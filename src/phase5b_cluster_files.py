import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

np.random.seed(42)

def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "processed"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df_lp = pd.read_csv(data_dir / "phase2_features_lp3500.csv", keep_default_na=False)
    df_lp["severity"] = df_lp["severity"].astype(str).str.strip().str.replace("\"", "").str.replace("'", "")
    assert len(df_lp) == 4643, f"Expected 4643 rows, got {len(df_lp)}"

    with open(results_dir / "feature_groups.json") as f:
        fg = json.load(f)
    ALL_21 = fg["time"] + fg["freq"] + fg["envelope"]
    DBSCAN_FEATURES = [feat for feat in ALL_21 if feat != "std"]
    assert len(DBSCAN_FEATURES) == 20, f"Expected 20 features, got {len(DBSCAN_FEATURES)}"

    with open(data_dir / "phase3_splits.json") as f:
        splits_json = json.load(f)

    df_dbscan_ref = pd.read_csv(results_dir / "phase5_dbscan.csv")

    LOLO_SPLITS = ["lolo_0", "lolo_1", "lolo_2", "lolo_3"]
    cluster_file_rows = []

    split_cluster_info = {}

    for s_name in LOLO_SPLITS:
        s_data = splits_json[s_name]
        te_files = s_data["test_files"]
        te_sub = df_lp[df_lp["file_id"].isin(te_files)].copy()
        N = len(te_sub)

        X_te = te_sub[DBSCAN_FEATURES].values

        # StandardScaler + PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_te)
        pca = PCA(n_components=0.95, svd_solver="full", random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        n_pcs = int(X_pca.shape[1])
        min_samples = int(np.clip(2 * n_pcs, 10, 30))

        # k-distance chord elbow
        nbrs = NearestNeighbors(n_neighbors=min_samples, metric="euclidean").fit(X_pca)
        k_dist = np.sort(nbrs.kneighbors()[0][:, -1])
        x_norm = np.arange(N) / (N - 1)
        y_norm = (k_dist - k_dist[0]) / max(k_dist[-1] - k_dist[0], 1e-12)
        chord_dist = np.abs(y_norm - x_norm) / np.sqrt(2.0)
        elbow_idx = int(np.argmax(chord_dist))
        eps_auto = float(k_dist[elbow_idx])

        # DBSCAN
        dbscan = DBSCAN(eps=eps_auto, min_samples=min_samples).fit(X_pca)
        labels = dbscan.labels_
        te_sub["cluster_label"] = labels

        n_clusters = int(len(set(labels) - {-1}))
        n_noise = int(np.sum(labels == -1))

        # Assertions
        ref_row = df_dbscan_ref[df_dbscan_ref["split"] == s_name].iloc[0]
        assert n_clusters == int(ref_row["n_clusters"]), f"n_clusters mismatch in {s_name}: {n_clusters} vs {ref_row['n_clusters']}"
        assert n_noise == int(ref_row["n_noise"]), f"n_noise mismatch in {s_name}: {n_noise} vs {ref_row['n_noise']}"

        # Per file window count
        file_win_counts = te_sub.groupby("file_id").size().to_dict()

        # Find largest cluster (excluding -1)
        non_noise_labels = sorted(list(set(labels) - {-1}))
        cluster_sizes = {c: int(np.sum(labels == c)) for c in non_noise_labels}
        largest_cluster = max(cluster_sizes, key=cluster_sizes.get)

        split_cluster_info[s_name] = {
            "largest_cluster": largest_cluster,
            "non_noise_clusters": non_noise_labels,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "te_sub": te_sub,
            "file_win_counts": file_win_counts,
        }

        # Build CSV rows
        # Group by split, cluster_label, file_id, fault_class, severity, load_hp
        grouped = te_sub.groupby(["cluster_label", "file_id", "fault_class", "severity", "load_hp"]).size().reset_index(name="n_windows_in_cluster")
        for _, r in grouped.iterrows():
            fid = int(r["file_id"])
            cluster_file_rows.append({
                "split": s_name,
                "cluster_label": int(r["cluster_label"]),
                "file_id": fid,
                "fault_class": str(r["fault_class"]),
                "severity": str(r["severity"]),
                "load_hp": int(r["load_hp"]),
                "n_windows_in_cluster": int(r["n_windows_in_cluster"]),
                "n_windows_of_file": int(file_win_counts[fid]),
            })

    df_cluster_files = pd.DataFrame(cluster_file_rows)
    out_csv = results_dir / "phase5_dbscan_cluster_files.csv"
    df_cluster_files.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} ({len(df_cluster_files)} rows)")

    # Reporting
    print("\n" + "=" * 80)
    print("DBSCAN CLUSTER COMPOSITION REPORT (NUMBERS ONLY)")
    print("=" * 80)

    for s_name in LOLO_SPLITS:
        info = split_cluster_info[s_name]
        largest_c = info["largest_cluster"]
        non_noise_c = info["non_noise_clusters"]
        te_sub = info["te_sub"]
        file_win_counts = info["file_win_counts"]

        print(f"\n--- Split: {s_name} (Total Clusters: {info['n_clusters']}, Total Noise Windows: {info['n_noise']}) ---")
        
        # 1. Non-largest clusters
        other_clusters = [c for c in non_noise_c if c != largest_c]
        print(f"Small Clusters (other than largest cluster {largest_c}):")
        if not other_clusters:
            print("  None")
        for c in other_clusters:
            c_rows = df_cluster_files[(df_cluster_files["split"] == s_name) & (df_cluster_files["cluster_label"] == c)]
            single_file = (len(c_rows) == 1)
            print(f"  Cluster {c} (Total windows: {c_rows['n_windows_in_cluster'].sum()}, Single file only: {single_file}):")
            for _, r in c_rows.iterrows():
                print(f"    file_id={r['file_id']}, class={r['fault_class']}, severity={r['severity']}, load={r['load_hp']} hp: {r['n_windows_in_cluster']}/{r['n_windows_of_file']} windows")

        # 2. Largest cluster
        print(f"\nLargest Cluster (Cluster {largest_c}, Total windows: {df_cluster_files[(df_cluster_files['split'] == s_name) & (df_cluster_files['cluster_label'] == largest_c)]['n_windows_in_cluster'].sum()}):")
        lg_rows = df_cluster_files[(df_cluster_files["split"] == s_name) & (df_cluster_files["cluster_label"] == largest_c)]
        
        # Get noise windows per file
        noise_rows = df_cluster_files[(df_cluster_files["split"] == s_name) & (df_cluster_files["cluster_label"] == -1)]
        noise_dict = {int(r["file_id"]): int(r["n_windows_in_cluster"]) for _, r in noise_rows.iterrows()}

        for _, r in lg_rows.iterrows():
            fid = int(r["file_id"])
            noise_w = noise_dict.get(fid, 0)
            print(f"    file_id={fid}, class={r['fault_class']}, severity={r['severity']}, load={r['load_hp']} hp: {r['n_windows_in_cluster']}/{r['n_windows_of_file']} windows in cluster, {noise_w}/{r['n_windows_of_file']} noise windows")

        # Files in test fold not in largest cluster:
        other_files = [f for f in te_sub["file_id"].unique() if f not in lg_rows["file_id"].values]
        if other_files:
            print(f"  Files outside largest cluster (in small clusters or pure noise):")
            for fid in sorted(other_files):
                f_sub = te_sub[te_sub["file_id"] == fid]
                cls = f_sub["fault_class"].iloc[0]
                sev = f_sub["severity"].iloc[0]
                ld = f_sub["load_hp"].iloc[0]
                tot = len(f_sub)
                noise_w = noise_dict.get(fid, 0)
                # find which cluster it is in
                cl_assigned = sorted(list(f_sub["cluster_label"].unique()))
                print(f"    file_id={fid}, class={cls}, severity={sev}, load={ld} hp: assigned clusters={cl_assigned}, {noise_w}/{tot} noise windows")

if __name__ == "__main__":
    main()
