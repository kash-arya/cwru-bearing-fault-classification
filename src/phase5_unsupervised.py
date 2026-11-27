"""Phase 5: Unsupervised Learning - Novelty Detection (IsolationForest) & Clustering (DBSCAN)

Evaluates out-of-distribution novelty detection across 4 LOLO test folds with 5 seeds,
and evaluates unsupervised clustering via DBSCAN with automated k-distance elbow selection.
"""
# Reads: data/processed/phase2_features_lp3500.csv, data/processed/phase2_features_nolp.csv, results/feature_groups.json, data/processed/phase3_splits.json, results/phase3_splits_summary.csv
# Writes: results/phase5_novelty_A.csv, results/phase5_novelty_B.csv, results/diagnostic_phase5_kdistance.png, results/phase5_dbscan.csv, results/phase5_dbscan_crosstab.csv, results/phase5_dbscan_eps_sensitivity.csv, results/phase5_summary.md

from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import roc_auc_score, adjusted_rand_score, normalized_mutual_info_score

np.random.seed(42)

CLASS_NAMES = ["Normal", "Inner Race", "Ball", "Outer Race"]
FAULT_CLASSES = ["Inner Race", "Ball", "Outer Race"]
SEVERITIES = ["0.007", "0.014", "0.021"]
LOLO_SPLITS = ["lolo_0", "lolo_1", "lolo_2", "lolo_3"]
SEEDS = [0, 1, 2, 3, 4]


def to_markdown_table(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in df.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "processed"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Inputs & Mandatory Assertions
    # -------------------------------------------------------------------------
    df_lp = pd.read_csv(data_dir / "phase2_features_lp3500.csv", keep_default_na=False)
    df_lp["severity"] = df_lp["severity"].astype(str).str.strip().str.replace("\"", "").str.replace("'", "")
    assert len(df_lp) == 4643, f"Expected 4643 rows in lp3500, got {len(df_lp)}"

    df_no = pd.read_csv(data_dir / "phase2_features_nolp.csv", keep_default_na=False)
    df_no["severity"] = df_no["severity"].astype(str).str.strip().str.replace("\"", "").str.replace("'", "")
    assert len(df_no) == 4643, f"Expected 4643 rows in nolp, got {len(df_no)}"

    with open(results_dir / "feature_groups.json") as f:
        fg = json.load(f)
    ALL_21 = fg["time"] + fg["freq"] + fg["envelope"]
    assert len(ALL_21) == 21, f"Expected 21 features, got {len(ALL_21)}"
    NO_LOUDNESS_18 = [feat for feat in ALL_21 if feat not in ("rms", "std", "peak_to_peak")]
    assert len(NO_LOUDNESS_18) == 18, f"Expected 18 features, got {len(NO_LOUDNESS_18)}"

    # Addendum A: DBSCAN features without std (20 features)
    DBSCAN_FEATURES = [feat for feat in ALL_21 if feat != "std"]
    assert len(DBSCAN_FEATURES) == 20, f"Expected 20 DBSCAN features, got {len(DBSCAN_FEATURES)}"

    with open(data_dir / "phase3_splits.json") as f:
        splits_json = json.load(f)

    df_sum = pd.read_csv(results_dir / "phase3_splits_summary.csv")

    # Mandatory Assertions
    print("Verifying mandatory assertions on LOLO splits...")
    for s_name in LOLO_SPLITS:
        s_data = splits_json[s_name]
        tr_files = set(s_data["train_files"])
        te_files = set(s_data["test_files"])

        # Assertion 1: disjoint files
        assert tr_files.isdisjoint(te_files), f"Assertion 1 failed: files overlap in {s_name}"

        tr_df = df_lp[df_lp["file_id"].isin(tr_files)]
        te_df = df_lp[df_lp["file_id"].isin(te_files)]

        # Assertion 3: all 4 classes present, window counts match summary exactly
        assert set(te_df["fault_class"].unique()) == set(CLASS_NAMES), f"Assertion 3 failed: missing class in {s_name} test"
        assert set(tr_df["fault_class"].unique()) == set(CLASS_NAMES), f"Assertion 3 failed: missing class in {s_name} train"

        row_tr = df_sum[(df_sum["split"] == s_name) & (df_sum["kind"] == "outer") & (df_sum["set"] == "train")].iloc[0]
        row_te = df_sum[(df_sum["split"] == s_name) & (df_sum["kind"] == "outer") & (df_sum["set"] == "test")].iloc[0]

        assert len(tr_df) == int(row_tr["n_windows_total"]), f"Assertion 3 failed: {s_name} train total mismatch"
        assert len(te_df) == int(row_te["n_windows_total"]), f"Assertion 3 failed: {s_name} test total mismatch"

        for c in CLASS_NAMES:
            assert len(tr_df[tr_df["fault_class"] == c]) == int(row_tr[f"windows_{c}"]), f"Assertion 3 failed: {s_name} train {c} mismatch"
            assert len(te_df[te_df["fault_class"] == c]) == int(row_te[f"windows_{c}"]), f"Assertion 3 failed: {s_name} test {c} mismatch"

    print("All input pre-checks and mandatory assertions passed successfully!")

    # -------------------------------------------------------------------------
    # 2. Experiment 1A: Unseen Fault Class (results/phase5_novelty_A.csv)
    # -------------------------------------------------------------------------
    print("\n--- Running Experiment 1A: Unseen Fault Class ---")
    t0_1a = time.time()
    novelty_a_rows = []

    exp1a_variants = [
        ("lp3500_all21", df_lp, ALL_21),
        ("lp3500_no_loudness18", df_lp, NO_LOUDNESS_18),
    ]

    for var_name, df_src, feat_cols in exp1a_variants:
        for s_name in LOLO_SPLITS:
            s_data = splits_json[s_name]
            tr_files = s_data["train_files"]
            te_files = s_data["test_files"]

            tr_sub = df_src[df_src["file_id"].isin(tr_files)]
            te_sub = df_src[df_src["file_id"].isin(te_files)]

            # Assertion 2: Scored windows come strictly from test_files
            assert set(te_sub["file_id"].unique()).issubset(set(te_files))
            assert set(te_sub["file_id"].unique()).isdisjoint(set(tr_files))

            for held_out_class in FAULT_CLASSES:
                # 3 known classes: Normal + 2 other fault classes
                known_classes = [c for c in CLASS_NAMES if c != held_out_class]
                tr_known = tr_sub[tr_sub["fault_class"].isin(known_classes)]

                # Assertion 4: tr_known contains strictly 3 known classes and 0 rows of held_out_class
                assert len(tr_known[tr_known["fault_class"] == held_out_class]) == 0, f"Assertion 4 failed: {held_out_class} in train for {s_name}"
                assert set(tr_known["fault_class"].unique()) == set(known_classes), f"Assertion 4 failed: missing known class in {s_name}"

                X_tr = tr_known[feat_cols].values
                X_te = te_sub[feat_cols].values
                y_te_cls = te_sub["fault_class"].values

                # Fit 5 seeds
                seed_scores = []
                for seed in SEEDS:
                    model = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1)
                    model.fit(X_tr)
                    # IsolationForest.score_samples gives higher values to inliers; negate so higher score indicates anomaly (positive class 1)
                    scores = -model.score_samples(X_te)
                    seed_scores.append(scores)

                # Comparisons
                other_faults = [c for c in FAULT_CLASSES if c != held_out_class]
                comparisons = [
                    ("overall", None),
                    ("vs_Normal", "Normal"),
                    (f"vs_{other_faults[0]}", other_faults[0]),
                    (f"vs_{other_faults[1]}", other_faults[1]),
                ]

                for comp_name, comp_target in comparisons:
                    if comp_target is None:
                        # overall: held_out_class (1) vs all 3 known classes (0)
                        mask = np.ones(len(y_te_cls), dtype=bool)
                        y_bin = (y_te_cls == held_out_class).astype(int)
                    else:
                        mask = np.isin(y_te_cls, [held_out_class, comp_target])
                        y_bin = (y_te_cls[mask] == held_out_class).astype(int)

                    n_pos = int(np.sum(y_bin == 1))
                    n_neg = int(np.sum(y_bin == 0))
                    assert n_pos > 0 and n_neg > 0, f"Zero samples for {comp_name} in {s_name}"

                    seed_aurocs = []
                    for s_idx in range(len(SEEDS)):
                        sc_sub = seed_scores[s_idx][mask]
                        # Threshold-free ROC-AUC: positive class 1 = held_out_class, negative class 0 = comparison target
                        auc = float(roc_auc_score(y_bin, sc_sub))
                        seed_aurocs.append(auc)

                    row_dict = {
                        "variant": var_name,
                        "split": s_name,
                        "held_out_class": held_out_class,
                        "comparison": comp_name,
                        "n_pos_windows": n_pos,
                        "n_neg_windows": n_neg,
                        # Population standard deviation (ddof=0) across 5 seeds ([0, 1, 2, 3, 4])
                        "auroc_seed_mean": float(np.mean(seed_aurocs)),
                        "auroc_seed_std": float(np.std(seed_aurocs, ddof=0)),
                    }
                    for s_idx, seed in enumerate(SEEDS):
                        row_dict[f"auroc_seed{seed}"] = seed_aurocs[s_idx]

                    novelty_a_rows.append(row_dict)

    df_novelty_a = pd.DataFrame(novelty_a_rows)
    df_novelty_a.to_csv(results_dir / "phase5_novelty_A.csv", index=False)
    runtime_1a = time.time() - t0_1a
    print(f"Saved: {results_dir / 'phase5_novelty_A.csv'} ({len(df_novelty_a)} rows, runtime: {runtime_1a:.2f}s)")

    # -------------------------------------------------------------------------
    # 3. Experiment 1B: Normal-Only Detector (results/phase5_novelty_B.csv)
    # -------------------------------------------------------------------------
    print("\n--- Running Experiment 1B: Normal-Only Detector ---")
    t0_1b = time.time()
    novelty_b_rows = []

    exp1b_variants = [
        ("lp3500_all21", df_lp, ALL_21, False),
        ("lp3500_no_loudness18", df_lp, NO_LOUDNESS_18, False),
        ("rms_only_baseline", df_lp, ["rms"], True),
        ("nolp_all21", df_no, ALL_21, False),
    ]

    for var_name, df_src, feat_cols, is_rms_baseline in exp1b_variants:
        for s_name in LOLO_SPLITS:
            s_data = splits_json[s_name]
            tr_files = s_data["train_files"]
            te_files = s_data["test_files"]

            tr_sub = df_src[df_src["file_id"].isin(tr_files)]
            te_sub = df_src[df_src["file_id"].isin(te_files)]

            # Assertion 2: Scored windows come strictly from test_files
            assert set(te_sub["file_id"].unique()).issubset(set(te_files))
            assert set(te_sub["file_id"].unique()).isdisjoint(set(tr_files))

            tr_norm = tr_sub[tr_sub["fault_class"] == "Normal"]
            # Assertion 5: tr_norm contains strictly Normal rows
            assert len(tr_norm) > 0, f"No normal rows in train for {s_name}"
            assert (tr_norm["fault_class"] == "Normal").all(), f"Assertion 5 failed: non-Normal in train for {s_name}"

            X_tr_norm = tr_norm[feat_cols].values
            X_te = te_sub[feat_cols].values
            y_te_cls = te_sub["fault_class"].values
            y_te_sev = te_sub["severity"].values

            seed_scores = []
            if is_rms_baseline:
                # Deterministic baseline: raw rms directly as score
                # Higher window RMS predicts fault class 1 vs Normal class 0
                raw_rms_scores = te_sub["rms"].values
                for _ in SEEDS:
                    seed_scores.append(raw_rms_scores)
            else:
                for seed in SEEDS:
                    model = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1)
                    model.fit(X_tr_norm)
                    # Negate score_samples so anomalous (fault) windows score higher than Normal training baseline
                    scores = -model.score_samples(X_te)
                    seed_scores.append(scores)

            # Subsets to evaluate:
            # 1. overall: Normal vs all faults
            # 2. per_class: Normal vs each fault class (3 classes)
            # 3. per_class_severity: Normal vs each (fault_class, severity) pair (9 pairs)
            eval_tasks = [("overall", "ALL", "ALL", None, None)]
            for fc in FAULT_CLASSES:
                eval_tasks.append(("per_class", fc, "ALL", fc, None))
            for fc in FAULT_CLASSES:
                for sev in SEVERITIES:
                    eval_tasks.append(("per_class_severity", fc, sev, fc, sev))

            for sub_type, fc_label, sev_label, target_fc, target_sev in eval_tasks:
                if sub_type == "overall":
                    mask = np.ones(len(y_te_cls), dtype=bool)
                    y_bin = (y_te_cls != "Normal").astype(int)
                elif sub_type == "per_class":
                    mask = (y_te_cls == "Normal") | (y_te_cls == target_fc)
                    y_bin = (y_te_cls[mask] == target_fc).astype(int)
                elif sub_type == "per_class_severity":
                    mask = (y_te_cls == "Normal") | ((y_te_cls == target_fc) & (y_te_sev == target_sev))
                    y_bin = ((y_te_cls[mask] == target_fc) & (y_te_sev[mask] == target_sev)).astype(int)

                n_norm = int(np.sum(y_bin == 0))
                n_fault = int(np.sum(y_bin == 1))
                assert n_norm > 0 and n_fault > 0, f"Missing class in {sub_type} for {s_name}"

                seed_aurocs = []
                for s_idx in range(len(SEEDS)):
                    sc_sub = seed_scores[s_idx][mask]
                    auc = float(roc_auc_score(y_bin, sc_sub))
                    seed_aurocs.append(auc)

                row_dict = {
                    "variant": var_name,
                    "split": s_name,
                    "subset_type": sub_type,
                    "fault_class": fc_label,
                    "severity": sev_label,
                    "n_normal_windows": n_norm,
                    "n_fault_windows": n_fault,
                    "auroc_seed_mean": float(np.mean(seed_aurocs)),
                    "auroc_seed_std": float(np.std(seed_aurocs, ddof=0)) if not is_rms_baseline else 0.0,
                }
                for s_idx, seed in enumerate(SEEDS):
                    row_dict[f"auroc_seed{seed}"] = seed_aurocs[s_idx]

                novelty_b_rows.append(row_dict)

    df_novelty_b = pd.DataFrame(novelty_b_rows)
    df_novelty_b.to_csv(results_dir / "phase5_novelty_B.csv", index=False)
    runtime_1b = time.time() - t0_1b
    print(f"Saved: {results_dir / 'phase5_novelty_B.csv'} ({len(df_novelty_b)} rows, runtime: {runtime_1b:.2f}s)")

    # -------------------------------------------------------------------------
    # 4. Experiment 2: Unsupervised Clustering (DBSCAN)
    # -------------------------------------------------------------------------
    print("\n--- Running Experiment 2: Unsupervised Clustering (DBSCAN) ---")
    t0_2 = time.time()
    dbscan_rows = []
    crosstab_rows = []
    sensitivity_rows = []

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, s_name in enumerate(LOLO_SPLITS):
        s_data = splits_json[s_name]
        te_files = s_data["test_files"]
        tr_files = s_data["train_files"]

        te_sub = df_lp[df_lp["file_id"].isin(te_files)]
        # Assertion 2: Scored windows come strictly from test_files
        assert set(te_sub["file_id"].unique()).issubset(set(te_files))
        assert set(te_sub["file_id"].unique()).isdisjoint(set(tr_files))

        X_te = te_sub[DBSCAN_FEATURES].values
        y_true = te_sub["fault_class"].values
        N = len(X_te)

        # 1. Fit StandardScaler and PCA on X_test (intentional for unsupervised test-fold clustering)
        # Standardize 20 features to zero mean and unit variance, then retain >= 95% cumulative variance via full-SVD PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_te)
        pca = PCA(n_components=0.95, svd_solver="full", random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        n_pcs = int(X_pca.shape[1])
        var_retained = float(np.sum(pca.explained_variance_ratio_))

        # 2. Set min_samples
        # Clamp DBSCAN min_samples = clip(2 * n_pcs, 10, 30)
        min_samples = int(np.clip(2 * n_pcs, 10, 30))

        # 3. Automatic k-distance elbow selection
        # Sort distance to k-th nearest Euclidean neighbor (k = min_samples, including self at column 0)
        nbrs = NearestNeighbors(n_neighbors=min_samples, metric="euclidean").fit(X_pca)
        k_dist = np.sort(nbrs.kneighbors()[0][:, -1])

        # Normalize rank and k-distance to [0, 1]; elbow maximizes orthogonal distance |y_norm - x_norm| / sqrt(2) to diagonal chord
        x_norm = np.arange(N) / (N - 1)
        y_norm = (k_dist - k_dist[0]) / max(k_dist[-1] - k_dist[0], 1e-12)
        chord_dist = np.abs(y_norm - x_norm) / np.sqrt(2.0)
        elbow_idx = int(np.argmax(chord_dist))
        eps_auto = float(k_dist[elbow_idx])

        # Plot k-distance panel
        ax = axes[idx]
        ax.plot(np.arange(N), k_dist, color="steelblue", lw=2, label=f"k-dist (k={min_samples})")
        ax.plot([0, N - 1], [k_dist[0], k_dist[-1]], "--", color="gray", label="Chord line")
        ax.scatter([elbow_idx], [eps_auto], color="crimson", s=60, zorder=5, label=f"Elbow (eps={eps_auto:.3f})")
        ax.set_title(f"{s_name} (N={N}, d={n_pcs})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Window Rank (Sorted)", fontsize=9)
        ax.set_ylabel("k-NN Distance", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        # 4. Run DBSCAN with eps_auto
        dbscan = DBSCAN(eps=eps_auto, min_samples=min_samples).fit(X_pca)
        labels = dbscan.labels_
        unique_labels = sorted(list(set(labels)))
        # Exclude DBSCAN noise label (-1) when counting distinct discovered clusters
        n_clusters = int(len(set(labels) - {-1}))
        n_noise = int(np.sum(labels == -1))
        noise_frac = float(np.mean(labels == -1))

        # Clustering alignment against ground-truth fault_class including noise label -1 as a separate cluster
        ari_noise = float(adjusted_rand_score(y_true, labels))
        nmi_noise = float(normalized_mutual_info_score(y_true, labels))

        # Guard: if np.sum(mask) < 2 OR n_clusters < 2, set to np.nan
        mask = (labels != -1)
        if np.sum(mask) < 2 or n_clusters < 2:
            ari_no_noise = np.nan
            nmi_no_noise = np.nan
        else:
            # ARI and NMI evaluated exclusively on core/border points (excluding label -1)
            ari_no_noise = float(adjusted_rand_score(y_true[mask], labels[mask]))
            nmi_no_noise = float(normalized_mutual_info_score(y_true[mask], labels[mask]))

        dbscan_rows.append({
            "split": s_name,
            "n_test_windows": N,
            "n_features": len(DBSCAN_FEATURES),
            "n_pcs": n_pcs,
            "variance_retained": var_retained,
            "min_samples": min_samples,
            "elbow_idx": elbow_idx,
            "eps_auto": eps_auto,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_fraction": noise_frac,
            "ari_noise_as_cluster": ari_noise,
            "nmi_noise_as_cluster": nmi_noise,
            "ari_no_noise": ari_no_noise,
            "nmi_no_noise": nmi_no_noise,
        })

        # Crosstab rows for multiplier == 1.0
        for lbl in unique_labels:
            mask_lbl = (labels == lbl)
            cnt_norm = int(np.sum(mask_lbl & (y_true == "Normal")))
            cnt_ir = int(np.sum(mask_lbl & (y_true == "Inner Race")))
            cnt_ball = int(np.sum(mask_lbl & (y_true == "Ball")))
            cnt_or = int(np.sum(mask_lbl & (y_true == "Outer Race")))
            tot = cnt_norm + cnt_ir + cnt_ball + cnt_or
            crosstab_rows.append({
                "split": s_name,
                "cluster_label": lbl,
                "count_Normal": cnt_norm,
                "count_Inner Race": cnt_ir,
                "count_Ball": cnt_ball,
                "count_Outer Race": cnt_or,
                "total": tot,
            })

        # 6. Eps Sensitivity Sweep: [0.5, 0.75, 1.0, 1.25, 1.5]
        for mult in [0.5, 0.75, 1.0, 1.25, 1.5]:
            eps_sw = float(eps_auto * mult)
            db_sw = DBSCAN(eps=eps_sw, min_samples=min_samples).fit(X_pca)
            lbl_sw = db_sw.labels_
            n_cl_sw = int(len(set(lbl_sw) - {-1}))
            n_noise_sw = int(np.sum(lbl_sw == -1))
            noise_frac_sw = float(np.mean(lbl_sw == -1))

            ari_n_sw = float(adjusted_rand_score(y_true, lbl_sw))
            nmi_n_sw = float(normalized_mutual_info_score(y_true, lbl_sw))

            mask_sw = (lbl_sw != -1)
            if np.sum(mask_sw) < 2 or n_cl_sw < 2:
                ari_no_sw = np.nan
                nmi_no_sw = np.nan
            else:
                ari_no_sw = float(adjusted_rand_score(y_true[mask_sw], lbl_sw[mask_sw]))
                nmi_no_sw = float(normalized_mutual_info_score(y_true[mask_sw], lbl_sw[mask_sw]))

            sensitivity_rows.append({
                "split": s_name,
                "multiplier": mult,
                "eps": eps_sw,
                "min_samples": min_samples,
                "n_clusters": n_cl_sw,
                "noise_fraction": noise_frac_sw,
                "ari_noise_as_cluster": ari_n_sw,
                "nmi_noise_as_cluster": nmi_n_sw,
                "ari_no_noise": ari_no_sw,
                "nmi_no_noise": nmi_no_sw,
            })

    plt.tight_layout()
    fig.savefig(results_dir / "diagnostic_phase5_kdistance.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {results_dir / 'diagnostic_phase5_kdistance.png'}")

    df_dbscan = pd.DataFrame(dbscan_rows)
    df_dbscan.to_csv(results_dir / "phase5_dbscan.csv", index=False)
    print(f"Saved: {results_dir / 'phase5_dbscan.csv'}")

    df_crosstab = pd.DataFrame(crosstab_rows)
    df_crosstab.to_csv(results_dir / "phase5_dbscan_crosstab.csv", index=False)
    print(f"Saved: {results_dir / 'phase5_dbscan_crosstab.csv'}")

    df_sens = pd.DataFrame(sensitivity_rows)
    df_sens.to_csv(results_dir / "phase5_dbscan_eps_sensitivity.csv", index=False)
    print(f"Saved: {results_dir / 'phase5_dbscan_eps_sensitivity.csv'}")

    runtime_2 = time.time() - t0_2

    # -------------------------------------------------------------------------
    # 5. Summary Markdown (results/phase5_summary.md)
    # -------------------------------------------------------------------------
    print("\n--- Generating results/phase5_summary.md ---")

    # Table 1: Exp A summary
    # columns: Variant, Held-out Class, Comparison, lolo_0, lolo_1, lolo_2, lolo_3, Fold Mean +- Std, Mean Seed Std
    t1_rows = []
    for var_name, _, _ in exp1a_variants:
        for hoc in FAULT_CLASSES:
            other_faults = [c for c in FAULT_CLASSES if c != hoc]
            comps = ["overall", "vs_Normal", f"vs_{other_faults[0]}", f"vs_{other_faults[1]}"]
            for comp in comps:
                sub = df_novelty_a[
                    (df_novelty_a["variant"] == var_name) &
                    (df_novelty_a["held_out_class"] == hoc) &
                    (df_novelty_a["comparison"] == comp)
                ]
                fold_seed_means = []
                fold_seed_stds = []
                row_dict = {
                    "Variant": var_name,
                    "Held-out Class": hoc,
                    "Comparison": comp,
                }
                for s in LOLO_SPLITS:
                    val_mean = sub[sub["split"] == s]["auroc_seed_mean"].iloc[0]
                    val_std = sub[sub["split"] == s]["auroc_seed_std"].iloc[0]
                    fold_seed_means.append(val_mean)
                    fold_seed_stds.append(val_std)
                    row_dict[s] = f"{val_mean:.4f}"

                f_mean = float(np.mean(fold_seed_means))
                f_std = float(np.std(fold_seed_means, ddof=0))
                s_std_m = float(np.mean(fold_seed_stds))

                row_dict["Fold Mean +- Std"] = f"{f_mean:.4f} +- {f_std:.4f}"
                row_dict["Mean Seed Std"] = f"{s_std_m:.4f}"
                t1_rows.append(row_dict)

    df_t1 = pd.DataFrame(t1_rows)

    # Table 2: Exp B summary
    # columns: Variant, Subset Type, Fault Class, Severity, lolo_0, lolo_1, lolo_2, lolo_3, Fold Mean +- Std, Mean Seed Std
    t2_rows = []
    for var_name, _, _, _ in exp1b_variants:
        eval_tasks_summary = [("overall", "ALL", "ALL")]
        for fc in FAULT_CLASSES:
            eval_tasks_summary.append(("per_class", fc, "ALL"))
        for fc in FAULT_CLASSES:
            for sev in SEVERITIES:
                eval_tasks_summary.append(("per_class_severity", fc, sev))

        for sub_type, fc_label, sev_label in eval_tasks_summary:
            sub = df_novelty_b[
                (df_novelty_b["variant"] == var_name) &
                (df_novelty_b["subset_type"] == sub_type) &
                (df_novelty_b["fault_class"] == fc_label) &
                (df_novelty_b["severity"] == sev_label)
            ]
            fold_seed_means = []
            fold_seed_stds = []
            row_dict = {
                "Variant": var_name,
                "Subset Type": sub_type,
                "Fault Class": fc_label,
                "Severity": sev_label,
            }
            for s in LOLO_SPLITS:
                val_mean = sub[sub["split"] == s]["auroc_seed_mean"].iloc[0]
                val_std = sub[sub["split"] == s]["auroc_seed_std"].iloc[0]
                fold_seed_means.append(val_mean)
                fold_seed_stds.append(val_std)
                row_dict[s] = f"{val_mean:.4f}"

            f_mean = float(np.mean(fold_seed_means))
            f_std = float(np.std(fold_seed_means, ddof=0))
            s_std_m = float(np.mean(fold_seed_stds))

            row_dict["Fold Mean +- Std"] = f"{f_mean:.4f} +- {f_std:.4f}"
            row_dict["Mean Seed Std"] = f"{s_std_m:.4f}"
            t2_rows.append(row_dict)

    df_t2 = pd.DataFrame(t2_rows)

    # Table 3: DBSCAN per-fold summary + Mean +- Std
    t3_rows = []
    for _, row in df_dbscan.iterrows():
        t3_rows.append({
            "Split": row["split"],
            "N": int(row["n_test_windows"]),
            "Features": int(row["n_features"]),
            "PCs": int(row["n_pcs"]),
            "Var Retained": f"{row['variance_retained']:.4f}",
            "min_samples": int(row["min_samples"]),
            "eps_auto": f"{row['eps_auto']:.4f}",
            "Clusters": int(row["n_clusters"]),
            "Noise Frac": f"{row['noise_fraction']:.4f}",
            "ARI (Noise Clust)": f"{row['ari_noise_as_cluster']:.4f}",
            "NMI (Noise Clust)": f"{row['nmi_noise_as_cluster']:.4f}",
            "ARI (No Noise)": f"{row['ari_no_noise']:.4f}" if not np.isnan(row["ari_no_noise"]) else "NaN",
            "NMI (No Noise)": f"{row['nmi_no_noise']:.4f}" if not np.isnan(row["nmi_no_noise"]) else "NaN",
        })

    # Mean +- Std row
    mean_ari_nc = df_dbscan["ari_noise_as_cluster"].mean()
    std_ari_nc = df_dbscan["ari_noise_as_cluster"].std(ddof=0)
    mean_nmi_nc = df_dbscan["nmi_noise_as_cluster"].mean()
    std_nmi_nc = df_dbscan["nmi_noise_as_cluster"].std(ddof=0)
    mean_noise_f = df_dbscan["noise_fraction"].mean()
    std_noise_f = df_dbscan["noise_fraction"].std(ddof=0)
    valid_ari_nn = df_dbscan["ari_no_noise"].dropna()
    mean_ari_nn = valid_ari_nn.mean() if len(valid_ari_nn) > 0 else np.nan
    std_ari_nn = valid_ari_nn.std(ddof=0) if len(valid_ari_nn) > 0 else np.nan
    valid_nmi_nn = df_dbscan["nmi_no_noise"].dropna()
    mean_nmi_nn = valid_nmi_nn.mean() if len(valid_nmi_nn) > 0 else np.nan
    std_nmi_nn = valid_nmi_nn.std(ddof=0) if len(valid_nmi_nn) > 0 else np.nan

    t3_rows.append({
        "Split": "Mean +- Std",
        "N": "-",
        "Features": 20,
        "PCs": f"{df_dbscan['n_pcs'].mean():.1f}",
        "Var Retained": f"{df_dbscan['variance_retained'].mean():.4f}",
        "min_samples": f"{df_dbscan['min_samples'].mean():.1f}",
        "eps_auto": f"{df_dbscan['eps_auto'].mean():.4f} +- {df_dbscan['eps_auto'].std(ddof=0):.4f}",
        "Clusters": f"{df_dbscan['n_clusters'].mean():.2f}",
        "Noise Frac": f"{mean_noise_f:.4f} +- {std_noise_f:.4f}",
        "ARI (Noise Clust)": f"{mean_ari_nc:.4f} +- {std_ari_nc:.4f}",
        "NMI (Noise Clust)": f"{mean_nmi_nc:.4f} +- {std_nmi_nc:.4f}",
        "ARI (No Noise)": f"{mean_ari_nn:.4f} +- {std_ari_nn:.4f}" if not np.isnan(mean_ari_nn) else "NaN",
        "NMI (No Noise)": f"{mean_nmi_nn:.4f} +- {std_nmi_nn:.4f}" if not np.isnan(mean_nmi_nn) else "NaN",
    })
    df_t3 = pd.DataFrame(t3_rows)

    # Table 4: DBSCAN eps sensitivity summary (multiplier 0.5..1.5 per fold and 4-fold mean)
    t4_rows = []
    for mult in [0.5, 0.75, 1.0, 1.25, 1.5]:
        sub_sens = df_sens[df_sens["multiplier"] == mult]
        row_dict = {"Multiplier": f"{mult:.2f}"}
        ari_vals = []
        noise_vals = []
        clusters_vals = []
        for s in LOLO_SPLITS:
            val_ari = sub_sens[sub_sens["split"] == s]["ari_noise_as_cluster"].iloc[0]
            val_noise = sub_sens[sub_sens["split"] == s]["noise_fraction"].iloc[0]
            val_cl = sub_sens[sub_sens["split"] == s]["n_clusters"].iloc[0]
            ari_vals.append(val_ari)
            noise_vals.append(val_noise)
            clusters_vals.append(val_cl)
            row_dict[f"{s} (ARI / Noise)"] = f"{val_ari:.4f} / {val_noise:.4f}"

        row_dict["Mean Clusters"] = f"{np.mean(clusters_vals):.1f}"
        row_dict["Mean ARI"] = f"{np.mean(ari_vals):.4f} +- {np.std(ari_vals, ddof=0):.4f}"
        row_dict["Mean Noise"] = f"{np.mean(noise_vals):.4f} +- {np.std(noise_vals, ddof=0):.4f}"
        t4_rows.append(row_dict)
    df_t4 = pd.DataFrame(t4_rows)

    # Write results/phase5_summary.md
    with open(results_dir / "phase5_summary.md", "w", encoding="utf-8") as f:
        f.write("# Phase 5: Unsupervised Novelty Detection & Clustering Summary\n\n")

        # Caveat block
        f.write("> [!NOTE]\n")
        f.write("> **Evaluation Caveats:**\n")
        f.write("> 1. Each LOLO test fold includes only 1 Normal file (58 windows in lolo_0, 117 windows in lolo_1..3).\n")
        f.write("> 2. Normal files originate from 48 kHz recordings resampled to 12 kHz, whereas fault files were acquired natively at 12 kHz.\n")
        f.write("> 3. DBSCAN feature standardization and PCA are fitted directly on each test fold, as unsupervised clustering evaluates natural grouping without train/test partitions.\n")
        f.write("> 4. Overlapping windows from the same continuous recording share high mutual information; the effective degrees of freedom for clustering and AUROC metrics align with the number of test files (10 per fold) rather than raw window counts. per_class_severity scores compare a single fault file against a single Normal file.\n\n")

        f.write("## Table 1: Experiment 1A — Unseen Fault Class Detection (IsolationForest AUROC)\n\n")
        f.write(to_markdown_table(df_t1) + "\n\n")

        f.write("## Table 2: Experiment 1B — Normal-Only Fault Detection (AUROC)\n\n")
        f.write(to_markdown_table(df_t2) + "\n\n")

        f.write("## Table 3: Experiment 2 — DBSCAN Test-Fold Clustering Summary\n\n")
        f.write(to_markdown_table(df_t3) + "\n\n")

        f.write("## Table 4: Experiment 2 — DBSCAN Epsilon Sensitivity Sweep\n\n")
        f.write(to_markdown_table(df_t4) + "\n\n")

    print(f"Saved: {results_dir / 'phase5_summary.md'}")

    # -------------------------------------------------------------------------
    # 6. Concise Terminal Output
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 5 CONCISE TERMINAL REPORT")
    print("=" * 80)
    print("Execution Runtimes:")
    print(f"  Experiment 1A (Unseen Fault Class): {runtime_1a:.2f}s")
    print(f"  Experiment 1B (Normal-Only Detector): {runtime_1b:.2f}s")
    print(f"  Experiment 2 (DBSCAN Clustering): {runtime_2:.2f}s")
    print("Confirmation that all 5 assertions passed: True")

    print("\nTable 1: Experiment A Summary (AUROC fold_mean +- fold_std [mean_seed_std]):")
    print(df_t1[["Variant", "Held-out Class", "Comparison", "Fold Mean +- Std", "Mean Seed Std"]].to_string(index=False))

    print("\nTable 2: Experiment B Summary (AUROC fold_mean +- fold_std [mean_seed_std]):")
    print(df_t2[["Variant", "Subset Type", "Fault Class", "Severity", "Fold Mean +- Std", "Mean Seed Std"]].to_string(index=False))

    print("\nTable 3: DBSCAN per-fold Summary:")
    print(df_t3.to_string(index=False))

    print("\nTable 4: DBSCAN Epsilon Sensitivity Summary:")
    print(df_t4[["Multiplier", "Mean Clusters", "Mean ARI", "Mean Noise"]].to_string(index=False))


if __name__ == "__main__":
    main()
