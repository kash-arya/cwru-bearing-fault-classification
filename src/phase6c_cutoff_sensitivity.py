"""Phase 6C: Cutoff Sensitivity Analysis for Low-Pass Anti-Aliasing Filter.

Evaluates cutoff frequencies [3000, 3500, 4000, 4500, 5000, 'none'] using
Random Forest across 7 outer splits and 3 random seeds.
"""
# Reads: results/phase1_file_manifest.csv, data/processed/phase1_signals_12k.pkl, data/processed/phase3_splits.json, data/processed/phase2_features_lp3500.csv, data/processed/phase2_features_nolp.csv, results/phase4_ablation.csv, cache data/processed/phase6c_cache/features_*.pkl
# Writes: cache data/processed/phase6c_cache/features_*.pkl, results/phase6c_power_removed.csv, results/phase6c_cutoff_metrics.csv, results/phase6c_per_file.csv, results/phase6c_cutoff_summary.md, results/fig_phase6c_cutoff_vs_macro_f1.png

import os
import sys
import time
import pickle
import json
from pathlib import Path

# Configure headless matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
np.random.seed(42)

import pandas as pd
import scipy.signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_features import compute_features_for_dataset, ALL_FEATURES, FEATURE_GROUPS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = DATA_DIR / "phase6c_cache"

CUTOFFS = [3000, 3500, 4000, 4500, 5000, "none"]
SEEDS = [42, 0, 1]
SPLIT_NAMES = ["lolo_0", "lolo_1", "lolo_2", "lolo_3", "sev_0.007", "sev_0.014", "sev_0.021"]
CLASS_NAMES = ["Normal", "Inner Race", "Ball", "Outer Race"]


def main():
    t_start = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Stage 1: Load Inputs and Build/Load Feature Tables
    # -------------------------------------------------------------------------
    t_stage1 = time.time()
    print("=== Stage 1: Load Inputs and Build/Load Feature Tables ===")

    manifest_df = pd.read_csv(RESULTS_DIR / "phase1_file_manifest.csv", keep_default_na=False)
    manifest_df["file_id"] = manifest_df["file_id"].astype(int)
    manifest_df = manifest_df.set_index("file_id")

    with open(DATA_DIR / "phase1_signals_12k.pkl", "rb") as f:
        signals = pickle.load(f)

    with open(DATA_DIR / "phase3_splits.json", "r") as f:
        splits_json = json.load(f)

    df_lp3500_ref = pd.read_csv(DATA_DIR / "phase2_features_lp3500.csv", keep_default_na=False)
    df_nolp_ref = pd.read_csv(DATA_DIR / "phase2_features_nolp.csv", keep_default_na=False)
    df_ablation_ref = pd.read_csv(RESULTS_DIR / "phase4_ablation.csv", keep_default_na=False)

    feature_tables = {}
    for cutoff in CUTOFFS:
        cache_file = CACHE_DIR / f"features_{cutoff}.pkl"
        if cache_file.exists():
            print(f"Loading cached feature table for cutoff {cutoff}...")
            with open(cache_file, "rb") as f:
                df_feat = pickle.load(f)
        else:
            t_feat_start = time.time()
            print(f"Computing feature table for cutoff {cutoff}...")
            if cutoff == "none":
                df_feat, _ = compute_features_for_dataset(signals, manifest_df, use_lp3500=False)
            else:
                # 8th-order Butterworth low-pass at c_val Hz (fs=12000 Hz) via zero-phase sosfiltfilt; use_lp3500=False skips the default 3500 Hz filter
                c_val = int(cutoff)
                sos = scipy.signal.butter(8, c_val, btype="lowpass", fs=12000, output="sos")
                filtered_signals = {}
                for fid in sorted(signals.keys()):
                    filtered_signals[fid] = {"x": scipy.signal.sosfiltfilt(sos, signals[fid]["x"])}
                df_feat, _ = compute_features_for_dataset(filtered_signals, manifest_df, use_lp3500=False)

            with open(cache_file, "wb") as f:
                pickle.dump(df_feat, f)
            print(f"Computed and cached cutoff {cutoff} in {time.time() - t_feat_start:.2f} s")

        feature_tables[cutoff] = df_feat

    # Fail-Fast Check (a): C = 3500 equals phase2_features_lp3500.csv
    print("\n--- Fail-Fast Check (a): C = 3500 vs phase2_features_lp3500.csv ---")
    df_3500 = feature_tables[3500]
    assert len(df_3500) == len(df_lp3500_ref), "Check (a) failed: row count mismatch"
    assert (df_3500["file_id"].values == df_lp3500_ref["file_id"].values).all(), "Check (a) failed: file_id order mismatch"
    assert (df_3500["window_index"].values == df_lp3500_ref["window_index"].values).all(), "Check (a) failed: window_index order mismatch"

    diff_3500 = 0.0
    for feat in ALL_FEATURES:
        max_d = float(np.max(np.abs(df_3500[feat].values.astype(float) - df_lp3500_ref[feat].values.astype(float))))
        if max_d > diff_3500:
            diff_3500 = max_d
    print(f"Check (a) Max absolute difference over 21 features: {diff_3500:.4e}")
    assert diff_3500 < 1e-9, f"Check (a) failed: max diff {diff_3500} >= 1e-9"
    print("Check (a) PASSED: C = 3500 matches phase2_features_lp3500.csv exactly (< 1e-9).")

    # Fail-Fast Check (b): 'none' equals phase2_features_nolp.csv
    print("\n--- Fail-Fast Check (b): 'none' vs phase2_features_nolp.csv ---")
    df_none = feature_tables["none"]
    assert len(df_none) == len(df_nolp_ref), "Check (b) failed: row count mismatch"
    assert (df_none["file_id"].values == df_nolp_ref["file_id"].values).all(), "Check (b) failed: file_id order mismatch"
    assert (df_none["window_index"].values == df_nolp_ref["window_index"].values).all(), "Check (b) failed: window_index order mismatch"

    diff_none = 0.0
    for feat in ALL_FEATURES:
        max_d = float(np.max(np.abs(df_none[feat].values.astype(float) - df_nolp_ref[feat].values.astype(float))))
        if max_d > diff_none:
            diff_none = max_d
    print(f"Check (b) Max absolute difference over 21 features: {diff_none:.4e}")
    assert diff_none < 1e-9, f"Check (b) failed: max diff {diff_none} >= 1e-9"
    print("Check (b) PASSED: 'none' matches phase2_features_nolp.csv exactly (< 1e-9).")

    stage1_time = time.time() - t_stage1
    print(f"Stage 1 completed in {stage1_time:.2f} s\n")

    # -------------------------------------------------------------------------
    # Stage 2: Power Removed Calculation per Cutoff & Fault Class
    # -------------------------------------------------------------------------
    t_stage2 = time.time()
    print("=== Stage 2: Power Removed by Filter per Cutoff ===")

    power_removed_rows = []
    # Pre-compute P_pre for all 40 files
    p_pre_dict = {}
    for fid in sorted(signals.keys()):
        raw_x = signals[fid]["x"]
        # Pre-filter AC signal power: mean((x - mean(x))^2)
        x_demean = raw_x - np.mean(raw_x)
        p_pre_dict[fid] = float(np.mean(x_demean ** 2))

    for cutoff in CUTOFFS:
        file_fracs = []
        if cutoff == "none":
            for fid in sorted(signals.keys()):
                meta = manifest_df.loc[fid]
                file_fracs.append({
                    "file_id": fid,
                    "fault_class": str(meta["fault_class"]),
                    "frac_removed": 0.0,
                })
        else:
            c_val = int(cutoff)
            sos = scipy.signal.butter(8, c_val, btype="lowpass", fs=12000, output="sos")
            for fid in sorted(signals.keys()):
                raw_x = signals[fid]["x"]
                x_demean = raw_x - np.mean(raw_x)
                # Zero-phase 8th-order low-pass filtered AC power and removed power fraction: 1 - P_post / P_pre
                x_filt = scipy.signal.sosfiltfilt(sos, x_demean)
                p_post = float(np.mean(x_filt ** 2))
                frac = float(1.0 - p_post / p_pre_dict[fid])
                meta = manifest_df.loc[fid]
                file_fracs.append({
                    "file_id": fid,
                    "fault_class": str(meta["fault_class"]),
                    "frac_removed": frac,
                })

        df_fracs = pd.DataFrame(file_fracs)
        for cls in CLASS_NAMES:
            cls_mean = float(df_fracs[df_fracs["fault_class"] == cls]["frac_removed"].mean())
            power_removed_rows.append({
                "cutoff": str(cutoff),
                "fault_class": cls,
                "mean_frac_removed_by_lp": cls_mean,
            })

    df_power_removed = pd.DataFrame(power_removed_rows)
    df_power_removed.to_csv(RESULTS_DIR / "phase6c_power_removed.csv", index=False)
    print(f"Saved: {RESULTS_DIR / 'phase6c_power_removed.csv'}")

    stage2_time = time.time() - t_stage2
    print(f"Stage 2 completed in {stage2_time:.2f} s\n")

    # -------------------------------------------------------------------------
    # Stage 3: Model Training and Evaluation (6 Cutoffs x 3 Seeds x 7 Splits)
    # -------------------------------------------------------------------------
    t_stage3 = time.time()
    print("=== Stage 3: Random Forest Evaluation across Cutoffs, Seeds, and Splits ===")

    metrics_rows = []
    per_file_rows = []

    total_fits = len(CUTOFFS) * len(SEEDS) * len(SPLIT_NAMES)
    fit_count = 0

    for cutoff in CUTOFFS:
        df_feat = feature_tables[cutoff]
        for seed in SEEDS:
            for s_name in SPLIT_NAMES:
                fit_count += 1
                s_data = splits_json[s_name]
                tr_files = s_data["train_files"]
                te_files = s_data["test_files"]

                tr_df = df_feat[df_feat["file_id"].isin(tr_files)]
                te_df = df_feat[df_feat["file_id"].isin(te_files)]

                X_train = tr_df[ALL_FEATURES].values
                y_train = tr_df["fault_class"].values
                X_test = te_df[ALL_FEATURES].values
                y_test = te_df["fault_class"].values

                # 300-tree class-balanced Random Forest; evaluate unweighted macro-F1 and per-class recall
                clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
                rec_norm = float(recall_score(y_test == "Normal", y_pred == "Normal", zero_division=0))
                rec_ir = float(recall_score(y_test == "Inner Race", y_pred == "Inner Race", zero_division=0))
                rec_ball = float(recall_score(y_test == "Ball", y_pred == "Ball", zero_division=0))
                rec_or = float(recall_score(y_test == "Outer Race", y_pred == "Outer Race", zero_division=0))

                split_group = "lolo" if s_name.startswith("lolo") else "sev"

                metrics_rows.append({
                    "cutoff": str(cutoff),
                    "seed": seed,
                    "split": s_name,
                    "split_group": split_group,
                    "macro_f1": macro_f1,
                    "recall_Normal": rec_norm,
                    "recall_Inner Race": rec_ir,
                    "recall_Ball": rec_ball,
                    "recall_Outer Race": rec_or,
                })

                if seed == 42:
                    for fid in te_files:
                        meta = manifest_df.loc[fid]
                        file_mask = (te_df["file_id"].values == fid)
                        n_win = int(np.sum(file_mask))
                        n_corr = int(np.sum((y_test[file_mask] == y_pred[file_mask])))
                        acc = float(n_corr / n_win) if n_win > 0 else 0.0
                        per_file_rows.append({
                            "cutoff": str(cutoff),
                            "split": s_name,
                            "file_id": int(fid),
                            "fault_class": str(meta["fault_class"]),
                            "severity": str(meta["severity"]) if pd.notna(meta["severity"]) else "none",
                            "load_hp": int(meta["load_hp"]),
                            "n_windows": n_win,
                            "n_correct": n_corr,
                            "window_accuracy": acc,
                        })

    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(RESULTS_DIR / "phase6c_cutoff_metrics.csv", index=False)
    print(f"Saved: {RESULTS_DIR / 'phase6c_cutoff_metrics.csv'} ({len(df_metrics)} rows)")

    df_per_file = pd.DataFrame(per_file_rows)
    df_per_file.to_csv(RESULTS_DIR / "phase6c_per_file.csv", index=False)
    print(f"Saved: {RESULTS_DIR / 'phase6c_per_file.csv'} ({len(df_per_file)} rows)")

    # -------------------------------------------------------------------------
    # Fail-Fast Check (c): Parity with Phase 4 Ablation (14 Comparisons)
    # -------------------------------------------------------------------------
    print("\n--- Fail-Fast Check (c): Parity with Phase 4 Ablation (Seed 42) ---")
    df_s42 = df_metrics[df_metrics["seed"] == 42]

    # Map ablation reference rows
    abl_5 = df_ablation_ref[df_ablation_ref["ablation_id"] == "5_full_lp3500"].set_index("split")["macro_f1"].to_dict()
    abl_6 = df_ablation_ref[df_ablation_ref["ablation_id"] == "6_full_nolp"].set_index("split")["macro_f1"].to_dict()

    print(f"{'Cutoff':<8} | {'Split':<10} | {'Calculated F1':<14} | {'Phase 4 Ablation':<16} | {'Diff':<10} | {'Status'}")
    print("-" * 75)

    check_c_passed = True
    for s_name in SPLIT_NAMES:
        # C = 3500 vs 5_full_lp3500
        calc_f1_3500 = float(df_s42[(df_s42["cutoff"] == "3500") & (df_s42["split"] == s_name)]["macro_f1"].iloc[0])
        ref_f1_3500 = float(abl_5[s_name])
        diff_3500 = abs(calc_f1_3500 - ref_f1_3500)
        status_3500 = "PASS" if diff_3500 <= 0.001 else "FAIL"
        print(f"{'3500':<8} | {s_name:<10} | {calc_f1_3500:<14.6f} | {ref_f1_3500:<16.6f} | {diff_3500:<10.6f} | {status_3500}")
        if diff_3500 > 0.001:
            check_c_passed = False

    for s_name in SPLIT_NAMES:
        # 'none' vs 6_full_nolp
        calc_f1_none = float(df_s42[(df_s42["cutoff"] == "none") & (df_s42["split"] == s_name)]["macro_f1"].iloc[0])
        ref_f1_none = float(abl_6[s_name])
        diff_none = abs(calc_f1_none - ref_f1_none)
        status_none = "PASS" if diff_none <= 0.001 else "FAIL"
        print(f"{'none':<8} | {s_name:<10} | {calc_f1_none:<14.6f} | {ref_f1_none:<16.6f} | {diff_none:<10.6f} | {status_none}")
        if diff_none > 0.001:
            check_c_passed = False

    assert check_c_passed, "Check (c) failed: one or more splits differ by > 0.001 from phase4_ablation.csv"
    print("Check (c) PASSED: all 14 comparisons match phase4_ablation.csv within 0.001.")

    stage3_time = time.time() - t_stage3
    print(f"Stage 3 completed in {stage3_time:.2f} s\n")

    # -------------------------------------------------------------------------
    # Stage 4: Generate Summary Markdown and Diagnostic Plot
    # -------------------------------------------------------------------------
    t_stage4 = time.time()
    print("=== Stage 4: Generate Summary Report and Plot ===")

    # Table 1: Macro-F1 per split for seed 42, LOLO mean +- std, Sev mean +- std, mean seed std
    t1_rows = []
    lolo_splits = ["lolo_0", "lolo_1", "lolo_2", "lolo_3"]
    sev_splits = ["sev_0.007", "sev_0.014", "sev_0.021"]

    for cutoff in CUTOFFS:
        c_str = str(cutoff)
        sub_s42 = df_s42[df_s42["cutoff"] == c_str].set_index("split")

        lolo_scores = [float(sub_s42.loc[s, "macro_f1"]) for s in lolo_splits]
        sev_scores = [float(sub_s42.loc[s, "macro_f1"]) for s in sev_splits]

        # Population mean and population std (ddof=0) across splits; seed std uses ddof=0 across SEEDS=[42, 0, 1]
        lolo_mean = float(np.mean(lolo_scores))
        lolo_std = float(np.std(lolo_scores, ddof=0))
        sev_mean = float(np.mean(sev_scores))
        sev_std = float(np.std(sev_scores, ddof=0))

        # Seed std across splits
        sub_all_seeds = df_metrics[df_metrics["cutoff"] == c_str]
        seed_stds_all = []
        for s in SPLIT_NAMES:
            s_f1s = sub_all_seeds[sub_all_seeds["split"] == s]["macro_f1"].values
            seed_stds_all.append(float(np.std(s_f1s, ddof=0)))
        mean_seed_std = float(np.mean(seed_stds_all))

        t1_rows.append({
            "cutoff": c_str,
            "lolo_0": lolo_scores[0],
            "lolo_1": lolo_scores[1],
            "lolo_2": lolo_scores[2],
            "lolo_3": lolo_scores[3],
            "lolo_mean": lolo_mean,
            "lolo_std": lolo_std,
            "sev_0.007": sev_scores[0],
            "sev_0.014": sev_scores[1],
            "sev_0.021": sev_scores[2],
            "sev_mean": sev_mean,
            "sev_std": sev_std,
            "mean_seed_std": mean_seed_std,
        })

    # Table 2: Per-class recall (LOLO mean, Severity mean, sev_0.014) per cutoff (seed 42)
    t2_rows = []
    for cutoff in CUTOFFS:
        c_str = str(cutoff)
        sub_s42 = df_s42[df_s42["cutoff"] == c_str]

        for cls in CLASS_NAMES:
            col_rec = f"recall_{cls}"
            lolo_mean_rec = float(sub_s42[sub_s42["split"].isin(lolo_splits)][col_rec].mean())
            sev_mean_rec = float(sub_s42[sub_s42["split"].isin(sev_splits)][col_rec].mean())
            s014_rec = float(sub_s42[sub_s42["split"] == "sev_0.014"][col_rec].iloc[0])

            t2_rows.append({
                "cutoff": c_str,
                "fault_class": cls,
                "lolo_mean_recall": lolo_mean_rec,
                "sev_mean_recall": sev_mean_rec,
                "sev_0.014_recall": s014_rec,
            })

    # Table 3: Mean power removed per class and cutoff
    t3_pivot = df_power_removed.pivot(index="cutoff", columns="fault_class", values="mean_frac_removed_by_lp")
    # Reindex cutoffs in order
    t3_pivot = t3_pivot.reindex([str(c) for c in CUTOFFS])

    # Table 4: Window accuracy of severity 0.014 files in sev_0.014 and lolo_3
    # 0.014 files: Inner Race (169, 170, 171, 172), Outer Race (197, 198, 199, 200)
    t4_target_files = [169, 170, 171, 172, 197, 198, 199, 200]
    sub_per_file = df_per_file[df_per_file["file_id"].isin(t4_target_files)]
    # Target combinations: (sev_0.014, all 8 files) and (lolo_3, files 172 and 200)
    sub_t4 = sub_per_file[
        ((sub_per_file["split"] == "sev_0.014")) |
        ((sub_per_file["split"] == "lolo_3") & (sub_per_file["file_id"].isin([172, 200])))
    ].copy()

    # Build Markdown Content
    summary_md = []
    summary_md.append("# Phase 6C: Low-Pass Anti-Aliasing Cutoff Sensitivity Report\n")
    summary_md.append("> [!NOTE]")
    summary_md.append("> **Caveats:**")
    summary_md.append("> - Evaluated strictly with Random Forest using fixed pilot settings (300 trees, `class_weight=\"balanced\"`, seeds [42, 0, 1]).")
    summary_md.append("> - The envelope band (1500–3000 Hz) and the 3500 Hz cutoff were chosen together in Phase 2, so 3500 may have a home advantage.")
    summary_md.append("> - At $C = 3000\\text{ Hz}$, the filter also attenuates the upper edge of the 1500–3000 Hz envelope demodulation band.")
    summary_md.append("> - The cross-severity results rest on 3 splits with large spreads.")
    summary_md.append("> - Standard deviations are population standard deviations (`ddof=0`), consistent with Phase 4 and Phase 5 summaries.\n")

    summary_md.append("## Table 1: Macro-F1 across Outer Splits per Cutoff (Seed 42)")
    summary_md.append("| Cutoff | lolo_0 | lolo_1 | lolo_2 | lolo_3 | LOLO Mean ± Std | sev_0.007 | sev_0.014 | sev_0.021 | Severity Mean ± Std | Mean Seed Std |")
    summary_md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in t1_rows:
        summary_md.append(
            f"| **{r['cutoff']}** | {r['lolo_0']:.4f} | {r['lolo_1']:.4f} | {r['lolo_2']:.4f} | {r['lolo_3']:.4f} | "
            f"{r['lolo_mean']:.4f} ± {r['lolo_std']:.4f} | {r['sev_0.007']:.4f} | {r['sev_0.014']:.4f} | {r['sev_0.021']:.4f} | "
            f"{r['sev_mean']:.4f} ± {r['sev_std']:.4f} | {r['mean_seed_std']:.4f} |"
        )
    summary_md.append("")

    summary_md.append("## Table 2: Per-Class Recall per Cutoff (Seed 42)")
    summary_md.append("| Cutoff | Fault Class | LOLO Mean Recall | Severity Mean Recall | sev_0.014 Recall |")
    summary_md.append("| :--- | :--- | :--- | :--- | :--- |")
    for r in t2_rows:
        summary_md.append(f"| **{r['cutoff']}** | {r['fault_class']} | {r['lolo_mean_recall']:.4f} | {r['sev_mean_recall']:.4f} | {r['sev_0.014_recall']:.4f} |")
    summary_md.append("")

    summary_md.append("## Table 3: Mean Fraction of Power Removed by Low-Pass per Fault Class")
    summary_md.append("| Cutoff | Ball | Inner Race | Normal | Outer Race |")
    summary_md.append("| :--- | :--- | :--- | :--- | :--- |")
    for c in CUTOFFS:
        c_str = str(c)
        summary_md.append(
            f"| **{c_str}** | {t3_pivot.loc[c_str, 'Ball']:.4f} | {t3_pivot.loc[c_str, 'Inner Race']:.4f} | "
            f"{t3_pivot.loc[c_str, 'Normal']:.4f} | {t3_pivot.loc[c_str, 'Outer Race']:.4f} |"
        )
    summary_md.append("")

    summary_md.append("## Table 4: Window Accuracy for Severity 0.014\" Files in sev_0.014 and lolo_3 (Seed 42)")
    summary_md.append("| Split | File ID | Fault Class | Load (HP) | Cutoff 3000 | Cutoff 3500 | Cutoff 4000 | Cutoff 4500 | Cutoff 5000 | Cutoff none |")
    summary_md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    # Reorganize table 4 by (split, file_id)
    t4_items = [
        ("sev_0.014", 169, "Inner Race", 0),
        ("sev_0.014", 170, "Inner Race", 1),
        ("sev_0.014", 171, "Inner Race", 2),
        ("sev_0.014", 172, "Inner Race", 3),
        ("sev_0.014", 197, "Outer Race", 0),
        ("sev_0.014", 198, "Outer Race", 1),
        ("sev_0.014", 199, "Outer Race", 2),
        ("sev_0.014", 200, "Outer Race", 3),
        ("lolo_3", 172, "Inner Race", 3),
        ("lolo_3", 200, "Outer Race", 3),
    ]

    for s_id, fid, f_cls, load_hp in t4_items:
        accs = []
        for c in CUTOFFS:
            c_str = str(c)
            match = sub_t4[(sub_t4["cutoff"] == c_str) & (sub_t4["split"] == s_id) & (sub_t4["file_id"] == fid)]
            acc_val = float(match["window_accuracy"].iloc[0]) if len(match) > 0 else 0.0
            accs.append(f"{acc_val:.4f}")
        summary_md.append(f"| {s_id} | {fid} | {f_cls} | {load_hp} | {' | '.join(accs)} |")
    summary_md.append("")

    with open(RESULTS_DIR / "phase6c_cutoff_summary.md", "w") as f:
        f.write("\n".join(summary_md))
    print(f"Saved: {RESULTS_DIR / 'phase6c_cutoff_summary.md'}")

    # Plot: results/fig_phase6c_cutoff_vs_macro_f1.png
    print("Generating figure results/fig_phase6c_cutoff_vs_macro_f1.png...")
    x_positions = np.arange(len(CUTOFFS))
    x_labels = [str(c) for c in CUTOFFS]

    # LOLO mean and error bar
    lolo_means = [r["lolo_mean"] for r in t1_rows]
    lolo_seed_stds = []
    for c in CUTOFFS:
        c_str = str(c)
        sub = df_metrics[(df_metrics["cutoff"] == c_str) & (df_metrics["split_group"] == "lolo")]
        # seed std per split, averaged
        stds = [float(np.std(sub[sub["split"] == s]["macro_f1"], ddof=0)) for s in lolo_splits]
        lolo_seed_stds.append(float(np.mean(stds)))

    # Severity splits
    sev_007_vals = [r["sev_0.007"] for r in t1_rows]
    sev_007_stds = [float(np.std(df_metrics[(df_metrics["cutoff"] == str(c)) & (df_metrics["split"] == "sev_0.007")]["macro_f1"], ddof=0)) for c in CUTOFFS]

    sev_014_vals = [r["sev_0.014"] for r in t1_rows]
    sev_014_stds = [float(np.std(df_metrics[(df_metrics["cutoff"] == str(c)) & (df_metrics["split"] == "sev_0.014")]["macro_f1"], ddof=0)) for c in CUTOFFS]

    sev_021_vals = [r["sev_0.021"] for r in t1_rows]
    sev_021_stds = [float(np.std(df_metrics[(df_metrics["cutoff"] == str(c)) & (df_metrics["split"] == "sev_0.021")]["macro_f1"], ddof=0)) for c in CUTOFFS]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    ax.errorbar(x_positions, lolo_means, yerr=lolo_seed_stds, fmt="-o", color="#1f77b4", label="LOLO Mean (4 splits)", capsize=4, linewidth=2)
    ax.errorbar(x_positions, sev_007_vals, yerr=sev_007_stds, fmt="--s", color="#2ca02c", label="sev_0.007", capsize=4, alpha=0.85)
    ax.errorbar(x_positions, sev_014_vals, yerr=sev_014_stds, fmt="--^", color="#d62728", label="sev_0.014", capsize=4, alpha=0.85)
    ax.errorbar(x_positions, sev_021_vals, yerr=sev_021_stds, fmt="--d", color="#9467bd", label="sev_0.021", capsize=4, alpha=0.85)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_xlabel("Low-Pass Filter Cutoff Frequency (Hz)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Macro-F1 Score", fontsize=12, fontweight="bold")
    ax.set_title("Random Forest Macro-F1 vs Low-Pass Anti-Aliasing Cutoff", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, loc="lower right", fontsize=10)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_phase6c_cutoff_vs_macro_f1.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {RESULTS_DIR / 'fig_phase6c_cutoff_vs_macro_f1.png'}")

    stage4_time = time.time() - t_stage4
    print(f"Stage 4 completed in {stage4_time:.2f} s\n")

    total_time = time.time() - t_start
    print(f"=== All stages finished successfully in {total_time:.2f} s ===")


if __name__ == "__main__":
    main()
