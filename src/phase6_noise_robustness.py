"""Phase 6: Noise Robustness & Additive Gaussian Noise Stress-Testing.

Executes:
1. Imports, headless matplotlib configuration, and data loading
2. Deterministic noise model: Variant A (per-file SNR) & Variant B (fixed noise level from fault median)
3. Fail-Fast checks:
   (a) Noise power check (|10*log10(P_meas/noise_var)| <= 0.2 dB)
   (b) Clean feature parity check (max abs diff < 1e-9 against phase2_features_lp3500.csv)
   (c) Clean model parity check (Macro-F1 matches phase4_metrics.csv within 0.001 across all 21 models)
4. Pre-LP and post-LP effective SNR calculation and CSV exports
5. Feature extraction across 24 noisy passes with disk caching in data/processed/phase6_cache/features/
6. Test fold inference across all 21 clean-trained models x 13 conditions x 2 variants
7. Feature degradation vs. noise analysis
8. Generation of publication figures and results/phase6_summary.md
"""
# Reads: data/processed/phase1_signals_12k.pkl, results/phase1_file_manifest.csv, data/processed/phase2_features_lp3500.csv, data/processed/phase3_splits.json, data/processed/phase4_best_params.json, results/phase4_metrics.csv, cache data/processed/phase6_cache/features/*.pkl
# Writes: cache data/processed/phase6_cache/features/*.pkl, results/phase6_effective_snr_variant_a.csv, results/phase6_effective_snr_variant_b.csv, results/phase6_metrics.csv, results/phase6_envelope_vs_snr.csv, results/fig_phase6_macro_f1_vs_snr.png, results/fig_phase6_envelope_vs_snr.png, results/phase6_summary.md

import json
import os
from pathlib import Path
import pickle
import sys
import time
import zlib

import numpy as np
import pandas as pd
import scipy.signal
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# Headless matplotlib configuration
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add src to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from phase2_features import extract_window_features, FEATURE_GROUPS, ALL_FEATURES, compute_features_for_dataset
from phase4_train_eval import CorrelationFilter, CLASS_NAMES, CLASS_TO_IDX, SPLIT_NAMES, to_markdown_table

np.random.seed(42)

SNR_ORDER = ["clean", "30", "20", "10", "5"]
NUMERIC_SNRS = [30, 20, 10, 5]
REPETITIONS = [0, 1, 2]
VARIANTS = ["variant_A", "variant_B"]
MODELS = ["svm", "rf", "xgb"]
NORMAL_FIDS = {97, 98, 99, 100}


def generate_noise_for_file(variant: str, snr_db: float, rep: int, fid: int, n_samples: int, p_signal: float, p_ref: float) -> tuple[np.ndarray, float]:
    """Deterministically generate white Gaussian noise for a single file."""
    if variant == "variant_A":
        # Variant A: noise_var = P_file / 10^(snr_db / 10)
        noise_var = p_signal / (10.0 ** (snr_db / 10.0))
    elif variant == "variant_B":
        # Variant B: fixed noise_var = P_ref / 10^(snr_db / 10)
        noise_var = p_ref / (10.0 ** (snr_db / 10.0))
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Deterministic 32-bit CRC32 seed per (variant, snr_db, rep, fid); N(0, sqrt(noise_var)) Gaussian sample
    seed = zlib.crc32(f"{variant}|{int(snr_db)}|{int(rep)}|{int(fid)}".encode("utf-8")) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.sqrt(noise_var), size=n_samples)
    return noise, noise_var


def main():
    t_start_total = time.time()
    data_dir = root_dir / "data" / "processed"
    results_dir = root_dir / "results"
    cache_dir = data_dir / "phase6_cache" / "features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 6: NOISE ROBUSTNESS BENCHMARK")
    print("=" * 80)

    # 1. Load data
    print("\n[1/8] Loading inputs...")
    with open(data_dir / "phase1_signals_12k.pkl", "rb") as f:
        signals = pickle.load(f)
    assert len(signals) == 40, f"Expected 40 signals, got {len(signals)}"

    manifest_df = pd.read_csv(results_dir / "phase1_file_manifest.csv", keep_default_na=False)
    manifest_df["file_id"] = manifest_df["file_id"].astype(int)
    manifest_df = manifest_df.set_index("file_id")

    df_lp3500_ref = pd.read_csv(data_dir / "phase2_features_lp3500.csv", keep_default_na=False)
    assert len(df_lp3500_ref) == 4643, f"Expected 4643 rows in df_lp3500, got {len(df_lp3500_ref)}"

    with open(data_dir / "phase3_splits.json", "r") as f:
        splits_json = json.load(f)

    with open(data_dir / "phase4_best_params.json", "r") as f:
        best_params = json.load(f)

    df_p4_metrics = pd.read_csv(results_dir / "phase4_metrics.csv")

    # 2. Pre-compute powers & filter setup
    print("\n[2/8] Computing signal powers & filter setups...")
    # 8th-order Butterworth low-pass filter at 3500 Hz (fs=12000 Hz)
    lp_sos = scipy.signal.butter(8, 3500, btype="lowpass", fs=12000, output="sos")

    P_file = {}
    P_file_post_lp = {}
    for fid in sorted(signals.keys()):
        raw_x = signals[fid]["x"]
        # AC signal power pre-LP and post zero-phase LP: mean((x - mean(x))^2)
        P_file[fid] = float(np.mean((raw_x - np.mean(raw_x)) ** 2))
        x_clean_lp = scipy.signal.sosfiltfilt(lp_sos, raw_x)
        P_file_post_lp[fid] = float(np.mean((x_clean_lp - np.mean(x_clean_lp)) ** 2))

    fault_fids = [fid for fid in sorted(signals.keys()) if fid not in NORMAL_FIDS]
    assert len(fault_fids) == 36, f"Expected 36 fault files, got {len(fault_fids)}"
    # Fixed reference power P_ref = median(P_file) across all 36 fault files (excluding Normal {97,98,99,100})
    P_ref = float(np.median([P_file[fid] for fid in fault_fids]))
    print(f"  P_ref (median fault power pre-LP): {P_ref:.6f} (RMS: {np.sqrt(P_ref):.4f})")

    # 3. Fail-Fast Check (b): Clean feature parity
    print("\n[3/8] Running Fail-Fast Check (b): Clean feature parity...")
    t0_check_b = time.time()
    clean_signals_dict = {fid: {"x": signals[fid]["x"]} for fid in sorted(signals.keys())}
    df_clean_computed, _ = compute_features_for_dataset(clean_signals_dict, manifest_df, use_lp3500=True)
    assert len(df_clean_computed) == 4643, f"Check (b) failed: expected 4643 rows, got {len(df_clean_computed)}"

    df_clean_sorted = df_clean_computed.sort_values(["file_id", "window_index"]).reset_index(drop=True)
    df_ref_sorted = df_lp3500_ref.sort_values(["file_id", "window_index"]).reset_index(drop=True)

    max_diff_feats = 0.0
    for feat in ALL_FEATURES:
        diff = float(np.max(np.abs(df_clean_sorted[feat].values - df_ref_sorted[feat].values)))
        if diff > max_diff_feats:
            max_diff_feats = diff
    print(f"  Check (b) Max absolute difference across all 21 features: {max_diff_feats:.4e} (runtime: {time.time() - t0_check_b:.2f}s)")
    assert max_diff_feats < 1e-9, f"Check (b) failed: max diff {max_diff_feats} >= 1e-9"
    print("  Check (b) PASSED.")

    # 4. Fail-Fast Check (c): Clean model parity & in-memory fitting
    print("\n[4/8] Running Fail-Fast Check (c): Clean model parity across 21 estimators...")
    t0_check_c = time.time()
    models_in_memory = {}
    clean_baseline_metrics = {}

    for s_name in SPLIT_NAMES:
        s_data = splits_json[s_name]
        tr_files = s_data["train_files"]
        te_files = s_data["test_files"]

        tr_df = df_lp3500_ref[df_lp3500_ref["file_id"].isin(tr_files)]
        te_df = df_lp3500_ref[df_lp3500_ref["file_id"].isin(te_files)]

        X_tr = tr_df[ALL_FEATURES].values
        y_tr = np.array([CLASS_TO_IDX[c] for c in tr_df["fault_class"]], dtype=int)
        X_te = te_df[ALL_FEATURES].values
        y_te = np.array([CLASS_TO_IDX[c] for c in te_df["fault_class"]], dtype=int)

        # 1. SVM
        p_svm = best_params["svm"][s_name]["best_params"]
        svm_est = Pipeline([
            ("corr", CorrelationFilter(0.95)),
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=p_svm["C"], gamma=p_svm["gamma"], class_weight="balanced", random_state=42))
        ])
        svm_est.fit(X_tr, y_tr)
        y_pred_svm = svm_est.predict(X_te)
        f1_svm = float(f1_score(y_te, y_pred_svm, average="macro", zero_division=0.0))
        ref_f1_svm = float(df_p4_metrics[(df_p4_metrics["split"] == s_name) & (df_p4_metrics["model"] == "svm") & (df_p4_metrics["class"] == "macro")]["f1"].iloc[0])
        assert abs(f1_svm - ref_f1_svm) < 0.001, f"SVM f1 mismatch in {s_name}: {f1_svm:.4f} vs {ref_f1_svm:.4f}"
        models_in_memory[("svm", s_name)] = svm_est
        clean_baseline_metrics[("svm", s_name)] = {
            "macro_f1": f1_svm,
            "recalls": recall_score(y_te, y_pred_svm, labels=[0, 1, 2, 3], average=None, zero_division=0.0)
        }

        # 2. RF
        p_rf = best_params["rf"][s_name]["best_params"]
        rf_est = RandomForestClassifier(n_estimators=300, min_samples_leaf=p_rf["min_samples_leaf"], max_depth=p_rf["max_depth"], class_weight="balanced", random_state=42, n_jobs=-1)
        rf_est.fit(X_tr, y_tr)
        y_pred_rf = rf_est.predict(X_te)
        f1_rf = float(f1_score(y_te, y_pred_rf, average="macro", zero_division=0.0))
        ref_f1_rf = float(df_p4_metrics[(df_p4_metrics["split"] == s_name) & (df_p4_metrics["model"] == "rf") & (df_p4_metrics["class"] == "macro")]["f1"].iloc[0])
        assert abs(f1_rf - ref_f1_rf) < 0.001, f"RF f1 mismatch in {s_name}: {f1_rf:.4f} vs {ref_f1_rf:.4f}"
        models_in_memory[("rf", s_name)] = rf_est
        clean_baseline_metrics[("rf", s_name)] = {
            "macro_f1": f1_rf,
            "recalls": recall_score(y_te, y_pred_rf, labels=[0, 1, 2, 3], average=None, zero_division=0.0)
        }

        # 3. XGBoost
        p_xgb = best_params["xgb"][s_name]["best_params"]
        sw = compute_sample_weight("balanced", y_tr)
        xgb_est = XGBClassifier(n_estimators=200, max_depth=p_xgb["max_depth"], learning_rate=p_xgb["learning_rate"], tree_method="hist", eval_metric="mlogloss", random_state=42, n_jobs=-1)
        xgb_est.fit(X_tr, y_tr, sample_weight=sw)
        y_pred_xgb = xgb_est.predict(X_te)
        f1_xgb = float(f1_score(y_te, y_pred_xgb, average="macro", zero_division=0.0))
        ref_f1_xgb = float(df_p4_metrics[(df_p4_metrics["split"] == s_name) & (df_p4_metrics["model"] == "xgb") & (df_p4_metrics["class"] == "macro")]["f1"].iloc[0])
        assert abs(f1_xgb - ref_f1_xgb) < 0.001, f"XGB f1 mismatch in {s_name}: {f1_xgb:.4f} vs {ref_f1_xgb:.4f}"
        models_in_memory[("xgb", s_name)] = xgb_est
        clean_baseline_metrics[("xgb", s_name)] = {
            "macro_f1": f1_xgb,
            "recalls": recall_score(y_te, y_pred_xgb, labels=[0, 1, 2, 3], average=None, zero_division=0.0)
        }

    print(f"  Check (c) PASSED across all 21 models (runtime: {time.time() - t0_check_c:.2f}s). Models preserved in memory.")

    # 5. Effective SNR Tables (Variant A & Variant B) & Fail-Fast Check (a)
    print("\n[5/8] Computing Effective SNR Tables & running Fail-Fast Check (a)...")
    snr_tables = {"variant_A": [], "variant_B": []}

    for variant in VARIANTS:
        for snr_db in NUMERIC_SNRS:
            for fid in sorted(signals.keys()):
                meta = manifest_df.loc[fid]
                p_file_val = P_file[fid]
                p_ref_val = P_ref
                n_samples = len(signals[fid]["x"])

                if variant == "variant_A":
                    noise_var = p_file_val / (10.0 ** (snr_db / 10.0))
                    target_eff_pre = float(snr_db)
                else:
                    noise_var = p_ref_val / (10.0 ** (snr_db / 10.0))
                    # Power ratio in dB for Variant B target SNR: 10*log10(P_file / noise_var)
                    target_eff_pre = float(10.0 * np.log10(p_file_val / noise_var))

                meas_pre_reps = []
                meas_post_reps = []

                for rep in REPETITIONS:
                    noise, n_var = generate_noise_for_file(variant, snr_db, rep, fid, n_samples, p_file_val, p_ref_val)
                    meas_var = float(np.mean(noise ** 2))

                    # Fail-Fast Check (a): Noise power check
                    # Assert |10*log10(mean(noise^2) / target_var)| <= 0.2 dB
                    ratio_db = 10.0 * np.log10(meas_var / n_var)
                    assert abs(ratio_db) <= 0.2, (
                        f"Check (a) failed for {variant}, SNR={snr_db}, rep={rep}, fid={fid}: ratio_db={ratio_db:.4f} > 0.2"
                    )

                    # Zero-phase filter applied to noise before measuring post-LP noise AC power
                    noise_lp = scipy.signal.sosfiltfilt(lp_sos, noise)
                    meas_post_var = float(np.mean((noise_lp - np.mean(noise_lp)) ** 2))

                    # Empirical pre-LP and post-LP SNR (dB) via 10*log10(signal_power / noise_power)
                    meas_pre_reps.append(10.0 * np.log10(p_file_val / meas_var))
                    meas_post_reps.append(10.0 * np.log10(P_file_post_lp[fid] / meas_post_var))

                snr_tables[variant].append({
                    "nominal_snr_db": int(snr_db),
                    "file_id": int(fid),
                    "fault_class": str(meta["fault_class"]),
                    "severity": str(meta["severity"]) if pd.notna(meta["severity"]) else "none",
                    "load_hp": int(meta["load_hp"]),
                    "p_file": p_file_val,
                    "p_ref": p_ref_val,
                    "target_effective_snr_pre_lp_db": target_eff_pre,
                    "measured_effective_snr_pre_lp_db": float(np.mean(meas_pre_reps)),
                    "measured_effective_snr_post_lp_db": float(np.mean(meas_post_reps)),
                })

    print("  Check (a) PASSED for all 40 files x 4 SNRs x 3 reps x 2 variants (960 checks).")

    df_eff_a = pd.DataFrame(snr_tables["variant_A"])
    df_eff_b = pd.DataFrame(snr_tables["variant_B"])

    df_eff_a.to_csv(results_dir / "phase6_effective_snr_variant_a.csv", index=False)
    df_eff_b.to_csv(results_dir / "phase6_effective_snr_variant_b.csv", index=False)
    print(f"  Saved: {results_dir / 'phase6_effective_snr_variant_a.csv'} ({len(df_eff_a)} rows)")
    print(f"  Saved: {results_dir / 'phase6_effective_snr_variant_b.csv'} ({len(df_eff_b)} rows)")

    # 6. Feature Extraction Loop with Caching
    print("\n[6/8] Extracting features for 24 noisy passes (with caching)...")
    noisy_features = {}
    pass_times = []

    total_passes = len(VARIANTS) * len(NUMERIC_SNRS) * len(REPETITIONS)
    current_pass = 0

    for variant in VARIANTS:
        for snr_db in NUMERIC_SNRS:
            for rep in REPETITIONS:
                current_pass += 1
                cache_file = cache_dir / f"{variant}__snr{snr_db}__rep{rep}.pkl"

                if cache_file.exists():
                    with open(cache_file, "rb") as f:
                        df_noisy = pickle.load(f)
                    noisy_features[(variant, snr_db, rep)] = df_noisy
                else:
                    t_pass_start = time.time()
                    noisy_signals_dict = {}
                    for fid in sorted(signals.keys()):
                        noise, _ = generate_noise_for_file(
                            variant, snr_db, rep, fid, len(signals[fid]["x"]), P_file[fid], P_ref
                        )
                        noisy_signals_dict[fid] = {"x": signals[fid]["x"] + noise}

                    df_noisy, _ = compute_features_for_dataset(noisy_signals_dict, manifest_df, use_lp3500=True)
                    t_pass_elapsed = time.time() - t_pass_start
                    pass_times.append(t_pass_elapsed)

                    with open(cache_file, "wb") as f:
                        pickle.dump(df_noisy, f, protocol=pickle.HIGHEST_PROTOCOL)
                    noisy_features[(variant, snr_db, rep)] = df_noisy

                    if current_pass == 1:
                        est_total_s = t_pass_elapsed * total_passes
                        print(f"  Pass 1/24 ({variant}, {snr_db}dB, rep={rep}) extracted in {t_pass_elapsed:.2f}s. Estimated total feature extraction: {est_total_s:.1f}s ({est_total_s / 60:.2f} min)")

    print(f"  All 24 passes ready in memory and cached in {cache_dir}.")

    # 7. Model Evaluation Loop (Clean + Noisy)
    print("\n[7/8] Evaluating 21 clean-trained models across all conditions...")
    metrics_rows = []

    # Clean evaluations (rep = 0, reused for both variants)
    for variant in VARIANTS:
        for (model_name, s_name), est in models_in_memory.items():
            base_m = clean_baseline_metrics[(model_name, s_name)]
            rec = base_m["recalls"]
            s_grp = "lolo" if "lolo" in s_name else "sev"
            metrics_rows.append({
                "variant": variant,
                "model": model_name,
                "split": s_name,
                "split_group": s_grp,
                "snr_db": "clean",
                "repetition": 0,
                "macro_f1": base_m["macro_f1"],
                "recall_Normal": float(rec[0]),
                "recall_Inner Race": float(rec[1]),
                "recall_Ball": float(rec[2]),
                "recall_Outer Race": float(rec[3]),
            })

    # Noisy evaluations
    for variant in VARIANTS:
        for snr_db in NUMERIC_SNRS:
            for rep in REPETITIONS:
                df_noisy = noisy_features[(variant, snr_db, rep)]

                for (model_name, s_name), est in models_in_memory.items():
                    te_files = splits_json[s_name]["test_files"]
                    te_df = df_noisy[df_noisy["file_id"].isin(te_files)]

                    X_te = te_df[ALL_FEATURES].values
                    y_te = np.array([CLASS_TO_IDX[c] for c in te_df["fault_class"]], dtype=int)

                    y_pred = est.predict(X_te)
                    f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0.0))
                    rec = recall_score(y_te, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0.0)

                    s_grp = "lolo" if "lolo" in s_name else "sev"
                    metrics_rows.append({
                        "variant": variant,
                        "model": model_name,
                        "split": s_name,
                        "split_group": s_grp,
                        "snr_db": str(snr_db),
                        "repetition": int(rep),
                        "macro_f1": f1,
                        "recall_Normal": float(rec[0]),
                        "recall_Inner Race": float(rec[1]),
                        "recall_Ball": float(rec[2]),
                        "recall_Outer Race": float(rec[3]),
                    })

    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(results_dir / "phase6_metrics.csv", index=False)
    print(f"  Saved: {results_dir / 'phase6_metrics.csv'} ({len(df_metrics)} rows)")

    # 8. Feature Degradation vs Noise
    print("\n[8/8] Computing feature degradation vs. SNR...")
    feat_deg_rows = []
    track_feats = ["rms", "kurtosis", "spectral_centroid", "env_snr_bpfo", "env_snr_2bpfo", "env_snr_bpfi", "env_snr_2bpfi", "env_snr_bsf", "env_snr_2bsf"]

    for variant in VARIANTS:
        # Clean
        for cls in CLASS_NAMES:
            sub = df_lp3500_ref[df_lp3500_ref["fault_class"] == cls]
            row_dict = {
                "variant": variant,
                "snr_db": "clean",
                "fault_class": cls,
                "n_windows": len(sub),
            }
            for feat in track_feats:
                row_dict[feat] = float(np.mean(sub[feat].values))
            feat_deg_rows.append(row_dict)

        # Noisy SNRs
        for snr_db in NUMERIC_SNRS:
            for cls in CLASS_NAMES:
                sub_dfs = [noisy_features[(variant, snr_db, rep)] for rep in REPETITIONS]
                cls_windows = pd.concat([df[df["fault_class"] == cls] for df in sub_dfs], ignore_index=True)
                row_dict = {
                    "variant": variant,
                    "snr_db": str(snr_db),
                    "fault_class": cls,
                    "n_windows": len(cls_windows) // len(REPETITIONS),
                }
                for feat in track_feats:
                    row_dict[feat] = float(np.mean(cls_windows[feat].values))
                feat_deg_rows.append(row_dict)

    df_feat_deg = pd.DataFrame(feat_deg_rows)
    df_feat_deg.to_csv(results_dir / "phase6_envelope_vs_snr.csv", index=False)
    print(f"  Saved: {results_dir / 'phase6_envelope_vs_snr.csv'} ({len(df_feat_deg)} rows)")

    # -------------------------------------------------------------------------
    # Plots: results/fig_phase6_macro_f1_vs_snr.png & fig_phase6_envelope_vs_snr.png
    # -------------------------------------------------------------------------
    print("\nGenerating figures...")
    snr_labels = ["Clean", "30 dB", "20 dB", "10 dB", "5 dB"]
    x_positions = np.arange(len(snr_labels))

    # Figure 1: Macro-F1 vs SNR (2x2 panels)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    model_styles = {
        "svm": {"color": "#1f77b4", "marker": "o", "label": "SVM"},
        "rf": {"color": "#2ca02c", "marker": "s", "label": "Random Forest"},
        "xgb": {"color": "#ff7f0e", "marker": "^", "label": "XGBoost"},
    }

    for r_idx, variant in enumerate(VARIANTS):
        for c_idx, s_grp in enumerate(["lolo", "sev"]):
            ax = axes[r_idx, c_idx]
            grp_name = "LOLO Mean (Loads 0-3)" if s_grp == "lolo" else "Cross-Severity Mean"
            v_name = "Variant A (Per-File SNR)" if variant == "variant_A" else "Variant B (Fixed Noise Floor)"
            ax.set_title(f"{v_name} — {grp_name}", fontsize=11, fontweight="bold")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(snr_labels, fontsize=10)
            ax.set_ylabel("Macro-F1", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_ylim(-0.05, 1.05)

            for model_name in MODELS:
                y_means = []
                y_errs = []
                for snr_val in SNR_ORDER:
                    sub = df_metrics[
                        (df_metrics["variant"] == variant) &
                        (df_metrics["split_group"] == s_grp) &
                        (df_metrics["model"] == model_name) &
                        (df_metrics["snr_db"] == snr_val)
                    ]
                    # Average over repetitions per split first
                    rep_means = sub.groupby("split")["macro_f1"].mean().values
                    split_m = float(np.mean(rep_means))
                    # Sample standard deviation across splits using ddof=1
                    split_s = float(np.std(rep_means, ddof=1)) if len(rep_means) > 1 else 0.0
                    y_means.append(split_m)
                    y_errs.append(split_s)

                st = model_styles[model_name]
                ax.errorbar(
                    x_positions, y_means, yerr=y_errs,
                    color=st["color"], marker=st["marker"], label=st["label"],
                    capsize=4, lw=1.8, markersize=6
                )

            if r_idx == 0 and c_idx == 0:
                ax.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    fig.savefig(results_dir / "fig_phase6_macro_f1_vs_snr.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {results_dir / 'fig_phase6_macro_f1_vs_snr.png'}")

    # Figure 2: Envelope Features & Kurtosis vs SNR (2x2 panels)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    cls_styles = {
        "Normal": {"color": "#1f77b4", "marker": "o", "label": "Normal"},
        "Inner Race": {"color": "#ff7f0e", "marker": "s", "label": "Inner Race"},
        "Ball": {"color": "#2ca02c", "marker": "^", "label": "Ball"},
        "Outer Race": {"color": "#d62728", "marker": "D", "label": "Outer Race"},
    }

    matched_env_feat = {
        "Normal": "env_snr_bpfi",
        "Inner Race": "env_snr_bpfi",
        "Ball": "env_snr_bsf",
        "Outer Race": "env_snr_bpfo",
    }

    for c_idx, variant in enumerate(VARIANTS):
        v_name = "Variant A (Per-File SNR)" if variant == "variant_A" else "Variant B (Fixed Noise Floor)"

        # Row 0: Class-Matched Envelope SNR
        ax_env = axes[0, c_idx]
        ax_env.set_title(f"{v_name} — Class-Matched Envelope SNR", fontsize=11, fontweight="bold")
        ax_env.set_xticks(x_positions)
        ax_env.set_xticklabels(snr_labels, fontsize=10)
        ax_env.set_ylabel("Envelope SNR", fontsize=10)
        ax_env.grid(True, linestyle="--", alpha=0.5)

        for cls in CLASS_NAMES:
            feat_name = matched_env_feat[cls]
            vals = []
            for snr_val in SNR_ORDER:
                row = df_feat_deg[
                    (df_feat_deg["variant"] == variant) &
                    (df_feat_deg["fault_class"] == cls) &
                    (df_feat_deg["snr_db"] == snr_val)
                ].iloc[0]
                vals.append(float(row[feat_name]))
            st = cls_styles[cls]
            lbl = f"{cls} ({feat_name})"
            ax_env.plot(x_positions, vals, color=st["color"], marker=st["marker"], label=lbl, lw=1.8, markersize=6)

        ax_env.legend(loc="upper right", fontsize=8)

        # Row 1: Kurtosis
        ax_kurt = axes[1, c_idx]
        ax_kurt.set_title(f"{v_name} — Class-Wise Kurtosis", fontsize=11, fontweight="bold")
        ax_kurt.set_xticks(x_positions)
        ax_kurt.set_xticklabels(snr_labels, fontsize=10)
        ax_kurt.set_ylabel("Kurtosis", fontsize=10)
        ax_kurt.grid(True, linestyle="--", alpha=0.5)

        for cls in CLASS_NAMES:
            vals = []
            for snr_val in SNR_ORDER:
                row = df_feat_deg[
                    (df_feat_deg["variant"] == variant) &
                    (df_feat_deg["fault_class"] == cls) &
                    (df_feat_deg["snr_db"] == snr_val)
                ].iloc[0]
                vals.append(float(row["kurtosis"]))
            st = cls_styles[cls]
            ax_kurt.plot(x_positions, vals, color=st["color"], marker=st["marker"], label=cls, lw=1.8, markersize=6)

        ax_kurt.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(results_dir / "fig_phase6_envelope_vs_snr.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {results_dir / 'fig_phase6_envelope_vs_snr.png'}")

    # -------------------------------------------------------------------------
    # Markdown Summary: results/phase6_summary.md
    # -------------------------------------------------------------------------
    print("\nGenerating results/phase6_summary.md...")

    # Build Tables
    # Table 1: Macro-F1 vs SNR (Variant A & Variant B)
    t1_rows = []
    for variant in VARIANTS:
        for model in MODELS:
            for snr_val in SNR_ORDER:
                row = {"Variant": variant, "Model": model, "SNR": snr_val}
                for s_grp in ["lolo", "sev"]:
                    sub = df_metrics[
                        (df_metrics["variant"] == variant) &
                        (df_metrics["model"] == model) &
                        (df_metrics["split_group"] == s_grp) &
                        (df_metrics["snr_db"] == snr_val)
                    ]
                    splits_in_grp = ["lolo_0", "lolo_1", "lolo_2", "lolo_3"] if s_grp == "lolo" else ["sev_0.007", "sev_0.014", "sev_0.021"]
                    rep_means = []
                    rep_stds = []
                    for sp in splits_in_grp:
                        sp_sub = sub[sub["split"] == sp]["macro_f1"].values
                        rm = float(np.mean(sp_sub))
                        # Sample standard deviation across repetitions/splits using ddof=1
                        rs = float(np.std(sp_sub, ddof=1)) if len(sp_sub) > 1 else 0.0
                        rep_means.append(rm)
                        rep_stds.append(rs)
                        row[sp] = f"{rm:.4f}"

                    split_m = float(np.mean(rep_means))
                    split_s = float(np.std(rep_means, ddof=1)) if len(rep_means) > 1 else 0.0
                    mean_rs = float(np.mean(rep_stds))
                    grp_prefix = "LOLO" if s_grp == "lolo" else "Sev"
                    row[f"{grp_prefix} Mean +- Std"] = f"{split_m:.4f} +- {split_s:.4f}"
                    row[f"{grp_prefix} Rep Std"] = f"{mean_rs:.4f}"
                t1_rows.append(row)

    df_t1 = pd.DataFrame(t1_rows)

    # Table 2: Normal Recall vs SNR
    t2_rows = []
    for variant in VARIANTS:
        for model in MODELS:
            for snr_val in SNR_ORDER:
                row = {"Variant": variant, "Model": model, "SNR": snr_val}
                for s_grp in ["lolo", "sev"]:
                    sub = df_metrics[
                        (df_metrics["variant"] == variant) &
                        (df_metrics["model"] == model) &
                        (df_metrics["split_group"] == s_grp) &
                        (df_metrics["snr_db"] == snr_val)
                    ]
                    splits_in_grp = ["lolo_0", "lolo_1", "lolo_2", "lolo_3"] if s_grp == "lolo" else ["sev_0.007", "sev_0.014", "sev_0.021"]
                    rep_means = []
                    for sp in splits_in_grp:
                        sp_sub = sub[sub["split"] == sp]["recall_Normal"].values
                        rm = float(np.mean(sp_sub))
                        rep_means.append(rm)
                        row[sp] = f"{rm:.4f}"
                    split_m = float(np.mean(rep_means))
                    split_s = float(np.std(rep_means, ddof=1)) if len(rep_means) > 1 else 0.0
                    grp_prefix = "LOLO" if s_grp == "lolo" else "Sev"
                    row[f"{grp_prefix} Recall (Mean +- Std)"] = f"{split_m:.4f} +- {split_s:.4f}"
                t2_rows.append(row)
    df_t2 = pd.DataFrame(t2_rows)

    # Table 3: Effective SNR Table for Variant B
    t3_rows = []
    for snr_db in NUMERIC_SNRS:
        sub_b = df_eff_b[df_eff_b["nominal_snr_db"] == snr_db]
        fault_sub = sub_b[~sub_b["file_id"].isin(NORMAL_FIDS)]

        for fid in [97, 98, 99, 100]:
            r_norm = sub_b[sub_b["file_id"] == fid].iloc[0]
            t3_rows.append({
                "Nominal SNR (dB)": snr_db,
                "Entity / File": f"Normal (fid={fid})",
                "Load": f"{int(r_norm['load_hp'])} HP",
                "Pre-LP Effective SNR (dB)": f"{r_norm['measured_effective_snr_pre_lp_db']:.2f}",
                "Post-LP Effective SNR (dB)": f"{r_norm['measured_effective_snr_post_lp_db']:.2f}",
                "Post vs Pre Shift (dB)": f"{r_norm['measured_effective_snr_post_lp_db'] - r_norm['measured_effective_snr_pre_lp_db']:+.2f}",
            })

        t3_rows.append({
            "Nominal SNR (dB)": snr_db,
            "Entity / File": "Faults Min",
            "Load": "-",
            "Pre-LP Effective SNR (dB)": f"{fault_sub['measured_effective_snr_pre_lp_db'].min():.2f}",
            "Post-LP Effective SNR (dB)": f"{fault_sub['measured_effective_snr_post_lp_db'].min():.2f}",
            "Post vs Pre Shift (dB)": f"{fault_sub['measured_effective_snr_post_lp_db'].min() - fault_sub['measured_effective_snr_pre_lp_db'].min():+.2f}",
        })
        t3_rows.append({
            "Nominal SNR (dB)": snr_db,
            "Entity / File": "Faults Median",
            "Load": "-",
            "Pre-LP Effective SNR (dB)": f"{fault_sub['measured_effective_snr_pre_lp_db'].median():.2f}",
            "Post-LP Effective SNR (dB)": f"{fault_sub['measured_effective_snr_post_lp_db'].median():.2f}",
            "Post vs Pre Shift (dB)": f"{fault_sub['measured_effective_snr_post_lp_db'].median() - fault_sub['measured_effective_snr_pre_lp_db'].median():+.2f}",
        })
        t3_rows.append({
            "Nominal SNR (dB)": snr_db,
            "Entity / File": "Faults Max",
            "Load": "-",
            "Pre-LP Effective SNR (dB)": f"{fault_sub['measured_effective_snr_pre_lp_db'].max():.2f}",
            "Post-LP Effective SNR (dB)": f"{fault_sub['measured_effective_snr_post_lp_db'].max():.2f}",
            "Post vs Pre Shift (dB)": f"{fault_sub['measured_effective_snr_post_lp_db'].max() - fault_sub['measured_effective_snr_pre_lp_db'].max():+.2f}",
        })
    df_t3 = pd.DataFrame(t3_rows)

    # Format Markdown
    summary_md = f"""# Phase 6: Noise Robustness & Additive Gaussian Stress-Testing Summary

> [!NOTE]
> **Evaluation Caveats:**
> 1. Noise is synthetic white Gaussian added before the 3500 Hz low-pass (so post-low-pass effective SNR is higher; approximately +2.34 dB for white noise only; the measured values are in the effective SNR tables).
> 2. Models are trained on clean data only (zero retraining or noise augmentation).
> 3. Normal files originate from 48 kHz resampled recordings and are quieter (`rms = 0.045`), receiving weaker absolute noise in Variant A and lower effective SNR in Variant B.
> 4. Each LOLO test fold contains 10 test files and each Severity test fold contains 13 test files (with only 1 Normal test file per fold).

## Table 1: Macro-F1 vs. Additive Noise Level (Variant A & Variant B)

{to_markdown_table(df_t1)}

## Table 2: Normal Recall vs. Additive Noise Level

{to_markdown_table(df_t2)}

## Table 3: Variant B Effective SNR Distribution (Pre-LP vs. Post-LP)

{to_markdown_table(df_t3)}

## Table 4: Class-Wise Feature Means across Noise Levels

{to_markdown_table(df_feat_deg)}
"""

    with open(results_dir / "phase6_summary.md", "w") as f:
        f.write(summary_md)
    print(f"  Saved: {results_dir / 'phase6_summary.md'}")

    total_time = time.time() - t_start_total
    print("\n" + "=" * 80)
    print("PHASE 6 EXECUTION COMPLETE")
    print(f"Total Runtime: {total_time:.2f}s ({total_time / 60:.2f} min)")
    print("=" * 80)


if __name__ == "__main__":
    main()
