"""Phase 2: Physics-Informed Vibration Feature Extraction & LOLO Fold 3 Pilot Check.

Extracts 21 physical features per 2048-sample window (50% overlap, 1024 step):
- 9 Time-Domain: rms, kurtosis, crest_factor, skewness, peak_to_peak, std, shape_factor, impulse_factor, margin_factor
- 6 Frequency-Domain: spectral_centroid, spectral_entropy, band_energy_bpfo, band_energy_bpfi, band_energy_bsf, band_energy_2bsf
- 6 Envelope Demodulation (1500-3000 Hz bandpass): env_snr_bpfo, env_snr_2bpfo, env_snr_bpfi, env_snr_2bpfi, env_snr_bsf, env_snr_2bsf

Two dataset variants are constructed:
1. "lp3500": Common 8th-order Butterworth low-pass filter at 3500 Hz applied to each whole signal prior to windowing.
2. "nolp": Raw signals without the 3500 Hz low-pass filter.

Saves:
- data/processed/phase2_features_lp3500.csv
- data/processed/phase2_features_nolp.csv
- results/feature_groups.json
- results/phase2_features_summary.txt
- results/pilot_lolo_fold3.txt
"""

import json
import os
import sys
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

np.random.seed(42)

FEATURE_GROUPS = {
    "time": [
        "rms",
        "kurtosis",
        "crest_factor",
        "skewness",
        "peak_to_peak",
        "std",
        "shape_factor",
        "impulse_factor",
        "margin_factor"
    ],
    "freq": [
        "spectral_centroid",
        "spectral_entropy",
        "band_energy_bpfo",
        "band_energy_bpfi",
        "band_energy_bsf",
        "band_energy_2bsf"
    ],
    "envelope": [
        "env_snr_bpfo",
        "env_snr_2bpfo",
        "env_snr_bpfi",
        "env_snr_2bpfi",
        "env_snr_bsf",
        "env_snr_2bsf"
    ]
}
ALL_FEATURES = FEATURE_GROUPS["time"] + FEATURE_GROUPS["freq"] + FEATURE_GROUPS["envelope"]


def extract_window_features(
    win_raw: np.ndarray,
    freqs: np.ndarray,
    hann: np.ndarray,
    bp_sos: np.ndarray,
    bpfo: float,
    bpfi: float,
    bsf: float
) -> dict:
    """Extract 21 physics-informed features from a single 2048-sample window."""
    # Precaution: demean the window so per-recording DC offset cannot be a shortcut
    x = win_raw - np.mean(win_raw)

    # 1. Time-Domain (9 features)
    rms = float(np.sqrt(np.mean(x ** 2)))
    kurtosis = float(scipy.stats.kurtosis(x, fisher=False))  # Pearson definition
    crest_factor = float(np.max(np.abs(x)) / (rms + 1e-12))
    skewness = float(scipy.stats.skew(x))
    peak_to_peak = float(np.ptp(x))
    std = float(np.std(x))
    mean_abs = float(np.mean(np.abs(x)))
    shape_factor = float(rms / (mean_abs + 1e-12))
    impulse_factor = float(np.max(np.abs(x)) / (mean_abs + 1e-12))
    mean_sqrt_abs = float(np.mean(np.sqrt(np.abs(x))))
    margin_factor = float(np.max(np.abs(x)) / ((mean_sqrt_abs ** 2) + 1e-12))

    # 2. Frequency-Domain (6 features)
    spec = np.abs(np.fft.rfft(x * hann))
    power = spec ** 2
    total_power = float(np.sum(power))

    spectral_centroid = float(np.sum(freqs * power) / (total_power + 1e-12))
    p = power / (total_power + 1e-12)
    spectral_entropy = float(-np.sum(p * np.log(p + 1e-12)))

    tol = 6.0
    def get_band_energy_frac(f_target: float) -> float:
        mask = (freqs >= f_target - tol) & (freqs <= f_target + tol)
        return float(np.sum(power[mask]) / (total_power + 1e-12))

    band_energy_bpfo = get_band_energy_frac(bpfo)
    band_energy_bpfi = get_band_energy_frac(bpfi)
    band_energy_bsf = get_band_energy_frac(bsf)
    band_energy_2bsf = get_band_energy_frac(2.0 * bsf)

    # 3. Envelope Demodulation (6 features)
    # Bandpass 1500-3000 Hz, demean, Hilbert envelope, demean envelope, Hann, FFT
    filt = scipy.signal.sosfiltfilt(bp_sos, x)
    demeaned_filt = filt - np.mean(filt)
    env = np.abs(scipy.signal.hilbert(demeaned_filt))
    env_demeaned = env - np.mean(env)
    env_spec = np.abs(np.fft.rfft(env_demeaned * hann))

    noise_median = float(np.median(env_spec[(freqs >= 10.0) & (freqs <= 350.0)]))

    def get_env_snr(f_target: float) -> float:
        mask = (freqs >= f_target - tol) & (freqs <= f_target + tol)
        return float(np.max(env_spec[mask]) / (noise_median + 1e-12))

    env_snr_bpfo = get_env_snr(bpfo)
    env_snr_2bpfo = get_env_snr(2.0 * bpfo)
    env_snr_bpfi = get_env_snr(bpfi)
    env_snr_2bpfi = get_env_snr(2.0 * bpfi)
    env_snr_bsf = get_env_snr(bsf)
    env_snr_2bsf = get_env_snr(2.0 * bsf)

    return {
        "rms": rms,
        "kurtosis": kurtosis,
        "crest_factor": crest_factor,
        "skewness": skewness,
        "peak_to_peak": peak_to_peak,
        "std": std,
        "shape_factor": shape_factor,
        "impulse_factor": impulse_factor,
        "margin_factor": margin_factor,
        "spectral_centroid": spectral_centroid,
        "spectral_entropy": spectral_entropy,
        "band_energy_bpfo": band_energy_bpfo,
        "band_energy_bpfi": band_energy_bpfi,
        "band_energy_bsf": band_energy_bsf,
        "band_energy_2bsf": band_energy_2bsf,
        "env_snr_bpfo": env_snr_bpfo,
        "env_snr_2bpfo": env_snr_2bpfo,
        "env_snr_bpfi": env_snr_bpfi,
        "env_snr_2bpfi": env_snr_2bpfi,
        "env_snr_bsf": env_snr_bsf,
        "env_snr_2bsf": env_snr_2bsf,
    }


def compute_features_for_dataset(signals: dict, manifest_df: pd.DataFrame, use_lp3500: bool) -> tuple[pd.DataFrame, list[float]]:
    """Build feature DataFrame across all files and windows. Also computes hf_frac."""
    window_size = 2048
    step = 1024
    hann = np.hanning(window_size)
    freqs = np.fft.rfftfreq(window_size, d=1.0 / 12000.0)

    lp_sos = scipy.signal.butter(8, 3500, btype="lowpass", fs=12000, output="sos")
    bp_sos = scipy.signal.butter(4, [1500, 3000], btype="bandpass", fs=12000, output="sos")

    rows = []
    hf_fracs = []

    for fid in sorted(signals.keys()):
        raw_x = signals[fid]["x"]
        meta = manifest_df.loc[fid]
        expected_windows = int(meta["window_count"])

        # Variant filtering: apply low-pass to the whole signal if requested
        if use_lp3500:
            sig_x = scipy.signal.sosfiltfilt(lp_sos, raw_x)
        else:
            sig_x = raw_x

        n_windows = (len(sig_x) - window_size) // step + 1
        assert n_windows == expected_windows, (
            f"File {fid}: computed windows ({n_windows}) != manifest window_count ({expected_windows})"
        )

        rpm = float(meta["rpm"])
        f_shaft = rpm / 60.0
        bpfo = 3.5848 * f_shaft
        bpfi = 5.4152 * f_shaft
        bsf = 4.7135 * f_shaft

        fault_class = meta["fault_class"]
        severity = str(meta["severity"]) if pd.notna(meta["severity"]) else "none"
        load_hp = int(meta["load_hp"])

        for win_idx in range(n_windows):
            win_raw = sig_x[win_idx * step : win_idx * step + window_size]
            feats = extract_window_features(win_raw, freqs, hann, bp_sos, bpfo, bpfi, bsf)

            # Metadata
            row = {
                "file_id": fid,
                "fault_class": fault_class,
                "severity": severity,
                "load_hp": load_hp,
                "window_index": win_idx,
                "rpm": rpm,
            }
            row.update(feats)
            rows.append(row)

            # Leakage diagnostic feature: hf_frac (power in 4000-5900 Hz / total power)
            w_demean = win_raw - np.mean(win_raw)
            w_spec = np.abs(np.fft.rfft(w_demean * hann))
            w_pow = w_spec ** 2
            tot_p = np.sum(w_pow)
            mask_hf = (freqs >= 4000.0) & (freqs <= 5900.0)
            hf_frac = float(np.sum(w_pow[mask_hf]) / (tot_p + 1e-12))
            hf_fracs.append(hf_frac)

    df = pd.DataFrame(rows)
    return df, hf_fracs


def main():
    root_dir = Path(__file__).resolve().parent.parent
    proc_dir = root_dir / "data" / "processed"
    results_dir = root_dir / "results"
    proc_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 2: FEATURE EXTRACTION (21 FEATURES, VARIANTS: lp3500 & nolp)")
    print("=" * 80)

    # 1. Load data
    pkl_path = proc_dir / "phase1_signals_12k.pkl"
    with open(pkl_path, "rb") as f:
        signals = pickle.load(f)

    manifest_path = results_dir / "phase1_file_manifest.csv"
    manifest_df = pd.read_csv(manifest_path).set_index("file_id")

    # 2. Extract features for both variants
    print("Extracting features for lp3500 variant (common 3500 Hz low-pass)...")
    df_lp3500, hf_lp = compute_features_for_dataset(signals, manifest_df, use_lp3500=True)

    print("Extracting features for nolp variant (unfiltered 12k signals)...")
    df_nolp, hf_nolp = compute_features_for_dataset(signals, manifest_df, use_lp3500=False)

    # Assert expected row counts
    assert len(df_lp3500) == 4643, f"Expected 4643 rows, got {len(df_lp3500)}"
    assert len(df_nolp) == 4643, f"Expected 4643 rows, got {len(df_nolp)}"
    assert len(df_lp3500[df_lp3500["fault_class"] == "Normal"]) == 409, "Expected 409 Normal rows"
    assert len(df_nolp[df_nolp["fault_class"] == "Normal"]) == 409, "Expected 409 Normal rows"

    # Assert 0 NaNs and 0 Infs
    assert df_lp3500[ALL_FEATURES].isna().sum().sum() == 0, "NaN found in df_lp3500"
    assert np.isinf(df_lp3500[ALL_FEATURES].values).sum() == 0, "Inf found in df_lp3500"
    assert df_nolp[ALL_FEATURES].isna().sum().sum() == 0, "NaN found in df_nolp"
    assert np.isinf(df_nolp[ALL_FEATURES].values).sum() == 0, "Inf found in df_nolp"

    # Save CSVs
    csv_lp3500_path = proc_dir / "phase2_features_lp3500.csv"
    csv_nolp_path = proc_dir / "phase2_features_nolp.csv"
    df_lp3500.to_csv(csv_lp3500_path, index=False)
    df_nolp.to_csv(csv_nolp_path, index=False)
    print(f"Saved: {csv_lp3500_path}")
    print(f"Saved: {csv_nolp_path}")

    # Save feature_groups.json
    groups_path = results_dir / "feature_groups.json"
    with open(groups_path, "w", encoding="utf-8") as f:
        json.dump(FEATURE_GROUPS, f, indent=2)
    print(f"Saved: {groups_path}")

    # 3. Create phase2_features_summary.txt
    summary_txt_path = results_dir / "phase2_features_summary.txt"
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("PHASE 2 FEATURE EXTRACTION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write("1. ROW COUNTS PER CLASS\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Class':<15} | {'lp3500 Rows':<15} | {'nolp Rows':<15}\n")
        f.write("-" * 50 + "\n")
        for cls in ["Normal", "Inner Race", "Ball", "Outer Race"]:
            cnt_lp = len(df_lp3500[df_lp3500["fault_class"] == cls])
            cnt_no = len(df_nolp[df_nolp["fault_class"] == cls])
            f.write(f"{cls:<15} | {cnt_lp:<15} | {cnt_no:<15}\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'TOTAL':<15} | {len(df_lp3500):<15} | {len(df_nolp):<15}\n\n")

        f.write("2. DATA INTEGRITY AUDIT\n")
        f.write("-" * 50 + "\n")
        f.write(f"lp3500 NaN count: {df_lp3500[ALL_FEATURES].isna().sum().sum()}\n")
        f.write(f"lp3500 Inf count: {np.isinf(df_lp3500[ALL_FEATURES].values).sum()}\n")
        f.write(f"nolp NaN count:   {df_nolp[ALL_FEATURES].isna().sum().sum()}\n")
        f.write(f"nolp Inf count:   {np.isinf(df_nolp[ALL_FEATURES].values).sum()}\n\n")

        f.write("3. FEATURE MEANS PER CLASS (rms, kurtosis, spectral_centroid, env_snr_bpfi)\n")
        f.write("=" * 95 + "\n")
        f.write(f"{'Variant':<8} | {'Class':<12} | {'RMS':>10} | {'Kurtosis':>10} | {'Spec Centroid':>15} | {'BPFI Env SNR':>14}\n")
        f.write("-" * 95 + "\n")
        for var_name, df_var in [("lp3500", df_lp3500), ("nolp", df_nolp)]:
            for cls in ["Normal", "Inner Race", "Ball", "Outer Race"]:
                sub = df_var[df_var["fault_class"] == cls]
                m_rms = sub["rms"].mean()
                m_kurt = sub["kurtosis"].mean()
                m_sc = sub["spectral_centroid"].mean()
                m_bpfi = sub["env_snr_bpfi"].mean()
                f.write(f"{var_name:<8} | {cls:<12} | {m_rms:10.4f} | {m_kurt:10.4f} | {m_sc:15.2f} | {m_bpfi:14.2f}\n")
            f.write("-" * 95 + "\n")

    print(f"Saved: {summary_txt_path}")

    # 4. PART 3: Pilot Check (LOLO Fold 3: Train 0,1,2 HP; Test 3 HP) & Leakage Diagnostic
    print("\n" + "=" * 80)
    print("PART 3: PILOT CHECK (LOLO FOLD 3) & LEAKAGE DIAGNOSTIC")
    print("=" * 80)

    pilot_txt_path = results_dir / "pilot_lolo_fold3.txt"
    pilot_lines = []

    classes_ordered = ["Normal", "Inner Race", "Ball", "Outer Race"]

    for var_name, df_var in [("lp3500", df_lp3500), ("nolp", df_nolp)]:
        pilot_lines.append("=" * 80)
        pilot_lines.append(f"PILOT EVALUATION: VARIANT '{var_name}' (Train: 0, 1, 2 HP | Test: 3 HP)")
        pilot_lines.append("=" * 80)

        train_mask = df_var["load_hp"].isin([0, 1, 2])
        test_mask = df_var["load_hp"] == 3

        X_train = df_var.loc[train_mask, ALL_FEATURES].values
        y_train = df_var.loc[train_mask, "fault_class"].values
        X_test = df_var.loc[test_mask, ALL_FEATURES].values
        y_test = df_var.loc[test_mask, "fault_class"].values
        test_meta = df_var.loc[test_mask, ["file_id", "fault_class", "severity"]].copy()

        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
        pilot_lines.append(f"\nMacro-F1: {macro_f1:.4f}")

        # Per-class metrics
        pilot_lines.append("\nPer-Class Metrics:")
        pilot_lines.append(f"{'Class':<12} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'Test Windows':>12}")
        pilot_lines.append("-" * 65)
        for cls in classes_ordered:
            cls_mask = y_test == cls
            n_win = int(np.sum(cls_mask))
            p = float(precision_score(y_test == cls, y_pred == cls, zero_division=0))
            r = float(recall_score(y_test == cls, y_pred == cls, zero_division=0))
            f1 = float(f1_score(y_test == cls, y_pred == cls, zero_division=0))
            pilot_lines.append(f"{cls:<12} | {p:10.4f} | {r:10.4f} | {f1:10.4f} | {n_win:12d}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=classes_ordered)
        pilot_lines.append("\nConfusion Matrix (Rows=True, Cols=Pred):")
        pilot_lines.append(f"{'':<12} | " + " | ".join([f"{c[:7]:>7}" for c in classes_ordered]))
        pilot_lines.append("-" * 50)
        for idx, cls in enumerate(classes_ordered):
            row_str = " | ".join([f"{cm[idx, j]:7d}" for j in range(len(classes_ordered))])
            pilot_lines.append(f"{cls:<12} | {row_str}")

        # Top 5 features
        importances = rf.feature_importances_
        top5_indices = np.argsort(importances)[::-1][:5]
        pilot_lines.append("\nTop 5 Most Important Features:")
        for rank, idx in enumerate(top5_indices, 1):
            pilot_lines.append(f"  {rank}. {ALL_FEATURES[idx]}: {importances[idx]:.4f}")

        # 3b. Per-file view
        pilot_lines.append("\nPer-File Test Performance (Load 3 HP):")
        pilot_lines.append(f"{'File ID':<8} | {'Class':<12} | {'Severity':<8} | {'Correct Win':>12} | {'Total Win':>10} | {'Accuracy':>10} | {'Majority Vote':<15}")
        pilot_lines.append("-" * 88)

        test_meta["pred"] = y_pred
        file_counts_per_class = {c: 0 for c in classes_ordered}
        for fid in sorted(test_meta["file_id"].unique()):
            f_sub = test_meta[test_meta["file_id"] == fid]
            cls = f_sub["fault_class"].iloc[0]
            sev = f_sub["severity"].iloc[0]
            tot_w = len(f_sub)
            corr_w = int(np.sum(f_sub["pred"] == cls))
            acc = corr_w / tot_w
            maj_vote = f_sub["pred"].mode().iloc[0]
            file_counts_per_class[cls] += 1
            pilot_lines.append(f"{fid:<8} | {cls:<12} | {sev:<8} | {corr_w:12d} | {tot_w:10d} | {acc:10.2%} | {maj_vote:<15}")

        pilot_lines.append(f"\nTest Files Per Class: " + ", ".join([f"{c}: {file_counts_per_class[c]}" for c in classes_ordered]))
        pilot_lines.append("\n")

    # 3c. Leakage diagnostic (using df_nolp and hf_nolp)
    pilot_lines.append("=" * 80)
    pilot_lines.append("LEAKAGE DIAGNOSTIC (High-Frequency Power Fraction in 4000-5900 Hz, nolp variant)")
    pilot_lines.append("=" * 80)

    df_nolp["hf_frac"] = hf_nolp
    pilot_lines.append("\nMean hf_frac per class (nolp windows):")
    for cls in classes_ordered:
        mean_hf = float(df_nolp[df_nolp["fault_class"] == cls]["hf_frac"].mean())
        pilot_lines.append(f"  {cls:<12}: {mean_hf:.6f} ({mean_hf*100:.3f}% of total power)")

    # AUROC computations: Normal = positive class (1), faults = 0
    y_normal_binary = (df_nolp["fault_class"] == "Normal").astype(int).values
    scores_all = df_nolp["hf_frac"].values
    auroc_all = float(roc_auc_score(y_normal_binary, scores_all))

    # Per fault class comparisons
    aurocs_per_fault = {}
    for fault_c in ["Inner Race", "Ball", "Outer Race"]:
        sub = df_nolp[df_nolp["fault_class"].isin(["Normal", fault_c])]
        y_bin = (sub["fault_class"] == "Normal").astype(int).values
        sc = sub["hf_frac"].values
        aurocs_per_fault[fault_c] = float(roc_auc_score(y_bin, sc))

    pilot_lines.append("\nAUROC of hf_frac for detecting Normal (positive class):")
    pilot_lines.append(f"  Normal vs All Faults:       {auroc_all:.4f}")
    for fault_c, auc_val in aurocs_per_fault.items():
        pilot_lines.append(f"  Normal vs {fault_c:<15}: {auc_val:.4f}")

    # Write pilot_lines to pilot_lolo_fold3.txt
    with open(pilot_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(pilot_lines) + "\n")
    print(f"Saved: {pilot_txt_path}")

    # Print summary output to stdout
    print("\n--- Pilot Results Summary ---")
    for line in pilot_lines:
        print(line)


if __name__ == "__main__":
    main()
