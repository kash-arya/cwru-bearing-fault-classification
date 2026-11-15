"""Phase 2b: Common Low-Pass Leakage Fix & Envelope Band Evaluation.

Evaluates:
- TEST A: 8th-order Butterworth low-pass filter at cutoffs C in [3000, 3500, 4000] Hz.
          Control ratio (109r / 105) in 500 Hz sub-bands, spread, and slope.
- TEST B: Envelope label matching across candidate bands on low-passed signals.
          Harmonics: BPFO, 2xBPFO, BPFI, 2xBPFI, BSF, 2xBSF (±6 Hz tolerance).
          Sensitivity check across cutoffs C.

Outputs:
- results/phase2b_leakage_fix_summary.txt
- results/diagnostic_leakage_fix.png
"""

import os
import sys
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import scipy.io
import scipy.signal

# Configure headless matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)


class SummaryLogger:
    """Tee logger to write both to stdout and to a summary file."""
    def __init__(self, filepath: Path):
        self.terminal = sys.stdout
        self.logfile = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        self.logfile.close()


def compute_average_spectrum(x: np.ndarray, window_size: int = 2048, step: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Compute average magnitude spectrum across 2048-sample windows with 50% overlap."""
    n_windows = (len(x) - window_size) // step + 1
    hann = np.hanning(window_size)
    specs = []
    for i in range(n_windows):
        win = x[i * step : i * step + window_size]
        spec = np.abs(np.fft.rfft(win * hann))
        specs.append(spec)
    avg_spec = np.mean(specs, axis=0)
    freqs = np.fft.rfftfreq(window_size, d=1.0 / 12000.0)
    return freqs, avg_spec


def compute_envelope_spectrum(x: np.ndarray, sos: np.ndarray, window_size: int = 2048, step: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Compute demodulated envelope spectrum averaged across windows."""
    n_windows = (len(x) - window_size) // step + 1
    hann = np.hanning(window_size)
    env_specs = []
    for i in range(n_windows):
        win = x[i * step : i * step + window_size]
        filtered = scipy.signal.sosfiltfilt(sos, win)
        demeaned = filtered - np.mean(filtered)
        env = np.abs(scipy.signal.hilbert(demeaned))
        env_demeaned = env - np.mean(env)
        env_spec = np.abs(np.fft.rfft(env_demeaned * hann))
        env_specs.append(env_spec)
    avg_env_spec = np.mean(env_specs, axis=0)
    freqs = np.fft.rfftfreq(window_size, d=1.0 / 12000.0)
    return freqs, avg_env_spec


def main():
    root_dir = Path(__file__).resolve().parent.parent
    proc_dir = root_dir / "data" / "processed"
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "phase2b_leakage_fix_summary.txt"
    logger = SummaryLogger(summary_path)
    sys.stdout = logger

    print("=" * 100)
    print("PHASE 2B: COMMON LOW-PASS FILTER LEAKAGE FIX & ENVELOPE BAND EVALUATION")
    print("=" * 100)

    # 1. Load Phase 1 processed signals
    pkl_path = proc_dir / "phase1_signals_12k.pkl"
    with open(pkl_path, "rb") as f:
        signals = pickle.load(f)

    # 2. Load manifest
    manifest_path = results_dir / "phase1_file_manifest.csv"
    manifest_df = pd.read_csv(manifest_path).set_index("file_id")

    # 3. Locate and rebuild resampled 109 ("109r")
    candidate_109 = [
        root_dir / "109.mat",
        root_dir / "data" / "raw" / "109.mat",
        root_dir / "verification" / "109.mat"
    ]
    path_109 = None
    for cand in candidate_109:
        if cand.exists():
            path_109 = cand
            break

    if path_109 is None:
        print("[ERROR] 109.mat not found in workspace root, data/raw, or verification/ folder.")
        sys.exit(1)

    mat_109 = scipy.io.loadmat(str(path_109))
    if "X109_DE_time" not in mat_109:
        print("[ERROR] Missing required key 'X109_DE_time' in 109.mat.")
        sys.exit(1)

    x_109_raw = mat_109["X109_DE_time"].squeeze().astype(np.float64)
    x_109r = scipy.signal.resample_poly(x_109_raw, 1, 4)

    # Base signals list: all 40 files + 109r
    sorted_fids = sorted(signals.keys())
    x_105 = signals[105]["x"]

    # -------------------------------------------------------------------------
    # TEST A: Common Low-Pass Filter Evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("TEST A: COMMON LOW-PASS FILTER AUDIT (Cutoffs C in [3000, 3500, 4000] Hz)")
    print("=" * 100)

    cutoffs = [3000, 3500, 4000]
    test_a_results = {}
    spectra_for_plots = {}

    for C in cutoffs:
        lp_sos = scipy.signal.butter(8, C, btype="lowpass", fs=12000, output="sos")

        # Filter 105 and 109r
        x_105_lp = scipy.signal.sosfiltfilt(lp_sos, x_105)
        x_109r_lp = scipy.signal.sosfiltfilt(lp_sos, x_109r)

        freqs, s_105 = compute_average_spectrum(x_105_lp)
        _, s_109r = compute_average_spectrum(x_109r_lp)
        spectra_for_plots[C] = (freqs, s_105, s_109r)

        # 500 Hz sub-bands from 500 Hz up to C - 500 Hz
        sub_bands = []
        low = 500
        while low + 500 <= C - 500:
            sub_bands.append((low, low + 500))
            low += 500

        ratios_db = []
        centres_khz = []
        sub_band_labels = []

        for low_f, high_f in sub_bands:
            mask = (freqs >= low_f) & (freqs <= high_f)
            mean_105 = np.mean(s_105[mask])
            mean_109r = np.mean(s_109r[mask])
            ratio = 20.0 * np.log10(max(mean_109r, 1e-12) / max(mean_105, 1e-12))
            ratios_db.append(ratio)
            centres_khz.append((low_f + high_f) / 2000.0)
            sub_band_labels.append(f"{low_f}-{high_f}Hz")

        spread = max(ratios_db) - min(ratios_db)
        slope = float(np.polyfit(centres_khz, ratios_db, 1)[0])
        status = "CONTROL OK" if spread <= 3.0 else "CONTROL NOT OK"

        test_a_results[C] = {
            "sub_bands": sub_band_labels,
            "ratios_db": ratios_db,
            "spread": spread,
            "slope": slope,
            "status": status
        }

        print(f"\nCutoff C = {C} Hz:")
        print(f"  Sub-bands: {', '.join([f'{lbl}: {r:+.2f} dB' for lbl, r in zip(sub_band_labels, ratios_db)])}")
        print(f"  Spread (max - min): {spread:.2f} dB")
        print(f"  Slope (ratio_dB vs centre kHz): {slope:+.2f} dB/kHz")
        print(f"  Verdict: {status}")
        print("  Note: 3 dB is a rule of thumb. 109 and 105 are separate recordings. A steadily rising or falling ratio (large slope) is more worrying than a small constant offset.")

    # -------------------------------------------------------------------------
    # Generate diagnostic plot: results/diagnostic_leakage_fix.png
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, C in enumerate(cutoffs):
        ax = axes[idx]
        freqs, s_105, s_109r = spectra_for_plots[C]
        mask = (freqs >= 10.0) & (freqs <= float(C))
        f_sub = freqs[mask]
        ratio_curve = 20.0 * np.log10(np.maximum(s_109r[mask], 1e-12) / np.maximum(s_105[mask], 1e-12))

        ax.plot(f_sub, ratio_curve, "b-", linewidth=1.0, label="109r / 105 (dB)")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(f"Cutoff C = {C} Hz (10 Hz to {C} Hz)\nSpread: {test_a_results[C]['spread']:.2f} dB | Slope: {test_a_results[C]['slope']:+.2f} dB/kHz", fontsize=10)
        ax.set_xlabel("Frequency (Hz)", fontsize=9)
        ax.set_ylabel("Ratio 109r / 105 (dB)", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plot_path = results_dir / "diagnostic_leakage_fix.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved diagnostic plot: {plot_path}")

    # -------------------------------------------------------------------------
    # TEST B: Envelope Label Check on Low-Passed Signals
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("TEST B: ENVELOPE LABEL CHECK ON LOW-PASSED SIGNALS")
    print("=" * 100)

    candidate_bands = [
        (1000, 2500),
        (1500, 3000),
        (1500, 3500),
        (2000, 3500)
    ]

    # Store full SNR records for sensitivity check: (C, band) -> {fid: {col: val}}
    snr_store = {}
    test_b_summaries = {}

    for C in cutoffs:
        lp_sos = scipy.signal.butter(8, C, btype="lowpass", fs=12000, output="sos")

        # Pre-filter all 40 signals + 109r with common low-pass C
        filtered_signals = {}
        for fid in sorted_fids:
            filtered_signals[fid] = scipy.signal.sosfiltfilt(lp_sos, signals[fid]["x"])
        filtered_signals["109r"] = scipy.signal.sosfiltfilt(lp_sos, x_109r)

        valid_bands = [b for b in candidate_bands if b[1] <= C - 500]

        for band in valid_bands:
            band_tuple = tuple(band)
            bp_sos = scipy.signal.butter(4, list(band), btype="bandpass", fs=12000, output="sos")

            print("\n" + "-" * 100)
            print(f"Cutoff C = {C} Hz | Bandpass = [{band[0]}, {band[1]}] Hz")
            print("-" * 100)
            header = f"{'file_id':<8} | {'class':<11} | {'sev':<6} | {'hp':>2} | {'BPFO':>6} | {'2xBPFO':>6} | {'BPFI':>6} | {'2xBPFI':>6} | {'BSF':>6} | {'2xBSF':>6} | {'strongest':<9} | {'old':<6} | {'matches':<7}"
            print(header)
            print("-" * 100)

            snr_store[(C, band_tuple)] = {}
            table_rows = []

            # Process all 40 files + 109r
            all_eval_ids = list(sorted_fids) + ["109r"]

            for fid in all_eval_ids:
                x_sig = filtered_signals[fid]
                if fid == "109r":
                    cls = "Inner Race"
                    sev = "0.007"
                    hp = 0
                    rpm = 1797.0
                    display_cls = "Inner (ref)"
                else:
                    meta = manifest_df.loc[fid]
                    cls = meta["fault_class"]
                    sev = str(meta["severity"]) if pd.notna(meta["severity"]) else "n/a"
                    hp = int(meta["load_hp"])
                    rpm = float(meta["rpm"])
                    display_cls = cls

                f_shaft = rpm / 60.0
                bpfo = 3.5848 * f_shaft
                bpfi = 5.4152 * f_shaft
                bsf = 4.7135 * f_shaft

                env_f, avg_env = compute_envelope_spectrum(x_sig, bp_sos)
                noise_median = np.median(avg_env[(env_f >= 10.0) & (env_f <= 350.0)])

                tol = 6.0
                def get_snr(f_target):
                    band_mask = (env_f >= f_target - tol) & (env_f <= f_target + tol)
                    return float(np.max(avg_env[band_mask]) / noise_median)

                s_bpfo = get_snr(bpfo)
                s_2bpfo = get_snr(2.0 * bpfo)
                s_bpfi = get_snr(bpfi)
                s_2bpfi = get_snr(2.0 * bpfi)
                s_bsf = get_snr(bsf)
                s_2bsf = get_snr(2.0 * bsf)

                outer = max(s_bpfo, s_2bpfo)
                inner = max(s_bpfi, s_2bpfi)
                ball = max(s_bsf, s_2bsf)

                # strongest
                if outer >= inner and outer >= ball:
                    strongest = "outer"
                elif inner >= outer and inner >= ball:
                    strongest = "inner"
                else:
                    strongest = "ball"

                # strongest_old
                if s_bpfo >= s_bpfi and s_bpfo >= ball:
                    strongest_old = "BPFO"
                elif s_bpfi >= s_bpfo and s_bpfi >= ball:
                    strongest_old = "BPFI"
                else:
                    strongest_old = "ball"

                # matches_label
                if cls == "Normal" or fid == "109r":
                    matches_label = "n/a"
                elif cls == "Inner Race":
                    matches_label = "yes" if strongest == "inner" else "no"
                elif cls == "Outer Race":
                    matches_label = "yes" if strongest == "outer" else "no"
                elif cls == "Ball":
                    matches_label = "yes" if strongest == "ball" else "no"
                else:
                    matches_label = "n/a"

                snr_store[(C, band_tuple)][fid] = {
                    "bpfo": s_bpfo, "2bpfo": s_2bpfo,
                    "bpfi": s_bpfi, "2bpfi": s_2bpfi,
                    "bsf": s_bsf, "2bsf": s_2bsf,
                    "outer": outer, "inner": inner, "ball": ball,
                    "strongest": strongest, "strongest_old": strongest_old,
                    "matches_label": matches_label,
                    "class": cls, "severity": sev, "load_hp": hp
                }

                fid_str = str(fid)
                row_str = f"{fid_str:<8} | {display_cls:<11} | {sev:<6} | {hp:2d} | {s_bpfo:6.2f} | {s_2bpfo:6.2f} | {s_bpfi:6.2f} | {s_2bpfi:6.2f} | {s_bsf:6.2f} | {s_2bsf:6.2f} | {strongest:<9} | {strongest_old:<6} | {matches_label:<7}"
                print(row_str)
                table_rows.append(row_str)

            # --- Summaries for (C, band) ---
            # Exclude Normal and 109r from fault file calculations
            fault_fids = [fid for fid in sorted_fids if manifest_df.loc[fid, "fault_class"] != "Normal"]

            # (a) how many of the 36 fault files have strongest = outer / inner / ball
            counts_fault = {"outer": 0, "inner": 0, "ball": 0}
            for fid in fault_fids:
                counts_fault[snr_store[(C, band_tuple)][fid]["strongest"]] += 1

            chance_outer = 12.0 * (counts_fault["outer"] / 36.0)
            chance_inner = 12.0 * (counts_fault["inner"] / 36.0)
            chance_ball = 12.0 * (counts_fault["ball"] / 36.0)

            # (b) per class: matches out of 12, non-matching files, matches broken down by severity
            class_stats = {}
            for target_cls, target_strongest, old_target in [
                ("Inner Race", "inner", "BPFI"),
                ("Ball", "ball", "ball"),
                ("Outer Race", "outer", "BPFO")
            ]:
                cls_fids = [fid for fid in fault_fids if manifest_df.loc[fid, "fault_class"] == target_cls]
                matches = [fid for fid in cls_fids if snr_store[(C, band_tuple)][fid]["strongest"] == target_strongest]
                non_matches = [fid for fid in cls_fids if snr_store[(C, band_tuple)][fid]["strongest"] != target_strongest]

                # Breakdown by severity (0.007", 0.014", 0.021")
                sev_matches = {}
                for s in ["0.007", "0.014", "0.021"]:
                    s_fids = [fid for fid in cls_fids if str(manifest_df.loc[fid, "severity"]) == s]
                    s_m = [fid for fid in s_fids if snr_store[(C, band_tuple)][fid]["strongest"] == target_strongest]
                    sev_matches[s] = f"{len(s_m)}/{len(s_fids)}"

                # Match count using strongest_old
                old_matches = [fid for fid in cls_fids if snr_store[(C, band_tuple)][fid]["strongest_old"] == old_target]

                class_stats[target_cls] = {
                    "matches": len(matches),
                    "non_matches": non_matches,
                    "sev_breakdown": sev_matches,
                    "old_matches": len(old_matches)
                }

            # (c) mean value of strongest SNR over 4 Normal files
            normal_fids = [fid for fid in sorted_fids if manifest_df.loc[fid, "fault_class"] == "Normal"]
            normal_strongest_snrs = [
                max(snr_store[(C, band_tuple)][fid]["outer"], snr_store[(C, band_tuple)][fid]["inner"], snr_store[(C, band_tuple)][fid]["ball"])
                for fid in normal_fids
            ]
            mean_normal_snr = float(np.mean(normal_strongest_snrs))

            # (d) whether 109r has inner as its strongest
            inner_109r = (snr_store[(C, band_tuple)]["109r"]["strongest"] == "inner")

            test_b_summaries[(C, band_tuple)] = {
                "counts_fault": counts_fault,
                "chance": {"Outer Race": chance_outer, "Inner Race": chance_inner, "Ball": chance_ball},
                "class_stats": class_stats,
                "mean_normal_snr": mean_normal_snr,
                "inner_109r": inner_109r
            }

            print(f"\nSummary for Cutoff C={C} Hz, Band={band}:")
            print(f"  Fault files strongest counts (total 36): outer={counts_fault['outer']}, inner={counts_fault['inner']}, ball={counts_fault['ball']}")
            print(f"  Chance levels (12 files): Outer Race={chance_outer:.2f}, Inner Race={chance_inner:.2f}, Ball={chance_ball:.2f}")
            for c_name in ["Inner Race", "Ball", "Outer Race"]:
                st = class_stats[c_name]
                print(f"  {c_name:<11}: {st['matches']}/12 matched (chance: {test_b_summaries[(C, band_tuple)]['chance'][c_name]:.2f}/12) | Non-matching: {st['non_matches']} | Severity: 0.007={st['sev_breakdown']['0.007']}, 0.014={st['sev_breakdown']['0.014']}, 0.021={st['sev_breakdown']['0.021']} | Old rule: {st['old_matches']}/12")
            print(f"  Normal mean strongest SNR (4 files): {mean_normal_snr:.2f}")
            print(f"  109r strongest is inner: {inner_109r}")

    # -------------------------------------------------------------------------
    # Sensitivity Check
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("SENSITIVITY CHECK ACROSS CUTOFFS C")
    print("=" * 100)

    # For each band that was run at more than one C
    bands_multi_c = []
    for band in candidate_bands:
        band_tuple = tuple(band)
        cs_for_band = [C for C in cutoffs if (C, band_tuple) in snr_store]
        if len(cs_for_band) > 1:
            bands_multi_c.append((band_tuple, cs_for_band))

    snr_cols = ["bpfo", "2bpfo", "bpfi", "2bpfi", "bsf", "2bsf"]
    all_fids_eval = list(sorted_fids) + ["109r"]

    for band_tuple, cs_for_band in bands_multi_c:
        max_abs_diff = 0.0
        exceeds_5pct = False

        for fid in all_fids_eval:
            for col in snr_cols:
                vals = [snr_store[(C, band_tuple)][fid][col] for C in cs_for_band]
                abs_diff = max(vals) - min(vals)
                if abs_diff > max_abs_diff:
                    max_abs_diff = abs_diff
                min_val = min(vals)
                pct_diff = (abs_diff / min_val) * 100.0 if min_val > 1e-12 else 0.0
                if pct_diff >= 5.0:
                    exceeds_5pct = True

        out_line = f"Band {list(band_tuple)} across cutoffs {cs_for_band}: largest absolute SNR difference = {max_abs_diff:.4f}"
        if not exceeds_5pct:
            out_line += " -> C HAS LITTLE EFFECT ON THIS BAND"
        print(out_line)

    logger.close()
    sys.stdout = sys.__stdout__
    print(f"\nExecution finished successfully. Results logged to {summary_path}")


if __name__ == "__main__":
    main()
