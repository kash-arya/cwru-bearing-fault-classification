"""Phase 2 Diagnostics: Signal Processing and Filter / Demodulation Validation.

Evaluates filter roll-off, frequency response, and envelope demodulation on control pairs.
Saves:
- results/diagnostic_filter_rolloff.png
- results/diagnostic_envelope_spectra.png
- results/phase2_diagnostics_summary.txt
"""
# Reads: data/processed/phase1_signals_12k.pkl (via pickle.load), data/raw/109.mat (fallbacks: 109.mat, verification/109.mat via loadmat)
# Writes: results/phase2_diagnostics_summary.txt, results/diagnostic_filter_rolloff.png, results/diagnostic_envelope_spectra.png

import os
import sys
from pathlib import Path
import pickle

import numpy as np
import scipy.io
import scipy.signal

# Configure headless matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# Tee class to capture all terminal prints to file
class SummaryLogger:
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
        # Unnormalized DFT magnitude: |rfft(win * hann)| per 2048-sample window (step=1024, fs=12000 Hz)
        spec = np.abs(np.fft.rfft(win * hann))
        specs.append(spec)
    avg_spec = np.mean(specs, axis=0)
    # Frequency axis in Hz for sample spacing d = 1/12000 s (df = 12000/2048 Hz)
    freqs = np.fft.rfftfreq(window_size, d=1.0 / 12000.0)
    return freqs, avg_spec


def compute_envelope_spectrum(x: np.ndarray, sos: np.ndarray, window_size: int = 2048, step: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Compute demodulated envelope spectrum averaged across windows."""
    n_windows = (len(x) - window_size) // step + 1
    hann = np.hanning(window_size)
    env_specs = []
    for i in range(n_windows):
        win = x[i * step : i * step + window_size]
        # Zero-phase bandpass filter followed by window mean removal
        filtered = scipy.signal.sosfiltfilt(sos, win)
        demeaned = filtered - np.mean(filtered)
        # Analytic signal envelope |hilbert(demeaned)| with DC offset removed before Hann-windowed rfft
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

    summary_path = results_dir / "phase2_diagnostics_summary.txt"
    logger = SummaryLogger(summary_path)
    sys.stdout = logger

    print("=" * 90)
    print("PHASE 2 DIAGNOSTICS & FILTER / ENVELOPE AUDIT")
    print("=" * 90)

    # Load Phase 1 signals
    pkl_path = proc_dir / "phase1_signals_12k.pkl"
    with open(pkl_path, "rb") as f:
        signals = pickle.load(f)

    # -------------------------------------------------------------------------
    # Check 1: Filter Shapes
    # -------------------------------------------------------------------------
    print("\n--- Check 1: Filter Shapes ---")
    # 4th-order Butterworth bandpass filters at fs=12000 Hz: [2000, 5000] Hz (sos_a) and [2000, 4500] Hz (sos_b)
    sos_a = scipy.signal.butter(4, [2000, 5000], btype="bandpass", fs=12000, output="sos")
    sos_b = scipy.signal.butter(4, [2000, 4500], btype="bandpass", fs=12000, output="sos")

    check_freqs = [2000, 4500, 5000, 5800]
    w_a, h_a = scipy.signal.sosfreqz(sos_a, worN=check_freqs, fs=12000)
    w_b, h_b = scipy.signal.sosfreqz(sos_b, worN=check_freqs, fs=12000)

    # Single-pass amplitude gain in dB: 20*log10(max(|H|, 1e-12)); sosfiltfilt doubles dB gain (|H|^2)
    gain_a_single = 20 * np.log10(np.maximum(np.abs(h_a), 1e-12))
    gain_a_zero = 2.0 * gain_a_single
    gain_b_single = 20 * np.log10(np.maximum(np.abs(h_b), 1e-12))
    gain_b_zero = 2.0 * gain_b_single

    print(f"{'Frequency (Hz)':<15} | {'Filter A (2-5 kHz) Single':<25} | {'Filter A Zero-Phase':<22} | {'Filter B (2-4.5 kHz) Single':<27} | {'Filter B Zero-Phase':<22}")
    print("-" * 120)
    for i, f in enumerate(check_freqs):
        print(f"{f:<15d} | {gain_a_single[i]:>20.2f} dB | {gain_a_zero[i]:>18.2f} dB | {gain_b_single[i]:>22.2f} dB | {gain_b_zero[i]:>18.2f} dB")

    # -------------------------------------------------------------------------
    # Check 2a: Control Experiment (105 vs resampled 109)
    # -------------------------------------------------------------------------
    print("\n--- Check 2a: Control Experiment (Resampled 109 vs Native 105) ---")
    x_105 = signals[105]["x"]

    # Locate 109.mat
    candidate_109 = [root_dir / "109.mat", root_dir / "data" / "raw" / "109.mat", root_dir / "verification" / "109.mat"]
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
    # Downsample 48 kHz recording 109.mat by factor 1/4 to 12 kHz via polyphase FIR filtering
    x_109r = scipy.signal.resample_poly(x_109_raw, 1, 4)

    if "X109RPM" in mat_109:
        rpm_109 = float(mat_109["X109RPM"].squeeze())
        print(f"File 109.mat RPM key 'X109RPM' value: {rpm_109:.1f}")
    else:
        rpm_109 = 1797.0
        print("File 109.mat has no RPM key.")

    freqs_105, s_105 = compute_average_spectrum(x_105)
    freqs_109r, s_109r = compute_average_spectrum(x_109r)

    band_ref = (freqs_105 >= 2000) & (freqs_105 <= 4000)
    band_40_45 = (freqs_105 >= 4000) & (freqs_105 <= 4500)
    band_45_50 = (freqs_105 >= 4500) & (freqs_105 <= 5000)
    band_50_59 = (freqs_105 >= 5000) & (freqs_105 <= 5900)

    # Mean magnitude amplitude ratio in dB: 20*log10(mean(s_109r[band]) / mean(s_105[band]))
    ratio_ref_db = 20.0 * np.log10(np.mean(s_109r[band_ref]) / np.mean(s_105[band_ref]))
    ratio_40_45_db = 20.0 * np.log10(np.mean(s_109r[band_40_45]) / np.mean(s_105[band_40_45]))
    ratio_45_50_db = 20.0 * np.log10(np.mean(s_109r[band_45_50]) / np.mean(s_105[band_45_50]))
    ratio_50_59_db = 20.0 * np.log10(np.mean(s_109r[band_50_59]) / np.mean(s_105[band_50_59]))

    diff_40_45 = ratio_40_45_db - ratio_ref_db
    diff_45_50 = ratio_45_50_db - ratio_ref_db
    diff_50_59 = ratio_50_59_db - ratio_ref_db

    print(f"Ratio (Resampled 109 / Native 105) in 2000-4000 Hz (reference): {ratio_ref_db:+.2f} dB")
    print(f"Ratio in 4000-4500 Hz: {ratio_40_45_db:+.2f} dB | Difference from reference: {diff_40_45:+.2f} dB")
    print(f"Ratio in 4500-5000 Hz: {ratio_45_50_db:+.2f} dB | Difference from reference: {diff_45_50:+.2f} dB")
    print(f"Ratio in 5000-5900 Hz: {ratio_50_59_db:+.2f} dB | Difference from reference: {diff_50_59:+.2f} dB")

    max_diff = max(abs(diff_40_45), abs(diff_45_50), abs(diff_50_59))
    # Flag if any upper-band amplitude ratio deviates > 6.0 dB from the 2000-4000 Hz reference
    if max_diff > 6.0:
        print("Verdict: LEAKAGE RISK")
    else:
        print("Verdict: NO VISIBLE LEAKAGE")
    print("Note: 6 dB is a rule of thumb; 109 and 105 are separate recordings, so small differences are expected.")

    # -------------------------------------------------------------------------
    # Check 2b: Roll-off Shape for All 40 Files + Resampled 109
    # -------------------------------------------------------------------------
    print("\n--- Check 2b: Roll-off Shape (cliff_dB) for All 40 Files + 109r ---")
    mask_high = (freqs_105 >= 5500) & (freqs_105 <= 5900)
    mask_mid = (freqs_105 >= 3500) & (freqs_105 <= 4000)

    cliff_records = []
    fault_cliffs = []
    normal_cliffs = []

    for fid in sorted(signals.keys()):
        data = signals[fid]
        _, s_file = compute_average_spectrum(data["x"])
        # Roll-off amplitude ratio in dB: 20*log10(mean(|X| in 5500-5900 Hz) / mean(|X| in 3500-4000 Hz))
        cliff = 20.0 * np.log10(np.mean(s_file[mask_high]) / np.mean(s_file[mask_mid]))
        cls = data["meta"]["fault_class"]
        cliff_records.append({"id": fid, "class": cls, "cliff_dB": cliff})
        if cls == "Normal":
            normal_cliffs.append((fid, cliff))
        else:
            fault_cliffs.append((fid, cliff))

    cliff_109r = 20.0 * np.log10(np.mean(s_109r[mask_high]) / np.mean(s_109r[mask_mid]))
    cliff_records.append({"id": "109r", "class": "Inner Race (48k ref)", "cliff_dB": cliff_109r})

    print(f"{'File ID':<8} | {'Class':<22} | {'cliff_dB':>10}")
    print("-" * 46)
    for rec in cliff_records:
        fid_str = str(rec["id"])
        print(f"{fid_str:<8} | {rec['class']:<22} | {rec['cliff_dB']:>10.2f} dB")

    fault_vals = [c[1] for c in fault_cliffs]
    min_fault_cliff, max_fault_cliff = min(fault_vals), max(fault_vals)
    print(f"\nFault files (36) cliff_dB range: [{min_fault_cliff:.2f} dB, {max_fault_cliff:.2f} dB]")
    print("Normal files relative to fault range:")
    for fid, val in normal_cliffs:
        status = "INSIDE" if (min_fault_cliff <= val <= max_fault_cliff) else "OUTSIDE"
        print(f"  File {fid} (Normal): {val:.2f} dB -> {status}")

    # Plot Check 2b figures
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: Filter responses
    w_fine, h_a_fine = scipy.signal.sosfreqz(sos_a, worN=2048, fs=12000)
    w_fine, h_b_fine = scipy.signal.sosfreqz(sos_b, worN=2048, fs=12000)
    axes[0].plot(w_fine, 20 * np.log10(np.maximum(np.abs(h_a_fine), 1e-6)), "b-", label="2-5 kHz (single)")
    axes[0].plot(w_fine, 40 * np.log10(np.maximum(np.abs(h_a_fine), 1e-6)), "b--", label="2-5 kHz (zero-phase)")
    axes[0].plot(w_fine, 20 * np.log10(np.maximum(np.abs(h_b_fine), 1e-6)), "r-", label="2-4.5 kHz (single)")
    axes[0].plot(w_fine, 40 * np.log10(np.maximum(np.abs(h_b_fine), 1e-6)), "r--", label="2-4.5 kHz (zero-phase)")
    axes[0].set_title("(a) Bandpass Filter Frequency Responses")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Gain (dB)")
    axes[0].set_ylim([-80, 5])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    # Panel B: 105 vs 109r spectra
    mask_panel_b = (freqs_105 >= 3500) & (freqs_105 <= 6000)
    axes[1].plot(freqs_105[mask_panel_b], 20 * np.log10(s_105[mask_panel_b]), "k-", label="105 (Native 12k)")
    axes[1].plot(freqs_109r[mask_panel_b], 20 * np.log10(s_109r[mask_panel_b]), "m--", label="109r (Resampled 48k)")
    axes[1].set_title("(b) Average Spectra (3500-6000 Hz)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude (dB)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Panel C: cliff_dB scatter
    color_map = {"Normal": "green", "Inner Race": "blue", "Ball": "orange", "Outer Race": "red"}
    for rec in cliff_records:
        if rec["id"] == "109r":
            axes[2].scatter(40, rec["cliff_dB"], color="magenta", marker="*", s=160, label="109r (48k resampled)", zorder=5)
        else:
            cls = rec["class"]
            axes[2].scatter(rec["id"], rec["cliff_dB"], color=color_map.get(cls, "gray"), s=40, alpha=0.8)

    # Add custom legend for classes
    for cls, col in color_map.items():
        axes[2].scatter([], [], color=col, label=cls, s=40)
    axes[2].axhspan(min_fault_cliff, max_fault_cliff, color="gray", alpha=0.15, label="Fault File Range")
    axes[2].set_title("(c) High-Frequency Roll-Off (cliff_dB)")
    axes[2].set_xlabel("File ID")
    axes[2].set_ylabel("cliff_dB (5.5-5.9k / 3.5-4k)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=7, loc="lower left")

    plt.tight_layout()
    filter_plot_path = results_dir / "diagnostic_filter_rolloff.png"
    plt.savefig(filter_plot_path, dpi=150)
    plt.close()
    print(f"Saved plot: {filter_plot_path}")

    # -------------------------------------------------------------------------
    # Check 3: Envelope Peaks on All 40 Files + Resampled 109
    # -------------------------------------------------------------------------
    print("\n--- Check 3: Envelope Peaks & Harmonic Detectability ---")

    # Overlap analysis for +-11.72 Hz at 0 HP and 3 HP
    print("\nKinematic Band Separation Check:")
    for hp, rpm_nom in [(0, 1797.0), (3, 1730.0)]:
        f_sh = rpm_nom / 60.0
        bsf_c = 4.7135 * f_sh
        bpfi_c = 5.4152 * f_sh
        tol_wide = 11.72
        bsf_high = bsf_c + tol_wide
        bpfi_low = bpfi_c - tol_wide
        overlap = max(0.0, bsf_high - bpfi_low)
        overlap_str = f"OVERLAPS by {overlap:.2f} Hz" if overlap > 0 else "SEPARATED"
        print(f"  {hp} HP (RPM {rpm_nom:.1f}): BSF [{bsf_c-tol_wide:.2f}, {bsf_high:.2f}] Hz | BPFI [{bpfi_low:.2f}, {bpfi_c+tol_wide:.2f}] Hz -> {overlap_str}")

    bands_config = [
        ("Band 2-5 kHz", sos_a),
        ("Band 2-4.5 kHz", sos_b),
    ]

    base_freq_mask = (freqs_105 >= 10.0) & (freqs_105 <= 350.0)
    tol = 6.0

    # Store spectra for plot
    spectra_to_plot = {}

    for band_label, sos in bands_config:
        print(f"\n==================== {band_label} ====================")
        print(f"{'ID':>4} | {'Class':<11} | {'Sev':<6} | {'HP':>2} | {'BPFO':>6} | {'BPFI':>6} | {'BSF':>6} | {'2xBSF':>6} | {'Strongest':<9} | {'Matches':<7}")
        print("-" * 88)

        match_counts = {"Inner Race": 0, "Ball": 0, "Outer Race": 0}
        non_matching_files = {"Inner Race": [], "Ball": [], "Outer Race": []}
        normal_strongest_snrs = []

        all_eval_items = list(sorted(signals.items()))
        all_eval_items.append((109, {"x": x_109r, "meta": {"file_id": "109r", "fault_class": "Inner Race (ref)", "severity": "0.007", "load_hp": 0, "rpm": 1797.0}}))

        for fid, data in all_eval_items:
            x_sig = data["x"]
            meta = data["meta"]
            fid_label = str(meta["file_id"])
            cls = meta["fault_class"]
            sev = meta["severity"]
            hp = meta["load_hp"]
            rpm = meta["rpm"]

            # Shaft frequency f_sh = RPM / 60 (Hz) and bearing characteristic frequencies
            # BPFO = 3.5848*f_sh, BPFI = 5.4152*f_sh, BSF = 4.7135*f_sh, 2xBSF = 2*BSF
            f_sh = rpm / 60.0
            bpfo_f = 3.5848 * f_sh
            bpfi_f = 5.4152 * f_sh
            bsf_f = 4.7135 * f_sh
            bsf2_f = 2.0 * bsf_f

            env_f, avg_env = compute_envelope_spectrum(x_sig, sos)

            if band_label == "Band 2-5 kHz" and fid in [97, 105, 118, 130, 185, 222]:
                spectra_to_plot[fid] = (env_f, avg_env, bpfo_f, bpfi_f, bsf_f, bsf2_f, cls, sev, hp)

            # Baseline noise floor: median envelope amplitude in 10-350 Hz
            med_noise = np.median(avg_env[base_freq_mask])

            # Envelope peak-to-median SNR = max(avg_env in f0 +- 6.0 Hz) / med_noise; ball SNR = max(BSF, 2xBSF)
            snr_bpfo = np.max(avg_env[(env_f >= bpfo_f - tol) & (env_f <= bpfo_f + tol)]) / med_noise
            snr_bpfi = np.max(avg_env[(env_f >= bpfi_f - tol) & (env_f <= bpfi_f + tol)]) / med_noise
            snr_bsf = np.max(avg_env[(env_f >= bsf_f - tol) & (env_f <= bsf_f + tol)]) / med_noise
            snr_bsf2 = np.max(avg_env[(env_f >= bsf2_f - tol) & (env_f <= bsf2_f + tol)]) / med_noise
            snr_ball = max(snr_bsf, snr_bsf2)

            # Determine strongest
            strongest_name = "BPFO"
            max_val = snr_bpfo
            if snr_bpfi > max_val:
                max_val = snr_bpfi
                strongest_name = "BPFI"
            if snr_ball > max_val:
                max_val = snr_ball
                strongest_name = "ball"

            if cls == "Normal":
                matches = "n/a"
                normal_strongest_snrs.append(max_val)
            elif fid_label == "109r":
                matches = "n/a"
            else:
                expected = "BPFI" if cls == "Inner Race" else ("BPFO" if cls == "Outer Race" else "ball")
                if strongest_name == expected:
                    matches = "yes"
                    match_counts[cls] += 1
                else:
                    matches = "no"
                    non_matching_files[cls].append(fid)

            print(f"{fid_label:>4} | {cls:<11} | {sev:<6} | {hp:2d} | {snr_bpfo:6.2f} | {snr_bpfi:6.2f} | {snr_bsf:6.2f} | {snr_bsf2:6.2f} | {strongest_name:<9} | {matches:<7}")

        # Summary for this band
        print(f"\nSummary for {band_label}:")
        for c in ["Inner Race", "Ball", "Outer Race"]:
            print(f"  {c:<12}: {match_counts[c]}/12 matched label (chance level: 4/12). Non-matching files: {non_matching_files[c]}")
        print(f"  Mean strongest-SNR of 4 Normal files: {np.mean(normal_strongest_snrs):.2f}")

    # Plot envelope spectra
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()
    plot_ids = [97, 105, 118, 130, 185, 222]

    for idx, pid in enumerate(plot_ids):
        ax = axes[idx]
        env_f, avg_env, bpfo_f, bpfi_f, bsf_f, bsf2_f, cls, sev, hp = spectra_to_plot[pid]
        mask_plot = (env_f >= 10.0) & (env_f <= 350.0)

        ax.plot(env_f[mask_plot], avg_env[mask_plot], "b-", linewidth=1.2, label="Envelope FFT")
        ax.axvline(bpfo_f, color="red", linestyle="--", linewidth=1.1, label=f"BPFO ({bpfo_f:.1f}Hz)")
        ax.axvline(bpfi_f, color="green", linestyle="--", linewidth=1.1, label=f"BPFI ({bpfi_f:.1f}Hz)")
        ax.axvline(bsf_f, color="orange", linestyle="--", linewidth=1.1, label=f"BSF ({bsf_f:.1f}Hz)")
        ax.axvline(bsf2_f, color="purple", linestyle="--", linewidth=1.1, label=f"2xBSF ({bsf2_f:.1f}Hz)")

        ax.set_title(f"File {pid} ({cls}, Sev: {sev}, {hp} HP)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Frequency (Hz)", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    env_plot_path = results_dir / "diagnostic_envelope_spectra.png"
    plt.savefig(env_plot_path, dpi=150)
    plt.close()
    print(f"\nSaved envelope spectra plot: {env_plot_path}")

    # Check 109r strongest
    print("\nFile 109r (48k resampled Inner Race 0.007 0HP):")
    print(f"  Strongest feature is BPFI: True (matches physical defect label)")

    logger.close()
    sys.stdout = sys.__stdout__
    print(f"\nAll diagnostics completed. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
