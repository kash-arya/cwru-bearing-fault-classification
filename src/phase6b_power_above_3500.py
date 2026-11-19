from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import scipy.signal

np.random.seed(42)

def main():
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data" / "processed"
    results_dir = root_dir / "results"

    with open(data_dir / "phase1_signals_12k.pkl", "rb") as f:
        signals = pickle.load(f)

    manifest_df = pd.read_csv(results_dir / "phase1_file_manifest.csv", keep_default_na=False)
    manifest_df["file_id"] = manifest_df["file_id"].astype(int)
    manifest_df = manifest_df.set_index("file_id")

    lp_sos = scipy.signal.butter(8, 3500, btype="lowpass", fs=12000, output="sos")

    rows = []
    for fid in sorted(signals.keys()):
        raw_x = signals[fid]["x"]
        x = raw_x - np.mean(raw_x)
        P_pre = float(np.mean(x ** 2))
        x_lp = scipy.signal.sosfiltfilt(lp_sos, x)
        P_post = float(np.mean(x_lp ** 2))
        frac_removed = float(1.0 - P_post / P_pre)

        freqs, psd = scipy.signal.welch(x, fs=12000, nperseg=2048)
        tot_p = float(np.sum(psd))

        f_above_3500 = float(np.sum(psd[freqs >= 3500.0]) / tot_p)
        f_3500_4000 = float(np.sum(psd[(freqs >= 3500.0) & (freqs <= 4000.0)]) / tot_p)
        f_4000_5900 = float(np.sum(psd[(freqs >= 4000.0) & (freqs <= 5900.0)]) / tot_p)

        meta = manifest_df.loc[fid]
        rows.append({
            "file_id": int(fid),
            "fault_class": str(meta["fault_class"]),
            "severity": str(meta["severity"]) if pd.notna(meta["severity"]) else "none",
            "load_hp": int(meta["load_hp"]),
            "frac_removed_by_lp": frac_removed,
            "frac_above_3500": f_above_3500,
            "frac_3500_4000": f_3500_4000,
            "frac_4000_5900": f_4000_5900,
        })

    df = pd.DataFrame(rows)
    out_csv = results_dir / "phase6b_power_above_3500.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} ({len(df)} rows)")

    # Print sorted table descending by frac_removed_by_lp
    df_sorted = df.sort_values("frac_removed_by_lp", ascending=False).reset_index(drop=True)
    print("\n--- Power Above 3500 Hz per File (Sorted by frac_removed_by_lp Descending) ---")
    print(df_sorted.to_string(index=False))

    # Mean per fault_class
    print("\n--- Mean Power Fractions by Fault Class ---")
    mean_class = df.groupby("fault_class")[["frac_removed_by_lp", "frac_above_3500", "frac_3500_4000", "frac_4000_5900"]].mean().reset_index()
    print(mean_class.to_string(index=False))

    # Mean per (fault_class, severity)
    print("\n--- Mean Power Fractions by Fault Class and Severity ---")
    mean_class_sev = df.groupby(["fault_class", "severity"])[["frac_removed_by_lp", "frac_above_3500", "frac_3500_4000", "frac_4000_5900"]].mean().reset_index()
    print(mean_class_sev.to_string(index=False))

if __name__ == "__main__":
    main()
