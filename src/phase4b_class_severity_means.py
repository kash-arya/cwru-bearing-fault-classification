"""Phase 4b: Feature Means by Fault Class, Severity, and Load

Computes mean physical features per fault_class and severity,
and mean RMS per fault_class, severity, and load_hp.
"""
# Reads: data/processed/phase2_features_lp3500.csv (pd.read_csv)
# Writes: results/phase4b_class_severity_means.csv (to_csv)

from pathlib import Path
import pandas as pd


def main():
    root = Path(__file__).resolve().parent.parent
    csv_path = root / "data" / "processed" / "phase2_features_lp3500.csv"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, keep_default_na=False)

    features = [
        "rms",
        "kurtosis",
        "peak_to_peak",
        "crest_factor",
        "spectral_centroid",
        "env_snr_bpfi",
        "env_snr_bpfo",
        "env_snr_bsf",
    ]

    class_order = {"Normal": 0, "Inner Race": 1, "Ball": 2, "Outer Race": 3}
    sev_order = {"none": 0, "0.007": 1, "0.014": 2, "0.021": 3}

    # Table 1: Feature means per fault_class and severity
    # Arithmetic mean of each of the 8 selected features across all windows in each (fault_class, severity) group
    t1 = df.groupby(["fault_class", "severity"], as_index=False)[features].mean()
    t1["c_ord"] = t1["fault_class"].map(class_order)
    t1["s_ord"] = t1["severity"].map(sev_order)
    t1 = t1.sort_values(["c_ord", "s_ord"]).drop(columns=["c_ord", "s_ord"]).reset_index(drop=True)

    out_csv = results_dir / "phase4b_class_severity_means.csv"
    t1.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    print("\nTable 1: Mean Features per Fault Class and Severity:")
    print(t1.to_string(index=False))

    # Table 2: RMS means per fault_class, severity, and load_hp
    # Arithmetic mean of window RMS across each (fault_class, severity, load_hp) group
    t2 = df.groupby(["fault_class", "severity", "load_hp"], as_index=False)["rms"].mean()
    t2["c_ord"] = t2["fault_class"].map(class_order)
    t2["s_ord"] = t2["severity"].map(sev_order)
    t2 = t2.sort_values(["c_ord", "s_ord", "load_hp"]).drop(columns=["c_ord", "s_ord"]).reset_index(drop=True)

    print("\nTable 2: Mean RMS per Fault Class, Severity, and Load HP:")
    print(t2.to_string(index=False))


if __name__ == "__main__":
    main()
