"""Phase 3: Dataset Splitting & Leakage-Free Evaluation Scheme.

Constructs:
A. Outer Leave-One-Load-Out (LOLO):
   4 splits: lolo_0, lolo_1, lolo_2, lolo_3.
   Inner folds: Leave-one-load-out across the 3 training loads.
B. Outer Cross-Severity:
   3 splits: sev_0.007, sev_0.014, sev_0.021.
   Inner folds: 3 folds held out by load l in [0, 1, 2] ensuring 4-class balance.

Outputs:
- data/processed/phase3_splits.json
- results/phase3_splits_summary.csv
"""
# Reads: results/phase1_file_manifest.csv, data/processed/phase2_features_lp3500.csv, data/processed/phase2_features_nolp.csv (pd.read_csv)
# Writes: data/processed/phase3_splits.json (json.dump), results/phase3_splits_summary.csv (to_csv)

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

CLASSES = ["Normal", "Inner Race", "Ball", "Outer Race"]


def normalize_severity(sev_series: pd.Series) -> pd.Series:
    """Normalize severity strings by stripping quotes and inch marks."""
    return sev_series.astype(str).str.strip("\"'")


def main():
    root_dir = Path(__file__).resolve().parent.parent
    proc_dir = root_dir / "data" / "processed"
    results_dir = root_dir / "results"
    proc_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 3: SPLIT GENERATION & VALIDATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Read & Verify Manifest
    # -------------------------------------------------------------------------
    manifest_path = results_dir / "phase1_file_manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    manifest["file_id"] = manifest["file_id"].astype(int)
    manifest["load_hp"] = manifest["load_hp"].astype(int)
    manifest["window_count"] = manifest["window_count"].astype(int)
    manifest["severity"] = normalize_severity(manifest["severity"])

    unique_severities = sorted(manifest["severity"].unique().tolist())
    print(f"Unique severities found in manifest: {unique_severities}")
    if unique_severities != ["0.007", "0.014", "0.021", "none"]:
        print(f"[ERROR] Unexpected severities: {unique_severities}")
        sys.exit(1)

    file_meta = manifest.set_index("file_id")
    all_fids = set(manifest["file_id"].tolist())
    if len(all_fids) != 40:
        print(f"[ERROR] Expected 40 unique file IDs, got {len(all_fids)}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 2. Verify Feature Files Metadata
    # -------------------------------------------------------------------------
    for feat_name, feat_path in [
        ("lp3500", proc_dir / "phase2_features_lp3500.csv"),
        ("nolp", proc_dir / "phase2_features_nolp.csv"),
    ]:
        print(f"Verifying metadata consistency for {feat_name} ({feat_path})...")
        feat_df = pd.read_csv(
            feat_path,
            usecols=["file_id", "fault_class", "severity", "load_hp"],
            dtype=str,
            keep_default_na=False,
        )
        feat_df["file_id"] = feat_df["file_id"].astype(int)
        feat_df["load_hp"] = feat_df["load_hp"].astype(int)
        feat_df["severity"] = normalize_severity(feat_df["severity"])

        feat_fids = set(feat_df["file_id"].unique())
        if feat_fids != all_fids:
            print(f"[ERROR] File ID mismatch in {feat_name}")
            sys.exit(1)

        for fid, group in feat_df.groupby("file_id"):
            m_row = file_meta.loc[fid]
            if len(group) != m_row["window_count"]:
                print(f"[ERROR] Window count mismatch for file {fid} in {feat_name}: {len(group)} vs {m_row['window_count']}")
                sys.exit(1)
            if group["fault_class"].iloc[0] != m_row["fault_class"]:
                print(f"[ERROR] Class mismatch for file {fid} in {feat_name}")
                sys.exit(1)
            if group["load_hp"].iloc[0] != m_row["load_hp"]:
                print(f"[ERROR] Load mismatch for file {fid} in {feat_name}")
                sys.exit(1)
            if group["severity"].iloc[0] != m_row["severity"]:
                print(f"[ERROR] Severity mismatch for file {fid} in {feat_name}")
                sys.exit(1)

    print("Manifest and feature files verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Generate Outer and Inner Splits
    # -------------------------------------------------------------------------
    splits = {}

    # A. Outer Leave-One-Load-Out (LOLO)
    # Partition by motor load_hp in [0, 1, 2, 3]: 10 test files (3 per fault class + 1 Normal), 30 outer train files
    for L in [0, 1, 2, 3]:
        s_name = f"lolo_{L}"
        test_f = sorted([int(fid) for fid in manifest[manifest["load_hp"] == L]["file_id"].tolist()])
        train_f = sorted([int(fid) for fid in manifest[manifest["load_hp"] != L]["file_id"].tolist()])

        # Inner CV holds out one load l from the 3 training loads (10 val files, 20 inner train files)
        inner_folds = []
        train_loads = sorted([x for x in [0, 1, 2, 3] if x != L])
        for l in train_loads:
            val_f = sorted([int(fid) for fid in manifest[manifest["load_hp"] == l]["file_id"].tolist()])
            in_train_f = sorted([int(fid) for fid in train_f if fid not in val_f])
            inner_folds.append({
                "name": f"val_load_{l}",
                "train_files": in_train_f,
                "val_files": val_f,
            })

        splits[s_name] = {
            "name": s_name,
            "description": f"Outer Leave-One-Load-Out: test on load {L} HP (10 files), train on remaining 3 loads (30 files)",
            "train_files": train_f,
            "test_files": test_f,
            "inner_folds": inner_folds,
        }

    # B. Outer Cross-Severity
    rules_text = (
        "Cross-severity inner validation rule: Every validation fold must contain all four classes, and Normal has no severity label. "
        "Therefore, for each held-out load l in [0, 1, 2], the inner validation set consists of the 6 fault files of the two training "
        "severities at load l plus the 1 Normal file at load l (7 validation files total), while inner train contains the remaining 20 files "
        "of the outer train set (including the 6 load-3 fault files of the training severities). "
        "Limitation: Hyperparameter tuning across these inner folds optimizes for operating load shift, not for crack severity shift."
    )
    print(f"\n[RULES]: {rules_text}\n")

    for s in ["0.007", "0.014", "0.021"]:
        s_name = f"sev_{s}"
        # Outer test: 12 fault files at severity s + load-3 Normal file (file 100, 13 test files total)
        fault_test = manifest[(manifest["fault_class"] != "Normal") & (manifest["severity"] == s)]["file_id"].tolist()
        norm_test = manifest[(manifest["fault_class"] == "Normal") & (manifest["load_hp"] == 3)]["file_id"].tolist()
        test_f = sorted([int(fid) for fid in (fault_test + norm_test)])

        # Outer train: 24 remaining fault files + loads 0, 1, 2 Normal files (27 train files total)
        fault_train = manifest[(manifest["fault_class"] != "Normal") & (manifest["severity"] != s)]["file_id"].tolist()
        norm_train = manifest[(manifest["fault_class"] == "Normal") & (manifest["load_hp"].isin([0, 1, 2]))]["file_id"].tolist()
        train_f = sorted([int(fid) for fid in (fault_train + norm_train)])

        # Inner CV holds out load l in [0, 1, 2] across training severities + Normal (7 val files, 20 train files)
        inner_folds = []
        for l in [0, 1, 2]:
            val_fault = manifest[(manifest["fault_class"] != "Normal") & (manifest["severity"] != s) & (manifest["load_hp"] == l)]["file_id"].tolist()
            val_norm = manifest[(manifest["fault_class"] == "Normal") & (manifest["load_hp"] == l)]["file_id"].tolist()
            val_f = sorted([int(fid) for fid in (val_fault + val_norm)])
            in_train_f = sorted([int(fid) for fid in train_f if fid not in val_f])
            inner_folds.append({
                "name": f"val_load_{l}",
                "train_files": in_train_f,
                "val_files": val_f,
            })

        splits[s_name] = {
            "name": s_name,
            "description": f"Outer Cross-Severity: test on severity {s}\" (12 fault files + load 3 Normal 100.mat), train on remaining severities + loads 0-2 Normal",
            "train_files": train_f,
            "test_files": test_f,
            "inner_folds": inner_folds,
        }

    # -------------------------------------------------------------------------
    # 4. Assertions (stop on failure)
    # -------------------------------------------------------------------------
    for s_name, s_data in splits.items():
        tr = set(s_data["train_files"])
        te = set(s_data["test_files"])

        # Assertion 1: No file in both train and test
        if len(tr & te) > 0:
            print(f"[ASSERTION FAILED] File in both train and test in {s_name}: {tr & te}")
            sys.exit(1)

        # Assertion 2: Union of train and test is exactly the 40 files
        if (tr | te) != all_fids:
            print(f"[ASSERTION FAILED] Train | Test != 40 files in {s_name}")
            sys.exit(1)

        # Assertion 3: All 4 classes present in test
        te_classes = {file_meta.loc[f, "fault_class"] for f in te}
        if not set(CLASSES).issubset(te_classes):
            print(f"[ASSERTION FAILED] Test set missing class in {s_name}: {set(CLASSES) - te_classes}")
            sys.exit(1)

        # Assertion 4: Expected sizes
        if s_name.startswith("lolo"):
            if len(te) != 10 or len(tr) != 30:
                print(f"[ASSERTION FAILED] LOLO outer file count wrong in {s_name}: test={len(te)}, train={len(tr)}")
                sys.exit(1)
            for c in ["Inner Race", "Ball", "Outer Race"]:
                if sum(file_meta.loc[f, "fault_class"] == c for f in te) != 3 or sum(file_meta.loc[f, "fault_class"] == c for f in tr) != 9:
                    print(f"[ASSERTION FAILED] LOLO class count wrong for {c} in {s_name}")
                    sys.exit(1)
            if sum(file_meta.loc[f, "fault_class"] == "Normal" for f in te) != 1 or sum(file_meta.loc[f, "fault_class"] == "Normal" for f in tr) != 3:
                print(f"[ASSERTION FAILED] LOLO Normal count wrong in {s_name}")
                sys.exit(1)
        else:
            if len(te) != 13 or len(tr) != 27:
                print(f"[ASSERTION FAILED] Cross-Severity outer file count wrong in {s_name}: test={len(te)}, train={len(tr)}")
                sys.exit(1)
            if te.intersection(manifest[manifest["fault_class"] == "Normal"]["file_id"]) != {100}:
                print(f"[ASSERTION FAILED] Cross-Severity test Normal file != 100 in {s_name}")
                sys.exit(1)
            for c in ["Inner Race", "Ball", "Outer Race"]:
                if sum(file_meta.loc[f, "fault_class"] == c for f in te) != 4 or sum(file_meta.loc[f, "fault_class"] == c for f in tr) != 8:
                    print(f"[ASSERTION FAILED] Cross-Severity class count wrong for {c} in {s_name}")
                    sys.exit(1)
            if sum(file_meta.loc[f, "fault_class"] == "Normal" for f in te) != 1 or sum(file_meta.loc[f, "fault_class"] == "Normal" for f in tr) != 3:
                print(f"[ASSERTION FAILED] Cross-Severity Normal count wrong in {s_name}")
                sys.exit(1)

        # Assertion 5: Inner folds
        val_sets = []
        for fold in s_data["inner_folds"]:
            i_tr = set(fold["train_files"])
            i_va = set(fold["val_files"])

            # Assertion 1 for inner: disjoint
            if len(i_tr & i_va) > 0:
                print(f"[ASSERTION FAILED] Inner train & val overlap in {s_name} {fold['name']}: {i_tr & i_va}")
                sys.exit(1)

            # Assertion 5 part 1: union is outer train
            if (i_tr | i_va) != tr:
                print(f"[ASSERTION FAILED] Inner train | val != outer train in {s_name} {fold['name']}")
                sys.exit(1)

            # Assertion 3 for inner: all 4 classes present
            va_classes = {file_meta.loc[f, "fault_class"] for f in i_va}
            if not set(CLASSES).issubset(va_classes):
                print(f"[ASSERTION FAILED] Inner val missing class in {s_name} {fold['name']}")
                sys.exit(1)

            # Assertion 4 for inner: sizes
            if s_name.startswith("lolo"):
                if len(i_va) != 10 or len(i_tr) != 20:
                    print(f"[ASSERTION FAILED] LOLO inner sizes wrong in {s_name} {fold['name']}")
                    sys.exit(1)
                for c in ["Inner Race", "Ball", "Outer Race"]:
                    if sum(file_meta.loc[f, "fault_class"] == c for f in i_va) != 3 or sum(file_meta.loc[f, "fault_class"] == c for f in i_tr) != 6:
                        print(f"[ASSERTION FAILED] LOLO inner class count wrong in {s_name} {fold['name']}")
                        sys.exit(1)
                if sum(file_meta.loc[f, "fault_class"] == "Normal" for f in i_va) != 1 or sum(file_meta.loc[f, "fault_class"] == "Normal" for f in i_tr) != 2:
                    print(f"[ASSERTION FAILED] LOLO inner Normal count wrong in {s_name} {fold['name']}")
                    sys.exit(1)
            else:
                if len(i_va) != 7 or len(i_tr) != 20:
                    print(f"[ASSERTION FAILED] Cross-Severity inner sizes wrong in {s_name} {fold['name']}: val={len(i_va)}, tr={len(i_tr)}")
                    sys.exit(1)
                for c in ["Inner Race", "Ball", "Outer Race"]:
                    if sum(file_meta.loc[f, "fault_class"] == c for f in i_va) != 2 or sum(file_meta.loc[f, "fault_class"] == c for f in i_tr) != 6:
                        print(f"[ASSERTION FAILED] Cross-Severity inner class count wrong in {s_name} {fold['name']}")
                        sys.exit(1)
                if sum(file_meta.loc[f, "fault_class"] == "Normal" for f in i_va) != 1 or sum(file_meta.loc[f, "fault_class"] == "Normal" for f in i_tr) != 2:
                    print(f"[ASSERTION FAILED] Cross-Severity inner Normal count wrong in {s_name} {fold['name']}")
                    sys.exit(1)

            val_sets.append(i_va)

        # Assertion 5 part 2: pairwise disjoint inner val sets
        if len(val_sets[0] & val_sets[1]) > 0 or len(val_sets[0] & val_sets[2]) > 0 or len(val_sets[1] & val_sets[2]) > 0:
            print(f"[ASSERTION FAILED] Inner val sets not pairwise disjoint in {s_name}")
            sys.exit(1)

    print("ALL ASSERTIONS PASSED.")

    # -------------------------------------------------------------------------
    # 5. Save phase3_splits.json
    # -------------------------------------------------------------------------
    json_path = proc_dir / "phase3_splits.json"
    json_payload = {
        "rules": rules_text,
    }
    for s_name in ["lolo_0", "lolo_1", "lolo_2", "lolo_3", "sev_0.007", "sev_0.014", "sev_0.021"]:
        json_payload[s_name] = splits[s_name]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)
    print(f"Saved splits JSON: {json_path}")

    # -------------------------------------------------------------------------
    # 6. Save results/phase3_splits_summary.csv
    # -------------------------------------------------------------------------
    def get_row_stats(split_name, kind, inner_fold, set_name, fids):
        sub = file_meta.loc[fids]
        n_files = int(len(fids))
        n_windows = int(sub["window_count"].sum())

        f_counts = {}
        w_counts = {}
        for c in CLASSES:
            c_sub = sub[sub["fault_class"] == c]
            f_counts[c] = int(len(c_sub))
            w_counts[c] = int(c_sub["window_count"].sum())

        return {
            "split": split_name,
            "kind": kind,
            "inner_fold": inner_fold,
            "set": set_name,
            "n_files_total": n_files,
            "n_windows_total": n_windows,
            "files_Normal": f_counts["Normal"],
            "files_Inner Race": f_counts["Inner Race"],
            "files_Ball": f_counts["Ball"],
            "files_Outer Race": f_counts["Outer Race"],
            "windows_Normal": w_counts["Normal"],
            "windows_Inner Race": w_counts["Inner Race"],
            "windows_Ball": w_counts["Ball"],
            "windows_Outer Race": w_counts["Outer Race"],
        }

    summary_rows = []
    outer_split_order = ["lolo_0", "lolo_1", "lolo_2", "lolo_3", "sev_0.007", "sev_0.014", "sev_0.021"]

    for s_name in outer_split_order:
        s_data = splits[s_name]

        # Outer train and test
        summary_rows.append(get_row_stats(s_name, "outer", "", "train", s_data["train_files"]))
        summary_rows.append(get_row_stats(s_name, "outer", "", "test", s_data["test_files"]))

        # Inner train and val
        for fold in s_data["inner_folds"]:
            summary_rows.append(get_row_stats(s_name, "inner", fold["name"], "train", fold["train_files"]))
            summary_rows.append(get_row_stats(s_name, "inner", fold["name"], "val", fold["val_files"]))

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = results_dir / "phase3_splits_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved splits summary CSV: {summary_csv_path}")
    print(f"Summary table shape: {summary_df.shape} (expect 56 rows, 14 columns)")


if __name__ == "__main__":
    main()
