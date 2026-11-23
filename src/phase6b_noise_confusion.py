"""Phase 6b: Four-Class Confusion Matrices Under Additive Sensor Noise.

Evaluates clean-trained classifiers under 10 dB and 5 dB Gaussian noise across LOLO splits.
"""

from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, confusion_matrix

import sys
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from phase2_features import ALL_FEATURES
from phase4_train_eval import CorrelationFilter, CLASS_NAMES, CLASS_TO_IDX

np.random.seed(42)

def main():
    data_dir = root_dir / "data" / "processed"
    results_dir = root_dir / "results"
    cache_dir = data_dir / "phase6_cache" / "features"

    df_clean = pd.read_csv(data_dir / "phase2_features_lp3500.csv", keep_default_na=False)
    with open(data_dir / "phase3_splits.json", "r") as f:
        splits_json = json.load(f)
    with open(data_dir / "phase4_best_params.json", "r") as f:
        best_params = json.load(f)
    df_p6_metrics = pd.read_csv(results_dir / "phase6_metrics.csv")

    LOLO_SPLITS = ["lolo_0", "lolo_1", "lolo_2", "lolo_3"]
    MODELS = ["svm", "rf", "xgb"]
    models = {}

    print("Refitting clean models on LOLO training windows and verifying parity...")
    for s_name in LOLO_SPLITS:
        s_data = splits_json[s_name]
        tr_df = df_clean[df_clean["file_id"].isin(s_data["train_files"])]
        te_df = df_clean[df_clean["file_id"].isin(s_data["test_files"])]

        X_tr = tr_df[ALL_FEATURES].values
        y_tr = np.array([CLASS_TO_IDX[c] for c in tr_df["fault_class"]], dtype=int)
        X_te = te_df[ALL_FEATURES].values
        y_te = np.array([CLASS_TO_IDX[c] for c in te_df["fault_class"]], dtype=int)

        # 1. SVM
        p_svm = best_params["svm"][s_name]["best_params"]
        svm = Pipeline([
            ("corr", CorrelationFilter(0.95)),
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=p_svm["C"], gamma=p_svm["gamma"], class_weight="balanced", random_state=42))
        ])
        svm.fit(X_tr, y_tr)
        f1_svm = float(f1_score(y_te, svm.predict(X_te), average="macro", zero_division=0.0))
        ref_svm = float(df_p6_metrics[
            (df_p6_metrics["split"] == s_name) &
            (df_p6_metrics["model"] == "svm") &
            (df_p6_metrics["snr_db"] == "clean") &
            (df_p6_metrics["variant"] == "variant_A")
        ]["macro_f1"].iloc[0])
        assert abs(f1_svm - ref_svm) < 0.001, f"SVM f1 mismatch in {s_name}: {f1_svm:.4f} vs {ref_svm:.4f}"
        models[("svm", s_name)] = svm

        # 2. RF
        p_rf = best_params["rf"][s_name]["best_params"]
        rf = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=p_rf["min_samples_leaf"],
            max_depth=p_rf["max_depth"], class_weight="balanced", random_state=42, n_jobs=-1
        )
        rf.fit(X_tr, y_tr)
        f1_rf = float(f1_score(y_te, rf.predict(X_te), average="macro", zero_division=0.0))
        ref_rf = float(df_p6_metrics[
            (df_p6_metrics["split"] == s_name) &
            (df_p6_metrics["model"] == "rf") &
            (df_p6_metrics["snr_db"] == "clean") &
            (df_p6_metrics["variant"] == "variant_A")
        ]["macro_f1"].iloc[0])
        assert abs(f1_rf - ref_rf) < 0.001, f"RF f1 mismatch in {s_name}: {f1_rf:.4f} vs {ref_rf:.4f}"
        models[("rf", s_name)] = rf

        # 3. XGBoost
        p_xgb = best_params["xgb"][s_name]["best_params"]
        sw = compute_sample_weight("balanced", y_tr)
        xgb = XGBClassifier(
            n_estimators=200, max_depth=p_xgb["max_depth"],
            learning_rate=p_xgb["learning_rate"], tree_method="hist",
            eval_metric="mlogloss", random_state=42, n_jobs=-1
        )
        xgb.fit(X_tr, y_tr, sample_weight=sw)
        f1_xgb = float(f1_score(y_te, xgb.predict(X_te), average="macro", zero_division=0.0))
        ref_xgb = float(df_p6_metrics[
            (df_p6_metrics["split"] == s_name) &
            (df_p6_metrics["model"] == "xgb") &
            (df_p6_metrics["snr_db"] == "clean") &
            (df_p6_metrics["variant"] == "variant_A")
        ]["macro_f1"].iloc[0])
        assert abs(f1_xgb - ref_xgb) < 0.001, f"XGB f1 mismatch in {s_name}: {f1_xgb:.4f} vs {ref_xgb:.4f}"
        models[("xgb", s_name)] = xgb

    print("All 12 LOLO models fitted and verified.")

    # Target conditions in exact requested order
    target_conditions = [
        ("variant_B", 10),
        ("variant_B", 5),
        ("variant_A", 10),
        ("variant_A", 5),
    ]

    csv_rows = []
    print("\nComputing LOLO-summed confusion matrices for target conditions...")

    for variant, snr_db in target_conditions:
        cache_file = cache_dir / f"{variant}__snr{snr_db}__rep0.pkl"
        with open(cache_file, "rb") as f:
            df_noisy = pickle.load(f)

        for model_name in MODELS:
            cm_total = np.zeros((4, 4), dtype=int)
            for s_name in LOLO_SPLITS:
                s_data = splits_json[s_name]
                te_df = df_noisy[df_noisy["file_id"].isin(s_data["test_files"])]
                X_te = te_df[ALL_FEATURES].values
                y_te = np.array([CLASS_TO_IDX[c] for c in te_df["fault_class"]], dtype=int)
                y_pred = models[(model_name, s_name)].predict(X_te)
                cm_total += confusion_matrix(y_te, y_pred, labels=[0, 1, 2, 3])

            print(f"\n=== {variant} | {snr_db} dB | {model_name} (LOLO Total Confusion Matrix) ===")
            print(f"{'True Class':<12} | {'Pred Normal':<12} | {'Pred Inner Race':<16} | {'Pred Ball':<10} | {'Pred Outer Race':<16}")
            print("-" * 75)
            for idx, cls in enumerate(CLASS_NAMES):
                print(f"{cls:<12} | {cm_total[idx, 0]:<12} | {cm_total[idx, 1]:<16} | {cm_total[idx, 2]:<10} | {cm_total[idx, 3]:<16}")
                csv_rows.append({
                    "variant": variant,
                    "snr_db": snr_db,
                    "model": model_name,
                    "true_class": cls,
                    "pred_Normal": int(cm_total[idx, 0]),
                    "pred_Inner Race": int(cm_total[idx, 1]),
                    "pred_Ball": int(cm_total[idx, 2]),
                    "pred_Outer Race": int(cm_total[idx, 3]),
                })

    df_out = pd.DataFrame(csv_rows)
    out_csv = results_dir / "phase6b_noise_confusion.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv} ({len(df_out)} rows)")

if __name__ == "__main__":
    main()
