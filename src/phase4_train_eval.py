"""Phase 4: Multi-Model Benchmark, Nested LOLO/Cross-Severity Tuning, and Physics Ablation.

Executes:
1. Environment & XGBoost availability check
2. Data loading & split consistency assertions
3. Fail-fast reproducibility check on lolo_3 (Macro-F1 0.9478 and 0.9803)
4. Nested cross-validation tuning for SVM, Random Forest, and XGBoost across 7 splits
5. Model evaluation (Precision, Recall, Specificity, F1, ROC-AUC, Confusion Matrix, Per-File Accuracy, Feature Importances)
6. 7-row Feature Ablation on Random Forest
7. Outputs generation:
   - results/phase4_metrics.csv
   - results/phase4_confusion.csv
   - results/phase4_per_file.csv
   - results/phase4_importance.csv
   - results/phase4_ablation.csv
   - results/phase4_summary.md
   - data/processed/phase4_predictions.csv
   - data/processed/phase4_best_params.json
"""

import itertools
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

# Set random seeds at top
np.random.seed(42)
random.seed(42)

# Verify / import xgboost
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    import xgboost as xgb
    from xgboost import XGBClassifier

print(f"xgboost version: {xgb.__version__}")

CLASS_NAMES = ["Normal", "Inner Race", "Ball", "Outer Race"]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
SPLIT_NAMES = ["lolo_0", "lolo_1", "lolo_2", "lolo_3", "sev_0.007", "sev_0.014", "sev_0.021"]


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drops collinear features with pairwise absolute Pearson correlation > threshold."""
    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        df_x = pd.DataFrame(X)
        corr = df_x.corr().abs().fillna(0.0).values
        n = corr.shape[0]
        dropped = set()
        for i in range(n):
            if i not in dropped:
                for j in range(i + 1, n):
                    if corr[i, j] > self.threshold:
                        dropped.add(j)
        self.keep_indices_ = np.array([i for i in range(n) if i not in dropped], dtype=int)
        self.dropped_indices_ = np.array(sorted(list(dropped)), dtype=int)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.values[:, self.keep_indices_]
        return np.asarray(X)[:, self.keep_indices_]


def to_markdown_table(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in df.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def compute_multiclass_specificity(cm: np.ndarray) -> np.ndarray:
    """Compute per-class specificity = TN / (TN + FP) from 4x4 confusion matrix."""
    n_classes = cm.shape[0]
    total = np.sum(cm)
    specs = np.zeros(n_classes, dtype=float)
    for k in range(n_classes):
        tp = cm[k, k]
        fp = np.sum(cm[:, k]) - tp
        fn = np.sum(cm[k, :]) - tp
        tn = total - tp - fp - fn
        denom = tn + fp
        specs[k] = (tn / denom) if denom > 0 else 0.0
    return specs


def get_majority_vote(y_pred_file: np.ndarray, scores_file: np.ndarray) -> int:
    """Majority vote prediction with ties broken by highest mean class score."""
    counts = np.bincount(y_pred_file, minlength=4)
    max_count = np.max(counts)
    candidate_classes = np.where(counts == max_count)[0]
    if len(candidate_classes) == 1:
        return int(candidate_classes[0])
    mean_scores = np.mean(scores_file, axis=0)
    best_cand = candidate_classes[0]
    best_score = mean_scores[best_cand]
    for c in candidate_classes[1:]:
        if mean_scores[c] > best_score:
            best_score = mean_scores[c]
            best_cand = c
    return int(best_cand)


def main():
    root_dir = Path(__file__).resolve().parent.parent
    proc_dir = root_dir / "data" / "processed"
    results_dir = root_dir / "results"
    cache_dir = proc_dir / "phase4_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 4: MULTI-MODEL BENCHMARK, NESTED TUNING, & PHYSICS ABLATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Inputs and Constants
    # -------------------------------------------------------------------------
    manifest_path = results_dir / "phase1_file_manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    manifest["file_id"] = manifest["file_id"].astype(int)
    manifest["load_hp"] = manifest["load_hp"].astype(int)
    manifest["window_count"] = manifest["window_count"].astype(int)

    with open(results_dir / "feature_groups.json") as f:
        fg = json.load(f)
    ALL_FEATURES = fg["time"] + fg["freq"] + fg["envelope"]

    with open(proc_dir / "phase3_splits.json") as f:
        splits_json = json.load(f)

    summary_df = pd.read_csv(results_dir / "phase3_splits_summary.csv")

    df_lp = pd.read_csv(proc_dir / "phase2_features_lp3500.csv")
    df_no = pd.read_csv(proc_dir / "phase2_features_nolp.csv")

    df_lp["class_idx"] = df_lp["fault_class"].map(CLASS_TO_IDX)
    df_no["class_idx"] = df_no["fault_class"].map(CLASS_TO_IDX)

    # -------------------------------------------------------------------------
    # 2. Assertions
    # -------------------------------------------------------------------------
    # d. Tables identical row order & 21 feature names equal feature_groups.json
    assert (df_lp["file_id"] == df_no["file_id"]).all(), "File ID order mismatch between lp3500 and nolp"
    assert (df_lp["window_index"] == df_no["window_index"]).all(), "Window index mismatch between lp3500 and nolp"
    feat_cols_lp = [c for c in df_lp.columns if c not in ["file_id", "fault_class", "severity", "load_hp", "window_index", "rpm", "class_idx"]]
    assert feat_cols_lp == ALL_FEATURES, f"Feature mismatch: {feat_cols_lp} vs {ALL_FEATURES}"

    # e. Print max |rms - std|
    diff_lp = float(np.max(np.abs(df_lp["rms"] - df_lp["std"])))
    diff_no = float(np.max(np.abs(df_no["rms"] - df_no["std"])))
    max_rms_std_diff = max(diff_lp, diff_no)
    print(f"Max absolute difference between rms and std over all rows: {max_rms_std_diff:.6e}")

    # a, b, c assertions for every outer split and inner fold
    for s_name in SPLIT_NAMES:
        s = splits_json[s_name]
        tr_files = set(s["train_files"])
        te_files = set(s["test_files"])

        # a. train and test share no file_id
        assert len(tr_files & te_files) == 0, f"{s_name}: file_id shared between train and test"

        # b. every outer test set contains all 4 classes
        te_df = df_lp[df_lp.file_id.isin(te_files)]
        tr_df = df_lp[df_lp.file_id.isin(tr_files)]
        assert set(te_df["fault_class"].unique()) == set(CLASS_NAMES), f"{s_name}: test missing class"
        assert set(tr_df["fault_class"].unique()) == set(CLASS_NAMES), f"{s_name}: train missing class"

        # c. window row counts match summary CSV
        sum_tr = int(summary_df[(summary_df["split"] == s_name) & (summary_df["kind"] == "outer") & (summary_df["set"] == "train")]["n_windows_total"].iloc[0])
        sum_te = int(summary_df[(summary_df["split"] == s_name) & (summary_df["kind"] == "outer") & (summary_df["set"] == "test")]["n_windows_total"].iloc[0])
        assert len(tr_df) == sum_tr, f"{s_name}: outer train windows {len(tr_df)} != {sum_tr}"
        assert len(te_df) == sum_te, f"{s_name}: outer test windows {len(te_df)} != {sum_te}"

        for fold in s["inner_folds"]:
            in_tr_f = set(fold["train_files"])
            in_va_f = set(fold["val_files"])

            # a. inner train and val share no file_id
            assert len(in_tr_f & in_va_f) == 0, f"{s_name} {fold['name']}: inner train and val share file_id"

            # b. inner val and inner train contain all 4 classes
            in_tr_df = df_lp[df_lp.file_id.isin(in_tr_f)]
            in_va_df = df_lp[df_lp.file_id.isin(in_va_f)]
            assert set(in_tr_df["fault_class"].unique()) == set(CLASS_NAMES), f"{s_name} {fold['name']}: inner train missing class"
            assert set(in_va_df["fault_class"].unique()) == set(CLASS_NAMES), f"{s_name} {fold['name']}: inner val missing class"

            # c. inner window counts match summary CSV
            sum_in_tr = int(summary_df[(summary_df["split"] == s_name) & (summary_df["kind"] == "inner") & (summary_df["inner_fold"] == fold["name"]) & (summary_df["set"] == "train")]["n_windows_total"].iloc[0])
            sum_in_va = int(summary_df[(summary_df["split"] == s_name) & (summary_df["kind"] == "inner") & (summary_df["inner_fold"] == fold["name"]) & (summary_df["set"] == "val")]["n_windows_total"].iloc[0])
            assert len(in_tr_df) == sum_in_tr, f"{s_name} {fold['name']}: inner train windows {len(in_tr_df)} != {sum_in_tr}"
            assert len(in_va_df) == sum_in_va, f"{s_name} {fold['name']}: inner val windows {len(in_va_df)} != {sum_in_va}"

    print("Confirmation that all assertions passed: True")

    # -------------------------------------------------------------------------
    # 3. Fail-Fast Reproducibility Check on lolo_3
    # -------------------------------------------------------------------------
    print("\n--- Fail-Fast Reproducibility Check on lolo_3 ---")
    lolo3_s = splits_json["lolo_3"]
    l3_tr_files = lolo3_s["train_files"]
    l3_te_files = lolo3_s["test_files"]

    # Test lp3500
    t0_lp = time.time()
    rf_l3_lp = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    df_l3_tr_lp = df_lp[df_lp.file_id.isin(l3_tr_files)]
    df_l3_te_lp = df_lp[df_lp.file_id.isin(l3_te_files)]
    rf_l3_lp.fit(df_l3_tr_lp[ALL_FEATURES], df_l3_tr_lp["fault_class"])
    y_pred_l3_lp = rf_l3_lp.predict(df_l3_te_lp[ALL_FEATURES])
    runtime_l3_lp = time.time() - t0_lp
    f1_l3_lp = float(f1_score(df_l3_te_lp["fault_class"], y_pred_l3_lp, average="macro"))

    # Test nolp
    t0_no = time.time()
    rf_l3_no = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    df_l3_tr_no = df_no[df_no.file_id.isin(l3_tr_files)]
    df_l3_te_no = df_no[df_no.file_id.isin(l3_te_files)]
    rf_l3_no.fit(df_l3_tr_no[ALL_FEATURES], df_l3_tr_no["fault_class"])
    y_pred_l3_no = rf_l3_no.predict(df_l3_te_no[ALL_FEATURES])
    runtime_l3_no = time.time() - t0_no
    f1_l3_no = float(f1_score(df_l3_te_no["fault_class"], y_pred_l3_no, average="macro"))

    print(f"lolo_3 lp3500 test macro-F1: {f1_l3_lp:.4f} (expected: 0.9478, runtime: {runtime_l3_lp:.2f}s)")
    print(f"lolo_3 nolp   test macro-F1: {f1_l3_no:.4f} (expected: 0.9803, runtime: {runtime_l3_no:.2f}s)")

    off_lp = abs(f1_l3_lp - 0.9478) > 0.001
    off_no = abs(f1_l3_no - 0.9803) > 0.001
    which_off = "none"
    if off_lp and off_no:
        which_off = "both lp3500 and nolp"
    elif off_lp:
        which_off = "lp3500"
    elif off_no:
        which_off = "nolp"
    print(f"Reproducibility off: {which_off}")

    if off_lp or off_no:
        print("[ERROR] Reproducibility check failed! Stopping.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 4. Model Grids & Definitions
    # -------------------------------------------------------------------------
    models_config = {
        "svm": {
            "param_names": ["C", "gamma"],
            "grid": list(itertools.product([1, 10, 100], ["scale", 0.01])),
        },
        "rf": {
            "param_names": ["min_samples_leaf", "max_depth"],
            "grid": list(itertools.product([5, 1], [20, None])),
        },
        "xgb": {
            "param_names": ["max_depth", "learning_rate"],
            "grid": list(itertools.product([3, 5], [0.1, 0.3])),
        }
    }

    # Store results for 7 splits x 3 models
    all_results = {}
    runtimes = {}
    best_params_json = {"svm": {}, "rf": {}, "xgb": {}}

    print("\n--- Starting Nested Cross-Validation Tuning (7 Splits x 3 Models) ---")

    for model_name in ["svm", "rf", "xgb"]:
        for s_name in SPLIT_NAMES:
            pair_key = f"{model_name}__{s_name}"
            cache_file = cache_dir / f"{pair_key}.pkl"

            if cache_file.exists():
                print(f"Loading cached result for {pair_key}...")
                with open(cache_file, "rb") as f:
                    res = pickle.load(f)
                all_results[pair_key] = res
                runtimes[pair_key] = res["runtime_seconds"]
                best_params_json[model_name][s_name] = {
                    "best_params": res["best_params"],
                    "best_inner_macro_f1": float(res["best_inner_macro_f1"]),
                }
                if model_name == "svm":
                    best_params_json[model_name][s_name]["dropped_features"] = res["dropped_features"]
                print(f"  {pair_key} runtime: {res['runtime_seconds']:.2f}s (cached) | test macro-F1: {res['macro_f1']:.4f}")
                continue

            t0 = time.time()
            s_data = splits_json[s_name]
            inner_folds = s_data["inner_folds"]
            cfg = models_config[model_name]
            param_names = cfg["param_names"]

            best_inner_f1 = -1.0
            best_cand = None

            for cand in cfg["grid"]:
                fold_f1s = []
                for fold in inner_folds:
                    in_tr_f = fold["train_files"]
                    in_va_f = fold["val_files"]

                    tr_sub = df_lp[df_lp.file_id.isin(in_tr_f)]
                    va_sub = df_lp[df_lp.file_id.isin(in_va_f)]

                    X_tr = tr_sub[ALL_FEATURES].values
                    y_tr = tr_sub["class_idx"].values
                    X_va = va_sub[ALL_FEATURES].values
                    y_va = va_sub["class_idx"].values

                    if model_name == "svm":
                        c_val, gamma_val = cand
                        clf = Pipeline([
                            ("corr", CorrelationFilter(0.95)),
                            ("scaler", StandardScaler()),
                            ("svc", SVC(kernel="rbf", C=c_val, gamma=gamma_val, class_weight="balanced", random_state=42))
                        ])
                        clf.fit(X_tr, y_tr)
                    elif model_name == "rf":
                        leaf_val, depth_val = cand
                        clf = RandomForestClassifier(
                            n_estimators=300,
                            min_samples_leaf=leaf_val,
                            max_depth=depth_val,
                            class_weight="balanced",
                            random_state=42,
                            n_jobs=-1
                        )
                        clf.fit(X_tr, y_tr)
                    elif model_name == "xgb":
                        depth_val, lr_val = cand
                        sw = compute_sample_weight("balanced", y_tr)
                        clf = XGBClassifier(
                            n_estimators=200,
                            max_depth=depth_val,
                            learning_rate=lr_val,
                            tree_method="hist",
                            eval_metric="mlogloss",
                            random_state=42,
                            n_jobs=-1
                        )
                        clf.fit(X_tr, y_tr, sample_weight=sw)

                    pred_va = clf.predict(X_va)
                    fold_f1 = float(f1_score(y_va, pred_va, average="macro"))
                    fold_f1s.append(fold_f1)

                mean_f1 = float(np.mean(fold_f1s))
                # Tie rule: update winner only on strictly greater scores
                if mean_f1 > best_inner_f1:
                    best_inner_f1 = mean_f1
                    best_cand = cand

            # Refit winner on full outer train set
            out_tr_sub = df_lp[df_lp.file_id.isin(s_data["train_files"])]
            out_te_sub = df_lp[df_lp.file_id.isin(s_data["test_files"])]

            X_out_tr = out_tr_sub[ALL_FEATURES].values
            y_out_tr = out_tr_sub["class_idx"].values
            X_out_te = out_te_sub[ALL_FEATURES].values
            y_out_te = out_te_sub["class_idx"].values

            dropped_features = []
            importances = None

            if model_name == "svm":
                c_val, gamma_val = best_cand
                final_model = Pipeline([
                    ("corr", CorrelationFilter(0.95)),
                    ("scaler", StandardScaler()),
                    ("svc", SVC(kernel="rbf", C=c_val, gamma=gamma_val, class_weight="balanced", random_state=42))
                ])
                final_model.fit(X_out_tr, y_out_tr)
                scores_te = final_model.decision_function(X_out_te)
                y_pred_te = final_model.predict(X_out_te)
                corr_filter = final_model.named_steps["corr"]
                dropped_features = [ALL_FEATURES[i] for i in corr_filter.dropped_indices_]
                classes_ret = final_model.named_steps["svc"].classes_
            elif model_name == "rf":
                leaf_val, depth_val = best_cand
                final_model = RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=leaf_val,
                    max_depth=depth_val,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
                final_model.fit(X_out_tr, y_out_tr)
                scores_te = final_model.predict_proba(X_out_te)
                y_pred_te = final_model.predict(X_out_te)
                importances = final_model.feature_importances_
                classes_ret = final_model.classes_
            elif model_name == "xgb":
                depth_val, lr_val = best_cand
                sw_out = compute_sample_weight("balanced", y_out_tr)
                final_model = XGBClassifier(
                    n_estimators=200,
                    max_depth=depth_val,
                    learning_rate=lr_val,
                    tree_method="hist",
                    eval_metric="mlogloss",
                    random_state=42,
                    n_jobs=-1
                )
                final_model.fit(X_out_tr, y_out_tr, sample_weight=sw_out)
                scores_te = final_model.predict_proba(X_out_te)
                y_pred_te = final_model.predict(X_out_te)
                importances = final_model.feature_importances_
                classes_ret = final_model.classes_

            assert scores_te.shape == (len(X_out_te), 4), f"{pair_key}: score shape != ({len(X_out_te)}, 4)"
            assert list(classes_ret) == [0, 1, 2, 3], f"{pair_key}: classes_ != [0, 1, 2, 3]"

            elapsed = time.time() - t0
            runtimes[pair_key] = elapsed

            best_param_dict = dict(zip(param_names, best_cand))
            macro_f1 = float(f1_score(y_out_te, y_pred_te, average="macro"))

            res = {
                "model": model_name,
                "split": s_name,
                "best_params": best_param_dict,
                "best_inner_macro_f1": best_inner_f1,
                "runtime_seconds": elapsed,
                "y_true": y_out_te,
                "y_pred": y_pred_te,
                "scores": scores_te,
                "test_file_ids": out_te_sub["file_id"].values,
                "test_window_indices": out_te_sub["window_index"].values,
                "test_severities": out_te_sub["severity"].values,
                "test_load_hps": out_te_sub["load_hp"].values,
                "feature_importances": importances,
                "dropped_features": dropped_features,
                "macro_f1": macro_f1,
            }

            with open(cache_file, "wb") as f:
                pickle.dump(res, f)

            all_results[pair_key] = res
            best_params_json[model_name][s_name] = {
                "best_params": best_param_dict,
                "best_inner_macro_f1": best_inner_f1,
            }
            if model_name == "svm":
                best_params_json[model_name][s_name]["dropped_features"] = dropped_features

            print(f"  {pair_key} runtime: {elapsed:.2f}s | best params: {best_param_dict} | test macro-F1: {macro_f1:.4f}")

    # Save phase4_best_params.json
    with open(proc_dir / "phase4_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params_json, f, indent=2)
    print(f"Saved: {proc_dir / 'phase4_best_params.json'}")

    # -------------------------------------------------------------------------
    # 5. Build Evaluation Tables
    # -------------------------------------------------------------------------
    metrics_rows = []
    confusion_rows = []
    per_file_rows = []
    importance_rows = []
    predictions_rows = []

    for model_name in ["svm", "rf", "xgb"]:
        for s_name in SPLIT_NAMES:
            res = all_results[f"{model_name}__{s_name}"]
            y_true = res["y_true"]
            y_pred = res["y_pred"]
            scores = res["scores"]
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
            specs = compute_multiclass_specificity(cm)

            # Metrics per class
            f1_per_c = []
            prec_per_c = []
            rec_per_c = []
            auc_per_c = []

            for k, c_name in enumerate(CLASS_NAMES):
                y_bin = (y_true == k).astype(int)
                p = float(precision_score(y_true == k, y_pred == k, zero_division=0))
                r = float(recall_score(y_true == k, y_pred == k, zero_division=0))
                f1 = float(f1_score(y_true == k, y_pred == k, zero_division=0))
                sp = float(specs[k])
                auc = float(roc_auc_score(y_bin, scores[:, k]))
                supp = int(np.sum(y_true == k))

                prec_per_c.append(p)
                rec_per_c.append(r)
                f1_per_c.append(f1)
                auc_per_c.append(auc)

                metrics_rows.append({
                    "model": model_name,
                    "split": s_name,
                    "class": c_name,
                    "precision": p,
                    "recall": r,
                    "specificity": sp,
                    "f1": f1,
                    "roc_auc": auc,
                    "support": supp,
                })

            # Macro row
            metrics_rows.append({
                "model": model_name,
                "split": s_name,
                "class": "macro",
                "precision": float(np.mean(prec_per_c)),
                "recall": float(np.mean(rec_per_c)),
                "specificity": float(np.mean(specs)),
                "f1": float(np.mean(f1_per_c)),
                "roc_auc": float(np.mean(auc_per_c)),
                "support": int(len(y_true)),
            })

            # Confusion matrix rows
            for k, c_name in enumerate(CLASS_NAMES):
                confusion_rows.append({
                    "model": model_name,
                    "split": s_name,
                    "true_class": c_name,
                    "pred_Normal": int(cm[k, 0]),
                    "pred_Inner Race": int(cm[k, 1]),
                    "pred_Ball": int(cm[k, 2]),
                    "pred_Outer Race": int(cm[k, 3]),
                })

            # Per-file evaluation
            test_fids = res["test_file_ids"]
            test_win_idxs = res["test_window_indices"]
            test_sevs = res["test_severities"]
            test_loads = res["test_load_hps"]

            unique_fids = sorted(list(set(test_fids)))
            for fid in unique_fids:
                mask_fid = test_fids == fid
                y_t_f = y_true[mask_fid]
                y_p_f = y_pred[mask_fid]
                sc_f = scores[mask_fid]

                n_win = int(len(y_t_f))
                n_corr = int(np.sum(y_t_f == y_p_f))
                acc = float(n_corr / n_win)
                maj_pred_idx = get_majority_vote(y_p_f, sc_f)
                maj_pred_name = CLASS_NAMES[maj_pred_idx]
                true_name = CLASS_NAMES[y_t_f[0]]
                maj_corr = int(maj_pred_name == true_name)

                per_file_rows.append({
                    "model": model_name,
                    "split": s_name,
                    "file_id": int(fid),
                    "fault_class": true_name,
                    "severity": str(test_sevs[mask_fid][0]),
                    "load_hp": int(test_loads[mask_fid][0]),
                    "n_windows": n_win,
                    "n_correct": n_corr,
                    "window_accuracy": acc,
                    "majority_vote_pred": maj_pred_name,
                    "majority_vote_correct": maj_corr,
                })

            # Feature importances
            if model_name in ["rf", "xgb"]:
                imp_vals = res["feature_importances"]
                imp_type = "impurity_decrease" if model_name == "rf" else "gain"
                sorted_idx = np.argsort(imp_vals)[::-1]
                for rank, idx in enumerate(sorted_idx, 1):
                    importance_rows.append({
                        "model": model_name,
                        "split": s_name,
                        "importance_type": imp_type,
                        "rank": rank,
                        "feature": ALL_FEATURES[idx],
                        "importance": float(imp_vals[idx]),
                    })

            # Predictions table
            for i in range(len(y_true)):
                predictions_rows.append({
                    "model": model_name,
                    "split": s_name,
                    "file_id": int(test_fids[i]),
                    "window_index": int(test_win_idxs[i]),
                    "y_true": CLASS_NAMES[y_true[i]],
                    "y_pred": CLASS_NAMES[y_pred[i]],
                })

    # Save CSVs
    pd.DataFrame(metrics_rows).to_csv(results_dir / "phase4_metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(results_dir / "phase4_confusion.csv", index=False)
    pd.DataFrame(per_file_rows).to_csv(results_dir / "phase4_per_file.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(results_dir / "phase4_importance.csv", index=False)
    pd.DataFrame(predictions_rows).to_csv(proc_dir / "phase4_predictions.csv", index=False)
    print("Saved metrics, confusion, per-file, importance, and predictions CSVs.")

    # -------------------------------------------------------------------------
    # 6. Feature Ablation (7 Rows x 7 Splits)
    # -------------------------------------------------------------------------
    ablation_cache_file = cache_dir / "ablation_results.pkl"
    ablation_rows = []

    if ablation_cache_file.exists():
        print("\nLoading cached ablation results...")
        with open(ablation_cache_file, "rb") as f:
            ablation_rows = pickle.load(f)
    else:
        print("\n--- Running Feature Ablation (7 Rows x 7 Splits) ---")
        ablation_configs = [
            ("1_dummy_majority", "lp3500", "none", 0, None),
            ("2_rms_only", "lp3500", "rms", 1, ["rms"]),
            ("3_time_only", "lp3500", "time", 9, fg["time"]),
            ("4_time_freq", "lp3500", "time+freq", 15, fg["time"] + fg["freq"]),
            ("5_full_lp3500", "lp3500", "full", 21, ALL_FEATURES),
            ("6_full_nolp", "nolp", "full", 21, ALL_FEATURES),
            ("7_lp3500_no_loudness", "lp3500", "no_loudness", 18, [f for f in ALL_FEATURES if f not in ["rms", "std", "peak_to_peak"]]),
        ]

        for ab_id, var_name, f_source, n_feats, feat_list in ablation_configs:
            for s_name in SPLIT_NAMES:
                s_data = splits_json[s_name]
                tr_f = s_data["train_files"]
                te_f = s_data["test_files"]

                df_src = df_lp if var_name == "lp3500" else df_no

                # Reuse section 4 runs on lolo_3
                if s_name == "lolo_3" and ab_id == "5_full_lp3500":
                    y_pred_ab = y_pred_l3_lp
                    te_df = df_src[df_src.file_id.isin(te_f)]
                    y_true_ab = te_df["fault_class"].values
                elif s_name == "lolo_3" and ab_id == "6_full_nolp":
                    y_pred_ab = y_pred_l3_no
                    te_df = df_src[df_src.file_id.isin(te_f)]
                    y_true_ab = te_df["fault_class"].values
                else:
                    tr_df = df_src[df_src.file_id.isin(tr_f)]
                    te_df = df_src[df_src.file_id.isin(te_f)]
                    y_true_ab = te_df["fault_class"].values

                    if ab_id == "1_dummy_majority":
                        clf = DummyClassifier(strategy="most_frequent")
                        clf.fit(tr_df[["file_id"]], tr_df["fault_class"])
                        y_pred_ab = clf.predict(te_df[["file_id"]])
                    else:
                        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
                        clf.fit(tr_df[feat_list], tr_df["fault_class"])
                        y_pred_ab = clf.predict(te_df[feat_list])

                macro_f1 = float(f1_score(y_true_ab, y_pred_ab, average="macro"))
                rec_norm = float(recall_score(y_true_ab == "Normal", y_pred_ab == "Normal", zero_division=0))
                rec_ir = float(recall_score(y_true_ab == "Inner Race", y_pred_ab == "Inner Race", zero_division=0))
                rec_ball = float(recall_score(y_true_ab == "Ball", y_pred_ab == "Ball", zero_division=0))
                rec_or = float(recall_score(y_true_ab == "Outer Race", y_pred_ab == "Outer Race", zero_division=0))

                ablation_rows.append({
                    "ablation_id": ab_id,
                    "variant_name": var_name,
                    "feature_source": f_source,
                    "n_features": n_feats,
                    "split": s_name,
                    "macro_f1": macro_f1,
                    "recall_Normal": rec_norm,
                    "recall_Inner Race": rec_ir,
                    "recall_Ball": rec_ball,
                    "recall_Outer Race": rec_or,
                })

        with open(ablation_cache_file, "wb") as f:
            pickle.dump(ablation_rows, f)
        print("Completed feature ablation and saved cache.")

    df_ablation = pd.DataFrame(ablation_rows)
    df_ablation.to_csv(results_dir / "phase4_ablation.csv", index=False)
    print(f"Saved: {results_dir / 'phase4_ablation.csv'}")

    # -------------------------------------------------------------------------
    # 7. Generate results/phase4_summary.md
    # -------------------------------------------------------------------------
    df_metrics = pd.DataFrame(metrics_rows)
    df_macro = df_metrics[df_metrics["class"] == "macro"].copy()

    # Table 1: Macro-F1 for svm, rf, xgb
    t1_rows = []
    lolo_splits = ["lolo_0", "lolo_1", "lolo_2", "lolo_3"]
    sev_splits = ["sev_0.007", "sev_0.014", "sev_0.021"]

    for s_name in SPLIT_NAMES:
        row_dict = {"Split": s_name}
        for m in ["svm", "rf", "xgb"]:
            val = df_macro[(df_macro["model"] == m) & (df_macro["split"] == s_name)]["f1"].iloc[0]
            row_dict[m.upper()] = f"{val:.4f}"
        t1_rows.append(row_dict)

    # LOLO mean +- std
    lolo_row = {"Split": "LOLO mean +- std"}
    for m in ["svm", "rf", "xgb"]:
        vals = df_macro[(df_macro["model"] == m) & (df_macro["split"].isin(lolo_splits))]["f1"].values
        lolo_row[m.upper()] = f"{np.mean(vals):.4f} +- {np.std(vals):.4f}"
    t1_rows.append(lolo_row)

    # Severity mean +- std
    sev_row = {"Split": "Severity mean +- std"}
    for m in ["svm", "rf", "xgb"]:
        vals = df_macro[(df_macro["model"] == m) & (df_macro["split"].isin(sev_splits))]["f1"].values
        sev_row[m.upper()] = f"{np.mean(vals):.4f} +- {np.std(vals):.4f}"
    t1_rows.append(sev_row)

    df_t1 = pd.DataFrame(t1_rows)

    # Table 2: Per-class mean recall and F1 over LOLO and Severity splits
    t2_rows = []
    for m in ["svm", "rf", "xgb"]:
        for c in CLASS_NAMES:
            sub_lolo = df_metrics[(df_metrics["model"] == m) & (df_metrics["class"] == c) & (df_metrics["split"].isin(lolo_splits))]
            sub_sev = df_metrics[(df_metrics["model"] == m) & (df_metrics["class"] == c) & (df_metrics["split"].isin(sev_splits))]

            rec_lolo = f"{sub_lolo['recall'].mean():.4f} +- {sub_lolo['recall'].std():.4f}"
            f1_lolo = f"{sub_lolo['f1'].mean():.4f} +- {sub_lolo['f1'].std():.4f}"
            rec_sev = f"{sub_sev['recall'].mean():.4f} +- {sub_sev['recall'].std():.4f}"
            f1_sev = f"{sub_sev['f1'].mean():.4f} +- {sub_sev['f1'].std():.4f}"

            t2_rows.append({
                "Model": m.upper(),
                "Class": c,
                "LOLO Recall": rec_lolo,
                "LOLO F1": f1_lolo,
                "Severity Recall": rec_sev,
                "Severity F1": f1_sev,
            })
    df_t2 = pd.DataFrame(t2_rows)

    # Table 3: Ablation rows summary
    t3_rows = []
    ab_order = [
        "1_dummy_majority",
        "2_rms_only",
        "3_time_only",
        "4_time_freq",
        "5_full_lp3500",
        "6_full_nolp",
        "7_lp3500_no_loudness"
    ]
    for ab_id in ab_order:
        sub_ab = df_ablation[df_ablation["ablation_id"] == ab_id]
        lolo_vals = sub_ab[sub_ab["split"].isin(lolo_splits)]["macro_f1"].values
        sev_vals = sub_ab[sub_ab["split"].isin(sev_splits)]["macro_f1"].values

        row_d = {
            "Ablation ID": ab_id,
            "Features": sub_ab["feature_source"].iloc[0],
            "N": sub_ab["n_features"].iloc[0],
        }
        for s in SPLIT_NAMES:
            row_d[s] = f"{sub_ab[sub_ab['split'] == s]['macro_f1'].iloc[0]:.4f}"
        row_d["LOLO Mean+-Std"] = f"{np.mean(lolo_vals):.4f} +- {np.std(lolo_vals):.4f}"
        row_d["Sev Mean+-Std"] = f"{np.mean(sev_vals):.4f} +- {np.std(sev_vals):.4f}"
        t3_rows.append(row_d)

    df_t3 = pd.DataFrame(t3_rows)

    # Write results/phase4_summary.md
    summary_md_path = results_dir / "phase4_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 4 Benchmark & Ablation Summary\n\n")

        f.write("## Table 1: Model Benchmark Across Splits (Macro-F1)\n\n")
        f.write(to_markdown_table(df_t1) + "\n\n")

        f.write("## Table 2: Per-Class Generalization Metrics\n\n")
        f.write("> Note: Normal class has exactly 1 test file per split (58 windows in lolo_0, 117 windows in all other splits).\n\n")
        f.write(to_markdown_table(df_t2) + "\n\n")

        f.write("## Table 3: Physics-Informed Feature Ablation (Random Forest)\n\n")
        f.write(to_markdown_table(df_t3) + "\n\n")

        if max_rms_std_diff < 1e-9:
            f.write("> **RMS / STD Identity:** rms and std are identical by construction, so ablation row 2 is equivalent to std only.\n")
        else:
            f.write(f"> **RMS / STD Max Difference:** measured maximum difference is {max_rms_std_diff:.6e}.\n")

    print(f"Saved: {summary_md_path}")

    # -------------------------------------------------------------------------
    # 8. Concise Terminal Output for Section 10
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CONCISE TERMINAL REPORT")
    print("=" * 80)
    print(f"xgboost version: {xgb.__version__}")
    print(f"Reproducibility check on lolo_3:")
    print(f"  lp3500 Macro-F1: {f1_l3_lp:.4f} (runtime: {runtime_l3_lp:.2f}s)")
    print(f"  nolp   Macro-F1: {f1_l3_no:.4f} (runtime: {runtime_l3_no:.2f}s)")
    print(f"Confirmation that all assertions passed: True")
    print(f"Max |rms - std|: {max_rms_std_diff:.6e}")

    print("\nPer-Pair Runtimes (seconds):")
    for pair_k in sorted(runtimes.keys()):
        print(f"  {pair_k:<18}: {runtimes[pair_k]:.2f}s")

    print("\nTable 1: Macro-F1 (SVM, RF, XGB):")
    print(df_t1.to_string(index=False))

    print("\nTable 3: Ablation Rows Macro-F1:")
    print(df_t3[["Ablation ID", "Features", "N", "LOLO Mean+-Std", "Sev Mean+-Std"]].to_string(index=False))

    print("\nDropped Features by CorrelationFilter (SVM 7 Splits):")
    for s_name in SPLIT_NAMES:
        dropped = best_params_json["svm"][s_name]["dropped_features"]
        print(f"  {s_name:<10}: {dropped}")


if __name__ == "__main__":
    main()
