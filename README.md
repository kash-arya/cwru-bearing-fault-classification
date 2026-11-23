# Bearing Fault Classification with Generalization Benchmarking

## What This Project Does

Vibration-based fault diagnosis often looks solved in the lab, until operational conditions shift. This benchmark stresses supervised classifiers (SVM, Random Forest, XGBoost) and unsupervised detectors (Isolation Forest, DBSCAN) against motor load variations, defect size shifts, and additive Gaussian sensor noise on the Case Western Reserve University (CWRU) bearing dataset. 

Signals are segmented into 2048-sample windows (50% overlap) across 40 drive-end accelerometer recordings. To prevent high-frequency sampling-rate discrepancies from resampled baseline files, a fixed 3500 Hz low-pass filter preprocesses the data before extracting 21 time, spectral, and envelope features. All evaluations enforce strict file-level isolation: overlapping windows from the same recording never cross a train/test boundary.

---

## Dataset and Data Access

Data comes from the Bearing Data Center at the Case School of Engineering, Case Western Reserve University (CWRU). The benchmark uses 40 drive-end accelerometer recordings:
- **Classes:** Healthy Normal baseline, Ball fault, Inner Race fault, and Outer Race fault (centered at 6 o'clock).
- **Fault Severities:** 0.007", 0.014", and 0.021" diameter electro-discharge machined defects. (The 0.028" depth is omitted because Outer Race data was never collected at that size).
- **Motor Loads:** 0, 1, 2, and 3 HP (approximate rotational speeds of 1797, 1772, 1750, and 1730 RPM).
- **Sampling Rates:** The 36 fault recordings were acquired at 12 kHz. The four Normal baseline files (`97.mat` to `100.mat`) lack explicit sampling rate metadata in the CWRU tables but share the exact length (243,938 samples) of 48 kHz recordings like `109.mat`. They are treated as 48 kHz recordings and resampled to 12 kHz using `scipy.signal.resample_poly(x, 1, 4)`.

Raw `.mat` files are excluded from git tracking. Running `src/phase1_data_verify.py` downloads the 40 required files directly from the CWRU repository (`https://engineering.case.edu/sites/default/files/{file_id}.mat`). File `109.mat` serves solely as a 48 kHz diagnostic control pair for `105.mat` (0.007" Inner Race at 0 HP) to evaluate anti-aliasing filter roll-off; `src/download_control_file.py` downloads it into `data/raw/109.mat`.

---

## Evaluation Design

Evaluating time-series models requires strict boundary enforcement to prevent data leakage (`results/phase3_splits_summary.csv`):
- **File-Level Splitting:** Adjacent overlapping windows from the same continuous recording share nearly identical phase information. Splitting windows randomly inflates test scores artificially. All train/test partitions are grouped strictly by file ID: windows from any single file never cross fold boundaries.
- **Leave-One-Load-Out (LOLO):** Four outer folds hold out all recordings from one motor load (0, 1, 2, or 3 HP) for testing (10 test files: 3 per fault class, 1 Normal file; 30 training files). Three inner folds across the remaining training loads tune hyperparameters.
- **Cross-Severity Generalization:** Three outer rotations hold out one defect size (0.007", 0.014", or 0.021") across all motor loads (13 test files: 4 Inner Race, 4 Ball, 4 Outer Race, and 1 Normal file; 27 training files). Inner cross-validation across loads selects hyperparameters.
- **Normal Class Balance:** Every test fold contains exactly one Normal recording (58 windows at 0 HP; 117 windows at 1, 2, and 3 HP). Consequently, per-fold Normal recall and precision reflect single-recording behavior.
- **Evaluation Metrics:** Multiclass macro-averaged F1 score (`macro_f1`) provides the primary benchmark metric, tracked alongside per-class recall, precision, and confusion matrices. Class weights are balanced for SVM and Random Forest; sample weights are balanced for XGBoost.

---

## Pipeline

The pipeline consists of 14 sequential scripts:

| Step | Script | Purpose | Main Inputs | Main Outputs |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `src/phase1_data_verify.py` | Download 40 `.mat` files, resample Normal files from 48 kHz to 12 kHz, count windows, audit metadata | Remote server CWRU URLs / `data/raw/` | `data/processed/phase1_signals_12k.pkl`, `results/phase1_file_manifest.csv` |
| 2 | `src/download_control_file.py` | Download and validate CWRU control file 109.mat (48 kHz Drive-End fault) | Remote server (`https://engineering.case.edu/sites/default/files/109.mat`) | `data/raw/109.mat` |
| 3 | `src/phase2_diagnostics.py` | Diagnose sampling rate roll-off discrepancy between resampled and native files via control pair 109 vs 105 | `data/processed/phase1_signals_12k.pkl`, `data/raw/109.mat` | `results/phase2_diagnostics_summary.txt`, `results/diagnostic_filter_rolloff.png`, `results/diagnostic_envelope_spectra.png` |
| 4 | `src/phase2b_leakage_fix.py` | Evaluate 3500 Hz low-pass filter and 1500-3000 Hz envelope band on control pair and 40 files | `data/processed/phase1_signals_12k.pkl`, `results/phase1_file_manifest.csv`, `data/raw/109.mat` | `results/phase2b_leakage_fix_summary.txt`, `results/diagnostic_leakage_fix.png` |
| 5 | `src/phase2_features.py` | Extract 21 time, frequency, and envelope features across 4,643 windows for filtered (`lp3500`) and unfiltered (`nolp`) data; run pilot evaluation on fold 3 | `data/processed/phase1_signals_12k.pkl`, `results/phase1_file_manifest.csv` | `data/processed/phase2_features_lp3500.csv`, `data/processed/phase2_features_nolp.csv`, `results/feature_groups.json`, `results/phase2_features_summary.txt`, `results/pilot_lolo_fold3.txt` |
| 6 | `src/phase3_splits.py` | Generate file-isolated outer and inner cross-validation folds for LOLO and severity splits | `results/phase1_file_manifest.csv`, `data/processed/phase2_features_lp3500.csv`, `data/processed/phase2_features_nolp.csv` | `data/processed/phase3_splits.json`, `results/phase3_splits_summary.csv` |
| 7 | `src/phase4_train_eval.py` | Benchmark SVM, Random Forest, and XGBoost with nested tuning across 7 splits; run 7-row ablation | `data/processed/phase2_features_lp3500.csv`, `data/processed/phase2_features_nolp.csv`, `data/processed/phase3_splits.json`, `results/phase3_splits_summary.csv`, `results/feature_groups.json`, `results/phase1_file_manifest.csv` | `results/phase4_metrics.csv`, `results/phase4_confusion.csv`, `results/phase4_per_file.csv`, `results/phase4_importance.csv`, `data/processed/phase4_predictions.csv`, `results/phase4_ablation.csv`, `data/processed/phase4_best_params.json`, `results/phase4_summary.md` |
| 8 | `src/phase4b_class_severity_means.py` | Compute mean feature values by class and severity to diagnose cross-severity transfer drop | `data/processed/phase2_features_lp3500.csv` | `results/phase4b_class_severity_means.csv` |
| 9 | `src/phase5_unsupervised.py` | Evaluate Isolation Forest novelty detection (Exp 1A and 1B) and DBSCAN clustering with PCA | `data/processed/phase2_features_lp3500.csv`, `data/processed/phase2_features_nolp.csv`, `data/processed/phase3_splits.json`, `results/phase3_splits_summary.csv`, `results/feature_groups.json` | `results/phase5_novelty_A.csv`, `results/phase5_novelty_B.csv`, `results/phase5_dbscan.csv`, `results/phase5_dbscan_crosstab.csv`, `results/phase5_dbscan_eps_sensitivity.csv`, `results/phase5_summary.md`, `results/diagnostic_phase5_kdistance.png` |
| 10 | `src/phase5b_cluster_files.py` | Audit DBSCAN cluster composition per file across the four LOLO test folds | `data/processed/phase2_features_lp3500.csv`, `results/feature_groups.json`, `data/processed/phase3_splits.json`, `results/phase5_dbscan.csv` | `results/phase5_dbscan_cluster_files.csv` |
| 11 | `src/phase6_noise_robustness.py` | Evaluate clean models under additive Gaussian noise (Variants A & B at 30, 20, 10, 5 dB) | `data/processed/phase1_signals_12k.pkl`, `results/phase1_file_manifest.csv`, `data/processed/phase2_features_lp3500.csv`, `data/processed/phase3_splits.json`, `data/processed/phase4_best_params.json`, `results/phase4_metrics.csv` | `results/phase6_metrics.csv`, `results/phase6_effective_snr_variant_a.csv`, `results/phase6_effective_snr_variant_b.csv`, `results/phase6_envelope_vs_snr.csv`, `results/phase6_summary.md`, `results/fig_phase6_macro_f1_vs_snr.png`, `results/fig_phase6_envelope_vs_snr.png`, `data/processed/phase6_cache/` |
| 12 | `src/phase6b_power_above_3500.py` | Quantify signal power removed by 3500 Hz low-pass filter and high-frequency spectral power fractions | `data/processed/phase1_signals_12k.pkl`, `results/phase1_file_manifest.csv` | `results/phase6b_power_above_3500.csv` |
| 13 | `src/phase6b_noise_confusion.py` | Generate 4-class confusion matrices under noise (Variants A and B at 10 and 5 dB) across LOLO splits | `data/processed/phase2_features_lp3500.csv`, `data/processed/phase3_splits.json`, `data/processed/phase4_best_params.json`, `results/phase6_metrics.csv`, `data/processed/phase6_cache/` | `results/phase6b_noise_confusion.csv` |
| 14 | `src/phase6c_cutoff_sensitivity.py` | Test Random Forest sensitivity to filter cutoffs (3000, 3500, 4000, 4500, 5000 Hz, and none) | `results/phase1_file_manifest.csv`, `data/processed/phase1_signals_12k.pkl`, `data/processed/phase3_splits.json`, `data/processed/phase2_features_lp3500.csv`, `data/processed/phase2_features_nolp.csv`, `results/phase4_ablation.csv` | `results/phase6c_cutoff_metrics.csv`, `results/phase6c_per_file.csv`, `results/phase6c_power_removed.csv`, `results/phase6c_cutoff_summary.md`, `results/fig_phase6c_cutoff_vs_macro_f1.png` |

---


## How to Run

### Setup Environment

Use a fresh virtual environment without system site packages (the default of python -m venv).

**Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/phase1_data_verify.py
python src/download_control_file.py
```

**Windows:**
```cmd
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src\phase1_data_verify.py
python src\download_control_file.py
```

**Platform Note:** Developed and benchmarked on Ubuntu Linux with Python 3.14.4. Scripts use POSIX `/tmp` paths for temporary matplotlib cache directories. While dependencies install cross-platform, execution under native Windows environments is untested. Make sure to install dependencies via `requirements.txt` before executing the pipeline.

### Run Pipeline

To run the complete pipeline sequentially on Linux:
```bash
python src/phase1_data_verify.py
python src/download_control_file.py
python src/phase2_diagnostics.py
python src/phase2b_leakage_fix.py
python src/phase2_features.py
python src/phase3_splits.py
python src/phase4_train_eval.py
python src/phase4b_class_severity_means.py
python src/phase5_unsupervised.py
python src/phase5b_cluster_files.py
python src/phase6_noise_robustness.py
python src/phase6b_power_above_3500.py
python src/phase6b_noise_confusion.py
python src/phase6c_cutoff_sensitivity.py
```

---


## Main Results

### 1. Supervised Benchmark Across Splits
The table below reports macro-F1 across the four load folds and three severity rotations. The Phase 4 summary uses population standard deviations (`ddof=0`), while the Phase 6 baseline summary uses sample standard deviations (`ddof=1`):

| Model | LOLO Mean Macro-F1 (ddof=0) | LOLO Mean Macro-F1 (ddof=1) | Severity Mean Macro-F1 (ddof=0) | Severity Mean Macro-F1 (ddof=1) | Weakest Split |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SVM** | 0.9760 +- 0.0249 | 0.9760 +- 0.0288 | 0.5968 +- 0.2470 | 0.5968 +- 0.3026 | `sev_0.014` (0.2482) |
| **Random Forest** | 0.9852 +- 0.0216 | 0.9852 +- 0.0249 | 0.6112 +- 0.2696 | 0.6112 +- 0.3302 | `sev_0.014` (0.2691) |
| **XGBoost** | 0.9835 +- 0.0217 | 0.9835 +- 0.0251 | 0.4871 +- 0.1169 | 0.4871 +- 0.1431 | `sev_0.014` (0.3747) |

*(Data source: `results/phase4_summary.md` Table 1; `results/phase6_summary.md` Table 1).*

### 2. Feature Ablation Benchmark (Random Forest)
| Ablation Configuration | Features | Feature Count | LOLO Mean Macro-F1 | Severity Mean Macro-F1 |
| :--- | :--- | :--- | :--- | :--- |
| **1_dummy_majority** | none | 0 | 0.1166 +- 0.0021 | 0.1177 +- 0.0002 |
| **2_rms_only** | rms | 1 | 0.7715 +- 0.0130 | 0.4716 +- 0.1582 |
| **3_time_only** | time | 9 | 0.9605 +- 0.0287 | 0.4487 +- 0.1654 |
| **4_time_freq** | time + freq | 15 | 0.9723 +- 0.0347 | 0.4509 +- 0.1639 |
| **5_full_lp3500** | full (filtered) | 21 | 0.9854 +- 0.0217 | 0.5790 +- 0.2330 |
| **6_full_nolp** | full (unfiltered) | 21 | 0.9857 +- 0.0143 | 0.6921 +- 0.2451 |
| **7_lp3500_no_loudness** | no_loudness | 18 | 0.9852 +- 0.0183 | 0.4707 +- 0.1257 |

*(Data source: `results/phase4_summary.md` Table 3; `results/phase4_ablation.csv`).*

### 3. Unsupervised Novelty Detection (Isolation Forest)
| Experiment | Task | Variant | Comparison / Class | Mean AUROC +- Std |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 1A** | Unseen Fault Detection | `lp3500_all21` | Inner Race (overall) | 0.9458 +- 0.0249 |
| **Exp 1A** | Unseen Fault Detection | `lp3500_all21` | Ball (overall) | 0.4464 +- 0.0242 |
| **Exp 1A** | Unseen Fault Detection | `lp3500_all21` | Outer Race (overall) | 0.9318 +- 0.0078 |
| **Exp 1B** | Normal-Only Detector | `lp3500_all21` | All Faults (overall) | 0.9650 +- 0.0319 |
| **Exp 1B** | Normal-Only Detector | `rms_only_baseline` | All Faults (overall) | 1.0000 +- 0.0000 |

*(Data source: `results/phase5_summary.md` Tables 1 and 2).*

### 4. Unsupervised Clustering (DBSCAN across LOLO Test Folds)
| Metric | `lolo_0` | `lolo_1` | `lolo_2` | `lolo_3` | Mean +- Std |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Clusters Found** | 2 | 5 | 5 | 4 | 4.00 |
| **Noise Fraction** | 0.0188 | 0.0128 | 0.0094 | 0.0195 | 0.0151 +- 0.0042 |
| **Adjusted Rand Index (ARI)** | 0.0554 | 0.2175 | 0.2172 | 0.1436 | 0.1584 +- 0.0667 |
| **Normalized Mutual Info (NMI)** | 0.1857 | 0.4449 | 0.4454 | 0.3683 | 0.3611 +- 0.1060 |

*(Data source: `results/phase5_summary.md` Table 3).*

### 5. Noise Robustness across SNR Levels (LOLO Macro-F1)
| Variant | Model | Clean | 20 dB | 10 dB | 5 dB |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Variant A (Proportional)** | SVM | 0.9760 +- 0.0288 | 0.9661 +- 0.0247 | 0.6158 +- 0.0869 | 0.2720 +- 0.0579 |
| **Variant A (Proportional)** | Random Forest | 0.9852 +- 0.0249 | 0.9848 +- 0.0268 | 0.9819 +- 0.0302 | 0.9216 +- 0.0405 |
| **Variant A (Proportional)** | XGBoost | 0.9835 +- 0.0251 | 0.9826 +- 0.0265 | 0.9740 +- 0.0223 | 0.8998 +- 0.0534 |
| **Variant B (Fixed Ref)** | SVM | 0.9760 +- 0.0288 | 0.9501 +- 0.0246 | 0.6143 +- 0.0546 | 0.4253 +- 0.0276 |
| **Variant B (Fixed Ref)** | Random Forest | 0.9852 +- 0.0249 | 0.9848 +- 0.0260 | 0.6596 +- 0.0445 | 0.6185 +- 0.0249 |
| **Variant B (Fixed Ref)** | XGBoost | 0.9835 +- 0.0251 | 0.9825 +- 0.0274 | 0.7871 +- 0.1036 | 0.6262 +- 0.0163 |

Normal recall in Variant B at 10 dB: SVM 0.9272, Random Forest 0.0029, XGBoost 0.2533; at 5 dB: 0.9315, 0.0000, 0.0000 (results/phase6_summary.md Table 2).

*(Data source: `results/phase6_summary.md` Table 1; sample standard deviations with ddof=1).*

### 6. Filter Cutoff Sensitivity (Random Forest, Seed 42)
| Cutoff Frequency | LOLO Mean Macro-F1 | Severity Mean Macro-F1 | Mean Seed Std | Normal Power Removed | Fault Power Removed (OR / IR / Ball) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **3000 Hz** | 0.9761 +- 0.0220 | 0.5190 +- 0.1905 | 0.0120 | 49.02% | 76.11% / 55.83% / 71.57% |
| **3500 Hz** | 0.9854 +- 0.0217 | 0.5790 +- 0.2330 | 0.0035 | 45.17% | 44.36% / 26.89% / 23.36% |
| **4000 Hz** | 0.9915 +- 0.0086 | 0.6826 +- 0.2412 | 0.0063 | 42.07% | 5.81% / 3.04% / 1.88% |
| **4500 Hz** | 0.9904 +- 0.0085 | 0.7097 +- 0.2526 | 0.0064 | 7.37% | 2.13% / 0.38% / 0.78% |
| **5000 Hz** | 0.9828 +- 0.0150 | 0.6743 +- 0.2401 | 0.0070 | 2.40% | 1.85% / 0.15% / 0.42% |
| **none (unfiltered)** | 0.9857 +- 0.0143 | 0.6921 +- 0.2451 | 0.0117 | 0.00% | 0.00% / 0.00% / 0.00% |

*(Data source: `results/phase6c_cutoff_summary.md` Tables 1 and 3; `results/phase6b_power_above_3500.csv`).*

---

## Figures

![Macro-F1 vs Additive Noise Level](results/fig_phase6_macro_f1_vs_snr.png)
*Figure 1: Classifier macro-F1 across signal-to-noise ratios from clean down to 5 dB for proportional noise (Variant A) and fixed reference noise (Variant B).*

![Envelope Feature Peak-to-Median Ratios vs Additive Noise Level](results/fig_phase6_envelope_vs_snr.png)
*Figure 2: Top row: class-matched envelope peak-to-median ratios; bottom row: class-wise kurtosis, from clean to 5 dB, for Variant A and Variant B.*

![Macro-F1 vs Filter Cutoff Frequency](results/fig_phase6c_cutoff_vs_macro_f1.png)
*Figure 3: Random Forest macro-F1 across low-pass filter cutoffs from 3000 Hz to unfiltered data across load and severity splits.*

---

## Limitations

1. **Ball Fault Kinematics & Labeling:** Envelope spectra match theoretical ball spin frequencies on only a subset of recordings. Across severities, the mean envelope ratio at the ball spin frequency spans 1.36 to 1.90, barely above the 1.38 baseline seen in healthy bearings (`results/phase4b_class_severity_means.csv`). Because rolling elements spin and tumble in three dimensions rather than striking a fixed location, impact pulses smear across time. Unsupervised DBSCAN merges Ball recordings into a single diffuse cluster with Normal baseline files (`results/phase5_dbscan_cluster_files.csv`).
2. **Pre-Filter Energy Loss:** The 3500 Hz low-pass filter removes substantial high-frequency vibroacoustic energy. Outer Race files lose an average of 44.36% of their raw power (peaking at 52.31% for 0.014" defects), Inner Race files lose 26.89%, Ball files lose 23.36%, and Normal files lose 45.17% (`results/phase6b_power_above_3500.csv`). Any diagnostic fault signatures above 3500 Hz are permanently attenuated before feature extraction.
3. **Control Pair Evidence for Resampling Leakage:** The sampling-rate roll-off discrepancy was diagnosed using a single control pair: 48 kHz file 109 downsampled to 12 kHz versus native 12 kHz file 105 (`results/phase2_diagnostics_summary.txt`). With only one paired condition available, we cannot isolate filter roll-off characteristics from the underlying physical differences between test benches.
4. **Normal Class Representation:** Each test fold contains exactly one Normal recording (58 windows at 0 HP; 117 windows at 1, 2, and 3 HP; `results/phase3_splits_summary.csv`). Per-fold Normal metrics are tied to the specific dynamics of that single continuous file rather than a distribution of healthy machines.
5. **Single-Fold Pilot Scope:** Initial low-pass filter validation relied exclusively on fold `lolo_3` with 10 test files (`results/pilot_lolo_fold3.txt`). The filter serves as a conservative safeguard against high-frequency leakage rather than a universally proven optimization.
6. **Loudness Dominance:** Amplitude features carry disproportionate weight across operating loads. RMS alone achieves 0.7715 macro-F1 across load folds (`results/phase4_summary.md`). However, stripping the three loudness features (RMS, standard deviation, peak-to-peak) leaves load macro-F1 steady at 0.9852 while cross-severity macro-F1 drops from 0.5790 to 0.4707, proving that severity generalization failure cannot be pinned on loudness bias alone.
7. **Cross-Severity Inner Fold Tuning:** Hyperparameter optimization for severity splits used leave-one-load-out inner validation folds (`data/processed/phase3_splits.json`). Consequently, grid search selected model parameters that generalize across motor loads rather than across unseen defect sizes.
8. **Disparate Baseline Recording Conditions:** Healthy Normal files originate from 48 kHz recordings, whereas fault files were sampled natively at 12 kHz. Downsampled file 109 shows elevated high-frequency energy relative to file 105 (+5.10 dB at 2000–4000 Hz, jumping to +16.24 dB at 4000–4500 Hz; `results/phase2_diagnostics_summary.txt`). Healthy signals are noticeably quieter than fault recordings (mean RMS of 0.0446 vs 0.0663 for the quietest fault group), allowing an elementary RMS threshold to achieve 1.0 AUROC in novelty detection without learning fault dynamics.
9. **Divergent Noise Degradation Modes:** Under additive Gaussian noise at 10 dB and 5 dB, tree ensembles (Random Forest, XGBoost) and SVM fail in opposite directions (`results/phase6b_noise_confusion.csv`). Trees yield false alarms by misclassifying noisy Normal windows as faults, but never miss a damaged bearing. SVM does the reverse: it misclassifies 14.95% to 52.22% of damaged bearing windows as healthy, introducing critical false negatives.
10. **Filter Cutoff Trade-offs:** Raising the low-pass cutoff to 4000 Hz or 4500 Hz improves cross-severity macro-F1 to 0.6826 and 0.7097 (compared to 0.5790 at 3500 Hz), while load generalization remains essentially unchanged at ~0.99 (`results/phase6c_cutoff_summary.md`). The persistent failure on 0.014" defects across every cutoff confirms that filter attenuation is not the primary driver of severity breakdown. Because the 3500 Hz cutoff was frozen during initial diagnostics, altering it after observing benchmark results would constitute test-set leakage.

---


## Reproducibility

- **Random Seeds:**
  - `np.random.seed(42)` is set at the top of 11 scripts: `src/phase2_diagnostics.py`, `src/phase2_features.py`, `src/phase2b_leakage_fix.py`, `src/phase3_splits.py`, `src/phase4_train_eval.py`, `src/phase5_unsupervised.py`, `src/phase5b_cluster_files.py`, `src/phase6_noise_robustness.py`, `src/phase6b_power_above_3500.py`, `src/phase6b_noise_confusion.py`, and `src/phase6c_cutoff_sensitivity.py`.
  - `random_state=42` is used for estimators and PCA in `src/phase2_features.py`, `src/phase4_train_eval.py`, `src/phase5_unsupervised.py`, `src/phase5b_cluster_files.py`, `src/phase6_noise_robustness.py`, `src/phase6b_noise_confusion.py`, and `src/phase6c_cutoff_sensitivity.py`. (Note: `src/phase3_splits.py` and `src/phase6b_power_above_3500.py` set `np.random.seed(42)` but contain no `random_state` parameter).
  - `random.seed(42)` is set in `src/phase4_train_eval.py`.
  - Seeds `[0, 1, 2, 3, 4]` are evaluated in `src/phase5_unsupervised.py` for Isolation Forest novelty detection.
  - Seeds `[42, 0, 1]` are evaluated in `src/phase6c_cutoff_sensitivity.py` for Random Forest cutoff sensitivity.
  - Deterministic pseudo-random noise in `src/phase6_noise_robustness.py` is seeded per file, SNR level, and repetition using `zlib.crc32`.
  - Results were produced on Ubuntu Linux with Python 3.14.4. The packages numpy, scipy, scikit-learn and xgboost came from the virtual environment (PyPI builds); matplotlib and pandas came from the Ubuntu system packages (same version numbers as in requirements.txt). Other versions or builds may change numbers slightly.
  - A re-run of phases 1 to 4 in a clean environment with the PyPI packages of requirements.txt reproduced the macro-F1 values of results/phase4_metrics.csv and results/phase4_ablation.csv identically (maximum absolute difference 0, on the same machine). Phases 5 and 6 were not re-run.
- **Tracked vs. Local Files:**
  - Tracked in Git: All code in `src/`, package requirements in `requirements.txt`, documentation in `README.md`, and all final result tables and figures in `results/` (`.csv`, `.md`, `.json`, and `fig_*.png`).
  - Created Locally (not tracked): Raw `.mat` downloads in `data/raw/`, processed signal pickles and cached feature tables in `data/processed/`, and intermediate text summaries and diagnostic figures (`results/*.txt`, `results/diagnostic_*.png`).

---


## Repository Layout

```
.
├── .gitignore               # Git untracked pattern specifications
├── data/
│   ├── processed/           # Processed signals, splits, features, and model caches (local, not tracked)
│   └── raw/                 # Downloaded .mat files (local, not tracked)
├── results/                 # Metrics, summaries, figures, and analysis reports (tracked)
├── src/                     # Pipeline implementation scripts (tracked)
├── README.md                # Project documentation
└── requirements.txt         # Environment package specifications
```

---


## Data Credit

Vibration recordings were collected by the Bearing Data Center at the Case School of Engineering, Case Western Reserve University (CWRU).
