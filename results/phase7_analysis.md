# Synthesis and Technical Analysis

This document provides a systematic technical analysis of the Case Western Reserve University (CWRU) bearing fault classification benchmark across data acquisition, feature engineering, supervised classification, unsupervised novelty detection, and noise stress-testing.

---

## 1. Where Does Performance Drop?

Model evaluation was conducted under two split regimes: leave-one-load-out (LOLO, four folds holding out motor loads 0, 1, 2, or 3 HP) and cross-severity (three rotations holding out fault diameter 0.007", 0.014", or 0.021"). The load-split scores measure only transfer to a new motor load at fault sizes already seen during training. Under load transfer, all three classifiers maintain high scores: Support Vector Machine (SVM) achieves mean macro-F1 of 0.9760 +- 0.0249, Random Forest (RF) achieves 0.9852 +- 0.0216, and XGBoost achieves 0.9835 +- 0.0217 (population standard deviations, ddof=0, from `results/phase4_summary.md`). In `results/phase6_summary.md`, where sample standard deviations (ddof=1) are reported across the four folds, these identical baseline runs are recorded as 0.9760 +- 0.0288 for SVM, 0.9852 +- 0.0249 for RF, and 0.9835 +- 0.0251 for XGBoost. The weakest load split for all models is `lolo_3` (macro-F1: SVM 0.9332, RF 0.9478, XGBoost 0.9461 from `results/phase4_summary.md`).

Under fault-size shift, performance drops sharply across all model architectures. Mean severity macro-F1 falls to 0.5968 +- 0.2470 for SVM, 0.6112 +- 0.2696 for RF, and 0.4871 +- 0.1169 for XGBoost (ddof=0, from `results/phase4_summary.md`; ddof=1 values in `results/phase6_summary.md` are 0.5968 +- 0.3026 for SVM, 0.6112 +- 0.3302 for RF, and 0.4871 +- 0.1431 for XGBoost). The weakest split across every model is `sev_0.014` (macro-F1: SVM 0.2482, RF 0.2691, XGBoost 0.3747 from `results/phase4_summary.md`).

Per-class recall across severity splits shows that healthy bearings and Ball faults remain identifiable, whereas race faults fail to generalize (`results/phase4_summary.md`, `results/phase4_metrics.csv`). Normal recall is 1.0000 +- 0.0000 across all three models. Ball recall remains high: SVM 0.9568 +- 0.0447, RF 0.9993 +- 0.0012, and XGBoost 0.9922 +- 0.0075. In contrast, Inner Race recall drops to 0.6172 +- 0.5277 for SVM, 0.3333 +- 0.5774 for RF, and 0.3800 +- 0.5410 for XGBoost. Outer Race recall collapses to 0.2681 +- 0.2884 for SVM, 0.5218 +- 0.4556 for RF, and 0.0000 +- 0.0000 for XGBoost.

| Model | LOLO Mean Macro-F1 (ddof=0) | LOLO Mean Macro-F1 (ddof=1) | Weakest LOLO (`lolo_3`) | Severity Mean Macro-F1 (ddof=0) | Severity Mean Macro-F1 (ddof=1) | Weakest Severity (`sev_0.014`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SVM** | 0.9760 +- 0.0249 | 0.9760 +- 0.0288 | 0.9332 | 0.5968 +- 0.2470 | 0.5968 +- 0.3026 | 0.2482 |
| **RF** | 0.9852 +- 0.0216 | 0.9852 +- 0.0249 | 0.9478 | 0.6112 +- 0.2696 | 0.6112 +- 0.3302 | 0.2691 |
| **XGBoost** | 0.9835 +- 0.0217 | 0.9835 +- 0.0251 | 0.9461 | 0.4871 +- 0.1169 | 0.4871 +- 0.1431 | 0.3747 |

*(Data source: `results/phase4_summary.md` for ddof=0 and fold scores; `results/phase6_summary.md` for ddof=1).*

---

## 2. What Does Each Pipeline Stage Add?

A 7-row feature ablation benchmark was conducted using Random Forest across all seven outer splits to quantify the contribution of each signal processing stage (`results/phase4_summary.md`, `results/phase4_ablation.csv`). The dummy majority baseline yields a macro-F1 of 0.1166 +- 0.0021 on load splits and 0.1177 +- 0.0002 on severity splits. Root mean square (RMS) amplitude alone (Row 2, 1 feature) achieves a macro-F1 of 0.7715 +- 0.0130 on load splits and 0.4716 +- 0.1582 on severity splits. This demonstrates that pure vibration loudness carries a substantial baseline across motor loads. Because each window is demeaned before feature calculation, RMS and standard deviation (std) are identical by construction, making Row 2 equivalent to evaluating standard deviation alone.

Expanding to nine time-domain statistics (Row 3) increases load macro-F1 to 0.9605 +- 0.0287, while severity macro-F1 remains low at 0.4487 +- 0.1654. Adding six fast Fourier transform (FFT) spectral features (Row 4, 15 features) further raises load macro-F1 to 0.9723 +- 0.0347 (severity: 0.4509 +- 0.1639). The full feature set with the 3500 Hz low-pass filter and six envelope features (Row 5, 21 features) achieves 0.9854 +- 0.0217 on load splits and 0.5790 +- 0.2330 on severity splits.

Removing loudness features (Row 7, 18 features without RMS, standard deviation, and peak-to-peak amplitude) preserves load macro-F1 at 0.9852 +- 0.0183, but severity macro-F1 drops to 0.4707 +- 0.1257. Evaluating the full 21 features without the 3500 Hz low-pass filter (Row 6, unfiltered signals) yields 0.9857 +- 0.0143 on load splits and 0.6921 +- 0.2451 on severity splits. Whether this unfiltered severity advantage stems from high-frequency filter roll-off discrepancies or from real physical vibroacoustic information above 3500 Hz was not determined in Phase 4 (Section 6 reports the cutoff sweep, which does not settle it either).

| Ablation ID | Description | Features | N | LOLO Mean Macro-F1 | Severity Mean Macro-F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1_dummy_majority** | Majority class baseline | none | 0 | 0.1166 +- 0.0021 | 0.1177 +- 0.0002 |
| **2_rms_only** | Single amplitude feature | rms | 1 | 0.7715 +- 0.0130 | 0.4716 +- 0.1582 |
| **3_time_only** | Time domain statistics | time | 9 | 0.9605 +- 0.0287 | 0.4487 +- 0.1654 |
| **4_time_freq** | Time + FFT spectral features | time+freq | 15 | 0.9723 +- 0.0347 | 0.4509 +- 0.1639 |
| **5_full_lp3500** | Full filtered pipeline | full | 21 | 0.9854 +- 0.0217 | 0.5790 +- 0.2330 |
| **6_full_nolp** | Full unfiltered pipeline | full | 21 | 0.9857 +- 0.0143 | 0.6921 +- 0.2451 |
| **7_lp3500_no_loudness** | Filtered without amplitude | no_loudness | 18 | 0.9852 +- 0.0183 | 0.4707 +- 0.1257 |

*(Data source: `results/phase4_summary.md` Table 3; `results/phase4_ablation.csv`).*

---

## 3. Why Does the Fault-Size Shift Fail?

The collapse on the `sev_0.014` split coincides with a non-monotonic relationship between defect diameter and vibration amplitude (`results/phase4b_class_severity_means.csv`). When trained on 0.007" and 0.021" faults, Random Forest, trained on the 0.007" and 0.021" Outer Race windows (large amplitudes), predicts the quiet 0.014" Outer Race windows as Normal or Ball; which features the models rely on was not tested here: 0.007" Outer Race has mean RMS of 0.4646 and envelope signal-to-noise ratio at the ball pass frequency outer race (BPFO) of 13.3187; 0.021" Outer Race has mean RMS of 0.4384 and BPFO envelope SNR of 4.4449. In contrast, 0.014" Outer Race exhibits an average RMS of only 0.0663 (a seven-fold reduction) and BPFO envelope SNR of 1.6482, with a Gaussian kurtosis of 3.0046 (all measured in `results/phase4b_class_severity_means.csv`).

Confusion matrices on `sev_0.014` confirm that classifiers misallocate these quiet 0.014" Outer Race windows to classes with similar amplitudes (`results/phase4_confusion.csv`). For Random Forest, 227 Outer Race windows are predicted as Normal and 243 as Ball (0 out of 470 predicted as Outer Race, giving 0.0000 recall). Inner Race 0.014" has a mean RMS of 0.1371, which is close to, and just above, the Ball range (0.1027 to 0.1279); its mean kurtosis (16.4151) is much higher than Ball's (2.9358 to 7.2975), so loudness alone does not explain the Ball predictions (not tested) (Ball RMS is 0.1213 at 0.007", 0.1279 at 0.014", and 0.1027 at 0.021"). Random Forest classifies all 468 test windows of 0.014" Inner Race as Ball (0.0000 recall). In `results/phase4_per_file.csv`, every single 0.014" test file across both Inner Race (files 169, 170, 171, 172) and Outer Race (files 197, 198, 199, 200) achieves 0.0000 window accuracy for Random Forest on `sev_0.014`.

Unsupervised clustering provides independent confirmation of this feature overlap (`results/phase5_dbscan_cluster_files.csv`). Density-Based Spatial Clustering of Applications with Noise (DBSCAN) isolates 0.007" and 0.021" race faults into separate single-file clusters, but in every fold, the 0.014" Inner Race and Outer Race recordings merge into the largest cluster alongside Normal and Ball files. Statements regarding measured RMS, kurtosis, envelope SNR, confusion counts, and cluster memberships are measured facts. No physical explanation for the lower amplitude at 0.014" was tested.

| Fault Class | Severity | Mean RMS | Mean Kurtosis | Mean Peak-to-Peak | Mean BPFI Envelope SNR | Mean BPFO Envelope SNR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Normal** | none | 0.0446 | 2.9939 | 0.3053 | 1.5650 | 1.4658 |
| **Inner Race** | 0.007" | 0.2415 | 4.7491 | 1.9943 | 9.7919 | 3.3271 |
| **Inner Race** | 0.014" | 0.1371 | 16.4151 | 2.1328 | 2.5296 | 1.8039 |
| **Inner Race** | 0.021" | 0.4589 | 7.4917 | 4.8290 | 5.4378 | 1.0663 |
| **Ball** | 0.007" | 0.1213 | 2.9358 | 0.8144 | 1.6674 | 1.8174 |
| **Ball** | 0.014" | 0.1279 | 7.2975 | 1.4658 | 1.3323 | 1.5891 |
| **Ball** | 0.021" | 0.1027 | 3.5732 | 0.8065 | 1.8298 | 1.8302 |
| **Outer Race** | 0.007" | 0.4646 | 6.4782 | 4.0201 | 1.2144 | 13.3187 |
| **Outer Race** | 0.014" | 0.0663 | 3.0046 | 0.4571 | 1.4258 | 1.6482 |
| **Outer Race** | 0.021" | 0.4384 | 18.6565 | 6.9289 | 1.7292 | 4.4449 |

*(Data source: `results/phase4b_class_severity_means.csv`).*

---

## 4. Novelty Detection and Clustering

Out-of-distribution novelty detection was evaluated with Isolation Forest using threshold-free area under the receiver operating characteristic curve (AUROC) across the four load folds (`results/phase5_summary.md`). In Experiment 1A (detecting an unseen fault class), Inner Race faults achieve a mean AUROC of 0.9458 +- 0.0249 and Outer Race faults achieve 0.9318 +- 0.0078. Ball faults yield an AUROC of 0.4464 +- 0.0242, which is below the random-guessing baseline of 0.5000 (and 0.1596 +- 0.0979 against Normal), confirming that Ball faults are difficult to isolate in this feature space.

In Experiment 1B (training only on healthy Normal baseline windows to detect any fault), the full 21-feature pipeline achieves an overall mean AUROC of 0.9650 +- 0.0319. However, the one-feature RMS baseline achieves a score of 1.0000 +- 0.0000 across all four folds and all sub-comparisons. This score arises because the Normal windows are quieter than every fault window, whereas the quietest fault group (Outer Race 0.014", mean rms 0.0663) is louder. Because Normal is the only class from 48 kHz recordings, its absolute loudness may not be directly comparable with the fault classes (limitation 8), so this experiment says little about generalisation. An energy-based detector therefore separates healthy bearings from damaged bearings without learning fault kinematics.

Unsupervised DBSCAN clustering across the test folds yields a mean Adjusted Rand Index (ARI) of 0.1584 +- 0.0667 and Normalized Mutual Information (NMI) of 0.3611 +- 0.1060 across the four load splits (`results/phase5_summary.md`). The mean noise fraction is 0.0151 +- 0.0042 (11 to 23 windows per fold). Detailed file breakdown (`results/phase5_dbscan_cluster_files.csv`) reveals that every cluster other than the largest consists entirely of windows from a single recording (specifically 0.007" and 0.021" Inner Race and Outer Race files), while Normal, Ball, and 0.014" files coalesce into a single diffuse cluster.

| Experiment | Task | Variant | Target / Comparison | Fold Mean AUROC +- Std |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 1A** | Unseen Fault Class Detection | `lp3500_all21` | Inner Race (overall) | 0.9458 +- 0.0249 |
| **Exp 1A** | Unseen Fault Class Detection | `lp3500_all21` | Ball (overall) | 0.4464 +- 0.0242 |
| **Exp 1A** | Unseen Fault Class Detection | `lp3500_all21` | Ball vs Normal | 0.1596 +- 0.0979 |
| **Exp 1A** | Unseen Fault Class Detection | `lp3500_all21` | Outer Race (overall) | 0.9318 +- 0.0078 |
| **Exp 1B** | Normal-Only Detector | `lp3500_all21` | All Faults (overall) | 0.9650 +- 0.0319 |
| **Exp 1B** | Normal-Only Detector | `rms_only_baseline` | All Faults (overall) | 1.0000 +- 0.0000 |

*(Data source: `results/phase5_summary.md` Tables 1 and 2).*

---

## 5. Noise Robustness

Clean-trained classifiers were stress-tested with additive white Gaussian noise injected prior to the 3500 Hz low-pass filter across four signal-to-noise ratios (SNR: 30, 20, 10, and 5 dB) under two formulations (`results/phase6_summary.md`). In Variant A, noise power scales proportionally to each individual file's alternating current (AC) power ($P_{\text{file}}$). Under Variant A, tree ensembles degrade gradually across load splits: Random Forest macro-F1 moves from clean 0.9852 +- 0.0249 to 0.9848 +- 0.0268 at 20 dB, 0.9819 +- 0.0302 at 10 dB, and 0.9216 +- 0.0405 at 5 dB (sample standard deviations, ddof=1). XGBoost moves from 0.9835 +- 0.0251 clean to 0.9826 +- 0.0265 at 20 dB, 0.9740 +- 0.0223 at 10 dB, and 0.8998 +- 0.0534 at 5 dB. SVM maintains 0.9661 +- 0.0247 at 20 dB but drops to 0.6158 +- 0.0869 at 10 dB and 0.2720 +- 0.0579 at 5 dB (the mechanism behind this SVM drop was not tested).

In Variant B, noise variance is fixed to the median power of all 36 fault recordings ($P_{\text{ref}} = 0.0300$) across all files (`results/phase6_summary.md`). Because Normal baseline files have lower power (mean rms 0.0446, against a reference power of 0.0300, i.e. rms about 0.173), fixed noise reduces their effective post-filter SNR to -0.16 to 1.05 dB at nominal 10 dB. As Normal windows absorb noise, their mean RMS rises from 0.0446 clean to 0.0607 at 10 dB. Random Forest Normal recall collapses from 1.0000 clean to 0.0029 +- 0.0057 at 10 dB and 0.0000 at 5 dB, reducing overall load macro-F1 to 0.6596 +- 0.0445 (10 dB) and 0.6185 +- 0.0249 (5 dB). XGBoost Normal recall falls to 0.2533 +- 0.1943 at 10 dB and 0.0000 at 5 dB, with macro-F1 of 0.7871 +- 0.1036 (10 dB) and 0.6262 +- 0.0163 (5 dB). SVM retains Normal recall of 0.9272 +- 0.0223 at 10 dB and 0.9315 +- 0.0300 at 5 dB, with macro-F1 of 0.6143 +- 0.0546 (10 dB) and 0.4253 +- 0.0276 (5 dB).

The confusion matrices reveal opposing operational failure modes (`results/phase6b_noise_confusion.csv`). Random Forest and XGBoost produce false alarms: at 10 and 5 dB in both variants (repetition 0, the conditions stored in results/phase6b_noise_confusion.csv), they never classify a damaged bearing window as Normal (0 missed faults). At 10 dB in Variant B, Random Forest predicts all 409 Normal windows as Outer Race; at 5 dB, it predicts 396 as Ball and 13 as Outer Race. In contrast, SVM exhibits missed detections: under noise, SVM predicts damaged bearings as Normal. In Variant A, SVM classifies 633 of 4,234 fault windows (14.95%) as Normal at 10 dB (140 Inner Race, 203 Ball, 290 Outer Race) and 2,211 of 4,234 fault windows (52.22%) as Normal at 5 dB (675 Inner Race, 725 Ball, 811 Outer Race) (computed from `results/phase6b_noise_confusion.csv`). In Variant B, SVM classifies 748 fault windows (17.67%) as Normal at 10 dB and 1,596 fault windows (37.70%) as Normal at 5 dB. Caveats apply: noise was synthetic white Gaussian, and models were evaluated without noise-augmented training or denoising filters.

| Variant | Model | Clean LOLO F1 | 20 dB LOLO F1 | 10 dB LOLO F1 | 5 dB LOLO F1 | 10 dB Normal Recall | 5 dB Normal Recall | Missed Fault Rate (10 dB) | Missed Fault Rate (5 dB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A (Proportional)** | SVM | 0.9760 +- 0.0288 | 0.9661 +- 0.0247 | 0.6158 +- 0.0869 | 0.2720 +- 0.0579 | 1.0000 +- 0.0000 | 0.9864 +- 0.0190 | 14.95% (633/4234) | 52.22% (2211/4234) |
| **A (Proportional)** | RF | 0.9852 +- 0.0249 | 0.9848 +- 0.0268 | 0.9819 +- 0.0302 | 0.9216 +- 0.0405 | 1.0000 +- 0.0000 | 0.9915 +- 0.0121 | 0.00% (0/4234) | 0.00% (0/4234) |
| **A (Proportional)** | XGB | 0.9835 +- 0.0251 | 0.9826 +- 0.0265 | 0.9740 +- 0.0223 | 0.8998 +- 0.0534 | 0.9964 +- 0.0054 | 0.8304 +- 0.1871 | 0.00% (0/4234) | 0.00% (0/4234) |
| **B (Fixed Ref)** | SVM | 0.9760 +- 0.0288 | 0.9501 +- 0.0246 | 0.6143 +- 0.0546 | 0.4253 +- 0.0276 | 0.9272 +- 0.0223 | 0.9315 +- 0.0300 | 17.67% (748/4234) | 37.70% (1596/4234) |
| **B (Fixed Ref)** | RF | 0.9852 +- 0.0249 | 0.9848 +- 0.0260 | 0.6596 +- 0.0445 | 0.6185 +- 0.0249 | 0.0029 +- 0.0057 | 0.0000 +- 0.0000 | 0.00% (0/4234) | 0.00% (0/4234) |
| **B (Fixed Ref)** | XGB | 0.9835 +- 0.0251 | 0.9825 +- 0.0274 | 0.7871 +- 0.1036 | 0.6262 +- 0.0163 | 0.2533 +- 0.1943 | 0.0000 +- 0.0000 | 0.00% (0/4234) | 0.00% (0/4234) |

*(Data source: `results/phase6_summary.md` Tables 1 and 2; missed fault rates computed from `results/phase6b_noise_confusion.csv`).*

---

## 6. Cutoff Sensitivity and Power Removed by Low-Pass

Filter cutoff sensitivity was evaluated with Random Forest across five cutoffs (3000, 3500, 4000, 4500, and 5000 Hz) and unfiltered data ("none") across the seven outer splits (`results/phase6c_cutoff_summary.md`). Across load splits, mean macro-F1 is stable across all conditions: 0.9761 +- 0.0220 at 3000 Hz, 0.9854 +- 0.0217 at 3500 Hz, 0.9915 +- 0.0086 at 4000 Hz, 0.9904 +- 0.0085 at 4500 Hz, 0.9828 +- 0.0150 at 5000 Hz, and 0.9857 +- 0.0143 for unfiltered data (seed 42, ddof=0). The differences between cutoffs lie within the cross-split spread (0.0085 to 0.0220) and seed spread (0.0035 to 0.0120).

Across severity splits, cutoffs of 4000 Hz (0.6826 +- 0.2412) and 4500 Hz (0.7097 +- 0.2526) yield higher mean macro-F1 than 3500 Hz (0.5790 +- 0.2330) and 3000 Hz (0.5190 +- 0.1905). This difference is driven by improved transfer on 0.007" (0.6364 at 3500 Hz vs 0.7082 at 4000 Hz and 0.8364 at 4500 Hz) and 0.021" (0.8313 at 3500 Hz vs 0.9645 at 4000 Hz and 0.9355 at 4500 Hz). In contrast, the `sev_0.014` split remains weak across every condition: 0.2584 at 3000 Hz, 0.2693 at 3500 Hz, 0.3753 at 4000 Hz, 0.3571 at 4500 Hz, 0.3485 at 5000 Hz, and 0.3526 unfiltered, with 0.0000 window accuracy for all eight 0.014" files. Normal recall on load splits is 1.0000 up to 4000 Hz, 0.9957 at 4500 Hz, and 0.9483 at 5000 Hz and unfiltered.

Power measurements are consistent with this response (`results/phase6b_power_above_3500.csv`, `results/phase6c_power_removed.csv`). The 3500 Hz low-pass filter removes on average 44.36% of Outer Race power, 26.89% of Inner Race power, 23.36% of Ball power, and 45.17% of Normal power. For Outer Race and Inner Race most of the removed power lies between 3500 and 4000 Hz (23.63% and 18.94% of the total power; a smaller part lies just below 3500 Hz, where the filter rolls off). Ball loses 23.36%, but only 3.84% of its power lies above 3500 Hz, so its loss is mainly the roll-off just below the cutoff. For Normal bearings, power above 3500 Hz resides almost entirely above 4000 Hz (42.68% out of 44.52%). A 4000 Hz cutoff keeps the 3500-4000 Hz band that the 3500 Hz filter removes, while removing energy above 4000 Hz. This does not show that the band carries fault information. However, the 3500 Hz cutoff was specified early in Phase 2 as a safeguard prior to these sensitivity tests, and altering the cutoff based on test performance would constitute tuning on test data.

| Cutoff (Hz) | LOLO Mean Macro-F1 | Severity Mean Macro-F1 | Mean Seed Std | Normal Power Removed | Outer Race Power Removed | Inner Race Power Removed | Ball Power Removed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **3000** | 0.9761 +- 0.0220 | 0.5190 +- 0.1905 | 0.0120 | 49.02% | 76.11% | 55.83% | 71.57% |
| **3500** | 0.9854 +- 0.0217 | 0.5790 +- 0.2330 | 0.0035 | 45.17% | 44.36% | 26.89% | 23.36% |
| **4000** | 0.9915 +- 0.0086 | 0.6826 +- 0.2412 | 0.0063 | 42.07% | 5.81% | 3.04% | 1.88% |
| **4500** | 0.9904 +- 0.0085 | 0.7097 +- 0.2526 | 0.0064 | 7.37% | 2.13% | 0.38% | 0.78% |
| **5000** | 0.9828 +- 0.0150 | 0.6743 +- 0.2401 | 0.0070 | 2.40% | 1.85% | 0.15% | 0.42% |
| **none** | 0.9857 +- 0.0143 | 0.6921 +- 0.2451 | 0.0117 | 0.00% | 0.00% | 0.00% | 0.00% |

*(Data source: `results/phase6c_cutoff_summary.md` Tables 1 and 3; `results/phase6b_power_above_3500.csv`).*

---

## 7. Data Notes and Limitations

Metadata verification identified anomalies in the raw recordings (`results/phase1_file_manifest.csv`). File `99.mat` (Normal at 2 HP) contains an extraneous variable `X098_DE_time` leftover from workspace leakage in addition to its valid variable `X099_DE_time`. Files `98.mat` (1 HP) and `99.mat` (2 HP) lack an internal shaft speed variable (`X098RPM`, `X099RPM`); nominal rotational speeds (1772 RPM and 1750 RPM) were applied deterministically. The four Normal baseline files are treated as 48 kHz recordings (inferred from sample counts: 97.mat has the same length as the 48 kHz fault file 109.mat; CWRU does not state it), whereas the fault recordings are 12 kHz; Normal files were downsampled by a factor of 4 with polyphase finite impulse response filtering (`scipy.signal.resample_poly(x, 1, 4)`). File `97.mat` (Normal at 0 HP) contains only 243,938 raw samples (~5 seconds, yielding 58 windows of 2048 samples at 50% overlap), whereas files 98, 99, and 100 contain ~485,000 raw samples (~10 seconds, yielding 117 windows). Under the 3500 Hz low-pass filter, file 97 loses 67.29% of its power, with 65.53% of its total spectral energy located between 4000 and 5900 Hz (`results/phase6b_power_above_3500.csv`).

Ten specific limitations apply to this benchmark, each with a concrete operational consequence:


1. Ball fault labels. The envelope check supports the Ball labels only on part of the files (mean envelope ratio at the ball spin frequency 1.36 to 1.90 across severities, against 1.38 for Normal; results/phase4b_class_severity_means.csv). Consequence: Ball labels rest on the CWRU file numbering, and the Ball files merge into one large cluster with Normal (results/phase5_dbscan_cluster_files.csv).
2. Low-pass attenuation. The common 3500 Hz low-pass removes on average 44.36% of the power of Outer Race files (52.31% at 0.014"), 26.89% of Inner Race (34.61% at 0.007", 39.20% at 0.014", 6.86% at 0.021"), 23.36% of Ball and 45.17% of Normal (results/phase6b_power_above_3500.csv). Consequence: content above 3500 Hz, and part of the content just below it, is not available to the models; whether it carries fault information was not tested beyond the cutoff sweep.
3. Resampling leakage evidence. It rests on one control pair (file 109 resampled vs file 105; local output results/phase2_diagnostics_summary.txt, regenerated by src/phase2_diagnostics.py). Consequence: the effect of the resampling filter cannot be separated from physical differences between healthy and damaged bearings.
4. Normal sample size. Exactly one Normal recording is used per load (58 windows at 0 HP, 117 at 1, 2 and 3 HP; results/phase3_splits_summary.csv). Consequence: per-fold Normal metrics depend on a single recording.
5. Pilot scope. The pilot used one fold (lolo_3) with 10 test files (local output results/pilot_lolo_fold3.txt, regenerated by src/phase2_features.py). Consequence: it cannot show whether the low-pass matters; the low-pass is a safeguard, not a demonstrated improvement.
6. Loudness. Amplitude features can dominate: rms alone reaches 0.7715 macro-F1 across load folds (results/phase4_summary.md). Removing the three loudness features leaves load macro-F1 unchanged (0.9852) and does not improve severity macro-F1 (0.4707 vs 0.5790). Consequence: loudness is not the only reason for the severity failure.
7. Inner cross-validation in the severity splits. Tuning uses leave-one-load-out inner folds (data/processed/phase3_splits.json). Consequence: model selection optimizes for load transfer, not for transfer to an unseen fault size.
8. Amplitude scale and roll-off. Ratio (Resampled 109 / Native 105) in 2000-4000 Hz (reference): +5.10 dB; Ratio in 4000-4500 Hz: +16.24 dB; Ratio in 4500-5000 Hz: +11.55 dB; Ratio in 5000-5900 Hz: +10.86 dB (local output results/phase2_diagnostics_summary.txt, regenerated by src/phase2_diagnostics.py). Consequence: the absolute loudness and high-frequency content of the Normal class, the only class from 48 kHz recordings, may not be directly comparable with the fault classes; an rms-only score reaches AUROC 1.0 in Experiment 1B, which says little about generalisation.
9. Failure modes under noise. At 10 and 5 dB, Random Forest and XGBoost raised false alarms (Normal windows predicted as fault classes) and never predicted a fault window as Normal, while the SVM predicted 14.95% to 52.22% of fault windows as Normal (results/phase6b_noise_confusion.csv). Consequence: in this synthetic-noise test the SVM missed faults and the tree models raised false alarms.
10. Cutoff sensitivity. Cutoffs of 4000 and 4500 Hz give a higher severity macro-F1 (0.6826 and 0.7097) than 3500 Hz (0.5790), with load macro-F1 unchanged within the spread (0.9915 and 0.9904 vs 0.9854); the severity differences lie within the spread of the three splits (about 0.19 to 0.25), and the 0.014" weakness persists at every cutoff (results/phase6c_cutoff_summary.md). Consequence: the 3500 Hz filter may remove fault information in the 3500-4000 Hz band; the cutoff was fixed before these tests and only Random Forest was tested.
