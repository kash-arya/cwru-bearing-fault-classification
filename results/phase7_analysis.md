# Benchmark Synthesis and Technical Analysis

Vibration diagnostics frequently show near-perfect accuracy under homogeneous bench conditions. The real test begins when models encounter operating domain shifts: changing motor loads, unobserved defect dimensions, and sensor noise. This report synthesizes findings from across the benchmark pipeline, tracking where classifiers hold, where they break, and the physical mechanisms behind those breakdowns.

---

## 1. Where Does Performance Drop?

We evaluated classifiers across two operational shifts: Leave-One-Load-Out (LOLO, testing on unseen motor torque at known defect sizes) and Cross-Severity (testing on unseen defect dimensions across all loads).

Load transfer proves straightforward. All three architectures sustain strong classification across operating torques: Support Vector Machines achieve 0.9760 ± 0.0249 macro-F1, Random Forest reaches 0.9852 ± 0.0216, and XGBoost attains 0.9835 ± 0.0217 (population std, ddof=0; sample std ddof=1 values in `results/phase6_summary.md` are 0.9760 ± 0.0288, 0.9852 ± 0.0249, and 0.9835 ± 0.0251). Performance dips slightly on fold `lolo_3` (macro-F1: SVM 0.9332, RF 0.9478, XGBoost 0.9461), but remains solid throughout.

Transfer across fault dimensions fails completely. When models encounter an unseen defect size, macro-F1 plummets: SVM drops to 0.5968 ± 0.2470, RF falls to 0.6112 ± 0.2696, and XGBoost sinks to 0.4871 ± 0.1169. The collapse centers on `sev_0.014`, where macro-F1 crashes to 0.2482 for SVM, 0.2691 for RF, and 0.3747 for XGBoost.

Per-class breakdowns isolate the culprit (`results/phase4_summary.md`, `results/phase4_metrics.csv`). Healthy baseline bearings and Ball faults retain near-perfect identification (Normal recall: 1.0000 across all models; Ball recall: SVM 0.9568, RF 0.9993, XGBoost 0.9922). But raceway faults disintegrate under size shifts. Inner Race recall drops to 0.6172 for SVM, 0.3333 for RF, and 0.3800 for XGBoost. Outer Race recall collapses entirely: 0.2681 for SVM, 0.5218 for RF, and a flat 0.0000 for XGBoost.

| Model | LOLO Mean Macro-F1 (ddof=0) | LOLO Mean Macro-F1 (ddof=1) | Weakest LOLO (`lolo_3`) | Severity Mean Macro-F1 (ddof=0) | Severity Mean Macro-F1 (ddof=1) | Weakest Severity (`sev_0.014`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SVM** | 0.9760 +- 0.0249 | 0.9760 +- 0.0288 | 0.9332 | 0.5968 +- 0.2470 | 0.5968 +- 0.3026 | 0.2482 |
| **RF** | 0.9852 +- 0.0216 | 0.9852 +- 0.0249 | 0.9478 | 0.6112 +- 0.2696 | 0.6112 +- 0.3302 | 0.2691 |
| **XGBoost** | 0.9835 +- 0.0217 | 0.9835 +- 0.0251 | 0.9461 | 0.4871 +- 0.1169 | 0.4871 +- 0.1431 | 0.3747 |

*(Data source: `results/phase4_summary.md` for ddof=0 and fold scores; `results/phase6_summary.md` for ddof=1).*

---

## 2. What Does Each Pipeline Stage Add?

To dissect how each signal processing stage contributes to classification, we benchmarked seven feature configurations using Random Forest across all outer splits (`results/phase4_ablation.csv`).

Predicting the majority class yields a trivial baseline: 0.1166 on loads, 0.1177 on severities. Feeding the model a single amplitude feature—RMS—jumps load macro-F1 straight to 0.7715 ± 0.0130 (and 0.4716 ± 0.1582 on severities). Because windows are demeaned prior to calculation, RMS and standard deviation are mathematically identical here. Sheer vibration energy alone provides a remarkably strong load-invariant signal.

Layering additional physical representations systematically closes the remaining gap. Adding nine time-domain statistics elevates load performance to 0.9605 ± 0.0287. Introducing six FFT spectral band energies pushes it to 0.9723 ± 0.0347. Incorporating 1500–3000 Hz envelope demodulation features alongside the 3500 Hz low-pass filter (21 features total) peaks load macro-F1 at 0.9854 ± 0.0217 and brings severity transfer to 0.5790 ± 0.2330.

Two counterfactual tests clarify what drives these numbers:
1. *Stripping loudness:* Removing the three amplitude features (RMS, standard deviation, peak-to-peak) leaves load performance untouched (0.9852 ± 0.0183), but degrades severity generalization to 0.4707 ± 0.1257. Loudness helps bridge defect sizes, even if models cannot rely on it exclusively.
2. *Removing the low-pass filter:* Evaluating all 21 features on raw, unfiltered signals yields 0.9857 ± 0.0143 on loads and an apparent boost to 0.6921 ± 0.2451 on severities. As explored in Section 6, this unfiltered gain is entangled with sampling-rate roll-off differences between 48 kHz and 12 kHz recordings.

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

Why does severity transfer collapse on 0.014" defects? The root cause lies in a non-monotonic physical response: defect diameter does not scale linearly with vibration amplitude (`results/phase4b_class_severity_means.csv`).

When training on 0.007" and 0.021" defects, classifiers learn to associate Outer Race damage with intense vibration: 0.007" Outer Race produces a mean RMS of 0.4646 with a pronounced BPFO envelope SNR of 13.3187, while 0.021" Outer Race exhibits 0.4384 RMS and 4.4449 envelope SNR. The intermediate 0.014" Outer Race recording behaves completely differently. Its mean RMS drops seven-fold to 0.0663, its BPFO envelope SNR registers a modest 1.6482, and its kurtosis settles at a near-Gaussian 3.0046.

Classifiers misallocate these muted vibrations to classes with matching amplitudes (`results/phase4_confusion.csv`). On `sev_0.014`, Random Forest predicts 227 Outer Race test windows as Normal and 243 as Ball (0 out of 470 predicted correctly, yielding 0.0000 recall). Inner Race faults exhibit a parallel failure: 0.014" Inner Race produces a mean RMS of 0.1371, placing it right within the Ball amplitude range (0.1027 to 0.1279). Random Forest misclassifies all 468 windows of 0.014" Inner Race as Ball (0.0000 recall). Across all eight 0.014" test files covering both raceway classes (files 169–172 and 197–200), window accuracy is exactly 0.0000 (`results/phase4_per_file.csv`).

Unsupervised DBSCAN confirms this overlap independently (`results/phase5_dbscan_cluster_files.csv`). While DBSCAN isolates 0.007" and 0.021" raceway faults into distinct clusters, 0.014" Inner Race and Outer Race recordings merge directly into one large cluster alongside Normal and Ball files. Whether the 0.014" amplitude drop reflects plastic deformation smoothing the defect edges or mechanical mounting differences on the test bench remains unverified, but the failure mode is consistent across all models.

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

We evaluated out-of-distribution novelty detection using Isolation Forest across the four load folds with threshold-free AUROC (`results/phase5_summary.md`).

In Experiment 1A (detecting an unseen fault class), Inner Race and Outer Race faults are readily flagged as anomalies, reaching mean AUROCs of 0.9458 ± 0.0249 and 0.9318 ± 0.0078. Ball faults fail to separate, returning a mean AUROC of 0.4464 ± 0.0242—worse than random guessing (and collapsing to 0.1596 ± 0.0979 against Normal). Because rolling elements tumble dynamically through the load zone, Ball fault energy smears across the spectrum, making Ball windows appear less anomalous to the tree model than baseline healthy recordings.

In Experiment 1B (training exclusively on healthy Normal baseline windows to detect any damage), the full 21-feature pipeline scores an AUROC of 0.9650 ± 0.0319. But an elementary one-feature RMS baseline achieves a perfect 1.0000 ± 0.0000 across all four folds and sub-comparisons. This pristine score is deceptive: Normal recordings are uniformly quieter than all fault recordings, while the quietest fault group (0.014" Outer Race, mean RMS 0.0663) remains louder than the healthy baseline (mean RMS 0.0446). Because Normal files originate from 48 kHz recordings, this loudness gap reflects recording conditions rather than learned fault kinematics.

Unsupervised DBSCAN clustering yields a modest Adjusted Rand Index (ARI) of 0.1584 ± 0.0667 and Normalized Mutual Information (NMI) of 0.3611 ± 0.1060 across the four load splits (`results/phase5_summary.md`). The mean noise fraction remains small at 0.0151 ± 0.0042 (11 to 23 windows per fold). Cluster auditing (`results/phase5_dbscan_cluster_files.csv`) reveals that every cluster other than the primary component consists solely of windows from a single recording (specifically 0.007" and 0.021" Inner and Outer Race files), while Normal, Ball, and 0.014" files blend into one diffuse mass.

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

Clean-trained classifiers were stress-tested under additive white Gaussian noise injected prior to the 3500 Hz low-pass filter across four signal-to-noise ratios (30, 20, 10, and 5 dB) under two formulations (`results/phase6_summary.md`). In Variant A, noise power scales proportionally to each file's AC power ($P_{\text{file}}$). Tree ensembles degrade gracefully under this proportional noise: Random Forest macro-F1 shifts from clean 0.9852 ± 0.0249 to 0.9848 at 20 dB, 0.9819 at 10 dB, and 0.9216 at 5 dB (sample std, ddof=1). XGBoost follows a similar trajectory (0.9835 clean to 0.8998 at 5 dB). SVM holds at 0.9661 at 20 dB, but drops steeply to 0.6158 at 10 dB and 0.2720 at 5 dB.

In Variant B, noise variance is pinned to the median power of all 36 fault recordings ($P_{\text{ref}} = 0.0300$). Because Normal baseline files have lower baseline energy (mean RMS 0.0446 vs ~0.173 for $P_{\text{ref}}$), fixed noise pushes their effective post-filter SNR down to -0.16 to 1.05 dB at nominal 10 dB. As noise inflates Normal window RMS from 0.0446 to 0.0607, tree models lose track of the healthy baseline. Random Forest Normal recall collapses from 1.0000 clean to 0.0029 ± 0.0057 at 10 dB and 0.0000 at 5 dB, dragging load macro-F1 down to 0.6596 (10 dB) and 0.6185 (5 dB). XGBoost shows the same vulnerability. Conversely, SVM maintains Normal recall of 0.9272 at 10 dB and 0.9315 at 5 dB, but its overall macro-F1 falls to 0.6143 and 0.4253.

Analyzing confusion matrices reveals opposing operational failure modes (`results/phase6b_noise_confusion.csv`):
- *Tree ensembles generate false alarms:* At 10 and 5 dB in both variants, Random Forest and XGBoost never classify a damaged bearing as Normal (zero missed faults). In Variant B at 10 dB, Random Forest labels all 409 Normal windows as Outer Race; at 5 dB, it flags 396 as Ball and 13 as Outer Race.
- *SVM produces missed detections:* Under severe noise, SVM defaults toward predicting damaged bearings as healthy Normal. In Variant A at 10 dB, SVM misclassifies 14.95% of fault windows (633/4,234) as Normal; at 5 dB, that rate jumps to 52.22% (2,211/4,234). In Variant B, SVM misses 17.67% of faults at 10 dB and 37.70% at 5 dB.

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

Filter cutoff sensitivity was benchmarked using Random Forest across five cutoff frequencies (3000, 3500, 4000, 4500, and 5000 Hz) plus unfiltered data across all seven outer splits (`results/phase6c_cutoff_summary.md`). Across load splits, macro-F1 remains rock solid: 0.9761 at 3000 Hz, 0.9854 at 3500 Hz, 0.9915 at 4000 Hz, 0.9904 at 4500 Hz, 0.9828 at 5000 Hz, and 0.9857 unfiltered (seed 42, ddof=0). The small differences between cutoffs fall well within cross-split and seed variance.

Cross-severity transfer presents a different story. Higher cutoffs of 4000 Hz (0.6826 ± 0.2412) and 4500 Hz (0.7097 ± 0.2526) noticeably outperform 3500 Hz (0.5790 ± 0.2330) and 3000 Hz (0.5190 ± 0.1905). This gain comes from improved transfer on 0.007" (0.6364 at 3500 Hz vs 0.7082 at 4000 Hz) and 0.021" defects (0.8313 at 3500 Hz vs 0.9645 at 4000 Hz). The `sev_0.014` split, however, stays completely broken under every filter setting (0.2584 at 3000 Hz, 0.2693 at 3500 Hz, 0.3753 at 4000 Hz, 0.3526 unfiltered), with 0.0000 window accuracy across all eight 0.014" files. Normal recall on load splits holds at 1.0000 up to 4000 Hz, dipping slightly to 0.9957 at 4500 Hz and 0.9483 at 5000 Hz and unfiltered.

Power spectral measurements explain why (`results/phase6b_power_above_3500.csv`, `results/phase6c_power_removed.csv`). The 3500 Hz filter attenuates 44.36% of Outer Race power, 26.89% of Inner Race, 23.36% of Ball, and 45.17% of Normal. For raceway faults, the bulk of this removed energy sits between 3500 and 4000 Hz (23.63% for Outer Race, 18.94% for Inner Race). A 4000 Hz cutoff preserves this 3500–4000 Hz band while rejecting frequencies above 4000 Hz. While this suggests that high-frequency raceway harmonics aid severity transfer, the 3500 Hz cutoff was frozen during initial diagnostics as a safeguard against sampling-rate artefacts; altering cutoffs based on test results would constitute retrospective optimization.

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

Ten specific limitations apply to this benchmark:

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
