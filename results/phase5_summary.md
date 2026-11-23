# Phase 5: Unsupervised Novelty Detection & Clustering Summary

> [!NOTE]
> **Evaluation Caveats:**
> 1. Each LOLO test fold includes only 1 Normal file (58 windows in `lolo_0`, 117 windows in `lolo_1`..`lolo_3`).
> 2. Normal files originate from 48 kHz recordings resampled to 12 kHz, whereas fault files were acquired natively at 12 kHz.
> 3. DBSCAN feature standardization and PCA are fitted directly on each test fold, as unsupervised clustering evaluates natural grouping without train/test partitions.
> 4. Overlapping windows from the same continuous recording share high mutual information; the effective degrees of freedom for clustering and AUROC metrics align with the number of test files (10 per fold) rather than raw window counts. per_class_severity scores compare a single fault file against a single Normal file.

## Table 1: Experiment 1A — Unseen Fault Class Detection (IsolationForest AUROC)

| Variant | Held-out Class | Comparison | lolo_0 | lolo_1 | lolo_2 | lolo_3 | Fold Mean +- Std | Mean Seed Std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lp3500_all21 | Inner Race | overall | 0.9323 | 0.9769 | 0.9611 | 0.9128 | 0.9458 +- 0.0249 | 0.0085 |
| lp3500_all21 | Inner Race | vs_Normal | 0.7707 | 0.9567 | 0.9996 | 0.9531 | 0.9200 +- 0.0881 | 0.0335 |
| lp3500_all21 | Inner Race | vs_Ball | 0.9521 | 0.9984 | 0.9878 | 0.9595 | 0.9744 +- 0.0192 | 0.0025 |
| lp3500_all21 | Inner Race | vs_Outer Race | 0.9391 | 0.9624 | 0.9214 | 0.8530 | 0.9190 +- 0.0408 | 0.0131 |
| lp3500_all21 | Ball | overall | 0.4241 | 0.4806 | 0.4579 | 0.4231 | 0.4464 +- 0.0242 | 0.0378 |
| lp3500_all21 | Ball | vs_Normal | 0.0650 | 0.0599 | 0.2392 | 0.2742 | 0.1596 +- 0.0979 | 0.0369 |
| lp3500_all21 | Ball | vs_Inner Race | 0.6038 | 0.6035 | 0.6161 | 0.5061 | 0.5824 +- 0.0443 | 0.0517 |
| lp3500_all21 | Ball | vs_Outer Race | 0.3039 | 0.4974 | 0.3723 | 0.3892 | 0.3907 +- 0.0694 | 0.0418 |
| lp3500_all21 | Outer Race | overall | 0.9362 | 0.9220 | 0.9420 | 0.9270 | 0.9318 +- 0.0078 | 0.0095 |
| lp3500_all21 | Outer Race | vs_Normal | 0.7234 | 0.7892 | 0.8756 | 0.8402 | 0.8071 +- 0.0573 | 0.0357 |
| lp3500_all21 | Outer Race | vs_Inner Race | 0.9390 | 0.9034 | 0.9191 | 0.9148 | 0.9191 +- 0.0128 | 0.0141 |
| lp3500_all21 | Outer Race | vs_Ball | 0.9684 | 0.9846 | 0.9869 | 0.9680 | 0.9770 +- 0.0088 | 0.0028 |
| lp3500_no_loudness18 | Inner Race | overall | 0.9194 | 0.9467 | 0.9588 | 0.8984 | 0.9308 +- 0.0235 | 0.0143 |
| lp3500_no_loudness18 | Inner Race | vs_Normal | 0.6146 | 0.8128 | 0.9976 | 0.9241 | 0.8373 +- 0.1444 | 0.0414 |
| lp3500_no_loudness18 | Inner Race | vs_Ball | 0.9476 | 0.9950 | 0.9858 | 0.9507 | 0.9698 +- 0.0209 | 0.0040 |
| lp3500_no_loudness18 | Inner Race | vs_Outer Race | 0.9413 | 0.9429 | 0.9188 | 0.8377 | 0.9102 +- 0.0429 | 0.0213 |
| lp3500_no_loudness18 | Ball | overall | 0.3961 | 0.3475 | 0.3413 | 0.3221 | 0.3518 +- 0.0273 | 0.0236 |
| lp3500_no_loudness18 | Ball | vs_Normal | 0.0418 | 0.0225 | 0.1368 | 0.1795 | 0.0952 +- 0.0651 | 0.0207 |
| lp3500_no_loudness18 | Ball | vs_Inner Race | 0.5769 | 0.4932 | 0.5098 | 0.4149 | 0.4987 +- 0.0576 | 0.0244 |
| lp3500_no_loudness18 | Ball | vs_Outer Race | 0.2741 | 0.3101 | 0.2408 | 0.2763 | 0.2753 +- 0.0245 | 0.0305 |
| lp3500_no_loudness18 | Outer Race | overall | 0.9215 | 0.9038 | 0.9459 | 0.9192 | 0.9226 +- 0.0151 | 0.0073 |
| lp3500_no_loudness18 | Outer Race | vs_Normal | 0.5465 | 0.7010 | 0.8793 | 0.8068 | 0.7334 +- 0.1252 | 0.0175 |
| lp3500_no_loudness18 | Outer Race | vs_Inner Race | 0.9506 | 0.8951 | 0.9313 | 0.9235 | 0.9251 +- 0.0199 | 0.0131 |
| lp3500_no_loudness18 | Outer Race | vs_Ball | 0.9542 | 0.9799 | 0.9826 | 0.9521 | 0.9672 +- 0.0141 | 0.0033 |

## Table 2: Experiment 1B — Normal-Only Fault Detection (AUROC)

| Variant | Subset Type | Fault Class | Severity | lolo_0 | lolo_1 | lolo_2 | lolo_3 | Fold Mean +- Std | Mean Seed Std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lp3500_all21 | overall | ALL | ALL | 0.9109 | 0.9744 | 0.9934 | 0.9811 | 0.9650 +- 0.0319 | 0.0093 |
| lp3500_all21 | per_class | Inner Race | ALL | 0.9977 | 1.0000 | 1.0000 | 1.0000 | 0.9994 +- 0.0010 | 0.0006 |
| lp3500_all21 | per_class | Ball | ALL | 0.8159 | 0.9445 | 0.9859 | 0.9508 | 0.9243 +- 0.0645 | 0.0218 |
| lp3500_all21 | per_class | Outer Race | ALL | 0.9194 | 0.9787 | 0.9944 | 0.9923 | 0.9712 +- 0.0305 | 0.0063 |
| lp3500_all21 | per_class_severity | Inner Race | 0.007 | 0.9997 | 1.0000 | 1.0000 | 1.0000 | 0.9999 +- 0.0001 | 0.0001 |
| lp3500_all21 | per_class_severity | Inner Race | 0.014 | 0.9934 | 1.0000 | 1.0000 | 1.0000 | 0.9983 +- 0.0029 | 0.0017 |
| lp3500_all21 | per_class_severity | Inner Race | 0.021 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +- 0.0000 | 0.0000 |
| lp3500_all21 | per_class_severity | Ball | 0.007 | 0.6375 | 0.8815 | 0.9780 | 0.9401 | 0.8593 +- 0.1326 | 0.0421 |
| lp3500_all21 | per_class_severity | Ball | 0.014 | 0.9552 | 0.9989 | 0.9995 | 0.9972 | 0.9877 +- 0.0188 | 0.0028 |
| lp3500_all21 | per_class_severity | Ball | 0.021 | 0.8562 | 0.9527 | 0.9801 | 0.9150 | 0.9260 +- 0.0464 | 0.0225 |
| lp3500_all21 | per_class_severity | Outer Race | 0.007 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +- 0.0000 | 0.0000 |
| lp3500_all21 | per_class_severity | Outer Race | 0.014 | 0.7570 | 0.9362 | 0.9833 | 0.9770 | 0.9134 +- 0.0921 | 0.0189 |
| lp3500_all21 | per_class_severity | Outer Race | 0.021 | 0.9999 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +- 0.0001 | 0.0000 |
| lp3500_no_loudness18 | overall | ALL | ALL | 0.8476 | 0.8857 | 0.9662 | 0.9196 | 0.9048 +- 0.0437 | 0.0089 |
| lp3500_no_loudness18 | per_class | Inner Race | ALL | 0.9896 | 1.0000 | 1.0000 | 1.0000 | 0.9974 +- 0.0045 | 0.0005 |
| lp3500_no_loudness18 | per_class | Ball | ALL | 0.6943 | 0.7572 | 0.9170 | 0.7998 | 0.7921 +- 0.0813 | 0.0204 |
| lp3500_no_loudness18 | per_class | Outer Race | ALL | 0.8593 | 0.8997 | 0.9817 | 0.9588 | 0.9249 +- 0.0482 | 0.0072 |
| lp3500_no_loudness18 | per_class_severity | Inner Race | 0.007 | 0.9956 | 1.0000 | 1.0000 | 1.0000 | 0.9989 +- 0.0019 | 0.0005 |
| lp3500_no_loudness18 | per_class_severity | Inner Race | 0.014 | 0.9734 | 1.0000 | 1.0000 | 1.0000 | 0.9933 +- 0.0115 | 0.0012 |
| lp3500_no_loudness18 | per_class_severity | Inner Race | 0.021 | 0.9996 | 1.0000 | 1.0000 | 1.0000 | 0.9999 +- 0.0002 | 0.0001 |
| lp3500_no_loudness18 | per_class_severity | Ball | 0.007 | 0.4292 | 0.5129 | 0.8678 | 0.7371 | 0.6368 +- 0.1746 | 0.0374 |
| lp3500_no_loudness18 | per_class_severity | Ball | 0.014 | 0.9133 | 0.9932 | 0.9978 | 0.9856 | 0.9725 +- 0.0344 | 0.0028 |
| lp3500_no_loudness18 | per_class_severity | Ball | 0.021 | 0.7421 | 0.7635 | 0.8850 | 0.6762 | 0.7667 +- 0.0755 | 0.0254 |
| lp3500_no_loudness18 | per_class_severity | Outer Race | 0.007 | 0.9977 | 1.0000 | 1.0000 | 1.0000 | 0.9994 +- 0.0010 | 0.0006 |
| lp3500_no_loudness18 | per_class_severity | Outer Race | 0.014 | 0.5808 | 0.6991 | 0.9449 | 0.8764 | 0.7753 +- 0.1437 | 0.0222 |
| lp3500_no_loudness18 | per_class_severity | Outer Race | 0.021 | 0.9972 | 1.0000 | 1.0000 | 1.0000 | 0.9993 +- 0.0012 | 0.0005 |
| rms_only_baseline | overall | ALL | ALL | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +- 0.0000 | 0.0000 |
| rms_only_baseline | per_class | Inner Race | ALL | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +- 0.0000 | 0.0000 |
| rms_only_baseline | per_class | Ball | ALL | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +- 0.0000 | 0.0000 |
| rms_only_baseline | per_class | Outer Race | ALL | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +- 0.0000 | 0.0000 |
| rms_only_baseline | per_class_severity | Inner Race | 0.007 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +- 0.0000 | 0.0000 |
| rms_only_baseline | per_class_severity | Inner Race | 0.014 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +- 0.0000 | 0.0000 |
| rms_only_baseline | per_class_severity | Inner Race | 0.021 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +- 0.0000 | 0.0000 |
| rms_only_baseline | per_class_severity | Ball | 0.007 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +- 0.0000 | 0.0000 |
| rms_only_baseline | per_class_severity | Ball | 0.014 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +- 0.0000 | 0.0000 |
| rms_only_baseline | per_class_severity | Ball | 0.021 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +- 0.0000 | 0.0000 |
| rms_only_baseline | per_class_severity | Outer Race | 0.007 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +- 0.0000 | 0.0000 |
| rms_only_baseline | per_class_severity | Outer Race | 0.014 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +- 0.0000 | 0.0000 |
| rms_only_baseline | per_class_severity | Outer Race | 0.021 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +- 0.0000 | 0.0000 |
| nolp_all21 | overall | ALL | ALL | 0.8987 | 0.9883 | 0.9994 | 0.9805 | 0.9667 +- 0.0399 | 0.0033 |
| nolp_all21 | per_class | Inner Race | ALL | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +- 0.0000 | 0.0000 |
| nolp_all21 | per_class | Ball | ALL | 0.7863 | 0.9763 | 0.9985 | 0.9483 | 0.9274 +- 0.0834 | 0.0070 |
| nolp_all21 | per_class | Outer Race | ALL | 0.9101 | 0.9885 | 0.9998 | 0.9932 | 0.9729 +- 0.0365 | 0.0031 |
| nolp_all21 | per_class_severity | Inner Race | 0.007 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +- 0.0000 | 0.0000 |
| nolp_all21 | per_class_severity | Inner Race | 0.014 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +- 0.0000 | 0.0000 |
| nolp_all21 | per_class_severity | Inner Race | 0.021 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +- 0.0000 | 0.0000 |
| nolp_all21 | per_class_severity | Ball | 0.007 | 0.4620 | 0.9461 | 0.9968 | 0.9188 | 0.8310 +- 0.2148 | 0.0119 |
| nolp_all21 | per_class_severity | Ball | 0.014 | 0.9419 | 0.9992 | 1.0000 | 0.9973 | 0.9846 +- 0.0247 | 0.0020 |
| nolp_all21 | per_class_severity | Ball | 0.021 | 0.9563 | 0.9833 | 0.9987 | 0.9286 | 0.9667 +- 0.0267 | 0.0082 |
| nolp_all21 | per_class_severity | Outer Race | 0.007 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +- 0.0000 | 0.0000 |
| nolp_all21 | per_class_severity | Outer Race | 0.014 | 0.7287 | 0.9656 | 0.9994 | 0.9795 | 0.9183 +- 0.1101 | 0.0094 |
| nolp_all21 | per_class_severity | Outer Race | 0.021 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +- 0.0000 | 0.0000 |

## Table 3: Experiment 2 — DBSCAN Test-Fold Clustering Summary

| Split | N | Features | PCs | Var Retained | min_samples | eps_auto | Clusters | Noise Frac | ARI (Noise Clust) | NMI (Noise Clust) | ARI (No Noise) | NMI (No Noise) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lolo_0 | 1116 | 20 | 10 | 0.9632 | 20 | 2.5430 | 2 | 0.0188 | 0.0554 | 0.1857 | 0.0452 | 0.1769 |
| lolo_1 | 1175 | 20 | 9 | 0.9501 | 18 | 2.0194 | 5 | 0.0128 | 0.2175 | 0.4449 | 0.2212 | 0.4556 |
| lolo_2 | 1174 | 20 | 9 | 0.9515 | 18 | 2.1828 | 5 | 0.0094 | 0.2172 | 0.4454 | 0.2179 | 0.4523 |
| lolo_3 | 1178 | 20 | 10 | 0.9642 | 20 | 2.3737 | 4 | 0.0195 | 0.1436 | 0.3683 | 0.1441 | 0.3816 |
| Mean +- Std | - | 20 | 9.5 | 0.9573 | 19.0 | 2.2797 +- 0.1970 | 4.00 | 0.0151 +- 0.0042 | 0.1584 +- 0.0667 | 0.3611 +- 0.1060 | 0.1571 +- 0.0716 | 0.3666 +- 0.1134 |

## Table 4: Experiment 2 — DBSCAN Epsilon Sensitivity Sweep

| Multiplier | lolo_0 (ARI / Noise) | lolo_1 (ARI / Noise) | lolo_2 (ARI / Noise) | lolo_3 (ARI / Noise) | Mean Clusters | Mean ARI | Mean Noise |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.50 | 0.3552 / 0.3065 | 0.1954 / 0.5209 | 0.2903 / 0.4446 | 0.3353 / 0.4194 | 5.2 | 0.2940 +- 0.0616 | 0.4228 +- 0.0769 |
| 0.75 | 0.4392 / 0.1013 | 0.3522 / 0.0885 | 0.3638 / 0.0801 | 0.2598 / 0.1358 | 5.8 | 0.3537 +- 0.0637 | 0.1014 +- 0.0212 |
| 1.00 | 0.0554 / 0.0188 | 0.2175 / 0.0128 | 0.2172 / 0.0094 | 0.1436 / 0.0195 | 4.0 | 0.1584 +- 0.0667 | 0.0151 +- 0.0042 |
| 1.25 | 0.0458 / 0.0072 | 0.2131 / 0.0043 | 0.1257 / 0.0034 | 0.0342 / 0.0042 | 3.0 | 0.1047 +- 0.0718 | 0.0048 +- 0.0014 |
| 1.50 | 0.0413 / 0.0018 | 0.0324 / 0.0017 | 0.0329 / 0.0009 | 0.0321 / 0.0000 | 2.0 | 0.0347 +- 0.0038 | 0.0011 +- 0.0007 |

