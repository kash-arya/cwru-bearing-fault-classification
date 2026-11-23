# Phase 4 Benchmark & Ablation Summary

## Table 1: Model Benchmark Across Splits (Macro-F1)

| Split | SVM | RF | XGB |
| --- | --- | --- | --- |
| lolo_0 | 0.9865 | 0.9972 | 0.9929 |
| lolo_1 | 0.9950 | 0.9993 | 0.9993 |
| lolo_2 | 0.9893 | 0.9965 | 0.9957 |
| lolo_3 | 0.9332 | 0.9478 | 0.9461 |
| sev_0.007 | 0.7906 | 0.6364 | 0.4384 |
| sev_0.014 | 0.2482 | 0.2691 | 0.3747 |
| sev_0.021 | 0.7517 | 0.9281 | 0.6483 |
| LOLO mean +- std | 0.9760 +- 0.0249 | 0.9852 +- 0.0216 | 0.9835 +- 0.0217 |
| Severity mean +- std | 0.5968 +- 0.2470 | 0.6112 +- 0.2696 | 0.4871 +- 0.1169 |

## Table 2: Per-Class Generalization Metrics

> **Sample size caveat:** Each test split contains exactly 1 Normal recording (58 windows in `lolo_0`, 117 windows across all other splits).

| Model | Class | LOLO Recall | LOLO F1 | Severity Recall | Severity F1 |
| --- | --- | --- | --- | --- | --- |
| SVM | Normal | 1.0000 +- 0.0000 | 0.9989 +- 0.0021 | 1.0000 +- 0.0000 | 0.6260 +- 0.3386 |
| SVM | Inner Race | 0.9393 +- 0.1177 | 0.9559 +- 0.0617 | 0.6172 +- 0.5277 | 0.5556 +- 0.4818 |
| SVM | Ball | 0.9759 +- 0.0232 | 0.9560 +- 0.0492 | 0.9568 +- 0.0447 | 0.8428 +- 0.1797 |
| SVM | Outer Race | 0.9901 +- 0.0082 | 0.9932 +- 0.0038 | 0.2681 +- 0.2884 | 0.3627 +- 0.3624 |
| RF | Normal | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | 0.8359 +- 0.2843 |
| RF | Inner Race | 0.9541 +- 0.0899 | 0.9723 +- 0.0488 | 0.3333 +- 0.5774 | 0.3197 +- 0.5538 |
| RF | Ball | 0.9929 +- 0.0067 | 0.9727 +- 0.0452 | 0.9993 +- 0.0012 | 0.7046 +- 0.1830 |
| RF | Outer Race | 0.9936 +- 0.0127 | 0.9957 +- 0.0061 | 0.5218 +- 0.4556 | 0.5846 +- 0.5076 |
| XGB | Normal | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | 0.9972 +- 0.0049 |
| XGB | Inner Race | 0.9491 +- 0.0979 | 0.9694 +- 0.0537 | 0.3800 +- 0.5410 | 0.3553 +- 0.4201 |
| XGB | Ball | 0.9887 +- 0.0120 | 0.9696 +- 0.0450 | 0.9922 +- 0.0075 | 0.5962 +- 0.1515 |
| XGB | Outer Race | 0.9965 +- 0.0071 | 0.9951 +- 0.0052 | 0.0000 +- 0.0000 | 0.0000 +- 0.0000 |

## Table 3: Physics-Informed Feature Ablation (Random Forest)

| Ablation ID | Features | N | lolo_0 | lolo_1 | lolo_2 | lolo_3 | sev_0.007 | sev_0.014 | sev_0.021 | LOLO Mean+-Std | Sev Mean+-Std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1_dummy_majority | none | 0 | 0.1201 | 0.1153 | 0.1153 | 0.1155 | 0.1174 | 0.1177 | 0.1179 | 0.1166 +- 0.0021 | 0.1177 +- 0.0002 |
| 2_rms_only | rms | 1 | 0.7865 | 0.7822 | 0.7590 | 0.7581 | 0.4928 | 0.2681 | 0.6539 | 0.7715 +- 0.0130 | 0.4716 +- 0.1582 |
| 3_time_only | time | 9 | 0.9645 | 0.9704 | 0.9929 | 0.9143 | 0.4163 | 0.2643 | 0.6656 | 0.9605 +- 0.0287 | 0.4487 +- 0.1654 |
| 4_time_freq | time+freq | 15 | 0.9815 | 0.9993 | 0.9950 | 0.9132 | 0.4163 | 0.2697 | 0.6667 | 0.9723 +- 0.0347 | 0.4509 +- 0.1639 |
| 5_full_lp3500 | full | 21 | 0.9972 | 1.0000 | 0.9965 | 0.9478 | 0.6364 | 0.2693 | 0.8313 | 0.9854 +- 0.0217 | 0.5790 +- 0.2330 |
| 6_full_nolp | full | 21 | 0.9649 | 1.0000 | 0.9979 | 0.9803 | 0.8013 | 0.3526 | 0.9223 | 0.9857 +- 0.0143 | 0.6921 +- 0.2451 |
| 7_lp3500_no_loudness | no_loudness | 18 | 0.9993 | 1.0000 | 0.9865 | 0.9550 | 0.3747 | 0.3891 | 0.6483 | 0.9852 +- 0.0183 | 0.4707 +- 0.1257 |

> **RMS / STD equivalence:** Because windows are demeaned prior to feature extraction, RMS and standard deviation are mathematically identical. Ablation row 2 therefore reflects evaluating standard deviation alone.
