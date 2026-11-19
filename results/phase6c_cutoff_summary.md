# Phase 6C: Low-Pass Anti-Aliasing Cutoff Sensitivity Report

> [!NOTE]
> **Caveats:**
> - Evaluated strictly with Random Forest using fixed pilot settings (300 trees, `class_weight="balanced"`, seeds [42, 0, 1]).
> - The envelope band (1500–3000 Hz) and the 3500 Hz cutoff were chosen together in Phase 2, so 3500 may have a home advantage.
> - At $C = 3000\text{ Hz}$, the filter also attenuates the upper edge of the 1500–3000 Hz envelope demodulation band.
> - The cross-severity results rest on 3 splits with large spreads.
> - Standard deviations are population standard deviations (`ddof=0`), consistent with Phase 4 and Phase 5 summaries.

## Table 1: Macro-F1 across Outer Splits per Cutoff (Seed 42)
| Cutoff | lolo_0 | lolo_1 | lolo_2 | lolo_3 | LOLO Mean ± Std | sev_0.007 | sev_0.014 | sev_0.021 | Severity Mean ± Std | Mean Seed Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **3000** | 0.9993 | 0.9499 | 0.9965 | 0.9589 | 0.9761 ± 0.0220 | 0.7086 | 0.2584 | 0.5900 | 0.5190 ± 0.1905 | 0.0120 |
| **3500** | 0.9972 | 1.0000 | 0.9965 | 0.9478 | 0.9854 ± 0.0217 | 0.6364 | 0.2693 | 0.8313 | 0.5790 ± 0.2330 | 0.0035 |
| **4000** | 0.9915 | 1.0000 | 0.9972 | 0.9775 | 0.9915 ± 0.0086 | 0.7082 | 0.3753 | 0.9645 | 0.6826 ± 0.2412 | 0.0063 |
| **4500** | 0.9918 | 1.0000 | 0.9929 | 0.9767 | 0.9904 ± 0.0085 | 0.8364 | 0.3571 | 0.9355 | 0.7097 ± 0.2526 | 0.0064 |
| **5000** | 0.9613 | 1.0000 | 0.9929 | 0.9768 | 0.9828 ± 0.0150 | 0.7542 | 0.3485 | 0.9202 | 0.6743 ± 0.2401 | 0.0070 |
| **none** | 0.9649 | 1.0000 | 0.9979 | 0.9803 | 0.9857 ± 0.0143 | 0.8013 | 0.3526 | 0.9223 | 0.6921 ± 0.2451 | 0.0117 |

## Table 2: Per-Class Recall per Cutoff (Seed 42)
| Cutoff | Fault Class | LOLO Mean Recall | Severity Mean Recall | sev_0.014 Recall |
| :--- | :--- | :--- | :--- | :--- |
| **3000** | Normal | 1.0000 | 1.0000 | 1.0000 |
| **3000** | Inner Race | 0.9753 | 0.5702 | 0.0000 |
| **3000** | Ball | 0.9460 | 0.9271 | 1.0000 |
| **3000** | Outer Race | 0.9831 | 0.0432 | 0.0000 |
| **3500** | Normal | 1.0000 | 1.0000 | 1.0000 |
| **3500** | Inner Race | 0.9548 | 0.1816 | 0.0000 |
| **3500** | Ball | 0.9929 | 0.9993 | 0.9979 |
| **3500** | Outer Race | 0.9936 | 0.5472 | 0.0000 |
| **4000** | Normal | 1.0000 | 1.0000 | 1.0000 |
| **4000** | Inner Race | 0.9936 | 0.3588 | 0.0000 |
| **4000** | Ball | 0.9908 | 0.9667 | 1.0000 |
| **4000** | Outer Race | 0.9816 | 0.6540 | 0.0000 |
| **4500** | Normal | 0.9957 | 1.0000 | 1.0000 |
| **4500** | Inner Race | 0.9972 | 0.4675 | 0.0000 |
| **4500** | Ball | 0.9887 | 0.9278 | 0.9745 |
| **4500** | Outer Race | 0.9781 | 0.6455 | 0.0000 |
| **5000** | Normal | 0.9483 | 1.0000 | 1.0000 |
| **5000** | Inner Race | 0.9958 | 0.3934 | 0.0000 |
| **5000** | Ball | 0.9894 | 0.9278 | 0.9660 |
| **5000** | Outer Race | 0.9788 | 0.6222 | 0.0000 |
| **none** | Normal | 0.9483 | 1.0000 | 1.0000 |
| **none** | Inner Race | 0.9972 | 0.4343 | 0.0000 |
| **none** | Ball | 0.9958 | 0.9342 | 0.9660 |
| **none** | Outer Race | 0.9830 | 0.6179 | 0.0000 |

## Table 3: Mean Fraction of Power Removed by Low-Pass per Fault Class
| Cutoff | Ball | Inner Race | Normal | Outer Race |
| :--- | :--- | :--- | :--- | :--- |
| **3000** | 0.7157 | 0.5583 | 0.4902 | 0.7611 |
| **3500** | 0.2336 | 0.2689 | 0.4517 | 0.4436 |
| **4000** | 0.0188 | 0.0304 | 0.4207 | 0.0581 |
| **4500** | 0.0078 | 0.0038 | 0.0737 | 0.0213 |
| **5000** | 0.0042 | 0.0015 | 0.0240 | 0.0185 |
| **none** | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Table 4: Window Accuracy for Severity 0.014" Files in sev_0.014 and lolo_3 (Seed 42)
| Split | File ID | Fault Class | Load (HP) | Cutoff 3000 | Cutoff 3500 | Cutoff 4000 | Cutoff 4500 | Cutoff 5000 | Cutoff none |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| sev_0.014 | 169 | Inner Race | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sev_0.014 | 170 | Inner Race | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sev_0.014 | 171 | Inner Race | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sev_0.014 | 172 | Inner Race | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sev_0.014 | 197 | Outer Race | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sev_0.014 | 198 | Outer Race | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sev_0.014 | 199 | Outer Race | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sev_0.014 | 200 | Outer Race | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| lolo_3 | 172 | Inner Race | 3 | 0.7009 | 0.4530 | 0.9231 | 0.9658 | 0.9487 | 0.9658 |
| lolo_3 | 200 | Outer Race | 3 | 0.7966 | 0.9237 | 0.8220 | 0.7712 | 0.7797 | 0.8051 |
