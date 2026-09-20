# PHY-ML R0 Canonical Baseline: Final Packaging & Freeze Report

**Freeze Status:** 🟢 **PASS**  
**Generated At (UTC):** `2026-09-20T16:04:07.960078+00:00`  
**Source Git Commit:** `193d864f5570664a6bfc8714ca3c8766bd1aef8c`  
**Target Architecture:** Canonical R0 Baseline (Slow Rayleigh Flat Block Fading, Perfect CSI, Uncoded AMC)  
**Output Directory:** `results/r0_final`  

---

## 1. Executive Summary & Provenance Manifest

This document certifies the final packaging and freeze of the **PHY-ML R0 baseline**.
The canonical dataset merges the verified **Deep 70k blocks/seed** (140,000 pooled blocks/point) Monte Carlo calibration table with targeted **0.25-dB transition refinement points** at switching boundaries, forming a unified **28-point operating grid**.

### Cryptographic Artifact Provenance

| Artifact / Parameter | Value / Hash |
|:---------------------|:-------------|
| Source Git Commit | `193d864f5570664a6bfc8714ca3c8766bd1aef8c` |
| Deep Calibration CSV | `results/r0_mc_deep_70k/calibration_1d_cuda_pooled.csv` |
| Deep CSV SHA-256 | `bcca900dbc410d0ab274a2b893eb5c1fa478ce2ef560882828f85a1c3f27fd7e` |
| Refinement Calibration CSV | `results/r0_snr_grid_refine_025/calibration_1d_cuda_pooled.csv` |
| Refinement CSV SHA-256 | `f9de0818c85ab54eacdffb31d784ac62affd87594c916bfdedcdc141b253b1c2` |
| Error Constraint | $\text{BER}_{\text{target}} = 0.01$ |
| Confidence Multiplier | $k = 1.96$ (Conservative Upper 95% CI) |
| Monte Carlo Budget | Deep 70,000 blocks/seed (140,000 pooled) |
| Canonical RNG Seeds | Seed A = `20260918`, Seed B = `20260919` |
| Transmission Block Size | `1536` complex symbols/block |
| Total Grid Points | `28` unique SNR evaluations |

---

## 2. R0 Freeze Gate Audit

- **Overall Freeze Decision:** 🟢 **PASS**
- **Grid Integrity:** `PASS` (28 / 28 points present, zero duplicates).
- **BestMode Monotonicity:** `PASS` (0 regressions detected).
- **CART Model Selection Fidelity:** `PASS` (100.00% training accuracy).
- **Ground Truth Label Uncertainty:** `PASS (0 uncertain points)`.
- **Material Target-Overlapping CIs:** `PASS (0 material overlaps)`.
- **Non-Material Candidate Overlaps:** `0` instances recorded (dominated lower-rate modes; cannot affect selected policy).

---

## 3. Decision Tree (CART) Model Selection

Under the canonical model-selection policy, maximum tree depth was swept over $d \in [1, 5]$ to select the smallest depth achieving 100% fidelity to the canonical Ground Truth labels:

| Max Depth | Actual Depth | Node Count | Leaf Count | Training Fidelity | Learned Split Thresholds (dB) |
|:---------:|:------------:|:----------:|:----------:|:-----------------:|:------------------------------|
|         1            |            1 |          3 |          2 |           64.29% | `[16.875]` |
|         2            |            2 |          5 |          3 |           82.14% | `[16.875, 22.875]` |
|         3 **(Selected)** |            3 |          7 |          4 |          100.00% | `[16.875, 22.875, 28.125]` |
|         4            |            3 |          7 |          4 |          100.00% | `[16.875, 22.875, 28.125]` |
|         5            |            3 |          7 |          4 |          100.00% | `[16.875, 22.875, 28.125]` |

- **Optimal Selected Depth:** `3`
- **Actual Tree Depth:** `3`
- **Learned Switching Thresholds:** `[16.875, 22.875, 28.125]`
- **Classification Fidelity:** `100.00%`

---

## 4. Policy & Baseline Comparative Analysis

> [!IMPORTANT]
> **Note on Sampled-Grid Averages:** Averages reported below represent the **sampled-grid average** across the discrete 28 evaluation points in this calibration set. They do NOT represent the ergodic or network-wide expected throughput over a continuous Rayleigh channel.

| Adaptation Policy | Sampled-Grid Mean Spectral Efficiency (bpcu) | Reliability Violations (BER > 0.01) | Violation Rate | Fidelity to Ground Truth | Structural Complexity |
|:-------------------|:--------------------------------------------:|:-----------------------------------:|:--------------:|:------------------------:|:----------------------|
| Ground Truth / 1D LUT |                                        2.679 |                                   7 |         25.0% | 100.0%                   | 3 switching intervals |
| Learned Decision Tree |                                        2.679 |                                   7 |         25.0% |                  100.0% | Depth 3 (4 leaves) |
| Fixed Robust (BPSK)   |                                        1.000 |                                   7 |         25.0% | N/A (Fixed)              | Static single-mode |
| Fixed High-TP (64QAM) |                                        6.000 |                                  23 |         82.1% | N/A (Fixed)              | Static single-mode |

---

## 5. Canonical Operating Points Table

| SNR (dB) | BestMode (GT) | Spectral Eff. | BPSK BER | QPSK BER | 16QAM BER | 64QAM BER | DT Prediction | DT Correct? |
|:--------:|:-------------:|:-------------:|:--------:|:--------:|:---------:|:---------:|:-------------:|:-----------:|
|     0.00 | BPSK          |             1 | 1.46e-01 | 2.11e-01 |  3.16e-01 |  3.74e-01 | BPSK          | YES         |
|     2.00 | BPSK          |             1 | 1.08e-01 | 1.68e-01 |  2.78e-01 |  3.46e-01 | BPSK          | YES         |
|     4.00 | BPSK          |             1 | 7.72e-02 | 1.27e-01 |  2.37e-01 |  3.13e-01 | BPSK          | YES         |
|     6.00 | BPSK          |             1 | 5.34e-02 | 9.26e-02 |  1.96e-01 |  2.78e-01 | BPSK          | YES         |
|     8.00 | BPSK          |             1 | 3.55e-02 | 6.44e-02 |  1.56e-01 |  2.41e-01 | BPSK          | YES         |
|    10.00 | BPSK          |             1 | 2.34e-02 | 4.39e-02 |  1.20e-01 |  2.04e-01 | BPSK          | YES         |
|    12.00 | BPSK          |             1 | 1.52e-02 | 2.91e-02 |  8.81e-02 |  1.67e-01 | BPSK          | YES         |
|    14.00 | BPSK          |             1 | 9.50e-03 | 1.86e-02 |  6.18e-02 |  1.32e-01 | BPSK          | YES         |
|    16.00 | BPSK          |             1 | 6.20e-03 | 1.22e-02 |  4.23e-02 |  1.00e-01 | BPSK          | YES         |
|    16.50 | BPSK          |             1 | 5.45e-03 | 1.08e-02 |  3.83e-02 |  9.32e-02 | BPSK          | YES         |
|    16.75 | BPSK          |             1 | 5.23e-03 | 1.02e-02 |  3.62e-02 |  8.95e-02 | BPSK          | YES         |
|    17.00 | QPSK          |             2 | 4.84e-03 | 9.66e-03 |  3.49e-02 |  8.69e-02 | QPSK          | YES         |
|    17.50 | QPSK          |             2 | 4.39e-03 | 8.68e-03 |  3.13e-02 |  7.98e-02 | QPSK          | YES         |
|    18.00 | QPSK          |             2 | 4.11e-03 | 7.98e-03 |  2.85e-02 |  7.40e-02 | QPSK          | YES         |
|    20.00 | QPSK          |             2 | 2.42e-03 | 4.81e-03 |  1.82e-02 |  5.17e-02 | QPSK          | YES         |
|    22.00 | QPSK          |             2 | 1.54e-03 | 3.08e-03 |  1.19e-02 |  3.56e-02 | QPSK          | YES         |
|    22.50 | QPSK          |             2 | 1.37e-03 | 2.73e-03 |  1.06e-02 |  3.23e-02 | QPSK          | YES         |
|    22.75 | QPSK          |             2 | 1.37e-03 | 2.70e-03 |  1.01e-02 |  3.06e-02 | QPSK          | YES         |
|    23.00 | 16QAM         |             4 | 1.25e-03 | 2.48e-03 |  9.56e-03 |  2.92e-02 | 16QAM         | YES         |
|    23.50 | 16QAM         |             4 | 1.13e-03 | 2.25e-03 |  8.57e-03 |  2.64e-02 | 16QAM         | YES         |
|    24.00 | 16QAM         |             4 | 1.00e-03 | 1.99e-03 |  7.64e-03 |  2.37e-02 | 16QAM         | YES         |
|    26.00 | 16QAM         |             4 | 6.55e-04 | 1.30e-03 |  4.97e-03 |  1.57e-02 | 16QAM         | YES         |
|    28.00 | 16QAM         |             4 | 3.93e-04 | 7.75e-04 |  3.11e-03 |  1.02e-02 | 16QAM         | YES         |
|    28.25 | 64QAM         |             6 | 4.03e-04 | 7.77e-04 |  2.96e-03 |  9.58e-03 | 64QAM         | YES         |
|    28.50 | 64QAM         |             6 | 3.40e-04 | 6.92e-04 |  2.77e-03 |  9.07e-03 | 64QAM         | YES         |
|    29.00 | 64QAM         |             6 | 2.89e-04 | 5.79e-04 |  2.40e-03 |  8.02e-03 | 64QAM         | YES         |
|    29.50 | 64QAM         |             6 | 3.05e-04 | 5.94e-04 |  2.28e-03 |  7.41e-03 | 64QAM         | YES         |
|    30.00 | 64QAM         |             6 | 2.78e-04 | 5.27e-04 |  1.99e-03 |  6.53e-03 | 64QAM         | YES         |

---

## 6. Artifact Package Manifest

- `r0_calibration_merged.csv`: Consolidated 28-point pooled Monte Carlo calibration table.
- `r0_ground_truth.csv`: Complete ground-truth labels and candidate mode confidence statistics.
- `r0_cart_predictions.csv`: Decision Tree predictions across all 28 operating points.
- `r0_cart_depth_sweep.csv`: Depth sweep metrics ($d \in [1, 5]$).
- `r0_policy_evaluation.csv`: Detailed point-by-point comparison of all 4 policies.
- `r0_freeze_manifest.json`: Machine-readable provenance and cryptographic audit manifest.
- `r0_final_report.md`: This comprehensive publication-grade report.
- `r0_cart_model.joblib`: Serialized scikit-learn Decision Tree model object.
