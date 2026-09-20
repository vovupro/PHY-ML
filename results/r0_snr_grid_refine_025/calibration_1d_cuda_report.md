# PHY-ML L2 Fixed-Budget Monte Carlo Calibration Report (CUDA / RTX 3060)

**Target Platform:** Intel Core i5-14400F (16 threads) | NVIDIA GeForce RTX 3060 (12 GB VRAM)  
**Software Stack:** PyTorch `2.9.1+cu126` | CUDA Runtime `12.6` | Sionna `2.0.0`  
**Execution Configuration:** Backend: `cuda:0` | Precision: `double` (torch.float64) | Batch Size: `500`  
**Methodology:** **FIXED-BUDGET MONTE CARLO (NO ADAPTIVE STOPPING)**  
**Simulation Profile / Budget:** `deep` | Requested: `70,000 blocks/seed` | Actual: `70,000 blocks/seed`  
**Date:** 2026-09-20  

---

## 1. Execution & Timing Summary

- **Methodology:** `FIXED-BUDGET MONTE CARLO`
- **Stopping Rule:** `NO ADAPTIVE STOPPING (every SNR evaluated strictly to fixed budget)`
- **Requested Blocks per Seed:** `70,000 blocks/seed`
- **Actual Blocks per Seed:** `70,000 blocks/seed` (`140,000 pooled blocks per SNR point`)
- **Total Wall-Clock Time:** `202.10 seconds`
  - **Initialization & Warm-Up:** `0.466 seconds`
  - **Pure Monte Carlo Simulation:** `201.64 seconds`
- **Total Blocks Evaluated:** `420,000 independent blocks`
- **Total Independent Channel Realizations ($h$):** `420,000`
- **Overall Average Throughput:** `2,082.9 blocks/second`
- **Symbol Throughput:** `3,199,394 symbols/second`
- **Speedup vs. Reference CPU Implementation:** `N/A` (no persisted benchmark provided; hard-coded reference disallowed)

---

## 2. Analytical Sanity Validation (BPSK Theoretical Rayleigh Baseline)

Theoretical analytical benchmark: $P_b = \frac{1}{2}\left(1 - \sqrt{\frac{\bar{\gamma}}{1 + \bar{\gamma}}}\right)$ where $\bar{\gamma} = 10^{SNR/10}$.

| SNR (dB) | Empirical Pooled BER | Theoretical BER | Abs Error | Block SE | z-score | Status |
|:--------:|:--------------------:|:---------------:|:---------:|:--------:|:-------:|:------:|
|     16.8 |         5.233691e-03 |    5.201418e-03 | 3.227e-05 | 8.24e-05 |    0.39 | CONSISTENT |
|     22.8 |         1.367829e-03 |    1.321950e-03 | 4.588e-05 | 4.21e-05 |    1.09 | CONSISTENT |
|     28.2 |         4.034040e-04 |    3.736397e-04 | 2.976e-05 | 2.37e-05 |    1.25 | CONSISTENT |

---

## 3. Fixed-Budget Monte Carlo Calibration Table across All Modulations

Methodology: **FIXED-BUDGET MONTE CARLO (NO ADAPTIVE STOPPING)**  
Requested blocks/seed: `70,000` | Actual blocks/seed: `70,000` (`140,000` pooled blocks/point)  

| SNR (dB) | Modulation | Requested Blks/Seed | Actual Blks/Seed | Actual Pooled Blks | Pooled BER | Pooled SE | 95% Confidence Interval | Target Overlap (0.01)? | z_AB |
|:--------:|:----------:|:-------------------:|:----------------:|:------------------:|:----------:|:---------:|:-----------------------:|:----------------------:|:----:|
|     16.8 | BPSK       |              70,000 |           70,000 |            140,000 | 5.23369e-03 |  8.24e-05 | [5.07219e-03, 5.39520e-03] | NO                     | 0.89 |
|     16.8 | QPSK       |              70,000 |           70,000 |            140,000 | 1.02311e-02 |  1.14e-04 | [1.00084e-02, 1.04539e-02] | NO                     | 0.69 |
|     16.8 | 16QAM      |              70,000 |           70,000 |            140,000 | 3.62480e-02 |  1.89e-04 | [3.58769e-02, 3.66191e-02] | NO                     | 0.19 |
|     16.8 | 64QAM      |              70,000 |           70,000 |            140,000 | 8.95292e-02 |  2.40e-04 | [8.90580e-02, 9.00005e-02] | NO                     | 0.24 |
|     22.8 | BPSK       |              70,000 |           70,000 |            140,000 | 1.36783e-03 |  4.21e-05 | [1.28538e-03, 1.45028e-03] | NO                     | 0.35 |
|     22.8 | QPSK       |              70,000 |           70,000 |            140,000 | 2.69502e-03 |  5.93e-05 | [2.57885e-03, 2.81118e-03] | NO                     | 0.76 |
|     22.8 | 16QAM      |              70,000 |           70,000 |            140,000 | 1.01385e-02 |  1.06e-04 | [9.93002e-03, 1.03469e-02] | YES                    | 1.22 |
|     22.8 | 64QAM      |              70,000 |           70,000 |            140,000 | 3.05979e-02 |  1.62e-04 | [3.02794e-02, 3.09164e-02] | NO                     | 1.61 |
|     28.2 | BPSK       |              70,000 |           70,000 |            140,000 | 4.03404e-04 |  2.37e-05 | [3.56911e-04, 4.49897e-04] | NO                     | 0.56 |
|     28.2 | QPSK       |              70,000 |           70,000 |            140,000 | 7.77193e-04 |  3.26e-05 | [7.13274e-04, 8.41111e-04] | NO                     | 0.11 |
|     28.2 | 16QAM      |              70,000 |           70,000 |            140,000 | 2.96401e-03 |  5.86e-05 | [2.84922e-03, 3.07880e-03] | NO                     | 0.49 |
|     28.2 | 64QAM      |              70,000 |           70,000 |            140,000 | 9.57815e-03 |  9.55e-05 | [9.39099e-03, 9.76531e-03] | NO                     | 0.91 |

---

## 4. Uncertainty & Boundary Semantics

Evaluation against $BER_{target} = 0.0100$ using empirical 95% confidence intervals ($k = 1.96$):
- Methodology: `FIXED-BUDGET MONTE CARLO (NO ADAPTIVE STOPPING)`
- Requested blocks/seed: `70,000` | Actual blocks/seed: `70,000`

| SNR (dB) | Modulation | Pooled BER | Pooled SE | 95% Confidence Interval | Overlaps 0.01? | Label / Semantics |
|:--------:|:----------:|:----------:|:---------:|:-----------------------:|:--------------:|:-----------------:|
|     16.8 | BPSK       | 5.23369e-03 |  8.24e-05 | [5.07219e-03, 5.39520e-03] | NO             | `resolved` |
|     16.8 | QPSK       | 1.02311e-02 |  1.14e-04 | [1.00084e-02, 1.04539e-02] | NO             | `resolved` |
|     16.8 | 16QAM      | 3.62480e-02 |  1.89e-04 | [3.58769e-02, 3.66191e-02] | NO             | `resolved` |
|     16.8 | 64QAM      | 8.95292e-02 |  2.40e-04 | [8.90580e-02, 9.00005e-02] | NO             | `resolved` |
|     22.8 | BPSK       | 1.36783e-03 |  4.21e-05 | [1.28538e-03, 1.45028e-03] | NO             | `resolved` |
|     22.8 | QPSK       | 2.69502e-03 |  5.93e-05 | [2.57885e-03, 2.81118e-03] | NO             | `resolved` |
|     22.8 | 16QAM      | 1.01385e-02 |  1.06e-04 | [9.93002e-03, 1.03469e-02] | YES            | `reliability_uncertain` |
|     22.8 | 64QAM      | 3.05979e-02 |  1.62e-04 | [3.02794e-02, 3.09164e-02] | NO             | `resolved` |
|     28.2 | BPSK       | 4.03404e-04 |  2.37e-05 | [3.56911e-04, 4.49897e-04] | NO             | `resolved` |
|     28.2 | QPSK       | 7.77193e-04 |  3.26e-05 | [7.13274e-04, 8.41111e-04] | NO             | `resolved` |
|     28.2 | 16QAM      | 2.96401e-03 |  5.86e-05 | [2.84922e-03, 3.07880e-03] | NO             | `resolved` |
|     28.2 | 64QAM      | 9.57815e-03 |  9.55e-05 | [9.39099e-03, 9.76531e-03] | NO             | `resolved` |

---

## 5. Checkpoint Trajectory Convergence Summary (Human Inspection Only)

Checkpoints were recorded every `5,000` blocks/seed strictly for human observational inspection and convergence analysis. No algorithmic stopping rules were applied during execution.

---

## 6. Summary & Verification Status

- **Methodology:** FIXED-BUDGET MONTE CARLO (NO ADAPTIVE STOPPING).
- All 3 SNR grid points evaluated strictly to the requested budget of 70,000 blocks/seed (140,000 pooled blocks per point).
- Analytical Rayleigh sanity test: **PASS** across the grid.
- No smoothing or synthetic alterations applied to BER estimates.
- Output artifacts successfully written to `results/r0_snr_grid_refine_025/`.
