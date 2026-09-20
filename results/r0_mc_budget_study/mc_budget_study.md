# PHY-ML R0 Monte Carlo Budget Study: Light (10k) vs. Deep (70k)

**Topic:** Empirical Statistical Resolution, Uncertainty Reduction, and Policy Invariance under Fixed-Budget Monte Carlo Calibration  
**Layer:** R0 Post-Processing Downstream Verification & Budget Tradeoff Analysis  
**Target Platform:** NVIDIA GeForce RTX 3060 (12 GB VRAM) | PyTorch 2.9+ / CUDA FP64  
**Decision Policy:** Canonical Conservative 95% Confidence Interval Upper Bound ($\text{BER} + 1.96 \cdot \text{SE} \le 0.0100$)  
**Date:** 2026-09-20  

---

## 1. Executive Summary: Core Research Questions

### Q1: How much uncertainty is reduced from 10k to 70k?
- **Theoretical Expectation:** By the Central Limit Theorem ($SE = \sigma / \sqrt{N_{blocks}}$), scaling the Monte Carlo budget from $10,000$ blocks/seed to $70,000$ blocks/seed ($7\times$ compute) yields a theoretical standard error scaling factor of $\frac{1}{\sqrt{7}} \approx 0.3780$, representing an exact **62.20% reduction** in standard error and 95% confidence interval width.
- **Empirical Measurement:** Across all 100 evaluated operating points, the measured average SE reduction is **62.05%** (median: 62.20%, min: 53.49%, max: 68.55%).
- **Practical Significance:** The 95% CI width near the critical $BER_{target} = 0.0100$ boundary drops from $\approx \pm 0.00030$ (Light) to $\approx \pm 0.00011$ (Deep), providing the statistical precision needed to resolve ambiguous operating points.

### Q2: Which operating points change overlap status?
- **Light Overlapping Points:** `2` operating points have 95% confidence intervals overlapping $BER_{target} = 0.0100$.
- **Deep Overlapping Points:** `0` operating points have 95% confidence intervals overlapping $BER_{target} = 0.0100$.
- **Net Status Changes:** `2` operating point(s) experienced overlap status transitions:
  - **22.5 dB | 16QAM**: Light = `OVERLAPPING` (BER=1.0417e-02, CI=[9.8735e-03, 1.0961e-02]) $\to$ Deep = `RESOLVED` (BER=1.0634e-02, CI=[1.0423e-02, 1.0844e-02])
  - **23.0 dB | 16QAM**: Light = `OVERLAPPING` (BER=9.9229e-03, CI=[9.3793e-03, 1.0467e-02]) $\to$ Deep = `RESOLVED` (BER=9.5557e-03, CI=[9.3546e-03, 9.7568e-03])

### Q3: Does any BestMode label change?
- **Label Flips:** Across the 25 evaluated SNR points, **1 label flip(s)** were observed (**96.0% label agreement**).
  - **SNR 23.0 dB:** Light BestMode = **QPSK** (2 bpcu) $\to$ Deep BestMode = **16QAM** (4 bpcu). Reason: *Max rate mode satisfying upper 95% CI (BER + 1.96*SE) <= 0.0100*.

### Q4: Do switching thresholds move?
| Transition | From | To | Light 1D LUT (dB) | Deep 1D LUT (dB) | LUT Shift (dB) | Light CART (dB) | Deep CART (dB) | CART Shift (dB) | Note |
|:----------:|:----:|:--:|:-----------------:|:----------------:|:--------------:|:---------------:|:--------------:|:---------------:|:----:|
| BPSK->QPSK | BPSK | QPSK | 16.75 | 16.75 | +0.00 | 16.75 | 16.75 | +0.00 | Boundary exactly invariant across budgets. |
| QPSK->16QAM | QPSK | 16QAM | 23.25 | 22.75 | -0.50 | 23.25 | 22.75 | -0.50 | Boundary shifted -0.50 dB under Deep reference. |
| 16QAM->64QAM | 16QAM | 64QAM | 28.25 | 28.25 | +0.00 | 28.25 | 28.25 | +0.00 | Boundary exactly invariant across budgets. |

### Q5: Does the learned Decision Tree policy change?
- **Model Selection Methodology:** Depth was swept independently over $d \in [1, 5]$ for Light and Deep; the smallest depth achieving 100% fidelity to each profile's ground-truth labels was selected (model selection, NOT ablation).
- **Light-Trained CART Tree:** Selected Depth = `3`, Actual Depth = `3`, Fidelity = `100.0%` (1.0000), Thresholds = `[16.75, 23.25, 28.25]`, Leaf Nodes = `4`
- **Deep-Trained CART Tree:** Selected Depth = `3`, Actual Depth = `3`, Fidelity = `100.0%` (1.0000), Thresholds = `[16.75, 22.75, 28.25]`, Leaf Nodes = `4`
- **Cross-Policy Agreement:** Predictions between Light-trained and Deep-trained decision trees agree on **96.0%** of the evaluation grid points.

### Q6: What compute cost is paid?
- **Simulated Channel Blocks:** Light = `500,000` blocks (`20,000` pooled/SNR) vs. Deep = `3,500,000` blocks (`140,000` pooled/SNR).
- **Compute Multiplier:** Exactly **7.00x** independent channel draws.
- **Measured Pure Monte Carlo Time:** Light = `238.95s` vs. Deep = `1681.14s`.
- **Measured GPU Throughput:** Light = `2,092.5 blk/s` vs. Deep = `2,081.9 blk/s`.

---

## 2. Scientific Methodological Stance

1. **Light Profile (10,000 blocks/seed = 20,000 pooled blocks/point):**
   - Serves as a high-speed, resource-limited empirical measurement.
   - Sufficient to identify all modulation operating regimes and detect broad switching boundaries with high fidelity.
   - **Validity:** Light is **not scientifically invalid**; its estimates are strictly unbiased point estimates with slightly wider confidence intervals ($SE \approx \sqrt{7} \times SE_{deep}$).

2. **Deep Profile (70,000 blocks/seed = 140,000 pooled blocks/point):**
   - Serves as the authoritative high-budget reference dataset.
   - Tighter standard errors eliminate boundary ambiguity at sensitive transition points (such as 23.0 dB where upper CI approaches 0.0100).
   - Recommended for final frozen ground-truth publication and downstream hardware lookup tables.

---

## 3. Detailed Operating Point Comparisons

| SNR (dB) | Mode | Light BER (SE) | Deep BER (SE) | Abs Diff | SE Reduction | Light 95% CI | Deep 95% CI | Overlap Changed? | CI Eligible (Light $\to$ Deep) |
|:--------:|:----:|:--------------:|:-------------:|:--------:|:------------:|:------------:|:-----------:|:----------------:|:------------------------------:|
|      0.0 | BPSK   | 1.452e-01 (8.22e-04) | 1.463e-01 (3.11e-04) | 1.08e-03 |   62.1% | [1.44e-01, 1.47e-01] | [1.46e-01, 1.47e-01] | No               | NO                             |
|      0.0 | QPSK   | 2.100e-01 (8.03e-04) | 2.112e-01 (3.04e-04) | 1.22e-03 |   62.2% | [2.08e-01, 2.12e-01] | [2.11e-01, 2.12e-01] | No               | NO                             |
|      0.0 | 16QAM  | 3.148e-01 (6.23e-04) | 3.157e-01 (2.35e-04) | 9.27e-04 |   62.2% | [3.14e-01, 3.16e-01] | [3.15e-01, 3.16e-01] | No               | NO                             |
|      0.0 | 64QAM  | 3.734e-01 (4.43e-04) | 3.740e-01 (1.67e-04) | 5.82e-04 |   62.2% | [3.73e-01, 3.74e-01] | [3.74e-01, 3.74e-01] | No               | NO                             |
|      2.0 | BPSK   | 1.095e-01 (7.85e-04) | 1.085e-01 (2.96e-04) | 1.03e-03 |   62.3% | [1.08e-01, 1.11e-01] | [1.08e-01, 1.09e-01] | No               | NO                             |
|      2.0 | QPSK   | 1.690e-01 (8.25e-04) | 1.677e-01 (3.12e-04) | 1.32e-03 |   62.2% | [1.67e-01, 1.71e-01] | [1.67e-01, 1.68e-01] | No               | NO                             |
|      2.0 | 16QAM  | 2.789e-01 (6.90e-04) | 2.777e-01 (2.61e-04) | 1.18e-03 |   62.1% | [2.78e-01, 2.80e-01] | [2.77e-01, 2.78e-01] | No               | NO                             |
|      2.0 | 64QAM  | 3.464e-01 (5.12e-04) | 3.455e-01 (1.94e-04) | 8.70e-04 |   62.0% | [3.45e-01, 3.47e-01] | [3.45e-01, 3.46e-01] | No               | NO                             |
|      4.0 | BPSK   | 7.622e-02 (7.10e-04) | 7.717e-02 (2.71e-04) | 9.50e-04 |   61.9% | [7.48e-02, 7.76e-02] | [7.66e-02, 7.77e-02] | No               | NO                             |
|      4.0 | QPSK   | 1.260e-01 (8.05e-04) | 1.269e-01 (3.06e-04) | 8.87e-04 |   61.9% | [1.24e-01, 1.28e-01] | [1.26e-01, 1.28e-01] | No               | NO                             |
|      4.0 | 16QAM  | 2.366e-01 (7.40e-04) | 2.371e-01 (2.82e-04) | 4.98e-04 |   61.9% | [2.35e-01, 2.38e-01] | [2.37e-01, 2.38e-01] | No               | NO                             |
|      4.0 | 64QAM  | 3.127e-01 (5.82e-04) | 3.129e-01 (2.21e-04) | 2.76e-04 |   61.9% | [3.12e-01, 3.14e-01] | [3.13e-01, 3.13e-01] | No               | NO                             |
|      6.0 | BPSK   | 5.488e-02 (6.44e-04) | 5.341e-02 (2.37e-04) | 1.46e-03 |   63.1% | [5.36e-02, 5.61e-02] | [5.29e-02, 5.39e-02] | No               | NO                             |
|      6.0 | QPSK   | 9.390e-02 (7.70e-04) | 9.256e-02 (2.86e-04) | 1.33e-03 |   62.9% | [9.24e-02, 9.54e-02] | [9.20e-02, 9.31e-02] | No               | NO                             |
|      6.0 | 16QAM  | 1.972e-01 (7.84e-04) | 1.965e-01 (2.93e-04) | 7.51e-04 |   62.7% | [1.96e-01, 1.99e-01] | [1.96e-01, 1.97e-01] | No               | NO                             |
|      6.0 | 64QAM  | 2.785e-01 (6.53e-04) | 2.780e-01 (2.44e-04) | 5.13e-04 |   62.6% | [2.77e-01, 2.80e-01] | [2.78e-01, 2.78e-01] | No               | NO                             |
|      8.0 | BPSK   | 3.528e-02 (5.30e-04) | 3.553e-02 (2.01e-04) | 2.43e-04 |   62.1% | [3.42e-02, 3.63e-02] | [3.51e-02, 3.59e-02] | No               | NO                             |
|      8.0 | QPSK   | 6.420e-02 (6.69e-04) | 6.442e-02 (2.54e-04) | 2.19e-04 |   62.1% | [6.29e-02, 6.55e-02] | [6.39e-02, 6.49e-02] | No               | NO                             |
|      8.0 | 16QAM  | 1.561e-01 (7.72e-04) | 1.564e-01 (2.92e-04) | 2.86e-04 |   62.1% | [1.55e-01, 1.58e-01] | [1.56e-01, 1.57e-01] | No               | NO                             |
|      8.0 | 64QAM  | 2.407e-01 (6.84e-04) | 2.409e-01 (2.59e-04) | 2.34e-04 |   62.2% | [2.39e-01, 2.42e-01] | [2.40e-01, 2.41e-01] | No               | NO                             |
|     10.0 | BPSK   | 2.385e-02 (4.48e-04) | 2.344e-02 (1.67e-04) | 4.08e-04 |   62.8% | [2.30e-02, 2.47e-02] | [2.31e-02, 2.38e-02] | No               | NO                             |
|     10.0 | QPSK   | 4.435e-02 (5.86e-04) | 4.386e-02 (2.19e-04) | 4.89e-04 |   62.6% | [4.32e-02, 4.55e-02] | [4.34e-02, 4.43e-02] | No               | NO                             |
|     10.0 | 16QAM  | 1.203e-01 (7.51e-04) | 1.198e-01 (2.83e-04) | 4.91e-04 |   62.4% | [1.19e-01, 1.22e-01] | [1.19e-01, 1.20e-01] | No               | NO                             |
|     10.0 | 64QAM  | 2.040e-01 (7.12e-04) | 2.036e-01 (2.68e-04) | 4.32e-04 |   62.4% | [2.03e-01, 2.05e-01] | [2.03e-01, 2.04e-01] | No               | NO                             |
|     12.0 | BPSK   | 1.590e-02 (3.78e-04) | 1.519e-02 (1.36e-04) | 7.03e-04 |   64.0% | [1.52e-02, 1.66e-02] | [1.49e-02, 1.55e-02] | No               | NO                             |
|     12.0 | QPSK   | 2.968e-02 (5.01e-04) | 2.913e-02 (1.84e-04) | 5.56e-04 |   63.4% | [2.87e-02, 3.07e-02] | [2.88e-02, 2.95e-02] | No               | NO                             |
|     12.0 | 16QAM  | 8.825e-02 (7.02e-04) | 8.811e-02 (2.62e-04) | 1.44e-04 |   62.6% | [8.69e-02, 8.96e-02] | [8.76e-02, 8.86e-02] | No               | NO                             |
|     12.0 | 64QAM  | 1.670e-01 (7.17e-04) | 1.670e-01 (2.69e-04) | 3.45e-05 |   62.4% | [1.66e-01, 1.68e-01] | [1.66e-01, 1.68e-01] | No               | NO                             |
|     14.0 | BPSK   | 9.445e-03 (2.80e-04) | 9.496e-03 (1.09e-04) | 5.12e-05 |   61.3% | [8.90e-03, 9.99e-03] | [9.28e-03, 9.71e-03] | No               | YES                            |
|     14.0 | QPSK   | 1.859e-02 (3.93e-04) | 1.859e-02 (1.49e-04) | 1.73e-06 |   62.0% | [1.78e-02, 1.94e-02] | [1.83e-02, 1.89e-02] | No               | NO                             |
|     14.0 | 16QAM  | 6.141e-02 (6.13e-04) | 6.183e-02 (2.32e-04) | 4.23e-04 |   62.2% | [6.02e-02, 6.26e-02] | [6.14e-02, 6.23e-02] | No               | NO                             |
|     14.0 | 64QAM  | 1.313e-01 (6.91e-04) | 1.320e-01 (2.61e-04) | 6.72e-04 |   62.2% | [1.30e-01, 1.33e-01] | [1.31e-01, 1.33e-01] | No               | NO                             |
|     16.0 | BPSK   | 6.196e-03 (2.33e-04) | 6.195e-03 (8.81e-05) | 7.11e-07 |   62.2% | [5.74e-03, 6.65e-03] | [6.02e-03, 6.37e-03] | No               | YES                            |
|     16.0 | QPSK   | 1.224e-02 (3.25e-04) | 1.218e-02 (1.23e-04) | 6.77e-05 |   62.2% | [1.16e-02, 1.29e-02] | [1.19e-02, 1.24e-02] | No               | NO                             |
|     16.0 | 16QAM  | 4.253e-02 (5.36e-04) | 4.234e-02 (2.02e-04) | 1.95e-04 |   62.3% | [4.15e-02, 4.36e-02] | [4.19e-02, 4.27e-02] | No               | NO                             |
|     16.0 | 64QAM  | 1.006e-01 (6.59e-04) | 1.004e-01 (2.48e-04) | 1.90e-04 |   62.3% | [9.93e-02, 1.02e-01] | [9.99e-02, 1.01e-01] | No               | NO                             |
|     16.5 | BPSK   | 5.735e-03 (2.28e-04) | 5.450e-03 (8.35e-05) | 2.85e-04 |   63.4% | [5.29e-03, 6.18e-03] | [5.29e-03, 5.61e-03] | No               | YES                            |
|     16.5 | QPSK   | 1.103e-02 (3.15e-04) | 1.077e-02 (1.16e-04) | 2.58e-04 |   63.3% | [1.04e-02, 1.17e-02] | [1.05e-02, 1.10e-02] | No               | NO                             |
|     16.5 | 16QAM  | 3.796e-02 (5.15e-04) | 3.826e-02 (1.93e-04) | 2.99e-04 |   62.5% | [3.70e-02, 3.90e-02] | [3.79e-02, 3.86e-02] | No               | NO                             |
|     16.5 | 64QAM  | 9.287e-02 (6.43e-04) | 9.323e-02 (2.43e-04) | 3.58e-04 |   62.2% | [9.16e-02, 9.41e-02] | [9.27e-02, 9.37e-02] | No               | NO                             |
|     17.0 | BPSK   | 4.558e-03 (1.95e-04) | 4.842e-03 (7.79e-05) | 2.84e-04 |   60.0% | [4.18e-03, 4.94e-03] | [4.69e-03, 4.99e-03] | No               | YES                            |
|     17.0 | QPSK   | 9.358e-03 (2.78e-04) | 9.664e-03 (1.09e-04) | 3.07e-04 |   60.8% | [8.81e-03, 9.90e-03] | [9.45e-03, 9.88e-03] | No               | YES                            |
|     17.0 | 16QAM  | 3.462e-02 (4.83e-04) | 3.487e-02 (1.85e-04) | 2.56e-04 |   61.7% | [3.37e-02, 3.56e-02] | [3.45e-02, 3.52e-02] | No               | NO                             |
|     17.0 | 64QAM  | 8.665e-02 (6.26e-04) | 8.691e-02 (2.38e-04) | 2.55e-04 |   62.0% | [8.54e-02, 8.79e-02] | [8.64e-02, 8.74e-02] | No               | NO                             |
|     17.5 | BPSK   | 4.543e-03 (2.02e-04) | 4.387e-03 (7.51e-05) | 1.56e-04 |   62.8% | [4.15e-03, 4.94e-03] | [4.24e-03, 4.53e-03] | No               | YES                            |
|     17.5 | QPSK   | 8.984e-03 (2.81e-04) | 8.683e-03 (1.04e-04) | 3.02e-04 |   62.8% | [8.43e-03, 9.54e-03] | [8.48e-03, 8.89e-03] | No               | YES                            |
|     17.5 | 16QAM  | 3.169e-02 (4.76e-04) | 3.130e-02 (1.78e-04) | 3.90e-04 |   62.7% | [3.08e-02, 3.26e-02] | [3.09e-02, 3.16e-02] | No               | NO                             |
|     17.5 | 64QAM  | 8.006e-02 (6.21e-04) | 7.981e-02 (2.33e-04) | 2.57e-04 |   62.5% | [7.88e-02, 8.13e-02] | [7.94e-02, 8.03e-02] | No               | NO                             |
|     18.0 | BPSK   | 4.221e-03 (1.98e-04) | 4.111e-03 (7.33e-05) | 1.10e-04 |   63.0% | [3.83e-03, 4.61e-03] | [3.97e-03, 4.25e-03] | No               | YES                            |
|     18.0 | QPSK   | 8.110e-03 (2.73e-04) | 7.982e-03 (1.01e-04) | 1.29e-04 |   62.8% | [7.58e-03, 8.64e-03] | [7.78e-03, 8.18e-03] | No               | YES                            |
|     18.0 | 16QAM  | 2.866e-02 (4.57e-04) | 2.847e-02 (1.72e-04) | 1.88e-04 |   62.5% | [2.78e-02, 2.96e-02] | [2.81e-02, 2.88e-02] | No               | NO                             |
|     18.0 | 64QAM  | 7.424e-02 (6.06e-04) | 7.403e-02 (2.28e-04) | 2.04e-04 |   62.4% | [7.30e-02, 7.54e-02] | [7.36e-02, 7.45e-02] | No               | NO                             |
|     20.0 | BPSK   | 2.327e-03 (1.47e-04) | 2.418e-03 (5.53e-05) | 9.05e-05 |   62.3% | [2.04e-03, 2.61e-03] | [2.31e-03, 2.53e-03] | No               | YES                            |
|     20.0 | QPSK   | 4.642e-03 (2.04e-04) | 4.813e-03 (7.81e-05) | 1.71e-04 |   61.7% | [4.24e-03, 5.04e-03] | [4.66e-03, 4.97e-03] | No               | YES                            |
|     20.0 | 16QAM  | 1.793e-02 (3.61e-04) | 1.819e-02 (1.39e-04) | 2.65e-04 |   61.6% | [1.72e-02, 1.86e-02] | [1.79e-02, 1.85e-02] | No               | NO                             |
|     20.0 | 64QAM  | 5.140e-02 (5.23e-04) | 5.168e-02 (1.99e-04) | 2.78e-04 |   61.9% | [5.04e-02, 5.24e-02] | [5.13e-02, 5.21e-02] | No               | NO                             |
|     22.0 | BPSK   | 1.661e-03 (1.23e-04) | 1.543e-03 (4.48e-05) | 1.17e-04 |   63.5% | [1.42e-03, 1.90e-03] | [1.46e-03, 1.63e-03] | No               | YES                            |
|     22.0 | QPSK   | 3.254e-03 (1.73e-04) | 3.079e-03 (6.28e-05) | 1.74e-04 |   63.6% | [2.92e-03, 3.59e-03] | [2.96e-03, 3.20e-03] | No               | YES                            |
|     22.0 | 16QAM  | 1.215e-02 (3.07e-04) | 1.188e-02 (1.13e-04) | 2.70e-04 |   63.1% | [1.16e-02, 1.28e-02] | [1.17e-02, 1.21e-02] | No               | NO                             |
|     22.0 | 64QAM  | 3.598e-02 (4.61e-04) | 3.564e-02 (1.72e-04) | 3.48e-04 |   62.6% | [3.51e-02, 3.69e-02] | [3.53e-02, 3.60e-02] | No               | NO                             |
|     22.5 | BPSK   | 1.248e-03 (1.05e-04) | 1.365e-03 (4.21e-05) | 1.18e-04 |   60.0% | [1.04e-03, 1.45e-03] | [1.28e-03, 1.45e-03] | No               | YES                            |
|     22.5 | QPSK   | 2.582e-03 (1.50e-04) | 2.729e-03 (5.92e-05) | 1.47e-04 |   60.6% | [2.29e-03, 2.88e-03] | [2.61e-03, 2.85e-03] | No               | YES                            |
|     22.5 | 16QAM  | 1.042e-02 (2.78e-04) | 1.063e-02 (1.07e-04) | 2.16e-04 |   61.3% | [9.87e-03, 1.10e-02] | [1.04e-02, 1.08e-02] | **YES**          | NO                             |
|     22.5 | 64QAM  | 3.209e-02 (4.32e-04) | 3.232e-02 (1.65e-04) | 2.37e-04 |   61.8% | [3.12e-02, 3.29e-02] | [3.20e-02, 3.26e-02] | No               | NO                             |
|     23.0 | BPSK   | 1.302e-03 (1.07e-04) | 1.251e-03 (4.06e-05) | 5.04e-05 |   61.9% | [1.09e-03, 1.51e-03] | [1.17e-03, 1.33e-03] | No               | YES                            |
|     23.0 | QPSK   | 2.619e-03 (1.52e-04) | 2.481e-03 (5.67e-05) | 1.38e-04 |   62.7% | [2.32e-03, 2.92e-03] | [2.37e-03, 2.59e-03] | No               | YES                            |
|     23.0 | 16QAM  | 9.923e-03 (2.77e-04) | 9.556e-03 (1.03e-04) | 3.67e-04 |   63.0% | [9.38e-03, 1.05e-02] | [9.35e-03, 9.76e-03] | **YES**          | NO -> YES                      |
|     23.0 | 64QAM  | 2.979e-02 (4.27e-04) | 2.918e-02 (1.58e-04) | 6.13e-04 |   62.9% | [2.90e-02, 3.06e-02] | [2.89e-02, 2.95e-02] | No               | NO                             |
|     23.5 | BPSK   | 1.265e-03 (1.07e-04) | 1.134e-03 (3.83e-05) | 1.31e-04 |   64.4% | [1.05e-03, 1.48e-03] | [1.06e-03, 1.21e-03] | No               | YES                            |
|     23.5 | QPSK   | 2.385e-03 (1.50e-04) | 2.245e-03 (5.41e-05) | 1.39e-04 |   64.0% | [2.09e-03, 2.68e-03] | [2.14e-03, 2.35e-03] | No               | YES                            |
|     23.5 | 16QAM  | 8.819e-03 (2.66e-04) | 8.572e-03 (9.77e-05) | 2.47e-04 |   63.2% | [8.30e-03, 9.34e-03] | [8.38e-03, 8.76e-03] | No               | YES                            |
|     23.5 | 64QAM  | 2.670e-02 (4.07e-04) | 2.637e-02 (1.52e-04) | 3.34e-04 |   62.7% | [2.59e-02, 2.75e-02] | [2.61e-02, 2.67e-02] | No               | NO                             |
|     24.0 | BPSK   | 9.617e-04 (9.13e-05) | 1.003e-03 (3.63e-05) | 4.10e-05 |   60.3% | [7.83e-04, 1.14e-03] | [9.32e-04, 1.07e-03] | No               | YES                            |
|     24.0 | QPSK   | 1.959e-03 (1.32e-04) | 1.986e-03 (5.08e-05) | 2.68e-05 |   61.4% | [1.70e-03, 2.22e-03] | [1.89e-03, 2.09e-03] | No               | YES                            |
|     24.0 | 16QAM  | 7.674e-03 (2.43e-04) | 7.642e-03 (9.23e-05) | 3.14e-05 |   62.0% | [7.20e-03, 8.15e-03] | [7.46e-03, 7.82e-03] | No               | YES                            |
|     24.0 | 64QAM  | 2.388e-02 (3.83e-04) | 2.372e-02 (1.45e-04) | 1.58e-04 |   62.2% | [2.31e-02, 2.46e-02] | [2.34e-02, 2.40e-02] | No               | NO                             |
|     26.0 | BPSK   | 6.895e-04 (8.12e-05) | 6.546e-04 (2.89e-05) | 3.49e-05 |   64.4% | [5.30e-04, 8.49e-04] | [5.98e-04, 7.11e-04] | No               | YES                            |
|     26.0 | QPSK   | 1.313e-03 (1.12e-04) | 1.303e-03 (4.11e-05) | 1.05e-05 |   63.4% | [1.09e-03, 1.53e-03] | [1.22e-03, 1.38e-03] | No               | YES                            |
|     26.0 | 16QAM  | 4.889e-03 (1.99e-04) | 4.968e-03 (7.53e-05) | 7.84e-05 |   62.1% | [4.50e-03, 5.28e-03] | [4.82e-03, 5.12e-03] | No               | YES                            |
|     26.0 | 64QAM  | 1.565e-02 (3.17e-04) | 1.568e-02 (1.21e-04) | 3.58e-05 |   62.0% | [1.50e-02, 1.63e-02] | [1.54e-02, 1.59e-02] | No               | NO                             |
|     28.0 | BPSK   | 4.864e-04 (7.45e-05) | 3.926e-04 (2.34e-05) | 9.38e-05 |   68.6% | [3.40e-04, 6.32e-04] | [3.47e-04, 4.38e-04] | No               | YES                            |
|     28.0 | QPSK   | 9.022e-04 (9.68e-05) | 7.746e-04 (3.22e-05) | 1.28e-04 |   66.7% | [7.12e-04, 1.09e-03] | [7.11e-04, 8.38e-04] | No               | YES                            |
|     28.0 | 16QAM  | 3.406e-03 (1.67e-04) | 3.108e-03 (5.88e-05) | 2.98e-04 |   64.7% | [3.08e-03, 3.73e-03] | [2.99e-03, 3.22e-03] | No               | YES                            |
|     28.0 | 64QAM  | 1.075e-02 (2.70e-04) | 1.020e-02 (9.75e-05) | 5.50e-04 |   63.9% | [1.02e-02, 1.13e-02] | [1.00e-02, 1.04e-02] | No               | NO                             |
|     28.5 | BPSK   | 2.965e-04 (5.23e-05) | 3.404e-04 (2.07e-05) | 4.39e-05 |   60.4% | [1.94e-04, 3.99e-04] | [3.00e-04, 3.81e-04] | No               | YES                            |
|     28.5 | QPSK   | 6.189e-04 (7.28e-05) | 6.921e-04 (2.97e-05) | 7.32e-05 |   59.3% | [4.76e-04, 7.62e-04] | [6.34e-04, 7.50e-04] | No               | YES                            |
|     28.5 | 16QAM  | 2.663e-03 (1.41e-04) | 2.768e-03 (5.56e-05) | 1.05e-04 |   60.5% | [2.39e-03, 2.94e-03] | [2.66e-03, 2.88e-03] | No               | YES                            |
|     28.5 | 64QAM  | 8.952e-03 (2.40e-04) | 9.074e-03 (9.24e-05) | 1.21e-04 |   61.5% | [8.48e-03, 9.42e-03] | [8.89e-03, 9.25e-03] | No               | YES                            |
|     29.0 | BPSK   | 2.123e-04 (4.18e-05) | 2.886e-04 (1.94e-05) | 7.63e-05 |   53.5% | [1.30e-04, 2.94e-04] | [2.51e-04, 3.27e-04] | No               | YES                            |
|     29.0 | QPSK   | 4.459e-04 (6.14e-05) | 5.792e-04 (2.72e-05) | 1.33e-04 |   55.6% | [3.26e-04, 5.66e-04] | [5.26e-04, 6.33e-04] | No               | YES                            |
|     29.0 | 16QAM  | 2.071e-03 (1.21e-04) | 2.400e-03 (5.13e-05) | 3.29e-04 |   57.5% | [1.83e-03, 2.31e-03] | [2.30e-03, 2.50e-03] | No               | YES                            |
|     29.0 | 64QAM  | 7.603e-03 (2.13e-04) | 8.021e-03 (8.63e-05) | 4.18e-04 |   59.6% | [7.18e-03, 8.02e-03] | [7.85e-03, 8.19e-03] | No               | YES                            |
|     29.5 | BPSK   | 2.867e-04 (5.21e-05) | 3.053e-04 (2.04e-05) | 1.87e-05 |   60.8% | [1.85e-04, 3.89e-04] | [2.65e-04, 3.45e-04] | No               | YES                            |
|     29.5 | QPSK   | 5.556e-04 (7.19e-05) | 5.938e-04 (2.83e-05) | 3.82e-05 |   60.7% | [4.15e-04, 6.97e-04] | [5.38e-04, 6.49e-04] | No               | YES                            |
|     29.5 | 16QAM  | 2.214e-03 (1.32e-04) | 2.276e-03 (5.12e-05) | 6.27e-05 |   61.3% | [1.95e-03, 2.47e-03] | [2.18e-03, 2.38e-03] | No               | YES                            |
|     29.5 | 64QAM  | 7.333e-03 (2.20e-04) | 7.408e-03 (8.44e-05) | 7.44e-05 |   61.6% | [6.90e-03, 7.76e-03] | [7.24e-03, 7.57e-03] | No               | YES                            |
|     30.0 | BPSK   | 2.521e-04 (4.85e-05) | 2.779e-04 (2.02e-05) | 2.58e-05 |   58.3% | [1.57e-04, 3.47e-04] | [2.38e-04, 3.18e-04] | No               | YES                            |
|     30.0 | QPSK   | 5.049e-04 (6.69e-05) | 5.266e-04 (2.73e-05) | 2.17e-05 |   59.2% | [3.74e-04, 6.36e-04] | [4.73e-04, 5.80e-04] | No               | YES                            |
|     30.0 | 16QAM  | 1.988e-03 (1.26e-04) | 1.991e-03 (4.82e-05) | 3.40e-06 |   61.6% | [1.74e-03, 2.23e-03] | [1.90e-03, 2.09e-03] | No               | YES                            |
|     30.0 | 64QAM  | 6.587e-03 (2.09e-04) | 6.529e-03 (7.93e-05) | 5.80e-05 |   62.1% | [6.18e-03, 7.00e-03] | [6.37e-03, 6.68e-03] | No               | YES                            |

---

## 4. Artifact Manifest

- `mc_budget_comparison.csv`: Point-by-point comparison table (100 rows across SNR x mode).
- `label_comparison.csv`: Ground truth label and eligibility comparison (25 rows across grid).
- `transition_comparison.csv`: Derived LUT and CART switching boundary shift analysis (3 transitions).
- `mc_budget_study.md`: This comprehensive comparative study.
