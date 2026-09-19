# PHY-AMC Ground Truth & Link Adaptation Baseline Report

**Topic:** 1D Ground Truth BestMode Synthesis and Baseline Adaptation under Slow Rayleigh Block Fading  
**Layer:** L3 (Link Adaptation Ground Truth & Baseline Policies)  
**Date:** 2026-09-19  
**Source Calibration Data:** `results/l2_cuda_rtx3060_final/calibration_1d_cuda_pooled.csv`  

---

## 1. Physical Model & Adaptation Problem Formulation

- **Channel & Noise Model:** Single-carrier transmission over slow Rayleigh flat block fading ($y = h \cdot x + n$). Channel scalar $h \sim \mathcal{CN}(0, 1)$ held constant over $N_s = 1,536$ symbols. Noise $n \sim \mathcal{CN}(0, N_0)$ with setup $SNR_{setup} = E_s / N_0$ ($E_s = 1.0$).
- **Fading Averaging:** Fading is averaged across independent Monte Carlo transmission blocks at each nominal SNR.
- **Simulation Depth:** Adaptive dual-seed Monte Carlo calibration ranging from 30,000 to 80,000 independent fading blocks per operating point (46,080,000 to 122,880,000 symbols/point), totaling 1,140,000 blocks (1,751,040,000 symbols) across all 25 SNR operating points.
- **Target Reliability Constraint:** $\text{BER} \le 0.0100$ (1.00%)
- **Fallback Rule:** `robustest_mode` (BPSK is selected if no modulation meets the target BER).
- **Operating Grid:** 25 points from 0.0 dB to 30.0 dB (locally refined to 0.5 dB resolution at 16-18 dB, 22-24 dB, and 28-30 dB).

---

## 2. Ground Truth BestMode Table

| SNR (dB) | BPSK BER (SE) | QPSK BER (SE) | 16-QAM BER (SE) | 64-QAM BER (SE) | BestMode | Rate (bpcu) | Fallback | Reliability Unc. | Label Unc. | Blocks |
|:--------:|:-------------:|:-------------:|:---------------:|:---------------:|:--------:|:-----------:|:--------:|:----------------:|:----------:|:------:|
|      0.0 | 1.46411e-01 (6.75e-04) | 2.11069e-01 (6.58e-04) | 3.15548e-01 (5.10e-04) | 3.73917e-01 (3.63e-04) | BPSK     |           1 | YES      | Clear            | Clear      | 30,000 |
|      2.0 | 1.09141e-01 (6.40e-04) | 1.68499e-01 (6.74e-04) | 2.78429e-01 (5.64e-04) | 3.46014e-01 (4.19e-04) | BPSK     |           1 | YES      | Clear            | Clear      | 30,000 |
|      4.0 | 7.67434e-02 (5.81e-04) | 1.26686e-01 (6.59e-04) | 2.37151e-01 (6.05e-04) | 3.13067e-01 (4.75e-04) | BPSK     |           1 | YES      | Clear            | Clear      | 30,000 |
|      6.0 | 5.40320e-02 (5.20e-04) | 9.30207e-02 (6.23e-04) | 1.96535e-01 (6.37e-04) | 2.77996e-01 (5.30e-04) | BPSK     |           1 | YES      | Clear            | Clear      | 30,000 |
|      8.0 | 3.54376e-02 (4.33e-04) | 6.44331e-02 (5.47e-04) | 1.56394e-01 (6.31e-04) | 2.40917e-01 (5.59e-04) | BPSK     |           1 | YES      | Clear            | Clear      | 30,000 |
|     10.0 | 2.35603e-02 (3.63e-04) | 4.38867e-02 (4.75e-04) | 1.19634e-01 (6.12e-04) | 2.03294e-01 (5.80e-04) | BPSK     |           1 | YES      | Clear            | Clear      | 30,000 |
|     12.0 | 1.57990e-02 (3.06e-04) | 2.96847e-02 (4.07e-04) | 8.85336e-02 (5.72e-04) | 1.67325e-01 (5.85e-04) | BPSK     |           1 | YES      | Clear            | Clear      | 30,000 |
|     14.0 | 9.57730e-03 (2.32e-04) | 1.87982e-02 (3.23e-04) | 6.18582e-02 (5.03e-04) | 1.31768e-01 (5.66e-04) | BPSK     |           1 | No       | FLAGGED          | Clear      | 30,000 |
|     16.0 | 6.39182e-03 (1.95e-04) | 1.24357e-02 (2.70e-04) | 4.26219e-02 (4.40e-04) | 1.00616e-01 (5.40e-04) | BPSK     |           1 | No       | Clear            | Clear      | 30,000 |
|     16.5 | 5.71895e-03 (1.87e-04) | 1.10119e-02 (2.58e-04) | 3.80200e-02 (4.20e-04) | 9.28435e-02 (5.25e-04) | BPSK     |           1 | No       | Clear            | Clear      | 30,000 |
|     17.0 | 4.71072e-03 (1.64e-04) | 9.54489e-03 (2.31e-04) | 3.48970e-02 (3.97e-04) | 8.70489e-02 (5.14e-04) | QPSK     |           2 | No       | Clear            | Clear      | 30,000 |
|     17.5 | 4.62307e-03 (1.67e-04) | 9.04715e-03 (2.32e-04) | 3.17368e-02 (3.90e-04) | 8.02772e-02 (5.07e-04) | QPSK     |           2 | No       | Clear            | Clear      | 30,000 |
|     18.0 | 4.18064e-03 (1.61e-04) | 8.04642e-03 (2.22e-04) | 2.85090e-02 (3.72e-04) | 7.40663e-02 (4.93e-04) | QPSK     |           2 | No       | Clear            | Clear      | 30,000 |
|     20.0 | 2.33965e-03 (1.03e-04) | 4.65968e-03 (1.44e-04) | 1.78500e-02 (2.56e-04) | 5.11242e-02 (3.70e-04) | QPSK     |           2 | No       | Clear            | Clear      | 40,000 |
|     22.0 | 1.47045e-03 (6.62e-05) | 2.98245e-03 (9.33e-05) | 1.18109e-02 (1.71e-04) | 3.56603e-02 (2.62e-04) | QPSK     |           2 | No       | Clear            | Clear      | 60,000 |
|     22.5 | 1.34167e-03 (5.84e-05) | 2.72012e-03 (8.26e-05) | 1.07113e-02 (1.52e-04) | 3.24622e-02 (2.34e-04) | QPSK     |           2 | No       | Clear            | Clear      | 70,000 |
|     23.0 | 1.27401e-03 (5.78e-05) | 2.53873e-03 (8.10e-05) | 9.71396e-03 (1.47e-04) | 2.93977e-02 (2.26e-04) | 16QAM    |           4 | No       | FLAGGED          | FLAGGED    | 70,000 |
|     23.5 | 1.17443e-03 (5.45e-05) | 2.29928e-03 (7.75e-05) | 8.66214e-03 (1.40e-04) | 2.64806e-02 (2.16e-04) | 16QAM    |           4 | No       | Clear            | Clear      | 70,000 |
|     24.0 | 9.81380e-04 (4.73e-05) | 1.95969e-03 (6.65e-05) | 7.63136e-03 (1.21e-04) | 2.37302e-02 (1.91e-04) | 16QAM    |           4 | No       | Clear            | Clear      | 80,000 |
|     26.0 | 7.20433e-04 (4.51e-05) | 1.36043e-03 (6.16e-05) | 4.96176e-03 (1.08e-04) | 1.56020e-02 (1.71e-04) | 16QAM    |           4 | No       | Clear            | Clear      | 70,000 |
|     28.0 | 3.45573e-04 (3.07e-05) | 7.17076e-04 (4.26e-05) | 3.05435e-03 (8.09e-05) | 1.00951e-02 (1.37e-04) | 16QAM    |           4 | No       | FLAGGED          | FLAGGED    | 70,000 |
|     28.5 | 3.21636e-04 (3.03e-05) | 6.74061e-04 (4.39e-05) | 2.77662e-03 (8.42e-05) | 9.11527e-03 (1.41e-04) | 64QAM    |           6 | No       | Clear            | Clear      | 60,000 |
|     29.0 | 2.43372e-04 (2.94e-05) | 5.00970e-04 (4.15e-05) | 2.25794e-03 (8.10e-05) | 7.84018e-03 (1.40e-04) | 64QAM    |           6 | No       | Clear            | Clear      | 50,000 |
|     29.5 | 2.91693e-04 (3.19e-05) | 5.90143e-04 (4.55e-05) | 2.30833e-03 (8.57e-05) | 7.49623e-03 (1.42e-04) | 64QAM    |           6 | No       | Clear            | Clear      | 50,000 |
|     30.0 | 3.06239e-04 (3.16e-05) | 5.79845e-04 (4.33e-05) | 2.09154e-03 (7.67e-05) | 6.67846e-03 (1.24e-04) | 64QAM    |           6 | No       | Clear            | Clear      | 60,000 |

---

## 3. Link Adaptation Switching Regions & LUT Thresholds

The 1D Look-Up Table (LUT) defines **sampled-grid-derived switching thresholds** at the midpoints between adjacent sampled SNRs where the selected modulation transitions.
These thresholds reflect the midpoints of the discrete empirical calibration grid and are not claimed to be exact continuous physical BER-crossing thresholds.
Calibration grid resolution is 0.5 dB in all active transition zones (16-18 dB, 22-24 dB, 28-30 dB).

### Derived Switching Thresholds:
- **16.75 dB**: Transition from **BPSK** (1 bpcu) $\to$ **QPSK** (2 bpcu)
- **22.75 dB**: Transition from **QPSK** (2 bpcu) $\to$ **16QAM** (4 bpcu)
- **28.25 dB**: Transition from **16QAM** (4 bpcu) $\to$ **64QAM** (6 bpcu)

### Operating Intervals:
- **[-inf, 16.75 dB)** $\to$ **BPSK** (1 bpcu)
- **[16.75 dB, 22.75 dB)** $\to$ **QPSK** (2 bpcu)
- **[22.75 dB, 28.25 dB)** $\to$ **16QAM** (4 bpcu)
- **[28.25 dB, +inf)** $\to$ **64QAM** (6 bpcu)

---

## 4. Boundary Uncertainty & Statistical Confidence Analysis

We distinguish two levels of uncertainty:
1. **Reliability Uncertainty (`reliability_uncertain`):** A candidate mode's empirical 95% confidence interval ($[BER - 1.96 \cdot SE, BER + 1.96 \cdot SE]$) overlaps $BER_{target} = 0.01$.
2. **Label Uncertainty (`label_uncertain`):** Confidence interval variance is sufficient to alter the selected $BestMode$ under optimistic vs. pessimistic bound evaluations (i.e. $BestMode_{optimistic} \ne BestMode_{pessimistic}$).

### Reliability Uncertainty (3 points):
- **SNR = 14.0 dB**: Empirical 95% CI of a candidate modulation overlaps $BER_{target} = 0.0100$. (Selected BestMode: **BPSK**)
- **SNR = 23.0 dB**: Empirical 95% CI of a candidate modulation overlaps $BER_{target} = 0.0100$. (Selected BestMode: **16QAM**)
- **SNR = 28.0 dB**: Empirical 95% CI of a candidate modulation overlaps $BER_{target} = 0.0100$. (Selected BestMode: **16QAM**)

### Label Uncertainty (2 points):
- **SNR = 23.0 dB**: BestMode selection is sensitive to statistical confidence intervals between candidate modes.
- **SNR = 28.0 dB**: BestMode selection is sensitive to statistical confidence intervals between candidate modes.

### Factual Empirical Observations at Boundary Points:
- **14.0 dB (`reliability_uncertain = True`, `label_uncertain = False`):** BPSK empirical BER is $9.577 \times 10^{-3}$ with 95% CI $[9.123 \times 10^{-3}, 1.003 \times 10^{-2}]$, overlapping $BER_{target} = 0.0100$. However, because BPSK is also the fallback mode if no mode qualifies, BestMode remains BPSK under both optimistic and pessimistic bounds, leaving the label robustly invariant.
- **23.0 dB (`reliability_uncertain = True`, `label_uncertain = True`):** 16-QAM empirical BER is $9.714 \times 10^{-3}$ with 95% CI $[9.427 \times 10^{-3}, 1.0001 \times 10^{-2}]$. At point estimate, 16-QAM meets the reliability constraint and is selected (4 bpcu). Under the pessimistic CI bound, 16-QAM BER marginally exceeds 0.0100, which would disqualify 16-QAM in favor of QPSK (2 bpcu).
- **28.0 dB (`reliability_uncertain = True`, `label_uncertain = True`):** 64-QAM empirical BER is $1.0095 \times 10^{-2}$ with 95% CI $[9.827 \times 10^{-3}, 1.036 \times 10^{-2}]$. At point estimate, 64-QAM slightly exceeds $BER_{target} = 0.0100$, so 16-QAM is selected as the highest eligible mode (4 bpcu). Under the optimistic CI bound, 64-QAM drops below 0.0100 and would be selected (6 bpcu).
- **Unambiguous Operating Points (22 points):** All other 22 SNR points exhibit non-overlapping confidence intervals relative to $BER_{target} = 0.0100$, yielding identical BestMode selections across nominal, optimistic, and pessimistic evaluations.

### Fallback Events (7 points):
- **SNR = 0.0 dB**: No candidate modulation satisfied $\text{BER} \le 0.0100$. Fallback policy `robustest_mode` selected **BPSK**.
- **SNR = 2.0 dB**: No candidate modulation satisfied $\text{BER} \le 0.0100$. Fallback policy `robustest_mode` selected **BPSK**.
- **SNR = 4.0 dB**: No candidate modulation satisfied $\text{BER} \le 0.0100$. Fallback policy `robustest_mode` selected **BPSK**.
- **SNR = 6.0 dB**: No candidate modulation satisfied $\text{BER} \le 0.0100$. Fallback policy `robustest_mode` selected **BPSK**.
- **SNR = 8.0 dB**: No candidate modulation satisfied $\text{BER} \le 0.0100$. Fallback policy `robustest_mode` selected **BPSK**.
- **SNR = 10.0 dB**: No candidate modulation satisfied $\text{BER} \le 0.0100$. Fallback policy `robustest_mode` selected **BPSK**.
- **SNR = 12.0 dB**: No candidate modulation satisfied $\text{BER} \le 0.0100$. Fallback policy `robustest_mode` selected **BPSK**.

### Quality Verification:
- Non-decreasing selected spectral efficiency with SNR is strictly verified across the entire 25-point grid (Monotonicity: PASS).
- LUT thresholds form well-defined, non-overlapping switching intervals derived from the sampled calibration grid.
- Both Fixed baselines (Fixed Robust: BPSK, Fixed High-Throughput: 64-QAM) are defined and ready for comparative benchmarking.
