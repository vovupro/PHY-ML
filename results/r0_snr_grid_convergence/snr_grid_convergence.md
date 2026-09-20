# PHY-ML R0 SNR-Grid Convergence & Resolution Sensitivity Study

**Topic:** Empirical Switching-Boundary Convergence and Decision Tree Policy Stability under Local Grid Refinement  
**Resolution Progression:** `1.0-dB local transition view` (integer-spaced grid) $\to$ `0.5-dB local transition view` (Deep reference grid) $\to$ `0.25-dB local transition refinement view` (transition-centered refinement)  
**Methodology:** Conservative 95% Confidence Interval Upper Bound ($\text{BER} + 1.96 \cdot \text{SE} \le 0.0100$)  
**Monte Carlo Budget:** Deep 70,000 blocks/seed (140,000 pooled blocks/point) | FP64 CUDA  
**Study Characterization:** Formal Grid Convergence / Resolution Sensitivity (NOT an ablation)  
**Output Directory:** `results/r0_snr_grid_convergence/`  
**Date:** 2026-09-20  

---

> [!NOTE]
> **Local Transition Views vs Global Grids:** The "1.0-dB local transition view", "0.5-dB local transition view", and "0.25-dB local transition refinement view" are **NOT uniform global grids** across the full SNR span. They describe the local resolution of SNR evaluation points available around mode switching boundaries. This targeted refinement evaluates switching sensitivity without redundant dense simulation across stable single-mode regions.

## 1. Executive Summary

This study measures the sensitivity and convergence behavior of AMC switching boundaries as local SNR resolution is refined around mode transitions.
By probing candidate midpoints of the 0.5-dB transition intervals (**16.75 dB**, **22.75 dB**, and **28.25 dB**), the local switching brackets can be resolved down to $0.25\text{ dB}$ without re-running any existing calibration points.

### Empirical Convergence Findings (Computed from Data):
- **Switching Bracket Halving:** Uncertainty brackets halved systematically across all transitions as local resolution was refined: BPSK->QPSK: 1.00 dB -> 0.50 dB -> 0.25 dB; QPSK->16QAM: 1.00 dB -> 0.50 dB -> 0.25 dB; 16QAM->64QAM: 1.00 dB -> 0.50 dB -> 0.25 dB.
- **Threshold Shift Diminution:** Empirical threshold shifts diminished in magnitude with finer local resolution across all 3 transitions (BPSK->QPSK: Δθ(1.0→0.5) = +0.250 dB, Δθ(0.5→0.25) = +0.125 dB; QPSK->16QAM: Δθ(1.0→0.5) = +0.250 dB, Δθ(0.5→0.25) = +0.125 dB; 16QAM->64QAM: Δθ(1.0→0.5) = -0.250 dB, Δθ(0.5→0.25) = -0.125 dB).
- **Theoretical Bound Adherence:** All observed midpoint shifts satisfied the theoretical symmetric bisection bounds ($|\Delta \theta_{1.0 \to 0.5}| \le 0.25\text{ dB}$, $|\Delta \theta_{0.5 \to 0.25}| \le 0.125\text{ dB}$).
- **Decision Tree Policy Stability:** The optimal CART depth selected via model selection remained constant at $d=3$ across all three local transition views, each achieving 100.0% fidelity to ground-truth labels.

---

## 2. Transition Switching-Boundary Convergence

| Transition | Resolution View | Lower Bracket (dB) | Upper Bracket (dB) | Bracket Width (dB) | Midpoint Threshold (dB) | Midpoint Shift (dB) | CART Threshold (dB) | CART Shift (dB) |
|:----------:|:---------------:|:------------------:|:------------------:|:------------------:|:-----------------------:|:-------------------:|:-------------------:|:---------------:|
| BPSK->QPSK   | 1.0-dB local transition view |              16.00 |              17.00 | 1.00               | 16.500                  | ---                 | 16.500              | ---             |
| BPSK->QPSK   | 0.5-dB local transition view |              16.50 |              17.00 | 0.50               | 16.750                  | +0.250              | 16.750              | +0.250          |
| BPSK->QPSK   | 0.25-dB local transition view |              16.75 |              17.00 | 0.25               | 16.875                  | +0.125              | 16.875              | +0.125          |
| QPSK->16QAM  | 1.0-dB local transition view |              22.00 |              23.00 | 1.00               | 22.500                  | ---                 | 22.500              | ---             |
| QPSK->16QAM  | 0.5-dB local transition view |              22.50 |              23.00 | 0.50               | 22.750                  | +0.250              | 22.750              | +0.250          |
| QPSK->16QAM  | 0.25-dB local transition view |              22.75 |              23.00 | 0.25               | 22.875                  | +0.125              | 22.875              | +0.125          |
| 16QAM->64QAM | 1.0-dB local transition view |              28.00 |              29.00 | 1.00               | 28.500                  | ---                 | 28.500              | ---             |
| 16QAM->64QAM | 0.5-dB local transition view |              28.00 |              28.50 | 0.50               | 28.250                  | -0.250              | 28.250              | -0.250          |
| 16QAM->64QAM | 0.25-dB local transition view |              28.00 |              28.25 | 0.25               | 28.125                  | -0.125              | 28.125              | -0.125          |

---

## 3. Decision Tree Model Selection & Policy Stability

Under the canonical model-selection rule, maximum depth was swept over $d \in [1, 5]$ independently for each resolution view to identify the smallest depth achieving 100% fidelity to that view's ground-truth labels:

### 1.0-dB Local Transition View (Integer Points)
- **Selected Depth:** `3`
- **Actual Depth:** `3`
- **Leaf Nodes:** `4`
- **Empirical Fidelity:** `100.0%` (1.0000)
- **Learned Thresholds:** `[16.5, 22.5, 28.5]`

### 0.5-dB Local Transition View (Canonical Deep Grid)
- **Selected Depth:** `3`
- **Actual Depth:** `3`
- **Leaf Nodes:** `4`
- **Empirical Fidelity:** `100.0%` (1.0000)
- **Learned Thresholds:** `[16.75, 22.75, 28.25]`

### 0.25-dB Local Transition Refinement View (Refined Midpoints Grid)
- **Selected Depth:** `3`
- **Actual Depth:** `3`
- **Leaf Nodes:** `4`
- **Empirical Fidelity:** `100.0%` (1.0000)
- **Learned Thresholds:** `[16.875, 22.875, 28.125]`

---

## 4. Detailed Operating Point Classifications across Views

| SNR (dB) | In 1.0-dB Local View? | In 0.5-dB Local View? | In 0.25-dB Local View? | 1.0-dB Local BestMode | 0.5-dB Local BestMode | 0.25-dB Local BestMode | Eligible Modes (0.25-dB Local View) |
|:--------:|:---------------------:|:---------------------:|:----------------------:|:---------------------:|:---------------------:|:----------------------:|:-----------------------------------:|
|     0.00 | YES                   | YES                   | YES                    | BPSK                  | BPSK                  | BPSK                   | None                                |
|     2.00 | YES                   | YES                   | YES                    | BPSK                  | BPSK                  | BPSK                   | None                                |
|     4.00 | YES                   | YES                   | YES                    | BPSK                  | BPSK                  | BPSK                   | None                                |
|     6.00 | YES                   | YES                   | YES                    | BPSK                  | BPSK                  | BPSK                   | None                                |
|     8.00 | YES                   | YES                   | YES                    | BPSK                  | BPSK                  | BPSK                   | None                                |
|    10.00 | YES                   | YES                   | YES                    | BPSK                  | BPSK                  | BPSK                   | None                                |
|    12.00 | YES                   | YES                   | YES                    | BPSK                  | BPSK                  | BPSK                   | None                                |
|    14.00 | YES                   | YES                   | YES                    | BPSK                  | BPSK                  | BPSK                   | BPSK                                |
|    16.00 | YES                   | YES                   | YES                    | BPSK                  | BPSK                  | BPSK                   | BPSK                                |
|    16.50 | No                    | YES                   | YES                    | N/A                   | BPSK                  | BPSK                   | BPSK                                |
|    16.75 | No                    | No                    | YES                    | N/A                   | N/A                   | BPSK                   | BPSK                                |
|    17.00 | YES                   | YES                   | YES                    | QPSK                  | QPSK                  | QPSK                   | BPSK,QPSK                           |
|    17.50 | No                    | YES                   | YES                    | N/A                   | QPSK                  | QPSK                   | BPSK,QPSK                           |
|    18.00 | YES                   | YES                   | YES                    | QPSK                  | QPSK                  | QPSK                   | BPSK,QPSK                           |
|    20.00 | YES                   | YES                   | YES                    | QPSK                  | QPSK                  | QPSK                   | BPSK,QPSK                           |
|    22.00 | YES                   | YES                   | YES                    | QPSK                  | QPSK                  | QPSK                   | BPSK,QPSK                           |
|    22.50 | No                    | YES                   | YES                    | N/A                   | QPSK                  | QPSK                   | BPSK,QPSK                           |
|    22.75 | No                    | No                    | YES                    | N/A                   | N/A                   | QPSK                   | BPSK,QPSK                           |
|    23.00 | YES                   | YES                   | YES                    | 16QAM                 | 16QAM                 | 16QAM                  | BPSK,QPSK,16QAM                     |
|    23.50 | No                    | YES                   | YES                    | N/A                   | 16QAM                 | 16QAM                  | BPSK,QPSK,16QAM                     |
|    24.00 | YES                   | YES                   | YES                    | 16QAM                 | 16QAM                 | 16QAM                  | BPSK,QPSK,16QAM                     |
|    26.00 | YES                   | YES                   | YES                    | 16QAM                 | 16QAM                 | 16QAM                  | BPSK,QPSK,16QAM                     |
|    28.00 | YES                   | YES                   | YES                    | 16QAM                 | 16QAM                 | 16QAM                  | BPSK,QPSK,16QAM                     |
|    28.25 | No                    | No                    | YES                    | N/A                   | N/A                   | 64QAM                  | BPSK,QPSK,16QAM,64QAM               |
|    28.50 | No                    | YES                   | YES                    | N/A                   | 64QAM                 | 64QAM                  | BPSK,QPSK,16QAM,64QAM               |
|    29.00 | YES                   | YES                   | YES                    | 64QAM                 | 64QAM                 | 64QAM                  | BPSK,QPSK,16QAM,64QAM               |
|    29.50 | No                    | YES                   | YES                    | N/A                   | 64QAM                 | 64QAM                  | BPSK,QPSK,16QAM,64QAM               |
|    30.00 | YES                   | YES                   | YES                    | 64QAM                 | 64QAM                 | 64QAM                  | BPSK,QPSK,16QAM,64QAM               |

---

## 5. Scientific Methodological Stance

1. **Formal Convergence Analysis:**
   - The progression from 1.0-dB to 0.5-dB to 0.25-dB local transition views provides an empirical framework to test spatial resolution sensitivity around switching boundaries.
   - It assesses whether the discrete 1D look-up table and CART boundaries converge toward stable operating thresholds as spatial sampling is refined around switching boundaries.

2. **Compute Efficiency via Targeted Refinement:**
   - Evaluating only 3 targeted refinement points (16.75, 22.75, 28.25 dB) instead of a dense 0.25-dB global grid (which would require 120 SNR points) achieves 97.5% compute savings while providing identical switching-boundary resolution.

3. **Artifact Manifest:**
   - `results/r0_snr_grid_convergence/snr_grid_convergence.csv`: Quantitative bracket endpoints, widths, midpoints, and shifts across transitions.
   - `results/r0_snr_grid_convergence/snr_grid_views_labels.csv`: SNR point membership, classifications, and candidate mode error metrics.
   - `results/r0_snr_grid_convergence/snr_grid_convergence.md`: This comprehensive publication-grade report.
