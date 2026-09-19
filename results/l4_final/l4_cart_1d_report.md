# PHY-ML L4: CART 1D Machine Learning Baseline Report

**Topic:** Supervised Decision Tree Classification for 1D Uncoded Link Adaptation  
**Layer:** L4 (Machine Learning - CART 1D Baseline)  
**Date:** 2026-09-19  
**Source Dataset:** `results/l3_final/ground_truth_1d.csv`  
**Framework Architecture:** Feature-Agnostic Generic `CARTClassifier`  

---

## 1. Machine Learning Formulation & Setup

- **Supervised Learning Task:** Multiclass Classification ($X \to y$)
- **Input Feature Space:** $X = [SNR_{setup}]$ (1D continuous scalar, nominal $E_s/N_0$ in dB)
- **Target Space:** $y \in \{\text{BPSK}, \text{QPSK}, \text{16QAM}, \text{64QAM}\}$
- **Ground Truth Basis:** Empirically calibrated uncoded PHY under slow Rayleigh flat block fading (25 operating points from 0.0 to 30.0 dB, adaptive dual-seed depth totaling 1,140,000 blocks, $BER_{target} = 0.0100$).
- **Algorithm:** Classification and Regression Trees (CART) via generic `CARTClassifier` (`sklearn.tree.DecisionTreeClassifier`).
- **Split Criterion:** Gini Impurity ($I_G(p) = 1 - \sum_k p_k^2$).
- **Random State:** `20260918` (guarantees deterministic tree induction).
- **Baseline for Comparison:** 1D Look-Up Table (LUT) derived from the exact same sampled calibration grid.

---

## 2. Decision Tree Depth Sweep (max_depth = 1..5)

| max_depth | Actual Depth | Total Nodes | Leaf Nodes | Training Fidelity | Learned Split Thresholds (dB) | LUT Equivalence |
|:---------:|:------------:|:-----------:|:----------:|:-----------------:|:------------------------------:|:---------------:|
|         1 |            1 |           3 |          2 |              64.0% | [16.75] | Sub-optimal     |
|         2 |            2 |           5 |          3 |              84.0% | [16.75, 22.75] | Sub-optimal     |
|         3 |            3 |           7 |          4 |             100.0% | [16.75, 22.75, 28.25] | **EXACT MATCH** |
|         4 |            3 |           7 |          4 |             100.0% | [16.75, 22.75, 28.25] | **EXACT MATCH** |
|         5 |            3 |           7 |          4 |             100.0% | [16.75, 22.75, 28.25] | **EXACT MATCH** |

---

## 3. Comparative Threshold Analysis: CART vs. 1D LUT Baseline

The 1D Look-Up Table defines **sampled-grid-derived switching thresholds** at the midpoints between adjacent sampled SNRs where BestMode transitions:
- **16.75 dB**: Transition from **BPSK** (1 bpcu) $\to$ **QPSK** (2 bpcu)
- **22.75 dB**: Transition from **QPSK** (2 bpcu) $\to$ **16QAM** (4 bpcu)
- **28.25 dB**: Transition from **16QAM** (4 bpcu) $\to$ **64QAM** (6 bpcu)

### Selected Baseline Model: `max_depth = 3`
- **Learned Split Thresholds:** [16.75, 22.75, 28.25] dB
- **LUT Switching Thresholds:** [16.75, 22.75, 28.25] dB
- **Structural Complexity:** 7 total nodes (4 leaves), actual tree depth = 3
- **Threshold Parity:** The CART decision tree with `max_depth = 3` reproduces the exact 1D Look-Up Table sampled-grid-derived switching thresholds without manual heuristic intervention.
- **Physical Interpretation:** These switching thresholds reflect discrete midpoints of the empirical calibration grid (0.5 dB resolution in active zones); they are not claimed to be exact continuous physical BER-crossing thresholds.

---

## 4. Textual Decision Tree Export (max_depth = 3)

```text
|--- snr_db <= 16.75
|   |--- class: BPSK
|--- snr_db >  16.75
|   |--- snr_db <= 22.75
|   |   |--- class: QPSK
|   |--- snr_db >  22.75
|   |   |--- snr_db <= 28.25
|   |   |   |--- class: 16QAM
|   |   |--- snr_db >  28.25
|   |   |   |--- class: 64QAM
```

---

## 5. Point-by-Point Ground Truth vs. CART Classification (25 Operating Points)

| SNR (dB) | Ground Truth BestMode | CART Prediction | Match? | Reliability Unc. | Label Unc. | Blocks |
|:--------:|:---------------------:|:---------------:|:------:|:----------------:|:----------:|:------:|
|      0.0 | BPSK                  | BPSK            | MATCH  | Clear            | Clear      | 30,000 |
|      2.0 | BPSK                  | BPSK            | MATCH  | Clear            | Clear      | 30,000 |
|      4.0 | BPSK                  | BPSK            | MATCH  | Clear            | Clear      | 30,000 |
|      6.0 | BPSK                  | BPSK            | MATCH  | Clear            | Clear      | 30,000 |
|      8.0 | BPSK                  | BPSK            | MATCH  | Clear            | Clear      | 30,000 |
|     10.0 | BPSK                  | BPSK            | MATCH  | Clear            | Clear      | 30,000 |
|     12.0 | BPSK                  | BPSK            | MATCH  | Clear            | Clear      | 30,000 |
|     14.0 | BPSK                  | BPSK            | MATCH  | FLAGGED          | Clear      | 30,000 |
|     16.0 | BPSK                  | BPSK            | MATCH  | Clear            | Clear      | 30,000 |
|     16.5 | BPSK                  | BPSK            | MATCH  | Clear            | Clear      | 30,000 |
|     17.0 | QPSK                  | QPSK            | MATCH  | Clear            | Clear      | 30,000 |
|     17.5 | QPSK                  | QPSK            | MATCH  | Clear            | Clear      | 30,000 |
|     18.0 | QPSK                  | QPSK            | MATCH  | Clear            | Clear      | 30,000 |
|     20.0 | QPSK                  | QPSK            | MATCH  | Clear            | Clear      | 40,000 |
|     22.0 | QPSK                  | QPSK            | MATCH  | Clear            | Clear      | 60,000 |
|     22.5 | QPSK                  | QPSK            | MATCH  | Clear            | Clear      | 70,000 |
|     23.0 | 16QAM                 | 16QAM           | MATCH  | FLAGGED          | FLAGGED    | 70,000 |
|     23.5 | 16QAM                 | 16QAM           | MATCH  | Clear            | Clear      | 70,000 |
|     24.0 | 16QAM                 | 16QAM           | MATCH  | Clear            | Clear      | 80,000 |
|     26.0 | 16QAM                 | 16QAM           | MATCH  | Clear            | Clear      | 70,000 |
|     28.0 | 16QAM                 | 16QAM           | MATCH  | FLAGGED          | FLAGGED    | 70,000 |
|     28.5 | 64QAM                 | 64QAM           | MATCH  | Clear            | Clear      | 60,000 |
|     29.0 | 64QAM                 | 64QAM           | MATCH  | Clear            | Clear      | 50,000 |
|     29.5 | 64QAM                 | 64QAM           | MATCH  | Clear            | Clear      | 50,000 |
|     30.0 | 64QAM                 | 64QAM           | MATCH  | Clear            | Clear      | 60,000 |

---

## 6. Uncertainty & Sensitivity Analysis at Operating Boundaries

The model is trained strictly on nominal point-estimate ground-truth labels without discarding or heuristically modifying uncertain points. Statistical uncertainty is preserved as evaluation metadata:

- **SNR = 14.0 dB (`reliability_uncertain = True`, `label_uncertain = False`):** BPSK 95% confidence interval $[9.123 \times 10^{-3}, 1.003 \times 10^{-2}]$ overlaps $BER_{target} = 0.0100$. However, because BPSK is also the fallback policy if no mode qualifies, BestMode remains invariant to CI bounds. CART correctly predicts BPSK.
- **SNR = 23.0 dB (`reliability_uncertain = True`, `label_uncertain = True`):** 16-QAM point-estimate BER is $9.714 \times 10^{-3}$ (compliant), but its upper 95% CI bound reaches $1.0001 \times 10^{-2}$. Under pessimistic bounds, QPSK would be selected; under nominal point estimate, 16-QAM is selected. CART reproduces the nominal label (16QAM, 4 bpcu).
- **SNR = 28.0 dB (`reliability_uncertain = True`, `label_uncertain = True`):** 64-QAM point-estimate BER is $1.0095 \times 10^{-2}$ (non-compliant), but its lower 95% CI bound reaches $9.827 \times 10^{-3}$. Under optimistic bounds, 64-QAM would qualify; under nominal point estimate, 16-QAM is selected. CART reproduces the nominal label (16QAM, 4 bpcu).

---

## 7. Domain & Extrapolation Behavior

- **Empirically Calibrated Domain:** $0 \le SNR_{setup} \le 30\text{ dB}$. The model is grounded in Monte Carlo simulations strictly within this 25-point operating window.
- **Extrapolation Behavior:** For $SNR < 0\text{ dB}$, the tree evaluates $SNR \le 16.75\text{ dB}$ and predicts BPSK (1 bpcu). For $SNR > 30\text{ dB}$, the tree evaluates $SNR > 28.25\text{ dB}$ and predicts 64-QAM (6 bpcu). This constant leaf extension is an inherent property of axis-aligned decision trees (model extrapolation / endpoint behavior), not validated physical ground truth.

---

## 8. Generalization Scope & Thesis Baseline Synthesis

- **Representation Completeness:** With 4 distinct modulation modes, a binary decision tree requires a theoretical minimum depth of $\lceil \log_2 4 \rceil = 2$ and 3 decision splits to partition a 1D scalar space into 4 contiguous intervals.
- **Representation Fidelity Baseline:** The current 1D experiment functions as a baseline representation and fidelity test, verifying that CART exactly learns the discrete 1D Look-Up Table policy. It is not presented as a broad statistical generalization study across continuous or unobserved channel profiles.
- **Extensibility:** The refactored `CARTClassifier` abstraction is feature-agnostic, ready to ingest multi-feature inputs (such as instantaneous channel magnitude $|h|$) without altering the core learning or threshold attribution logic.
