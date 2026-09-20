# PHY-ML R1: Monte Carlo Experiment Design Proposal
**Stage:** Pre-Monte Carlo Architecture & Methodology Proposal  
**Target:** R1 LDPC Coded Link Adaptation Baseline  
**Base Commit:** `939cd89e017a537f2e9fbd0316ab258103a384b9`  
**Parent Reference:** Frozen R0 Uncoded AMC Baseline  

> [!IMPORTANT]
> **GOVERNANCE NOTICE: NO PRODUCTION MONTE CARLO EXECUTION**  
> This document is an architectural and experimental design proposal. In strict compliance with project quality rules, **no production SNR sweeps, deep calibrations, or Ground Truth policies** may be executed until human review and approval of the unresolved design decisions detailed in [Section M](#m-approval-checklist).

---

## A. Scientific Question

### Primary Investigation
> **"What changes in the optimal learned link-adaptation policy when 5G NR LDPC channel coding and code-rate selection are introduced, while physical channel fading, coherent receiver CSI, state representation, learning algorithm, and nominal SNR setup axis remain strictly fixed?"**

### Detailed Inquiries
1. **Coding Gain & Boundary Displacement:**  
   By how many decibels do the optimal mode switching thresholds shift toward lower SNRs under 5G NR LDPC coding relative to the frozen R0 uncoded root baseline ($[16.88, 22.88, 28.38]\text{ dB}$)?
2. **Action Granularity (Modulation vs Code Rate):**  
   Does an information-theoretically optimal decision tree prefer switching code rates within the same constellation (e.g., QPSK-1/2 $\to$ QPSK-2/3) or switching modulation orders at a fixed code rate (e.g., QPSK-1/2 $\to$ 16QAM-1/2)?
3. **Action Dominance at Identical Spectral Efficiency:**  
   For candidate actions sharing identical spectral efficiency ($\eta = 3.00$ info bits/symbol: `16QAM-3/4` vs `64QAM-1/2`), which action dominates under slow Rayleigh flat fading, or do their operational reliability regions partition the SNR axis?
4. **Policy Parsimony & Tree Complexity:**  
   Can the expanded coded action space be represented with 100% classification fidelity by a parsimonious, hardware-friendly Decision Tree of depth $d \le 4$, or does the waterfall behavior of LDPC codes require greater depth?

---

## B. Proposed Action Set

### Full Candidate Action Catalog (8 Actions)
Each action is defined as a tuple `(modulation, code_rate)` evaluated over a canonical transmission block of $N_{\text{symbols}} = 1536$ complex symbols:

| Action Name | Modulation ($M$) | Bits/Symbol ($m$) | Requested $R_c$ | Actual $k$ | Actual $n$ | Effective $R_c$ ($k/n$) | Spectral Efficiency ($\eta$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `BPSK-1/2` | BPSK (2) | 1 | 1/2 (0.500) | 768 | 1536 | 0.500000 | **0.50** |
| `QPSK-1/2` | QPSK (4) | 2 | 1/2 (0.500) | 1536 | 3072 | 0.500000 | **1.00** |
| `QPSK-2/3` | QPSK (4) | 2 | 2/3 (0.667) | 2048 | 3072 | 0.666667 | **1.33** |
| `16QAM-1/2` | 16QAM (16) | 4 | 1/2 (0.500) | 3072 | 6144 | 0.500000 | **2.00** |
| `16QAM-3/4` | 16QAM (16) | 4 | 3/4 (0.750) | 4608 | 6144 | 0.750000 | **3.00** |
| `64QAM-1/2` | 64QAM (64) | 6 | 1/2 (0.500) | 4608 | 9216 | 0.500000 | **3.00** |
| `64QAM-2/3` | 64QAM (64) | 6 | 2/3 (0.667) | 6144 | 9216 | 0.666667 | **4.00** |
| `64QAM-3/4` | 64QAM (64) | 6 | 3/4 (0.750) | 6912 | 9216 | 0.750000 | **4.50** |

### Redundancy & Dominance Collision: $\eta = 3.00$
Both `16QAM-3/4` and `64QAM-1/2` deliver exactly $\eta = 3.00$ information bits per complex symbol:
- **`16QAM-3/4`:** Smaller constellation (16-QAM) with weaker rate-3/4 LDPC code.
- **`64QAM-1/2`:** Denser constellation (64-QAM) with stronger rate-1/2 LDPC code.

Under additive Gaussian noise or fading, one candidate typically exhibits lower BLER or sharper waterfall transition than the other. Rather than deleting one a priori, both are retained in the full candidate catalog to allow empirical investigation.

### Catalog Recommendations for Human Review
- **Option 1 (Full Exploratory Catalog):** Retain all 8 actions. Resolves the $\eta=3.00$ competition empirically.
- **Option 2 (Strictly Monotonic Throughput Catalog - 7 Actions):** Prune one of the $\eta=3.00$ actions prior to calibration once a preliminary tie-break preference is decided.

---

## C. Reliability Metric Options

In coded communications, bit errors within a single codeword are strongly correlated due to parity-check constraints and block channel fading. Therefore, reliability metrics must be selected carefully:

1. **Candidate 1: Codeword BLER Only (Recommended)**  
   - $\text{BLER} = \frac{N_{\text{err\_codewords}}}{N_{\text{codewords}}}$.  
   - Direct correspondence with transport-block packet reception in wireless standards (3GPP NR / LTE).  
   - Well-defined Bernoulli trial statistics across independent fading realizations.
2. **Candidate 2: Codeword BLER Primary with Information BER Auditing**  
   - Selection rule governed by $\text{BLER} \le \text{BLER}_{\text{target}}$.  
   - Information BER is recorded as an observational diagnostic for reporting.
3. **Candidate 3: Dual Constraint ($\text{BLER} \le \text{BLER}_{\text{target}}$ and $\text{BER}_{\text{info}} \le \text{BER}_{\text{target}}$)**  
   - Requires simultaneous compliance with both packet-level and bit-level constraints.

---

## D. BLER Target Options

Candidate operating targets for human consideration:

| Candidate Target | Target Domain | Scientific Relevance | Monte Carlo Sample Size Floor |
|:---:|:---|:---|:---:|
| **$\text{BLER}_{\text{target}} = 0.10$ (10%)** | 3GPP NR Initial Transmission Target | Standard cellular link adaptation target; balances spectrum efficiency with retransmission overhead. | $\sim 500 - 1,000$ codewords |
| **$\text{BLER}_{\text{target}} = 0.01$ (1%)** | High-Reliability / Ultra-Reliable | Direct numerical parity with uncoded R0 target ($\text{BER}_{\text{target}} = 0.01$); sharp waterfall threshold. | $\sim 5,000 - 10,000$ codewords |
| **$\text{BLER}_{\text{target}} = 0.05$ (5%)** | Intermediate Low-Latency | Balanced compromise between 10% and 1%. | $\sim 2,000 - 5,000$ codewords |

---

## E. Confidence Interval Method

For Monte Carlo evaluation of Bernoulli codeword errors ($x$ errors out of $N$ blocks), three confidence interval formulations are available:

### 1. Wald Normal Approximation (Diagnostic Only)
$$\text{CI}_{\text{Wald}} = \hat{p} \pm z_{1-\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{N}}$$
- **Flaws:** At $\hat{p} = 0$, $\text{SE} = 0$, giving a degenerate zero-width interval $[0, 0]$ that severely underestimates rare-event uncertainty. Produces negative lower bounds near zero.

### 2. Wilson Score Interval (Recommended Candidate)
$$\tilde{p} = \frac{\hat{p} + \frac{z^2}{2N}}{1 + \frac{z^2}{N}}, \quad \text{Margin} = \frac{z}{1 + \frac{z^2}{N}} \sqrt{\frac{\hat{p}(1-\hat{p})}{N} + \frac{z^2}{4N^2}}$$
$$\text{CI}_{\text{Wilson}} = \left[\max\left(0, \tilde{p} - \text{Margin}\right), \, \min\left(1, \tilde{p} + \text{Margin}\right)\right]$$
- **Advantages:** Asymmetric, bounded in $[0, 1]$, and maintains an accurate non-zero upper bound at $\hat{p} = 0$ ($z^2 / (N + z^2)$).

### 3. Clopper-Pearson Exact Interval (Conservative Candidate)
$$\text{CI}_{\text{CP}} = \left[\text{Beta}\left(\frac{\alpha}{2}; x, N - x + 1\right), \, \text{Beta}\left(1 - \frac{\alpha}{2}; x + 1, N - x\right)\right]$$
- **Advantages:** Guaranteed exact minimum coverage $\ge 1 - \alpha$. Strictly conservative.

---

## F. Ground Truth Rule

### Proposed Decision Engine Formulation
For operating point $\text{SNR}_j$ and candidate action $a$:
$$\text{Eligible}(a, \text{SNR}_j) \iff \text{Upper\_CI}_{95\%}\left(\text{BLER}(a, \text{SNR}_j)\right) \le \text{BLER}_{\text{target}}$$
$$\text{BestAction}(\text{SNR}_j) = \arg\max_{a \in \text{Eligible}} \eta(a)$$

### Unresolved Policy Decisions Requiring Approval
1. **Equal-$\eta$ Tie-Break Strategy (e.g. 16QAM-3/4 vs 64QAM-1/2):**
   - **Strategy A (Lowest BLER):** Choose candidate with smaller empirical BLER.
   - **Strategy B (Largest Margin):** Choose candidate with largest gap below target.
   - **Strategy C (Lower Modulation):** Favor smaller constellation (16-QAM) for RF/PA linearity.
   - **Strategy D (Stronger Coding):** Favor lower code rate (Rc=1/2) for fading robustness.
   - **Strategy E (Deterministic Predefined Ordering):** Fixed hierarchy list.
2. **Fallback Strategy When Eligible Set is Empty:**
   - **Option 1:** Lowest spectral efficiency candidate (`BPSK-1/2`).
   - **Option 2:** Fixed robust candidate action.
3. **Boundary Uncertainty Handling:**
   - How to audit points where an action's lower CI is below target but upper CI is above target.

---

## G. SNR Range Discovery

Because LDPC coding provides significant coding gain, the useful operating SNR range will shift substantially to the left compared to R0 ($0 \text{ dB} \dots 30 \text{ dB}$):

### Recommended Coarse Reconnaissance Plan
- **Grid Step:** Coarse $2.0 \text{ dB}$ steps.
- **Estimated Window:** $-4.0 \text{ dB}$ to $+24.0 \text{ dB}$ (15 evaluation points).
- **Block Budget:** Extremely light ($N = 200$ blocks/point) purely to detect waterfall cliff locations.
- **Strict Constraint:** Reconnaissance is used only to bound the production grid window and will NOT be used for Ground Truth training.

---

## H. Grid Refinement Plan

Following coarse reconnaissance:
1. **Global Uniform Grid:** $1.0 \text{ dB}$ spacing across the active operational envelope.
2. **Local Switching Boundary Refinement:**
   - Detect mode switching brackets $[s_k, s_{k+1}]$ between consecutive BestActions.
   - Evaluate $0.5 \text{ dB}$ midpoints.
   - Evaluate $0.25 \text{ dB}$ quarter-points if needed to resolve boundary parsimony.
3. **Waterfall Consideration:** Because LDPC coded BLER exhibits much steeper slopes than uncoded BER, switching intervals are typically narrower.

---

## I. Monte Carlo Budget Study

Using the measured throughput from the CPU micro-benchmark ($\sim 5 - 25 \text{ ms/codeword}$) and expected CUDA acceleration ($\sim 0.2 - 1.0 \text{ ms/codeword}$):

| Budget Level | Codewords / Seed ($N$) | Pooled Codewords ($2N$) | Statistical Resolution Floor ($1/2N$) | $95\%$ CI Width at $\text{BLER}=0.10$ |
|:---|:---:|:---:|:---:|:---:|
| **Reconnaissance** | 200 | 400 | $2.5 \times 10^{-3}$ | $\pm 0.029$ |
| **Light MC** | 1,000 | 2,000 | $5.0 \times 10^{-4}$ | $\pm 0.013$ |
| **Medium MC** | 5,000 | 10,000 | $1.0 \times 10^{-4}$ | $\pm 0.006$ |
| **Deep Reference** | 20,000 | 40,000 | $2.5 \times 10^{-5}$ | $\pm 0.003$ |

---

## J. Deep Reference Strategy

To preserve the proven scientific methodology of R0:
- **Two Canonical Independent Seeds:** Seed A (`20260918`) and Seed B (`20260919`).
- **Fixed Budget Calibration:** No adaptive early stopping. Every evaluated SNR point executes the complete requested budget.
- **Targeted Extension:** Points exhibiting boundary overlap uncertainty can be extended via targeted refinement rather than modifying global budgets.

---

## K. Compute Cost Matrix

Total computation cost formula:
$$\text{GPU Hours} = \frac{N_{\text{actions}} \times N_{\text{SNR}} \times N_{\text{blocks}} \times T_{\text{CW}}}{3600}$$

Assuming $T_{\text{CW}} \approx 0.5 \text{ ms}$ on RTX 3060 CUDA ($2000 \text{ CW/s}$), 8 actions, and a 25-point SNR grid:

| Scenario | Codewords / Action / SNR | Total Codewords | Estimated GPU Time | Estimated CPU Time |
|:---|:---:|:---:|:---:|:---:|
| **Pilot / Reconnaissance** | 200 | 40,000 | $\sim 20 \text{ seconds}$ | $\sim 5 \text{ minutes}$ |
| **Light MC** | 1,000 | 200,000 | $\sim 1.7 \text{ minutes}$ | $\sim 25 \text{ minutes}$ |
| **Medium MC** | 5,000 | 1,000,000 | $\sim 8.3 \text{ minutes}$ | $\sim 2.1 \text{ hours}$ |
| **Deep MC** | 20,000 | 4,000,000 | $\sim 33.3 \text{ minutes}$ | $\sim 8.5 \text{ hours}$ |

---

## L. Production Artifact Plan

Future production runs will produce structured directories parallel to R0:
- `results/r1_mc_light/`: Initial 1k/seed fixed-budget calibration.
- `results/r1_mc_deep/`: Full fixed-budget reference calibration.
- `results/r1_mc_budget_study/`: Light-vs-Deep convergence and fidelity audit.
- `results/r1_snr_grid_convergence/`: Boundary resolution study ($1.0 \text{ dB} \to 0.5 \text{ dB} \to 0.25 \text{ dB}$).
- `results/r1_final/`: Unified frozen R1 baseline package, serialized CART model, and cryptographic manifest.

---

## M. Approval Checklist

Before any production Monte Carlo run is initiated, the following decisions must be explicitly reviewed and approved:

- [ ] **1. Action Catalog:** Approve full 8-action catalog or 7-action reduced catalog.
- [ ] **2. Decoder Iterations:** Confirm fixed setting of $15$ iterations (or alternative $10 / 20$).
- [ ] **3. Primary Reliability Metric:** Confirm Codeword BLER as primary gating metric.
- [ ] **4. Target BLER:** Select $\text{BLER}_{\text{target}} \in \{0.10, 0.05, 0.01\}$.
- [ ] **5. CI Method:** Select Wilson, Clopper-Pearson, or Wald for production Ground Truth.
- [ ] **6. Confidence Level:** Confirm $95\%$ confidence level ($k = 1.96$).
- [ ] **7. Eligibility Rule:** Confirm conservative upper-CI eligibility ($\text{Upper\_CI} \le \text{Target}$).
- [ ] **8. Equal-$\eta$ Tie-Break Rule:** Select Strategy A (lowest BLER), C (lower modulation), D (stronger coding), or F (predefined).
- [ ] **9. Fallback Rule:** Confirm lowest spectral efficiency action as fallback.
- [ ] **10. Coarse SNR Reconnaissance:** Approve coarse range (e.g. $-4 \text{ dB} \dots 24 \text{ dB}$ at $2 \text{ dB}$ steps).
- [ ] **11. Production MC Budgets:** Select Light budget ($N_{\text{light}}$) and Deep budget ($N_{\text{deep}}$).
- [ ] **12. Grid Refinement Schedule:** Approve transition refinement stages ($1.0 \text{ dB} \to 0.5 \text{ dB} \to 0.25 \text{ dB}$).
- [ ] **13. Baseline Policy Definitions:** Confirm Fixed Robust, Fixed High-Throughput, 1D LUT, and learned CART baseline definitions.
