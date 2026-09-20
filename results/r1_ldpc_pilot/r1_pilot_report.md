# PHY-ML R1: LDPC Coded PHY Foundation Pilot Verification Report

**Generated At (UTC):** `2026-09-20T16:14:19.032218+00:00`  
**Source Git Commit:** `193d864f5570664a6bfc8714ca3c8766bd1aef8c`  
**Execution Runtime:** `10.64 s`  
**Stage:** R1 LDPC Coded Foundation (Verification Pilot)  

---

## 1. Executive Summary & Design Scope

This report verifies the initial integration of **5G NR LDPC channel coding** into the PHY-ML research core, directly inheriting from the frozen **R0 uncoded baseline root**.

### Invariant Physical Foundations (Preserved from R0)
- **Channel Model:** Slow Rayleigh flat block fading (constant complex gain $h \sim \mathcal{CN}(0, 1)$ over block symbols, independent between blocks).
- **CSI Knowledge:** Perfect coherent CSI known at the receiver.
- **Simulation Setup Axis:** Nominal $E_s/N_0$ in dB held constant ($E_s = 1$). No silent axis shift.
- **Constellation Primitives:** Unit average energy Gray-mapped constellations (BPSK, QPSK, 16QAM, 64QAM).
- **Seeding Discipline:** Exact deterministic reproducibility using PyTorch generators.
- **R0 Protection:** R0 frozen artifacts, calibrations, and decision trees remain completely untouched.

### Single Communication-System Complexity Added
- **Channel Coding:** 5G NR LDPC encoder and soft belief-propagation decoder (`sionna.phy.fec.ldpc.LDPC5GEncoder` / `LDPC5GDecoder`).
- **Demapper Interface:** Pure soft log-likelihood ratios (LLRs). Absolutely no hard decisions before LDPC decoding.

---

## 2. R1 Action Representation & Spectral Efficiency

Actions in R1 are parameterized as tuples `(modulation, code_rate)` with effective spectral efficiency:
$$\eta = R_c \cdot \log_2(M) \quad [\text{information bits / complex symbol}]$$

The derived relationship to information-bit $E_b/N_0$ is recorded for fairness:
$$\left(\frac{E_b}{N_0}\right)_{\text{dB}} = \left(\frac{E_s}{N_0}\right)_{\text{dB}} - 10 \log_{10}(\eta)$$

### Initial Validated Action Catalog

| Action Name | Modulation ($M$) | Bits/Symbol ($\log_2 M$) | Target Code Rate ($R_c$) | Info Bits ($k$) | Coded Bits ($n$) | Spectral Efficiency ($\eta$) |
|:------------|:----------------:|:------------------------:|:------------------------:|:---------------:|:----------------:|:----------------------------:|
| `BPSK-1/2` | BPSK (2) | 1 | `1/2` (0.500) | 768 | 1536 | **0.50** |
| `QPSK-1/2` | QPSK (4) | 2 | `1/2` (0.500) | 1536 | 3072 | **1.00** |
| `QPSK-2/3` | QPSK (4) | 2 | `2/3` (0.667) | 2048 | 3072 | **1.33** |
| `16QAM-1/2` | 16QAM (16) | 4 | `1/2` (0.500) | 3072 | 6144 | **2.00** |
| `16QAM-3/4` | 16QAM (16) | 4 | `3/4` (0.750) | 4608 | 6144 | **3.00** |
| `64QAM-1/2` | 64QAM (64) | 6 | `1/2` (0.500) | 4608 | 9216 | **3.00** |
| `64QAM-2/3` | 64QAM (64) | 6 | `2/3` (0.667) | 6144 | 9216 | **4.00** |
| `64QAM-3/4` | 64QAM (64) | 6 | `3/4` (0.750) | 6912 | 9216 | **4.50** |

---

## 3. Independent Statistical Unit & Reliability Metrics

- **Independent Statistical Unit:** Under slow block fading, each codeword of $n = 1536 \cdot m$ bits spans exactly one transmission block and experiences one independent channel realization $h$. The **codeword / transport-block is the independent statistical unit**.
- **Codeword BLER:** Modeled as independent Bernoulli trials across blocks ($E_i \in \{0, 1\}$). Standard error: $\text{SE}(\text{BLER}) = \sqrt{\frac{\text{BLER}(1 - \text{BLER})}{N_{\text{blocks}}}}$.
- **Information BER:** Bit errors within a codeword are correlated. Standard error is computed from the empirical sample standard deviation of per-block BERs divided by $\sqrt{N_{\text{blocks}}}$.

---

## 4. Pilot Verification Results

| Action | Nominal $E_s/N_0$ | Derived $E_b/N_0$ | Blocks | Info Bits ($k$) | Coded Bits ($n$) | Bit Errors | Info BER | Codeword Errors | Codeword BLER |
|:-------|:-----------------:|:-----------------:|:------:|:---------------:|:----------------:|:----------:|:--------:|:---------------:|:-------------:|
| `BPSK-1/2` | 10.0 dB | 13.01 dB | 20 | 768 | 1536 | 0 | 0.0000e+00 | 0 | **0.0000** |
| `BPSK-1/2` | 18.0 dB | 21.01 dB | 20 | 768 | 1536 | 0 | 0.0000e+00 | 0 | **0.0000** |
| `BPSK-1/2` | 26.0 dB | 29.01 dB | 20 | 768 | 1536 | 0 | 0.0000e+00 | 0 | **0.0000** |
| `QPSK-1/2` | 10.0 dB | 10.00 dB | 20 | 1536 | 3072 | 0 | 0.0000e+00 | 0 | **0.0000** |
| `QPSK-1/2` | 18.0 dB | 18.00 dB | 20 | 1536 | 3072 | 0 | 0.0000e+00 | 0 | **0.0000** |
| `QPSK-1/2` | 26.0 dB | 26.00 dB | 20 | 1536 | 3072 | 0 | 0.0000e+00 | 0 | **0.0000** |
| `QPSK-2/3` | 10.0 dB | 8.75 dB | 20 | 2048 | 3072 | 0 | 0.0000e+00 | 0 | **0.0000** |
| `QPSK-2/3` | 18.0 dB | 16.75 dB | 20 | 2048 | 3072 | 168 | 4.1016e-03 | 2 | **0.1000** |
| `QPSK-2/3` | 26.0 dB | 24.75 dB | 20 | 2048 | 3072 | 0 | 0.0000e+00 | 0 | **0.0000** |
| `16QAM-1/2` | 10.0 dB | 6.99 dB | 20 | 3072 | 6144 | 3050 | 4.9642e-02 | 4 | **0.2000** |
| `16QAM-1/2` | 18.0 dB | 14.99 dB | 20 | 3072 | 6144 | 0 | 0.0000e+00 | 0 | **0.0000** |
| `16QAM-1/2` | 26.0 dB | 22.99 dB | 20 | 3072 | 6144 | 0 | 0.0000e+00 | 0 | **0.0000** |
| `16QAM-3/4` | 10.0 dB | 5.23 dB | 20 | 4608 | 6144 | 6164 | 6.6884e-02 | 8 | **0.4000** |
| `16QAM-3/4` | 18.0 dB | 13.23 dB | 20 | 4608 | 6144 | 0 | 0.0000e+00 | 0 | **0.0000** |
| `16QAM-3/4` | 26.0 dB | 21.23 dB | 20 | 4608 | 6144 | 1441 | 1.5636e-02 | 2 | **0.1000** |
| `64QAM-1/2` | 10.0 dB | 5.23 dB | 20 | 4608 | 9216 | 10135 | 1.0997e-01 | 10 | **0.5000** |
| `64QAM-1/2` | 18.0 dB | 13.23 dB | 20 | 4608 | 9216 | 2020 | 2.1918e-02 | 3 | **0.1500** |
| `64QAM-1/2` | 26.0 dB | 21.23 dB | 20 | 4608 | 9216 | 553 | 6.0004e-03 | 1 | **0.0500** |
| `64QAM-2/3` | 10.0 dB | 3.98 dB | 20 | 6144 | 9216 | 15585 | 1.2683e-01 | 15 | **0.7500** |
| `64QAM-2/3` | 18.0 dB | 11.98 dB | 20 | 6144 | 9216 | 7876 | 6.4095e-02 | 9 | **0.4500** |
| `64QAM-2/3` | 26.0 dB | 19.98 dB | 20 | 6144 | 9216 | 3764 | 3.0632e-02 | 2 | **0.1000** |
| `64QAM-3/4` | 10.0 dB | 3.47 dB | 20 | 6912 | 9216 | 25318 | 1.8315e-01 | 19 | **0.9500** |
| `64QAM-3/4` | 18.0 dB | 11.47 dB | 20 | 6912 | 9216 | 4806 | 3.4766e-02 | 7 | **0.3500** |
| `64QAM-3/4` | 26.0 dB | 19.47 dB | 20 | 6912 | 9216 | 5378 | 3.8903e-02 | 5 | **0.2500** |

---

## 5. Next Steps for R1

1. Complete code rate and modulation expansion.
2. Establish R1 Monte Carlo calibration protocol (balancing compute cost with LDPC decoding latency).
3. Define Ground Truth policy for coded transmission (BLER target vs BER target).

