# PHY-ML: Joint Modulation, Coding & Decoder Effort Adaptation (PHY-AMC)

> **Lightweight Machine Learning for Joint Link Adaptation & Decoder Resource Optimization on Fading Wireless Channels**

---

## 📌 Overview

This repository contains the numerical research core (`research_core`) implementing **PHY-AMC**: an end-to-end framework for joint adaptation of:
1. **Modulation Scheme**: BPSK, QPSK, 16-QAM, 64-QAM, 256-QAM.
2. **Channel Coding**: Uncoded transmission or LDPC (5G NR compliant code rates $R \in \{1/2, 2/3, 3/4, 5/6\}$ via Sionna/PyTorch).
3. **Decoder Effort Allocation**: Adaptive Belief Propagation (BP) iteration capping ($I_{\max} \in \{5, 10\}$) to optimize decoding energy and latency budget.

Adaptation is driven by interpretable, low-latency **Lightweight Machine Learning** models (CART Decision Trees, Random Forests) operating on pilot-based SNR/CSI estimates under real-time constraints ($\mu$s-scale inference latency).

---

## 🏗️ Architecture & Philosophy

The project follows a **3-Tier Architecture** adhering to rigorous empirical evaluation protocols:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Tier 1: Soundness Verification Gate (Physics & Anchor Checks)               │
│ - AWGN erfc analytical SER/BER                                              │
│ - Rayleigh fading integration bounds                                        │
│ - Cramér-Rao lower bound on pilot-based channel estimation                  │
│ - Disjoint seed & partition integrity                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier 2: Physical Layer Engine (Sionna / NumPy PHY Link)                      │
│ - Single-carrier link with block fading channels                            │
│ - Configurable pilots, symbols, SNR range, and coherence times              │
│ - Uncoded & LDPC 5G NR code rates with BP decoders                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier 3: Adaptive Policy & ML Decision Engine                                │
│ - Baselines: Fixed Modulation, Lookup Table (LUT), Empirical Oracle        │
│ - Lightweight ML: Decision Tree (DT) & Random Forest (RF)                   │
│ - Strict pre-registration: Disjoint train/validation/test sets, no leakage │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Installation

Clone the repository:
```bash
git clone https://github.com/vovupro/PHY-ML.git
cd PHY-ML
```

Install core dependencies (NumPy, SciPy, Scikit-Learn, Joblib):
```bash
pip install -r requirements.txt
```

*(Optional)* If you wish to run 5G LDPC simulations with GPU/PyTorch acceleration and Sionna:
```bash
pip install -r requirements-ldpc.txt
```

---

### 2. Verify Physics & System Integrity

Run the verification test suite to ensure mathematical anchors and physical layer calculations match theoretical bounds:
```bash
python test_physics.py
```

---

### 3. Run Benchmarks

Run the standard numerical AMC evaluation:
```bash
# Fast uncoded benchmark across SNR grid
python run_benchmark.py --coding uncoded --snrs 0 6 12 18 24 30 --output results/uncoded.json

# LDPC joint adaptation benchmark
python run_benchmark.py --coding ldpc --caps 5 10 --budget 10 --output results/ldpc.json
```

Verify generated benchmark results:
```bash
python verify_results.py
```

---

## 📂 Project Structure

```
├── channel.py                     # Channel models (AWGN, Rayleigh) & channel estimation
├── phy_engine.py                  # Core PHY transceiver engine & modulation catalog
├── policies.py                    # Adaptation policies (Fixed, LUT, Oracle, DT, RF)
├── experiment.py                  # Protocol enforcement, data split & evaluation logic
├── metrics.py                     # Performance metrics (BER, BLER, Goodput, EE)
├── run_benchmark.py               # CLI entry point for running empirical studies
├── test_physics.py                # Verification gate testing against analytical anchors
├── verify_results.py              # Integrity checks on experiment outputs
├── PHUONG_PHAP_LUAN_VA_THAM_KHAO.md # Full methodology, theory, and references (in Vietnamese)
├── ml_model_selection_review.txt  # Model selection rationale and review
├── requirements.txt               # Core dependencies
└── requirements-ldpc.txt          # Optional LDPC & Sionna dependencies
```

---

## 📄 Documentation

For deep technical details on the methodology, system assumptions, baseline comparisons, and theoretical background, refer to [PHUONG_PHAP_LUAN_VA_THAM_KHAO.md](PHUONG_PHAP_LUAN_VA_THAM_KHAO.md).

---

## 📜 License

MIT License.
