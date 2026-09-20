# PHY-ML: Minimal Verified Uncoded PHY Research Core

A transparent, reproducible, and mathematically verified physical layer (PHY) foundation for wireless communication and link adaptation studies, powered by **Sionna 2.0** as the canonical PHY backend.

---

## 📌 Architecture & Design Philosophy

The project separates **experimental control** from **canonical PHY primitives**:

```text
                   OUR RESEARCH CODE
                         │
             ┌───────────┴───────────┐
             ↓                       ↓
     experiment control        reproducibility
     SNR, |h|, seeds           block identity
             │                       │
             └───────────┬───────────┘
                         ↓
                    SIONNA 2.0
                  (PyTorch backend)
                         ↓
              bits → symbols → channel
                         ↓
             coherent receiver (y / h)
                         ↓
                  Sionna Demapper
                         ↓
                    raw metrics
```

- **Sionna 2.0**: Canonical PHY primitive library for constellation definitions, Gray mapping, and maximum-likelihood demapping.
- **Our Experiment Control Layer**: Governs block identities, RNG seeding (`SeedSequence`), channel realization pairing, and 2D parameter sweeps ($E_s/N_0$ vs $|h|$).

---

## 📌 Current Stage: R0 Canonical Frozen Research Root

`PHY-ML R0` represents the finalized, frozen uncoded baseline root for adaptive modulation and coding (AMC) link adaptation:
- **Physical Channel:** Slow Rayleigh flat block fading ($\mathbb{E}[|h|^2] = 1.0$, held strictly constant over 1536 complex symbols/block, independent realizations between blocks).
- **CSI Knowledge:** Perfect coherent channel state information (CSI) at the receiver.
- **State Space:** $s = [\text{SNR}_{\text{dB}}]$ (nominal setup $E_s/N_0$).
- **Action Space:** $\mathcal{A} = \{\text{BPSK}, \text{QPSK}, \text{16QAM}, \text{64QAM}\}$ (uncoded, Gray-mapped, unit average energy).
- **Monte Carlo Calibration:** Fixed-budget Monte Carlo with canonical Deep reference budget of 70,000 blocks/seed (140,000 pooled blocks/point across Seeds A & B).
- **Ground Truth Policy:** Conservative 95% confidence upper bound ($\hat{\text{BER}} + 1.96 \cdot \text{SE} \le \text{BER}_{\text{target}}$ with $\text{BER}_{\text{target}} = 0.0100$).
- **Local Transition Resolution:** Targeted 0.25-dB refinement at candidate mode switching boundaries (16.75 dB, 22.75 dB, 28.25 dB), forming a unified 28-point operating grid.
- **Decision Tree Architecture:** Classification Decision Tree (CART) trained via minimum-depth model selection ($d \in [1, 5]$) to maximize policy parsimony and transparency.

---

## ⚙️ Computational Runtime & Artifact Taxonomy

### Computational Platforms
- **Reference Simulation Environment (CKEY)**:
  - Python: `3.13.5`
  - PyTorch: `2.9.1+cu126`
  - CUDA Runtime: `12.6`
  - Sionna: `2.0.0`
  - Hardware: NVIDIA GeForce RTX 3060 (`sm_86`)

### Canonical Artifact Pipeline
1. **L2 Fixed-Budget Monte Carlo Calibration**:
   - Light profile: 10,000 blocks/seed $\to$ `results/r0_mc_light_10k/`
   - Deep profile: 70,000 blocks/seed $\to$ `results/r0_mc_deep_70k/`
2. **Targeted 0.25-dB Transition Refinement**:
   - Evaluates 16.75, 22.75, 28.25 dB at Deep 70k budget $\to$ `results/r0_snr_grid_refine_025/`
3. **Downstream Comparative Studies**:
   - Light vs Deep budget comparison $\to$ `results/r0_mc_budget_study/`
   - SNR-grid convergence study $\to$ `results/r0_snr_grid_convergence/`
4. **Canonical R0 Freeze Package**:
   - Unified 28-point dataset, ground truth, DT model selection, policy comparison, and cryptographic manifest $\to$ `results/r0_final/`

---

## ✅ Implemented Features

- **PHY Backend**:
  - Powered by **Sionna 2.0** (PyTorch backend, 64-bit double precision).
- **Modulation Schemes**:
  - BPSK ($1\text{ bit/symbol}$) — via Sionna PAM
  - QPSK ($2\text{ bits/symbol}$) — via Sionna QAM
  - 16-QAM ($4\text{ bits/symbol}$) — via Sionna QAM
  - 64-QAM ($6\text{ bits/symbol}$) — via Sionna QAM
- **Constellation Properties**:
  - Exact unit average symbol energy normalization: $\mathbb{E}[|s|^2] = 1.0$.
  - Canonical 2D Gray mapping: adjacent nearest-neighbour constellation points differ by exactly 1 bit.
- **Physical Channel Models**:
  - **AWGN**: Additive white Gaussian noise channel with unit gain ($h = 1.0 + 0.0j$).
  - **Slow Rayleigh Block Fading**: One complex scalar channel coefficient generated via Sionna's `GenerateFlatFadingChannel(num_tx_ant=1, num_rx_ant=1, precision='double')` with $\mathbb{E}[|h|^2] = 1.0$, held strictly **constant across the entire transmission block** ($y = h \cdot x + n$). Independent realizations drawn between independent blocks.
  - Future-proofed 2D conditioning interface ($h\_magnitude$ parameter) for subsequent fading envelope sweeps.
- **Receiver Architecture**:
  - Perfect coherent channel state information (CSI) baseline.
  - Sionna hard APP demapping: true channel equalization $\hat{x} = y / h$ followed by Sionna `Demapper('app', hard_out=True)` with effective noise variance $N_{0, \text{eff}} = N_0 / |h|^2$.
- **SNR Convention**:
  - Explicitly defined as nominal/setup $E_s / N_0$ in dB ($snr\_db$).
  - With unit average constellation energy $E_s = 1.0$, the complex noise variance is $N_0 = 1 / \text{db\_to\_lin}(snr\_db)$.
- **RNG Handling & Paired Evaluations**:
  - Fully deterministic and reproducible using explicit Torch `Generator` streams (`bit_rng`, `fading_rng`, `noise_rng`).
  - Paired physical evaluations across candidate modulation modes share identical $h$ and noise realizations, strictly invariant to evaluation order.
- **Raw PHY Metrics**:
  - Transparent error counters: bit error count, total bits, BER, block error count, total blocks, BLER, bits_per_symbol.
- **Physics Acceptance Suite**:
  - 15 automated tests validating Sionna Mapper $\to$ Demapper roundtrip, constellation energy, Gray consistency, noiseless round-trip, coherent demodulation, exact analytical AWGN BER ($Q$-function/erfc and exact PAM/2D decision-region integration), analytical Rayleigh average BER, $h$ constancy, independence, and deterministic reproducibility.

---

## 🚫 Deferred Beyond R0 (Future Milestone Extensions)

The frozen R0 uncoded root deliberately excludes:
- Channel coding (LDPC, Polar, Convolutional, Turbo)
- 2D feature representations ($|h|$ vs $E_s/N_0$)
- Pilot symbol transmission & practical channel/SNR estimation
- CRC verification, ACK/NACK feedback, and HARQ retransmissions
- FPGA export and fixed-point quantization

---

## 📂 Active Repository Structure

```text
PHY-ML/
├── .gitignore                      # Clean Git ignore definitions
├── README.md                       # Canonical R0 project overview and scope
├── __init__.py                     # Package root exposing public PHY API
├── requirements.txt                # Minimal scientific Python dependencies
├── requirements-ckey.txt           # Dedicated CUDA/Sionna requirements for CKEY
├── channel.py                      # Slow Rayleigh block fading and AWGN channel model
├── phy_engine.py                   # Sionna 2.0-backed modulation, equalization, and demapping
├── metrics.py                      # Raw PHY counters and error rate calculations
├── calibration_1d.py               # L2 CPU reference prototype
├── calibration_l2_cuda.py          # L2 Canonical fixed-budget CUDA Monte Carlo engine
├── refine_snr_grid_cuda.py         # Targeted 0.25-dB transition refinement runner
├── ground_truth.py                 # Conservative CI-based Ground Truth & baseline policies
├── cart_1d.py                      # Supervised CART classification framework & depth evaluation
├── compare_mc_budgets.py           # Light vs Deep budget comparative analysis tool
├── analyze_snr_grid_convergence.py # Transition-centered SNR grid convergence study tool
├── package_r0_freeze.py            # Canonical R0 packaging, freeze gate, and manifest builder
└── test_*.py                       # Comprehensive acceptance and regression test suites
```

---

## 🚀 Quick Start

### 1. Installation

Install minimal dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Acceptance Test Suite

Execute the physical validation gate:
```bash
python -m unittest -v test_physics.py
```

All 15 tests must pass:
```text
test_00_sionna_mapper_demapper_noiseless_roundtrip ... ok
test_01_unit_average_constellation_energy ... ok
test_02_constellation_sizes ... ok
test_03_gray_nearest_neighbour_consistency ... ok
test_04_noiseless_round_trip ... ok
test_05_coherent_demodulation_with_known_complex_h ... ok
test_06_awgn_ber_sanity_analytical ... ok
test_07_rayleigh_average_ber_analytical ... ok
test_08_rayleigh_h_unit_average_power ... ok
test_09_h_constant_across_entire_block ... ok
test_10_independent_h_across_blocks ... ok
test_11_deterministic_reproducibility ... ok
test_12_different_block_identity_different_realization ... ok
test_13_batch_and_individual_evaluation_agree ... ok
test_14_modulation_order_invariance_for_paired_eval ... ok
```

---

## 📜 Public API Overview

### `channel.py`
- `make_block_rng(master_seed: int, block_id: int) -> BlockRNG`
- `generate_channel_coefficient(rng, channel_type="rayleigh", h_magnitude=None) -> complex`
- `generate_standard_noise(rng, num_symbols: int) -> np.ndarray`
- `apply_channel(transmitted_symbols, snr_db, channel_type="rayleigh", h=None, ...) -> ChannelOutput`

### `phy_engine.py`
- `Mode` / `ModulationMode`: Dataclass `(mode_id, modulation, bits_per_symbol)`
- `MODES`: Tuple of supported uncoded modes (`BPSK`, `QPSK`, `16QAM`, `64QAM`)
- `constellation(bits_per_symbol: int) -> tuple[np.ndarray, np.ndarray]`
- `modulate(bits: np.ndarray, bits_per_symbol: int) -> np.ndarray`: Sionna 2.0 Mapper
- `demodulate(received_symbols: np.ndarray, h: complex, bits_per_symbol: int, n0: float = 1.0) -> np.ndarray`: Coherent equalization + Sionna 2.0 Demapper
- `PHYEngine(block_symbols: int)`: Core evaluation engine (`evaluate_mode`, `evaluate_paired_block`)

### `metrics.py`
- `RawPHYCounters`: Stateful accumulator for bits, blocks, BER, BLER, mean block BER, and block standard error ($SE$).
- `count_errors`, `count_block_errors`, `compute_ber`, `compute_bler`: Sionna 2.0 canonical functions.

### `calibration_1d.py` (L2 CPU Prototype)
- `Calibration1DConfig`: Configuration parameters (SNR grid, num_blocks=5000, batch_blocks=500, symbols_per_block=1536, master_seed).
- `run_1d_calibration(config) -> List[CalibrationRecord]`: Vectorized chunk-batching Monte Carlo calibration preserving paired realizations.
- `save_calibration_csv(records, output_path)`: Exports raw counts, block variance, and SE to CSV.
- `generate_verification_report(records, config, records_seed_b, ...) -> str`: Rigorous telecom report comparing empirical curves against independent SciPy analytical theory using block standard errors.

### `calibration_l2_cuda.py` (L2 Canonical Fixed-Budget CUDA Engine)
- **Methodology:** Fixed-budget Monte Carlo calibration across 25 SNR operating points (NO adaptive stopping).
- **Profiles:**
  - `light`: 10,000 blocks/seed (20,000 pooled blocks/point) $\to$ `results/r0_mc_light_10k/`
  - `deep`: 70,000 blocks/seed (140,000 pooled blocks/point) $\to$ `results/r0_mc_deep_70k/`
  - Arbitrary override: `--blocks-per-seed N` $\to$ deterministic custom directory `results/r0_mc_custom_{N}k/`
- **Observational Trajectory:** Checkpoints recorded every 5,000 blocks/seed for human convergence inspection only (strictly non-stopping).
- **CLI Commands:**
  - `python calibration_l2_cuda.py --profile light`
  - `python calibration_l2_cuda.py --profile deep`
  - `python calibration_l2_cuda.py --blocks-per-seed 100000`

### `ground_truth.py` (Conservative CI-Based Ground Truth)
- `GroundTruthConfig`: Parameters (`ber_target=0.01`, `fallback_policy="robustest_mode"`, `confidence_k=1.96`).
- `compute_ground_truth(calibration_data, config) -> List[GroundTruthRow]`: Synthesizes BestMode strictly from calibration tables using conservative upper CI ($\hat{\text{BER}} + 1.96 \cdot \text{SE} \le 0.0100$).
- `FixedRobustPolicy`: Baseline unconditionally selecting BPSK (1 bpcu).
- `FixedHighThroughputPolicy`: Baseline unconditionally selecting 64-QAM (6 bpcu).
- `LookupTable1D`: Derived 1D LUT with thresholds at ground-truth mode transition midpoints.

### `refine_snr_grid_cuda.py` (Targeted 0.25-dB Refinement)
- `REFINEMENT_025_SNRS`: Midpoints of candidate 0.5-dB transition intervals `(16.75, 22.75, 28.25 dB)`.
- `run_refinement_calibration(...)`: Evaluates exclusively the 3 refinement points at Deep 70k budget.

### `cart_1d.py` (Supervised CART Classifier)
- `CARTClassifier`: Feature-agnostic decision tree classifier supporting arbitrary input dimensions.
- `CART1DClassifier`: Thin wrapper for 1D scalar SNR link adaptation.
- `TreeMetrics`: Data container for actual depth, node/leaf counts, accuracy, and learned split thresholds.

### `compare_mc_budgets.py` (Budget Study Tool)
- `run_budget_comparison(...)`: Standalone tool comparing Light 10k vs Deep 70k calibration tables.
- `select_optimal_cart_depth(...)`: Sweeps depths 1..5 to select the smallest tree achieving 100% fidelity.

### `analyze_snr_grid_convergence.py` (Grid Convergence Tool)
- `run_snr_grid_convergence_study(...)`: Merges Deep 70k and refinement datasets to evaluate local transition views (1.0 dB, 0.5 dB, 0.25 dB).
- `compute_data_driven_findings(...)`: Dynamically formats findings on bracket widths, threshold shifts, and tree depth stability.

### `package_r0_freeze.py` (Canonical R0 Freeze Package Builder)
- `package_r0_freeze(...)`: Consolidates the canonical 28-point R0 baseline package into `results/r0_final/`.
- `run_freeze_gate(...)`: Data-driven audit certifying grid integrity, BestMode monotonicity, and 100% DT fidelity before declaring PASS.
- Artifacts produced: `r0_calibration_merged.csv`, `r0_ground_truth.csv`, `r0_cart_predictions.csv`, `r0_cart_depth_sweep.csv`, `r0_policy_evaluation.csv`, `r0_freeze_manifest.json`, `r0_final_report.md`, `r0_cart_model.joblib`.
