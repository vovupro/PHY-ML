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

## 📌 Current Stage

**L3 Ground Truth & Baseline Policies Completed**
Workflow:
`L1 Verified PHY Core` $\to$ `L2 1D Monte Carlo Calibration (5,000 blocks / point)` $\to$ `L3 Ground Truth (BestMode) & Baselines (Fixed Robust, Fixed High-Throughput, 1D LUT)`.

---

## ⚙️ Verified Computational Runtime

### Computational Platforms
- **Historical Frozen Reference**:
  - PyTorch: `2.6.0+cu124`
  - CUDA Runtime: `12.4`
- **Current Qualified CKEY Runtime**:
  - Python: `3.13.5`
  - PyTorch: `2.9.1+cu126`
  - CUDA Runtime: `12.6`
  - Sionna: `2.0.0`
  - Hardware: NVIDIA RTX 3060 (`sm_86`)

### Reproduced Scientific Invariance
Full reproduction across the computational chain demonstrates 100% scientific invariance:
- **L2 Calibration**: 1,140,000 blocks simulated across 25 SNR points
- **L3 LUT Thresholds**: 16.75, 22.75, 28.25 dB
- **L4 CART Decision Tree**:
  - Depth 1 = 64%
  - Depth 2 = 84%
  - Depth 3 = 100%
  - Selected depth = 3 (7 nodes / 4 leaves)
  - Switching thresholds = 16.75, 22.75, 28.25 dB

### Artifact Taxonomy
- **Historical Frozen Baselines**:
  - `results/l2_cuda_rtx3060_final`
  - `results/l3_final`
  - `results/l4_final`
- **Qualified Reproduced CKEY Chain**:
  - `results/l2_ckey_torch291_cu126`
  - `results/l3_ckey_torch291_cu126`
  - `results/l4_ckey_torch291_cu126`
  *(Note: Versioned CKEY outputs are generated on the compute node and managed as runtime reproduction artifacts).*

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

## 🚫 Explicitly Not Implemented Yet (Deferred to Later Levels)

This minimal core deliberately excludes:
- Channel coding (LDPC, Polar, Convolutional, Turbo) — *LDPC is not implemented yet*
- AMC policy selection & Monte Carlo calibration
- Lookup Table (LUT) baselines
- Decision Tree (CART) & Random Forest classifiers
- 2D feature study ($|h|$ vs $E_s/N_0$)
- Pilot transmission & practical channel/SNR estimation
- CRC verification, ACK/NACK feedback, and HARQ protocols
- FPGA export and fixed-point quantization

---

## 📂 Active Repository Structure

```text
PHY-ML/
├── .gitignore          # Clean Git ignore definitions
├── README.md           # Project overview and scope
├── __init__.py         # Package root exposing public PHY API
├── requirements.txt    # Minimal scientific Python dependencies + sionna-no-rt
├── channel.py          # Slow Rayleigh block fading and AWGN channel model
├── phy_engine.py       # Sionna 2.0-backed modulation, equalization, and demapping
├── metrics.py          # Raw PHY counters and error rate calculations
└── test_physics.py     # 15-point physical layer acceptance test gate
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

### `calibration_1d.py` (L2)
- `Calibration1DConfig`: Configuration parameters (SNR grid, num_blocks=5000, batch_blocks=500, symbols_per_block=1536, master_seed).
- `run_1d_calibration(config) -> List[CalibrationRecord]`: Vectorized chunk-batching Monte Carlo calibration preserving paired realizations.
- `save_calibration_csv(records, output_path)`: Exports raw counts, block variance, and SE to CSV.
- `generate_verification_report(records, config, records_seed_b, ...) -> str`: Rigorous telecom report comparing empirical curves against independent SciPy analytical theory using block standard errors.

### `ground_truth.py` (L3)
- `GroundTruthConfig`: Explicit parameters (`ber_target=0.01`, `fallback_policy="robustest_mode"`, `confidence_k=1.96`).
- `compute_ground_truth(calibration_data, config) -> List[GroundTruthRow]`: Synthesizes BestMode strictly from calibration tables without re-running PHY.
- `FixedRobustPolicy`: Baseline unconditionally selecting BPSK (1 bpcu).
- `FixedHighThroughputPolicy`: Baseline unconditionally selecting 64-QAM (6 bpcu).
- `LookupTable1D`: Derived 1D LUT with thresholds at ground-truth mode transition midpoints.
- `generate_l3_report(rows, lut, config, ...) -> str`: Generates telecom verification report documenting switching regions, thresholds, boundary ambiguity, and fallback events.
