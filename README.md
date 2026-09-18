# PHY-ML: Minimal Verified Uncoded PHY Research Core

A transparent, reproducible, and mathematically verified physical layer (PHY) foundation for wireless communication and link adaptation studies.

---

## 📌 Current Stage

**Verified Uncoded PHY Research Core** (Clean baseline prior to L1 Monte Carlo calibration and ML link adaptation).

---

## ✅ Implemented Features

- **Modulation Schemes**:
  - BPSK ($1\text{ bit/symbol}$)
  - QPSK ($2\text{ bits/symbol}$)
  - 16-QAM ($4\text{ bits/symbol}$)
  - 64-QAM ($6\text{ bits/symbol}$)
- **Constellation Properties**:
  - Exact unit average symbol energy normalization: $\mathbb{E}[|s|^2] = 1.0$.
  - Exact 2D Gray mapping: adjacent nearest-neighbour constellation points differ by exactly 1 bit.
- **Physical Channel Models**:
  - **AWGN**: Additive white Gaussian noise channel with unit gain ($h = 1.0 + 0.0j$).
  - **Slow Rayleigh Block Fading**: One complex scalar channel coefficient $h \sim \mathcal{CN}(0, 1)$ with $\mathbb{E}[|h|^2] = 1.0$, held strictly **constant across the entire transmission block** ($y = h \cdot x + n$). Independent realizations drawn between independent blocks.
  - Future-proofed 2D conditioning interface ($h\_magnitude$ parameter) for subsequent fading envelope sweeps.
- **Receiver Architecture**:
  - Perfect coherent channel state information (CSI) baseline.
  - Unconditionally stable coherent maximum likelihood (ML) hard demodulation: $\arg\min_{s \in \mathcal{C}} |y - h \cdot s|^2$.
- **SNR Convention**:
  - Explicitly defined as nominal/setup $E_s / N_0$ in dB ($snr\_db$).
  - With unit average constellation energy $E_s = 1.0$, the complex noise variance is $N_0 = 10^{-snr\_db / 10}$.
- **RNG Handling & Paired Evaluations**:
  - Fully deterministic and reproducible using explicit NumPy `SeedSequence` and child `Generator` streams (`bit_rng`, `fading_rng`, `noise_rng`).
  - Paired physical evaluations across candidate modulation modes share identical $h$ and noise realizations, strictly invariant to evaluation order.
- **Raw PHY Metrics**:
  - Transparent error counters: bit error count, total bits, BER, block error count, total blocks, BLER, nominal throughput/goodput.
- **Physics Acceptance Suite**:
  - 14 automated tests validating constellation energy, Gray consistency, noiseless round-trip, coherent demodulation, exact analytical AWGN BER ($Q$-function/erfc and exact PAM decision-region integration), analytical Rayleigh average BER, $h$ constancy, independence, and deterministic reproducibility.

---

## 🚫 Explicitly Not Implemented Yet (Deferred to Later Levels)

This minimal core deliberately excludes:
- AMC policy selection & Monte Carlo calibration
- Lookup Table (LUT) baselines
- Decision Tree (CART) & Random Forest classifiers
- 2D feature study ($|h|$ vs $E_s/N_0$)
- Pilot transmission & practical channel/SNR estimation
- Channel coding (LDPC, Polar, Convolutional, Turbo)
- CRC verification, ACK/NACK feedback, and HARQ protocols
- FPGA export and fixed-point quantization

---

## 📂 Active Repository Structure

```text
PHY-ML/
├── .gitignore          # Clean Git ignore definitions
├── README.md           # Project overview and scope
├── __init__.py         # Package root exposing public PHY API
├── requirements.txt    # Minimal scientific Python dependencies
├── channel.py          # Slow Rayleigh block fading and AWGN channel model
├── phy_engine.py       # Constellations, modulation, and coherent demodulation
├── metrics.py          # Raw PHY counters and error rate calculations
└── test_physics.py     # 14-point physical layer acceptance test gate
```

---

## 🚀 Quick Start

### 1. Installation

Install minimal dependencies (NumPy, SciPy, Scikit-Learn, Joblib):
```bash
pip install -r requirements.txt
```

### 2. Run Acceptance Test Suite

Execute the physical validation gate:
```bash
python -m unittest -v test_physics.py
```

All 14 tests must pass:
```text
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
- `MODES`: Tuple of supported uncoded modes (`BPSK`, `QPSK`, `16QAM`, `64QAM`)
- `constellation(bits_per_symbol: int) -> tuple[np.ndarray, np.ndarray]`
- `modulate(bits: np.ndarray, bits_per_symbol: int) -> np.ndarray`
- `demodulate(received_symbols: np.ndarray, h: complex, bits_per_symbol: int) -> np.ndarray`
- `PHYEngine(block_symbols: int)`: Core evaluation engine (`evaluate_mode`, `evaluate_paired_block`)

### `metrics.py`
- `RawPHYCounters`: Stateful accumulator for bits, blocks, BER, and BLER
- `compute_ber(bit_errors, total_bits) -> float`
- `compute_bler(block_errors, total_blocks) -> float`
- `compute_raw_goodput(bler, bits_per_symbol) -> float`
