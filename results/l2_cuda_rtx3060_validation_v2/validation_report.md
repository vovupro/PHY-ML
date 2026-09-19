# L2 CUDA Validation & Benchmark Audit Report (v2)

> [!IMPORTANT]
> **Audit Notice & Boundary Invariance:**
> - Historical L2 FP64 calibration was **NOT** rerun.
> - Historical calibration artifacts in `results/l2_cuda_rtx3060/` remain strictly **unchanged** and preserved.
> - The artifacts in this directory (`results/l2_cuda_rtx3060_validation_v2/`) represent **lightweight validation and benchmark results only**.

---

## 1. Audit Issue Resolutions Summary

| Issue | Status | Applied Solution |
|:------|:------:|:-----------------|
| **1. Uncertainty Semantics** | **CLOSED** | Formalized `reliability_uncertain` as strictly `ci_low <= BER_target <= ci_high` ($k=1.96$). Removed heuristic windows `[0.009, 0.011]` and `[0.008, 0.012]`. Unified via shared helper `metrics.is_reliability_uncertain`. |
| **2. Reporting / Speedup** | **CLOSED** | Replaced 'Peak Throughput' label with 'Overall Average Throughput' ($total\_blocks / total\_mc\_time$). Removed hard-coded `52.0 blocks/s` reference; speedups only computed when backed by persisted benchmark. |
| **3. Benchmark Batch Bug** | **CLOSED** | Decoupled CUDA Double and CUDA Single batch configurations in benchmark suite. CUDA FP64 uses tuned FP64 batch; CUDA FP32 uses tuned FP32 batch. |
| **4. True Exact-Realization Parity** | **CLOSED** | Implemented `PHYRealization` and `evaluate_realization`. Physical realization is generated once on CPU in FP64 and reused identically across CPU FP64, CUDA FP64, and CUDA FP32. Exact bit parity: CUDA FP64 is **VERIFIED** (0 bit discrepancies); CUDA FP32 is **VERIFIED** (0 bit discrepancies on test set; FP64 remains standard for bit-exact invariance). |
| **5. Persisted Evidence** | **CLOSED** | Persisted `benchmark.json`, `parity.json`, and `validation_report.md` exclusively in isolated directory `results/l2_cuda_rtx3060_validation_v2/`. |
| **6. CUDA Hot Path** | **CLOSED** | Refactored `evaluate_chunk` to aggregate 16 per-mode `.item()` host synchronizations into a single device-to-host transfer. Exact bitwise numerical equivalence preserved ($diff = 0$). Throughput: 3084.6 blk/s (before) vs 3076.1 blk/s (after). |

---

## 2. True Exact-Realization Physical Parity

- **Methodology:** Physical channel realization (transmitted bits, Rayleigh fading $h$, AWGN noise $z$) generated once on host CPU in FP64 and evaluated identically.
- **CUDA FP64 vs. CPU FP64 Parity:** `VERIFIED` (0 bit discrepancies across all evaluated modulations and SNRs).
- **CUDA FP32 vs. CPU FP64 Parity:** `VERIFIED`.

| SNR (dB) | Modulation | Total Bits | CPU FP64 Errs | CUDA FP64 Errs | CUDA FP32 Errs | FP64 Discrepancies | FP32 Discrepancies | FP64 Match | FP32 Match |
|:--------:|:----------:|:----------:|:-------------:|:--------------:|:--------------:|:------------------:|:------------------:|:----------:|:----------:|
|      6.0 | BPSK       |  1,536,000 |        74,545 |         74,545 |         74,545 |                  0 |                  0 | 100.000000% | 100.000000% |
|      6.0 | QPSK       |  3,072,000 |       266,001 |        266,001 |        266,001 |                  0 |                  0 | 100.000000% | 100.000000% |
|      6.0 | 16QAM      |  6,144,000 |     1,173,553 |      1,173,553 |      1,173,553 |                  0 |                  0 | 100.000000% | 100.000000% |
|      6.0 | 64QAM      |  9,216,000 |     2,524,044 |      2,524,044 |      2,524,044 |                  0 |                  0 | 100.000000% | 100.000000% |
|     18.0 | BPSK       |  1,536,000 |         8,693 |          8,693 |          8,693 |                  0 |                  0 | 100.000000% | 100.000000% |
|     18.0 | QPSK       |  3,072,000 |        31,903 |         31,903 |         31,903 |                  0 |                  0 | 100.000000% | 100.000000% |
|     18.0 | 16QAM      |  6,144,000 |       198,923 |        198,923 |        198,923 |                  0 |                  0 | 100.000000% | 100.000000% |
|     18.0 | 64QAM      |  9,216,000 |       726,453 |        726,453 |        726,453 |                  0 |                  0 | 100.000000% | 100.000000% |
|     28.0 | BPSK       |  1,536,000 |           611 |            611 |            611 |                  0 |                  0 | 100.000000% | 100.000000% |
|     28.0 | QPSK       |  3,072,000 |         2,505 |          2,505 |          2,505 |                  0 |                  0 | 100.000000% | 100.000000% |
|     28.0 | 16QAM      |  6,144,000 |        17,920 |         17,920 |         17,920 |                  0 |                  0 | 100.000000% | 100.000000% |
|     28.0 | 64QAM      |  9,216,000 |        91,628 |         91,628 |         91,628 |                  0 |                  0 | 100.000000% | 100.000000% |

---

## 3. Fair Benchmark Results (Independent Batches)

- **Hardware Environment:** Intel Core i5-14400F (16 threads) | NVIDIA GeForce RTX 3060 (12.00 GB (12288 MiB))
- **PyTorch / CUDA / Sionna:** PyTorch `2.6.0+cu124` | CUDA Runtime `12.4` | Sionna `2.0.0`
- **Tuned Batches:** CUDA FP64 Batch = `500`, CUDA FP32 Batch = `500`

| Backend & Precision | Device | Precision | Batch Size | Total Time (s) | Sec / 5000 Blks | Throughput (blks/s) | Measured Speedup |
|:--------------------|:------:|:---------:|:----------:|:--------------:|:---------------:|:-------------------:|:----------------:|
| Reference CPU Double | CPU | FP64 | 500 | 51.531 | 17.177 | 291.1 | 1.00x (1.00x ref) |
| Optimized CPU Double (16T) | CPU | FP64 | 1000 | 375.273 | 125.091 | 40.0 | 0.14x |
| CUDA Double | CUDA | FP64 | 500 | 81.740 | 27.247 | 183.5 | 0.63x |
| CUDA Single | CUDA | FP32 | 500 | 21.919 | 7.306 | 684.3 | 2.35x |

---

## 4. CUDA Hot Path Sync Aggregation

- **Inspection:** `cuda_engine.py` previously performed 4 individual `.item()` calls per modulation mode (16 device-to-host synchronizations per chunk batch).
- **Optimization:** Mode metric tensors (`total_errs`, `total_blk_errs`, `mean_ber`, `m2_ber`) are stacked into `(num_modes, 4)` on device and retrieved via a single device-to-host transfer.
- **Numerical Equivalence:** 100% bit-exact numerical match across all modes (`err_diff = 0`, `blk_diff = 0`, `mean_diff = 0.00e+00`, `m2_diff = 0.00e+00`).
- **Benchmark:** 3084.6 blocks/sec (before, 16 syncs) vs 3076.1 blocks/sec (after, 1 sync). Computation and demapping dominate chunk runtime.
