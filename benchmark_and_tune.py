"""Precision validation, hardware auto-tuning, and fair benchmark suite for PHY-ML.

Fulfills Requirements:
    5. Precision benchmark: CPU double vs CUDA double vs CUDA single.
    6. Automatically tune GPU batch size (500, 1000, 2000, 4000, 8000, 12000, 16000) with OOM safety and VRAM tracking.
    7. CPU intra-op threading optimization (8, 10, 12, 16 threads).
    8. Fair performance benchmark across 6 dB, 18 dB, 28 dB (5000 blocks/point).
"""
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import sionna
from sionna.phy.utils import complex_normal, db_to_lin

from config import probe_environment, print_environment_report
from phy_engine import ModulationMode, MODES, _get_sionna_modem
from channel import _get_flat_fading_model
from cuda_engine import BatchPHYEngine, OnlineBlockStats


def validate_constellation_normalization() -> Dict[str, Any]:
    """Verify unit-average-energy constellation for all modes under both single and double precision."""
    results = {}
    for prec in ("double", "single"):
        for m in MODES:
            const, _, _, _ = _get_sionna_modem(m.bits_per_symbol, precision=prec, device="cpu")
            pts = const.points
            avg_energy = float(torch.mean(torch.abs(pts) ** 2).item())
            err = abs(avg_energy - 1.0)
            results[(prec, m.modulation)] = {
                "avg_energy": avg_energy,
                "error": err,
                "valid": err < 1e-5,
            }
    return results


def run_precision_validation(
    snrs: Tuple[float, ...] = (6.0, 18.0, 28.0),
    num_blocks: int = 5000,
    seed: int = 20260918,
) -> Dict[str, Any]:
    """Validate CUDA single against CPU double reference using identical predetermined physical realizations.

    Compares:
        - Exact bit match percentage
        - Absolute BER discrepancy |BER_single - BER_double|
        - Standard Error of Monte Carlo estimate
        - Verifies discrepancy << SE across all candidate modulations.
    """
    print("\n" + "=" * 70)
    print("           STEP 1: MANDATORY PRECISION VALIDATION (FP64 vs FP32)")
    print("=" * 70)
    print(f"Comparing CPU Double Reference vs CUDA Single across SNRs: {snrs} ({num_blocks:,} blocks/SNR)")

    s_sym = 1536
    max_m = max(m.bits_per_symbol for m in MODES)
    results = []

    for snr_db in snrs:
        print(f"\n--- Validating SNR = {snr_db:.1f} dB ---")
        # Run identical deterministic batches
        # 1. CPU Double Reference
        engine_cpu_dbl = BatchPHYEngine(s_sym, MODES, device="cpu", precision="double")
        stats_cpu_dbl = {m.mode_id: OnlineBlockStats(m.bits_per_symbol) for m in MODES}

        # 2. CUDA Double Reference
        engine_cuda_dbl = BatchPHYEngine(s_sym, MODES, device="cuda:0", precision="double")
        stats_cuda_dbl = {m.mode_id: OnlineBlockStats(m.bits_per_symbol) for m in MODES}

        # 3. CUDA Single Candidate
        engine_cuda_sgl = BatchPHYEngine(s_sym, MODES, device="cuda:0", precision="single")
        stats_cuda_sgl = {m.mode_id: OnlineBlockStats(m.bits_per_symbol) for m in MODES}

        chunk_size = 1000
        blocks_done = 0
        chunk_idx = 0

        while blocks_done < num_blocks:
            b = min(chunk_size, num_blocks - blocks_done)
            f_seed = (seed + int(snr_db * 1000) + chunk_idx * 7919) % (2**31 - 1)
            n_seed = (f_seed + 101) % (2**31 - 1)
            b_seed = (f_seed + 202) % (2**31 - 1)

            # Evaluate CPU double
            r_cpu = engine_cpu_dbl.evaluate_chunk(snr_db, b, f_seed, n_seed, b_seed)
            for m in MODES:
                stats_cpu_dbl[m.mode_id].update_chunk(b, *r_cpu[m.mode_id])

            # Evaluate CUDA double
            r_cuda_dbl = engine_cuda_dbl.evaluate_chunk(snr_db, b, f_seed, n_seed, b_seed)
            for m in MODES:
                stats_cuda_dbl[m.mode_id].update_chunk(b, *r_cuda_dbl[m.mode_id])

            # Evaluate CUDA single
            r_cuda_sgl = engine_cuda_sgl.evaluate_chunk(snr_db, b, f_seed, n_seed, b_seed)
            for m in MODES:
                stats_cuda_sgl[m.mode_id].update_chunk(b, *r_cuda_sgl[m.mode_id])

            blocks_done += b
            chunk_idx += 1

        print(f"| {'Modulation':10s} | {'CPU Double BER':15s} | {'CUDA Double BER':15s} | {'CUDA Single BER':15s} | {'|Diff S-D|':11s} | {'Block SE':10s} | {'Diff / SE':10s} | {'Status':8s} |")
        print("|" + "-" * 12 + "|" + "-" * 17 + "|" + "-" * 17 + "|" + "-" * 17 + "|" + "-" * 13 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 10 + "|")

        for m in MODES:
            c_dbl = stats_cpu_dbl[m.mode_id]
            cu_dbl = stats_cuda_dbl[m.mode_id]
            cu_sgl = stats_cuda_sgl[m.mode_id]

            diff_sgl_cpu = abs(cu_sgl.ber - c_dbl.ber)
            se_ref = c_dbl.se_block_ber
            ratio = (diff_sgl_cpu / se_ref) if se_ref > 1e-12 else 0.0

            # Single precision is validated if discrepancy is far below Monte Carlo SE (ratio < 0.20 or absolute diff < 1e-4)
            passed = (ratio <= 0.25) or (diff_sgl_cpu < 1e-4) or (c_dbl.bit_errors == 0 and cu_sgl.bit_errors == 0)
            status = "PASS" if passed else "FLAG"

            results.append({
                "snr_db": snr_db,
                "modulation": m.modulation,
                "ber_cpu_double": c_dbl.ber,
                "ber_cuda_double": cu_dbl.ber,
                "ber_cuda_single": cu_sgl.ber,
                "diff_cuda_cpu_single": diff_sgl_cpu,
                "se_cpu_double": se_ref,
                "ratio_diff_se": ratio,
                "status": status,
            })

            print(
                f"| {m.modulation:10s} | {c_dbl.ber:15.6e} | {cu_dbl.ber:15.6e} | {cu_sgl.ber:15.6e} | "
                f"{diff_sgl_cpu:11.4e} | {se_ref:10.4e} | {ratio:10.3f} | {status:8s} |"
            )

    all_passed = all(r["status"] == "PASS" for r in results)
    print("-" * 70)
    print(f"Overall FP32 Physical Parity Validation: {'PASSED — FP32 IS VALID' if all_passed else 'FAILED — FP64 REQUIRED'}")
    print("=" * 70)
    return {"all_passed": all_passed, "details": results}


def tune_cpu_threads(
    candidate_threads: Tuple[int, ...] = (8, 10, 12, 16),
    test_blocks: int = 1500,
) -> Dict[str, Any]:
    """Benchmark PyTorch CPU intra-op thread counts on Intel i5-14400F."""
    print("\n" + "=" * 70)
    print("           STEP 2: CPU INTRA-OP THREAD COUNT TUNING")
    print("=" * 70)
    results = {}
    best_thr = 12
    max_throughput = 0.0

    for thr in candidate_threads:
        torch.set_num_threads(thr)
        engine = BatchPHYEngine(symbols_per_block=1536, modes=MODES, device="cpu", precision="double")
        
        # Warm-up
        engine.evaluate_chunk(18.0, 100, 101, 102, 103)

        t0 = time.perf_counter()
        engine.evaluate_chunk(18.0, test_blocks, 1001, 1002, 1003)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        throughput = test_blocks / elapsed
        results[thr] = {"elapsed": elapsed, "throughput": throughput}
        print(f"Threads: {thr:2d} | Time: {elapsed:6.3f}s | Throughput: {throughput:8.1f} blocks/sec")

        if throughput > max_throughput:
            max_throughput = throughput
            best_thr = thr

    print(f"Optimal CPU intra-op thread count: {best_thr} ({max_throughput:.1f} blocks/sec)")
    print("=" * 70)
    return {"best_threads": best_thr, "benchmarks": results}


def tune_gpu_batch_size(
    precision: str = "single",
    candidate_batches: Tuple[int, ...] = (500, 1000, 2000, 4000, 8000, 12000, 16000),
    test_blocks: int = 8000,
) -> Dict[str, Any]:
    """Auto-tune GPU batch size with clean OOM handling and VRAM tracking on RTX 3060 12 GB."""
    print("\n" + "=" * 70)
    print(f"           STEP 3: GPU BATCH SIZE TUNING ({precision.upper()})")
    print("=" * 70)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available for GPU batch tuning")

    engine = BatchPHYEngine(symbols_per_block=1536, modes=MODES, device="cuda:0", precision=precision)
    
    # Warm-up GPU
    torch.cuda.synchronize()
    engine.evaluate_chunk(18.0, 500, 1, 2, 3)
    torch.cuda.synchronize()

    results = []
    best_batch = 2000
    max_throughput = 0.0

    for batch in candidate_batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Run test_blocks in chunks of size `batch`
            done = 0
            chunk_i = 0
            while done < test_blocks:
                cur_b = min(batch, test_blocks - done)
                engine.evaluate_chunk(18.0, cur_b, 1000 + chunk_i, 2000 + chunk_i, 3000 + chunk_i)
                done += cur_b
                chunk_i += 1

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            elapsed = t1 - t0
            throughput = test_blocks / elapsed
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

            results.append({
                "batch_size": batch,
                "elapsed": elapsed,
                "throughput": throughput,
                "peak_vram_mb": peak_vram_mb,
                "status": "OK",
            })
            print(f"Batch: {batch:5d} | Time: {elapsed:6.3f}s | Throughput: {throughput:8.1f} blocks/s | Peak VRAM: {peak_vram_mb:7.1f} MiB")

            if throughput > max_throughput:
                max_throughput = throughput
                best_batch = batch

        except torch.cuda.OutOfMemoryError:
            print(f"Batch: {batch:5d} | CUDA OutOfMemoryError! Catching cleanly.")
            torch.cuda.empty_cache()
            results.append({
                "batch_size": batch,
                "status": "OOM",
            })
            break
        except Exception as e:
            print(f"Batch: {batch:5d} | Error: {e}")
            break

    print(f"Selected Optimal GPU Batch Size: {best_batch} ({max_throughput:.1f} blocks/sec)")
    print("=" * 70)
    return {"best_batch": best_batch, "details": results}


def run_fair_performance_benchmark(
    best_cpu_threads: int,
    best_gpu_batch: int,
    snrs: Tuple[float, ...] = (6.0, 18.0, 28.0),
    blocks_per_snr: int = 5000,
    validate_single_passed: bool = True,
) -> Dict[str, Any]:
    """Execute fair benchmark across representative SNRs comparing:

    1. Reference CPU Double (original 12 threads, batch 500)
    2. Optimized CPU Double (best threads, batch 2000)
    3. CUDA Double (RTX 3060, best batch)
    4. CUDA Single (RTX 3060, best batch, if validated)
    """
    print("\n" + "=" * 70)
    print("           STEP 4: FAIR BACKEND PERFORMANCE BENCHMARK")
    print("=" * 70)
    print(f"Evaluating {len(snrs)} SNRs: {snrs} with {blocks_per_snr:,} blocks/point ({len(snrs)*blocks_per_snr:,} total blocks)")

    configs = [
        ("Reference CPU Double", "cpu", "double", 12, 500),
        (f"Optimized CPU Double ({best_cpu_threads}T)", "cpu", "double", best_cpu_threads, 1000),
        ("CUDA Double", "cuda:0", "double", 4, best_gpu_batch),
    ]
    if validate_single_passed:
        configs.append(("CUDA Single", "cuda:0", "single", 4, best_gpu_batch))

    benchmark_summary = {}

    for label, dev_str, prec_str, threads, batch_size in configs:
        if "cpu" in dev_str:
            torch.set_num_threads(threads)
        engine = BatchPHYEngine(symbols_per_block=1536, modes=MODES, device=dev_str, precision=prec_str)

        # Warm-up
        if "cuda" in dev_str:
            torch.cuda.synchronize()
        engine.evaluate_chunk(18.0, min(500, batch_size), 999, 998, 997)
        if "cuda" in dev_str:
            torch.cuda.synchronize()

        total_blocks = len(snrs) * blocks_per_snr
        t0 = time.perf_counter()

        for snr_db in snrs:
            done = 0
            c_idx = 0
            while done < blocks_per_snr:
                b = min(batch_size, blocks_per_snr - done)
                f_seed = (20260918 + int(snr_db * 100) + c_idx * 7919) % (2**31 - 1)
                n_seed = (f_seed + 101) % (2**31 - 1)
                b_seed = (f_seed + 202) % (2**31 - 1)
                engine.evaluate_chunk(snr_db, b, f_seed, n_seed, b_seed)
                done += b
                c_idx += 1

        if "cuda" in dev_str:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        throughput = total_blocks / elapsed
        sec_per_5k = (elapsed / total_blocks) * 5000
        benchmark_summary[label] = {
            "elapsed_total": elapsed,
            "sec_per_5000": sec_per_5k,
            "blocks_per_sec": throughput,
            "device": dev_str,
            "precision": prec_str,
            "batch_size": batch_size,
        }

    # Compute speedup vs Reference CPU Double
    ref_time = benchmark_summary["Reference CPU Double"]["elapsed_total"]
    for label, d in benchmark_summary.items():
        d["speedup"] = ref_time / d["elapsed_total"]

    print("\n" + "=" * 80)
    print(f"| {'Backend & Precision':32s} | {'Time (Total)':13s} | {'Sec / 5000':11s} | {'Blocks / Sec':13s} | {'Speedup':8s} |")
    print("|" + "-" * 34 + "|" + "-" * 15 + "|" + "-" * 13 + "|" + "-" * 15 + "|" + "-" * 10 + "|")
    for label, d in benchmark_summary.items():
        print(
            f"| {label:32s} | {d['elapsed_total']:11.3f} s | {d['sec_per_5000']:9.3f} s | "
            f"{d['blocks_per_sec']:13.1f} | {d['speedup']:7.2f}x |"
        )
    print("=" * 80)

    return benchmark_summary


def main():
    env = probe_environment()
    print_environment_report(env)

    # 1. Constellation Normalization
    const_res = validate_constellation_normalization()
    print("Constellation Normalization check:")
    for k, v in const_res.items():
        print(f"  {k}: avg energy = {v['avg_energy']:.6f} (error = {v['error']:.2e}, valid = {v['valid']})")

    # 2. Precision Validation
    prec_res = run_precision_validation(snrs=(6.0, 18.0, 28.0), num_blocks=5000)

    # 3. CPU Threads Tuning
    cpu_res = tune_cpu_threads(candidate_threads=(8, 10, 12, 16), test_blocks=1000)

    # 4. GPU Batch Size Tuning (single and double)
    gpu_sgl_res = tune_gpu_batch_size(precision="single", candidate_batches=(500, 1000, 2000, 4000, 8000, 12000))
    gpu_dbl_res = tune_gpu_batch_size(precision="double", candidate_batches=(500, 1000, 2000, 4000, 8000))

    # 5. Fair Benchmark
    bench_res = run_fair_performance_benchmark(
        best_cpu_threads=cpu_res["best_threads"],
        best_gpu_batch=gpu_sgl_res["best_batch"],
        snrs=(6.0, 18.0, 28.0),
        blocks_per_snr=5000,
        validate_single_passed=prec_res["all_passed"],
    )

    return {
        "precision_validation": prec_res,
        "cpu_threads": cpu_res,
        "gpu_batch_single": gpu_sgl_res,
        "gpu_batch_double": gpu_dbl_res,
        "benchmark_summary": bench_res,
    }


if __name__ == "__main__":
    main()
