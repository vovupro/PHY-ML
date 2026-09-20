"""L2: Fixed-Budget Monte Carlo Calibration Engine using Vectorized CUDA Hot-Path.

Features:
    - Fixed-budget Monte Carlo calibration across 25-point canonical SNR grid [0, 30] dB.
    - NO ADAPTIVE STOPPING: every requested operating point runs strictly to the configured budget.
    - Two canonical fixed-budget profiles:
        * 'light' = 10,000 blocks/seed (20,000 blocks pooled per SNR)
        * 'deep'  = 70,000 blocks/seed (140,000 blocks pooled per SNR)
    - Arbitrary budget overrides supported (e.g. --blocks-per-seed 100000, 200000).
    - Deterministic output isolation:
        * 'light' -> results/r0_mc_light_10k/
        * 'deep'  -> results/r0_mc_deep_70k/
        * custom  -> results/r0_mc_custom_{N}k/
    - Checkpoint convergence trajectory recorded every 5,000 blocks/seed:
      BER, SE, 95% CI lower/upper, target overlap (BER_target = 0.0100).
      This trajectory is purely for human observational analysis; it never affects stopping.
    - True CUDA batch hot-path (RTX 3060, batch size = 500, FP64 double precision).
    - Independent Seed A (20260918) and Seed B (20260919) Monte Carlo streams.
    - Online Chan/Welford parallel variance accumulation across fading blocks.
"""
import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
from sionna.phy.utils import db_to_lin

from config import (
    ExecutionConfig,
    FIXED_MC_PROFILES,
    resolve_results_dir,
    probe_environment,
    print_environment_report,
)
from phy_engine import ModulationMode, MODES
from cuda_engine import BatchPHYEngine, OnlineBlockStats
from calibration_1d import theoretical_bpsk_rayleigh
from metrics import is_reliability_uncertain, BER_TARGET, CONFIDENCE_K


GRID_25_POINTS: Tuple[float, ...] = (
    0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0,
    16.0, 16.5, 17.0, 17.5, 18.0, 20.0,
    22.0, 22.5, 23.0, 23.5, 24.0, 26.0,
    28.0, 28.5, 29.0, 29.5, 30.0,
)


@dataclass
class ConvergenceCheckpointState:
    """State of an SNR operating point at an observational convergence checkpoint.

    Note: Checkpoints are recorded strictly for human convergence inspection.
    They do not affect fixed-budget simulation stopping.
    """
    checkpoint_idx: int
    blocks_per_seed: int
    total_blocks_pooled: int
    mod_stats_a: Dict[int, OnlineBlockStats]
    mod_stats_b: Dict[int, OnlineBlockStats]
    pooled_ber: Dict[int, float]
    pooled_se: Dict[int, float]
    z_ab: Dict[int, float]
    ci_low: Optional[Dict[int, float]] = None
    ci_high: Optional[Dict[int, float]] = None
    overlaps_target: Optional[Dict[int, bool]] = None
    crit_seed: Optional[Dict[int, bool]] = None
    crit_prec: Optional[Dict[int, bool]] = None
    crit_move: Optional[Dict[int, bool]] = None
    all_crit_passed: bool = False


# Alias for clean semantic naming
TrajectoryCheckpointRecord = ConvergenceCheckpointState


def pool_two_streams(
    stats_a: OnlineBlockStats,
    stats_b: OnlineBlockStats,
) -> Tuple[float, float, float]:
    """Pool two independent Monte Carlo streams using Chan's parallel variance algorithm.

    Returns
    -------
    ber_pooled : float (macro bit error rate)
    mean_ber_pooled : float (mean of block-level BER)
    se_ber_pooled : float (standard error of mean block-level BER)
    """
    n_a = stats_a.count
    n_b = stats_b.count
    n_pooled = n_a + n_b
    if n_pooled == 0:
        return 0.0, 0.0, 0.0

    tot_errs = stats_a.bit_errors + stats_b.bit_errors
    tot_bits = stats_a.total_bits + stats_b.total_bits
    ber_macro = float(tot_errs / tot_bits) if tot_bits > 0 else 0.0

    delta = stats_b.mean_ber - stats_a.mean_ber
    m2_pooled = stats_a.m2_ber + stats_b.m2_ber + (delta ** 2) * (n_a * n_b / n_pooled)
    var_pooled = (m2_pooled / (n_pooled - 1)) if n_pooled > 1 else 0.0
    std_pooled = math.sqrt(max(0.0, var_pooled))
    se_pooled = std_pooled / math.sqrt(n_pooled)

    mean_pooled = stats_a.mean_ber + delta * (n_b / n_pooled)
    return ber_macro, mean_pooled, se_pooled


def run_fresh_l2_cuda_calibration(
    config: Optional[ExecutionConfig] = None,
    profile: Optional[str] = None,
    blocks_per_seed: Optional[int] = None,
    snrs: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Execute complete fixed-budget L2 calibration on NVIDIA RTX 3060 (NO ADAPTIVE STOPPING)."""
    if config is None:
        p_name = profile if profile in FIXED_MC_PROFILES else "deep"
        b_count = blocks_per_seed if blocks_per_seed is not None else FIXED_MC_PROFILES[p_name]
        out_dir = resolve_results_dir(profile=p_name if blocks_per_seed is None else None, blocks_per_seed=b_count)
        config = ExecutionConfig(
            backend="cuda",
            precision="double",
            batch_blocks=500,
            results_dir=out_dir,
            blocks_per_seed=b_count,
            profile=p_name if blocks_per_seed is None else None,
        )
    else:
        if blocks_per_seed is not None:
            config.blocks_per_seed = blocks_per_seed
            config.profile = profile
            config.results_dir = resolve_results_dir(profile, blocks_per_seed, config.results_dir)
        elif profile is not None:
            config.profile = profile
            config.blocks_per_seed = FIXED_MC_PROFILES[profile]
            config.results_dir = resolve_results_dir(profile, config.blocks_per_seed, config.results_dir)
        else:
            config.results_dir = resolve_results_dir(config.profile, config.blocks_per_seed, config.results_dir)

    out_dir = Path(config.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = probe_environment(config)
    print_environment_report(env)

    device_str = env["selected_device"]
    prec_str = env["sionna_precision"]
    batch_size = config.batch_blocks
    snr_grid = GRID_25_POINTS if snrs is None else tuple(snrs)

    target_budget = config.blocks_per_seed
    step_size = config.checkpoint_step

    print("\n" + "=" * 75)
    print(f"   STARTING FIXED-BUDGET L2 RUN: {len(snr_grid)} SNR POINTS (NO ADAPTIVE STOPPING)")
    print(f"   Profile: {config.profile or 'custom'} | Budget: {target_budget:,} blocks/seed ({target_budget * 2:,} pooled)")
    print(f"   Device: {device_str} | Precision: {prec_str} | Batch Size: {batch_size}")
    print(f"   Seed A: {config.master_seed_a} | Seed B: {config.master_seed_b}")
    print(f"   Checkpoint Step: {step_size:,} blocks/seed (Human Inspection Trajectory Only)")
    print(f"   Output Directory: {config.results_dir}")
    print("=" * 75, flush=True)

    # 1. Warm-up GPU
    t_init_0 = time.perf_counter()
    engine = BatchPHYEngine(
        symbols_per_block=config.symbols_per_block,
        modes=MODES,
        device=device_str,
        precision=prec_str,
    )
    if "cuda" in device_str:
        torch.cuda.synchronize()
    engine.evaluate_chunk(18.0, min(500, batch_size), 1, 2, 3)
    if "cuda" in device_str:
        torch.cuda.synchronize()
    t_init_1 = time.perf_counter()
    warmup_time = t_init_1 - t_init_0
    print(f"Modem & GPU Warm-up complete in {warmup_time:.3f} s\n")

    snr_results = {}
    total_blocks_simulated = 0

    t_mc_start = time.perf_counter()

    for idx, snr_db in enumerate(snr_grid):
        print(f"\n>>> [{idx+1:2d}/{len(snr_grid):2d}] Simulating SNR = {snr_db:4.1f} dB "
              f"(Fixed Budget: {target_budget:,} blks/seed | {target_budget * 2:,} pooled) ...", flush=True)
        t_snr_0 = time.perf_counter()

        stats_a = {m.mode_id: OnlineBlockStats(m.bits_per_symbol) for m in MODES}
        stats_b = {m.mode_id: OnlineBlockStats(m.bits_per_symbol) for m in MODES}

        blocks_done_a = 0
        blocks_done_b = 0
        chunk_idx_a = 0
        chunk_idx_b = 0
        checkpoint_idx = 0

        checkpoint_history: List[ConvergenceCheckpointState] = []

        target_blocks = 0
        # STRICT FIXED BUDGET: Loop until target_blocks reaches target_budget
        while target_blocks < target_budget:
            checkpoint_idx += 1
            target_blocks = min(target_blocks + step_size, target_budget)

            # --- 1. Run Stream A until target_blocks ---
            while blocks_done_a < target_blocks:
                cur_b = min(batch_size, target_blocks - blocks_done_a)
                f_seed = (config.master_seed_a + int(snr_db * 1000) + chunk_idx_a * 7919) % (2**31 - 1)
                n_seed = (f_seed + 101) % (2**31 - 1)
                b_seed = (f_seed + 202) % (2**31 - 1)

                r_a = engine.evaluate_chunk(snr_db, cur_b, f_seed, n_seed, b_seed)
                for m in MODES:
                    stats_a[m.mode_id].update_chunk(cur_b, *r_a[m.mode_id])

                blocks_done_a += cur_b
                chunk_idx_a += 1

            # --- 2. Run Stream B until target_blocks ---
            while blocks_done_b < target_blocks:
                cur_b = min(batch_size, target_blocks - blocks_done_b)
                f_seed = (config.master_seed_b + int(snr_db * 1000) + chunk_idx_b * 7919) % (2**31 - 1)
                n_seed = (f_seed + 101) % (2**31 - 1)
                b_seed = (f_seed + 202) % (2**31 - 1)

                r_b = engine.evaluate_chunk(snr_db, cur_b, f_seed, n_seed, b_seed)
                for m in MODES:
                    stats_b[m.mode_id].update_chunk(cur_b, *r_b[m.mode_id])

                blocks_done_b += cur_b
                chunk_idx_b += 1

            if "cuda" in device_str:
                torch.cuda.synchronize()

            # --- 3. Evaluate Checkpoint Statistics (Human Inspection Trajectory Only) ---
            cur_pooled_ber = {}
            cur_pooled_se = {}
            cur_ci_low = {}
            cur_ci_high = {}
            cur_overlaps = {}
            z_ab_map = {}

            for m in MODES:
                sa = stats_a[m.mode_id]
                sb = stats_b[m.mode_id]
                ber_pool, mean_pool, se_pool = pool_two_streams(sa, sb)
                cur_pooled_ber[m.mode_id] = ber_pool
                cur_pooled_se[m.mode_id] = se_pool

                ci_hw = CONFIDENCE_K * se_pool
                cur_ci_low[m.mode_id] = max(0.0, float(ber_pool - ci_hw))
                cur_ci_high[m.mode_id] = float(ber_pool + ci_hw)
                cur_overlaps[m.mode_id] = is_reliability_uncertain(
                    ber_pool, se_pool, target=BER_TARGET, k=CONFIDENCE_K
                )

                comb_se = math.sqrt(sa.se_block_ber**2 + sb.se_block_ber**2)
                z_ab = (abs(sa.ber - sb.ber) / comb_se) if comb_se > 1e-12 else 0.0
                z_ab_map[m.mode_id] = z_ab

            chk_state = ConvergenceCheckpointState(
                checkpoint_idx=checkpoint_idx,
                blocks_per_seed=blocks_done_a,
                total_blocks_pooled=blocks_done_a + blocks_done_b,
                mod_stats_a={m.mode_id: stats_a[m.mode_id] for m in MODES},
                mod_stats_b={m.mode_id: stats_b[m.mode_id] for m in MODES},
                pooled_ber=cur_pooled_ber,
                pooled_se=cur_pooled_se,
                ci_low=cur_ci_low,
                ci_high=cur_ci_high,
                overlaps_target=cur_overlaps,
                z_ab=z_ab_map,
            )
            checkpoint_history.append(chk_state)

            ov_bpsk = "YES" if cur_overlaps[0] else "NO"
            ov_64qam = "YES" if cur_overlaps[3] else "NO"
            print(
                f"   [Chk #{checkpoint_idx:02d} | {blocks_done_a:6,d} blks/seed | {blocks_done_a + blocks_done_b:7,d} pooled] "
                f"BPSK: {cur_pooled_ber[0]:.4e} (CI: [{cur_ci_low[0]:.4e}, {cur_ci_high[0]:.4e}], ov={ov_bpsk}) | "
                f"64QAM: {cur_pooled_ber[3]:.4e} (CI: [{cur_ci_low[3]:.4e}, {cur_ci_high[3]:.4e}], ov={ov_64qam})",
                flush=True,
            )

        if "cuda" in device_str:
            torch.cuda.synchronize()
        t_snr_1 = time.perf_counter()
        snr_elapsed = t_snr_1 - t_snr_0
        total_blocks_simulated += (blocks_done_a + blocks_done_b)

        print(
            f"   Finished SNR = {snr_db:.1f} dB in {snr_elapsed:.2f}s | "
            f"Blocks: {blocks_done_a:,} x 2 = {blocks_done_a + blocks_done_b:,} | "
            f"Status: FIXED_BUDGET_REACHED (No early stopping)",
            flush=True,
        )

        snr_results[snr_db] = {
            "snr_db": snr_db,
            "requested_blocks_per_seed": target_budget,
            "actual_blocks_per_seed": blocks_done_a,
            "final_blocks_a": blocks_done_a,
            "final_blocks_b": blocks_done_b,
            "total_blocks_pooled": blocks_done_a + blocks_done_b,
            "elapsed_seconds": snr_elapsed,
            "fixed_budget_reached": True,
            "stable": None,  # Deprecated: fixed-budget methodology does not evaluate adaptive stability
            "ceiling_hit": False,
            "checkpoints_count": checkpoint_idx,
            "checkpoint_history": checkpoint_history,
            "final_stats_a": stats_a,
            "final_stats_b": stats_b,
            "final_pooled_ber": cur_pooled_ber,
            "final_pooled_se": cur_pooled_se,
            "final_ci_low": cur_ci_low,
            "final_ci_high": cur_ci_high,
            "final_overlaps_target": cur_overlaps,
            "final_z_ab": z_ab_map,
        }

    t_mc_end = time.perf_counter()
    total_mc_time = t_mc_end - t_mc_start

    # --- 4. Write CSV files ---
    t_io_0 = time.perf_counter()
    csv_a_path = out_dir / "calibration_1d_cuda_seed_a.csv"
    csv_b_path = out_dir / "calibration_1d_cuda_seed_b.csv"
    csv_pooled_path = out_dir / "calibration_1d_cuda_pooled.csv"

    fieldnames = [
        "mode_id", "modulation", "bits_per_symbol", "snr_db", "num_blocks",
        "total_bits", "bit_errors", "ber", "block_errors", "bler",
        "mean_block_ber", "std_block_ber", "se_block_ber"
    ]

    # Write Seed A
    with open(csv_a_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for snr_db in snr_grid:
            res = snr_results[snr_db]
            for m in MODES:
                sa = res["final_stats_a"][m.mode_id]
                writer.writerow({
                    "mode_id": m.mode_id,
                    "modulation": m.modulation,
                    "bits_per_symbol": m.bits_per_symbol,
                    "snr_db": snr_db,
                    "num_blocks": sa.count,
                    "total_bits": sa.total_bits,
                    "bit_errors": sa.bit_errors,
                    "ber": sa.ber,
                    "block_errors": sa.block_errors,
                    "bler": sa.bler,
                    "mean_block_ber": sa.mean_ber,
                    "std_block_ber": sa.std_block_ber,
                    "se_block_ber": sa.se_block_ber,
                })

    # Write Seed B
    with open(csv_b_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for snr_db in snr_grid:
            res = snr_results[snr_db]
            for m in MODES:
                sb = res["final_stats_b"][m.mode_id]
                writer.writerow({
                    "mode_id": m.mode_id,
                    "modulation": m.modulation,
                    "bits_per_symbol": m.bits_per_symbol,
                    "snr_db": snr_db,
                    "num_blocks": sb.count,
                    "total_bits": sb.total_bits,
                    "bit_errors": sb.bit_errors,
                    "ber": sb.ber,
                    "block_errors": sb.block_errors,
                    "bler": sb.bler,
                    "mean_block_ber": sb.mean_ber,
                    "std_block_ber": sb.std_block_ber,
                    "se_block_ber": sb.se_block_ber,
                })

    # Write Pooled
    fieldnames_pooled = fieldnames + ["z_ab", "fixed_budget_reached", "is_stable", "reliability_uncertain"]
    with open(csv_pooled_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_pooled)
        writer.writeheader()
        for snr_db in snr_grid:
            res = snr_results[snr_db]
            for m in MODES:
                sa = res["final_stats_a"][m.mode_id]
                sb = res["final_stats_b"][m.mode_id]
                ber_pool, mean_pool, se_pool = pool_two_streams(sa, sb)
                z = res["final_z_ab"][m.mode_id]
                tot_bits = sa.total_bits + sb.total_bits
                tot_errs = sa.bit_errors + sb.bit_errors
                tot_blks = sa.count + sb.count
                tot_blk_errs = sa.block_errors + sb.block_errors

                uncertain = is_reliability_uncertain(ber_pool, se_pool)

                writer.writerow({
                    "mode_id": m.mode_id,
                    "modulation": m.modulation,
                    "bits_per_symbol": m.bits_per_symbol,
                    "snr_db": snr_db,
                    "num_blocks": tot_blks,
                    "total_bits": tot_bits,
                    "bit_errors": tot_errs,
                    "ber": ber_pool,
                    "block_errors": tot_blk_errs,
                    "bler": (tot_blk_errs / tot_blks) if tot_blks > 0 else 0.0,
                    "mean_block_ber": mean_pool,
                    "std_block_ber": se_pool * math.sqrt(tot_blks),
                    "se_block_ber": se_pool,
                    "z_ab": z,
                    "fixed_budget_reached": True,
                    "is_stable": "N/A (deprecated)",
                    "reliability_uncertain": uncertain,
                })

    # Write Checkpoint Trajectory (observational human inspection)
    csv_trajectory_path = out_dir / "checkpoint_trajectory.csv"
    fieldnames_trajectory = [
        "profile",
        "requested_blocks_per_seed",
        "snr_db",
        "checkpoint_idx",
        "blocks_per_seed",
        "pooled_blocks",
        "mode_id",
        "modulation",
        "pooled_ber",
        "pooled_se",
        "ci95_low",
        "ci95_high",
        "overlaps_ber_target",
        "z_ab",
    ]
    with open(csv_trajectory_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_trajectory)
        writer.writeheader()
        profile_name = config.profile if config.profile else "custom"
        for snr_db in snr_grid:
            res = snr_results[snr_db]
            req_budget = res.get("requested_blocks_per_seed", target_budget)
            for chk in res["checkpoint_history"]:
                for m in MODES:
                    writer.writerow({
                        "profile": profile_name,
                        "requested_blocks_per_seed": req_budget,
                        "snr_db": snr_db,
                        "checkpoint_idx": chk.checkpoint_idx,
                        "blocks_per_seed": chk.blocks_per_seed,
                        "pooled_blocks": chk.total_blocks_pooled,
                        "mode_id": m.mode_id,
                        "modulation": m.modulation,
                        "pooled_ber": chk.pooled_ber[m.mode_id],
                        "pooled_se": chk.pooled_se[m.mode_id],
                        "ci95_low": chk.ci_low[m.mode_id],
                        "ci95_high": chk.ci_high[m.mode_id],
                        "overlaps_ber_target": chk.overlaps_target[m.mode_id],
                        "z_ab": chk.z_ab[m.mode_id],
                    })

    # --- 5. Generate Markdown Report ---
    report_path = out_dir / "calibration_1d_cuda_report.md"
    generate_markdown_report(
        snr_results=snr_results,
        env=env,
        config=config,
        warmup_time=warmup_time,
        total_mc_time=total_mc_time,
        total_blocks_simulated=total_blocks_simulated,
        output_path=report_path,
    )
    t_io_1 = time.perf_counter()
    io_time = t_io_1 - t_io_0

    total_wall_clock = warmup_time + total_mc_time + io_time
    throughput_overall = total_blocks_simulated / total_mc_time
    symbols_sec = throughput_overall * config.symbols_per_block

    print("\n" + "=" * 75)
    print("      L2 CUDA FIXED-BUDGET MONTE CARLO CALIBRATION COMPLETE")
    print("=" * 75)
    print(f"Methodology:                FIXED-BUDGET MONTE CARLO (NO ADAPTIVE STOPPING)")
    print(f"Requested Blocks/Seed:      {target_budget:,} blocks/seed")
    print(f"Actual Blocks/Seed:         {target_budget:,} blocks/seed ({target_budget * 2:,} pooled/SNR)")
    print(f"Total Wall-Clock Time:      {total_wall_clock:.2f} s")
    print(f"  - Modem / GPU Warm-up:    {warmup_time:.3f} s")
    print(f"  - Pure Monte Carlo Time:  {total_mc_time:.2f} s")
    print(f"  - Artifacts & Report I/O: {io_time:.3f} s")
    print(f"Total Blocks Simulated:     {total_blocks_simulated:,} blocks")
    print(f"Total Independent h draws:  {total_blocks_simulated:,} channel realizations")
    print(f"Throughput (Monte Carlo):   {throughput_overall:.1f} blocks/sec")
    print(f"Symbols Throughput:         {symbols_sec:,.0f} symbols/sec")
    print(f"Saved Seed A CSV:           {csv_a_path}")
    print(f"Saved Seed B CSV:           {csv_b_path}")
    print(f"Saved Pooled CSV:           {csv_pooled_path}")
    print(f"Saved Trajectory CSV:       {csv_trajectory_path}")
    print(f"Saved Report:               {report_path}")
    print("=" * 75, flush=True)

    return {
        "total_wall_clock": total_wall_clock,
        "warmup_time": warmup_time,
        "total_mc_time": total_mc_time,
        "io_time": io_time,
        "total_blocks_simulated": total_blocks_simulated,
        "throughput_blocks_sec": throughput_overall,
        "symbols_sec": symbols_sec,
        "snr_results": snr_results,
    }


def generate_markdown_report(
    snr_results: Dict[float, Any],
    env: Dict[str, Any],
    config: ExecutionConfig,
    warmup_time: float,
    total_mc_time: float,
    total_blocks_simulated: int,
    output_path: Path,
    persisted_benchmark: Optional[Dict[str, Any]] = None,
) -> None:
    """Format thesis-quality fixed-budget calibration verification report."""
    throughput = (total_blocks_simulated / total_mc_time) if total_mc_time > 0 else 0.0
    symbols_sec = throughput * config.symbols_per_block

    sample_res = next(iter(snr_results.values())) if snr_results else {}
    actual_blocks = sample_res.get("final_blocks_a", config.blocks_per_seed)
    requested_blocks = sample_res.get("requested_blocks_per_seed", config.blocks_per_seed)

    speedup_line = "- **Speedup vs. Reference CPU Implementation:** `N/A` (no persisted benchmark provided; hard-coded reference disallowed)"
    if persisted_benchmark is not None:
        ref_cpu_throughput = None
        if "Reference CPU Double" in persisted_benchmark:
            ref_cpu_throughput = persisted_benchmark["Reference CPU Double"].get("blocks_per_sec")
        elif "ref_cpu_throughput" in persisted_benchmark:
            ref_cpu_throughput = persisted_benchmark["ref_cpu_throughput"]

        if ref_cpu_throughput and ref_cpu_throughput > 0:
            speedup = throughput / ref_cpu_throughput
            speedup_line = f"- **Speedup vs. Reference CPU Implementation:** `~{speedup:.2f}x` (from measured {ref_cpu_throughput:.1f} blocks/s to {throughput:.1f} blocks/s)"

    profile_str = config.profile if config.profile else "custom"

    lines = [
        "# PHY-ML L2 Fixed-Budget Monte Carlo Calibration Report (CUDA / RTX 3060)",
        "",
        "**Target Platform:** Intel Core i5-14400F (16 threads) | NVIDIA GeForce RTX 3060 (12 GB VRAM)  ",
        f"**Software Stack:** PyTorch `{env['torch_version']}` | CUDA Runtime `{env['cuda_runtime_version']}` | Sionna `{env['sionna_version']}`  ",
        f"**Execution Configuration:** Backend: `{env['selected_device']}` | Precision: `{env['sionna_precision']}` ({env['selected_real_dtype']}) | Batch Size: `{config.batch_blocks}`  ",
        "**Methodology:** **FIXED-BUDGET MONTE CARLO (NO ADAPTIVE STOPPING)**  ",
        f"**Simulation Profile / Budget:** `{profile_str}` | Requested: `{requested_blocks:,} blocks/seed` | Actual: `{actual_blocks:,} blocks/seed`  ",
        "**Date:** 2026-09-20  ",
        "",
        "---",
        "",
        "## 1. Execution & Timing Summary",
        "",
        "- **Methodology:** `FIXED-BUDGET MONTE CARLO`",
        "- **Stopping Rule:** `NO ADAPTIVE STOPPING (every SNR evaluated strictly to fixed budget)`",
        f"- **Requested Blocks per Seed:** `{requested_blocks:,} blocks/seed`",
        f"- **Actual Blocks per Seed:** `{actual_blocks:,} blocks/seed` (`{actual_blocks * 2:,} pooled blocks per SNR point`)",
        f"- **Total Wall-Clock Time:** `{warmup_time + total_mc_time:.2f} seconds`",
        f"  - **Initialization & Warm-Up:** `{warmup_time:.3f} seconds`",
        f"  - **Pure Monte Carlo Simulation:** `{total_mc_time:.2f} seconds`",
        f"- **Total Blocks Evaluated:** `{total_blocks_simulated:,} independent blocks`",
        f"- **Total Independent Channel Realizations ($h$):** `{total_blocks_simulated:,}`",
        f"- **Overall Average Throughput:** `{throughput:,.1f} blocks/second`",
        f"- **Symbol Throughput:** `{symbols_sec:,.0f} symbols/second`",
        speedup_line,
        "",
        "---",
        "",
        "## 2. Analytical Sanity Validation (BPSK Theoretical Rayleigh Baseline)",
        "",
        "Theoretical analytical benchmark: $P_b = \\frac{1}{2}\\left(1 - \\sqrt{\\frac{\\bar{\\gamma}}{1 + \\bar{\\gamma}}}\\right)$ where $\\bar{\\gamma} = 10^{SNR/10}$.",
        "",
        "| SNR (dB) | Empirical Pooled BER | Theoretical BER | Abs Error | Block SE | z-score | Status |",
        "|:--------:|:--------------------:|:---------------:|:---------:|:--------:|:-------:|:------:|",
    ]

    suspicious_points = []
    for snr_db in sorted(snr_results.keys()):
        res = snr_results[snr_db]
        emp_ber = res["final_pooled_ber"][0]
        emp_se = res["final_pooled_se"][0]
        theo = theoretical_bpsk_rayleigh(snr_db)
        diff = abs(emp_ber - theo)
        z = (diff / emp_se) if emp_se > 1e-12 else 0.0
        status = "CONSISTENT" if z <= 3.0 else "SUSPICIOUS"
        if z > 3.0:
            suspicious_points.append((snr_db, z))
        lines.append(
            f"| {snr_db:8.1f} | {emp_ber:20.6e} | {theo:15.6e} | {diff:9.3e} | {emp_se:8.2e} | {z:7.2f} | {status:10s} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Fixed-Budget Monte Carlo Calibration Table across All Modulations",
        "",
        "Methodology: **FIXED-BUDGET MONTE CARLO (NO ADAPTIVE STOPPING)**  ",
        f"Requested blocks/seed: `{requested_blocks:,}` | Actual blocks/seed: `{actual_blocks:,}` (`{actual_blocks * 2:,}` pooled blocks/point)  ",
        "",
        "| SNR (dB) | Modulation | Requested Blks/Seed | Actual Blks/Seed | Actual Pooled Blks | Pooled BER | Pooled SE | 95% Confidence Interval | Target Overlap (0.01)? | z_AB |",
        "|:--------:|:----------:|:-------------------:|:----------------:|:------------------:|:----------:|:---------:|:-----------------------:|:----------------------:|:----:|",
    ])

    for snr_db in sorted(snr_results.keys()):
        res = snr_results[snr_db]
        b_a = res["final_blocks_a"]
        tot_b = res.get("total_blocks_pooled", b_a * 2)
        req_b = res.get("requested_blocks_per_seed", requested_blocks)

        for m in MODES:
            p_ber = res["final_pooled_ber"][m.mode_id]
            p_se = res["final_pooled_se"][m.mode_id]
            z = res["final_z_ab"][m.mode_id]

            ci_hw = CONFIDENCE_K * p_se
            ci_low = max(0.0, float(p_ber - ci_hw))
            ci_high = float(p_ber + ci_hw)
            ov = is_reliability_uncertain(p_ber, p_se, target=BER_TARGET, k=CONFIDENCE_K)
            ov_str = "YES" if ov else "NO"

            lines.append(
                f"| {snr_db:8.1f} | {m.modulation:10s} | {req_b:19,d} | {b_a:16,d} | {tot_b:18,d} | "
                f"{p_ber:10.5e} | {p_se:9.2e} | [{ci_low:.5e}, {ci_high:.5e}] | {ov_str:22s} | {z:4.2f} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Uncertainty & Boundary Semantics",
        "",
        f"Evaluation against $BER_{{target}} = {BER_TARGET:.4f}$ using empirical 95% confidence intervals ($k = {CONFIDENCE_K}$):",
        "- Methodology: `FIXED-BUDGET MONTE CARLO (NO ADAPTIVE STOPPING)`",
        f"- Requested blocks/seed: `{requested_blocks:,}` | Actual blocks/seed: `{actual_blocks:,}`",
        "",
        "| SNR (dB) | Modulation | Pooled BER | Pooled SE | 95% Confidence Interval | Overlaps 0.01? | Label / Semantics |",
        "|:--------:|:----------:|:----------:|:---------:|:-----------------------:|:--------------:|:-----------------:|",
    ])

    for snr_db in sorted(snr_results.keys()):
        res = snr_results[snr_db]
        for m in MODES:
            p_ber = res["final_pooled_ber"][m.mode_id]
            p_se = res["final_pooled_se"][m.mode_id]
            low = max(0.0, p_ber - CONFIDENCE_K * p_se)
            high = p_ber + CONFIDENCE_K * p_se
            ov = is_reliability_uncertain(p_ber, p_se, target=BER_TARGET, k=CONFIDENCE_K)
            ov_str = "YES" if ov else "NO"
            label = "`reliability_uncertain`" if ov else "`resolved`"
            lines.append(
                f"| {snr_db:8.1f} | {m.modulation:10s} | {p_ber:10.5e} | {p_se:9.2e} | [{low:.5e}, {high:.5e}] | {ov_str:14s} | {label} |"
            )

    if not suspicious_points:
        sanity_summary_line = "- Analytical Rayleigh sanity test: **PASS** across the grid."
    else:
        pts_str = ", ".join([f"{snr:.1f} dB (z={z:.2f})" for snr, z in suspicious_points])
        sanity_summary_line = (
            f"- Analytical Rayleigh sanity test: **REVIEW** "
            f"({len(suspicious_points)} suspicious point(s) with z > 3.0: {pts_str})."
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Checkpoint Trajectory Convergence Summary (Human Inspection Only)",
        "",
        f"Checkpoints were recorded every `{config.checkpoint_step:,}` blocks/seed strictly for human observational inspection and convergence analysis. No algorithmic stopping rules were applied during execution.",
        "",
        "---",
        "",
        "## 6. Summary & Verification Status",
        "",
        f"- **Methodology:** FIXED-BUDGET MONTE CARLO (NO ADAPTIVE STOPPING).",
        f"- All {len(snr_results)} SNR grid points evaluated strictly to the requested budget of {actual_blocks:,} blocks/seed ({actual_blocks * 2:,} pooled blocks per point).",
        sanity_summary_line,
        "- No smoothing or synthetic alterations applied to BER estimates.",
        f"- Output artifacts successfully written to `{config.results_dir}/`.",
    ])

    report_text = "\n".join(lines) + "\n"
    output_path.write_text(report_text, encoding="utf-8")


def main() -> None:
    """CLI entry point supporting fixed-budget profiles and explicit overrides."""
    parser = argparse.ArgumentParser(
        description="PHY-ML L2 Fixed-Budget Monte Carlo Calibration Engine (CUDA FP64).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["light", "deep"],
        default=None,
        help="Fixed Monte Carlo profile: 'light' (10,000 blocks/seed) or 'deep' (70,000 blocks/seed).",
    )
    parser.add_argument(
        "--blocks-per-seed",
        type=int,
        default=None,
        help="Explicit arbitrary fixed budget override in blocks per seed (e.g. 100000, 200000).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Compute backend ('cuda' canonical for production run on RTX 3060).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="double",
        choices=["double", "single"],
        help="Numerical precision ('double' canonical FP64).",
    )
    parser.add_argument(
        "--batch-blocks",
        type=int,
        default=500,
        help="Batch chunk size in blocks (canonical: 500).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Custom results directory override.",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=None,
        help="Optional single SNR filter for focused evaluation.",
    )

    args = parser.parse_args()

    # Determine budget and profile
    if args.blocks_per_seed is not None:
        if args.blocks_per_seed <= 0:
            raise ValueError(f"--blocks-per-seed must be positive, got {args.blocks_per_seed}")
        budget = args.blocks_per_seed
        profile_name = args.profile
    elif args.profile is not None:
        budget = FIXED_MC_PROFILES[args.profile]
        profile_name = args.profile
    else:
        profile_name = "deep"
        budget = FIXED_MC_PROFILES["deep"]

    out_dir = resolve_results_dir(
        profile=profile_name if args.blocks_per_seed is None else None,
        blocks_per_seed=budget,
        explicit_dir=args.results_dir,
    )

    cfg = ExecutionConfig(
        backend=args.backend,
        precision=args.precision,
        batch_blocks=args.batch_blocks,
        results_dir=out_dir,
        blocks_per_seed=budget,
        profile=profile_name,
    )

    if args.snr is not None:
        snrs = (args.snr,)
    else:
        snrs = GRID_25_POINTS

    run_fresh_l2_cuda_calibration(config=cfg, snrs=snrs)


if __name__ == "__main__":
    main()
