"""Deep Stopping Uncertainty Diagnostic for PHY-ML L2 Monte Carlo Calibration.

This is a diagnostic experiment, not a replacement for canonical L2.

Goal:
Check whether the current L2 adaptive Monte Carlo stopping rule stops too early
near BER_target = 0.01 because it prioritizes compute efficiency, or whether the
remaining uncertainty is genuinely hard to resolve statistically.

Architecture & Constraints:
    - Isolated diagnostic script only (canonical L2 behavior and files untouched).
    - Preserves exact seed generation, RNG semantics, and simulation prefix of
      calibration_l2_cuda.py.
    - Reuses ExecutionConfig, BatchPHYEngine, OnlineBlockStats, MODES,
      pool_two_streams, and BER_TARGET.
    - Tests only the 3 previously uncertain boundary points:
        * SNR 14.0 dB, BPSK (mode_id: 0), old stop depth: 15,000 blocks/seed
        * SNR 23.0 dB, 16QAM (mode_id: 2), old stop depth: 35,000 blocks/seed
        * SNR 28.0 dB, 64QAM (mode_id: 3), old stop depth: 35,000 blocks/seed
    - Disables adaptive stopping; forces simulation to MAX_BLOCKS = 200,000 blocks/seed
      at STEP = 5,000 blocks/seed.
    - At each checkpoint, computes pooled BER, pooled SE, empirical 95% CI (BER +- 1.96*SE),
      and tests whether the CI overlaps BER_TARGET = 0.01.
    - Evaluates persistent resolution across subsequent checkpoints (transient non-overlaps
      are not called resolved unless they remain stable through the final checkpoint).
    - Outputs results to stdout only; does not overwrite canonical L2 artifacts.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Ensure project root is available on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config import ExecutionConfig, probe_environment, print_environment_report
from phy_engine import ModulationMode, MODES
from cuda_engine import BatchPHYEngine, OnlineBlockStats
from calibration_l2_cuda import pool_two_streams
from metrics import BER_TARGET, CONFIDENCE_K, is_reliability_uncertain


@dataclass(frozen=True)
class UncertainOperatingPoint:
    """Operating point identified as reliability_uncertain near BER_target=0.01 in canonical L2."""
    snr_db: float
    mode_id: int
    modulation: str
    old_stop_depth: int  # in blocks/seed


# Exactly the 3 operating points where canonical L2 converged with reliability_uncertain
UNCERTAIN_OPERATING_POINTS: Tuple[UncertainOperatingPoint, ...] = (
    UncertainOperatingPoint(snr_db=14.0, mode_id=0, modulation="BPSK", old_stop_depth=15_000),
    UncertainOperatingPoint(snr_db=23.0, mode_id=2, modulation="16QAM", old_stop_depth=35_000),
    UncertainOperatingPoint(snr_db=28.0, mode_id=3, modulation="64QAM", old_stop_depth=35_000),
)

# Mandatory checkpoint reporting schedule
MANDATORY_PRINT_CHECKPOINTS: Tuple[int, ...] = (
    5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 35_000,
    50_000, 75_000, 100_000, 125_000, 150_000, 175_000, 200_000,
)


@dataclass
class DiagnosticCheckpointRecord:
    """Evaluation state at a single blocks/seed checkpoint."""
    checkpoint_idx: int
    blocks_per_seed: int
    total_blocks_pooled: int
    ber_pooled: float
    se_pooled: float
    ci_low: float
    ci_high: float
    overlaps_target: bool
    is_old_stop: bool


def analyze_checkpoint_series(
    checkpoints: Sequence[DiagnosticCheckpointRecord],
    old_stop_blocks: int,
    target: float = BER_TARGET,
) -> Dict[str, Any]:
    """Analyze the trajectory of confidence intervals across checkpoints.

    Detects:
        - The old stopping checkpoint record.
        - The first checkpoint where CI no longer overlaps target.
        - The first persistently resolved checkpoint (i.e. CI no longer overlaps target
          for this and ALL subsequent checkpoints through the end of the run).
        - Stability analysis (verifies whether initial non-overlap was transient or permanent).
        - Final status at the maximum checkpoint.
        - Compute multiplier relative to old stopping depth.
    """
    if not checkpoints:
        raise ValueError("checkpoints sequence cannot be empty")

    n = len(checkpoints)

    # 1. Locate old stop checkpoint
    old_stop_chk: Optional[DiagnosticCheckpointRecord] = None
    for chk in checkpoints:
        if chk.blocks_per_seed == old_stop_blocks:
            old_stop_chk = chk
            break

    # 2. First non-overlap checkpoint (any point where CI does not cover target)
    first_non_overlap_chk: Optional[DiagnosticCheckpointRecord] = None
    for chk in checkpoints:
        if not chk.overlaps_target:
            first_non_overlap_chk = chk
            break

    # 3. First persistently resolved checkpoint:
    # Smallest index i such that for all j in [i, n-1], checkpoints[j].overlaps_target is False.
    first_persistent_chk: Optional[DiagnosticCheckpointRecord] = None
    for i in range(n):
        if all(not checkpoints[j].overlaps_target for j in range(i, n)):
            first_persistent_chk = checkpoints[i]
            break

    # 4. Stability classification
    if first_non_overlap_chk is None:
        resolution_stability = "NEVER_RESOLVED"
        resolution_desc = "CI continuously overlaps BER_target across all checkpoints"
    elif first_persistent_chk is None:
        resolution_stability = "TRANSIENT_ONLY"
        resolution_desc = (
            f"Transient non-overlap observed at {first_non_overlap_chk.blocks_per_seed:,} blks/seed, "
            "but later checkpoint(s) re-entered overlap"
        )
    elif first_non_overlap_chk.blocks_per_seed == first_persistent_chk.blocks_per_seed:
        resolution_stability = "STABLE"
        resolution_desc = (
            f"Resolved at {first_persistent_chk.blocks_per_seed:,} blks/seed "
            "and remained strictly separated through final depth"
        )
    else:
        resolution_stability = "PERSISTENT_AFTER_TRANSIENT"
        resolution_desc = (
            f"Initial transient non-overlap at {first_non_overlap_chk.blocks_per_seed:,} blks/seed; "
            f"permanently resolved from {first_persistent_chk.blocks_per_seed:,} blks/seed onward"
        )

    final_chk = checkpoints[-1]
    final_overlaps = final_chk.overlaps_target
    if final_overlaps:
        final_overlap_label = "OVERLAPS (Uncertain)"
    else:
        if final_chk.ci_high < target:
            final_overlap_label = "RESOLVED (Strictly Below 0.01)"
        else:
            final_overlap_label = "RESOLVED (Strictly Above 0.01)"

    compute_multiplier = float(final_chk.blocks_per_seed) / float(old_stop_blocks)

    return {
        "old_stop_chk": old_stop_chk,
        "first_non_overlap_chk": first_non_overlap_chk,
        "first_persistent_chk": first_persistent_chk,
        "resolution_stability": resolution_stability,
        "resolution_desc": resolution_desc,
        "final_chk": final_chk,
        "final_overlap_status": final_overlap_label,
        "compute_multiplier": compute_multiplier,
    }


def run_deep_stopping_diagnostic(
    config: Optional[ExecutionConfig] = None,
    target_points: Optional[Sequence[UncertainOperatingPoint]] = None,
    max_blocks_per_seed: int = 200_000,
    step_blocks_per_seed: int = 5_000,
    mandatory_print_checkpoints: Sequence[int] = MANDATORY_PRINT_CHECKPOINTS,
) -> Dict[float, Dict[str, Any]]:
    """Execute deep stopping diagnostic without adaptive stopping.

    Parameters
    ----------
    config : ExecutionConfig, optional
        Execution configuration. Canonical defaults: backend='cuda', precision='double',
        batch_blocks=500, symbols_per_block=1536.
    target_points : Sequence[UncertainOperatingPoint], optional
        Operating points to evaluate. Defaults to UNCERTAIN_OPERATING_POINTS.
    max_blocks_per_seed : int
        Maximum blocks per Monte Carlo stream (default 200,000).
    step_blocks_per_seed : int
        Block increment between diagnostic checkpoints (default 5,000).
    mandatory_print_checkpoints : Sequence[int]
        Checkpoints guaranteed to be printed in the detailed log.

    Returns
    -------
    all_results : Dict[float, Dict[str, Any]]
        Diagnostic results keyed by SNR in dB.
    """
    if config is None:
        config = ExecutionConfig(
            backend="cuda",
            precision="double",
            batch_blocks=500,
            results_dir="results/l2_cuda_rtx3060",
        )

    if target_points is None:
        target_points = UNCERTAIN_OPERATING_POINTS

    # Hardware & runtime environment probe
    env = probe_environment(config)
    print_environment_report(env)

    device_str = env["selected_device"]
    prec_str = env["sionna_precision"]
    batch_size = config.batch_blocks
    is_cuda = "cuda" in device_str

    print("\n" + "=" * 80)
    print("   PHY-ML L2 DEEP STOPPING UNCERTAINTY DIAGNOSTIC")
    print("   (Diagnostic experiment only; canonical L2 artifacts are not overwritten)")
    print(f"   Target Operating Points : {len(target_points)} points")
    print(f"   Device                  : {device_str} | Precision: {prec_str} | Batch: {batch_size}")
    print(f"   Seeds                   : Seed A = {config.master_seed_a} | Seed B = {config.master_seed_b}")
    print(f"   Diagnostic Depth        : {max_blocks_per_seed:,} blocks/seed (Step: {step_blocks_per_seed:,})")
    print(f"   Target Metric           : BER_target = {BER_TARGET:.4f} (95% CI with k = {CONFIDENCE_K})")
    print("=" * 80, flush=True)

    # 1. GPU / Engine warm-up (identical to calibration_l2_cuda.py)
    t_w0 = time.perf_counter()
    engine = BatchPHYEngine(
        symbols_per_block=config.symbols_per_block,
        modes=MODES,
        device=device_str,
        precision=prec_str,
    )
    if is_cuda:
        torch.cuda.synchronize()
    engine.evaluate_chunk(18.0, min(500, batch_size), 1, 2, 3)
    if is_cuda:
        torch.cuda.synchronize()
    t_w1 = time.perf_counter()
    print(f"Engine & Hardware Warm-up completed in {t_w1 - t_w0:.3f} s\n", flush=True)

    mandatory_set = set(mandatory_print_checkpoints)
    all_snr_results: Dict[float, Dict[str, Any]] = {}
    t_diag_start = time.perf_counter()

    for p_idx, pt in enumerate(target_points):
        print("\n" + "#" * 80)
        print(f"### [{p_idx + 1}/{len(target_points)}] OPERATING POINT: SNR = {pt.snr_db:.1f} dB | "
              f"Modulation = {pt.modulation} (mode_id: {pt.mode_id})")
        print(f"### Canonical L2 Old Stopping Depth: {pt.old_stop_depth:,} blocks/seed")
        print(f"### Deep Diagnostic Target Depth  : {max_blocks_per_seed:,} blocks/seed "
              f"({max_blocks_per_seed / pt.old_stop_depth:.2f}x compute multiplier)")
        print("#" * 80, flush=True)

        stats_a = {m.mode_id: OnlineBlockStats(m.bits_per_symbol) for m in MODES}
        stats_b = {m.mode_id: OnlineBlockStats(m.bits_per_symbol) for m in MODES}

        blocks_done_a = 0
        blocks_done_b = 0
        chunk_idx_a = 0
        chunk_idx_b = 0
        checkpoint_idx = 0

        checkpoints: List[DiagnosticCheckpointRecord] = []
        t_snr_start = time.perf_counter()

        target_blocks = 0
        while target_blocks < max_blocks_per_seed:
            checkpoint_idx += 1
            target_blocks = min(target_blocks + step_blocks_per_seed, max_blocks_per_seed)

            # --- Stream A execution: strictly preserve canonical seed logic ---
            while blocks_done_a < target_blocks:
                cur_b = min(batch_size, target_blocks - blocks_done_a)
                f_seed = (config.master_seed_a + int(pt.snr_db * 1000) + chunk_idx_a * 7919) % (2**31 - 1)
                n_seed = (f_seed + 101) % (2**31 - 1)
                b_seed = (f_seed + 202) % (2**31 - 1)

                r_a = engine.evaluate_chunk(pt.snr_db, cur_b, f_seed, n_seed, b_seed)
                for m in MODES:
                    stats_a[m.mode_id].update_chunk(cur_b, *r_a[m.mode_id])

                blocks_done_a += cur_b
                chunk_idx_a += 1

            # --- Stream B execution: strictly preserve canonical seed logic ---
            while blocks_done_b < target_blocks:
                cur_b = min(batch_size, target_blocks - blocks_done_b)
                f_seed = (config.master_seed_b + int(pt.snr_db * 1000) + chunk_idx_b * 7919) % (2**31 - 1)
                n_seed = (f_seed + 101) % (2**31 - 1)
                b_seed = (f_seed + 202) % (2**31 - 1)

                r_b = engine.evaluate_chunk(pt.snr_db, cur_b, f_seed, n_seed, b_seed)
                for m in MODES:
                    stats_b[m.mode_id].update_chunk(cur_b, *r_b[m.mode_id])

                blocks_done_b += cur_b
                chunk_idx_b += 1

            if is_cuda:
                torch.cuda.synchronize()

            # --- Evaluate statistics for focus modulation at this checkpoint ---
            sa = stats_a[pt.mode_id]
            sb = stats_b[pt.mode_id]
            ber_pool, mean_pool, se_pool = pool_two_streams(sa, sb)

            ci_half_width = CONFIDENCE_K * se_pool
            ci_low = max(0.0, float(ber_pool - ci_half_width))
            ci_high = float(ber_pool + ci_half_width)
            overlaps = is_reliability_uncertain(ber_pool, se_pool, target=BER_TARGET, k=CONFIDENCE_K)
            is_old_stop = (blocks_done_a == pt.old_stop_depth)

            rec = DiagnosticCheckpointRecord(
                checkpoint_idx=checkpoint_idx,
                blocks_per_seed=blocks_done_a,
                total_blocks_pooled=blocks_done_a + blocks_done_b,
                ber_pooled=ber_pool,
                se_pooled=se_pool,
                ci_low=ci_low,
                ci_high=ci_high,
                overlaps_target=overlaps,
                is_old_stop=is_old_stop,
            )
            checkpoints.append(rec)

            # Determine whether this checkpoint is in the mandatory print list or an event point
            is_mandatory = (blocks_done_a in mandatory_set)
            is_final = (blocks_done_a == max_blocks_per_seed)

            # Build tag markers
            tags = []
            if is_old_stop:
                tags.append("OLD STOP")
            if not overlaps:
                tags.append("NO OVERLAP")
            if is_final:
                tags.append(f"FINAL ({blocks_done_a:,} blks)")
            tag_str = f" <-- [{', '.join(tags)}]" if tags else ""

            overlap_str = "YES (Uncertain)" if overlaps else "NO  (Resolved)"
            print(
                f"   [Chk #{checkpoint_idx:02d} | {blocks_done_a:7,d} blks/seed | {blocks_done_a * 2:7,d} pooled] "
                f"BER: {ber_pool:.5e} | SE: {se_pool:.2e} | "
                f"95% CI: [{ci_low:.5e}, {ci_high:.5e}] | Overlaps 0.01: {overlap_str}{tag_str}",
                flush=True,
            )

        if is_cuda:
            torch.cuda.synchronize()
        t_snr_end = time.perf_counter()
        snr_elapsed = t_snr_end - t_snr_start

        # Post-run analysis of the trajectory
        analysis = analyze_checkpoint_series(checkpoints, pt.old_stop_depth, target=BER_TARGET)

        # Print structured Milestone Trajectory Table for this operating point
        print("\n   " + "-" * 76)
        print(f"   CHECKPOINT PROGRESSION TABLE: SNR {pt.snr_db:.1f} dB ({pt.modulation})")
        print("   " + "-" * 76)
        print(f"   {'Chk':>4} | {'Blks/Seed':>9} | {'Pooled':>9} | {'Pooled BER':>12} | {'SE':>9} | {'95% CI Range':>27} | {'Overlaps?':>9} | {'Notes'}")
        print("   " + "-" * 76)

        # Select checkpoints to display: mandatory set + old stop + first non-overlap + persistent + final
        display_blocks = set(mandatory_print_checkpoints)
        display_blocks.add(pt.old_stop_depth)
        display_blocks.add(max_blocks_per_seed)
        if analysis["first_non_overlap_chk"]:
            display_blocks.add(analysis["first_non_overlap_chk"].blocks_per_seed)
        if analysis["first_persistent_chk"]:
            display_blocks.add(analysis["first_persistent_chk"].blocks_per_seed)

        for chk in checkpoints:
            if chk.blocks_per_seed in display_blocks:
                note_items = []
                if chk.blocks_per_seed == pt.old_stop_depth:
                    note_items.append("Old Canonical Stop")
                if analysis["first_non_overlap_chk"] and chk.blocks_per_seed == analysis["first_non_overlap_chk"].blocks_per_seed:
                    note_items.append("First Non-Overlap")
                if analysis["first_persistent_chk"] and chk.blocks_per_seed == analysis["first_persistent_chk"].blocks_per_seed:
                    note_items.append("First Persistently Resolved")
                if chk.blocks_per_seed == max_blocks_per_seed:
                    note_items.append("Final Depth")
                notes = ", ".join(note_items)

                ov_sym = "YES" if chk.overlaps_target else "NO "
                ci_str = f"[{chk.ci_low:.5e}, {chk.ci_high:.5e}]"
                print(f"   {chk.checkpoint_idx:4d} | {chk.blocks_per_seed:9,d} | {chk.total_blocks_pooled:9,d} | "
                      f"{chk.ber_pooled:12.5e} | {chk.se_pooled:9.2e} | {ci_str:>27} | {ov_sym:>9} | {notes}")
        print("   " + "-" * 76)

        # Print Operating Point Summary
        old_chk = analysis["old_stop_chk"]
        first_no_chk = analysis["first_non_overlap_chk"]
        first_pers_chk = analysis["first_persistent_chk"]
        final_chk = analysis["final_chk"]

        if old_chk is not None:
            old_stop_summary = (
                f"{pt.old_stop_depth:,} blocks/seed "
                f"(BER = {old_chk.ber_pooled:.5e}, SE = {old_chk.se_pooled:.2e}, Overlaps = {old_chk.overlaps_target})"
            )
        else:
            old_stop_summary = f"{pt.old_stop_depth:,} blocks/seed (not reached in this run)"

        first_no_str = (
            f"{first_no_chk.blocks_per_seed:,} blocks/seed (Chk #{first_no_chk.checkpoint_idx})"
            if first_no_chk else "None (overlaps 0.01 across all evaluated blocks)"
        )
        first_pers_str = (
            f"{first_pers_chk.blocks_per_seed:,} blocks/seed (Chk #{first_pers_chk.checkpoint_idx})"
            if first_pers_chk else "None (not persistently resolved)"
        )

        print(f"\n   >>> SNR = {pt.snr_db:.1f} dB ({pt.modulation}) SUMMARY:")
        print(f"       - Old Stop Depth                  : {old_stop_summary}")
        print(f"       - First Non-Overlap Checkpoint    : {first_no_str}")
        print(f"       - First Persistently Resolved     : {first_pers_str}")
        print(f"       - Stability Evaluation            : {analysis['resolution_desc']}")
        print(f"       - Final Pooled BER                : {final_chk.ber_pooled:.5e}")
        print(f"       - Final Pooled SE                 : {final_chk.se_pooled:.2e}")
        print(f"       - Final 95% Confidence Interval   : [{final_chk.ci_low:.5e}, {final_chk.ci_high:.5e}]")
        print(f"       - Final Overlap Status            : {analysis['final_overlap_status']}")
        print(f"       - Compute Multiplier vs Old Stop  : {analysis['compute_multiplier']:.2f}x "
              f"({final_chk.blocks_per_seed:,} / {pt.old_stop_depth:,})")
        print(f"       - Wall-Clock Time                 : {snr_elapsed:.2f} s", flush=True)

        all_snr_results[pt.snr_db] = {
            "point": pt,
            "checkpoints": checkpoints,
            "analysis": analysis,
            "elapsed_seconds": snr_elapsed,
        }

    t_diag_end = time.perf_counter()
    total_diag_time = t_diag_end - t_diag_start

    # Print Final Summary Report across all 3 SNRs (Requirement 10)
    print_final_diagnostic_summary(all_snr_results, total_diag_time)

    return all_snr_results


def print_final_diagnostic_summary(
    results: Dict[float, Dict[str, Any]],
    total_time: float,
) -> None:
    """Print the final comprehensive comparison table and scientific evaluation."""
    print("\n\n" + "=" * 95)
    print("                          FINAL DEEP STOPPING DIAGNOSTIC SUMMARY REPORT")
    print("=" * 95)
    print(f"Total Simulation Time: {total_time:.2f} s | Target BER: {BER_TARGET:.4f} | Confidence: 95% (k={CONFIDENCE_K})")
    print("-" * 95)

    for snr_db, data in results.items():
        pt: UncertainOperatingPoint = data["point"]
        analysis = data["analysis"]
        old_chk: Optional[DiagnosticCheckpointRecord] = analysis["old_stop_chk"]
        first_no: Optional[DiagnosticCheckpointRecord] = analysis["first_non_overlap_chk"]
        first_pers: Optional[DiagnosticCheckpointRecord] = analysis["first_persistent_chk"]
        final_chk: DiagnosticCheckpointRecord = analysis["final_chk"]

        first_no_text = f"{first_no.blocks_per_seed:,} blks/seed" if first_no else "None"
        first_pers_text = f"{first_pers.blocks_per_seed:,} blks/seed" if first_pers else "None"
        old_stop_text = (
            f"{pt.old_stop_depth:7,d} blocks/seed ({pt.old_stop_depth * 2:7,d} pooled)"
            if old_chk is not None else f"{pt.old_stop_depth:7,d} blocks/seed (not reached)"
        )

        print(f"\n[Operating Point] SNR = {pt.snr_db:4.1f} dB | Modulation = {pt.modulation} (mode_id: {pt.mode_id})")
        print(f"  * Old Stop Depth               : {old_stop_text}")
        print(f"  * First Non-Overlap Checkpoint : {first_no_text}")
        print(f"  * First Persistently Resolved  : {first_pers_text}")
        print(f"  * Stability Behavior           : {analysis['resolution_desc']}")
        print(f"  * Final Pooled BER ({final_chk.blocks_per_seed:,} blks/seed) : {final_chk.ber_pooled:.5e}")
        print(f"  * Final Pooled SE ({final_chk.blocks_per_seed:,} blks/seed)  : {final_chk.se_pooled:.2e}")
        print(f"  * Final 95% Confidence Interval: [{final_chk.ci_low:.5e}, {final_chk.ci_high:.5e}]")
        print(f"  * Final Overlap Status         : {analysis['final_overlap_status']}")
        print(f"  * Compute Multiplier           : {analysis['compute_multiplier']:.2f}x "
              f"({final_chk.blocks_per_seed:,} / {pt.old_stop_depth:,})")

    # Comparative Table
    print("\n" + "-" * 95)
    print("COMPARATIVE TABLE ACROSS UNCERTAIN OPERATING POINTS:")
    print("-" * 95)
    header = (
        f"| {'SNR (dB)':^8} | {'Mod':^6} | {'Old Stop':^10} | {'1st Non-Overlap':^15} | "
        f"{'1st Persist Res':^15} | {'Final BER':^12} | {'Final SE':^9} | {'Final 95% CI':^25} | "
        f"{'Final Status':^20} | {'Multiplier':^10} |"
    )
    separator = (
        f"|:{'-'*8}:|:{'-'*6}:|:{'-'*10}:|:{'-'*15}:|:{'-'*15}:|:{'-'*12}:|:{'-'*9}:|:{'-'*25}:|:{'-'*20}:|:{'-'*10}:|"
    )
    print(header)
    print(separator)

    for snr_db, data in results.items():
        pt = data["point"]
        analysis = data["analysis"]
        first_no = analysis["first_non_overlap_chk"]
        first_pers = analysis["first_persistent_chk"]
        final_chk = analysis["final_chk"]

        no_str = f"{first_no.blocks_per_seed:,}" if first_no else "None"
        pers_str = f"{first_pers.blocks_per_seed:,}" if first_pers else "None"
        ci_str = f"[{final_chk.ci_low:.4e}, {final_chk.ci_high:.4e}]"

        status_short = "OVERLAPS" if final_chk.overlaps_target else ("BELOW" if final_chk.ci_high < BER_TARGET else "ABOVE")

        row = (
            f"| {pt.snr_db:8.1f} | {pt.modulation:^6} | {pt.old_stop_depth:10,d} | {no_str:^15} | "
            f"{pers_str:^15} | {final_chk.ber_pooled:12.5e} | {final_chk.se_pooled:9.2e} | {ci_str:^25} | "
            f"{status_short:^20} | {analysis['compute_multiplier']:9.2f}x |"
        )
        print(row)

    print("-" * 95)

    # Scientific Conclusion Synthesis
    print("\nSCIENTIFIC INTERPRETATION & CONCLUSION:")
    all_resolved = all(data["analysis"]["first_persistent_chk"] is not None for data in results.values())
    none_resolved = all(data["analysis"]["first_persistent_chk"] is None for data in results.values())
    max_eval_b = max(data["analysis"]["final_chk"].blocks_per_seed for data in results.values())

    if all_resolved:
        print(f"  - ALL previously uncertain operating points were PERSISTENTLY RESOLVED by extending sampling to {max_eval_b:,} blocks/seed.")
        print("  - Conclusion: The canonical L2 stopping rule terminated early due to compute efficiency trade-offs.")
        print("    Additional Monte Carlo sampling successfully reduced SE enough to separate the CI from BER_target = 0.01.")
    elif none_resolved:
        print(f"  - NONE of the uncertain operating points resolved, even after {max_eval_b:,} blocks/seed ({max_eval_b * 2:,} pooled).")
        print("  - Conclusion: The remaining uncertainty is genuinely hard to resolve statistically because the true")
        print("    operating BER is virtually indistinguishable from 0.01. Stopping at the canonical L2 depth was optimal.")
    else:
        print("  - MIXED RESOLUTION: Some operating points resolved while others remain statistically ambiguous:")
        for snr_db, data in results.items():
            pt = data["point"]
            pers = data["analysis"]["first_persistent_chk"]
            final_b = data["analysis"]["final_chk"].blocks_per_seed
            if pers:
                print(f"    * SNR {pt.snr_db:.1f} dB ({pt.modulation}): Resolved at {pers.blocks_per_seed:,} blocks/seed "
                      f"({pers.blocks_per_seed / pt.old_stop_depth:.2f}x compute).")
            else:
                print(f"    * SNR {pt.snr_db:.1f} dB ({pt.modulation}): Genuinely uncertain even at {final_b:,} blocks/seed "
                      f"({data['analysis']['compute_multiplier']:.2f}x compute).")
    print("=" * 95 + "\n", flush=True)


def main() -> None:
    """Entry point for CLI execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Deep Stopping Uncertainty Diagnostic for PHY-ML L2 Monte Carlo Calibration.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Compute backend ('cuda' canonical for production run on RTX 3060; 'cpu' for smoke testing).",
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
        help="Batch size in blocks per chunk (default 500).",
    )
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=200_000,
        help="Maximum blocks per seed to evaluate (default 200,000).",
    )
    parser.add_argument(
        "--step-blocks",
        type=int,
        default=5_000,
        help="Checkpoint increment in blocks per seed (default 5,000).",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=None,
        help="Optional single SNR filter (14.0, 23.0, or 28.0).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a lightweight test (CPU, max 1000 blocks, step 500) to verify code paths locally.",
    )

    args = parser.parse_args()

    if args.smoke_test:
        print("[SMOKE TEST MODE ENABLED] Running lightweight verification on CPU...")
        cfg = ExecutionConfig(
            backend="cpu",
            precision="double",
            batch_blocks=500,
        )
        max_b = 1_000
        step_b = 500
        targets = (UNCERTAIN_OPERATING_POINTS[0],)
        mandatory = (500, 1000)
    else:
        cfg = ExecutionConfig(
            backend=args.backend,
            precision=args.precision,
            batch_blocks=args.batch_blocks,
        )
        max_b = args.max_blocks
        step_b = args.step_blocks
        mandatory = MANDATORY_PRINT_CHECKPOINTS

        if args.snr is not None:
            filtered = [pt for pt in UNCERTAIN_OPERATING_POINTS if abs(pt.snr_db - args.snr) < 1e-3]
            if not filtered:
                raise ValueError(f"Specified SNR {args.snr} is not one of the uncertain points (14.0, 23.0, 28.0)")
            targets = tuple(filtered)
        else:
            targets = UNCERTAIN_OPERATING_POINTS

    run_deep_stopping_diagnostic(
        config=cfg,
        target_points=targets,
        max_blocks_per_seed=max_b,
        step_blocks_per_seed=step_b,
        mandatory_print_checkpoints=mandatory,
    )


if __name__ == "__main__":
    main()
