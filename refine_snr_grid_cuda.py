"""R0 Transition-Centered SNR Grid Refinement Calibration Runner.

Evaluates exclusively the 0.25-dB transition midpoint candidate points:
    - 16.75 dB (BPSK -> QPSK midpoint)
    - 22.75 dB (QPSK -> 16QAM midpoint)
    - 28.25 dB (16QAM -> 64QAM midpoint)

Uses the exact canonical Deep Monte Carlo methodology:
    - Budget: 70,000 blocks/seed (140,000 pooled blocks/SNR)
    - Seeds: Master Seed A = 20260918, Master Seed B = 20260919
    - Batch size: 500 blocks (768,000 symbols/batch)
    - Backend: CUDA FP64 (double precision)
    - Zero modification to existing Deep 70k points

Artifacts saved to:
    results/r0_snr_grid_refine_025/
"""
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from config import ExecutionConfig, FIXED_MC_PROFILES
from calibration_l2_cuda import run_fresh_l2_cuda_calibration

REFINEMENT_025_SNRS: Tuple[float, ...] = (16.75, 22.75, 28.25)
DEFAULT_REFINE_RESULTS_DIR: str = "results/r0_snr_grid_refine_025"
DEFAULT_REFINE_BUDGET: int = FIXED_MC_PROFILES["deep"]  # 70,000 blocks/seed


def run_refinement_calibration(
    snrs: Sequence[float] = REFINEMENT_025_SNRS,
    results_dir: str = DEFAULT_REFINE_RESULTS_DIR,
    blocks_per_seed: int = DEFAULT_REFINE_BUDGET,
    backend: str = "cuda",
    precision: str = "double",
    batch_blocks: int = 500,
    config: Optional[ExecutionConfig] = None,
) -> Dict[str, Any]:
    """Execute fixed-budget Monte Carlo calibration for transition refinement points."""
    if config is None:
        out_dir = Path(results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        config = ExecutionConfig(
            backend=backend,
            precision=precision,
            batch_blocks=batch_blocks,
            results_dir=str(out_dir),
            blocks_per_seed=blocks_per_seed,
            profile="deep",
        )
    return run_fresh_l2_cuda_calibration(config=config, snrs=snrs)


def main() -> None:
    """CLI entry point for transition-centered 0.25-dB grid refinement runner."""
    parser = argparse.ArgumentParser(
        description="PHY-ML R0: Transition-Centered 0.25-dB SNR Grid Refinement Runner.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=DEFAULT_REFINE_RESULTS_DIR,
        help=f"Output directory for refinement artifacts (default: {DEFAULT_REFINE_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--blocks-per-seed",
        type=int,
        default=DEFAULT_REFINE_BUDGET,
        help=f"Fixed Monte Carlo budget in blocks per seed (default: {DEFAULT_REFINE_BUDGET:,}).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Compute backend ('cuda' canonical for production run).",
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

    args = parser.parse_args()

    run_refinement_calibration(
        results_dir=args.results_dir,
        blocks_per_seed=args.blocks_per_seed,
        backend=args.backend,
        precision=args.precision,
        batch_blocks=args.batch_blocks,
    )


if __name__ == "__main__":
    main()
