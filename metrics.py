"""Physical Layer (PHY) metrics powered by Sionna 2.0 canonical functions.

Exports:
    - Sionna canonical error evaluators:
          count_errors(b, b_hat)
          count_block_errors(b, b_hat)
          compute_ber(b, b_hat)
          compute_bler(b, b_hat)
    - RawPHYCounters: Accumulator across independent transmission blocks.
"""
from dataclasses import dataclass
from typing import Any, Dict
import torch
from sionna.phy.utils import (
    count_errors,
    count_block_errors,
    compute_ber,
    compute_bler,
)


@dataclass
class RawPHYCounters:
    """Accumulator for raw physical layer counters across transmission blocks."""
    bit_errors: int = 0
    total_bits: int = 0
    block_errors: int = 0
    total_blocks: int = 0
    bits_per_symbol: int = 0
    sum_block_ber: float = 0.0
    sum_sq_block_ber: float = 0.0

    def update(
        self,
        bit_errors: int,
        total_bits: int,
        block_error: int = 0,
        blocks: int = 1,
        sum_ber: float = 0.0,
        sum_sq_ber: float = 0.0,
    ) -> None:
        """Increment counters with observations from one or more blocks."""
        b_err = int(bit_errors)
        tot_b = int(total_bits)
        blk_err = int(block_error)
        blks = int(blocks)

        if b_err < 0 or tot_b < 0 or b_err > tot_b:
            raise ValueError(f"Invalid bit counts: errors={b_err}, total={tot_b}")
        if blks < 0 or blk_err < 0 or blk_err > blks:
            raise ValueError(f"Invalid block counts: block_error={blk_err}, blocks={blks}")

        self.bit_errors += b_err
        self.total_bits += tot_b
        self.block_errors += blk_err
        self.total_blocks += blks
        self.sum_block_ber += float(sum_ber)
        self.sum_sq_block_ber += float(sum_sq_ber)

    @property
    def ber(self) -> float:
        """Macro Bit Error Rate (total bit errors / total bits)."""
        return float(self.bit_errors / self.total_bits) if self.total_bits > 0 else 0.0

    @property
    def bler(self) -> float:
        """Block Error Rate."""
        return float(self.block_errors / self.total_blocks) if self.total_blocks > 0 else 0.0

    @property
    def mean_block_ber(self) -> float:
        """Mean of per-block BER across independent fading blocks."""
        return float(self.sum_block_ber / self.total_blocks) if self.total_blocks > 0 else 0.0

    @property
    def var_block_ber(self) -> float:
        """Sample variance of per-block BER across independent fading blocks."""
        if self.total_blocks <= 1:
            return 0.0
        n = self.total_blocks
        val = (self.sum_sq_block_ber - (self.sum_block_ber ** 2) / n) / (n - 1)
        return float(max(0.0, val))

    @property
    def std_block_ber(self) -> float:
        """Sample standard deviation of per-block BER across independent fading blocks."""
        return float(self.var_block_ber ** 0.5)

    @property
    def se_block_ber(self) -> float:
        """Standard error of the mean block BER across independent fading blocks."""
        return float(self.std_block_ber / (self.total_blocks ** 0.5)) if self.total_blocks > 0 else 0.0

    def summary(self) -> Dict[str, Any]:
        """Return a plain dictionary summary of raw counters and calculated rates."""
        return {
            "bit_errors": int(self.bit_errors),
            "total_bits": int(self.total_bits),
            "ber": self.ber,
            "block_errors": int(self.block_errors),
            "total_blocks": int(self.total_blocks),
            "bler": self.bler,
            "bits_per_symbol": int(self.bits_per_symbol),
            "mean_block_ber": self.mean_block_ber,
            "std_block_ber": self.std_block_ber,
            "se_block_ber": self.se_block_ber,
        }



BER_TARGET: float = 0.01
CONFIDENCE_K: float = 1.96


def is_reliability_uncertain(
    ber: float,
    se: float,
    target: float = BER_TARGET,
    k: float = CONFIDENCE_K,
) -> bool:
    """Determine whether the empirical 95% confidence interval overlaps BER_target.

    Exact criterion:
        ci_low <= target <= ci_high
    where ci_low = max(0.0, ber - k*se) and ci_high = ber + k*se.
    Heuristic windows such as [0.009, 0.011] or [0.008, 0.012] are strictly excluded.
    """
    ci_low = max(0.0, float(ber - k * se))
    ci_high = float(ber + k * se)
    return ci_low <= float(target) <= ci_high


__all__ = [
    "count_errors",
    "count_block_errors",
    "compute_ber",
    "compute_bler",
    "RawPHYCounters",
    "BER_TARGET",
    "CONFIDENCE_K",
    "is_reliability_uncertain",
]
