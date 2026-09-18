"""Raw Physical Layer (PHY) metrics and accumulators.

Contains transparent counters and error rate calculations for link performance analysis.
Deliberately decoupled from mode selection, calibration, or decision policies.
"""
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class RawPHYCounters:
    """Accumulator for raw physical layer counters across transmission blocks."""
    bit_errors: int = 0
    total_bits: int = 0
    block_errors: int = 0
    total_blocks: int = 0
    bits_per_symbol: int = 0

    def update(
        self,
        bit_errors: int,
        total_bits: int,
        block_error: int = 0,
        blocks: int = 1,
    ) -> None:
        """Increment counters with observations from one or more blocks."""
        if bit_errors < 0 or total_bits < 0 or bit_errors > total_bits:
            raise ValueError(f"Invalid bit counts: errors={bit_errors}, total={total_bits}")
        if blocks < 0 or block_error < 0 or block_error > blocks:
            raise ValueError(f"Invalid block counts: block_error={block_error}, blocks={blocks}")

        self.bit_errors += bit_errors
        self.total_bits += total_bits
        self.block_errors += block_error
        self.total_blocks += blocks

    @property
    def ber(self) -> float:
        """Bit Error Rate."""
        return float(self.bit_errors / self.total_bits) if self.total_bits > 0 else 0.0

    @property
    def bler(self) -> float:
        """Block Error Rate."""
        return float(self.block_errors / self.total_blocks) if self.total_blocks > 0 else 0.0

    @property
    def raw_goodput(self) -> float:
        """Nominal correct bits per symbol: (1 - BLER) * bits_per_symbol."""
        return float((1.0 - self.bler) * self.bits_per_symbol)

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
            "raw_goodput": self.raw_goodput,
        }


def compute_ber(bit_errors: int, total_bits: int) -> float:
    """Calculate Bit Error Rate (BER) from raw counts."""
    if total_bits <= 0:
        return 0.0
    if bit_errors < 0 or bit_errors > total_bits:
        raise ValueError(f"bit_errors ({bit_errors}) must be in [0, total_bits ({total_bits})]")
    return float(bit_errors / total_bits)


def compute_bler(block_errors: int, total_blocks: int) -> float:
    """Calculate Block Error Rate (BLER) from raw counts."""
    if total_blocks <= 0:
        return 0.0
    if block_errors < 0 or block_errors > total_blocks:
        raise ValueError(f"block_errors ({block_errors}) must be in [0, total_blocks ({total_blocks})]")
    return float(block_errors / total_blocks)


def compute_raw_goodput(bler: float, bits_per_symbol: int) -> float:
    """Calculate raw delivered bits per symbol: (1 - bler) * bits_per_symbol."""
    if not 0.0 <= bler <= 1.0:
        raise ValueError(f"bler must be in [0, 1], got {bler}")
    if bits_per_symbol < 0:
        raise ValueError(f"bits_per_symbol must be non-negative, got {bits_per_symbol}")
    return float((1.0 - bler) * bits_per_symbol)
