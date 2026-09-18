"""Minimal uncoded coherent physical layer (PHY) engine.

Supports:
    - Modulations: BPSK (1 bit/sym), QPSK (2 bits/sym), 16-QAM (4 bits/sym), 64-QAM (6 bits/sym).
    - Unit-average-energy normalization: E[|s|^2] = 1 for all constellations.
    - True 2D Gray mapping with single-bit differences between adjacent constellation points.
    - Hard coherent maximum likelihood (ML) demodulation with known complex channel coefficient h.
    - Paired physical evaluation across candidate modulation modes on identical channel and noise realizations.
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence
import numpy as np

from channel import apply_channel, ChannelOutput


@dataclass(frozen=True)
class ModulationMode:
    """Representation of an uncoded modulation mode."""
    mode_id: int
    modulation: str
    bits_per_symbol: int

    @property
    def name(self) -> str:
        return f"{self.modulation}-uncoded"


# Catalogue of active uncoded PHY modes
MODES: tuple[ModulationMode, ...] = (
    ModulationMode(mode_id=0, modulation="BPSK", bits_per_symbol=1),
    ModulationMode(mode_id=1, modulation="QPSK", bits_per_symbol=2),
    ModulationMode(mode_id=2, modulation="16QAM", bits_per_symbol=4),
    ModulationMode(mode_id=3, modulation="64QAM", bits_per_symbol=6),
)

MODE_BY_ID: dict[int, ModulationMode] = {m.mode_id: m for m in MODES}
MODE_BY_NAME: dict[str, ModulationMode] = {m.modulation.upper(): m for m in MODES}


@dataclass(frozen=True)
class BlockEvaluationResult:
    """Outcome of evaluating one transmission block under a specific modulation mode."""
    mode_id: int
    modulation: str
    bits_per_symbol: int
    bit_errors: int
    total_bits: int
    block_error: int   # 1 if bit_errors > 0, else 0
    ber: float
    h: complex
    snr_db: float


def validate_binary(bits: np.ndarray) -> np.ndarray:
    """Ensure input is a 1D array of binary ints (0 or 1)."""
    b = np.asarray(bits)
    if b.ndim != 1:
        raise ValueError(f"Expected 1D binary array, got ndim={b.ndim}")
    if len(b) == 0:
        raise ValueError("Binary array must not be empty")
    if not np.isin(b, [0, 1]).all():
        raise ValueError("Array elements must be binary 0 or 1")
    return b.astype(np.int32)


@lru_cache(maxsize=8)
def constellation(bits_per_symbol: int) -> tuple[np.ndarray, np.ndarray]:
    """Return unit-energy constellation points and bit labels with Gray mapping.

    Parameters
    ----------
    bits_per_symbol : int
        1 (BPSK), 2 (QPSK), 4 (16-QAM), or 6 (64-QAM).

    Returns
    -------
    points : np.ndarray
        Complex128 array of length 2^m with E[|points|^2] = 1.0.
    labels : np.ndarray
        Int32 array of shape (2^m, m) containing the binary label for each point.
    """
    m = bits_per_symbol
    if m not in (1, 2, 4, 6):
        raise ValueError(f"Supported bits_per_symbol: 1 (BPSK), 2 (QPSK), 4 (16QAM), 6 (64QAM). Got {m}")

    labels = ((np.arange(2**m)[:, None] >> np.arange(m - 1, -1, -1)) & 1).astype(np.int32)

    if m == 1:
        # BPSK: bit 0 -> +1, bit 1 -> -1
        points = (1.0 - 2.0 * labels[:, 0]).astype(np.complex128)
    elif m == 2:
        # QPSK: Gray mapped across I and Q
        points = ((1.0 - 2.0 * labels[:, 0]) + 1j * (1.0 - 2.0 * labels[:, 1])) / np.sqrt(2.0)
    elif m == 4:
        # 16-QAM: PAM-4 Gray levels [-3, -1, 3, 1] for 2 bits each
        levels = np.array([-3.0, -1.0, 3.0, 1.0])
        i_comp = levels[labels[:, 0] * 2 + labels[:, 1]]
        q_comp = levels[labels[:, 2] * 2 + labels[:, 3]]
        points = (i_comp + 1j * q_comp) / np.sqrt(10.0)
    else:
        # 64-QAM: PAM-8 Gray levels for 3 bits each
        levels_8 = np.array([-7.0, -5.0, -1.0, -3.0, 7.0, 5.0, 1.0, 3.0])
        i_idx = labels[:, 0] * 4 + labels[:, 1] * 2 + labels[:, 2]
        q_idx = labels[:, 3] * 4 + labels[:, 4] * 2 + labels[:, 5]
        points = (levels_8[i_idx] + 1j * levels_8[q_idx]) / np.sqrt(42.0)

    points = np.asarray(points, dtype=np.complex128)
    points.setflags(write=False)
    labels.setflags(write=False)
    return points, labels


def modulate(bits: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    """Map binary bits to unit-energy complex modulation symbols.

    Parameters
    ----------
    bits : np.ndarray
        1D array of binary bits (0 or 1). Length must be divisible by bits_per_symbol.
    bits_per_symbol : int
        Modulation order (1, 2, 4, or 6).

    Returns
    -------
    symbols : np.ndarray
        Complex128 array of length len(bits) // bits_per_symbol.
    """
    b = validate_binary(bits)
    m = bits_per_symbol
    if len(b) % m != 0:
        raise ValueError(f"Bit length {len(b)} must divide evenly into symbols of size {m}")

    points, _ = constellation(m)
    indices = b.reshape(-1, m) @ (1 << np.arange(m - 1, -1, -1))
    symbols = points[indices]
    symbols.setflags(write=False)
    return symbols


def demodulate(
    received_symbols: np.ndarray,
    h: complex,
    bits_per_symbol: int,
) -> np.ndarray:
    """Hard coherent maximum likelihood demodulation with known channel coefficient h.

    Metric: argmin_{s in C} |y - h * s|^2.
    Vectorized and unconditionally stable for any complex h.

    Parameters
    ----------
    received_symbols : np.ndarray
        1D complex array of received symbols y = h * x + n.
    h : complex
        Scalar complex channel coefficient known to the coherent receiver.
    bits_per_symbol : int
        Modulation order (1, 2, 4, or 6).

    Returns
    -------
    detected_bits : np.ndarray
        1D int32 array of detected bits.
    """
    y = np.asarray(received_symbols, dtype=np.complex128)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("received_symbols must be a non-empty 1D array")
    if not np.isfinite(y).all():
        raise ValueError("received_symbols contains non-finite values")

    m = bits_per_symbol
    points, labels = constellation(m)

    # Coherent ML distance: |y - h * s|^2
    diff = y[:, None] - (h * points[None, :])
    dist_sq = np.real(diff * np.conj(diff))
    best_idx = np.argmin(dist_sq, axis=1)

    detected = labels[best_idx].reshape(-1)
    detected.setflags(write=False)
    return detected


class PHYEngine:
    """Core Physical Layer Engine for single-carrier block transmissions."""

    def __init__(self, block_symbols: int = 1536):
        if not isinstance(block_symbols, int) or block_symbols <= 0:
            raise ValueError(f"block_symbols must be a positive integer, got {block_symbols}")
        self.block_symbols = block_symbols

    def evaluate_mode(
        self,
        mode: ModulationMode,
        snr_db: float,
        h: complex,
        payload_bits: np.ndarray,
        standard_noise: np.ndarray,
    ) -> BlockEvaluationResult:
        """Evaluate a single transmission block under a specified mode.

        Parameters
        ----------
        mode : ModulationMode
            The uncoded modulation mode to evaluate.
        snr_db : float
            Nominal setup Es/N0 in dB.
        h : complex
            The scalar channel coefficient constant across the block.
        payload_bits : np.ndarray
            Binary payload array of length block_symbols * mode.bits_per_symbol.
        standard_noise : np.ndarray
            CN(0, 1) standard noise vector of length block_symbols.

        Returns
        -------
        BlockEvaluationResult
            Error counts and metrics for this block.
        """
        req_bits = self.block_symbols * mode.bits_per_symbol
        if len(payload_bits) < req_bits:
            raise ValueError(f"Mode {mode.name} requires {req_bits} bits, got {len(payload_bits)}")

        tx_bits = payload_bits[:req_bits]
        tx_symbols = modulate(tx_bits, mode.bits_per_symbol)

        ch_out = apply_channel(
            transmitted_symbols=tx_symbols,
            snr_db=snr_db,
            h=h,
            standard_noise=standard_noise,
        )

        rx_bits = demodulate(ch_out.received_symbols, h=h, bits_per_symbol=mode.bits_per_symbol)
        bit_errors = int(np.count_nonzero(tx_bits != rx_bits))
        block_error = 1 if bit_errors > 0 else 0
        ber = float(bit_errors / req_bits)

        return BlockEvaluationResult(
            mode_id=mode.mode_id,
            modulation=mode.modulation,
            bits_per_symbol=mode.bits_per_symbol,
            bit_errors=bit_errors,
            total_bits=req_bits,
            block_error=block_error,
            ber=ber,
            h=h,
            snr_db=float(snr_db),
        )

    def evaluate_paired_block(
        self,
        modes: Sequence[ModulationMode],
        snr_db: float,
        h: complex,
        payload_pool: np.ndarray,
        standard_noise: np.ndarray,
    ) -> dict[int, BlockEvaluationResult]:
        """Evaluate candidate modulation modes paired on the exact same physical channel & noise.

        Ordering invariant: each mode slices its required bits from the start of payload_pool,
        and uses the identical channel coefficient h and standard_noise vector.

        Parameters
        ----------
        modes : Sequence[ModulationMode]
            List or tuple of ModulationMode instances to evaluate.
        snr_db : float
            Nominal Es/N0 in dB.
        h : complex
            Scalar complex channel coefficient constant across the block.
        payload_pool : np.ndarray
            Binary bit pool of length >= block_symbols * max(m.bits_per_symbol).
        standard_noise : np.ndarray
            CN(0, 1) noise vector of length block_symbols.

        Returns
        -------
        dict[int, BlockEvaluationResult]
            Dictionary keyed by mode_id with individual evaluation outcomes.
        """
        results: dict[int, BlockEvaluationResult] = {}
        for m in modes:
            res = self.evaluate_mode(
                mode=m,
                snr_db=snr_db,
                h=h,
                payload_bits=payload_pool,
                standard_noise=standard_noise,
            )
            results[m.mode_id] = res
        return results
