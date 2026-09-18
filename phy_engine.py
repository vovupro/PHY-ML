"""Physical layer (PHY) engine wrapping Sionna 2.0 primitives.

Canonical PHY modem backend:
    - Modulation: BPSK (1 bit/sym), QPSK (2 bits/sym), 16-QAM (4 bits/sym), 64-QAM (6 bits/sym).
    - Constellation mapping & demapping are delegated entirely to Sionna 2.0 (PyTorch backend).
    - Receiver architecture:
          y  ->  divide/equalize by true h  ->  Sionna Demapper  ->  hard bits
    - Coherent ML detection with perfect CSI baseline.
    - Paired physical evaluation across candidate modulation modes on identical channel and noise realizations.
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence, Tuple
import numpy as np
import torch
from sionna.phy.mapping import Mapper, Demapper, Constellation

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


# Convenient alias requested by thesis architecture
Mode = ModulationMode

# Active canonical catalog: uncoded BPSK, QPSK, 16QAM, 64QAM
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


@lru_cache(maxsize=4)
def _get_sionna_modem(bits_per_symbol: int) -> Tuple[Constellation, Mapper, Demapper]:
    """Instantiate and cache canonical Sionna 2.0 Constellation, Mapper, and Demapper.

    BPSK uses PAM (1 bit), QPSK/16QAM/64QAM use QAM (2, 4, 6 bits).
    Precision is set to 'double' for 64-bit numerical stability.
    """
    m = bits_per_symbol
    if m not in (1, 2, 4, 6):
        raise ValueError(f"Supported bits_per_symbol: 1, 2, 4, 6. Got {m}")

    constellation_type = "pam" if m == 1 else "qam"
    const = Constellation(constellation_type, m, precision="double")
    mapper = Mapper(constellation=const, precision="double")
    demapper = Demapper("app", constellation=const, hard_out=True, precision="double")
    return const, mapper, demapper


def constellation(bits_per_symbol: int) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve unit-energy constellation points and bit labels from Sionna 2.0.

    Parameters
    ----------
    bits_per_symbol : int
        1 (BPSK), 2 (QPSK), 4 (16-QAM), or 6 (64-QAM).

    Returns
    -------
    points : np.ndarray
        Complex128 array of length 2^m.
    labels : np.ndarray
        Int32 array of shape (2^m, m) matching Sionna's bit indexing.
    """
    const, _, _ = _get_sionna_modem(bits_per_symbol)
    m = bits_per_symbol
    points = const.points.detach().cpu().numpy().astype(np.complex128)
    labels = ((np.arange(2**m)[:, None] >> np.arange(m - 1, -1, -1)) & 1).astype(np.int32)
    points.setflags(write=False)
    labels.setflags(write=False)
    return points, labels


def modulate(bits: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    """Map binary bits to unit-energy complex modulation symbols using Sionna 2.0 Mapper.

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

    _, mapper, _ = _get_sionna_modem(m)
    bits_tensor = torch.from_numpy(b.astype(np.float64))
    symbols_tensor = mapper(bits_tensor)
    symbols = symbols_tensor.detach().cpu().numpy().astype(np.complex128)
    symbols.setflags(write=False)
    return symbols


def demodulate(
    received_symbols: np.ndarray,
    h: complex,
    bits_per_symbol: int,
    n0: float = 1.0,
) -> np.ndarray:
    """Coherent demodulation using true channel equalization and Sionna 2.0 Demapper.

    Architecture:
        y  ->  divide/equalize by true h (y_eq = y / h)  ->  Sionna Demapper(y_eq, N0_eff)  ->  hard bits

    Parameters
    ----------
    received_symbols : np.ndarray
        1D complex array of received symbols y = h * x + n.
    h : complex
        Scalar complex channel coefficient known to the coherent receiver.
    bits_per_symbol : int
        Modulation order (1, 2, 4, or 6).
    n0 : float, default 1.0
        Channel noise variance N0 = 10^(-snr_db / 10).

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
    _, _, demapper = _get_sionna_modem(m)

    h_scalar = complex(h)
    abs_h_sq = abs(h_scalar) ** 2

    # Zero gain edge-case: cannot equalize, return default zeros
    if abs_h_sq < 1e-15:
        detected = np.zeros(len(y) * m, dtype=np.int32)
        detected.setflags(write=False)
        return detected

    # Coherent equalization by true scalar h: y_eq = y / h
    y_eq = y / h_scalar

    # Effective noise variance after equalization: N0_eff = N0 / |h|^2
    no_eff_val = max(float(n0) / abs_h_sq, 1e-12)

    y_eq_t = torch.as_tensor(y_eq, dtype=torch.complex128)
    no_eff_t = torch.tensor(no_eff_val, dtype=torch.float64)

    detected_t = demapper(y_eq_t, no_eff_t)
    detected = detected_t.detach().cpu().numpy().astype(np.int32)
    detected.setflags(write=False)
    return detected


class PHYEngine:
    """Core Physical Layer Engine wrapping Sionna 2.0 for single-carrier block transmissions."""

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

        rx_bits = demodulate(
            ch_out.received_symbols,
            h=h,
            bits_per_symbol=mode.bits_per_symbol,
            n0=ch_out.noise_variance,
        )
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
