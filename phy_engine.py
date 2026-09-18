"""Physical layer (PHY) engine with pure Torch tensors and Sionna 2.0 primitives.

Canonical PHY modem backend:
    - Modulation: BPSK (1 bit/sym), QPSK (2 bits/sym), 16-QAM (4 bits/sym), 64-QAM (6 bits/sym).
    - Constellation mapping & demapping are delegated entirely to Sionna 2.0 (PyTorch backend).
    - End-to-end Torch tensor pipeline inside the PHY core:
          bits: torch.Tensor
            ↓
          Sionna Mapper
            ↓
          x: torch.complex128
            ↓
          Channel y = h * x + n: torch.complex128
            ↓
          Equalization y_eq = y / h: torch.complex128
            ↓
          Sionna Demapper
            ↓
          bits_hat: torch.Tensor
            ↓
          Sionna error metrics (count_errors, count_block_errors, compute_ber)
    - Coherent ML detection with perfect CSI baseline.
    - Paired physical evaluation across candidate modulation modes on identical channel and noise realizations.
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence, Tuple, Union
import numpy as np
import torch
from sionna.phy.mapping import Mapper, Demapper, Constellation, SymbolInds2Bits

from channel import apply_channel, ChannelOutput
from metrics import count_errors, count_block_errors, compute_ber


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


def validate_binary_tensor(bits: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Ensure input is a 1D Torch tensor of binary values (0.0 or 1.0)."""
    if isinstance(bits, np.ndarray):
        t = torch.from_numpy(bits)
    else:
        t = torch.as_tensor(bits)
    if t.ndim != 1 or t.numel() == 0:
        raise ValueError(f"Expected non-empty 1D tensor, got shape={tuple(t.shape)}")
    return t.to(dtype=torch.float64)


@lru_cache(maxsize=4)
def _get_sionna_modem(bits_per_symbol: int) -> Tuple[Constellation, Mapper, Demapper, SymbolInds2Bits]:
    """Instantiate and cache canonical Sionna 2.0 Constellation, Mapper, Demapper, and SymbolInds2Bits.

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
    s2b = SymbolInds2Bits(num_bits_per_symbol=m)
    return const, mapper, demapper, s2b


def constellation(bits_per_symbol: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Retrieve unit-energy constellation points and bit labels from Sionna 2.0.

    Parameters
    ----------
    bits_per_symbol : int
        1 (BPSK), 2 (QPSK), 4 (16-QAM), or 6 (64-QAM).

    Returns
    -------
    points : torch.Tensor
        1D complex128 torch.Tensor of length 2^m.
    labels : torch.Tensor
        2D float64 torch.Tensor of shape (2^m, m) generated via Sionna SymbolInds2Bits.
    """
    const, _, _, s2b = _get_sionna_modem(bits_per_symbol)
    m = bits_per_symbol
    points = const.points
    inds = torch.arange(2**m)
    labels = s2b(inds).to(torch.float64)
    return points, labels


def modulate(bits: Union[torch.Tensor, np.ndarray], bits_per_symbol: int) -> torch.Tensor:
    """Map binary bits to unit-energy complex modulation symbols using Sionna 2.0 Mapper.

    Parameters
    ----------
    bits : torch.Tensor or np.ndarray
        1D binary bits (0 or 1). Length must be divisible by bits_per_symbol.
    bits_per_symbol : int
        Modulation order (1, 2, 4, or 6).

    Returns
    -------
    symbols : torch.Tensor
        1D complex128 torch.Tensor of length len(bits) // bits_per_symbol.
    """
    b = validate_binary_tensor(bits)
    m = bits_per_symbol
    if len(b) % m != 0:
        raise ValueError(f"Bit length {len(b)} must divide evenly into symbols of size {m}")

    _, mapper, _, _ = _get_sionna_modem(m)
    return mapper(b)


def demodulate(
    received_symbols: Union[torch.Tensor, np.ndarray],
    h: Union[torch.Tensor, complex],
    bits_per_symbol: int,
    n0: Union[torch.Tensor, float] = 1.0,
) -> torch.Tensor:
    """Coherent demodulation using true channel equalization and Sionna 2.0 Demapper.

    Architecture:
        y  ->  divide/equalize by true h (y_eq = y / h)  ->  Sionna Demapper(y_eq, N0_eff)  ->  hard bits

    Parameters
    ----------
    received_symbols : torch.Tensor or np.ndarray
        1D complex received symbols y = h * x + n.
    h : torch.Tensor or complex
        Scalar complex channel coefficient known to the coherent receiver.
    bits_per_symbol : int
        Modulation order (1, 2, 4, or 6).
    n0 : torch.Tensor or float, default 1.0
        Channel noise variance N0.

    Returns
    -------
    detected_bits : torch.Tensor
        1D float64 torch.Tensor of detected bits.
    """
    y = torch.as_tensor(received_symbols, dtype=torch.complex128)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("received_symbols must be a non-empty 1D tensor")

    m = bits_per_symbol
    _, _, demapper, _ = _get_sionna_modem(m)

    h_t = torch.as_tensor(h, dtype=torch.complex128).squeeze()
    abs_h_sq = torch.abs(h_t) ** 2

    # Zero gain edge-case: return zeros
    if abs_h_sq < 1e-15:
        return torch.zeros(len(y) * m, dtype=torch.float64)

    # Coherent equalization by true scalar h: y_eq = y / h
    y_eq = y / h_t

    # Effective noise variance after equalization: N0_eff = N0 / |h|^2
    n0_t = torch.as_tensor(n0, dtype=torch.float64)
    no_eff = torch.clamp(n0_t / abs_h_sq, min=1e-12)

    return demapper(y_eq, no_eff)


class PHYEngine:
    """Core Physical Layer Engine wrapping Sionna 2.0 with end-to-end Torch tensors."""

    def __init__(self, block_symbols: int = 1536):
        if not isinstance(block_symbols, int) or block_symbols <= 0:
            raise ValueError(f"block_symbols must be a positive integer, got {block_symbols}")
        self.block_symbols = block_symbols

    def evaluate_mode(
        self,
        mode: ModulationMode,
        snr_db: float,
        h: Union[torch.Tensor, complex],
        payload_bits: torch.Tensor,
        standard_noise: torch.Tensor,
    ) -> BlockEvaluationResult:
        """Evaluate a single transmission block under a specified mode.

        Parameters
        ----------
        mode : ModulationMode
            The uncoded modulation mode to evaluate.
        snr_db : float
            Nominal setup Es/N0 in dB.
        h : torch.Tensor or complex
            The scalar channel coefficient constant across the block.
        payload_bits : torch.Tensor
            Binary payload tensor of length >= block_symbols * mode.bits_per_symbol.
        standard_noise : torch.Tensor
            CN(0, 1) standard noise tensor of length block_symbols.

        Returns
        -------
        BlockEvaluationResult
            Error counts and metrics for this block evaluated via Sionna metrics.
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

        # Canonical Sionna error calculations
        bit_errors = int(count_errors(tx_bits, rx_bits).item())
        block_error = int(count_block_errors(tx_bits, rx_bits).item())
        ber = float(compute_ber(tx_bits, rx_bits).item())

        h_complex = complex(torch.as_tensor(h).item() if isinstance(h, torch.Tensor) and h.numel() == 1 else h)

        return BlockEvaluationResult(
            mode_id=mode.mode_id,
            modulation=mode.modulation,
            bits_per_symbol=mode.bits_per_symbol,
            bit_errors=bit_errors,
            total_bits=req_bits,
            block_error=block_error,
            ber=ber,
            h=h_complex,
            snr_db=float(snr_db),
        )

    def evaluate_paired_block(
        self,
        modes: Sequence[ModulationMode],
        snr_db: float,
        h: Union[torch.Tensor, complex],
        payload_pool: torch.Tensor,
        standard_noise: torch.Tensor,
    ) -> dict[int, BlockEvaluationResult]:
        """Evaluate candidate modulation modes paired on the exact same physical channel & noise.

        Ordering invariant: each mode slices its required bits from the start of payload_pool,
        and uses the identical channel coefficient h and standard_noise tensor.

        Parameters
        ----------
        modes : Sequence[ModulationMode]
            List or tuple of ModulationMode instances to evaluate.
        snr_db : float
            Nominal Es/N0 in dB.
        h : torch.Tensor or complex
            Scalar complex channel coefficient constant across the block.
        payload_pool : torch.Tensor
            Binary bit pool of length >= block_symbols * max(m.bits_per_symbol).
        standard_noise : torch.Tensor
            CN(0, 1) noise tensor of length block_symbols.

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
