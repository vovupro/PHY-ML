"""Slow Rayleigh block fading and AWGN channel model with Sionna 2.0 primitives.

Baseline physical channel:
    y = h * x + n

Where:
    - x is a block of transmission symbols with unit average energy (Es = 1).
    - h is a single complex scalar channel coefficient constant over the entire block.
      Generated using Sionna's canonical RayleighBlockFading for 1D Rayleigh baseline.
      For 2D experimental conditioning sweeps (|h| fixed), samples random phase theta.
    - n is complex additive white Gaussian noise with variance N0 per complex symbol,
      generated via Sionna's complex_normal().
    - SNR convention: snr_db is nominal/setup Es/N0 in dB.
      With Es = 1, N0 = 1 / db_to_lin(snr_db).
    - End-to-end Torch tensor operations throughout the physical channel.
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional
import numpy as np
import torch
from sionna.phy.channel import RayleighBlockFading
from sionna.phy.utils import complex_normal, db_to_lin


@dataclass(frozen=True)
class BlockRNG:
    """Explicit Torch Generator streams for a single transmission block."""
    bit_rng: torch.Generator
    fading_rng: torch.Generator
    noise_rng: torch.Generator


def make_block_rng(master_seed: int, block_id: int) -> BlockRNG:
    """Create reproducible, independent Torch Generator streams for a specific block.

    Derives three child generators from (master_seed, block_id) deterministically.
    Same (master_seed, block_id) strictly reproduces the same streams.
    Different block_id produces independent streams.
    """
    if master_seed < 0 or block_id < 0:
        raise ValueError("master_seed and block_id must be non-negative integers")
    base = (master_seed * 1_000_003 + block_id * 1_009) & 0x7FFFFFFF
    g_bit = torch.Generator().manual_seed((base + 101) & 0x7FFFFFFF)
    g_fading = torch.Generator().manual_seed((base + 202) & 0x7FFFFFFF)
    g_noise = torch.Generator().manual_seed((base + 303) & 0x7FFFFFFF)
    return BlockRNG(bit_rng=g_bit, fading_rng=g_fading, noise_rng=g_noise)


@dataclass(frozen=True)
class ChannelOutput:
    """Result of passing a transmission block through the physical channel."""
    received_symbols: torch.Tensor  # y = h * x + n, 1D complex128 torch.Tensor
    h: torch.Tensor                 # scalar complex128 torch.Tensor
    noise_variance: torch.Tensor    # scalar float64 torch.Tensor: N0 = 1 / db_to_lin(snr_db)
    snr_db: float                   # nominal Es/N0 in dB
    noise: torch.Tensor             # actual complex noise added, 1D complex128 torch.Tensor


@lru_cache(maxsize=1)
def _get_rbf_model() -> RayleighBlockFading:
    """Instantiate and cache canonical Sionna RayleighBlockFading model."""
    return RayleighBlockFading(
        num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant=1, precision="double"
    )


def generate_channel_coefficient(
    generator: Optional[torch.Generator] = None,
    channel_type: str = "rayleigh",
    h_magnitude: Optional[float] = None,
) -> torch.Tensor:
    """Generate one scalar channel coefficient h constant for an entire block.

    Parameters
    ----------
    generator : torch.Generator, optional
        Torch random number generator for fading.
    channel_type : str, default 'rayleigh'
        'rayleigh' or 'awgn'.
    h_magnitude : float, optional
        If provided for Rayleigh channel, fixes |h| = h_magnitude and randomizes
        phase uniformly in [0, 2*pi). Experimental conditioning for 2D sweeps.

    Returns
    -------
    torch.Tensor
        Scalar complex128 channel coefficient h.
    """
    ch = channel_type.lower().strip()
    if ch == "awgn":
        return torch.tensor(1.0 + 0.0j, dtype=torch.complex128)
    if ch == "rayleigh":
        if h_magnitude is not None:
            if h_magnitude < 0:
                raise ValueError("h_magnitude must be non-negative")
            theta = torch.rand(1, generator=generator, dtype=torch.float64).item() * (2.0 * np.pi)
            return torch.tensor(
                complex(h_magnitude * np.cos(theta), h_magnitude * np.sin(theta)),
                dtype=torch.complex128,
            )
        # Canonical Sionna RayleighBlockFading primitive
        rbf = _get_rbf_model()
        if generator is not None:
            seed = torch.randint(0, 2**31 - 1, (1,), generator=generator).item()
            rbf.torch_rng.manual_seed(seed)
        a, _ = rbf(batch_size=1, num_time_steps=1)
        return a.squeeze().to(torch.complex128)

    raise ValueError(f"Unsupported channel_type: {channel_type}. Supported: 'rayleigh', 'awgn'")


def generate_standard_noise(
    num_symbols: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate unit-variance standard complex Gaussian noise CN(0, 1) using Sionna."""
    if num_symbols < 1:
        raise ValueError("num_symbols must be >= 1")
    return complex_normal([num_symbols], precision="double", generator=generator)


def apply_channel(
    transmitted_symbols: torch.Tensor,
    snr_db: float,
    channel_type: str = "rayleigh",
    h: Optional[torch.Tensor] = None,
    h_magnitude: Optional[float] = None,
    noise_generator: Optional[torch.Generator] = None,
    fading_generator: Optional[torch.Generator] = None,
    standard_noise: Optional[torch.Tensor] = None,
) -> ChannelOutput:
    """Pass a block of transmitted symbols through a slow block fading / AWGN channel.

    Physical Model:
        y[k] = h * x[k] + n[k],  k = 0, ..., N - 1

    Parameters
    ----------
    transmitted_symbols : torch.Tensor
        1D complex128 torch.Tensor of transmitted symbols (unit average energy Es = 1).
    snr_db : float
        Nominal setup Es/N0 in dB. N0 = 1 / db_to_lin(snr_db).
    channel_type : str, default 'rayleigh'
        'rayleigh' or 'awgn'. Ignored if h is explicitly provided.
    h : torch.Tensor, optional
        Pre-determined channel coefficient. If None, generated using fading_generator.
    h_magnitude : float, optional
        Fixed |h| magnitude for 2D conditioning (Rayleigh channel only).
    noise_generator : torch.Generator, optional
        RNG for noise generation. Required if standard_noise is None.
    fading_generator : torch.Generator, optional
        RNG for fading generation. Required if h is None.
    standard_noise : torch.Tensor, optional
        Pre-generated CN(0, 1) standard noise vector for paired evaluations.

    Returns
    -------
    ChannelOutput
        Dataclass containing received symbols y, channel coefficient h,
        noise variance N0, nominal snr_db, and the added noise n.
    """
    x = torch.as_tensor(transmitted_symbols, dtype=torch.complex128)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("transmitted_symbols must be a non-empty 1D tensor")

    # Determine scalar h constant over the entire block
    if h is None:
        h_val = generate_channel_coefficient(
            generator=fading_generator,
            channel_type=channel_type,
            h_magnitude=h_magnitude,
        )
    else:
        h_val = torch.as_tensor(h, dtype=torch.complex128).squeeze()

    # Determine noise variance via Sionna db_to_lin: N0 = 1 / db_to_lin(snr_db)
    esno_lin = db_to_lin(snr_db, precision="double")
    n0 = 1.0 / esno_lin

    if standard_noise is not None:
        std_n = torch.as_tensor(standard_noise, dtype=torch.complex128)
        if std_n.shape != x.shape:
            raise ValueError(f"standard_noise shape {std_n.shape} does not match symbols shape {x.shape}")
        actual_noise = torch.sqrt(n0) * std_n
    else:
        std_n = generate_standard_noise(len(x), generator=noise_generator)
        actual_noise = torch.sqrt(n0) * std_n

    # Physical model: y = h * x + n (one scalar h for the entire block)
    y = h_val * x + actual_noise

    return ChannelOutput(
        received_symbols=y,
        h=h_val,
        noise_variance=n0,
        snr_db=float(snr_db),
        noise=actual_noise,
    )
