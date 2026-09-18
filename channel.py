"""Slow Rayleigh block fading and AWGN channel model.

Baseline physical channel:
    y = h * x + n

Where:
    - x is a block of transmission symbols with unit average energy (Es = 1).
    - h is a single complex scalar channel coefficient constant over the entire block.
    - n is complex additive white Gaussian noise with variance N0 per complex symbol.
    - SNR convention: snr_db is the nominal/setup Es/N0 in dB.
      With Es = 1, N0 = 10^(-snr_db / 10).
    - For Rayleigh fading: h ~ CN(0, 1) with E[|h|^2] = 1.
    - For AWGN: h = 1.0 + 0.0j.
    - Future-proofing 2D calibration: h_magnitude can be specified to fix |h|
      while randomizing phase.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class BlockRNG:
    """Explicit RNG streams for a single transmission block."""
    bit_rng: np.random.Generator
    fading_rng: np.random.Generator
    noise_rng: np.random.Generator


def make_block_rng(master_seed: int, block_id: int) -> BlockRNG:
    """Create reproducible, independent RNG streams for a specific block.

    Derives three child streams (bits, fading, noise) from SeedSequence(master_seed, block_id).
    Same (master_seed, block_id) strictly reproduces the same streams.
    Different block_id produces independent streams.
    """
    if master_seed < 0 or block_id < 0:
        raise ValueError("master_seed and block_id must be non-negative integers")
    ss = np.random.SeedSequence([master_seed, block_id])
    children = ss.spawn(3)
    return BlockRNG(
        bit_rng=np.random.default_rng(children[0]),
        fading_rng=np.random.default_rng(children[1]),
        noise_rng=np.random.default_rng(children[2]),
    )


@dataclass(frozen=True)
class ChannelOutput:
    """Result of passing a transmission block through the physical channel."""
    received_symbols: np.ndarray  # y = h * x + n, complex128 array of length N
    h: complex                    # The exact single scalar h applied to the block
    noise_variance: float         # N0 = 10^(-snr_db / 10)
    snr_db: float                 # Nominal Es/N0 in dB
    noise: np.ndarray             # The actual noise vector n added to symbols


def generate_channel_coefficient(
    rng: np.random.Generator,
    channel_type: str = "rayleigh",
    h_magnitude: Optional[float] = None,
) -> complex:
    """Generate one scalar channel coefficient h constant for an entire block.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator for fading.
    channel_type : str, default 'rayleigh'
        'rayleigh' for circular complex Gaussian CN(0, 1), or 'awgn' for h = 1.
    h_magnitude : float, optional
        If provided for Rayleigh channel, fixes |h| = h_magnitude and randomizes
        phase uniformly in [0, 2*pi). Designed for 2D conditioning sweeps.

    Returns
    -------
    complex
        Channel coefficient h satisfying E[|h|^2] = 1 for unconstrained Rayleigh.
    """
    ch = channel_type.lower().strip()
    if ch == "awgn":
        return 1.0 + 0.0j
    if ch == "rayleigh":
        if h_magnitude is not None:
            if h_magnitude < 0:
                raise ValueError("h_magnitude must be non-negative")
            theta = rng.uniform(0.0, 2.0 * np.pi)
            return complex(h_magnitude * np.cos(theta), h_magnitude * np.sin(theta))
        hr = rng.standard_normal()
        hi = rng.standard_normal()
        return complex(hr / np.sqrt(2.0), hi / np.sqrt(2.0))
    raise ValueError(f"Unsupported channel_type: {channel_type}. Supported: 'rayleigh', 'awgn'")


def generate_standard_noise(rng: np.random.Generator, num_symbols: int) -> np.ndarray:
    """Generate unit-variance standard complex Gaussian noise CN(0, 1)."""
    if num_symbols < 1:
        raise ValueError("num_symbols must be >= 1")
    nr = rng.standard_normal(num_symbols)
    ni = rng.standard_normal(num_symbols)
    noise = (nr + 1j * ni) / np.sqrt(2.0)
    noise.setflags(write=False)
    return noise


def apply_channel(
    transmitted_symbols: np.ndarray,
    snr_db: float,
    channel_type: str = "rayleigh",
    h: Optional[complex] = None,
    h_magnitude: Optional[float] = None,
    noise_rng: Optional[np.random.Generator] = None,
    fading_rng: Optional[np.random.Generator] = None,
    standard_noise: Optional[np.ndarray] = None,
) -> ChannelOutput:
    """Pass a block of transmitted symbols through a slow block fading / AWGN channel.

    Physical Model:
        y[k] = h * x[k] + n[k],  k = 0, ..., N - 1

    Parameters
    ----------
    transmitted_symbols : np.ndarray
        1D complex array of transmitted symbols (unit average energy Es = 1).
    snr_db : float
        Nominal setup Es/N0 in dB. N0 = 10^(-snr_db / 10).
    channel_type : str, default 'rayleigh'
        'rayleigh' or 'awgn'. Ignored if h is explicitly provided.
    h : complex, optional
        Pre-determined channel coefficient. If None, generated from fading_rng.
    h_magnitude : float, optional
        Fixed |h| magnitude for 2D conditioning (Rayleigh channel only).
    noise_rng : np.random.Generator, optional
        RNG for noise generation. Required if standard_noise is None.
    fading_rng : np.random.Generator, optional
        RNG for fading generation. Required if h is None.
    standard_noise : np.ndarray, optional
        Pre-generated CN(0, 1) standard noise vector for paired evaluations.

    Returns
    -------
    ChannelOutput
        Dataclass containing received symbols y, channel coefficient h,
        noise variance N0, nominal snr_db, and the added noise n.
    """
    x = np.asarray(transmitted_symbols, dtype=np.complex128)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("transmitted_symbols must be a non-empty 1D array")
    if not np.isfinite(x).all():
        raise ValueError("transmitted_symbols contains non-finite values")
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")

    # Determine scalar h constant over the entire block
    if h is None:
        ch_lower = channel_type.lower().strip()
        if ch_lower == "awgn":
            h_val = 1.0 + 0.0j
        else:
            if fading_rng is None:
                raise ValueError("fading_rng is required when channel coefficient h is not provided for fading channels")
            h_val = generate_channel_coefficient(fading_rng, channel_type=channel_type, h_magnitude=h_magnitude)
    else:
        h_val = complex(h)

    # Determine noise: N0 = 10^(-snr_db / 10)
    n0 = float(10.0 ** (-snr_db / 10.0))

    if standard_noise is not None:
        std_n = np.asarray(standard_noise, dtype=np.complex128)
        if std_n.shape != x.shape:
            raise ValueError(f"standard_noise shape {std_n.shape} does not match symbols shape {x.shape}")
        actual_noise = np.sqrt(n0) * std_n
    else:
        if noise_rng is None:
            raise ValueError("noise_rng is required when standard_noise is not provided")
        std_n = generate_standard_noise(noise_rng, len(x))
        actual_noise = np.sqrt(n0) * std_n

    # Apply physical model: y = h * x + n (one scalar h for the entire block)
    y = h_val * x + actual_noise

    y.setflags(write=False)
    actual_noise.setflags(write=False)

    return ChannelOutput(
        received_symbols=y,
        h=h_val,
        noise_variance=n0,
        snr_db=float(snr_db),
        noise=actual_noise,
    )
