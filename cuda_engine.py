"""High-throughput CUDA batch hot-path engine with online Chan/Welford statistics.

Architecture:
    - Complete hot path executed on device (GPU or CPU):
      bits -> mapper -> symbols -> physical channel y = h*x + sqrt(N0)*z
      -> coherent equalization y_eq = y/h -> demapper -> error masks -> block BER statistics.
    - Zero CPU<->GPU copies inside inner per-block operations.
    - Vectorized chunk processing over B blocks simultaneously.
    - Modems and channel models pre-instantiated and cached on device.
    - Online block-level Chan-Welford parallel variance accumulation.
    - Full support for 'cuda' and 'cpu' backends, and 'double' and 'single' precisions.
"""
from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
import sionna
from sionna.phy.utils import complex_normal, db_to_lin

from channel import _get_flat_fading_model
from phy_engine import ModulationMode, MODES, _get_sionna_modem


@dataclass
class OnlineBlockStats:
    """Online accumulator for block-level Monte Carlo statistics using Chan's parallel variance algorithm."""
    bits_per_symbol: int
    count: int = 0                  # Total blocks N
    bit_errors: int = 0             # Total bit errors
    total_bits: int = 0             # Total bits evaluated
    block_errors: int = 0           # Total erroneous blocks (at least 1 bit error)
    mean_ber: float = 0.0           # Online mean of per-block BER
    m2_ber: float = 0.0             # Online sum of squared differences from mean: M2 = sum((x_i - mean)^2)

    def update_chunk(
        self,
        chunk_blocks: int,
        chunk_errors: int,
        chunk_total_bits: int,
        chunk_block_errors: int,
        chunk_mean_ber: float,
        chunk_m2_ber: float,
    ) -> None:
        """Merge a batch/chunk of block observations using Chan's parallel Welford algorithm."""
        n_a = self.count
        n_b = int(chunk_blocks)
        if n_b <= 0:
            return

        self.bit_errors += int(chunk_errors)
        self.total_bits += int(chunk_total_bits)
        self.block_errors += int(chunk_block_errors)

        if n_a == 0:
            self.count = n_b
            self.mean_ber = float(chunk_mean_ber)
            self.m2_ber = float(chunk_m2_ber)
            return

        # Chan et al. (1979) parallel update
        n_ab = n_a + n_b
        delta = float(chunk_mean_ber) - self.mean_ber
        self.mean_ber += delta * (n_b / n_ab)
        self.m2_ber += float(chunk_m2_ber) + (delta ** 2) * (n_a * n_b / n_ab)
        self.count = n_ab

    @property
    def ber(self) -> float:
        """Macro BER = total_bit_errors / total_bits."""
        return float(self.bit_errors / self.total_bits) if self.total_bits > 0 else 0.0

    @property
    def bler(self) -> float:
        """Block Error Rate = block_errors / total_blocks."""
        return float(self.block_errors / self.count) if self.count > 0 else 0.0

    @property
    def var_block_ber(self) -> float:
        """Unbiased sample variance of per-block BER."""
        if self.count <= 1:
            return 0.0
        return max(0.0, float(self.m2_ber / (self.count - 1)))

    @property
    def std_block_ber(self) -> float:
        """Sample standard deviation of per-block BER."""
        return math.sqrt(self.var_block_ber)

    @property
    def se_block_ber(self) -> float:
        """Standard Error of the mean per-block BER = std / sqrt(N)."""
        if self.count <= 0:
            return 0.0
        return float(self.std_block_ber / math.sqrt(self.count))


class BatchPHYEngine:
    """High-throughput vectorized Physical Layer engine for GPU and CPU backends."""

    def __init__(
        self,
        symbols_per_block: int = 1536,
        modes: Sequence[ModulationMode] = MODES,
        device: Union[torch.device, str] = "cuda:0",
        precision: str = "double",
    ):
        self.symbols_per_block = symbols_per_block
        self.modes = tuple(modes)
        self.device = torch.device(device)
        self.precision = precision.lower().strip()

        if self.precision == "double":
            self.real_dtype = torch.float64
            self.complex_dtype = torch.complex128
            self.sionna_precision = "double"
        elif self.precision == "single":
            self.real_dtype = torch.float32
            self.complex_dtype = torch.complex64
            self.sionna_precision = "single"
        else:
            raise ValueError(f"Unsupported precision: '{precision}'. Supported: 'double', 'single'")

        # Pre-instantiate and warm modem objects on target device
        self.modems = {}
        for m in self.modes:
            const, mapper, demapper, s2b = _get_sionna_modem(
                m.bits_per_symbol,
                precision=self.sionna_precision,
                device=self.device,
            )
            self.modems[m.mode_id] = (const, mapper, demapper, s2b)

        # Pre-instantiate flat fading model on target device
        self.gfc = _get_flat_fading_model(precision=self.sionna_precision, device=self.device)
        self.max_m = max(m.bits_per_symbol for m in self.modes)

    @torch.inference_mode()
    def evaluate_chunk(
        self,
        snr_db: float,
        num_blocks: int,
        fading_seed: int,
        noise_seed: int,
        bit_seed: int,
    ) -> Dict[int, Tuple[int, int, int, float, float]]:
        """Evaluate a vectorized chunk of transmission blocks entirely on-device.

        Parameters
        ----------
        snr_db : float
            Nominal setup Es/N0 in dB.
        num_blocks : int
            Number of blocks in this chunk batch (B).
        fading_seed, noise_seed, bit_seed : int
            Deterministic seeds for independent RNG streams.

        Returns
        -------
        results : Dict[mode_id, (chunk_errors, chunk_total_bits, chunk_block_errors, chunk_mean_ber, chunk_m2_ber)]
            Compact summary metrics transferred to host CPU.
        """
        b_size = num_blocks
        s_sym = self.symbols_per_block
        dev = self.device
        r_dtype = self.real_dtype
        c_dtype = self.complex_dtype
        prec_str = self.sionna_precision

        # 1. Setup channel noise variance N0 = 1 / db_to_lin(snr_db)
        esno_lin = db_to_lin(snr_db, precision=prec_str, device=str(dev))
        n0_t = (1.0 / esno_lin).to(dtype=r_dtype, device=dev)
        sqrt_n0 = torch.sqrt(n0_t)

        # 2. Sample independent Rayleigh flat fading coefficients on device: Shape (B, 1)
        self.gfc.torch_rng.manual_seed(fading_seed)
        h = self.gfc(batch_size=b_size).squeeze().unsqueeze(-1).to(dtype=c_dtype, device=dev)

        # 3. Sample standardized complex Gaussian noise CN(0, 1) on device: Shape (B, s_sym)
        gen_noise = torch.Generator(device=dev).manual_seed(noise_seed) if dev.type == "cuda" else torch.Generator().manual_seed(noise_seed)
        z = complex_normal([b_size, s_sym], precision=prec_str, device=str(dev), generator=gen_noise)

        # 4. Sample common bit payload pool on device: Shape (B, s_sym * max_m)
        gen_bits = torch.Generator(device=dev).manual_seed(bit_seed) if dev.type == "cuda" else torch.Generator().manual_seed(bit_seed)
        payload_pool = torch.randint(
            0, 2, (b_size, s_sym * self.max_m),
            generator=gen_bits,
            dtype=r_dtype,
            device=dev,
        )

        # 5. Precompute channel terms on device
        abs_h_sq = torch.abs(h) ** 2
        n0_eff = torch.clamp(n0_t / abs_h_sq, min=1e-12)
        actual_noise = sqrt_n0 * z

        chunk_metrics: Dict[int, Tuple[int, int, int, float, float]] = {}

        # 6. Paired evaluation across candidate modulations
        for m in self.modes:
            req_bits = s_sym * m.bits_per_symbol
            tx_bits = payload_pool[:, :req_bits]

            _, mapper, demapper, _ = self.modems[m.mode_id]
            tx_symbols = mapper(tx_bits)

            # Physical channel equation: y = h * x + sqrt(N0) * z
            y = h * tx_symbols + actual_noise

            # Coherent equalization: y_eq = y / h
            y_eq = y / h

            # Sionna hard APP demapping
            rx_bits = demapper(y_eq, n0_eff)

            # On-device error counts and block statistics
            err_mask = (tx_bits != rx_bits)
            bit_errors_per_block = err_mask.sum(dim=-1)
            block_errors_per_block = (bit_errors_per_block > 0)
            ber_per_block = bit_errors_per_block.to(torch.float64) / float(req_bits)

            # Compact reduction on device
            total_errs = int(bit_errors_per_block.sum().item())
            total_blk_errs = int(block_errors_per_block.sum().item())
            mean_ber = float(ber_per_block.mean().item())
            m2_ber = float(((ber_per_block - mean_ber) ** 2).sum().item())
            total_bits_chunk = b_size * req_bits

            chunk_metrics[m.mode_id] = (total_errs, total_bits_chunk, total_blk_errs, mean_ber, m2_ber)

        return chunk_metrics
