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



@dataclass
class PHYRealization:
    """Exact physical layer realization containing payload bits, fading channel, and noise.

    Reused across backends (CPU, CUDA) and precisions (FP64, FP32) to evaluate true exact-realization parity.
    """
    payload_pool: torch.Tensor  # shape: (B, s_sym * max_m), binary bits
    h: torch.Tensor             # shape: (B, 1), complex Rayleigh channel coefficients
    z: torch.Tensor             # shape: (B, s_sym), complex standard normal noise


def generate_physical_realization(
    num_blocks: int,
    symbols_per_block: int = 1536,
    max_bits_per_symbol: int = 6,
    seed: int = 42,
    dtype_real: torch.dtype = torch.float64,
    dtype_complex: torch.dtype = torch.complex128,
) -> PHYRealization:
    """Generate a single deterministic physical realization once on CPU in high precision."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    payload = torch.randint(
        0, 2, (num_blocks, symbols_per_block * max_bits_per_symbol),
        generator=gen,
        dtype=dtype_real,
    )
    # Rayleigh fading: CN(0, 1) -> Var(real) = Var(imag) = 1/2
    h_r = torch.randn((num_blocks, 1), generator=gen, dtype=dtype_real) / math.sqrt(2.0)
    h_i = torch.randn((num_blocks, 1), generator=gen, dtype=dtype_real) / math.sqrt(2.0)
    h = torch.complex(h_r, h_i)

    # Standard complex Gaussian noise: CN(0, 1) -> Var(real) = Var(imag) = 1/2
    z_r = torch.randn((num_blocks, symbols_per_block), generator=gen, dtype=dtype_real) / math.sqrt(2.0)
    z_i = torch.randn((num_blocks, symbols_per_block), generator=gen, dtype=dtype_real) / math.sqrt(2.0)
    z = torch.complex(z_r, z_i)

    return PHYRealization(payload_pool=payload, h=h, z=z)


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
    def _evaluate_payload_and_channel(
        self,
        snr_db: float,
        payload_pool: torch.Tensor,
        h: torch.Tensor,
        z: torch.Tensor,
        return_rx_bits: bool = False,
    ) -> Tuple[Dict[int, Tuple[int, int, int, float, float]], Optional[Dict[int, torch.Tensor]]]:
        """Core physical layer transmission, detection, and reduction pipeline."""
        b_size = payload_pool.shape[0]
        s_sym = self.symbols_per_block
        dev = self.device
        r_dtype = self.real_dtype
        prec_str = self.sionna_precision

        # Setup channel noise variance N0 = 1 / db_to_lin(snr_db)
        esno_lin = db_to_lin(snr_db, precision=prec_str, device=str(dev))
        n0_t = (1.0 / esno_lin).to(dtype=r_dtype, device=dev)
        sqrt_n0 = torch.sqrt(n0_t)

        abs_h_sq = torch.abs(h) ** 2
        n0_eff = torch.clamp(n0_t / abs_h_sq, min=1e-12)
        actual_noise = sqrt_n0 * z

        chunk_metrics: Dict[int, Tuple[int, int, int, float, float]] = {}
        rx_bits_map: Optional[Dict[int, torch.Tensor]] = {} if return_rx_bits else None

        mode_tensors = []
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
            if return_rx_bits and rx_bits_map is not None:
                rx_bits_map[m.mode_id] = rx_bits.cpu()

            # On-device error counts and block statistics
            err_mask = (tx_bits != rx_bits)
            bit_errors_per_block = err_mask.sum(dim=-1)
            block_errors_per_block = (bit_errors_per_block > 0)
            ber_per_block = bit_errors_per_block.to(torch.float64) / float(req_bits)

            total_errs_t = bit_errors_per_block.sum()
            total_blk_errs_t = block_errors_per_block.sum()
            mean_ber_t = ber_per_block.mean()
            m2_ber_t = ((ber_per_block - mean_ber_t) ** 2).sum()

            mode_tensors.append(torch.stack([
                total_errs_t.to(torch.float64),
                total_blk_errs_t.to(torch.float64),
                mean_ber_t,
                m2_ber_t,
            ]))

        # Single device-to-host transfer for all candidate modulations
        all_metrics = torch.stack(mode_tensors).cpu().tolist()
        for idx, m in enumerate(self.modes):
            t_errs, t_blk_errs, m_ber, m2_b = all_metrics[idx]
            req_bits = s_sym * m.bits_per_symbol
            chunk_metrics[m.mode_id] = (
                int(t_errs),
                b_size * req_bits,
                int(t_blk_errs),
                float(m_ber),
                float(m2_b),
            )

        return chunk_metrics, rx_bits_map

    @torch.inference_mode()
    def evaluate_chunk(
        self,
        snr_db: float,
        num_blocks: int,
        fading_seed: int,
        noise_seed: int,
        bit_seed: int,
    ) -> Dict[int, Tuple[int, int, int, float, float]]:
        """Evaluate a vectorized chunk of transmission blocks entirely on-device with RNG seeds."""
        b_size = num_blocks
        s_sym = self.symbols_per_block
        dev = self.device
        r_dtype = self.real_dtype
        c_dtype = self.complex_dtype
        prec_str = self.sionna_precision

        # Sample channel, noise, and bits on device
        self.gfc.torch_rng.manual_seed(fading_seed)
        h = self.gfc(batch_size=b_size).squeeze().unsqueeze(-1).to(dtype=c_dtype, device=dev)

        gen_noise = torch.Generator(device=dev).manual_seed(noise_seed) if dev.type == "cuda" else torch.Generator().manual_seed(noise_seed)
        z = complex_normal([b_size, s_sym], precision=prec_str, device=str(dev), generator=gen_noise)

        gen_bits = torch.Generator(device=dev).manual_seed(bit_seed) if dev.type == "cuda" else torch.Generator().manual_seed(bit_seed)
        payload_pool = torch.randint(
            0, 2, (b_size, s_sym * self.max_m),
            generator=gen_bits,
            dtype=r_dtype,
            device=dev,
        )

        metrics, _ = self._evaluate_payload_and_channel(snr_db, payload_pool, h, z, return_rx_bits=False)
        return metrics

    @torch.inference_mode()
    def evaluate_realization(
        self,
        snr_db: float,
        realization: PHYRealization,
        return_rx_bits: bool = False,
    ) -> Union[
        Dict[int, Tuple[int, int, int, float, float]],
        Tuple[Dict[int, Tuple[int, int, int, float, float]], Dict[int, torch.Tensor]],
    ]:
        """Evaluate an externally supplied exact physical realization.

        Ensures bit-for-bit physical parity across hardware backends and precisions.
        """
        dev = self.device
        h_dev = realization.h.to(device=dev, dtype=self.complex_dtype)
        z_dev = realization.z.to(device=dev, dtype=self.complex_dtype)
        payload_dev = realization.payload_pool.to(device=dev, dtype=self.real_dtype)

        metrics, rx_bits_map = self._evaluate_payload_and_channel(
            snr_db, payload_dev, h_dev, z_dev, return_rx_bits=return_rx_bits
        )
        if return_rx_bits and rx_bits_map is not None:
            return metrics, rx_bits_map
        return metrics
