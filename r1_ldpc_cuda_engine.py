"""R1: High-Throughput Batched CUDA LDPC Coded PHY Engine.

Dedicated CUDA implementation for R1 5G NR LDPC-coded transmission under
slow Rayleigh flat block fading with perfect coherent CSI.

Architectural Principles:
- Full on-device execution: LDPC encoder, constellation mapper, Rayleigh channel,
  perfect-CSI equalizer, soft APP demapper, and LDPC BP decoder execute entirely on GPU.
- On-device error reduction: bit error and codeword error reductions are performed on GPU
  using Torch tensor ops, eliminating PCIe transfer of raw codeword tensors in the hot loop.
- Paired physical realizations: shared (h, z) channel states are allocated and applied on device,
  guaranteeing that candidate actions evaluate on bit-identical physical fading and noise.
- Precision: FP64 (double precision) is the scientific baseline. FP32 is optionally supported.
- Graceful device fallback: automatically falls back to CPU if CUDA is unavailable.
"""
from dataclasses import dataclass
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from sionna.phy.utils import complex_normal, db_to_lin

from ldpc_phy import (
    ACTION_BY_NAME,
    BLOCK_SYMBOLS_DEFAULT,
    CodedAction,
    CodedBlockStats,
    CodedModem,
    PairedChannelRealization,
    R1_INITIAL_ACTIONS,
    generate_paired_channel_realization,
)


@dataclass(frozen=True)
class CUDABatchResult:
    """Outcome of one batched evaluation on CUDA."""
    action_name: str
    snr_db: float
    num_blocks: int
    info_bits_per_block: int
    coded_bits_per_block: int
    total_info_bits: int
    total_coded_bits: int
    info_bit_errors: int
    info_ber: float
    block_errors: int
    bler: float
    per_block_bit_errors: Tuple[int, ...]
    wall_clock_seconds: float
    device: str
    precision: str


class CUDACodedPHYEngine:
    """High-performance batched engine executing R1 LDPC transmission on CUDA/GPU."""

    def __init__(
        self,
        block_symbols: int = BLOCK_SYMBOLS_DEFAULT,
        num_iter: int = 20,
        cn_update: str = "boxplus-phi",
        demapping_method: str = "app",
        precision: str = "double",
        device: Optional[Union[torch.device, str]] = None,
    ):
        if device is None:
            self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_str = str(device)

        self.device = torch.device(self.device_str)
        self.block_symbols = block_symbols
        self.num_iter = num_iter
        self.cn_update = cn_update
        self.demapping_method = demapping_method
        self.precision = precision.lower()

        if self.precision not in ("double", "single"):
            raise ValueError(f"Precision must be 'double' or 'single', got {precision}")

        self.r_dtype = torch.float64 if self.precision == "double" else torch.float32
        self.c_dtype = torch.complex128 if self.precision == "double" else torch.complex64

        # Cache of initialized modems per action
        self._modem_cache: Dict[str, CodedModem] = {}

    def get_modem(self, action: CodedAction) -> CodedModem:
        """Retrieve or instantiate device modem for action."""
        key = f"{action.name}_{self.num_iter}_{self.precision}_{self.device_str}"
        if key not in self._modem_cache:
            self._modem_cache[key] = CodedModem(
                action=action,
                block_symbols=self.block_symbols,
                num_iter=self.num_iter,
                cn_update=self.cn_update,
                demapping_method=self.demapping_method,
                precision=self.precision,
                device=self.device,
            )
        return self._modem_cache[key]

    def run_batch(
        self,
        action: CodedAction,
        snr_db: float,
        num_blocks: int,
        paired_realization: Optional[PairedChannelRealization] = None,
        master_seed: Optional[int] = None,
    ) -> CUDABatchResult:
        """Execute one complete batched coded transmission on device.

        Parameters
        ----------
        action : CodedAction
            Modulation and code rate action.
        snr_db : float
            Nominal Es/N0 in dB.
        num_blocks : int
            Number of transmission blocks (codewords) in the batch.
        paired_realization : PairedChannelRealization, optional
            Pre-allocated shared physical channel realization (h, z).
        master_seed : int, optional
            Seed for deterministic generation.

        Returns
        -------
        CUDABatchResult
            Structured result containing error counts, rates, and timing.
        """
        modem = self.get_modem(action)
        t0 = time.perf_counter()

        # 1. Channel realization (h, z)
        if paired_realization is not None:
            h = paired_realization.h.to(dtype=self.c_dtype, device=self.device)
            z = paired_realization.z.to(dtype=self.c_dtype, device=self.device)
        else:
            realization = generate_paired_channel_realization(
                num_blocks=num_blocks,
                block_symbols=self.block_symbols,
                master_seed=master_seed,
                channel_type="rayleigh",
                precision=self.precision,
                device=self.device,
            )
            h = realization.h
            z = realization.z

        # 2. Draw information payload bits on separate stream
        if master_seed is not None:
            p_gen = torch.Generator(device="cpu").manual_seed(((master_seed * 1_000_003) + 101) & 0x7FFFFFFF)
            info_bits = torch.randint(0, 2, (num_blocks, modem.k), generator=p_gen, device="cpu").to(dtype=self.r_dtype, device=self.device)
        else:
            info_bits = torch.randint(0, 2, (num_blocks, modem.k), device=self.device).to(self.r_dtype)

        # 3. LDPC Encode & Constellation Map on device
        coded_bits = modem.encode(info_bits)
        x = modem.modulate(coded_bits)

        # 4. Physical Channel: y = h * x + sqrt(N0) * z
        n0_val = 1.0 / db_to_lin(snr_db)
        n0 = torch.as_tensor(n0_val, dtype=self.r_dtype, device=self.device)
        noise = torch.sqrt(n0) * z
        y = h * x + noise

        # 5. Coherent Equalization & Soft Demapping on device
        llrs = modem.equalize_and_demap_soft(y, h, n0)

        # 6. LDPC Decoding on device
        u_hat = modem.decode(llrs)

        # 7. On-device error reduction
        bit_diffs = (info_bits != u_hat).to(torch.int64)
        per_block_errs = bit_diffs.sum(dim=-1)
        total_bit_errors = int(per_block_errs.sum().item())
        block_errors = int((per_block_errs > 0).sum().item())

        per_block_list = tuple(int(e) for e in per_block_errs.cpu().tolist())
        elapsed_s = time.perf_counter() - t0

        total_info_bits = num_blocks * modem.k
        total_coded_bits = num_blocks * modem.n
        info_ber = float(total_bit_errors) / float(total_info_bits) if total_info_bits > 0 else 0.0
        bler = float(block_errors) / float(num_blocks) if num_blocks > 0 else 0.0

        return CUDABatchResult(
            action_name=action.name,
            snr_db=float(snr_db),
            num_blocks=num_blocks,
            info_bits_per_block=modem.k,
            coded_bits_per_block=modem.n,
            total_info_bits=total_info_bits,
            total_coded_bits=total_coded_bits,
            info_bit_errors=total_bit_errors,
            info_ber=info_ber,
            block_errors=block_errors,
            bler=bler,
            per_block_bit_errors=per_block_list,
            wall_clock_seconds=elapsed_s,
            device=self.device_str,
            precision=self.precision,
        )

    def evaluate_action_at_snr(
        self,
        action: CodedAction,
        snr_db: float,
        total_blocks: int,
        batch_size: int = 50,
        master_seed: int = 20260920,
    ) -> CodedBlockStats:
        """Evaluate an action across multiple batches, accumulating exact block statistics."""
        modem = self.get_modem(action)
        stats = CodedBlockStats(action, snr_db, modem.k, modem.n, self.block_symbols)

        remaining = total_blocks
        batch_idx = 0

        while remaining > 0:
            current_batch = min(remaining, batch_size)
            batch_seed = (master_seed + batch_idx * 10007) & 0x7FFFFFFF

            res = self.run_batch(
                action=action,
                snr_db=snr_db,
                num_blocks=current_batch,
                master_seed=batch_seed,
            )

            for err in res.per_block_bit_errors:
                stats.update(err)

            remaining -= current_batch
            batch_idx += 1

        return stats
