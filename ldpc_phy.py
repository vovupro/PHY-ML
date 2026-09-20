"""R1: LDPC-Coded Physical Layer (PHY) Foundation with Sionna 2.0 Primitives.

Extends the frozen R0 uncoded root baseline by introducing 5G NR LDPC channel coding
as the single communication-system enhancement, while strictly preserving:
    - Slow Rayleigh flat block fading (constant over transmission block, independent between blocks)
    - Coherent reception with perfect CSI baseline (h known at receiver)
    - Simulation setup axis: Nominal Es/N0 in dB (held constant with Es = 1)
    - Constellation mapping (BPSK, QPSK, 16QAM, 64QAM) with unit average energy
    - Pure Torch tensor operations end-to-end
    - Deterministic pseudo-random seeding discipline (SeedSequence / BlockRNG)

Coded Physical Transmission Chain:
    information bits u  [k bits]
            ↓
    Sionna LDPC5GEncoder  [3GPP TS 38.212 rate matching]
            ↓
    coded bits c  [n bits]
            ↓
    Sionna Mapper  (unit-energy constellation)
            ↓
    transmitted symbols x  [N_symbols = n // bits_per_symbol]
            ↓
    Rayleigh block channel: y = h * x + sqrt(N0) * z  where z ~ CN(0, 1), E[|z|^2] = 1
            ↓
    Coherent perfect-CSI equalization: y_eq = y / h, N0_eff = N0 / |h|^2
            ↓
    Sionna Demapper (hard_out=False): SOFT log-likelihood ratios (LLRs)
            ↓
    Sionna LDPC5GDecoder (Belief Propagation: boxplus-phi / minsum)
            ↓
    decoded information bits u_hat  [k bits]
            ↓
    Bit Error & Codeword Block Error Counters (BER_info, BLER)

Terminology & Scope Note:
    In current R1 AMC link adaptation, the transmission unit is termed:
        "LDPC information block / codeword"
    It is NOT termed a 3GPP Transport Block because current R1 intentionally DOES NOT include:
        - CRC attachment or parity checks
        - Transport block segmentation / code block concatenation
        - Hybrid ARQ (HARQ) or soft bit combining
        - Automatic Repeat reQuest (ACK/NACK)
        - Temporal feedback or retransmissions

Independent Monte Carlo Statistical Unit:
    Under slow Rayleigh flat block fading, each transmission block of N_symbols (default 1536)
    experiences an independent complex channel fading draw h ~ CN(0, 1) held strictly constant
    across all symbols of that block.
    In the canonical 1-codeword-per-block mapping (n = N_symbols * bits_per_symbol), each
    codeword experiences exactly one fading realization.
    Therefore, the CODEWORD is the independent statistical unit for reliability:
        Codeword error: decoded information block contains >= 1 erroneous information bit.
        Codeword BLER = N_error_codewords / N_total_codewords.
    Codeword BLER constitutes an independent Bernoulli trial per block.
    Information-bit BER standard error is calculated from the empirical sample standard deviation
    of per-block BERs divided by sqrt(N_blocks), accounting for intra-codeword bit error correlation.

Complex Noise Scaling Convention (Parent-Compatible with R0):
    z ~ CN(0, 1) with E[|z|^2] = 1 (generated via Sionna complex_normal).
    With symbol energy Es = 1, nominal noise variance is N0 = 1 / 10^(snr_db / 10).
    Physical channel noise is:
        n_w = sqrt(N0) * z
    giving E[|n_w|^2] = N0 * E[|z|^2] = N0.
    The factor sqrt(N0/2) must NOT be applied to complex_normal(), as complex_normal()
    already distributes variance 0.5 to real and 0.5 to imaginary components.

Fairness and SNR Axis Convention:
    The simulation axis remains strictly nominal Es/N0 in dB, preserving complete continuity
    with the frozen R0 baseline.
    The derived relationship to information-bit Eb/N0 is:
        Eb/N0 = (Es/N0) / eta
        (Eb/N0)_dB = (Es/N0)_dB - 10 * log10(eta)
    where eta = (k/n) * log2(M) is the effective spectral efficiency in information bits per symbol.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.stats
import torch
from sionna.phy.channel import GenerateFlatFadingChannel
from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
from sionna.phy.mapping import Constellation, Demapper, Mapper, SymbolInds2Bits
from sionna.phy.utils import complex_normal, db_to_lin


CONFIDENCE_K: float = 1.96
BLOCK_SYMBOLS_DEFAULT: int = 1536

MODULATION_BPS: Dict[str, int] = {
    "BPSK": 1,
    "QPSK": 2,
    "16QAM": 4,
    "64QAM": 6,
}


@dataclass(frozen=True)
class CodedAction:
    """Representation of an LDPC-coded modulation action (modulation, code_rate)."""
    modulation: str          # "BPSK", "QPSK", "16QAM", "64QAM"
    code_rate: float         # Requested code rate, e.g. 0.5, 0.6666666666666666, 0.75
    code_rate_str: str       # "1/2", "2/3", "3/4"
    name: str                # e.g. "BPSK-1/2", "QPSK-2/3"

    @property
    def bits_per_symbol(self) -> int:
        mod_upper = self.modulation.upper()
        if mod_upper not in MODULATION_BPS:
            raise ValueError(f"Unknown modulation: {self.modulation}. Supported: {list(MODULATION_BPS.keys())}")
        return MODULATION_BPS[mod_upper]

    @property
    def nominal_spectral_efficiency(self) -> float:
        """Nominal spectral efficiency Rc_requested * log2(M)."""
        return float(self.code_rate * self.bits_per_symbol)

    @property
    def spectral_efficiency(self) -> float:
        """Alias to nominal_spectral_efficiency."""
        return self.nominal_spectral_efficiency

    def get_code_params(self, block_symbols: int = BLOCK_SYMBOLS_DEFAULT) -> Tuple[int, int, float, float]:
        """Compute exact (k, n, effective_code_rate, effective_spectral_efficiency).

        Parameters
        ----------
        block_symbols : int, default 1536
            Number of complex symbols in one transmission block.

        Returns
        -------
        k : int
            Number of information bits per codeword.
        n : int
            Number of coded bits per codeword (n = block_symbols * bits_per_symbol).
        eff_rc : float
            Effective code rate k / n.
        eff_eta : float
            Effective spectral efficiency eff_rc * bits_per_symbol.
        """
        m = self.bits_per_symbol
        n = block_symbols * m
        k = int(round(n * self.code_rate))
        eff_rc = float(k) / float(n)
        eff_eta = float(eff_rc * m)
        return k, n, eff_rc, eff_eta

    def compute_eb_n0_db(self, es_n0_db: float, block_symbols: int = BLOCK_SYMBOLS_DEFAULT) -> float:
        """Derived relationship to information-bit Eb/N0: (Eb/N0)_dB = (Es/N0)_dB - 10*log10(eta_eff)."""
        _, _, _, eff_eta = self.get_code_params(block_symbols)
        if eff_eta <= 0:
            raise ValueError(f"Effective spectral efficiency must be positive, got {eff_eta}")
        return float(es_n0_db - 10.0 * math.log10(eff_eta))

    def to_dict(self, block_symbols: int = BLOCK_SYMBOLS_DEFAULT) -> Dict[str, Any]:
        k, n, eff_rc, eff_eta = self.get_code_params(block_symbols)
        return {
            "name": self.name,
            "modulation": self.modulation,
            "constellation_size": 2 ** self.bits_per_symbol,
            "bits_per_symbol": self.bits_per_symbol,
            "requested_code_rate": self.code_rate,
            "code_rate_str": self.code_rate_str,
            "actual_k": k,
            "actual_n": n,
            "effective_code_rate": eff_rc,
            "effective_spectral_efficiency": eff_eta,
            "symbols_per_block": block_symbols,
        }


# Initial validated action set suitable for correctness testing across rates 1/2, 2/3, 3/4
R1_INITIAL_ACTIONS: Tuple[CodedAction, ...] = (
    CodedAction(modulation="BPSK", code_rate=1.0 / 2.0, code_rate_str="1/2", name="BPSK-1/2"),
    CodedAction(modulation="QPSK", code_rate=1.0 / 2.0, code_rate_str="1/2", name="QPSK-1/2"),
    CodedAction(modulation="QPSK", code_rate=2.0 / 3.0, code_rate_str="2/3", name="QPSK-2/3"),
    CodedAction(modulation="16QAM", code_rate=1.0 / 2.0, code_rate_str="1/2", name="16QAM-1/2"),
    CodedAction(modulation="16QAM", code_rate=3.0 / 4.0, code_rate_str="3/4", name="16QAM-3/4"),
    CodedAction(modulation="64QAM", code_rate=1.0 / 2.0, code_rate_str="1/2", name="64QAM-1/2"),
    CodedAction(modulation="64QAM", code_rate=2.0 / 3.0, code_rate_str="2/3", name="64QAM-2/3"),
    CodedAction(modulation="64QAM", code_rate=3.0 / 4.0, code_rate_str="3/4", name="64QAM-3/4"),
)

ACTION_BY_NAME: Dict[str, CodedAction] = {a.name: a for a in R1_INITIAL_ACTIONS}


def get_ldpc_code_params(
    action: CodedAction,
    block_symbols: int = BLOCK_SYMBOLS_DEFAULT,
) -> Tuple[int, int]:
    """Compute (k, n) code parameters for a transmission block."""
    k, n, _, _ = action.get_code_params(block_symbols)
    return k, n


def analyze_action_space(
    actions: Sequence[CodedAction] = R1_INITIAL_ACTIONS,
    block_symbols: int = BLOCK_SYMBOLS_DEFAULT,
) -> Dict[str, Any]:
    """Analyze R1 candidate action space for spectral efficiencies, equal-eta conflicts, and dominance."""
    action_records = [a.to_dict(block_symbols) for a in actions]
    unique_etas = sorted(list(set(r["effective_spectral_efficiency"] for r in action_records)))

    # Identify equal-eta conflicts
    equal_eta_groups: Dict[float, List[str]] = {}
    for r in action_records:
        eta = r["effective_spectral_efficiency"]
        if eta not in equal_eta_groups:
            equal_eta_groups[eta] = []
        equal_eta_groups[eta].append(r["name"])

    # Filter to groups with > 1 action
    equal_eta_conflicts = {eta: names for eta, names in equal_eta_groups.items() if len(names) > 1}

    dominance_analysis = [
        {
            "spectral_efficiency": 3.0,
            "actions": ["16QAM-3/4", "64QAM-1/2"],
            "comparison": (
                "16QAM-3/4 (m=4, Rc=3/4) uses smaller constellation but weaker code; "
                "64QAM-1/2 (m=6, Rc=1/2) uses denser constellation but stronger rate-1/2 LDPC code. "
                "Relative BLER depends on SNR operating point and channel fading realization."
            ),
        }
    ]

    return {
        "action_records": action_records,
        "unique_spectral_efficiencies": unique_etas,
        "equal_eta_conflicts": equal_eta_conflicts,
        "dominance_analysis": dominance_analysis,
    }


@dataclass(frozen=True)
class PairedChannelRealization:
    """Shared physical channel realization (h, z) paired across candidate actions for a block batch.

    Attributes
    ----------
    h : torch.Tensor
        Complex channel coefficients [num_blocks, 1] with E[|h|^2] = 1.
    z : torch.Tensor
        Normalized complex noise vector [num_blocks, block_symbols] with E[|z|^2] = 1.
    num_blocks : int
        Number of transmission blocks in this batch.
    block_symbols : int
        Number of complex symbols per transmission block (default 1536).
    """
    h: torch.Tensor
    z: torch.Tensor
    num_blocks: int
    block_symbols: int = BLOCK_SYMBOLS_DEFAULT


def generate_paired_channel_realization(
    num_blocks: int,
    block_symbols: int = BLOCK_SYMBOLS_DEFAULT,
    master_seed: Optional[int] = None,
    channel_type: str = "rayleigh",
    precision: str = "double",
    device: Union[torch.device, str] = "cpu",
) -> PairedChannelRealization:
    """Deterministically generate shared physical channel realization (h, z) paired across actions.

    The RNG streams for fading and noise are derived independently from the master seed,
    ensuring that information payload bit drawing CANNOT perturb the channel or noise realizations.

    Parameters
    ----------
    num_blocks : int
        Number of transmission blocks.
    block_symbols : int, default 1536
        Complex symbols per block.
    master_seed : int, optional
        Seed for deterministic generation.
    channel_type : str, default 'rayleigh'
        'rayleigh' or 'awgn'.
    precision : str, default 'double'
        'double' (complex128) or 'single' (complex64).
    device : torch.device or str, default 'cpu'
        Target PyTorch device.

    Returns
    -------
    PairedChannelRealization
        Shared physical realization (h, z) to be reused across all candidate actions.
    """
    c_dtype = torch.complex128 if precision == "double" else torch.complex64
    dev_str = str(device)

    g_fading: Optional[torch.Generator] = None
    g_noise: Optional[torch.Generator] = None

    if master_seed is not None:
        base = (master_seed * 1_000_003) & 0x7FFFFFFF
        g_fading = torch.Generator(device="cpu").manual_seed((base + 202) & 0x7FFFFFFF)
        g_noise = torch.Generator(device="cpu").manual_seed((base + 303) & 0x7FFFFFFF)

    # 1. Fading coefficient h [num_blocks, 1]
    if channel_type.lower() == "awgn":
        h = torch.ones((num_blocks, 1), dtype=c_dtype, device=device)
    else:
        # Rayleigh flat fading: h ~ CN(0, 1), E[|h|^2] = 1.0
        h = complex_normal([num_blocks, 1], precision=precision, device=dev_str, generator=g_fading)
        if not isinstance(h, torch.Tensor):
            h = torch.as_tensor(h, dtype=c_dtype, device=device)

    # 2. Normalized complex noise z [num_blocks, block_symbols]
    # z ~ CN(0, 1) with E[|z|^2] = 1.0
    z = complex_normal([num_blocks, block_symbols], precision=precision, device=dev_str, generator=g_noise)
    if not isinstance(z, torch.Tensor):
        z = torch.as_tensor(z, dtype=c_dtype, device=device)

    return PairedChannelRealization(
        h=h,
        z=z,
        num_blocks=num_blocks,
        block_symbols=block_symbols,
    )


class CodedModem:
    """Manages Sionna 2.0 LDPC encoder/decoder and constellation mapper/demapper."""

    def __init__(
        self,
        action: CodedAction,
        block_symbols: int = BLOCK_SYMBOLS_DEFAULT,
        num_iter: int = 20,
        cn_update: str = "boxplus-phi",
        demapping_method: str = "app",
        precision: str = "double",
        device: Union[torch.device, str] = "cpu",
    ):
        self.action = action
        self.block_symbols = block_symbols
        self.num_iter = num_iter
        self.cn_update = cn_update
        self.demapping_method = demapping_method
        self.precision = precision
        self.device = torch.device(device)
        self.device_str = str(device)

        # Code dimension bookkeeping
        self.k, self.n, self.effective_code_rate, self.effective_spectral_efficiency = action.get_code_params(block_symbols)
        self.bits_per_symbol = action.bits_per_symbol

        # Constellation type: PAM for BPSK (1 bit), QAM for QPSK/16QAM/64QAM
        const_type = "pam" if self.bits_per_symbol == 1 else "qam"
        self.constellation = Constellation(
            const_type,
            self.bits_per_symbol,
            precision=self.precision,
            device=self.device_str,
        )

        self.mapper = Mapper(
            constellation=self.constellation,
            precision=self.precision,
            device=self.device_str,
        )

        # Soft demapper (hard_out=False returns real-valued LLRs)
        self.demapper = Demapper(
            self.demapping_method,
            constellation=self.constellation,
            hard_out=False,
            precision=self.precision,
            device=self.device_str,
        )

        # 5G NR LDPC Encoder & Decoder
        self.encoder = LDPC5GEncoder(
            k=self.k,
            n=self.n,
            precision=self.precision,
            device=self.device_str,
        )

        self.decoder = LDPC5GDecoder(
            self.encoder,
            cn_update=self.cn_update,
            num_iter=self.num_iter,
            hard_out=True,
            return_infobits=True,
            precision=self.precision,
            device=self.device_str,
        )

    def encode(self, info_bits: torch.Tensor) -> torch.Tensor:
        """Encode information bits u [..., k] into codeword c [..., n]."""
        r_dtype = torch.float64 if self.precision == "double" else torch.float32
        u = torch.as_tensor(info_bits, dtype=r_dtype, device=self.device)
        return self.encoder(u)

    def modulate(self, coded_bits: torch.Tensor) -> torch.Tensor:
        """Map binary coded bits c [..., n] to complex modulation symbols x [..., N_symbols]."""
        r_dtype = torch.float64 if self.precision == "double" else torch.float32
        c = torch.as_tensor(coded_bits, dtype=r_dtype, device=self.device)
        return self.mapper(c)

    def equalize_and_demap_soft(
        self,
        received_symbols: torch.Tensor,
        h: torch.Tensor,
        n0: Union[torch.Tensor, float],
    ) -> torch.Tensor:
        """Coherent perfect-CSI equalization and soft demapping to LLRs.

        Mathematical Model:
            y = h * x + sqrt(N0) * z
            y_eq = y / h
            N0_eff = N0 / |h|^2
            llr = Demapper(y_eq, N0_eff)

        Sign convention (Sionna native):
            LLR_j = ln( P(c_j = 1 | y) / P(c_j = 0 | y) )
            LLR < 0 -> bit 0 likely
            LLR > 0 -> bit 1 likely

        Parameters
        ----------
        received_symbols : torch.Tensor
            Complex received symbols [..., N_symbols].
        h : torch.Tensor
            Complex channel coefficient [..., 1] or scalar.
        n0 : torch.Tensor or float
            Nominal noise variance N0 = 1 / db_to_lin(snr_db).

        Returns
        -------
        llrs : torch.Tensor
            Soft log-likelihood ratios [..., n].
        """
        c_dtype = torch.complex128 if self.precision == "double" else torch.complex64
        r_dtype = torch.float64 if self.precision == "double" else torch.float32

        y = torch.as_tensor(received_symbols, dtype=c_dtype, device=self.device)
        h_t = torch.as_tensor(h, dtype=c_dtype, device=self.device)

        abs_h_sq = torch.abs(h_t) ** 2
        abs_h_sq_safe = torch.clamp(abs_h_sq, min=1e-12)

        # Equalization by scalar channel coefficient
        y_eq = y / h_t

        # Scaled noise variance after scalar division
        n0_t = torch.as_tensor(n0, dtype=r_dtype, device=self.device)
        n0_eff = torch.clamp(n0_t / abs_h_sq_safe, min=1e-12)

        return self.demapper(y_eq, n0_eff)

    def decode(self, llrs: torch.Tensor) -> torch.Tensor:
        """Decode soft LLRs [..., n] to hard information bits u_hat [..., k]."""
        r_dtype = torch.float64 if self.precision == "double" else torch.float32
        l = torch.as_tensor(llrs, dtype=r_dtype, device=self.device)
        return self.decoder(l)


@dataclass(frozen=True)
class CodedBlockEvaluationResult:
    """Outcome of evaluating one or more transmission blocks under an LDPC-coded mode."""
    action_name: str
    modulation: str
    code_rate: float
    bits_per_symbol: int
    spectral_efficiency: float
    info_bits_per_block: int
    coded_bits_per_block: int
    num_blocks: int
    total_info_bits: int
    total_coded_bits: int
    info_bit_errors: int
    info_ber: float
    block_errors: int
    bler: float
    decoder_iterations: int
    snr_db: float
    eb_n0_db: float
    channel_type: str = "rayleigh"
    per_block_bit_errors: Sequence[int] = ()


def simulate_coded_transmission(
    modem: CodedModem,
    snr_db: float,
    num_blocks: int = 1,
    channel_type: str = "rayleigh",
    paired_realization: Optional[PairedChannelRealization] = None,
    h_custom: Optional[torch.Tensor] = None,
    z_custom: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    payload_generator: Optional[torch.Generator] = None,
) -> CodedBlockEvaluationResult:
    """Simulate complete LDPC-coded transmission chain over slow Rayleigh block fading.

    Chain:
        u -> LDPC Encoder -> Mapper -> Rayleigh Channel -> Equalizer -> Soft Demapper -> Decoder -> u_hat

    Complex noise model:
        n_w = sqrt(N0) * z   with z ~ CN(0, 1), E[|z|^2] = 1

    Parameters
    ----------
    modem : CodedModem
        Configured LDPC modem.
    snr_db : float
        Nominal setup Es/N0 in dB.
    num_blocks : int, default 1
        Number of transmission blocks (codewords) to evaluate.
    channel_type : str, default 'rayleigh'
        'rayleigh' or 'awgn'.
    paired_realization : PairedChannelRealization, optional
        Pre-generated paired physical channel realization (h, z).
    h_custom : torch.Tensor, optional
        Explicit channel realizations [num_blocks, 1].
    z_custom : torch.Tensor, optional
        Explicit standard noise realizations [num_blocks, block_symbols].
    seed : int, optional
        Deterministic master seed.
    payload_generator : torch.Generator, optional
        RNG stream for information bits u.

    Returns
    -------
    CodedBlockEvaluationResult
        Structured outcome containing BER, BLER, and code bookkeeping.
    """
    dev = modem.device
    r_dtype = torch.float64 if modem.precision == "double" else torch.float32
    c_dtype = torch.complex128 if modem.precision == "double" else torch.complex64

    # 1. Payload Generator for information bits u [num_blocks, k]
    p_gen = payload_generator
    if p_gen is None and seed is not None:
        p_gen = torch.Generator(device="cpu").manual_seed(((seed * 1_000_003) + 101) & 0x7FFFFFFF)

    if p_gen is not None:
        info_bits = torch.randint(0, 2, (num_blocks, modem.k), generator=p_gen, device="cpu").to(dtype=r_dtype, device=dev)
    else:
        info_bits = torch.randint(0, 2, (num_blocks, modem.k), device=dev).to(r_dtype)

    # 2. LDPC Encode -> c [num_blocks, n]
    coded_bits = modem.encode(info_bits)

    # 3. Constellation Mapping -> x [num_blocks, N_symbols]
    x = modem.modulate(coded_bits)

    # 4. Physical Channel
    # Noise variance N0 = 1 / db_to_lin(snr_db) (Es = 1)
    n0_val = 1.0 / db_to_lin(snr_db)
    n0 = torch.as_tensor(n0_val, dtype=r_dtype, device=dev)

    if paired_realization is not None:
        h = paired_realization.h.to(dtype=c_dtype, device=dev)
        z = paired_realization.z.to(dtype=c_dtype, device=dev)
    elif h_custom is not None and z_custom is not None:
        h = torch.as_tensor(h_custom, dtype=c_dtype, device=dev)
        z = torch.as_tensor(z_custom, dtype=c_dtype, device=dev)
        if h.ndim == 1:
            h = h.unsqueeze(-1)
    else:
        realization = generate_paired_channel_realization(
            num_blocks=num_blocks,
            block_symbols=modem.block_symbols,
            master_seed=seed,
            channel_type=channel_type,
            precision=modem.precision,
            device=dev,
        )
        h = realization.h
        z = realization.z

    # Physical noise scaling: n_w = sqrt(N0) * z   (E[|z|^2] = 1, E[|n_w|^2] = N0)
    noise = torch.sqrt(n0) * z

    # Received signal y = h * x + noise
    y = h * x + noise

    # 5. Perfect-CSI Equalization & Soft Demapping -> LLRs [num_blocks, n]
    llrs = modem.equalize_and_demap_soft(y, h, n0)

    # 6. LDPC Decoding -> u_hat [num_blocks, k]
    u_hat = modem.decode(llrs)

    # 7. Error Counting
    bit_diffs = (info_bits != u_hat).to(torch.int64)
    per_block_bit_errors = bit_diffs.sum(dim=-1).cpu().numpy()
    total_bit_errors = int(np.sum(per_block_bit_errors))
    total_info_bits = num_blocks * modem.k
    total_coded_bits = num_blocks * modem.n

    info_ber = float(total_bit_errors) / float(total_info_bits)
    block_errors = int(np.sum(per_block_bit_errors > 0))
    bler = float(block_errors) / float(num_blocks)

    eb_n0_db = modem.action.compute_eb_n0_db(snr_db, modem.block_symbols)

    return CodedBlockEvaluationResult(
        action_name=modem.action.name,
        modulation=modem.action.modulation,
        code_rate=modem.effective_code_rate,
        bits_per_symbol=modem.bits_per_symbol,
        spectral_efficiency=modem.effective_spectral_efficiency,
        info_bits_per_block=modem.k,
        coded_bits_per_block=modem.n,
        num_blocks=num_blocks,
        total_info_bits=total_info_bits,
        total_coded_bits=total_coded_bits,
        info_bit_errors=total_bit_errors,
        info_ber=info_ber,
        block_errors=block_errors,
        bler=bler,
        decoder_iterations=modem.num_iter,
        snr_db=float(snr_db),
        eb_n0_db=eb_n0_db,
        channel_type=channel_type,
        per_block_bit_errors=tuple(int(e) for e in per_block_bit_errors),
    )


class CodedBlockStats:
    """Online statistical accumulator for coded block Monte Carlo trials.

    Supports candidate confidence interval methods for human review:
        - Wald normal approximation (diagnostic baseline)
        - Wilson score interval (asymmetric, stable at zero errors)
        - Clopper-Pearson exact binomial interval (conservative coverage guarantee)
    """

    def __init__(self, action: CodedAction, snr_db: float, k: int, n: int, block_symbols: int = BLOCK_SYMBOLS_DEFAULT):
        self.action = action
        self.snr_db = float(snr_db)
        self.k = k
        self.n = n
        self.block_symbols = block_symbols
        self.eb_n0_db = action.compute_eb_n0_db(snr_db, block_symbols)

        self.num_blocks: int = 0
        self.total_info_bits: int = 0
        self.total_coded_bits: int = 0
        self.info_bit_errors: int = 0
        self.block_errors: int = 0

        # For block-level empirical BER variance tracking (Welford accumulator)
        self._block_ber_mean: float = 0.0
        self._block_ber_m2: float = 0.0

    def update(self, bit_errors: int) -> None:
        """Update statistics with one completed block result."""
        self.num_blocks += 1
        self.total_info_bits += self.k
        self.total_coded_bits += self.n
        self.info_bit_errors += bit_errors

        block_err = 1 if bit_errors > 0 else 0
        self.block_errors += block_err

        # Per-block BER for variance
        block_ber = float(bit_errors) / float(self.k)
        delta = block_ber - self._block_ber_mean
        self._block_ber_mean += delta / float(self.num_blocks)
        delta2 = block_ber - self._block_ber_mean
        self._block_ber_m2 += delta * delta2

    @property
    def info_ber(self) -> float:
        if self.total_info_bits == 0:
            return 0.0
        return float(self.info_bit_errors) / float(self.total_info_bits)

    @property
    def bler(self) -> float:
        if self.num_blocks == 0:
            return 0.0
        return float(self.block_errors) / float(self.num_blocks)

    @property
    def bler_se_wald(self) -> float:
        """Codeword/transport-block Bernoulli standard error (Wald)."""
        if self.num_blocks <= 1:
            return 0.0
        b = self.bler
        return math.sqrt(max(0.0, b * (1.0 - b) / float(self.num_blocks)))

    @property
    def bler_se(self) -> float:
        """Alias to bler_se_wald."""
        return self.bler_se_wald

    @property
    def info_ber_se(self) -> float:
        """Information bit BER standard error derived from sample variance across blocks."""
        if self.num_blocks <= 1:
            return 0.0
        sample_var = self._block_ber_m2 / float(self.num_blocks - 1)
        return math.sqrt(max(0.0, sample_var / float(self.num_blocks)))

    def bler_ci_wald(self, k: float = CONFIDENCE_K) -> Tuple[float, float]:
        """Wald normal approximation CI for BLER."""
        se = self.bler_se_wald
        low = max(0.0, self.bler - k * se)
        high = min(1.0, self.bler + k * se)
        return low, high

    def bler_ci_wilson(self, k: float = CONFIDENCE_K) -> Tuple[float, float]:
        """Wilson score interval for BLER (stable at zero errors)."""
        n = self.num_blocks
        if n == 0:
            return 0.0, 1.0
        p = self.bler
        z = k
        denom = 1.0 + (z ** 2) / float(n)
        center = (p + (z ** 2) / (2.0 * float(n))) / denom
        margin = (z / denom) * math.sqrt((p * (1.0 - p) / float(n)) + ((z ** 2) / (4.0 * float(n ** 2))))
        low = max(0.0, center - margin)
        high = min(1.0, center + margin)
        return low, high

    def bler_ci_clopper_pearson(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Exact Clopper-Pearson interval for BLER."""
        n = self.num_blocks
        x = self.block_errors
        if n == 0:
            return 0.0, 1.0
        alpha = 1.0 - confidence
        low = 0.0 if x == 0 else float(scipy.stats.beta.ppf(alpha / 2.0, x, n - x + 1))
        high = 1.0 if x == n else float(scipy.stats.beta.ppf(1.0 - alpha / 2.0, x + 1, n - x))
        return low, high

    def bler_ci95(self, method: str = "wald") -> Tuple[float, float]:
        """95% Confidence Interval for BLER under chosen method."""
        m = method.lower()
        if m == "wald":
            return self.bler_ci_wald()
        elif m == "wilson":
            return self.bler_ci_wilson()
        elif m in ("clopper_pearson", "exact"):
            return self.bler_ci_clopper_pearson()
        else:
            raise ValueError(f"Unknown CI method: {method}. Choose from 'wald', 'wilson', 'clopper_pearson'")

    def info_ber_ci95(self, k: float = CONFIDENCE_K) -> Tuple[float, float]:
        """95% Confidence Interval for Information BER."""
        se = self.info_ber_se
        low = max(0.0, self.info_ber - k * se)
        high = min(1.0, self.info_ber + k * se)
        return low, high

    def to_dict(self) -> Dict[str, Any]:
        b_wald_low, b_wald_high = self.bler_ci_wald()
        b_wilson_low, b_wilson_high = self.bler_ci_wilson()
        b_cp_low, b_cp_high = self.bler_ci_clopper_pearson()
        i_low, i_high = self.info_ber_ci95()

        return {
            "action": self.action.name,
            "modulation": self.action.modulation,
            "requested_code_rate": self.action.code_rate,
            "effective_code_rate": float(self.k) / float(self.n),
            "effective_spectral_efficiency": (float(self.k) / float(self.n)) * self.action.bits_per_symbol,
            "snr_db": self.snr_db,
            "eb_n0_db": self.eb_n0_db,
            "info_bits_per_block": self.k,
            "coded_bits_per_block": self.n,
            "num_blocks": self.num_blocks,
            "total_info_bits": self.total_info_bits,
            "total_coded_bits": self.total_coded_bits,
            "info_bit_errors": self.info_bit_errors,
            "info_ber": self.info_ber,
            "info_ber_se": self.info_ber_se,
            "info_ber_ci95_low": i_low,
            "info_ber_ci95_high": i_high,
            "block_errors": self.block_errors,
            "bler": self.bler,
            "bler_se": self.bler_se,
            "bler_ci95_low": b_wald_low,
            "bler_ci95_high": b_wald_high,
            "bler_wilson_ci95_low": b_wilson_low,
            "bler_wilson_ci95_high": b_wilson_high,
            "bler_clopper_pearson_ci95_low": b_cp_low,
            "bler_clopper_pearson_ci95_high": b_cp_high,
        }
