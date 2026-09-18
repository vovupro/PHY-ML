"""Unit-energy coherent PHY. Custom MCS; optional Sionna NR LDPC, not full NR."""
from dataclasses import dataclass
from functools import lru_cache
import time
import numpy as np
from scipy.special import logsumexp

NUM_SYMBOLS_PER_FRAME = 1536
SYMBOL_RATE = 1_000_000.0
CRC_LENGTH = 24


@dataclass(frozen=True)
class Mode:
    mode_id: int
    modulation: str
    bits_per_symbol: int
    rate: float
    coded: bool

    @property
    def name(self):
        return self.modulation + (f'-LDPC-{self.rate:.3f}' if self.coded else '-uncoded')

    def information_bits(self, symbols):
        return int(symbols * self.bits_per_symbol * self.rate)

    def payload_bits(self, symbols):
        return self.information_bits(symbols) - (CRC_LENGTH if self.coded else 0)


@dataclass(frozen=True, order=True)
class Action:
    mode_id: int
    iterations: int = 0


MODES = tuple(Mode(i, name, m, 1., False) for i, (name, m) in enumerate(
    [('BPSK', 1), ('QPSK', 2), ('16QAM', 4), ('64QAM', 6)])) + tuple(
    Mode(4 + mod * 3 + j, name, m, rate, True)
    for mod, (name, m) in enumerate([('BPSK', 1), ('QPSK', 2), ('16QAM', 4), ('64QAM', 6)])
    for j, rate in enumerate([.5, 2/3, .75]))


def action_catalog(coding='uncoded', caps=(5, 10), budget=10):
    if coding not in ('uncoded', 'ldpc', 'mixed') or budget < 0:
        raise ValueError('Invalid coding or iteration budget')
    if not caps or any(not isinstance(i, int) or i <= 0 for i in caps):
        raise ValueError('Iteration caps must be positive integers')
    modes = [m for m in MODES if coding == 'mixed' or m.coded == (coding == 'ldpc')]
    actions = [Action(m.mode_id, it) for m in modes
               for it in (sorted(set(i for i in caps if i <= budget)) if m.coded else [0])]
    if not actions:
        raise ValueError('No action fits the iteration budget')
    return tuple(actions)


def _binary(bits):
    bits = np.asarray(bits)
    if bits.ndim != 1 or not np.isin(bits, [0, 1]).all():
        raise ValueError('Expected a one-dimensional binary array')
    return bits.astype(np.int32)


def compute_crc24a(bits):
    reg = 0
    for bit in _binary(bits):
        feedback = ((reg >> 23) & 1) ^ int(bit)
        reg = (reg << 1) & 0xffffff
        if feedback:
            reg ^= 0x864cfb
    return np.array([(reg >> i) & 1 for i in range(23, -1, -1)], dtype=np.int32)


def verify_crc24a(bits):
    bits = _binary(bits)
    if len(bits) < CRC_LENGTH:
        raise ValueError('CRC word is shorter than 24 bits')
    return bool(np.array_equal(compute_crc24a(bits[:-24]), bits[-24:]))


@lru_cache(maxsize=1)
def _crc_byte_table():
    table = np.arange(256, dtype=np.uint32) << 16
    for _ in range(8):
        table = ((table << 1) & 0xffffff) ^ np.where(table & 0x800000, 0x864cfb, 0).astype(np.uint32)
    return table


def crc24a_many(bits):
    """Same polynomial as compute_crc24a; byte-wise batch implementation."""
    bits = np.asarray(bits, dtype=np.int32)
    if bits.ndim != 2 or not np.isin(bits, [0, 1]).all():
        raise ValueError('Expected binary matrix')
    reg = np.zeros(len(bits), dtype=np.uint32)
    full = bits.shape[1]//8*8
    for byte in np.packbits(bits[:, :full], axis=1).T:
        reg = ((reg << 8) & 0xffffff) ^ _crc_byte_table()[((reg >> 16) ^ byte) & 255]
    for bit in bits[:, full:].T:
        feedback = (reg >> 23) ^ bit.astype(np.uint32)
        reg = ((reg << 1) & 0xffffff) ^ (feedback * 0x864cfb)
    return ((reg[:, None] >> np.arange(23, -1, -1, dtype=np.uint32)) & 1).astype(np.int32)


@lru_cache(maxsize=4)
def constellation(m):
    if m not in (1, 2, 4, 6):
        raise ValueError('Supported modulations: BPSK, QPSK, 16QAM, 64QAM')
    labels = ((np.arange(2**m)[:, None] >> np.arange(m-1, -1, -1)) & 1).astype(np.int32)
    if m == 1:
        points = 1 - 2*labels[:, 0]
    elif m == 2:
        points = ((1-2*labels[:, 0]) + 1j*(1-2*labels[:, 1])) / np.sqrt(2)
    elif m == 4:
        # Gray axis: 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3.
        levels = np.array([-3, -1, 3, 1])
        points = (levels[labels[:, 0]*2+labels[:, 1]] +
                  1j*levels[labels[:, 2]*2+labels[:, 3]]) / np.sqrt(10)
    else:
        # Unit-energy Cartesian Gray 64-QAM (custom bit ordering).
        levels_8 = np.array([-7, -5, -1, -3, 7, 5, 1, 3])
        i_idx = labels[:, 0]*4 + labels[:, 1]*2 + labels[:, 2]
        q_idx = labels[:, 3]*4 + labels[:, 4]*2 + labels[:, 5]
        points = (levels_8[i_idx] + 1j*levels_8[q_idx]) / np.sqrt(42)
    points = np.asarray(points, dtype=np.complex128)
    points.setflags(write=False)
    labels.setflags(write=False)
    return points, labels


def modulate(bits, m):
    bits = _binary(bits)
    if len(bits) % m:
        raise ValueError('Bit count must divide into complete symbols')
    indices = bits.reshape(-1, m) @ (1 << np.arange(m-1, -1, -1))
    return constellation(m)[0][indices]


def demodulate(y, h_est, n0, m, soft=False):
    """Plugin coherent likelihood with known N0; no division by channel gain.

    Pilot CSI gives a mismatched receiver, not a Bayesian-optimal claim.
    Zero gain produces zero LLRs rather than NaNs.
    """
    points, labels = constellation(m)
    y = np.asarray(y, dtype=np.complex128)
    h_est = np.broadcast_to(np.asarray(h_est), y.shape)
    if y.ndim != 1 or not np.isfinite(y).all() or not np.isfinite(h_est).all() or not np.isfinite(n0) or n0 <= 0:
        raise ValueError('Nonfinite signal/CSI or nonpositive noise variance')
    metric = -np.abs(y[:, None] - h_est[:, None]*points[None, :])**2 / n0
    if not soft:
        return labels[np.argmax(metric, axis=1)].reshape(-1)
    return np.stack([logsumexp(metric[:, labels[:, j] == 1], axis=1) -
                     logsumexp(metric[:, labels[:, j] == 0], axis=1)
                     for j in range(m)], axis=1).reshape(-1).astype(np.float32)


class PHYEngine:
    def __init__(self, symbols=NUM_SYMBOLS_PER_FRAME):
        if not isinstance(symbols, int) or symbols < 96 or symbols % 12:
            raise ValueError('Data symbols must be an integer multiple of 12, at least 96')
        self.symbols = symbols
        self._encoders = {}
        self._decoders = {}

    def _encoder(self, mode):
        if mode.mode_id not in self._encoders:
            from sionna.phy.fec.ldpc import LDPC5GEncoder
            self._encoders[mode.mode_id] = LDPC5GEncoder(
                mode.information_bits(self.symbols), self.symbols*mode.bits_per_symbol,
                num_bits_per_symbol=mode.bits_per_symbol, device='cpu')
        return self._encoders[mode.mode_id]

    def _decoder(self, action):
        if action not in self._decoders:
            from sionna.phy.fec.ldpc import LDPC5GDecoder
            self._decoders[action] = LDPC5GDecoder(self._encoder(MODES[action.mode_id]),
                num_iter=action.iterations, return_infobits=True, device='cpu')
        return self._decoders[action]

    def warm_up(self, actions):
        rng = np.random.default_rng(7104)
        self.evaluate(actions, h=1+0j, h_est=1+0j, n0=.001,
                      payload_pool=rng.integers(0, 2, 6*self.symbols, dtype=np.int32),
                      standard_noise=np.zeros(self.symbols, dtype=np.complex128))

    def complexity(self, action):
        mode = MODES[action.mode_id]
        enc = self._encoder(mode) if mode.coded else None
        return {'action': [action.mode_id, action.iterations], 'mode': mode.name,
                'payload_bits': mode.payload_bits(self.symbols),
                'coded_bits': self.symbols*mode.bits_per_symbol,
                'base_graph': enc._bg if enc else None,
                'mother_graph_edges': int(enc.pcm.nnz) if enc else 0,
                'mother_edge_iterations': int(enc.pcm.nnz)*action.iterations if enc else 0}

    def evaluate_batch(self, actions, sources):
        """Paired action outcomes; batching changes execution only, not the PHY."""
        size = len(sources)
        if not size:
            raise ValueError('Empty batch')
        hs = np.stack([s.h for s in sources])
        estimates = np.stack([s.h_est for s in sources])
        n0 = np.array([s.n0 for s in sources])[:, None, None]
        noise = np.stack([s.standard_noise for s in sources])
        payload_pool = np.stack([s.payload_pool for s in sources])
        output = {k: np.empty((size, len(actions)), dtype=np.int64 if k in
                  ('errors', 'bits', 'blocks', 'undetected') else float)
                  for k in ('errors', 'bits', 'blocks', 'undetected', 'phy_ms', 'decode_ms')}
        prepared = {}
        for i, action in enumerate(actions):
            mode = MODES[action.mode_id]
            if mode.coded:
                import torch
                self._decoder(action)
            if mode.mode_id not in prepared:
                started = time.perf_counter()
                payload = payload_pool[:, :mode.payload_bits(self.symbols)]
                if mode.coded:
                    info = np.concatenate([payload, crc24a_many(payload)], axis=1)
                    with torch.inference_mode():
                        encoded = self._encoder(mode)(torch.from_numpy(info.astype(np.float32))).numpy().astype(np.int32)
                else:
                    encoded = payload
                tx = modulate(encoded.ravel(), mode.bits_per_symbol).reshape(size, self.symbols)
                y = hs*tx + np.sqrt(n0[..., 0])*noise
                points, labels = constellation(mode.bits_per_symbol)
                metric = -np.abs(y[..., None] - estimates[..., None]*points)**2/n0
                if mode.coded:
                    received = np.stack([logsumexp(metric[..., labels[:, j] == 1], axis=-1) -
                                         logsumexp(metric[..., labels[:, j] == 0], axis=-1)
                                         for j in range(mode.bits_per_symbol)], axis=-1).reshape(size, -1).astype(np.float32)
                else:
                    received = labels[np.argmax(metric, axis=-1)].reshape(size, -1)
                prep_ms = (time.perf_counter()-started)*1000/size
                prepared[mode.mode_id] = payload, received, prep_ms
            payload, received, prep_ms = prepared[mode.mode_id]
            started = time.perf_counter()
            if mode.coded:
                with torch.inference_mode():
                    decoded = self._decoder(action)(torch.from_numpy(received)).numpy().astype(np.int32)
                decode_ms = (time.perf_counter()-started)*1000/size
                payload_hat = decoded[:, :-CRC_LENGTH]
                crc_ok = np.all(crc24a_many(payload_hat) == decoded[:, -CRC_LENGTH:], axis=1)
            else:
                payload_hat, decode_ms = received, 0.0
                crc_ok = np.ones(size, dtype=bool)
            errors = np.count_nonzero(payload != payload_hat, axis=1)
            output['errors'][:, i] = errors
            output['bits'][:, i] = payload.shape[1]
            output['blocks'][:, i] = (errors > 0) | ~crc_ok
            output['undetected'][:, i] = (errors > 0) & crc_ok if mode.coded else 0
            output['phy_ms'][:, i] = prep_ms + decode_ms
            output['decode_ms'][:, i] = decode_ms
        return output

    def evaluate(self, actions, *, h, h_est, n0, payload_pool, standard_noise):
        """All actions share physical noise/bits; caps share the exact LLR.

        Only the evaluator owns these counterfactual outcomes. Timings exclude
        initialization. Each cap decodes anew, without early stopping.
        """
        if not np.isfinite(h).all() or not np.isfinite(h_est).all() or not np.isfinite(n0) or n0 <= 0:
            raise ValueError('Invalid channel/receiver parameters')
        payload_pool = _binary(payload_pool)
        standard_noise = np.asarray(standard_noise)
        if len(payload_pool) < max(MODES[a.mode_id].payload_bits(self.symbols) for a in actions):
            raise ValueError('Insufficient payload pool')
        if standard_noise.shape != (self.symbols,) or not np.isfinite(standard_noise).all():
            raise ValueError('Invalid standard complex noise')
        prepared = {}
        outcomes = []
        for action in actions:
            mode = MODES[action.mode_id]
            if (mode.coded and action.iterations <= 0) or (not mode.coded and action.iterations != 0):
                raise ValueError('Iterations inconsistent with coding')
            if mode.coded:
                self._decoder(action)
            if mode.mode_id not in prepared:
                start = time.perf_counter()
                payload = payload_pool[:mode.payload_bits(self.symbols)]
                if mode.coded:
                    import torch
                    info = np.concatenate([payload, compute_crc24a(payload)])
                    with torch.inference_mode():
                        encoded = self._encoder(mode)(torch.tensor(info, dtype=torch.float32)[None]).numpy().ravel().astype(np.int32)
                else:
                    encoded = payload
                tx = modulate(encoded, mode.bits_per_symbol)
                tx_ms = (time.perf_counter()-start)*1000
                y = h*tx + np.sqrt(n0)*standard_noise
                start = time.perf_counter()
                received = demodulate(y, h_est, n0, mode.bits_per_symbol, soft=mode.coded)
                demap_ms = (time.perf_counter()-start)*1000
                prepared[mode.mode_id] = (payload, received, tx_ms, demap_ms)
            payload, received, tx_ms, demap_ms = prepared[mode.mode_id]
            start = time.perf_counter()
            if mode.coded:
                import torch
                with torch.inference_mode():
                    decoded = self._decoder(action)(torch.from_numpy(received)[None]).numpy().ravel().astype(np.int32)
                decode_ms = (time.perf_counter()-start)*1000
                crc_ok = verify_crc24a(decoded)
                payload_hat = decoded[:-CRC_LENGTH]
            else:
                decode_ms, crc_ok, payload_hat = 0., None, received
            errors = int(np.count_nonzero(payload != payload_hat))
            # No CRC for uncoded modes: error-free-block goodput is evaluator
            # knowledge, NOT an implemented packet acceptance/ACK mechanism.
            block_error = int(errors > 0 or (mode.coded and not crc_ok))
            outcomes.append({'bit_errors': errors, 'payload_bits': len(payload),
                             'block_error': block_error, 'crc_ok': crc_ok,
                             'undetected_error': int(errors > 0 and crc_ok is True),
                             'tx_ms': tx_ms, 'demap_ms': demap_ms, 'decode_ms': decode_ms,
                             'phy_ms': tx_ms+demap_ms+decode_ms})
        return outcomes
