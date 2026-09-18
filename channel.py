"""Flat Gauss-Markov Rayleigh bursts with causal pilot tracking.
Independent bursts are the sampling units. P segments each contain one pilot and
N/P data symbols; gain is constant inside each segment. Exponential correlation
time is 20 / 0.1 burst durations for slow / fast. This is NOT a Jakes spectrum.
Only the first pilot reaches the transmitter before its single burst decision.
The receiver tracks using preceding segment pilots. AWGN has known unit gain.
"""
from dataclasses import dataclass
import numpy as np

CHANNELS = ('AWGN', 'Rayleigh slow', 'Rayleigh fast')
SPLITS = {'train': 0, 'validation': 1, 'test': 2, 'development': 3}
COHERENCE_FRAMES = {1: 20.0, 2: 0.1}


@dataclass(frozen=True)
class Observation:
    snr_est_db: float
    nominal_snr_db: float
    channel_id: int

    def features(self):
        return [self.snr_est_db, self.nominal_snr_db, float(self.channel_id)]


@dataclass(frozen=True)
class FrameInput:
    observation: Observation
    h: np.ndarray
    h_est: np.ndarray
    n0: float
    payload_pool: np.ndarray
    standard_noise: np.ndarray
    seed_key: tuple


def complex_normal(rng, size=None):
    return (rng.normal(size=size) + 1j*rng.normal(size=size))/np.sqrt(2)


def correlation(channel_id, pilots):
    return 1.0 if channel_id == 0 else float(np.exp(-1/(COHERENCE_FRAMES[channel_id]*pilots)))


def posterior_initial(observation, rng, csi='pilot'):
    """Exact Gaussian h0 posterior given first pilot; circular phase canonicalized."""
    n0 = 10.**(-observation.nominal_snr_db/10)
    z = np.sqrt(n0 * 10.**(observation.snr_est_db/10))
    if csi == 'perfect':
        return complex(z), z
    variance = n0/(1+n0)
    return z/(1+n0) + np.sqrt(variance)*complex_normal(rng), z


def frame_input(*, master_seed, split, channel_id, snr_index, snr_db, frame,
                symbols=1536, pilots=32, csi='pilot', condition=None, repeat=None):
    if split not in SPLITS or channel_id not in range(len(CHANNELS)) or csi not in ('pilot', 'perfect'):
        raise ValueError('Unknown split, channel or CSI mode')
    if min(master_seed, snr_index, frame) < 0 or pilots < 1 or symbols < 1 or symbols % pilots or not np.isfinite(snr_db):
        raise ValueError('Pilot count must divide data symbols; valid seeds/SNR required')
    key = (master_seed, SPLITS[split], channel_id, snr_index, frame)
    rng_key = key if repeat is None else key + (701, int(repeat))
    ch_rng, pilot_rng, bit_rng, noise_rng = [
        np.random.default_rng(s) for s in np.random.SeedSequence(rng_key).spawn(4)]
    n0 = 10.**(-float(snr_db)/10)
    rho = correlation(channel_id, pilots)
    gains = np.ones(pilots, dtype=complex)
    estimates = np.ones(pilots, dtype=complex)
    if channel_id:
        if condition is not None:
            if condition.channel_id != channel_id or condition.nominal_snr_db != snr_db:
                raise ValueError('Conditional context does not match scenario')
            gains[0], estimates[0] = posterior_initial(condition, ch_rng, csi)
        else:
            h0 = complex_normal(ch_rng)
            z = h0 if csi == 'perfect' else h0 + np.sqrt(n0)*complex_normal(pilot_rng)
            rotation = np.exp(-1j*np.angle(z))
            gains[0], estimates[0] = h0*rotation, abs(z)
        innovations = complex_normal(ch_rng, pilots-1)
        for j in range(1, pilots):
            gains[j] = rho*gains[j-1] + np.sqrt(1-rho*rho)*innovations[j-1]
        estimates[1:] = gains[1:] + np.sqrt(n0)*complex_normal(pilot_rng, pilots-1)
        if csi == 'perfect':
            estimates = gains.copy()
    obs = Observation(float(10*np.log10(max(abs(estimates[0])**2/n0, np.finfo(float).tiny))),
                      float(snr_db), channel_id) if condition is None else condition
    payload = bit_rng.integers(0, 2, 6*symbols, dtype=np.int32)
    noise = complex_normal(noise_rng, symbols)
    h, h_est = (np.repeat(v, symbols//pilots) for v in (gains, estimates))
    for value in (h, h_est, payload, noise):
        value.setflags(write=False)
    return FrameInput(obs, h, h_est, n0, payload, noise, key)

