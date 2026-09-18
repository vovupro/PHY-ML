"""Raw-count metrics and finite-sample bounds across independent FRAME trials.

Bits within an LDPC/fading frame are not treated as independent trials.
Empirical Bernstein bound: Maurer & Pontil (2009), Theorem 4,
https://arxiv.org/abs/0907.3740 . No optional stopping or test-based policy fit.
"""
import numpy as np
from scipy.stats import beta


def bernstein_radius(values, delta, width):
    x = np.asarray(values, dtype=float)
    if not 0 < delta < 1 or width < 0 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Bound needs >=2 finite independent trials and valid delta/range')
    log = np.log(2/delta)
    return float(np.sqrt(2*np.var(x, ddof=1)*log/len(x)) + 7*width*log/(3*(len(x)-1)))


def binomial_interval(errors, trials, alpha=.05):
    if trials < 1:
        return [None, None]
    return [0. if errors == 0 else float(beta.ppf(alpha/2, errors, trials-errors+1)),
            1. if errors == trials else float(beta.ppf(1-alpha/2, errors+1, trials-errors))]


def summarize(errors, bits, blocks, *, total_symbols, target_ber, max_payload, alpha=.05):
    errors, bits, blocks = [np.asarray(v) for v in (errors, bits, blocks)]
    if len(bits) < 2 or errors.shape != bits.shape or blocks.shape != bits.shape:
        raise ValueError('At least two aligned frames required')
    if np.any((errors < 0) | (errors > bits)) or np.any(bits < 0) or np.any(bits > max_payload):
        raise ValueError('Invalid bit counters')
    if not np.isin(blocks, [0, 1]).all() or np.any((errors > 0) & (blocks == 0)):
        raise ValueError('Block errors inconsistent with bit errors')
    n = len(bits)
    tx_mask = bits > 0
    n_tx = int(tx_mask.sum())
    if n_tx > 0:
        ber = float(errors[tx_mask].sum()/bits[tx_mask].sum())
        bler = float(blocks[tx_mask].mean())
        violation_rate = float(np.mean(errors[tx_mask]/bits[tx_mask] > target_ber))
        bler_interval = binomial_interval(int(blocks[tx_mask].sum()), n_tx, alpha)
        observed_meets_target = bool(ber <= target_ber)
    else:
        ber = None
        bler = None
        violation_rate = None
        bler_interval = [None, None]
        observed_meets_target = None
    # Ratio of expectations. Bound numerator and denominator separately;
    # union bound, not a binomial interval over correlated decoded bits.
    x, y = errors/max_payload, bits/max_payload
    numerator = min(1., float(x.mean())+bernstein_radius(x, alpha/2, 1.))
    denominator = max(0., float(y.mean())-bernstein_radius(y, alpha/2, 1.))
    ber_upper = min(1., numerator/denominator) if denominator > 0 else 1.
    gp = bits*(1-blocks)/total_symbols
    gp_radius = bernstein_radius(gp, alpha/2, max_payload/total_symbols)
    return {'frames': n, 'bit_errors': int(errors.sum()), 'payload_bits': int(bits.sum()),
            'block_errors': int(blocks.sum()), 'ber': ber,
            'ber_upper': ber_upper, 'confidence': 1-alpha,
            'observed_ber_meets_target': observed_meets_target,
            'ber_certified_at_stated_confidence': bool(ber_upper <= target_ber),
            'bler': bler, 'bler_interval': bler_interval,
            'violation_rate': violation_rate,
            'offered_bits_per_symbol': float(bits.sum()/(n*total_symbols)),
            'correct_bits_per_symbol': float((bits-errors).sum()/(n*total_symbols)),
            'goodput_bits_per_symbol': float(gp.mean()),
            'goodput_interval': [max(0., float(gp.mean())-gp_radius),
                                 min(max_payload/total_symbols, float(gp.mean())+gp_radius)]}


def paired_goodput_difference(a, b, max_goodput, alpha=.05):
    diff = np.asarray(a)-np.asarray(b)
    radius = bernstein_radius(diff, alpha/2, 2*max_goodput)
    return {'mean_difference': float(diff.mean()),
            'interval': [max(-max_goodput, float(diff.mean())-radius),
                         min(max_goodput, float(diff.mean())+radius)],
            'confidence': 1-alpha, 'paired_frames': len(diff)}
