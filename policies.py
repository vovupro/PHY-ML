"""Frozen lightweight baselines and one common service rule.

Direct CART learns ex-ante Monte Carlo action labels. RF regresses conditional
BER/BLER. Both see identical transmitter features and the same training trials.
A training-only empirical LUT checks the selected action for every policy.
Raw counterparts isolate the effect of this common DEFER gate.
"""
from dataclasses import dataclass, replace
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, export_text

if __package__:
    from .phy_engine import MODES
    from .channel import CHANNELS
else:
    from phy_engine import MODES
    from channel import CHANNELS

FEATURES = ['snr_est_db', 'nominal_snr_db', 'channel_id']
DT_CONFIG = dict(max_depth=7, min_samples_leaf=12, random_state=1729)
RF_CONFIG = dict(n_estimators=64, max_depth=10, min_samples_leaf=12, random_state=1729, n_jobs=1)


@dataclass(frozen=True)
class Decision:
    action_index: int
    predicted_ber: float | None = None
    predicted_bler: float | None = None
    predicted_feasible: bool | None = None
    confidence: float | None = None
    gate_predicted_ber: float | None = None


def choose(predictions, payload, objective='goodput', target_ber=.001, allow_defer=False):
    p = np.asarray(predictions, dtype=float)
    if objective not in ('goodput', 'correct_bits') or not 0 < target_ber < 1:
        raise ValueError('Invalid objective or target')
    if p.shape != (len(payload), 2) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Expected finite BER/BLER probabilities for each action')
    score = np.asarray(payload)*(1-p[:, 1 if objective == 'goodput' else 0])
    feasible = np.flatnonzero(p[:, 0] <= target_ber)
    if not len(feasible):
        if allow_defer:
            return len(payload)
        feasible = np.flatnonzero(p[:, 0] == p[:, 0].min())
    # Action catalogue orders iteration caps ascending; ties prefer cheaper action.
    return int(max(feasible, key=lambda i: (score[i], -i)))


def compute_oracle_labels(errors, bits, blocks, trials, num_actions, target_ber,
                          allow_defer=False, objective='goodput'):
    """Finite-MC teacher labels, not absolute truth or a hindsight certificate."""
    errors, bits, blocks = map(np.asarray, (errors, bits, blocks))
    trials = np.asarray(trials).reshape(-1, 1)
    if errors.shape != bits.shape or blocks.shape != bits.shape or bits.shape[1] != num_actions or np.any(trials < 1):
        raise ValueError('Unaligned counts/trials')
    predictions = np.stack([errors/bits, blocks/trials], axis=-1)
    payload = bits/trials
    return np.array([choose(p, b, objective, target_ber, allow_defer) for p,b in zip(predictions, payload)])


def select_predictions(predictions, actions, symbols, target_ber, objective='goodput', allow_defer=False):
    payload = [MODES[a.mode_id].payload_bits(symbols) for a in actions]
    index = choose(predictions, payload, objective, target_ber, allow_defer)
    if index == len(actions):
        return Decision(index, predicted_feasible=False)
    ber, bler = map(float, predictions[index])
    return Decision(index, ber, bler, ber <= target_ber)


class FixedPolicy:
    def __init__(self, actions, robust=True):
        mode_ids = sorted(set(a.mode_id for a in actions))
        mid = min(mode_ids, key=lambda i: (MODES[i].bits_per_symbol, MODES[i].rate)) if robust else max(
            mode_ids, key=lambda i: MODES[i].bits_per_symbol*MODES[i].rate)
        self.index = max((i for i,a in enumerate(actions) if a.mode_id == mid), key=lambda i: actions[i].iterations)

    def decide_many(self, observations):
        return [Decision(self.index) for _ in observations]

    def decide(self, obs):
        return self.decide_many([obs])[0]


class AdaptivePolicy:
    def __init__(self, actions, symbols, target_ber, objective='goodput'):
        self.actions, self.symbols = actions, symbols
        self.target_ber, self.objective = target_ber, objective

    def decide_many(self, observations):
        return [select_predictions(p, self.actions, self.symbols, self.target_ber, self.objective)
                for p in self.predict_many(observations)]

    def decide(self, obs):
        return self.decide_many([obs])[0]


class EmpiricalLUT(AdaptivePolicy):
    """Fixed 2 dB cells; action-outcome means, with nearest-cell extrapolation."""
    def fit(self, observations, targets):
        x = np.array([o.features() for o in observations])
        self.centers, self.tables, self.counts = {}, {}, {}
        for ch in range(len(CHANNELS)):
            mask = x[:, 2] == ch
            if not mask.any():
                raise ValueError('Training must cover each declared channel')
            bins = np.floor(x[mask, :2]/2).astype(int)
            unique, inverse = np.unique(bins, axis=0, return_inverse=True)
            self.centers[ch] = unique*2+1
            self.tables[ch] = np.stack([targets[mask][inverse == i].mean(axis=0) for i in range(len(unique))])
            self.counts[ch] = np.bincount(inverse)
        return self

    def predict_many(self, observations):
        result = []
        for o in observations:
            query = 2*np.floor(np.array([o.snr_est_db, o.nominal_snr_db])/2)+1
            d = np.sum((self.centers[o.channel_id]-query)**2, axis=1)
            result.append(self.tables[o.channel_id][np.argmin(d)])
        return np.stack(result)


class RandomForestPolicy(AdaptivePolicy):
    def fit(self, observations, targets):
        self.model = RandomForestRegressor(**RF_CONFIG)
        self.model.fit([o.features() for o in observations], targets.reshape(len(observations), -1))
        return self

    def predict_many(self, observations):
        return self.model.predict([o.features() for o in observations]).reshape(-1, len(self.actions), 2)


class DecisionTreeClassifierPolicy:
    def __init__(self, actions, **config):
        self.actions = actions
        self.model = DecisionTreeClassifier(**(DT_CONFIG | config))

    def fit(self, observations, labels):
        self.model.fit([o.features() for o in observations], labels)
        return self

    def predict_many(self, observations):
        return self.model.predict([o.features() for o in observations])

    def decide_many(self, observations):
        x = [o.features() for o in observations]
        predictions = self.model.predict(x)
        confidence = self.model.predict_proba(x).max(axis=1)
        # Class probability is not a BER feasibility probability.
        return [Decision(int(i), confidence=float(c)) for i,c in zip(predictions, confidence)]

    def decide(self, obs):
        return self.decide_many([obs])[0]

    def export_rules(self, action_names=None):
        names = None if action_names is None else [action_names[i] for i in self.model.classes_]
        return export_text(self.model, feature_names=FEATURES, class_names=names)


class ServicePolicy:
    """Identical action-specific gate for Fixed, LUT, CART and RF; no test fitting."""
    def __init__(self, base, gate, target_ber, actions):
        self.base, self.gate = base, gate
        self.target_ber, self.actions = target_ber, actions

    def decide_many(self, observations):
        decisions = self.base.decide_many(observations)
        estimates = self.gate.predict_many(observations)
        output = []
        for d,p in zip(decisions, estimates):
            if not 0 <= d.action_index < len(self.actions):
                raise ValueError('Base policy must choose a real transmission action')
            ber = float(p[d.action_index, 0])
            feasible = ber <= self.target_ber
            output.append(replace(d, action_index=d.action_index if feasible else len(self.actions),
                                  predicted_feasible=feasible, gate_predicted_ber=ber))
        return output

    def decide(self, obs):
        return self.decide_many([obs])[0]


def fit_policies(data, actions, config):
    args = (actions, config['symbols'], config['target_ber'], config['objective'])
    lut = EmpiricalLUT(*args).fit(data.observations, data.targets)
    rf = RandomForestPolicy(*args).fit(data.observations, data.targets)
    labels = data.oracle_labels(config['target_ber'], objective=config['objective'])
    dt = DecisionTreeClassifierPolicy(actions).fit(data.observations, labels)
    bases = {'Fixed robust': FixedPolicy(actions), 'Fixed high throughput': FixedPolicy(actions, False),
             'Empirical LUT': lut, 'Decision Tree': dt, 'Random Forest': rf}
    policies = {}
    for name, base in bases.items():
        policies[name+' / raw'] = base
        policies[name+' / gated'] = ServicePolicy(base, lut, config['target_ber'], actions)
    return policies

