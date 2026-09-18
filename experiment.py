"""Independent conditional-MC training contexts and held-out physical bursts."""
from dataclasses import dataclass, fields
from pathlib import Path
import json
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold

if __package__:
    from .channel import CHANNELS, Observation, frame_input
    from .phy_engine import MODES
    from .policies import compute_oracle_labels, fit_policies
    from .metrics import summarize, paired_goodput_difference
else:
    from channel import CHANNELS, Observation, frame_input
    from phy_engine import MODES
    from policies import compute_oracle_labels, fit_policies
    from metrics import summarize, paired_goodput_difference


def save_json(path, obj):
    def cast(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False, default=cast), encoding='utf-8')


@dataclass
class Dataset:
    observation: np.ndarray
    keys: np.ndarray
    snr_index: np.ndarray
    errors: np.ndarray
    bits: np.ndarray
    blocks: np.ndarray
    undetected: np.ndarray
    phy_ms: np.ndarray
    decode_ms: np.ndarray
    trials: np.ndarray

    @property
    def observations(self):
        return [Observation(float(v[0]), float(v[1]), int(v[2])) for v in self.observation]

    @property
    def targets(self):
        return np.stack([self.errors/self.bits, self.blocks/self.trials[:, None]], axis=-1)

    def oracle_labels(self, target_ber, objective='goodput'):
        return compute_oracle_labels(self.errors, self.bits, self.blocks, self.trials,
                                     self.errors.shape[1], target_ber, objective=objective)

    def take(self, indices):
        return Dataset(**{f.name: getattr(self, f.name)[indices] for f in fields(self)})

    def save(self, path):
        np.savez_compressed(path, **vars(self))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as arrays:
            return cls(**dict(arrays))


def collect(engine, actions, config, split, count, mc_trials=1):
    if count < 2 or mc_trials < 1:
        raise ValueError('At least two contexts and positive MC count required')
    n = len(CHANNELS)*len(config['snrs'])*count
    shape = (n, len(actions))
    data = Dataset(np.empty((n, 3)), np.empty((n, 5), dtype=np.int64), np.empty(n, dtype=np.int32),
                   *(np.zeros(shape, dtype=np.int64) for _ in range(4)),
                   np.zeros(shape), np.zeros(shape), np.full(n, mc_trials, dtype=np.int64))
    sources, owners = [], []
    def flush():
        if not sources: return
        out = engine.evaluate_batch(actions, sources)
        for k in ('errors', 'bits', 'blocks', 'undetected', 'phy_ms', 'decode_ms'):
            np.add.at(getattr(data, k), owners, out[k])
        sources.clear()
        owners.clear()
    index = 0
    for ch, channel in enumerate(CHANNELS):
        for s, snr in enumerate(config['snrs']):
            for frame in range(count):
                args = dict(master_seed=config['seed'], split=split, channel_id=ch,
                            snr_index=s, snr_db=snr, frame=frame, symbols=engine.symbols,
                            pilots=config['pilots'], csi=config['csi'])
                source = frame_input(**args)
                data.observation[index] = source.observation.features()
                data.keys[index], data.snr_index[index] = source.seed_key, s
                for repeat in range(mc_trials):
                    # ALL train repetitions resample hidden h conditional on observed
                    # pilot, including repetition zero. No realized-h teacher shortcut.
                    trial = frame_input(**args, condition=source.observation, repeat=repeat) if split == 'train' else source
                    if split != 'train' and mc_trials != 1:
                        raise ValueError('Held-out physical data must contain actual independent bursts')
                    sources.append(trial)
                    owners.append(index)
                    if len(sources) == config['batch_size']: flush()
                index += 1
            flush()
            print(f'{split}: {channel}, SNR={snr:g}, {index}/{n} contexts', flush=True)
    return data


def assert_disjoint(*datasets):
    seen = set()
    for data in datasets:
        keys = set(map(tuple, data.keys))
        if len(keys) != len(data.keys) or seen.intersection(keys):
            raise ValueError('Duplicate or overlapping independent context keys')
        seen.update(keys)


def chosen_counts(policy, data, actions):
    decisions = policy.decide_many(data.observations)
    choices = np.array([d.action_index for d in decisions], dtype=int)
    if len(choices) != len(data.keys) or np.any((choices < 0) | (choices > len(actions))):
        raise ValueError('Invalid action or decision count')
    tx = choices < len(actions)
    safe = np.minimum(choices, len(actions)-1)
    out = {k: np.where(tx, getattr(data, k)[np.arange(len(choices)), safe], 0)
           for k in ('errors', 'bits', 'blocks', 'undetected', 'phy_ms', 'decode_ms')}
    return decisions, choices, tx, out


def evaluate(policies, data, actions, config):
    if np.any(data.trials != 1):
        raise ValueError('Metrics require independent single-burst test observations')
    rows, trace, paired = [], {}, []
    max_payload = max(MODES[a.mode_id].payload_bits(config['symbols']) for a in actions)
    total_symbols = config['symbols']+config['pilots']
    scenario_count = len(CHANNELS)*len(config['snrs'])
    gp = {}
    ceiling = np.max(data.bits*(1-data.blocks), axis=1)/total_symbols
    for name, policy in policies.items():
        decisions, choices, tx, out = chosen_counts(policy, data, actions)
        gp[name] = out['bits']*(1-out['blocks'])/total_symbols
        trace[name] = {'action_index': choices}
        for field in ('predicted_ber', 'predicted_bler', 'predicted_feasible', 'gate_predicted_ber'):
            trace[name][field] = np.array([-1 if getattr(d, field) is None else getattr(d, field) for d in decisions])
        for ch, channel in enumerate(CHANNELS):
            for s,snr in enumerate(config['snrs']):
                mask = (data.observation[:, 2] == ch) & (data.snr_index == s)
                args = dict(total_symbols=total_symbols, target_ber=config['target_ber'], max_payload=max_payload)
                summary = summarize(out['errors'][mask], out['bits'][mask], out['blocks'][mask], **args)
                family = summarize(out['errors'][mask], out['bits'][mask], out['blocks'][mask],
                                   alpha=.05/(len(policies)*scenario_count), **args)
                rows.append({'policy': name, 'channel': channel, 'snr_db': snr, **summary,
                             'defer_fraction': float(1-tx[mask].mean()),
                             'transmitted_frames': int(tx[mask].sum()),
                             'action_counts': np.bincount(choices[mask], minlength=len(actions)+1).tolist(),
                             'familywise_ber_upper': family['ber_upper'],
                             'familywise_ber_certified': family['ber_certified_at_stated_confidence'],
                             'hindsight_unconstrained_goodput_ceiling': float(ceiling[mask].mean()),
                             'undetected_errors': int(out['undetected'][mask].sum())})
    comparisons = []
    for suffix in (' / raw', ' / gated'):
        for model in ('Decision Tree', 'Random Forest'):
            for reference in ('Empirical LUT', 'Fixed robust', 'Fixed high throughput'):
                comparisons.append((model+suffix, reference+suffix))
        comparisons.append(('Decision Tree'+suffix, 'Random Forest'+suffix))
    for left,right in comparisons:
        for ch,channel in enumerate(CHANNELS):
            for s,snr in enumerate(config['snrs']):
                mask = (data.observation[:,2] == ch) & (data.snr_index == s)
                paired.append({'comparison': left+' - '+right, 'channel': channel, 'snr_db': snr,
                    **paired_goodput_difference(gp[left][mask], gp[right][mask], max_payload/total_symbols,
                                               alpha=.05/(len(comparisons)*scenario_count))})
    return rows, trace, paired


def prediction_diagnostics(policies, data, actions, config):
    output = []
    total = config['symbols']+config['pilots']
    for name,policy in policies.items():
        decisions, choices, tx, out = chosen_counts(policy, data, actions)
        result = {'policy': name, 'frames': len(tx), 'transmitted': int(tx.sum()),
                  'defer_fraction': float(1-tx.mean()),
                  'observed_pooled_ber': float(out['errors'].sum()/out['bits'].sum()) if tx.any() else None,
                  'observed_goodput': float(np.mean(out['bits']*(1-out['blocks'])/total)),
                  'action_counts': np.bincount(choices, minlength=len(actions)+1).tolist()}
        gate = np.array([-1 if d.gate_predicted_ber is None else d.gate_predicted_ber for d in decisions])
        bins = []
        for lo,hi in zip([0., .0001, .001, .01, .1], [.0001, .001, .01, .1, 1.000001]):
            mask = tx & (gate >= lo) & (gate < hi)
            if mask.any():
                bins.append({'gate_ber_range': [lo,min(1.,hi)], 'frames': int(mask.sum()),
                             'observed_ber': float(out['errors'][mask].sum()/out['bits'][mask].sum()),
                             'predicted_ber': float(np.average(gate[mask], weights=out['bits'][mask]))})
        result['gate_calibration_bins'] = bins
        if hasattr(policy, 'predict_many'):
            p = np.asarray(policy.predict_many(data.observations))
            if p.ndim == 3:
                result['ber_mse'] = float(np.mean((p[...,0]-data.targets[...,0])**2))
                result['bler_brier'] = float(np.mean((p[...,1]-data.targets[...,1])**2))
        output.append(result)
    return output


def cross_validate(train, actions, config):
    """Fold-local LUT, gate and model fitting; replicas never cross folds."""
    strata = train.observation[:,2].astype(int)*len(config['snrs']) + train.snr_index
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=741)
    rows = []
    for fold,(fit_index, held_index) in enumerate(cv.split(train.observation, strata)):
        fit, held = train.take(fit_index), train.take(held_index)
        models = fit_policies(fit, actions, config)
        for name,policy in models.items():
            _,_,tx,out = chosen_counts(policy, held, actions)
            # MC counts estimate performance on held-out contexts. These are CV
            # development diagnostics, never single-frame BER certificates.
            gp = (out['bits']/held.trials)*(1-out['blocks']/held.trials)
            rows.append({'fold': fold, 'policy': name,
                         'mc_goodput': float(np.mean(gp/(config['symbols']+config['pilots']))),
                         'mc_ber': float(out['errors'].sum()/out['bits'].sum()) if tx.any() else None,
                         'defer_fraction': float(1-tx.mean())})
    return rows


def runtime_table(engine, actions, policies, config):
    observations = [frame_input(master_seed=9001, split='development', channel_id=i%len(CHANNELS),
                    snr_index=i%len(config['snrs']), snr_db=config['snrs'][i%len(config['snrs'])], frame=i,
                    symbols=config['symbols'], pilots=config['pilots'], csi=config['csi']).observation
                    for i in range(config['runtime_repetitions'])]
    rows = []
    for name,policy in policies.items():
        for obs in observations[:5]: policy.decide(obs)
        times = []
        for obs in observations:
            started = time.perf_counter()
            policy.decide(obs)
            times.append((time.perf_counter()-started)*1000)
        model = getattr(getattr(policy, 'base', policy), 'model', None)
        nodes = None
        if hasattr(model, 'tree_'): nodes = model.tree_.node_count
        elif hasattr(model, 'estimators_'): nodes = sum(t.tree_.node_count for t in model.estimators_)
        rows.append({'kind': 'policy', 'name': name, 'mean_ms': float(np.mean(times)),
                     'p95_ms': float(np.quantile(times,.95)), 'model_nodes': nodes,
                     'repetitions': len(times), 'scope': 'single decision, including gate when enabled'})
    source = frame_input(master_seed=9002, split='development', channel_id=2, snr_index=0,
                         snr_db=18., frame=0, symbols=config['symbols'], pilots=config['pilots'], csi=config['csi'])
    for action in actions:
        times = []
        for _ in range(5):
            started = time.perf_counter()
            engine.evaluate([action], h=source.h, h_est=source.h_est, n0=source.n0,
                            payload_pool=source.payload_pool, standard_noise=source.standard_noise)
            times.append((time.perf_counter()-started)*1000)
        rows.append({'kind': 'phy', 'name': MODES[action.mode_id].name+f'/I={action.iterations}',
                     **engine.complexity(action), 'mean_ms': float(np.mean(times)), 'p95_ms': float(np.quantile(times,.95)),
                     'repetitions': len(times), 'scope': 'complete individual PHY evaluation, excludes channel generation'})
    return rows

