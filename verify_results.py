"""Verify numerical artifacts and independently replay saved PHY trials.
Use only with trusted local policies.joblib. Hash checks establish provenance,
not scientific validity; physics tests and held-out performance remain separate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np

if __package__:
    from .channel import frame_input, CHANNELS
    from .phy_engine import Action, PHYEngine
    from .experiment import Dataset, assert_disjoint, evaluate, save_json
else:
    from channel import frame_input, CHANNELS
    from phy_engine import Action, PHYEngine
    from experiment import Dataset, assert_disjoint, evaluate, save_json


def verify(directory):
    out = Path(directory).resolve()
    manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 5 or manifest['status'] != 'complete':
        raise ValueError('Requires completed schema-5 study; legacy runs retain their source.zip')
    for name,expected in manifest['artifact_sha256'].items():
        if hashlib.sha256((out/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Artifact hash mismatch: {name}')
    root = Path(__file__).resolve().parent
    for name,expected in manifest['source_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Source differs from frozen study: {name}')
    if manifest['policy_hash_before_validation'] != hashlib.sha256((out/'policies.joblib').read_bytes()).hexdigest():
        raise ValueError('Model was changed after training')
    config = manifest['config']
    datasets = [Dataset.load(out/f'{s}.npz') for s in ('train','validation','test')]
    for split,data in zip(('train','validation','test'),datasets):
        expected = len(CHANNELS)*len(config['snrs'])*config[f'frames_{split}']
        if len(data.keys) != expected:
            raise ValueError('Incorrect independent context count')
        trials = config['mc_trials_train'] if split == 'train' else 1
        if not np.all(data.trials == trials):
            raise ValueError('Incorrect trial counts')
    assert_disjoint(*datasets)
    actions = tuple(Action(a['mode_id'],a['iterations']) for a in manifest['actions'])
    policies = joblib.load(out/'policies.joblib')
    stored = json.loads((out/'results.json').read_text(encoding='utf-8'))
    rows,trace,comparisons = evaluate(policies,datasets[-1],actions,config)
    plain = json.loads(json.dumps(rows,default=lambda v:v.item()))
    if plain != stored['policy_metrics'] or comparisons != stored['paired_goodput_comparisons']:
        raise ValueError('Saved metrics do not reproduce')
    with np.load(out/'decisions.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['frame_keys'],datasets[-1].keys)
        for i,(_,fields) in enumerate(trace.items()):
            for key,values in fields.items():
                np.testing.assert_array_equal(values,saved[f'policy_{i}_{key}'])
    if config['coding'] != 'uncoded':
        import torch
        torch.set_num_threads(config['torch_threads'])
    engine = PHYEngine(config['symbols'])
    replayed = 0
    # First train context and first/last test burst of EACH scenario.
    for split,data in [('train',datasets[0]),('test',datasets[-1])]:
        for ch in range(len(CHANNELS)):
            for si,snr in enumerate(config['snrs']):
                indices = np.flatnonzero((data.observation[:,2] == ch)&(data.snr_index == si))
                for index in ([indices[0]] if split == 'train' else [indices[0],indices[-1]]):
                    args = dict(master_seed=config['seed'],split=split,channel_id=ch,
                                snr_index=si,snr_db=snr,frame=int(data.keys[index,-1]),
                                symbols=config['symbols'],pilots=config['pilots'],csi=config['csi'])
                    original = frame_input(**args)
                    np.testing.assert_array_equal(original.observation.features(),data.observation[index])
                    sources = [frame_input(**args,condition=original.observation,repeat=r)
                               for r in range(config['mc_trials_train'])] if split == 'train' else [original]
                    accumulated = {k: np.zeros(len(actions),dtype=np.int64) for k in ('errors','bits','blocks','undetected')}
                    for start in range(0,len(sources),config['batch_size']):
                        result = engine.evaluate_batch(actions,sources[start:start+config['batch_size']])
                        for k in accumulated: accumulated[k] += result[k].sum(axis=0)
                    for k,value in accumulated.items():
                        np.testing.assert_array_equal(value,getattr(data,k)[index])
                    replayed += len(sources)
    report = {'status':'passed','schema_version':5,'source_and_artifact_hashes':'matched',
              'policy_frozen_before_validation':True,'independent_context_keys':True,
              'all_policy_metrics_and_decisions_recomputed':True,'physical_trials_replayed':replayed,
              'actions_per_trial':len(actions),'timing':'not replayed; machine-load dependent',
              'scope':'Provenance and sampled implementation replay, not a guarantee that BER meets target'}
    save_json(out/'verification.json',report)
    print(json.dumps(report))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    verify(parser.parse_args().directory)

