"""Numeric-only PHY/AMC study. Frozen models, fold-local CV, independent final test.
Custom single-carrier link; not a complete 5G NR waveform. No test-based tuning.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import zipfile
import joblib
import numpy as np

if __package__:
    from .phy_engine import PHYEngine, MODES, action_catalog
    from .channel import CHANNELS, COHERENCE_FRAMES
    from .policies import fit_policies, DT_CONFIG, RF_CONFIG
    from .experiment import (collect, assert_disjoint, evaluate, prediction_diagnostics,
                             runtime_table, save_json, cross_validate)
    from .metrics import summarize
else:
    from phy_engine import PHYEngine, MODES, action_catalog
    from channel import CHANNELS, COHERENCE_FRAMES
    from policies import fit_policies, DT_CONFIG, RF_CONFIG
    from experiment import (collect, assert_disjoint, evaluate, prediction_diagnostics,
                            runtime_table, save_json, cross_validate)
    from metrics import summarize

ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coding', choices=['uncoded', 'ldpc', 'mixed'], default='mixed')
    p.add_argument('--caps', type=int, nargs='+', default=[5, 10])
    p.add_argument('--budget', type=int, default=10)
    p.add_argument('--symbols', type=int, default=1536)
    p.add_argument('--pilots', type=int, default=32)
    p.add_argument('--csi', choices=['pilot', 'perfect'], default='pilot')
    p.add_argument('--target-ber', type=float, default=1e-3)
    p.add_argument('--objective', choices=['goodput', 'correct_bits'], default='goodput')
    p.add_argument('--snrs', type=float, nargs='+', default=[0., 6., 12., 18., 24., 30.])
    p.add_argument('--frames-train', type=int, default=24)
    p.add_argument('--mc-trials-train', type=int, default=32)
    p.add_argument('--frames-validation', type=int, default=128)
    p.add_argument('--frames-test', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=20260910)
    p.add_argument('--torch-threads', type=int, default=1)
    p.add_argument('--runtime-repetitions', type=int, default=60)
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--skip-cv', action='store_true', help='Development smoke runs only; recorded in protocol')
    args = p.parse_args(argv)
    if args.frames_train < 5 or min(args.frames_validation, args.frames_test) < 2 or args.mc_trials_train < 2:
        p.error('Need >=5 train contexts, >=2 validation/test bursts and >=2 train MC trials')
    if args.pilots < 1 or args.symbols % args.pilots or args.seed < 0 or not 0 < args.target_ber < 1:
        p.error('Invalid pilot count, symbol divisibility, seed or BER target')
    if min(args.batch_size, args.torch_threads) < 1 or args.runtime_repetitions < 2:
        p.error('Invalid execution limits')
    if len(args.snrs) < 2 or not np.isfinite(args.snrs).all() or sorted(set(args.snrs)) != args.snrs:
        p.error('SNR values must be finite, unique and increasing')
    return args


def protocol(config, actions):
    packages = ['numpy', 'scipy', 'scikit-learn', 'joblib']
    if config['coding'] != 'uncoded': packages += ['sionna', 'torch']
    return {'schema_version': 5, 'status': 'running', 'config': config,
        'python': sys.version, 'platform': platform.platform(),
        'dependencies': {p: importlib.metadata.version(p) for p in packages},
        'source_sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')},
        'actions': [{**asdict(a), 'mode': MODES[a.mode_id].name,
                     'payload_bits': MODES[a.mode_id].payload_bits(config['symbols'])} for a in actions],
        'models': {'Decision Tree': DT_CONFIG, 'Random Forest': RF_CONFIG},
        'assumptions': {
            'channel': 'Independent bursts; flat Gauss-Markov gain across pilot/data segments, constant inside a segment. Not Jakes/OFDM.',
            'channels': list(CHANNELS), 'correlation_time_in_frames': COHERENCE_FRAMES,
            'observation': 'First pilot magnitude, nominal Es/N0, known channel regime. Circular Rayleigh phase canonicalization.',
            'receiver': 'Causal one-pilot LS per segment; calibrated known unit gain for AWGN; perfect CSI only when explicitly selected.',
            'feedback': 'Ideal feedback of first pilot before action; later pilots available only to receiver.',
            'frame_symbols': config['symbols']+config['pilots'], 'symbol_rate': 1000000.,
            'training': 'Independent conditional-MC draws of hidden channel, future pilots, payload and noise given transmitter context.',
            'labels': 'Estimated expected objective with estimated BER constraint; finite-MC labels, not absolute ground truth.',
            'fallback': 'If no estimated feasible action, choose minimum estimated BER; ties objective then catalogue order.',
            'gate': 'All gated policies use the same train-only LUT for selected-action BER. Failing selected action defers. No certificate claim.',
            'controls': 'Raw counterparts remove the gate for attribution. Fixed policies have identical gate access.',
            'objective': config['objective'], 'goodput': 'Error-free accepted payload per ALL data+pilot symbols, including deferred time.',
            'uncoded': 'Error-free packet success uses evaluator truth; no implemented uncoded ACK or CRC claim.',
            'validation': '5-fold stratified context CV, fold-local teacher/gate/model fitting, fixed hyperparameters; independent validation diagnostics.',
            'test': 'Fresh independent bursts, same predeclared SNR/regimes, no tuning or optional stopping; conditional on this fitted training run.',
            'statistics': 'Frame empirical Bernstein bounds; pooled BER; frame-binomial BLER interval. Familywise Bonferroni paired intervals.',
            'hindsight': 'Unconstrained realized goodput maximum, diagnostic ceiling only; not a deployed policy or measured runtime.',
            'runtime': 'Single-decision CPU time includes shared gate, separate complete PHY time; no deadline guarantee.',
            'scope': 'Joint modulation/coding/iteration adaptation; custom framing and modulation labels; no full NR, queue, ARQ, or energy model.',
            'sources': ['https://scikit-learn.org/stable/modules/cross_validation.html',
                        'https://doi.org/10.1109/TVT.2009.2029693',
                        'https://doi.org/10.1186/s13638-020-01668-7',
                        'https://arxiv.org/abs/0907.3740',
                        'https://web.stanford.edu/~dntse/Chapters_PDF/Fundamentals_Wireless_Communication_chapter2.pdf']
        }}


def run(args):
    config = vars(args).copy()
    output = config.pop('output')
    actions = action_catalog(config['coding'], tuple(config['caps']), config['budget'])
    engine = PHYEngine(config['symbols'])
    if any(MODES[a.mode_id].coded for a in actions):
        import torch
        torch.set_num_threads(config['torch_threads'])
    out = Path(output or ROOT/'results'/time.strftime('study_%Y%m%d_%H%M%S')).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save_json(out/'config.json', config)
    manifest = protocol(config, actions)
    save_json(out/'manifest.json', manifest)
    with zipfile.ZipFile(out/'source.zip', 'w', compression=zipfile.ZIP_DEFLATED) as snapshot:
        for source in sorted([*ROOT.glob('*.py'), *ROOT.glob('requirements*.txt')]):
            snapshot.write(source, source.name)
    started = time.perf_counter()
    train = collect(engine, actions, config, 'train', config['frames_train'], config['mc_trials_train'])
    train.save(out/'train.npz')
    cv = [] if config['skip_cv'] else cross_validate(train, actions, config)
    fit_started = time.perf_counter()
    policies = fit_policies(train, actions, config)
    fit_seconds = time.perf_counter()-fit_started
    # Freeze model bytes before collecting validation or test outcomes.
    joblib.dump(policies, out/'policies.joblib')
    model_hash = hashlib.sha256((out/'policies.joblib').read_bytes()).hexdigest()
    validation = collect(engine, actions, config, 'validation', config['frames_validation'])
    assert_disjoint(train, validation)
    validation.save(out/'validation.npz')
    diagnostics = {'cross_validation': cv,
                   'validation': prediction_diagnostics(policies, validation, actions, config)}
    save_json(out/'diagnostics.json', diagnostics)
    # No fit, calibration or threshold selection below this line.
    test = collect(engine, actions, config, 'test', config['frames_test'])
    assert_disjoint(train, validation, test)
    test.save(out/'test.npz')
    rows, trace, comparisons = evaluate(policies, test, actions, config)
    trace_arrays = {'frame_keys': test.keys}
    for i, (_, values) in enumerate(trace.items()):
        for field,array in values.items(): trace_arrays[f'policy_{i}_{field}'] = array
    np.savez_compressed(out/'decisions.npz', **trace_arrays)
    phy_rows = []
    for i,action in enumerate(actions):
        for ch,channel in enumerate(CHANNELS):
            for s,snr in enumerate(config['snrs']):
                mask = (test.observation[:,2] == ch) & (test.snr_index == s)
                phy_rows.append({'action_index': i, 'mode': MODES[action.mode_id].name,
                    'iterations': action.iterations, 'channel': channel, 'snr_db': snr,
                    **summarize(test.errors[mask,i], test.bits[mask,i], test.blocks[mask,i],
                                total_symbols=config['symbols']+config['pilots'],
                                target_ber=config['target_ber'],
                                max_payload=MODES[action.mode_id].payload_bits(config['symbols']))})
    runtime = runtime_table(engine, actions, policies, config)
    results = {'schema_version': 5, 'config': config, 'policy_index': list(policies),
               'policy_metrics': rows, 'paired_goodput_comparisons': comparisons,
               'phy_curves': phy_rows, 'runtime': runtime, 'fit_seconds': fit_seconds,
               'test_diagnostics': prediction_diagnostics(policies, test, actions, config),
               'interpretation': 'Estimated BER feasibility is not a guarantee. Report raw/gated coverage and goodput together; no claims of universal optimality.'}
    save_json(out/'results.json', results)
    manifest.update({'status': 'complete', 'elapsed_seconds': time.perf_counter()-started,
        'policy_hash_before_validation': model_hash,
        'dataset_frames': {'train': len(train.keys), 'validation': len(validation.keys), 'test': len(test.keys)},
        'artifact_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()
                            if p.is_file() and p.name != 'manifest.json'}})
    save_json(out/'manifest.json', manifest)
    print(f'Completed: {out}', flush=True)
    return out


if __name__ == '__main__':
    run(parse_args())

