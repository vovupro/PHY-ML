"""Independent PHY references, posterior/causality checks and end-to-end tests."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from scipy.special import erfc, ndtr

from channel import CHANNELS, Observation, frame_input, posterior_initial, correlation
from phy_engine import (PHYEngine, MODES, action_catalog, constellation, modulate,
                        demodulate, compute_crc24a, verify_crc24a, crc24a_many)
from policies import (choose, compute_oracle_labels, FixedPolicy, EmpiricalLUT,
                      DecisionTreeClassifierPolicy, ServicePolicy, fit_policies)
from metrics import summarize
from experiment import collect, assert_disjoint, cross_validate, evaluate


class TestPHY(unittest.TestCase):
    def test_catalogue_and_gray_modems(self):
        for coded in (False, True):
            self.assertEqual({m.bits_per_symbol for m in MODES if m.coded == coded}, {1,2,4,6})
        for m in (1,2,4,6):
            points, labels = constellation(m)
            self.assertAlmostEqual(np.mean(abs(points)**2), 1., places=12)
            h = np.linspace(.2, 1., len(points)) + .3j
            np.testing.assert_array_equal(demodulate(h*points, h, .001, m), labels.ravel())
            np.testing.assert_array_equal(demodulate(h*points, h, .001, m, True)>0, labels.ravel())
            d = abs(points[:,None]-points[None,:])
            for i,j in np.argwhere(np.isclose(d, d[d>0].min())):
                self.assertEqual(np.count_nonzero(labels[i] != labels[j]), 1)

    def test_crc_external_known_vector_and_batch(self):
        bits = np.unpackbits(np.frombuffer(b'123456789', dtype=np.uint8))
        self.assertEqual(int(''.join(map(str, compute_crc24a(bits))),2), 0xcde703)
        rng = np.random.default_rng(9)
        for length in (1,7,8,31,96,2304):
            values = rng.integers(0,2,(5,length))
            np.testing.assert_array_equal(crc24a_many(values), np.stack([compute_crc24a(v) for v in values]))
        word = np.r_[bits, compute_crc24a(bits)]
        self.assertTrue(verify_crc24a(word))
        for i in range(len(word)):
            broken = word.copy()
            broken[i] ^= 1
            self.assertFalse(verify_crc24a(broken))

    def test_awgn_analytic_all_modulations(self):
        rng = np.random.default_rng(101)
        for m,snr in ((1,0.),(2,4.),(4,8.),(6,14.)):
            n0 = 10**(-snr/10)
            bits = rng.integers(0,2,240000)
            tx = modulate(bits,m)
            noise = (rng.normal(size=len(tx))+1j*rng.normal(size=len(tx)))*np.sqrt(n0/2)
            actual = np.mean(demodulate(tx+noise,1.,n0,m) != bits)
            if m == 1:
                expected = .5*erfc(np.sqrt(1/n0))
            elif m == 2:
                expected = .5*erfc(np.sqrt(.5/n0))
            else:
                # Exact integration over PAM decision regions, including distant
                # symbol errors. Independent of demodulator distance implementation.
                width = m//2
                levels = constellation(m)[0][np.arange(2**width)*(2**width)].real
                order = np.argsort(levels)
                levels = levels[order]
                labels = ((order[:,None] >> np.arange(width-1,-1,-1)) & 1)
                bounds = np.r_[-np.inf,(levels[:-1]+levels[1:])/2,np.inf]
                prob = np.diff(ndtr((bounds[None,:]-levels[:,None])/np.sqrt(n0/2)),axis=1)
                distance = np.count_nonzero(labels[:,None,:] != labels[None,:,:],axis=2)
                expected = float(np.sum(prob*distance)/(len(levels)*width))
            self.assertAlmostEqual(actual, expected, delta=.0035)

    def test_rayleigh_mean_ber_slow_and_fast(self):
        engine, actions = PHYEngine(96), action_catalog()[:1]
        n0 = 10**(-4/10)
        expected = .5*(1-np.sqrt(1/(1+n0)))
        for ch in (1,2):
            samples = [frame_input(master_seed=58,split='development',channel_id=ch,
                        snr_index=0,snr_db=4.,frame=i,symbols=96,csi='perfect') for i in range(1500)]
            result = engine.evaluate_batch(actions,samples)
            ber = result['errors'].sum()/result['bits'].sum()
            self.assertAlmostEqual(ber,expected,delta=.012)

    def test_zero_gain(self):
        for m in (1,2,4,6):
            np.testing.assert_allclose(demodulate(np.array([1j,0j]),0j,1.,m,True),0,atol=1e-6)

    def test_uncoded_batch_matches_individual_and_action_order(self):
        engine, actions = PHYEngine(96), action_catalog()
        sources = [frame_input(master_seed=67,split='development',channel_id=i%3,
                    snr_index=0,snr_db=10.,frame=i,symbols=96) for i in range(6)]
        batch = engine.evaluate_batch(actions,sources)
        reverse = engine.evaluate_batch(actions[::-1],sources)
        for key in ('errors','bits','blocks','undetected'):
            np.testing.assert_array_equal(batch[key],reverse[key][:,::-1])
        for i,s in enumerate(sources):
            one = engine.evaluate(actions,h=s.h,h_est=s.h_est,n0=s.n0,payload_pool=s.payload_pool,standard_noise=s.standard_noise)
            for key,attribute in [('errors','bit_errors'),('bits','payload_bits'),('blocks','block_error')]:
                np.testing.assert_array_equal(batch[key][i],[r[attribute] for r in one])


class TestChannel(unittest.TestCase):
    def test_rayleigh_moments_correlation_and_tracking_mse(self):
        for ch in (1,2):
            sources = [frame_input(master_seed=119,split='development',channel_id=ch,
                        snr_index=0,snr_db=8.,frame=i,symbols=96) for i in range(3000)]
            h = np.stack([s.h[::3] for s in sources])
            h_est = np.stack([s.h_est[::3] for s in sources])
            self.assertAlmostEqual(np.mean(abs(h)**2),1.,delta=.06)
            self.assertAlmostEqual(np.mean(abs(h)**4),2.,delta=.2)
            for lag in (1,16,31):
                actual = np.mean(h[:,lag]*h[:,0].conj()).real
                self.assertAlmostEqual(actual,correlation(ch,32)**lag,delta=.07)
            self.assertAlmostEqual(np.mean(abs(h-h_est)**2),sources[0].n0,delta=.012)

    def test_conditional_posterior_and_future_resampling(self):
        obs = Observation(3.,6.,1)
        rng = np.random.default_rng(3)
        samples = np.array([posterior_initial(obs,rng)[0] for _ in range(20000)])
        n0 = 10**(-.6)
        z = np.sqrt(n0*10**(.3))
        self.assertAlmostEqual(samples.mean().real,z/(1+n0),delta=.012)
        self.assertAlmostEqual(np.mean(abs(samples-z/(1+n0))**2),n0/(1+n0),delta=.008)
        args = dict(master_seed=44,split='train',channel_id=1,snr_index=0,snr_db=6.,
                    frame=2,symbols=96,condition=obs)
        a,b = [frame_input(**args,repeat=i) for i in (0,1)]
        self.assertEqual(a.observation,b.observation)
        self.assertNotEqual(a.h[0],b.h[0])
        self.assertNotEqual(a.h[-1],b.h[-1])
        np.testing.assert_array_equal(a.h_est[:3],b.h_est[:3])
        self.assertFalse(np.array_equal(a.payload_pool,b.payload_pool))
        self.assertFalse(np.array_equal(a.standard_noise,b.standard_noise))

    def test_action_independent_observation_and_split_keys(self):
        args = dict(master_seed=81,channel_id=2,snr_index=0,snr_db=6.,frame=1)
        a = frame_input(**args,split='train',symbols=96)
        b = frame_input(**args,split='train',symbols=192)
        self.assertEqual(a.observation,b.observation)
        self.assertEqual(set(vars(a.observation)),{'snr_est_db','nominal_snr_db','channel_id'})
        keys = [frame_input(**args,split=s,symbols=96).seed_key for s in ('train','validation','test')]
        self.assertEqual(len(set(keys)),3)


class TestMethod(unittest.TestCase):
    def config(self):
        return dict(seed=151,snrs=[0.,12.],symbols=96,pilots=32,csi='pilot',
                    target_ber=.001,objective='goodput',batch_size=8)

    def test_objective_shared_by_labels_and_selector(self):
        # Same BER, very different block delivery rates.
        bits = np.array([[1000,2000,4000]])
        trials = np.array([10])
        errors = np.array([[0,1,2]])
        blocks = np.array([[0,8,9]])
        for objective,wanted in [('goodput',0),('correct_bits',2)]:
            result = compute_oracle_labels(errors,bits,blocks,trials,3,.001,objective=objective)
            self.assertEqual(result[0],wanted)

    def test_classifier_never_claims_ber_probability(self):
        obs = [Observation(float(i),12.,0) for i in range(20)]
        model = DecisionTreeClassifierPolicy(action_catalog(),min_samples_leaf=1).fit(obs,np.zeros(20,dtype=int))
        self.assertTrue(all(d.predicted_feasible is None for d in model.decide_many(obs)))

    def test_identical_gate_for_all_base_policies(self):
        actions = action_catalog()
        obs = [Observation(10.,10.,ch) for ch in range(3)]
        targets = np.full((3,len(actions),2),.5)
        lut = EmpiricalLUT(actions,96,.001).fit(obs,targets)
        for base in (FixedPolicy(actions),FixedPolicy(actions,False),lut):
            gated = ServicePolicy(base,lut,.001,actions)
            decisions = gated.decide_many(obs)
            self.assertTrue(all(d.action_index == len(actions) and not d.predicted_feasible for d in decisions))

    def test_defer_metrics_and_pilot_accounting(self):
        row = summarize([0,0],[0,0],[0,0],total_symbols=128,target_ber=.001,max_payload=576)
        self.assertIsNone(row['ber'])
        self.assertIsNone(row['violation_rate'])
        self.assertEqual(row['goodput_bits_per_symbol'],0.)
        self.assertFalse(row['ber_certified_at_stated_confidence'])
        row = summarize([0,0],[96,0],[0,0],total_symbols=128,target_ber=.001,max_payload=576)
        self.assertAlmostEqual(row['goodput_bits_per_symbol'],96/256)

    def test_conditional_collect_and_independent_splits(self):
        config,actions,engine = self.config(),action_catalog(),PHYEngine(96)
        train = collect(engine,actions,config,'train',5,mc_trials=4)
        validation = collect(engine,actions,config,'validation',2)
        assert_disjoint(train,validation)
        with self.assertRaises(ValueError): assert_disjoint(train,train)
        self.assertTrue(np.all((train.targets>=0)&(train.targets<=1)))
        policies = fit_policies(train,actions,config)
        for p in policies.values():
            a = p.decide_many(validation.observations)
            b = [p.decide(o) for o in validation.observations]
            self.assertEqual(a,b)
        cv = cross_validate(train,actions,config)
        self.assertEqual(len(cv),50)
        rows,trace,paired = evaluate(policies,validation,actions,config)
        self.assertEqual(len(rows),60)
        self.assertEqual(len(paired),84)
        with self.assertRaises(ValueError): evaluate(policies,train,actions,config)


class TestLDPC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_crc_against_sionna_and_noisy_llr_reference(self):
        import torch
        from sionna.phy.fec.crc import CRCEncoder
        from sionna.phy.mapping import Constellation, Demapper
        rng = np.random.default_rng(5)
        bits = rng.integers(0,2,(3,127),dtype=np.int32)
        native = CRCEncoder('CRC24A',device='cpu')(torch.tensor(bits,dtype=torch.float32)).numpy()[:,-24:]
        np.testing.assert_array_equal(crc24a_many(bits),native)
        for m in (1,2,4,6):
            points,_ = constellation(m)
            reference = Demapper('app', constellation=Constellation('custom',m,points=points,
                                 normalize=False,center=False,precision='double',device='cpu'),
                                 precision='double',device='cpu')
            h = .3 + .4j
            y = rng.normal(size=40)+1j*rng.normal(size=40)
            ref = reference(torch.tensor(y/h)[None],torch.tensor(.2/abs(h)**2,dtype=torch.float64)).numpy().ravel()
            np.testing.assert_allclose(demodulate(y,h,.2,m,True),ref,rtol=3e-6,atol=3e-6)

    def test_all_coded_modes_roundtrip_and_batch_parity(self):
        actions,engine = action_catalog('ldpc'),PHYEngine(96)
        sources = [frame_input(master_seed=7,split='development',channel_id=ch,snr_index=0,
                   snr_db=snr,frame=i,symbols=96) for i,(ch,snr) in enumerate([(0,60.),(1,14.),(2,18.)])]
        batch = engine.evaluate_batch(actions,sources)
        self.assertTrue(np.all(batch['errors'][0] == 0))
        self.assertTrue(np.all(batch['blocks'][0] == 0))
        for i,s in enumerate(sources):
            values = engine.evaluate(actions,h=s.h,h_est=s.h_est,n0=s.n0,
                                     payload_pool=s.payload_pool,standard_noise=s.standard_noise)
            for key,attribute in [('errors','bit_errors'),('bits','payload_bits'),('blocks','block_error'),('undetected','undetected_error')]:
                np.testing.assert_array_equal(batch[key][i],[r[attribute] for r in values])


class TestCLI(unittest.TestCase):
    def test_numeric_only_artifacts_and_verification(self):
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'study'
            result = subprocess.run([sys.executable,str(root/'run_benchmark.py'),'--coding','uncoded',
                '--symbols','96','--snrs','0','12','--frames-train','5','--mc-trials-train','2',
                '--frames-validation','2','--frames-test','2','--runtime-repetitions','2',
                '--output',str(out)],capture_output=True,text=True,timeout=120,cwd=temp)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertFalse(list(out.glob('*.png')))
            self.assertFalse(list(out.glob('*.md')))
            self.assertTrue((out/'diagnostics.json').exists())
            replay = subprocess.run([sys.executable,str(root/'verify_results.py'),str(out)],
                                    capture_output=True,text=True,timeout=120,cwd=temp)
            self.assertEqual(replay.returncode,0,replay.stderr)
            report = json.loads((out/'verification.json').read_text())
            self.assertEqual(report['status'],'passed')
            # Missing numerical outputs must be rejected, not just a stale green flag.
            (out/'test.npz').write_bytes(b'corrupted')
            failed = subprocess.run([sys.executable,str(root/'verify_results.py'),str(out)],
                                    capture_output=True,text=True,timeout=30,cwd=temp)
            self.assertNotEqual(failed.returncode,0)


if __name__ == '__main__':
    unittest.main()

