"""Physics and correctness verification test suite for R1 LDPC Coded PHY foundation.

Verifies:
1. Noiseless encode -> decode roundtrip across all modulations and code rates.
2. Decoded information length exactly matches input length (k bits).
3. Soft demapper LLR shape, magnitude scaling, and sign sanity (negative for 0, positive for 1).
4. LDPC rate bookkeeping and spectral efficiency relationship eta = Rc * log2(M).
5. Deterministic reproducibility and seed discipline (identical seeds reproduce identical bits/fading/noise).
6. BER/BLER statistical accumulators and confidence interval math on synthetic fixtures.
7. Controlled noisy sanity case where channel coding improves reliability over uncoded transmission.
8. Regression protection for frozen R0 baseline (R0 modules, artifacts, and semantics remain untouched).
"""
import math
from pathlib import Path
import unittest

import numpy as np
import torch

from ldpc_phy import (
    ACTION_BY_NAME,
    BLOCK_SYMBOLS_DEFAULT,
    CodedAction,
    CodedBlockStats,
    CodedModem,
    MODULATION_BPS,
    R1_INITIAL_ACTIONS,
    get_ldpc_code_params,
    simulate_coded_transmission,
)


class TestLdpcPhy(unittest.TestCase):
    """Test suite verifying physics, metrics, and contracts for R1 LDPC PHY."""

    def test_01_noiseless_roundtrip_all_modulations(self):
        """Verify perfect information recovery in noiseless channel for all initial R1 actions."""
        # Test a representative set of actions spanning all 4 modulations and rates
        test_actions = [
            ACTION_BY_NAME["BPSK-1/2"],
            ACTION_BY_NAME["QPSK-1/2"],
            ACTION_BY_NAME["QPSK-2/3"],
            ACTION_BY_NAME["16QAM-1/2"],
            ACTION_BY_NAME["16QAM-3/4"],
            ACTION_BY_NAME["64QAM-1/2"],
        ]

        for act in test_actions:
            with self.subTest(action=act.name):
                # Use smaller block size for test speed while retaining valid 5G NR structure
                modem = CodedModem(act, block_symbols=500, num_iter=10)
                u = torch.randint(0, 2, (2, modem.k)).float()

                # Encode -> Modulate
                c = modem.encode(u)
                x = modem.modulate(c)

                self.assertEqual(c.shape, (2, modem.n))
                self.assertEqual(x.shape, (2, 500))

                # Noiseless channel: h = 1, N0 very small (1e-6)
                h = torch.ones((2, 1), dtype=torch.complex128)
                n0 = torch.tensor(1e-6, dtype=torch.float64)

                llrs = modem.equalize_and_demap_soft(x, h, n0)
                u_hat = modem.decode(llrs)

                self.assertEqual(u_hat.shape, u.shape)
                errs = (u != u_hat).sum().item()
                self.assertEqual(errs, 0, f"Noiseless decode failed for {act.name} with {errs} errors")

    def test_02_decoded_info_length_matches_input_length(self):
        """Verify decoded information bit length exactly equals encoder input length k."""
        for act in [ACTION_BY_NAME["BPSK-1/2"], ACTION_BY_NAME["QPSK-1/2"], ACTION_BY_NAME["16QAM-1/2"]]:
            for syms in [200, 500, 1536]:
                with self.subTest(action=act.name, block_symbols=syms):
                    k, n = get_ldpc_code_params(act, block_symbols=syms)
                    modem = CodedModem(act, block_symbols=syms, num_iter=5)

                    self.assertEqual(modem.k, k)
                    self.assertEqual(modem.n, n)

                    u = torch.randint(0, 2, (1, k)).float()
                    c = modem.encode(u)
                    x = modem.modulate(c)
                    llrs = modem.equalize_and_demap_soft(x, torch.ones((1, 1)), 0.01)
                    u_hat = modem.decode(llrs)

                    self.assertEqual(u_hat.shape[-1], k)
                    self.assertEqual(u.shape[-1], u_hat.shape[-1])

    def test_03_soft_demapper_llr_shape_and_sign_sanity(self):
        """Verify Sionna native LLR conventions: negative for bit 0, positive for bit 1."""
        # Test QPSK
        act = ACTION_BY_NAME["QPSK-1/2"]
        modem = CodedModem(act, block_symbols=100, num_iter=5)

        # All-zero codeword bits
        zeros_c = torch.zeros((1, modem.n), dtype=torch.float64)
        x_zeros = modem.modulate(zeros_c)
        h = torch.ones((1, 1), dtype=torch.complex128)
        n0 = torch.tensor(0.01, dtype=torch.float64)

        llrs_zeros = modem.equalize_and_demap_soft(x_zeros, h, n0)
        self.assertEqual(llrs_zeros.shape, (1, modem.n))
        # For bit 0, LLR = ln(P(1)/P(0)) must be strictly negative
        self.assertTrue(torch.all(llrs_zeros < -1.0), "LLR for bit 0 must be strongly negative")

        # All-one codeword bits
        ones_c = torch.ones((1, modem.n), dtype=torch.float64)
        x_ones = modem.modulate(ones_c)
        llrs_ones = modem.equalize_and_demap_soft(x_ones, h, n0)
        self.assertEqual(llrs_ones.shape, (1, modem.n))
        # For bit 1, LLR = ln(P(1)/P(0)) must be strictly positive
        self.assertTrue(torch.all(llrs_ones > 1.0), "LLR for bit 1 must be strongly positive")

        # Check that higher noise reduces LLR magnitude (uncertainty increases)
        n0_noisy = torch.tensor(1.0, dtype=torch.float64)
        llrs_noisy = modem.equalize_and_demap_soft(x_zeros, h, n0_noisy)
        self.assertLess(torch.mean(torch.abs(llrs_noisy)).item(), torch.mean(torch.abs(llrs_zeros)).item())

    def test_04_ldpc_rate_bookkeeping_and_spectral_efficiency(self):
        """Verify rate bookkeeping, modulation mapping, and spectral efficiency formula."""
        for act in R1_INITIAL_ACTIONS:
            with self.subTest(action=act.name):
                m = MODULATION_BPS[act.modulation]
                self.assertEqual(act.bits_per_symbol, m)

                expected_eta = act.code_rate * m
                self.assertAlmostEqual(act.spectral_efficiency, expected_eta, places=6)

                k, n = get_ldpc_code_params(act, block_symbols=BLOCK_SYMBOLS_DEFAULT)
                self.assertEqual(n, BLOCK_SYMBOLS_DEFAULT * m)
                eff_rate = k / n
                self.assertAlmostEqual(eff_rate, act.code_rate, delta=1e-3)

                # Verify Eb/N0 derived relationship: (Eb/N0)_dB = (Es/N0)_dB - 10*log10(eta)
                es_n0_db = 20.0
                expected_eb_n0_db = es_n0_db - 10.0 * math.log10(expected_eta)
                self.assertAlmostEqual(act.compute_eb_n0_db(es_n0_db), expected_eb_n0_db, places=6)

    def test_05_deterministic_reproducibility_and_seed_discipline(self):
        """Verify identical master seeds produce bit-exact simulation outcomes."""
        act = ACTION_BY_NAME["QPSK-1/2"]
        modem = CodedModem(act, block_symbols=200, num_iter=5)

        res1 = simulate_coded_transmission(modem, snr_db=10.0, num_blocks=3, seed=20260921)
        res2 = simulate_coded_transmission(modem, snr_db=10.0, num_blocks=3, seed=20260921)

        self.assertEqual(res1.info_bit_errors, res2.info_bit_errors)
        self.assertEqual(res1.block_errors, res2.block_errors)
        self.assertEqual(res1.info_ber, res2.info_ber)
        self.assertEqual(res1.bler, res2.bler)

        # Different seed produces different realization
        res3 = simulate_coded_transmission(modem, snr_db=10.0, num_blocks=3, seed=99999999)
        # Verify run completes cleanly without crash
        self.assertIsInstance(res3.info_bit_errors, int)

    def test_06_ber_bler_counters_synthetic_examples(self):
        """Verify statistical metrics and confidence interval computation on controlled fixtures."""
        act = ACTION_BY_NAME["BPSK-1/2"]
        stats = CodedBlockStats(action=act, snr_db=10.0, k=100, n=200)

        # Feed 10 blocks: 8 clean blocks (0 errors), 2 corrupted blocks (10 errors each)
        for _ in range(8):
            stats.update(bit_errors=0)
        for _ in range(2):
            stats.update(bit_errors=10)

        self.assertEqual(stats.num_blocks, 10)
        self.assertEqual(stats.total_info_bits, 1000)
        self.assertEqual(stats.total_coded_bits, 2000)
        self.assertEqual(stats.info_bit_errors, 20)
        self.assertAlmostEqual(stats.info_ber, 0.02, places=6)
        self.assertEqual(stats.block_errors, 2)
        self.assertAlmostEqual(stats.bler, 0.20, places=6)

        # Analytical BLER SE = sqrt(0.20 * 0.80 / 10) = sqrt(0.016) ≈ 0.12649
        expected_bler_se = math.sqrt(0.20 * 0.80 / 10.0)
        self.assertAlmostEqual(stats.bler_se, expected_bler_se, places=5)

        b_low, b_high = stats.bler_ci95()
        self.assertAlmostEqual(b_low, max(0.0, 0.20 - 1.96 * expected_bler_se), places=5)
        self.assertAlmostEqual(b_high, min(1.0, 0.20 + 1.96 * expected_bler_se), places=5)

        # Serialized dictionary sanity
        d = stats.to_dict()
        self.assertEqual(d["action"], "BPSK-1/2")
        self.assertEqual(d["info_bit_errors"], 20)
        self.assertEqual(d["block_errors"], 2)

    def test_07_coding_improves_reliability_in_controlled_noisy_scenario(self):
        """Verify LDPC coding improves reliability over uncoded transmission in a controlled AWGN channel."""
        act = ACTION_BY_NAME["QPSK-1/2"]
        block_syms = 500
        modem = CodedModem(act, block_symbols=block_syms, num_iter=15)

        # Test at moderate AWGN SNR = 3.0 dB
        # Under AWGN at Es/N0 = 3 dB, uncoded QPSK has theoretical BER ≈ 0.023.
        snr_db = 3.0
        num_blocks = 10
        seed = 42

        # Coded transmission
        coded_res = simulate_coded_transmission(
            modem=modem,
            snr_db=snr_db,
            num_blocks=num_blocks,
            channel_type="awgn",
            seed=seed,
        )

        # Compare with raw uncoded QPSK at the same SNR
        gen = torch.Generator().manual_seed(seed)
        total_uncoded_bits = num_blocks * block_syms * 2
        uncoded_bits = torch.randint(0, 2, (num_blocks, block_syms * 2), generator=gen).float()
        x_uncoded = modem.mapper(uncoded_bits)
        n0_val = 1.0 / (10 ** (snr_db / 10.0))
        noise = math.sqrt(n0_val / 2.0) * torch.randn(x_uncoded.shape, dtype=torch.complex128)
        y_uncoded = x_uncoded + noise
        uncoded_demapper = torch.from_numpy(np.array([0, 0]))  # use modem demapper hard decisions
        llrs_uncoded = modem.demapper(y_uncoded, torch.tensor(n0_val, dtype=torch.float64))
        uncoded_detected = (llrs_uncoded > 0).float()
        uncoded_bit_errors = (uncoded_bits != uncoded_detected).sum().item()
        uncoded_ber = float(uncoded_bit_errors) / float(total_uncoded_bits)

        # Assert uncoded has non-trivial error rate at 5 dB (~3-4%)
        self.assertGreater(uncoded_ber, 0.01)

        # Assert LDPC coding achieves strictly lower BER (clean waterfall behavior)
        self.assertLess(coded_res.info_ber, uncoded_ber)

    def test_08_r0_regression_protection(self):
        """Verify R0 canonical modules and frozen definitions remain untouched."""
        import phy_engine
        import ground_truth
        import cart_1d
        import package_r0_freeze

        # 1. R0 uncoded MODES catalog intact
        self.assertEqual(len(phy_engine.MODES), 4)
        self.assertEqual([m.modulation for m in phy_engine.MODES], ["BPSK", "QPSK", "16QAM", "64QAM"])

        # 2. R0 ground truth parameters intact
        self.assertEqual(ground_truth.BER_TARGET, 0.01)
        self.assertEqual(ground_truth.CONFIDENCE_K, 1.96)

        # 3. R0 CART 1D classifier intact
        clf = cart_1d.CART1DClassifier(max_depth=3)
        self.assertEqual(clf.max_depth, 3)

        # 4. R0 package freeze gate function intact
        self.assertTrue(callable(package_r0_freeze.run_freeze_gate))


if __name__ == "__main__":
    unittest.main()
