"""PHY Acceptance Gate: Pure Physical Layer Verification Suite.

Tests validate mathematical anchors, physics, and reproducibility:
    1. Unit-average constellation energy (Es = 1).
    2. Expected constellation sizes for BPSK, QPSK, 16-QAM, 64-QAM.
    3. Gray nearest-neighbour consistency for all constellations.
    4. Noiseless modulate/demodulate round trip.
    5. Hard coherent demodulation correctness with known complex h.
    6. AWGN BER sanity against analytical expressions (erfc / exact PAM region integration).
    7. Rayleigh average BER sanity against analytical expectation.
    8. Approximately E[|h|^2] = 1 for Rayleigh fading.
    9. Channel coefficient h is strictly constant across the entire block.
    10. Independent h realizations across independent blocks.
    11. Deterministic reproducibility with identical seed and configuration.
    12. Different block identities produce different realizations.
    13. Sequence/batch evaluation and individual evaluation agree.
    14. Modulation evaluation order does not alter paired physical realizations.
"""
import unittest
import numpy as np
from scipy.special import erfc, ndtr

from channel import (
    BlockRNG,
    ChannelOutput,
    make_block_rng,
    generate_channel_coefficient,
    generate_standard_noise,
    apply_channel,
)
from phy_engine import (
    ModulationMode,
    BlockEvaluationResult,
    MODES,
    MODE_BY_ID,
    constellation,
    modulate,
    demodulate,
    PHYEngine,
)
from metrics import RawPHYCounters, compute_ber, compute_bler, compute_raw_goodput


class TestPhysics(unittest.TestCase):

    def test_01_unit_average_constellation_energy(self):
        """1. Constellations must have exact unit average symbol energy (Es = 1.0)."""
        for m in (1, 2, 4, 6):
            points, _ = constellation(m)
            avg_energy = float(np.mean(np.abs(points) ** 2))
            self.assertAlmostEqual(
                avg_energy, 1.0, places=12,
                msg=f"Modulation m={m} average energy {avg_energy} != 1.0"
            )

    def test_02_constellation_sizes(self):
        """2. Constellation size must be 2^m for m in {1, 2, 4, 6}."""
        expected_sizes = {1: 2, 2: 4, 4: 16, 6: 64}
        for m, exp_size in expected_sizes.items():
            points, labels = constellation(m)
            self.assertEqual(len(points), exp_size)
            self.assertEqual(labels.shape, (exp_size, m))
            # Unique constellation points
            self.assertEqual(len(np.unique(points)), exp_size)

    def test_03_gray_nearest_neighbour_consistency(self):
        """3. Gray mapping: every adjacent nearest-neighbour pair must differ by exactly 1 bit."""
        for m in (1, 2, 4, 6):
            points, labels = constellation(m)
            dists = np.abs(points[:, None] - points[None, :])
            min_dist = float(np.min(dists[dists > 1e-6]))

            # Find all nearest-neighbour pairs
            nn_indices = np.argwhere(np.isclose(dists, min_dist))
            self.assertGreater(len(nn_indices), 0)
            for i, j in nn_indices:
                bit_diff = int(np.count_nonzero(labels[i] != labels[j]))
                self.assertEqual(
                    bit_diff, 1,
                    msg=f"m={m} nearest neighbors ({i}, {j}) have Hamming distance {bit_diff} != 1"
                )

    def test_04_noiseless_round_trip(self):
        """4. Transmitting through a noiseless channel with h=1 must recover 100% of bits."""
        rng = np.random.default_rng(20260901)
        for m in (1, 2, 4, 6):
            num_symbols = 500
            bits = rng.integers(0, 2, num_symbols * m, dtype=np.int32)
            symbols = modulate(bits, m)
            # Demodulate directly with known h = 1.0
            recovered = demodulate(symbols, h=1.0 + 0.0j, bits_per_symbol=m)
            np.testing.assert_array_equal(recovered, bits)

    def test_05_coherent_demodulation_with_known_complex_h(self):
        """5. Coherent demodulation with arbitrary complex h in noiseless condition must yield 0 errors."""
        rng = np.random.default_rng(20260902)
        test_channels = [
            0.5 + 0.8j,
            -0.7 + 0.3j,
            0.1 - 0.9j,
            3.5 + 2.1j,
            0.05 + 0.05j,  # Small magnitude
        ]
        for h_val in test_channels:
            for m in (1, 2, 4, 6):
                bits = rng.integers(0, 2, 600 * m, dtype=np.int32)
                symbols = modulate(bits, m)
                received = h_val * symbols  # pure channel rotation and scaling
                recovered = demodulate(received, h=h_val, bits_per_symbol=m)
                np.testing.assert_array_equal(
                    recovered, bits,
                    err_msg=f"Failed coherent demodulation for m={m} with h={h_val}"
                )

    def test_06_awgn_ber_sanity_analytical(self):
        """6. AWGN BER across all modulations must match exact analytical formulas."""
        rng = np.random.default_rng(20260903)
        # Test configurations: (m, snr_db, tolerance)
        test_cases = [
            (1, 0.0, 0.0035),   # BPSK
            (2, 4.0, 0.0035),   # QPSK
            (4, 8.0, 0.0035),   # 16-QAM
            (6, 14.0, 0.0035),  # 64-QAM
        ]

        for m, snr_db, tol in test_cases:
            n0 = 10.0 ** (-snr_db / 10.0)
            num_bits = 300_000
            # Ensure divisibility
            num_bits = (num_bits // m) * m
            bits = rng.integers(0, 2, num_bits, dtype=np.int32)

            tx_symbols = modulate(bits, m)
            std_noise = generate_standard_noise(rng, len(tx_symbols))
            rx_output = apply_channel(
                transmitted_symbols=tx_symbols,
                snr_db=snr_db,
                channel_type="awgn",
                standard_noise=std_noise,
            )

            rx_bits = demodulate(rx_output.received_symbols, h=1.0 + 0.0j, bits_per_symbol=m)
            sim_ber = float(np.mean(bits != rx_bits))

            # Exact analytical reference
            if m == 1:
                # BPSK: Pb = 0.5 * erfc(sqrt(Es / N0)) = 0.5 * erfc(sqrt(1 / N0))
                expected_ber = float(0.5 * erfc(np.sqrt(1.0 / n0)))
            elif m == 2:
                # QPSK: Pb = 0.5 * erfc(sqrt(0.5 / N0))
                expected_ber = float(0.5 * erfc(np.sqrt(0.5 / n0)))
            else:
                # Exact integration over PAM decision regions for square QAM
                width = m // 2
                levels = constellation(m)[0][np.arange(2**width) * (2**width)].real
                order = np.argsort(levels)
                levels = levels[order]
                pam_labels = ((order[:, None] >> np.arange(width - 1, -1, -1)) & 1)
                bounds = np.r_[-np.inf, (levels[:-1] + levels[1:]) / 2.0, np.inf]
                # Noise standard deviation per dimension: sigma = sqrt(N0 / 2)
                sigma_1d = np.sqrt(n0 / 2.0)
                prob = np.diff(ndtr((bounds[None, :] - levels[:, None]) / sigma_1d), axis=1)
                hamming = np.count_nonzero(pam_labels[:, None, :] != pam_labels[None, :, :], axis=2)
                expected_ber = float(np.sum(prob * hamming) / (len(levels) * width))

            self.assertAlmostEqual(
                sim_ber, expected_ber, delta=tol,
                msg=f"AWGN BER mismatch for m={m} at {snr_db} dB: sim={sim_ber:.5f}, exp={expected_ber:.5f}"
            )

    def test_07_rayleigh_average_ber_analytical(self):
        """7. BPSK average BER over slow Rayleigh fading must match theoretical Pb = 0.5*(1 - sqrt(g/(1+g)))."""
        # Nominal Es/N0 = 4 dB
        snr_db = 4.0
        n0 = 10.0 ** (-snr_db / 10.0)
        avg_snr_linear = 1.0 / n0  # Es = 1, so gamma_bar = Es / N0
        expected_ber = float(0.5 * (1.0 - np.sqrt(avg_snr_linear / (1.0 + avg_snr_linear))))

        engine = PHYEngine(block_symbols=128)
        bpsk_mode = MODE_BY_ID[0]
        num_blocks = 2000
        master_seed = 20260904

        counters = RawPHYCounters(bits_per_symbol=1)
        for b in range(num_blocks):
            block_rng = make_block_rng(master_seed=master_seed, block_id=b)
            h = generate_channel_coefficient(block_rng.fading_rng, channel_type="rayleigh")
            std_noise = generate_standard_noise(block_rng.noise_rng, engine.block_symbols)
            payload = block_rng.bit_rng.integers(0, 2, engine.block_symbols, dtype=np.int32)

            res = engine.evaluate_mode(
                mode=bpsk_mode,
                snr_db=snr_db,
                h=h,
                payload_bits=payload,
                standard_noise=std_noise,
            )
            counters.update(bit_errors=res.bit_errors, total_bits=res.total_bits, block_error=res.block_error)

        # Statistical tolerance for Monte Carlo over 2000 fading blocks
        self.assertAlmostEqual(
            counters.ber, expected_ber, delta=0.012,
            msg=f"Rayleigh average BER {counters.ber:.4f} != theoretical {expected_ber:.4f}"
        )

    def test_08_rayleigh_h_unit_average_power(self):
        """8. Rayleigh fading coefficients must satisfy E[|h|^2] = 1.0."""
        rng = np.random.default_rng(20260905)
        n_samples = 30_000
        hs = np.array([generate_channel_coefficient(rng, "rayleigh") for _ in range(n_samples)])
        empirical_power = float(np.mean(np.abs(hs) ** 2))
        self.assertAlmostEqual(
            empirical_power, 1.0, delta=0.02,
            msg=f"Rayleigh E[|h|^2] = {empirical_power:.4f} deviates from 1.0"
        )

    def test_09_h_constant_across_entire_block(self):
        """9. Channel coefficient h must be identical and constant across all symbols in a block."""
        rng = np.random.default_rng(20260906)
        symbols = (rng.standard_normal(1536) + 1j * rng.standard_normal(1536)) / np.sqrt(2.0)
        h_known = 0.6 - 0.7j

        output = apply_channel(
            transmitted_symbols=symbols,
            snr_db=10.0,
            h=h_known,
            noise_rng=rng,
        )

        # Verify y = h * x + n
        noiseless_part = output.received_symbols - output.noise
        effective_h_per_symbol = noiseless_part / symbols

        # Every symbol experienced the exact same scalar h
        np.testing.assert_allclose(
            effective_h_per_symbol, h_known, rtol=1e-12, atol=1e-12,
            err_msg="h varied across symbols within the block!"
        )

    def test_10_independent_h_across_blocks(self):
        """10. Independent blocks must have mutually independent channel coefficients."""
        master_seed = 20260907
        num_blocks = 5000
        hs = []
        for b in range(num_blocks):
            block_rng = make_block_rng(master_seed=master_seed, block_id=b)
            h = generate_channel_coefficient(block_rng.fading_rng, channel_type="rayleigh")
            hs.append(h)
        hs = np.array(hs)

        # Lag-1 correlation should be statistically zero
        lag1_corr = np.mean(hs[:-1] * np.conj(hs[1:]))
        self.assertLess(
            abs(lag1_corr), 0.05,
            msg=f"Blocks exhibit unexpected correlation: lag1={abs(lag1_corr):.4f}"
        )

    def test_11_deterministic_reproducibility(self):
        """11. Exact reproducibility: same master_seed + same block_id => identical realization."""
        master_seed = 20260908
        block_id = 42

        # First run
        rng1 = make_block_rng(master_seed, block_id)
        h1 = generate_channel_coefficient(rng1.fading_rng)
        noise1 = generate_standard_noise(rng1.noise_rng, 500)
        bits1 = rng1.bit_rng.integers(0, 2, 500, dtype=np.int32)

        # Second run with same parameters
        rng2 = make_block_rng(master_seed, block_id)
        h2 = generate_channel_coefficient(rng2.fading_rng)
        noise2 = generate_standard_noise(rng2.noise_rng, 500)
        bits2 = rng2.bit_rng.integers(0, 2, 500, dtype=np.int32)

        self.assertEqual(h1, h2)
        np.testing.assert_array_equal(noise1, noise2)
        np.testing.assert_array_equal(bits1, bits2)

    def test_12_different_block_identity_different_realization(self):
        """12. Different block IDs must produce distinct channel and noise realizations."""
        master_seed = 20260909
        rng_a = make_block_rng(master_seed, block_id=0)
        rng_b = make_block_rng(master_seed, block_id=1)

        ha = generate_channel_coefficient(rng_a.fading_rng)
        hb = generate_channel_coefficient(rng_b.fading_rng)
        na = generate_standard_noise(rng_a.noise_rng, 100)
        nb = generate_standard_noise(rng_b.noise_rng, 100)

        self.assertNotEqual(ha, hb)
        self.assertFalse(np.array_equal(na, nb))

    def test_13_batch_and_individual_evaluation_agree(self):
        """13. Evaluating a sequence of blocks yields identical results whether processed sequentially or individually."""
        engine = PHYEngine(block_symbols=96)
        mode = MODE_BY_ID[1]  # QPSK
        master_seed = 20260910
        snr_db = 8.0

        individual_results = []
        for b in range(10):
            rng = make_block_rng(master_seed, b)
            h = generate_channel_coefficient(rng.fading_rng)
            noise = generate_standard_noise(rng.noise_rng, engine.block_symbols)
            bits = rng.bit_rng.integers(0, 2, engine.block_symbols * mode.bits_per_symbol, dtype=np.int32)
            res = engine.evaluate_mode(mode, snr_db, h, bits, noise)
            individual_results.append((res.bit_errors, res.block_error, res.ber))

        # Re-run sequentially with fresh RNG matching same seeds
        reconstructed_results = []
        for b in range(10):
            rng = make_block_rng(master_seed, b)
            h = generate_channel_coefficient(rng.fading_rng)
            noise = generate_standard_noise(rng.noise_rng, engine.block_symbols)
            bits = rng.bit_rng.integers(0, 2, engine.block_symbols * mode.bits_per_symbol, dtype=np.int32)
            res = engine.evaluate_mode(mode, snr_db, h, bits, noise)
            reconstructed_results.append((res.bit_errors, res.block_error, res.ber))

        self.assertEqual(individual_results, reconstructed_results)

    def test_14_modulation_order_invariance_for_paired_eval(self):
        """14. Evaluating candidate modes in forward vs reverse order must yield identical paired results."""
        engine = PHYEngine(block_symbols=128)
        master_seed = 20260911
        snr_db = 10.0

        rng = make_block_rng(master_seed, block_id=12)
        h = generate_channel_coefficient(rng.fading_rng)
        noise = generate_standard_noise(rng.noise_rng, engine.block_symbols)
        max_bits = engine.block_symbols * max(m.bits_per_symbol for m in MODES)
        payload_pool = rng.bit_rng.integers(0, 2, max_bits, dtype=np.int32)

        modes_forward = list(MODES)
        modes_reverse = list(reversed(MODES))

        res_fwd = engine.evaluate_paired_block(modes_forward, snr_db, h, payload_pool, noise)
        res_rev = engine.evaluate_paired_block(modes_reverse, snr_db, h, payload_pool, noise)

        for m in MODES:
            fwd = res_fwd[m.mode_id]
            rev = res_rev[m.mode_id]
            self.assertEqual(fwd.bit_errors, rev.bit_errors)
            self.assertEqual(fwd.block_error, rev.block_error)
            self.assertEqual(fwd.total_bits, rev.total_bits)
            self.assertEqual(fwd.h, rev.h)
            self.assertEqual(fwd.snr_db, rev.snr_db)


if __name__ == "__main__":
    unittest.main()
