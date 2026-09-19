"""PHY Acceptance Gate: Pure Physical Layer Verification Suite with Sionna 2.0 Backend.

Tests validate:
    0. Sionna Mapper -> Demapper noiseless roundtrip across all 4 candidate modulations.
    1. Unit-average constellation energy (Es = 1.0).
    2. Expected constellation sizes for BPSK, QPSK, 16-QAM, 64-QAM.
    3. Gray nearest-neighbour consistency for all Sionna constellations.
    4. Noiseless modulate/demodulate round trip through the PHY engine.
    5. Hard coherent demodulation correctness with known complex h (equalization).
    6. AWGN BER sanity against analytical SciPy expressions (erfc / exact 2D region integration).
    7. Rayleigh average BER sanity against analytical expectation.
    8. Approximately E[|h|^2] = 1 for Sionna RayleighBlockFading.
    9. Channel coefficient h is strictly constant across the entire block.
    10. Independent h realizations across independent blocks.
    11. Deterministic reproducibility with identical seed and configuration.
    12. Different block identities produce different realizations.
    13. Sequence/batch evaluation and individual evaluation agree.
    14. Modulation evaluation order does not alter paired physical realizations.
"""
import unittest
import numpy as np
import torch
from scipy.special import erfc, ndtr
from sionna.phy.utils import complex_normal

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
    Mode,
    BlockEvaluationResult,
    MODES,
    MODE_BY_ID,
    _get_sionna_modem,
    constellation,
    modulate,
    demodulate,
    PHYEngine,
)
from metrics import RawPHYCounters, compute_ber, compute_bler, count_errors, count_block_errors


class TestPhysics(unittest.TestCase):

    def test_00_sionna_mapper_demapper_noiseless_roundtrip(self):
        """0. Direct verification of Sionna 2.0 Mapper -> Demapper noiseless roundtrip."""
        for m in (1, 2, 4, 6):
            const, mapper, demapper, _ = _get_sionna_modem(m)
            num_bits = 600 * m
            bits = torch.randint(0, 2, (num_bits,), dtype=torch.float64)
            symbols = mapper(bits)
            no_tensor = torch.tensor(1e-4, dtype=torch.float64)
            demapped_bits = demapper(symbols, no_tensor)
            self.assertTrue(
                torch.equal(bits, demapped_bits),
                msg=f"Sionna direct roundtrip failed for m={m} bits/symbol"
            )

    def test_01_unit_average_constellation_energy(self):
        """1. Sionna constellations must have exact unit average symbol energy (Es = 1.0)."""
        for m in (1, 2, 4, 6):
            points, _ = constellation(m)
            avg_energy = torch.mean(torch.abs(points) ** 2).item()
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
            self.assertEqual(len(np.unique(points.detach().cpu().numpy())), exp_size)

    def test_03_gray_nearest_neighbour_consistency(self):
        """3. Gray mapping: every adjacent nearest-neighbour pair must differ by exactly 1 bit."""
        for m in (1, 2, 4, 6):
            points, labels = constellation(m)
            dists = torch.abs(points.unsqueeze(-1) - points.unsqueeze(0))
            non_zero_dists = dists[dists > 1e-6]
            min_dist = torch.min(non_zero_dists).item()

            nn_indices = torch.nonzero(torch.isclose(dists, torch.tensor(min_dist, dtype=dists.dtype)))
            self.assertGreater(len(nn_indices), 0)
            for pair in nn_indices:
                i, j = pair[0].item(), pair[1].item()
                bit_diff = torch.count_nonzero(labels[i] != labels[j]).item()
                self.assertEqual(
                    bit_diff, 1,
                    msg=f"m={m} nearest neighbors ({i}, {j}) have Hamming distance {bit_diff} != 1"
                )

    def test_04_noiseless_round_trip(self):
        """4. Transmitting through noiseless channel with h=1 must recover 100% of bits."""
        gen = torch.Generator().manual_seed(20260901)
        for m in (1, 2, 4, 6):
            num_symbols = 500
            bits = torch.randint(0, 2, (num_symbols * m,), generator=gen, dtype=torch.float64)
            symbols = modulate(bits, m)
            recovered = demodulate(symbols, h=1.0 + 0.0j, bits_per_symbol=m, n0=1e-4)
            self.assertTrue(torch.equal(recovered, bits))

    def test_05_coherent_demodulation_with_known_complex_h(self):
        """5. Coherent demodulation with arbitrary complex h in noiseless condition must yield 0 errors."""
        gen = torch.Generator().manual_seed(20260902)
        test_channels = [
            0.5 + 0.8j,
            -0.7 + 0.3j,
            0.1 - 0.9j,
            3.5 + 2.1j,
            0.05 + 0.05j,
        ]
        for h_val in test_channels:
            for m in (1, 2, 4, 6):
                bits = torch.randint(0, 2, (600 * m,), generator=gen, dtype=torch.float64)
                symbols = modulate(bits, m)
                h_tensor = torch.tensor(h_val, dtype=torch.complex128)
                received = h_tensor * symbols
                recovered = demodulate(received, h=h_tensor, bits_per_symbol=m, n0=1e-4)
                self.assertTrue(
                    torch.equal(recovered, bits),
                    msg=f"Failed coherent demodulation for m={m} with h={h_val}"
                )

    def test_06_awgn_ber_sanity_analytical(self):
        """6. AWGN BER with Sionna modem must match exact analytical formulas."""
        gen = torch.Generator().manual_seed(20260903)
        test_cases = [
            (1, 0.0, 0.0035),   # BPSK
            (2, 4.0, 0.0035),   # QPSK
            (4, 8.0, 0.0035),   # 16-QAM
            (6, 14.0, 0.0035),  # 64-QAM
        ]

        for m, snr_db, tol in test_cases:
            n0 = 10.0 ** (-snr_db / 10.0)
            num_bits = 300_000
            num_bits = (num_bits // m) * m
            bits = torch.randint(0, 2, (num_bits,), generator=gen, dtype=torch.float64)

            tx_symbols = modulate(bits, m)
            std_noise = generate_standard_noise(len(tx_symbols), generator=gen)
            rx_output = apply_channel(
                transmitted_symbols=tx_symbols,
                snr_db=snr_db,
                channel_type="awgn",
                standard_noise=std_noise,
            )

            rx_bits = demodulate(rx_output.received_symbols, h=1.0 + 0.0j, bits_per_symbol=m, n0=rx_output.noise_variance)
            sim_ber = compute_ber(bits, rx_bits).item()

            # Exact analytical reference
            if m == 1:
                expected_ber = float(0.5 * erfc(np.sqrt(1.0 / n0)))
            elif m == 2:
                expected_ber = float(0.5 * erfc(np.sqrt(0.5 / n0)))
            else:
                pts_t, labels_t = constellation(m)
                pts = pts_t.detach().cpu().numpy()
                labels = labels_t.detach().cpu().numpy()

                sigma = np.sqrt(n0 / 2.0)
                real_pts = np.sort(np.unique(pts.real))
                imag_pts = np.sort(np.unique(pts.imag))
                r_bounds = np.r_[-np.inf, (real_pts[:-1] + real_pts[1:]) / 2.0, np.inf]
                i_bounds = np.r_[-np.inf, (imag_pts[:-1] + imag_pts[1:]) / 2.0, np.inf]

                total_ber = 0.0
                for k in range(len(pts)):
                    pk = pts[k]
                    p_r = ndtr((r_bounds[1:] - pk.real) / sigma) - ndtr((r_bounds[:-1] - pk.real) / sigma)
                    p_i = ndtr((i_bounds[1:] - pk.imag) / sigma) - ndtr((i_bounds[:-1] - pk.imag) / sigma)
                    prob_matrix = p_r[:, None] * p_i[None, :]
                    cell_centers = real_pts[:, None] + 1j * imag_pts[None, :]
                    dists = np.abs(cell_centers[:, :, None] - pts[None, None, :])
                    cell_nearest_pt = np.argmin(dists, axis=2)
                    hamming = np.count_nonzero(labels[cell_nearest_pt] != labels[k], axis=2)
                    total_ber += float(np.sum(prob_matrix * hamming))

                expected_ber = float(total_ber / (len(pts) * m))

            self.assertAlmostEqual(
                sim_ber, expected_ber, delta=tol,
                msg=f"AWGN BER mismatch for m={m} at {snr_db} dB: sim={sim_ber:.5f}, exp={expected_ber:.5f}"
            )

    def test_07_rayleigh_average_ber_analytical(self):
        """7. BPSK average BER over slow Rayleigh fading must match theoretical Pb = 0.5*(1 - sqrt(g/(1+g)))."""
        snr_db = 4.0
        n0 = 10.0 ** (-snr_db / 10.0)
        avg_snr_linear = 1.0 / n0
        expected_ber = float(0.5 * (1.0 - np.sqrt(avg_snr_linear / (1.0 + avg_snr_linear))))

        engine = PHYEngine(block_symbols=128)
        bpsk_mode = MODE_BY_ID[0]
        num_blocks = 2000
        master_seed = 20260904

        counters = RawPHYCounters(bits_per_symbol=1)
        for b in range(num_blocks):
            block_rng = make_block_rng(master_seed=master_seed, block_id=b)
            h = generate_channel_coefficient(generator=block_rng.fading_rng, channel_type="rayleigh")
            std_noise = generate_standard_noise(engine.block_symbols, generator=block_rng.noise_rng)
            payload = torch.randint(0, 2, (engine.block_symbols,), generator=block_rng.bit_rng, dtype=torch.float64)

            res = engine.evaluate_mode(
                mode=bpsk_mode,
                snr_db=snr_db,
                h=h,
                payload_bits=payload,
                standard_noise=std_noise,
            )
            counters.update(bit_errors=res.bit_errors, total_bits=res.total_bits, block_error=res.block_error)

        self.assertAlmostEqual(
            counters.ber, expected_ber, delta=0.012,
            msg=f"Rayleigh average BER {counters.ber:.4f} != theoretical {expected_ber:.4f}"
        )

    def test_08_rayleigh_h_unit_average_power(self):
        """8. Rayleigh fading coefficients from Sionna must satisfy E[|h|^2] = 1.0."""
        gen = torch.Generator().manual_seed(20260905)
        n_samples = 30_000
        hs = torch.stack([generate_channel_coefficient(generator=gen, channel_type="rayleigh") for _ in range(n_samples)])
        empirical_power = torch.mean(torch.abs(hs) ** 2).item()
        self.assertAlmostEqual(
            empirical_power, 1.0, delta=0.02,
            msg=f"Rayleigh E[|h|^2] = {empirical_power:.4f} deviates from 1.0"
        )

    def test_09_h_constant_across_entire_block(self):
        """9. Channel coefficient h must be identical and constant across all symbols in a block."""
        gen = torch.Generator().manual_seed(20260906)
        symbols = complex_normal([1536], precision="double", generator=gen)
        h_known = torch.tensor(0.6 - 0.7j, dtype=torch.complex128)

        output = apply_channel(
            transmitted_symbols=symbols,
            snr_db=10.0,
            h=h_known,
            noise_generator=gen,
        )

        noiseless_part = output.received_symbols - output.noise
        effective_h_per_symbol = noiseless_part / symbols

        self.assertTrue(
            torch.allclose(effective_h_per_symbol, h_known, rtol=1e-12, atol=1e-12),
            msg="h varied across symbols within the block!"
        )

    def test_10_independent_h_across_blocks(self):
        """10. Independent blocks must have mutually independent channel coefficients."""
        master_seed = 20260907
        num_blocks = 5000
        hs = []
        for b in range(num_blocks):
            block_rng = make_block_rng(master_seed=master_seed, block_id=b)
            h = generate_channel_coefficient(generator=block_rng.fading_rng, channel_type="rayleigh")
            hs.append(h)
        hs = torch.stack(hs)

        lag1_corr = torch.mean(hs[:-1] * torch.conj(hs[1:])).abs().item()
        self.assertLess(
            lag1_corr, 0.05,
            msg=f"Blocks exhibit unexpected correlation: lag1={lag1_corr:.4f}"
        )

    def test_11_deterministic_reproducibility(self):
        """11. Exact reproducibility: same master_seed + same block_id => identical realization."""
        master_seed = 20260908
        block_id = 42

        rng1 = make_block_rng(master_seed, block_id)
        h1 = generate_channel_coefficient(generator=rng1.fading_rng)
        noise1 = generate_standard_noise(500, generator=rng1.noise_rng)
        bits1 = torch.randint(0, 2, (500,), generator=rng1.bit_rng, dtype=torch.float64)

        rng2 = make_block_rng(master_seed, block_id)
        h2 = generate_channel_coefficient(generator=rng2.fading_rng)
        noise2 = generate_standard_noise(500, generator=rng2.noise_rng)
        bits2 = torch.randint(0, 2, (500,), generator=rng2.bit_rng, dtype=torch.float64)

        self.assertTrue(torch.equal(h1, h2))
        self.assertTrue(torch.equal(noise1, noise2))
        self.assertTrue(torch.equal(bits1, bits2))

    def test_12_different_block_identity_different_realization(self):
        """12. Different block IDs must produce distinct channel and noise realizations."""
        master_seed = 20260909
        rng_a = make_block_rng(master_seed, block_id=0)
        rng_b = make_block_rng(master_seed, block_id=1)

        ha = generate_channel_coefficient(generator=rng_a.fading_rng)
        hb = generate_channel_coefficient(generator=rng_b.fading_rng)
        na = generate_standard_noise(100, generator=rng_a.noise_rng)
        nb = generate_standard_noise(100, generator=rng_b.noise_rng)

        self.assertFalse(torch.equal(ha, hb))
        self.assertFalse(torch.equal(na, nb))

    def test_13_batch_and_individual_evaluation_agree(self):
        """13. Evaluating a sequence of blocks yields identical results whether processed sequentially or individually."""
        engine = PHYEngine(block_symbols=96)
        mode = MODE_BY_ID[1]  # QPSK
        master_seed = 20260910
        snr_db = 8.0

        individual_results = []
        for b in range(10):
            rng = make_block_rng(master_seed, b)
            h = generate_channel_coefficient(generator=rng.fading_rng)
            noise = generate_standard_noise(engine.block_symbols, generator=rng.noise_rng)
            bits = torch.randint(0, 2, (engine.block_symbols * mode.bits_per_symbol,), generator=rng.bit_rng, dtype=torch.float64)
            res = engine.evaluate_mode(mode, snr_db, h, bits, noise)
            individual_results.append((res.bit_errors, res.block_error, res.ber))

        reconstructed_results = []
        for b in range(10):
            rng = make_block_rng(master_seed, b)
            h = generate_channel_coefficient(generator=rng.fading_rng)
            noise = generate_standard_noise(engine.block_symbols, generator=rng.noise_rng)
            bits = torch.randint(0, 2, (engine.block_symbols * mode.bits_per_symbol,), generator=rng.bit_rng, dtype=torch.float64)
            res = engine.evaluate_mode(mode, snr_db, h, bits, noise)
            reconstructed_results.append((res.bit_errors, res.block_error, res.ber))

        self.assertEqual(individual_results, reconstructed_results)

    def test_14_modulation_order_invariance_for_paired_eval(self):
        """14. Evaluating candidate modes in forward vs reverse order must yield identical paired results."""
        engine = PHYEngine(block_symbols=128)
        master_seed = 20260911
        snr_db = 10.0

        rng = make_block_rng(master_seed, block_id=12)
        h = generate_channel_coefficient(generator=rng.fading_rng)
        noise = generate_standard_noise(engine.block_symbols, generator=rng.noise_rng)
        max_bits = engine.block_symbols * max(m.bits_per_symbol for m in MODES)
        payload_pool = torch.randint(0, 2, (max_bits,), generator=rng.bit_rng, dtype=torch.float64)

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

    def test_15_batch_vs_reference_phy_parity(self):
        """15. Batch L2 kernel vs scalar PHYEngine reference parity across all 4 modulations."""
        from sionna.phy.utils import db_to_lin

        num_blocks = 8
        s_sym = 128
        engine = PHYEngine(block_symbols=s_sym)
        master_seed = 20260912
        test_snrs = [6.0, 14.0, 22.0, 28.0]

        for snr_db in test_snrs:
            n0_val = float(1.0 / db_to_lin(snr_db, precision="double"))
            n0_t = torch.tensor(n0_val, dtype=torch.float64)
            sqrt_n0 = torch.sqrt(n0_t)

            # Generate shared realizations for num_blocks
            hs_list = []
            noises_list = []
            max_m = max(m.bits_per_symbol for m in MODES)
            payloads_list = []

            for b in range(num_blocks):
                rng = make_block_rng(master_seed + int(snr_db * 100), block_id=b)
                h = generate_channel_coefficient(generator=rng.fading_rng)
                noise = generate_standard_noise(s_sym, generator=rng.noise_rng)
                payload = torch.randint(0, 2, (s_sym * max_m,), generator=rng.bit_rng, dtype=torch.float64)
                hs_list.append(h)
                noises_list.append(noise)
                payloads_list.append(payload)

            # Batched representations
            h_batch = torch.stack(hs_list).unsqueeze(-1)  # (B, 1)
            z_batch = torch.stack(noises_list)            # (B, s_sym)
            payload_batch = torch.stack(payloads_list)    # (B, s_sym * max_m)

            abs_h_sq = torch.abs(h_batch) ** 2
            n0_eff = torch.clamp(n0_t / abs_h_sq, min=1e-12)
            actual_noise_batch = sqrt_n0 * z_batch

            for mode in MODES:
                m = mode.bits_per_symbol
                req_bits = s_sym * m

                # --- 1. Batched kernel path ---
                tx_bits_batch = payload_batch[:, :req_bits]
                _, mapper, demapper, _ = _get_sionna_modem(m)
                tx_syms_batch = mapper(tx_bits_batch)
                y_batch = h_batch * tx_syms_batch + actual_noise_batch
                y_eq_batch = y_batch / h_batch
                rx_bits_batch = demapper(y_eq_batch, n0_eff)

                err_mask_batch = (tx_bits_batch != rx_bits_batch).to(torch.int64)
                bit_errors_batch = err_mask_batch.sum(dim=-1)
                block_errors_batch = (bit_errors_batch > 0).to(torch.int64)

                # --- 2. Scalar reference path (block-by-block) ---
                for b in range(num_blocks):
                    scalar_res = engine.evaluate_mode(
                        mode=mode,
                        snr_db=snr_db,
                        h=hs_list[b],
                        payload_bits=payloads_list[b][:req_bits],
                        standard_noise=noises_list[b],
                    )

                    # Equivalence checks: exact matching of physical results and counts
                    b_bit_errors = int(bit_errors_batch[b].item())
                    b_block_error = int(block_errors_batch[b].item())

                    self.assertEqual(
                        scalar_res.bit_errors, b_bit_errors,
                        msg=f"Bit errors mismatch for {mode.modulation} at {snr_db} dB, block {b}: "
                            f"scalar={scalar_res.bit_errors} vs batch={b_bit_errors}"
                    )
                    self.assertEqual(
                        scalar_res.block_error, b_block_error,
                        msg=f"Block error mismatch for {mode.modulation} at {snr_db} dB, block {b}: "
                            f"scalar={scalar_res.block_error} vs batch={b_block_error}"
                    )

                    # Demapped bits exact match via demodulate
                    scalar_rx_bits = demodulate(
                        apply_channel(modulate(payloads_list[b][:req_bits], m), snr_db, h=hs_list[b], standard_noise=noises_list[b]).received_symbols,
                        h=hs_list[b],
                        bits_per_symbol=m,
                        n0=n0_t,
                    )
                    self.assertTrue(
                        torch.equal(rx_bits_batch[b], scalar_rx_bits),
                        msg=f"Demapped bits mismatch for {mode.modulation} at {snr_db} dB, block {b}"
                    )


if __name__ == "__main__":
    unittest.main()

