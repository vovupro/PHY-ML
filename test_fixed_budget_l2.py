"""Comprehensive acceptance tests for L2 Fixed-Budget Monte Carlo methodology.

Verifies:
1. 'light' profile resolves to exactly 10,000 blocks/seed and results/r0_mc_light_10k/.
2. 'deep' profile resolves to exactly 70,000 blocks/seed and results/r0_mc_deep_70k/.
3. Explicit arbitrary budget overrides (e.g. 100k, 200k, arbitrary N) resolve deterministically.
4. 'light' execution runs to exactly 10k/seed.
5. 'deep' configuration executes to exactly 70k/seed.
6. Arbitrary budget execution runs to exactly N blocks/seed.
7. NO early stopping exists: simulation never stops early regardless of convergence criteria.
8. Checkpoint trajectory statistics are purely observational and do not control execution length.
9. Real non-mocked CPU end-to-end execution.
10. Reports clearly state: FIXED-BUDGET MONTE CARLO, NO ADAPTIVE STOPPING, requested/actual blocks.
11. Canonical PHY artifacts in results/l2_cuda_rtx3060_final/ are untouched.
"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from config import (
    ExecutionConfig,
    FIXED_MC_PROFILES,
    resolve_results_dir,
    probe_environment,
)
from phy_engine import MODES
from calibration_l2_cuda import (
    run_fresh_l2_cuda_calibration,
    generate_markdown_report,
)


class TestFixedBudgetMonteCarlo(unittest.TestCase):
    """Test suite verifying fixed-budget Monte Carlo calibration requirements."""

    def setUp(self):
        self.canonical_final_dir = Path("results/l2_cuda_rtx3060_final")
        if self.canonical_final_dir.exists():
            self.canonical_mtimes = {
                p: p.stat().st_mtime for p in self.canonical_final_dir.glob("*") if p.is_file()
            }
        else:
            self.canonical_mtimes = {}

    def tearDown(self):
        # Verify canonical PHY artifacts were never modified by any test
        if self.canonical_mtimes:
            current_mtimes = {
                p: p.stat().st_mtime for p in self.canonical_final_dir.glob("*") if p.is_file()
            }
            self.assertEqual(
                self.canonical_mtimes,
                current_mtimes,
                "Canonical artifacts in results/l2_cuda_rtx3060_final/ were altered by tests!"
            )

    # -------------------------------------------------------------------------
    # 1. Profile & Override Configuration Resolution
    # -------------------------------------------------------------------------

    def test_01_light_profile_configuration(self):
        """1. 'light' profile resolves to exactly 10k/seed and results/r0_mc_light_10k."""
        self.assertEqual(FIXED_MC_PROFILES["light"], 10_000)
        self.assertEqual(resolve_results_dir(profile="light"), "results/r0_mc_light_10k")

        cfg = ExecutionConfig(profile="light")
        self.assertEqual(cfg.blocks_per_seed, 10_000)
        self.assertEqual(cfg.results_dir, "results/r0_mc_light_10k")

    def test_02_deep_profile_configuration(self):
        """2. 'deep' profile resolves to exactly 70k/seed and results/r0_mc_deep_70k."""
        self.assertEqual(FIXED_MC_PROFILES["deep"], 70_000)
        self.assertEqual(resolve_results_dir(profile="deep"), "results/r0_mc_deep_70k")

        cfg_default = ExecutionConfig()
        self.assertEqual(cfg_default.blocks_per_seed, 70_000)
        self.assertEqual(cfg_default.results_dir, "results/r0_mc_deep_70k")

        cfg_deep = ExecutionConfig(profile="deep")
        self.assertEqual(cfg_deep.blocks_per_seed, 70_000)
        self.assertEqual(cfg_deep.results_dir, "results/r0_mc_deep_70k")

    def test_03_arbitrary_budget_override(self):
        """3. Arbitrary fixed budget override (e.g. 100k, 200k, arbitrary N) resolves deterministically."""
        self.assertEqual(resolve_results_dir(blocks_per_seed=100_000), "results/r0_mc_custom_100k")
        self.assertEqual(resolve_results_dir(blocks_per_seed=200_000), "results/r0_mc_custom_200k")
        self.assertEqual(resolve_results_dir(blocks_per_seed=15_000), "results/r0_mc_custom_15k")
        self.assertEqual(resolve_results_dir(blocks_per_seed=12_345), "results/r0_mc_custom_12345")

        cfg_100k = ExecutionConfig(blocks_per_seed=100_000, profile=None)
        self.assertEqual(cfg_100k.blocks_per_seed, 100_000)
        self.assertEqual(cfg_100k.results_dir, "results/r0_mc_custom_100k")

        cfg_200k = ExecutionConfig(blocks_per_seed=200_000, profile=None)
        self.assertEqual(cfg_200k.blocks_per_seed, 200_000)
        self.assertEqual(cfg_200k.results_dir, "results/r0_mc_custom_200k")

    # -------------------------------------------------------------------------
    # 2. Execution Exactness (No Early Stopping, Checkpoint Non-Interference)
    # -------------------------------------------------------------------------

    def _make_mock_chunk_metrics(self, batch_size: int):
        """Generate dummy chunk metrics for fast simulation testing."""
        return {
            m.mode_id: (
                int(batch_size * 1536 * m.bits_per_symbol * 0.01),  # errors
                batch_size * 1536 * m.bits_per_symbol,              # total bits
                int(batch_size * 0.05),                             # block errors
                0.01,                                               # mean BER
                0.0001,                                             # m2 BER
            )
            for m in MODES
        }

    def test_04_light_executes_exactly_10k_seed(self):
        """4. 'light' profile executes to exactly 10,000 blocks/seed with 2 checkpoints."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = ExecutionConfig(
                backend="cpu",
                profile="light",
                batch_blocks=500,
                checkpoint_step=5000,
                results_dir=tmp_dir,
            )

            with patch("calibration_l2_cuda.BatchPHYEngine") as MockEngine:
                mock_inst = MockEngine.return_value
                mock_inst.evaluate_chunk.side_effect = lambda snr, cur_b, f, n, b: self._make_mock_chunk_metrics(cur_b)

                res = run_fresh_l2_cuda_calibration(config=cfg, snrs=(18.0,))

            snr_res = res["snr_results"][18.0]
            self.assertEqual(snr_res["final_blocks_a"], 10_000)
            self.assertEqual(snr_res["final_blocks_b"], 10_000)
            self.assertEqual(snr_res["total_blocks_pooled"], 20_000)
            self.assertEqual(snr_res["checkpoints_count"], 2)
            self.assertEqual(len(snr_res["checkpoint_history"]), 2)
            self.assertEqual(snr_res["checkpoint_history"][0].blocks_per_seed, 5000)
            self.assertEqual(snr_res["checkpoint_history"][1].blocks_per_seed, 10000)

    def test_05_deep_executes_exactly_70k_seed(self):
        """5. 'deep' profile executes to exactly 70,000 blocks/seed with 14 checkpoints."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = ExecutionConfig(
                backend="cpu",
                profile="deep",
                batch_blocks=500,
                checkpoint_step=5000,
                results_dir=tmp_dir,
            )

            with patch("calibration_l2_cuda.BatchPHYEngine") as MockEngine:
                mock_inst = MockEngine.return_value
                mock_inst.evaluate_chunk.side_effect = lambda snr, cur_b, f, n, b: self._make_mock_chunk_metrics(cur_b)

                res = run_fresh_l2_cuda_calibration(config=cfg, snrs=(18.0,))

            snr_res = res["snr_results"][18.0]
            self.assertEqual(snr_res["final_blocks_a"], 70_000)
            self.assertEqual(snr_res["final_blocks_b"], 70_000)
            self.assertEqual(snr_res["total_blocks_pooled"], 140_000)
            self.assertEqual(snr_res["checkpoints_count"], 14)
            self.assertEqual(len(snr_res["checkpoint_history"]), 14)
            self.assertEqual(snr_res["checkpoint_history"][-1].blocks_per_seed, 70_000)

    def test_06_arbitrary_override_executes_exactly_requested_budget(self):
        """6. Arbitrary budget override executes to exactly the requested blocks/seed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_budget = 25_000
            cfg = ExecutionConfig(
                backend="cpu",
                blocks_per_seed=custom_budget,
                profile=None,
                batch_blocks=500,
                checkpoint_step=5000,
                results_dir=tmp_dir,
            )

            with patch("calibration_l2_cuda.BatchPHYEngine") as MockEngine:
                mock_inst = MockEngine.return_value
                mock_inst.evaluate_chunk.side_effect = lambda snr, cur_b, f, n, b: self._make_mock_chunk_metrics(cur_b)

                res = run_fresh_l2_cuda_calibration(config=cfg, snrs=(18.0,))

            snr_res = res["snr_results"][18.0]
            self.assertEqual(snr_res["final_blocks_a"], custom_budget)
            self.assertEqual(snr_res["final_blocks_b"], custom_budget)
            self.assertEqual(snr_res["total_blocks_pooled"], custom_budget * 2)
            self.assertEqual(snr_res["checkpoints_count"], 5)

    def test_07_no_early_stopping_when_criteria_pass(self):
        """7. No early stopping exists: simulation does NOT exit early when stats are stable."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Request 20,000 blocks/seed. Even if checkpoint 1 & 2 have identical zero-error or stable stats,
            # it must continue all the way to 20,000 blocks/seed.
            cfg = ExecutionConfig(
                backend="cpu",
                blocks_per_seed=20_000,
                profile=None,
                batch_blocks=500,
                checkpoint_step=5000,
                results_dir=tmp_dir,
            )

            with patch("calibration_l2_cuda.BatchPHYEngine") as MockEngine:
                mock_inst = MockEngine.return_value
                # Completely identical stationary metrics across all chunks
                mock_inst.evaluate_chunk.side_effect = lambda snr, cur_b, f, n, b: self._make_mock_chunk_metrics(cur_b)

                res = run_fresh_l2_cuda_calibration(config=cfg, snrs=(18.0,))

            snr_res = res["snr_results"][18.0]
            # Old adaptive stopping would have stopped at checkpoint 2 (10k).
            # Fixed-budget MUST continue to 20k!
            self.assertEqual(snr_res["final_blocks_a"], 20_000)
            self.assertEqual(snr_res["final_blocks_b"], 20_000)
            self.assertEqual(snr_res["checkpoints_count"], 4)

    def test_08_checkpoint_statistics_do_not_control_execution_length(self):
        """8. Checkpoint statistics are purely observational and have zero impact on execution length."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = ExecutionConfig(
                backend="cpu",
                blocks_per_seed=15_000,
                profile=None,
                batch_blocks=500,
                checkpoint_step=5000,
                results_dir=tmp_dir,
            )

            # Execution 1: zero errors
            with patch("calibration_l2_cuda.BatchPHYEngine") as MockEngine:
                mock_inst = MockEngine.return_value
                mock_inst.evaluate_chunk.side_effect = lambda snr, cur_b, f, n, b: {
                    m.mode_id: (0, cur_b * 1536 * m.bits_per_symbol, 0, 0.0, 0.0) for m in MODES
                }
                res1 = run_fresh_l2_cuda_calibration(config=cfg, snrs=(18.0,))

            # Execution 2: high error rate
            with patch("calibration_l2_cuda.BatchPHYEngine") as MockEngine:
                mock_inst = MockEngine.return_value
                mock_inst.evaluate_chunk.side_effect = lambda snr, cur_b, f, n, b: {
                    m.mode_id: (int(cur_b * 1536 * m.bits_per_symbol * 0.5), cur_b * 1536 * m.bits_per_symbol, cur_b, 0.5, 0.01) for m in MODES
                }
                res2 = run_fresh_l2_cuda_calibration(config=cfg, snrs=(18.0,))

            # Both must execute exactly 15,000 blocks/seed despite wildly different stats
            self.assertEqual(res1["snr_results"][18.0]["final_blocks_a"], 15_000)
            self.assertEqual(res2["snr_results"][18.0]["final_blocks_a"], 15_000)
            self.assertEqual(res1["snr_results"][18.0]["checkpoints_count"], 3)
            self.assertEqual(res2["snr_results"][18.0]["checkpoints_count"], 3)

    # -------------------------------------------------------------------------
    # 3. Real Non-Mocked Lightweight CPU End-to-End Execution
    # -------------------------------------------------------------------------

    def test_09_real_cpu_lightweight_execution(self):
        """9. Real non-mocked BatchPHYEngine end-to-end execution on CPU."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = ExecutionConfig(
                backend="cpu",
                precision="double",
                batch_blocks=250,
                blocks_per_seed=500,
                checkpoint_step=500,
                profile=None,
                results_dir=tmp_dir,
            )

            res = run_fresh_l2_cuda_calibration(config=cfg, snrs=(14.0,))
            self.assertIn(14.0, res["snr_results"])
            snr_res = res["snr_results"][14.0]
            self.assertEqual(snr_res["final_blocks_a"], 500)
            self.assertEqual(snr_res["final_blocks_b"], 500)
            self.assertEqual(snr_res["total_blocks_pooled"], 1000)
            self.assertEqual(snr_res["checkpoints_count"], 1)

            # Check CSV artifacts
            csv_a = Path(tmp_dir) / "calibration_1d_cuda_seed_a.csv"
            csv_b = Path(tmp_dir) / "calibration_1d_cuda_seed_b.csv"
            csv_pooled = Path(tmp_dir) / "calibration_1d_cuda_pooled.csv"
            report_md = Path(tmp_dir) / "calibration_1d_cuda_report.md"

            self.assertTrue(csv_a.exists())
            self.assertTrue(csv_b.exists())
            self.assertTrue(csv_pooled.exists())
            self.assertTrue(report_md.exists())

    # -------------------------------------------------------------------------
    # 4. Report Statements Verification (Requirement 9)
    # -------------------------------------------------------------------------

    def test_10_report_statements_verification(self):
        """10. Reports clearly state FIXED-BUDGET MONTE CARLO, NO ADAPTIVE STOPPING, and requested/actual blocks."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "test_report.md"
            cfg = ExecutionConfig(
                backend="cpu",
                profile="deep",
                blocks_per_seed=70_000,
                results_dir=tmp_dir,
            )
            env = probe_environment(cfg)

            mock_snr_results = {
                18.0: {
                    "final_pooled_ber": {0: 0.004, 1: 0.008, 2: 0.03, 3: 0.08},
                    "final_pooled_se": {0: 0.0003, 1: 0.0005, 2: 0.001, 3: 0.002},
                    "requested_blocks_per_seed": 70_000,
                    "actual_blocks_per_seed": 70_000,
                    "final_blocks_a": 70_000,
                    "final_blocks_b": 70_000,
                    "total_blocks_pooled": 140_000,
                    "checkpoints_count": 14,
                    "stable": True,
                    "final_stats_a": {m: type("Stats", (), {"ber": 0.01})() for m in range(4)},
                    "final_stats_b": {m: type("Stats", (), {"ber": 0.01})() for m in range(4)},
                    "final_z_ab": {m: 0.5 for m in range(4)},
                }
            }

            generate_markdown_report(
                snr_results=mock_snr_results,
                env=env,
                config=cfg,
                warmup_time=0.1,
                total_mc_time=10.0,
                total_blocks_simulated=140_000,
                output_path=report_path,
                persisted_benchmark=None,
            )

            report_text = report_path.read_text(encoding="utf-8")

            # Must contain required fixed-budget declarations
            self.assertIn("FIXED-BUDGET MONTE CARLO", report_text)
            self.assertIn("NO ADAPTIVE STOPPING", report_text)
            self.assertIn("Requested Blocks per Seed", report_text)
            self.assertIn("Actual Blocks per Seed", report_text)
            self.assertIn("70,000", report_text)
            self.assertIn("140,000", report_text)
            self.assertIn("Target Overlap", report_text)
            self.assertIn("95% Confidence Interval", report_text)
            self.assertNotIn("Predefined 3-criterion rule", report_text)


if __name__ == "__main__":
    unittest.main()
