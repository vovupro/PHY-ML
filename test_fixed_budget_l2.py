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
import csv
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
        self.canonical_dirs = [
            Path("results/l2_cuda_rtx3060"),
            Path("results/l2_cuda_rtx3060_final"),
        ]
        self.canonical_snapshots = {}
        for d in self.canonical_dirs:
            if d.exists():
                self.canonical_snapshots[d] = {
                    p: (p.stat().st_mtime, p.stat().st_size) for p in d.glob("*") if p.is_file()
                }
            else:
                self.canonical_snapshots[d] = {}

    def tearDown(self):
        # Verify canonical historical PHY artifacts were never modified by any test
        for d, snap in self.canonical_snapshots.items():
            if snap:
                current_snap = {
                    p: (p.stat().st_mtime, p.stat().st_size) for p in d.glob("*") if p.is_file()
                }
                self.assertEqual(
                    snap,
                    current_snap,
                    f"Historical canonical artifacts in {d} were altered by tests!"
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
            csv_traj = Path(tmp_dir) / "checkpoint_trajectory.csv"
            report_md = Path(tmp_dir) / "calibration_1d_cuda_report.md"

            self.assertTrue(csv_a.exists())
            self.assertTrue(csv_b.exists())
            self.assertTrue(csv_pooled.exists())
            self.assertTrue(csv_traj.exists())
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
                    "fixed_budget_reached": True,
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

    # -------------------------------------------------------------------------
    # 5. Patch Audit Verifications (Trajectory CSV, Semantics, Sanity Summary)
    # -------------------------------------------------------------------------

    def test_11_checkpoint_trajectory_csv_and_all_four_modes(self):
        """11. checkpoint_trajectory.csv exists with 14 columns and all 4 modes at every checkpoint."""
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
                run_fresh_l2_cuda_calibration(config=cfg, snrs=(18.0,))

            traj_path = Path(tmp_dir) / "checkpoint_trajectory.csv"
            self.assertTrue(traj_path.exists(), "checkpoint_trajectory.csv does not exist!")

            expected_columns = [
                "profile",
                "requested_blocks_per_seed",
                "snr_db",
                "checkpoint_idx",
                "blocks_per_seed",
                "pooled_blocks",
                "mode_id",
                "modulation",
                "pooled_ber",
                "pooled_se",
                "ci95_low",
                "ci95_high",
                "overlaps_ber_target",
                "z_ab",
            ]

            with open(traj_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, expected_columns)
                rows = list(reader)

            # Light profile: 10,000 blocks / 5,000 step = 2 checkpoints
            # 2 checkpoints * 4 modulations = 8 rows
            self.assertEqual(len(rows), 8)

            # Group rows by (snr_db, checkpoint_idx)
            grouped = {}
            for r in rows:
                key = (float(r["snr_db"]), int(r["checkpoint_idx"]))
                grouped.setdefault(key, []).append(r)

            for key, group in grouped.items():
                self.assertEqual(len(group), 4, f"Checkpoint {key} must have exactly 4 modulation rows!")
                modes_present = {r["modulation"] for r in group}
                mode_ids_present = {int(r["mode_id"]) for r in group}
                self.assertEqual(modes_present, {"BPSK", "QPSK", "16QAM", "64QAM"})
                self.assertEqual(mode_ids_present, {0, 1, 2, 3})

    def test_12_deep_profile_fourteen_checkpoints_and_trajectory(self):
        """12. Deep 70k profile with 5k step implies exactly 14 checkpoints and 56 rows per SNR."""
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
            self.assertEqual(snr_res["checkpoints_count"], 14)
            self.assertEqual(len(snr_res["checkpoint_history"]), 14)

            expected_depths = [i * 5000 for i in range(1, 15)]
            actual_depths = [chk.blocks_per_seed for chk in snr_res["checkpoint_history"]]
            self.assertEqual(actual_depths, expected_depths)

            traj_path = Path(tmp_dir) / "checkpoint_trajectory.csv"
            with open(traj_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # 14 checkpoints * 4 modulations = 56 rows
            self.assertEqual(len(rows), 56)
            chk_indices = {int(r["checkpoint_idx"]) for r in rows}
            self.assertEqual(chk_indices, set(range(1, 15)))

    def test_13_no_stale_stability_semantics(self):
        """13. Ensure no stale is_stable=True / stable=True semantics remain in results or CSV."""
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
            # Must report explicit fixed_budget_reached=True
            self.assertTrue(snr_res.get("fixed_budget_reached"))
            # Must NOT report stable=True
            self.assertNotEqual(snr_res.get("stable"), True)
            self.assertIsNone(snr_res.get("stable"))

            # Check pooled CSV
            csv_pooled = Path(tmp_dir) / "calibration_1d_cuda_pooled.csv"
            with open(csv_pooled, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                pooled_rows = list(reader)

            self.assertGreater(len(pooled_rows), 0)
            for row in pooled_rows:
                self.assertEqual(row.get("fixed_budget_reached"), "True")
                self.assertNotEqual(row.get("is_stable"), "True")
                self.assertNotEqual(row.get("is_stable"), True)
                self.assertIn(row.get("is_stable"), ("N/A (deprecated)", None, ""))

    def test_14_historical_canonical_directories_untouched(self):
        """14. Execution must never touch historical canonical directories."""
        dir_1 = Path("results/l2_cuda_rtx3060")
        dir_2 = Path("results/l2_cuda_rtx3060_final")

        snap_before = {}
        for d in [dir_1, dir_2]:
            if d.exists():
                snap_before[d] = {
                    p.name: (p.stat().st_mtime, p.stat().st_size)
                    for p in d.glob("*") if p.is_file()
                }

        # Run calibration in an isolated temporary directory
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
                run_fresh_l2_cuda_calibration(config=cfg, snrs=(14.0,))

        # Verify neither directory changed
        for d, before in snap_before.items():
            after = {
                p.name: (p.stat().st_mtime, p.stat().st_size)
                for p in d.glob("*") if p.is_file()
            }
            self.assertEqual(before, after, f"Directory {d} was modified!")

    def test_15_analytical_summary_review_on_suspicious_point(self):
        """15. Analytical summary dynamically prints PASS or REVIEW based on BPSK z-scores."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = ExecutionConfig(
                backend="cpu",
                profile="light",
                blocks_per_seed=10_000,
                results_dir=tmp_dir,
            )
            env = probe_environment(cfg)

            # Scenario A: Consistent point (z <= 3.0) -> PASS
            pass_report_path = Path(tmp_dir) / "report_pass.md"
            snr_results_pass = {
                18.0: {
                    # Theoretical BPSK at 18 dB is ~0.00392
                    "final_pooled_ber": {0: 0.00392, 1: 0.008, 2: 0.03, 3: 0.08},
                    "final_pooled_se": {0: 0.0003, 1: 0.0005, 2: 0.001, 3: 0.002},
                    "requested_blocks_per_seed": 10_000,
                    "actual_blocks_per_seed": 10_000,
                    "final_blocks_a": 10_000,
                    "final_blocks_b": 10_000,
                    "total_blocks_pooled": 20_000,
                    "checkpoints_count": 2,
                    "fixed_budget_reached": True,
                    "final_stats_a": {m: type("Stats", (), {"ber": 0.01})() for m in range(4)},
                    "final_stats_b": {m: type("Stats", (), {"ber": 0.01})() for m in range(4)},
                    "final_z_ab": {m: 0.5 for m in range(4)},
                }
            }
            generate_markdown_report(
                snr_results=snr_results_pass,
                env=env,
                config=cfg,
                warmup_time=0.1,
                total_mc_time=1.0,
                total_blocks_simulated=20_000,
                output_path=pass_report_path,
            )
            pass_text = pass_report_path.read_text(encoding="utf-8")
            self.assertIn("Analytical Rayleigh sanity test: **PASS** across the grid.", pass_text)
            self.assertNotIn("**REVIEW**", pass_text)

            # Scenario B: Suspicious point (z > 3.0) -> REVIEW with details
            review_report_path = Path(tmp_dir) / "report_review.md"
            snr_results_review = {
                18.0: {
                    # Empirical BER is 0.020 (far from ~0.00392, diff=0.016, SE=0.001 -> z=16.0)
                    "final_pooled_ber": {0: 0.020, 1: 0.008, 2: 0.03, 3: 0.08},
                    "final_pooled_se": {0: 0.001, 1: 0.0005, 2: 0.001, 3: 0.002},
                    "requested_blocks_per_seed": 10_000,
                    "actual_blocks_per_seed": 10_000,
                    "final_blocks_a": 10_000,
                    "final_blocks_b": 10_000,
                    "total_blocks_pooled": 20_000,
                    "checkpoints_count": 2,
                    "fixed_budget_reached": True,
                    "final_stats_a": {m: type("Stats", (), {"ber": 0.01})() for m in range(4)},
                    "final_stats_b": {m: type("Stats", (), {"ber": 0.01})() for m in range(4)},
                    "final_z_ab": {m: 0.5 for m in range(4)},
                }
            }
            generate_markdown_report(
                snr_results=snr_results_review,
                env=env,
                config=cfg,
                warmup_time=0.1,
                total_mc_time=1.0,
                total_blocks_simulated=20_000,
                output_path=review_report_path,
            )
            review_text = review_report_path.read_text(encoding="utf-8")
            self.assertNotIn("Analytical Rayleigh sanity test: **PASS** across the grid.", review_text)
            self.assertIn("Analytical Rayleigh sanity test: **REVIEW**", review_text)
            self.assertIn("1 suspicious point(s) with z > 3.0: 18.0 dB", review_text)


if __name__ == "__main__":
    unittest.main()
