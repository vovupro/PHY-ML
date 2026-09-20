"""Unit tests for R0 SNR-Grid Convergence Study & Refinement Runner.

Verifies:
1. Refinement runner specification:
   - Evaluates ONLY 16.75, 22.75, 28.25 dB.
   - Targets Deep budget (70,000 blocks/seed = 140,000 pooled blocks/point).
   - Correctly passes execution parameters to the calibration core.
2. Dataset merging and three resolution views:
   - 1.0-dB local transition view contains integer-spaced points around transitions.
   - 0.5-dB local transition view contains canonical Deep 70k points.
   - 0.25-dB local transition refinement view contains all points plus the 3 refinement midpoints.
3. Transition bracket computation mechanics:
   - Extracts valid bracket endpoints and midpoints on synthetic fixtures.
4. Decision Tree model selection mechanics:
   - Verifies optimal depth search and threshold extraction without enforcing production outcomes.
5. Strict post-processing isolation:
   - Analysis tool NEVER imports Sionna, BatchPHYEngine, calibration_l2_cuda, or phy_engine.
"""
import csv
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple
import unittest
from unittest.mock import MagicMock, patch

from ground_truth import GroundTruthConfig, compute_ground_truth
from refine_snr_grid_cuda import (
    DEFAULT_REFINE_BUDGET,
    DEFAULT_REFINE_RESULTS_DIR,
    REFINEMENT_025_SNRS,
    run_refinement_calibration,
)
from analyze_snr_grid_convergence import (
    TRANSITION_SPECS,
    construct_resolution_views,
    find_transition_bracket,
    load_and_merge_datasets,
    run_snr_grid_convergence_study,
)

TEST_GRID_25_POINTS: Tuple[float, ...] = (
    0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0,
    16.0, 16.5, 17.0, 17.5, 18.0, 20.0,
    22.0, 22.5, 23.0, 23.5, 24.0, 26.0,
    28.0, 28.5, 29.0, 29.5, 30.0,
)

TEST_MODES: Tuple[Dict[str, Any], ...] = (
    {"mode_id": 0, "modulation": "BPSK", "bits_per_symbol": 1},
    {"mode_id": 1, "modulation": "QPSK", "bits_per_symbol": 2},
    {"mode_id": 2, "modulation": "16QAM", "bits_per_symbol": 4},
    {"mode_id": 3, "modulation": "64QAM", "bits_per_symbol": 6},
)


class TestSNRGridConvergence(unittest.TestCase):
    """Test suite for transition-centered SNR grid refinement and convergence study."""

    def _generate_synthetic_cal_csv(
        self,
        output_path: Path,
        snrs: Sequence[float],
        num_blocks: int = 140000,
    ) -> None:
        """Generate synthetic calibration table with realistic Rayleigh BER progression."""
        fieldnames = [
            "mode_id", "modulation", "bits_per_symbol", "snr_db", "num_blocks",
            "total_bits", "bit_errors", "ber", "block_errors", "bler",
            "mean_block_ber", "std_block_ber", "se_block_ber", "z_ab",
            "fixed_budget_reached", "is_stable", "reliability_uncertain",
        ]

        rows = []
        for snr in snrs:
            for m in TEST_MODES:
                mid = m["mode_id"]
                mod = m["modulation"]
                bps = m["bits_per_symbol"]

                # Calibrated synthetic decay parameters to produce standard transitions:
                # BPSK -> QPSK around 16.75 dB
                # QPSK -> 16QAM around 22.75 dB
                # 16QAM -> 64QAM around 28.25 dB
                if mid == 0:  # BPSK
                    ber = 0.5 * math.exp(-0.25 * (10 ** (snr / 10.0)))
                elif mid == 1:  # QPSK
                    ber = 0.5 * math.exp(-0.080 * (10 ** (snr / 10.0)))
                elif mid == 2:  # 16QAM
                    ber = 0.5 * math.exp(-0.020 * (10 ** (snr / 10.0)))
                else:  # 64QAM
                    ber = 0.5 * math.exp(-0.0057 * (10 ** (snr / 10.0)))

                ber = max(1e-6, min(0.5, ber))
                se = math.sqrt(ber * (1.0 - ber) / num_blocks)
                ci_hw = 1.96 * se
                uncertain = (ber - ci_hw <= 0.01 <= ber + ci_hw)
                tot_bits = num_blocks * 1536 * bps

                rows.append({
                    "mode_id": mid,
                    "modulation": mod,
                    "bits_per_symbol": bps,
                    "snr_db": snr,
                    "num_blocks": num_blocks,
                    "total_bits": tot_bits,
                    "bit_errors": int(tot_bits * ber),
                    "ber": ber,
                    "block_errors": int(num_blocks * min(1.0, ber * 10)),
                    "bler": min(1.0, ber * 10),
                    "mean_block_ber": ber,
                    "std_block_ber": se * math.sqrt(num_blocks),
                    "se_block_ber": se,
                    "z_ab": 0.15,
                    "fixed_budget_reached": True,
                    "is_stable": "N/A (deprecated)",
                    "reliability_uncertain": uncertain,
                })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # -------------------------------------------------------------------------
    # 1. Refinement Runner Specification & Parameter Tests
    # -------------------------------------------------------------------------

    def test_01_refinement_runner_constants(self):
        """Verify refinement runner constants evaluate exclusively the 3 midpoint SNRs."""
        self.assertEqual(REFINEMENT_025_SNRS, (16.75, 22.75, 28.25))
        self.assertEqual(DEFAULT_REFINE_RESULTS_DIR, "results/r0_snr_grid_refine_025")
        self.assertEqual(DEFAULT_REFINE_BUDGET, 70000)

    @patch("refine_snr_grid_cuda.run_fresh_l2_cuda_calibration")
    def test_02_refinement_runner_execution_call(self, mock_cal):
        """Verify refinement runner invokes fixed-budget calibration with exact Deep parameters."""
        mock_cal.return_value = {"status": "SUCCESS"}
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = run_refinement_calibration(
                results_dir=tmp_dir,
                blocks_per_seed=70000,
                backend="cpu",
                precision="double",
                batch_blocks=500,
            )
            mock_cal.assert_called_once()
            call_kwargs = mock_cal.call_args[1]
            cfg = call_kwargs["config"]
            snrs = call_kwargs["snrs"]

            self.assertEqual(snrs, (16.75, 22.75, 28.25))
            self.assertEqual(cfg.blocks_per_seed, 70000)
            self.assertEqual(cfg.profile, "deep")
            self.assertEqual(cfg.precision, "double")
            self.assertEqual(cfg.batch_blocks, 500)
            self.assertEqual(cfg.results_dir, tmp_dir)

    # -------------------------------------------------------------------------
    # 2. Dataset Merging & Resolution Views Tests
    # -------------------------------------------------------------------------

    def test_03_load_and_merge_datasets(self):
        """Verify merging Deep (25 SNRs) and Refinement (3 SNRs) produces 28 unified SNRs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            deep_csv = tmp_p / "deep" / "calibration_1d_cuda_pooled.csv"
            refine_csv = tmp_p / "refine" / "calibration_1d_cuda_pooled.csv"

            self._generate_synthetic_cal_csv(deep_csv, TEST_GRID_25_POINTS)
            self._generate_synthetic_cal_csv(refine_csv, REFINEMENT_025_SNRS)

            deep_cal, refine_cal, merged_cal = load_and_merge_datasets(deep_csv, refine_csv)

            self.assertEqual(len(deep_cal), 25)
            self.assertEqual(len(refine_cal), 3)
            self.assertEqual(len(merged_cal), 28)
            for snr in REFINEMENT_025_SNRS:
                self.assertIn(snr, merged_cal)

    def test_04_construct_resolution_views(self):
        """Verify construction of 1.0-dB, 0.5-dB, and 0.25-dB resolution views."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            deep_csv = tmp_p / "deep" / "calibration_1d_cuda_pooled.csv"
            refine_csv = tmp_p / "refine" / "calibration_1d_cuda_pooled.csv"

            self._generate_synthetic_cal_csv(deep_csv, TEST_GRID_25_POINTS)
            self._generate_synthetic_cal_csv(refine_csv, REFINEMENT_025_SNRS)

            deep_cal, refine_cal, merged_cal = load_and_merge_datasets(deep_csv, refine_csv)
            cal_1_0, cal_0_5, cal_0_25 = construct_resolution_views(deep_cal, merged_cal)

            # 1.0-dB view: integer points only
            for s in cal_1_0.keys():
                self.assertAlmostEqual(s, round(s), places=3, msg=f"Non-integer point in 1.0-dB view: {s}")
            self.assertEqual(len(cal_1_0), 19)

            # 0.5-dB view: canonical 25 points
            self.assertEqual(len(cal_0_5), 25)

            # 0.25-dB view: all 28 points
            self.assertEqual(len(cal_0_25), 28)

    # -------------------------------------------------------------------------
    # 3. Transition Bracket Halving & Shift Tests
    # -------------------------------------------------------------------------

    def test_05_transition_bracket_computation_mechanics(self):
        """Verify transition bracket extraction mechanics using a calibrated synthetic fixture."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            deep_csv = tmp_p / "deep" / "calibration_1d_cuda_pooled.csv"
            refine_csv = tmp_p / "refine" / "calibration_1d_cuda_pooled.csv"

            self._generate_synthetic_cal_csv(deep_csv, TEST_GRID_25_POINTS)
            self._generate_synthetic_cal_csv(refine_csv, REFINEMENT_025_SNRS)

            deep_cal, refine_cal, merged_cal = load_and_merge_datasets(deep_csv, refine_csv)
            cal_1_0, cal_0_5, cal_0_25 = construct_resolution_views(deep_cal, merged_cal)

            gt_cfg = GroundTruthConfig()
            gt_1_0 = compute_ground_truth(cal_1_0, gt_cfg)
            gt_0_5 = compute_ground_truth(cal_0_5, gt_cfg)
            gt_0_25 = compute_ground_truth(cal_0_25, gt_cfg)

            for tr_name, from_m, to_m, _ in TRANSITION_SPECS:
                b_1_0 = find_transition_bracket(gt_1_0, from_m, to_m)
                b_0_5 = find_transition_bracket(gt_0_5, from_m, to_m)
                b_0_25 = find_transition_bracket(gt_0_25, from_m, to_m)

                self.assertIsNotNone(b_1_0, f"Bracket not found for {tr_name} in 1.0-dB local view")
                self.assertIsNotNone(b_0_5, f"Bracket not found for {tr_name} in 0.5-dB local view")
                self.assertIsNotNone(b_0_25, f"Bracket not found for {tr_name} in 0.25-dB local view")

                w_1_0 = b_1_0["bracket_width"]
                w_0_5 = b_0_5["bracket_width"]
                w_025 = b_0_25["bracket_width"]

                # Mechanics verification: positive widths and correct interval ordering
                self.assertGreater(w_1_0, 0.0)
                self.assertGreater(w_0_5, 0.0)
                self.assertGreater(w_025, 0.0)
                self.assertLess(b_1_0["lower_endpoint"], b_1_0["upper_endpoint"])
                self.assertLess(b_0_5["lower_endpoint"], b_0_5["upper_endpoint"])
                self.assertLess(b_0_25["lower_endpoint"], b_0_25["upper_endpoint"])

                # Synthetic fixture bisection mechanics check
                self.assertLessEqual(w_0_5, w_1_0)
                self.assertLessEqual(w_025, w_0_5)

    # -------------------------------------------------------------------------
    # 4. End-to-End Convergence Study Pipeline & Artifact Tests
    # -------------------------------------------------------------------------

    def test_06_convergence_study_artifacts_generation(self):
        """Verify run_snr_grid_convergence_study produces all three artifacts with complete schema."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            deep_dir = tmp_p / "deep"
            refine_dir = tmp_p / "refine"
            out_dir = tmp_p / "study"

            self._generate_synthetic_cal_csv(deep_dir / "calibration_1d_cuda_pooled.csv", TEST_GRID_25_POINTS)
            self._generate_synthetic_cal_csv(refine_dir / "calibration_1d_cuda_pooled.csv", REFINEMENT_025_SNRS)

            res = run_snr_grid_convergence_study(
                deep_dir=deep_dir,
                refine_dir=refine_dir,
                output_dir=out_dir,
            )

            csv_conv = out_dir / "snr_grid_convergence.csv"
            csv_labels = out_dir / "snr_grid_views_labels.csv"
            md_report = out_dir / "snr_grid_convergence.md"

            self.assertTrue(csv_conv.exists(), "snr_grid_convergence.csv must exist")
            self.assertTrue(csv_labels.exists(), "snr_grid_views_labels.csv must exist")
            self.assertTrue(md_report.exists(), "snr_grid_convergence.md must exist")

            with open(csv_conv, "r", encoding="utf-8") as f:
                conv_rows = list(csv.DictReader(f))
            self.assertEqual(len(conv_rows), 3, "3 transitions expected")

            with open(csv_labels, "r", encoding="utf-8") as f:
                label_rows = list(csv.DictReader(f))
            self.assertEqual(len(label_rows), 28, "28 unique SNR points expected")

            report_text = md_report.read_text(encoding="utf-8")
            self.assertIn("1.0-dB local transition view", report_text)
            self.assertIn("0.5-dB local transition view", report_text)
            self.assertIn("0.25-dB local transition", report_text)
            self.assertIn("Decision Tree Model Selection", report_text)
            self.assertIn("Transition Switching-Boundary", report_text)
            self.assertIn("results/r0_snr_grid_convergence/", report_text)

    def test_07_cart_model_selection_mechanics(self):
        """Verify CART depth model selection mechanics (valid depth range, bounded fidelity)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            deep_dir = tmp_p / "deep"
            refine_dir = tmp_p / "refine"
            out_dir = tmp_p / "study"

            self._generate_synthetic_cal_csv(deep_dir / "calibration_1d_cuda_pooled.csv", TEST_GRID_25_POINTS)
            self._generate_synthetic_cal_csv(refine_dir / "calibration_1d_cuda_pooled.csv", REFINEMENT_025_SNRS)

            res = run_snr_grid_convergence_study(
                deep_dir=deep_dir,
                refine_dir=refine_dir,
                output_dir=out_dir,
            )

            dt = res["dt_metrics"]
            for v_key in ["view_1_0", "view_0_5", "view_0_25"]:
                dm = dt[v_key]
                self.assertIn(dm["selected_depth"], range(1, 6))
                self.assertGreaterEqual(dm["actual_depth"], 1)
                self.assertLessEqual(dm["actual_depth"], dm["selected_depth"])
                self.assertGreaterEqual(dm["fidelity"], 0.0)
                self.assertLessEqual(dm["fidelity"], 1.0)
                self.assertIsInstance(dm["thresholds"], list)
                self.assertGreaterEqual(len(dm["thresholds"]), 1)

    # -------------------------------------------------------------------------
    # 5. Genuine Post-Processing Isolation Test
    # -------------------------------------------------------------------------

    def test_08_post_processing_isolation_no_phy_imports(self):
        """Verify analyze_snr_grid_convergence does not import Sionna, BatchPHYEngine, calibration_l2_cuda, or phy_engine."""
        code = (
            "import analyze_snr_grid_convergence, sys\n"
            "assert 'sionna' not in sys.modules, 'sionna illegally loaded'\n"
            "assert 'calibration_l2_cuda' not in sys.modules, 'calibration_l2_cuda illegally loaded'\n"
            "assert 'phy_engine' not in sys.modules, 'phy_engine illegally loaded'\n"
            "assert 'cuda_engine' not in sys.modules, 'cuda_engine illegally loaded'\n"
            "print('ISOLATION_OK')"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Isolation check failed with stderr:\n{res.stderr}")
        self.assertIn("ISOLATION_OK", res.stdout)


if __name__ == "__main__":
    unittest.main()
