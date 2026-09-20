"""Unit tests for Light vs Deep MC Budget Study & CI-based Ground Truth.

Verifies:
1. CI upper-bound eligibility:
   - Mode accepted when upper CI <= BER_target.
   - Mode rejected when nominal BER passes but upper CI exceeds BER_target.
   - Mode rejected when nominal BER exceeds BER_target.
2. Light vs Deep comparison tool:
   - Successfully consumes synthetic CSVs and produces all 4 artifacts.
   - mc_budget_comparison.csv contains all expected metrics.
   - label_comparison.csv detects label flips accurately.
   - transition_comparison.csv detects switching threshold shifts accurately.
   - mc_budget_study.md explicitly answers all 6 research questions.
3. Strict isolation:
   - Comparison tool NEVER invokes PHY simulation.
"""
import csv
import math
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Tuple
import unittest
from unittest.mock import patch

from ground_truth import (
    BER_TARGET,
    CONFIDENCE_K,
    GroundTruthConfig,
    compute_ground_truth,
)
from compare_mc_budgets import (
    run_budget_comparison,
    select_optimal_cart_depth,
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


class TestMCBudgetStudy(unittest.TestCase):
    """Test suite for CI-based Ground Truth and Light vs Deep budget study."""

    def setUp(self):
        # Sample synthetic calibration generator helper
        pass

    # -------------------------------------------------------------------------
    # 1. CI Upper-Bound Eligibility Tests
    # -------------------------------------------------------------------------

    def test_01_mode_accepted_when_upper_ci_below_target(self):
        """Mode accepted when full upper 95% CI is below BER_target (0.0100)."""
        # Nominal = 0.0070, SE = 0.0010 -> upper CI = 0.0070 + 1.96*0.0010 = 0.00896 <= 0.0100
        test_data = {
            20.0: {
                "BPSK": {"ber": 0.0010, "se": 0.0001, "num_blocks": 20000},
                "QPSK": {"ber": 0.0070, "se": 0.0010, "num_blocks": 20000},
                "16QAM": {"ber": 0.0300, "se": 0.0020, "num_blocks": 20000},
                "64QAM": {"ber": 0.0800, "se": 0.0040, "num_blocks": 20000},
            }
        }
        cfg = GroundTruthConfig(ber_target=0.0100, confidence_k=1.96)
        rows = compute_ground_truth(test_data, cfg)
        r = rows[0]

        self.assertTrue(r.qpsk_eligible, "QPSK upper CI 0.00896 <= 0.0100 must be eligible")
        self.assertEqual(r.best_mode, "QPSK")
        self.assertFalse(r.fallback_used)

    def test_02_mode_rejected_when_nominal_passes_but_upper_ci_exceeds_target(self):
        """Mode rejected when nominal BER passes but upper 95% CI exceeds target."""
        # Nominal = 0.0090 <= 0.0100 (nominal pass!)
        # SE = 0.0010 -> upper CI = 0.0090 + 1.96*0.0010 = 0.01096 > 0.0100 (CI reject!)
        test_data = {
            20.0: {
                "BPSK": {"ber": 0.0010, "se": 0.0001, "num_blocks": 20000},
                "QPSK": {"ber": 0.0090, "se": 0.0010, "num_blocks": 20000},
                "16QAM": {"ber": 0.0300, "se": 0.0020, "num_blocks": 20000},
                "64QAM": {"ber": 0.0800, "se": 0.0040, "num_blocks": 20000},
            }
        }
        cfg = GroundTruthConfig(ber_target=0.0100, confidence_k=1.96)
        rows = compute_ground_truth(test_data, cfg)
        r = rows[0]

        self.assertTrue(r.bpsk_eligible)
        self.assertFalse(
            r.qpsk_eligible,
            "QPSK has nominal BER=0.0090 but upper CI=0.01096 > 0.0100; MUST be rejected under CI rule!"
        )
        # Because QPSK is rejected, BPSK (1 bpcu) must be selected
        self.assertEqual(r.best_mode, "BPSK")

    def test_03_mode_rejected_when_nominal_exceeds_target(self):
        """Mode rejected when nominal BER exceeds target."""
        test_data = {
            20.0: {
                "BPSK": {"ber": 0.0020, "se": 0.0002, "num_blocks": 20000},
                "QPSK": {"ber": 0.0150, "se": 0.0010, "num_blocks": 20000},
                "16QAM": {"ber": 0.0400, "se": 0.0020, "num_blocks": 20000},
                "64QAM": {"ber": 0.0900, "se": 0.0040, "num_blocks": 20000},
            }
        }
        cfg = GroundTruthConfig(ber_target=0.0100, confidence_k=1.96)
        rows = compute_ground_truth(test_data, cfg)
        r = rows[0]

        self.assertFalse(r.qpsk_eligible)
        self.assertFalse(r.qam16_eligible)
        self.assertEqual(r.best_mode, "BPSK")

    # -------------------------------------------------------------------------
    # 2. Light vs Deep Comparison Synthetic Generation Helper
    # -------------------------------------------------------------------------

    def _generate_synthetic_cal_csv(
        self,
        output_path: Path,
        num_blocks: int,
        se_scale: float = 1.0,
        inject_flip_at_23: bool = False,
    ) -> None:
        """Generate a clean synthetic calibration CSV with realistic SNR progression."""
        fieldnames = [
            "mode_id", "modulation", "bits_per_symbol", "snr_db", "num_blocks",
            "total_bits", "bit_errors", "ber", "block_errors", "bler",
            "mean_block_ber", "std_block_ber", "se_block_ber", "z_ab",
            "fixed_budget_reached", "is_stable", "reliability_uncertain",
        ]

        rows = []
        for snr in TEST_GRID_25_POINTS:
            for m in TEST_MODES:
                mid = m["mode_id"]
                mod = m["modulation"]
                bps = m["bits_per_symbol"]

                # Nominal synthetic BER functions
                if mid == 0:  # BPSK
                    ber = 0.5 * math.exp(-0.25 * (10 ** (snr / 10.0)))
                elif mid == 1:  # QPSK
                    ber = 0.5 * math.exp(-0.12 * (10 ** (snr / 10.0)))
                elif mid == 2:  # 16QAM
                    ber = 0.5 * math.exp(-0.020 * (10 ** (snr / 10.0)))
                else:  # 64QAM
                    ber = 0.5 * math.exp(-0.0050 * (10 ** (snr / 10.0)))

                ber = max(1e-6, min(0.5, ber))
                # Standard error scales inversely with sqrt(num_blocks)
                se = math.sqrt(ber * (1.0 - ber) / num_blocks) * se_scale

                # If inject_flip_at_23 is requested (for testing flip detection):
                # At 23.0 dB for 16QAM:
                # In Light: ber=0.0095, se=0.0010 -> upper CI = 0.01146 > 0.0100 (ineligible)
                # In Deep:  ber=0.0095, se=0.0001 -> upper CI = 0.00970 <= 0.0100 (eligible)
                if inject_flip_at_23 and abs(snr - 23.0) < 1e-3 and mid == 2:
                    ber = 0.0095
                    se = 0.0010 if num_blocks <= 20000 else 0.0001

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
                    "z_ab": 0.25,
                    "fixed_budget_reached": True,
                    "is_stable": "N/A (deprecated)",
                    "reliability_uncertain": uncertain,
                })

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # -------------------------------------------------------------------------
    # 3. Light vs Deep Comparison Tool Execution Tests
    # -------------------------------------------------------------------------

    def test_04_compare_mc_budgets_produces_all_four_artifacts(self):
        """run_budget_comparison produces all 4 artifacts with complete schema."""
        with tempfile.TemporaryDirectory() as tmp_root:
            light_dir = Path(tmp_root) / "light_10k"
            deep_dir = Path(tmp_root) / "deep_70k"
            study_dir = Path(tmp_root) / "budget_study"

            light_dir.mkdir()
            deep_dir.mkdir()

            self._generate_synthetic_cal_csv(light_dir / "calibration_1d_cuda_pooled.csv", num_blocks=20000)
            self._generate_synthetic_cal_csv(deep_dir / "calibration_1d_cuda_pooled.csv", num_blocks=140000)

            # Run comparison tool
            res = run_budget_comparison(
                light_dir=light_dir,
                deep_dir=deep_dir,
                output_dir=study_dir,
            )

            csv_budget = study_dir / "mc_budget_comparison.csv"
            csv_label = study_dir / "label_comparison.csv"
            csv_transition = study_dir / "transition_comparison.csv"
            md_study = study_dir / "mc_budget_study.md"

            self.assertTrue(csv_budget.exists(), "mc_budget_comparison.csv must exist")
            self.assertTrue(csv_label.exists(), "label_comparison.csv must exist")
            self.assertTrue(csv_transition.exists(), "transition_comparison.csv must exist")
            self.assertTrue(md_study.exists(), "mc_budget_study.md must exist")

            # Check rows count
            with open(csv_budget, "r", encoding="utf-8") as f:
                budget_rows = list(csv.DictReader(f))
            self.assertEqual(len(budget_rows), 100, "25 SNRs x 4 modes = 100 rows expected")

            with open(csv_label, "r", encoding="utf-8") as f:
                label_rows = list(csv.DictReader(f))
            self.assertEqual(len(label_rows), 25, "25 SNR points expected")

            with open(csv_transition, "r", encoding="utf-8") as f:
                transition_rows = list(csv.DictReader(f))
            self.assertEqual(len(transition_rows), 3, "3 mode transitions expected")

            # Check Markdown answers all 6 questions
            report_text = md_study.read_text(encoding="utf-8")
            self.assertIn("Q1: How much uncertainty is reduced from 10k to 70k?", report_text)
            self.assertIn("Q2: Which operating points change overlap status?", report_text)
            self.assertIn("Q3: Does any BestMode label change?", report_text)
            self.assertIn("Q4: Do switching thresholds move?", report_text)
            self.assertIn("Q5: Does the learned Decision Tree policy change?", report_text)
            self.assertIn("Q6: What compute cost is paid?", report_text)
            self.assertIn("62.20% reduction", report_text)
            self.assertIn("7.00x", report_text)

    def test_05_label_flip_detection(self):
        """Verify comparison tool reliably detects BestMode label flips."""
        with tempfile.TemporaryDirectory() as tmp_root:
            light_dir = Path(tmp_root) / "light_10k"
            deep_dir = Path(tmp_root) / "deep_70k"
            study_dir = Path(tmp_root) / "budget_study"

            light_dir.mkdir()
            deep_dir.mkdir()

            # Inject label flip at 23.0 dB
            self._generate_synthetic_cal_csv(
                light_dir / "calibration_1d_cuda_pooled.csv",
                num_blocks=20000,
                inject_flip_at_23=True,
            )
            self._generate_synthetic_cal_csv(
                deep_dir / "calibration_1d_cuda_pooled.csv",
                num_blocks=140000,
                inject_flip_at_23=True,
            )

            res = run_budget_comparison(
                light_dir=light_dir,
                deep_dir=deep_dir,
                output_dir=study_dir,
            )

            self.assertGreaterEqual(res["flips_count"], 1)

            csv_label = study_dir / "label_comparison.csv"
            with open(csv_label, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            row_23 = next(r for r in rows if abs(float(r["snr_db"]) - 23.0) < 1e-3)
            self.assertEqual(row_23["label_flip"], "True")
            self.assertEqual(row_23["light_best_mode"], "QPSK")
            self.assertEqual(row_23["deep_best_mode"], "16QAM")

    def test_06_transition_shift_detection(self):
        """Verify comparison tool detects switching boundary threshold shifts."""
        with tempfile.TemporaryDirectory() as tmp_root:
            light_dir = Path(tmp_root) / "light_10k"
            deep_dir = Path(tmp_root) / "deep_70k"
            study_dir = Path(tmp_root) / "budget_study"

            light_dir.mkdir()
            deep_dir.mkdir()

            self._generate_synthetic_cal_csv(
                light_dir / "calibration_1d_cuda_pooled.csv",
                num_blocks=20000,
                inject_flip_at_23=True,
            )
            self._generate_synthetic_cal_csv(
                deep_dir / "calibration_1d_cuda_pooled.csv",
                num_blocks=140000,
                inject_flip_at_23=True,
            )

            run_budget_comparison(
                light_dir=light_dir,
                deep_dir=deep_dir,
                output_dir=study_dir,
            )

            csv_transition = study_dir / "transition_comparison.csv"
            with open(csv_transition, "r", encoding="utf-8") as f:
                transitions = {r["transition"]: r for r in csv.DictReader(f)}

            qpsk_16qam = transitions["QPSK->16QAM"]
            # Light was QPSK at 23.0 -> threshold is between 23.0 and 23.5 dB (23.25)
            # Deep was 16QAM at 23.0 -> threshold is between 22.5 and 23.0 dB (22.75)
            self.assertIsNotNone(qpsk_16qam["lut_threshold_shift_db"])
            shift = float(qpsk_16qam["lut_threshold_shift_db"])
            self.assertAlmostEqual(shift, -0.50, places=2)

    def test_07_no_phy_invocation_by_comparison_tool(self):
        """Verify comparison tool executes purely as post-processing without invoking BatchPHYEngine."""
        with tempfile.TemporaryDirectory() as tmp_root:
            light_dir = Path(tmp_root) / "light_10k"
            deep_dir = Path(tmp_root) / "deep_70k"
            study_dir = Path(tmp_root) / "budget_study"

            light_dir.mkdir()
            deep_dir.mkdir()

            self._generate_synthetic_cal_csv(light_dir / "calibration_1d_cuda_pooled.csv", num_blocks=20000)
            self._generate_synthetic_cal_csv(deep_dir / "calibration_1d_cuda_pooled.csv", num_blocks=140000)

            # Patch BatchPHYEngine in cuda_engine and PHYEngine in phy_engine: must NEVER be called
            with patch("cuda_engine.BatchPHYEngine") as MockCudaEngine, \
                 patch("phy_engine.PHYEngine") as MockPhyEngine:
                MockCudaEngine.side_effect = RuntimeError("BatchPHYEngine was illegally invoked by comparison tool!")
                MockPhyEngine.side_effect = RuntimeError("PHYEngine was illegally invoked by comparison tool!")

                res = run_budget_comparison(
                    light_dir=light_dir,
                    deep_dir=deep_dir,
                    output_dir=study_dir,
                )

                MockCudaEngine.assert_not_called()
                MockPhyEngine.assert_not_called()
                self.assertTrue((study_dir / "mc_budget_study.md").exists())

    def test_08_uncertainty_reduction_ratio_and_scaling(self):
        """Verify SE reduction ratio across 140k vs 20k blocks scales according to sqrt(7)."""
        with tempfile.TemporaryDirectory() as tmp_root:
            light_dir = Path(tmp_root) / "light_10k"
            deep_dir = Path(tmp_root) / "deep_70k"
            study_dir = Path(tmp_root) / "budget_study"

            light_dir.mkdir()
            deep_dir.mkdir()

            self._generate_synthetic_cal_csv(light_dir / "calibration_1d_cuda_pooled.csv", num_blocks=20000)
            self._generate_synthetic_cal_csv(deep_dir / "calibration_1d_cuda_pooled.csv", num_blocks=140000)

            res = run_budget_comparison(
                light_dir=light_dir,
                deep_dir=deep_dir,
                output_dir=study_dir,
            )

            budget_rows = res["mc_budget_rows"]
            # Theoretical reduction is 1 - 1/sqrt(7) = ~62.20%
            expected_reduction = (1.0 - 1.0 / math.sqrt(7.0)) * 100.0

            measured_reductions = [r["se_reduction_pct"] for r in budget_rows if r["light_se"] > 1e-12]
            self.assertGreater(len(measured_reductions), 0)
            mean_reduction = sum(measured_reductions) / len(measured_reductions)
            self.assertAlmostEqual(mean_reduction, expected_reduction, delta=0.5)

    def test_09_post_processing_isolation_no_sionna_or_phy_imports(self):
        """Verify compare_mc_budgets does not import Sionna, BatchPHYEngine, calibration_l2_cuda, or phy_engine."""
        import subprocess
        import sys

        code = (
            "import compare_mc_budgets, sys\n"
            "assert 'sionna' not in sys.modules, 'sionna illegally loaded'\n"
            "assert 'calibration_l2_cuda' not in sys.modules, 'calibration_l2_cuda illegally loaded'\n"
            "assert 'phy_engine' not in sys.modules, 'phy_engine illegally loaded'\n"
            "assert 'cuda_engine' not in sys.modules, 'cuda_engine illegally loaded'\n"
            "print('ISOLATION_OK')"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Isolation check failed with stderr:\n{res.stderr}")
        self.assertIn("ISOLATION_OK", res.stdout)

    def test_10_cart_depth_model_selection(self):
        """Verify select_optimal_cart_depth sweeps 1..5 and chooses smallest depth with 100% fidelity."""
        # 1. Two-class dataset: depth 1 must achieve 100% fidelity and be selected
        snrs_2class = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
        labels_2class = ["BPSK", "BPSK", "BPSK", "BPSK", "QPSK", "QPSK", "QPSK", "QPSK"]
        clf_2c, m_2c = select_optimal_cart_depth(snrs_2class, labels_2class)
        self.assertEqual(m_2c.max_depth_param, 1, "Smallest depth for 2 classes must be 1")
        self.assertEqual(m_2c.actual_depth, 1)
        self.assertEqual(m_2c.accuracy, 1.0)
        self.assertEqual(len(m_2c.learned_thresholds), 1)

        # 2. Four-class dataset: depth 3 must achieve 100% fidelity and be selected over depth 4 and 5
        snrs_4class = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
        labels_4class = ["BPSK", "BPSK", "QPSK", "QPSK", "16QAM", "16QAM", "64QAM", "64QAM"]
        clf_4c, m_4c = select_optimal_cart_depth(snrs_4class, labels_4class)
        self.assertEqual(m_4c.max_depth_param, 3, "Smallest depth for 4 classes must be 3")
        self.assertEqual(m_4c.accuracy, 1.0)
        self.assertEqual(len(m_4c.learned_thresholds), 3)


if __name__ == "__main__":
    unittest.main()
