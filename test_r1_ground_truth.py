"""Tests for R1 Ground Truth decision engine using synthetic fixtures only."""
import unittest

from r1_ground_truth import (
    CodedCandidateRecord,
    R1GroundTruthConfig,
    R1GroundTruthRow,
    compute_confidence_interval,
    compute_r1_ground_truth,
    evaluate_r1_baselines,
    resolve_equal_eta_tie,
)


class TestR1GroundTruth(unittest.TestCase):
    """Test suite for R1 configurable ground truth generator."""

    def test_01_confidence_interval_methods_synthetic(self):
        """Verify Wald, Wilson, and Clopper-Pearson methods on zero and non-zero error fixtures."""
        # 0 errors out of 100 trials
        w_low, w_high = compute_confidence_interval(0, 100, method="wald")
        self.assertEqual(w_low, 0.0)
        self.assertEqual(w_high, 0.0)  # Wald collapses to 0 width at p=0

        wil_low, wil_high = compute_confidence_interval(0, 100, method="wilson")
        self.assertEqual(wil_low, 0.0)
        self.assertGreater(wil_high, 0.0)  # Wilson maintains valid positive upper bound

        cp_low, cp_high = compute_confidence_interval(0, 100, method="clopper_pearson")
        self.assertEqual(cp_low, 0.0)
        self.assertGreater(cp_high, 0.03)
        self.assertGreater(wil_high, 0.03)

        # 10 errors out of 100 trials (p = 0.10)
        w_l, w_h = compute_confidence_interval(10, 100, method="wald")
        self.assertAlmostEqual(w_l, 0.10 - 1.96 * 0.03, places=3)
        self.assertAlmostEqual(w_h, 0.10 + 1.96 * 0.03, places=3)

    def test_02_single_and_multiple_eligible_actions(self):
        """Verify single eligible action and argmax spectral efficiency selection."""
        cal_data = {
            10.0: {
                "BPSK-1/2": {"bler": 0.02, "num_blocks": 100, "spectral_efficiency": 0.50, "modulation": "BPSK"},
                "QPSK-1/2": {"bler": 0.25, "num_blocks": 100, "spectral_efficiency": 1.00, "modulation": "QPSK"},
            },
            15.0: {
                "BPSK-1/2": {"bler": 0.00, "num_blocks": 100, "spectral_efficiency": 0.50, "modulation": "BPSK"},
                "QPSK-1/2": {"bler": 0.01, "num_blocks": 100, "spectral_efficiency": 1.00, "modulation": "QPSK"},
                "16QAM-1/2": {"bler": 0.30, "num_blocks": 100, "spectral_efficiency": 2.00, "modulation": "16QAM"},
            },
        }

        cfg = R1GroundTruthConfig(bler_target=0.10, eligibility_rule="nominal")
        rows = compute_r1_ground_truth(cal_data, cfg)

        self.assertEqual(len(rows), 2)
        # At 10 dB, only BPSK-1/2 is eligible
        self.assertEqual(rows[0].best_action, "BPSK-1/2")
        self.assertEqual(rows[0].best_spectral_efficiency, 0.50)
        self.assertFalse(rows[0].fallback_used)

        # At 15 dB, both BPSK-1/2 and QPSK-1/2 are eligible; QPSK-1/2 has higher eta
        self.assertEqual(rows[1].best_action, "QPSK-1/2")
        self.assertEqual(rows[1].best_spectral_efficiency, 1.00)
        self.assertFalse(rows[1].fallback_used)

    def test_03_no_eligible_action_activates_fallback(self):
        """Verify fallback triggers when all candidates exceed BLER target."""
        cal_data = {
            5.0: {
                "BPSK-1/2": {"bler": 0.30, "num_blocks": 100, "spectral_efficiency": 0.50, "modulation": "BPSK"},
                "QPSK-1/2": {"bler": 0.60, "num_blocks": 100, "spectral_efficiency": 1.00, "modulation": "QPSK"},
            }
        }

        cfg = R1GroundTruthConfig(bler_target=0.10, fallback_rule="lowest_rate")
        rows = compute_r1_ground_truth(cal_data, cfg)

        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].fallback_used)
        self.assertEqual(rows[0].best_action, "BPSK-1/2")
        self.assertIn("fallback:lowest_spectral_efficiency", rows[0].selection_reason)

    def test_04_equal_eta_tie_break_strategies(self):
        """Verify equal-eta resolution (16QAM-3/4 vs 64QAM-1/2, both eta=3.0) across all 4 candidate rules."""
        cal_data = {
            20.0: {
                "16QAM-3/4": {"bler": 0.05, "num_blocks": 100, "spectral_efficiency": 3.00, "modulation": "16QAM"},
                "64QAM-1/2": {"bler": 0.02, "num_blocks": 100, "spectral_efficiency": 3.00, "modulation": "64QAM"},
            }
        }

        # Rule A: lowest_bler -> selects 64QAM-1/2 (BLER 0.02 < 0.05)
        cfg_a = R1GroundTruthConfig(bler_target=0.10, tie_break_rule="lowest_bler")
        row_a = compute_r1_ground_truth(cal_data, cfg_a)[0]
        self.assertEqual(row_a.best_action, "64QAM-1/2")
        self.assertTrue(row_a.equal_eta_tie_broken)

        # Rule C: lower_modulation -> selects 16QAM-3/4 (16QAM < 64QAM)
        cfg_c = R1GroundTruthConfig(bler_target=0.10, tie_break_rule="lower_modulation")
        row_c = compute_r1_ground_truth(cal_data, cfg_c)[0]
        self.assertEqual(row_c.best_action, "16QAM-3/4")
        self.assertTrue(row_c.equal_eta_tie_broken)

        # Rule D: stronger_coding -> selects 64QAM-1/2 (Rc=1/2 is stronger than 3/4)
        cfg_d = R1GroundTruthConfig(bler_target=0.10, tie_break_rule="stronger_coding")
        row_d = compute_r1_ground_truth(cal_data, cfg_d)[0]
        self.assertEqual(row_d.best_action, "64QAM-1/2")
        self.assertTrue(row_d.equal_eta_tie_broken)

        # Rule F: predefined -> follows predefined preference list
        cfg_f1 = R1GroundTruthConfig(bler_target=0.10, tie_break_rule="predefined", predefined_action_order=["16QAM-3/4", "64QAM-1/2"])
        row_f1 = compute_r1_ground_truth(cal_data, cfg_f1)[0]
        self.assertEqual(row_f1.best_action, "16QAM-3/4")

        cfg_f2 = R1GroundTruthConfig(bler_target=0.10, tie_break_rule="predefined", predefined_action_order=["64QAM-1/2", "16QAM-3/4"])
        row_f2 = compute_r1_ground_truth(cal_data, cfg_f2)[0]
        self.assertEqual(row_f2.best_action, "64QAM-1/2")

    def test_05_ci_overlap_uncertainty_flag(self):
        """Verify label_uncertain=True when CI interval straddles the selection boundary."""
        cal_data = {
            18.0: {
                "BPSK-1/2": {"bler": 0.00, "num_blocks": 100, "spectral_efficiency": 0.50, "modulation": "BPSK"},
                # QPSK-1/2 has bler=0.08, but 95% CI is [0.03, 0.13] which crosses bler_target=0.10
                "QPSK-1/2": {"bler": 0.08, "num_blocks": 100, "spectral_efficiency": 1.00, "modulation": "QPSK"},
            }
        }

        # Under conservative CI, QPSK-1/2 upper CI (0.13) > 0.10, so it is ineligible -> BPSK-1/2 chosen
        cfg_cons = R1GroundTruthConfig(bler_target=0.10, eligibility_rule="conservative_ci")
        row_cons = compute_r1_ground_truth(cal_data, cfg_cons)[0]
        self.assertEqual(row_cons.best_action, "BPSK-1/2")
        # But optimistically, lower CI (0.03) <= 0.10, so label uncertainty is True!
        self.assertTrue(row_cons.label_uncertain)

    def test_06_baseline_evaluation_metrics(self):
        """Verify baseline evaluation computes valid averages and violation counts."""
        cal_data = {
            10.0: {
                "BPSK-1/2": {"bler": 0.01, "num_blocks": 100, "spectral_efficiency": 0.50, "modulation": "BPSK"},
                "64QAM-3/4": {"bler": 0.80, "num_blocks": 100, "spectral_efficiency": 4.50, "modulation": "64QAM"},
            },
            25.0: {
                "BPSK-1/2": {"bler": 0.00, "num_blocks": 100, "spectral_efficiency": 0.50, "modulation": "BPSK"},
                "64QAM-3/4": {"bler": 0.02, "num_blocks": 100, "spectral_efficiency": 4.50, "modulation": "64QAM"},
            },
        }

        cfg = R1GroundTruthConfig(bler_target=0.10)
        gt_rows = compute_r1_ground_truth(cal_data, cfg)
        summary = evaluate_r1_baselines(gt_rows, cfg)

        self.assertEqual(summary["num_operating_points"], 2)
        self.assertAlmostEqual(summary["mean_spectral_efficiency"]["fixed_robust"], 0.50)
        self.assertAlmostEqual(summary["mean_spectral_efficiency"]["fixed_high_throughput"], 4.50)
        self.assertEqual(summary["total_violations"]["fixed_robust"], 0)
        self.assertEqual(summary["total_violations"]["fixed_high_throughput"], 1)  # 64QAM-3/4 failed at 10 dB


if __name__ == "__main__":
    unittest.main()
