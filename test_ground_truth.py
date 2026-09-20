"""Unit tests for L3 Ground Truth & Baseline Policies."""
from pathlib import Path
import unittest
import numpy as np

from ground_truth import (
    GroundTruthConfig,
    GroundTruthRow,
    compute_ground_truth,
    load_calibration_csv,
    FixedRobustPolicy,
    FixedHighThroughputPolicy,
    LookupTable1D,
)


class TestGroundTruth(unittest.TestCase):
    """Test suite for BestMode decision logic, fallbacks, boundary checks, and baselines."""

    def setUp(self):
        # Synthetic calibration data with known physical behavior
        self.cal_data = {
            0.0: {
                "BPSK": {"ber": 0.15, "se": 0.005, "bit_errors": 10000, "total_bits": 7680000},
                "QPSK": {"ber": 0.22, "se": 0.006, "bit_errors": 20000, "total_bits": 7680000},
                "16QAM": {"ber": 0.32, "se": 0.007, "bit_errors": 30000, "total_bits": 7680000},
                "64QAM": {"ber": 0.38, "se": 0.008, "bit_errors": 40000, "total_bits": 7680000},
            },
            10.0: {
                "BPSK": {"ber": 0.005, "se": 0.001, "bit_errors": 500, "total_bits": 7680000},
                "QPSK": {"ber": 0.025, "se": 0.002, "bit_errors": 2500, "total_bits": 7680000},
                "16QAM": {"ber": 0.12, "se": 0.004, "bit_errors": 12000, "total_bits": 7680000},
                "64QAM": {"ber": 0.20, "se": 0.005, "bit_errors": 20000, "total_bits": 7680000},
            },
            16.0: {
                "BPSK": {"ber": 0.001, "se": 0.0003, "bit_errors": 100, "total_bits": 7680000},
                "QPSK": {"ber": 0.004, "se": 0.0008, "bit_errors": 400, "total_bits": 7680000},
                "16QAM": {"ber": 0.035, "se": 0.002, "bit_errors": 3500, "total_bits": 7680000},
                "64QAM": {"ber": 0.095, "se": 0.004, "bit_errors": 9500, "total_bits": 7680000},
            },
            22.0: {
                "BPSK": {"ber": 0.0001, "se": 0.00005, "bit_errors": 10, "total_bits": 7680000},
                "QPSK": {"ber": 0.0008, "se": 0.0002, "bit_errors": 80, "total_bits": 7680000},
                "16QAM": {"ber": 0.008, "se": 0.0008, "bit_errors": 800, "total_bits": 7680000},
                "64QAM": {"ber": 0.030, "se": 0.0025, "bit_errors": 3000, "total_bits": 7680000},
            },
            30.0: {
                "BPSK": {"ber": 0.00002, "se": 0.00001, "bit_errors": 2, "total_bits": 7680000},
                "QPSK": {"ber": 0.0001, "se": 0.00005, "bit_errors": 10, "total_bits": 7680000},
                "16QAM": {"ber": 0.0012, "se": 0.0004, "bit_errors": 120, "total_bits": 7680000},
                "64QAM": {"ber": 0.0050, "se": 0.0009, "bit_errors": 500, "total_bits": 7680000},
            },
        }

    def test_best_mode_selection_under_ber_target(self):
        """Verify BestMode selects the maximum spectral rate among eligible modes."""
        cfg = GroundTruthConfig(ber_target=0.01, fallback_policy="robustest_mode")
        rows = compute_ground_truth(self.cal_data, cfg)

        row_map = {r.snr_db: r for r in rows}

        # SNR 0 dB: No mode satisfies BER <= 0.01 -> Fallback BPSK
        self.assertTrue(row_map[0.0].fallback_used)
        self.assertEqual(row_map[0.0].best_mode, "BPSK")

        # SNR 10 dB: Only BPSK (0.005) satisfies BER <= 0.01
        self.assertFalse(row_map[10.0].fallback_used)
        self.assertEqual(row_map[10.0].best_mode, "BPSK")
        self.assertEqual(row_map[10.0].best_mode_bps, 1)

        # SNR 16 dB: BPSK (0.001) and QPSK (0.004) satisfy BER <= 0.01 -> Max rate is QPSK
        self.assertFalse(row_map[16.0].fallback_used)
        self.assertEqual(row_map[16.0].best_mode, "QPSK")
        self.assertEqual(row_map[16.0].best_mode_bps, 2)

        # SNR 22 dB: BPSK, QPSK, 16QAM (0.008) satisfy BER <= 0.01 -> Max rate is 16QAM
        self.assertFalse(row_map[22.0].fallback_used)
        self.assertEqual(row_map[22.0].best_mode, "16QAM")
        self.assertEqual(row_map[22.0].best_mode_bps, 4)

        # SNR 30 dB: All modes satisfy BER <= 0.01 -> Max rate is 64QAM
        self.assertFalse(row_map[30.0].fallback_used)
        self.assertEqual(row_map[30.0].best_mode, "64QAM")
        self.assertEqual(row_map[30.0].best_mode_bps, 6)

    def test_lut_thresholds_and_predictions(self):
        """Verify 1D LUT thresholds are exact midpoints and predict correctly."""
        cfg = GroundTruthConfig(ber_target=0.01)
        rows = compute_ground_truth(self.cal_data, cfg)
        lut = LookupTable1D.from_ground_truth(rows)

        # Transitions:
        # BPSK at 10 dB -> QPSK at 16 dB => Threshold = 13.0 dB
        # QPSK at 16 dB -> 16QAM at 22 dB => Threshold = 19.0 dB
        # 16QAM at 22 dB -> 64QAM at 30 dB => Threshold = 26.0 dB
        expected_thresholds = [13.0, 19.0, 26.0]
        actual_thresholds = [th for th, _, _ in lut.thresholds]
        self.assertEqual(actual_thresholds, expected_thresholds)

        # Prediction test
        self.assertEqual(lut.predict(5.0), "BPSK")
        self.assertEqual(lut.predict(13.0), "QPSK")
        self.assertEqual(lut.predict(18.9), "QPSK")
        self.assertEqual(lut.predict(19.1), "16QAM")
        self.assertEqual(lut.predict(26.1), "64QAM")
        self.assertEqual(lut.predict(40.0), "64QAM")

    def test_fixed_baselines(self):
        """Verify Fixed Robust and Fixed High-Throughput always return constant decisions."""
        p_rob = FixedRobustPolicy()
        p_high = FixedHighThroughputPolicy()

        for snr in [-10.0, 0.0, 15.0, 30.0, 50.0]:
            self.assertEqual(p_rob.predict(snr), "BPSK")
            self.assertEqual(p_high.predict(snr), "64QAM")

        # Array prediction
        snrs = np.array([0.0, 10.0, 20.0])
        self.assertEqual(p_rob.predict(snrs), ["BPSK", "BPSK", "BPSK"])
        self.assertEqual(p_high.predict(snrs), ["64QAM", "64QAM", "64QAM"])

    def test_boundary_uncertainty_flag(self):
        """Verify boundary_uncertain flag triggers when BER is within k*SE of target."""
        test_data = {
            22.0: {
                "BPSK": {"ber": 0.0001, "se": 0.00005},
                "QPSK": {"ber": 0.0008, "se": 0.0002},
                "16QAM": {"ber": 0.008, "se": 0.0012},  # 0.008 + 1.96*0.0012 = 0.01035 > 0.01, 0.008 - 1.96*0.0012 = 0.00565 < 0.01
                "64QAM": {"ber": 0.030, "se": 0.0025},
            },
            0.0: {
                "BPSK": {"ber": 0.15, "se": 0.005},
                "QPSK": {"ber": 0.22, "se": 0.006},
                "16QAM": {"ber": 0.32, "se": 0.007},
                "64QAM": {"ber": 0.38, "se": 0.008},
            },
        }
        cfg = GroundTruthConfig(ber_target=0.01, confidence_k=1.96)
        rows = compute_ground_truth(test_data, cfg)
        row_map = {r.snr_db: r for r in rows}

        self.assertTrue(row_map[22.0].boundary_uncertain)
        # At 0 dB: BER is 0.15, diff = 0.14 >> 1.96 * 0.005 -> boundary_uncertain False
        self.assertFalse(row_map[0.0].boundary_uncertain)

    def test_reliability_uncertain_vs_label_uncertain(self):
        """Verify distinct semantics: reliability_uncertain (CI overlaps target) vs label_uncertain (BestMode can alter)."""
        test_data = {
            14.0: {
                # BPSK CI overlaps target 0.01, but whether it passes or fails, fallback chooses BPSK!
                "BPSK": {"ber": 0.0095, "se": 0.0010, "bit_errors": 70, "total_bits": 7680000},
                "QPSK": {"ber": 0.025, "se": 0.0020, "bit_errors": 200, "total_bits": 7680000},
                "16QAM": {"ber": 0.080, "se": 0.0040, "bit_errors": 800, "total_bits": 7680000},
                "64QAM": {"ber": 0.150, "se": 0.0060, "bit_errors": 1500, "total_bits": 7680000},
            },
            28.0: {
                # 16QAM clearly passes. 64QAM CI overlaps target 0.01, altering BestMode between 16QAM and 64QAM!
                "BPSK": {"ber": 0.0001, "se": 0.00005, "bit_errors": 1, "total_bits": 7680000},
                "QPSK": {"ber": 0.0005, "se": 0.0001, "bit_errors": 5, "total_bits": 7680000},
                "16QAM": {"ber": 0.0030, "se": 0.0003, "bit_errors": 30, "total_bits": 7680000},
                "64QAM": {"ber": 0.0105, "se": 0.0008, "bit_errors": 105, "total_bits": 7680000},
            },
        }

        cfg = GroundTruthConfig(ber_target=0.01, fallback_policy="robustest_mode", confidence_k=1.96)
        rows = compute_ground_truth(test_data, cfg)
        row_map = {r.snr_db: r for r in rows}

        # SNR 14.0 dB: BPSK is marginally uncertain, but BestMode remains BPSK regardless -> label_uncertain is False
        r14 = row_map[14.0]
        self.assertTrue(r14.reliability_uncertain, "14 dB BPSK CI overlaps 0.0100 -> reliability_uncertain should be True")
        self.assertFalse(r14.label_uncertain, "14 dB BestMode is BPSK under both bounds -> label_uncertain should be False")
        self.assertEqual(r14.best_mode, "BPSK")

        # SNR 28.0 dB: 64QAM CI overlaps target, choice swings between 16QAM and 64QAM -> label_uncertain is True
        r28 = row_map[28.0]
        self.assertTrue(r28.reliability_uncertain, "28 dB 64QAM CI overlaps 0.0100 -> reliability_uncertain should be True")
        self.assertTrue(r28.label_uncertain, "28 dB BestMode alternates 16QAM vs 64QAM -> label_uncertain should be True")

    def test_refined_lut_switching_thresholds(self):
        """Verify refined LUT switching thresholds at 0.5 dB transition zones: 16.75, 22.75, 28.25 dB."""
        refined_data = {
            16.0: {"BPSK": {"ber": 0.007, "se": 0.0005}, "QPSK": {"ber": 0.014, "se": 0.0008}, "16QAM": {"ber": 0.045, "se": 0.001}, "64QAM": {"ber": 0.10, "se": 0.001}},
            16.5: {"BPSK": {"ber": 0.005, "se": 0.0004}, "QPSK": {"ber": 0.011, "se": 0.0007}, "16QAM": {"ber": 0.038, "se": 0.001}, "64QAM": {"ber": 0.09, "se": 0.001}},
            17.0: {"BPSK": {"ber": 0.004, "se": 0.0003}, "QPSK": {"ber": 0.008, "se": 0.0005}, "16QAM": {"ber": 0.032, "se": 0.001}, "64QAM": {"ber": 0.08, "se": 0.001}},
            22.0: {"BPSK": {"ber": 0.001, "se": 0.0001}, "QPSK": {"ber": 0.003, "se": 0.0002}, "16QAM": {"ber": 0.012, "se": 0.0006}, "64QAM": {"ber": 0.037, "se": 0.0009}},
            22.5: {"BPSK": {"ber": 0.0009, "se": 0.0001}, "QPSK": {"ber": 0.0025, "se": 0.0002}, "16QAM": {"ber": 0.0108, "se": 0.0005}, "64QAM": {"ber": 0.033, "se": 0.0008}},
            23.0: {"BPSK": {"ber": 0.0008, "se": 0.0001}, "QPSK": {"ber": 0.0020, "se": 0.0002}, "16QAM": {"ber": 0.0090, "se": 0.0005}, "64QAM": {"ber": 0.029, "se": 0.0007}},
            28.0: {"BPSK": {"ber": 0.0004, "se": 0.0001}, "QPSK": {"ber": 0.0008, "se": 0.0001}, "16QAM": {"ber": 0.0034, "se": 0.0003}, "64QAM": {"ber": 0.0106, "se": 0.0005}},
            28.5: {"BPSK": {"ber": 0.0003, "se": 0.0001}, "QPSK": {"ber": 0.0007, "se": 0.0001}, "16QAM": {"ber": 0.0028, "se": 0.0003}, "64QAM": {"ber": 0.0089, "se": 0.0004}},
        }

        cfg = GroundTruthConfig(ber_target=0.01)
        rows = compute_ground_truth(refined_data, cfg)
        lut = LookupTable1D.from_ground_truth(rows)

        # Expected transitions:
        # 16.5 dB (BPSK) -> 17.0 dB (QPSK): threshold = 16.75 dB
        # 22.5 dB (QPSK) -> 23.0 dB (16QAM): threshold = 22.75 dB
        # 28.0 dB (16QAM) -> 28.5 dB (64QAM): threshold = 28.25 dB
        expected = [(16.75, "BPSK", "QPSK"), (22.75, "QPSK", "16QAM"), (28.25, "16QAM", "64QAM")]
        self.assertEqual(lut.thresholds, expected)

        # Test predictions around boundaries
        self.assertEqual(lut.predict(16.5), "BPSK")
        self.assertEqual(lut.predict(16.74), "BPSK")
        self.assertEqual(lut.predict(16.75), "QPSK")
        self.assertEqual(lut.predict(22.74), "QPSK")
        self.assertEqual(lut.predict(22.75), "16QAM")
        self.assertEqual(lut.predict(28.24), "16QAM")
        self.assertEqual(lut.predict(28.25), "64QAM")

    def test_ci_upper_bound_eligibility(self):
        """Verify candidate mode is eligible iff upper 95% CI (BER + 1.96*SE) <= 0.0100."""
        test_data = {
            20.0: {
                # BPSK: upper CI = 0.005 + 1.96 * 0.0005 = 0.00598 <= 0.01 -> eligible
                "BPSK": {"ber": 0.005, "se": 0.0005},
                # QPSK: nominal BER = 0.008 <= 0.01, upper CI = 0.008 + 1.96 * 0.001 = 0.00996 <= 0.01 -> eligible
                "QPSK": {"ber": 0.008, "se": 0.0010},
                # 16QAM: nominal BER = 0.009 <= 0.01 passes, but upper CI = 0.009 + 1.96 * 0.001 = 0.01096 > 0.01 -> REJECTED
                "16QAM": {"ber": 0.009, "se": 0.0010},
                # 64QAM: nominal BER = 0.050 > 0.01 -> REJECTED
                "64QAM": {"ber": 0.050, "se": 0.0010},
            }
        }
        cfg = GroundTruthConfig(ber_target=0.01)
        rows = compute_ground_truth(test_data, cfg)
        r = rows[0]

        self.assertTrue(r.bpsk_eligible)
        self.assertTrue(r.qpsk_eligible, "Upper CI <= 0.0100 must be eligible")
        self.assertFalse(r.qam16_eligible, "Nominal passes but upper CI > 0.0100 must be REJECTED")
        self.assertFalse(r.qam64_eligible)
        self.assertEqual(r.best_mode, "QPSK", "Highest rate eligible mode under CI rule is QPSK (2 bpcu)")

    def test_fallback_behavior(self):
        """Verify fallback selects BPSK with fallback_used=True when no mode qualifies."""
        test_data = {
            5.0: {
                "BPSK": {"ber": 0.05, "se": 0.002},
                "QPSK": {"ber": 0.08, "se": 0.003},
                "16QAM": {"ber": 0.15, "se": 0.004},
                "64QAM": {"ber": 0.25, "se": 0.005},
            }
        }
        cfg = GroundTruthConfig(ber_target=0.01, fallback_policy="robustest_mode")
        rows = compute_ground_truth(test_data, cfg)
        r = rows[0]

        self.assertTrue(r.fallback_used)
        self.assertEqual(r.best_mode, "BPSK")
        self.assertEqual(r.best_mode_bps, 1)

    def test_lut_midpoint_construction(self):
        """Verify LUT switching thresholds are exactly the midpoint of adjacent sampled grid points."""
        test_data = {
            10.0: {"BPSK": {"ber": 0.005, "se": 0.0005}, "QPSK": {"ber": 0.02, "se": 0.001}},
            15.0: {"BPSK": {"ber": 0.001, "se": 0.0002}, "QPSK": {"ber": 0.006, "se": 0.0005}},
        }
        cfg = GroundTruthConfig(ber_target=0.01)
        rows = compute_ground_truth(test_data, cfg)
        lut = LookupTable1D.from_ground_truth(rows)

        self.assertEqual(len(lut.thresholds), 1)
        th, p_m, n_m = lut.thresholds[0]
        self.assertEqual(th, 12.5, "Midpoint between 10.0 and 15.0 must be exactly 12.5 dB")
        self.assertEqual(p_m, "BPSK")
        self.assertEqual(n_m, "QPSK")

    def test_monotonic_mode_progression(self):
        """Verify detection of monotonic and non-monotonic mode progressions."""
        # Case 1: Monotonic
        mono_data = {
            10.0: {"BPSK": {"ber": 0.005, "se": 0.0005}, "QPSK": {"ber": 0.02, "se": 0.001}},
            15.0: {"BPSK": {"ber": 0.001, "se": 0.0002}, "QPSK": {"ber": 0.006, "se": 0.0005}},
        }
        lut_mono = LookupTable1D.from_ground_truth(compute_ground_truth(mono_data, GroundTruthConfig(ber_target=0.01)))
        self.assertEqual(len(lut_mono.non_monotonic_transitions), 0)

        # Case 2: Inversion / non-monotonic
        inversion_data = {
            10.0: {"BPSK": {"ber": 0.001, "se": 0.0001}, "QPSK": {"ber": 0.004, "se": 0.0003}},  # QPSK (2 bpcu)
            15.0: {"BPSK": {"ber": 0.008, "se": 0.0005}, "QPSK": {"ber": 0.015, "se": 0.0010}},  # BPSK (1 bpcu)
        }
        lut_inv = LookupTable1D.from_ground_truth(compute_ground_truth(inversion_data, GroundTruthConfig(ber_target=0.01)))
        self.assertEqual(len(lut_inv.non_monotonic_transitions), 1)
        th, p_m, n_m = lut_inv.non_monotonic_transitions[0]
        self.assertEqual(p_m, "QPSK")
        self.assertEqual(n_m, "BPSK")

    def test_frozen_final_l2_dataset_acceptance(self):
        """Acceptance test verifying complete integration with frozen final L2 dataset."""
        cal_path = Path("results/l2_cuda_rtx3060_final/calibration_1d_cuda_pooled.csv")
        if not cal_path.exists():
            self.skipTest("Frozen final L2 calibration dataset not present")

        cal_data = load_calibration_csv(cal_path)

        # 1. Exactly 25 SNR points and 4 modulations per SNR
        self.assertEqual(len(cal_data), 25, "Expected exactly 25 SNR points")
        for snr, mods in cal_data.items():
            self.assertEqual(len(mods), 4, f"SNR {snr} must have exactly 4 modulation modes")
            self.assertEqual(set(mods.keys()), {"BPSK", "QPSK", "16QAM", "64QAM"})

        cfg = GroundTruthConfig(ber_target=0.01, fallback_policy="robustest_mode", confidence_k=1.96)
        rows = compute_ground_truth(cal_data, cfg)
        self.assertEqual(len(rows), 25)

        # 2. Mode counts under canonical conservative CI rule:
        # At 23.0 dB, 16QAM has upper CI = 0.0100011 > 0.0100 -> QPSK selected
        mode_counts = {}
        for r in rows:
            mode_counts[r.best_mode] = mode_counts.get(r.best_mode, 0) + 1
        self.assertEqual(mode_counts.get("BPSK"), 10)
        self.assertEqual(mode_counts.get("QPSK"), 7)
        self.assertEqual(mode_counts.get("16QAM"), 4)
        self.assertEqual(mode_counts.get("64QAM"), 4)

        # 3. Fallback points (0.0 to 14.0 dB: at 14 dB BPSK upper CI marginally exceeds 0.01)
        fallback_snrs = [r.snr_db for r in rows if r.fallback_used]
        self.assertEqual(fallback_snrs, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])

        # 4. Uncertainty points
        rel_unc_snrs = [r.snr_db for r in rows if r.reliability_uncertain]
        lbl_unc_snrs = [r.snr_db for r in rows if r.label_uncertain]
        self.assertEqual(rel_unc_snrs, [14.0, 23.0, 28.0])
        self.assertEqual(lbl_unc_snrs, [23.0, 28.0])

        # 5. LUT thresholds & monotonicity
        # QPSK -> 16QAM boundary shifts from 22.75 to 23.25 dB due to conservative CI rule at 23.0 dB
        lut = LookupTable1D.from_ground_truth(rows)
        self.assertEqual(len(lut.non_monotonic_transitions), 0, "Monotonicity must PASS")
        self.assertEqual(
            lut.thresholds,
            [(16.75, "BPSK", "QPSK"), (23.25, "QPSK", "16QAM"), (28.25, "16QAM", "64QAM")]
        )


if __name__ == "__main__":
    unittest.main()
