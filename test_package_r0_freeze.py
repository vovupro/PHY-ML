"""Fixture-based tests for canonical R0 baseline packaging and freeze gate.

Verifies:
1. 25 + 3 -> 28-point calibration dataset merging.
2. Duplicate protection and schema integrity.
3. Conservative CI Ground Truth synthesis.
4. CART depth model selection (smallest depth with 100% fidelity, fallback handling).
5. Policy and baseline evaluation (sampled-grid averages, violation counts).
6. Manifest generation and SHA-256 cryptographic hashing.
7. Data-driven freeze gate (PASS under valid conditions, REVIEW under regressions).
8. Strict post-processing isolation (no imports of Sionna or PHY engines).
9. Full end-to-end packaging pipeline and artifact generation.
"""
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Sequence, Tuple
import unittest

import joblib

from ground_truth import BER_TARGET, CONFIDENCE_K, GroundTruthConfig, compute_ground_truth
from cart_1d import CART1DClassifier
from package_r0_freeze import (
    compute_file_sha256,
    evaluate_policies_and_baselines,
    get_git_commit,
    load_and_merge_r0_calibration,
    package_r0_freeze,
    run_freeze_gate,
    sweep_and_select_cart_depth,
)

SYNTHETIC_DEEP_25_SNRS: Tuple[float, ...] = (
    0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0,
    16.0, 16.5, 17.0, 17.5, 18.0, 20.0,
    22.0, 22.5, 23.0, 23.5, 24.0, 26.0,
    28.0, 28.5, 29.0, 29.5, 30.0,
)

SYNTHETIC_REFINE_3_SNRS: Tuple[float, ...] = (
    16.75, 22.75, 28.25,
)

MODES = [
    {"mode_id": 0, "modulation": "BPSK", "bits_per_symbol": 1},
    {"mode_id": 1, "modulation": "QPSK", "bits_per_symbol": 2},
    {"mode_id": 2, "modulation": "16QAM", "bits_per_symbol": 4},
    {"mode_id": 3, "modulation": "64QAM", "bits_per_symbol": 6},
]


def create_synthetic_calibration_csv(
    output_path: Path,
    snrs: Sequence[float],
    num_blocks: int = 140000,
) -> None:
    """Generate synthetic calibration table with realistic physical transitions."""
    fieldnames = [
        "mode_id", "modulation", "bits_per_symbol", "snr_db", "num_blocks",
        "total_bits", "bit_errors", "ber", "block_errors", "bler",
        "mean_block_ber", "std_block_ber", "se_block_ber", "z_ab",
        "fixed_budget_reached", "is_stable", "reliability_uncertain",
    ]

    rows = []
    for snr in snrs:
        for m in MODES:
            mid = m["mode_id"]
            mod = m["modulation"]
            bps = m["bits_per_symbol"]

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
                "is_stable": "N/A",
                "reliability_uncertain": False,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestPackageR0Freeze(unittest.TestCase):
    """Test suite for R0 freeze packaging pipeline."""

    def test_01_merge_25_plus_3_to_28_points(self):
        """Verify merging Deep 25 SNRs and Refine 3 SNRs produces exactly 28 unique SNRs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            deep_csv = tmp_p / "deep" / "calibration_1d_cuda_pooled.csv"
            refine_csv = tmp_p / "refine" / "calibration_1d_cuda_pooled.csv"

            create_synthetic_calibration_csv(deep_csv, SYNTHETIC_DEEP_25_SNRS)
            create_synthetic_calibration_csv(refine_csv, SYNTHETIC_REFINE_3_SNRS)

            raw_rows, merged_cal, audit = load_and_merge_r0_calibration(deep_csv, refine_csv)

            self.assertEqual(len(merged_cal), 28)
            self.assertEqual(len(raw_rows), 28 * 4)
            self.assertEqual(audit["deep_point_count"], 25)
            self.assertEqual(audit["refine_point_count"], 3)
            self.assertEqual(audit["merged_point_count"], 28)
            for snr in SYNTHETIC_REFINE_3_SNRS:
                self.assertIn(snr, merged_cal)

    def test_02_duplicate_protection(self):
        """Verify duplicate SNRs across files are detected and reconciled cleanly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            deep_csv = tmp_p / "deep" / "calibration_1d_cuda_pooled.csv"
            refine_csv = tmp_p / "refine" / "calibration_1d_cuda_pooled.csv"

            # Overlap at 16.0 dB
            overlap_refine = (16.0, 16.75, 22.75, 28.25)
            create_synthetic_calibration_csv(deep_csv, SYNTHETIC_DEEP_25_SNRS)
            create_synthetic_calibration_csv(refine_csv, overlap_refine)

            raw_rows, merged_cal, audit = load_and_merge_r0_calibration(deep_csv, refine_csv)
            # 25 + 4 - 1 = 28 unique SNRs
            self.assertEqual(len(merged_cal), 28)
            self.assertEqual(len(raw_rows), 28 * 4)

    def test_03_ci_ground_truth_computation(self):
        """Verify canonical conservative CI ground-truth synthesis across 28 points."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            deep_csv = tmp_p / "deep" / "calibration_1d_cuda_pooled.csv"
            refine_csv = tmp_p / "refine" / "calibration_1d_cuda_pooled.csv"

            create_synthetic_calibration_csv(deep_csv, SYNTHETIC_DEEP_25_SNRS)
            create_synthetic_calibration_csv(refine_csv, SYNTHETIC_REFINE_3_SNRS)

            _, merged_cal, _ = load_and_merge_r0_calibration(deep_csv, refine_csv)
            gt_cfg = GroundTruthConfig()
            gt_rows = compute_ground_truth(merged_cal, gt_cfg)

            self.assertEqual(len(gt_rows), 28)
            modes_seen = {r.best_mode for r in gt_rows}
            self.assertIn("BPSK", modes_seen)
            self.assertIn("64QAM", modes_seen)
            for r in gt_rows:
                self.assertGreater(r.best_mode_bps, 0)
                self.assertGreaterEqual(r.bpsk_ci_high, r.bpsk_ber)

    def test_04_cart_depth_sweep_and_selection(self):
        """Verify depth sweep 1..5 selects smallest depth achieving 100% fidelity."""
        snrs = [float(i) for i in range(10, 31)]
        # Perfectly separable step function
        labels = ["BPSK"] * 6 + ["QPSK"] * 6 + ["16QAM"] * 5 + ["64QAM"] * 4

        chosen_clf, metrics, sweep_rows, achieved_100 = sweep_and_select_cart_depth(snrs, labels)

        self.assertEqual(len(sweep_rows), 5)
        self.assertTrue(achieved_100)
        self.assertEqual(metrics.accuracy, 1.0)
        self.assertIn(metrics.max_depth_param, range(1, 6))
        self.assertLessEqual(metrics.actual_depth, metrics.max_depth_param)

    def test_05_policy_and_baseline_evaluation(self):
        """Verify policy evaluation generates per-SNR rows and valid sampled-grid averages."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            deep_csv = tmp_p / "deep" / "calibration_1d_cuda_pooled.csv"
            refine_csv = tmp_p / "refine" / "calibration_1d_cuda_pooled.csv"

            create_synthetic_calibration_csv(deep_csv, SYNTHETIC_DEEP_25_SNRS)
            create_synthetic_calibration_csv(refine_csv, SYNTHETIC_REFINE_3_SNRS)

            _, merged_cal, _ = load_and_merge_r0_calibration(deep_csv, refine_csv)
            gt_cfg = GroundTruthConfig()
            gt_rows = compute_ground_truth(merged_cal, gt_cfg)

            snrs = [r.snr_db for r in gt_rows]
            labels = [r.best_mode for r in gt_rows]
            chosen_clf, _, _, _ = sweep_and_select_cart_depth(snrs, labels)

            per_snr_rows, summary = evaluate_policies_and_baselines(gt_rows, chosen_clf, merged_cal)

            self.assertEqual(len(per_snr_rows), 28)
            means = summary["sampled_grid_mean_spectral_efficiency"]
            self.assertAlmostEqual(means["fixed_robust_bpsk"], 1.0, places=3)
            self.assertAlmostEqual(means["fixed_high_tp_64qam"], 6.0, places=3)
            self.assertGreaterEqual(means["ground_truth_lut"], 1.0)
            self.assertLessEqual(means["ground_truth_lut"], 6.0)

    def test_06_freeze_manifest_and_sha256_creation(self):
        """Verify SHA-256 computation and git commit extraction."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = Path(tmp_dir) / "test.txt"
            test_file.write_text("reproducible_sha256_test", encoding="utf-8")

            sha = compute_file_sha256(test_file)
            self.assertEqual(len(sha), 64)
            self.assertNotEqual(sha, "FILE_NOT_FOUND")

            commit = get_git_commit()
            self.assertIsInstance(commit, str)
            self.assertGreater(len(commit), 0)

    def test_07_freeze_gate_data_driven_pass_and_review(self):
        """Verify freeze gate yields PASS for valid data and REVIEW when regressions are injected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            deep_csv = tmp_p / "deep" / "calibration_1d_cuda_pooled.csv"
            refine_csv = tmp_p / "refine" / "calibration_1d_cuda_pooled.csv"

            create_synthetic_calibration_csv(deep_csv, SYNTHETIC_DEEP_25_SNRS)
            create_synthetic_calibration_csv(refine_csv, SYNTHETIC_REFINE_3_SNRS)

            _, merged_cal, _ = load_and_merge_r0_calibration(deep_csv, refine_csv)
            gt_cfg = GroundTruthConfig()
            gt_rows = compute_ground_truth(merged_cal, gt_cfg)
            snrs = [r.snr_db for r in gt_rows]
            labels = [r.best_mode for r in gt_rows]
            _, dt_metrics, _, _ = sweep_and_select_cart_depth(snrs, labels)

            # Case A: Standard Monotonic Data -> PASS
            pass_audit = run_freeze_gate(
                gt_rows=gt_rows,
                dt_metrics=dt_metrics,
                merged_cal=merged_cal,
                expected_snrs=snrs,
            )
            self.assertEqual(pass_audit["status"], "PASS")
            self.assertTrue(pass_audit["monotonicity_passed"])
            self.assertTrue(pass_audit["dt_fidelity_passed"])

            # Case B: Injected Non-Monotonic Regression -> REVIEW
            corrupted_gt = list(gt_rows)
            # Invert highest SNR label to BPSK (bps=1) after a higher mode
            last_r = corrupted_gt[-1]
            corrupted_last = last_r.__class__(
                snr_db=last_r.snr_db,
                bpsk_ber=last_r.bpsk_ber,
                qpsk_ber=last_r.qpsk_ber,
                qam16_ber=last_r.qam16_ber,
                qam64_ber=last_r.qam64_ber,
                bpsk_se=last_r.bpsk_se,
                qpsk_se=last_r.qpsk_se,
                qam16_se=last_r.qam16_se,
                qam64_se=last_r.qam64_se,
                bpsk_eligible=True,
                qpsk_eligible=False,
                qam16_eligible=False,
                qam64_eligible=False,
                best_mode="BPSK",
                best_mode_bps=1,
                fallback_used=False,
                selection_reason="injected_regression",
                reliability_uncertain=False,
                label_uncertain=False,
                boundary_uncertain=False,
            )
            corrupted_gt[-1] = corrupted_last

            review_audit = run_freeze_gate(
                gt_rows=corrupted_gt,
                dt_metrics=dt_metrics,
                merged_cal=merged_cal,
                expected_snrs=snrs,
            )
            self.assertEqual(review_audit["status"], "REVIEW")
            self.assertFalse(review_audit["monotonicity_passed"])
            self.assertGreater(len(review_audit["review_reasons"]), 0)

    def test_08_post_processing_isolation_no_phy_imports(self):
        """Verify package_r0_freeze does not import Sionna, BatchPHYEngine, or cuda_engine."""
        code = (
            "import package_r0_freeze, sys\n"
            "assert 'sionna' not in sys.modules, 'sionna illegally loaded'\n"
            "assert 'calibration_l2_cuda' not in sys.modules, 'calibration_l2_cuda illegally loaded'\n"
            "assert 'phy_engine' not in sys.modules, 'phy_engine illegally loaded'\n"
            "assert 'cuda_engine' not in sys.modules, 'cuda_engine illegally loaded'\n"
            "print('ISOLATION_OK')"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Isolation check failed:\n{res.stderr}")
        self.assertIn("ISOLATION_OK", res.stdout)

    def test_09_end_to_end_package_generation(self):
        """Verify package_r0_freeze generates all 8 required artifacts with valid schema."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            deep_csv = tmp_p / "deep" / "calibration_1d_cuda_pooled.csv"
            refine_csv = tmp_p / "refine" / "calibration_1d_cuda_pooled.csv"
            out_dir = tmp_p / "r0_final"

            create_synthetic_calibration_csv(deep_csv, SYNTHETIC_DEEP_25_SNRS)
            create_synthetic_calibration_csv(refine_csv, SYNTHETIC_REFINE_3_SNRS)

            res = package_r0_freeze(
                deep_csv=deep_csv,
                refine_csv=refine_csv,
                output_dir=out_dir,
            )

            expected_artifacts = [
                "r0_calibration_merged.csv",
                "r0_ground_truth.csv",
                "r0_cart_predictions.csv",
                "r0_cart_depth_sweep.csv",
                "r0_policy_evaluation.csv",
                "r0_freeze_manifest.json",
                "r0_final_report.md",
                "r0_cart_model.joblib",
            ]

            for fname in expected_artifacts:
                fpath = out_dir / fname
                self.assertTrue(fpath.exists(), f"Missing expected artifact: {fname}")
                self.assertGreater(fpath.stat().st_size, 0, f"Empty artifact: {fname}")

            # Verify manifest schema
            manifest = json.loads((out_dir / "r0_freeze_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("source_git_commit", manifest)
            self.assertIn("inputs", manifest)
            self.assertIn("scientific_specification", manifest)
            self.assertIn("cart_decision_tree", manifest)
            self.assertIn("freeze_status", manifest)
            self.assertEqual(manifest["final_dataset"]["num_operating_points"], 28)

            # Verify serialized model functionality
            loaded_clf = joblib.load(out_dir / "r0_cart_model.joblib")
            test_preds = loaded_clf.predict([10.0, 20.0, 30.0])
            self.assertEqual(len(test_preds), 3)


if __name__ == "__main__":
    unittest.main()
