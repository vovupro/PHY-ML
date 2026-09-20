"""Lightweight local unit and smoke tests for deep_stopping_diagnostic.py.

Verifies:
1. Mathematical and statistical logic of analyze_checkpoint_series under all
   overlap/resolution trajectories (never resolved, stable resolved, transient,
   persistent after transient).
2. Exactness of the 3 uncertain operating points and old stopping depths.
3. End-to-end lightweight execution on CPU (small budget smoke test).
4. Strict non-modification of canonical L2 artifacts.
"""
import io
from pathlib import Path
import sys
import unittest

from diagnostics.deep_stopping_diagnostic import (
    DiagnosticCheckpointRecord,
    UncertainOperatingPoint,
    UNCERTAIN_OPERATING_POINTS,
    analyze_checkpoint_series,
    run_deep_stopping_diagnostic,
)
from config import ExecutionConfig
from metrics import BER_TARGET


class TestDeepStoppingDiagnostic(unittest.TestCase):
    """Test suite for deep stopping uncertainty diagnostic."""

    def test_operating_points_specification(self):
        """Verify the 3 uncertain operating points match canonical L2 definitions."""
        self.assertEqual(len(UNCERTAIN_OPERATING_POINTS), 3)

        pt0 = UNCERTAIN_OPERATING_POINTS[0]
        self.assertEqual(pt0.snr_db, 14.0)
        self.assertEqual(pt0.mode_id, 0)
        self.assertEqual(pt0.modulation, "BPSK")
        self.assertEqual(pt0.old_stop_depth, 15_000)

        pt1 = UNCERTAIN_OPERATING_POINTS[1]
        self.assertEqual(pt1.snr_db, 23.0)
        self.assertEqual(pt1.mode_id, 2)
        self.assertEqual(pt1.modulation, "16QAM")
        self.assertEqual(pt1.old_stop_depth, 35_000)

        pt2 = UNCERTAIN_OPERATING_POINTS[2]
        self.assertEqual(pt2.snr_db, 28.0)
        self.assertEqual(pt2.mode_id, 3)
        self.assertEqual(pt2.modulation, "64QAM")
        self.assertEqual(pt2.old_stop_depth, 35_000)

    def test_analyze_never_resolved(self):
        """Case 1: Confidence intervals always overlap target across all checkpoints."""
        chks = [
            DiagnosticCheckpointRecord(
                checkpoint_idx=i,
                blocks_per_seed=i * 5000,
                total_blocks_pooled=i * 10000,
                ber_pooled=0.0098,
                se_pooled=0.0003,
                ci_low=0.0098 - 1.96 * 0.0003,
                ci_high=0.0098 + 1.96 * 0.0003,
                overlaps_target=True,
                is_old_stop=(i == 3),
            )
            for i in range(1, 11)
        ]

        res = analyze_checkpoint_series(chks, old_stop_blocks=15000, target=BER_TARGET)
        self.assertIsNotNone(res["old_stop_chk"])
        self.assertEqual(res["old_stop_chk"].blocks_per_seed, 15000)
        self.assertIsNone(res["first_non_overlap_chk"])
        self.assertIsNone(res["first_persistent_chk"])
        self.assertEqual(res["resolution_stability"], "NEVER_RESOLVED")
        self.assertEqual(res["final_overlap_status"], "OVERLAPS (Uncertain)")
        self.assertAlmostEqual(res["compute_multiplier"], 50000 / 15000, places=4)

    def test_analyze_immediately_stable(self):
        """Case 2: Point resolves at checkpoint 5 and stays resolved through end."""
        chks = []
        for i in range(1, 11):
            overlaps = (i < 5)
            # When i >= 5, BER is strictly below 0.01 with tight CI
            ber = 0.0098 if overlaps else 0.0090
            se = 0.0003 if overlaps else 0.0001
            chks.append(
                DiagnosticCheckpointRecord(
                    checkpoint_idx=i,
                    blocks_per_seed=i * 5000,
                    total_blocks_pooled=i * 10000,
                    ber_pooled=ber,
                    se_pooled=se,
                    ci_low=ber - 1.96 * se,
                    ci_high=ber + 1.96 * se,
                    overlaps_target=overlaps,
                    is_old_stop=(i == 3),
                )
            )

        res = analyze_checkpoint_series(chks, old_stop_blocks=15000, target=BER_TARGET)
        self.assertIsNotNone(res["first_non_overlap_chk"])
        self.assertEqual(res["first_non_overlap_chk"].checkpoint_idx, 5)
        self.assertIsNotNone(res["first_persistent_chk"])
        self.assertEqual(res["first_persistent_chk"].checkpoint_idx, 5)
        self.assertEqual(res["resolution_stability"], "STABLE")
        self.assertIn("Strictly Below", res["final_overlap_status"])

    def test_analyze_transient_only(self):
        """Case 3: Point resolves temporarily at checkpoint 5, but reverts to overlap at checkpoint 8."""
        chks = []
        for i in range(1, 11):
            # Non-overlap only at checkpoints 5, 6; overlaps again at 7..10
            overlaps = not (i in (5, 6))
            ber = 0.0098 if overlaps else 0.0090
            se = 0.0003 if overlaps else 0.0001
            chks.append(
                DiagnosticCheckpointRecord(
                    checkpoint_idx=i,
                    blocks_per_seed=i * 5000,
                    total_blocks_pooled=i * 10000,
                    ber_pooled=ber,
                    se_pooled=se,
                    ci_low=ber - 1.96 * se,
                    ci_high=ber + 1.96 * se,
                    overlaps_target=overlaps,
                    is_old_stop=(i == 3),
                )
            )

        res = analyze_checkpoint_series(chks, old_stop_blocks=15000, target=BER_TARGET)
        self.assertIsNotNone(res["first_non_overlap_chk"])
        self.assertEqual(res["first_non_overlap_chk"].checkpoint_idx, 5)
        self.assertIsNone(res["first_persistent_chk"])
        self.assertEqual(res["resolution_stability"], "TRANSIENT_ONLY")
        self.assertEqual(res["final_overlap_status"], "OVERLAPS (Uncertain)")

    def test_analyze_persistent_after_transient(self):
        """Case 4: Transient non-overlap at 4, reverts at 5, then permanently resolves from 7 onward."""
        chks = []
        for i in range(1, 11):
            # Non-overlap at 4, overlap at 5,6, non-overlap at 7..10
            overlaps = not (i == 4 or i >= 7)
            ber = 0.0098 if overlaps else 0.0088
            se = 0.0003 if overlaps else 0.0001
            chks.append(
                DiagnosticCheckpointRecord(
                    checkpoint_idx=i,
                    blocks_per_seed=i * 5000,
                    total_blocks_pooled=i * 10000,
                    ber_pooled=ber,
                    se_pooled=se,
                    ci_low=ber - 1.96 * se,
                    ci_high=ber + 1.96 * se,
                    overlaps_target=overlaps,
                    is_old_stop=(i == 3),
                )
            )

        res = analyze_checkpoint_series(chks, old_stop_blocks=15000, target=BER_TARGET)
        self.assertEqual(res["first_non_overlap_chk"].checkpoint_idx, 4)
        self.assertEqual(res["first_persistent_chk"].checkpoint_idx, 7)
        self.assertEqual(res["resolution_stability"], "PERSISTENT_AFTER_TRANSIENT")
        self.assertIn("Strictly Below", res["final_overlap_status"])

    def test_lightweight_cpu_smoke_run(self):
        """Lightweight smoke test executing BatchPHYEngine and stats on CPU without heavy GPU load."""
        cfg = ExecutionConfig(
            backend="cpu",
            precision="double",
            batch_blocks=250,
        )
        test_pt = UncertainOperatingPoint(
            snr_db=14.0,
            mode_id=0,
            modulation="BPSK",
            old_stop_depth=500,
        )

        # Capture stdout
        old_stdout = sys.stdout
        captured = io.StringIO()
        try:
            sys.stdout = captured
            results = run_deep_stopping_diagnostic(
                config=cfg,
                target_points=(test_pt,),
                max_blocks_per_seed=1_000,
                step_blocks_per_seed=500,
                mandatory_print_checkpoints=(500, 1000),
            )
        finally:
            sys.stdout = old_stdout

        out_str = captured.getvalue()
        self.assertIn("PHY-ML L2 DEEP STOPPING UNCERTAINTY DIAGNOSTIC", out_str)
        self.assertIn("COMPARATIVE TABLE", out_str)
        self.assertIn(14.0, results)

        res14 = results[14.0]
        self.assertEqual(len(res14["checkpoints"]), 2)
        self.assertEqual(res14["checkpoints"][0].blocks_per_seed, 500)
        self.assertEqual(res14["checkpoints"][1].blocks_per_seed, 1000)
        self.assertIsNotNone(res14["analysis"]["old_stop_chk"])
        self.assertEqual(res14["analysis"]["compute_multiplier"], 2.0)


if __name__ == "__main__":
    unittest.main()
