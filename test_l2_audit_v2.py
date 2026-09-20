"""Focused acceptance tests for L2 audit repairs.

Fulfills Acceptance Criteria:
A. CI overlaps 0.01 => uncertain = true
B. BER may be numerically near 0.01, but if the whole CI is below or above 0.01: uncertain = false
C. CUDA FP64 and FP32 benchmark batch choices are independent
D. Report generation cannot silently use a hard-coded CPU throughput
E. Exact-realization parity is VERIFIED or explicitly NOT VERIFIED
"""
import io
from pathlib import Path
import tempfile
import unittest
import torch

from metrics import is_reliability_uncertain, BER_TARGET, CONFIDENCE_K
from benchmark_and_tune import run_fair_performance_benchmark, run_exact_realization_parity
from calibration_l2_cuda import generate_markdown_report
from config import ExecutionConfig, probe_environment


class TestL2AuditRepairs(unittest.TestCase):
    """Test suite verifying all 5 audit repair issues."""

    def test_a_ci_overlaps_0_01_uncertain_true(self):
        """A. CI overlaps 0.01 => uncertain = true."""
        # Case 1: BER slightly below 0.01, but CI extends above 0.01
        ber = 0.0098
        se = 0.0002  # CI = [0.0098 - 0.000392, 0.0098 + 0.000392] = [0.009408, 0.010192]
        self.assertTrue(
            is_reliability_uncertain(ber, se),
            "CI straddles 0.01 from below -> uncertain must be True"
        )

        # Case 2: BER slightly above 0.01, but CI extends below 0.01
        ber = 0.0102
        se = 0.0002  # CI = [0.009808, 0.010592]
        self.assertTrue(
            is_reliability_uncertain(ber, se),
            "CI straddles 0.01 from above -> uncertain must be True"
        )

        # Case 3: BER exactly at 0.01
        ber = 0.0100
        se = 0.0001
        self.assertTrue(
            is_reliability_uncertain(ber, se),
            "BER at target with nonzero SE -> uncertain must be True"
        )

    def test_b_ber_near_0_01_ci_outside_uncertain_false(self):
        """B. BER may be numerically near 0.01, but if the whole CI is below or above 0.01: uncertain = false."""
        # Case 1: BER = 0.0095 is in old heuristic [0.009, 0.011], but tight SE = 0.0001 puts CI in [0.009304, 0.009696]
        # Since upper bound 0.009696 < 0.0100, uncertain must be False!
        ber_below = 0.0095
        se = 0.0001
        self.assertFalse(
            is_reliability_uncertain(ber_below, se),
            "Whole CI strictly below 0.01 despite being in [0.009, 0.011] -> uncertain must be False"
        )

        # Case 2: BER = 0.0105 is in old heuristic [0.009, 0.011], but tight SE = 0.0001 puts CI in [0.010304, 0.010696]
        # Since lower bound 0.010304 > 0.0100, uncertain must be False!
        ber_above = 0.0105
        self.assertFalse(
            is_reliability_uncertain(ber_above, se),
            "Whole CI strictly above 0.01 despite being in [0.009, 0.011] -> uncertain must be False"
        )

        # Case 3: In old heuristic window [0.008, 0.012] at 0.0085 with SE 0.0001
        self.assertFalse(
            is_reliability_uncertain(0.0085, 0.0001),
            "BER in [0.008, 0.012] with CI below 0.01 -> uncertain must be False"
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for batch independence test")
    def test_c_cuda_fp64_and_fp32_batches_are_independent(self):
        """C. CUDA FP64 and FP32 benchmark batch choices are independent."""
        batch_fp64 = 1000
        batch_fp32 = 2000

        res = run_fair_performance_benchmark(
            best_cpu_threads=8,
            best_gpu_batch_double=batch_fp64,
            best_gpu_batch_single=batch_fp32,
            snrs=(18.0,),
            blocks_per_snr=100,
            validate_single_passed=True,
        )

        self.assertIn("CUDA Double", res)
        self.assertIn("CUDA Single", res)
        self.assertEqual(res["CUDA Double"]["batch_size"], batch_fp64)
        self.assertEqual(res["CUDA Single"]["batch_size"], batch_fp32)
        self.assertNotEqual(
            res["CUDA Double"]["batch_size"],
            res["CUDA Single"]["batch_size"],
            "CUDA Double and CUDA Single must have independent batch configurations"
        )

    def test_d_report_generation_cannot_silently_use_hardcoded_cpu_throughput(self):
        """D. Report generation cannot silently use a hard-coded CPU throughput."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "test_report.md"
            cfg = ExecutionConfig(backend="cuda" if torch.cuda.is_available() else "cpu")
            env = probe_environment(cfg)

            # Minimal snr_results for testing
            snr_results = {
                18.0: {
                    "final_pooled_ber": {0: 0.004, 1: 0.008, 2: 0.03, 3: 0.08},
                    "final_pooled_se": {0: 0.0003, 1: 0.0005, 2: 0.001, 3: 0.002},
                    "final_blocks_a": 500,
                    "final_blocks_b": 500,
                    "checkpoints_count": 2,
                    "stable": True,
                    "final_stats_a": {m: type("Stats", (), {"ber": 0.01})() for m in range(4)},
                    "final_stats_b": {m: type("Stats", (), {"ber": 0.01})() for m in range(4)},
                    "final_z_ab": {m: 0.5 for m in range(4)},
                }
            }

            # 1. Generate report WITHOUT persisted benchmark
            generate_markdown_report(
                snr_results=snr_results,
                env=env,
                config=cfg,
                warmup_time=0.1,
                total_mc_time=10.0,
                total_blocks_simulated=5000,
                output_path=out_path,
                persisted_benchmark=None,
            )

            report_content = out_path.read_text(encoding="utf-8")

            # Check 1: Peak Throughput is forbidden
            self.assertNotIn("Peak Throughput", report_content)
            self.assertIn("Overall Average Throughput", report_content)

            # Check 2: Hard-coded 52.0 blocks/s is forbidden
            self.assertNotIn("52.0", report_content)
            self.assertIn("N/A", report_content)

            # 2. Generate report WITH explicit persisted benchmark
            mock_bench = {
                "Reference CPU Double": {
                    "blocks_per_sec": 80.0,
                }
            }
            generate_markdown_report(
                snr_results=snr_results,
                env=env,
                config=cfg,
                warmup_time=0.1,
                total_mc_time=10.0,
                total_blocks_simulated=5000,
                output_path=out_path,
                persisted_benchmark=mock_bench,
            )
            report_content_bench = out_path.read_text(encoding="utf-8")
            self.assertIn("80.0 blocks/s", report_content_bench)
            self.assertNotIn("52.0", report_content_bench)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for exact-realization parity test")
    def test_e_exact_realization_parity_verified_or_explicitly_not_verified(self):
        """E. Exact-realization parity is VERIFIED or explicitly NOT VERIFIED."""
        parity_res = run_exact_realization_parity(snrs=(18.0,), num_blocks=100)

        self.assertIn("cuda_fp64_exact_parity", parity_res)
        self.assertIn("cuda_fp32_exact_parity", parity_res)

        # FP64 must be VERIFIED with 0 bit discrepancies
        self.assertEqual(parity_res["cuda_fp64_exact_parity"], "VERIFIED")

        # FP32 must be either VERIFIED (if all bits match on sample) or explicitly NOT VERIFIED (if any discrepancy)
        fp32_status = parity_res["cuda_fp32_exact_parity"]
        self.assertTrue(
            fp32_status == "VERIFIED" or fp32_status.startswith("NOT VERIFIED"),
            f"FP32 parity status must be VERIFIED or explicitly NOT VERIFIED, got: {fp32_status}"
        )


if __name__ == "__main__":
    unittest.main()
