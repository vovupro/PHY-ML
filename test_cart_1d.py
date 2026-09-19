"""Unit tests for L4 CART 1D Classifier."""
import unittest
import numpy as np

from cart_1d import (
    CART1DClassifier,
    sweep_cart_depths,
    TreeMetrics,
)
from ground_truth import LookupTable1D, GroundTruthRow


class TestCART1D(unittest.TestCase):
    """Test suite for CART 1D classifier induction, metrics, and LUT equivalence."""

    def setUp(self):
        # 25-point refined calibration grid pattern
        self.snrs = [
            0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 16.5,
            17.0, 17.5, 18.0, 20.0, 22.0, 22.5,
            23.0, 23.5, 24.0, 26.0, 28.0,
            28.5, 29.0, 29.5, 30.0,
        ]
        self.labels = [
            # 0 to 16.5 dB: BPSK (10 points)
            "BPSK", "BPSK", "BPSK", "BPSK", "BPSK", "BPSK", "BPSK", "BPSK", "BPSK", "BPSK",
            # 17.0 to 22.5 dB: QPSK (6 points)
            "QPSK", "QPSK", "QPSK", "QPSK", "QPSK", "QPSK",
            # 23.0 to 28.0 dB: 16QAM (5 points)
            "16QAM", "16QAM", "16QAM", "16QAM", "16QAM",
            # 28.5 to 30.0 dB: 64QAM (4 points)
            "64QAM", "64QAM", "64QAM", "64QAM",
        ]

    def test_cart_fit_and_predict_scalar_and_array(self):
        """Verify CART1DClassifier fits and predicts scalar and array inputs."""
        clf = CART1DClassifier(max_depth=3, random_state=20260918)
        clf.fit(self.snrs, self.labels)

        self.assertTrue(clf.is_fitted)
        self.assertEqual(clf.predict(10.0), "BPSK")
        self.assertEqual(clf.predict(20.0), "QPSK")
        self.assertEqual(clf.predict(25.0), "16QAM")
        self.assertEqual(clf.predict(29.0), "64QAM")

        # Array prediction
        preds = clf.predict([5.0, 19.0, 24.0, 29.0])
        self.assertEqual(preds, ["BPSK", "QPSK", "16QAM", "64QAM"])

    def test_cart_depth_sweep_and_accuracy(self):
        """Verify depth sweep progression from underfitting to 100% training accuracy."""
        results = sweep_cart_depths(self.snrs, self.labels, depths=(1, 2, 3, 4, 5))

        # Depth 1 can only split once -> 2 classes max -> accuracy < 1.0
        self.assertLess(results[1][1].accuracy, 1.0)

        # Depth 3 has sufficient splits for 4 classes -> 100% accuracy
        self.assertEqual(results[3][1].accuracy, 1.0)
        self.assertEqual(results[4][1].accuracy, 1.0)
        self.assertEqual(results[5][1].accuracy, 1.0)

    def test_cart_exact_lut_threshold_equivalence(self):
        """Verify learned CART split thresholds match exact LUT midpoints at 16.75, 22.75, 28.25 dB."""
        clf = CART1DClassifier(max_depth=3, random_state=20260918)
        clf.fit(self.snrs, self.labels)
        thresholds = [round(t, 2) for t in clf.get_learned_thresholds()]

        expected_thresholds = [16.75, 22.75, 28.25]
        self.assertEqual(thresholds, expected_thresholds)

    def test_tree_export_text(self):
        """Verify exported tree text contains expected feature name and threshold tokens."""
        clf = CART1DClassifier(max_depth=3, random_state=20260918)
        clf.fit(self.snrs, self.labels)
        text = clf.export_tree_text()

        self.assertIn("SNR_dB", text)
        self.assertIn("16.75", text)
        self.assertIn("22.75", text)
        self.assertIn("28.25", text)


if __name__ == "__main__":
    unittest.main()
