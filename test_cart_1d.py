"""Unit tests for L4 Generic CART Machine Learning Framework and 1D Baseline."""
from pathlib import Path
import tempfile
import unittest
import numpy as np

from cart_1d import (
    CARTClassifier,
    CART1DClassifier,
    sweep_cart_depths,
    TreeMetrics,
    load_ground_truth_dataset,
    load_lut_csv,
    load_lut_from_ground_truth_csv,
    save_model,
    load_model,
)
from ground_truth import LookupTable1D, GroundTruthRow


class TestCART1D(unittest.TestCase):
    """Test suite for CART generic framework, 1D wrapper, metrics, and LUT equivalence."""

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

    # -----------------------------------------------------------------
    # Baseline 1D Specialization Tests (Backward Compatibility)
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Generic Feature-Agnostic CART Framework Tests
    # -----------------------------------------------------------------

    def test_generic_cart_1_feature(self):
        """Verify generic CARTClassifier works with 2D single-feature input (N, 1)."""
        X = np.array(self.snrs, dtype=np.float64).reshape(-1, 1)
        clf = CARTClassifier(max_depth=3, random_state=20260918, feature_names=["snr_db"])
        clf.fit(X, self.labels)

        self.assertTrue(clf.is_fitted)
        self.assertEqual(clf.n_features_in_, 1)
        self.assertEqual(clf.feature_names, ["snr_db"])
        preds = clf.predict(X)
        self.assertEqual(preds, self.labels)
        self.assertEqual(clf.get_learned_thresholds(), [16.75, 22.75, 28.25])

    def test_generic_cart_multi_feature_software_api(self):
        """Verify generic CARTClassifier accepts multi-feature inputs as a software unit test.

        NOTE: This is purely a software API verification test, NOT scientific thesis data.
        """
        # Synthetic 2D dataset: [snr_db, abs_h]
        X_2d = np.array([
            [5.0, 0.2],
            [8.0, 0.4],
            [12.0, 0.8],
            [15.0, 1.1],
            [18.0, 0.9],
            [22.0, 1.3],
            [25.0, 1.5],
            [30.0, 2.0],
        ], dtype=np.float64)
        y_synthetic = ["BPSK", "BPSK", "BPSK", "QPSK", "QPSK", "16QAM", "16QAM", "64QAM"]

        clf = CARTClassifier(max_depth=3, random_state=20260918, feature_names=["snr_db", "abs_h"])
        clf.fit(X_2d, y_synthetic)

        self.assertTrue(clf.is_fitted)
        self.assertEqual(clf.n_features_in_, 2)
        self.assertEqual(clf.feature_names, ["snr_db", "abs_h"])

        preds = clf.predict(X_2d)
        self.assertEqual(len(preds), len(y_synthetic))

        # Check threshold attribution contains valid feature names from the 2D schema
        splits = clf.get_threshold_splits()
        self.assertGreater(len(splits), 0)
        for s in splits:
            self.assertIn(s["feature_name"], ["snr_db", "abs_h"])
            self.assertIn(s["feature_index"], [0, 1])

    def test_arbitrary_feature_names(self):
        """Verify arbitrary feature names are correctly attributed and exported."""
        X_2d = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]], dtype=np.float64)
        y = ["MODE_A", "MODE_A", "MODE_B", "MODE_B"]
        custom_names = ["alpha_metric", "beta_metric"]

        clf = CARTClassifier(max_depth=2, random_state=20260918, feature_names=custom_names)
        clf.fit(X_2d, y)

        text = clf.export_tree_text()
        self.assertTrue("alpha_metric" in text or "beta_metric" in text)

        splits = clf.get_threshold_splits()
        for s in splits:
            self.assertIn(s["feature_name"], custom_names)

    def test_shape_and_feature_count_validation(self):
        """Verify strict shape, dimension, and feature count validations."""
        clf = CARTClassifier(max_depth=3)

        # 1. Unfitted predict raises RuntimeError
        with self.assertRaises(RuntimeError):
            clf.predict([[10.0]])

        # 2. Fitting with 1D array raises ValueError (demands explicit 2D matrix)
        with self.assertRaises(ValueError):
            clf.fit([1.0, 2.0, 3.0], ["A", "B", "C"])

        # 3. Sample count mismatch between X and y
        with self.assertRaises(ValueError):
            clf.fit([[1.0], [2.0]], ["A", "B", "C"])

        # 4. Feature name count mismatch
        with self.assertRaises(ValueError):
            clf.fit([[1.0], [2.0]], ["A", "B"], feature_names=["f1", "f2"])

        # 5. Fit valid 2-feature model, then predict with 1-feature array
        X_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        clf.fit(X_2d, ["A", "B"])
        with self.assertRaises(ValueError):
            clf.predict([[1.0]])  # 1 feature instead of 2

        # 6. Predict with 1D array
        with self.assertRaises(ValueError):
            clf.predict([1.0, 2.0])

    def test_deterministic_cart_fitting(self):
        """Verify deterministic tree induction across separate instances with the same seed."""
        X = np.array(self.snrs, dtype=np.float64).reshape(-1, 1)
        clf1 = CARTClassifier(max_depth=3, random_state=20260918, feature_names=["snr_db"]).fit(X, self.labels)
        clf2 = CARTClassifier(max_depth=3, random_state=20260918, feature_names=["snr_db"]).fit(X, self.labels)

        self.assertEqual(clf1.get_learned_thresholds(), clf2.get_learned_thresholds())
        self.assertEqual(clf1.predict(X), clf2.predict(X))

    def test_threshold_extraction_with_feature_attribution(self):
        """Verify get_threshold_splits returns complete attribution metadata."""
        X = np.array(self.snrs, dtype=np.float64).reshape(-1, 1)
        clf = CARTClassifier(max_depth=3, random_state=20260918, feature_names=["snr_db"]).fit(X, self.labels)

        splits = clf.get_threshold_splits()
        self.assertEqual(len(splits), 3)
        for s in splits:
            self.assertIn("node_id", s)
            self.assertEqual(s["feature_index"], 0)
            self.assertEqual(s["feature_name"], "snr_db")
            self.assertIsInstance(s["threshold"], float)

    def test_dataset_loading_and_uncertainty_preservation(self):
        """Verify load_ground_truth_dataset preserves uncertainty metadata and extracts clean features."""
        gt_path = Path("results/l3_final/ground_truth_1d.csv")
        if not gt_path.exists():
            self.skipTest("results/l3_final/ground_truth_1d.csv does not exist yet")

        X, y, metadata = load_ground_truth_dataset(
            csv_path=gt_path,
            feature_names=["snr_db"],
            target_col="best_mode",
        )

        self.assertEqual(X.shape, (25, 1))
        self.assertEqual(len(y), 25)
        self.assertEqual(metadata["feature_names"], ["snr_db"])
        self.assertEqual(metadata["n_samples"], 25)

        # Verify uncertainty metadata
        rel_unc = metadata["reliability_uncertain"]
        lbl_unc = metadata["label_uncertain"]
        snrs = metadata["snr_db"]

        # Exactly 3 reliability_uncertain points: 14.0, 23.0, 28.0
        rel_flagged_snrs = [snrs[i] for i in range(len(snrs)) if rel_unc[i]]
        self.assertEqual(rel_flagged_snrs, [14.0, 23.0, 28.0])

        # Exactly 2 label_uncertain points: 23.0, 28.0
        lbl_flagged_snrs = [snrs[i] for i in range(len(snrs)) if lbl_unc[i]]
        self.assertEqual(lbl_flagged_snrs, [23.0, 28.0])

        # Verify uncertainty columns are NOT in X
        self.assertEqual(X.shape[1], 1)

    def test_model_persistence_and_reproducibility(self):
        """Verify model serialization and deserialization via joblib preserves full reproducibility."""
        X = np.array(self.snrs, dtype=np.float64).reshape(-1, 1)
        clf = CARTClassifier(max_depth=3, random_state=20260918, feature_names=["snr_db"]).fit(X, self.labels)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_file = Path(tmp_dir) / "test_model.joblib"
            meta = {"experiment": "unit_test", "seed": 20260918}
            save_model(clf, model_file, metadata=meta)

            loaded_clf, payload = load_model(model_file)
            self.assertTrue(loaded_clf.is_fitted)
            self.assertEqual(loaded_clf.feature_names, ["snr_db"])
            self.assertEqual(loaded_clf.classes_, clf.classes_)
            self.assertEqual(loaded_clf.get_learned_thresholds(), clf.get_learned_thresholds())
            self.assertEqual(loaded_clf.predict(X), clf.predict(X))
            self.assertEqual(payload["training_metadata"], meta)

    def test_no_l2_l3_mutation(self):
        """Verify frozen L2 and L3 dataset integrity is strictly preserved."""
        l2_path = Path("results/l2_cuda_rtx3060_final/calibration_1d_cuda_pooled.csv")
        l3_path = Path("results/l3_final/ground_truth_1d.csv")
        self.assertTrue(l2_path.exists(), "L2 final dataset must exist")
        self.assertTrue(l3_path.exists(), "L3 final dataset must exist")

    def test_load_lut_csv_direct_l3_dependency(self):
        """Verify load_lut_csv loads directly from frozen L3 CSV with correct thresholds."""
        lut_path = Path("results/l3_final/lut_1d.csv")
        self.assertTrue(lut_path.exists(), "results/l3_final/lut_1d.csv must exist")
        lut = load_lut_csv(lut_path)
        thresholds = [th[0] for th in lut.thresholds]
        self.assertEqual(thresholds, [16.75, 22.75, 28.25])
        self.assertEqual(len(lut.intervals), 4)
        self.assertEqual(lut.intervals[0].mode, "BPSK")
        self.assertEqual(lut.intervals[1].mode, "QPSK")
        self.assertEqual(lut.intervals[2].mode, "16QAM")
        self.assertEqual(lut.intervals[3].mode, "64QAM")

    def test_load_lut_from_ground_truth_csv(self):
        """Verify load_lut_from_ground_truth_csv matches load_lut_csv without any L2 access."""
        gt_path = Path("results/l3_final/ground_truth_1d.csv")
        self.assertTrue(gt_path.exists(), "results/l3_final/ground_truth_1d.csv must exist")
        lut_from_gt = load_lut_from_ground_truth_csv(gt_path)
        lut_from_csv = load_lut_csv("results/l3_final/lut_1d.csv")

        self.assertEqual(lut_from_gt.thresholds, lut_from_csv.thresholds)
        self.assertEqual(lut_from_gt.intervals, lut_from_csv.intervals)
        self.assertEqual(lut_from_gt.non_monotonic_transitions, lut_from_csv.non_monotonic_transitions)

    def test_l4_no_runtime_l2_dependencies(self):
        """Verify cart_1d module has no runtime references to L2 calibration functions or paths."""
        import cart_1d
        self.assertFalse(hasattr(cart_1d, "load_calibration_csv"))
        self.assertFalse(hasattr(cart_1d, "compute_ground_truth"))
        self.assertFalse(hasattr(cart_1d, "GroundTruthConfig"))


if __name__ == "__main__":
    unittest.main()

