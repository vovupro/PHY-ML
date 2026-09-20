"""Tests for R1 Decision Tree (CART) model selection pipeline."""
import unittest

from r1_cart import R1CARTClassifier, sweep_and_select_r1_cart_depth


class TestR1CART(unittest.TestCase):
    """Test suite for R1 1D CART link adaptation pipeline."""

    def test_01_multiclass_coded_action_training(self):
        """Verify CART trains and predicts across multiclass discrete coded actions."""
        # 4 distinct action regions
        snrs = [float(i) for i in range(10, 31)]
        labels = (
            ["BPSK-1/2"] * 5 +
            ["QPSK-1/2"] * 5 +
            ["16QAM-1/2"] * 5 +
            ["64QAM-3/4"] * 6
        )

        clf = R1CARTClassifier(max_depth=3)
        clf.fit(snrs, labels)

        self.assertTrue(clf.is_fitted)
        self.assertEqual(len(clf.classes_), 4)

        preds = clf.predict(snrs)
        self.assertEqual(preds, labels)
        self.assertEqual(clf.score(snrs, labels), 1.0)

        metrics = clf.get_metrics(snrs, labels)
        self.assertEqual(metrics.accuracy, 1.0)
        self.assertEqual(len(metrics.learned_thresholds), 3)

    def test_02_depth_sweep_model_selection(self):
        """Verify sweep selects smallest depth achieving 100% fidelity without hard-coding depth."""
        snrs = [float(i) for i in range(10, 26)]
        # 3 regions: separable by depth 2
        labels = ["BPSK-1/2"] * 5 + ["QPSK-1/2"] * 5 + ["16QAM-1/2"] * 6

        chosen_clf, chosen_metrics, sweep_rows, achieved = sweep_and_select_r1_cart_depth(
            snrs=snrs,
            labels=labels,
            max_depth_range=(1, 2, 3, 4, 5),
            target_fidelity=1.0,
        )

        self.assertTrue(achieved)
        self.assertEqual(len(sweep_rows), 5)
        # Depth 1 cannot separate 3 classes (accuracy < 1.0)
        self.assertLess(sweep_rows[0]["accuracy"], 1.0)
        # Depth 2 can separate 3 classes (accuracy = 1.0)
        self.assertEqual(sweep_rows[1]["accuracy"], 1.0)
        # Model selection chooses depth 2 as the smallest qualifying depth
        self.assertEqual(chosen_metrics.max_depth_param, 2)
        self.assertEqual(chosen_metrics.accuracy, 1.0)

    def test_03_fallback_when_target_fidelity_unachievable(self):
        """Verify fallback selects maximum accuracy when 100% cannot be achieved within depth bound."""
        # Non-separable noisy labels (identical SNR has conflicting labels)
        snrs = [10.0, 10.0, 20.0, 20.0]
        labels = ["BPSK-1/2", "QPSK-1/2", "16QAM-1/2", "64QAM-3/4"]

        chosen_clf, chosen_metrics, _, achieved = sweep_and_select_r1_cart_depth(
            snrs=snrs,
            labels=labels,
            max_depth_range=(1, 2),
            target_fidelity=1.0,
        )

        self.assertFalse(achieved)
        self.assertLess(chosen_metrics.accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
