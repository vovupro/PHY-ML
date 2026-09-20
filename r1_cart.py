"""R1: 1D Decision Tree (CART) Model Selection Pipeline for Coded Link Adaptation.

Trains and evaluates 1D Classification Trees predicting discrete coded actions
from continuous SNR operating points:
    X = [SNR_setup]  ->  y = Coded Action (e.g. 'BPSK-1/2', 'QPSK-2/3', ...)

Model Selection Principle:
    Sweeps maximum tree depth d in [1, 2, 3, 4, 5, 6] to find the SMALLEST depth
    achieving 100% fidelity to the Ground Truth labels, maximizing policy parsimony
    and hardware implementability as a low-latency threshold comparator.

Strictly post-processing infrastructure: Evaluated only on synthetic fixtures.
Does NOT train production policies from uncalibrated pilot data.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class R1TreeMetrics:
    """Structural and accuracy metrics for an R1 Decision Tree."""
    max_depth_param: int
    actual_depth: int
    node_count: int
    leaf_count: int
    accuracy: float
    learned_thresholds: Tuple[float, ...]
    action_classes: Tuple[str, ...]


class R1CARTClassifier:
    """1D Decision Tree classifier predicting discrete coded actions from SNR."""

    def __init__(
        self,
        max_depth: Optional[int] = None,
        criterion: str = "gini",
        random_state: int = 20260920,
    ):
        self.max_depth = max_depth
        self.criterion = criterion
        self.random_state = random_state
        self._clf = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            random_state=random_state,
        )
        self.is_fitted: bool = False
        self.classes_: np.ndarray = np.array([])
        self.learned_thresholds_: List[float] = []

    def fit(self, snrs: Sequence[float], labels: Sequence[str]) -> "R1CARTClassifier":
        """Fit decision tree on 1D SNR values and discrete action labels."""
        X = np.array(snrs, dtype=np.float64).reshape(-1, 1)
        y = np.array(labels, dtype=str)

        self._clf.fit(X, y)
        self.classes_ = self._clf.classes_
        self.is_fitted = True

        # Extract internal split thresholds
        tree = self._clf.tree_
        thresholds = []
        for i in range(tree.node_count):
            if tree.children_left[i] != tree.children_right[i]:  # Non-leaf node
                thresholds.append(float(tree.threshold[i]))

        self.learned_thresholds_ = sorted(thresholds)
        return self

    def predict(self, snrs: Union[Sequence[float], np.ndarray, float]) -> List[str]:
        """Predict coded actions for given SNR points."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before predicting.")

        if isinstance(snrs, (int, float)):
            arr = np.array([[float(snrs)]])
        else:
            arr = np.array(snrs, dtype=np.float64).reshape(-1, 1)

        preds = self._clf.predict(arr)
        return [str(p) for p in preds]

    def score(self, snrs: Sequence[float], labels: Sequence[str]) -> float:
        """Compute training fidelity (accuracy) against true labels."""
        preds = self.predict(snrs)
        correct = sum(1 for p, y in zip(preds, labels) if p == y)
        return float(correct) / float(len(labels)) if len(labels) > 0 else 0.0

    def get_metrics(self, snrs: Sequence[float], labels: Sequence[str]) -> R1TreeMetrics:
        """Extract full complexity and fidelity metrics."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted to get metrics.")

        tree = self._clf.tree_
        acc = self.score(snrs, labels)

        return R1TreeMetrics(
            max_depth_param=self.max_depth if self.max_depth is not None else int(tree.max_depth),
            actual_depth=int(tree.max_depth),
            node_count=int(tree.node_count),
            leaf_count=int(tree.n_leaves),
            accuracy=acc,
            learned_thresholds=tuple(self.learned_thresholds_),
            action_classes=tuple(str(c) for c in self.classes_),
        )


def sweep_and_select_r1_cart_depth(
    snrs: Sequence[float],
    labels: Sequence[str],
    max_depth_range: Sequence[int] = (1, 2, 3, 4, 5, 6),
    target_fidelity: float = 1.0,
    random_state: int = 20260920,
) -> Tuple[R1CARTClassifier, R1TreeMetrics, List[Dict[str, Any]], bool]:
    """Sweep maximum depth and select the most parsimonious tree meeting target fidelity.

    Parameters
    ----------
    snrs : Sequence[float]
        Operating SNR values.
    labels : Sequence[str]
        Target Ground Truth coded actions.
    max_depth_range : Sequence[int], default (1..6)
        Candidate maximum depths to evaluate.
    target_fidelity : float, default 1.0
        Required classification accuracy (1.0 = 100% fidelity).
    random_state : int, default 20260920
        RNG seed for deterministic tree induction.

    Returns
    -------
    chosen_clf : R1CARTClassifier
        Best selected classifier.
    chosen_metrics : R1TreeMetrics
        Metrics for selected classifier.
    sweep_rows : List[Dict[str, Any]]
        Tabular metrics across all evaluated depths.
    achieved_target : bool
        Whether target fidelity was achieved.
    """
    sweep_rows: List[Dict[str, Any]] = []
    candidates: List[Tuple[R1CARTClassifier, R1TreeMetrics]] = []

    for d in max_depth_range:
        clf = R1CARTClassifier(max_depth=d, random_state=random_state)
        clf.fit(snrs, labels)
        metrics = clf.get_metrics(snrs, labels)

        sweep_rows.append({
            "max_depth_param": d,
            "actual_depth": metrics.actual_depth,
            "node_count": metrics.node_count,
            "leaf_count": metrics.leaf_count,
            "accuracy": metrics.accuracy,
            "fidelity_percent": metrics.accuracy * 100.0,
            "learned_thresholds": list(metrics.learned_thresholds),
            "achieved_target": (metrics.accuracy >= target_fidelity - 1e-9),
        })
        candidates.append((clf, metrics))

    # Model Selection Rule:
    # 1. Choose smallest depth achieving target fidelity (100%)
    qualifying = [c for c in candidates if c[1].accuracy >= target_fidelity - 1e-9]
    if qualifying:
        # Smallest depth parameter
        chosen_clf, chosen_metrics = min(qualifying, key=lambda c: c[1].max_depth_param)
        achieved_target = True
    else:
        # Fallback: largest accuracy, then smallest depth
        chosen_clf, chosen_metrics = max(candidates, key=lambda c: (c[1].accuracy, -c[1].max_depth_param))
        achieved_target = False

    return chosen_clf, chosen_metrics, sweep_rows, achieved_target
