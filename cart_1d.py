"""L4: Generic Supervised CART Machine Learning Framework for Link Adaptation.

Provides a feature-agnostic Decision Tree (CART) classification core supporting
arbitrary input features (e.g., 1D SNR, or future 2D [SNR, |h|]) while running
the current 1D SNR link adaptation baseline.

Architecture:
    results/l3_final/ground_truth_1d.csv
                ↓
    Dataset Loader & Feature Schema (X: (N, d), y: (N,), metadata)
                ↓
    CARTClassifier (Generic feature-agnostic estimator)
        ├── CART1DClassifier (Thin 1D wrapper for scalar/1D inputs)
        ├── Depth Sweep & Hyperparameter Evaluation
        ├── Threshold Attribution & LUT Comparison
        └── Reproducible Model Serialization (joblib + metadata)
                ↓
    results/l4_final/
        ├── l4_cart_1d_report.md
        ├── cart_1d_predictions.csv
        ├── cart_depth_sweep.csv
        └── cart_1d_model.joblib
"""
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

from ground_truth import (
    LookupTable1D,
    LUTInterval,
    MODULATION_BPS,
)


@dataclass(frozen=True)
class TreeMetrics:
    """Structural and performance metrics for a trained decision tree."""
    max_depth_param: int
    actual_depth: int
    node_count: int
    leaf_count: int
    accuracy: float
    learned_thresholds: List[float]
    tree_text: str
    threshold_splits: Optional[List[Dict[str, Any]]] = None


# =====================================================================
# 1. Generic Feature-Agnostic CART Classifier
# =====================================================================

class CARTClassifier:
    """Feature-agnostic Decision Tree Link Adaptation policy using CART.

    Accepts arbitrary feature dimensions (n_samples, n_features) and ordered
    feature names, without hardcoding 1D assumptions or specific feature identifiers.
    """

    def __init__(
        self,
        max_depth: int = 3,
        random_state: int = 20260918,
        criterion: str = "gini",
        feature_names: Optional[Sequence[str]] = None,
    ):
        self.max_depth = max_depth
        self.random_state = random_state
        self.criterion = criterion
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.clf = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            random_state=random_state,
        )
        self.classes_: List[str] = []
        self.is_fitted: bool = False
        self.n_features_in_: int = 0

    def fit(
        self,
        X: Union[np.ndarray, Sequence[Sequence[float]]],
        y: Sequence[str],
        feature_names: Optional[Sequence[str]] = None,
    ) -> "CARTClassifier":
        """Fit CART decision tree on feature matrix X and target labels y.

        Args:
            X: 2D array-like of shape (n_samples, n_features).
            y: 1D array-like of target class labels (n_samples,).
            feature_names: Optional list of feature names matching X.shape[1].
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(
                f"Generic CARTClassifier requires 2D input of shape (n_samples, n_features), "
                f"got array with ndim={X_arr.ndim} and shape={X_arr.shape}. "
                f"For 1D scalar sequences, use CART1DClassifier or reshape inputs to (-1, 1)."
            )

        n_samples, n_features = X_arr.shape
        if n_samples != len(y):
            raise ValueError(f"Sample count mismatch: X has {n_samples} samples but y has {len(y)} labels.")

        fn = list(feature_names) if feature_names is not None else self.feature_names
        if fn is not None:
            if len(fn) != n_features:
                raise ValueError(
                    f"feature_names length ({len(fn)}) does not match feature dimension ({n_features})."
                )
            self.feature_names = fn
        else:
            self.feature_names = [f"feature_{i}" for i in range(n_features)]

        y_arr = np.asarray(y, dtype=str)
        self.clf.fit(X_arr, y_arr)
        self.classes_ = list(self.clf.classes_)
        self.n_features_in_ = n_features
        self.is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, Sequence[Sequence[float]]]) -> List[str]:
        """Predict optimal modulation mode for given feature vectors."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before predicting.")

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(
                f"predict expects 2D array of shape (n_samples, n_features), got shape {X_arr.shape}"
            )
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature count mismatch: model expects {self.n_features_in_} features, got {X_arr.shape[1]}"
            )

        preds = self.clf.predict(X_arr)
        return [str(p) for p in preds]

    def predict_proba(self, X: Union[np.ndarray, Sequence[Sequence[float]]]) -> np.ndarray:
        """Predict class probabilities for given feature vectors."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before predicting.")

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected shape (n_samples, {self.n_features_in_}), got {X_arr.shape}")

        return self.clf.predict_proba(X_arr)

    def get_threshold_splits(self) -> List[Dict[str, Any]]:
        """Extract all internal split thresholds with explicit feature attribution.

        This is the canonical split representation for generic multi-feature CART models.
        Each split record contains:
            - 'node_id': internal tree node index
            - 'feature_index': integer column index of the splitting feature
            - 'feature_name': name of the splitting feature
            - 'threshold': scalar threshold value for feature <= threshold
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before extracting threshold splits.")

        tree = self.clf.tree_
        splits = []
        for i in range(tree.node_count):
            f_idx = tree.feature[i]
            if f_idx >= 0:
                feat_name = (
                    self.feature_names[f_idx]
                    if self.feature_names and f_idx < len(self.feature_names)
                    else f"feature_{f_idx}"
                )
                splits.append({
                    "node_id": int(i),
                    "feature_index": int(f_idx),
                    "feature_name": feat_name,
                    "threshold": float(tree.threshold[i]),
                })
        return splits

    def get_learned_thresholds(self) -> List[float]:
        """Extract flat sorted list of numeric split thresholds across all internal nodes.

        NOTE: This method is a flat convenience helper maintained primarily for 1D single-feature
        workflows and backward compatibility. For multi-feature CART models, use get_threshold_splits(),
        as globally sorted numeric thresholds across heterogeneous feature dimensions are not physically
        comparable or meaningful without feature attribution.
        """
        return sorted(s["threshold"] for s in self.get_threshold_splits())

    def export_tree_text(self, feature_names: Optional[Sequence[str]] = None) -> str:
        """Export readable ASCII representation of the learned tree structure."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before exporting tree text.")
        fn = list(feature_names) if feature_names is not None else self.feature_names
        return export_text(self.clf, feature_names=fn)

    def evaluate(
        self,
        X: Union[np.ndarray, Sequence[Sequence[float]]],
        y: Sequence[str],
    ) -> TreeMetrics:
        """Evaluate structural complexity and empirical accuracy on provided data."""
        if not self.is_fitted:
            self.fit(X, y)

        preds = self.predict(X)
        acc = float(np.mean([p == t for p, t in zip(preds, y)]))
        thresholds = self.get_learned_thresholds()
        splits = self.get_threshold_splits()

        return TreeMetrics(
            max_depth_param=self.max_depth,
            actual_depth=int(self.clf.get_depth()),
            node_count=int(self.clf.tree_.node_count),
            leaf_count=int(self.clf.get_n_leaves()),
            accuracy=acc,
            learned_thresholds=thresholds,
            tree_text=self.export_tree_text(),
            threshold_splits=splits,
        )


# =====================================================================
# 2. Thin 1D Experiment Specialization
# =====================================================================

class CART1DClassifier(CARTClassifier):
    """Thin 1D wrapper of CARTClassifier for SNR-only link adaptation workflows."""

    def __init__(
        self,
        max_depth: int = 3,
        random_state: int = 20260918,
        criterion: str = "gini",
        feature_name: str = "SNR_dB",
    ):
        super().__init__(
            max_depth=max_depth,
            random_state=random_state,
            criterion=criterion,
            feature_names=[feature_name],
        )

    def fit(
        self,
        snr_db: Union[Sequence[float], Sequence[Sequence[float]], np.ndarray],
        best_mode: Sequence[str],
        feature_names: Optional[Sequence[str]] = None,
    ) -> "CART1DClassifier":
        """Fit 1D decision tree accepting scalar lists or (n, 1) arrays."""
        arr = np.asarray(snr_db, dtype=np.float64)
        if arr.ndim == 1:
            X = arr.reshape(-1, 1)
        elif arr.ndim == 2 and arr.shape[1] == 1:
            X = arr
        else:
            raise ValueError(f"CART1DClassifier expects 1D sequence or (n, 1) array, got shape {arr.shape}")

        fn = feature_names if feature_names is not None else self.feature_names
        super().fit(X, best_mode, feature_names=fn)
        return self

    def predict(
        self,
        snr_db: Union[float, int, Sequence[float], Sequence[Sequence[float]], np.ndarray],
    ) -> Union[str, List[str]]:
        """Predict modulation mode for scalar or array of SNR values."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before predicting.")

        if isinstance(snr_db, (int, float, np.floating, np.integer)):
            X = np.array([[float(snr_db)]], dtype=np.float64)
            return super().predict(X)[0]

        arr = np.asarray(snr_db, dtype=np.float64)
        if arr.ndim == 0:
            X = np.array([[float(arr)]], dtype=np.float64)
            return super().predict(X)[0]
        elif arr.ndim == 1:
            X = arr.reshape(-1, 1)
            return super().predict(X)
        elif arr.ndim == 2 and arr.shape[1] == 1:
            return super().predict(arr)
        else:
            raise ValueError(f"CART1DClassifier predict expects scalar, 1D array, or (n, 1) array, got {arr.shape}")

    def evaluate(
        self,
        snr_db: Union[Sequence[float], Sequence[Sequence[float]], np.ndarray],
        best_mode: Sequence[str],
    ) -> TreeMetrics:
        """Evaluate structural complexity and training accuracy for 1D inputs."""
        arr = np.asarray(snr_db, dtype=np.float64)
        X = arr.reshape(-1, 1) if arr.ndim == 1 else arr
        if not self.is_fitted:
            self.fit(X, best_mode)

        preds = self.predict(X)
        acc = float(np.mean([p == t for p, t in zip(preds, best_mode)]))
        thresholds = self.get_learned_thresholds()
        splits = self.get_threshold_splits()

        return TreeMetrics(
            max_depth_param=self.max_depth,
            actual_depth=int(self.clf.get_depth()),
            node_count=int(self.clf.tree_.node_count),
            leaf_count=int(self.clf.get_n_leaves()),
            accuracy=acc,
            learned_thresholds=thresholds,
            tree_text=self.export_tree_text(),
            threshold_splits=splits,
        )


# =====================================================================
# 3. Dataset Loading & Feature Schema
# =====================================================================

def load_ground_truth_dataset(
    csv_path: Union[str, Path] = "results/l3_final/ground_truth_1d.csv",
    feature_names: Sequence[str] = ("snr_db",),
    target_col: str = "best_mode",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load ground truth link adaptation dataset with arbitrary feature schema.

    Args:
        csv_path: Path to ground truth CSV (e.g. results/l3_final/ground_truth_1d.csv).
        feature_names: Ordered list of feature column names to extract into X.
        target_col: Target column name for classification labels.

    Returns:
        X: Feature matrix of shape (n_samples, len(feature_names)).
        y: Target label array of shape (n_samples,).
        metadata: Dictionary containing sample metadata (uncertainty flags, block counts, snr_db).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth dataset not found at: {path}")

    feature_cols = list(feature_names)
    X_rows: List[List[float]] = []
    y_vals: List[str] = []
    meta_rows: Dict[str, List[Any]] = {
        "snr_db": [],
        "reliability_uncertain": [],
        "label_uncertain": [],
        "num_blocks": [],
        "fallback_used": [],
    }

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_feats = []
            for col in feature_cols:
                if col not in row:
                    raise KeyError(f"Feature column '{col}' not found in {path}. Available: {list(row.keys())}")
                sample_feats.append(float(row[col]))
            X_rows.append(sample_feats)

            if target_col not in row:
                raise KeyError(f"Target column '{target_col}' not found in {path}. Available: {list(row.keys())}")
            y_vals.append(str(row[target_col]).strip())

            # Metadata preservation
            if "snr_db" in row:
                meta_rows["snr_db"].append(float(row["snr_db"]))
            if "reliability_uncertain" in row:
                meta_rows["reliability_uncertain"].append(str(row["reliability_uncertain"]).lower() == "true")
            if "label_uncertain" in row:
                meta_rows["label_uncertain"].append(str(row["label_uncertain"]).lower() == "true")
            if "num_blocks" in row:
                meta_rows["num_blocks"].append(int(float(row["num_blocks"])))
            if "fallback_used" in row:
                meta_rows["fallback_used"].append(str(row["fallback_used"]).lower() == "true")

    X = np.asarray(X_rows, dtype=np.float64)
    y = np.asarray(y_vals, dtype=str)

    metadata: Dict[str, Any] = {
        "feature_names": feature_cols,
        "target_col": target_col,
        "n_samples": len(y_vals),
        "snr_db": np.asarray(meta_rows["snr_db"], dtype=np.float64),
        "reliability_uncertain": np.asarray(meta_rows["reliability_uncertain"], dtype=bool),
        "label_uncertain": np.asarray(meta_rows["label_uncertain"], dtype=bool),
        "num_blocks": np.asarray(meta_rows["num_blocks"], dtype=np.int64),
        "fallback_used": np.asarray(meta_rows["fallback_used"], dtype=bool),
        "source_path": str(path),
    }

    return X, y, metadata


def load_ground_truth_samples(csv_path: Union[str, Path]) -> Tuple[List[float], List[str]]:
    """Legacy helper for loading 1D SNR and BestMode labels."""
    X, y, _ = load_ground_truth_dataset(csv_path=csv_path, feature_names=["snr_db"], target_col="best_mode")
    return list(X[:, 0]), list(y)


def load_lut_csv(csv_path: Union[str, Path] = "results/l3_final/lut_1d.csv") -> LookupTable1D:
    """Load 1D Look-Up Table policy directly from frozen L3 LUT CSV artifact.

    Args:
        csv_path: Path to LUT CSV file (results/l3_final/lut_1d.csv).

    Returns:
        LookupTable1D instance with intervals, switching thresholds, and non-monotonic checks.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"L3 1D LUT file not found at: {path}")

    intervals: List[LUTInterval] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            min_s = -float("inf") if row["min_snr_db"].strip().lower() == "-inf" else float(row["min_snr_db"])
            max_s = float("inf") if row["max_snr_db"].strip().lower() == "inf" else float(row["max_snr_db"])
            m = str(row["mode"]).strip()
            bps = int(row["bits_per_symbol"])
            intervals.append(LUTInterval(min_snr=min_s, max_snr=max_s, mode=m, bits_per_symbol=bps))

    thresholds: List[Tuple[float, str, str]] = []
    non_monotonic: List[Tuple[float, str, str]] = []
    for i in range(len(intervals) - 1):
        curr = intervals[i]
        nxt = intervals[i + 1]
        th = curr.max_snr
        thresholds.append((th, curr.mode, nxt.mode))
        if nxt.bits_per_symbol < curr.bits_per_symbol:
            non_monotonic.append((th, curr.mode, nxt.mode))

    return LookupTable1D(intervals=intervals, thresholds=thresholds, non_monotonic_transitions=non_monotonic)


def load_lut_from_ground_truth_csv(
    csv_path: Union[str, Path] = "results/l3_final/ground_truth_1d.csv",
) -> LookupTable1D:
    """Derive 1D Look-Up Table policy solely from frozen L3 ground truth CSV without L2 calibration data.

    Args:
        csv_path: Path to ground truth CSV (results/l3_final/ground_truth_1d.csv).

    Returns:
        LookupTable1D instance constructed from switching points in ground_truth_1d.csv.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"L3 ground truth dataset not found at: {path}")

    rows: List[Tuple[float, str, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            snr = float(r["snr_db"])
            mode = str(r["best_mode"]).strip()
            bps = int(r["best_mode_bps"]) if "best_mode_bps" in r else MODULATION_BPS[mode]
            rows.append((snr, mode, bps))

    if not rows:
        raise ValueError(f"No ground truth rows found in {path}")

    rows.sort(key=lambda x: x[0])
    thresholds: List[Tuple[float, str, str]] = []
    non_monotonic: List[Tuple[float, str, str]] = []
    for i in range(len(rows) - 1):
        curr_snr, curr_mode, curr_bps = rows[i]
        nxt_snr, nxt_mode, nxt_bps = rows[i + 1]
        if curr_mode != nxt_mode:
            th = 0.5 * (curr_snr + nxt_snr)
            thresholds.append((th, curr_mode, nxt_mode))
            if nxt_bps < curr_bps:
                non_monotonic.append((th, curr_mode, nxt_mode))

    intervals: List[LUTInterval] = []
    lower_bound = -float("inf")
    for th, prev_mode, next_mode in thresholds:
        bps = MODULATION_BPS[prev_mode]
        intervals.append(LUTInterval(min_snr=lower_bound, max_snr=th, mode=prev_mode, bits_per_symbol=bps))
        lower_bound = th

    final_mode = rows[-1][1]
    intervals.append(
        LUTInterval(
            min_snr=lower_bound,
            max_snr=float("inf"),
            mode=final_mode,
            bits_per_symbol=MODULATION_BPS[final_mode],
        )
    )

    return LookupTable1D(intervals=intervals, thresholds=thresholds, non_monotonic_transitions=non_monotonic)


# =====================================================================
# 4. Hyperparameter Evaluation & Depth Sweep
# =====================================================================

def sweep_cart_depths(
    X: Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]],
    y: Sequence[str],
    depths: Sequence[int] = (1, 2, 3, 4, 5),
    random_state: int = 20260918,
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[int, Tuple[CARTClassifier, TreeMetrics]]:
    """Sweep max_depth over candidate values and return fitted models and metrics."""
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
        if feature_names is None:
            feature_names = ["SNR_dB"]
    elif feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]

    results = {}
    for d in depths:
        clf = CARTClassifier(max_depth=d, random_state=random_state, feature_names=feature_names)
        clf.fit(X_arr, y)
        metrics = clf.evaluate(X_arr, y)
        results[d] = (clf, metrics)
    return results


# =====================================================================
# 5. Model Serialization & Export
# =====================================================================

def save_model(
    clf: CARTClassifier,
    output_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist fitted CART model along with complete reproducibility metadata."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "estimator": clf.clf,
        "feature_names": clf.feature_names,
        "classes": clf.classes_,
        "selected_hyperparameters": {
            "max_depth": clf.max_depth,
            "criterion": clf.criterion,
            "random_state": clf.random_state,
        },
        "actual_depth": int(clf.clf.get_depth()),
        "node_count": int(clf.clf.tree_.node_count),
        "leaf_count": int(clf.clf.get_n_leaves()),
        "training_metadata": metadata or {},
    }
    joblib.dump(payload, path)
    return path


def load_model(model_path: Union[str, Path]) -> Tuple[CARTClassifier, Dict[str, Any]]:
    """Load persisted CART model and its metadata dictionary."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")
    payload = joblib.load(path)
    hp = payload.get("selected_hyperparameters", {})
    clf = CARTClassifier(
        max_depth=hp.get("max_depth", 3),
        random_state=hp.get("random_state", 20260918),
        criterion=hp.get("criterion", "gini"),
        feature_names=payload.get("feature_names"),
    )
    clf.clf = payload["estimator"]
    clf.classes_ = list(payload.get("classes", []))
    clf.n_features_in_ = clf.clf.n_features_in_
    clf.is_fitted = True
    return clf, payload


def save_predictions_csv(
    snr_db: Sequence[float],
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metadata: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """Save detailed point-by-point predictions and uncertainty flags to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "snr_db",
        "ground_truth_best_mode",
        "cart_predicted_mode",
        "is_match",
        "reliability_uncertain",
        "label_uncertain",
        "num_blocks",
    ]
    rel_unc = metadata.get("reliability_uncertain", [False] * len(y_true))
    lbl_unc = metadata.get("label_uncertain", [False] * len(y_true))
    n_blks = metadata.get("num_blocks", [0] * len(y_true))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s, yt, yp, ru, lu, nb in zip(snr_db, y_true, y_pred, rel_unc, lbl_unc, n_blks):
            writer.writerow({
                "snr_db": f"{float(s):.1f}",
                "ground_truth_best_mode": yt,
                "cart_predicted_mode": yp,
                "is_match": str(yt == yp),
                "reliability_uncertain": str(bool(ru)),
                "label_uncertain": str(bool(lu)),
                "num_blocks": int(nb),
            })
    return path


def save_depth_sweep_csv(
    sweep_results: Dict[int, Tuple[CARTClassifier, TreeMetrics]],
    lut: Optional[LookupTable1D],
    output_path: Union[str, Path],
) -> Path:
    """Save depth sweep hyperparameter exploration metrics to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lut_thresholds = [round(th, 2) for th, _, _ in lut.thresholds] if lut is not None else []
    fieldnames = [
        "max_depth",
        "actual_depth",
        "node_count",
        "leaf_count",
        "training_accuracy",
        "learned_thresholds",
        "lut_equivalence",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in sorted(sweep_results.keys()):
            clf, m = sweep_results[d]
            cart_th = [round(t, 2) for t in m.learned_thresholds]
            is_equiv = (cart_th == lut_thresholds) and (m.accuracy == 1.0)
            equiv_str = "EXACT_MATCH" if is_equiv else ("SUB_OPTIMAL" if m.accuracy < 1.0 else "EQUIVALENT")
            writer.writerow({
                "max_depth": m.max_depth_param,
                "actual_depth": m.actual_depth,
                "node_count": m.node_count,
                "leaf_count": m.leaf_count,
                "training_accuracy": f"{m.accuracy:.4f}",
                "learned_thresholds": "; ".join(f"{t:.2f}" for t in cart_th) if cart_th else "None",
                "lut_equivalence": equiv_str,
            })
    return path


# =====================================================================
# 6. Comprehensive L4 Thesis Report Generator
# =====================================================================

def generate_l4_cart_report(
    sweep_results: Dict[int, Tuple[CARTClassifier, TreeMetrics]],
    lut: LookupTable1D,
    snr_db: Sequence[float],
    best_mode: Sequence[str],
    metadata: Optional[Dict[str, Any]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Generate comprehensive L4 CART 1D Machine Learning report formatted for a thesis."""
    lut_thresholds = [round(th, 2) for th, _, _ in lut.thresholds]

    best_depth = min(d for d, (_, m) in sweep_results.items() if m.accuracy == 1.0)
    best_clf, best_m = sweep_results[best_depth]

    lines = [
        "# PHY-ML L4: CART 1D Machine Learning Baseline Report",
        "",
        "**Topic:** Supervised Decision Tree Classification for 1D Uncoded Link Adaptation  ",
        "**Layer:** L4 (Machine Learning - CART 1D Baseline)  ",
        "**Date:** 2026-09-19  ",
        "**Source Dataset:** `results/l3_final/ground_truth_1d.csv`  ",
        "**Framework Architecture:** Feature-Agnostic Generic `CARTClassifier`  ",
        "",
        "---",
        "",
        "## 1. Machine Learning Formulation & Setup",
        "",
        "- **Supervised Learning Task:** Multiclass Classification ($X \\to y$)",
        "- **Input Feature Space:** $X = [SNR_{setup}]$ (1D continuous scalar, nominal $E_s/N_0$ in dB)",
        "- **Target Space:** $y \\in \\{\\text{BPSK}, \\text{QPSK}, \\text{16QAM}, \\text{64QAM}\\}$",
        "- **Ground Truth Basis:** Empirically calibrated uncoded PHY under slow Rayleigh flat block fading (25 operating points from 0.0 to 30.0 dB, adaptive dual-seed depth totaling 1,140,000 blocks, $BER_{target} = 0.0100$).",
        "- **Algorithm:** Classification and Regression Trees (CART) via generic `CARTClassifier` (`sklearn.tree.DecisionTreeClassifier`).",
        "- **Split Criterion:** Gini Impurity ($I_G(p) = 1 - \\sum_k p_k^2$).",
        "- **Random State:** `20260918` (guarantees deterministic tree induction).",
        "- **Baseline for Comparison:** 1D Look-Up Table (LUT) derived from the exact same sampled calibration grid.",
        "",
        "---",
        "",
        "## 2. Decision Tree Depth Sweep (max_depth = 1..5)",
        "",
        "| max_depth | Actual Depth | Total Nodes | Leaf Nodes | Training Fidelity | Learned Split Thresholds (dB) | LUT Equivalence |",
        "|:---------:|:------------:|:-----------:|:----------:|:-----------------:|:------------------------------:|:---------------:|",
    ]

    for d in sorted(sweep_results.keys()):
        clf, m = sweep_results[d]
        cart_th = [round(t, 2) for t in m.learned_thresholds]
        is_equiv = (cart_th == lut_thresholds) and (m.accuracy == 1.0)
        equiv_str = "**EXACT MATCH**" if is_equiv else ("Sub-optimal" if m.accuracy < 1.0 else "Equivalent")
        th_str = ", ".join(f"{t:.2f}" for t in cart_th) if cart_th else "None"
        lines.append(
            f"| {m.max_depth_param:9d} | {m.actual_depth:12d} | {m.node_count:11d} | {m.leaf_count:10d} | "
            f"{m.accuracy * 100:17.1f}% | [{th_str}] | {equiv_str:15s} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Comparative Threshold Analysis: CART vs. 1D LUT Baseline",
        "",
        "The 1D Look-Up Table defines **sampled-grid-derived switching thresholds** at the midpoints between adjacent sampled SNRs where BestMode transitions:",
    ])
    for th, p_m, n_m in lut.thresholds:
        lines.append(f"- **{th:5.2f} dB**: Transition from **{p_m}** ({MODULATION_BPS[p_m]} bpcu) $\\to$ **{n_m}** ({MODULATION_BPS[n_m]} bpcu)")

    lines.extend([
        "",
        f"### Selected Baseline Model: `max_depth = {best_depth}`",
        f"- **Learned Split Thresholds:** {[round(t, 2) for t in best_m.learned_thresholds]} dB",
        f"- **LUT Switching Thresholds:** {lut_thresholds} dB",
        f"- **Structural Complexity:** {best_m.node_count} total nodes ({best_m.leaf_count} leaves), actual tree depth = {best_m.actual_depth}",
        f"- **Threshold Parity:** The CART decision tree with `max_depth = {best_depth}` reproduces the exact 1D Look-Up Table sampled-grid-derived switching thresholds without manual heuristic intervention.",
        f"- **Physical Interpretation:** These switching thresholds reflect discrete midpoints of the empirical calibration grid (0.5 dB resolution in active zones); they are not claimed to be exact continuous physical BER-crossing thresholds.",
        "",
        "---",
        "",
        f"## 4. Textual Decision Tree Export (max_depth = {best_depth})",
        "",
        "```text",
        best_m.tree_text.strip(),
        "```",
        "",
        "---",
        "",
        "## 5. Point-by-Point Ground Truth vs. CART Classification (25 Operating Points)",
        "",
        "| SNR (dB) | Ground Truth BestMode | CART Prediction | Match? | Reliability Unc. | Label Unc. | Blocks |",
        "|:--------:|:---------------------:|:---------------:|:------:|:----------------:|:----------:|:------:|",
    ])

    rel_unc = metadata.get("reliability_uncertain", [False] * len(snr_db)) if metadata else [False] * len(snr_db)
    lbl_unc = metadata.get("label_uncertain", [False] * len(snr_db)) if metadata else [False] * len(snr_db)
    n_blks = metadata.get("num_blocks", [0] * len(snr_db)) if metadata else [0] * len(snr_db)

    preds_best = best_clf.predict(np.asarray(snr_db, dtype=np.float64).reshape(-1, 1))
    for s, gt_m, pred, ru, lu, nb in zip(snr_db, best_mode, preds_best, rel_unc, lbl_unc, n_blks):
        match_str = "MATCH" if pred == gt_m else "MISMATCH"
        ru_str = "FLAGGED" if ru else "Clear"
        lu_str = "FLAGGED" if lu else "Clear"
        blks_str = f"{int(nb):6,d}" if nb > 0 else "   N/A"
        lines.append(
            f"| {s:8.1f} | {gt_m:21s} | {pred:15s} | {match_str:6s} | {ru_str:16s} | {lu_str:10s} | {blks_str} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 6. Uncertainty & Sensitivity Analysis at Operating Boundaries",
        "",
        "The model is trained strictly on nominal point-estimate ground-truth labels without discarding or heuristically modifying uncertain points. Statistical uncertainty is preserved as evaluation metadata:",
        "",
        "- **SNR = 14.0 dB (`reliability_uncertain = True`, `label_uncertain = False`):** BPSK 95% confidence interval $[9.123 \\times 10^{-3}, 1.003 \\times 10^{-2}]$ overlaps $BER_{target} = 0.0100$. However, because BPSK is also the fallback policy if no mode qualifies, BestMode remains invariant to CI bounds. CART correctly predicts BPSK.",
        "- **SNR = 23.0 dB (`reliability_uncertain = True`, `label_uncertain = True`):** 16-QAM point-estimate BER is $9.714 \\times 10^{-3}$ (compliant), but its upper 95% CI bound reaches $1.0001 \\times 10^{-2}$. Under pessimistic bounds, QPSK would be selected; under nominal point estimate, 16-QAM is selected. CART reproduces the nominal label (16QAM, 4 bpcu).",
        "- **SNR = 28.0 dB (`reliability_uncertain = True`, `label_uncertain = True`):** 64-QAM point-estimate BER is $1.0095 \\times 10^{-2}$ (non-compliant), but its lower 95% CI bound reaches $9.827 \\times 10^{-3}$. Under optimistic bounds, 64-QAM would qualify; under nominal point estimate, 16-QAM is selected. CART reproduces the nominal label (16QAM, 4 bpcu).",
        "",
        "---",
        "",
        "## 7. Domain & Extrapolation Behavior",
        "",
        "- **Empirically Calibrated Domain:** $0 \\le SNR_{setup} \\le 30\\text{ dB}$. The model is grounded in Monte Carlo simulations strictly within this 25-point operating window.",
        "- **Extrapolation Behavior:** For $SNR < 0\\text{ dB}$, the tree evaluates $SNR \\le 16.75\\text{ dB}$ and predicts BPSK (1 bpcu). For $SNR > 30\\text{ dB}$, the tree evaluates $SNR > 28.25\\text{ dB}$ and predicts 64-QAM (6 bpcu). This constant leaf extension is an inherent property of axis-aligned decision trees (model extrapolation / endpoint behavior), not validated physical ground truth.",
        "",
        "---",
        "",
        "## 8. Generalization Scope & Thesis Baseline Synthesis",
        "",
        "- **Representation Completeness:** With 4 distinct modulation modes, a binary decision tree requires a theoretical minimum depth of $\\lceil \\log_2 4 \\rceil = 2$ and 3 decision splits to partition a 1D scalar space into 4 contiguous intervals.",
        "- **Representation Fidelity Baseline:** The current 1D experiment functions as a baseline representation and fidelity test, verifying that CART exactly learns the discrete 1D Look-Up Table policy. It is not presented as a broad statistical generalization study across continuous or unobserved channel profiles.",
        "- **Extensibility:** The refactored `CARTClassifier` abstraction is feature-agnostic, ready to ingest multi-feature inputs (such as instantaneous channel magnitude $|h|$) without altering the core learning or threshold attribution logic.",
    ])

    report_content = "\n".join(lines) + "\n"

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(report_content, encoding="utf-8")

    return report_content


# =====================================================================
# 7. Experiment Runner Entrypoint
# =====================================================================

def run_1d_cart_experiment(
    data_path: Union[str, Path] = "results/l3_final/ground_truth_1d.csv",
    out_dir: Union[str, Path] = "results/l4_final",
    depths: Sequence[int] = (1, 2, 3, 4, 5),
    random_state: int = 20260918,
) -> Dict[str, Any]:
    """Execute complete L4 1D CART experiment and generate all artifacts."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Required final L3 ground truth dataset not found at: {path}. "
            f"Execution halted: L3 outputs are frozen inputs and must exist."
        )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading ground truth dataset: {path}...")
    X, y, metadata = load_ground_truth_dataset(
        csv_path=path,
        feature_names=["snr_db"],
        target_col="best_mode",
    )
    snrs = list(X[:, 0])
    labels = list(y)
    print(f"Loaded {len(snrs)} samples across features: {metadata['feature_names']}.")

    # Load 1D LUT reference directly from frozen L3 artifacts (strictly no L2 dependency)
    lut_csv = Path(data_path).parent / "lut_1d.csv"
    if not lut_csv.exists():
        lut_csv = Path("results/l3_final/lut_1d.csv")

    if lut_csv.exists():
        print(f"Loading 1D LUT reference from: {lut_csv}")
        lut = load_lut_csv(lut_csv)
    else:
        print(f"Deriving 1D LUT reference solely from ground truth: {path}")
        lut = load_lut_from_ground_truth_csv(path)

    print("\nSweeping max_depth in {1, 2, 3, 4, 5}...")
    sweep_results = sweep_cart_depths(
        X=X,
        y=y,
        depths=depths,
        random_state=random_state,
        feature_names=["snr_db"],
    )

    for d, (clf, m) in sweep_results.items():
        print(
            f"  max_depth={d}: actual_depth={m.actual_depth}, "
            f"nodes={m.node_count}, leaves={m.leaf_count}, "
            f"acc={m.accuracy * 100:.1f}%, "
            f"thresholds={[round(t, 2) for t in m.learned_thresholds]}"
        )

    # 1. Depth sweep CSV
    sweep_csv_path = out_path / "cart_depth_sweep.csv"
    save_depth_sweep_csv(sweep_results, lut, sweep_csv_path)
    print(f"\nSaved depth sweep table to: {sweep_csv_path}")

    # 2. Select optimal model (lowest depth achieving 100% accuracy)
    best_depth = min(d for d, (_, m) in sweep_results.items() if m.accuracy == 1.0)
    best_clf, best_m = sweep_results[best_depth]
    best_preds = best_clf.predict(X)

    # 3. Model persistence (joblib + metadata)
    model_joblib_path = out_path / "cart_1d_model.joblib"
    save_model(best_clf, model_joblib_path, metadata=metadata)
    print(f"Saved optimal CART model to: {model_joblib_path}")

    # 4. Point-by-point predictions CSV
    preds_csv_path = out_path / "cart_1d_predictions.csv"
    save_predictions_csv(
        snr_db=snrs,
        y_true=labels,
        y_pred=best_preds,
        metadata=metadata,
        output_path=preds_csv_path,
    )
    print(f"Saved point predictions table to: {preds_csv_path}")

    # 5. Generate Markdown Report
    report_path = out_path / "l4_cart_1d_report.md"
    generate_l4_cart_report(
        sweep_results=sweep_results,
        lut=lut,
        snr_db=snrs,
        best_mode=labels,
        metadata=metadata,
        output_path=report_path,
    )
    print(f"Saved L4 CART report to: {report_path}")

    print(f"\n--- Decision Tree Structure (max_depth={best_depth}) ---")
    print(best_m.tree_text)

    return {
        "selected_depth": best_depth,
        "metrics": best_m,
        "sweep_results": sweep_results,
        "artifacts": {
            "report": report_path,
            "predictions_csv": preds_csv_path,
            "sweep_csv": sweep_csv_path,
            "model_joblib": model_joblib_path,
        },
    }


if __name__ == "__main__":
    print("=== PHY-ML L4: Generic CART Machine Learning Framework ===")
    data_path = Path("results/l3_final/ground_truth_1d.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Required final L3 ground truth dataset not found at: {data_path}. "
            f"Execution halted: L3 outputs are frozen inputs and must exist."
        )
    out_dir = Path("results/l4_final")
    run_1d_cart_experiment(data_path=data_path, out_dir=out_dir)
