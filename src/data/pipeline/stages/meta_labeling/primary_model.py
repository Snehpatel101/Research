"""
Primary Classifier for Meta-Labeling.

Implements a high-recall classifier for direction prediction. The primary model
focuses on maximizing recall (catching most true signals) rather than precision,
since the meta-model will filter out false positives.

Key design principles (Lopez de Prado AFML Ch. 3):
1. Optimize for recall, not accuracy
2. Accept more false positives (they'll be filtered by meta-model)
3. Avoid false negatives (missed opportunities are costly)

Engineering Rules Applied:
- Clear contracts with type hints
- Input validation at boundaries
- Fail fast with explicit errors
- Modular design with single responsibility
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# =============================================================================
# CONSTANTS
# =============================================================================

# Default recall target for primary classifier
DEFAULT_RECALL_TARGET = 0.70

# Minimum acceptable recall (below this, model is too conservative)
MIN_RECALL_THRESHOLD = 0.50

# Maximum decision threshold search range
THRESHOLD_SEARCH_MIN = 0.01
THRESHOLD_SEARCH_MAX = 0.50
THRESHOLD_SEARCH_STEPS = 50


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class PrimaryModelConfig:
    """Configuration for the primary classifier.

    Attributes:
        recall_target: Target recall to achieve (default 0.70)
        min_recall: Minimum acceptable recall (default 0.50)
        base_model: Base model type ('logistic', 'lightgbm', 'xgboost')
        calibrate: Whether to calibrate probabilities
        cv_folds: Number of CV folds for threshold optimization
        random_state: Random seed for reproducibility
        class_weight: Class weight strategy ('balanced' or None)
        verbose: Logging verbosity
    """

    recall_target: float = DEFAULT_RECALL_TARGET
    min_recall: float = MIN_RECALL_THRESHOLD
    base_model: str = "logistic"
    calibrate: bool = True
    cv_folds: int = 5
    random_state: int = 42
    class_weight: str | None = "balanced"
    verbose: bool = True
    model_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0 < self.recall_target <= 1.0:
            raise ValueError(f"recall_target must be in (0, 1], got {self.recall_target}")

        if not 0 < self.min_recall < self.recall_target:
            raise ValueError(
                f"min_recall must be in (0, recall_target), "
                f"got {self.min_recall} with target {self.recall_target}"
            )

        if self.cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {self.cv_folds}")

        valid_models = {"logistic", "lightgbm", "xgboost", "random_forest"}
        if self.base_model not in valid_models:
            raise ValueError(f"base_model must be one of {valid_models}, got {self.base_model}")


# =============================================================================
# RECALL OPTIMIZER
# =============================================================================


class RecallOptimizer:
    """
    Optimizes decision threshold to achieve target recall.

    This class finds the optimal probability threshold that maximizes precision
    while meeting a minimum recall constraint.

    Attributes:
        recall_target: Target recall to achieve
        optimal_threshold: Optimized threshold (set after fit)
        achieved_recall: Actual recall achieved (set after fit)
        achieved_precision: Precision at optimal threshold (set after fit)

    Example:
        >>> optimizer = RecallOptimizer(recall_target=0.70)
        >>> threshold = optimizer.fit(y_true, y_proba)
        >>> y_pred = optimizer.apply_threshold(y_proba)
    """

    def __init__(
        self,
        recall_target: float = DEFAULT_RECALL_TARGET,
        search_min: float = THRESHOLD_SEARCH_MIN,
        search_max: float = THRESHOLD_SEARCH_MAX,
        n_steps: int = THRESHOLD_SEARCH_STEPS,
    ):
        """
        Initialize RecallOptimizer.

        Args:
            recall_target: Target recall to achieve (default 0.70)
            search_min: Minimum threshold to search (default 0.01)
            search_max: Maximum threshold to search (default 0.50)
            n_steps: Number of threshold steps to evaluate (default 50)
        """
        if not 0 < recall_target <= 1.0:
            raise ValueError(f"recall_target must be in (0, 1], got {recall_target}")

        self.recall_target = recall_target
        self.search_min = search_min
        self.search_max = search_max
        self.n_steps = n_steps

        # Set after fit
        self.optimal_threshold: float | None = None
        self.achieved_recall: float | None = None
        self.achieved_precision: float | None = None
        self._threshold_results: list[dict[str, float]] = []

    def fit(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        positive_class: int = 1,
    ) -> float:
        """
        Find optimal threshold that achieves target recall.

        Searches threshold space to find the lowest threshold that achieves
        the target recall, maximizing precision subject to recall constraint.

        Args:
            y_true: True labels (must be binary: 0 or 1)
            y_proba: Predicted probabilities for positive class
            positive_class: Label value for positive class (default 1)

        Returns:
            Optimal threshold value

        Raises:
            ValueError: If inputs are invalid or target recall unachievable
        """
        # Validate inputs
        y_true = np.asarray(y_true).ravel()
        y_proba = np.asarray(y_proba).ravel()

        if len(y_true) != len(y_proba):
            raise ValueError(
                f"y_true and y_proba must have same length, "
                f"got {len(y_true)} and {len(y_proba)}"
            )

        if len(y_true) == 0:
            raise ValueError("Input arrays are empty")

        unique_labels = np.unique(y_true)
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError(f"y_true must be binary (0 or 1), got unique values: {unique_labels}")

        if not y_proba.min() >= 0 and y_proba.max() <= 1:
            raise ValueError("y_proba must be in range [0, 1]")

        # Search thresholds
        thresholds = np.linspace(self.search_min, self.search_max, self.n_steps)
        self._threshold_results = []

        best_threshold = self.search_min
        best_precision = 0.0
        best_recall = 0.0

        for threshold in thresholds:
            # Apply threshold
            y_pred = (y_proba >= threshold).astype(int)

            # Compute metrics
            n_positive_pred = y_pred.sum()
            if n_positive_pred == 0:
                continue

            n_true_positive = ((y_pred == positive_class) & (y_true == positive_class)).sum()
            n_actual_positive = (y_true == positive_class).sum()

            if n_actual_positive == 0:
                continue

            recall = n_true_positive / n_actual_positive
            precision = n_true_positive / n_positive_pred

            self._threshold_results.append(
                {
                    "threshold": threshold,
                    "recall": recall,
                    "precision": precision,
                    "n_positive": n_positive_pred,
                }
            )

            # Check if meets recall target
            if recall >= self.recall_target and precision > best_precision:
                best_precision = precision
                best_threshold = threshold
                best_recall = recall

        # Validate we found a valid threshold
        if best_recall < self.recall_target:
            # Fall back to threshold that maximizes recall
            if self._threshold_results:
                max_recall_result = max(self._threshold_results, key=lambda x: x["recall"])
                best_threshold = max_recall_result["threshold"]
                best_recall = max_recall_result["recall"]
                best_precision = max_recall_result["precision"]
                logger.warning(
                    f"Could not achieve target recall {self.recall_target:.2%}. "
                    f"Using threshold {best_threshold:.3f} with recall {best_recall:.2%}"
                )

        self.optimal_threshold = best_threshold
        self.achieved_recall = best_recall
        self.achieved_precision = best_precision

        logger.info(
            f"RecallOptimizer: threshold={best_threshold:.3f}, "
            f"recall={best_recall:.2%}, precision={best_precision:.2%}"
        )

        return best_threshold

    def apply_threshold(self, y_proba: np.ndarray) -> np.ndarray:
        """
        Apply optimal threshold to probabilities.

        Args:
            y_proba: Predicted probabilities

        Returns:
            Binary predictions (0 or 1)

        Raises:
            RuntimeError: If fit() hasn't been called
        """
        if self.optimal_threshold is None:
            raise RuntimeError("Must call fit() before apply_threshold()")

        return (np.asarray(y_proba) >= self.optimal_threshold).astype(int)


# =============================================================================
# PRIMARY CLASSIFIER
# =============================================================================


class PrimaryClassifier(BaseEstimator, ClassifierMixin):
    """
    High-recall classifier for direction prediction in meta-labeling.

    This classifier is optimized for recall rather than accuracy, following
    the meta-labeling approach from Lopez de Prado's AFML. The goal is to
    catch as many true trading signals as possible, accepting that some
    false positives will be filtered by the downstream meta-model.

    The classifier works in two stages:
    1. Train a base model (logistic regression, etc.)
    2. Optimize decision threshold to achieve target recall

    For multi-class labels (-1, 0, 1), the classifier treats the problem
    as binary by focusing on "has signal" (1 or -1) vs "no signal" (0).

    Attributes:
        config: PrimaryModelConfig with hyperparameters
        base_model_: Trained base model (after fit)
        recall_optimizer_: Threshold optimizer (after fit)
        classes_: Array of class labels

    Example:
        >>> primary = PrimaryClassifier(recall_target=0.70)
        >>> primary.fit(X_train, y_train)
        >>> sides = primary.predict_side(X_test)  # Returns {-1, 1}
        >>> proba = primary.predict_proba(X_test)  # Returns probabilities
    """

    def __init__(
        self,
        recall_target: float = DEFAULT_RECALL_TARGET,
        base_model: str = "logistic",
        calibrate: bool = True,
        cv_folds: int = 5,
        random_state: int = 42,
        class_weight: str | None = "balanced",
        verbose: bool = True,
        **model_params: Any,
    ):
        """
        Initialize PrimaryClassifier.

        Args:
            recall_target: Target recall to achieve (default 0.70)
            base_model: Base model type ('logistic', 'lightgbm', 'xgboost')
            calibrate: Whether to calibrate probabilities (default True)
            cv_folds: Number of CV folds for optimization (default 5)
            random_state: Random seed (default 42)
            class_weight: Class weight strategy ('balanced' or None)
            verbose: Enable verbose logging (default True)
            **model_params: Additional parameters for base model
        """
        self.recall_target = recall_target
        self.base_model = base_model
        self.calibrate = calibrate
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.class_weight = class_weight
        self.verbose = verbose
        self.model_params = model_params

        # Set during fit
        self.base_model_: Any = None
        self.recall_optimizer_: RecallOptimizer | None = None
        self.classes_: np.ndarray | None = None
        self._is_fitted: bool = False

    def _create_base_model(self) -> Any:
        """Create the base model instance."""
        if self.base_model == "logistic":
            return LogisticRegression(
                class_weight=self.class_weight,
                random_state=self.random_state,
                max_iter=1000,
                solver="lbfgs",
                **self.model_params,
            )
        elif self.base_model == "lightgbm":
            try:
                import lightgbm as lgb

                return lgb.LGBMClassifier(
                    class_weight=self.class_weight,
                    random_state=self.random_state,
                    n_estimators=100,
                    verbose=-1,
                    **self.model_params,
                )
            except ImportError as err:
                raise ImportError("lightgbm not installed. Use pip install lightgbm") from err
        elif self.base_model == "xgboost":
            try:
                import xgboost as xgb

                # XGBoost uses scale_pos_weight for imbalanced classes
                return xgb.XGBClassifier(
                    random_state=self.random_state,
                    n_estimators=100,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    **self.model_params,
                )
            except ImportError as err:
                raise ImportError("xgboost not installed. Use pip install xgboost") from err
        elif self.base_model == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                class_weight=self.class_weight,
                random_state=self.random_state,
                n_estimators=100,
                **self.model_params,
            )
        else:
            raise ValueError(f"Unknown base_model: {self.base_model}")

    def _convert_to_binary(self, y: np.ndarray) -> npt.NDArray[np.int_]:
        """
        Convert multi-class labels to binary (has signal vs no signal).

        Maps: {-1, 1} -> 1 (has signal), {0} -> 0 (no signal)
        """
        y_arr = np.asarray(y).ravel()
        return np.asarray((y_arr != 0).astype(int))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "PrimaryClassifier":
        """
        Fit the primary classifier.

        Training proceeds in two stages:
        1. Fit base model on binary "has signal" prediction
        2. Optimize threshold to achieve target recall

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Labels, values in {-1, 0, 1}
            sample_weight: Optional sample weights

        Returns:
            self

        Raises:
            ValueError: If input validation fails
        """
        # Validate inputs
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        if X.shape[0] != len(y):
            raise ValueError(
                f"X and y must have same number of samples, got {X.shape[0]} and {len(y)}"
            )

        if len(y) == 0:
            raise ValueError("Empty training data")

        # Store original classes
        self.classes_ = np.unique(y)

        # Convert to binary for training
        y_binary = self._convert_to_binary(y)

        if self.verbose:
            n_signal = y_binary.sum()
            n_total = len(y_binary)
            logger.info(
                f"PrimaryClassifier training: {n_total} samples, "
                f"{n_signal} with signal ({n_signal/n_total:.1%})"
            )

        # Create and fit base model
        base_model = self._create_base_model()

        if self.calibrate:
            # Calibrate probabilities using CV
            base_model = CalibratedClassifierCV(
                base_model,
                method="isotonic",
                cv=min(self.cv_folds, len(np.unique(y_binary))),
            )

        if sample_weight is not None:
            base_model.fit(X, y_binary, sample_weight=sample_weight)
        else:
            base_model.fit(X, y_binary)

        self.base_model_ = base_model

        # Get cross-validated probabilities for threshold optimization
        try:
            y_proba_cv = cross_val_predict(
                self._create_base_model(),  # Use fresh model for CV
                X,
                y_binary,
                cv=min(self.cv_folds, len(np.unique(y_binary))),
                method="predict_proba",
            )[:, 1]
        except Exception as e:
            logger.warning(f"CV prediction failed: {e}. Using training probabilities.")
            y_proba_cv = base_model.predict_proba(X)[:, 1]

        # Optimize threshold for target recall
        self.recall_optimizer_ = RecallOptimizer(recall_target=self.recall_target)
        self.recall_optimizer_.fit(y_binary, y_proba_cv)

        self._is_fitted = True

        if self.verbose:
            logger.info(
                f"PrimaryClassifier fitted: threshold={self.recall_optimizer_.optimal_threshold:.3f}, "
                f"recall={self.recall_optimizer_.achieved_recall:.2%}, "
                f"precision={self.recall_optimizer_.achieved_precision:.2%}"
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels (has signal vs no signal).

        Uses the optimized threshold for high recall.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Binary predictions, values in {0, 1}
        """
        self._check_is_fitted()
        X = np.asarray(X)

        proba = self.base_model_.predict_proba(X)[:, 1]
        if self.recall_optimizer_ is None:
            raise RuntimeError("Model not fitted yet")
        return self.recall_optimizer_.apply_threshold(proba)

    def predict_proba(self, X: np.ndarray) -> npt.NDArray[np.float64]:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Probability matrix, shape (n_samples, 2)
            Column 0: P(no signal), Column 1: P(has signal)
        """
        self._check_is_fitted()
        X = np.asarray(X)
        return np.asarray(self.base_model_.predict_proba(X))

    def predict_side(
        self,
        X: np.ndarray,
        y_direction: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Predict trading side (-1 or 1).

        This is the primary output for meta-labeling. If y_direction is provided,
        uses it to determine the side for samples predicted to have a signal.
        Otherwise, returns 1 for all predicted signals.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y_direction: Optional true direction labels {-1, 0, 1}
                        Used to determine side when signal is predicted

        Returns:
            Side predictions, values in {-1, 1}
            Samples with no predicted signal get side=0 (will be filtered)
        """
        self._check_is_fitted()

        # Get binary predictions (has signal)
        has_signal = self.predict(X)

        if y_direction is not None:
            # Use true direction for samples predicted to have signal
            y_direction = np.asarray(y_direction).ravel()
            sides = np.where(has_signal == 1, np.sign(y_direction), 0)
        else:
            # Default to long for all signals (caller should provide direction)
            sides = has_signal.copy()
            logger.warning(
                "predict_side called without y_direction. "
                "Returning binary has_signal instead of actual sides."
            )

        return sides.astype(int)

    def _check_is_fitted(self) -> None:
        """Check if the model has been fitted."""
        if not self._is_fitted:
            raise RuntimeError("PrimaryClassifier has not been fitted. Call fit() first.")

    def get_recall_metrics(self) -> dict[str, float]:
        """
        Get recall optimization metrics.

        Returns:
            Dictionary with threshold, recall, and precision
        """
        self._check_is_fitted()
        if self.recall_optimizer_ is None:
            raise RuntimeError("Model not fitted yet")
        # Provide defaults for optional float values (set after fit())
        threshold = self.recall_optimizer_.optimal_threshold
        recall = self.recall_optimizer_.achieved_recall
        precision = self.recall_optimizer_.achieved_precision
        return {
            "optimal_threshold": float(threshold) if threshold is not None else 0.5,
            "achieved_recall": float(recall) if recall is not None else 0.0,
            "achieved_precision": float(precision) if precision is not None else 0.0,
            "recall_target": self.recall_target,
        }
