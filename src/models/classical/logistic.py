"""
Logistic Regression Model - Linear classifier for 3-class prediction.

CPU-only implementation using scikit-learn. Useful as a meta-learner
for stacking ensembles due to its simplicity and calibrated outputs.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss

from ..base import BaseModel, PredictionOutput, TrainingMetrics
from ..registry import register

logger = logging.getLogger(__name__)

# Label mapping: -1,0,1 (short/neutral/long) -> 0,1,2 (sklearn classes)
LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}
CLASS_TO_LABEL = {0: -1, 1: 0, 2: 1}


@register(
    name="logistic",
    family="classical",
    description="Logistic Regression linear classifier (CPU-only)",
    aliases=["lr", "logistic_regression"],
)
class LogisticModel(BaseModel):
    """
    Logistic Regression classifier with sample weight support.

    Uses scikit-learn's LogisticRegression with L2 regularization.
    Requires scaled features for optimal performance.
    Commonly used as a meta-learner in stacking ensembles.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._model: Optional[LogisticRegression] = None
        self._feature_names: Optional[List[str]] = None
        self._n_classes: int = 3

    @property
    def model_family(self) -> str:
        return "classical"

    @property
    def requires_scaling(self) -> bool:
        # Logistic Regression benefits from scaling
        return True

    @property
    def requires_sequences(self) -> bool:
        return False

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 500,
            "multi_class": "multinomial",
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": 42,
            "tol": 1e-4,
            "verbose": 0,
            "warm_start": False,
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> TrainingMetrics:
        """
        Train Logistic Regression model.

        Note: Logistic Regression does not use early stopping.
        Validation set is used only for computing validation metrics.
        """
        self._validate_input_shape(X_train, "X_train")
        self._validate_input_shape(X_val, "X_val")
        start_time = time.time()

        # Merge config
        train_config = self._config.copy()
        if config:
            train_config.update(config)

        # Convert labels: -1,0,1 -> 0,1,2
        y_train_sk = self._convert_labels_to_sklearn(y_train)
        y_val_sk = self._convert_labels_to_sklearn(y_val)

        # Build model
        self._model = LogisticRegression(
            penalty=train_config.get("penalty", "l2"),
            C=train_config.get("C", 1.0),
            solver=train_config.get("solver", "lbfgs"),
            max_iter=train_config.get("max_iter", 500),
            multi_class=train_config.get("multi_class", "multinomial"),
            class_weight=train_config.get("class_weight", "balanced"),
            n_jobs=train_config.get("n_jobs", -1),
            random_state=train_config.get("random_state", 42),
            tol=train_config.get("tol", 1e-4),
            verbose=train_config.get("verbose", 0),
            warm_start=train_config.get("warm_start", False),
        )

        logger.info(
            f"Training LogisticRegression: C={train_config.get('C', 1.0)}, "
            f"solver={train_config.get('solver', 'lbfgs')}, "
            f"max_iter={train_config.get('max_iter', 500)}"
        )

        # Train model
        self._model.fit(X_train, y_train_sk, sample_weight=sample_weights)

        training_time = time.time() - start_time

        # Compute metrics
        train_metrics = self._compute_metrics(X_train, y_train)
        val_metrics = self._compute_metrics(X_val, y_val)

        # Compute log loss
        train_proba = self._model.predict_proba(X_train)
        val_proba = self._model.predict_proba(X_val)
        train_loss = float(log_loss(y_train_sk, train_proba))
        val_loss = float(log_loss(y_val_sk, val_proba))

        self._is_fitted = True

        # Get number of iterations used
        n_iter = getattr(self._model, "n_iter_", [0])
        n_iterations = n_iter[0] if isinstance(n_iter, np.ndarray) else n_iter

        logger.info(
            f"Training complete: iterations={n_iterations}, "
            f"val_f1={val_metrics['f1']:.4f}, "
            f"time={training_time:.1f}s"
        )

        return TrainingMetrics(
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_metrics["accuracy"],
            val_accuracy=val_metrics["accuracy"],
            train_f1=train_metrics["f1"],
            val_f1=val_metrics["f1"],
            epochs_trained=int(n_iterations),
            training_time_seconds=training_time,
            early_stopped=int(n_iterations) < train_config.get("max_iter", 500),
            best_epoch=None,
            history={},
            metadata={
                "n_features": X_train.shape[1],
                "n_train_samples": len(X_train),
                "n_val_samples": len(X_val),
                "n_iterations": int(n_iterations),
                "converged": int(n_iterations) < train_config.get("max_iter", 500),
            },
        )

    def predict(self, X: np.ndarray) -> PredictionOutput:
        """Generate predictions with class probabilities."""
        self._validate_fitted()
        self._validate_input_shape(X, "X")

        probabilities = self._model.predict_proba(X)
        class_predictions_sk = np.argmax(probabilities, axis=1)
        class_predictions = self._convert_labels_from_sklearn(class_predictions_sk)
        confidence = np.max(probabilities, axis=1)

        return PredictionOutput(
            class_predictions=class_predictions,
            class_probabilities=probabilities,
            confidence=confidence,
            metadata={"model": "logistic"},
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return raw class probabilities."""
        self._validate_fitted()
        self._validate_input_shape(X, "X")
        return self._model.predict_proba(X)

    def save(self, path: Path) -> None:
        """Save model and metadata to directory."""
        self._validate_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self._model, path / "model.joblib")

        metadata = {
            "config": self._config,
            "feature_names": self._feature_names,
            "n_classes": self._n_classes,
        }
        joblib.dump(metadata, path / "metadata.joblib")

        logger.info(f"Saved LogisticRegression model to {path}")

    def load(self, path: Path) -> None:
        """Load model from directory."""
        path = Path(path)
        model_path = path / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._model = joblib.load(model_path)

        metadata_path = path / "metadata.joblib"
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            self._config = metadata.get("config", self._config)
            self._feature_names = metadata.get("feature_names")
            self._n_classes = metadata.get("n_classes", 3)

        self._is_fitted = True
        logger.info(f"Loaded LogisticRegression model from {path}")

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Return feature importances (absolute coefficient values)."""
        if not self._is_fitted:
            return None

        # For multi-class, average absolute coefficients across classes
        coefs = np.abs(self._model.coef_).mean(axis=0)
        feature_names = self._feature_names or [
            f"f{i}" for i in range(len(coefs))
        ]

        return dict(zip(feature_names, coefs.tolist()))

    def set_feature_names(self, names: List[str]) -> None:
        """Set feature names for interpretability."""
        self._feature_names = names

    def get_coefficients(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Return raw model coefficients per class.

        Returns:
            Dict with 'coef' (n_classes, n_features) and 'intercept' (n_classes,)
        """
        if not self._is_fitted:
            return None

        return {
            "coef": self._model.coef_,
            "intercept": self._model.intercept_,
        }

    def _convert_labels_to_sklearn(self, labels: np.ndarray) -> np.ndarray:
        """Convert labels from -1,0,1 to 0,1,2."""
        return np.array([LABEL_TO_CLASS.get(int(l), 1) for l in labels])

    def _convert_labels_from_sklearn(self, labels: np.ndarray) -> np.ndarray:
        """Convert labels from 0,1,2 to -1,0,1."""
        return np.array([CLASS_TO_LABEL.get(int(l), 0) for l in labels])

    def _compute_metrics(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Compute accuracy and F1 for a dataset."""
        y_pred_sk = self._model.predict(X)
        y_pred = self._convert_labels_from_sklearn(y_pred_sk)

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }


__all__ = ["LogisticModel"]
