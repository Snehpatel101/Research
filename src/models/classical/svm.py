"""
SVM Model - Support Vector Machine for 3-class prediction.

CPU-only implementation using scikit-learn. Provides probability
estimates via Platt scaling and sample weight support.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.svm import SVC

from ..base import BaseModel, PredictionOutput, TrainingMetrics
from ..common import map_classes_to_labels, map_labels_to_classes
from ..registry import register

logger = logging.getLogger(__name__)


@register(
    name="svm",
    family="classical",
    description="Support Vector Machine classifier (CPU-only)",
    aliases=["svc"],
)
class SVMModel(BaseModel):
    """
    Support Vector Machine classifier with sample weight support.

    Uses scikit-learn's SVC with RBF kernel. Probability estimates
    are enabled via Platt scaling. Requires scaled features.

    Note: SVM can be slow on large datasets. Consider using a
    subset of training data or LinearSVC for very large datasets.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._model: SVC | None = None
        self._feature_names: list[str] | None = None
        self._n_classes: int = 3

    @property
    def model_family(self) -> str:
        return "classical"

    @property
    def requires_scaling(self) -> bool:
        # SVM requires scaling for optimal performance
        return True

    @property
    def requires_sequences(self) -> bool:
        return False

    def get_default_config(self) -> dict[str, Any]:
        return {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "class_weight": "balanced",
            "probability": True,
            "max_iter": 10000,
            "cache_size": 1000,  # MB
            "random_state": 42,
            "tol": 1e-3,
            "verbose": False,
            "decision_function_shape": "ovr",
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: np.ndarray | None = None,
        config: dict[str, Any] | None = None,
    ) -> TrainingMetrics:
        """
        Train SVM model.

        Note: SVM does not use early stopping. Validation set is
        used only for computing validation metrics.

        Warning: Training time scales O(n^2) to O(n^3) with sample count.
        Consider subsampling for large datasets.
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
        self._model = SVC(
            kernel=train_config.get("kernel", "rbf"),
            C=train_config.get("C", 1.0),
            gamma=train_config.get("gamma", "scale"),
            class_weight=train_config.get("class_weight", "balanced"),
            probability=train_config.get("probability", True),
            max_iter=train_config.get("max_iter", 10000),
            cache_size=train_config.get("cache_size", 1000),
            random_state=train_config.get("random_state", 42),
            tol=train_config.get("tol", 1e-3),
            verbose=train_config.get("verbose", False),
            decision_function_shape=train_config.get("decision_function_shape", "ovr"),
        )

        n_samples = len(X_train)
        logger.info(
            f"Training SVM: kernel={train_config.get('kernel', 'rbf')}, "
            f"C={train_config.get('C', 1.0)}, "
            f"gamma={train_config.get('gamma', 'scale')}, "
            f"n_samples={n_samples}"
        )

        if n_samples > 50000:
            logger.warning(
                f"SVM training on {n_samples} samples may be slow. "
                f"Consider subsampling or using a different model."
            )

        # Train model
        self._model.fit(X_train, y_train_sk, sample_weight=sample_weights)

        training_time = time.time() - start_time

        # Compute metrics
        train_metrics = self._compute_metrics(X_train, y_train)
        val_metrics = self._compute_metrics(X_val, y_val)

        # Compute log loss (requires probability=True)
        train_loss = 0.0
        val_loss = 0.0
        if train_config.get("probability", True):
            train_proba = self._model.predict_proba(X_train)
            val_proba = self._model.predict_proba(X_val)
            train_loss = float(log_loss(y_train_sk, train_proba))
            val_loss = float(log_loss(y_val_sk, val_proba))

        self._is_fitted = True

        # Get number of support vectors
        n_support_vectors = sum(self._model.n_support_)

        logger.info(
            f"Training complete: n_support_vectors={n_support_vectors}, "
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
            epochs_trained=1,  # SVM trains in one pass
            training_time_seconds=training_time,
            early_stopped=False,
            best_epoch=None,
            history={},
            metadata={
                "n_features": X_train.shape[1],
                "n_train_samples": len(X_train),
                "n_val_samples": len(X_val),
                "n_support_vectors": n_support_vectors,
                "n_support_per_class": self._model.n_support_.tolist(),
                "converged": self._model.fit_status_ == 0,
            },
        )

    def predict(self, X: np.ndarray) -> PredictionOutput:
        """Generate predictions with class probabilities."""
        self._validate_fitted()
        self._validate_input_shape(X, "X")

        # Check if probability estimation is enabled
        if hasattr(self._model, "predict_proba"):
            probabilities = self._model.predict_proba(X)
        else:
            # Fallback: use decision function and softmax
            decision = self._model.decision_function(X)
            exp_decision = np.exp(decision - decision.max(axis=1, keepdims=True))
            probabilities = exp_decision / exp_decision.sum(axis=1, keepdims=True)

        class_predictions_sk = np.argmax(probabilities, axis=1)
        class_predictions = self._convert_labels_from_sklearn(class_predictions_sk)
        confidence = np.max(probabilities, axis=1)

        return PredictionOutput(
            class_predictions=class_predictions,
            class_probabilities=probabilities,
            confidence=confidence,
            metadata={"model": "svm"},
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return raw class probabilities."""
        self._validate_fitted()
        self._validate_input_shape(X, "X")

        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        else:
            # Fallback using decision function
            decision = self._model.decision_function(X)
            exp_decision = np.exp(decision - decision.max(axis=1, keepdims=True))
            return exp_decision / exp_decision.sum(axis=1, keepdims=True)

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

        logger.info(f"Saved SVM model to {path}")

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
        logger.info(f"Loaded SVM model from {path}")

    def get_feature_importance(self) -> dict[str, float] | None:
        """
        Return feature importances.

        Note: For non-linear kernels (RBF, poly), feature importance
        is not directly available. Returns None in these cases.
        For linear kernel, returns absolute coefficient values.
        """
        if not self._is_fitted:
            return None

        # Only linear kernel has interpretable coefficients
        if self._model.kernel != "linear":
            logger.debug(f"Feature importance not available for {self._model.kernel} kernel")
            return None

        # For linear kernel, average absolute coefficients across classes
        coefs = np.abs(self._model.coef_).mean(axis=0)
        feature_names = self._feature_names or [f"f{i}" for i in range(len(coefs))]

        return dict(zip(feature_names, coefs.tolist(), strict=False))

    def set_feature_names(self, names: list[str]) -> None:
        """Set feature names for interpretability."""
        self._feature_names = names

    def get_support_vectors(self) -> np.ndarray | None:
        """Return the support vectors."""
        if not self._is_fitted:
            return None
        return self._model.support_vectors_

    def get_support_indices(self) -> np.ndarray | None:
        """Return indices of support vectors."""
        if not self._is_fitted:
            return None
        return self._model.support_

    def _convert_labels_to_sklearn(self, labels: np.ndarray) -> np.ndarray:
        """Convert labels from -1,0,1 to 0,1,2."""
        return map_labels_to_classes(labels)

    def _convert_labels_from_sklearn(self, labels: np.ndarray) -> np.ndarray:
        """Convert labels from 0,1,2 to -1,0,1."""
        return map_classes_to_labels(labels)

    def _compute_metrics(self, X: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
        """Compute accuracy and F1 for a dataset."""
        y_pred_sk = self._model.predict(X)
        y_pred = self._convert_labels_from_sklearn(y_pred_sk)

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }


__all__ = ["SVMModel"]
