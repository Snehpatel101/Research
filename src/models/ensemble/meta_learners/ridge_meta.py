"""
Ridge regression meta-learner for stacking ensembles.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

from ...base import BaseModel, PredictionOutput, TrainingMetrics
from ...common import map_classes_to_labels, map_labels_to_classes
from ...registry import register
from .base import softmax

logger = logging.getLogger(__name__)


@register(
    name="ridge_meta",
    family="ensemble",
    description="Ridge regression meta-learner for combining OOF predictions",
    aliases=["ridge_meta_learner", "ridge_stacking"],
)
class RidgeMetaLearner(BaseModel):
    """
    Ridge regression meta-learner for stacking ensembles.

    Uses Ridge regularization (L2) to combine base model OOF predictions
    into final class predictions. Effective for linear combination of
    well-calibrated base models.

    Input shape: (n_samples, n_base_models * n_classes) for probability inputs
                 or (n_samples, n_base_models) for class predictions

    Advantages:
    - Fast training, closed-form solution
    - Robust to multicollinearity in base model predictions
    - Interpretable weights show relative model contribution
    - Effective when base models are well-calibrated

    Example:
        meta = RidgeMetaLearner(config={"alpha": 1.0})
        meta.fit(oof_features, y_train, oof_val, y_val)
        output = meta.predict(stacking_features)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._model: RidgeClassifier | None = None
        self._scaler: StandardScaler | None = None
        self._feature_names: list[str] | None = None
        self._n_classes: int = 3

    @property
    def model_family(self) -> str:
        return "ensemble"

    @property
    def requires_scaling(self) -> bool:
        # Internal scaling is handled
        return False

    @property
    def requires_sequences(self) -> bool:
        return False

    def get_default_config(self) -> dict[str, Any]:
        return {
            "alpha": 1.0,  # Regularization strength
            "fit_intercept": True,
            "class_weight": "balanced",
            "random_state": 42,
            "tol": 1e-4,
            "solver": "auto",  # 'auto', 'svd', 'cholesky', 'lsqr', etc.
            "scale_features": True,  # Scale input features internally
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
        Train Ridge meta-learner on OOF predictions.

        Args:
            X_train: OOF predictions, shape (n_samples, n_features)
            y_train: True labels (-1, 0, 1)
            X_val: Validation OOF predictions
            y_val: Validation labels
            sample_weights: Optional sample weights
            config: Optional config overrides
        """
        self._validate_input_shape(X_train, "X_train")
        self._validate_input_shape(X_val, "X_val")
        start_time = time.time()

        train_config = self._config.copy()
        if config:
            train_config.update(config)

        # Convert labels: -1,0,1 -> 0,1,2
        y_train_sk = map_labels_to_classes(y_train)
        y_val_sk = map_labels_to_classes(y_val)

        # Optional feature scaling
        X_train_scaled = X_train
        X_val_scaled = X_val
        if train_config.get("scale_features", True):
            self._scaler = StandardScaler()
            X_train_scaled = self._scaler.fit_transform(X_train)
            X_val_scaled = self._scaler.transform(X_val)

        # Build Ridge classifier
        self._model = RidgeClassifier(
            alpha=train_config.get("alpha", 1.0),
            fit_intercept=train_config.get("fit_intercept", True),
            class_weight=train_config.get("class_weight", "balanced"),
            random_state=train_config.get("random_state", 42),
            tol=train_config.get("tol", 1e-4),
            solver=train_config.get("solver", "auto"),
        )

        logger.info(
            f"Training RidgeMetaLearner: alpha={train_config.get('alpha', 1.0)}, "
            f"n_features={X_train.shape[1]}"
        )

        # Train model
        self._model.fit(X_train_scaled, y_train_sk, sample_weight=sample_weights)

        training_time = time.time() - start_time

        # Compute metrics
        train_metrics = self._compute_metrics(X_train_scaled, y_train)
        val_metrics = self._compute_metrics(X_val_scaled, y_val)

        # Compute pseudo-loss using decision function distance
        train_loss = self._compute_loss(X_train_scaled, y_train_sk)
        val_loss = self._compute_loss(X_val_scaled, y_val_sk)

        self._is_fitted = True

        logger.info(
            f"Training complete: val_f1={val_metrics['f1']:.4f}, " f"time={training_time:.1f}s"
        )

        return TrainingMetrics(
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_metrics["accuracy"],
            val_accuracy=val_metrics["accuracy"],
            train_f1=train_metrics["f1"],
            val_f1=val_metrics["f1"],
            epochs_trained=1,
            training_time_seconds=training_time,
            early_stopped=False,
            best_epoch=None,
            history={},
            metadata={
                "meta_learner": "ridge",
                "n_features": X_train.shape[1],
                "n_train_samples": len(X_train),
                "n_val_samples": len(X_val),
                "alpha": train_config.get("alpha", 1.0),
            },
        )

    def predict(self, X: np.ndarray) -> PredictionOutput:
        """Generate predictions with class probabilities."""
        self._validate_fitted()
        self._validate_input_shape(X, "X")

        # Scale if scaler was used during training
        X_scaled = X
        if self._scaler is not None:
            X_scaled = self._scaler.transform(X)

        # Get decision function and convert to pseudo-probabilities
        decision = self._model.decision_function(X_scaled)

        # Convert decision function to probabilities using softmax
        probabilities = softmax(decision)
        class_predictions_sk = np.argmax(probabilities, axis=1)
        class_predictions = map_classes_to_labels(class_predictions_sk)
        confidence = np.max(probabilities, axis=1)

        return PredictionOutput(
            class_predictions=class_predictions,
            class_probabilities=probabilities,
            confidence=confidence,
            metadata={"meta_learner": "ridge"},
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return pseudo-probabilities from decision function."""
        output = self.predict(X)
        return output.class_probabilities

    def save(self, path: Path) -> None:
        """Save model and metadata to directory."""
        self._validate_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self._model, path / "model.joblib")
        if self._scaler is not None:
            joblib.dump(self._scaler, path / "scaler.joblib")

        metadata = {
            "config": self._config,
            "feature_names": self._feature_names,
            "n_classes": self._n_classes,
        }
        joblib.dump(metadata, path / "metadata.joblib")

        logger.info(f"Saved RidgeMetaLearner to {path}")

    def load(self, path: Path) -> None:
        """Load model from directory."""
        path = Path(path)
        model_path = path / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._model = joblib.load(model_path)

        scaler_path = path / "scaler.joblib"
        if scaler_path.exists():
            self._scaler = joblib.load(scaler_path)

        metadata_path = path / "metadata.joblib"
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            self._config = metadata.get("config", self._config)
            self._feature_names = metadata.get("feature_names")
            self._n_classes = metadata.get("n_classes", 3)

        self._is_fitted = True
        logger.info(f"Loaded RidgeMetaLearner from {path}")

    def get_feature_importance(self) -> dict[str, float] | None:
        """Return coefficient magnitudes as feature importance."""
        if not self._is_fitted:
            return None

        # Average absolute coefficients across classes
        coefs = np.abs(self._model.coef_).mean(axis=0)
        feature_names = self._feature_names or [f"f{i}" for i in range(len(coefs))]

        return dict(zip(feature_names, coefs.tolist(), strict=False))

    def set_feature_names(self, names: list[str]) -> None:
        """Set feature names for interpretability."""
        self._feature_names = names

    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute hinge-like loss from decision function."""
        decision = self._model.decision_function(X)
        # Use negative log softmax as loss proxy
        probs = softmax(decision)
        return float(log_loss(y, probs))

    def _compute_metrics(self, X: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
        """Compute accuracy and F1 for a dataset."""
        y_pred_sk = self._model.predict(X)
        y_pred = map_classes_to_labels(y_pred_sk)

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }


__all__ = ["RidgeMetaLearner"]
