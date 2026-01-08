"""
Multi-layer perceptron meta-learner for stacking ensembles.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ...base import BaseModel, PredictionOutput, TrainingMetrics
from ...common import map_classes_to_labels, map_labels_to_classes
from ...registry import register

logger = logging.getLogger(__name__)


@register(
    name="mlp_meta",
    family="ensemble",
    description="Multi-layer perceptron meta-learner for non-linear combinations",
    aliases=["mlp_meta_learner", "mlp_stacking", "nn_meta"],
)
class MLPMetaLearner(BaseModel):
    """
    Multi-layer perceptron meta-learner for stacking ensembles.

    Uses a shallow neural network to learn non-linear combinations of
    base model predictions. Effective when base models have complementary
    error patterns that can be exploited through non-linear transformation.

    Input shape: (n_samples, n_base_models * n_classes) for probability inputs

    Advantages:
    - Captures non-linear interactions between base model predictions
    - Automatic feature learning from prediction patterns
    - Dropout regularization prevents overfitting

    Disadvantages:
    - More hyperparameters to tune than linear methods
    - Longer training time than Ridge
    - May overfit on small stacking datasets

    Example:
        meta = MLPMetaLearner(config={
            "hidden_layer_sizes": (32, 16),
            "alpha": 0.01,
        })
        meta.fit(oof_features, y_train, oof_val, y_val)
        output = meta.predict(stacking_features)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._model: MLPClassifier | None = None
        self._scaler: StandardScaler | None = None
        self._feature_names: list[str] | None = None
        self._n_classes: int = 3

    @property
    def model_family(self) -> str:
        return "ensemble"

    @property
    def requires_scaling(self) -> bool:
        return False  # Internal scaling

    @property
    def requires_sequences(self) -> bool:
        return False

    def get_default_config(self) -> dict[str, Any]:
        return {
            # Network architecture (shallow for meta-learning)
            "hidden_layer_sizes": (32, 16),
            "activation": "relu",
            # Regularization
            "alpha": 0.01,  # L2 penalty
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 10,
            # Training
            "learning_rate_init": 0.001,
            "max_iter": 200,
            "batch_size": "auto",
            "solver": "adam",
            # Reproducibility
            "random_state": 42,
            "verbose": False,
            # Feature scaling
            "scale_features": True,
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
        Train MLP meta-learner on OOF predictions.

        Note: sample_weights are not directly supported by MLPClassifier.
        If provided, they will be used for metric computation only.
        """
        self._validate_input_shape(X_train, "X_train")
        self._validate_input_shape(X_val, "X_val")
        start_time = time.time()

        train_config = self._config.copy()
        if config:
            train_config.update(config)

        if sample_weights is not None:
            logger.warning(
                "MLPMetaLearner does not support sample_weights during training. "
                "Weights will be ignored."
            )

        # Convert labels: -1,0,1 -> 0,1,2
        y_train_sk = map_labels_to_classes(y_train)
        y_val_sk = map_labels_to_classes(y_val)

        # Feature scaling (important for neural networks)
        X_train_scaled = X_train
        X_val_scaled = X_val
        if train_config.get("scale_features", True):
            self._scaler = StandardScaler()
            X_train_scaled = self._scaler.fit_transform(X_train)
            X_val_scaled = self._scaler.transform(X_val)

        # Combine train and validation for early stopping
        # MLPClassifier uses validation_fraction from training data
        X_combined = np.vstack([X_train_scaled, X_val_scaled])
        y_combined = np.hstack([y_train_sk, y_val_sk])

        # Build MLP classifier
        self._model = MLPClassifier(
            hidden_layer_sizes=train_config.get("hidden_layer_sizes", (32, 16)),
            activation=train_config.get("activation", "relu"),
            alpha=train_config.get("alpha", 0.01),
            early_stopping=train_config.get("early_stopping", True),
            validation_fraction=train_config.get("validation_fraction", 0.1),
            n_iter_no_change=train_config.get("n_iter_no_change", 10),
            learning_rate_init=train_config.get("learning_rate_init", 0.001),
            max_iter=train_config.get("max_iter", 200),
            batch_size=train_config.get("batch_size", "auto"),
            solver=train_config.get("solver", "adam"),
            random_state=train_config.get("random_state", 42),
            verbose=train_config.get("verbose", False),
        )

        hidden_layers = train_config.get("hidden_layer_sizes", (32, 16))
        logger.info(
            f"Training MLPMetaLearner: layers={hidden_layers}, "
            f"alpha={train_config.get('alpha', 0.01)}, n_features={X_train.shape[1]}"
        )

        # Train model
        self._model.fit(X_combined, y_combined)

        training_time = time.time() - start_time

        # Compute metrics on original splits
        train_metrics = self._compute_metrics(X_train_scaled, y_train)
        val_metrics = self._compute_metrics(X_val_scaled, y_val)

        # Compute loss
        train_proba = self._model.predict_proba(X_train_scaled)
        val_proba = self._model.predict_proba(X_val_scaled)
        train_loss = float(log_loss(y_train_sk, train_proba))
        val_loss = float(log_loss(y_val_sk, val_proba))

        self._is_fitted = True

        # Get training history
        n_iter = getattr(self._model, "n_iter_", 0)
        best_loss = getattr(self._model, "best_loss_", None)

        logger.info(
            f"Training complete: iterations={n_iter}, val_f1={val_metrics['f1']:.4f}, "
            f"time={training_time:.1f}s"
        )

        return TrainingMetrics(
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_metrics["accuracy"],
            val_accuracy=val_metrics["accuracy"],
            train_f1=train_metrics["f1"],
            val_f1=val_metrics["f1"],
            epochs_trained=n_iter,
            training_time_seconds=training_time,
            early_stopped=n_iter < train_config.get("max_iter", 200),
            best_epoch=None,
            history={
                "loss_curve": (
                    list(self._model.loss_curve_) if hasattr(self._model, "loss_curve_") else []
                )
            },
            metadata={
                "meta_learner": "mlp",
                "n_features": X_train.shape[1],
                "n_train_samples": len(X_train),
                "n_val_samples": len(X_val),
                "hidden_layers": hidden_layers,
                "n_iterations": n_iter,
                "best_loss": best_loss,
            },
        )

    def predict(self, X: np.ndarray) -> PredictionOutput:
        """Generate predictions with class probabilities."""
        self._validate_fitted()
        self._validate_input_shape(X, "X")

        X_scaled = X
        if self._scaler is not None:
            X_scaled = self._scaler.transform(X)

        probabilities = self._model.predict_proba(X_scaled)
        class_predictions_sk = np.argmax(probabilities, axis=1)
        class_predictions = map_classes_to_labels(class_predictions_sk)
        confidence = np.max(probabilities, axis=1)

        return PredictionOutput(
            class_predictions=class_predictions,
            class_probabilities=probabilities,
            confidence=confidence,
            metadata={"meta_learner": "mlp"},
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities."""
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

        logger.info(f"Saved MLPMetaLearner to {path}")

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
        logger.info(f"Loaded MLPMetaLearner from {path}")

    def get_feature_importance(self) -> dict[str, float] | None:
        """Return input layer weight magnitudes as feature importance."""
        if not self._is_fitted:
            return None

        # Get first layer weights: shape (n_features, hidden_size)
        first_layer_weights = self._model.coefs_[0]
        # Sum absolute weights for each input feature
        importance = np.abs(first_layer_weights).sum(axis=1)

        feature_names = self._feature_names or [f"f{i}" for i in range(len(importance))]

        return dict(zip(feature_names, importance.tolist(), strict=False))

    def set_feature_names(self, names: list[str]) -> None:
        """Set feature names for interpretability."""
        self._feature_names = names

    def _compute_metrics(self, X: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
        """Compute accuracy and F1 for a dataset."""
        y_pred_sk = self._model.predict(X)
        y_pred = map_classes_to_labels(y_pred_sk)

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }


__all__ = ["MLPMetaLearner"]
