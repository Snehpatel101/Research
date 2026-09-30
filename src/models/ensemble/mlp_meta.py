"""
Multi-layer perceptron meta-learner for stacking ensembles.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.core.utils.safe_pickle import safe_pickle_dump, safe_pickle_load

from ..base import BaseModel, PredictionResult, TrainingMetrics
from ..common import (
    full_class_probabilities,
    map_classes_to_labels,
    map_labels_to_classes,
)
from ..registry import register

logger = logging.getLogger(__name__)


@register(
    name="mlp_meta",
    family="meta_learner",
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
        self._epochs_used: int = 0

    @property
    def model_family(self) -> str:
        return "meta_learner"

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
            # Temporal early stopping on X_val log loss (best epoch restored)
            "early_stopping": True,
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

        Fits on X_train only. With ``early_stopping`` the network trains one
        epoch at a time and stops once X_val log loss has not improved for
        ``n_iter_no_change`` epochs, restoring the best epoch's weights; X_val
        is never trained on, so the reported val metrics are out-of-sample.
        With ``early_stopping=False`` it trains a fixed ``max_iter`` epochs.

        Note: sample_weights are not supported by MLPClassifier and are ignored.
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
        y_train_sk = map_labels_to_classes(y_train, self._n_classes)
        y_val_sk = map_labels_to_classes(y_val, self._n_classes)
        classes = np.arange(self._n_classes)

        # Feature scaling (important for neural networks), fit on train only
        X_train_scaled = X_train
        X_val_scaled = X_val
        if train_config.get("scale_features", True):
            self._scaler = StandardScaler()
            X_train_scaled = self._scaler.fit_transform(X_train)
            X_val_scaled = self._scaler.transform(X_val)

        # sklearn's own early stopping carves a RANDOM validation split out of
        # the training rows; epochs are driven here instead so stopping is
        # temporal (on X_val) and every train row is used for fitting.
        self._model = MLPClassifier(
            hidden_layer_sizes=train_config.get("hidden_layer_sizes", (32, 16)),
            activation=train_config.get("activation", "relu"),
            alpha=train_config.get("alpha", 0.01),
            early_stopping=False,
            learning_rate_init=train_config.get("learning_rate_init", 0.001),
            batch_size=train_config.get("batch_size", "auto"),
            solver=train_config.get("solver", "adam"),
            random_state=train_config.get("random_state", 42),
            verbose=train_config.get("verbose", False),
        )

        hidden_layers = train_config.get("hidden_layer_sizes", (32, 16))
        max_iter = int(train_config.get("max_iter", 200))
        early_stopping = bool(train_config.get("early_stopping", True))
        patience = int(train_config.get("n_iter_no_change", 10))
        logger.info(
            f"Training MLPMetaLearner: layers={hidden_layers}, "
            f"alpha={train_config.get('alpha', 0.01)}, n_features={X_train.shape[1]}, "
            f"early_stopping={early_stopping}"
        )

        val_loss_curve: list[float] = []
        best_loss = np.inf
        best_epoch = max_iter - 1
        best_weights: tuple[list[np.ndarray], list[np.ndarray]] | None = None
        for epoch in range(max_iter):
            self._model.partial_fit(X_train_scaled, y_train_sk, classes=classes)
            if not early_stopping:
                continue
            epoch_loss = float(
                log_loss(y_val_sk, self._model.predict_proba(X_val_scaled), labels=classes)
            )
            val_loss_curve.append(epoch_loss)
            if epoch_loss < best_loss:
                best_loss, best_epoch = epoch_loss, epoch
                best_weights = (
                    [w.copy() for w in self._model.coefs_],
                    [b.copy() for b in self._model.intercepts_],
                )
            elif epoch - best_epoch >= patience:
                break
        n_iter = len(val_loss_curve) if early_stopping else max_iter
        if best_weights is not None:
            self._model.coefs_, self._model.intercepts_ = best_weights
        self._epochs_used = best_epoch + 1

        training_time = time.time() - start_time

        self._is_fitted = True
        train_metrics = self._compute_metrics(X_train_scaled, y_train)
        val_metrics = self._compute_metrics(X_val_scaled, y_val)
        train_loss = float(
            log_loss(y_train_sk, self._model.predict_proba(X_train_scaled), labels=classes)
        )
        val_loss = float(
            log_loss(y_val_sk, self._model.predict_proba(X_val_scaled), labels=classes)
        )

        logger.info(
            f"Training complete: epochs={n_iter} (kept={self._epochs_used}), "
            f"val_f1={val_metrics['f1']:.4f}, time={training_time:.1f}s"
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
            early_stopped=early_stopping and n_iter < max_iter,
            best_epoch=best_epoch if early_stopping else None,
            history={"val_loss": val_loss_curve},
            metadata={
                "meta_learner": "mlp",
                "n_features": X_train.shape[1],
                "n_train_samples": len(X_train),
                "n_val_samples": len(X_val),
                "hidden_layers": hidden_layers,
                "n_iterations": n_iter,
                "best_loss": None if best_weights is None else best_loss,
            },
        )

    @property
    def uses_early_stopping(self) -> bool:
        """Whether fit() selects its stopping epoch on X_val."""
        return bool(self._config.get("early_stopping", True))

    def refit_config(self) -> dict[str, Any]:
        """Config for refitting on more rows: the chosen epoch count, fixed."""
        self._validate_fitted()
        return {"early_stopping": False, "max_iter": self._epochs_used}

    def predict(self, X: np.ndarray) -> PredictionResult:
        """Generate predictions with class probabilities."""
        self._validate_fitted()
        self._validate_input_shape(X, "X")

        if self._model is None:
            raise RuntimeError("Meta model is not fitted")

        X_scaled = X
        if self._scaler is not None:
            X_scaled = self._scaler.transform(X)

        probabilities = full_class_probabilities(
            self._model.predict_proba(X_scaled), self._model.classes_, self._n_classes
        )
        class_predictions_sk = np.argmax(probabilities, axis=1)
        class_predictions = map_classes_to_labels(class_predictions_sk, self._n_classes)
        confidence = np.max(probabilities, axis=1)

        return PredictionResult(
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

        safe_pickle_dump(self._model, path / "model.pkl")
        if self._scaler is not None:
            safe_pickle_dump(self._scaler, path / "scaler.pkl")

        metadata = {
            "config": self._config,
            "feature_names": self._feature_names,
            "n_classes": self._n_classes,
        }
        safe_pickle_dump(metadata, path / "metadata.pkl")

        logger.info(f"Saved MLPMetaLearner to {path}")

    def load(self, path: Path) -> None:
        """Load model from directory."""
        path = Path(path)
        model_path = path / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._model = safe_pickle_load(model_path)

        scaler_path = path / "scaler.pkl"
        if scaler_path.exists():
            self._scaler = safe_pickle_load(scaler_path)

        metadata_path = path / "metadata.pkl"
        if metadata_path.exists():
            metadata = safe_pickle_load(metadata_path)
            self._config = metadata.get("config", self._config)
            self._feature_names = metadata.get("feature_names")
            self._n_classes = metadata.get("n_classes", 3)

        self._is_fitted = True
        logger.info(f"Loaded MLPMetaLearner from {path}")

    def get_feature_importance(self) -> dict[str, float] | None:
        """Return input layer weight magnitudes as feature importance."""
        if not self._is_fitted or self._model is None:
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
        if self._model is None:
            raise RuntimeError("Meta model is not fitted")
        y_pred_sk = self._model.predict(X)
        y_pred = map_classes_to_labels(y_pred_sk, self._n_classes)

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }


__all__ = ["MLPMetaLearner"]
