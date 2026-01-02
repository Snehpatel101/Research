"""
Meta-learner models for stacking ensembles.

This module provides specialized meta-learner implementations designed to
combine out-of-fold (OOF) predictions from base models. These models accept
input of shape (n_samples, n_base_models * n_classes) and produce final
3-class predictions.

Available Meta-Learners:
- RidgeMetaLearner: Ridge regression for linear combination of predictions
- MLPMetaLearner: Multi-layer perceptron for non-linear combinations
- CalibratedMetaLearner: Calibration wrapper using Isotonic or Platt scaling
- XGBoostMeta: XGBoost as meta-learner for gradient boosted stacking

All meta-learners implement the BaseModel interface and support:
- Sample weights for imbalanced data
- Training on stacking datasets from OOF predictions
- Returning PredictionOutput with probabilities and confidence
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ..base import BaseModel, PredictionOutput, TrainingMetrics
from ..common import map_classes_to_labels, map_labels_to_classes
from ..registry import register

logger = logging.getLogger(__name__)


# =============================================================================
# RIDGE META-LEARNER
# =============================================================================


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
        probabilities = self._softmax(decision)
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

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute hinge-like loss from decision function."""
        decision = self._model.decision_function(X)
        # Use negative log softmax as loss proxy
        probs = self._softmax(decision)
        return float(log_loss(y, probs))

    def _compute_metrics(self, X: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
        """Compute accuracy and F1 for a dataset."""
        y_pred_sk = self._model.predict(X)
        y_pred = map_classes_to_labels(y_pred_sk)

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }


# =============================================================================
# MLP META-LEARNER
# =============================================================================


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


# =============================================================================
# CALIBRATED META-LEARNER
# =============================================================================


@register(
    name="calibrated_meta",
    family="ensemble",
    description="Calibration wrapper meta-learner using Isotonic/Platt scaling",
    aliases=["calibrated_meta_learner", "isotonic_meta", "platt_meta"],
)
class CalibratedMetaLearner(BaseModel):
    """
    Calibrated meta-learner using Isotonic or Platt scaling.

    Wraps a base classifier with probability calibration to ensure
    predicted probabilities reflect true class frequencies. Essential
    for trading applications where probability thresholds drive decisions.

    Calibration methods:
    - "isotonic": Non-parametric, monotonic calibration (recommended for >1000 samples)
    - "sigmoid": Platt scaling using logistic regression (better for small datasets)

    Input shape: (n_samples, n_base_models * n_classes) for probability inputs

    Advantages:
    - Produces well-calibrated probabilities
    - Works with any base classifier
    - Essential for threshold-based trading decisions

    Example:
        meta = CalibratedMetaLearner(config={
            "base_estimator": "logistic",
            "method": "isotonic",
            "cv": 5,
        })
        meta.fit(oof_features, y_train, oof_val, y_val)
        output = meta.predict(stacking_features)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._model: CalibratedClassifierCV | None = None
        self._scaler: StandardScaler | None = None
        self._feature_names: list[str] | None = None
        self._n_classes: int = 3
        self._base_estimator_name: str = ""

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
            # Base estimator (simple classifier to calibrate)
            "base_estimator": "logistic",  # "logistic", "ridge", "svm"
            "base_estimator_config": {},
            # Calibration method
            "method": "isotonic",  # "isotonic" or "sigmoid"
            "cv": 5,  # Cross-validation folds for calibration
            "ensemble": True,  # Average predictions from CV folds
            # Feature scaling
            "scale_features": True,
            # Random state
            "random_state": 42,
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
        Train calibrated meta-learner on OOF predictions.

        Note: CalibratedClassifierCV uses internal CV for calibration.
        The validation set is used for metric computation only.
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

        # Feature scaling
        X_train_scaled = X_train
        X_val_scaled = X_val
        if train_config.get("scale_features", True):
            self._scaler = StandardScaler()
            X_train_scaled = self._scaler.fit_transform(X_train)
            X_val_scaled = self._scaler.transform(X_val)

        # Create base estimator
        base_estimator = self._create_base_estimator(train_config)
        self._base_estimator_name = train_config.get("base_estimator", "logistic")

        # Create calibrated classifier
        method = train_config.get("method", "isotonic")
        cv = train_config.get("cv", 5)
        ensemble = train_config.get("ensemble", True)

        self._model = CalibratedClassifierCV(
            estimator=base_estimator,
            method=method,
            cv=cv,
            ensemble=ensemble,
        )

        logger.info(
            f"Training CalibratedMetaLearner: base={self._base_estimator_name}, "
            f"method={method}, cv={cv}, n_features={X_train.shape[1]}"
        )

        # Train model (sample weights passed to base estimator if supported)
        if sample_weights is not None:
            # CalibratedClassifierCV passes fit_params to base estimator
            self._model.fit(X_train_scaled, y_train_sk, sample_weight=sample_weights)
        else:
            self._model.fit(X_train_scaled, y_train_sk)

        training_time = time.time() - start_time

        # Compute metrics
        train_metrics = self._compute_metrics(X_train_scaled, y_train)
        val_metrics = self._compute_metrics(X_val_scaled, y_val)

        # Compute calibrated loss
        train_proba = self._model.predict_proba(X_train_scaled)
        val_proba = self._model.predict_proba(X_val_scaled)
        train_loss = float(log_loss(y_train_sk, train_proba))
        val_loss = float(log_loss(y_val_sk, val_proba))

        self._is_fitted = True

        logger.info(
            f"Training complete: val_f1={val_metrics['f1']:.4f}, "
            f"val_loss={val_loss:.4f}, time={training_time:.1f}s"
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
                "meta_learner": "calibrated",
                "base_estimator": self._base_estimator_name,
                "calibration_method": method,
                "cv_folds": cv,
                "n_features": X_train.shape[1],
                "n_train_samples": len(X_train),
                "n_val_samples": len(X_val),
            },
        )

    def predict(self, X: np.ndarray) -> PredictionOutput:
        """Generate predictions with calibrated probabilities."""
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
            metadata={
                "meta_learner": "calibrated",
                "base_estimator": self._base_estimator_name,
            },
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated class probabilities."""
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
            "base_estimator_name": self._base_estimator_name,
        }
        joblib.dump(metadata, path / "metadata.joblib")

        logger.info(f"Saved CalibratedMetaLearner to {path}")

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
            self._base_estimator_name = metadata.get("base_estimator_name", "logistic")

        self._is_fitted = True
        logger.info(f"Loaded CalibratedMetaLearner from {path}")

    def get_feature_importance(self) -> dict[str, float] | None:
        """Feature importance not available for calibrated classifier."""
        return None

    def set_feature_names(self, names: list[str]) -> None:
        """Set feature names for interpretability."""
        self._feature_names = names

    def _create_base_estimator(self, config: dict[str, Any]) -> Any:
        """Create the base estimator for calibration."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC

        estimator_name = config.get("base_estimator", "logistic")
        estimator_config = config.get("base_estimator_config", {})
        random_state = config.get("random_state", 42)

        if estimator_name == "logistic":
            return LogisticRegression(
                C=estimator_config.get("C", 1.0),
                solver="saga",
                max_iter=estimator_config.get("max_iter", 500),
                class_weight="balanced",
                random_state=random_state,
            )
        elif estimator_name == "ridge":
            return RidgeClassifier(
                alpha=estimator_config.get("alpha", 1.0),
                class_weight="balanced",
                random_state=random_state,
            )
        elif estimator_name == "svm":
            return SVC(
                C=estimator_config.get("C", 1.0),
                kernel=estimator_config.get("kernel", "rbf"),
                class_weight="balanced",
                random_state=random_state,
                probability=True,  # Required for calibration
            )
        else:
            raise ValueError(
                f"Unknown base_estimator: {estimator_name}. "
                f"Supported: 'logistic', 'ridge', 'svm'"
            )

    def _compute_metrics(self, X: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
        """Compute accuracy and F1 for a dataset."""
        y_pred_sk = self._model.predict(X)
        y_pred = map_classes_to_labels(y_pred_sk)

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }


# =============================================================================
# XGBOOST META-LEARNER
# =============================================================================


@register(
    name="xgboost_meta",
    family="ensemble",
    description="XGBoost gradient boosting as meta-learner for stacking",
    aliases=["xgb_meta", "xgboost_stacking"],
)
class XGBoostMeta(BaseModel):
    """
    XGBoost as meta-learner for stacking ensembles.

    Uses gradient boosted trees to learn complex, non-linear relationships
    between base model predictions. Particularly effective when base models
    have diverse error patterns that require tree-based decision boundaries.

    Input shape: (n_samples, n_base_models * n_classes) for probability inputs

    Advantages:
    - Captures complex non-linear interactions
    - Built-in regularization (L1, L2, tree constraints)
    - Feature importance for model contribution analysis
    - Handles imbalanced classes well

    Configuration Notes:
    - Uses shallow trees (max_depth=3-4) to prevent overfitting
    - Lower learning rate with early stopping
    - Sample weights supported for quality-weighted training

    Example:
        meta = XGBoostMeta(config={
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.05,
        })
        meta.fit(oof_features, y_train, oof_val, y_val)
        output = meta.predict(stacking_features)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._model: Any = None  # xgb.Booster
        self._feature_names: list[str] | None = None
        self._n_classes: int = 3
        self._use_gpu: bool = False

    @property
    def model_family(self) -> str:
        return "ensemble"

    @property
    def requires_scaling(self) -> bool:
        # XGBoost doesn't require scaling
        return False

    @property
    def requires_sequences(self) -> bool:
        return False

    def get_default_config(self) -> dict[str, Any]:
        return {
            # Shallow trees for meta-learning
            "n_estimators": 100,
            "max_depth": 3,  # Shallow to prevent overfitting
            "min_child_weight": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "learning_rate": 0.05,
            # Regularization
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            # Early stopping
            "early_stopping_rounds": 20,
            # Training
            "eval_metric": "mlogloss",
            "use_gpu": False,
            "random_state": 42,
            "verbosity": 0,
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
        """Train XGBoost meta-learner on OOF predictions."""
        import xgboost as xgb

        self._validate_input_shape(X_train, "X_train")
        self._validate_input_shape(X_val, "X_val")
        start_time = time.time()

        train_config = self._config.copy()
        if config:
            train_config.update(config)

        # GPU check
        self._use_gpu = train_config.get("use_gpu", False)
        if self._use_gpu:
            self._use_gpu = self._check_cuda_available()

        # Convert labels: -1,0,1 -> 0,1,2
        y_train_xgb = map_labels_to_classes(y_train)
        y_val_xgb = map_labels_to_classes(y_val)

        # Apply balanced class weights
        final_weights = sample_weights
        unique_classes, class_counts = np.unique(y_train_xgb, return_counts=True)
        n_samples = len(y_train_xgb)
        n_classes = len(unique_classes)
        class_weight_values = n_samples / (n_classes * class_counts)
        class_weight_dict = dict(zip(unique_classes, class_weight_values, strict=False))
        sample_class_weights = np.array([class_weight_dict[int(c)] for c in y_train_xgb])

        if sample_weights is not None:
            final_weights = sample_weights * sample_class_weights
        else:
            final_weights = sample_class_weights

        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train, label=y_train_xgb, weight=final_weights)
        dval = xgb.DMatrix(X_val, label=y_val_xgb)

        # Build parameters
        params = self._build_params(train_config)
        n_estimators = train_config.get("n_estimators", 100)
        early_stopping = train_config.get("early_stopping_rounds", 20)

        evals = [(dtrain, "train"), (dval, "val")]
        evals_result: dict[str, dict[str, list[float]]] = {}

        logger.info(
            f"Training XGBoostMeta: n_estimators={n_estimators}, "
            f"max_depth={params['max_depth']}, n_features={X_train.shape[1]}"
        )

        self._model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=evals,
            early_stopping_rounds=early_stopping,
            evals_result=evals_result,
            verbose_eval=False,
        )

        training_time = time.time() - start_time

        # Extract metrics
        metric_name = train_config.get("eval_metric", "mlogloss")
        train_losses = evals_result.get("train", {}).get(metric_name, [])
        val_losses = evals_result.get("val", {}).get(metric_name, [])

        best_iteration = self._model.best_iteration
        epochs_trained = best_iteration + 1

        # Compute accuracy and F1
        train_metrics = self._compute_metrics(dtrain, y_train)
        val_metrics = self._compute_metrics(dval, y_val)

        self._is_fitted = True

        logger.info(
            f"Training complete: epochs={epochs_trained}, val_f1={val_metrics['f1']:.4f}, "
            f"time={training_time:.1f}s"
        )

        return TrainingMetrics(
            train_loss=train_losses[-1] if train_losses else 0.0,
            val_loss=val_losses[-1] if val_losses else 0.0,
            train_accuracy=train_metrics["accuracy"],
            val_accuracy=val_metrics["accuracy"],
            train_f1=train_metrics["f1"],
            val_f1=val_metrics["f1"],
            epochs_trained=epochs_trained,
            training_time_seconds=training_time,
            early_stopped=epochs_trained < n_estimators,
            best_epoch=best_iteration,
            history={f"train_{metric_name}": train_losses, f"val_{metric_name}": val_losses},
            metadata={
                "meta_learner": "xgboost",
                "best_iteration": best_iteration,
                "n_features": X_train.shape[1],
                "n_train_samples": len(X_train),
                "n_val_samples": len(X_val),
                "use_gpu": self._use_gpu,
            },
        )

    def predict(self, X: np.ndarray) -> PredictionOutput:
        """Generate predictions with class probabilities."""
        import xgboost as xgb

        self._validate_fitted()
        self._validate_input_shape(X, "X")

        dmatrix = xgb.DMatrix(X)
        probabilities = self._model.predict(dmatrix)
        class_predictions_xgb = np.argmax(probabilities, axis=1)
        class_predictions = map_classes_to_labels(class_predictions_xgb)
        confidence = np.max(probabilities, axis=1)

        return PredictionOutput(
            class_predictions=class_predictions,
            class_probabilities=probabilities,
            confidence=confidence,
            metadata={"meta_learner": "xgboost"},
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities."""
        output = self.predict(X)
        return output.class_probabilities

    def save(self, path: Path) -> None:
        """Save model to directory."""
        import pickle

        self._validate_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._model.save_model(str(path / "model.json"))

        metadata = {
            "config": self._config,
            "feature_names": self._feature_names,
            "n_classes": self._n_classes,
            "use_gpu": self._use_gpu,
        }
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Saved XGBoostMeta to {path}")

    def load(self, path: Path) -> None:
        """Load model from directory."""
        import pickle

        import xgboost as xgb

        path = Path(path)
        model_path = path / "model.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._model = xgb.Booster()
        self._model.load_model(str(model_path))

        metadata_path = path / "metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            self._config = metadata.get("config", self._config)
            self._feature_names = metadata.get("feature_names")
            self._n_classes = metadata.get("n_classes", 3)
            self._use_gpu = metadata.get("use_gpu", False)

        self._is_fitted = True
        logger.info(f"Loaded XGBoostMeta from {path}")

    def get_feature_importance(self) -> dict[str, float] | None:
        """Return feature importances by gain."""
        if not self._is_fitted:
            return None

        importance = self._model.get_score(importance_type="gain")

        if self._feature_names:
            result = {}
            for key, value in importance.items():
                if key.startswith("f"):
                    idx = int(key[1:])
                    if idx < len(self._feature_names):
                        result[self._feature_names[idx]] = value
                else:
                    result[key] = value
            return result

        return importance

    def set_feature_names(self, names: list[str]) -> None:
        """Set feature names for interpretability."""
        self._feature_names = names

    def _build_params(self, config: dict[str, Any]) -> dict[str, Any]:
        """Build XGBoost parameter dict."""
        params = {
            "objective": "multi:softprob",
            "num_class": self._n_classes,
            "eval_metric": config.get("eval_metric", "mlogloss"),
            "max_depth": config.get("max_depth", 3),
            "min_child_weight": config.get("min_child_weight", 10),
            "subsample": config.get("subsample", 0.8),
            "colsample_bytree": config.get("colsample_bytree", 0.8),
            "eta": config.get("learning_rate", 0.05),
            "gamma": config.get("gamma", 0.1),
            "alpha": config.get("reg_alpha", 0.1),
            "lambda": config.get("reg_lambda", 1.0),
            "tree_method": "hist",
            "seed": config.get("random_state", 42),
            "verbosity": config.get("verbosity", 0),
        }

        if self._use_gpu:
            params["device"] = "cuda"
        else:
            params["device"] = "cpu"

        return params

    def _check_cuda_available(self) -> bool:
        """Check if CUDA is available for XGBoost."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _compute_metrics(self, dmatrix: Any, y_true: np.ndarray) -> dict[str, float]:
        """Compute accuracy and F1 for a dataset."""
        probabilities = self._model.predict(dmatrix)
        y_pred_xgb = np.argmax(probabilities, axis=1)
        y_pred = map_classes_to_labels(y_pred_xgb)

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }


__all__ = [
    "RidgeMetaLearner",
    "MLPMetaLearner",
    "CalibratedMetaLearner",
    "XGBoostMeta",
]
