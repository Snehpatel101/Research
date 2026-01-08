"""
Calibrated meta-learner using Isotonic or Platt scaling.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

from ...base import BaseModel, PredictionOutput, TrainingMetrics
from ...common import map_classes_to_labels, map_labels_to_classes
from ...registry import register

logger = logging.getLogger(__name__)


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


__all__ = ["CalibratedMetaLearner"]
