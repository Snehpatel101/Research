"""
L2-regularized logistic (ridge) meta-learner for stacking ensembles.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

from src.core.utils.safe_pickle import safe_pickle_dump, safe_pickle_load

from ..base import BaseModel, PredictionResult, TrainingMetrics
from ..common import map_classes_to_labels, map_labels_to_classes
from ..registry import register

logger = logging.getLogger(__name__)


@register(
    name="ridge_meta",
    family="meta_learner",
    description="L2-regularized logistic (ridge) meta-learner",
    aliases=["ridge_meta_learner", "ridge_stacking"],
)
class RidgeMetaLearner(BaseModel):
    """
    L2-regularized (ridge) multinomial logistic regression meta-learner.

    Combines base-model OOF probabilities with a linear model whose output is
    a proper probability distribution (softmax of a model fit by maximum
    likelihood), so ensemble probabilities can drive position sizing and
    conformal sets directly.

    Config:
        C: Inverse L2 strength (default 1.0; smaller = stronger shrinkage).
        class_weight: None (default) keeps the empirical class priors in the
            probabilities; "balanced" reweights classes (probabilities are
            then no longer calibrated to the base rates).

    Input shape: (n_samples, n_base_models * n_classes + derived features)

    Example:
        meta = RidgeMetaLearner(config={"C": 0.5})
        meta.fit(oof_features, y_train, oof_val, y_val)
        output = meta.predict(stacking_features)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._model: LogisticRegression | None = None
        self._scaler: StandardScaler | None = None
        self._feature_names: list[str] | None = None

    @property
    def model_family(self) -> str:
        return "meta_learner"

    @property
    def requires_scaling(self) -> bool:
        # Internal scaling is handled
        return False

    @property
    def requires_sequences(self) -> bool:
        return False

    def get_default_config(self) -> dict[str, Any]:
        return {
            "C": 1.0,  # Inverse L2 regularization strength
            "fit_intercept": True,
            "class_weight": None,  # None keeps priors; "balanced" reweights
            "max_iter": 1000,
            "tol": 1e-4,
            "random_state": 42,
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
        Train the logistic meta-learner on OOF predictions.

        X_val is used for reporting only (the fit is closed over X_train).

        Args:
            X_train: OOF predictions, shape (n_samples, n_features)
            y_train: True labels (-1, 0, 1), or (0, 1) in binary mode
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

        y_train_sk = map_labels_to_classes(y_train, self._n_classes)
        y_val_sk = map_labels_to_classes(y_val, self._n_classes)

        X_train_scaled = X_train
        X_val_scaled = X_val
        if train_config.get("scale_features", True):
            self._scaler = StandardScaler()
            X_train_scaled = self._scaler.fit_transform(X_train)
            X_val_scaled = self._scaler.transform(X_val)

        # lbfgs with more than two classes fits the multinomial (softmax) model
        self._model = LogisticRegression(
            C=float(train_config.get("C", 1.0)),
            fit_intercept=train_config.get("fit_intercept", True),
            class_weight=train_config.get("class_weight"),
            max_iter=int(train_config.get("max_iter", 1000)),
            tol=float(train_config.get("tol", 1e-4)),
            random_state=train_config.get("random_state", 42),
            solver="lbfgs",
        )

        logger.info(
            f"Training RidgeMetaLearner (L2 logistic): C={train_config.get('C', 1.0)}, "
            f"class_weight={train_config.get('class_weight')}, n_features={X_train.shape[1]}"
        )

        self._model.fit(X_train_scaled, y_train_sk, sample_weight=sample_weights)
        self._is_fitted = True

        training_time = time.time() - start_time

        labels = list(range(self._n_classes))
        train_probs = self._probabilities(X_train_scaled)
        val_probs = self._probabilities(X_val_scaled)
        train_pred = map_classes_to_labels(train_probs.argmax(axis=1), self._n_classes)
        val_pred = map_classes_to_labels(val_probs.argmax(axis=1), self._n_classes)
        val_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))

        logger.info(f"Training complete: val_f1={val_f1:.4f}, time={training_time:.1f}s")

        return TrainingMetrics(
            train_loss=float(log_loss(y_train_sk, train_probs, labels=labels)),
            val_loss=float(log_loss(y_val_sk, val_probs, labels=labels)),
            train_accuracy=float(accuracy_score(y_train, train_pred)),
            val_accuracy=float(accuracy_score(y_val, val_pred)),
            train_f1=float(f1_score(y_train, train_pred, average="macro", zero_division=0)),
            val_f1=val_f1,
            epochs_trained=int(np.max(self._model.n_iter_)),
            training_time_seconds=training_time,
            early_stopped=False,
            best_epoch=None,
            history={},
            metadata={
                "meta_learner": "ridge",
                "n_features": X_train.shape[1],
                "n_train_samples": len(X_train),
                "n_val_samples": len(X_val),
                "C": train_config.get("C", 1.0),
                "class_weight": train_config.get("class_weight"),
            },
        )

    def predict(self, X: np.ndarray) -> PredictionResult:
        """Generate predictions with class probabilities."""
        self._validate_fitted()
        self._validate_input_shape(X, "X")

        X_scaled = X
        if self._scaler is not None:
            X_scaled = self._scaler.transform(X)

        probabilities = self._probabilities(X_scaled)
        class_predictions = map_classes_to_labels(probabilities.argmax(axis=1), self._n_classes)

        return PredictionResult(
            class_predictions=class_predictions,
            class_probabilities=probabilities,
            confidence=probabilities.max(axis=1),
            metadata={"meta_learner": "ridge"},
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

        logger.info(f"Saved RidgeMetaLearner to {path}")

    def load(self, path: Path) -> None:
        """Load model from directory."""
        path = Path(path)
        model_path = path / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._model = safe_pickle_load(model_path, allowed_types=(LogisticRegression,))

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
        logger.info(f"Loaded RidgeMetaLearner from {path}")

    def get_feature_importance(self) -> dict[str, float] | None:
        """Return coefficient magnitudes (averaged over classes) as importance."""
        if not self._is_fitted or self._model is None:
            return None

        coefs = np.abs(self._model.coef_).mean(axis=0)
        feature_names = self._feature_names or [f"f{i}" for i in range(len(coefs))]

        return dict(zip(feature_names, coefs.tolist(), strict=False))

    def set_feature_names(self, names: list[str]) -> None:
        """Set feature names for interpretability."""
        self._feature_names = names

    def _probabilities(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        (n, n_classes) probabilities in class-index order.

        A class absent from the training labels gets probability 0 — the
        model has no evidence for it — so the output width always matches
        the run's class count.
        """
        if self._model is None:
            raise RuntimeError("Meta model is not fitted")
        fitted = np.asarray(self._model.predict_proba(X_scaled))
        probabilities = np.zeros((len(X_scaled), self._n_classes), dtype=np.float64)
        probabilities[:, np.asarray(self._model.classes_, dtype=int)] = fitted
        return probabilities


__all__ = ["RidgeMetaLearner"]
