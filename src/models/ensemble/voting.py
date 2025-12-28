"""
Voting Ensemble Model - Combine predictions from multiple base models.

Supports both hard voting (majority) and soft voting (probability averaging).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from ..base import BaseModel, PredictionOutput, TrainingMetrics
from ..common import map_classes_to_labels
from ..registry import ModelRegistry, register

logger = logging.getLogger(__name__)


@register(
    name="voting",
    family="ensemble",
    description="Voting ensemble combining multiple base models",
    aliases=["voting_ensemble", "vote"],
)
class VotingEnsemble(BaseModel):
    """
    Voting Ensemble that combines predictions from multiple base models.

    Supports two voting strategies:
    - "hard": Majority vote of class predictions
    - "soft": Average of class probabilities

    Base models can be provided as:
    - List of trained model instances
    - List of model names to train during fit()

    Example:
        # With pre-trained models
        ensemble = VotingEnsemble(config={"voting": "soft"})
        ensemble.set_base_models([xgb_model, lgb_model])
        output = ensemble.predict(X_test)

        # Training from scratch
        ensemble = VotingEnsemble(config={
            "voting": "soft",
            "base_model_names": ["xgboost", "lightgbm", "random_forest"],
        })
        metrics = ensemble.fit(X_train, y_train, X_val, y_val)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._base_models: List[BaseModel] = []
        self._base_model_names: List[str] = []
        self._weights: Optional[np.ndarray] = None
        self._feature_names: Optional[List[str]] = None

    @property
    def model_family(self) -> str:
        return "ensemble"

    @property
    def requires_scaling(self) -> bool:
        # Check if any base model requires scaling
        if not self._base_models:
            return False
        return any(m.requires_scaling for m in self._base_models)

    @property
    def requires_sequences(self) -> bool:
        # Check if any base model requires sequences
        if not self._base_models:
            return False
        return any(m.requires_sequences for m in self._base_models)

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "voting": "soft",  # "hard" or "soft"
            "weights": None,  # Optional model weights [w1, w2, ...]
            "base_model_names": [],  # Model names to train
            "base_model_configs": {},  # Per-model config overrides
        }

    def set_base_models(
        self,
        models: List[BaseModel],
        weights: Optional[List[float]] = None,
    ) -> None:
        """
        Set pre-trained base models for the ensemble.

        Args:
            models: List of trained BaseModel instances
            weights: Optional weights for each model (must sum to 1 for soft voting)
        """
        if not models:
            raise ValueError("Must provide at least one base model")

        for i, model in enumerate(models):
            if not isinstance(model, BaseModel):
                raise TypeError(f"Model {i} is not a BaseModel instance")
            if not model.is_fitted:
                raise RuntimeError(f"Model {i} is not fitted")

        self._base_models = list(models)
        self._base_model_names = [
            type(m).__name__ for m in models
        ]

        if weights is not None:
            if len(weights) != len(models):
                raise ValueError(
                    f"Weights length ({len(weights)}) != models length ({len(models)})"
                )
            self._weights = np.array(weights, dtype=np.float32)
            # Normalize weights
            self._weights = self._weights / self._weights.sum()
        else:
            self._weights = None

        self._is_fitted = True
        logger.info(f"Set {len(models)} base models for VotingEnsemble")

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
        Train all base models from scratch.

        Requires base_model_names to be specified in config.
        """
        self._validate_input_shape(X_train, "X_train")
        self._validate_input_shape(X_val, "X_val")
        start_time = time.time()

        train_config = self._config.copy()
        if config:
            train_config.update(config)

        base_model_names = train_config.get("base_model_names", [])
        if not base_model_names:
            raise ValueError(
                "No base_model_names specified. Either set base_model_names in config "
                "or use set_base_models() with pre-trained models."
            )

        base_model_configs = train_config.get("base_model_configs", {})

        logger.info(f"Training VotingEnsemble with base models: {base_model_names}")

        # Train each base model
        self._base_models = []
        self._base_model_names = base_model_names
        all_val_f1 = []

        for model_name in base_model_names:
            model_config = base_model_configs.get(model_name, {})
            model = ModelRegistry.create(model_name, config=model_config)

            logger.info(f"  Training base model: {model_name}")
            metrics = model.fit(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                sample_weights=sample_weights,
            )

            self._base_models.append(model)
            all_val_f1.append(metrics.val_f1)
            logger.info(f"    {model_name} val_f1={metrics.val_f1:.4f}")

        # Set weights based on validation performance if requested
        weights = train_config.get("weights")
        if weights is not None:
            if len(weights) != len(self._base_models):
                raise ValueError(
                    f"Weights length ({len(weights)}) != models ({len(self._base_models)})"
                )
            self._weights = np.array(weights, dtype=np.float32)
            self._weights = self._weights / self._weights.sum()
        else:
            self._weights = None

        training_time = time.time() - start_time
        self._is_fitted = True

        # Compute ensemble metrics
        train_metrics = self._compute_metrics(X_train, y_train)
        val_metrics = self._compute_metrics(X_val, y_val)

        logger.info(
            f"VotingEnsemble training complete: val_f1={val_metrics['f1']:.4f}, "
            f"time={training_time:.1f}s"
        )

        return TrainingMetrics(
            train_loss=0.0,
            val_loss=0.0,
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
                "base_models": self._base_model_names,
                "n_base_models": len(self._base_models),
                "voting": train_config.get("voting", "soft"),
                "base_model_val_f1": all_val_f1,
            },
        )

    def predict(self, X: np.ndarray) -> PredictionOutput:
        """Generate ensemble predictions."""
        self._validate_fitted()
        self._validate_input_shape(X, "X")

        voting_strategy = self._config.get("voting", "soft")

        if voting_strategy == "soft":
            return self._soft_vote(X)
        else:
            return self._hard_vote(X)

    def _soft_vote(self, X: np.ndarray) -> PredictionOutput:
        """Soft voting: average class probabilities."""
        n_samples = X.shape[0]
        n_classes = 3

        # Collect probabilities from all models
        all_probs = []
        for model in self._base_models:
            output = model.predict(X)
            all_probs.append(output.class_probabilities)

        all_probs = np.array(all_probs)  # (n_models, n_samples, n_classes)

        # Apply weights if specified
        if self._weights is not None:
            weights = self._weights.reshape(-1, 1, 1)
            weighted_probs = all_probs * weights
            avg_probs = weighted_probs.sum(axis=0)
        else:
            avg_probs = all_probs.mean(axis=0)

        # Get class predictions from averaged probabilities
        class_predictions_idx = np.argmax(avg_probs, axis=1)
        # Convert 0,1,2 -> -1,0,1 using canonical mapping
        class_predictions = map_classes_to_labels(class_predictions_idx)

        confidence = np.max(avg_probs, axis=1)

        return PredictionOutput(
            class_predictions=class_predictions,
            class_probabilities=avg_probs,
            confidence=confidence,
            metadata={
                "voting": "soft",
                "n_models": len(self._base_models),
                "model_names": self._base_model_names,
            },
        )

    def _hard_vote(self, X: np.ndarray) -> PredictionOutput:
        """Hard voting: majority vote of class predictions."""
        n_samples = X.shape[0]

        # Collect predictions from all models
        all_preds = []
        all_probs = []
        for model in self._base_models:
            output = model.predict(X)
            all_preds.append(output.class_predictions)
            all_probs.append(output.class_probabilities)

        all_preds = np.array(all_preds)  # (n_models, n_samples)
        all_probs = np.array(all_probs)  # (n_models, n_samples, n_classes)

        # Majority vote
        class_predictions = np.zeros(n_samples, dtype=np.int32)
        confidence = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            sample_preds = all_preds[:, i]
            # Count votes for each class
            unique, counts = np.unique(sample_preds, return_counts=True)
            majority_idx = np.argmax(counts)
            class_predictions[i] = unique[majority_idx]
            # Confidence = fraction of models that agree
            confidence[i] = counts[majority_idx] / len(self._base_models)

        # Average probabilities for output (even though we use hard voting)
        avg_probs = all_probs.mean(axis=0)

        return PredictionOutput(
            class_predictions=class_predictions,
            class_probabilities=avg_probs,
            confidence=confidence,
            metadata={
                "voting": "hard",
                "n_models": len(self._base_models),
                "model_names": self._base_model_names,
            },
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return averaged class probabilities."""
        output = self.predict(X)
        return output.class_probabilities

    def save(self, path: Path) -> None:
        """Save ensemble and all base models."""
        self._validate_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save each base model
        for i, model in enumerate(self._base_models):
            model_dir = path / f"base_model_{i}"
            model.save(model_dir)

        # Save ensemble metadata
        metadata = {
            "config": self._config,
            "base_model_names": self._base_model_names,
            "weights": self._weights.tolist() if self._weights is not None else None,
            "feature_names": self._feature_names,
            "n_base_models": len(self._base_models),
        }
        joblib.dump(metadata, path / "ensemble_metadata.joblib")

        logger.info(f"Saved VotingEnsemble to {path}")

    def load(self, path: Path) -> None:
        """Load ensemble and all base models."""
        path = Path(path)
        metadata_path = path / "ensemble_metadata.joblib"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Ensemble metadata not found: {metadata_path}")

        metadata = joblib.load(metadata_path)
        self._config = metadata.get("config", self._config)
        self._base_model_names = metadata.get("base_model_names", [])
        weights = metadata.get("weights")
        self._weights = np.array(weights) if weights is not None else None
        self._feature_names = metadata.get("feature_names")
        n_base_models = metadata.get("n_base_models", 0)

        # Load base models
        self._base_models = []
        for i in range(n_base_models):
            model_dir = path / f"base_model_{i}"
            model_name = self._base_model_names[i]

            # Create empty model and load state
            model = ModelRegistry.create(model_name)
            model.load(model_dir)
            self._base_models.append(model)

        self._is_fitted = True
        logger.info(f"Loaded VotingEnsemble from {path}")

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Aggregate feature importances from base models."""
        if not self._is_fitted:
            return None

        # Collect importances from models that support it
        all_importances = []
        for model in self._base_models:
            imp = model.get_feature_importance()
            if imp is not None:
                all_importances.append(imp)

        if not all_importances:
            return None

        # Get all unique features
        all_features = set()
        for imp in all_importances:
            all_features.update(imp.keys())

        # Average importances across models
        avg_importance = {}
        for feat in all_features:
            values = [imp.get(feat, 0.0) for imp in all_importances]
            avg_importance[feat] = float(np.mean(values))

        return avg_importance

    def set_feature_names(self, names: List[str]) -> None:
        """Set feature names for interpretability."""
        self._feature_names = names
        for model in self._base_models:
            if hasattr(model, "set_feature_names"):
                model.set_feature_names(names)

    def _compute_metrics(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Compute accuracy and F1 for a dataset."""
        output = self.predict(X)
        y_pred = output.class_predictions

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }


__all__ = ["VotingEnsemble"]
