"""
Blending Ensemble Model - Two-layer model with holdout set predictions.

Similar to stacking but uses a holdout validation set instead of
out-of-fold predictions. Simpler and faster than full stacking.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from ..base import BaseModel, PredictionOutput, TrainingMetrics
from ..registry import ModelRegistry, register
from .validator import validate_base_model_compatibility

logger = logging.getLogger(__name__)


@register(
    name="blending",
    family="ensemble",
    description="Blending ensemble using holdout set for meta-learner training",
    aliases=["blending_ensemble", "blend"],
)
class BlendingEnsemble(BaseModel):
    """
    Blending Ensemble using holdout validation predictions.

    Architecture:
    1. Split training data into blend_train and blend_holdout
    2. Train base models on blend_train
    3. Generate predictions on blend_holdout
    4. Train meta-learner on holdout predictions + labels
    5. Retrain base models on full training data

    Advantages over Stacking:
    - Faster (no k-fold training)
    - Simpler implementation
    - Good for large datasets

    Disadvantages:
    - Uses less data for base model training
    - May have higher variance

    Example:
        ensemble = BlendingEnsemble(config={
            "base_model_names": ["xgboost", "lightgbm"],
            "meta_learner_name": "logistic",
            "holdout_fraction": 0.2,
        })
        metrics = ensemble.fit(X_train, y_train, X_val, y_val)
        output = ensemble.predict(X_test)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._base_models: List[BaseModel] = []
        self._meta_learner: Optional[BaseModel] = None
        self._base_model_names: List[str] = []
        self._meta_learner_name: str = ""
        self._feature_names: Optional[List[str]] = None

    @property
    def model_family(self) -> str:
        return "ensemble"

    @property
    def requires_scaling(self) -> bool:
        return False

    @property
    def requires_sequences(self) -> bool:
        if not self._base_models:
            return False
        return any(m.requires_sequences for m in self._base_models)

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "base_model_names": ["xgboost", "lightgbm"],
            "base_model_configs": {},
            "meta_learner_name": "logistic",
            "meta_learner_config": {},
            "holdout_fraction": 0.2,  # Fraction for meta-learner training
            "use_probabilities": True,
            "passthrough": False,
            "retrain_on_full": True,  # Retrain base models on full data
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        config: Optional[Dict[str, Any]] = None,
        label_end_times: Optional[Any] = None,
    ) -> TrainingMetrics:
        """
        Train blending ensemble.

        1. Split training data into blend_train and blend_holdout (time-based)
        2. Train base models on blend_train
        3. Generate predictions on blend_holdout
        4. Train meta-learner on holdout predictions
        5. Optionally retrain base models on full data

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            sample_weights: Sample weights for training
            config: Optional config overrides
            label_end_times: Unused (for API compatibility with stacking)

        Note:
            Uses time-based split: the LAST `holdout_fraction` of training data
            is used as holdout set to preserve temporal ordering and prevent
            future data leakage into past training.
        """
        self._validate_input_shape(X_train, "X_train")
        self._validate_input_shape(X_val, "X_val")
        start_time = time.time()

        train_config = self._config.copy()
        if config:
            train_config.update(config)

        self._base_model_names = train_config.get("base_model_names", [])
        if not self._base_model_names:
            raise ValueError("No base_model_names specified in config")

        # Validate base model compatibility (tabular vs sequence)
        validate_base_model_compatibility(self._base_model_names)

        self._meta_learner_name = train_config.get("meta_learner_name", "logistic")
        holdout_fraction = train_config.get("holdout_fraction", 0.2)
        base_model_configs = train_config.get("base_model_configs", {})
        meta_learner_config = train_config.get("meta_learner_config", {})
        use_probabilities = train_config.get("use_probabilities", True)
        passthrough = train_config.get("passthrough", False)
        retrain_on_full = train_config.get("retrain_on_full", True)

        n_samples = X_train.shape[0]
        n_holdout = int(n_samples * holdout_fraction)
        n_blend_train = n_samples - n_holdout

        if n_holdout < 100:
            logger.warning(
                f"Holdout set is small ({n_holdout} samples). "
                "Consider using Stacking for small datasets."
            )

        logger.info(
            f"Training BlendingEnsemble: base_models={self._base_model_names}, "
            f"meta_learner={self._meta_learner_name}, "
            f"blend_train={n_blend_train}, holdout={n_holdout}"
        )

        # Step 1: TIME-BASED SPLIT (critical for preventing data leakage)
        # Train on FIRST (1 - holdout_fraction) of data
        # Use LAST holdout_fraction as holdout set
        # This preserves temporal ordering: no future data leaks into past training
        X_blend_train = X_train[:n_blend_train]
        y_blend_train = y_train[:n_blend_train]
        X_holdout = X_train[n_blend_train:]  # Last portion (most recent data)
        y_holdout = y_train[n_blend_train:]

        w_blend_train = None
        w_holdout = None
        if sample_weights is not None:
            w_blend_train = sample_weights[:n_blend_train]
            w_holdout = sample_weights[n_blend_train:]

        # Step 2: Train base models on blend_train
        base_models_initial = []
        for model_name in self._base_model_names:
            model_config = base_model_configs.get(model_name, {})
            model = ModelRegistry.create(model_name, config=model_config)

            logger.info(f"  Training base model on blend_train: {model_name}")
            model.fit(
                X_train=X_blend_train,
                y_train=y_blend_train,
                X_val=X_holdout,
                y_val=y_holdout,
                sample_weights=w_blend_train,
            )
            base_models_initial.append(model)

        # Step 3: Generate predictions on holdout
        holdout_predictions = self._generate_predictions(
            X_holdout, base_models_initial, use_probabilities
        )

        # Step 4: Train meta-learner
        meta_features_holdout = holdout_predictions
        if passthrough:
            meta_features_holdout = np.hstack([X_holdout, holdout_predictions])

        logger.info(
            f"Training meta-learner on {meta_features_holdout.shape[1]} features"
        )

        self._meta_learner = ModelRegistry.create(
            self._meta_learner_name,
            config=meta_learner_config,
        )

        # Use validation set for meta-learner validation
        val_predictions = self._generate_predictions(
            X_val, base_models_initial, use_probabilities
        )
        meta_features_val = val_predictions
        if passthrough:
            meta_features_val = np.hstack([X_val, val_predictions])

        meta_metrics = self._meta_learner.fit(
            X_train=meta_features_holdout,
            y_train=y_holdout,
            X_val=meta_features_val,
            y_val=y_val,
        )

        # Step 5: Retrain base models on full data (optional)
        if retrain_on_full:
            logger.info("Retraining base models on full training data")
            self._base_models = []
            for model_name in self._base_model_names:
                model_config = base_model_configs.get(model_name, {})
                model = ModelRegistry.create(model_name, config=model_config)

                model.fit(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    sample_weights=sample_weights,
                )
                self._base_models.append(model)
        else:
            self._base_models = base_models_initial

        training_time = time.time() - start_time
        self._is_fitted = True

        # Compute ensemble metrics
        train_metrics = self._compute_metrics(X_train, y_train)
        val_metrics = self._compute_metrics(X_val, y_val)

        logger.info(
            f"BlendingEnsemble training complete: "
            f"val_f1={val_metrics['f1']:.4f}, time={training_time:.1f}s"
        )

        return TrainingMetrics(
            train_loss=meta_metrics.train_loss,
            val_loss=meta_metrics.val_loss,
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
                "meta_learner": self._meta_learner_name,
                "n_base_models": len(self._base_model_names),
                "holdout_fraction": holdout_fraction,
                "meta_features_dim": meta_features_holdout.shape[1],
                "retrain_on_full": retrain_on_full,
            },
        )

    def _generate_predictions(
        self,
        X: np.ndarray,
        models: List[BaseModel],
        use_probabilities: bool,
    ) -> np.ndarray:
        """Generate predictions from base models."""
        n_samples = X.shape[0]
        n_classes = 3
        n_models = len(models)

        if use_probabilities:
            predictions = np.zeros((n_samples, n_models * n_classes))
        else:
            predictions = np.zeros((n_samples, n_models))

        for model_idx, model in enumerate(models):
            output = model.predict(X)

            if use_probabilities:
                start_col = model_idx * n_classes
                end_col = start_col + n_classes
                predictions[:, start_col:end_col] = output.class_probabilities
            else:
                predictions[:, model_idx] = output.class_predictions

        return predictions

    def predict(self, X: np.ndarray) -> PredictionOutput:
        """Generate blending ensemble predictions."""
        self._validate_fitted()
        self._validate_input_shape(X, "X")

        use_probabilities = self._config.get("use_probabilities", True)
        passthrough = self._config.get("passthrough", False)

        # Generate base model predictions
        base_predictions = self._generate_predictions(
            X, self._base_models, use_probabilities
        )

        # Build meta-learner input
        meta_features = base_predictions
        if passthrough:
            meta_features = np.hstack([X, base_predictions])

        # Get meta-learner predictions
        output = self._meta_learner.predict(meta_features)

        return PredictionOutput(
            class_predictions=output.class_predictions,
            class_probabilities=output.class_probabilities,
            confidence=output.confidence,
            metadata={
                "ensemble": "blending",
                "base_models": self._base_model_names,
                "meta_learner": self._meta_learner_name,
            },
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities from meta-learner."""
        output = self.predict(X)
        return output.class_probabilities

    def save(self, path: Path) -> None:
        """Save blending ensemble and all component models."""
        self._validate_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save base models
        for i, model in enumerate(self._base_models):
            model_dir = path / f"base_model_{i}"
            model.save(model_dir)

        # Save meta-learner
        meta_dir = path / "meta_learner"
        self._meta_learner.save(meta_dir)

        # Save ensemble metadata
        metadata = {
            "config": self._config,
            "base_model_names": self._base_model_names,
            "meta_learner_name": self._meta_learner_name,
            "feature_names": self._feature_names,
            "n_base_models": len(self._base_models),
        }
        joblib.dump(metadata, path / "ensemble_metadata.joblib")

        logger.info(f"Saved BlendingEnsemble to {path}")

    def load(self, path: Path) -> None:
        """Load blending ensemble and all component models."""
        path = Path(path)
        metadata_path = path / "ensemble_metadata.joblib"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Ensemble metadata not found: {metadata_path}")

        metadata = joblib.load(metadata_path)
        self._config = metadata.get("config", self._config)
        self._base_model_names = metadata.get("base_model_names", [])
        self._meta_learner_name = metadata.get("meta_learner_name", "logistic")
        self._feature_names = metadata.get("feature_names")
        n_base_models = metadata.get("n_base_models", 0)

        # Load base models
        self._base_models = []
        for i in range(n_base_models):
            model_dir = path / f"base_model_{i}"
            model_name = self._base_model_names[i]
            model = ModelRegistry.create(model_name)
            model.load(model_dir)
            self._base_models.append(model)

        # Load meta-learner
        meta_dir = path / "meta_learner"
        self._meta_learner = ModelRegistry.create(self._meta_learner_name)
        self._meta_learner.load(meta_dir)

        self._is_fitted = True
        logger.info(f"Loaded BlendingEnsemble from {path}")

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Aggregate feature importances from base models."""
        if not self._is_fitted:
            return None

        all_importances = []
        for model in self._base_models:
            imp = model.get_feature_importance()
            if imp is not None:
                all_importances.append(imp)

        if not all_importances:
            return None

        all_features = set()
        for imp in all_importances:
            all_features.update(imp.keys())

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


__all__ = ["BlendingEnsemble"]
