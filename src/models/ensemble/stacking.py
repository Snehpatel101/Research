"""
Stacking Ensemble Model - Two-layer model with out-of-fold predictions.

Uses base models to generate OOF predictions, then trains a meta-learner
on those predictions to produce final output.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.cross_validation.purged_kfold import PurgedKFold, PurgedKFoldConfig
from ..base import BaseModel, PredictionOutput, TrainingMetrics
from ..registry import ModelRegistry, register
from .validator import validate_base_model_compatibility

logger = logging.getLogger(__name__)


@register(
    name="stacking",
    family="ensemble",
    description="Stacking ensemble with out-of-fold predictions and meta-learner",
    aliases=["stacking_ensemble", "stack"],
)
class StackingEnsemble(BaseModel):
    """
    Stacking Ensemble using out-of-fold predictions.

    Architecture:
    1. Layer 1: Base models generate out-of-fold predictions
       - Each sample is predicted by a model trained without that sample
       - Prevents overfitting in meta-learner training
    2. Layer 2: Meta-learner trains on OOF predictions
       - Default: LogisticRegression (calibrated probabilities)
       - Can use any registered model

    Example:
        ensemble = StackingEnsemble(config={
            "base_model_names": ["xgboost", "lightgbm", "random_forest"],
            "meta_learner_name": "logistic",
            "n_folds": 5,
        })
        metrics = ensemble.fit(X_train, y_train, X_val, y_val)
        output = ensemble.predict(X_test)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._base_models: List[List[BaseModel]] = []  # [model_idx][fold_idx]
        self._meta_learner: Optional[BaseModel] = None
        self._base_model_names: List[str] = []
        self._meta_learner_name: str = ""
        self._n_folds: int = 5
        self._feature_names: Optional[List[str]] = None

    @property
    def model_family(self) -> str:
        return "ensemble"

    @property
    def requires_scaling(self) -> bool:
        # Stacking handles scaling internally via meta-learner
        return False

    @property
    def requires_sequences(self) -> bool:
        # Check if any base model requires sequences
        if self._base_models:
            return any(
                m.requires_sequences
                for fold_models in self._base_models
                for m in fold_models
            )
        # If base models not yet instantiated, check config for base_model_names
        # This is critical for Trainer to prepare correct data shape (2D vs 3D)
        base_model_names = self._config.get("base_model_names", [])
        if base_model_names:
            return self._check_configured_models_require_sequences(base_model_names)
        return False

    def _check_configured_models_require_sequences(
        self, model_names: list[str]
    ) -> bool:
        """
        Check if any configured base model requires sequences.

        This enables correct data preparation before base models are instantiated.
        """
        for name in model_names:
            if ModelRegistry.is_registered(name):
                info = ModelRegistry.get_model_info(name)
                if info.get("requires_sequences", False):
                    return True
        return False

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "base_model_names": ["xgboost", "lightgbm"],
            "base_model_configs": {},  # Per-model config overrides for FINAL models
            "meta_learner_name": "logistic",  # LogisticModel from classical
            "meta_learner_config": {},
            "n_folds": 5,
            "use_probabilities": True,  # Use probs vs class predictions
            "passthrough": False,  # Include original features in meta-learner
            "purge_bars": 60,  # Prevent overlapping label leakage
            "embargo_bars": 1440,  # Break serial correlation (5 days at 5min)
            # CRITICAL: Use default configs for OOF to prevent leakage
            "use_default_configs_for_oof": True,
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        config: Optional[Dict[str, Any]] = None,
        label_end_times: Optional[pd.Series] = None,
    ) -> TrainingMetrics:
        """
        Train stacking ensemble with OOF predictions.

        1. Generate OOF predictions for each base model using PurgedKFold
        2. Train meta-learner on OOF predictions
        3. Train final base models on full training data

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            sample_weights: Sample weights for training
            config: Optional config overrides
            label_end_times: When each label's outcome is known (for purging overlapping labels)
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
        self._n_folds = train_config.get("n_folds", 5)
        base_model_configs = train_config.get("base_model_configs", {})
        meta_learner_config = train_config.get("meta_learner_config", {})
        use_probabilities = train_config.get("use_probabilities", True)
        passthrough = train_config.get("passthrough", False)
        purge_bars = train_config.get("purge_bars", 60)
        embargo_bars = train_config.get("embargo_bars", 1440)

        # ==================================================================
        # LEAKAGE PREVENTION: Use default configs for OOF generation
        # ==================================================================
        # When use_default_configs_for_oof=True (default), OOF predictions
        # are generated with default model configurations, NOT tuned configs.
        # This prevents the meta-learner from training on optimistically
        # biased OOF predictions from tuned base models.
        #
        # The tuned configs (base_model_configs) are only used for:
        # 1. Final base models stored for inference
        # 2. NOT for OOF generation that trains the meta-learner
        # ==================================================================
        use_default_for_oof = train_config.get("use_default_configs_for_oof", True)

        if use_default_for_oof:
            # Use empty configs (defaults) for OOF to prevent leakage
            oof_base_configs: Dict[str, Dict] = {}
            logger.info(
                "Using DEFAULT configs for OOF generation (leakage prevention enabled)"
            )
        else:
            # Legacy behavior: use provided configs (NOT RECOMMENDED)
            oof_base_configs = base_model_configs
            logger.warning(
                "Using tuned configs for OOF generation. "
                "This may cause leakage - set use_default_configs_for_oof=True"
            )

        logger.info(
            f"Training StackingEnsemble: base_models={self._base_model_names}, "
            f"meta_learner={self._meta_learner_name}, n_folds={self._n_folds}"
        )

        # Step 1: Generate OOF predictions with PurgedKFold (prevents label leakage)
        # IMPORTANT: Uses oof_base_configs (defaults) not base_model_configs (tuned)
        oof_predictions, fold_models = self._generate_oof_predictions(
            X_train=X_train,
            y_train=y_train,
            base_model_names=self._base_model_names,
            base_model_configs=oof_base_configs,  # Use defaults for OOF
            sample_weights=sample_weights,
            use_probabilities=use_probabilities,
            label_end_times=label_end_times,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
        )

        # Step 2: Train meta-learner on OOF predictions
        meta_features_train = oof_predictions
        if passthrough:
            meta_features_train = np.hstack([X_train, oof_predictions])

        logger.info(
            f"Training meta-learner on {meta_features_train.shape[1]} features"
        )

        self._meta_learner = ModelRegistry.create(
            self._meta_learner_name,
            config=meta_learner_config,
        )

        # Use a portion of validation set for meta-learner validation
        val_predictions = self._generate_base_predictions(
            X_val, fold_models, use_probabilities
        )
        meta_features_val = val_predictions
        if passthrough:
            meta_features_val = np.hstack([X_val, val_predictions])

        meta_metrics = self._meta_learner.fit(
            X_train=meta_features_train,
            y_train=y_train,
            X_val=meta_features_val,
            y_val=y_val,
            sample_weights=sample_weights,
        )

        # Step 3: Keep fold models for inference (use average of folds)
        self._base_models = fold_models

        training_time = time.time() - start_time
        self._is_fitted = True

        # Compute ensemble metrics
        train_metrics = self._compute_metrics(X_train, y_train)
        val_metrics = self._compute_metrics(X_val, y_val)

        logger.info(
            f"StackingEnsemble training complete: "
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
                "n_folds": self._n_folds,
                "n_base_models": len(self._base_model_names),
                "meta_features_dim": meta_features_train.shape[1],
                "use_probabilities": use_probabilities,
                "passthrough": passthrough,
                "use_default_configs_for_oof": use_default_for_oof,
                "leakage_prevention": "enabled" if use_default_for_oof else "disabled",
            },
        )

    def _generate_oof_predictions(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        base_model_names: List[str],
        base_model_configs: Dict[str, Dict],
        sample_weights: Optional[np.ndarray],
        use_probabilities: bool,
        label_end_times: Optional[pd.Series] = None,
        purge_bars: int = 60,
        embargo_bars: int = 1440,
    ) -> Tuple[np.ndarray, List[List[BaseModel]]]:
        """
        Generate out-of-fold predictions for all base models.

        Uses PurgedKFold to prevent label leakage from overlapping labels
        and serial correlation.

        Args:
            X_train: Training features
            y_train: Training labels
            base_model_names: Names of base models to train
            base_model_configs: Config overrides per model
            sample_weights: Sample weights
            use_probabilities: If True, use class probabilities; else use class predictions
            label_end_times: When each label is resolved (enables overlapping label purging)
            purge_bars: Number of bars to purge around validation set
            embargo_bars: Number of bars to embargo after validation set

        Returns:
            Tuple of (oof_predictions, fold_models)
        """
        n_samples = X_train.shape[0]
        n_classes = 3

        # Initialize OOF storage
        if use_probabilities:
            # n_models * n_classes features
            n_features = len(base_model_names) * n_classes
        else:
            # n_models features (class predictions)
            n_features = len(base_model_names)

        oof_predictions = np.zeros((n_samples, n_features))
        fold_models: List[List[BaseModel]] = [[] for _ in base_model_names]

        # Create PurgedKFold splitter to prevent label leakage
        purged_kfold_config = PurgedKFoldConfig(
            n_splits=self._n_folds,
            purge_bars=purge_bars,  # Prevent overlapping label leakage
            embargo_bars=embargo_bars,  # Break serial correlation
        )
        kfold = PurgedKFold(purged_kfold_config)

        # Convert to DataFrame for PurgedKFold (requires index for label_end_times)
        X_df = pd.DataFrame(X_train)

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_df, label_end_times=label_end_times)):
            logger.debug(f"  Fold {fold_idx + 1}/{self._n_folds}")

            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]

            w_train = None
            if sample_weights is not None:
                w_train = sample_weights[train_idx]

            # Train each base model on this fold
            for model_idx, model_name in enumerate(base_model_names):
                model_config = base_model_configs.get(model_name, {})
                model = ModelRegistry.create(model_name, config=model_config)

                model.fit(
                    X_train=X_fold_train,
                    y_train=y_fold_train,
                    X_val=X_fold_val,
                    y_val=y_fold_val,
                    sample_weights=w_train,
                )

                # Generate predictions for OOF samples
                output = model.predict(X_fold_val)

                if use_probabilities:
                    start_col = model_idx * n_classes
                    end_col = start_col + n_classes
                    oof_predictions[val_idx, start_col:end_col] = (
                        output.class_probabilities
                    )
                else:
                    oof_predictions[val_idx, model_idx] = output.class_predictions

                fold_models[model_idx].append(model)

        return oof_predictions, fold_models

    def _generate_base_predictions(
        self,
        X: np.ndarray,
        fold_models: List[List[BaseModel]],
        use_probabilities: bool,
    ) -> np.ndarray:
        """Generate predictions from base models (averaging across folds)."""
        n_samples = X.shape[0]
        n_classes = 3
        n_models = len(fold_models)

        if use_probabilities:
            predictions = np.zeros((n_samples, n_models * n_classes))
        else:
            predictions = np.zeros((n_samples, n_models))

        for model_idx, models in enumerate(fold_models):
            # Average predictions across folds
            if use_probabilities:
                probs = np.zeros((n_samples, n_classes))
                for model in models:
                    output = model.predict(X)
                    probs += output.class_probabilities
                probs /= len(models)

                start_col = model_idx * n_classes
                end_col = start_col + n_classes
                predictions[:, start_col:end_col] = probs
            else:
                # For class predictions, use voting
                all_preds = []
                for model in models:
                    output = model.predict(X)
                    all_preds.append(output.class_predictions)
                all_preds = np.array(all_preds)
                # Mode across folds
                from scipy import stats
                mode_result = stats.mode(all_preds, axis=0, keepdims=False)
                predictions[:, model_idx] = mode_result.mode

        return predictions

    def predict(self, X: np.ndarray) -> PredictionOutput:
        """Generate stacking ensemble predictions."""
        self._validate_fitted()
        self._validate_input_shape(X, "X")

        use_probabilities = self._config.get("use_probabilities", True)
        passthrough = self._config.get("passthrough", False)

        # Generate base model predictions
        base_predictions = self._generate_base_predictions(
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
                "ensemble": "stacking",
                "base_models": self._base_model_names,
                "meta_learner": self._meta_learner_name,
            },
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities from meta-learner."""
        output = self.predict(X)
        return output.class_probabilities

    def save(self, path: Path) -> None:
        """Save stacking ensemble and all component models."""
        self._validate_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save base models (all folds)
        base_models_dir = path / "base_models"
        base_models_dir.mkdir(exist_ok=True)

        for model_idx, models in enumerate(self._base_models):
            model_dir = base_models_dir / f"model_{model_idx}"
            model_dir.mkdir(exist_ok=True)
            for fold_idx, model in enumerate(models):
                fold_dir = model_dir / f"fold_{fold_idx}"
                model.save(fold_dir)

        # Save meta-learner
        meta_dir = path / "meta_learner"
        self._meta_learner.save(meta_dir)

        # Save ensemble metadata
        metadata = {
            "config": self._config,
            "base_model_names": self._base_model_names,
            "meta_learner_name": self._meta_learner_name,
            "n_folds": self._n_folds,
            "feature_names": self._feature_names,
        }
        joblib.dump(metadata, path / "ensemble_metadata.joblib")

        logger.info(f"Saved StackingEnsemble to {path}")

    def load(self, path: Path) -> None:
        """Load stacking ensemble and all component models."""
        path = Path(path)
        metadata_path = path / "ensemble_metadata.joblib"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Ensemble metadata not found: {metadata_path}")

        metadata = joblib.load(metadata_path)
        self._config = metadata.get("config", self._config)
        self._base_model_names = metadata.get("base_model_names", [])
        self._meta_learner_name = metadata.get("meta_learner_name", "logistic")
        self._n_folds = metadata.get("n_folds", 5)
        self._feature_names = metadata.get("feature_names")

        # Load base models
        base_models_dir = path / "base_models"
        self._base_models = []

        for model_idx, model_name in enumerate(self._base_model_names):
            model_dir = base_models_dir / f"model_{model_idx}"
            fold_models = []

            for fold_idx in range(self._n_folds):
                fold_dir = model_dir / f"fold_{fold_idx}"
                if fold_dir.exists():
                    model = ModelRegistry.create(model_name)
                    model.load(fold_dir)
                    fold_models.append(model)

            self._base_models.append(fold_models)

        # Load meta-learner
        meta_dir = path / "meta_learner"
        self._meta_learner = ModelRegistry.create(self._meta_learner_name)
        self._meta_learner.load(meta_dir)

        self._is_fitted = True
        logger.info(f"Loaded StackingEnsemble from {path}")

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Return meta-learner feature importances."""
        if not self._is_fitted or self._meta_learner is None:
            return None

        # Meta-learner importance tells which base model predictions matter
        return self._meta_learner.get_feature_importance()

    def set_feature_names(self, names: List[str]) -> None:
        """Set feature names for base models."""
        self._feature_names = names
        for fold_models in self._base_models:
            for model in fold_models:
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


__all__ = ["StackingEnsemble"]
