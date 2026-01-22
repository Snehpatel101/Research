"""
Stacking Ensemble Model - Two-layer model with out-of-fold predictions.

Uses base models to generate OOF predictions, then trains a meta-learner
on those predictions to produce final output.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.core.utils.memory import estimate_array_size

from ..base import BaseModel, PredictionOutput, TrainingMetrics
from ..registry import ModelRegistry, register
from .diversity import DiversityAnalyzer, DiversityMetrics
from .validator import (
    classify_base_models,
    is_heterogeneous_ensemble,
    validate_base_model_compatibility,
)

logger = logging.getLogger(__name__)

# Memory warning threshold: warn if cached sequence data exceeds this size
_CACHE_SIZE_WARNING_THRESHOLD_BYTES = 1024 * 1024 * 1024  # 1GB


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

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._base_models: list[list[BaseModel]] = []  # [model_idx][fold_idx]
        self._meta_learner: BaseModel | None = None
        self._base_model_names: list[str] = []
        self._meta_learner_name: str = ""
        self._n_folds: int = 5
        self._feature_names: list[str] | None = None
        # Heterogeneous ensemble support
        self._is_heterogeneous: bool = False
        self._tabular_models: set[str] = set()
        self._sequence_models: set[str] = set()
        # Cached sequence data for heterogeneous ensembles (MOD-007)
        # These are cleared after training via clear_cache() to free memory
        self._X_train_seq: np.ndarray | None = None
        self._X_val_seq: np.ndarray | None = None
        self._seq_cache_size_bytes: int = 0  # Track cached data size
        # Diversity analysis
        self._diversity_metrics: DiversityMetrics | None = None
        self._diversity_analyzer: DiversityAnalyzer | None = None

    @property
    def model_family(self) -> str:
        return "ensemble"

    @property
    def ensemble_type(self) -> str:
        """Return the ensemble type identifier."""
        return "stacking"

    @property
    def requires_scaling(self) -> bool:
        # Stacking handles scaling internally via meta-learner
        return False

    @property
    def requires_sequences(self) -> bool:
        # Check if any base model requires sequences
        if self._base_models:
            return any(
                m.requires_sequences for fold_models in self._base_models for m in fold_models
            )
        # If base models not yet instantiated, check config for base_model_names
        # This is critical for Trainer to prepare correct data shape (2D vs 3D)
        base_model_names = self._config.get("base_model_names", [])
        if base_model_names:
            return self._check_configured_models_require_sequences(base_model_names)
        return False

    def _check_configured_models_require_sequences(self, model_names: list[str]) -> bool:
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

    def _validate_input_shape(self, X: np.ndarray, context: str = "input") -> None:
        """
        Validate input array shape for stacking ensemble.

        For heterogeneous ensembles, the main X parameter is always 2D tabular data.
        Sequence models receive 3D data via separate X_seq parameter.

        Args:
            X: Input array to validate
            context: Context for error messages

        Raises:
            ValueError: If shape is invalid
        """
        if X.ndim == 1:
            raise ValueError(f"{context} must be 2D or 3D, got 1D array with shape {X.shape}")

        # For heterogeneous ensembles, main X is always 2D tabular
        # Sequence data is passed separately via X_seq parameter
        if self._is_heterogeneous:
            if X.ndim != 2:
                raise ValueError(
                    f"{context} must be 2D (n_samples, n_features) for heterogeneous "
                    f"stacking ensemble (sequence data passed separately), got shape {X.shape}"
                )
        elif self.requires_sequences:
            # Homogeneous sequence ensemble
            if X.ndim != 3:
                raise ValueError(
                    f"{context} must be 3D (n_samples, seq_len, n_features) "
                    f"for sequence models, got shape {X.shape}"
                )
        else:
            # Homogeneous tabular ensemble
            if X.ndim != 2:
                raise ValueError(
                    f"{context} must be 2D (n_samples, n_features) "
                    f"for tabular models, got shape {X.shape}"
                )

    def get_default_config(self) -> dict[str, Any]:
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
            # Diversity analysis settings
            "analyze_diversity": True,  # Enable diversity analysis
            "min_diversity_threshold": 0.3,  # Minimum acceptable diversity score
            "correlation_threshold": 0.8,  # Max correlation before warning
            "reject_low_diversity": False,  # If True, reject ensembles below threshold
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: np.ndarray | None = None,
        config: dict[str, Any] | None = None,
        label_end_times: pd.Series | None = None,
        X_train_seq: np.ndarray | None = None,
        X_val_seq: np.ndarray | None = None,
    ) -> TrainingMetrics:
        """
        Train stacking ensemble with OOF predictions.

        1. Generate OOF predictions for each base model using PurgedKFold
        2. Train meta-learner on OOF predictions
        3. Train final base models on full training data

        Args:
            X_train: Training features (2D for tabular models)
            y_train: Training labels
            X_val: Validation features (2D for tabular models)
            y_val: Validation labels
            sample_weights: Sample weights for training
            config: Optional config overrides
            label_end_times: When each label's outcome is known (for purging overlapping labels)
            X_train_seq: Training sequences (3D) for sequence models in heterogeneous ensembles.
                If None in heterogeneous mode, falls back to X_train for all models.
            X_val_seq: Validation sequences (3D) for sequence models in heterogeneous ensembles.
                If None in heterogeneous mode, falls back to X_val for all models.

        Note:
            For heterogeneous stacking (mixed tabular + sequence models):
            - Tabular models receive X_train/X_val (2D)
            - Sequence models receive X_train_seq/X_val_seq (3D) if provided
            - The meta-learner always receives 2D OOF predictions regardless of base model type
        """
        start_time = time.time()

        train_config = self._config.copy()
        if config:
            train_config.update(config)

        self._base_model_names = train_config.get("base_model_names", [])
        if not self._base_model_names:
            raise ValueError("No base_model_names specified in config")

        # Validate base model compatibility - stacking allows heterogeneous models
        # because the meta-learner always receives 2D OOF predictions
        validate_base_model_compatibility(
            self._base_model_names,
            ensemble_type="stacking",  # Enables heterogeneous support
        )

        # Check if this is a heterogeneous ensemble and log appropriately
        self._is_heterogeneous = is_heterogeneous_ensemble(self._base_model_names)
        if self._is_heterogeneous:
            tabular, sequence = classify_base_models(self._base_model_names)
            self._tabular_models = set(tabular)
            self._sequence_models = set(sequence)
            logger.info(
                f"Heterogeneous stacking ensemble: "
                f"tabular={tabular}, sequence={sequence}. "
                f"Each model family will receive appropriately shaped data."
            )

            # For heterogeneous ensembles:
            # - X_train/X_val must be 2D (for tabular models)
            # - X_train_seq/X_val_seq must be 3D (for sequence models)
            if X_train.ndim != 2:
                raise ValueError(
                    f"For heterogeneous stacking, X_train must be 2D for tabular models, "
                    f"got shape {X_train.shape}. Provide sequence data via X_train_seq."
                )
            if X_val.ndim != 2:
                raise ValueError(
                    f"For heterogeneous stacking, X_val must be 2D for tabular models, "
                    f"got shape {X_val.shape}. Provide sequence data via X_val_seq."
                )

            # Validate sequence data is provided for heterogeneous ensembles
            if X_train_seq is None or X_val_seq is None:
                logger.warning(
                    "Heterogeneous ensemble requires both 2D (tabular) and 3D (sequence) data. "
                    "X_train_seq/X_val_seq not provided - sequence models will receive 2D data "
                    "which may cause errors. Consider providing sequence data or using "
                    "homogeneous ensembles."
                )
            else:
                # Validate sequence data shapes
                if X_train_seq.ndim != 3:
                    raise ValueError(
                        f"X_train_seq must be 3D (n_samples, seq_len, n_features), "
                        f"got shape {X_train_seq.shape}"
                    )
                if X_val_seq.ndim != 3:
                    raise ValueError(
                        f"X_val_seq must be 3D (n_samples, seq_len, n_features), "
                        f"got shape {X_val_seq.shape}"
                    )
        else:
            self._tabular_models = set()
            self._sequence_models = set()
            # For homogeneous ensembles, use standard validation
            self._validate_input_shape(X_train, "X_train")
            self._validate_input_shape(X_val, "X_val")

        # Store sequence data for heterogeneous ensembles (MOD-007)
        # Cache with memory size tracking and warnings
        self._X_train_seq = X_train_seq
        self._X_val_seq = X_val_seq
        self._seq_cache_size_bytes = 0

        if X_train_seq is not None or X_val_seq is not None:
            train_size = estimate_array_size(X_train_seq)
            val_size = estimate_array_size(X_val_seq)
            self._seq_cache_size_bytes = train_size + val_size

            # Warn if cached data exceeds threshold (MOD-007)
            if self._seq_cache_size_bytes > _CACHE_SIZE_WARNING_THRESHOLD_BYTES:
                logger.warning(
                    f"MOD-007: Caching large sequence data for heterogeneous ensemble: "
                    f"{self._seq_cache_size_bytes / 1024**3:.2f}GB "
                    f"(train={train_size / 1024**2:.1f}MB, val={val_size / 1024**2:.1f}MB). "
                    f"Consider calling clear_cache() after training to free memory."
                )
            else:
                logger.debug(
                    f"MOD-007: Caching sequence data: {self._seq_cache_size_bytes / 1024**2:.1f}MB"
                )

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
            oof_base_configs: dict[str, dict] = {}
            logger.info("Using DEFAULT configs for OOF generation (leakage prevention enabled)")
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
        # For heterogeneous ensembles, pass both 2D and 3D data
        oof_predictions, fold_models, oof_per_model = self._generate_oof_predictions(
            X_train=X_train,
            y_train=y_train,
            base_model_names=self._base_model_names,
            base_model_configs=oof_base_configs,  # Use defaults for OOF
            sample_weights=sample_weights,
            use_probabilities=use_probabilities,
            label_end_times=label_end_times,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            X_train_seq=X_train_seq,  # For sequence models in heterogeneous ensembles
        )

        # Step 1.5: Analyze diversity of base model predictions
        analyze_diversity = train_config.get("analyze_diversity", True)
        min_diversity_threshold = train_config.get("min_diversity_threshold", 0.3)
        correlation_threshold = train_config.get("correlation_threshold", 0.8)
        reject_low_diversity = train_config.get("reject_low_diversity", False)

        if analyze_diversity:
            self._diversity_metrics = self._analyze_oof_diversity(
                oof_per_model=oof_per_model,
                y_train=y_train,
                min_diversity_threshold=min_diversity_threshold,
                correlation_threshold=correlation_threshold,
            )

            # Reject ensemble if diversity is too low and rejection is enabled
            if reject_low_diversity:
                if self._diversity_metrics.diversity_score < min_diversity_threshold:
                    raise ValueError(
                        f"Ensemble diversity too low: {self._diversity_metrics.diversity_score:.3f} "
                        f"< {min_diversity_threshold}. Recommendations: "
                        f"{self._diversity_metrics.recommendations}"
                    )

        # Step 2: Train meta-learner on OOF predictions
        # MOD-001 FIX: When passthrough=True, X_train must be trimmed to match
        # oof_predictions size (sequence models produce fewer samples due to lookback)
        meta_features_train = oof_predictions
        y_train_meta = y_train
        sample_weights_meta = sample_weights
        if passthrough:
            if oof_predictions.shape[0] < X_train.shape[0]:
                offset = X_train.shape[0] - oof_predictions.shape[0]

                # MOD-001: Validate offset is reasonable (should match seq_len-1 for sequence models)
                # If offset exceeds 50% of training data, something is wrong
                max_reasonable_offset = X_train.shape[0] // 2
                if offset > max_reasonable_offset:
                    raise ValueError(
                        f"MOD-001: Passthrough offset {offset} exceeds 50% of training data "
                        f"({X_train.shape[0]} samples). This indicates a severe data alignment issue. "
                        f"Expected offset should match sequence_length-1 for sequence models."
                    )

                X_train_trimmed = X_train[offset:]
                y_train_meta = y_train[offset:]
                if sample_weights is not None:
                    sample_weights_meta = sample_weights[offset:]

                # MOD-001: Validate shapes after trimming
                if X_train_trimmed.shape[0] != oof_predictions.shape[0]:
                    raise ValueError(
                        f"MOD-001: Shape mismatch after passthrough trimming - "
                        f"X_train_trimmed.shape[0]={X_train_trimmed.shape[0]} != "
                        f"oof_predictions.shape[0]={oof_predictions.shape[0]}"
                    )
                if len(y_train_meta) != oof_predictions.shape[0]:
                    raise ValueError(
                        f"MOD-001: Label alignment error after passthrough trimming - "
                        f"len(y_train_meta)={len(y_train_meta)} != "
                        f"oof_predictions.shape[0]={oof_predictions.shape[0]}"
                    )

                logger.info(
                    f"MOD-001: Passthrough mode - trimmed X_train from {X_train.shape[0]} to "
                    f"{X_train_trimmed.shape[0]} samples (offset={offset}) to match OOF predictions"
                )
                meta_features_train = np.hstack([X_train_trimmed, oof_predictions])
            else:
                meta_features_train = np.hstack([X_train, oof_predictions])

        logger.info(f"Training meta-learner on {meta_features_train.shape[1]} features")

        self._meta_learner = ModelRegistry.create(
            self._meta_learner_name,
            config=meta_learner_config,
        )

        # Use a portion of validation set for meta-learner validation
        # For heterogeneous ensembles, pass both 2D and 3D data
        val_predictions = self._generate_base_predictions(
            X_val, fold_models, use_probabilities, X_seq=X_val_seq
        )

        # Align validation labels with predictions (may be trimmed for heterogeneous ensembles)
        y_val_aligned = y_val
        X_val_aligned = X_val
        if val_predictions.shape[0] < len(y_val):
            offset = len(y_val) - val_predictions.shape[0]
            y_val_aligned = y_val[offset:]
            X_val_aligned = X_val[offset:]

        meta_features_val = val_predictions
        if passthrough:
            meta_features_val = np.hstack([X_val_aligned, val_predictions])

        meta_metrics = self._meta_learner.fit(
            X_train=meta_features_train,
            y_train=y_train_meta,
            X_val=meta_features_val,
            y_val=y_val_aligned,
            sample_weights=sample_weights_meta,
        )

        # Step 3: Keep fold models for inference (use average of folds)
        self._base_models = fold_models

        training_time = time.time() - start_time
        self._is_fitted = True

        # Compute ensemble metrics (use stored sequence data for heterogeneous ensembles)
        train_metrics = self._compute_metrics(X_train, y_train, X_seq=X_train_seq)
        val_metrics = self._compute_metrics(X_val, y_val, X_seq=X_val_seq)

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
                "is_heterogeneous": self._is_heterogeneous,
                "tabular_models": list(self._tabular_models) if self._is_heterogeneous else [],
                "sequence_models": list(self._sequence_models) if self._is_heterogeneous else [],
                "diversity_analysis": self._diversity_metrics.to_dict()
                if self._diversity_metrics
                else None,
            },
        )

    def _generate_oof_predictions(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        base_model_names: list[str],
        base_model_configs: dict[str, dict],
        sample_weights: np.ndarray | None,
        use_probabilities: bool,
        label_end_times: pd.Series | None = None,
        purge_bars: int = 60,
        embargo_bars: int = 1440,
        X_train_seq: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[list[BaseModel]], dict[str, dict[str, np.ndarray]]]:
        """
        Generate out-of-fold predictions for all base models.

        Uses PurgedKFold to prevent label leakage from overlapping labels
        and serial correlation.

        For heterogeneous ensembles (mixed tabular + sequence models):
        - Tabular models receive X_train (2D)
        - Sequence models receive X_train_seq (3D) if provided

        Args:
            X_train: Training features (2D for tabular models)
            y_train: Training labels
            base_model_names: Names of base models to train
            base_model_configs: Config overrides per model
            sample_weights: Sample weights
            use_probabilities: If True, use class probabilities; else use class predictions
            label_end_times: When each label is resolved (enables overlapping label purging)
            purge_bars: Number of bars to purge around validation set
            embargo_bars: Number of bars to embargo after validation set
            X_train_seq: Training sequences (3D) for sequence models in heterogeneous ensembles

        Returns:
            Tuple of (oof_predictions, fold_models, oof_per_model)
            - oof_predictions: Stacked OOF predictions for meta-learner
            - fold_models: Trained models per fold
            - oof_per_model: Dict mapping model_name -> {"predictions": array, "probabilities": array}
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
        fold_models: list[list[BaseModel]] = [[] for _ in base_model_names]

        # Per-model OOF storage for diversity analysis
        oof_per_model: dict[str, dict[str, np.ndarray]] = {}
        for model_name in base_model_names:
            oof_per_model[model_name] = {
                "predictions": np.zeros(n_samples, dtype=np.int64),
                "probabilities": np.zeros((n_samples, n_classes)),
            }

        # Create PurgedKFold splitter to prevent label leakage
        # Deferred import to avoid circular dependency with src.validation
        from src.validation.cv.purged_kfold import PurgedKFold, PurgedKFoldConfig

        purged_kfold_config = PurgedKFoldConfig(
            n_splits=self._n_folds,
            purge_bars=purge_bars,  # Prevent overlapping label leakage
            embargo_bars=embargo_bars,  # Break serial correlation
        )
        kfold = PurgedKFold(purged_kfold_config)

        # Convert to DataFrame for PurgedKFold (requires index for label_end_times)
        X_df = pd.DataFrame(X_train)

        # Pre-slice sequence data for heterogeneous ensembles
        X_seq_fold_train_cache: np.ndarray | None = None
        X_seq_fold_val_cache: np.ndarray | None = None
        y_seq_fold_train: np.ndarray | None = None
        y_seq_fold_val: np.ndarray | None = None

        # Calculate offset between tabular and sequence data
        # Sequence data has fewer samples due to windowing (loses seq_len-1 samples at start)
        seq_offset = 0
        if self._is_heterogeneous and X_train_seq is not None:
            seq_offset = n_samples - X_train_seq.shape[0]

            # MOD-002: Validate seq_offset is reasonable
            # If offset exceeds 50% of data, sequence models will have insufficient samples
            max_reasonable_offset = n_samples // 2
            if seq_offset > max_reasonable_offset:
                raise ValueError(
                    f"MOD-002: Sequence offset {seq_offset} exceeds 50% of training data "
                    f"({n_samples} samples). Sequence data has only {X_train_seq.shape[0]} samples. "
                    f"This indicates the sequence_length is too large for the dataset size. "
                    f"Consider reducing sequence_length or using more training data."
                )

            if seq_offset < 0:
                raise ValueError(
                    f"MOD-002: Invalid negative sequence offset {seq_offset}. "
                    f"Sequence data ({X_train_seq.shape[0]}) has more samples than tabular data ({n_samples}). "
                    f"This indicates a data preparation error."
                )

            logger.info(
                f"MOD-002: Heterogeneous data alignment - tabular={n_samples}, "
                f"sequence={X_train_seq.shape[0]}, offset={seq_offset}"
            )

        for fold_idx, (train_idx, val_idx) in enumerate(
            kfold.split(X_df, label_end_times=label_end_times)
        ):
            logger.debug(f"  Fold {fold_idx + 1}/{self._n_folds}")

            # Tabular data slicing (always done)
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]

            # Initialize sequence fold variables (may be set below if heterogeneous)
            X_seq_fold_train_cache: np.ndarray | None = None
            X_seq_fold_val_cache: np.ndarray | None = None
            y_seq_fold_train: np.ndarray | None = None
            y_seq_fold_val: np.ndarray | None = None

            # Sequence data slicing (only if heterogeneous and seq data provided)
            # Must adjust indices by offset since sequence data is shorter
            if self._is_heterogeneous and X_train_seq is not None:
                # Filter indices to only those valid for sequence data (>= offset)
                # and adjust by subtracting offset
                seq_train_idx = train_idx[train_idx >= seq_offset] - seq_offset
                seq_val_idx = val_idx[val_idx >= seq_offset] - seq_offset

                # MOD-002: Validate minimum fold size for sequence models
                # If seq_offset removes more than 50% of a fold's samples, raise error
                original_train_size = len(train_idx)
                filtered_train_size = len(seq_train_idx)
                if original_train_size > 0:
                    train_retention_ratio = filtered_train_size / original_train_size
                    if train_retention_ratio < 0.5:
                        raise ValueError(
                            f"MOD-002: Fold {fold_idx + 1} sequence data retention too low. "
                            f"Original train samples: {original_train_size}, "
                            f"after seq_offset filter: {filtered_train_size} "
                            f"(retention: {train_retention_ratio:.1%}). "
                            f"Minimum required: 50%. seq_offset={seq_offset}. "
                            f"Consider reducing sequence_length or using more training data."
                        )

                original_val_size = len(val_idx)
                filtered_val_size = len(seq_val_idx)
                if original_val_size > 0:
                    val_retention_ratio = filtered_val_size / original_val_size
                    if val_retention_ratio < 0.5:
                        raise ValueError(
                            f"MOD-002: Fold {fold_idx + 1} sequence validation data retention too low. "
                            f"Original val samples: {original_val_size}, "
                            f"after seq_offset filter: {filtered_val_size} "
                            f"(retention: {val_retention_ratio:.1%}). "
                            f"Minimum required: 50%. seq_offset={seq_offset}. "
                            f"Consider reducing sequence_length or using more training data."
                        )

                # MOD-003: Validate that we have valid samples after filtering
                if len(seq_train_idx) == 0:
                    raise ValueError(
                        f"MOD-003: No valid sequence training samples in fold {fold_idx + 1} "
                        f"(all {len(train_idx)} train indices < seq_offset={seq_offset}). "
                        f"This indicates insufficient data for sequence models with the "
                        f"current lookback window. Consider reducing sequence_length or "
                        f"using more training data."
                    )
                if len(seq_val_idx) == 0:
                    raise ValueError(
                        f"MOD-003: No valid sequence validation samples in fold {fold_idx + 1} "
                        f"(all {len(val_idx)} val indices < seq_offset={seq_offset}). "
                        f"This indicates insufficient data for sequence models with the "
                        f"current lookback window. Consider reducing sequence_length or "
                        f"using more training data."
                    )

                X_seq_fold_train_cache = X_train_seq[seq_train_idx]
                X_seq_fold_val_cache = X_train_seq[seq_val_idx]
                # Also need aligned labels for sequence models
                y_seq_fold_train = y_train[train_idx[train_idx >= seq_offset]]
                y_seq_fold_val = y_train[val_idx[val_idx >= seq_offset]]

            w_train = None
            if sample_weights is not None:
                w_train = sample_weights[train_idx]

            # Train each base model on this fold
            for model_idx, model_name in enumerate(base_model_names):
                model_config = base_model_configs.get(model_name, {})
                model = ModelRegistry.create(model_name, config=model_config)

                # Select appropriate data format and labels based on model type
                # For heterogeneous ensembles: tabular models get 2D, sequence models get 3D
                is_seq_model = self._is_heterogeneous and model_name in self._sequence_models
                if is_seq_model:
                    if X_seq_fold_train_cache is not None:
                        # These are always set together in the if block above
                        assert X_seq_fold_val_cache is not None
                        assert y_seq_fold_train is not None
                        assert y_seq_fold_val is not None
                        model_X_train = X_seq_fold_train_cache
                        model_X_val = X_seq_fold_val_cache
                        model_y_train = y_seq_fold_train
                        model_y_val = y_seq_fold_val
                        # Sequence models use filtered val_idx for OOF storage
                        oof_val_idx = val_idx[val_idx >= seq_offset]

                        # MOD-003: Validate data dimensions for sequence models
                        if model_X_train.ndim != 3:
                            raise ValueError(
                                f"MOD-003: Sequence model '{model_name}' requires 3D data "
                                f"(n_samples, seq_len, n_features) but received {model_X_train.ndim}D "
                                f"data with shape {model_X_train.shape}. "
                                f"Ensure X_train_seq is properly prepared with sequence windows."
                            )
                    else:
                        # MOD-003 FIX: Raise error instead of silently falling back to wrong data type
                        # The original code logged a warning but proceeded with 2D tabular data,
                        # which would cause the sequence model to fail or produce incorrect results
                        raise ValueError(
                            f"MOD-003: Sequence model '{model_name}' requires 3D data but "
                            f"no sequence data (X_train_seq) was provided for fold {fold_idx + 1}. "
                            f"Heterogeneous stacking ensembles with sequence models require both "
                            f"2D tabular data (X_train) and 3D sequence data (X_train_seq). "
                            f"Provide X_train_seq parameter to fit() or remove sequence models "
                            f"from base_model_names."
                        )
                else:
                    # Tabular model or homogeneous ensemble
                    model_X_train = X_fold_train
                    model_X_val = X_fold_val
                    model_y_train = y_fold_train
                    model_y_val = y_fold_val
                    oof_val_idx = val_idx

                # Get sample weights aligned with model data
                model_w_train = None
                if sample_weights is not None:
                    if is_seq_model and seq_offset > 0:
                        # Sequence model weights need to be filtered
                        model_w_train = sample_weights[train_idx[train_idx >= seq_offset]]
                    else:
                        model_w_train = w_train

                model.fit(
                    X_train=model_X_train,
                    y_train=model_y_train,
                    X_val=model_X_val,
                    y_val=model_y_val,
                    sample_weights=model_w_train,
                )

                # Generate predictions for OOF samples
                output = model.predict(model_X_val)

                if use_probabilities:
                    start_col = model_idx * n_classes
                    end_col = start_col + n_classes
                    oof_predictions[oof_val_idx, start_col:end_col] = output.class_probabilities
                else:
                    oof_predictions[oof_val_idx, model_idx] = output.class_predictions

                # Store per-model predictions for diversity analysis
                oof_per_model[model_name]["predictions"][oof_val_idx] = output.class_predictions
                oof_per_model[model_name]["probabilities"][oof_val_idx] = output.class_probabilities

                fold_models[model_idx].append(model)

        return oof_predictions, fold_models, oof_per_model

    def _generate_base_predictions(
        self,
        X: np.ndarray,
        fold_models: list[list[BaseModel]],
        use_probabilities: bool,
        X_seq: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Generate predictions from base models (averaging across folds).

        For heterogeneous ensembles, uses appropriate data format for each model.

        Args:
            X: Tabular features (2D)
            fold_models: List of fold model lists per base model
            use_probabilities: Whether to use probabilities or class predictions
            X_seq: Sequence features (3D) for sequence models in heterogeneous ensembles

        Returns:
            Predictions array (2D) - always 2D regardless of input model types
        """
        n_classes = 3
        n_models = len(fold_models)

        # For heterogeneous ensembles, use the smaller of tabular/sequence size
        # since we can only make predictions where all models have data
        n_samples = X.shape[0]
        seq_offset = 0
        if self._is_heterogeneous and X_seq is not None and X_seq.shape[0] < n_samples:
            seq_offset = n_samples - X_seq.shape[0]
            original_n_samples = n_samples
            n_samples = X_seq.shape[0]  # Use sequence size as common denominator
            # MOD-008 FIX: Log warning when trimming occurs
            logger.warning(
                f"MOD-008: Trimming tabular data to match sequence data size. "
                f"Original: {original_n_samples}, After trim: {n_samples}, "
                f"Offset: {seq_offset} samples removed from start."
            )
            # Trim tabular data to match
            X = X[seq_offset:]
            # MOD-008 FIX: Validate shapes after trimming
            if X.shape[0] != n_samples:
                raise ValueError(
                    f"MOD-008: Shape mismatch after trimming - "
                    f"X.shape[0]={X.shape[0]} != expected n_samples={n_samples}"
                )
            if X_seq.shape[0] != n_samples:
                raise ValueError(
                    f"MOD-008: Shape mismatch after trimming - "
                    f"X_seq.shape[0]={X_seq.shape[0]} != expected n_samples={n_samples}"
                )

        if use_probabilities:
            predictions = np.zeros((n_samples, n_models * n_classes))
        else:
            predictions = np.zeros((n_samples, n_models))

        for model_idx, models in enumerate(fold_models):
            # Get model name from the first fold's model
            model_name = self._base_model_names[model_idx]

            # Select appropriate data format for this model
            if self._is_heterogeneous and model_name in self._sequence_models:
                model_X = X_seq if X_seq is not None else X
            else:
                model_X = X

            # Average predictions across folds
            if use_probabilities:
                probs = np.zeros((n_samples, n_classes))
                for model in models:
                    output = model.predict(model_X)
                    probs += output.class_probabilities
                probs /= len(models)

                start_col = model_idx * n_classes
                end_col = start_col + n_classes
                predictions[:, start_col:end_col] = probs
            else:
                # For class predictions, use voting
                all_preds = []
                for model in models:
                    output = model.predict(model_X)
                    all_preds.append(output.class_predictions)
                all_preds = np.array(all_preds)
                # Mode across folds
                from scipy import stats

                mode_result = stats.mode(all_preds, axis=0, keepdims=False)
                predictions[:, model_idx] = mode_result.mode

        return predictions

    def predict(
        self,
        X: np.ndarray,
        X_seq: np.ndarray | None = None,
        use_cache: bool = True,
    ) -> PredictionOutput:
        """
        Generate stacking ensemble predictions.

        Args:
            X: Tabular features (2D) for tabular models
            X_seq: Sequence features (3D) for sequence models in heterogeneous ensembles.
                If None and ensemble is heterogeneous, uses stored X_val_seq or falls back to X.
            use_cache: If True (default), fall back to cached X_val_seq when X_seq is None.
                If False, do not use cached data - X_seq must be provided for heterogeneous
                ensembles with sequence models, otherwise an error is raised. (MOD-007)

        Returns:
            PredictionOutput with class predictions, probabilities, and confidence
        """
        self._validate_fitted()
        self._validate_input_shape(X, "X")

        use_probabilities = self._config.get("use_probabilities", True)
        passthrough = self._config.get("passthrough", False)

        # For heterogeneous ensembles, determine sequence data to use (MOD-007)
        predict_X_seq: np.ndarray | None
        if X_seq is not None:
            predict_X_seq = X_seq
        elif use_cache:
            predict_X_seq = self._X_val_seq
            if predict_X_seq is not None:
                logger.debug(
                    "MOD-007: Using cached X_val_seq for prediction "
                    f"({estimate_array_size(predict_X_seq) / 1024**2:.1f}MB)"
                )
        else:
            # use_cache=False and X_seq not provided
            predict_X_seq = None
            if self._is_heterogeneous and self._sequence_models:
                raise ValueError(
                    "MOD-007: Heterogeneous ensemble has sequence models but X_seq not provided "
                    "and use_cache=False. Either provide X_seq or set use_cache=True to use "
                    "cached validation sequence data."
                )

        # Generate base model predictions
        # For heterogeneous ensembles, each model gets appropriate data format
        base_predictions = self._generate_base_predictions(
            X, self._base_models, use_probabilities, X_seq=predict_X_seq
        )

        # Build meta-learner input
        # MOD-001 FIX: When passthrough=True, X must be trimmed to match
        # base_predictions size (sequence models produce fewer samples)
        meta_features = base_predictions
        if passthrough:
            if base_predictions.shape[0] < X.shape[0]:
                offset = X.shape[0] - base_predictions.shape[0]
                X_trimmed = X[offset:]
                meta_features = np.hstack([X_trimmed, base_predictions])
            else:
                meta_features = np.hstack([X, base_predictions])

        # Get meta-learner predictions
        if self._meta_learner is None:
            raise ValueError("Cannot predict: meta-learner not fitted")
        output = self._meta_learner.predict(meta_features)

        return PredictionOutput(
            class_predictions=output.class_predictions,
            class_probabilities=output.class_probabilities,
            confidence=output.confidence,
            metadata={
                "ensemble": "stacking",
                "base_models": self._base_model_names,
                "meta_learner": self._meta_learner_name,
                "is_heterogeneous": self._is_heterogeneous,
            },
        )

    def predict_proba(
        self,
        X: np.ndarray,
        X_seq: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return class probabilities from meta-learner.

        Args:
            X: Tabular features (2D)
            X_seq: Sequence features (3D) for heterogeneous ensembles

        Returns:
            Class probabilities array
        """
        output = self.predict(X, X_seq=X_seq)
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
        if self._meta_learner is None:
            raise ValueError("Cannot save ensemble: meta-learner not fitted")
        self._meta_learner.save(meta_dir)

        # Save ensemble metadata
        metadata = {
            "config": self._config,
            "base_model_names": self._base_model_names,
            "meta_learner_name": self._meta_learner_name,
            "n_folds": self._n_folds,
            "feature_names": self._feature_names,
            "is_heterogeneous": self._is_heterogeneous,
            "tabular_models": list(self._tabular_models),
            "sequence_models": list(self._sequence_models),
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
        self._is_heterogeneous = metadata.get("is_heterogeneous", False)
        self._tabular_models = set(metadata.get("tabular_models", []))
        self._sequence_models = set(metadata.get("sequence_models", []))
        # MOD-007: Sequence data not stored (too large), cache cleared on load
        self._X_train_seq = None
        self._X_val_seq = None
        self._seq_cache_size_bytes = 0

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

    def clear_cache(self) -> int:
        """
        Clear cached sequence data to free memory (MOD-007).

        Call this after training is complete if you don't need the cached
        X_train_seq and X_val_seq data for subsequent predictions.
        After calling this method, predict() will require X_seq to be
        provided explicitly when use_cache=True (otherwise will fall back to None).

        Returns:
            Number of bytes freed
        """
        freed_bytes = self._seq_cache_size_bytes

        if freed_bytes > 0:
            logger.info(f"MOD-007: Clearing cached sequence data: {freed_bytes / 1024**2:.1f}MB")

        self._X_train_seq = None
        self._X_val_seq = None
        self._seq_cache_size_bytes = 0

        # Trigger garbage collection to actually free memory
        import gc

        gc.collect()

        return freed_bytes

    def get_cache_size(self) -> int:
        """
        Get size of cached sequence data in bytes (MOD-007).

        Returns:
            Size of cached data in bytes, or 0 if nothing is cached
        """
        return self._seq_cache_size_bytes

    def get_feature_importance(self) -> dict[str, float] | None:
        """Return meta-learner feature importances."""
        if not self._is_fitted or self._meta_learner is None:
            return None

        # Meta-learner importance tells which base model predictions matter
        return self._meta_learner.get_feature_importance()

    def set_feature_names(self, names: list[str]) -> None:
        """Set feature names for base models."""
        self._feature_names = names
        for fold_models in self._base_models:
            for model in fold_models:
                set_fn = getattr(model, "set_feature_names", None)
                if set_fn is not None:
                    set_fn(names)

    def _compute_metrics(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        X_seq: np.ndarray | None = None,
    ) -> dict[str, float]:
        """
        Compute accuracy and F1 for a dataset.

        Args:
            X: Tabular features (2D)
            y_true: True labels
            X_seq: Sequence features (3D) for heterogeneous ensembles
        """
        output = self.predict(X, X_seq=X_seq)
        y_pred = output.class_predictions

        # Align y_true with predictions (may be trimmed for heterogeneous ensembles)
        if len(y_pred) < len(y_true):
            offset = len(y_true) - len(y_pred)
            y_true = y_true[offset:]

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            # sklearn accepts 0, 1, or "warn" for zero_division, but type stubs are outdated
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),  # pyright: ignore[reportArgumentType]
        }

    def _analyze_oof_diversity(
        self,
        oof_per_model: dict[str, dict[str, np.ndarray]],
        y_train: np.ndarray,
        min_diversity_threshold: float = 0.3,
        correlation_threshold: float = 0.8,
    ) -> DiversityMetrics:
        """
        Analyze diversity of OOF predictions from base models.

        Args:
            oof_per_model: Dict mapping model_name -> {"predictions": array, "probabilities": array}
            y_train: Ground truth labels
            min_diversity_threshold: Minimum acceptable diversity score
            correlation_threshold: Maximum acceptable pairwise correlation

        Returns:
            DiversityMetrics with comprehensive diversity analysis
        """
        self._diversity_analyzer = DiversityAnalyzer(
            min_diversity_threshold=min_diversity_threshold,
            correlation_threshold=correlation_threshold,
            n_classes=3,
        )

        # Extract predictions and probabilities for diversity analysis
        base_predictions = {name: data["predictions"] for name, data in oof_per_model.items()}
        base_probabilities = {name: data["probabilities"] for name, data in oof_per_model.items()}

        metrics = self._diversity_analyzer.analyze(
            base_predictions=base_predictions,
            base_probabilities=base_probabilities,
            y_true=y_train,
        )

        # Log diversity analysis results
        logger.info(
            f"Diversity analysis: score={metrics.diversity_score:.3f}, "
            f"correlation={metrics.pairwise_correlation:.3f}, "
            f"disagreement={metrics.disagreement:.3f}, "
            f"q_statistic={metrics.q_statistic:.3f}"
        )

        if metrics.recommendations:
            for rec in metrics.recommendations[:3]:  # Limit to top 3 recommendations
                logger.warning(f"Diversity recommendation: {rec}")

        return metrics

    def get_diversity_metrics(self) -> DiversityMetrics | None:
        """
        Get the diversity metrics computed during training.

        Returns:
            DiversityMetrics if diversity analysis was performed, None otherwise
        """
        return self._diversity_metrics

    def suggest_model_removal(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_val_seq: np.ndarray | None = None,
    ) -> list[str]:
        """
        Suggest base models to remove due to redundancy.

        Uses validation set predictions to identify models that provide
        similar predictions and could be removed without losing diversity.

        Args:
            X_val: Validation features (2D)
            y_val: Validation labels
            X_val_seq: Validation sequences (3D) for heterogeneous ensembles

        Returns:
            List of model names that are redundant and could be removed
        """
        self._validate_fitted()

        if self._diversity_analyzer is None:
            self._diversity_analyzer = DiversityAnalyzer()

        # Generate predictions from each base model
        _use_probabilities = self._config.get("use_probabilities", True)  # Reserved for future use
        base_predictions: dict[str, np.ndarray] = {}

        for model_idx, model_name in enumerate(self._base_model_names):
            # Select appropriate data format
            if self._is_heterogeneous and model_name in self._sequence_models:
                model_X = X_val_seq if X_val_seq is not None else X_val
            else:
                model_X = X_val

            # Average predictions across folds
            all_preds = []
            for model in self._base_models[model_idx]:
                output = model.predict(model_X)
                all_preds.append(output.class_predictions)

            # Mode across folds for final prediction
            all_preds = np.array(all_preds)
            from scipy import stats as sp_stats

            mode_result = sp_stats.mode(all_preds, axis=0, keepdims=False)
            base_predictions[model_name] = mode_result.mode

        # Align y_val with predictions if needed
        n_samples = len(next(iter(base_predictions.values())))
        y_val_aligned = y_val
        if n_samples < len(y_val):
            offset = len(y_val) - n_samples
            y_val_aligned = y_val[offset:]

        return self._diversity_analyzer.suggest_model_removal(
            base_predictions=base_predictions,
            y_true=y_val_aligned,
        )


__all__ = ["StackingEnsemble"]
