# src/training/services/model_training.py
"""Service for training individual models."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.data.adapters import PreparedData

if TYPE_CHECKING:
    from src.core.container import TimeSeriesDataContainer

logger = logging.getLogger(__name__)


@dataclass
class ModelTrainingRequest:
    """Request to train a single model."""

    model_name: str
    horizon: int
    prepared_data: PreparedData
    sequence_length: int = 60
    output_dir: Path | None = None
    optimize_hyperparams: bool = False
    hyperparam_trials: int = 100
    n_splits: int = 5
    scoring: str = "f1_weighted"  # Optimization metric (from PipelineConfig.optuna_metric)
    use_feature_selection: bool = True
    max_epochs: int | None = None  # Cap epochs for neural models in Optuna
    cv_method: str = "purged_kfold"  # CV method: "purged_kfold" or "cpcv"
    batch_size: int | None = None  # Override batch size (used by OOM retry)
    embargo_bars: int | None = None  # Pipeline embargo (overrides horizon*2 default)


@dataclass
class ModelTrainingResult:
    """Result from training a single model."""

    model_name: str
    horizon: int
    trainer: Any
    metrics: dict[str, float] = field(default_factory=dict)
    training_time_seconds: float = 0.0
    n_features: int = 0
    data_rank: int = 2


class ModelTrainingService:
    """
    Service for training individual models.

    Responsibilities:
    - Create model trainer with appropriate configuration
    - Execute training with prepared data
    - Return structured result with metrics and trained model

    Does NOT:
    - Prepare data (handled by adapters/UnifiedDataPreparation)
    - Generate OOF predictions (handled by OOFGenerationService)
    - Save artifacts (handled by ArtifactManager)
    - Orchestrate multiple models (handled by UnifiedTrainingOrchestrator)
    """

    def __init__(self) -> None:
        """Initialize ModelTrainingService."""
        self._tuning_service: Any = None

    def _get_tuning_service(self) -> Any:
        """Lazy load HyperparameterTuningService to avoid circular imports."""
        if self._tuning_service is None:
            from .hyperparameter_tuning import (
                HyperparameterTuningService,
            )

            self._tuning_service = HyperparameterTuningService()
        return self._tuning_service

    def train_model(self, request: ModelTrainingRequest) -> ModelTrainingResult:
        """
        Train a single model with prepared data.

        Args:
            request: ModelTrainingRequest containing model name, horizon,
                    prepared data, and training configuration

        Returns:
            ModelTrainingResult with metrics and trained model
        """
        from src.models import Trainer, TrainerConfig

        start = time.time()

        prepared = request.prepared_data
        model_name = request.model_name
        horizon = request.horizon

        # Determine output directory
        # Note: callers already append h{horizon} to output_dir, so don't duplicate it
        output_dir = request.output_dir
        if output_dir is not None:
            model_output_dir = output_dir
        else:
            # Default output directory if none provided
            model_output_dir = Path("./output") / model_name / f"h{horizon}"

        # Create trainer config with training params in model_config
        # so neural models receive max_epochs, batch_size etc. via model.fit()
        _model_config = {}
        if request.max_epochs is not None:
            _model_config["max_epochs"] = request.max_epochs
            _model_config["early_stopping_patience"] = max(1, request.max_epochs // 2)
        if request.batch_size is not None:
            _model_config["batch_size"] = request.batch_size
        trainer_config = TrainerConfig(
            model_name=model_name,
            horizon=horizon,
            sequence_length=request.sequence_length,
            output_dir=model_output_dir,
            use_feature_selection=request.use_feature_selection,
            max_epochs=request.max_epochs if request.max_epochs is not None else 100,
            model_config=_model_config,
        )

        # CRITICAL: Filter invalid labels (-99) before any training
        # The sentinel value marks invalid/ambiguous samples that must be excluded
        prepared = prepared.filter_invalid_labels()

        # Hyperparameter optimization if enabled
        if request.optimize_hyperparams:
            trainer_config = self._optimize_hyperparams(trainer_config, prepared, request)

        # Create trainer and run
        trainer = Trainer(trainer_config)

        # For 3D/4D data, use run_prepared() to bypass container flattening
        # For 2D data, use traditional container path
        if prepared.data_rank >= 3:
            # Sequence/multi-stream models: use pre-shaped data directly
            logger.info(
                f"Using run_prepared() for {prepared.data_rank}D data "
                f"(shape: {prepared.X_train.shape})"
            )
            training_results = trainer.run_prepared(prepared)
        else:
            # Tabular models: use container path
            container = self._build_container(prepared, horizon)
            training_results = trainer.run(container)

        training_time = time.time() - start

        return ModelTrainingResult(
            model_name=model_name,
            horizon=horizon,
            metrics=training_results.get("evaluation_metrics", {}),
            trainer=trainer,
            training_time_seconds=training_time,
            n_features=prepared.n_features,
            data_rank=prepared.data_rank,
        )

    def _build_container(self, prepared: PreparedData, horizon: int) -> TimeSeriesDataContainer:
        """
        Build TimeSeriesDataContainer from PreparedData.

        Handles different data ranks (2D, 3D, 4D) by flattening higher
        dimensional data appropriately. The model handles any needed reshaping.

        Args:
            prepared: PreparedData from adapter
            horizon: Label horizon for column naming

        Returns:
            TimeSeriesDataContainer ready for training
        """
        from src.core.container import TimeSeriesDataContainer

        # Handle different data ranks
        if prepared.data_rank == 2:
            # Tabular: (n_samples, n_features)
            X_train_df = pd.DataFrame(
                prepared.X_train,
                columns=prepared.feature_names,
            )
            X_val_df = pd.DataFrame(
                prepared.X_val,
                columns=prepared.feature_names,
            )
            X_test_df = (
                pd.DataFrame(
                    prepared.X_test,
                    columns=prepared.feature_names,
                )
                if prepared.has_test
                else None
            )
        else:
            # Sequence (3D/4D): Flatten for container, model handles reshaping
            n_train = prepared.X_train.shape[0]
            n_val = prepared.X_val.shape[0]
            n_features = np.prod(prepared.X_train.shape[1:])

            X_train_df = pd.DataFrame(
                prepared.X_train.reshape(n_train, -1),
                columns=[f"f{i}" for i in range(n_features)],
            )
            X_val_df = pd.DataFrame(
                prepared.X_val.reshape(n_val, -1),
                columns=[f"f{i}" for i in range(n_features)],
            )
            if prepared.has_test and prepared.X_test is not None:
                n_test = prepared.X_test.shape[0]
                X_test_df = pd.DataFrame(
                    prepared.X_test.reshape(n_test, -1),
                    columns=[f"f{i}" for i in range(n_features)],
                )
            else:
                X_test_df = None

        # Build sample weights
        if prepared.has_weights:
            sample_weights = pd.Series(prepared.train_weights)
        else:
            sample_weights = pd.Series(np.ones(len(prepared.y_train)))

        container = TimeSeriesDataContainer.from_dataframes(
            train_df=self._build_container_df(
                X_train_df,
                pd.Series(prepared.y_train),
                sample_weights,
                horizon,
            ),
            val_df=self._build_container_df(
                X_val_df,
                pd.Series(prepared.y_val),
                None,
                horizon,
            ),
            test_df=(
                self._build_container_df(
                    X_test_df,
                    pd.Series(prepared.y_test) if prepared.has_test else None,
                    None,
                    horizon,
                )
                if prepared.has_test and X_test_df is not None
                else None
            ),
            horizon=horizon,
            feature_columns=list(X_train_df.columns),
        )

        return container

    def _build_container_df(
        self,
        X_df: pd.DataFrame | None,
        y_series: pd.Series | None,
        weights_series: pd.Series | None,
        horizon: int,
    ) -> pd.DataFrame | None:
        """
        Build a DataFrame suitable for TimeSeriesDataContainer.from_dataframes.

        Args:
            X_df: Feature DataFrame
            y_series: Label series
            weights_series: Sample weights series (optional)
            horizon: Label horizon for column naming

        Returns:
            Combined DataFrame with features, labels, and weights, or None
        """
        if X_df is None or y_series is None:
            return None

        df = X_df.copy()
        df[f"label_h{horizon}"] = y_series.values
        if weights_series is not None:
            df[f"sample_weight_h{horizon}"] = weights_series.values
        return df

    def _optimize_hyperparams(
        self,
        trainer_config: Any,
        prepared: PreparedData,
        request: ModelTrainingRequest,
    ) -> Any:
        """
        Run Optuna hyperparameter optimization.

        Args:
            trainer_config: TrainerConfig to update with best params
            prepared: PreparedData with training data
            request: Original training request with tuning params

        Returns:
            Updated trainer_config with optimized hyperparameters
        """
        from .hyperparameter_tuning import TuningRequest

        tuning_service = self._get_tuning_service()

        tuning_request = TuningRequest(
            model_name=trainer_config.model_name,
            horizon=trainer_config.horizon,
            prepared_data=prepared,
            n_splits=request.n_splits,
            n_trials=request.hyperparam_trials,
            scoring=request.scoring,
            max_epochs=request.max_epochs,
            cv_method=request.cv_method,
            embargo_bars=request.embargo_bars,
        )

        result = tuning_service.optimize(tuning_request)

        # Apply best params to model_config so they reach ModelRegistry.create().
        # Previously used setattr() which put params as ad-hoc TrainerConfig attributes,
        # but Trainer only passes model_config to ModelRegistry — so params were silently lost.
        if not hasattr(trainer_config, "model_config") or trainer_config.model_config is None:
            trainer_config.model_config = {}
        trainer_config.model_config.update(result.best_params)

        return trainer_config
