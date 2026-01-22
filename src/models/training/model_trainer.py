"""
ModelTrainer - Unified model training with adapter integration.

Bridges PHASE_2 adapters with existing model training infrastructure.
Uses PipelineConfig from src/core.

PHASE_3 Training Orchestration:
- Integrates with UnifiedDataPreparation from PHASE_2
- Uses PipelineConfig as the ONLY configuration source
- Handles different data ranks (2D/3D/4D) transparently
- Produces TrainedModelArtifact with comprehensive metadata
- Supports hyperparameter optimization via Optuna

Usage:
    from src.core import PipelineConfig
    from src.models.training import ModelTrainer

    config = PipelineConfig(...)
    trainer = ModelTrainer(config)

    # Train single model
    artifact = trainer.train_model("xgboost", df, horizon=20)

    # Train all configured models
    artifacts = trainer.train_all(df)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.core import (
    PipelineConfig,
    MODEL_DATA_RANKS,
    MODEL_ADAPTER_MAP,
    DataRank,
)
from src.data.adapters import (
    UnifiedDataPreparation,
    PreparedData,
    AdapterFactory,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainedModelArtifact:
    """
    Artifact from a trained model.

    Contains all metadata and results from a single model training run,
    including metrics, paths, and optionally the trainer instance.

    Attributes:
        model_name: Name of the model (e.g., "xgboost", "lstm")
        horizon: Prediction horizon in bars
        data_rank: Dimensionality of input data (2, 3, or 4)
        adapter_type: Type of adapter used ("tabular", "sequence", "multi_stream")
        feature_names: List of feature column names
        n_train_samples: Number of training samples
        n_val_samples: Number of validation samples
        metrics: Dictionary of evaluation metrics
        training_time_seconds: Total training time
        model_path: Path to saved model artifacts
        trainer: Optional Trainer instance (for inference)
        hyperparams: Dictionary of hyperparameters used
    """

    model_name: str
    horizon: int
    data_rank: int
    adapter_type: str
    feature_names: List[str]
    n_train_samples: int
    n_val_samples: int
    metrics: Dict[str, float]
    training_time_seconds: float
    model_path: Optional[Path] = None
    trainer: Optional[Any] = None  # Trainer instance
    hyperparams: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert artifact to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "data_rank": self.data_rank,
            "adapter_type": self.adapter_type,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "n_train_samples": self.n_train_samples,
            "n_val_samples": self.n_val_samples,
            "metrics": self.metrics,
            "training_time_seconds": self.training_time_seconds,
            "model_path": str(self.model_path) if self.model_path else None,
            "hyperparams": self.hyperparams,
        }

    @property
    def key(self) -> str:
        """Generate unique key for this artifact."""
        return f"{self.model_name}_h{self.horizon}"

    def summary(self) -> str:
        """Generate summary string for logging."""
        val_f1 = self.metrics.get("val_f1", self.metrics.get("f1", 0))
        return (
            f"TrainedModelArtifact({self.model_name}, h={self.horizon}): "
            f"train={self.n_train_samples}, val={self.n_val_samples}, "
            f"features={len(self.feature_names)}, rank={self.data_rank}D, "
            f"val_f1={val_f1:.4f}, time={self.training_time_seconds:.1f}s"
        )


class ModelTrainer:
    """
    Unified model training with adapter integration.

    Handles:
    - Data preparation via PHASE_2 adapters
    - Model instantiation via ModelRegistry
    - Training with proper CV
    - Artifact saving

    This class bridges PHASE_2 data preparation with the existing
    model training infrastructure. It uses PipelineConfig as the
    single source of truth for all configuration.

    Usage:
        from src.core import PipelineConfig
        from src.models.training import ModelTrainer

        config = PipelineConfig(...)
        trainer = ModelTrainer(config)

        # Train single model
        artifact = trainer.train_model("xgboost", df, horizon=20)

        # Train all configured models
        artifacts = trainer.train_all(df)

    Attributes:
        config: PipelineConfig instance
        output_dir: Path to output directory
    """

    def __init__(self, config: PipelineConfig) -> None:
        """
        Initialize ModelTrainer.

        Args:
            config: PipelineConfig instance with all training settings
        """
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data preparation (PHASE_2)
        self._data_prep = UnifiedDataPreparation(config)

        # Trained artifacts
        self._artifacts: Dict[str, TrainedModelArtifact] = {}

        logger.info("ModelTrainer initialized")
        logger.info(f"  Models: {config.models}")
        logger.info(f"  Horizons: {config.horizons}")

    def train_model(
        self,
        model_name: str,
        df: pd.DataFrame,
        horizon: int,
        additional_dfs: Optional[Dict[str, pd.DataFrame]] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> TrainedModelArtifact:
        """
        Train a single model.

        This method handles the complete training workflow:
        1. Determine model metadata (data rank, adapter type)
        2. Prepare data using PHASE_2 adapters
        3. Create and configure Trainer
        4. Run hyperparameter optimization if enabled
        5. Train model and evaluate
        6. Create and store artifact

        Args:
            model_name: Model name (e.g., "xgboost", "lstm")
            df: Raw OHLCV DataFrame with features and labels
            horizon: Prediction horizon in bars
            additional_dfs: Optional additional timeframe data for multi-stream models
            hyperparams: Optional hyperparameter overrides

        Returns:
            TrainedModelArtifact with trained model and metadata

        Raises:
            ValueError: If model_name is unknown
        """
        logger.info(f"Training {model_name} for horizon {horizon}...")
        start_time = time.time()

        # Get model metadata
        model_key = model_name.lower()
        data_rank = MODEL_DATA_RANKS.get(model_key, 2)
        adapter_type = MODEL_ADAPTER_MAP.get(model_key, "tabular")

        logger.info(f"  Data rank: {data_rank}D, Adapter: {adapter_type}")

        # Prepare data using PHASE_2 adapters
        prepared = self._data_prep.prepare(
            df=df,
            model_name=model_name,
            additional_dfs=additional_dfs,
        )

        logger.info(f"  Train shape: {prepared.X_train.shape}")
        logger.info(f"  Val shape: {prepared.X_val.shape}")

        # Create and train model
        from src.models import Trainer, TrainerConfig

        model_output_dir = self.output_dir / f"{model_name}_h{horizon}"
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # Build trainer config from PipelineConfig + optional overrides
        trainer_config_kwargs = {
            "model_name": model_name,
            "horizon": horizon,
            "sequence_length": self.config.sequence_length,
            "output_dir": model_output_dir,
            "batch_size": self.config.batch_size,
            "max_epochs": self.config.max_epochs,
            "early_stopping_patience": self.config.early_stopping_patience,
            "random_seed": self.config.random_state,
        }

        # Apply hyperparameter overrides if provided
        if hyperparams:
            trainer_config_kwargs.update(hyperparams)

        trainer_config = TrainerConfig(**trainer_config_kwargs)

        # Hyperparameter optimization if enabled and no overrides provided
        if self.config.optimize_hyperparams and hyperparams is None:
            trainer_config = self._run_hyperparam_optimization(
                trainer_config, prepared
            )

        trainer = Trainer(trainer_config)

        # Build container from prepared data
        from src.core.container import TimeSeriesDataContainer

        # Handle different data ranks for container
        # The container expects 2D data; model will reshape internally if needed
        if data_rank == 2:
            X_train_2d = prepared.X_train
            X_val_2d = prepared.X_val
            X_test_2d = prepared.X_test if prepared.X_test is not None else None
        else:
            # Flatten 3D/4D data for container (model will reshape internally)
            X_train_2d = prepared.X_train.reshape(prepared.X_train.shape[0], -1)
            X_val_2d = prepared.X_val.reshape(prepared.X_val.shape[0], -1)
            X_test_2d = (
                prepared.X_test.reshape(prepared.X_test.shape[0], -1)
                if prepared.X_test is not None
                else None
            )

        # Generate feature names for flattened data
        feature_names = prepared.feature_names or [
            f"f{i}" for i in range(X_train_2d.shape[1])
        ]

        # Ensure we have enough feature names for flattened data
        if len(feature_names) < X_train_2d.shape[1]:
            feature_names = [f"f{i}" for i in range(X_train_2d.shape[1])]

        # Create container with proper column names
        container = TimeSeriesDataContainer(
            X_train=pd.DataFrame(
                X_train_2d, columns=feature_names[: X_train_2d.shape[1]]
            ),
            y_train=pd.Series(prepared.y_train),
            X_val=pd.DataFrame(X_val_2d, columns=feature_names[: X_val_2d.shape[1]]),
            y_val=pd.Series(prepared.y_val),
            X_test=(
                pd.DataFrame(X_test_2d, columns=feature_names[: X_test_2d.shape[1]])
                if X_test_2d is not None
                else None
            ),
            y_test=(
                pd.Series(prepared.y_test) if prepared.y_test is not None else None
            ),
            sample_weights=pd.Series(np.ones(len(prepared.y_train))),
        )

        # Run training
        training_results = trainer.run(container)

        training_time = time.time() - start_time

        # Extract metrics from training results
        metrics = training_results.get("evaluation_metrics", {})

        # Log key metrics
        val_f1 = metrics.get("val_f1", metrics.get("f1", 0))
        logger.info(f"  Val F1: {val_f1:.4f}")
        logger.info(f"  Training time: {training_time:.1f}s")

        # Create artifact
        artifact = TrainedModelArtifact(
            model_name=model_name,
            horizon=horizon,
            data_rank=data_rank,
            adapter_type=adapter_type,
            feature_names=prepared.feature_names or feature_names,
            n_train_samples=len(prepared.y_train),
            n_val_samples=len(prepared.y_val),
            metrics=metrics,
            training_time_seconds=training_time,
            model_path=model_output_dir,
            trainer=trainer,
            hyperparams=(
                trainer_config.to_dict()
                if hasattr(trainer_config, "to_dict")
                else dict(trainer_config.__dict__)
            ),
        )

        # Store artifact
        key = f"{model_name}_h{horizon}"
        self._artifacts[key] = artifact

        logger.info(artifact.summary())

        return artifact

    def train_all(
        self,
        df: pd.DataFrame,
        additional_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, TrainedModelArtifact]:
        """
        Train all configured models.

        Iterates through all horizons and models specified in PipelineConfig,
        training each combination and collecting artifacts.

        Args:
            df: Raw OHLCV DataFrame with features and labels
            additional_dfs: Optional additional timeframe data for multi-stream models

        Returns:
            Dict mapping "{model_name}_h{horizon}" to TrainedModelArtifact
        """
        artifacts = {}

        total_models = len(self.config.horizons) * len(self.config.models)
        current = 0

        for horizon in self.config.horizons:
            logger.info(f"\n{'='*50}")
            logger.info(f"HORIZON: {horizon}")
            logger.info(f"{'='*50}")

            for model_name in self.config.models:
                current += 1
                logger.info(f"\n[{current}/{total_models}] Training {model_name}...")

                try:
                    artifact = self.train_model(
                        model_name=model_name,
                        df=df,
                        horizon=horizon,
                        additional_dfs=additional_dfs,
                    )
                    key = f"{model_name}_h{horizon}"
                    artifacts[key] = artifact
                except Exception as e:
                    logger.error(f"Failed to train {model_name} for horizon {horizon}: {e}")
                    if self.config.verbose > 0:
                        import traceback
                        traceback.print_exc()
                    # Continue with other models

        # Save summary
        self.save_artifacts_summary()

        return artifacts

    def _run_hyperparam_optimization(
        self,
        trainer_config,
        prepared: PreparedData,
    ):
        """
        Run Optuna hyperparameter optimization.

        Uses TimeSeriesOptunaTuner from cross_validation module to find
        optimal hyperparameters for the model.

        Args:
            trainer_config: Initial TrainerConfig to optimize
            prepared: PreparedData with training data

        Returns:
            Updated TrainerConfig with optimized parameters
        """
        from src.validation.cv import TimeSeriesOptunaTuner

        logger.info("  Running hyperparameter optimization...")

        tuner = TimeSeriesOptunaTuner(
            model_name=trainer_config.model_name,
            horizon=trainer_config.horizon,
            n_splits=self.config.n_splits,
        )

        # Flatten for Optuna (works with 2D)
        X_train_2d = prepared.X_train
        if X_train_2d.ndim > 2:
            X_train_2d = X_train_2d.reshape(X_train_2d.shape[0], -1)

        best_params = tuner.optimize(
            X=X_train_2d,
            y=prepared.y_train,
            n_trials=self.config.hyperparam_trials,
        )

        logger.info(f"    Best params: {best_params}")

        # Update trainer config with best params
        for param, value in best_params.items():
            if hasattr(trainer_config, param):
                setattr(trainer_config, param, value)
            elif hasattr(trainer_config, "model_config"):
                # Some params go into model_config dict
                trainer_config.model_config[param] = value

        return trainer_config

    def get_artifact(
        self, model_name: str, horizon: int
    ) -> Optional[TrainedModelArtifact]:
        """
        Get trained artifact by model name and horizon.

        Args:
            model_name: Name of the model
            horizon: Prediction horizon

        Returns:
            TrainedModelArtifact if found, None otherwise
        """
        key = f"{model_name}_h{horizon}"
        return self._artifacts.get(key)

    @property
    def artifacts(self) -> Dict[str, TrainedModelArtifact]:
        """Get all trained artifacts (copy)."""
        return self._artifacts.copy()

    def save_artifacts_summary(self, path: Optional[Path] = None) -> Path:
        """
        Save summary of all trained artifacts.

        Saves a JSON file with metadata for all trained models,
        including metrics, training times, and configuration.

        Args:
            path: Optional path. If None, saves to output_dir/training_summary.json

        Returns:
            Path to saved summary file
        """
        path = path or self.output_dir / "training_summary.json"

        summary = {
            "config": {
                "symbol": self.config.symbol,
                "models": self.config.models,
                "horizons": self.config.horizons,
                "optimize_hyperparams": self.config.optimize_hyperparams,
            },
            "artifacts": {},
        }

        for key, artifact in self._artifacts.items():
            summary["artifacts"][key] = {
                "model_name": artifact.model_name,
                "horizon": artifact.horizon,
                "data_rank": artifact.data_rank,
                "adapter_type": artifact.adapter_type,
                "n_features": len(artifact.feature_names),
                "n_train_samples": artifact.n_train_samples,
                "n_val_samples": artifact.n_val_samples,
                "metrics": artifact.metrics,
                "training_time_seconds": artifact.training_time_seconds,
                "model_path": str(artifact.model_path) if artifact.model_path else None,
            }

        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Training summary saved: {path}")
        return path

    def load_artifacts_summary(self, path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load artifacts summary from JSON file.

        Args:
            path: Optional path. If None, loads from output_dir/training_summary.json

        Returns:
            Dictionary with summary data
        """
        path = path or self.output_dir / "training_summary.json"

        with open(path, "r") as f:
            return json.load(f)


def train_models(
    config: PipelineConfig,
    df: pd.DataFrame,
    additional_dfs: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, TrainedModelArtifact]:
    """
    Convenience function to train all models.

    Creates a ModelTrainer and trains all models specified in the config.

    Usage:
        from src.core import PipelineConfig
        from src.models.training import train_models

        config = PipelineConfig(...)
        artifacts = train_models(config, df)

    Args:
        config: PipelineConfig with training settings
        df: Raw OHLCV DataFrame with features and labels
        additional_dfs: Optional additional timeframe data

    Returns:
        Dict mapping "{model_name}_h{horizon}" to TrainedModelArtifact
    """
    trainer = ModelTrainer(config)
    return trainer.train_all(df, additional_dfs)


__all__ = [
    "ModelTrainer",
    "TrainedModelArtifact",
    "train_models",
]
