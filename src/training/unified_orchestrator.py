"""
UnifiedTrainingOrchestrator - THE single entry point for all training.

Uses PipelineConfig from src/core as the ONLY configuration source.
Integrates with PHASE_2 adapters for data preparation.

PHASE_3: Training Orchestration

This module provides the unified training orchestrator that:
1. Uses PipelineConfig as the ONLY configuration source
2. Integrates with UnifiedDataPreparation from PHASE_2 adapters
3. Routes to appropriate training mode (standard, walk_forward, regime_aware, meta_labeling)
4. Generates OOF predictions and aligns them using OOFAligner
5. Builds heterogeneous ensembles with proper OOF alignment
6. Returns structured TrainingRunResult

Example:
    from src.core import PipelineConfig
    from src.training import UnifiedTrainingOrchestrator

    config = PipelineConfig(
        symbol="MES",
        data_path="./data/mes.parquet",
        output_dir="./experiments/exp_001",
        models=["xgboost", "lightgbm", "lstm"],
        build_ensemble=True,
    )

    orchestrator = UnifiedTrainingOrchestrator(config)
    result = orchestrator.train(df)  # Pass raw OHLCV DataFrame
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.adapters import (
    AlignedOOFResult,
    OOFAligner,
    PreparedData,
    UnifiedDataPreparation,
)
from src.core import (
    CVMethod,
    OOFResult,
    PipelineConfig,
    TrainingMode,
)
from src.cross_validation import (
    OOFGenerator,
    OOFPrediction,
    PurgedKFold,
    PurgedKFoldConfig,
    StackingDataset,
    TimeSeriesOptunaTuner,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT DATACLASSES
# =============================================================================


@dataclass
class ModelTrainingResult:
    """
    Result from training a single model.

    Attributes:
        model_name: Name of the model (e.g., "xgboost", "lstm")
        horizon: Prediction horizon in bars
        metrics: Validation metrics dict (val_f1, val_accuracy, etc.)
        oof_prediction: Optional OOF predictions from CV
        trainer: Optional Trainer instance (for inference)
        training_time_seconds: Time taken to train
        n_features: Number of features used
        data_rank: Data dimensionality (2, 3, or 4)
    """

    model_name: str
    horizon: int
    metrics: dict[str, float] = field(default_factory=dict)
    oof_prediction: OOFPrediction | None = None
    trainer: Any | None = None
    training_time_seconds: float = 0.0
    n_features: int = 0
    data_rank: int = 2

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "metrics": self.metrics,
            "training_time_seconds": self.training_time_seconds,
            "n_features": self.n_features,
            "data_rank": self.data_rank,
        }


@dataclass
class TrainingRunResult:
    """
    Result from a complete training run.

    Attributes:
        run_id: Unique identifier for this run
        config: PipelineConfig used for this run
        model_results: Dict mapping model key -> ModelTrainingResult
        ensemble_result: Optional ensemble training result
        stacking_dataset: Optional stacking dataset for meta-learner
        aligned_oof: Optional AlignedOOFResult for heterogeneous ensembles
        total_time_seconds: Total wall-clock time
        output_dir: Directory where results are saved
    """

    run_id: str
    config: PipelineConfig
    model_results: dict[str, ModelTrainingResult] = field(default_factory=dict)
    ensemble_result: ModelTrainingResult | None = None
    stacking_dataset: StackingDataset | None = None
    aligned_oof: AlignedOOFResult | None = None
    total_time_seconds: float = 0.0
    output_dir: Path = field(default_factory=lambda: Path("."))

    @property
    def n_models(self) -> int:
        """Number of trained models."""
        return len(self.model_results)

    @property
    def best_model(self) -> str | None:
        """Model with best validation F1 score."""
        if not self.model_results:
            return None
        return max(
            self.model_results.keys(),
            key=lambda k: self.model_results[k].metrics.get("val_f1", 0)
        )

    def get_metrics_summary(self) -> dict[str, dict[str, float]]:
        """Get summary of all model metrics."""
        return {
            key: result.metrics
            for key, result in self.model_results.items()
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "n_models": self.n_models,
            "best_model": self.best_model,
            "total_time_seconds": self.total_time_seconds,
            "output_dir": str(self.output_dir),
            "model_results": {
                key: result.to_dict()
                for key, result in self.model_results.items()
            },
        }


# =============================================================================
# UNIFIED TRAINING ORCHESTRATOR
# =============================================================================


class UnifiedTrainingOrchestrator:
    """
    THE single entry point for all training in the ML Factory.

    Uses PipelineConfig from src/core as the ONLY configuration source.

    Supports:
    - All training modes: standard, walk_forward, regime_aware, meta_labeling
    - All CV methods: purged_kfold, cpcv, pbo, walk_forward
    - Heterogeneous ensembles with OOF alignment
    - Integration with PHASE_2 adapters

    Usage:
        from src.core import PipelineConfig
        from src.training import UnifiedTrainingOrchestrator

        config = PipelineConfig(
            symbol="MES",
            data_path="./data/mes.parquet",
            output_dir="./experiments/exp_001",
            models=["xgboost", "lightgbm", "lstm"],
            build_ensemble=True,
        )

        orchestrator = UnifiedTrainingOrchestrator(config)
        result = orchestrator.train(df)  # Pass raw OHLCV DataFrame
    """

    def __init__(self, config: PipelineConfig) -> None:
        """
        Initialize with PipelineConfig.

        Args:
            config: PipelineConfig from src/core - THE ONLY config source
        """
        self.config = config
        self.run_id = self._generate_run_id()
        self.output_dir = config.output_dir / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data preparation (PHASE_2)
        self._data_prep = UnifiedDataPreparation(config)

        # Initialize CV based on config
        self._cv = self._create_cv()

        # Initialize OOF generator
        self._oof_generator = OOFGenerator(
            self._cv,
            cache_dir=self.output_dir / "oof_cache" if config.save_oof else None,
        )

        # Results storage
        self._model_results: dict[str, ModelTrainingResult] = {}
        self._oof_predictions: dict[str, OOFPrediction] = {}
        self._trained_models: dict[str, Any] = {}

        logger.info("Initialized UnifiedTrainingOrchestrator")
        logger.info(f"  Run ID: {self.run_id}")
        logger.info(f"  Mode: {config.training_mode}")
        logger.info(f"  CV: {config.cv_method}")
        logger.info(f"  Models: {config.models}")
        logger.info(f"  Output: {self.output_dir}")

    def _generate_run_id(self) -> str:
        """Generate unique run identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = self.config.training_mode
        return f"run_{mode}_{timestamp}"

    def _create_cv(self) -> PurgedKFold:
        """
        Create cross-validator based on config.

        Returns:
            PurgedKFold instance configured per PipelineConfig
        """
        cv_method = CVMethod(self.config.cv_method)

        # Base config for all CV methods
        cv_config = PurgedKFoldConfig(
            n_splits=self.config.n_splits,
            purge_bars=self.config.purge_bars,
            embargo_bars=self.config.embargo_bars,
        )

        # For now, use PurgedKFold for all methods
        # CPCV and WalkForward have specialized implementations
        if cv_method == CVMethod.PURGED_KFOLD:
            return PurgedKFold(cv_config)
        elif cv_method == CVMethod.CPCV:
            # CPCV wraps PurgedKFold with combinatorial paths
            return PurgedKFold(cv_config)
        elif cv_method == CVMethod.WALK_FORWARD:
            # Walk-forward uses different evaluator
            return PurgedKFold(cv_config)
        elif cv_method == CVMethod.PBO:
            # PBO is post-hoc validation, use purged k-fold for training
            return PurgedKFold(cv_config)
        else:
            logger.warning(f"Unknown CV method: {cv_method}, using PurgedKFold")
            return PurgedKFold(cv_config)

    def train(
        self,
        df: pd.DataFrame,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> TrainingRunResult:
        """
        Execute complete training pipeline.

        This is THE unified interface for all training.

        Args:
            df: Raw OHLCV DataFrame (features will be computed via adapters)
            additional_dfs: Optional dict of additional timeframe DataFrames
                for multi-stream models (e.g., {"5min": df_5min})

        Returns:
            TrainingRunResult with all training outputs
        """
        start_time = time.time()

        logger.info("=" * 60)
        logger.info(f"UNIFIED TRAINING: {self.config.training_mode.upper()}")
        logger.info("=" * 60)
        logger.info(f"Symbol: {self.config.symbol}")
        logger.info(f"Models: {self.config.models}")
        logger.info(f"Horizons: {self.config.horizons}")

        # Route to appropriate training mode
        mode = TrainingMode(self.config.training_mode)

        if mode == TrainingMode.STANDARD:
            self._train_standard(df, additional_dfs)
        elif mode == TrainingMode.WALK_FORWARD:
            self._train_walk_forward(df, additional_dfs)
        elif mode == TrainingMode.REGIME_AWARE:
            self._train_regime_aware(df, additional_dfs)
        elif mode == TrainingMode.META_LABELING:
            self._train_meta_labeling(df, additional_dfs)
        else:
            raise ValueError(f"Unknown training mode: {mode}")

        # Build ensemble if requested
        ensemble_result = None
        stacking_dataset = None
        aligned_oof = None

        if self.config.build_ensemble and len(self._model_results) > 1:
            logger.info("\n" + "=" * 60)
            logger.info("BUILDING ENSEMBLE")
            logger.info("=" * 60)
            aligned_oof, stacking_dataset, ensemble_result = self._build_ensemble(df)

        # Save results
        self._save_results()

        total_time = time.time() - start_time

        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Models trained: {len(self._model_results)}")
        logger.info(f"Output: {self.output_dir}")

        return TrainingRunResult(
            run_id=self.run_id,
            config=self.config,
            model_results=self._model_results,
            ensemble_result=ensemble_result,
            stacking_dataset=stacking_dataset,
            aligned_oof=aligned_oof,
            total_time_seconds=total_time,
            output_dir=self.output_dir,
        )

    def _train_standard(
        self,
        df: pd.DataFrame,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """
        Standard training with PurgedKFold CV and OOF generation.

        This is the default training mode that:
        1. Iterates through each horizon and model
        2. Prepares data using PHASE_2 adapters
        3. Optionally optimizes hyperparameters
        4. Trains model and generates OOF predictions

        Args:
            df: Input DataFrame with features and labels
            additional_dfs: Optional additional timeframe DataFrames
        """
        for horizon in self.config.horizons:
            logger.info(f"\n--- Horizon {horizon} ---")

            for model_name in self.config.models:
                logger.info(f"\nTraining {model_name}...")

                # Prepare data using PHASE_2 adapters
                prepared = self._data_prep.prepare(
                    df=df,
                    model_name=model_name,
                    additional_dfs=additional_dfs,
                )

                logger.info(f"  Data prepared: {prepared.summary()}")

                # Train model
                result = self._train_single_model(
                    model_name=model_name,
                    prepared=prepared,
                    horizon=horizon,
                )

                key = f"{model_name}_h{horizon}"
                self._model_results[key] = result

                # Generate OOF if enabled
                if self.config.save_oof:
                    oof = self._generate_oof(model_name, prepared, horizon)
                    if oof is not None:
                        self._oof_predictions[key] = oof
                        result.oof_prediction = oof

                logger.info(
                    f"  {model_name} complete: "
                    f"val_f1={result.metrics.get('val_f1', 0):.4f}, "
                    f"time={result.training_time_seconds:.1f}s"
                )

    def _train_single_model(
        self,
        model_name: str,
        prepared: PreparedData,
        horizon: int,
    ) -> ModelTrainingResult:
        """
        Train a single model with prepared data.

        Args:
            model_name: Name of the model to train
            prepared: PreparedData from PHASE_2 adapter
            horizon: Prediction horizon

        Returns:
            ModelTrainingResult with metrics and trained model
        """
        from src.models import Trainer, TrainerConfig

        start = time.time()

        # Create trainer config
        trainer_config = TrainerConfig(
            model_name=model_name,
            horizon=horizon,
            sequence_length=self.config.sequence_length,
            output_dir=self.output_dir / f"h{horizon}",
        )

        # Hyperparameter optimization if enabled
        if self.config.optimize_hyperparams:
            trainer_config = self._optimize_hyperparams(
                trainer_config, prepared
            )

        # Create trainer and run
        trainer = Trainer(trainer_config)

        # Build container from prepared data
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
            X_test_df = pd.DataFrame(
                prepared.X_test,
                columns=prepared.feature_names,
            ) if prepared.has_test else None
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
            if prepared.has_test:
                n_test = prepared.X_test.shape[0]
                X_test_df = pd.DataFrame(
                    prepared.X_test.reshape(n_test, -1),
                    columns=[f"f{i}" for i in range(n_features)],
                )
            else:
                X_test_df = None

        container = TimeSeriesDataContainer(
            X_train=X_train_df,
            y_train=pd.Series(prepared.y_train),
            X_val=X_val_df,
            y_val=pd.Series(prepared.y_val),
            X_test=X_test_df if X_test_df is not None else pd.DataFrame(),
            y_test=pd.Series(prepared.y_test) if prepared.has_test else pd.Series(dtype=float),
            sample_weights=pd.Series(
                prepared.train_weights if prepared.has_weights
                else np.ones(len(prepared.y_train))
            ),
        )

        training_results = trainer.run(container)

        training_time = time.time() - start

        # Store trained model for inference
        self._trained_models[f"{model_name}_h{horizon}"] = trainer

        return ModelTrainingResult(
            model_name=model_name,
            horizon=horizon,
            metrics=training_results.get("evaluation_metrics", {}),
            trainer=trainer,
            training_time_seconds=training_time,
            n_features=prepared.n_features,
            data_rank=prepared.data_rank,
        )

    def _generate_oof(
        self,
        model_name: str,
        prepared: PreparedData,
        horizon: int,
    ) -> OOFPrediction | None:
        """
        Generate OOF predictions for a model.

        Args:
            model_name: Name of the model
            prepared: PreparedData from PHASE_2
            horizon: Prediction horizon

        Returns:
            OOFPrediction or None if generation fails
        """
        try:
            # Flatten to 2D for OOF generation
            if prepared.data_rank > 2:
                X_train_2d = prepared.X_train.reshape(prepared.X_train.shape[0], -1)
            else:
                X_train_2d = prepared.X_train

            X_train_df = pd.DataFrame(
                X_train_2d,
                columns=[f"f{i}" for i in range(X_train_2d.shape[1])],
            )
            y_train = pd.Series(prepared.y_train)

            oof_predictions = self._oof_generator.generate_oof_predictions(
                X=X_train_df,
                y=y_train,
                model_configs={model_name: {}},
                use_cache=True,
            )

            return oof_predictions.get(model_name)

        except Exception as e:
            logger.warning(f"Failed to generate OOF for {model_name}: {e}")
            return None

    def _build_ensemble(
        self,
        df: pd.DataFrame,
    ) -> tuple[AlignedOOFResult | None, StackingDataset | None, ModelTrainingResult | None]:
        """
        Build ensemble from OOF predictions.

        Aligns OOF predictions from heterogeneous models (2D/3D/4D) and
        trains a meta-learner on the stacking dataset.

        Args:
            df: Original DataFrame (for label extraction)

        Returns:
            Tuple of (aligned_oof, stacking_dataset, ensemble_result)
        """
        if not self._oof_predictions:
            logger.warning("No OOF predictions available for ensemble")
            return None, None, None

        logger.info(f"Building ensemble from {len(self._oof_predictions)} models...")

        # Convert OOFPrediction dict to OOFResult list for alignment
        # OOFAligner expects OOFResult format
        oof_results: list[OOFResult] = []

        for _key, oof_pred in self._oof_predictions.items():
            # Extract probabilities and predictions
            probs = oof_pred.get_probabilities()
            preds = oof_pred.get_class_predictions()

            # Create indices array
            n_samples = len(probs)
            indices = np.arange(n_samples)

            # Create fold_ids from fold_info
            fold_ids = np.zeros(n_samples, dtype=int)
            # Simple assignment - all samples from same prediction run

            oof_result = OOFResult(
                predictions=preds.astype(int),
                probabilities=probs,
                indices=indices,
                fold_ids=fold_ids,
                model_name=oof_pred.model_name,
                coverage=oof_pred.coverage,
            )
            oof_results.append(oof_result)

        if len(oof_results) < 2:
            logger.warning("Need at least 2 models for ensemble")
            return None, None, None

        # Align OOF predictions
        aligner = OOFAligner()
        try:
            aligned = aligner.align(oof_results, strategy="intersection")
        except ValueError as e:
            logger.error(f"Failed to align OOF predictions: {e}")
            return None, None, None

        logger.info(f"Aligned {len(aligned.model_names)} models")
        logger.info(f"Valid samples: {aligned.n_common}")

        # Get y_true for aligned samples
        # This is a simplification - in practice, would extract from prepared data
        y_aligned = None
        for _key, oof_pred in self._oof_predictions.items():
            if "y_true" in oof_pred.predictions.columns:
                y_aligned = oof_pred.predictions["y_true"].values[:aligned.n_common]
                break

        if y_aligned is None:
            logger.warning("Could not extract aligned labels for meta-learner")
            return aligned, None, None

        # Build stacking dataset
        stacking_features = aligned.stacking_features

        stacking_df = pd.DataFrame(
            stacking_features,
            columns=aligned.get_feature_names(),
        )
        stacking_df["y_true"] = y_aligned

        stacking_dataset = StackingDataset(
            data=stacking_df,
            model_names=aligned.model_names,
            horizon=self.config.horizons[0],
            metadata={
                "n_common": aligned.n_common,
                "coverage": aligned.coverage,
            },
        )

        logger.info(f"Stacking dataset: {stacking_dataset.n_samples} samples")

        # Train meta-learner
        ensemble_result = self._train_meta_learner(stacking_dataset)

        return aligned, stacking_dataset, ensemble_result

    def _train_meta_learner(
        self,
        stacking_dataset: StackingDataset,
    ) -> ModelTrainingResult | None:
        """
        Train meta-learner on stacking dataset.

        Args:
            stacking_dataset: StackingDataset with aligned OOF features

        Returns:
            ModelTrainingResult for meta-learner or None
        """
        try:
            from src.models import Trainer, TrainerConfig

            start = time.time()

            # Get stacking features and labels
            X_stack = stacking_dataset.get_features()
            y_stack = stacking_dataset.get_labels()

            # Split into train/val
            n_samples = len(X_stack)
            n_train = int(n_samples * 0.8)

            X_train = X_stack.iloc[:n_train]
            X_val = X_stack.iloc[n_train:]
            y_train = y_stack.iloc[:n_train]
            y_val = y_stack.iloc[n_train:]

            # Create meta-learner config
            meta_config = TrainerConfig(
                model_name=self.config.meta_learner,
                horizon=stacking_dataset.horizon,
                output_dir=self.output_dir / "meta_learner",
            )

            from src.core.container import TimeSeriesDataContainer

            container = TimeSeriesDataContainer(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=pd.DataFrame(),
                y_test=pd.Series(dtype=float),
                sample_weights=pd.Series(np.ones(len(y_train))),
            )

            trainer = Trainer(meta_config)
            results = trainer.run(container)

            training_time = time.time() - start

            logger.info(
                f"Meta-learner ({self.config.meta_learner}) trained: "
                f"val_f1={results['evaluation_metrics'].get('val_f1', 0):.4f}"
            )

            return ModelTrainingResult(
                model_name=f"ensemble_{self.config.meta_learner}",
                horizon=stacking_dataset.horizon,
                metrics=results.get("evaluation_metrics", {}),
                trainer=trainer,
                training_time_seconds=training_time,
                n_features=X_stack.shape[1],
                data_rank=2,
            )

        except Exception as e:
            logger.error(f"Failed to train meta-learner: {e}")
            return None

    def _train_walk_forward(
        self,
        df: pd.DataFrame,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """
        Walk-forward training mode.

        Uses expanding or rolling windows for more realistic backtesting.

        Args:
            df: Input DataFrame
            additional_dfs: Optional additional timeframe DataFrames
        """
        from src.training.config import ExperimentConfig, ModelConfig
        from src.training.modes import WalkForwardTrainer, WalkForwardTrainerConfig

        logger.info("Walk-forward training mode")

        # Create ExperimentConfig from PipelineConfig
        model_configs = [
            ModelConfig(name=m) for m in self.config.models
        ]

        exp_config = ExperimentConfig(
            symbol=self.config.symbol,
            horizons=self.config.horizons,
            models=model_configs,
            data_dir=self.config.data_path.parent,
            output_dir=self.output_dir,
        )

        wf_config = WalkForwardTrainerConfig(
            n_windows=self.config.n_splits,
            window_type="expanding",
            min_train_pct=self.config.train_ratio,
            test_pct=self.config.val_ratio,
            gap_bars=self.config.purge_bars,
            embargo_bars=self.config.embargo_bars,
        )

        trainer = WalkForwardTrainer(exp_config, wf_config)

        # Prepare container from df
        from src.core.container import TimeSeriesDataContainer

        # Use first model to prepare data (for basic structure)
        prepared = self._data_prep.prepare(
            df=df,
            model_name=self.config.models[0],
            additional_dfs=additional_dfs,
        )

        X_train_df = pd.DataFrame(
            prepared.X_train.reshape(prepared.X_train.shape[0], -1) if prepared.data_rank > 2 else prepared.X_train,
            columns=prepared.feature_names if prepared.data_rank == 2 else [f"f{i}" for i in range(np.prod(prepared.X_train.shape[1:]))],
        )

        container = TimeSeriesDataContainer(
            X_train=X_train_df,
            y_train=pd.Series(prepared.y_train),
            X_val=pd.DataFrame(),
            y_val=pd.Series(dtype=float),
            X_test=pd.DataFrame(),
            y_test=pd.Series(dtype=float),
            sample_weights=pd.Series(np.ones(len(prepared.y_train))),
        )

        results = trainer.run(container)

        # Convert walk-forward results to our format
        for model_name, wf_result in results.get("model_results", {}).items():
            key = f"{model_name}_h{self.config.horizons[0]}"
            self._model_results[key] = ModelTrainingResult(
                model_name=model_name,
                horizon=self.config.horizons[0],
                metrics={
                    "val_f1": wf_result.aggregated_metrics.get("mean_f1", 0),
                    "val_accuracy": wf_result.aggregated_metrics.get("mean_accuracy", 0),
                },
                training_time_seconds=wf_result.total_time,
                n_features=prepared.n_features,
                data_rank=prepared.data_rank,
            )

    def _train_regime_aware(
        self,
        df: pd.DataFrame,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """
        Regime-aware training mode.

        Trains separate models for different market regimes using the new
        RegimeAwareTrainer that integrates with PipelineConfig.

        Uses PipelineConfig settings:
        - regime_detection_method: Detection method (volatility_percentile, trend_adx, combined)
        - regime_lookback: Lookback period for regime detection
        - n_regimes: Number of regime states (2 or 3)
        - regime_volatility_window: Rolling window for volatility
        - regime_adx_threshold: ADX threshold for trending
        - regime_min_samples: Minimum samples per regime
        - train_separate_regime_models: Whether to train separate models per regime

        Args:
            df: Input DataFrame with OHLCV data
            additional_dfs: Optional additional timeframe DataFrames
        """
        from src.training.regime_trainer import RegimeAwareTrainer, RegimeTrainingResult

        logger.info("Regime-aware training mode")
        logger.info(f"  Detection method: {self.config.regime_detection_method}")
        logger.info(f"  Number of regimes: {self.config.n_regimes}")
        logger.info(f"  Separate models: {self.config.train_separate_regime_models}")

        # Initialize the regime-aware trainer with PipelineConfig
        regime_trainer = RegimeAwareTrainer(self.config)

        # Store regime trainer for inference
        self._regime_trainer = regime_trainer

        for horizon in self.config.horizons:
            logger.info(f"\n--- Horizon {horizon} ---")

            for model_name in self.config.models:
                logger.info(f"\nTraining regime-aware: {model_name}...")

                # Prepare data using PHASE_2 adapters
                prepared = self._data_prep.prepare(
                    df=df,
                    model_name=model_name,
                    additional_dfs=additional_dfs,
                )

                logger.info(f"  Data prepared: {prepared.summary()}")

                # Train regime-aware model
                regime_result = regime_trainer.train(
                    prepared=prepared,
                    horizon=horizon,
                    model_name=model_name,
                    save_models=self.config.save_models,
                )

                # Convert regime results to ModelTrainingResult format
                for (trained_model_name, regime), regime_model_result in regime_result.regime_results.items():
                    if trained_model_name != model_name:
                        continue

                    # Create key that includes regime
                    key = f"{model_name}_h{horizon}_{regime}"

                    self._model_results[key] = ModelTrainingResult(
                        model_name=f"{model_name}_{regime}",
                        horizon=horizon,
                        metrics={
                            "val_f1": regime_model_result.val_f1,
                            "val_accuracy": regime_model_result.val_accuracy,
                            **regime_model_result.metrics,
                        },
                        trainer=regime_model_result.trainer,
                        training_time_seconds=regime_model_result.training_time_seconds,
                        n_features=prepared.n_features,
                        data_rank=prepared.data_rank,
                    )

                    # Store trained model for inference
                    self._trained_models[key] = regime_model_result.trainer

                    logger.info(
                        f"    {model_name}/{regime}: "
                        f"val_f1={regime_model_result.val_f1:.4f}, "
                        f"samples={regime_model_result.n_samples}"
                    )

                # Also create an aggregated result for this model
                aggregated_key = f"{model_name}_h{horizon}_aggregated"
                aggregated_metrics = {
                    k.replace(f"{model_name}_", ""): v
                    for k, v in regime_result.aggregated_metrics.items()
                    if model_name in k or "overall" in k
                }

                self._model_results[aggregated_key] = ModelTrainingResult(
                    model_name=f"{model_name}_regime_aware",
                    horizon=horizon,
                    metrics=aggregated_metrics,
                    training_time_seconds=regime_result.total_time_seconds,
                    n_features=prepared.n_features,
                    data_rank=prepared.data_rank,
                )

        logger.info("\nRegime-aware training complete")

    def _train_meta_labeling(
        self,
        df: pd.DataFrame,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """
        Meta-labeling training mode (Lopez de Prado 2018).

        Implements the complete meta-labeling methodology:
        1. Train primary model for direction prediction (+1, -1)
        2. Generate primary model predictions on training data
        3. Create meta-labels: 1 if primary was correct, 0 if wrong
        4. Train meta-model to predict probability of primary being correct
        5. At inference: final_position = primary_direction * meta_probability

        Uses PipelineConfig settings:
        - meta_labeling_primary_model: Model for direction (default: "xgboost")
        - meta_labeling_meta_model: Model for bet sizing (default: "logistic")
        - meta_labeling_threshold: Min probability to take trade (default: 0.5)

        Args:
            df: Input DataFrame
            additional_dfs: Optional additional timeframe DataFrames
        """
        logger.info("=" * 60)
        logger.info("META-LABELING TRAINING (Lopez de Prado 2018)")
        logger.info("=" * 60)
        logger.info(f"Primary model: {self.config.meta_labeling_primary_model}")
        logger.info(f"Meta model: {self.config.meta_labeling_meta_model}")
        logger.info(f"Bet threshold: {self.config.meta_labeling_threshold}")

        start_time = time.time()

        for horizon in self.config.horizons:
            logger.info(f"\n--- Horizon {horizon} ---")

            result = self._train_meta_labeling_for_horizon(
                df=df,
                horizon=horizon,
                additional_dfs=additional_dfs,
            )

            key = f"meta_labeling_h{horizon}"
            self._model_results[key] = result

            logger.info(
                f"  Meta-labeling complete: "
                f"primary_acc={result.metrics.get('primary_accuracy', 0):.4f}, "
                f"combined_acc={result.metrics.get('combined_accuracy', 0):.4f}, "
                f"trade_fraction={result.metrics.get('trade_fraction', 0)*100:.1f}%"
            )

        total_time = time.time() - start_time
        logger.info(f"\nMeta-labeling total time: {total_time:.1f}s")

    def _train_meta_labeling_for_horizon(
        self,
        df: pd.DataFrame,
        horizon: int,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> ModelTrainingResult:
        """
        Train meta-labeling system for a single horizon.

        Args:
            df: Input DataFrame
            horizon: Prediction horizon
            additional_dfs: Optional additional DataFrames

        Returns:
            ModelTrainingResult with combined metrics
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier

        start_time = time.time()

        # Get model names from config
        primary_model_name = self.config.meta_labeling_primary_model
        meta_model_name = self.config.meta_labeling_meta_model

        # =====================================================================
        # STAGE 1: PREPARE DATA
        # =====================================================================
        logger.info("\n  STAGE 1: Preparing data...")

        # Prepare data for primary model
        prepared = self._data_prep.prepare(
            df=df,
            model_name=primary_model_name,
            additional_dfs=additional_dfs,
        )

        logger.info(f"    Data: {prepared.n_train} train, {prepared.n_val} val samples")
        logger.info(f"    Features: {prepared.n_features}, Rank: {prepared.data_rank}")

        # Convert to 2D for training
        X_train = self._flatten_to_2d(prepared.X_train)
        X_val = self._flatten_to_2d(prepared.X_val)
        y_train = prepared.y_train
        y_val = prepared.y_val

        feature_names = (
            prepared.feature_names
            if prepared.data_rank == 2
            else [f"f{i}" for i in range(X_train.shape[1])]
        )

        # =====================================================================
        # STAGE 2: TRAIN PRIMARY MODEL (Direction)
        # =====================================================================
        logger.info("\n  STAGE 2: Training primary model (direction)...")

        primary_result = self._train_single_model(
            model_name=primary_model_name,
            prepared=prepared,
            horizon=horizon,
        )

        primary_trainer = primary_result.trainer
        if primary_trainer is None:
            raise RuntimeError("Primary model training failed")

        # Get primary predictions on training and validation sets
        primary_train_preds = primary_trainer.model.predict(X_train)
        primary_val_preds = primary_trainer.model.predict(X_val)

        # Extract class predictions
        primary_train_classes = primary_train_preds.class_predictions
        primary_val_classes = primary_val_preds.class_predictions

        # Primary model accuracy
        primary_train_acc = (primary_train_classes == y_train).mean()
        primary_val_acc = (primary_val_classes == y_val).mean()

        logger.info(f"    Primary train accuracy: {primary_train_acc:.4f}")
        logger.info(f"    Primary val accuracy: {primary_val_acc:.4f}")

        # =====================================================================
        # STAGE 3: CREATE META-LABELS
        # =====================================================================
        logger.info("\n  STAGE 3: Creating meta-labels...")

        # Meta-label = 1 if primary prediction was correct, 0 if wrong
        meta_labels_train = (primary_train_classes == y_train).astype(int)
        meta_labels_val = (primary_val_classes == y_val).astype(int)

        # Log meta-label distribution
        train_correct_pct = meta_labels_train.mean() * 100
        val_correct_pct = meta_labels_val.mean() * 100

        logger.info(f"    Train: {train_correct_pct:.1f}% correct (meta=1)")
        logger.info(f"    Val: {val_correct_pct:.1f}% correct (meta=1)")

        # =====================================================================
        # STAGE 4: TRAIN META-MODEL (Bet Sizing)
        # =====================================================================
        logger.info("\n  STAGE 4: Training meta-model (bet sizing)...")

        # Create meta-model based on config
        meta_model = self._create_meta_model(meta_model_name)

        # Train meta-model to predict if primary is correct
        meta_model.fit(X_train, meta_labels_train)

        # Get meta-model probabilities
        if hasattr(meta_model, 'predict_proba'):
            meta_proba_train = meta_model.predict_proba(X_train)[:, 1]
            meta_proba_val = meta_model.predict_proba(X_val)[:, 1]
        else:
            # Fall back to decision function if no predict_proba
            meta_proba_train = meta_model.predict(X_train).astype(float)
            meta_proba_val = meta_model.predict(X_val).astype(float)

        # Meta-model accuracy (thresholded at 0.5)
        meta_train_acc = ((meta_proba_train >= 0.5) == meta_labels_train).mean()
        meta_val_acc = ((meta_proba_val >= 0.5) == meta_labels_val).mean()

        logger.info(f"    Meta-model train accuracy: {meta_train_acc:.4f}")
        logger.info(f"    Meta-model val accuracy: {meta_val_acc:.4f}")

        # =====================================================================
        # STAGE 5: EVALUATE COMBINED SYSTEM
        # =====================================================================
        logger.info("\n  STAGE 5: Evaluating combined system...")

        threshold = self.config.meta_labeling_threshold

        # Apply threshold - only take trades where meta probability >= threshold
        trades_taken_train = meta_proba_train >= threshold
        trades_taken_val = meta_proba_val >= threshold

        # Combined accuracy (only on trades taken)
        if trades_taken_val.sum() > 0:
            combined_val_acc = (
                (primary_val_classes[trades_taken_val] == y_val[trades_taken_val]).mean()
            )
            trade_fraction = trades_taken_val.mean()
        else:
            combined_val_acc = 0.0
            trade_fraction = 0.0

        # Improvement over primary alone
        improvement = combined_val_acc - primary_val_acc if trade_fraction > 0 else 0.0

        logger.info(f"    Threshold: {threshold}")
        logger.info(f"    Trade fraction: {trade_fraction*100:.1f}%")
        logger.info(f"    Primary-only accuracy: {primary_val_acc:.4f}")
        logger.info(f"    Combined accuracy: {combined_val_acc:.4f}")
        logger.info(f"    Improvement: {improvement:+.4f}")

        # =====================================================================
        # STAGE 6: STORE MODELS FOR INFERENCE
        # =====================================================================
        logger.info("\n  STAGE 6: Storing models...")

        # Store both models for inference
        model_key = f"meta_labeling_h{horizon}"
        self._trained_models[f"{model_key}_primary"] = primary_trainer
        self._trained_models[f"{model_key}_meta"] = meta_model

        # Save models if enabled
        if self.config.save_models:
            models_dir = self.output_dir / "models"
            models_dir.mkdir(exist_ok=True)

            # Save meta-model (sklearn)
            import pickle
            meta_path = models_dir / f"{model_key}_meta.pkl"
            with open(meta_path, 'wb') as f:
                pickle.dump({
                    'meta_model': meta_model,
                    'primary_model_name': primary_model_name,
                    'meta_model_name': meta_model_name,
                    'threshold': threshold,
                    'feature_names': feature_names,
                }, f)
            logger.info(f"    Saved meta-model to: {meta_path}")

        training_time = time.time() - start_time

        # Build combined metrics
        metrics = {
            # Primary model metrics
            "primary_train_accuracy": float(primary_train_acc),
            "primary_val_accuracy": float(primary_val_acc),
            "primary_val_f1": primary_result.metrics.get("val_f1", 0),

            # Meta model metrics
            "meta_train_accuracy": float(meta_train_acc),
            "meta_val_accuracy": float(meta_val_acc),

            # Combined system metrics
            "combined_accuracy": float(combined_val_acc),
            "primary_accuracy": float(primary_val_acc),  # For comparison
            "trade_fraction": float(trade_fraction),
            "trades_taken": int(trades_taken_val.sum()),
            "total_samples": len(y_val),
            "threshold": threshold,
            "improvement": float(improvement),

            # For backward compatibility with val_f1 key
            "val_f1": float(combined_val_acc),  # Use combined accuracy as main metric
            "val_accuracy": float(combined_val_acc),
        }

        return ModelTrainingResult(
            model_name=f"meta_labeling_{primary_model_name}_{meta_model_name}",
            horizon=horizon,
            metrics=metrics,
            trainer=primary_trainer,  # Store primary trainer as main
            training_time_seconds=training_time,
            n_features=prepared.n_features,
            data_rank=prepared.data_rank,
        )

    def _flatten_to_2d(self, X: np.ndarray) -> np.ndarray:
        """Flatten array to 2D if needed."""
        if X.ndim == 2:
            return X
        return X.reshape(X.shape[0], -1)

    def _create_meta_model(self, model_name: str) -> Any:
        """
        Create meta-model for bet sizing.

        Args:
            model_name: Name of the model ("logistic", "xgboost", etc.)

        Returns:
            Fitted sklearn-compatible model
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier

        if model_name == "logistic":
            return LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight='balanced',
                random_state=self.config.random_state,
            )
        elif model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                class_weight='balanced',
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            )
        elif model_name == "xgboost":
            try:
                import xgboost as xgb
                return xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs,
                    use_label_encoder=False,
                    eval_metric='logloss',
                )
            except ImportError:
                logger.warning("XGBoost not available, falling back to logistic")
                return self._create_meta_model("logistic")
        elif model_name == "lightgbm":
            try:
                import lightgbm as lgb
                return lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs,
                    verbose=-1,
                )
            except ImportError:
                logger.warning("LightGBM not available, falling back to logistic")
                return self._create_meta_model("logistic")
        elif model_name == "catboost":
            try:
                import catboost as cb
                return cb.CatBoostClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=self.config.random_state,
                    verbose=False,
                )
            except ImportError:
                logger.warning("CatBoost not available, falling back to logistic")
                return self._create_meta_model("logistic")
        else:
            logger.warning(f"Unknown meta model: {model_name}, using logistic")
            return self._create_meta_model("logistic")

    def _optimize_hyperparams(
        self,
        trainer_config: Any,
        prepared: PreparedData,
    ) -> Any:
        """
        Run Optuna hyperparameter optimization.

        Args:
            trainer_config: TrainerConfig to update with best params
            prepared: PreparedData with training data

        Returns:
            Updated trainer_config with optimized hyperparameters
        """
        logger.info("  Running hyperparameter optimization...")

        tuner = TimeSeriesOptunaTuner(
            model_name=trainer_config.model_name,
            horizon=trainer_config.horizon,
            n_splits=self.config.n_splits,
        )

        # Flatten to 2D for tuning
        X_train_2d = (
            prepared.X_train.reshape(prepared.X_train.shape[0], -1)
            if prepared.X_train.ndim > 2
            else prepared.X_train
        )

        best_params = tuner.optimize(
            X=X_train_2d,
            y=prepared.y_train,
            n_trials=self.config.hyperparam_trials,
        )

        logger.info(f"  Best params: {best_params}")

        for param, value in best_params.items():
            setattr(trainer_config, param, value)

        return trainer_config

    def _save_results(self) -> None:
        """Save all results to disk."""
        logger.info(f"\nSaving results to: {self.output_dir}")

        # Save config
        self.config.save(self.output_dir / "config.json")

        # Save metrics summary
        metrics_summary: dict[str, Any] = {}
        for key, result in self._model_results.items():
            metrics_summary[key] = {
                "model_name": result.model_name,
                "horizon": result.horizon,
                "metrics": result.metrics,
                "training_time": result.training_time_seconds,
                "n_features": result.n_features,
                "data_rank": result.data_rank,
            }

        with open(self.output_dir / "metrics_summary.json", "w") as f:
            json.dump(metrics_summary, f, indent=2)

        # Save OOF predictions if available
        if self._oof_predictions and self.config.save_oof:
            oof_dir = self.output_dir / "oof"
            oof_dir.mkdir(exist_ok=True)

            for key, oof in self._oof_predictions.items():
                oof.predictions.to_parquet(oof_dir / f"{key}_oof.parquet")

        # Save trained models if enabled
        if self.config.save_models:
            models_dir = self.output_dir / "models"
            models_dir.mkdir(exist_ok=True)

            for key, trainer in self._trained_models.items():
                try:
                    trainer.save(models_dir / f"{key}.pkl")
                except Exception as e:
                    logger.warning(f"Failed to save model {key}: {e}")

        logger.info(f"Results saved to: {self.output_dir}")

    def get_trained_model(self, model_key: str) -> Any | None:
        """
        Get a trained model by key.

        Args:
            model_key: Key in format "model_name_hHORIZON" (e.g., "xgboost_h20")

        Returns:
            Trainer instance or None if not found
        """
        return self._trained_models.get(model_key)

    def get_oof_predictions(self, model_key: str) -> OOFPrediction | None:
        """
        Get OOF predictions for a model.

        Args:
            model_key: Key in format "model_name_hHORIZON"

        Returns:
            OOFPrediction or None if not found
        """
        return self._oof_predictions.get(model_key)

    def get_meta_labeling_models(
        self,
        horizon: int,
    ) -> tuple[Any, Any, float] | None:
        """
        Get meta-labeling models for a given horizon.

        Returns both the primary model (for direction) and meta-model (for bet sizing),
        along with the configured threshold.

        Args:
            horizon: Prediction horizon

        Returns:
            Tuple of (primary_trainer, meta_model, threshold) or None if not found

        Example:
            primary, meta, threshold = orchestrator.get_meta_labeling_models(20)
            if primary and meta:
                # Get direction from primary
                direction = primary.model.predict(X).class_predictions
                # Get bet probability from meta
                bet_prob = meta.predict_proba(X)[:, 1]
                # Final position = direction * bet_prob (where bet_prob >= threshold)
        """
        primary_key = f"meta_labeling_h{horizon}_primary"
        meta_key = f"meta_labeling_h{horizon}_meta"

        primary = self._trained_models.get(primary_key)
        meta = self._trained_models.get(meta_key)

        if primary is None or meta is None:
            return None

        return primary, meta, self.config.meta_labeling_threshold

    def predict_meta_labeling(
        self,
        X: np.ndarray | pd.DataFrame,
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Generate predictions using trained meta-labeling system.

        Implements Lopez de Prado's meta-labeling:
        1. Primary model predicts direction
        2. Meta-model predicts probability of primary being correct
        3. Only trade where probability >= threshold
        4. Position size = direction * probability

        Args:
            X: Feature array/DataFrame
            horizon: Prediction horizon to use

        Returns:
            Tuple of (directions, probabilities, positions) or None if models not trained
            - directions: Primary model predictions (-1, 0, +1 or class labels)
            - probabilities: Meta-model confidence [0, 1]
            - positions: Final position sizes (direction * probability, 0 if below threshold)

        Example:
            directions, probs, positions = orchestrator.predict_meta_labeling(X, 20)
            # positions contains the bet-sized positions ready for trading
        """
        models = self.get_meta_labeling_models(horizon)
        if models is None:
            logger.warning(f"Meta-labeling models not found for horizon {horizon}")
            return None

        primary_trainer, meta_model, threshold = models

        # Convert to numpy if needed
        X_arr = np.asarray(X)
        if X_arr.ndim > 2:
            X_arr = X_arr.reshape(X_arr.shape[0], -1)

        # Get primary predictions (direction)
        primary_preds = primary_trainer.model.predict(X_arr)
        directions = primary_preds.class_predictions

        # Get meta-model probabilities
        if hasattr(meta_model, 'predict_proba'):
            probabilities = meta_model.predict_proba(X_arr)[:, 1]
        else:
            probabilities = meta_model.predict(X_arr).astype(float)

        # Calculate positions: direction * probability (0 if below threshold)
        # Map class predictions to direction (-1, 0, +1)
        # Assuming 3-class: {0: short, 1: neutral, 2: long} -> {-1, 0, +1}
        direction_mapped = directions.astype(float) - 1.0

        # Position = direction * probability, but 0 if probability < threshold
        positions = np.where(
            probabilities >= threshold,
            direction_mapped * probabilities,
            0.0
        )

        return directions, probabilities, positions


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def train_pipeline(
    config: PipelineConfig,
    df: pd.DataFrame,
    **kwargs: Any,
) -> TrainingRunResult:
    """
    Convenience function for training.

    Usage:
        from src.core import PipelineConfig
        from src.training import train_pipeline

        config = PipelineConfig(
            symbol="MES",
            data_path="./data/mes.parquet",
            output_dir="./experiments",
            models=["xgboost", "lightgbm"],
        )

        result = train_pipeline(config, df)

    Args:
        config: PipelineConfig from src/core
        df: Raw OHLCV DataFrame
        **kwargs: Additional arguments passed to train()

    Returns:
        TrainingRunResult with all outputs
    """
    orchestrator = UnifiedTrainingOrchestrator(config)
    return orchestrator.train(df, **kwargs)


def train_meta_labeling(
    config: PipelineConfig,
    df: pd.DataFrame,
    **kwargs: Any,
) -> TrainingRunResult:
    """
    Convenience function for meta-labeling training.

    This is a shortcut that sets training_mode to "meta_labeling" and runs
    the unified training orchestrator.

    Usage:
        from src.core import PipelineConfig
        from src.training import train_meta_labeling

        config = PipelineConfig(
            symbol="MES",
            data_path="./data/mes.parquet",
            output_dir="./experiments",
            meta_labeling_primary_model="xgboost",
            meta_labeling_meta_model="logistic",
            meta_labeling_threshold=0.5,
        )

        result = train_meta_labeling(config, df)

        # Access results
        print(f"Trade fraction: {result.model_results['meta_labeling_h20'].metrics['trade_fraction']}")
        print(f"Combined accuracy: {result.model_results['meta_labeling_h20'].metrics['combined_accuracy']}")

    Args:
        config: PipelineConfig from src/core (training_mode will be overridden)
        df: Raw OHLCV DataFrame
        **kwargs: Additional arguments passed to train()

    Returns:
        TrainingRunResult with meta-labeling outputs
    """
    # Override training mode to meta_labeling
    config.training_mode = TrainingMode.META_LABELING.value

    orchestrator = UnifiedTrainingOrchestrator(config)
    return orchestrator.train(df, **kwargs)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "UnifiedTrainingOrchestrator",
    "TrainingRunResult",
    "ModelTrainingResult",
    "train_pipeline",
    "train_meta_labeling",
]
