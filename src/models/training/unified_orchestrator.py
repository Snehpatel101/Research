"""
UnifiedTrainingOrchestrator - Thin coordinator for training workflows.

NOW focuses ONLY on:
- Routing to training modes
- Coordinating services
- Managing workflow state

Delegates to:
- ModelTrainingService: Individual model training
- OOFGenerationService: Out-of-fold prediction generation
- ArtifactManager: Saving results
- FeatureSelectionMixin: Feature selection pipeline (feature_selection.py)
- TrainingOpsMixin: Training operations (training_ops.py)

Example:
    from src.core import PipelineConfig
    from src.models.training import UnifiedTrainingOrchestrator

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

import gc
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core import CVMethod, PipelineConfig, TrainingMode
from src.core.exceptions import PreTrainingValidationError
from src.data.adapters import AlignedOOFResult, PreparedData
from src.validation.cv import OOFPrediction, PurgedKFold, PurgedKFoldConfig, StackingDataset

from .feature_selection import FeatureSelectionMixin
from .services import (
    ArtifactManager,
    ArtifactSaveRequest,
    DataPreparer,
    EnsembleRequest,
    EnsembleService,
    ModelTrainingService,
    OOFGenerationService,
    ParallelTrainingService,  # Phase 12A-6: Add parallel training support
)
from .training_ops import TrainingOpsMixin

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
    calibrator: Any | None = None
    training_degraded: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "metrics": self.metrics,
            "training_time_seconds": self.training_time_seconds,
            "n_features": self.n_features,
            "data_rank": self.data_rank,
            "training_degraded": self.training_degraded,
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
            self.model_results.keys(), key=lambda k: self.model_results[k].metrics.get("val_f1", 0)
        )

    def get_metrics_summary(self) -> dict[str, dict[str, float]]:
        """Get summary of all model metrics."""
        return {key: result.metrics for key, result in self.model_results.items()}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "n_models": self.n_models,
            "best_model": self.best_model,
            "total_time_seconds": self.total_time_seconds,
            "output_dir": str(self.output_dir),
            "model_results": {key: result.to_dict() for key, result in self.model_results.items()},
        }


# =============================================================================
# UNIFIED TRAINING ORCHESTRATOR
# =============================================================================


class UnifiedTrainingOrchestrator(FeatureSelectionMixin, TrainingOpsMixin):
    """
    THE single entry point for all training in the ML Factory.

    Uses PipelineConfig from src/core as the ONLY configuration source.

    Supports:
    - All training modes: standard, walk_forward, regime_aware, meta_labeling
    - All CV methods: purged_kfold, cpcv, pbo, walk_forward
    - Heterogeneous ensembles with OOF alignment
    - Integration with PHASE_2 adapters

    Feature selection pipeline and training operations are provided by
    FeatureSelectionMixin (feature_selection.py) and TrainingOpsMixin
    (training_ops.py) respectively.

    Usage:
        from src.core import PipelineConfig
        from src.models.training import UnifiedTrainingOrchestrator

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
        self.output_dir = Path(config.output_dir) / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize services (all heavy lifting is delegated)
        self._data_preparer = DataPreparer(config)
        # Phase 12A-6: Use ParallelTrainingService for multi-model training
        self._model_service = ModelTrainingService()
        self._parallel_service = ParallelTrainingService(n_jobs=config.n_jobs)
        self._oof_service = OOFGenerationService(
            cache_dir=self.output_dir / "oof_cache" if config.save_oof else None
        )
        self._ensemble_service = EnsembleService()
        self._artifact_manager = ArtifactManager(self.output_dir)

        # Initialize CV based on config
        self._cv = self._create_cv()

        # Results storage
        self._model_results: dict[str, ModelTrainingResult] = {}
        self._oof_predictions: dict[str, OOFPrediction] = {}
        self._trained_models: dict[str, Any] = {}

        # PreparedData cache: keyed by contract properties (rank, seq_len, feature_mode,
        # mtf_mode, scaler, n_features) so models with identical data requirements
        # share preparation. Biggest win: 3 boosting models (all rank 2) prepare once,
        # reuse 3x.
        self._prepared_cache: dict[tuple, PreparedData] = {}

        # Per-model feature subsets: each model gets features appropriate for its
        # contract (top N by MDA importance where N = min(available, max_features)).
        # Populated by _pre_training_validation, consumed by _prepare_with_cache.
        self._per_model_features: dict[str, list[str]] = {}
        self._all_feature_names: list[str] = []

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
        if cv_method != CVMethod.PURGED_KFOLD:
            logger.warning(
                f"CV method '{cv_method.value}' requested but not yet implemented in orchestrator. "
                f"Falling back to PurgedKFold. Use walk-forward training mode for true walk-forward CV."
            )

        return PurgedKFold(cv_config)

    def _prepare_with_cache(
        self,
        df: pd.DataFrame,
        model_name: str,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> PreparedData:
        """
        Prepare data with caching by model contract properties.

        Models sharing the same data rank, sequence length, feature mode,
        MTF mode, and scaler type produce identical PreparedData, so we
        prepare once and reuse. This mainly benefits boosting models
        (all rank 2, identical contracts) — prepare once, reuse 3x.

        Neural models with different sequence lengths or feature modes
        get separate cache entries automatically.

        Args:
            df: Input DataFrame
            model_name: Model name (determines contract/adapter)
            additional_dfs: Optional additional timeframe DataFrames

        Returns:
            PreparedData (from cache if available)
        """
        from src.core.contracts import get_model_contract

        contract = get_model_contract(model_name)
        n_model_features = len(self._per_model_features.get(model_name, []))
        cache_key = (
            contract.input_rank.value,
            contract.sequence_length,
            contract.feature_mode.value,
            contract.mtf_mode.value,
            contract.scaler_type,
            n_model_features,
        )

        if cache_key in self._prepared_cache:
            logger.debug(
                f"Cache hit for {model_name} (rank={contract.input_rank.value}, "
                f"seq={contract.sequence_length})"
            )
            return self._prepared_cache[cache_key]

        # Filter DataFrame to per-model feature subset (if computed)
        if self._per_model_features and model_name in self._per_model_features:
            model_features = set(self._per_model_features[model_name])
            all_features = set(self._all_feature_names)
            drop_cols = [c for c in df.columns if c in all_features and c not in model_features]
            if drop_cols:
                df = df.drop(columns=drop_cols)
                logger.debug(f"Filtered to {len(model_features)} features for {model_name}")

        prepared = self._data_preparer.prepare(
            df=df,
            model_name=model_name,
            additional_dfs=additional_dfs,
        )
        self._prepared_cache[cache_key] = prepared
        logger.debug(
            f"Cached PreparedData for {model_name} (rank={contract.input_rank.value}, "
            f"seq={contract.sequence_length})"
        )
        return prepared

    def _clear_prepared_cache(self) -> None:
        """Clear PreparedData cache to free memory after a horizon completes."""
        if self._prepared_cache:
            n_entries = len(self._prepared_cache)
            self._prepared_cache.clear()
            gc.collect()
            logger.debug(f"Cleared {n_entries} PreparedData cache entries")

    def train(
        self,
        df: pd.DataFrame,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
        generate_financial_report: bool = True,
    ) -> TrainingRunResult:
        """
        Execute complete training pipeline.

        This is THE unified interface for all training.

        Args:
            df: Raw OHLCV DataFrame (features will be computed via adapters)
            additional_dfs: Optional dict of additional timeframe DataFrames
                for multi-stream models (e.g., {"5min": df_5min})
            generate_financial_report: Whether to generate financial report
                with visualizations after training (default: True)

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

        # Run pre-training validation (Phase 1: Contract Enforcement)
        # This will raise PreTrainingValidationError if validation fails
        self._pre_training_validation(df)

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

        # Save results (must happen BEFORE clearing OOF predictions, which _save_results uses)
        self._save_results()

        # Clear OOF predictions to free memory after saving (Phase 37 fix)
        # Each OOF dict entry is 50-200MB; clearing saves 750MB-1.5GB for typical runs
        if self._oof_predictions:
            oof_count = len(self._oof_predictions)
            self._oof_predictions.clear()
            logger.debug(f"Cleared {oof_count} OOF predictions from memory")

        # Free trained models and prepared data cache after saving
        model_count = len(self._trained_models)
        self._trained_models.clear()
        self._clear_prepared_cache()
        gc.collect()
        logger.debug(f"Cleared {model_count} trained models from memory")

        total_time = time.time() - start_time

        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Models trained: {len(self._model_results)}")
        logger.info(f"Output: {self.output_dir}")

        # Generate financial report if requested
        if generate_financial_report and self._model_results:
            self._generate_financial_reports(df)

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

    def _build_ensemble(
        self,
        df: pd.DataFrame,
    ) -> tuple[AlignedOOFResult | None, StackingDataset | None, ModelTrainingResult | None]:
        """
        Build ensemble from OOF predictions - delegates to EnsembleService.

        Also performs diversity analysis on base model predictions (Phase 16E).

        Args:
            df: Original DataFrame (for label extraction)

        Returns:
            Tuple of (aligned_oof, stacking_dataset, ensemble_result)
        """
        if not self._oof_predictions:
            logger.warning("No OOF predictions available for ensemble")
            return None, None, None

        # Filter OOF predictions to primary horizon only.
        # Keys are like "xgboost_h5", "xgboost_h20" — mixing horizons in a
        # single ensemble is invalid because labels differ across horizons.
        target_horizon = self.config.horizons[0] if self.config.horizons else 5
        horizon_suffix = f"_h{target_horizon}"
        filtered_oof = {
            k: v for k, v in self._oof_predictions.items() if k.endswith(horizon_suffix)
        }

        if not filtered_oof:
            logger.warning(
                f"No OOF predictions found for target horizon {target_horizon} "
                f"(available keys: {list(self._oof_predictions.keys())}). "
                f"Using all OOF predictions as fallback."
            )
            filtered_oof = self._oof_predictions

        if len(filtered_oof) < len(self._oof_predictions):
            logger.info(
                f"Filtered OOF predictions from {len(self._oof_predictions)} to "
                f"{len(filtered_oof)} for horizon h{target_horizon}"
            )

        # Delegate to EnsembleService
        request = EnsembleRequest(
            oof_predictions=filtered_oof,
            config=self.config,
            df=df,
        )
        result = self._ensemble_service.build_ensemble(request)

        # Phase 16E: Analyze ensemble diversity
        diversity_metrics = self._analyze_ensemble_diversity(result.aligned_oof, df)

        # Convert service result to orchestrator result
        ensemble_result = None
        if result.meta_learner is not None:
            # Merge diversity metrics into ensemble metrics
            ensemble_metrics = result.ensemble_metrics.copy()
            if diversity_metrics:
                ensemble_metrics.update(diversity_metrics)

            ensemble_result = ModelTrainingResult(
                model_name=f"ensemble_{self.config.meta_learner}",
                horizon=self.config.horizons[0] if self.config.horizons else 20,
                metrics=ensemble_metrics,
                trainer=result.meta_learner,
                training_time_seconds=result.training_time_seconds,
            )

        return result.aligned_oof, result.stacking_dataset, ensemble_result

    def _save_results(self) -> None:
        """Save all results - delegates to ArtifactManager."""
        request = ArtifactSaveRequest(
            output_dir=self.output_dir,
            config=self.config,
            model_results=self._model_results,
            oof_predictions=self._oof_predictions,
            trained_models=self._trained_models,
            save_models=self.config.save_models,
            save_oof=self.config.save_oof,
        )
        self._artifact_manager.save_all(request)

    # _analyze_ensemble_diversity and _generate_financial_reports are provided
    # by FeatureSelectionMixin (feature_selection.py)

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
        if hasattr(meta_model, "predict_proba"):
            probabilities = meta_model.predict_proba(X_arr)[:, 1]
        else:
            probabilities = meta_model.predict(X_arr).astype(float)

        # Calculate positions: direction * probability (0 if below threshold)
        # Map class predictions to direction (-1, 0, +1)
        # Assuming 3-class: {0: short, 1: neutral, 2: long} -> {-1, 0, +1}
        direction_mapped = directions.astype(float) - 1.0

        # Position = direction * probability, but 0 if probability < threshold
        positions = np.where(probabilities >= threshold, direction_mapped * probabilities, 0.0)

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
        from src.models.training import train_pipeline

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
        from src.models.training import train_meta_labeling

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
    "PreTrainingValidationError",
    "TrainingRunResult",
    "ModelTrainingResult",
    "train_pipeline",
    "train_meta_labeling",
]
