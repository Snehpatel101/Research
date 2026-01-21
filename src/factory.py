"""
MLFactory - THE single cohesive entry point for the entire ML pipeline.

UNIFIED PIPELINE ARCHITECTURE (PHASE_0):
MLFactory is the ONE entry point that internally delegates to PipelineRunner
(Phase 1) for data preparation, ensuring no duplicate implementations.

This module provides:
1. One entry point: `MLFactory`
2. One configuration: `PipelineConfig` from src/core
3. Complete pipeline from raw OHLCV to inference-ready bundle
4. Internal delegation to PipelineRunner for data preparation

ARCHITECTURE:
    Raw OHLCV DataFrame
           │
           ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  PHASE 1: DATA PREPARATION (via PipelineRunner)              │
    │  ┌──────────────────┐  ┌──────────────────┐                  │
    │  │  Feature Compute │→ │  Labeling        │→ Splits/Scaling  │
    │  │  162 base + MTF  │  │  Triple Barrier  │                  │
    │  └──────────────────┘  └──────────────────┘                  │
    └────────────────────────────┬─────────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────┐
    │  Adapters        │  PHASE_2: Convert to 2D/3D/4D per model
    │  (Tabular/Seq)   │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  Training        │  PHASE_3: Standard/WalkForward/Regime/MetaLabeling
    │  (+ OOF)         │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  Ensemble        │  PHASE_4: Heterogeneous stacking
    │  (Meta-learner)  │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  Bundle          │  PHASE_5: Inference-ready package
    │  (Inference)     │
    └──────────────────┘

Usage:
    from src.factory import MLFactory, run_pipeline
    from src.core import PipelineConfig

    # Option 1: Use the class
    config = PipelineConfig(
        symbol="MES",
        data_path="./data/mes_1min.parquet",
        output_dir="./experiments/exp_001",
        models=["xgboost", "lightgbm", "lstm"],
        build_ensemble=True,
        training_mode="standard",  # or "regime_aware", "meta_labeling"
    )

    factory = MLFactory(config)
    result = factory.run(df)

    # Option 2: Use the convenience function
    result = run_pipeline(config, df)

    # Access results
    print(f"Best model: {result.best_model}")
    print(f"Metrics: {result.metrics_summary}")

    # Get inference bundle
    bundle = result.get_inference_bundle()
    predictions = bundle.predict(new_data)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import pandas as pd

from src.core import (
    PipelineConfig,
    TrainingMode,
    LabelingMethod,
)

if TYPE_CHECKING:
    from src.training import TrainingRunResult

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT DATACLASSES
# =============================================================================


@dataclass
class FeatureComputeResult:
    """Result from feature computation stage."""

    features_df: pd.DataFrame
    base_feature_count: int
    mtf_feature_count: int
    total_feature_count: int
    computation_time_seconds: float

    def __repr__(self) -> str:
        return (
            f"FeatureComputeResult("
            f"base={self.base_feature_count}, "
            f"mtf={self.mtf_feature_count}, "
            f"total={self.total_feature_count}, "
            f"time={self.computation_time_seconds:.1f}s)"
        )


@dataclass
class LabelingResult:
    """Result from labeling stage."""

    labels: pd.Series
    labeling_method: str
    n_samples: int
    class_distribution: Dict[int, int]
    label_params: Dict[str, Any]
    computation_time_seconds: float

    def __repr__(self) -> str:
        dist_str = ", ".join(f"{k}:{v}" for k, v in self.class_distribution.items())
        return (
            f"LabelingResult("
            f"method={self.labeling_method}, "
            f"n={self.n_samples}, "
            f"dist=[{dist_str}], "
            f"time={self.computation_time_seconds:.1f}s)"
        )


@dataclass
class PipelineResult:
    """
    Complete result from the MLFactory pipeline.

    This is THE result object that contains everything from the pipeline run.

    Attributes:
        run_id: Unique identifier for this run
        config: PipelineConfig used
        feature_result: Feature computation result
        labeling_result: Labeling result
        training_result: Training result from UnifiedTrainingOrchestrator
        output_dir: Directory where results are saved
        total_time_seconds: Total pipeline time
    """

    run_id: str
    config: PipelineConfig
    feature_result: Optional[FeatureComputeResult] = None
    labeling_result: Optional[LabelingResult] = None
    training_result: Optional["TrainingRunResult"] = None
    output_dir: Path = field(default_factory=lambda: Path("."))
    total_time_seconds: float = 0.0

    @property
    def best_model(self) -> str | None:
        """Get the best performing model name."""
        if self.training_result:
            return self.training_result.best_model
        return None

    @property
    def n_models_trained(self) -> int:
        """Number of models trained."""
        if self.training_result:
            return self.training_result.n_models
        return 0

    @property
    def metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all model metrics."""
        if self.training_result:
            return self.training_result.get_metrics_summary()
        return {}

    def get_inference_bundle(self) -> Optional[Any]:
        """
        Get inference bundle for deployment.

        Returns the trained models packaged for inference, or None
        if no ensemble was built.
        """
        if self.training_result is None:
            return None

        # Check if we have an ensemble result
        if self.training_result.ensemble_result:
            return self._build_ensemble_bundle()

        # Otherwise return the best single model
        if self.best_model and self.training_result.model_results:
            best_key = self.best_model
            if best_key in self.training_result.model_results:
                return self.training_result.model_results[best_key].trainer
        return None

    def _build_ensemble_bundle(self) -> Any:
        """Build inference bundle from ensemble."""
        from src.inference import EnsembleBundle

        bundle = EnsembleBundle(
            base_models={
                key: result.trainer
                for key, result in self.training_result.model_results.items()
                if result.trainer is not None
            },
            meta_learner=self.training_result.ensemble_result.trainer,
            aligned_oof=self.training_result.aligned_oof,
            config=self.config,
        )
        return bundle

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "config": self.config.to_dict(),
            "output_dir": str(self.output_dir),
            "total_time_seconds": self.total_time_seconds,
            "n_models_trained": self.n_models_trained,
            "best_model": self.best_model,
            "metrics_summary": self.metrics_summary,
            "feature_result": {
                "base_count": self.feature_result.base_feature_count,
                "mtf_count": self.feature_result.mtf_feature_count,
                "total_count": self.feature_result.total_feature_count,
                "time": self.feature_result.computation_time_seconds,
            }
            if self.feature_result
            else None,
            "labeling_result": {
                "method": self.labeling_result.labeling_method,
                "n_samples": self.labeling_result.n_samples,
                "distribution": self.labeling_result.class_distribution,
            }
            if self.labeling_result
            else None,
        }

    def save(self, path: Optional[Path] = None) -> Path:
        """Save result summary to JSON."""
        save_path = path or (self.output_dir / "pipeline_result.json")
        with open(save_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return save_path


# =============================================================================
# ML FACTORY - THE SINGLE ENTRY POINT
# =============================================================================


class MLFactory:
    """
    THE single cohesive entry point for the entire ML pipeline.

    MLFactory orchestrates all phases of the ML pipeline:
    - PHASE_1: Feature computation (162 base + 240 MTF features)
    - PHASE_1B: Labeling (triple barrier, meta-labeling, Optuna optimization)
    - PHASE_1: Feature selection (Optuna-based pruning, SHAP)
    - PHASE_2: Data preparation via adapters (2D/3D/4D)
    - PHASE_3: Training (standard, walk-forward, regime-aware, meta-labeling)
    - PHASE_4: Ensemble building (heterogeneous stacking)
    - PHASE_5: Bundling for inference

    Configuration is centralized through PipelineConfig from src/core.

    Example:
        from src.factory import MLFactory
        from src.core import PipelineConfig

        config = PipelineConfig(
            symbol="MES",
            data_path="./data/mes.parquet",
            output_dir="./experiments",
            models=["xgboost", "lightgbm", "lstm"],
            training_mode="standard",
            build_ensemble=True,
            compute_mtf_features=True,
            optimize_labels=True,
        )

        factory = MLFactory(config)
        result = factory.run(df)

        # Access results
        print(f"Best model: {result.best_model}")
        print(f"Total time: {result.total_time_seconds:.1f}s")

        # Get inference bundle
        bundle = result.get_inference_bundle()
        predictions = bundle.predict(new_data)
    """

    def __init__(self, config: PipelineConfig) -> None:
        """
        Initialize MLFactory with configuration.

        Args:
            config: PipelineConfig from src/core - THE single config source
        """
        self.config = config
        self.run_id = self._generate_run_id()
        self.output_dir = config.output_dir / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info("ML FACTORY INITIALIZED")
        logger.info("=" * 70)
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Symbol: {config.symbol}")
        logger.info(f"Training mode: {config.training_mode}")
        logger.info(f"Models: {config.models}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("=" * 70)

    def _generate_run_id(self) -> str:
        """Generate unique run identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"ml_factory_{self.config.symbol}_{timestamp}"

    def run(
        self,
        df: pd.DataFrame,
        additional_dfs: Optional[Dict[str, pd.DataFrame]] = None,
        skip_features: bool = False,
        skip_labels: bool = False,
    ) -> PipelineResult:
        """
        Execute the complete ML pipeline.

        This is THE main entry point that runs all phases.

        Args:
            df: Raw OHLCV DataFrame with columns [open, high, low, close, volume]
                and DatetimeIndex
            additional_dfs: Optional dict of additional timeframe DataFrames
                for multi-stream models (e.g., {"5min": df_5min})
            skip_features: If True, assume df already has features computed
            skip_labels: If True, assume df already has labels

        Returns:
            PipelineResult containing all outputs from the pipeline
        """
        start_time = time.time()

        logger.info("\n" + "=" * 70)
        logger.info("STARTING ML FACTORY PIPELINE (UNIFIED)")
        logger.info("=" * 70)

        # Initialize result
        feature_result = None
        labeling_result = None
        training_result = None

        # =================================================================
        # PHASE 1: DATA PREPARATION (via PipelineRunner)
        # =================================================================
        # This is the UNIFIED approach - MLFactory delegates to PipelineRunner
        # for all data preparation (features, labels, splits, scaling)
        if not skip_features and not skip_labels:
            logger.info("\n" + "-" * 60)
            logger.info("PHASE 1: DATA PREPARATION (via PipelineRunner)")
            logger.info("-" * 60)
            feature_result, labeling_result, df = self._run_data_pipeline(df)
            logger.info(f"Feature computation complete: {feature_result}")
            logger.info(f"Labeling complete: {labeling_result}")
        elif not skip_features:
            # Legacy path: separate feature computation (deprecated)
            logger.info("\n" + "-" * 60)
            logger.info("PHASE 1: FEATURE COMPUTATION (legacy)")
            logger.info("-" * 60)
            feature_result, df = self._compute_features(df)
            logger.info(f"Feature computation complete: {feature_result}")
        else:
            logger.info("\nSkipping data preparation (skip_features=True)")

        # =================================================================
        # PHASE 1B: LABELING (legacy path, only if skip_features=True)
        # =================================================================
        if skip_features and not skip_labels:
            logger.info("\n" + "-" * 60)
            logger.info("PHASE 1B: LABELING (legacy)")
            logger.info("-" * 60)
            labeling_result, df = self._compute_labels(df)
            logger.info(f"Labeling complete: {labeling_result}")
        elif skip_labels and labeling_result is None:
            logger.info("\nSkipping labeling (skip_labels=True)")

        # =================================================================
        # PHASE 1: FEATURE SELECTION (optional, post-processing)
        # =================================================================
        if self.config.optimize_features and feature_result:
            logger.info("\n" + "-" * 60)
            logger.info("PHASE 1: FEATURE SELECTION")
            logger.info("-" * 60)
            df = self._select_features(df)

        # =================================================================
        # PHASE 2-4: TRAINING (includes adapters, training, ensemble)
        # =================================================================
        logger.info("\n" + "-" * 60)
        logger.info("PHASES 2-4: TRAINING PIPELINE")
        logger.info("-" * 60)
        training_result = self._run_training(df, additional_dfs)

        # =================================================================
        # PHASE 5: SAVE AND BUNDLE
        # =================================================================
        logger.info("\n" + "-" * 60)
        logger.info("PHASE 5: SAVING RESULTS")
        logger.info("-" * 60)

        total_time = time.time() - start_time

        result = PipelineResult(
            run_id=self.run_id,
            config=self.config,
            feature_result=feature_result,
            labeling_result=labeling_result,
            training_result=training_result,
            output_dir=self.output_dir,
            total_time_seconds=total_time,
        )

        # Save result summary
        result.save()

        logger.info("\n" + "=" * 70)
        logger.info("ML FACTORY PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Models trained: {result.n_models_trained}")
        logger.info(f"Best model: {result.best_model}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("=" * 70)

        return result

    def _run_data_pipeline(
        self,
        df: pd.DataFrame,
    ) -> tuple[FeatureComputeResult, LabelingResult, pd.DataFrame]:
        """
        Run Phase 1 data pipeline via PipelineRunner.

        This is the UNIFIED approach - delegates to PipelineRunner instead of
        having duplicate feature/labeling implementations.

        Args:
            df: Raw OHLCV DataFrame

        Returns:
            Tuple of (FeatureComputeResult, LabelingResult, DataFrame with features & labels)
        """
        start = time.time()

        # Convert core PipelineConfig to DataConfig for PipelineRunner
        data_config = self.config.to_data_config()

        # Save the input DataFrame to the expected location for PipelineRunner
        # PipelineRunner expects data in a specific directory structure
        raw_data_dir = data_config.data_raw_dir
        raw_data_dir.mkdir(parents=True, exist_ok=True)

        # Save as parquet for PipelineRunner
        raw_data_path = raw_data_dir / f"{self.config.symbol}_1min.parquet"
        df.to_parquet(raw_data_path)
        logger.info(f"  Saved raw data to: {raw_data_path}")

        # Run PipelineRunner for Phase 1 (data preparation)
        from src.pipeline.runner import PipelineRunner

        runner = PipelineRunner(data_config, resume=False)
        success = runner.run()

        if not success:
            raise RuntimeError("Phase 1 data pipeline failed")

        # Load the prepared data from PipelineRunner outputs
        features_dir = data_config.data_features_dir
        tf = data_config.target_timeframe

        # Load features DataFrame
        features_path = features_dir / f"features_{tf}.parquet"
        if not features_path.exists():
            # Try alternate naming
            features_path = features_dir / f"{self.config.symbol}_{tf}_features.parquet"

        result_df = pd.read_parquet(features_path)

        # Extract feature counts
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        label_cols = [c for c in result_df.columns if c.startswith("label") or c == "target"]
        feature_cols = [
            c for c in result_df.columns if c not in ohlcv_cols + label_cols + ["timestamp"]
        ]

        # Estimate base vs MTF features
        mtf_feature_cols = [
            c for c in feature_cols if any(tf in c for tf in ["5min", "15min", "60min", "1h"])
        ]
        base_feature_cols = [c for c in feature_cols if c not in mtf_feature_cols]

        computation_time = time.time() - start

        feature_result = FeatureComputeResult(
            features_df=result_df[feature_cols] if feature_cols else pd.DataFrame(),
            base_feature_count=len(base_feature_cols),
            mtf_feature_count=len(mtf_feature_cols),
            total_feature_count=len(feature_cols),
            computation_time_seconds=computation_time,
        )

        # Extract labeling result
        label_col = "label" if "label" in result_df.columns else "target"
        labels = result_df[label_col] if label_col in result_df.columns else pd.Series()

        labeling_result = LabelingResult(
            labels=labels,
            labeling_method=self.config.labeling_method,
            n_samples=len(labels.dropna()) if not labels.empty else 0,
            class_distribution={int(k): int(v) for k, v in labels.value_counts().items()}
            if not labels.empty
            else {},
            label_params={
                "horizons": self.config.horizons,
                "upper_mult": self.config.upper_mult,
                "lower_mult": self.config.lower_mult,
            },
            computation_time_seconds=0.0,  # Included in feature time
        )

        logger.info(
            f"  Phase 1 complete: {len(feature_cols)} features, {len(labels.dropna())} labeled samples"
        )

        return feature_result, labeling_result, result_df

    def _compute_features(
        self,
        df: pd.DataFrame,
    ) -> tuple[FeatureComputeResult, pd.DataFrame]:
        """
        Compute all features including MTF.

        DEPRECATED: This method is kept for backward compatibility.
        Use _run_data_pipeline() which delegates to PipelineRunner for unified execution.

        Args:
            df: Raw OHLCV DataFrame

        Returns:
            Tuple of (FeatureComputeResult, DataFrame with features)
        """
        import warnings

        warnings.warn(
            "MLFactory._compute_features is deprecated. "
            "The unified pipeline now uses PipelineRunner for data preparation.",
            DeprecationWarning,
            stacklevel=2,
        )

        from src.features.compute import (
            compute_all_features,
            compute_mtf_features,
            TOTAL_FEATURES,
        )

        start = time.time()

        # Compute base features (162)
        logger.info("  Computing base features...")
        base_features = compute_all_features(df)
        base_count = len(base_features.columns)
        logger.info(f"    Base features: {base_count}")

        # Compute MTF features if enabled
        mtf_count = 0
        if self.config.compute_mtf_features:
            logger.info("  Computing MTF features...")
            mtf_features = compute_mtf_features(
                df,
                timeframes=self.config.mtf_timeframes,
            )
            mtf_count = len(mtf_features.columns)
            logger.info(f"    MTF features: {mtf_count}")

            # Combine
            features_df = pd.concat([base_features, mtf_features], axis=1)
        else:
            features_df = base_features

        # Combine with original OHLCV
        result_df = pd.concat([df, features_df], axis=1)

        computation_time = time.time() - start

        feature_result = FeatureComputeResult(
            features_df=features_df,
            base_feature_count=base_count,
            mtf_feature_count=mtf_count,
            total_feature_count=len(features_df.columns),
            computation_time_seconds=computation_time,
        )

        return feature_result, result_df

    def _compute_labels(
        self,
        df: pd.DataFrame,
    ) -> tuple[LabelingResult, pd.DataFrame]:
        """
        Compute labels using configured method.

        DEPRECATED: This method is kept for backward compatibility.
        Use _run_data_pipeline() which delegates to PipelineRunner for unified execution.

        Args:
            df: DataFrame with OHLCV and features

        Returns:
            Tuple of (LabelingResult, DataFrame with labels)
        """
        import warnings

        warnings.warn(
            "MLFactory._compute_labels is deprecated. "
            "The unified pipeline now uses PipelineRunner for data preparation.",
            DeprecationWarning,
            stacklevel=2,
        )

        start = time.time()

        labeling_method = LabelingMethod(self.config.labeling_method)

        if labeling_method == LabelingMethod.TRIPLE_BARRIER:
            labels, params = self._compute_triple_barrier_labels(df)
        elif labeling_method == LabelingMethod.FIXED_HORIZON:
            labels, params = self._compute_fixed_horizon_labels(df)
        else:
            raise ValueError(f"Unknown labeling method: {labeling_method}")

        # Optimize labels with Optuna if enabled
        if self.config.optimize_labels:
            logger.info("  Optimizing labels with Optuna...")
            labels, params = self._optimize_labels(df, labels)

        # Add labels to DataFrame
        df["label"] = labels

        # Compute class distribution
        class_dist = {int(k): int(v) for k, v in labels.value_counts().items()}

        computation_time = time.time() - start

        result = LabelingResult(
            labels=labels,
            labeling_method=labeling_method.value,
            n_samples=len(labels.dropna()),
            class_distribution=class_dist,
            label_params=params,
            computation_time_seconds=computation_time,
        )

        return result, df

    def _compute_triple_barrier_labels(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.Series, Dict[str, Any]]:
        """Compute triple barrier labels."""
        from src.labeling import TripleBarrierLabeler, TripleBarrierConfig

        labeler_config = TripleBarrierConfig(
            upper_mult=self.config.upper_mult,
            lower_mult=self.config.lower_mult,
            atr_period=self.config.atr_period,
            horizon=self.config.max_holding_bars,
        )

        labeler = TripleBarrierLabeler(labeler_config)
        labels = labeler.create_labels(df)

        params = {
            "upper_mult": self.config.upper_mult,
            "lower_mult": self.config.lower_mult,
            "atr_period": self.config.atr_period,
            "max_holding_bars": self.config.max_holding_bars,
        }

        return labels, params

    def _compute_fixed_horizon_labels(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.Series, Dict[str, Any]]:
        """Compute fixed horizon labels."""
        horizon = self.config.horizons[0]

        # Simple fixed horizon labeling
        future_returns = df["close"].shift(-horizon) / df["close"] - 1

        # Use quantiles for thresholds
        upper_q = future_returns.quantile(0.6)
        lower_q = future_returns.quantile(0.4)

        labels = pd.Series(1, index=df.index)  # Neutral
        labels[future_returns > upper_q] = 2  # Long
        labels[future_returns < lower_q] = 0  # Short

        params = {
            "horizon": horizon,
            "upper_quantile": 0.6,
            "lower_quantile": 0.4,
        }

        return labels, params

    def _optimize_labels(
        self,
        df: pd.DataFrame,
        initial_labels: pd.Series,
    ) -> tuple[pd.Series, Dict[str, Any]]:
        """Optimize label parameters using Optuna."""
        from src.labeling import LabelOptimizer

        optimizer = LabelOptimizer(
            n_trials=self.config.label_optimization_trials,
        )

        result = optimizer.optimize(df)

        logger.info(f"    Optimized params: {result.best_config}")

        # Create new labels with optimized config
        from src.labeling import TripleBarrierLabeler

        optimized_labeler = TripleBarrierLabeler(result.best_config)
        optimized_labels = optimized_labeler.create_labels(df)

        return optimized_labels, result.best_config.__dict__ if hasattr(
            result.best_config, "__dict__"
        ) else {}

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select features using configured method.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with selected features
        """
        from src.features import FeatureSelector

        logger.info("  Running feature selection...")

        selector = FeatureSelector(
            method=self.config.feature_selection_method,
            n_trials=self.config.feature_selection_trials,
            min_features=self.config.min_features,
        )

        # Get feature columns (exclude OHLCV and label)
        exclude_cols = ["open", "high", "low", "close", "volume", "label"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        selected_features = selector.select(
            X=df[feature_cols],
            y=df["label"],
        )

        logger.info(f"    Selected {len(selected_features)} features")

        # Keep only selected features plus required columns
        keep_cols = exclude_cols + selected_features
        keep_cols = [c for c in keep_cols if c in df.columns]

        return df[keep_cols]

    def _run_training(
        self,
        df: pd.DataFrame,
        additional_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> "TrainingRunResult":
        """
        Run the training pipeline.

        This delegates to UnifiedTrainingOrchestrator which handles:
        - Data preparation via adapters (PHASE 2)
        - Model training with configured mode (PHASE 3)
        - Ensemble building if enabled (PHASE 4)

        Args:
            df: DataFrame with features and labels
            additional_dfs: Optional additional DataFrames for multi-stream

        Returns:
            TrainingRunResult from the orchestrator
        """
        from src.training import UnifiedTrainingOrchestrator

        # Update config output directory to use our run directory
        self.config.output_dir = self.output_dir

        orchestrator = UnifiedTrainingOrchestrator(self.config)
        return orchestrator.train(df, additional_dfs)

    def get_feature_names(self) -> list[str]:
        """Get list of all feature names that will be computed."""
        from src.features.compute import get_all_feature_names, get_mtf_feature_names

        base_names = get_all_feature_names()

        if self.config.compute_mtf_features:
            mtf_names = get_mtf_feature_names(
                timeframes=self.config.mtf_timeframes,
            )
            return base_names + mtf_names

        return base_names


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def run_pipeline(
    config: PipelineConfig,
    df: pd.DataFrame,
    additional_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    **kwargs: Any,
) -> PipelineResult:
    """
    Convenience function to run the complete ML pipeline.

    This is the simplest way to run the entire pipeline.

    Args:
        config: PipelineConfig from src/core
        df: Raw OHLCV DataFrame
        additional_dfs: Optional additional timeframe DataFrames
        **kwargs: Additional arguments passed to MLFactory.run()

    Returns:
        PipelineResult with all outputs

    Example:
        from src.factory import run_pipeline
        from src.core import PipelineConfig

        config = PipelineConfig(
            symbol="MES",
            data_path="./data/mes.parquet",
            output_dir="./experiments",
            models=["xgboost", "lightgbm"],
        )

        result = run_pipeline(config, df)
        print(f"Best model: {result.best_model}")
    """
    factory = MLFactory(config)
    return factory.run(df, additional_dfs, **kwargs)


def quick_run(
    df: pd.DataFrame,
    models: list[str] = None,
    output_dir: str = "./experiments",
    **config_kwargs: Any,
) -> PipelineResult:
    """
    Quick pipeline run with minimal configuration.

    Args:
        df: Raw OHLCV DataFrame
        models: List of model names (default: ["xgboost", "lightgbm"])
        output_dir: Output directory
        **config_kwargs: Additional PipelineConfig parameters

    Returns:
        PipelineResult

    Example:
        from src.factory import quick_run

        result = quick_run(df, models=["xgboost", "lstm"])
    """
    from src.core import quick_config

    config = quick_config(
        models=models or ["xgboost", "lightgbm"],
        **config_kwargs,
    )
    config.output_dir = Path(output_dir)

    return run_pipeline(config, df)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main class
    "MLFactory",
    # Result types
    "PipelineResult",
    "FeatureComputeResult",
    "LabelingResult",
    # Convenience functions
    "run_pipeline",
    "quick_run",
]
