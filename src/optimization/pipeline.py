"""
Unified Optimization Pipeline - PHASE_1B Full Optimization Orchestrator.

This module provides the OptimizationPipeline class that orchestrates all
optimization phases in the ML Factory pipeline:

1. Label Optimization (100 trials) - Triple-barrier parameter tuning
2. Feature Selection (100 trials) - Binary include/exclude optimization
3. Feature Pruning (50 trials) - Importance-based feature removal
4. Hyperparameter Optimization (100 trials per model) - Model-specific tuning

Integration:
    The pipeline integrates with PipelineConfig from src.core:
    - optimize_labels, label_optimization_trials
    - optimize_features, feature_selection_trials, feature_pruning_trials
    - optimize_hyperparams, hyperparam_trials
    - optuna_random_state

Usage:
    from src.optimization import OptimizationPipeline
    from src.core import PipelineConfig

    config = PipelineConfig(
        symbol="MES",
        data_path="./data/mes_1min.parquet",
        output_dir="./experiments/exp_001",
        models=["xgboost", "lightgbm", "lstm"],
        optimize_labels=True,
        optimize_features=True,
        optimize_hyperparams=True,
    )

    pipeline = OptimizationPipeline.from_config(config)
    result = pipeline.run_full_optimization(
        ohlcv_df=ohlcv_df,
        feature_df=feature_df,
        models=["xgboost", "lightgbm", "lstm"],
        model_factories=factories,
    )

    # Access results
    print(f"Final features: {len(result.final_features)}")
    print(f"Label config: {result.final_label_config}")
    print(f"Total trials: {result.total_trials}")

Reference: Bergstra et al. (2011) "Algorithms for Hyper-Parameter Optimization"
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core import (
    DEFAULT_FEATURE_PRUNING_TRIALS,
    DEFAULT_FEATURE_SELECTION_TRIALS,
    DEFAULT_HYPERPARAM_TRIALS,
    DEFAULT_LABEL_OPTIMIZATION_TRIALS,
    DEFAULT_MIN_FEATURES,
    DEFAULT_OPTUNA_RANDOM_STATE,
    PipelineConfig,
)
from src.optimization.features import (
    FeatureOptimizer,
    FeaturePruningResult,
    FeatureSelectionResult,
)
from src.optimization.hyperparameters import (
    HyperparameterOptimizer,
    HyperparameterResult,
)
from src.optimization.labels import (
    LabelOptimizationResult,
    LabelOptimizer,
    TripleBarrierConfig,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT DATACLASS
# =============================================================================


@dataclass
class FullOptimizationResult:
    """
    Combined result container for full optimization pipeline.

    Attributes:
        # Stage results
        label_result: Label optimization result (if run)
        selection_result: Feature selection result (if run)
        pruning_result: Feature pruning result (if run)
        hyperparam_results: Dict of model_name -> HyperparameterResult

        # Final configurations
        final_features: Final list of selected feature names
        final_label_config: Final triple-barrier configuration
        final_hyperparams: Dict of model_name -> best hyperparameters

        # Statistics
        total_trials: Total number of Optuna trials across all stages
        optimization_time_seconds: Total time taken for all optimization

        # Metadata
        stages_run: List of stages that were run
        config_used: PipelineConfig that was used
    """

    # Stage results
    label_result: LabelOptimizationResult | None = None
    selection_result: FeatureSelectionResult | None = None
    pruning_result: FeaturePruningResult | None = None
    hyperparam_results: dict[str, HyperparameterResult] = field(default_factory=dict)

    # Final configurations
    final_features: list[str] = field(default_factory=list)
    final_label_config: TripleBarrierConfig | None = None
    final_hyperparams: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Statistics
    total_trials: int = 0
    optimization_time_seconds: float = 0.0

    # Metadata
    stages_run: list[str] = field(default_factory=list)
    config_used: dict[str, Any] | None = None

    def __repr__(self) -> str:
        stages = ", ".join(self.stages_run) if self.stages_run else "none"
        return (
            f"FullOptimizationResult(\n"
            f"  stages_run=[{stages}],\n"
            f"  total_trials={self.total_trials},\n"
            f"  n_final_features={len(self.final_features)},\n"
            f"  n_models_optimized={len(self.hyperparam_results)},\n"
            f"  optimization_time={self.optimization_time_seconds:.1f}s\n"
            f")"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "final_features": self.final_features,
            "final_label_config": (
                self.final_label_config.to_dict() if self.final_label_config else None
            ),
            "final_hyperparams": self.final_hyperparams,
            "total_trials": self.total_trials,
            "optimization_time_seconds": self.optimization_time_seconds,
            "stages_run": self.stages_run,
        }

        if self.label_result:
            result["label_result"] = self.label_result.to_dict()
        if self.selection_result:
            result["selection_result"] = self.selection_result.to_dict()
        if self.pruning_result:
            result["pruning_result"] = self.pruning_result.to_dict()
        if self.hyperparam_results:
            result["hyperparam_results"] = {
                k: v.to_dict() for k, v in self.hyperparam_results.items()
            }

        return result

    def save(self, path: Path) -> None:
        """Save result to JSON file."""
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        logger.info(f"Saved optimization result to {path}")


# =============================================================================
# OPTIMIZATION PIPELINE
# =============================================================================


class OptimizationPipeline:
    """
    Unified Optuna optimization pipeline.

    Runs all optimization stages in sequence:
    1. Label optimization (triple-barrier parameters) - 100 trials
    2. Feature selection (binary include/exclude) - 100 trials
    3. Feature pruning (importance-based removal) - 50 trials
    4. Hyperparameter optimization (per model) - 100 trials each

    Args:
        label_trials: Trials for label optimization (default: 100)
        feature_trials: Trials for feature selection (default: 100)
        pruning_trials: Trials for feature pruning (default: 50)
        hyperparam_trials: Trials per model for hyperparameters (default: 100)
        min_features: Minimum features to retain (default: 20)
        random_state: Random seed for reproducibility (default: 42)
        verbose: Verbosity level (0=silent, 1=progress, 2=debug)

    Example:
        >>> pipeline = OptimizationPipeline(
        ...     label_trials=100,
        ...     feature_trials=100,
        ...     hyperparam_trials=100,
        ... )
        >>> result = pipeline.run_full_optimization(
        ...     ohlcv_df=ohlcv_df,
        ...     feature_df=feature_df,
        ...     models=["xgboost", "lightgbm"],
        ...     model_factories=factories,
        ... )
    """

    def __init__(
        self,
        label_trials: int = DEFAULT_LABEL_OPTIMIZATION_TRIALS,
        feature_trials: int = DEFAULT_FEATURE_SELECTION_TRIALS,
        pruning_trials: int = DEFAULT_FEATURE_PRUNING_TRIALS,
        hyperparam_trials: int = DEFAULT_HYPERPARAM_TRIALS,
        min_features: int = DEFAULT_MIN_FEATURES,
        random_state: int = DEFAULT_OPTUNA_RANDOM_STATE,
        scoring: str = "f1_weighted",
        verbose: int = 1,
    ):
        self.label_trials = label_trials
        self.feature_trials = feature_trials
        self.pruning_trials = pruning_trials
        self.hyperparam_trials = hyperparam_trials
        self.min_features = min_features
        self.random_state = random_state
        self.scoring = scoring
        self.verbose = verbose

        # Initialize sub-optimizers
        self._label_optimizer: LabelOptimizer | None = None
        self._feature_optimizer: FeatureOptimizer | None = None
        self._hyperparam_optimizer: HyperparameterOptimizer | None = None

    @classmethod
    def from_config(cls, config: PipelineConfig) -> OptimizationPipeline:
        """
        Create pipeline from PipelineConfig.

        Args:
            config: PipelineConfig with optimization settings

        Returns:
            Configured OptimizationPipeline
        """
        return cls(
            label_trials=config.label_optimization_trials,
            feature_trials=config.feature_selection_trials,
            pruning_trials=config.feature_pruning_trials,
            hyperparam_trials=config.hyperparam_trials,
            min_features=config.min_features,
            random_state=config.optuna_random_state,
            scoring=config.optuna_metric,
            verbose=config.verbose,
        )

    def _print_stage_header(self, stage_name: str, stage_num: int, total_stages: int) -> None:
        """Print a formatted stage header."""
        if self.verbose >= 1:
            print(f"\n{'='*70}")
            print(f"STAGE {stage_num}/{total_stages}: {stage_name}")
            print(f"{'='*70}")

    def _print_pipeline_header(self, stages: list[str]) -> None:
        """Print pipeline start header."""
        if self.verbose >= 1:
            print(f"\n{'#'*70}")
            print("# OPTIMIZATION PIPELINE - PHASE_1B")
            print(f"# Stages: {', '.join(stages)}")
            print(f"# Random state: {self.random_state}")
            print(f"{'#'*70}")

    def run_full_optimization(
        self,
        ohlcv_df: pd.DataFrame,
        feature_df: pd.DataFrame,
        models: list[str],
        model_factories: dict[str, Callable[[dict[str, Any]], Any]],
        labels: np.ndarray | None = None,
        optimize_labels: bool = True,
        optimize_features: bool = True,
        optimize_hyperparams: bool = True,
        target_distribution: dict[int, float] | None = None,
    ) -> FullOptimizationResult:
        """
        Run full optimization pipeline.

        Args:
            ohlcv_df: OHLCV DataFrame with required columns
            feature_df: Feature DataFrame (same index as ohlcv_df)
            models: List of model names to optimize
            model_factories: Dict mapping model_name -> factory function
            labels: Optional pre-computed labels (skips label optimization)
            optimize_labels: Whether to run label optimization
            optimize_features: Whether to run feature optimization
            optimize_hyperparams: Whether to run hyperparameter optimization
            target_distribution: Target class distribution for labels

        Returns:
            FullOptimizationResult with all optimization results
        """
        start_time = time.time()
        total_trials = 0
        stages_run = []

        # Determine which stages to run
        stages_to_run = []
        if optimize_labels and labels is None:
            stages_to_run.append("label_optimization")
        if optimize_features:
            stages_to_run.append("feature_selection")
            stages_to_run.append("feature_pruning")
        if optimize_hyperparams:
            stages_to_run.append("hyperparameter_optimization")

        self._print_pipeline_header(stages_to_run)

        # Initialize result
        result = FullOptimizationResult()

        # Stage counter
        stage_num = 0
        total_stages = len(stages_to_run)

        # =====================================================================
        # STAGE 1: Label Optimization
        # =====================================================================
        if optimize_labels and labels is None:
            stage_num += 1
            self._print_stage_header("Label Optimization", stage_num, total_stages)

            self._label_optimizer = LabelOptimizer(
                n_trials=self.label_trials,
                objective="class_balance",
                random_state=self.random_state,
                verbose=self.verbose,
            )

            label_result = self._label_optimizer.optimize(
                ohlcv_df=ohlcv_df,
                target_distribution=target_distribution,
            )

            result.label_result = label_result
            result.final_label_config = label_result.best_config
            total_trials += label_result.n_trials
            stages_run.append("label_optimization")

            # Compute labels with optimized config
            labels = self._label_optimizer._compute_labels(ohlcv_df, label_result.best_config)

            if self.verbose >= 1:
                print(f"\n  Label optimization complete: {label_result.n_trials} trials")
                print(f"  Best config: {label_result.best_config}")

        elif labels is None:
            # Use default label config
            default_config = TripleBarrierConfig()
            result.final_label_config = default_config

            # Initialize label optimizer just for computing labels
            self._label_optimizer = LabelOptimizer(
                random_state=self.random_state,
                verbose=0,
            )
            labels = self._label_optimizer._compute_labels(ohlcv_df, default_config)

        # Filter valid samples
        valid_mask = labels != -99
        X_all = feature_df.values[valid_mask]
        y_all = labels[valid_mask]
        feature_names = list(feature_df.columns)

        if self.verbose >= 1:
            print(f"\n  Valid samples: {len(y_all)}/{len(labels)}")
            print(f"  Features: {len(feature_names)}")

        # Split data for optimization (use first model for feature/label optimization)
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            X_all,
            y_all,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y_all,
        )

        # Get a representative model factory for feature optimization
        representative_model = models[0] if models else "xgboost"
        representative_factory = model_factories.get(
            representative_model, list(model_factories.values())[0] if model_factories else None
        )

        # =====================================================================
        # STAGE 2: Feature Selection
        # =====================================================================
        selected_features = feature_names.copy()
        feature_mask = np.ones(len(feature_names), dtype=bool)

        if optimize_features and representative_factory:
            stage_num += 1
            self._print_stage_header("Feature Selection", stage_num, total_stages)

            self._feature_optimizer = FeatureOptimizer(
                selection_trials=self.feature_trials,
                pruning_trials=self.pruning_trials,
                min_features=self.min_features,
                scoring=self.scoring,
                random_state=self.random_state,
                verbose=self.verbose,
            )

            selection_result = self._feature_optimizer.optimize_selection(
                X_train=X_train,
                y_train=y_train,
                feature_names=feature_names,
                model_factory=representative_factory,
                X_val=X_val,
                y_val=y_val,
            )

            result.selection_result = selection_result
            total_trials += selection_result.n_trials
            stages_run.append("feature_selection")

            # Update selected features
            selected_features = selection_result.selected_features
            feature_mask = selection_result.selection_mask

            if self.verbose >= 1:
                print(f"\n  Feature selection complete: {selection_result.n_trials} trials")
                print(f"  Selected: {len(selected_features)}/{len(feature_names)} features")

            # =====================================================================
            # STAGE 3: Feature Pruning
            # =====================================================================
            stage_num += 1
            self._print_stage_header("Feature Pruning", stage_num, total_stages)

            # Get indices of selected features
            selected_indices = np.where(feature_mask)[0]
            X_train_selected = X_train[:, selected_indices]
            X_val_selected = X_val[:, selected_indices]

            pruning_result = self._feature_optimizer.optimize_pruning(
                X_train=X_train_selected,
                y_train=y_train,
                feature_names=selected_features,
                model_factory=representative_factory,
                X_val=X_val_selected,
                y_val=y_val,
            )

            result.pruning_result = pruning_result
            total_trials += pruning_result.n_trials
            stages_run.append("feature_pruning")

            # Update final features
            selected_features = pruning_result.selected_features

            if self.verbose >= 1:
                print(f"\n  Feature pruning complete: {pruning_result.n_trials} trials")
                print(f"  Final features: {len(selected_features)}")

        result.final_features = selected_features

        # =====================================================================
        # STAGE 4: Hyperparameter Optimization
        # =====================================================================
        if optimize_hyperparams and model_factories:
            stage_num += 1
            self._print_stage_header("Hyperparameter Optimization", stage_num, total_stages)

            self._hyperparam_optimizer = HyperparameterOptimizer(
                n_trials=self.hyperparam_trials,
                cv_folds=3,
                scoring=self.scoring,
                use_pruning=True,
                random_state=self.random_state,
                verbose=self.verbose,
            )

            # Get final feature indices
            final_feature_indices = [
                feature_names.index(f) for f in selected_features if f in feature_names
            ]
            X_train_final = X_train[:, final_feature_indices]
            X_val_final = X_val[:, final_feature_indices]

            hyperparam_results = {}
            final_hyperparams = {}

            for i, model_name in enumerate(models):
                if model_name not in model_factories:
                    logger.warning(f"No factory for {model_name}, skipping")
                    continue

                if self.verbose >= 1:
                    print(f"\n  Optimizing {model_name} ({i+1}/{len(models)})...")

                hp_result = self._hyperparam_optimizer.optimize(
                    model_name=model_name,
                    X=X_train_final,
                    y=y_train,
                    model_factory=model_factories[model_name],
                    X_val=X_val_final,
                    y_val=y_val,
                )

                hyperparam_results[model_name] = hp_result
                final_hyperparams[model_name] = hp_result.best_params
                total_trials += hp_result.n_trials

            result.hyperparam_results = hyperparam_results
            result.final_hyperparams = final_hyperparams
            stages_run.append("hyperparameter_optimization")

            if self.verbose >= 1:
                print("\n  Hyperparameter optimization complete")
                print(f"  Models optimized: {len(hyperparam_results)}")

        # =====================================================================
        # FINALIZE
        # =====================================================================
        optimization_time = time.time() - start_time

        result.total_trials = total_trials
        result.optimization_time_seconds = optimization_time
        result.stages_run = stages_run

        if self.verbose >= 1:
            print(f"\n{'#'*70}")
            print("# OPTIMIZATION PIPELINE COMPLETE")
            print(f"# Total trials: {total_trials}")
            print(f"# Total time: {optimization_time:.1f}s")
            print(f"# Stages: {', '.join(stages_run)}")
            print(f"{'#'*70}\n")

        return result

    def run_label_optimization(
        self,
        ohlcv_df: pd.DataFrame,
        target_distribution: dict[int, float] | None = None,
    ) -> LabelOptimizationResult:
        """
        Run only label optimization.

        Args:
            ohlcv_df: OHLCV DataFrame
            target_distribution: Target class distribution

        Returns:
            LabelOptimizationResult
        """
        self._label_optimizer = LabelOptimizer(
            n_trials=self.label_trials,
            objective="class_balance",
            random_state=self.random_state,
            verbose=self.verbose,
        )

        return self._label_optimizer.optimize(
            ohlcv_df=ohlcv_df,
            target_distribution=target_distribution,
        )

    def run_feature_optimization(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str],
        model_factory: Callable[[dict[str, Any]], Any],
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> tuple[FeatureSelectionResult, FeaturePruningResult]:
        """
        Run only feature optimization (selection + pruning).

        Args:
            X_train: Training features
            y_train: Training labels
            feature_names: Feature names
            model_factory: Model factory function
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Tuple of (FeatureSelectionResult, FeaturePruningResult)
        """
        self._feature_optimizer = FeatureOptimizer(
            selection_trials=self.feature_trials,
            pruning_trials=self.pruning_trials,
            min_features=self.min_features,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        return self._feature_optimizer.optimize(
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names,
            model_factory=model_factory,
            X_val=X_val,
            y_val=y_val,
            method="both",
        )

    def run_hyperparameter_optimization(
        self,
        models: list[str],
        model_factories: dict[str, Callable[[dict[str, Any]], Any]],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, HyperparameterResult]:
        """
        Run only hyperparameter optimization.

        Args:
            models: List of model names
            model_factories: Model factory functions
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Dict mapping model_name -> HyperparameterResult
        """
        self._hyperparam_optimizer = HyperparameterOptimizer(
            n_trials=self.hyperparam_trials,
            cv_folds=3,
            scoring=self.scoring,
            use_pruning=True,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        return self._hyperparam_optimizer.optimize_multiple(
            model_names=models,
            X=X_train,
            y=y_train,
            model_factories=model_factories,
            X_val=X_val,
            y_val=y_val,
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "FullOptimizationResult",
    "OptimizationPipeline",
]
