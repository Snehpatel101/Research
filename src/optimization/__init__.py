"""
Optimization Package - PHASE_1B Unified Optuna Optimization.

This package provides comprehensive hyperparameter optimization for the ML Factory
pipeline, including:

1. Label Optimization - Triple-barrier parameter tuning
2. Feature Selection - Binary include/exclude optimization
3. Feature Pruning - Importance-based feature removal
4. Hyperparameter Optimization - Model-specific parameter tuning

Key Components:
    HyperparameterResult: Result container for hyperparameter optimization
    HyperparameterOptimizer: Optuna-based optimizer for all 23 models
    HYPERPARAMETER_SPACES: Complete search spaces for all models

    FullOptimizationResult: Combined result for full pipeline optimization
    OptimizationPipeline: Unified pipeline orchestrator

Integration:
    All optimizations integrate with PipelineConfig from src.core:
    - optimize_labels, label_optimization_trials
    - optimize_features, feature_selection_trials, feature_pruning_trials
    - optimize_hyperparams, hyperparam_trials
    - optuna_random_state

Usage:
    from src.optimization import (
        # Hyperparameter optimization
        HyperparameterOptimizer,
        HyperparameterResult,
        HYPERPARAMETER_SPACES,

        # Full pipeline
        OptimizationPipeline,
        FullOptimizationResult,

        # Label optimization results
        LabelOptimizationResult,
        TripleBarrierConfig,
    )

    # Single model optimization
    optimizer = HyperparameterOptimizer(n_trials=100)
    result = optimizer.optimize("xgboost", X, y, model_factory)

    # Full pipeline optimization
    pipeline = OptimizationPipeline.from_config(config)
    result = pipeline.run_full_optimization(ohlcv_df, feature_df, models, factories)

Reference: Bergstra et al. (2011) "Algorithms for Hyper-Parameter Optimization"
"""

# =============================================================================
# HYPERPARAMETER OPTIMIZATION
# =============================================================================
from src.optimization.hyperparameters import (
    # Result types
    HyperparameterResult,
    # Search spaces
    HYPERPARAMETER_SPACES,
    # Optimizer
    HyperparameterOptimizer,
    # Utility functions
    suggest_hyperparameters,
    get_default_hyperparameters,
)

# =============================================================================
# LABEL OPTIMIZATION
# =============================================================================
from src.optimization.labels import (
    LabelOptimizationResult,
    TripleBarrierConfig,
    LabelOptimizer,
)

# =============================================================================
# FEATURE OPTIMIZATION
# =============================================================================
from src.optimization.features import (
    FeatureSelectionResult,
    FeaturePruningResult,
    FeatureOptimizer,
)

# =============================================================================
# UNIFIED PIPELINE
# =============================================================================
from src.optimization.pipeline import (
    FullOptimizationResult,
    OptimizationPipeline,
)

# =============================================================================
# EXPORTS
# =============================================================================
__all__ = [
    # Hyperparameter optimization
    "HyperparameterResult",
    "HYPERPARAMETER_SPACES",
    "HyperparameterOptimizer",
    "suggest_hyperparameters",
    "get_default_hyperparameters",
    # Label optimization
    "LabelOptimizationResult",
    "TripleBarrierConfig",
    "LabelOptimizer",
    # Feature optimization
    "FeatureSelectionResult",
    "FeaturePruningResult",
    "FeatureOptimizer",
    # Full pipeline
    "FullOptimizationResult",
    "OptimizationPipeline",
]
