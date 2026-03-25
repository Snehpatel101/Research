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
# =============================================================================
# FEATURE OPTIMIZATION
# =============================================================================
# =============================================================================
# BASE FEATURE SETS (Phase 3)
# =============================================================================
# =============================================================================
# ARTIFACT SAVING (Phase 3)
# =============================================================================
from src.optimization.artifact_saver import (
    DEFAULT_OUTPUT_DIR,
    FEATURE_SPECS_SUBDIR,
    MANIFEST_FILENAME,
    list_runs,
    list_saved_specs,
    load_best_feature_spec,
    load_feature_spec,
    load_manifest,
    save_feature_specs,
    save_optimization_results,
)

# =============================================================================
# 5-DIMENSION OPTUNA OBJECTIVE (Phase 3)
# =============================================================================
from src.optimization.base_feature_sets import (
    BASE_FEATURE_SETS,
    DEFAULT_FEATURE_PARAMS,
    MOMENTUM_FEATURES,
    MOVING_AVERAGE_FEATURES,
    PRICE_FEATURES,
    RAW_FEATURES,
    TREND_FEATURES,
    VOLATILITY_FEATURES,
    VOLUME_FEATURES,
    get_all_features,
    get_base_features,
    get_extensive_feature_set,
    get_feature_category,
    get_feature_params,
    get_features_by_category,
    get_minimal_feature_set,
)

# =============================================================================
# ENSEMBLE-AWARE OBJECTIVE (Phase 16)
# =============================================================================
from src.optimization.ensemble_objective import (
    DEFAULT_DIVERSITY_WEIGHT,
    DEFAULT_MAX_OVERLAP,
    DEFAULT_SHARPE_WEIGHT,
    EnsembleAwareObjective,
    calculate_feature_overlap,
    calculate_feature_overlap_ratio,
    check_feature_diversity,
    compute_feature_diversity_score,
    compute_prediction_diversity,
    diversity_aware_objective,
)
from src.optimization.features import (
    FeatureOptimizer,
    FeaturePruningResult,
    OptunaSelectionResult,
)
from src.optimization.five_dimension_objective import (
    DEFAULT_TIMEFRAMES,
    LOWER_MULT_RANGE,
    MAX_FEATURES_TO_SEARCH,
    MAX_HOLDING_BARS_RANGE,
    MIN_FEATURES,
    UPPER_MULT_RANGE,
    FiveDimensionResult,
    clear_label_cache,
    create_5d_objective,
    generate_labels_for_trial,
    get_best_feature_spec,
    get_top_k_specs,
    run_5d_optimization,
    suggest_hyperparameters_by_family,
)
from src.optimization.hyperparameters import (
    # Search spaces
    HYPERPARAMETER_SPACES,
    # Optimizer
    HyperparameterOptimizer,
    # Result types
    HyperparameterResult,
    get_default_hyperparameters,
    # Utility functions
    suggest_hyperparameters,
)

# =============================================================================
# LABEL OPTIMIZATION
# =============================================================================
from src.optimization.labels import (
    LabelOptimizationResult,
    LabelOptimizer,
    TripleBarrierConfig,
)

# =============================================================================
# UNIFIED PIPELINE
# =============================================================================
from src.optimization.pipeline import (
    FullOptimizationResult,
    OptimizationPipeline,
)
from src.optimization.scoring import get_score_fn, purged_train_val_split

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
    "OptunaSelectionResult",
    "FeaturePruningResult",
    "FeatureOptimizer",
    # Base feature sets (Phase 3)
    "BASE_FEATURE_SETS",
    "DEFAULT_FEATURE_PARAMS",
    "MOMENTUM_FEATURES",
    "TREND_FEATURES",
    "VOLATILITY_FEATURES",
    "VOLUME_FEATURES",
    "RAW_FEATURES",
    "PRICE_FEATURES",
    "MOVING_AVERAGE_FEATURES",
    "get_base_features",
    "get_all_features",
    "get_feature_category",
    "get_features_by_category",
    "get_feature_params",
    "get_minimal_feature_set",
    "get_extensive_feature_set",
    # 5-Dimension Optuna Objective (Phase 3)
    "FiveDimensionResult",
    "create_5d_objective",
    "generate_labels_for_trial",
    "clear_label_cache",
    "run_5d_optimization",
    "get_best_feature_spec",
    "get_top_k_specs",
    "suggest_hyperparameters_by_family",
    "UPPER_MULT_RANGE",
    "LOWER_MULT_RANGE",
    "MAX_HOLDING_BARS_RANGE",
    "MIN_FEATURES",
    "MAX_FEATURES_TO_SEARCH",
    "DEFAULT_TIMEFRAMES",
    # Artifact saving (Phase 3)
    "save_feature_specs",
    "save_optimization_results",
    "load_best_feature_spec",
    "load_feature_spec",
    "load_manifest",
    "list_saved_specs",
    "list_runs",
    "DEFAULT_OUTPUT_DIR",
    "FEATURE_SPECS_SUBDIR",
    "MANIFEST_FILENAME",
    # Full pipeline
    "FullOptimizationResult",
    "OptimizationPipeline",
    # Ensemble-aware objective (Phase 16)
    "diversity_aware_objective",
    "compute_prediction_diversity",
    "calculate_feature_overlap",
    "calculate_feature_overlap_ratio",
    "check_feature_diversity",
    "compute_feature_diversity_score",
    "EnsembleAwareObjective",
    "DEFAULT_SHARPE_WEIGHT",
    "DEFAULT_DIVERSITY_WEIGHT",
    "DEFAULT_MAX_OVERLAP",
    # Scoring & splitting (Phase 22)
    "get_score_fn",
    "purged_train_val_split",
]
