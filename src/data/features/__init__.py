# Feature computation (PHASE_1)
# MTF (Multi-Timeframe) feature computation (PHASE_1)
from .compute import (
    DEFAULT_MTF_FEATURES,
    FEATURE_COMPUTE_MAP,
    MTFConfig,
    MTFFeatureComputer,
    compute_all_features,
    compute_features_by_family,
    compute_features_by_names,
    compute_mtf_features,
    compute_single_feature,
    get_all_feature_names,
    get_features_in_family,
    get_mtf_feature_names,
    resample_ohlcv,
    validate_mtf_config,
)
from .optimization import (
    FeatureOptimizer,
    OptimizationResult,
    optimize_features_for_model,
    suggest_features,
)
from .pruning import (
    FeaturePruner,
    FeaturePruningResult,
    prune_correlated_features,
    prune_features,
)
from .registry import (
    FEATURE_REGISTRY,
    FeatureDefinition,
    get_feature_families,
    get_features_by_families,
    get_features_by_model,
)
from .selection import (
    FeatureSelectionResult,
    FeatureSelector,
    select_features,
)
from .strategies import (
    MODEL_FEATURE_STRATEGIES,
    ModelFeatureStrategy,
    get_baseline_features,
    get_strategy_for_model,
)
from .strategy_manager import (
    FeatureStrategyManager,
    ResolvedFeatureSet,
    get_features_for_model,
)

__all__ = [
    # Registry
    "FeatureDefinition",
    "FEATURE_REGISTRY",
    "get_features_by_families",
    "get_features_by_model",
    "get_feature_families",
    # Compute (PHASE_1)
    "compute_all_features",
    "compute_features_by_family",
    "compute_features_by_names",
    "compute_single_feature",
    "FEATURE_COMPUTE_MAP",
    "get_all_feature_names",
    "get_features_in_family",
    # MTF Compute (PHASE_1)
    "MTFConfig",
    "MTFFeatureComputer",
    "compute_mtf_features",
    "get_mtf_feature_names",
    "validate_mtf_config",
    "resample_ohlcv",
    "DEFAULT_MTF_FEATURES",
    # Strategies
    "ModelFeatureStrategy",
    "MODEL_FEATURE_STRATEGIES",
    "get_strategy_for_model",
    "get_baseline_features",
    # Optimization (legacy)
    "OptimizationResult",
    "FeatureOptimizer",
    "optimize_features_for_model",
    "suggest_features",
    # Strategy Manager
    "ResolvedFeatureSet",
    "FeatureStrategyManager",
    "get_features_for_model",
    # Selection (PHASE_1B)
    "FeatureSelectionResult",
    "FeatureSelector",
    "select_features",
    # Pruning (PHASE_1B)
    "FeaturePruningResult",
    "FeaturePruner",
    "prune_features",
    "prune_correlated_features",
]
