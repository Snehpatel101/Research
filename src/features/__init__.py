"""
DEPRECATED: Import from 'src.data.features' instead.

This module will be removed in v2.0.0.

Migration:
    # Old (deprecated)
    from src.features import compute_all_features, FeatureDefinition

    # New (preferred)
    from src.data.features import compute_all_features, FeatureDefinition
"""

import warnings

_FEATURES_EXPORTS = [
    # Registry
    "FeatureDefinition",
    "FEATURE_REGISTRY",
    "get_features_by_families",
    "get_features_by_model",
    "get_feature_families",
    # Compute
    "compute_all_features",
    "compute_features_by_family",
    "compute_features_by_names",
    "compute_single_feature",
    "FEATURE_COMPUTE_MAP",
    "get_all_feature_names",
    "get_features_in_family",
    # MTF Compute
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
    # Optimization
    "OptimizationResult",
    "FeatureOptimizer",
    "optimize_features_for_model",
    "suggest_features",
    # Strategy Manager
    "ResolvedFeatureSet",
    "FeatureStrategyManager",
    "get_features_for_model",
    # Selection
    "FeatureSelectionResult",
    "FeatureSelector",
    "select_features",
    # Pruning
    "FeaturePruningResult",
    "FeaturePruner",
    "prune_features",
    "prune_correlated_features",
]


def __getattr__(name: str):
    """Lazy import to avoid circular imports."""
    if name in _FEATURES_EXPORTS:
        warnings.warn(
            f"Importing '{name}' from 'src.features' is deprecated. "
            f"Use 'from src.data.features import {name}' instead. "
            "This will be removed in v2.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        from src.data import features as features_module

        return getattr(features_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return _FEATURES_EXPORTS
