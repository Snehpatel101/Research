"""
Optimization Feature Selection - Re-export from src.feature_selection.

New import path:
    from src.optimization.feature_selection import WalkForwardFeatureSelector

Legacy import path (still works):
    from src.feature_selection import WalkForwardFeatureSelector
"""

from src.feature_selection import (
    # Result classes
    FeatureSelectionResult,
    PersistedFeatureSelection,
    # Configuration
    FeatureSelectionConfig,
    FeatureSelectorConfig,
    ModelFamilyDefaults,
    # Walk-forward selectors
    WalkForwardFeatureSelector,
    CVIntegratedFeatureSelector,
    # Manager (lazy loaded)
    FeatureSelectionManager,
    # OHLCV selectors
    OHLCVFeatureSelector,
    PurgedFeatureSelector,
    StabilityMetrics,
    # OHLCV utilities
    FEATURE_CATEGORIES,
    categorize_feature,
    filter_ohlcv_features,
    get_feature_categories,
    # Factory functions
    create_ohlcv_selector,
    create_purged_selector,
    # Filtering functions
    apply_feature_selection,
    build_correlation_groups,
    filter_correlated_features,
    filter_low_variance,
    identify_feature_columns,
    save_feature_selection_report,
    select_features,
    select_from_correlated_group,
    # Priority
    DEFAULT_PRIORITY,
    FEATURE_PRIORITY,
    get_feature_priority,
    # Optimization
    FeatureOptimizer,
    optimize_feature_subset_simple,
)

__all__ = [
    "FeatureSelectionResult",
    "PersistedFeatureSelection",
    "FeatureSelectionConfig",
    "FeatureSelectorConfig",
    "ModelFamilyDefaults",
    "WalkForwardFeatureSelector",
    "CVIntegratedFeatureSelector",
    "FeatureSelectionManager",
    "OHLCVFeatureSelector",
    "PurgedFeatureSelector",
    "StabilityMetrics",
    "FEATURE_CATEGORIES",
    "categorize_feature",
    "filter_ohlcv_features",
    "get_feature_categories",
    "create_ohlcv_selector",
    "create_purged_selector",
    "apply_feature_selection",
    "build_correlation_groups",
    "filter_correlated_features",
    "filter_low_variance",
    "identify_feature_columns",
    "save_feature_selection_report",
    "select_features",
    "select_from_correlated_group",
    "DEFAULT_PRIORITY",
    "FEATURE_PRIORITY",
    "get_feature_priority",
    "FeatureOptimizer",
    "optimize_feature_subset_simple",
]
