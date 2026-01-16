"""
Feature Selection Package for OHLCV Time-Series ML.

This is the canonical location for all feature selection functionality.
The package consolidates code from multiple locations:
- src/cross_validation/feature_selector.py (WalkForwardFeatureSelector, FeatureSelectionResult)
- src/models/feature_selection/ (FeatureSelectionManager, config, PersistedFeatureSelection)
- src/phase1/utils/feature_selection.py (filtering functions, FEATURE_PRIORITY)

Main Components:
    Result Classes:
        FeatureSelectionResult: Canonical result container for all selection operations
        PersistedFeatureSelection: Persisted result for model artifacts

    Configuration:
        FeatureSelectionConfig: Per-model feature selection configuration
        FeatureSelectorConfig: Walk-forward selector configuration
        ModelFamilyDefaults: Default settings per model family

    Selectors:
        WalkForwardFeatureSelector: Walk-forward feature selection with MDA/MDI
        CVIntegratedFeatureSelector: CV-integrated feature selection
        OHLCVFeatureSelector: Enhanced selector with stability, correlation, and regime support
        PurgedFeatureSelector: Selector integrated with PurgedKFold CV
        FeatureSelectionManager: High-level manager for model training integration

    Filtering Functions:
        filter_low_variance: Remove near-constant features
        filter_correlated_features: Remove highly correlated features
        select_features: Main feature selection function
        apply_feature_selection: Apply selection to DataFrame

    Priority & Categories:
        FEATURE_PRIORITY: Feature interpretability rankings
        FEATURE_CATEGORIES: OHLCV feature category patterns
        get_feature_priority: Get priority score for a feature
        categorize_feature: Categorize a feature by name pattern

Key Features:
    - Time-series aware MDA (walk-forward, no lookahead)
    - Stability scoring (ranking consistency across folds)
    - Correlation filtering (hierarchical clustering)
    - Regime-conditional importance (market state aware)
    - OHLCV category filtering (momentum, volatility, volume, etc.)
    - PurgedKFold integration (purge + embargo for label leakage prevention)
    - Model-family-aware configuration (boosting, neural, classical)

Example:
    >>> from src.feature_selection import (
    ...     FeatureSelectionManager,
    ...     WalkForwardFeatureSelector,
    ...     filter_ohlcv_features,
    ...     select_features,
    ... )
    >>>
    >>> # High-level: Use manager for model training integration
    >>> manager = FeatureSelectionManager.from_model_family("boosting")
    >>> result = manager.select_features(X_train_df, y_train, sample_weights)
    >>> X_selected = manager.apply_selection(X_train_df)
    >>>
    >>> # Mid-level: Use walk-forward selector directly
    >>> selector = WalkForwardFeatureSelector(n_features_to_select=50)
    >>> cv_splits = list(cv.split(X, y))
    >>> result = selector.select_features_walkforward(X, y, cv_splits)
    >>>
    >>> # Low-level: Use filtering functions
    >>> result = select_features(df, correlation_threshold=0.85)
    >>> filtered = filter_ohlcv_features(features, exclude_categories=['mtf'])

Reference: Lopez de Prado (2018) "Advances in Financial Machine Learning"
"""

# Result classes - these have no external dependencies
from src.feature_selection.result import (
    FeatureSelectionResult,
    PersistedFeatureSelection,
)

# Configuration - these have no external dependencies
from src.feature_selection.config import (
    FeatureSelectionConfig,
    FeatureSelectorConfig,
    ModelFamilyDefaults,
)

# Priority - no external dependencies
from src.feature_selection.priority import (
    DEFAULT_PRIORITY,
    FEATURE_PRIORITY,
    get_feature_priority,
)

# Walk-forward selectors - minimal external dependencies
from src.feature_selection.walk_forward import (
    CVIntegratedFeatureSelector,
    WalkForwardFeatureSelector,
)

# OHLCV-specific selectors
from src.feature_selection.ohlcv_selector import (
    FEATURE_CATEGORIES,
    OHLCVFeatureSelector,
    StabilityMetrics,
    categorize_feature,
    create_ohlcv_selector,
    filter_ohlcv_features,
    get_feature_categories,
)

# Purged selector
from src.feature_selection.purged_selector import (
    PurgedFeatureSelector,
    create_purged_selector,
)

# Filtering functions
from src.feature_selection.filtering import (
    apply_feature_selection,
    build_correlation_groups,
    filter_correlated_features,
    filter_low_variance,
    identify_feature_columns,
    save_feature_selection_report,
    select_features,
    select_from_correlated_group,
)

# Optimization
from src.feature_selection.optimization import (
    FeatureOptimizer,
    optimize_feature_subset_simple,
)

# Lazy import for FeatureSelectionManager to avoid circular dependency
# (it imports from src.cross_validation which can trigger src.models imports)
_FeatureSelectionManager = None


def __getattr__(name):
    """Lazy import for classes with complex dependencies."""
    global _FeatureSelectionManager

    if name == "FeatureSelectionManager":
        if _FeatureSelectionManager is None:
            from src.feature_selection.manager import FeatureSelectionManager

            _FeatureSelectionManager = FeatureSelectionManager
        return _FeatureSelectionManager

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Result classes
    "FeatureSelectionResult",
    "PersistedFeatureSelection",
    # Configuration
    "FeatureSelectionConfig",
    "FeatureSelectorConfig",
    "ModelFamilyDefaults",
    # Walk-forward selectors
    "WalkForwardFeatureSelector",
    "CVIntegratedFeatureSelector",
    # Manager (lazy loaded)
    "FeatureSelectionManager",
    # OHLCV selectors
    "OHLCVFeatureSelector",
    "PurgedFeatureSelector",
    "StabilityMetrics",
    # OHLCV utilities
    "FEATURE_CATEGORIES",
    "categorize_feature",
    "filter_ohlcv_features",
    "get_feature_categories",
    # Factory functions
    "create_ohlcv_selector",
    "create_purged_selector",
    # Filtering functions
    "apply_feature_selection",
    "build_correlation_groups",
    "filter_correlated_features",
    "filter_low_variance",
    "identify_feature_columns",
    "save_feature_selection_report",
    "select_features",
    "select_from_correlated_group",
    # Priority
    "DEFAULT_PRIORITY",
    "FEATURE_PRIORITY",
    "get_feature_priority",
    # Optimization
    "FeatureOptimizer",
    "optimize_feature_subset_simple",
]
