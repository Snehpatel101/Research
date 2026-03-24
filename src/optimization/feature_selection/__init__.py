"""
Feature Selection Package for OHLCV Time-Series ML.

Import paths:
    # Canonical path:
    from src.optimization.feature_selection import WalkForwardFeatureSelector

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

Reference: Lopez de Prado (2018) "Advances in Financial Machine Learning"
"""

# Result classes - these have no external dependencies
# Configuration - these have no external dependencies
from .config import (
    FeatureSelectionConfig,
    FeatureSelectorConfig,
    ModelFamilyDefaults,
)

# Filtering functions
from .filtering import (
    apply_feature_selection,
    build_correlation_groups,
    filter_correlated_features,
    filter_low_variance,
    identify_feature_columns,
    save_feature_selection_report,
    select_features,
    select_from_correlated_group,
)

# OHLCV-specific selectors
from .ohlcv_selector import (
    FEATURE_CATEGORIES,
    OHLCVFeatureSelector,
    StabilityMetrics,
    categorize_feature,
    create_ohlcv_selector,
    filter_ohlcv_features,
    get_feature_categories,
)

# Priority - no external dependencies
from .priority import (
    DEFAULT_PRIORITY,
    FEATURE_PRIORITY,
    get_feature_priority,
)
from .result import (
    FeatureSelectionResult,
    PersistedFeatureSelection,
)

# Walk-forward selectors - minimal external dependencies
from .walk_forward import (
    CVIntegratedFeatureSelector,
    WalkForwardFeatureSelector,
)

# Lazy import for FeatureSelectionManager to avoid circular dependency
# (it imports from src.validation.cv which can trigger src.models imports)
_FeatureSelectionManager = None


def __getattr__(name):
    """Lazy import for classes with complex dependencies."""
    global _FeatureSelectionManager

    if name == "FeatureSelectionManager":
        if _FeatureSelectionManager is None:
            from .manager import FeatureSelectionManager

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
    "StabilityMetrics",
    # OHLCV utilities
    "FEATURE_CATEGORIES",
    "categorize_feature",
    "filter_ohlcv_features",
    "get_feature_categories",
    # Factory functions
    "create_ohlcv_selector",
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
]
