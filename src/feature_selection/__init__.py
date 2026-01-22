"""
DEPRECATED: Import from 'src.optimization.feature_selection' instead.

This module will be removed in v2.0.0.

Migration:
    # Old (deprecated)
    from src.feature_selection import WalkForwardFeatureSelector

    # New (preferred)
    from src.optimization.feature_selection import WalkForwardFeatureSelector
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

# Names that will be lazily loaded from src.optimization.feature_selection
_FEATURE_SELECTION_EXPORTS = [
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


def __getattr__(name: str):
    """Lazy import to avoid circular imports."""
    if name in _FEATURE_SELECTION_EXPORTS:
        # Show deprecation warning
        warnings.warn(
            f"Importing '{name}' from 'src.feature_selection' is deprecated. "
            f"Use 'from src.optimization.feature_selection import {name}' instead. "
            "This will be removed in v2.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Import from new location
        from src.optimization import feature_selection as fs_module

        return getattr(fs_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Return list of available names."""
    return _FEATURE_SELECTION_EXPORTS


__all__ = _FEATURE_SELECTION_EXPORTS
