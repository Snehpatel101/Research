"""
Walk-Forward Feature Selection for Time Series.

DEPRECATED: This module has been moved to src.feature_selection.
Please update your imports:

    # Old (deprecated):
    from src.cross_validation.feature_selector import WalkForwardFeatureSelector

    # New (preferred):
    from src.feature_selection import WalkForwardFeatureSelector

This module will be removed in a future version.

Reference: Lopez de Prado (2018) "Advances in Financial Machine Learning", Chapter 8
"""

from __future__ import annotations


def __getattr__(name):
    """Lazy import to avoid circular dependency."""
    import warnings

    warnings.warn(
        "Importing from src.cross_validation.feature_selector is deprecated. "
        "Use src.feature_selection instead. "
        "Example: from src.feature_selection import WalkForwardFeatureSelector",
        DeprecationWarning,
        stacklevel=2,
    )

    # Import from canonical location
    from src.feature_selection import (
        CVIntegratedFeatureSelector,
        FeatureSelectionResult,
        FeatureSelectorConfig,
        WalkForwardFeatureSelector,
    )

    _exports = {
        "FeatureSelectorConfig": FeatureSelectorConfig,
        "FeatureSelectionResult": FeatureSelectionResult,
        "WalkForwardFeatureSelector": WalkForwardFeatureSelector,
        "CVIntegratedFeatureSelector": CVIntegratedFeatureSelector,
    }

    if name in _exports:
        return _exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FeatureSelectorConfig",
    "FeatureSelectionResult",
    "WalkForwardFeatureSelector",
    "CVIntegratedFeatureSelector",
]
