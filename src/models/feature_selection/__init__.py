"""
Feature Selection Integration for Model Training.

DEPRECATED: This module has been moved to src.feature_selection.
Please update your imports:

    # Old (deprecated):
    from src.models.feature_selection import FeatureSelectionManager

    # New (preferred):
    from src.feature_selection import FeatureSelectionManager

This module will be removed in a future version.
"""

from __future__ import annotations


def __getattr__(name):
    """Lazy import to avoid circular dependency."""
    import warnings

    warnings.warn(
        "Importing from src.models.feature_selection is deprecated. "
        "Use src.feature_selection instead. "
        "Example: from src.feature_selection import FeatureSelectionManager",
        DeprecationWarning,
        stacklevel=2,
    )

    # Import from canonical location
    from src.feature_selection import (
        FeatureSelectionConfig,
        FeatureSelectionManager,
        ModelFamilyDefaults,
        PersistedFeatureSelection,
    )

    _exports = {
        "FeatureSelectionConfig": FeatureSelectionConfig,
        "FeatureSelectionManager": FeatureSelectionManager,
        "ModelFamilyDefaults": ModelFamilyDefaults,
        "PersistedFeatureSelection": PersistedFeatureSelection,
    }

    if name in _exports:
        return _exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FeatureSelectionConfig",
    "FeatureSelectionManager",
    "ModelFamilyDefaults",
    "PersistedFeatureSelection",
]
