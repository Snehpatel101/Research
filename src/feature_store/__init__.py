"""
DEPRECATED: Import from 'src.data.store' instead.

This module will be removed in v2.0.0.

Migration:
    # Old (deprecated)
    from src.feature_store import FeatureStore, FeatureCache

    # New (preferred)
    from src.data.store import FeatureStore, FeatureCache
"""

import warnings

_FEATURE_STORE_EXPORTS = [
    # Main store
    "FeatureStore",
    "FeatureStoreError",
    "FeatureNotFoundError",
    "FeatureIntegrityError",
    # Cache
    "FeatureCache",
    "CacheMetadata",
    "compute_file_checksum",
    "compute_dataframe_checksum",
    # Lineage
    "LineageTracker",
    "FeatureLineage",
    "DataSource",
    "Transformation",
    "TransformationType",
    # Versioning
    "SemanticVersion",
    "VersionInfo",
    "VersionManager",
    "compute_schema_hash",
    "compute_config_hash",
]


def __getattr__(name: str):
    """Lazy import to avoid circular imports."""
    if name in _FEATURE_STORE_EXPORTS:
        warnings.warn(
            f"Importing '{name}' from 'src.feature_store' is deprecated. "
            f"Use 'from src.data.store import {name}' instead. "
            "This will be removed in v2.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        from src.data import store as store_module

        return getattr(store_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return _FEATURE_STORE_EXPORTS
