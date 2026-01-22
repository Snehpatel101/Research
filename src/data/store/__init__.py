"""
Feature Store module for managing feature storage, versioning, and lineage.

This module provides a unified interface for:
1. **Feature Caching**: Parquet-based storage with content-addressable keys
2. **Versioning**: Semantic versioning (major.minor.patch) for feature definitions
3. **Lineage Tracking**: Full audit trail from raw data to computed features
4. **Point-in-Time Retrieval**: Backtesting-safe feature queries
5. **Integrity Validation**: SHA256 checksums for data integrity

Main Components
---------------
FeatureStore : class
    Main entry point for feature storage operations
FeatureCache : class
    Low-level Parquet caching with hash-based invalidation
VersionManager : class
    Semantic version management for feature sets
LineageTracker : class
    Lineage tracking and querying

Quick Start
-----------
>>> from src.data.store import FeatureStore
>>>
>>> # Initialize store
>>> store = FeatureStore(cache_dir="data/feature_cache")
>>>
>>> # Check if features exist
>>> if store.has_features(symbol="MES", feature_set="core_full"):
...     df = store.get_features(symbol="MES", feature_set="core_full")
... else:
...     df = compute_features(...)  # your pipeline
...     store.put_features(
...         df,
...         symbol="MES",
...         feature_set="core_full",
...         lineage={"raw_path": "data/raw/MES_1m.parquet", "config": {...}}
...     )
>>>
>>> # Point-in-time retrieval for backtesting
>>> from datetime import datetime
>>> df = store.get_features_as_of(
...     symbol="MES",
...     feature_set="core_full",
...     as_of_date=datetime(2024, 6, 1)
... )

Integration with Pipeline
-------------------------
The feature store integrates with the existing feature engineering pipeline:

>>> from src.pipeline.stages.features import FeatureEngineer
>>> from src.data.store import FeatureStore
>>>
>>> store = FeatureStore(cache_dir="data/feature_cache")
>>>
>>> # Check cache before computing
>>> if not store.has_features(symbol="MES", feature_set="core_full", version="1.0.0"):
...     # Run feature engineering
...     engineer = FeatureEngineer(input_dir="data/clean", output_dir="data/features")
...     df_features, report = engineer.engineer_features(df, "MES")
...
...     # Cache computed features
...     store.put_features(
...         df_features,
...         symbol="MES",
...         feature_set="core_full",
...         lineage={
...             "raw_path": "data/clean/MES_5min_clean.parquet",
...             "config": {
...                 "enable_wavelets": True,
...                 "enable_mtf": True,
...                 "mtf_timeframes": ["15min", "60min"],
...             },
...             "transformations": [
...                 {"type": "feature_engineering", "name": "core_indicators"}
...             ]
...         }
...     )
"""

# Main store class
from .store import (
    FeatureIntegrityError,
    FeatureNotFoundError,
    FeatureStore,
    FeatureStoreError,
)

# Cache components
from .cache import (
    CacheMetadata,
    FeatureCache,
    compute_dataframe_checksum,
    compute_file_checksum,
)

# Lineage tracking
from .lineage import (
    DataSource,
    FeatureLineage,
    LineageTracker,
    Transformation,
    TransformationType,
    compute_file_checksum as compute_lineage_checksum,
)

# Version management
from .versioning import (
    SemanticVersion,
    VersionInfo,
    VersionManager,
    compute_config_hash,
    compute_schema_hash,
)

__all__ = [
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
