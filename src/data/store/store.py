"""
Main FeatureStore class for managing feature storage, retrieval, and versioning.

The FeatureStore provides:
1. Parquet-based feature caching with content-addressable storage
2. Point-in-time feature retrieval for backtesting
3. Feature lineage tracking for auditability
4. Semantic versioning for feature definitions
5. SHA256 checksums for data integrity

Usage
-----
>>> store = FeatureStore(cache_dir="data/feature_cache")
>>>
>>> # Check if features exist
>>> if store.has_features(symbol="MES", feature_set="core_full", version="1.0.0"):
...     df = store.get_features(symbol="MES", feature_set="core_full")
... else:
...     df = compute_features(...)  # existing pipeline
...     store.put_features(
...         df,
...         symbol="MES",
...         feature_set="core_full",
...         lineage={"raw_path": "...", "config": {...}}
...     )
>>>
>>> # Point-in-time retrieval
>>> df = store.get_features_as_of(
...     symbol="MES",
...     feature_set="core_full",
...     as_of_date=datetime(2024, 1, 15),
... )
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.core.exceptions import FeatureIntegrityError, FeatureNotFoundError

from .cache import CacheMetadata, FeatureCache, compute_dataframe_checksum, compute_file_checksum
from .lineage import (
    FeatureLineage,
    LineageTracker,
    TransformationType,
)
from .versioning import (
    SemanticVersion,
    VersionInfo,
    VersionManager,
    compute_config_hash,
    compute_schema_hash,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Unified feature storage with versioning, lineage, and caching.

    The FeatureStore manages the complete lifecycle of computed features:
    - Storage in Parquet format with compression
    - Content-addressable caching for efficient retrieval
    - Semantic versioning for feature definitions
    - Lineage tracking for auditability
    - Point-in-time queries for backtesting

    Parameters
    ----------
    cache_dir : Path or str
        Root directory for feature storage
    lineage_dir : Path or str, optional
        Directory for lineage data. Defaults to cache_dir/lineage
    validate_on_read : bool, default True
        Whether to validate checksums when reading features
    compression : str, default "snappy"
        Parquet compression codec

    Examples
    --------
    >>> store = FeatureStore(cache_dir="data/feature_cache")
    >>>
    >>> # Store computed features
    >>> store.put_features(
    ...     df=features_df,
    ...     symbol="MES",
    ...     feature_set="core_full",
    ...     lineage={
    ...         "raw_path": "data/raw/MES_1m.parquet",
    ...         "config": {"enable_wavelets": True, "mtf_timeframes": ["15min", "60min"]}
    ...     }
    ... )
    >>>
    >>> # Retrieve features
    >>> df = store.get_features(symbol="MES", feature_set="core_full")
    >>>
    >>> # Point-in-time retrieval
    >>> df = store.get_features_as_of(
    ...     symbol="MES",
    ...     feature_set="core_full",
    ...     as_of_date=datetime(2024, 6, 1)
    ... )
    """

    def __init__(
        self,
        cache_dir: Path | str,
        lineage_dir: Path | str | None = None,
        validate_on_read: bool = True,
        compression: str = "snappy",
    ) -> None:
        """
        Initialize the FeatureStore.

        Parameters
        ----------
        cache_dir : Path or str
            Root directory for feature storage
        lineage_dir : Path or str, optional
            Directory for lineage data
        validate_on_read : bool, default True
            Validate checksums on read
        compression : str, default "snappy"
            Parquet compression codec
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sub-directories
        if lineage_dir is None:
            self.lineage_dir = self.cache_dir / "lineage"
        else:
            self.lineage_dir = Path(lineage_dir)
        self.lineage_dir.mkdir(parents=True, exist_ok=True)

        self.version_dir = self.cache_dir / "versions"
        self.version_dir.mkdir(parents=True, exist_ok=True)

        self.validate_on_read = validate_on_read
        self.compression = compression

        # Initialize components
        self._cache = FeatureCache(
            cache_dir=self.cache_dir / "data",
            compression=compression,
            validate_on_read=validate_on_read,
        )
        self._lineage_tracker = LineageTracker(storage_path=self.lineage_dir)
        self._version_managers: dict[str, VersionManager] = {}

        # Load existing version managers
        self._load_version_managers()

        logger.info(f"Initialized FeatureStore at {self.cache_dir}")

    def _load_version_managers(self) -> None:
        """Load existing version managers from storage."""
        for version_file in self.version_dir.glob("*.json"):
            try:
                with open(version_file) as f:
                    data = json.load(f)
                manager = VersionManager.from_dict(data)
                self._version_managers[manager.feature_set] = manager
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load version manager from {version_file}: {e}")

        logger.debug(f"Loaded {len(self._version_managers)} version managers")

    def _get_version_manager(self, feature_set: str) -> VersionManager:
        """Get or create version manager for feature set."""
        if feature_set not in self._version_managers:
            self._version_managers[feature_set] = VersionManager(feature_set)
        return self._version_managers[feature_set]

    def _save_version_manager(self, feature_set: str) -> None:
        """Save version manager to storage."""
        manager = self._version_managers.get(feature_set)
        if manager is None:
            return

        version_file = self.version_dir / f"{feature_set}.json"
        with open(version_file, "w") as f:
            json.dump(manager.to_dict(), f, indent=2, default=str)

    def has_features(
        self,
        symbol: str,
        feature_set: str,
        version: str | None = None,
    ) -> bool:
        """
        Check if features exist in the store.

        Parameters
        ----------
        symbol : str
            Symbol name (e.g., "MES", "MGC")
        feature_set : str
            Feature set name (e.g., "core_full", "mtf_enriched")
        version : str, optional
            Feature version. If None, checks for latest version.

        Returns
        -------
        bool
            True if features exist
        """
        # Get version to check
        if version is None:
            manager = self._get_version_manager(feature_set)
            latest = manager.latest_version
            if latest is None:
                return False
            version = str(latest)

        # Check cache for any entry matching feature_set/symbol/version
        entries = self._cache.list_entries(feature_set)
        for entry in entries:
            if entry.symbol == symbol and entry.version == version:
                return True

        return False

    def get_features(
        self,
        symbol: str,
        feature_set: str,
        version: str | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get features from the store.

        Parameters
        ----------
        symbol : str
            Symbol name
        feature_set : str
            Feature set name
        version : str, optional
            Feature version. If None, returns latest version.
        columns : list[str], optional
            Specific columns to load

        Returns
        -------
        DataFrame
            Feature data

        Raises
        ------
        FeatureNotFoundError
            If features are not found
        FeatureIntegrityError
            If checksum validation fails
        """
        # Get version
        if version is None:
            manager = self._get_version_manager(feature_set)
            latest = manager.latest_version
            if latest is None:
                raise FeatureNotFoundError(f"No versions found for feature set '{feature_set}'")
            version = str(latest)

        # Find cache entry
        entries = self._cache.list_entries(feature_set)
        matching_entry = None
        for entry in entries:
            if entry.symbol == symbol and entry.version == version:
                matching_entry = entry
                break

        if matching_entry is None:
            raise FeatureNotFoundError(f"Features not found: {feature_set}/{symbol}/{version}")

        # Get from cache
        try:
            result = self._cache.get(
                feature_set=feature_set,
                cache_key=matching_entry.cache_key,
                columns=columns,
            )
        except ValueError as e:
            raise FeatureIntegrityError(str(e)) from e

        if result is None:
            raise FeatureNotFoundError(f"Failed to load features: {feature_set}/{symbol}/{version}")

        df, metadata = result
        logger.info(
            f"Retrieved features: {feature_set}/{symbol}/{version} "
            f"({len(df)} rows, {len(df.columns)} columns)"
        )
        return df

    def get_features_with_metadata(
        self,
        symbol: str,
        feature_set: str,
        version: str | None = None,
        columns: list[str] | None = None,
    ) -> tuple[pd.DataFrame, CacheMetadata]:
        """
        Get features along with metadata.

        Parameters
        ----------
        symbol : str
            Symbol name
        feature_set : str
            Feature set name
        version : str, optional
            Feature version
        columns : list[str], optional
            Specific columns to load

        Returns
        -------
        tuple[DataFrame, CacheMetadata]
            Feature data and metadata
        """
        # Get version
        if version is None:
            manager = self._get_version_manager(feature_set)
            latest = manager.latest_version
            if latest is None:
                raise FeatureNotFoundError(f"No versions found for feature set '{feature_set}'")
            version = str(latest)

        # Find cache entry
        entries = self._cache.list_entries(feature_set)
        for entry in entries:
            if entry.symbol == symbol and entry.version == version:
                result = self._cache.get(
                    feature_set=feature_set,
                    cache_key=entry.cache_key,
                    columns=columns,
                )
                if result is not None:
                    return result

        raise FeatureNotFoundError(f"Features not found: {feature_set}/{symbol}/{version}")

    def get_features_as_of(
        self,
        symbol: str,
        feature_set: str,
        as_of_date: datetime,
        version: str | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get features with point-in-time filtering.

        Retrieves features and filters to only include data available
        as of the specified date. This is critical for backtesting to
        avoid look-ahead bias.

        Parameters
        ----------
        symbol : str
            Symbol name
        feature_set : str
            Feature set name
        as_of_date : datetime
            Point-in-time date - only returns data before this date
        version : str, optional
            Feature version
        columns : list[str], optional
            Specific columns to load

        Returns
        -------
        DataFrame
            Feature data filtered to as_of_date

        Raises
        ------
        FeatureNotFoundError
            If features are not found
        ValueError
            If no datetime column exists for filtering
        """
        df = self.get_features(
            symbol=symbol,
            feature_set=feature_set,
            version=version,
            columns=columns,
        )

        # Find datetime column
        datetime_col = None
        for col in ["datetime", "timestamp", "date", "time"]:
            if col in df.columns:
                datetime_col = col
                break

        if datetime_col is None:
            raise ValueError(
                f"No datetime column found in features. " f"Available columns: {list(df.columns)}"
            )

        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            df[datetime_col] = pd.to_datetime(df[datetime_col])

        # Filter to as_of_date
        mask = df[datetime_col] < as_of_date
        filtered_df = df[mask].copy()

        logger.info(
            f"Point-in-time retrieval: {feature_set}/{symbol} as of {as_of_date} "
            f"({len(filtered_df)}/{len(df)} rows)"
        )

        return filtered_df

    def put_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        feature_set: str,
        lineage: dict[str, Any] | None = None,
        version: str | None = None,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Store features in the feature store.

        Parameters
        ----------
        df : DataFrame
            Feature data to store
        symbol : str
            Symbol name
        feature_set : str
            Feature set name
        lineage : dict, optional
            Lineage information including:
            - raw_path: Path to raw data source
            - config: Feature engineering configuration
            - transformations: List of transformation steps
        version : str, optional
            Feature version. If None, auto-increments based on changes.
        description : str, optional
            Description of this version
        metadata : dict, optional
            Additional metadata

        Returns
        -------
        str
            Version string of stored features
        """
        if df.empty:
            raise ValueError("Cannot store empty DataFrame")

        # Validate required columns
        datetime_col = None
        for col in ["datetime", "timestamp", "date", "time"]:
            if col in df.columns:
                datetime_col = col
                break

        if datetime_col is None:
            logger.warning("No datetime column found. Point-in-time queries will not work.")

        # Compute hashes for versioning
        schema_hash = compute_schema_hash(
            columns=list(df.columns),
            dtypes={col: str(df[col].dtype) for col in df.columns},
        )
        config = lineage.get("config", {}) if lineage else {}
        config_hash = compute_config_hash(config)

        # Determine version
        manager = self._get_version_manager(feature_set)
        if version is None:
            suggested_version = manager.suggest_version(
                current_schema_hash=schema_hash,
                current_config_hash=config_hash,
            )
            version = str(suggested_version)
        else:
            suggested_version = SemanticVersion.parse(version)

        # Compute input checksum from raw data path or dataframe
        if lineage and "raw_path" in lineage:
            raw_path = lineage["raw_path"]
            if Path(raw_path).exists():
                input_checksum = compute_file_checksum(raw_path)
            else:
                input_checksum = compute_dataframe_checksum(df)
        else:
            input_checksum = compute_dataframe_checksum(df)

        # Get date range
        if datetime_col:
            date_col = df[datetime_col]
            if not pd.api.types.is_datetime64_any_dtype(date_col):
                date_col = pd.to_datetime(date_col)
            date_start = date_col.min()
            date_end = date_col.max()
        else:
            date_start = datetime.now()
            date_end = datetime.now()

        # Store in cache
        cache_metadata = self._cache.put(
            df=df,
            feature_set=feature_set,
            version=version,
            symbol=symbol,
            input_checksum=input_checksum,
            config_hash=config_hash,
            date_start=date_start,
            date_end=date_end,
            metadata=metadata,
        )

        # Register version
        if suggested_version not in manager:
            manager.register_version(
                version=suggested_version,
                description=description,
                schema_hash=schema_hash,
                config_hash=config_hash,
                metadata={
                    "column_count": len(df.columns),
                    "sample_columns": list(df.columns)[:20],
                },
            )
            self._save_version_manager(feature_set)

        # Track lineage
        if lineage:
            self._track_lineage(
                df=df,
                symbol=symbol,
                feature_set=feature_set,
                version=version,
                lineage=lineage,
                cache_metadata=cache_metadata,
            )

        logger.info(
            f"Stored features: {feature_set}/{symbol}/{version} "
            f"({len(df)} rows, {len(df.columns)} columns)"
        )

        return version

    def _track_lineage(
        self,
        df: pd.DataFrame,
        symbol: str,
        feature_set: str,
        version: str,
        lineage: dict[str, Any],
        cache_metadata: CacheMetadata,
    ) -> FeatureLineage:
        """Track lineage for stored features."""
        # Create lineage record
        feature_lineage = self._lineage_tracker.create_lineage(
            feature_set=feature_set,
            version=version,
            symbol=symbol,
            metadata=lineage.get("metadata", {}),
        )

        # Set source if raw_path provided
        raw_path = lineage.get("raw_path")
        if raw_path:
            path = Path(raw_path)
            if path.exists():
                checksum = compute_file_checksum(path)
                # Get date range from datetime column
                datetime_col = None
                for col in ["datetime", "timestamp", "date", "time"]:
                    if col in df.columns:
                        datetime_col = col
                        break

                if datetime_col:
                    date_col = df[datetime_col]
                    if not pd.api.types.is_datetime64_any_dtype(date_col):
                        date_col = pd.to_datetime(date_col)
                    date_start = date_col.min()
                    date_end = date_col.max()
                else:
                    date_start = datetime.now()
                    date_end = datetime.now()

                feature_lineage.set_source(
                    path=str(path),
                    checksum=checksum,
                    symbol=symbol,
                    timeframe=lineage.get("timeframe", "unknown"),
                    row_count=len(df),
                    date_start=date_start,
                    date_end=date_end,
                )

        # Add transformations
        for transform in lineage.get("transformations", []):
            feature_lineage.add_transformation(
                type=TransformationType(transform.get("type", "feature_engineering")),
                name=transform.get("name", "unknown"),
                config=transform.get("config", {}),
                input_columns=transform.get("input_columns", []),
                output_columns=transform.get("output_columns", []),
                duration_seconds=transform.get("duration_seconds", 0.0),
            )

        # Set output info
        feature_lineage.output_columns = list(df.columns)
        feature_lineage.output_checksum = cache_metadata.parquet_checksum

        # Save lineage
        self._lineage_tracker.save_lineage(feature_lineage)

        return feature_lineage

    def get_lineage(
        self,
        symbol: str,
        feature_set: str,
        version: str,
    ) -> FeatureLineage | None:
        """
        Get lineage for a feature set.

        Parameters
        ----------
        symbol : str
            Symbol name
        feature_set : str
            Feature set name
        version : str
            Feature version

        Returns
        -------
        FeatureLineage or None
            Lineage if found
        """
        return self._lineage_tracker.get_lineage_for_feature_set(
            feature_set=feature_set,
            version=version,
            symbol=symbol,
        )

    def get_version_info(
        self,
        feature_set: str,
        version: str | None = None,
    ) -> VersionInfo | None:
        """
        Get version information.

        Parameters
        ----------
        feature_set : str
            Feature set name
        version : str, optional
            Version to get info for. If None, returns latest.

        Returns
        -------
        VersionInfo or None
            Version info if found
        """
        manager = self._get_version_manager(feature_set)
        if version is None:
            latest = manager.latest_version
            if latest is None:
                return None
            version = str(latest)
        return manager.get_version(version)

    def list_versions(self, feature_set: str) -> list[str]:
        """
        List all versions for a feature set.

        Parameters
        ----------
        feature_set : str
            Feature set name

        Returns
        -------
        list[str]
            List of version strings in sorted order
        """
        manager = self._get_version_manager(feature_set)
        return [str(v) for v in manager.all_versions]

    def list_feature_sets(self) -> list[str]:
        """
        List all feature sets in the store.

        Returns
        -------
        list[str]
            List of feature set names
        """
        feature_sets: set[str] = set()

        # From version managers
        feature_sets.update(self._version_managers.keys())

        # From cache entries
        for entry in self._cache.list_entries():
            feature_sets.add(entry.feature_set)

        return sorted(feature_sets)

    def list_symbols(self, feature_set: str) -> list[str]:
        """
        List all symbols for a feature set.

        Parameters
        ----------
        feature_set : str
            Feature set name

        Returns
        -------
        list[str]
            List of symbol names
        """
        symbols = set()
        for entry in self._cache.list_entries(feature_set):
            symbols.add(entry.symbol)
        return sorted(symbols)

    def invalidate(
        self,
        symbol: str,
        feature_set: str,
        version: str | None = None,
    ) -> bool:
        """
        Invalidate cached features.

        Parameters
        ----------
        symbol : str
            Symbol name
        feature_set : str
            Feature set name
        version : str, optional
            Specific version to invalidate. If None, invalidates all versions.

        Returns
        -------
        bool
            True if any entries were invalidated
        """
        invalidated = False
        for entry in self._cache.list_entries(feature_set):
            if entry.symbol != symbol:
                continue
            if version is not None and entry.version != version:
                continue

            if self._cache.invalidate(feature_set, entry.cache_key):
                invalidated = True

        if invalidated:
            logger.info(
                f"Invalidated features: {feature_set}/{symbol}" + (f"/{version}" if version else "")
            )

        return invalidated

    def get_stats(self) -> dict[str, Any]:
        """
        Get feature store statistics.

        Returns
        -------
        dict
            Store statistics including cache stats and version counts
        """
        cache_stats = self._cache.get_cache_stats()

        version_stats = {}
        for feature_set, manager in self._version_managers.items():
            version_stats[feature_set] = {
                "version_count": len(manager),
                "latest_version": str(manager.latest_version) if manager.latest_version else None,
            }

        return {
            "cache": cache_stats,
            "versions": version_stats,
            "feature_sets": self.list_feature_sets(),
            "lineage_count": len(self._lineage_tracker),
        }

    def validate_integrity(
        self,
        symbol: str | None = None,
        feature_set: str | None = None,
    ) -> dict[str, list[str]]:
        """
        Validate integrity of stored features.

        Parameters
        ----------
        symbol : str, optional
            Filter by symbol
        feature_set : str, optional
            Filter by feature set

        Returns
        -------
        dict
            Validation results with 'valid' and 'invalid' lists
        """
        valid = []
        invalid = []

        for entry in self._cache.list_entries(feature_set):
            if symbol and entry.symbol != symbol:
                continue

            entry_key = f"{entry.feature_set}/{entry.symbol}/{entry.version}"

            try:
                # Try to load and validate
                result = self._cache.get(
                    feature_set=entry.feature_set,
                    cache_key=entry.cache_key,
                )
                if result is not None:
                    valid.append(entry_key)
                else:
                    invalid.append(entry_key)
            except (ValueError, Exception) as e:
                logger.warning(f"Integrity check failed for {entry_key}: {e}")
                invalid.append(entry_key)

        return {"valid": valid, "invalid": invalid}

    def cleanup(
        self,
        max_age_days: int = 30,
        feature_set: str | None = None,
    ) -> int:
        """
        Clean up old cache entries.

        Parameters
        ----------
        max_age_days : int, default 30
            Maximum age of entries to keep
        feature_set : str, optional
            Filter by feature set

        Returns
        -------
        int
            Number of entries cleaned up
        """
        return self._cache.cleanup_old_entries(
            max_age_days=max_age_days,
            feature_set=feature_set,
        )
