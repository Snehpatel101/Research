"""Feature manifest for tracking engineered features.

Part of Phase 7D: Feature Manifest System.
Enhanced in Phase 14G: Per-feature computation parameters for reproducibility.

The FeatureManifest provides a persistent record of what feature columns
were produced by feature engineering. This enables:
1. Explicit feature column lists for adapters (vs prefix matching)
2. Validation that expected features are present
3. Reproducibility auditing for ML pipelines
4. Per-feature computation parameters for exact reproduction
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureMetadata:
    """Metadata for a single feature including computation parameters.

    Stores all information needed to reproduce a feature's computation,
    including the parameters used and source columns.

    Attributes:
        name: Feature name (e.g., 'rsi_14')
        category: Feature category (e.g., 'momentum', 'volatility', 'mtf')
        params: Computation parameters (e.g., {'period': 14, 'method': 'sma'})
        source_columns: Input columns used to compute this feature
        created_at: ISO timestamp when feature was registered
        checksum: Hash of params for change detection
    """

    name: str
    category: str
    params: dict[str, Any] = field(default_factory=dict)
    source_columns: list[str] = field(default_factory=list)
    created_at: str = ""
    checksum: str | None = None

    def __post_init__(self) -> None:
        """Set created_at and compute checksum if not provided."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if self.checksum is None and self.params:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute a deterministic hash of the computation parameters.

        Returns:
            SHA-256 hash of the JSON-serialized params
        """
        # Sort keys for deterministic serialization
        params_json = json.dumps(self.params, sort_keys=True, default=str)
        return hashlib.sha256(params_json.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the feature metadata
        """
        return {
            "name": self.name,
            "category": self.category,
            "params": self.params,
            "source_columns": self.source_columns,
            "created_at": self.created_at,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureMetadata:
        """Create FeatureMetadata from dictionary.

        Args:
            data: Dictionary with feature metadata fields

        Returns:
            FeatureMetadata instance
        """
        return cls(
            name=data["name"],
            category=data["category"],
            params=data.get("params", {}),
            source_columns=data.get("source_columns", []),
            created_at=data.get("created_at", ""),
            checksum=data.get("checksum"),
        )


@dataclass
class FeatureManifest:
    """Manifest tracking feature columns produced by feature engineering.

    Enhanced to support per-feature computation parameters for reproducibility.

    Attributes:
        feature_columns: List of feature column names (e.g., 'rsi_14', 'sma_20')
        label_columns: List of label column names (e.g., 'label_h20')
        timestamp_column: Name of timestamp column (default: 'datetime')
        created_at: ISO timestamp of manifest creation
        pipeline_version: Version of the pipeline that created this manifest
        symbol: Trading symbol (e.g., 'MES')
        timeframe: Data timeframe (e.g., '5min')
        metadata: Additional metadata dictionary
        features: Map of feature name to FeatureMetadata with computation params
        computation_config: Global computation settings applied to all features
    """

    feature_columns: list[str] = field(default_factory=list)
    label_columns: list[str] = field(default_factory=list)
    timestamp_column: str = "datetime"
    created_at: str = ""
    pipeline_version: str = "1.0.0"
    symbol: str = ""
    timeframe: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    features: dict[str, FeatureMetadata] = field(default_factory=dict)
    computation_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set created_at if not provided and convert features from dict."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        # Convert dict representations to FeatureMetadata objects
        if self.features:
            converted = {}
            for name, meta in self.features.items():
                if isinstance(meta, dict):
                    converted[name] = FeatureMetadata.from_dict(meta)
                else:
                    converted[name] = meta
            self.features = converted

    def save(self, path: Path | str) -> None:
        """Save manifest to JSON file.

        Args:
            path: Path to save the manifest JSON file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved feature manifest to: {path}")

    @classmethod
    def load(cls, path: Path | str) -> FeatureManifest:
        """Load manifest from JSON file.

        Args:
            path: Path to the manifest JSON file

        Returns:
            FeatureManifest instance

        Raises:
            FileNotFoundError: If manifest file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        logger.debug(f"Loaded feature manifest from: {path}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the manifest
        """
        return {
            "feature_columns": self.feature_columns,
            "label_columns": self.label_columns,
            "timestamp_column": self.timestamp_column,
            "created_at": self.created_at,
            "pipeline_version": self.pipeline_version,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "metadata": self.metadata,
            "features": {name: meta.to_dict() for name, meta in self.features.items()},
            "computation_config": self.computation_config,
        }

    @property
    def n_features(self) -> int:
        """Number of feature columns."""
        return len(self.feature_columns)

    @property
    def n_labels(self) -> int:
        """Number of label columns."""
        return len(self.label_columns)

    def add_feature(
        self,
        name: str,
        category: str,
        params: dict[str, Any],
        source_columns: list[str],
    ) -> None:
        """Register a feature with its computation parameters.

        Args:
            name: Feature name (e.g., 'rsi_14')
            category: Feature category (e.g., 'momentum', 'volatility', 'mtf')
            params: Computation parameters (e.g., {'period': 14, 'method': 'sma'})
            source_columns: Input columns used to compute this feature
        """
        metadata = FeatureMetadata(
            name=name,
            category=category,
            params=params,
            source_columns=source_columns,
        )
        self.features[name] = metadata
        # Also add to feature_columns if not present
        if name not in self.feature_columns:
            self.feature_columns.append(name)
        logger.debug(f"Added feature '{name}' with category '{category}'")

    def get_feature_params(self, name: str) -> dict[str, Any] | None:
        """Get computation parameters for a feature.

        Args:
            name: Feature name to look up

        Returns:
            Computation parameters dict, or None if feature not found
        """
        metadata = self.features.get(name)
        if metadata is None:
            return None
        return metadata.params

    def get_feature_metadata(self, name: str) -> FeatureMetadata | None:
        """Get full metadata for a feature.

        Args:
            name: Feature name to look up

        Returns:
            FeatureMetadata instance, or None if feature not found
        """
        return self.features.get(name)

    def get_features_by_category(self, category: str) -> list[FeatureMetadata]:
        """Get all features in a specific category.

        Args:
            category: Category to filter by (e.g., 'momentum', 'volatility')

        Returns:
            List of FeatureMetadata instances in the category
        """
        return [meta for meta in self.features.values() if meta.category == category]

    def to_reproducibility_record(self) -> dict[str, Any]:
        """Export manifest for reproducibility tracking.

        Creates a comprehensive record containing all information needed
        to exactly reproduce the feature engineering process.

        Returns:
            Dictionary with complete reproducibility information
        """
        return {
            "manifest_version": "2.0.0",  # Version with params support
            "pipeline_version": self.pipeline_version,
            "created_at": self.created_at,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "computation_config": self.computation_config,
            "features": {
                name: {
                    "category": meta.category,
                    "params": meta.params,
                    "source_columns": meta.source_columns,
                    "checksum": meta.checksum,
                }
                for name, meta in self.features.items()
            },
            "feature_count": len(self.feature_columns),
            "label_columns": self.label_columns,
            "timestamp_column": self.timestamp_column,
        }

    def validate_reproducibility(self, other: FeatureManifest) -> tuple[bool, list[str]]:
        """Validate that another manifest can reproduce this one.

        Compares feature checksums to detect parameter changes.

        Args:
            other: Another FeatureManifest to compare against

        Returns:
            Tuple of (is_reproducible, list of differences)
        """
        differences = []

        # Check pipeline versions
        if self.pipeline_version != other.pipeline_version:
            differences.append(
                f"Pipeline version mismatch: {self.pipeline_version} vs "
                f"{other.pipeline_version}"
            )

        # Check computation configs
        if self.computation_config != other.computation_config:
            differences.append("Computation config differs")

        # Check feature checksums
        for name, meta in self.features.items():
            other_meta = other.features.get(name)
            if other_meta is None:
                differences.append(f"Feature '{name}' missing in other manifest")
            elif meta.checksum != other_meta.checksum:
                differences.append(
                    f"Feature '{name}' params differ: checksum "
                    f"{meta.checksum} vs {other_meta.checksum}"
                )

        # Check for extra features in other
        for name in other.features:
            if name not in self.features:
                differences.append(f"Extra feature '{name}' in other manifest")

        return len(differences) == 0, differences

    def validate_dataframe(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """Validate that a DataFrame contains expected columns.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list of missing columns)
        """
        import pandas as pd_module

        if not isinstance(df, pd_module.DataFrame):
            return False, ["Input is not a DataFrame"]

        df_cols = set(df.columns)
        missing_features = [c for c in self.feature_columns if c not in df_cols]
        missing_labels = [c for c in self.label_columns if c not in df_cols]

        issues = []
        if missing_features:
            issues.append(f"Missing features: {missing_features[:10]}")
        if missing_labels:
            issues.append(f"Missing labels: {missing_labels}")
        if self.timestamp_column not in df_cols:
            issues.append(f"Missing timestamp column: {self.timestamp_column}")

        return len(issues) == 0, issues

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        label_prefixes: tuple[str, ...] = ("label_",),
        exclude_prefixes: tuple[str, ...] = (
            "sample_weight_",
            "regime_",
            "timestamp",
            "datetime",
            "symbol",
            "timeframe",
            "split",
        ),
        exclude_exact: set[str] | None = None,
        symbol: str = "",
        timeframe: str = "",
        pipeline_version: str = "1.0.0",
    ) -> FeatureManifest:
        """Create manifest from DataFrame columns.

        Automatically classifies columns as features, labels, or excluded
        based on naming conventions.

        Args:
            df: DataFrame with feature and label columns
            label_prefixes: Column prefixes that indicate label columns
            exclude_prefixes: Column prefixes to exclude from features
            exclude_exact: Exact column names to exclude from features
            symbol: Trading symbol
            timeframe: Data timeframe
            pipeline_version: Pipeline version string

        Returns:
            FeatureManifest instance
        """
        exclude_exact = exclude_exact or {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "bar_index",
            "session_id",
        }

        feature_columns = []
        label_columns = []

        for col in df.columns:
            col_lower = col.lower()

            # Check if it's a label
            if any(col_lower.startswith(prefix) for prefix in label_prefixes):
                label_columns.append(col)
                continue

            # Check if it should be excluded
            if col_lower in exclude_exact:
                continue
            if any(col_lower.startswith(prefix) for prefix in exclude_prefixes):
                continue

            # It's a feature
            feature_columns.append(col)

        return cls(
            feature_columns=feature_columns,
            label_columns=label_columns,
            symbol=symbol,
            timeframe=timeframe,
            pipeline_version=pipeline_version,
        )


__all__ = ["FeatureManifest", "FeatureMetadata"]
