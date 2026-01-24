"""Feature manifest for tracking engineered features.

Part of Phase 7D: Feature Manifest System.

The FeatureManifest provides a persistent record of what feature columns
were produced by feature engineering. This enables:
1. Explicit feature column lists for adapters (vs prefix matching)
2. Validation that expected features are present
3. Reproducibility auditing for ML pipelines
"""

from __future__ import annotations

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
class FeatureManifest:
    """Manifest tracking feature columns produced by feature engineering.

    Attributes:
        feature_columns: List of feature column names (e.g., 'rsi_14', 'sma_20')
        label_columns: List of label column names (e.g., 'label_h20')
        timestamp_column: Name of timestamp column (default: 'datetime')
        created_at: ISO timestamp of manifest creation
        pipeline_version: Version of the pipeline that created this manifest
        symbol: Trading symbol (e.g., 'MES')
        timeframe: Data timeframe (e.g., '5min')
        metadata: Additional metadata dictionary
    """

    feature_columns: list[str]
    label_columns: list[str] = field(default_factory=list)
    timestamp_column: str = "datetime"
    created_at: str = ""
    pipeline_version: str = "1.0.0"
    symbol: str = ""
    timeframe: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set created_at if not provided."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

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
        }

    @property
    def n_features(self) -> int:
        """Number of feature columns."""
        return len(self.feature_columns)

    @property
    def n_labels(self) -> int:
        """Number of label columns."""
        return len(self.label_columns)

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


__all__ = ["FeatureManifest"]
