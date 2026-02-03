"""
Canonical Data Contract - Schema for all data flowing through the pipeline.

Every adapter, trainer, and validator uses this contract to ensure
data consistency and traceability.

NOTE: This is the CANONICAL data contract for model data requirements.
Do NOT confuse with:
- OHLCVValidationSchema (src/data/pipeline/stages/validation/data_contract.py):
  OHLCV-specific validation schema for pipeline data
- DatasetContract (src/core/data_contract.py):
  Pipeline stage data passing contract

Phase 0 of the SNwH (Unified Multi-Timeframe Model Factory) implementation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from src.core.exceptions import DataContractViolation
from src.core.types import DataRank


class FeatureMode(str, Enum):
    """Feature generation mode for models."""

    ENGINEERED = "engineered"  # Pre-computed indicators (~180 features)
    RAW = "raw"  # Raw OHLCV only (4-5 features)
    HYBRID = "hybrid"  # Mix of raw + selected indicators
    OOF_PROBS = "oof_probs"  # OOF predictions for meta-learners


class MTFMode(str, Enum):
    """
    Multi-timeframe mode for MODEL CONTRACTS.

    This describes what MTF mode a model expects as input, not what the
    pipeline generates. For pipeline generation modes, see src.config.data.MTFMode.

    Values:
        NONE: Model uses single timeframe only (no MTF features)
        INDICATORS: Model expects MTF indicator features flattened into primary TF
        MULTI_STREAM: Model expects 4D multi-timeframe tensor input
    """

    NONE = "none"  # Single timeframe, no MTF
    INDICATORS = "indicators"  # MTF indicator features added to primary TF
    MULTI_STREAM = "multi_stream"  # Multiple TF streams (4D input)


@dataclass(frozen=True)
class DataContractSchema:
    """
    Schema definition for canonical data.

    This defines the expected columns and their types for data
    at various stages of the pipeline.
    """

    # Required OHLCV columns
    REQUIRED_OHLCV: tuple[str, ...] = (
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
    )

    # Alternative datetime column names
    DATETIME_COLUMNS: tuple[str, ...] = ("timestamp", "datetime", "date", "time")

    # Label column pattern
    LABEL_PATTERN: str = "label_h{horizon}"
    WEIGHT_PATTERN: str = "sample_weight_h{horizon}"
    LABEL_END_TIME_PATTERN: str = "label_end_time_h{horizon}"

    # Metadata columns (not features)
    METADATA_COLUMNS: tuple[str, ...] = (
        "timestamp",
        "datetime",
        "symbol",
        "timeframe",
        "feature_version",
        "label_version",
        "split",
    )

    # Valid label values
    VALID_LABELS: frozenset[int] = frozenset({-1, 0, 1})
    INVALID_LABEL_SENTINEL: int = -99

    def get_label_column(self, horizon: int) -> str:
        """Get label column name for a horizon."""
        return self.LABEL_PATTERN.format(horizon=horizon)

    def get_weight_column(self, horizon: int) -> str:
        """Get sample weight column name for a horizon."""
        return self.WEIGHT_PATTERN.format(horizon=horizon)

    def get_label_end_time_column(self, horizon: int) -> str:
        """Get label end time column name for a horizon."""
        return self.LABEL_END_TIME_PATTERN.format(horizon=horizon)

    def is_metadata_column(self, column: str) -> bool:
        """Check if a column is a metadata column."""
        return column in self.METADATA_COLUMNS or column.startswith("label_")


# Singleton schema instance
DATA_SCHEMA = DataContractSchema()


@dataclass
class DataContract:
    """
    Canonical data contract for MODEL data requirements and lineage tracking.

    This contract captures:
    - Data shape and type information
    - Source lineage (symbol, timeframe, pipeline run)
    - Feature and label metadata
    - Schema hash for validation

    Every data artifact stores this contract alongside the data
    to ensure traceability and compatibility.

    NOTE: This is for MODEL data requirements. For OHLCV validation,
    use OHLCVValidationSchema (src/data/pipeline/stages/validation/data_contract.py).
    """

    # Identity
    symbol: str
    timeframe: str
    horizon: int
    split: str  # "train", "val", "test"

    # Shape
    n_samples: int
    n_features: int
    data_rank: DataRank
    sequence_length: int | None = None  # For 3D/4D data
    n_timeframes: int | None = None  # For 4D data

    # Feature metadata
    feature_mode: FeatureMode = FeatureMode.ENGINEERED
    feature_columns: list[str] = field(default_factory=list)
    feature_version: str = "v1"

    # Label metadata
    label_column: str = ""
    weight_column: str = ""
    label_version: str = "v1"
    n_classes: int = 3  # short, neutral, long

    # Lineage
    pipeline_run_id: str = ""
    source_file: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Schema hash (computed)
    schema_hash: str = ""

    def __post_init__(self):
        """Compute schema hash and set defaults."""
        # Set label column if not provided
        if not self.label_column:
            object.__setattr__(self, "label_column", DATA_SCHEMA.get_label_column(self.horizon))
        if not self.weight_column:
            object.__setattr__(self, "weight_column", DATA_SCHEMA.get_weight_column(self.horizon))
        if not self.schema_hash:
            object.__setattr__(self, "schema_hash", self._compute_schema_hash())

    def _compute_schema_hash(self) -> str:
        """Compute deterministic hash of schema-defining fields."""
        schema_dict = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "horizon": self.horizon,
            "data_rank": self.data_rank.value,
            "n_features": self.n_features,
            "sequence_length": self.sequence_length,
            "n_timeframes": self.n_timeframes,
            "feature_mode": self.feature_mode.value,
            "feature_columns": sorted(self.feature_columns),
            "feature_version": self.feature_version,
            "label_version": self.label_version,
            "n_classes": self.n_classes,
        }
        schema_str = json.dumps(schema_dict, sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]

    def validate_dataframe(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """
        Validate a DataFrame against this contract.

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        # Check sample count
        if len(df) != self.n_samples:
            issues.append(f"Sample count mismatch: expected {self.n_samples}, got {len(df)}")

        # Check feature columns exist
        if self.feature_columns:
            missing_features = set(self.feature_columns) - set(df.columns)
            if missing_features:
                sample = sorted(missing_features)[:10]
                issues.append(f"Missing feature columns: {sample}")

        # Check label column
        if self.label_column and self.label_column not in df.columns:
            issues.append(f"Missing label column: {self.label_column}")

        # Check weight column (warning only)
        if self.weight_column and self.weight_column not in df.columns:
            issues.append(f"Missing weight column: {self.weight_column}")

        return len(issues) == 0, issues

    def validate_dataframe_strict(self, df: pd.DataFrame) -> None:
        """
        Validate DataFrame, raising DataContractViolation on failure.

        This is the preferred method for validation in pipelines where
        invalid data should halt execution.

        Args:
            df: DataFrame to validate

        Raises:
            DataContractViolation: If validation fails with list of issues
        """
        is_valid, issues = self.validate_dataframe(df)
        if not is_valid:
            raise DataContractViolation(issues)

    def validate_array(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[bool, list[str]]:
        """
        Validate numpy arrays against this contract.

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        # Check rank
        if X.ndim != self.data_rank.value:
            issues.append(f"Rank mismatch: expected {self.data_rank.value}D, got {X.ndim}D")
            return False, issues

        # Check shape based on rank
        if self.data_rank == DataRank.TABULAR_2D:
            if X.shape[0] != self.n_samples:
                issues.append(f"Sample count mismatch: expected {self.n_samples}, got {X.shape[0]}")
            if X.shape[1] != self.n_features:
                issues.append(
                    f"Feature count mismatch: expected {self.n_features}, got {X.shape[1]}"
                )

        elif self.data_rank == DataRank.SEQUENCE_3D:
            # (n_samples, seq_len, n_features)
            if self.sequence_length and X.shape[1] != self.sequence_length:
                issues.append(
                    f"Sequence length mismatch: expected {self.sequence_length}, "
                    f"got {X.shape[1]}"
                )
            if X.shape[2] != self.n_features:
                issues.append(
                    f"Feature count mismatch: expected {self.n_features}, got {X.shape[2]}"
                )

        elif self.data_rank == DataRank.MULTI_TF_4D:
            # (n_samples, n_timeframes, seq_len, n_features)
            if self.n_timeframes and X.shape[1] != self.n_timeframes:
                issues.append(
                    f"Timeframe count mismatch: expected {self.n_timeframes}, " f"got {X.shape[1]}"
                )
            if self.sequence_length and X.shape[2] != self.sequence_length:
                issues.append(
                    f"Sequence length mismatch: expected {self.sequence_length}, "
                    f"got {X.shape[2]}"
                )
            if X.shape[3] != self.n_features:
                issues.append(
                    f"Feature count mismatch: expected {self.n_features}, got {X.shape[3]}"
                )

        # Validate y if provided
        if y is not None and y.shape[0] != X.shape[0]:
            issues.append(
                f"Label count mismatch: X has {X.shape[0]} samples, " f"y has {y.shape[0]}"
            )

        return len(issues) == 0, issues

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "horizon": self.horizon,
            "split": self.split,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "data_rank": self.data_rank.value,
            "sequence_length": self.sequence_length,
            "n_timeframes": self.n_timeframes,
            "feature_mode": self.feature_mode.value,
            "feature_columns": self.feature_columns,
            "feature_version": self.feature_version,
            "label_column": self.label_column,
            "weight_column": self.weight_column,
            "label_version": self.label_version,
            "n_classes": self.n_classes,
            "pipeline_run_id": self.pipeline_run_id,
            "source_file": self.source_file,
            "created_at": self.created_at,
            "schema_hash": self.schema_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataContract:
        """Deserialize from dictionary."""
        return cls(
            symbol=data["symbol"],
            timeframe=data["timeframe"],
            horizon=data["horizon"],
            split=data["split"],
            n_samples=data["n_samples"],
            n_features=data["n_features"],
            data_rank=DataRank(data["data_rank"]),
            sequence_length=data.get("sequence_length"),
            n_timeframes=data.get("n_timeframes"),
            feature_mode=FeatureMode(data.get("feature_mode", "engineered")),
            feature_columns=data.get("feature_columns", []),
            feature_version=data.get("feature_version", "v1"),
            label_column=data.get("label_column", ""),
            weight_column=data.get("weight_column", ""),
            label_version=data.get("label_version", "v1"),
            n_classes=data.get("n_classes", 3),
            pipeline_run_id=data.get("pipeline_run_id", ""),
            source_file=data.get("source_file", ""),
            created_at=data.get("created_at", ""),
            schema_hash=data.get("schema_hash", ""),
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        horizon: int,
        split: str,
        feature_columns: list[str],
        pipeline_run_id: str = "",
    ) -> DataContract:
        """Create contract from a DataFrame."""
        return cls(
            symbol=symbol,
            timeframe=timeframe,
            horizon=horizon,
            split=split,
            n_samples=len(df),
            n_features=len(feature_columns),
            data_rank=DataRank.TABULAR_2D,
            feature_columns=feature_columns,
            pipeline_run_id=pipeline_run_id,
        )

    @classmethod
    def from_array(
        cls,
        X: np.ndarray,
        symbol: str,
        timeframe: str,
        horizon: int,
        split: str,
        feature_columns: list[str] | None = None,
        pipeline_run_id: str = "",
    ) -> DataContract:
        """Create contract from numpy array."""
        if X.ndim == 2:
            return cls(
                symbol=symbol,
                timeframe=timeframe,
                horizon=horizon,
                split=split,
                n_samples=X.shape[0],
                n_features=X.shape[1],
                data_rank=DataRank.TABULAR_2D,
                feature_columns=feature_columns or [],
                pipeline_run_id=pipeline_run_id,
            )
        elif X.ndim == 3:
            return cls(
                symbol=symbol,
                timeframe=timeframe,
                horizon=horizon,
                split=split,
                n_samples=X.shape[0],
                n_features=X.shape[2],
                data_rank=DataRank.SEQUENCE_3D,
                sequence_length=X.shape[1],
                feature_columns=feature_columns or [],
                pipeline_run_id=pipeline_run_id,
            )
        elif X.ndim == 4:
            return cls(
                symbol=symbol,
                timeframe=timeframe,
                horizon=horizon,
                split=split,
                n_samples=X.shape[0],
                n_features=X.shape[3],
                data_rank=DataRank.MULTI_TF_4D,
                n_timeframes=X.shape[1],
                sequence_length=X.shape[2],
                feature_columns=feature_columns or [],
                pipeline_run_id=pipeline_run_id,
            )
        else:
            raise ValueError(f"Unsupported array rank: {X.ndim}. Must be 2, 3, or 4.")

    def __repr__(self) -> str:
        """Concise string representation."""
        shape = f"{self.n_samples}x{self.n_features}"
        if self.sequence_length:
            shape = f"{self.n_samples}x{self.sequence_length}x{self.n_features}"
        if self.n_timeframes:
            shape = (
                f"{self.n_samples}x{self.n_timeframes}x" f"{self.sequence_length}x{self.n_features}"
            )
        return (
            f"DataContract({self.symbol}/{self.timeframe}/h{self.horizon}, "
            f"split={self.split}, shape={shape}, hash={self.schema_hash[:8]})"
        )


__all__ = [
    "DataRank",
    "FeatureMode",
    "MTFMode",
    "DataContractSchema",
    "DATA_SCHEMA",
    "DataContract",
    "DataContractViolation",
]
