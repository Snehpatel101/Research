"""
Pipeline Lineage Tracking.

Canonical implementation for tracking dataset provenance and pipeline configuration.
This module provides data structures and utilities for:
- Recording pipeline run metadata and configuration
- Computing and validating dataset checksums
- Ensuring reproducibility through lineage tracking

Usage:
    from src.core import (
        PipelineLineage,
        DatasetChecksum,
        compute_dataframe_checksum,
        compute_file_checksum,
        create_dataset_checksum,
        validate_dataset_checksum,
    )
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class DatasetChecksum:
    """Checksum information for a dataset file.

    Attributes:
        file_path: Path to the dataset file.
        checksum: SHA256 checksum (truncated to 16 chars).
        n_rows: Number of rows in the dataset.
        n_cols: Number of columns in the dataset.
        columns: List of column names.
    """

    file_path: str
    checksum: str
    n_rows: int
    n_cols: int
    columns: list[str]


@dataclass
class PipelineLineage:
    """Lineage information for a pipeline run.

    Tracks all configuration and metadata for reproducibility.

    Attributes:
        pipeline_run_id: Unique identifier for this pipeline run.
        target_timeframe: Primary timeframe for the pipeline.
        output_timeframes: List of all output timeframes.
        symbols: List of symbols processed.
        feature_generation: Feature generation mode ('full', 'minimal', etc.).
        label_horizons: List of label horizon values.
        train_ratio: Training set ratio.
        val_ratio: Validation set ratio.
        test_ratio: Test set ratio.
        purge_bars: Number of bars to purge between sets.
        embargo_bars: Number of bars to embargo.
        random_seed: Random seed for reproducibility.
        dataset_checksums: Dictionary of dataset checksums by name.
        created_at: ISO timestamp of when lineage was created.
    """

    pipeline_run_id: str
    target_timeframe: str
    output_timeframes: list[str]
    symbols: list[str]
    feature_generation: str
    label_horizons: list[int]
    train_ratio: float
    val_ratio: float
    test_ratio: float
    purge_bars: int
    embargo_bars: int
    random_seed: int
    dataset_checksums: dict[str, DatasetChecksum]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """Convert lineage to a dictionary for serialization."""
        return {
            "pipeline_run_id": self.pipeline_run_id,
            "target_timeframe": self.target_timeframe,
            "output_timeframes": self.output_timeframes,
            "symbols": self.symbols,
            "feature_generation": self.feature_generation,
            "label_horizons": self.label_horizons,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "purge_bars": self.purge_bars,
            "embargo_bars": self.embargo_bars,
            "random_seed": self.random_seed,
            "dataset_checksums": {
                name: {
                    "file_path": cs.file_path,
                    "checksum": cs.checksum,
                    "n_rows": cs.n_rows,
                    "n_cols": cs.n_cols,
                    "columns": cs.columns,
                }
                for name, cs in self.dataset_checksums.items()
            },
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineLineage":
        """Create a PipelineLineage from a dictionary."""
        dataset_checksums = {}
        for name, cs_data in data.get("dataset_checksums", {}).items():
            dataset_checksums[name] = DatasetChecksum(
                file_path=cs_data["file_path"],
                checksum=cs_data["checksum"],
                n_rows=cs_data["n_rows"],
                n_cols=cs_data["n_cols"],
                columns=cs_data["columns"],
            )

        return cls(
            pipeline_run_id=data["pipeline_run_id"],
            target_timeframe=data["target_timeframe"],
            output_timeframes=data.get("output_timeframes", [data["target_timeframe"]]),
            symbols=data["symbols"],
            feature_generation=data.get("feature_generation", "full"),
            label_horizons=data["label_horizons"],
            train_ratio=data["train_ratio"],
            val_ratio=data["val_ratio"],
            test_ratio=data["test_ratio"],
            purge_bars=data["purge_bars"],
            embargo_bars=data["embargo_bars"],
            random_seed=data["random_seed"],
            dataset_checksums=dataset_checksums,
            created_at=data["created_at"],
        )

    def save(self, path: Path) -> None:
        """Save lineage to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "PipelineLineage":
        """Load lineage from a JSON file."""
        with path.open("r") as f:
            data = json.load(f)
        return cls.from_dict(data)


def compute_dataframe_checksum(df: pd.DataFrame) -> str:
    """Compute SHA256 checksum of a DataFrame.

    Args:
        df: DataFrame to checksum.

    Returns:
        Truncated (16 char) SHA256 hex digest.
    """
    hasher = hashlib.sha256()
    hasher.update(pd.util.hash_pandas_object(df).values.tobytes())
    return hasher.hexdigest()[:16]


def compute_file_checksum(path: Path, sample_size: int = 10000) -> str:
    """Compute checksum of a file.

    For large parquet/csv files, samples rows for efficiency.

    Args:
        path: Path to the file.
        sample_size: Maximum rows to sample for parquet/csv files.

    Returns:
        Truncated (16 char) SHA256 hex digest.
    """
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        return compute_dataframe_checksum(df)
    elif path.suffix == ".csv":
        df = pd.read_csv(path, nrows=sample_size)
        return compute_dataframe_checksum(df)
    else:
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]


def create_dataset_checksum(path: Path, name: str) -> DatasetChecksum:
    """Create a DatasetChecksum for a file.

    Args:
        path: Path to the dataset file.
        name: Name identifier for the dataset.

    Returns:
        DatasetChecksum with file metadata and checksum.

    Raises:
        ValueError: If file format is not supported.
    """
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return DatasetChecksum(
        file_path=str(path),
        checksum=compute_file_checksum(path),
        n_rows=len(df),
        n_cols=len(df.columns),
        columns=list(df.columns),
    )


def validate_dataset_checksum(
    current_path: Path,
    expected_checksum: DatasetChecksum,
    strict: bool = False,
) -> tuple[bool, list[str]]:
    """Validate a dataset against an expected checksum.

    Args:
        current_path: Path to the current dataset file.
        expected_checksum: Expected checksum to validate against.
        strict: If True, also validates exact checksum match.

    Returns:
        Tuple of (is_valid, list_of_issues).
    """
    issues = []

    if not current_path.exists():
        issues.append(f"File not found: {current_path}")
        return False, issues

    if current_path.suffix == ".parquet":
        df = pd.read_parquet(current_path)
    elif current_path.suffix == ".csv":
        df = pd.read_csv(current_path)
    else:
        issues.append(f"Unsupported file format: {current_path.suffix}")
        return False, issues

    if len(df) != expected_checksum.n_rows:
        issues.append(f"Row count mismatch: expected {expected_checksum.n_rows}, got {len(df)}")

    if len(df.columns) != expected_checksum.n_cols:
        issues.append(
            f"Column count mismatch: expected {expected_checksum.n_cols}, got {len(df.columns)}"
        )

    missing_cols = set(expected_checksum.columns) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")

    extra_cols = set(df.columns) - set(expected_checksum.columns)
    if extra_cols:
        issues.append(f"Extra columns: {extra_cols}")

    if strict:
        current_checksum = compute_file_checksum(current_path)
        if current_checksum != expected_checksum.checksum:
            issues.append(
                f"Checksum mismatch: expected {expected_checksum.checksum}, got {current_checksum}"
            )

    return len(issues) == 0, issues


__all__ = [
    "DatasetChecksum",
    "PipelineLineage",
    "compute_dataframe_checksum",
    "compute_file_checksum",
    "create_dataset_checksum",
    "validate_dataset_checksum",
]
