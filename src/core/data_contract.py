"""
Explicit data contracts for all pipeline stages.

This eliminates implicit assumptions about column names, indices, and metadata.
Every stage should accept and return DatasetContract objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any
import pandas as pd
import numpy as np


@dataclass
class DatasetContract:
    """
    Explicit contract for datasets passed between pipeline stages.

    This is THE standard format for all data transformations.

    Attributes:
        features: Feature DataFrame (columns = feature names)
        labels: Label Series (index must match features)
        indices: Explicit index (for OOF alignment)
        label_end_times: End time of label horizon (for purge calculation)
        split: Which split this belongs to ("train", "val", "test")
        metadata: Arbitrary metadata (symbol, horizon, etc.)
    """
    # Core data
    features: pd.DataFrame
    labels: pd.Series

    # Index tracking (EXPLICIT)
    indices: pd.Index | None = None
    label_end_times: Optional[pd.Series] = None

    # Metadata
    split: str = "train"  # "train", "val", "test"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate contract consistency."""
        # Check feature/label length match
        if len(self.features) != len(self.labels):
            raise ValueError(
                f"Feature/label length mismatch: "
                f"{len(self.features)} != {len(self.labels)}"
            )

        # Check index match
        if not self.features.index.equals(self.labels.index):
            raise ValueError("Feature and label indices must match")

        # Set indices if not provided
        if self.indices is None:
            self.indices = self.features.index

    @property
    def n_samples(self) -> int:
        """Number of samples in dataset."""
        return len(self.features)

    @property
    def n_features(self) -> int:
        """Number of features."""
        return len(self.features.columns)

    @property
    def feature_names(self) -> list[str]:
        """List of feature column names."""
        return list(self.features.columns)

    @property
    def has_label_end_times(self) -> bool:
        """Whether label end times are available."""
        return self.label_end_times is not None

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert to numpy arrays (for model training)."""
        return self.features.values, self.labels.values

    def filter_by_indices(self, indices: pd.Index) -> "DatasetContract":
        """Filter dataset to specific indices (for OOF alignment)."""
        mask = self.indices.isin(indices)

        return DatasetContract(
            features=self.features.loc[mask].copy(),
            labels=self.labels.loc[mask].copy(),
            indices=self.indices[mask],
            label_end_times=self.label_end_times.loc[mask].copy() if self.label_end_times is not None else None,
            split=self.split,
            metadata=self.metadata.copy(),
        )

    def copy(self) -> "DatasetContract":
        """Create a deep copy."""
        return DatasetContract(
            features=self.features.copy(),
            labels=self.labels.copy(),
            indices=self.indices.copy(),
            label_end_times=self.label_end_times.copy() if self.label_end_times is not None else None,
            split=self.split,
            metadata=self.metadata.copy(),
        )

    @classmethod
    def from_arrays(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        split: str = "train",
        **metadata,
    ) -> "DatasetContract":
        """Create from numpy arrays."""
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X.shape[1])]

        features = pd.DataFrame(X, columns=feature_names)
        labels = pd.Series(y, name="label")

        return cls(
            features=features,
            labels=labels,
            split=split,
            metadata=metadata,
        )


@dataclass
class SplitDatasetContract:
    """Contract for train/val/test splits."""
    train: DatasetContract
    val: DatasetContract
    test: DatasetContract | None = None

    @property
    def has_test(self) -> bool:
        return self.test is not None
