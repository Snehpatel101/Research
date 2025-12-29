"""
SequenceDataset - PyTorch Dataset for time series sequences.

This module provides a PyTorch Dataset implementation that generates
sliding window sequences for sequence models (LSTM, Transformer, etc.).

Key Features:
- Sliding window generation with configurable seq_len and stride
- Symbol isolation (sequences never cross symbol boundaries)
- Pre-computed indices for efficiency
- Returns (X_seq, y, weight) tuples

Usage:
------
    from src.phase1.stages.datasets.sequences import SequenceDataset

    dataset = SequenceDataset(
        df=train_df,
        feature_columns=["return_1", "rsi_14", ...],
        label_column="label_h20",
        weight_column="sample_weight_h20",
        symbol_column="symbol",
        seq_len=60,
        stride=1
    )

    # Use with DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for X_seq, y, weights in loader:
        # X_seq: (batch_size, seq_len, n_features)
        # y: (batch_size,)
        # weights: (batch_size,)
        pass
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SequenceConfig:
    """Configuration for sequence generation."""
    seq_len: int
    stride: int = 1
    feature_columns: list[str] = None
    label_column: str = "label_h20"
    weight_column: str = "sample_weight_h20"
    symbol_column: str | None = "symbol"

    def __post_init__(self) -> None:
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")
        if self.feature_columns is None:
            self.feature_columns = []


# =============================================================================
# SEQUENCE INDEX BUILDER
# =============================================================================

def build_sequence_indices(
    n_samples: int,
    seq_len: int,
    stride: int,
    symbol_boundaries: list[int] | None = None
) -> np.ndarray:
    """
    Build valid sequence start indices.

    Generates indices where each index i represents a valid sequence
    starting at position i and ending at position i + seq_len - 1.
    The label/target is taken from position i + seq_len - 1.

    If symbol_boundaries is provided, ensures sequences don't cross boundaries.

    Args:
        n_samples: Total number of samples
        seq_len: Sequence length
        stride: Step size between sequences
        symbol_boundaries: Sorted list of indices where symbols change

    Returns:
        Array of valid sequence start indices

    Example:
        >>> # 100 samples, seq_len=10, stride=1
        >>> indices = build_sequence_indices(100, 10, 1)
        >>> len(indices)  # 91 valid sequences (0-90)
        91

        >>> # With symbol boundary at index 50
        >>> indices = build_sequence_indices(100, 10, 1, [50])
        >>> 45 in indices  # Valid: uses samples 45-54
        True
        >>> 46 in indices  # Invalid: would cross boundary at 50
        False
    """
    if n_samples < seq_len:
        return np.array([], dtype=np.int64)

    # Without symbol boundaries, simple range
    if symbol_boundaries is None or len(symbol_boundaries) == 0:
        max_start = n_samples - seq_len
        indices = np.arange(0, max_start + 1, stride)
        return indices

    # With symbol boundaries, build valid ranges per segment
    # Add 0 and n_samples as implicit boundaries
    boundaries = [0] + sorted(symbol_boundaries) + [n_samples]
    valid_indices = []

    for i in range(len(boundaries) - 1):
        seg_start = boundaries[i]
        seg_end = boundaries[i + 1]
        seg_len = seg_end - seg_start

        if seg_len < seq_len:
            continue

        # Valid start indices within this segment
        max_start = seg_end - seq_len
        seg_indices = np.arange(seg_start, max_start + 1, stride)
        valid_indices.append(seg_indices)

    if not valid_indices:
        return np.array([], dtype=np.int64)

    return np.concatenate(valid_indices)


def find_symbol_boundaries(
    df: pd.DataFrame,
    symbol_column: str
) -> list[int]:
    """
    Find indices where symbol changes.

    Args:
        df: DataFrame with symbol column
        symbol_column: Name of the symbol column

    Returns:
        List of indices where symbol changes (sorted)

    Example:
        >>> df = pd.DataFrame({"symbol": ["MES", "MES", "MGC", "MGC", "MES"]})
        >>> find_symbol_boundaries(df, "symbol")
        [2, 4]  # Symbol changes at index 2 (MES->MGC) and 4 (MGC->MES)
    """
    if symbol_column not in df.columns:
        return []

    symbols = df[symbol_column].values
    boundaries = []

    for i in range(1, len(symbols)):
        if symbols[i] != symbols[i - 1]:
            boundaries.append(i)

    return boundaries


# =============================================================================
# SEQUENCE DATASET
# =============================================================================

class SequenceDataset(Dataset):
    """
    PyTorch Dataset for time series sequences.

    Generates sliding window sequences with configurable length and stride.
    Supports symbol isolation to prevent sequences from crossing symbol
    boundaries (e.g., MES data bleeding into MGC sequences).

    The dataset pre-computes valid sequence indices for efficiency,
    then generates sequences on-the-fly during iteration.

    Each sample returns:
        - X_seq: Feature sequence tensor, shape (seq_len, n_features)
        - y: Label scalar tensor
        - weight: Sample weight scalar tensor

    Attributes:
        config: SequenceConfig with generation parameters
        n_samples: Total number of valid sequences
        n_features: Number of features per time step

    Example:
        >>> dataset = SequenceDataset(
        ...     df=train_df,
        ...     feature_columns=["return_1", "rsi_14"],
        ...     label_column="label_h20",
        ...     seq_len=60
        ... )
        >>> X_seq, y, w = dataset[0]
        >>> X_seq.shape
        torch.Size([60, 2])
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        label_column: str,
        seq_len: int,
        weight_column: str | None = None,
        symbol_column: str | None = None,
        stride: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Initialize SequenceDataset.

        Args:
            df: Source DataFrame (should be sorted by datetime within symbols)
            feature_columns: List of feature column names
            label_column: Target/label column name
            seq_len: Sequence length (number of time steps)
            weight_column: Sample weight column (None for uniform weights)
            symbol_column: Symbol column for boundary detection (None to disable)
            stride: Step size between sequences (default 1)
            dtype: PyTorch tensor dtype (default float32)

        Raises:
            ValueError: If columns missing or seq_len/stride invalid
        """
        # Validate inputs
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if not feature_columns:
            raise ValueError("feature_columns cannot be empty")

        # Validate columns exist
        missing_features = [c for c in feature_columns if c not in df.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features[:5]}")
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found")

        # Store config
        self.config = SequenceConfig(
            seq_len=seq_len,
            stride=stride,
            feature_columns=feature_columns,
            label_column=label_column,
            weight_column=weight_column,
            symbol_column=symbol_column,
        )
        self.dtype = dtype

        # Pre-convert data to numpy arrays for efficiency
        self._features = df[feature_columns].values.astype(np.float32)
        self._labels = df[label_column].values.astype(np.float32)

        # Handle weights
        if weight_column and weight_column in df.columns:
            self._weights = df[weight_column].values.astype(np.float32)
        else:
            self._weights = np.ones(len(df), dtype=np.float32)

        # Find symbol boundaries if symbol isolation enabled
        if symbol_column and symbol_column in df.columns:
            boundaries = find_symbol_boundaries(df, symbol_column)
        else:
            boundaries = None

        # Build valid sequence indices
        self._indices = build_sequence_indices(
            n_samples=len(df),
            seq_len=seq_len,
            stride=stride,
            symbol_boundaries=boundaries
        )

        if len(self._indices) == 0:
            logger.warning(
                f"No valid sequences generated (n_rows={len(df)}, "
                f"seq_len={seq_len}, boundaries={len(boundaries or [])})"
            )

        logger.debug(
            f"SequenceDataset: {len(self._indices)} sequences from {len(df)} rows "
            f"(seq_len={seq_len}, stride={stride}, features={len(feature_columns)})"
        )

    def __len__(self) -> int:
        """Return number of valid sequences."""
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sequence sample.

        Args:
            idx: Sequence index (0 to len(self)-1)

        Returns:
            Tuple of (X_seq, y, weight):
                - X_seq: Feature tensor, shape (seq_len, n_features)
                - y: Label scalar tensor
                - weight: Sample weight scalar tensor
        """
        if idx < 0 or idx >= len(self._indices):
            raise IndexError(f"Index {idx} out of range [0, {len(self._indices)})")

        start_idx = self._indices[idx]
        end_idx = start_idx + self.config.seq_len

        # Extract sequence
        X_seq = self._features[start_idx:end_idx]

        # Label and weight from the last position in the sequence
        # This is the "prediction target" for the sequence
        target_idx = end_idx - 1
        y = self._labels[target_idx]
        weight = self._weights[target_idx]

        return (
            torch.tensor(X_seq, dtype=self.dtype),
            torch.tensor(y, dtype=self.dtype),
            torch.tensor(weight, dtype=self.dtype),
        )

    @property
    def n_samples(self) -> int:
        """Number of valid sequences."""
        return len(self._indices)

    @property
    def n_features(self) -> int:
        """Number of features per time step."""
        return len(self.config.feature_columns)

    @property
    def seq_len(self) -> int:
        """Sequence length."""
        return self.config.seq_len

    @property
    def feature_shape(self) -> tuple[int, int]:
        """Shape of feature tensor: (seq_len, n_features)."""
        return (self.config.seq_len, self.n_features)

    def get_all_labels(self) -> np.ndarray:
        """Get all labels for the valid sequences (for stratification)."""
        target_indices = self._indices + self.config.seq_len - 1
        return self._labels[target_indices]

    def get_all_weights(self) -> np.ndarray:
        """Get all weights for the valid sequences."""
        target_indices = self._indices + self.config.seq_len - 1
        return self._weights[target_indices]

    def get_label_distribution(self) -> dict:
        """Get label value counts for valid sequences."""
        labels = self.get_all_labels()
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist(), strict=False))

    def __repr__(self) -> str:
        return (
            f"SequenceDataset(n_sequences={len(self)}, "
            f"seq_len={self.seq_len}, features={self.n_features}, "
            f"stride={self.config.stride})"
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_sequence_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    seq_len: int,
    weight_column: str | None = None,
    symbol_column: str | None = "symbol",
    stride: int = 1,
) -> SequenceDataset:
    """
    Factory function to create SequenceDataset.

    Convenience wrapper around SequenceDataset constructor.

    Args:
        df: Source DataFrame
        feature_columns: List of feature column names
        label_column: Target/label column name
        seq_len: Sequence length
        weight_column: Sample weight column (optional)
        symbol_column: Symbol column for isolation (default "symbol")
        stride: Step between sequences (default 1)

    Returns:
        SequenceDataset instance
    """
    return SequenceDataset(
        df=df,
        feature_columns=feature_columns,
        label_column=label_column,
        seq_len=seq_len,
        weight_column=weight_column,
        symbol_column=symbol_column,
        stride=stride,
    )
