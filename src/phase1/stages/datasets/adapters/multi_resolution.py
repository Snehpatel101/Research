"""
MultiResolution4DAdapter - 4D Tensor Adapter for Multi-Timeframe Models.

Transforms multi-timeframe (MTF) features into 4D tensors suitable for
advanced multi-resolution neural networks that process multiple timeframes
simultaneously.

Output shape: (batch, n_timeframes, seq_len, features_per_tf)

This adapter is designed for models that:
- Process multiple timeframes in parallel (e.g., multi-scale CNNs)
- Use attention across timeframes (e.g., cross-timeframe transformers)
- Perform hierarchical temporal modeling

Key Features:
- Automatic timeframe detection from column naming conventions
- Configurable feature selection per timeframe
- Sliding window sequence generation with symbol isolation
- Padding/alignment for consistent tensor shapes
- Integration with TimeSeriesDataContainer

Usage:
------
    from src.phase1.stages.datasets.adapters import MultiResolution4DAdapter

    # Create adapter for 9-timeframe MTF data
    adapter = MultiResolution4DAdapter(
        timeframes=['1min', '5min', '10min', '15min', '20min',
                   '25min', '30min', '45min', '1h'],
        seq_len=60,
        stride=1
    )

    # Create PyTorch dataset
    dataset = adapter.create_dataset(
        df=train_df,
        label_column='label_h20',
        weight_column='sample_weight_h20'
    )

    # Use with DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for X_4d, y, weights in loader:
        # X_4d shape: (32, 9, 60, n_features_per_tf)
        model_output = model(X_4d)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.phase1.stages.mtf.constants import (
    DEFAULT_MTF_TIMEFRAMES,
    MTF_TIMEFRAMES,
)
from src.phase1.stages.datasets.adapters.utils import (
    DEFAULT_MTF_FEATURES,
    TIMEFRAME_SUFFIX_PATTERNS,
    build_4d_sequence_indices,
    build_timeframe_feature_map,
    extract_timeframe_columns,
    find_symbol_boundaries,
    get_timeframe_suffix,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MultiResolution4DConfig:
    """Configuration for Multi-Resolution 4D data generation."""

    timeframes: list[str] = field(default_factory=lambda: DEFAULT_MTF_TIMEFRAMES.copy())
    seq_len: int = 60
    stride: int = 1
    features_per_timeframe: list[str] | None = None
    label_column: str = "label_h20"
    weight_column: str = "sample_weight_h20"
    symbol_column: str | None = "symbol"
    datetime_column: str = "datetime"
    include_base_features: bool = True
    pad_missing_features: bool = True
    pad_value: float = 0.0

    def __post_init__(self) -> None:
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")
        if not self.timeframes:
            raise ValueError("timeframes list cannot be empty")

        for tf in self.timeframes:
            if tf not in MTF_TIMEFRAMES and tf not in TIMEFRAME_SUFFIX_PATTERNS:
                raise ValueError(
                    f"Unknown timeframe '{tf}'. Valid options: {list(MTF_TIMEFRAMES.keys())}"
                )

        if self.features_per_timeframe is None:
            self.features_per_timeframe = DEFAULT_MTF_FEATURES.copy()


# =============================================================================
# MULTI-RESOLUTION 4D ADAPTER
# =============================================================================

class MultiResolution4DAdapter:
    """
    Adapter that transforms MTF data into 4D tensors.

    Produces tensors of shape: (batch, n_timeframes, seq_len, features_per_tf)

    This adapter handles:
    - Automatic detection of timeframe-specific columns
    - Consistent feature ordering across timeframes
    - Padding for timeframes with fewer features
    - Symbol-isolated sequence generation

    Attributes:
        config: MultiResolution4DConfig with generation parameters
        timeframes: List of timeframes in order
        n_timeframes: Number of timeframes
        features_per_tf: Number of features per timeframe (after padding)

    Example:
        >>> adapter = MultiResolution4DAdapter(
        ...     timeframes=['1min', '5min', '15min', '30min', '1h'],
        ...     seq_len=60
        ... )
        >>> dataset = adapter.create_dataset(train_df, 'label_h20')
        >>> X_4d, y, w = dataset[0]
        >>> X_4d.shape  # (5, 60, n_features)
    """

    def __init__(
        self,
        timeframes: list[str] | None = None,
        seq_len: int = 60,
        stride: int = 1,
        features_per_timeframe: list[str] | None = None,
        include_base_features: bool = True,
        pad_missing_features: bool = True,
        pad_value: float = 0.0,
    ) -> None:
        """
        Initialize MultiResolution4DAdapter.

        Args:
            timeframes: List of timeframes to include. Default: 9-timeframe ladder
            seq_len: Sequence length for windowing
            stride: Step between sequences
            features_per_timeframe: Base feature names to extract per timeframe
            include_base_features: Include non-MTF features for base timeframe
            pad_missing_features: Pad timeframes to have same feature count
            pad_value: Value to use for padding
        """
        self.config = MultiResolution4DConfig(
            timeframes=timeframes or DEFAULT_MTF_TIMEFRAMES.copy(),
            seq_len=seq_len,
            stride=stride,
            features_per_timeframe=features_per_timeframe,
            include_base_features=include_base_features,
            pad_missing_features=pad_missing_features,
            pad_value=pad_value,
        )

        self._feature_map: dict[str, list[str]] | None = None
        self._max_features: int = 0

    @property
    def timeframes(self) -> list[str]:
        """List of timeframes."""
        return self.config.timeframes

    @property
    def n_timeframes(self) -> int:
        """Number of timeframes."""
        return len(self.config.timeframes)

    @property
    def seq_len(self) -> int:
        """Sequence length."""
        return self.config.seq_len

    def analyze_features(self, df: pd.DataFrame) -> dict[str, int]:
        """
        Analyze available features per timeframe.

        Args:
            df: DataFrame with MTF columns

        Returns:
            Dict mapping timeframe -> feature count
        """
        feature_map = build_timeframe_feature_map(
            df,
            self.config.timeframes,
            self.config.features_per_timeframe,
            self.config.include_base_features,
        )
        return {tf: len(cols) for tf, cols in feature_map.items()}

    def prepare(self, df: pd.DataFrame) -> None:
        """
        Prepare the adapter by analyzing the DataFrame structure.

        This must be called before create_dataset() to build the feature map.

        Args:
            df: DataFrame with MTF columns
        """
        self._feature_map = build_timeframe_feature_map(
            df,
            self.config.timeframes,
            self.config.features_per_timeframe,
            self.config.include_base_features,
        )

        feature_counts = [len(cols) for cols in self._feature_map.values()]
        self._max_features = max(feature_counts) if feature_counts else 0

        logger.info(
            f"MultiResolution4DAdapter prepared: {self.n_timeframes} timeframes, "
            f"max_features={self._max_features}"
        )
        for tf in self.config.timeframes:
            n_features = len(self._feature_map.get(tf, []))
            logger.debug(f"  {tf}: {n_features} features")

    def get_feature_map(self) -> dict[str, list[str]]:
        """
        Get the feature map (timeframe -> column names).

        Returns:
            Dict mapping timeframe to list of column names

        Raises:
            RuntimeError: If prepare() has not been called
        """
        if self._feature_map is None:
            raise RuntimeError("Adapter not prepared. Call prepare(df) first.")
        return self._feature_map.copy()

    def create_dataset(
        self,
        df: pd.DataFrame,
        label_column: str | None = None,
        weight_column: str | None = None,
        symbol_column: str | None = None,
        auto_prepare: bool = True,
    ) -> "MultiResolution4DDataset":
        """
        Create a PyTorch Dataset from the DataFrame.

        Args:
            df: DataFrame with MTF features
            label_column: Label column name (default from config)
            weight_column: Weight column name (default from config)
            symbol_column: Symbol column for isolation (None to disable)
            auto_prepare: Automatically call prepare() if not done

        Returns:
            MultiResolution4DDataset instance
        """
        if auto_prepare and self._feature_map is None:
            self.prepare(df)

        if self._feature_map is None:
            raise RuntimeError("Adapter not prepared. Call prepare(df) first.")

        label_col = label_column or self.config.label_column
        weight_col = weight_column or self.config.weight_column
        symbol_col = symbol_column if symbol_column is not None else self.config.symbol_column

        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in DataFrame")

        return MultiResolution4DDataset(
            df=df,
            feature_map=self._feature_map,
            timeframes=self.config.timeframes,
            seq_len=self.config.seq_len,
            stride=self.config.stride,
            label_column=label_col,
            weight_column=weight_col,
            symbol_column=symbol_col,
            pad_missing_features=self.config.pad_missing_features,
            pad_value=self.config.pad_value,
        )

    def get_output_shape(self, n_features: int | None = None) -> tuple[int, int, int]:
        """
        Get the output tensor shape (excluding batch dimension).

        Args:
            n_features: Features per timeframe (uses max from prepare() if None)

        Returns:
            Tuple of (n_timeframes, seq_len, n_features)
        """
        features = n_features if n_features is not None else self._max_features
        return (self.n_timeframes, self.config.seq_len, features)

    def __repr__(self) -> str:
        return (
            f"MultiResolution4DAdapter("
            f"timeframes={self.n_timeframes}, "
            f"seq_len={self.seq_len}, "
            f"prepared={self._feature_map is not None})"
        )


# =============================================================================
# MULTI-RESOLUTION 4D DATASET
# =============================================================================

class MultiResolution4DDataset(Dataset):
    """
    PyTorch Dataset for multi-resolution 4D time series sequences.

    Produces tensors of shape: (n_timeframes, seq_len, features)

    Each sample returns:
        - X_4d: 4D feature tensor (n_timeframes, seq_len, max_features)
        - y: Label scalar
        - weight: Sample weight scalar

    Attributes:
        n_samples: Number of valid sequences
        n_timeframes: Number of timeframes
        n_features: Max features per timeframe
        shape: Tuple of (n_timeframes, seq_len, n_features)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_map: dict[str, list[str]],
        timeframes: list[str],
        seq_len: int,
        stride: int = 1,
        label_column: str = "label_h20",
        weight_column: str | None = None,
        symbol_column: str | None = "symbol",
        pad_missing_features: bool = True,
        pad_value: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Initialize MultiResolution4DDataset.

        Args:
            df: Source DataFrame (sorted by datetime within symbols)
            feature_map: Dict mapping timeframe -> list of column names
            timeframes: Ordered list of timeframes
            seq_len: Sequence length
            stride: Step between sequences
            label_column: Label column name
            weight_column: Weight column name (None for uniform)
            symbol_column: Symbol column for boundary detection
            pad_missing_features: Pad to max features across timeframes
            pad_value: Value to use for padding
            dtype: PyTorch tensor dtype
        """
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if not timeframes:
            raise ValueError("timeframes list cannot be empty")
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found")

        self._timeframes = timeframes
        self._seq_len = seq_len
        self._stride = stride
        self._dtype = dtype
        self._pad_value = pad_value

        feature_counts = [len(feature_map.get(tf, [])) for tf in timeframes]
        self._max_features = max(feature_counts) if feature_counts else 0

        if self._max_features == 0:
            raise ValueError("No features found for any timeframe")

        self._timeframe_data = self._prepare_timeframe_arrays(
            df, feature_map, timeframes, pad_missing_features
        )

        self._labels = df[label_column].values.astype(np.float32)

        if weight_column and weight_column in df.columns:
            self._weights = df[weight_column].values.astype(np.float32)
        else:
            self._weights = np.ones(len(df), dtype=np.float32)

        if symbol_column and symbol_column in df.columns:
            boundaries = find_symbol_boundaries(df, symbol_column)
        else:
            boundaries = None

        self._indices = build_4d_sequence_indices(
            n_samples=len(df),
            seq_len=seq_len,
            stride=stride,
            symbol_boundaries=boundaries,
        )

        if len(self._indices) == 0:
            logger.warning(
                f"No valid sequences generated (n_rows={len(df)}, "
                f"seq_len={seq_len}, boundaries={len(boundaries or [])})"
            )

        logger.debug(
            f"MultiResolution4DDataset: {len(self._indices)} sequences from {len(df)} rows "
            f"(timeframes={len(timeframes)}, seq_len={seq_len}, max_features={self._max_features})"
        )

    def _prepare_timeframe_arrays(
        self,
        df: pd.DataFrame,
        feature_map: dict[str, list[str]],
        timeframes: list[str],
        pad_to_max: bool,
    ) -> np.ndarray:
        """Convert DataFrame to 3D numpy array: (n_timeframes, n_samples, max_features)."""
        n_samples = len(df)
        n_timeframes = len(timeframes)

        data = np.full(
            (n_timeframes, n_samples, self._max_features),
            self._pad_value,
            dtype=np.float32,
        )

        for tf_idx, tf in enumerate(timeframes):
            columns = feature_map.get(tf, [])
            if not columns:
                continue

            tf_data = df[columns].values.astype(np.float32)
            n_features = tf_data.shape[1]
            data[tf_idx, :, :n_features] = tf_data

        return data

    def __len__(self) -> int:
        """Return number of valid sequences."""
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single 4D sequence sample.

        Args:
            idx: Sequence index (0 to len(self)-1)

        Returns:
            Tuple of (X_4d, y, weight):
                - X_4d: Feature tensor, shape (n_timeframes, seq_len, n_features)
                - y: Label scalar tensor
                - weight: Sample weight scalar tensor
        """
        if idx < 0 or idx >= len(self._indices):
            raise IndexError(f"Index {idx} out of range [0, {len(self._indices)})")

        start_idx = self._indices[idx]
        end_idx = start_idx + self._seq_len

        X_4d = self._timeframe_data[:, start_idx:end_idx, :]

        target_idx = end_idx - 1
        y = self._labels[target_idx]
        weight = self._weights[target_idx]

        return (
            torch.tensor(X_4d, dtype=self._dtype),
            torch.tensor(y, dtype=self._dtype),
            torch.tensor(weight, dtype=self._dtype),
        )

    @property
    def n_samples(self) -> int:
        """Number of valid sequences."""
        return len(self._indices)

    @property
    def n_timeframes(self) -> int:
        """Number of timeframes."""
        return len(self._timeframes)

    @property
    def n_features(self) -> int:
        """Max features per timeframe."""
        return self._max_features

    @property
    def seq_len(self) -> int:
        """Sequence length."""
        return self._seq_len

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of feature tensor: (n_timeframes, seq_len, n_features)."""
        return (self.n_timeframes, self._seq_len, self._max_features)

    @property
    def timeframes(self) -> list[str]:
        """List of timeframes."""
        return self._timeframes.copy()

    def get_all_labels(self) -> np.ndarray:
        """Get all labels for valid sequences (for stratification)."""
        target_indices = self._indices + self._seq_len - 1
        return self._labels[target_indices]

    def get_all_weights(self) -> np.ndarray:
        """Get all weights for valid sequences."""
        target_indices = self._indices + self._seq_len - 1
        return self._weights[target_indices]

    def get_label_distribution(self) -> dict:
        """Get label value counts for valid sequences."""
        labels = self.get_all_labels()
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist(), strict=False))

    def get_timeframe_feature_counts(self) -> dict[str, int]:
        """Get actual feature counts per timeframe (before padding)."""
        counts = {}
        for tf_idx, tf in enumerate(self._timeframes):
            tf_data = self._timeframe_data[tf_idx]
            variances = np.var(tf_data, axis=0)
            counts[tf] = int(np.sum(variances > 1e-10))
        return counts

    def __repr__(self) -> str:
        return (
            f"MultiResolution4DDataset("
            f"n_sequences={len(self)}, "
            f"shape={self.shape}, "
            f"timeframes={self._timeframes})"
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_multi_resolution_dataset(
    df: pd.DataFrame,
    timeframes: list[str] | None = None,
    seq_len: int = 60,
    stride: int = 1,
    label_column: str = "label_h20",
    weight_column: str | None = "sample_weight_h20",
    symbol_column: str | None = "symbol",
    features_per_timeframe: list[str] | None = None,
    include_base_features: bool = True,
) -> MultiResolution4DDataset:
    """
    Factory function to create MultiResolution4DDataset.

    Convenience wrapper that creates an adapter and dataset in one call.

    Args:
        df: Source DataFrame with MTF features
        timeframes: List of timeframes (default: 9-timeframe ladder)
        seq_len: Sequence length
        stride: Step between sequences
        label_column: Label column name
        weight_column: Weight column name
        symbol_column: Symbol column for isolation
        features_per_timeframe: Base feature names per timeframe
        include_base_features: Include non-MTF features for base timeframe

    Returns:
        MultiResolution4DDataset instance
    """
    adapter = MultiResolution4DAdapter(
        timeframes=timeframes,
        seq_len=seq_len,
        stride=stride,
        features_per_timeframe=features_per_timeframe,
        include_base_features=include_base_features,
    )

    return adapter.create_dataset(
        df=df,
        label_column=label_column,
        weight_column=weight_column,
        symbol_column=symbol_column,
    )


__all__ = [
    "MultiResolution4DAdapter",
    "MultiResolution4DConfig",
    "MultiResolution4DDataset",
    "create_multi_resolution_dataset",
]
