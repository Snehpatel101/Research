"""
TimeSeriesDataContainer - Unified data container for model training and evaluation.

CANONICAL LOCATION: src/core/container.py

This module provides a container class over prepared train/val/test frames
and provides data in formats required by different model frameworks:
- sklearn: (X, y, weights) numpy arrays
- PyTorch: SequenceDataset with sliding windows
- NeuralForecast: DataFrames with [unique_id, ds, y, features...]

Usage:
------
    from src.core import TimeSeriesDataContainer

    container = TimeSeriesDataContainer.from_dataframes(
        train_df=train_df, val_df=val_df, horizon=20
    )

    # Get sklearn arrays
    X_train, y_train, w_train = container.get_sklearn_arrays("train")

    # Get PyTorch sequences
    train_dataset = container.get_pytorch_sequences("train", seq_len=60)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core.label_spans import (
    LabelSpans,
    frame_label_ends,
    label_end_column,
    remap_label_ends,
)

if TYPE_CHECKING:
    from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# =============================================================================
# METADATA CONSTANTS (inlined from phase1 to avoid circular dependency)
# =============================================================================

# Metadata columns to exclude from features
METADATA_COLUMNS = {
    # Temporal identifiers
    "datetime",
    "timestamp",
    "date",
    "time",
    # Symbol identifiers
    "symbol",
    # Raw OHLCV data
    "open",
    "high",
    "low",
    "close",
    "volume",
    # Pipeline metadata
    "timeframe",
    "session_id",
    "missing_bar",
    "roll_event",
    "roll_window",
    "filled",
}

# Label column prefixes (targets, not features)
_LABEL_PREFIXES = (
    "label_",
    "bars_to_hit_",
    "mae_",
    "mfe_",
    "quality_",
    "sample_weight_",
    "touch_type_",
    "pain_to_gain_",
    "time_weighted_dd_",
    "fwd_return_",
    "fwd_return_log_",
    "time_to_hit_",
)


def _is_label_column(name: str) -> bool:
    """Check if column name is a label/target column."""
    return any(name.startswith(prefix) for prefix in _LABEL_PREFIXES)


# =============================================================================
# CONSTANTS
# =============================================================================

# Valid split names
VALID_SPLITS = {"train", "val", "test"}

# Invalid label value (samples to exclude)
INVALID_LABEL = -99


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class DataContainerConfig:
    """Configuration for TimeSeriesDataContainer."""

    horizon: int
    feature_columns: list[str] = field(default_factory=list)
    label_column: str = ""
    weight_column: str = ""
    symbol_column: str = "symbol"
    datetime_column: str = "datetime"
    exclude_invalid_labels: bool = True
    # 3: {-1, 0, +1} (short/neutral/long); 2: binary labels {0, 1} (no move/move)
    n_classes: int = 3

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        if self.n_classes not in (2, 3):
            raise ValueError(f"n_classes must be 2 or 3, got {self.n_classes}")
        if not self.label_column:
            self.label_column = f"label_h{self.horizon}"
        if not self.weight_column:
            self.weight_column = f"sample_weight_h{self.horizon}"


@dataclass
class SplitData:
    """Data for a single split (train/val/test)."""

    df: pd.DataFrame
    feature_columns: list[str]
    label_column: str
    weight_column: str
    symbol_column: str
    datetime_column: str

    @property
    def n_samples(self) -> int:
        return len(self.df)

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    @property
    def symbols(self) -> list[str]:
        if self.symbol_column in self.df.columns:
            return list(self.df[self.symbol_column].unique())
        return []


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _extract_feature_columns(
    df: pd.DataFrame, horizon: int, explicit_features: list[str] | None = None
) -> list[str]:
    """
    Extract feature columns from DataFrame.

    If explicit_features is provided, validates and returns them.
    Otherwise, auto-detects features by excluding metadata and labels.
    """
    if explicit_features:
        missing = [col for col in explicit_features if col not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing[:10]}")
        return explicit_features

    # Auto-detect features: everything that's not metadata or labels
    features = [
        col for col in df.columns if col not in METADATA_COLUMNS and not _is_label_column(col)
    ]
    return features


def _drop_invalid_labels(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Drop rows with the invalid-label sentinel and renumber the rows.

    The label-end column (row positions, see ``label_end_column``) is re-mapped to
    the surviving rows, so every label span keeps covering the same bars.
    """
    keep = (df[label_col] != INVALID_LABEL).to_numpy()
    if keep.all():
        return df.reset_index(drop=True)

    ends = frame_label_ends(df, label_col)
    out = df.loc[keep].reset_index(drop=True)
    if ends is not None:
        out[label_end_column(label_col)] = remap_label_ends(ends[keep], np.flatnonzero(keep))
    return out


def _validate_split_name(split: str) -> None:
    """Validate split name."""
    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split '{split}'. Must be one of: {VALID_SPLITS}")


# =============================================================================
# TIMESERIES DATA CONTAINER
# =============================================================================


class TimeSeriesDataContainer:
    """
    Unified container for time series ML data.

    Holds prepared train/val/test frames and provides data in formats required
    by different model frameworks (sklearn, PyTorch, NeuralForecast).

    Attributes:
        config: DataContainerConfig with horizon and column settings
        splits: Dict mapping split names to SplitData objects
        metadata: Optional dict with run metadata

    Example:
        >>> container = TimeSeriesDataContainer.from_dataframes(
        ...     train_df=train_df, horizon=20
        ... )
        >>> X, y, w = container.get_sklearn_arrays("train")
        >>> print(X.shape, y.shape)
    """

    def __init__(
        self,
        config: DataContainerConfig,
        splits: dict[str, SplitData],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize TimeSeriesDataContainer.

        Args:
            config: Container configuration
            splits: Dict of split name to SplitData
            metadata: Optional metadata dict
        """
        if not splits:
            raise ValueError("At least one split must be provided")
        self.config = config
        self.splits = splits
        self.metadata = metadata or {}

    @classmethod
    def from_dataframes(
        cls,
        train_df: pd.DataFrame | None = None,
        val_df: pd.DataFrame | None = None,
        test_df: pd.DataFrame | None = None,
        horizon: int = 20,
        feature_columns: list[str] | None = None,
        exclude_invalid_labels: bool = True,
        n_classes: int = 3,
    ) -> TimeSeriesDataContainer:
        """
        Create container directly from DataFrames.

        Useful for testing or when data is already loaded. ``n_classes`` is 2 for
        binary labels (``LabelingConfig.binary_mode``), 3 otherwise.
        """
        config = DataContainerConfig(
            horizon=horizon,
            feature_columns=feature_columns or [],
            exclude_invalid_labels=exclude_invalid_labels,
            n_classes=n_classes,
        )

        splits: dict[str, SplitData] = {}
        split_dfs = {"train": train_df, "val": val_df, "test": test_df}

        for split_name, df in split_dfs.items():
            if df is None or df.empty:
                continue

            label_col = config.label_column
            weight_col = config.weight_column

            if label_col not in df.columns:
                raise ValueError(f"Label column '{label_col}' not found in {split_name}")

            if exclude_invalid_labels:
                df = _drop_invalid_labels(df, label_col)

            features = _extract_feature_columns(df, horizon, config.feature_columns or None)
            if not config.feature_columns:
                config.feature_columns = features

            splits[split_name] = SplitData(
                df=df,
                feature_columns=features,
                label_column=label_col,
                weight_column=weight_col,
                symbol_column=config.symbol_column,
                datetime_column=config.datetime_column,
            )

        if not splits:
            raise ValueError("At least one non-empty DataFrame must be provided")

        return cls(config, splits)

    # =========================================================================
    # SPLIT ACCESS
    # =========================================================================

    def get_split(self, split: str) -> SplitData:
        """Get SplitData for a specific split."""
        _validate_split_name(split)
        if split not in self.splits:
            raise KeyError(f"Split '{split}' not loaded. Available: {list(self.splits.keys())}")
        return self.splits[split]

    @property
    def feature_columns(self) -> list[str]:
        """Feature column names."""
        return self.config.feature_columns

    @property
    def n_features(self) -> int:
        """Number of features."""
        return len(self.config.feature_columns)

    @property
    def horizon(self) -> int:
        """Label horizon."""
        return self.config.horizon

    @property
    def n_classes(self) -> int:
        """Number of label classes (2 for binary labels, 3 for short/neutral/long)."""
        return self.config.n_classes

    # =========================================================================
    # LABEL SPANS (FOR PURGED CV)
    # =========================================================================

    def get_label_spans(self, split: str) -> LabelSpans | None:
        """
        Label spans (integer row positions) for purged cross-validation.

        Built from the split's ``label_end_h{horizon}`` column (see
        ``src.core.label_spans.label_end_column``), whose values are row
        positions of the split itself: the row at which each sample's
        triple-barrier label resolves.

        Args:
            split: Split name ("train", "val", "test")

        Returns:
            LabelSpans aligned with the rows of ``get_sklearn_arrays(split)``,
            or None when the split has no label-end column.
        """
        split_data = self.get_split(split)
        ends = frame_label_ends(split_data.df, split_data.label_column)
        if ends is None:
            return None
        return LabelSpans(starts=np.arange(len(ends), dtype=np.int64), ends=ends)

    # =========================================================================
    # SKLEARN FORMAT
    # =========================================================================

    def get_sklearn_arrays(
        self, split: str, return_df: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Get data in sklearn format: (X, y, weights).

        Args:
            split: Split name ("train", "val", "test")
            return_df: If True, return pandas objects instead of numpy arrays

        Returns:
            Tuple of (X, y, weights):
                - X: Features array/DataFrame, shape (n_samples, n_features)
                - y: Labels array/Series, shape (n_samples,)
                - weights: Sample weights array/Series, shape (n_samples,)

        Raises:
            KeyError: If split not found
        """
        split_data = self.get_split(split)
        df = split_data.df

        X = df[split_data.feature_columns]
        y = df[split_data.label_column]

        if split_data.weight_column in df.columns:
            weights = df[split_data.weight_column]
        else:
            weights = pd.Series(np.ones(len(df)), index=df.index)

        if return_df:
            if split_data.datetime_column in df.columns:
                # Frames carrying bar times are indexed by them
                times = pd.DatetimeIndex(df[split_data.datetime_column])
                X, y, weights = X.set_axis(times), y.set_axis(times), weights.set_axis(times)
            return X, y, weights

        return X.values, y.values, weights.values

    # =========================================================================
    # PYTORCH FORMAT
    # =========================================================================

    def get_pytorch_sequences(
        self,
        split: str,
        seq_len: int,
        stride: int = 1,
        symbol_isolated: bool = True,
        feature_columns: list[str] | None = None,
    ) -> Dataset:
        """
        Get PyTorch Dataset with sliding window sequences.

        Creates sequences suitable for LSTM, Transformer, and other
        sequence models. Symbol isolation prevents sequences from
        crossing symbol boundaries (e.g., no MES->MGC bleeding).

        Args:
            split: Split name ("train", "val", "test")
            seq_len: Sequence length (number of time steps)
            stride: Step size between sequences (default 1)
            symbol_isolated: If True, sequences don't cross symbol boundaries
            feature_columns: Optional list of feature columns to use.
                If None, uses all available features. Use this to filter
                to model-specific optimal feature sets.

        Returns:
            SequenceDataset instance (PyTorch Dataset)

        Raises:
            KeyError: If split not found
            ValueError: If seq_len <= 0 or stride <= 0
        """
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        from src.core.datasets import SequenceDataset

        split_data = self.get_split(split)

        # Use provided feature columns or default to all features
        if feature_columns is not None:
            # Filter to only features that exist in the data
            available = set(split_data.feature_columns)
            filtered_features = [f for f in feature_columns if f in available]
            missing = set(feature_columns) - available
            if missing:
                logger.warning(
                    f"Feature filtering: {len(missing)} features not found in data, "
                    f"using {len(filtered_features)}/{len(feature_columns)} requested features"
                )
            use_features = filtered_features
        else:
            use_features = split_data.feature_columns

        return SequenceDataset(
            df=split_data.df,
            feature_columns=use_features,
            label_column=split_data.label_column,
            weight_column=split_data.weight_column,
            symbol_column=split_data.symbol_column if symbol_isolated else None,
            seq_len=seq_len,
            stride=stride,
        )

    # =========================================================================
    # MULTI-RESOLUTION 4D FORMAT
    # =========================================================================

    def get_multi_resolution_4d(
        self,
        split: str,
        seq_len: int = 60,
        stride: int = 1,
        timeframes: list[str] | None = None,
        features_per_timeframe: list[str] | None = None,
        symbol_isolated: bool = True,
        include_base_features: bool = True,
    ) -> Dataset:
        """
        Get PyTorch Dataset with multi-resolution 4D sequences.

        Creates 4D tensors of shape (batch, n_timeframes, seq_len, features)
        suitable for multi-resolution models that process multiple timeframes
        simultaneously.

        This method is designed for:
        - Multi-scale CNNs that process timeframes in parallel
        - Cross-timeframe attention transformers
        - Hierarchical temporal models

        Args:
            split: Split name ("train", "val", "test")
            seq_len: Sequence length per timeframe (default 60)
            stride: Step size between sequences (default 1)
            timeframes: List of timeframes to include (default: 9-timeframe ladder)
            features_per_timeframe: Base feature names to extract per timeframe
            symbol_isolated: If True, sequences don't cross symbol boundaries
            include_base_features: Include non-MTF features for base timeframe

        Returns:
            MultiResolution4DDataset instance (PyTorch Dataset)

        Raises:
            KeyError: If split not found
            ValueError: If seq_len <= 0, stride <= 0, or no MTF features found

        Example:
            >>> container = TimeSeriesDataContainer.from_dataframes(
            ...     train_df=train_df, horizon=20
            ... )
            >>> dataset = container.get_multi_resolution_4d("train", seq_len=60)
            >>> X_4d, y, w = dataset[0]
            >>> X_4d.shape  # (9, 60, n_features)
        """
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        # Use factory function to avoid direct class import from data layer
        from src.data.adapters import create_multi_resolution_dataset

        split_data = self.get_split(split)

        return create_multi_resolution_dataset(
            df=split_data.df,
            label_column=split_data.label_column,
            weight_column=split_data.weight_column,
            symbol_column=split_data.symbol_column if symbol_isolated else None,
            timeframes=timeframes,
            seq_len=seq_len,
            stride=stride,
            features_per_timeframe=features_per_timeframe,
            include_base_features=include_base_features,
        )

    def __repr__(self) -> str:
        splits_info = ", ".join(f"{k}={v.n_samples}" for k, v in self.splits.items())
        return (
            f"TimeSeriesDataContainer(horizon={self.horizon}, "
            f"features={self.n_features}, splits=[{splits_info}])"
        )
