"""
TimeSeriesDataContainer - Unified data container for Phase 2 model training.

This module provides a container class that loads Phase 1 pipeline outputs
and provides data in formats required by different model frameworks:
- sklearn: (X, y, weights) numpy arrays
- PyTorch: SequenceDataset with sliding windows
- NeuralForecast: DataFrames with [unique_id, ds, y, features...]

Usage:
------
    from src.phase1.stages.datasets.container import TimeSeriesDataContainer

    # Load from Phase 1 outputs
    container = TimeSeriesDataContainer.from_parquet_dir(
        path="data/splits/scaled",
        horizon=20
    )

    # Get sklearn arrays
    X_train, y_train, w_train = container.get_sklearn_arrays("train")

    # Get PyTorch sequences
    train_dataset = container.get_pytorch_sequences("train", seq_len=60)

    # Get NeuralForecast DataFrame
    nf_df = container.get_neuralforecast_df("train")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from torch.utils.data import Dataset

# Import shared constants from canonical source
from src.phase1.utils.feature_sets import (
    METADATA_COLUMNS,
    LABEL_PREFIXES,
    _is_label_column,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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
    feature_columns: List[str] = field(default_factory=list)
    label_column: str = ""
    weight_column: str = ""
    symbol_column: str = "symbol"
    datetime_column: str = "datetime"
    exclude_invalid_labels: bool = True

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        if not self.label_column:
            self.label_column = f"label_h{self.horizon}"
        if not self.weight_column:
            self.weight_column = f"sample_weight_h{self.horizon}"


@dataclass
class SplitData:
    """Data for a single split (train/val/test)."""
    df: pd.DataFrame
    feature_columns: List[str]
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
    def symbols(self) -> List[str]:
        if self.symbol_column in self.df.columns:
            return list(self.df[self.symbol_column].unique())
        return []


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# _is_label_column imported from src.phase1.utils.feature_sets


def _extract_feature_columns(
    df: pd.DataFrame,
    horizon: int,
    explicit_features: Optional[List[str]] = None
) -> List[str]:
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
        col for col in df.columns
        if col not in METADATA_COLUMNS and not _is_label_column(col)
    ]
    return features


def _validate_split_name(split: str) -> None:
    """Validate split name."""
    if split not in VALID_SPLITS:
        raise ValueError(
            f"Invalid split '{split}'. Must be one of: {VALID_SPLITS}"
        )


def _load_parquet_with_validation(path: Path, split_name: str) -> pd.DataFrame:
    """Load parquet file with basic validation."""
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"Empty DataFrame loaded from {path}")
    logger.debug(f"Loaded {split_name}: {len(df)} rows, {len(df.columns)} columns")
    return df


# =============================================================================
# TIMESERIES DATA CONTAINER
# =============================================================================

class TimeSeriesDataContainer:
    """
    Unified container for time series ML data.

    Loads Phase 1 pipeline outputs and provides data in formats required
    by different model frameworks (sklearn, PyTorch, NeuralForecast).

    Attributes:
        config: DataContainerConfig with horizon and column settings
        splits: Dict mapping split names to SplitData objects
        metadata: Optional dict with scaling/run metadata

    Example:
        >>> container = TimeSeriesDataContainer.from_parquet_dir(
        ...     path="data/splits/scaled",
        ...     horizon=20
        ... )
        >>> X, y, w = container.get_sklearn_arrays("train")
        >>> print(X.shape, y.shape)
    """

    def __init__(
        self,
        config: DataContainerConfig,
        splits: Dict[str, SplitData],
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Initialize TimeSeriesDataContainer.

        Args:
            config: Container configuration
            splits: Dict of split name to SplitData
            metadata: Optional metadata dict (from scaling_metadata.json)
        """
        if not splits:
            raise ValueError("At least one split must be provided")
        self.config = config
        self.splits = splits
        self.metadata = metadata or {}

    @classmethod
    def from_parquet_dir(
        cls,
        path: Union[str, Path],
        horizon: int,
        feature_columns: Optional[List[str]] = None,
        exclude_invalid_labels: bool = True
    ) -> "TimeSeriesDataContainer":
        """
        Load container from Phase 1 scaled parquet directory.

        Expected directory structure:
            path/
                train_scaled.parquet
                val_scaled.parquet
                test_scaled.parquet
                scaling_metadata.json (optional)

        Args:
            path: Path to scaled splits directory
            horizon: Label horizon (e.g., 5, 10, 20)
            feature_columns: Explicit feature list, or None for auto-detect
            exclude_invalid_labels: Whether to filter out rows with label=-99

        Returns:
            TimeSeriesDataContainer instance

        Raises:
            FileNotFoundError: If parquet files don't exist
            ValueError: If horizon is invalid or data is empty
        """
        path = Path(path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        # Load metadata if available
        metadata_path = path / "scaling_metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

        # Create config
        config = DataContainerConfig(
            horizon=horizon,
            feature_columns=feature_columns or [],
            exclude_invalid_labels=exclude_invalid_labels
        )

        # Load splits
        splits: Dict[str, SplitData] = {}
        split_files = {
            "train": path / "train_scaled.parquet",
            "val": path / "val_scaled.parquet",
            "test": path / "test_scaled.parquet",
        }

        for split_name, split_path in split_files.items():
            if not split_path.exists():
                logger.warning(f"Split file not found, skipping: {split_path}")
                continue

            df = _load_parquet_with_validation(split_path, split_name)

            # Validate label column exists
            label_col = config.label_column
            if label_col not in df.columns:
                raise ValueError(
                    f"Label column '{label_col}' not found in {split_name}. "
                    f"Available label columns: {[c for c in df.columns if _is_label_column(c)]}"
                )

            # Validate weight column exists
            weight_col = config.weight_column
            if weight_col not in df.columns:
                logger.warning(
                    f"Weight column '{weight_col}' not found, using uniform weights"
                )

            # Filter invalid labels if requested
            if exclude_invalid_labels:
                invalid_mask = df[label_col] == INVALID_LABEL
                n_invalid = invalid_mask.sum()
                if n_invalid > 0:
                    logger.info(
                        f"{split_name}: Filtering {n_invalid} rows with "
                        f"invalid label ({INVALID_LABEL})"
                    )
                    df = df[~invalid_mask].reset_index(drop=True)

            # Extract feature columns (auto-detect if not provided)
            features = _extract_feature_columns(
                df, horizon, config.feature_columns or None
            )

            # Update config with detected features (on first split)
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
            raise ValueError(f"No valid split files found in {path}")

        logger.info(
            f"Loaded TimeSeriesDataContainer: horizon={horizon}, "
            f"splits={list(splits.keys())}, features={len(config.feature_columns)}"
        )

        return cls(config, splits, metadata)

    @classmethod
    def from_dataframes(
        cls,
        train_df: Optional[pd.DataFrame] = None,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        horizon: int = 20,
        feature_columns: Optional[List[str]] = None,
        exclude_invalid_labels: bool = True
    ) -> "TimeSeriesDataContainer":
        """
        Create container directly from DataFrames.

        Useful for testing or when data is already loaded.
        """
        config = DataContainerConfig(
            horizon=horizon,
            feature_columns=feature_columns or [],
            exclude_invalid_labels=exclude_invalid_labels
        )

        splits: Dict[str, SplitData] = {}
        split_dfs = {"train": train_df, "val": val_df, "test": test_df}

        for split_name, df in split_dfs.items():
            if df is None or df.empty:
                continue

            label_col = config.label_column
            weight_col = config.weight_column

            if label_col not in df.columns:
                raise ValueError(f"Label column '{label_col}' not found in {split_name}")

            if exclude_invalid_labels:
                df = df[df[label_col] != INVALID_LABEL].reset_index(drop=True)

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
    def available_splits(self) -> List[str]:
        """List of available split names."""
        return list(self.splits.keys())

    @property
    def feature_columns(self) -> List[str]:
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

    # =========================================================================
    # LABEL END TIMES (FOR PURGED CV)
    # =========================================================================

    def get_label_end_times(self, split: str) -> Optional[pd.Series]:
        """
        Get label end times for purged cross-validation.

        Label end times mark when each sample's label outcome is known.
        This enables proper purging of overlapping labels in PurgedKFold.

        Args:
            split: Split name ("train", "val", "test")

        Returns:
            Series of datetime when each label is resolved, or None if not available
        """
        split_data = self.get_split(split)
        label_end_time_col = f"label_end_time_h{self.config.horizon}"

        if label_end_time_col not in split_data.df.columns:
            logger.debug(
                f"Label end time column '{label_end_time_col}' not found in {split}. "
                "Overlapping label purging will be skipped."
            )
            return None

        return pd.to_datetime(split_data.df[label_end_time_col])

    # =========================================================================
    # SKLEARN FORMAT
    # =========================================================================

    def get_sklearn_arrays(
        self,
        split: str,
        return_df: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray],
               Tuple[pd.DataFrame, pd.Series, pd.Series]]:
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
        symbol_isolated: bool = True
    ) -> "Dataset":
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

        from src.phase1.stages.datasets.sequences import SequenceDataset

        split_data = self.get_split(split)

        return SequenceDataset(
            df=split_data.df,
            feature_columns=split_data.feature_columns,
            label_column=split_data.label_column,
            weight_column=split_data.weight_column,
            symbol_column=split_data.symbol_column if symbol_isolated else None,
            seq_len=seq_len,
            stride=stride,
        )

    # =========================================================================
    # NEURALFORECAST FORMAT
    # =========================================================================

    def get_neuralforecast_df(
        self,
        split: str,
        include_features: bool = True
    ) -> pd.DataFrame:
        """
        Get DataFrame in NeuralForecast format.

        NeuralForecast expects columns: [unique_id, ds, y, (optional features)]
        - unique_id: Identifier for each time series (symbol)
        - ds: Datetime column
        - y: Target variable

        Args:
            split: Split name ("train", "val", "test")
            include_features: If True, include feature columns

        Returns:
            DataFrame with NeuralForecast-compatible schema

        Raises:
            KeyError: If split not found
        """
        split_data = self.get_split(split)
        df = split_data.df.copy()

        # Build output columns
        output_cols = []

        # unique_id from symbol column
        if split_data.symbol_column in df.columns:
            df["unique_id"] = df[split_data.symbol_column]
        else:
            df["unique_id"] = "default"
        output_cols.append("unique_id")

        # ds from datetime column
        if split_data.datetime_column in df.columns:
            df["ds"] = pd.to_datetime(df[split_data.datetime_column])
        else:
            # Create synthetic datetime index
            df["ds"] = pd.date_range(
                start="2020-01-01",
                periods=len(df),
                freq="5min"
            )
        output_cols.append("ds")

        # y from label column
        df["y"] = df[split_data.label_column]
        output_cols.append("y")

        # Optional: sample weight
        if split_data.weight_column in df.columns:
            df["sample_weight"] = df[split_data.weight_column]
            output_cols.append("sample_weight")

        # Optional: features
        if include_features:
            output_cols.extend(split_data.feature_columns)

        return df[output_cols]

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def describe(self) -> Dict:
        """Return summary statistics for the container."""
        summary = {
            "horizon": self.config.horizon,
            "n_features": self.n_features,
            "label_column": self.config.label_column,
            "weight_column": self.config.weight_column,
            "splits": {},
        }

        for split_name, split_data in self.splits.items():
            labels = split_data.df[split_data.label_column]
            summary["splits"][split_name] = {
                "n_samples": split_data.n_samples,
                "symbols": split_data.symbols,
                "label_distribution": labels.value_counts().to_dict(),
            }

        return summary

    def __repr__(self) -> str:
        splits_info = ", ".join(
            f"{k}={v.n_samples}" for k, v in self.splits.items()
        )
        return (
            f"TimeSeriesDataContainer(horizon={self.horizon}, "
            f"features={self.n_features}, splits=[{splits_info}])"
        )
