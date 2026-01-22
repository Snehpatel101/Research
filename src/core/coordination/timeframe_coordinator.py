"""
TimeframeCoordinator - Manages multi-timeframe data loading and alignment.

This module provides coordinated access to data across multiple timeframes,
ensuring proper alignment and forward-fill propagation for multi-stream models.

Key responsibilities:
- Load and cache timeframe-specific data on demand
- Establish anchor timeframe (smallest TF) for timestamp alignment
- Forward-fill higher timeframes to anchor timestamps (no lookahead)
- Provide data views for different model configurations

Usage:
    from src.core.coordination import TimeframeCoordinator
    from src.core.contracts import get_model_contract

    coordinator = TimeframeCoordinator(
        data_dir="data/splits/scaled",
        split="train",
        horizon=20,
    )

    # Load required timeframes
    coordinator.load_timeframes(["1min", "5min", "15min"])

    # Get data for a specific model
    contract = get_model_contract("patchtst")
    df = coordinator.get_data_for_model("patchtst", contract)

    # Get aligned multi-stream data
    dfs = coordinator.get_multi_stream_dfs(["1min", "5min", "15min"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.core.common.timeframes import (
    get_timeframe_minutes,
    normalize_timeframe,
)
from src.core.contracts import ModelContract, get_model_contract

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Columns to exclude when auto-detecting features
_METADATA_COLUMNS = frozenset(
    {
        # Index/timestamp
        "timestamp",
        "date",
        "datetime",
        "time",
        "index",
        # Labels
        "label",
        "target",
        "y",
        "label_horizon_5",
        "label_horizon_10",
        "label_horizon_15",
        "label_horizon_20",
        # Weights
        "weight",
        "sample_weight",
        "weights",
        # Other metadata
        "symbol",
        "contract",
        "session",
        "split",
        "fold",
    }
)

# Label column patterns to exclude
_LABEL_PATTERNS = ("label_", "target_", "horizon_", "y_")

# Weight column patterns to exclude
_WEIGHT_PATTERNS = ("weight", "sample_weight")


@dataclass
class TimeframeData:
    """
    Container for loaded timeframe data.

    Attributes:
        timeframe: Canonical timeframe string (e.g., "5min")
        df: DataFrame containing the data
        feature_columns: List of feature column names
        n_samples: Number of samples (computed from df)
        start_time: First timestamp in the data (computed from df)
        end_time: Last timestamp in the data (computed from df)
    """

    timeframe: str
    df: pd.DataFrame
    feature_columns: list[str]
    n_samples: int = field(init=False)
    start_time: pd.Timestamp | None = field(init=False)
    end_time: pd.Timestamp | None = field(init=False)

    def __post_init__(self) -> None:
        """Compute derived attributes from DataFrame."""
        self.n_samples = len(self.df)

        # Extract timestamps from index or column
        if isinstance(self.df.index, pd.DatetimeIndex):
            self.start_time = self.df.index.min()
            self.end_time = self.df.index.max()
        elif "timestamp" in self.df.columns:
            ts_col = pd.to_datetime(self.df["timestamp"])
            self.start_time = ts_col.min()
            self.end_time = ts_col.max()
        else:
            self.start_time = None
            self.end_time = None


class TimeframeCoordinator:
    """
    Coordinates data loading and alignment across multiple timeframes.

    This class provides:
    - Lazy loading of timeframe-specific data files
    - Anchor timeframe determination (smallest loaded TF)
    - Forward-fill alignment for higher timeframes
    - Model-specific data views based on contracts

    Key design principles:
    - Lazy loading: Timeframes loaded only when requested
    - Anchor-based alignment: Smallest TF determines timestamp grid
    - Forward fill: Higher TFs propagate to anchor timestamps (no lookahead)
    - Contract-driven: Data views shaped by model requirements

    Attributes:
        data_dir: Base directory for data files
        split: Data split (train, val, test)
        horizon: Label horizon for loading corresponding labels
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        horizon: int = 20,
    ) -> None:
        """
        Initialize the TimeframeCoordinator.

        Args:
            data_dir: Base directory containing timeframe data files.
                      Expected structure varies; see load_timeframes() for path patterns.
            split: Data split to load (train, val, test). Default: "train"
            horizon: Label horizon for loading corresponding labels. Default: 20
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.horizon = horizon

        # Internal state
        self._timeframe_data: dict[str, TimeframeData] = {}
        self._anchor_timeframe: str | None = None
        self._common_timestamps: pd.DatetimeIndex | None = None

        logger.debug(
            "TimeframeCoordinator initialized: data_dir=%s, split=%s, horizon=%d",
            self.data_dir,
            self.split,
            self.horizon,
        )

    def load_timeframes(
        self,
        timeframes: list[str],
        feature_columns: list[str] | None = None,
    ) -> None:
        """
        Load data files for the given timeframes.

        Attempts to load from multiple path patterns in order:
        1. {data_dir}/{tf}/{split}_scaled.parquet
        2. {data_dir}/{split}_{tf}_scaled.parquet
        3. {data_dir}/features_{tf}.parquet

        After loading, sets the anchor timeframe to the smallest loaded TF.

        Args:
            timeframes: List of timeframe strings (will be normalized)
            feature_columns: Optional list of feature columns to select.
                            If None, auto-detects features excluding metadata/labels.

        Raises:
            FileNotFoundError: If data file cannot be found for a timeframe
            ValueError: If loaded DataFrame is empty
        """
        for tf in timeframes:
            tf_norm = normalize_timeframe(tf)

            # Skip if already loaded
            if tf_norm in self._timeframe_data:
                logger.debug("Timeframe %s already loaded, skipping", tf_norm)
                continue

            # Try multiple path patterns
            df = self._try_load_timeframe(tf_norm)
            if df is None:
                paths_tried = self._get_possible_paths(tf_norm)
                raise FileNotFoundError(
                    f"Could not find data file for timeframe '{tf_norm}'. "
                    f"Tried paths: {[str(p) for p in paths_tried]}"
                )

            if df.empty:
                raise ValueError(f"Loaded DataFrame for timeframe '{tf_norm}' is empty")

            # Detect or validate feature columns
            if feature_columns is not None:
                cols = [c for c in feature_columns if c in df.columns]
                if not cols:
                    raise ValueError(
                        f"None of the specified feature columns found in data for {tf_norm}. "
                        f"Available columns: {list(df.columns)[:20]}..."
                    )
            else:
                cols = self._detect_feature_columns(df)

            self._timeframe_data[tf_norm] = TimeframeData(
                timeframe=tf_norm,
                df=df,
                feature_columns=cols,
            )

            logger.info(
                "Loaded timeframe %s: %d samples, %d features, range [%s, %s]",
                tf_norm,
                len(df),
                len(cols),
                self._timeframe_data[tf_norm].start_time,
                self._timeframe_data[tf_norm].end_time,
            )

        # Update anchor after loading
        self._set_anchor_timeframe()

    def _try_load_timeframe(self, tf_norm: str) -> pd.DataFrame | None:
        """
        Try to load data from multiple path patterns.

        Args:
            tf_norm: Normalized timeframe string

        Returns:
            DataFrame if found, None otherwise
        """
        for path in self._get_possible_paths(tf_norm):
            if path.exists():
                logger.debug("Loading timeframe %s from %s", tf_norm, path)
                return pd.read_parquet(path)
        return None

    def _get_possible_paths(self, tf_norm: str) -> list[Path]:
        """
        Get list of possible file paths for a timeframe.

        Args:
            tf_norm: Normalized timeframe string

        Returns:
            List of Path objects to try
        """
        return [
            self.data_dir / tf_norm / f"{self.split}_scaled.parquet",
            self.data_dir / f"{self.split}_{tf_norm}_scaled.parquet",
            self.data_dir / f"features_{tf_norm}.parquet",
        ]

    def _detect_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Auto-detect feature columns by excluding metadata, labels, and weights.

        Args:
            df: DataFrame to analyze

        Returns:
            List of detected feature column names
        """
        feature_cols = []
        for col in df.columns:
            col_lower = col.lower()

            # Skip known metadata columns
            if col_lower in _METADATA_COLUMNS:
                continue

            # Skip label patterns
            if any(col_lower.startswith(pattern) for pattern in _LABEL_PATTERNS):
                continue

            # Skip weight patterns
            if any(pattern in col_lower for pattern in _WEIGHT_PATTERNS):
                continue

            feature_cols.append(col)

        logger.debug(
            "Auto-detected %d feature columns from %d total columns",
            len(feature_cols),
            len(df.columns),
        )
        return feature_cols

    def _set_anchor_timeframe(self) -> None:
        """
        Set the anchor timeframe to the smallest loaded timeframe by minutes.

        The anchor timeframe determines the timestamp grid for alignment.
        Higher timeframes will be forward-filled to match anchor timestamps.
        """
        if not self._timeframe_data:
            self._anchor_timeframe = None
            self._common_timestamps = None
            return

        # Find smallest timeframe by minutes
        tf_minutes = [(tf, get_timeframe_minutes(tf)) for tf in self._timeframe_data.keys()]
        tf_minutes.sort(key=lambda x: x[1])
        self._anchor_timeframe = tf_minutes[0][0]

        # Extract anchor timestamps
        anchor_data = self._timeframe_data[self._anchor_timeframe]
        if isinstance(anchor_data.df.index, pd.DatetimeIndex):
            self._common_timestamps = anchor_data.df.index
        elif "timestamp" in anchor_data.df.columns:
            self._common_timestamps = pd.DatetimeIndex(anchor_data.df["timestamp"])
        else:
            self._common_timestamps = None

        logger.info(
            "Anchor timeframe set to %s (%d minutes)",
            self._anchor_timeframe,
            get_timeframe_minutes(self._anchor_timeframe),
        )

    @property
    def loaded_timeframes(self) -> list[str]:
        """Return list of loaded timeframe keys."""
        return list(self._timeframe_data.keys())

    @property
    def anchor_timeframe(self) -> str | None:
        """Return the anchor timeframe (smallest loaded TF)."""
        return self._anchor_timeframe

    def get_timeframe_data(self, timeframe: str) -> TimeframeData:
        """
        Get the TimeframeData container for a specific timeframe.

        Args:
            timeframe: Timeframe string (will be normalized)

        Returns:
            TimeframeData for the requested timeframe

        Raises:
            KeyError: If timeframe not loaded
        """
        tf_norm = normalize_timeframe(timeframe)
        if tf_norm not in self._timeframe_data:
            raise KeyError(
                f"Timeframe '{tf_norm}' not loaded. " f"Available: {self.loaded_timeframes}"
            )
        return self._timeframe_data[tf_norm]

    def get_data_for_model(
        self,
        model_name: str,
        contract: ModelContract | None = None,
    ) -> pd.DataFrame:
        """
        Get DataFrame for a model based on its contract's primary_timeframe.

        If the model's primary timeframe is not loaded, raises an error.

        Args:
            model_name: Model name for logging and contract lookup
            contract: Optional ModelContract. If None, looks up via get_model_contract()

        Returns:
            DataFrame with features for the model's primary timeframe

        Raises:
            ValueError: If model contract not found
            KeyError: If required timeframe not loaded
        """
        if contract is None:
            contract = get_model_contract(model_name)

        primary_tf = normalize_timeframe(contract.primary_timeframe)

        if primary_tf not in self._timeframe_data:
            raise KeyError(
                f"Model '{model_name}' requires timeframe '{primary_tf}' which is not loaded. "
                f"Loaded: {self.loaded_timeframes}"
            )

        tf_data = self._timeframe_data[primary_tf]
        logger.debug(
            "Returning data for model %s: timeframe=%s, n_samples=%d, n_features=%d",
            model_name,
            primary_tf,
            tf_data.n_samples,
            len(tf_data.feature_columns),
        )

        return tf_data.df[tf_data.feature_columns].copy()

    def get_multi_stream_dfs(
        self,
        timeframes: list[str],
        align_timestamps: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Get DataFrames for multiple timeframes, optionally aligned to anchor timestamps.

        When aligned, higher timeframes are forward-filled to match the anchor
        timeframe's timestamp grid. This ensures no lookahead bias.

        Args:
            timeframes: List of timeframe strings
            align_timestamps: If True, align all DataFrames to anchor timestamps.
                             Default: True

        Returns:
            Dictionary mapping timeframe -> DataFrame

        Raises:
            KeyError: If any requested timeframe is not loaded
        """
        result = {}
        for tf in timeframes:
            tf_norm = normalize_timeframe(tf)
            if tf_norm not in self._timeframe_data:
                raise KeyError(
                    f"Timeframe '{tf_norm}' not loaded. Available: {self.loaded_timeframes}"
                )
            tf_data = self._timeframe_data[tf_norm]
            result[tf_norm] = tf_data.df[tf_data.feature_columns].copy()

        if align_timestamps and self._anchor_timeframe is not None:
            result = self._align_dataframes(result)

        return result

    def _align_dataframes(
        self,
        dfs: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """
        Forward-fill higher timeframes to anchor timestamps.

        Uses forward-fill (ffill) to propagate higher-TF values to the anchor
        timestamp grid. This ensures no lookahead: at each anchor timestamp,
        only information available up to that point is used.

        Args:
            dfs: Dictionary mapping timeframe -> DataFrame

        Returns:
            Dictionary with all DataFrames aligned to anchor timestamps
        """
        if self._common_timestamps is None:
            logger.warning(
                "No common timestamps available for alignment; returning unaligned DataFrames"
            )
            return dfs

        anchor_tf = self._anchor_timeframe
        anchor_minutes = get_timeframe_minutes(anchor_tf)

        aligned = {}
        for tf, df in dfs.items():
            tf_minutes = get_timeframe_minutes(tf)

            if tf_minutes == anchor_minutes:
                # Anchor timeframe - no alignment needed
                aligned[tf] = df
                logger.debug("Timeframe %s is anchor, no alignment needed", tf)
            else:
                # Higher timeframe - reindex and forward-fill
                logger.debug(
                    "Aligning timeframe %s (%d min) to anchor %s (%d min)",
                    tf,
                    tf_minutes,
                    anchor_tf,
                    anchor_minutes,
                )

                # Ensure DataFrame has datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    if "timestamp" in df.columns:
                        df = df.set_index("timestamp")
                    else:
                        logger.warning(
                            "Cannot align timeframe %s: no datetime index or timestamp column",
                            tf,
                        )
                        aligned[tf] = df
                        continue

                # Reindex to common timestamps with forward-fill
                df_aligned = df.reindex(self._common_timestamps, method="ffill")

                # Drop rows where forward-fill couldn't populate (at the start)
                df_aligned = df_aligned.dropna()

                aligned[tf] = df_aligned
                logger.debug(
                    "Aligned %s: %d -> %d samples",
                    tf,
                    len(df),
                    len(df_aligned),
                )

        return aligned

    def get_required_timeframes_for_ensemble(
        self,
        base_models: list[str],
    ) -> set[str]:
        """
        Get all timeframes required by a list of base models.

        Analyzes each model's contract to determine primary and MTF timeframes needed.

        Args:
            base_models: List of model names

        Returns:
            Set of all required timeframe strings (normalized)
        """
        required = set()

        for model_name in base_models:
            try:
                contract = get_model_contract(model_name)
            except ValueError:
                logger.warning(
                    "No contract found for model '%s', skipping timeframe detection",
                    model_name,
                )
                continue

            # Add primary timeframe
            primary = normalize_timeframe(contract.primary_timeframe)
            required.add(primary)

            # Add MTF timeframes if specified
            for mtf_tf in contract.mtf_timeframes:
                required.add(normalize_timeframe(mtf_tf))

        logger.debug(
            "Required timeframes for ensemble %s: %s",
            base_models,
            sorted(required),
        )
        return required

    def validate_timeframe_coverage(
        self,
        required: set[str],
    ) -> tuple[bool, list[str]]:
        """
        Check if all required timeframes are loaded.

        Args:
            required: Set of required timeframe strings (normalized)

        Returns:
            Tuple of (is_valid, list_of_missing_timeframes)
        """
        loaded = set(self.loaded_timeframes)
        missing = [tf for tf in required if normalize_timeframe(tf) not in loaded]

        is_valid = len(missing) == 0
        if not is_valid:
            logger.warning(
                "Missing required timeframes: %s. Loaded: %s",
                missing,
                self.loaded_timeframes,
            )

        return is_valid, missing

    def get_feature_columns_for_model(
        self,
        model_name: str,
        contract: ModelContract | None = None,
    ) -> list[str]:
        """
        Get the feature columns for a model's primary timeframe.

        Args:
            model_name: Model name for contract lookup
            contract: Optional ModelContract. If None, looks up via get_model_contract()

        Returns:
            List of feature column names

        Raises:
            ValueError: If model contract not found
            KeyError: If required timeframe not loaded
        """
        if contract is None:
            contract = get_model_contract(model_name)

        primary_tf = normalize_timeframe(contract.primary_timeframe)

        if primary_tf not in self._timeframe_data:
            raise KeyError(
                f"Model '{model_name}' requires timeframe '{primary_tf}' which is not loaded. "
                f"Loaded: {self.loaded_timeframes}"
            )

        return self._timeframe_data[primary_tf].feature_columns.copy()


__all__ = [
    "TimeframeData",
    "TimeframeCoordinator",
]
