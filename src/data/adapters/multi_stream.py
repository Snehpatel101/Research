"""
Multi-Stream Adapter - Convert multi-timeframe DataFrames to 4D arrays.

Phase 2 SNwH Implementation.

This adapter transforms multiple timeframe DataFrames into 4D arrays
with shape (n_samples, n_timeframes, seq_len, n_features) for models
that support multi-stream MTF mode like PatchTST and iTransformer.

Integration with Raw MTF Store
-------------------------------
This adapter can load multi-timeframe data directly from the raw MTF store
(created during pipeline execution). When `symbol` and `split` are provided
and `additional_dfs` is not, the adapter will automatically load all
required timeframes from the store.

Usage with Store:
    >>> adapter = MultiStreamAdapter(
    ...     symbol="MES",
    ...     split="train",
    ...     timeframes=["1min", "5min", "15min"],
    ... )
    >>> result = adapter.transform(df)  # df is anchor, others loaded from store

Usage with additional_dfs (legacy):
    >>> adapter = MultiStreamAdapter(timeframes=["1min", "5min", "15min"])
    >>> result = adapter.transform(df, additional_dfs={"5min": df_5, "15min": df_15})
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.core.common.timeframes import get_timeframe_minutes, normalize_timeframe
from src.core.contracts import DataContract, DataRank
from src.data.store import TIMEFRAMES as STORE_TIMEFRAMES
from src.data.store import load_all_timeframes, load_raw_mtf

from .base import AdapterResult, BaseAdapter
from .registry import AdapterRegistry

if TYPE_CHECKING:
    from src.core.contracts import ModelContract

logger = logging.getLogger(__name__)


@AdapterRegistry.register("multi_stream")
class MultiStreamAdapter(BaseAdapter):
    """
    Adapter for multi-timeframe 4D data.

    Transforms multiple timeframe DataFrames into a 4D numpy array
    with shape (n_samples, n_timeframes, seq_len, n_features).

    The anchor timeframe (smallest, first in list) determines the
    sample indices. Higher timeframes are aligned using ratio-based
    index mapping.

    This adapter is used by models with MTFMode.MULTI_STREAM such as
    PatchTST and iTransformer that benefit from seeing raw OHLCV data
    at multiple resolutions simultaneously.

    Integration with Raw MTF Store:
        When `symbol` and `split` are provided, the adapter can automatically
        load timeframe data from the raw MTF store (data/canonical/raw_mtf/).
        This is the recommended approach for 4D models.

    Attributes:
        adapter_id: Registry identifier ("multi_stream")
        output_rank: Output data rank (DataRank.MULTI_TF_4D)

    Example with store (recommended):
        >>> adapter = MultiStreamAdapter(
        ...     symbol="MES",
        ...     split="train",
        ...     timeframes=["1min", "5min", "15min"],
        ...     sequence_length=60,
        ... )
        >>> result = adapter.transform(df_anchor)
        >>> result.X.shape  # (n_sequences, 3, 60, 5)

    Example with additional_dfs (legacy):
        >>> adapter = MultiStreamAdapter(
        ...     feature_columns=["open", "high", "low", "close"],
        ...     timeframes=["1min", "5min", "15min"],
        ...     sequence_length=60,
        ... )
        >>> result = adapter.transform(df_1min, additional_dfs={
        ...     "5min": df_5min,
        ...     "15min": df_15min,
        ... })
        >>> result.X.shape  # (n_sequences, 3, 60, 4)
    """

    adapter_id: str = "multi_stream"
    output_rank: DataRank = DataRank.MULTI_TF_4D

    # Default raw OHLCV features for multi-stream models
    DEFAULT_FEATURE_COLUMNS = ["open", "high", "low", "close", "volume"]
    DEFAULT_TIMEFRAMES = ["1min", "5min", "15min"]

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        label_column: str = "label_h20",
        weight_column: str | None = "sample_weight_h20",
        sequence_length: int = 60,
        stride: int = 1,
        timeframes: list[str] | None = None,
        data_dir: Path | str | None = None,
        symbol: str | None = None,
        split: str | None = None,
        base_path: str | Path = "data/canonical",
    ):
        """
        Initialize the multi-stream adapter.

        Args:
            feature_columns: Feature columns to extract from each timeframe.
                Defaults to ["open", "high", "low", "close", "volume"] for
                raw OHLCV which is typical for multi-stream models.
            label_column: Column name for labels in the anchor DataFrame.
            weight_column: Column name for sample weights (None for uniform).
            sequence_length: Length of each sequence window.
            stride: Step size between consecutive sequences (1 = sliding window).
            timeframes: List of timeframes to use. First is anchor (smallest).
                Defaults to ["1min", "5min", "15min"].
            data_dir: Directory to load additional timeframe parquet files from
                if not provided in additional_dfs (legacy mode).
            symbol: Trading symbol (e.g., "MES") for loading from raw MTF store.
            split: Data split ("train", "val", "test") for loading from raw MTF store.
            base_path: Base path for the raw MTF store. Default: "data/canonical".
        """
        # Use raw OHLCV by default for multi-stream models
        if feature_columns is None:
            feature_columns = self.DEFAULT_FEATURE_COLUMNS.copy()

        super().__init__(
            feature_columns=feature_columns,
            label_column=label_column,
            weight_column=weight_column,
        )

        self.sequence_length = sequence_length
        self.stride = stride
        self.timeframes = (
            [normalize_timeframe(tf) for tf in timeframes]
            if timeframes
            else self.DEFAULT_TIMEFRAMES.copy()
        )
        self.data_dir = Path(data_dir) if data_dir else None

        # Raw MTF store integration
        self.symbol = symbol
        self.split = split
        self.base_path = Path(base_path)

    def transform(
        self,
        df: pd.DataFrame,
        model_contract: ModelContract | None = None,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> AdapterResult:
        """
        Transform multi-timeframe DataFrames to 4D array.

        Args:
            df: Primary DataFrame for the anchor timeframe (smallest TF).
            model_contract: Optional model contract to override timeframes
                and sequence_length from mtf_timeframes and sequence_length.
            additional_dfs: Dictionary mapping timeframe strings to DataFrames
                for the other timeframes. If not provided, attempts to load
                from data_dir.

        Returns:
            AdapterResult with:
                - X: shape (n_sequences, n_timeframes, seq_len, n_features)
                - y: shape (n_sequences,)
                - weights: shape (n_sequences,) or None
                - original_indices: indices into anchor DataFrame

        Raises:
            ValueError: If required timeframe data is missing or invalid.
        """
        # Validate input
        is_valid, issues = self.validate_input(df)
        if not is_valid:
            raise ValueError(f"Input validation failed: {issues}")

        # Override parameters from model contract if provided
        timeframes = self._resolve_timeframes(model_contract)
        seq_len = self._resolve_sequence_length(model_contract)
        feature_cols = self._resolve_feature_columns(df, model_contract)

        # Anchor timeframe is first (smallest)
        anchor_tf = timeframes[0]

        # Collect all timeframe DataFrames
        tf_dfs = self._collect_timeframe_dfs(
            df, anchor_tf, timeframes, additional_dfs, feature_cols
        )

        # Build 4D array
        X, y, weights, original_indices = self._build_multi_stream(
            tf_dfs=tf_dfs,
            timeframes=timeframes,
            feature_cols=feature_cols,
            seq_len=seq_len,
        )

        # Create data contract for 4D multi-stream data
        data_contract = DataContract.from_array(
            X=X,
            symbol=self._get_metadata_value(df, "symbol", "unknown"),
            timeframe=anchor_tf,
            horizon=self._parse_horizon_from_label_column(self.label_column),
            split=self._get_metadata_value(df, "split", "unknown"),
            feature_columns=feature_cols,
        )

        # Validate data contract against model requirements if provided
        if model_contract is not None:
            model_contract.validate_data_contract_strict(data_contract)

        return AdapterResult(
            X=X,
            y=y,
            weights=weights,
            n_samples=X.shape[0],
            n_features=X.shape[3],
            data_rank=DataRank.MULTI_TF_4D,
            sequence_length=seq_len,
            n_timeframes=len(timeframes),
            timeframe_names=timeframes,
            original_indices=original_indices,
            feature_columns=feature_cols,
            data_contract=data_contract,
            adapter_name=self.adapter_id,
        )

    def _resolve_timeframes(self, model_contract: ModelContract | None) -> list[str]:
        """Resolve timeframes from contract or instance config."""
        if model_contract is not None and model_contract.mtf_timeframes:
            # Contract specifies primary_timeframe + mtf_timeframes
            primary = normalize_timeframe(model_contract.primary_timeframe)
            mtf = [normalize_timeframe(tf) for tf in model_contract.mtf_timeframes]
            return [primary] + mtf

        return self.timeframes

    def _resolve_sequence_length(self, model_contract: ModelContract | None) -> int:
        """Resolve sequence length from contract or instance config."""
        if model_contract is not None:
            return model_contract.sequence_length
        return self.sequence_length

    def _resolve_feature_columns(
        self,
        df: pd.DataFrame,
        model_contract: ModelContract | None,
    ) -> list[str]:
        """Resolve feature columns, validating they exist."""
        feature_cols = self.feature_columns or self.DEFAULT_FEATURE_COLUMNS.copy()

        # For multi-stream, we typically use raw OHLCV so don't auto-detect
        # but we do validate the columns exist
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise ValueError(
                f"Missing feature columns in anchor DataFrame: {missing}. "
                f"Available columns: {list(df.columns)[:20]}..."
            )

        return feature_cols

    def _collect_timeframe_dfs(
        self,
        df: pd.DataFrame,
        anchor_tf: str,
        timeframes: list[str],
        additional_dfs: dict[str, pd.DataFrame] | None,
        feature_cols: list[str],
    ) -> dict[str, pd.DataFrame]:
        """
        Collect DataFrames for all timeframes.

        Tries loading in this order:
        1. Primary DataFrame (for anchor timeframe)
        2. additional_dfs if provided
        3. Raw MTF store if symbol and split are configured
        4. Legacy data_dir fallback

        Args:
            df: Primary DataFrame (anchor timeframe).
            anchor_tf: Anchor timeframe string.
            timeframes: All timeframes to collect.
            additional_dfs: Pre-loaded DataFrames for other timeframes.
            feature_cols: Feature columns that must exist.

        Returns:
            Dictionary mapping timeframe -> DataFrame.

        Raises:
            ValueError: If a required timeframe cannot be loaded.
        """
        additional_dfs = additional_dfs or {}
        tf_dfs: dict[str, pd.DataFrame] = {}

        for tf in timeframes:
            if tf == anchor_tf:
                # Primary DataFrame
                tf_dfs[tf] = df
            elif tf in additional_dfs:
                # Provided in additional_dfs
                tf_dfs[tf] = additional_dfs[tf]
            elif self.symbol is not None and self.split is not None:
                # Load from raw MTF store
                tf_df = self._load_timeframe_from_store(tf)
                if tf_df is not None:
                    tf_dfs[tf] = tf_df
                else:
                    raise ValueError(
                        f"Cannot find data for timeframe '{tf}' in raw MTF store. "
                        f"Expected at: {self.base_path}/raw_mtf/{self.symbol}_{tf}_{self.split}.parquet"
                    )
            elif self.data_dir is not None:
                # Legacy: Try to load from data_dir
                tf_df = self._load_timeframe_from_dir(tf, feature_cols)
                if tf_df is not None:
                    tf_dfs[tf] = tf_df
                else:
                    raise ValueError(
                        f"Cannot find data for timeframe '{tf}'. "
                        f"Either provide in additional_dfs or ensure "
                        f"'{tf}' parquet exists in data_dir={self.data_dir}"
                    )
            else:
                raise ValueError(
                    f"No data for timeframe '{tf}'. Options: "
                    f"1) Provide in additional_dfs, "
                    f"2) Set symbol and split for raw MTF store, "
                    f"3) Set data_dir for legacy loading."
                )

            # Validate feature columns exist in this timeframe's DataFrame
            tf_df = tf_dfs[tf]
            missing = [col for col in feature_cols if col not in tf_df.columns]
            if missing:
                raise ValueError(f"Timeframe '{tf}' DataFrame missing feature columns: {missing}")

        return tf_dfs

    def _load_timeframe_from_store(self, timeframe: str) -> pd.DataFrame | None:
        """
        Load a timeframe DataFrame from the raw MTF store.

        Uses the raw_mtf_store module to load pre-saved OHLCV data
        for the specified timeframe, symbol, and split.

        Args:
            timeframe: Timeframe string (e.g., "5min", "15min").

        Returns:
            DataFrame if found, None otherwise.
        """
        if self.symbol is None or self.split is None:
            return None

        try:
            df = load_raw_mtf(
                symbol=self.symbol,
                timeframe=timeframe,
                split=self.split,
                base_path=self.base_path,
            )
            logger.info(
                f"Loaded timeframe '{timeframe}' from raw MTF store: "
                f"{self.symbol}/{self.split} ({len(df)} rows)"
            )
            return df
        except Exception as e:
            logger.debug(f"Failed to load {timeframe} from raw MTF store: {e}")
            return None

    def _load_timeframe_from_dir(
        self, timeframe: str, feature_cols: list[str]
    ) -> pd.DataFrame | None:
        """
        Attempt to load a timeframe DataFrame from data_dir.

        Looks for parquet files matching patterns like:
        - {timeframe}.parquet
        - features_{timeframe}.parquet
        - scaled_{timeframe}.parquet

        Args:
            timeframe: Timeframe string (e.g., "5min").
            feature_cols: Feature columns that must exist.

        Returns:
            DataFrame if found and valid, None otherwise.
        """
        if self.data_dir is None:
            return None

        # Try common naming patterns
        patterns = [
            f"{timeframe}.parquet",
            f"features_{timeframe}.parquet",
            f"scaled_{timeframe}.parquet",
            f"data_{timeframe}.parquet",
        ]

        for pattern in patterns:
            path = self.data_dir / pattern
            if path.exists():
                try:
                    df = pd.read_parquet(path)
                    # Check it has required columns
                    if all(col in df.columns for col in feature_cols):
                        logger.info(f"Loaded timeframe '{timeframe}' from {path}")
                        return df
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
                    continue

        return None

    def _build_multi_stream(
        self,
        tf_dfs: dict[str, pd.DataFrame],
        timeframes: list[str],
        feature_cols: list[str],
        seq_len: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
        """
        Build 4D multi-stream array from timeframe DataFrames.

        The anchor timeframe (first in list, smallest granularity) determines
        the number of samples. Higher timeframes are aligned by mapping
        anchor indices to their corresponding indices using the timeframe ratio.

        Args:
            tf_dfs: Dictionary of timeframe -> DataFrame.
            timeframes: Ordered list of timeframes (anchor first).
            feature_cols: Feature columns to extract.
            seq_len: Sequence length for each window.

        Returns:
            Tuple of (X, y, weights, original_indices):
                - X: shape (n_sequences, n_timeframes, seq_len, n_features)
                - y: shape (n_sequences,)
                - weights: shape (n_sequences,) or None
                - original_indices: indices into anchor DataFrame
        """
        anchor_tf = timeframes[0]
        anchor_df = tf_dfs[anchor_tf]
        anchor_minutes = get_timeframe_minutes(anchor_tf)

        n_samples = len(anchor_df)
        n_tfs = len(timeframes)
        n_features = len(feature_cols)

        # Calculate number of valid sequences from anchor timeframe
        n_sequences = (n_samples - seq_len) // self.stride + 1
        if n_sequences <= 0:
            raise ValueError(
                f"Not enough samples ({n_samples}) for sequence_length={seq_len}. "
                f"Need at least {seq_len} samples."
            )

        # Pre-allocate output array
        X = np.zeros((n_sequences, n_tfs, seq_len, n_features), dtype=np.float32)
        y = np.zeros(n_sequences, dtype=np.int64)
        original_indices = np.zeros(n_sequences, dtype=np.int64)

        # Extract weights if available
        weights = self._get_weights(anchor_df)

        # Build sequences for each timeframe
        for tf_idx, tf in enumerate(timeframes):
            tf_df = tf_dfs[tf]
            tf_minutes = get_timeframe_minutes(tf)
            remainder = tf_minutes % anchor_minutes
            if remainder != 0:
                # Phase 31: Proper handling for non-integer ratios
                # Use ceiling to ensure we don't miss data from higher timeframes
                ratio = (tf_minutes + anchor_minutes - 1) // anchor_minutes  # ceiling division
                logger.warning(
                    f"Timeframe {tf} ({tf_minutes}min) is not an exact multiple of "
                    f"anchor ({anchor_minutes}min). Using ceiling ratio={ratio} "
                    f"(remainder={remainder}min). Consider using compatible timeframes."
                )
            else:
                ratio = tf_minutes // anchor_minutes

            # Extract feature values as numpy array for efficiency
            tf_values = tf_df[feature_cols].values.astype(np.float32)
            tf_n_samples = len(tf_values)

            for seq_idx in range(n_sequences):
                # Anchor indices for this sequence
                anchor_start = seq_idx * self.stride
                anchor_end = anchor_start + seq_len

                if tf_idx == 0:
                    # Anchor timeframe - direct extraction
                    X[seq_idx, tf_idx, :, :] = tf_values[anchor_start:anchor_end]

                    # Store label from end of sequence (target is next step)
                    y[seq_idx] = anchor_df[self.label_column].iloc[anchor_end - 1]
                    original_indices[seq_idx] = anchor_end - 1
                else:
                    # Higher timeframe - map indices
                    X[seq_idx, tf_idx, :, :] = self._extract_higher_tf_sequence(
                        tf_values=tf_values,
                        tf_n_samples=tf_n_samples,
                        anchor_start=anchor_start,
                        anchor_end=anchor_end,
                        ratio=ratio,
                        seq_len=seq_len,
                    )

        # Extract weights for sequences if available
        seq_weights = None
        if weights is not None:
            seq_weights = np.zeros(n_sequences, dtype=np.float32)
            for seq_idx in range(n_sequences):
                # Use weight at sequence end point
                anchor_end = seq_idx * self.stride + seq_len
                seq_weights[seq_idx] = weights[anchor_end - 1]

        return X, y, seq_weights, original_indices

    def _extract_higher_tf_sequence(
        self,
        tf_values: np.ndarray,
        tf_n_samples: int,
        anchor_start: int,
        anchor_end: int,
        ratio: int,
        seq_len: int,
    ) -> np.ndarray:
        """
        Extract a sequence from a higher timeframe aligned to anchor indices.

        Maps anchor timeframe indices to higher timeframe indices using the
        ratio between timeframes. Handles boundary cases with padding.

        Args:
            tf_values: Feature values for the higher timeframe.
            tf_n_samples: Number of samples in higher timeframe.
            anchor_start: Start index in anchor timeframe.
            anchor_end: End index in anchor timeframe.
            ratio: Ratio of higher TF minutes to anchor minutes.
            seq_len: Desired sequence length.

        Returns:
            Sequence array of shape (seq_len, n_features).
        """
        n_features = tf_values.shape[1]

        # Map anchor indices to higher timeframe
        tf_start = anchor_start // ratio
        tf_end = anchor_end // ratio

        # Calculate how many higher-TF bars cover the anchor sequence
        tf_seq_len = tf_end - tf_start
        if tf_seq_len <= 0:
            tf_seq_len = 1

        # Extract available data
        actual_start = max(0, tf_start)
        actual_end = min(tf_n_samples, tf_start + seq_len)

        if actual_start >= tf_n_samples:
            # All padding needed - beyond available data
            logger.warning(
                f"Higher TF sequence fully out of bounds: "
                f"tf_start={tf_start}, tf_n_samples={tf_n_samples}"
            )
            return np.zeros((seq_len, n_features), dtype=np.float32)

        # Extract what we can
        available = tf_values[actual_start:actual_end]
        available_len = len(available)

        # Handle case where higher TF has fewer bars than seq_len
        if available_len < seq_len:
            # Resample/repeat to fill sequence length
            result = self._resample_to_length(available, seq_len)
        elif available_len > seq_len:
            # Take last seq_len bars (most recent aligned with anchor end)
            result = available[-seq_len:]
        else:
            result = available

        return result.astype(np.float32)

    def _resample_to_length(self, arr: np.ndarray, target_len: int) -> np.ndarray:
        """
        Resample array to target length using repetition.

        For multi-stream, we typically repeat/pad rather than interpolate
        since we want to preserve the discrete bar values.

        Args:
            arr: Input array of shape (n, n_features).
            target_len: Desired length.

        Returns:
            Resampled array of shape (target_len, n_features).
        """
        current_len = len(arr)
        if current_len == 0:
            return np.zeros((target_len, arr.shape[1] if arr.ndim > 1 else 1), dtype=np.float32)

        if current_len >= target_len:
            return arr[:target_len]

        # Calculate repetition factor for each bar
        # E.g., if we have 5 bars and need 15, each bar repeats 3 times
        repeat_factor = target_len // current_len
        remainder = target_len % current_len

        # Repeat each bar
        result = np.repeat(arr, repeat_factor, axis=0)

        # Add remainder from the end (most recent bars)
        if remainder > 0:
            result = np.concatenate(
                [
                    arr[-remainder:],  # Pad at start with repeated end values
                    result,
                ],
                axis=0,
            )

        return result[:target_len]

    # NOTE: _get_metadata_value and _parse_horizon_from_label_column
    # are now inherited from BaseAdapter (Phase 31 consolidation)

    @classmethod
    def from_store(
        cls,
        symbol: str,
        split: str,
        timeframes: list[str] | None = None,
        sequence_length: int = 60,
        base_path: str | Path = "data/canonical",
        **kwargs,
    ) -> MultiStreamAdapter:
        """
        Create a MultiStreamAdapter configured to load from the raw MTF store.

        This is the recommended factory method for 4D models (PatchTST, iTransformer)
        that need multi-timeframe OHLCV data.

        Args:
            symbol: Trading symbol (e.g., "MES", "MGC").
            split: Data split ("train", "val", "test").
            timeframes: List of timeframes to use. If None, uses all standard
                timeframes from the store: ["1min", "3min", "5min", "10min",
                "15min", "30min", "60min", "2h", "4h"].
            sequence_length: Length of each sequence window.
            base_path: Base path for the raw MTF store.
            **kwargs: Additional arguments passed to constructor.

        Returns:
            Configured MultiStreamAdapter instance.

        Example:
            >>> adapter = MultiStreamAdapter.from_store(
            ...     symbol="MES",
            ...     split="train",
            ...     timeframes=["1min", "5min", "15min", "60min"],
            ...     sequence_length=60,
            ... )
            >>> result = adapter.transform(anchor_df)
        """
        # Use all store timeframes if not specified
        if timeframes is None:
            timeframes = STORE_TIMEFRAMES.copy()

        return cls(
            symbol=symbol,
            split=split,
            timeframes=timeframes,
            sequence_length=sequence_length,
            base_path=base_path,
            **kwargs,
        )

    def load_all_from_store(self) -> dict[str, pd.DataFrame]:
        """
        Load all configured timeframes from the raw MTF store.

        This is useful when you want to pre-load all data before calling transform().

        Returns:
            Dictionary mapping timeframe -> DataFrame.

        Raises:
            ValueError: If symbol or split not configured.
            TimeframeNotFoundError: If any timeframe is missing from store.

        Example:
            >>> adapter = MultiStreamAdapter.from_store("MES", "train")
            >>> all_data = adapter.load_all_from_store()
            >>> print(list(all_data.keys()))
            ['1min', '3min', '5min', ...]
        """
        if self.symbol is None or self.split is None:
            raise ValueError(
                "Cannot load from store: symbol and split must be configured. "
                "Use MultiStreamAdapter.from_store() to create a store-configured adapter."
            )

        return load_all_timeframes(
            symbol=self.symbol,
            split=self.split,
            base_path=self.base_path,
            timeframes=self.timeframes,
            missing_ok=False,
        )


# Re-export store timeframes for convenience
MTF_TIMEFRAMES = STORE_TIMEFRAMES


__all__ = [
    "MultiStreamAdapter",
    "MTF_TIMEFRAMES",
]
