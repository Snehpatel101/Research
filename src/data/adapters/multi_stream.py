"""
Multi-Stream Adapter - Convert multi-timeframe DataFrames to 4D arrays.

Phase 2 SNwH Implementation.

This adapter transforms multiple timeframe DataFrames into 4D arrays
with shape (n_samples, n_timeframes, seq_len, n_features) for models
that support multi-stream MTF mode like PatchTST and iTransformer.

Timeframes come from ``additional_dfs`` (MLFactory resamples the raw bars itself):

    >>> adapter = MultiStreamAdapter(timeframes=["1min", "5min", "15min"])
    >>> result = adapter.transform(df, additional_dfs={"5min": df_5, "15min": df_15})
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.core.common.timeframes import normalize_timeframe
from src.core.constants import DEFAULT_MTF_TIMEFRAMES
from src.core.contracts import DataContract, DataRank

from .base import AdapterResult, BaseAdapter
from .registry import AdapterRegistry

if TYPE_CHECKING:
    from src.core.contracts import ModelContract

logger = logging.getLogger(__name__)

_OHLCV_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def resample_higher_timeframes(
    bars: pd.DataFrame, timeframes: list[str]
) -> dict[str, pd.DataFrame]:
    """
    The multi-stream model's higher-timeframe OHLCV streams, from its own bars.

    Training (``MLFactory``) and serving (``ModelBundle``) both call this on the
    bars at the training bar timeframe, so every stream is built from the same
    rows. Each stream is lagged one bar: a row stamped T holds the higher-TF
    bar that COMPLETED before T, never the still-forming one.

    Args:
        bars: OHLCV bars with a DatetimeIndex (the model's anchor stream).
        timeframes: Higher timeframes to build.

    Returns:
        Dict of normalized timeframe key -> lagged OHLCV DataFrame.
    """
    streams: dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        resampled = (
            bars[list(_OHLCV_AGG)]
            .resample(tf, closed="left", label="left")
            .agg(_OHLCV_AGG)
            .dropna()
        )
        streams[normalize_timeframe(tf)] = resampled.shift(1).dropna()
    return streams


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

    Attributes:
        adapter_id: Registry identifier ("multi_stream")
        output_rank: Output data rank (DataRank.MULTI_TF_4D)

    Example:
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
    # Import from constants.py for single source of truth
    DEFAULT_TIMEFRAMES = DEFAULT_MTF_TIMEFRAMES

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        label_column: str = "label_h20",
        weight_column: str | None = "sample_weight_h20",
        sequence_length: int = 60,
        stride: int = 1,
        timeframes: list[str] | None = None,
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
                for the other timeframes (required).

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

        The anchor timeframe is the primary DataFrame; every other timeframe comes
        from ``additional_dfs``.

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
            else:
                raise ValueError(
                    f"No data for timeframe '{tf}': provide it in additional_dfs "
                    f"(available: {sorted(additional_dfs)})"
                )

            # Validate feature columns exist in this timeframe's DataFrame
            tf_df = tf_dfs[tf]
            missing = [col for col in feature_cols if col not in tf_df.columns]
            if missing:
                raise ValueError(f"Timeframe '{tf}' DataFrame missing feature columns: {missing}")

        return tf_dfs

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
        the number of samples. Higher timeframes are aligned using timestamp-based
        mapping (merge_asof) that correctly handles irregular gaps (overnight,
        weekends, holidays).

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

        # Pre-compute timestamp-based index mappings for higher timeframes
        # (Critical Fix #7: replaces ratio-based mapping that ignores gaps)
        tf_index_maps = self._build_timestamp_index_maps(
            anchor_df, tf_dfs, timeframes, feature_cols
        )

        # Pre-compute all anchor start/end indices (vectorized)
        anchor_starts = np.arange(n_sequences) * self.stride
        anchor_ends = anchor_starts + seq_len

        # Build sequences for each timeframe
        for tf_idx, tf in enumerate(timeframes):
            tf_values = tf_index_maps[tf]["values"]

            if tf_idx == 0:
                # Anchor timeframe — vectorized extraction using fancy indexing
                # Build index array: (n_sequences, seq_len) where each row is
                # [anchor_start, anchor_start+1, ..., anchor_start+seq_len-1]
                seq_indices = anchor_starts[:, np.newaxis] + np.arange(seq_len)[np.newaxis, :]
                X[:, tf_idx, :, :] = tf_values[seq_indices]

                # Store labels and original indices (vectorized)
                label_values = anchor_df[self.label_column].values
                y[:] = label_values[anchor_ends - 1]
                original_indices[:] = anchor_ends - 1
            else:
                # Higher timeframe - inline vectorized alignment (opt 2.6)
                # Eliminates per-iteration method dispatch overhead while
                # preserving the deduplication + forward-fill padding logic
                # that _extract_aligned_sequence implements.
                idx_map = tf_index_maps[tf]["anchor_to_tf"]

                if len(tf_values) == 0:
                    # Guard: no higher-TF bars in this split — fill NaN
                    X[:, tf_idx, :, :] = np.nan
                else:
                    for seq_idx in range(n_sequences):
                        a_start = anchor_starts[seq_idx]
                        a_end = anchor_ends[seq_idx]
                        mapped = idx_map[a_start:a_end]

                        # Deduplicate consecutive identical indices (vectorized)
                        unique_mask = np.empty(len(mapped), dtype=np.bool_)
                        unique_mask[0] = True
                        unique_mask[1:] = np.diff(mapped) != 0
                        unique_idx = np.clip(mapped[unique_mask], 0, len(tf_values) - 1)

                        unique_bars = tf_values[unique_idx]
                        n_unique = len(unique_bars)

                        if n_unique >= seq_len:
                            X[seq_idx, tf_idx, :, :] = unique_bars[-seq_len:]
                        elif n_unique > 0:
                            # Pad front with earliest bar, place actuals at end
                            X[seq_idx, tf_idx, -n_unique:, :] = unique_bars
                            X[seq_idx, tf_idx, : seq_len - n_unique, :] = unique_bars[0]
                        else:
                            X[seq_idx, tf_idx, :, :] = np.nan

        # Extract weights for sequences if available (vectorized)
        seq_weights = None
        if weights is not None:
            seq_weights = weights[anchor_ends - 1].astype(np.float32)

        return X, y, seq_weights, original_indices

    def _build_timestamp_index_maps(
        self,
        anchor_df: pd.DataFrame,
        tf_dfs: dict[str, pd.DataFrame],
        timeframes: list[str],
        feature_cols: list[str],
    ) -> dict[str, dict]:
        """
        Build timestamp-based index mappings from anchor to each timeframe.

        For each higher timeframe, uses merge_asof to map each anchor
        timestamp to the most recent higher-TF bar at or before that time.
        This correctly handles irregular gaps (overnight, weekends, holidays).

        Args:
            anchor_df: Anchor timeframe DataFrame (must have DatetimeIndex).
            tf_dfs: All timeframe DataFrames.
            timeframes: Ordered list of timeframes.
            feature_cols: Feature columns to extract.

        Returns:
            Dict mapping timeframe -> {"values": ndarray, "anchor_to_tf": ndarray}
            where anchor_to_tf[i] gives the higher-TF index for anchor row i.
        """
        result = {}

        for tf in timeframes:
            tf_df = tf_dfs[tf]
            tf_values = tf_df[feature_cols].values.astype(np.float32)

            if tf == timeframes[0]:
                # Anchor timeframe: identity mapping
                result[tf] = {
                    "values": tf_values,
                    "anchor_to_tf": np.arange(len(tf_values)),
                }
                continue

            # Higher timeframe: timestamp-based alignment
            anchor_has_dt_index = isinstance(anchor_df.index, pd.DatetimeIndex)
            tf_has_dt_index = isinstance(tf_df.index, pd.DatetimeIndex)

            if anchor_has_dt_index and tf_has_dt_index:
                # Proper timestamp alignment using merge_asof
                anchor_to_tf = self._timestamp_align(anchor_df, tf_df)
            else:
                # Ratio-based fallback silently corrupts alignment across
                # overnight/weekend gaps — require DatetimeIndex instead.
                missing_on = []
                if not anchor_has_dt_index:
                    missing_on.append(f"anchor ({type(anchor_df.index).__name__})")
                if not tf_has_dt_index:
                    missing_on.append(f"{tf} ({type(tf_df.index).__name__})")
                raise ValueError(
                    f"MultiStreamAdapter requires DatetimeIndex for accurate "
                    f"cross-timeframe alignment. Missing on: {', '.join(missing_on)}. "
                    f"Convert with df.index = pd.to_datetime(df.index)"
                )

            result[tf] = {
                "values": tf_values,
                "anchor_to_tf": anchor_to_tf,
            }

        return result

    def _timestamp_align(
        self,
        anchor_df: pd.DataFrame,
        higher_tf_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Map each anchor row to the most recent higher-TF bar via merge_asof.

        For each anchor timestamp, finds the higher-TF bar whose timestamp
        is <= the anchor timestamp (backward lookup). This handles gaps
        correctly: if a higher-TF bar is missing due to a gap, the previous
        bar is used rather than producing a misalignment.

        Args:
            anchor_df: Anchor DataFrame with DatetimeIndex.
            higher_tf_df: Higher-TF DataFrame with DatetimeIndex.

        Returns:
            Array of shape (len(anchor_df),) with the higher-TF index
            for each anchor row.
        """
        # Build lookup frames with positional indices
        anchor_ts = anchor_df.index.to_series().reset_index(drop=True)
        anchor_lookup = pd.DataFrame(
            {
                "anchor_ts": anchor_ts,
                "anchor_pos": np.arange(len(anchor_df)),
            }
        )

        higher_ts = higher_tf_df.index.to_series().reset_index(drop=True)
        higher_lookup = pd.DataFrame(
            {
                "higher_ts": higher_ts,
                "higher_pos": np.arange(len(higher_tf_df)),
            }
        )

        # merge_asof: for each anchor timestamp, find the most recent
        # higher-TF bar at or before that time
        merged = pd.merge_asof(
            anchor_lookup.rename(columns={"anchor_ts": "ts"}),
            higher_lookup.rename(columns={"higher_ts": "ts"}),
            on="ts",
            direction="backward",
        )

        # Fill any NaN at the start (anchor bars before first higher-TF bar)
        idx_map = merged["higher_pos"].fillna(0).astype(np.int64).values
        return idx_map

    def _extract_aligned_sequence(
        self,
        tf_values: np.ndarray,
        idx_map: np.ndarray,
        anchor_start: int,
        anchor_end: int,
        seq_len: int,
    ) -> np.ndarray:
        """
        Extract a higher-TF sequence using pre-computed timestamp alignment.

        For each anchor position in [anchor_start, anchor_end), looks up the
        corresponding higher-TF index via idx_map. Deduplicates consecutive
        identical indices to get unique higher-TF bars, then pads/truncates
        to seq_len using forward-fill (last known bar value).

        Args:
            tf_values: Higher-TF feature array (n_higher_samples, n_features).
            idx_map: Mapping from anchor index -> higher-TF index.
            anchor_start: Start of anchor sequence window.
            anchor_end: End of anchor sequence window.
            seq_len: Desired output sequence length.

        Returns:
            Array of shape (seq_len, n_features).
        """
        # Get the higher-TF indices for this anchor window
        mapped_indices = idx_map[anchor_start:anchor_end]

        # Get unique higher-TF bars in order (deduplicate consecutive same-bar refs)
        unique_mask = np.concatenate([[True], np.diff(mapped_indices) != 0])
        unique_indices = mapped_indices[unique_mask]

        # Guard against empty tf_values (e.g. small dataset where split has no bars)
        # Use NaN instead of zeros so models don't interpret missing data as signal
        if len(tf_values) == 0:
            n_feat = tf_values.shape[1] if tf_values.ndim > 1 else 1
            return np.full((seq_len, n_feat), np.nan, dtype=np.float32)

        # Clamp to valid range
        unique_indices = np.clip(unique_indices, 0, len(tf_values) - 1)

        # Extract the unique bars
        unique_bars = tf_values[unique_indices]
        n_unique = len(unique_bars)

        if n_unique >= seq_len:
            # Take the most recent seq_len bars (aligned with anchor end)
            return unique_bars[-seq_len:].astype(np.float32)

        # Fewer unique bars than seq_len: pad at the front with the earliest bar
        # (forward-fill from the oldest available bar)
        pad_len = seq_len - n_unique
        pad = np.repeat(unique_bars[:1], pad_len, axis=0)
        result = np.concatenate([pad, unique_bars], axis=0)
        return result.astype(np.float32)

    # NOTE: _get_metadata_value and _parse_horizon_from_label_column
    # are now inherited from BaseAdapter (Phase 31 consolidation)


__all__ = [
    "MultiStreamAdapter",
    "resample_higher_timeframes",
]
