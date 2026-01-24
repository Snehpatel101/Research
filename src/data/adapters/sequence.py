"""
Sequence Adapter - Converts DataFrame to 3D arrays for sequence models.

Transforms tabular DataFrames into (n_samples, seq_len, n_features) format
for LSTM, GRU, TCN, Transformer, PatchTST, and other sequence models.

Phase 2 SNwH Implementation.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.core.contracts import DataContract, DataRank

from .base import AdapterResult, BaseAdapter
from .registry import AdapterRegistry

if TYPE_CHECKING:
    from src.core.contracts import ModelContract

logger = logging.getLogger(__name__)


@AdapterRegistry.register("sequence")
class SequenceAdapter(BaseAdapter):
    """
    Adapter for sequence models requiring 3D input.

    Converts DataFrame with features and labels into windowed sequences
    of shape (n_samples, sequence_length, n_features).

    The label for each sequence corresponds to the LAST timestep of the window,
    ensuring no future information leaks into the features.

    Supports:
    - Configurable sequence length and stride
    - Per-symbol sequence building (maintains temporal isolation)
    - Optional contract-driven sequence length override
    - Proper index tracking for mapping predictions back to source
    """

    adapter_id: str = "sequence"
    output_rank: DataRank = DataRank.SEQUENCE_3D

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        label_column: str = "label_h20",
        weight_column: str | None = "sample_weight_h20",
        sequence_length: int = 60,
        stride: int = 1,
        symbol_column: str | None = "symbol",
    ):
        """
        Initialize sequence adapter.

        Args:
            feature_columns: Feature columns to use (None = auto-detect).
            label_column: Label column name.
            weight_column: Weight column name (None = uniform weights).
            sequence_length: Number of timesteps in each sequence window.
            stride: Step size between consecutive sequences.
            symbol_column: Column for symbol isolation. If present in DataFrame,
                sequences are built per-symbol then concatenated. Set to None
                to disable symbol isolation.
        """
        super().__init__(
            feature_columns=feature_columns,
            label_column=label_column,
            weight_column=weight_column,
        )
        self.sequence_length = sequence_length
        self.stride = stride
        self.symbol_column = symbol_column

    def transform(
        self,
        df: pd.DataFrame,
        model_contract: ModelContract | None = None,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> AdapterResult:
        """
        Transform DataFrame to 3D sequence arrays.

        Args:
            df: Source DataFrame with features, labels, and optionally weights.
                Must be sorted by timestamp within each symbol group.
            model_contract: Optional contract for sequence length override.

        Returns:
            AdapterResult with:
                - X: (n_samples, seq_len, n_features) float32 array
                - y: (n_samples,) label array
                - weights: (n_samples,) weights array or None
                - original_indices: Maps each sequence to source DataFrame index
                - data_contract: Contract describing the output

        Raises:
            ValueError: If input validation fails.
        """
        # Validate input DataFrame
        is_valid, issues = self.validate_input(df)
        if not is_valid:
            raise ValueError(f"Input validation failed: {issues}")

        # Get feature columns
        feature_cols = self._get_feature_columns(df)
        if not feature_cols:
            raise ValueError("No feature columns found or specified")

        n_features = len(feature_cols)
        logger.debug(
            f"SequenceAdapter: {len(feature_cols)} features, "
            f"seq_len={self.sequence_length}, stride={self.stride}"
        )

        # Override sequence length from contract if provided
        seq_len = self.sequence_length
        if model_contract is not None and model_contract.sequence_length:
            seq_len = model_contract.sequence_length
            logger.debug(f"Using contract sequence_length: {seq_len}")

        # Build sequences - either per-symbol or for entire DataFrame
        if self.symbol_column and self.symbol_column in df.columns:
            X, y, weights, original_indices = self._build_sequences_per_symbol(
                df, feature_cols, seq_len
            )
        else:
            X, y, weights, original_indices = self._build_sequences(df, feature_cols, seq_len)

        # Handle empty result
        if X.shape[0] == 0:
            logger.warning(
                f"No sequences created. DataFrame has {len(df)} rows, "
                f"sequence_length={seq_len}. Need at least {seq_len} rows."
            )

        # Create data contract with metadata
        data_contract = self._create_data_contract(
            df=df,
            X=X,
            feature_cols=feature_cols,
        )

        # Validate data contract against model requirements if provided
        if model_contract is not None:
            model_contract.validate_data_contract_strict(data_contract)

        # Build result
        result = AdapterResult(
            X=X,
            y=y,
            weights=weights,
            n_samples=X.shape[0],
            n_features=n_features,
            data_rank=DataRank.SEQUENCE_3D,
            sequence_length=seq_len,
            original_indices=original_indices,
            feature_columns=feature_cols,
            data_contract=data_contract,
            adapter_name=self.adapter_id,
        )

        # Validate result
        is_valid, issues = result.validate()
        if not is_valid:
            logger.warning(f"AdapterResult validation issues: {issues}")

        return result

    def _build_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        seq_len: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
        """
        Build sequence arrays from a single DataFrame.

        Creates sliding windows of length seq_len, where each window's label
        corresponds to the LAST timestep in that window.

        Args:
            df: Source DataFrame (should be time-sorted).
            feature_cols: List of feature column names.
            seq_len: Sequence length for each window.

        Returns:
            Tuple of (X, y, weights, original_indices):
                - X: (n_sequences, seq_len, n_features) float32 array
                - y: (n_sequences,) label array
                - weights: (n_sequences,) weight array or None
                - original_indices: (n_sequences,) array mapping to source indices
        """
        n_rows = len(df)
        n_features = len(feature_cols)

        # Return empty arrays if insufficient data
        if n_rows < seq_len:
            return (
                np.empty((0, seq_len, n_features), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                None,
                np.empty((0,), dtype=np.int64),
            )

        # Calculate number of sequences based on stride
        n_sequences = (n_rows - seq_len) // self.stride + 1

        # Pre-allocate arrays
        X = np.empty((n_sequences, seq_len, n_features), dtype=np.float32)
        y = np.empty((n_sequences,), dtype=np.int64)
        original_indices = np.empty((n_sequences,), dtype=np.int64)

        # Extract data arrays once (more efficient than repeated iloc)
        features = df[feature_cols].values.astype(np.float32)
        labels = df[self.label_column].values
        df_indices = df.index.values

        # Check for weights
        has_weights = self.weight_column and self.weight_column in df.columns
        if has_weights:
            weights_arr = df[self.weight_column].values.astype(np.float32)
            weights = np.empty((n_sequences,), dtype=np.float32)
        else:
            weights_arr = None
            weights = None

        # Build sequences
        for i in range(n_sequences):
            start = i * self.stride
            end = start + seq_len

            # Window of features: [start, end)
            X[i] = features[start:end]

            # Label at LAST timestep of window (end - 1)
            y[i] = labels[end - 1]

            # Track original index of the label row
            original_indices[i] = df_indices[end - 1]

            # Weight at label position
            if has_weights and weights_arr is not None and weights is not None:
                weights[i] = weights_arr[end - 1]

        return X, y, weights, original_indices

    def _build_sequences_per_symbol(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        seq_len: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
        """
        Build sequences separately for each symbol, then concatenate.

        This ensures that sequences never span across symbol boundaries,
        preventing data leakage between different trading instruments.

        Args:
            df: Source DataFrame with symbol column.
            feature_cols: List of feature column names.
            seq_len: Sequence length for each window.

        Returns:
            Tuple of (X, y, weights, original_indices) - concatenated across symbols.
        """
        assert self.symbol_column is not None

        X_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        weights_list: list[np.ndarray] = []
        indices_list: list[np.ndarray] = []

        symbols = df[self.symbol_column].unique()
        logger.debug(f"Building sequences for {len(symbols)} symbol(s)")

        for symbol in symbols:
            symbol_df = df[df[self.symbol_column] == symbol]

            X_symbol, y_symbol, weights_symbol, indices_symbol = self._build_sequences(
                symbol_df, feature_cols, seq_len
            )

            if X_symbol.shape[0] > 0:
                X_list.append(X_symbol)
                y_list.append(y_symbol)
                indices_list.append(indices_symbol)
                if weights_symbol is not None:
                    weights_list.append(weights_symbol)

                logger.debug(
                    f"Symbol {symbol}: {X_symbol.shape[0]} sequences from " f"{len(symbol_df)} rows"
                )

        # Handle case where no sequences were created
        if not X_list:
            n_features = len(feature_cols)
            return (
                np.empty((0, seq_len, n_features), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                None,
                np.empty((0,), dtype=np.int64),
            )

        # Concatenate all symbols
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        original_indices = np.concatenate(indices_list, axis=0)

        # Concatenate weights if available
        if weights_list:
            weights = np.concatenate(weights_list, axis=0)
        else:
            weights = None

        logger.debug(f"Total sequences: {X.shape[0]} across {len(symbols)} symbol(s)")

        return X, y, weights, original_indices

    def _create_data_contract(
        self,
        df: pd.DataFrame,
        X: np.ndarray,
        feature_cols: list[str],
    ) -> DataContract:
        """
        Create DataContract with metadata extracted from DataFrame.

        Args:
            df: Source DataFrame.
            X: Transformed feature array.
            feature_cols: List of feature column names.

        Returns:
            DataContract with appropriate metadata.
        """
        # Extract symbol from DataFrame if available
        symbol = self._get_metadata_value(df, "symbol", default="unknown")

        # Extract timeframe from DataFrame if available
        timeframe = self._get_metadata_value(df, "timeframe", default="5min")

        # Parse horizon from label column name (e.g., "label_h20" -> 20)
        horizon = self._parse_horizon_from_label_column(self.label_column)

        # Determine split if available
        split = self._get_metadata_value(df, "split", default="unknown")

        return DataContract(
            symbol=symbol,
            timeframe=timeframe,
            horizon=horizon,
            split=split,
            n_samples=X.shape[0],
            n_features=X.shape[2],  # (n_samples, seq_len, n_features)
            data_rank=DataRank.SEQUENCE_3D,
            sequence_length=X.shape[1],
            feature_columns=feature_cols,
            label_column=self.label_column,
            weight_column=self.weight_column or "",
        )

    def _get_metadata_value(
        self,
        df: pd.DataFrame,
        column: str,
        default: str,
    ) -> str:
        """
        Extract a single metadata value from DataFrame column.

        If the column exists and has a single unique non-null value,
        return that value. Otherwise return the default.

        Args:
            df: DataFrame to extract from.
            column: Column name to look for.
            default: Default value if column missing or ambiguous.

        Returns:
            Extracted metadata value or default.
        """
        if column not in df.columns:
            return default

        unique_values = df[column].dropna().unique()
        if len(unique_values) == 1:
            return str(unique_values[0])
        elif len(unique_values) == 0:
            return default
        else:
            # Multiple values - log warning and use first
            logger.warning(
                f"Multiple values for metadata column '{column}': {unique_values[:5]}. "
                f"Using first value."
            )
            return str(unique_values[0])

    def _parse_horizon_from_label_column(self, label_column: str) -> int:
        """
        Parse horizon from label column name.

        Examples:
            "label_h20" -> 20
            "label_h5" -> 5
            "label_h10" -> 10
            "label" -> 20 (default)

        Args:
            label_column: Label column name.

        Returns:
            Extracted horizon or default of 20.
        """
        # Pattern: label_h{number}
        match = re.search(r"label_h(\d+)", label_column)
        if match:
            return int(match.group(1))

        # Fallback: check for just a number at the end
        match = re.search(r"_(\d+)$", label_column)
        if match:
            return int(match.group(1))

        # Default horizon
        return 20


__all__ = ["SequenceAdapter"]
