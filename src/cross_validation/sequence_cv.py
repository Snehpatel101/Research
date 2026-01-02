"""
Sequence-aware Cross-Validation Utilities.

This module provides utilities for running cross-validation with sequence models
(LSTM, GRU, TCN, Transformer). The key challenge is that sequence models require
3D input (batch, seq_len, features), but CV splits data by individual sample indices.

Key Features:
- Build 3D sequences from CV fold indices
- Handle symbol boundaries (sequences don't cross symbols)
- Map sequence targets back to original indices for OOF storage
- Proper purging for sequence models

Usage:
    from src.cross_validation.sequence_cv import SequenceCVBuilder

    builder = SequenceCVBuilder(
        X=X_df,
        y=y_series,
        seq_len=60,
        symbol_column="symbol",
    )

    # For each CV fold
    for train_idx, val_idx in cv.split(X, y):
        X_train_3d, y_train, train_targets = builder.build_fold_sequences(train_idx)
        X_val_3d, y_val, val_targets = builder.build_fold_sequences(val_idx)
        # train_targets contains original indices for OOF storage
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Gap detection threshold for datetime boundaries
# A gap is considered a boundary if it's larger than this multiple of median bar spacing
GAP_DETECTION_MULTIPLIER = 2.0


@dataclass
class SequenceFoldResult:
    """
    Result from building sequences for a CV fold.

    Attributes:
        X_sequences: 3D array of shape (n_sequences, seq_len, n_features)
        y: 1D array of labels, shape (n_sequences,)
        weights: 1D array of sample weights, shape (n_sequences,)
        target_indices: Original DataFrame indices for each sequence target.
            Used to map OOF predictions back to original sample positions.
        n_dropped: Number of samples dropped (not enough lookback history)
    """

    X_sequences: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    target_indices: np.ndarray
    n_dropped: int

    @property
    def n_sequences(self) -> int:
        return len(self.y)

    @property
    def seq_len(self) -> int:
        return self.X_sequences.shape[1] if len(self.X_sequences) > 0 else 0

    @property
    def n_features(self) -> int:
        return self.X_sequences.shape[2] if len(self.X_sequences) > 0 else 0

    def __repr__(self) -> str:
        return (
            f"SequenceFoldResult(n_sequences={self.n_sequences}, "
            f"seq_len={self.seq_len}, features={self.n_features}, "
            f"dropped={self.n_dropped})"
        )


class SequenceCVBuilder:
    """
    Build 3D sequences from CV fold indices.

    For sequence models, each sample at index `i` requires lookback data
    from indices `[i - seq_len + 1, ..., i]`. This class handles:

    1. Identifying which fold samples have sufficient lookback history
    2. Building 3D sequences from those samples
    3. Respecting boundaries (symbol changes or time gaps) - no cross-boundary sequences
    4. Tracking original indices for OOF prediction storage

    Boundary Detection Methods:
    - symbol_column: If provided and exists in X, uses symbol changes as boundaries
    - datetime_gaps: If X has DatetimeIndex, detects large time gaps (>2x median)
    - none: No boundary detection (sequences can span entire dataset)

    The key insight is that we allow using historical data from OUTSIDE the fold
    for sequence construction (the lookback window), but the TARGET (label) must
    be within the fold. This maximizes data usage while maintaining proper CV.

    Example:
        >>> builder = SequenceCVBuilder(X, y, weights, seq_len=60)
        >>> for train_idx, val_idx in cv.split(X, y):
        ...     train_result = builder.build_fold_sequences(train_idx, allow_lookback_outside=True)
        ...     val_result = builder.build_fold_sequences(val_idx, allow_lookback_outside=True)
        ...     # Train model on train_result.X_sequences, train_result.y
        ...     # Predict on val_result.X_sequences
        ...     # Store predictions at val_result.target_indices
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        seq_len: int,
        weights: pd.Series | None = None,
        symbol_column: str | None = None,
    ) -> None:
        """
        Initialize SequenceCVBuilder.

        Args:
            X: Feature DataFrame (n_samples, n_features)
            y: Label Series
            seq_len: Sequence length
            weights: Optional sample weights
            symbol_column: Column name for symbol isolation (None to disable)

        Raises:
            ValueError: If seq_len <= 0 or data is empty
        """
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if len(X) == 0:
            raise ValueError("X cannot be empty")
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")

        self.seq_len = seq_len
        self.n_samples = len(X)

        # Build symbol boundary information FIRST (needs original X)
        self._symbol_boundaries: np.ndarray | None = None
        self._symbol_ids: np.ndarray | None = None
        self._boundary_detection_method = "none"

        if symbol_column and symbol_column in X.columns:
            self._build_symbol_info(X, symbol_column)
            self._boundary_detection_method = "symbol_column"
        elif symbol_column:
            logger.debug(f"Symbol column '{symbol_column}' not found, attempting gap detection")
            # Fallback to datetime gap detection if no symbol column
            if isinstance(X.index, pd.DatetimeIndex):
                self._build_gap_boundaries(X)
                self._boundary_detection_method = "datetime_gaps"
            else:
                logger.debug("No DatetimeIndex found, disabling boundary detection")
        else:
            # No symbol column requested, check for datetime gaps anyway
            if isinstance(X.index, pd.DatetimeIndex):
                self._build_gap_boundaries(X)
                self._boundary_detection_method = "datetime_gaps"

        # Exclude symbol column from features (it's non-numeric)
        feature_cols = [c for c in X.columns if c != symbol_column]
        X_features = X[feature_cols] if feature_cols else X

        self.n_features = X_features.shape[1]

        # Store data as numpy arrays for efficiency
        self._X = X_features.values.astype(np.float32)
        self._y = y.values.astype(np.float32)
        self._weights = (
            weights.values.astype(np.float32)
            if weights is not None
            else np.ones(len(X), dtype=np.float32)
        )

    def _build_symbol_info(self, X: pd.DataFrame, symbol_column: str) -> None:
        """Build symbol boundary and ID arrays."""
        # Get symbol for each sample
        symbols = X.reset_index(drop=True)[symbol_column].values

        # Assign integer IDs to symbols
        unique_symbols, symbol_ids = np.unique(symbols, return_inverse=True)
        self._symbol_ids = symbol_ids

        # Find boundary indices (where symbol changes)
        boundaries = []
        for i in range(1, len(symbols)):
            if symbols[i] != symbols[i - 1]:
                boundaries.append(i)

        self._symbol_boundaries = np.array(boundaries, dtype=np.int64)
        logger.debug(f"Found {len(unique_symbols)} symbols, {len(boundaries)} boundaries")

    def _build_gap_boundaries(self, X: pd.DataFrame) -> None:
        """
        Build boundaries from datetime gaps in the index.

        Detects large time gaps in a DatetimeIndex and treats them as implicit
        boundaries (similar to symbol changes). This prevents sequences from
        spanning data gaps.

        Args:
            X: DataFrame with DatetimeIndex
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            return

        # Calculate time deltas between consecutive samples
        time_diffs = X.index.to_series().diff()

        # Estimate normal bar resolution (median of time diffs)
        # Use median instead of mode to handle occasional gaps
        median_diff = time_diffs.median()

        # Define a gap as anything > GAP_DETECTION_MULTIPLIER * normal resolution
        # (conservative threshold to avoid false positives)
        gap_threshold = median_diff * GAP_DETECTION_MULTIPLIER

        # Find indices where gaps occur
        boundaries = []
        for i in range(1, len(time_diffs)):
            if time_diffs.iloc[i] > gap_threshold:
                boundaries.append(i)

        self._symbol_boundaries = np.array(boundaries, dtype=np.int64)

        if len(boundaries) > 0:
            logger.info(
                f"Detected {len(boundaries)} time gaps (using boundary detection). "
                f"Bar resolution: {median_diff}, gap threshold: {gap_threshold}"
            )
        else:
            logger.debug(f"No significant time gaps detected (resolution: {median_diff})")

    def _get_symbol_at(self, idx: int) -> int:
        """Get symbol ID at given index (-1 if no symbol info)."""
        if self._symbol_ids is None:
            return -1
        return int(self._symbol_ids[idx])

    def _sequence_crosses_boundary(self, start_idx: int, end_idx: int) -> bool:
        """Check if sequence [start_idx, end_idx] crosses a symbol boundary."""
        if self._symbol_boundaries is None or len(self._symbol_boundaries) == 0:
            return False

        # Check if any boundary falls within (start_idx, end_idx]
        # A boundary at position b means symbols[b-1] != symbols[b]
        for boundary in self._symbol_boundaries:
            if start_idx < boundary <= end_idx:
                return True
        return False

    def build_fold_sequences(
        self,
        fold_indices: np.ndarray,
        allow_lookback_outside: bool = True,
        stride: int = 1,
    ) -> SequenceFoldResult:
        """
        Build 3D sequences for a CV fold.

        For each target index in fold_indices, attempts to build a sequence
        using the previous seq_len samples. If allow_lookback_outside=True,
        the lookback window can include samples from outside the fold (but
        the TARGET must be in the fold).

        Args:
            fold_indices: Array of sample indices for this fold
            allow_lookback_outside: If True, allow lookback into samples
                outside this fold. If False, only use fold samples for lookback.
            stride: Step size between target indices (default 1)

        Returns:
            SequenceFoldResult with 3D sequences and mapping info

        Note:
            - Sequences that would cross symbol boundaries are skipped
            - Sequences with insufficient lookback history are skipped
        """
        if len(fold_indices) == 0:
            return SequenceFoldResult(
                X_sequences=np.array([]).reshape(0, self.seq_len, self.n_features),
                y=np.array([]),
                weights=np.array([]),
                target_indices=np.array([], dtype=np.int64),
                n_dropped=0,
            )

        # Sort indices for proper ordering
        sorted_indices = np.sort(fold_indices)

        # Build set for fast fold membership check
        fold_set = set(sorted_indices.tolist()) if not allow_lookback_outside else None

        sequences = []
        labels = []
        weights = []
        target_indices = []
        n_dropped = 0

        for target_idx in sorted_indices[::stride]:
            # Sequence spans [start_idx, target_idx] (inclusive)
            start_idx = target_idx - self.seq_len + 1

            # Check 1: Sufficient lookback history
            if start_idx < 0:
                n_dropped += 1
                continue

            # Check 2: If not allowing lookback outside, verify all lookback in fold
            if not allow_lookback_outside:
                lookback_in_fold = all(idx in fold_set for idx in range(start_idx, target_idx))
                if not lookback_in_fold:
                    n_dropped += 1
                    continue

            # Check 3: Symbol boundary check
            if self._sequence_crosses_boundary(start_idx, target_idx):
                n_dropped += 1
                continue

            # Build sequence
            seq = self._X[start_idx : target_idx + 1]  # [seq_len, n_features]
            sequences.append(seq)
            labels.append(self._y[target_idx])
            weights.append(self._weights[target_idx])
            target_indices.append(target_idx)

        # Stack into arrays
        if sequences:
            X_sequences = np.stack(sequences, axis=0)  # [n_seq, seq_len, n_features]
            y = np.array(labels, dtype=np.float32)
            w = np.array(weights, dtype=np.float32)
            targets = np.array(target_indices, dtype=np.int64)
        else:
            X_sequences = np.array([]).reshape(0, self.seq_len, self.n_features)
            y = np.array([], dtype=np.float32)
            w = np.array([], dtype=np.float32)
            targets = np.array([], dtype=np.int64)

        if n_dropped > 0:
            logger.debug(
                f"Built {len(sequences)} sequences from {len(fold_indices)} fold samples "
                f"(dropped {n_dropped}: insufficient history or symbol boundary)"
            )

        return SequenceFoldResult(
            X_sequences=X_sequences,
            y=y,
            weights=w,
            target_indices=targets,
            n_dropped=n_dropped,
        )

    def get_fold_coverage(
        self,
        fold_indices: np.ndarray,
        allow_lookback_outside: bool = True,
    ) -> float:
        """
        Calculate what fraction of fold samples can produce valid sequences.

        Args:
            fold_indices: Array of sample indices for this fold
            allow_lookback_outside: Whether lookback can use outside samples

        Returns:
            Coverage ratio (0.0 to 1.0)
        """
        if len(fold_indices) == 0:
            return 0.0

        result = self.build_fold_sequences(
            fold_indices,
            allow_lookback_outside=allow_lookback_outside,
        )

        return result.n_sequences / len(fold_indices)


def build_sequences_for_cv_fold(
    X: pd.DataFrame,
    y: pd.Series,
    fold_indices: np.ndarray,
    seq_len: int,
    weights: pd.Series | None = None,
    symbol_column: str | None = None,
    allow_lookback_outside: bool = True,
) -> SequenceFoldResult:
    """
    Convenience function to build sequences for a single CV fold.

    This is a wrapper around SequenceCVBuilder for one-off usage.

    Args:
        X: Feature DataFrame
        y: Label Series
        fold_indices: Indices for this fold
        seq_len: Sequence length
        weights: Optional sample weights
        symbol_column: Symbol column for isolation
        allow_lookback_outside: Allow lookback into non-fold samples

    Returns:
        SequenceFoldResult with 3D sequences
    """
    builder = SequenceCVBuilder(
        X=X,
        y=y,
        seq_len=seq_len,
        weights=weights,
        symbol_column=symbol_column,
    )
    return builder.build_fold_sequences(
        fold_indices,
        allow_lookback_outside=allow_lookback_outside,
    )


def validate_sequence_cv_coverage(
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    seq_len: int,
    symbol_column: str | None = None,
) -> dict:
    """
    Validate sequence coverage across all CV folds.

    Checks what fraction of samples in each fold can produce valid sequences.
    Low coverage may indicate issues with seq_len, data ordering, or symbol distribution.

    Args:
        X: Feature DataFrame
        y: Label Series
        cv: Cross-validator (must have split method)
        seq_len: Sequence length
        symbol_column: Symbol column for isolation

    Returns:
        Dict with coverage statistics per fold and overall
    """
    builder = SequenceCVBuilder(
        X=X,
        y=y,
        seq_len=seq_len,
        symbol_column=symbol_column,
    )

    fold_coverages = []
    fold_n_sequences = []
    fold_n_samples = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # Check validation fold coverage
        val_result = builder.build_fold_sequences(
            val_idx,
            allow_lookback_outside=True,
        )
        coverage = val_result.n_sequences / len(val_idx) if len(val_idx) > 0 else 0.0
        fold_coverages.append(coverage)
        fold_n_sequences.append(val_result.n_sequences)
        fold_n_samples.append(len(val_idx))

    return {
        "n_folds": len(fold_coverages),
        "fold_coverages": fold_coverages,
        "fold_n_sequences": fold_n_sequences,
        "fold_n_samples": fold_n_samples,
        "mean_coverage": float(np.mean(fold_coverages)),
        "min_coverage": float(np.min(fold_coverages)) if fold_coverages else 0.0,
        "total_sequences": sum(fold_n_sequences),
        "total_samples": sum(fold_n_samples),
    }


__all__ = [
    "SequenceFoldResult",
    "SequenceCVBuilder",
    "build_sequences_for_cv_fold",
    "validate_sequence_cv_coverage",
]
