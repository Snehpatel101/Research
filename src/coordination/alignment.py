"""
Temporal alignment utilities for multi-timeframe data coordination.

This module provides functions to align data from different timeframes,
ensuring proper temporal relationships and preventing lookahead bias.

Key functions:
- align_to_anchor: Align higher TF data to anchor timestamps
- apply_mtf_lag: Apply lag to MTF columns to prevent leakage
- compute_sequence_offset: Calculate offset between tabular and sequence data
- validate_timestamp_alignment: Validate two DataFrames have aligned timestamps

Usage:
    from src.coordination.alignment import (
        align_to_anchor,
        apply_mtf_lag,
        compute_sequence_offset,
        validate_timestamp_alignment,
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from src.core.common.timeframes import get_timeframe_minutes, normalize_timeframe

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def align_to_anchor(
    anchor_df: pd.DataFrame,
    higher_tf_df: pd.DataFrame,
    anchor_tf: str,
    higher_tf: str,
    datetime_col: str = "datetime",
) -> pd.DataFrame:
    """
    Align higher timeframe DataFrame to anchor timestamps using forward-fill.

    This function ensures we use the LAST COMPLETED bar from the higher timeframe
    at each anchor timestamp, preventing lookahead bias. The higher TF values are
    forward-filled to match the anchor's timestamp frequency.

    Parameters
    ----------
    anchor_df : pd.DataFrame
        The anchor (lower timeframe) DataFrame with target timestamps.
    higher_tf_df : pd.DataFrame
        The higher timeframe DataFrame to align.
    anchor_tf : str
        The anchor timeframe (e.g., "1min", "5min").
    higher_tf : str
        The higher timeframe (e.g., "15min", "60min").
    datetime_col : str, default "datetime"
        Name of the datetime column in both DataFrames.

    Returns
    -------
    pd.DataFrame
        Higher timeframe data aligned to anchor timestamps with forward-fill.
        Contains the same timestamps as anchor_df.

    Raises
    ------
    ValueError
        If datetime column is missing from either DataFrame.
        If higher timeframe is not actually higher than anchor.
        If DataFrames are empty.

    Examples
    --------
    >>> anchor = pd.DataFrame({
    ...     "datetime": pd.date_range("2024-01-01", periods=60, freq="1min"),
    ...     "close": range(60)
    ... })
    >>> higher = pd.DataFrame({
    ...     "datetime": pd.date_range("2024-01-01", periods=4, freq="15min"),
    ...     "sma_15": [100, 101, 102, 103]
    ... })
    >>> aligned = align_to_anchor(anchor, higher, "1min", "15min")
    >>> len(aligned) == len(anchor)
    True

    Notes
    -----
    - Forward-fill ensures we only use information available at each anchor timestamp
    - The first anchor timestamps before the first higher TF bar will have NaN values
    - This is the correct behavior to prevent lookahead bias
    """
    # Validate inputs
    if anchor_df.empty:
        logger.warning("align_to_anchor: anchor_df is empty, returning empty DataFrame")
        return pd.DataFrame()

    if higher_tf_df.empty:
        logger.warning("align_to_anchor: higher_tf_df is empty, returning empty DataFrame")
        return pd.DataFrame()

    if datetime_col not in anchor_df.columns:
        raise ValueError(
            f"Datetime column '{datetime_col}' not found in anchor_df. "
            f"Available columns: {list(anchor_df.columns)}"
        )

    if datetime_col not in higher_tf_df.columns:
        raise ValueError(
            f"Datetime column '{datetime_col}' not found in higher_tf_df. "
            f"Available columns: {list(higher_tf_df.columns)}"
        )

    # Normalize timeframes and validate relationship
    anchor_tf_norm = normalize_timeframe(anchor_tf)
    higher_tf_norm = normalize_timeframe(higher_tf)

    anchor_minutes = get_timeframe_minutes(anchor_tf_norm)
    higher_minutes = get_timeframe_minutes(higher_tf_norm)

    if higher_minutes <= anchor_minutes:
        raise ValueError(
            f"Higher timeframe ({higher_tf} = {higher_minutes}min) must be greater than "
            f"anchor timeframe ({anchor_tf} = {anchor_minutes}min)"
        )

    logger.debug(
        "Aligning %s (%d min) to anchor %s (%d min)",
        higher_tf,
        higher_minutes,
        anchor_tf,
        anchor_minutes,
    )

    # Create copies to avoid modifying originals
    anchor_times = anchor_df[[datetime_col]].copy()
    higher_copy = higher_tf_df.copy()

    # Ensure datetime columns are datetime type
    anchor_times[datetime_col] = pd.to_datetime(anchor_times[datetime_col])
    higher_copy[datetime_col] = pd.to_datetime(higher_copy[datetime_col])

    # Sort both by datetime
    anchor_times = anchor_times.sort_values(datetime_col).reset_index(drop=True)
    higher_copy = higher_copy.sort_values(datetime_col).reset_index(drop=True)

    # Set datetime as index for reindexing
    higher_copy = higher_copy.set_index(datetime_col)

    # Use merge_asof for efficient alignment
    # direction='backward' ensures we use the LAST completed bar (no lookahead)
    aligned = pd.merge_asof(
        anchor_times,
        higher_copy.reset_index(),
        on=datetime_col,
        direction="backward",
    )

    logger.debug(
        "Aligned %d higher TF rows to %d anchor rows",
        len(higher_tf_df),
        len(aligned),
    )

    return aligned


def apply_mtf_lag(
    df: pd.DataFrame,
    mtf_columns: list[str],
    shift: int = 1,
) -> pd.DataFrame:
    """
    Apply lag (shift) to MTF indicator columns to prevent lookahead leakage.

    MTF features from higher timeframes may contain information that is not yet
    available at the anchor timestamp. This function applies a configurable shift
    to ensure temporal correctness.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing MTF indicator columns.
    mtf_columns : list[str]
        List of column names to shift. Columns not found are skipped with warning.
    shift : int, default 1
        Number of periods to shift (positive = lag, looking backward).

    Returns
    -------
    pd.DataFrame
        DataFrame with shifted MTF columns. NaN values from shift are
        forward-filled with the first valid value.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "close": [100, 101, 102, 103, 104],
    ...     "sma_15m": [99, 100, 101, 102, 103],
    ...     "rsi_1h": [50, 51, 52, 53, 54]
    ... })
    >>> result = apply_mtf_lag(df, ["sma_15m", "rsi_1h"], shift=1)
    >>> # First row will have forward-filled values
    >>> result["sma_15m"].tolist()
    [99, 99, 100, 101, 102]

    Notes
    -----
    - Shift=1 means each row uses the previous row's MTF value
    - This ensures we only use information from completed bars
    - NaN values at the start are filled with the first valid value to avoid
      losing samples at the beginning of the dataset
    """
    if df.empty:
        logger.warning("apply_mtf_lag: input DataFrame is empty")
        return df.copy()

    if not mtf_columns:
        logger.debug("apply_mtf_lag: no MTF columns specified, returning unchanged")
        return df.copy()

    if shift <= 0:
        logger.warning("apply_mtf_lag: shift=%d is not positive, no lag applied", shift)
        return df.copy()

    result = df.copy()

    # Track which columns were actually shifted
    shifted_cols = []
    missing_cols = []

    for col in mtf_columns:
        if col not in result.columns:
            missing_cols.append(col)
            continue

        # Apply shift
        result[col] = result[col].shift(shift)

        # Forward-fill NaN from shift with first valid value
        # This fills the first `shift` rows with the first non-NaN value
        first_valid = result[col].dropna().iloc[0] if result[col].notna().any() else None
        if first_valid is not None:
            result[col] = result[col].fillna(first_valid)

        shifted_cols.append(col)

    if missing_cols:
        logger.warning(
            "apply_mtf_lag: columns not found in DataFrame: %s",
            missing_cols,
        )

    logger.debug(
        "Applied shift=%d to %d MTF columns: %s",
        shift,
        len(shifted_cols),
        shifted_cols[:5],  # Log first 5 to avoid verbose output
    )

    return result


def compute_sequence_offset(
    tabular_samples: int,
    sequence_samples: int,
    sequence_length: int,
) -> int:
    """
    Compute offset between tabular and sequence data due to windowing.

    Sequence models require a lookback window of `sequence_length` bars.
    This means the first (seq_len - 1) samples cannot form complete windows
    and are lost. This function computes and validates this offset.

    Parameters
    ----------
    tabular_samples : int
        Number of samples in tabular (2D) dataset.
    sequence_samples : int
        Number of samples in sequence (3D) dataset.
    sequence_length : int
        Length of the sequence window.

    Returns
    -------
    int
        The offset (number of samples lost from the start of tabular data).

    Raises
    ------
    ValueError
        If any input is negative.
        If sequence_samples > tabular_samples (impossible).

    Examples
    --------
    >>> compute_sequence_offset(1000, 971, 30)
    29
    >>> # Expected: 30 - 1 = 29 samples lost

    Notes
    -----
    - Expected offset is (sequence_length - 1)
    - If actual offset differs, a warning is logged
    - This offset should be used to align tabular predictions with sequence data
    """
    # Validate inputs
    if tabular_samples < 0:
        raise ValueError(f"tabular_samples must be non-negative, got {tabular_samples}")
    if sequence_samples < 0:
        raise ValueError(f"sequence_samples must be non-negative, got {sequence_samples}")
    if sequence_length < 1:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")

    if sequence_samples > tabular_samples:
        raise ValueError(
            f"sequence_samples ({sequence_samples}) cannot exceed "
            f"tabular_samples ({tabular_samples})"
        )

    # Compute actual offset
    actual_offset = tabular_samples - sequence_samples

    # Expected offset from windowing
    expected_offset = sequence_length - 1

    if actual_offset != expected_offset:
        logger.warning(
            "Sequence offset mismatch: expected %d (seq_len=%d - 1), "
            "actual %d (tabular=%d - sequence=%d). "
            "This may indicate data processing issues.",
            expected_offset,
            sequence_length,
            actual_offset,
            tabular_samples,
            sequence_samples,
        )

    logger.debug(
        "Computed sequence offset: %d (tabular=%d, sequence=%d, seq_len=%d)",
        actual_offset,
        tabular_samples,
        sequence_samples,
        sequence_length,
    )

    return actual_offset


def validate_timestamp_alignment(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    datetime_col: str = "datetime",
    tolerance_minutes: int = 1,
) -> tuple[bool, list[str]]:
    """
    Validate that two DataFrames have aligned timestamps.

    This function checks that two DataFrames have the same length and
    corresponding timestamps that are within a specified tolerance.

    Parameters
    ----------
    df1 : pd.DataFrame
        First DataFrame to compare.
    df2 : pd.DataFrame
        Second DataFrame to compare.
    datetime_col : str, default "datetime"
        Name of the datetime column in both DataFrames.
    tolerance_minutes : int, default 1
        Maximum allowed difference between corresponding timestamps in minutes.

    Returns
    -------
    tuple[bool, list[str]]
        - bool: True if timestamps are aligned, False otherwise
        - list[str]: List of issues found (empty if aligned)

    Examples
    --------
    >>> df1 = pd.DataFrame({
    ...     "datetime": pd.date_range("2024-01-01", periods=100, freq="5min"),
    ...     "feature1": range(100)
    ... })
    >>> df2 = pd.DataFrame({
    ...     "datetime": pd.date_range("2024-01-01", periods=100, freq="5min"),
    ...     "feature2": range(100)
    ... })
    >>> is_aligned, issues = validate_timestamp_alignment(df1, df2)
    >>> is_aligned
    True
    >>> len(issues)
    0

    Notes
    -----
    - Length mismatch is always reported as an issue
    - Timestamp differences are checked only if lengths match
    - Use tolerance_minutes=0 for exact timestamp matching
    """
    issues: list[str] = []

    # Check for empty DataFrames
    if df1.empty and df2.empty:
        logger.debug("Both DataFrames are empty, considered aligned")
        return True, []

    if df1.empty:
        issues.append("df1 is empty while df2 is not")
        return False, issues

    if df2.empty:
        issues.append("df2 is empty while df1 is not")
        return False, issues

    # Check datetime column existence
    if datetime_col not in df1.columns:
        issues.append(f"Datetime column '{datetime_col}' not found in df1")
    if datetime_col not in df2.columns:
        issues.append(f"Datetime column '{datetime_col}' not found in df2")

    if issues:
        return False, issues

    # Check length match
    len1, len2 = len(df1), len(df2)
    if len1 != len2:
        issues.append(f"Length mismatch: df1 has {len1} rows, df2 has {len2} rows")
        return False, issues

    # Convert to datetime if needed
    times1 = pd.to_datetime(df1[datetime_col])
    times2 = pd.to_datetime(df2[datetime_col])

    # Compute timestamp differences
    time_diffs = (times1.values - times2.values).astype("timedelta64[m]").astype(float)

    # Check tolerance
    tolerance_td = tolerance_minutes
    max_diff = abs(time_diffs).max()

    if max_diff > tolerance_td:
        # Find indices of misaligned timestamps
        misaligned_mask = abs(time_diffs) > tolerance_td
        misaligned_count = misaligned_mask.sum()
        misaligned_indices = misaligned_mask.nonzero()[0][:5]  # First 5 examples

        issues.append(
            f"Timestamp differences exceed tolerance ({tolerance_minutes} min): "
            f"max diff = {max_diff:.1f} min, {misaligned_count} rows affected. "
            f"First misaligned indices: {list(misaligned_indices)}"
        )

        # Log specific examples for debugging
        for idx in misaligned_indices[:3]:
            logger.debug(
                "Misaligned timestamp at index %d: df1=%s, df2=%s, diff=%.1f min",
                idx,
                times1.iloc[idx],
                times2.iloc[idx],
                time_diffs[idx],
            )

        return False, issues

    logger.debug(
        "Timestamp alignment validated: %d rows, max diff = %.2f min (tolerance = %d min)",
        len1,
        max_diff,
        tolerance_minutes,
    )

    return True, []
