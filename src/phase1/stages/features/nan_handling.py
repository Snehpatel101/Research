"""
NaN handling utilities for feature engineering.

This module provides functions for auditing and cleaning NaN values
in DataFrames, with configurable thresholds for column dropping.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def audit_nan_columns(
    df: pd.DataFrame,
    nan_threshold: float = 0.9
) -> Dict:
    """
    Audit NaN values per column and categorize by severity.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to audit
    nan_threshold : float, default 0.9
        Threshold for "high NaN" categorization

    Returns
    -------
    Dict
        Audit results with column categorizations
    """
    rows = len(df)
    if rows == 0:
        return {
            'total_rows': 0,
            'total_cols': len(df.columns),
            'all_nan_cols': [],
            'high_nan_cols': [],
            'moderate_nan_cols': [],
            'nan_rates': {}
        }

    nan_counts = df.isna().sum()
    nan_rates = nan_counts / rows

    all_nan_cols = nan_rates[nan_rates == 1.0].index.tolist()
    high_nan_cols = nan_rates[
        (nan_rates > nan_threshold) & (nan_rates < 1.0)
    ].index.tolist()
    moderate_nan_cols = nan_rates[
        (nan_rates > 0.5) & (nan_rates <= nan_threshold)
    ].index.tolist()

    return {
        'total_rows': rows,
        'total_cols': len(df.columns),
        'all_nan_cols': all_nan_cols,
        'high_nan_cols': high_nan_cols,
        'moderate_nan_cols': moderate_nan_cols,
        'nan_rates': nan_rates.to_dict()
    }


def clean_nan_columns(
    df: pd.DataFrame,
    symbol: str,
    nan_threshold: float = 0.9,
    protected_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Audit NaN values and clean problematic columns before row-wise dropna.

    This function:
    1. Computes NaN statistics for each column
    2. Logs columns with high NaN rates
    3. Drops columns exceeding the nan_threshold (except protected columns)
    4. Then drops rows with remaining NaN values

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to audit and clean
    symbol : str
        Symbol name for logging context
    nan_threshold : float, default 0.9
        Columns with NaN rate above this threshold are dropped.
        Range: 0.0 to 1.0. Set to 1.0 to disable column dropping.
    protected_columns : List[str], optional
        Columns that should never be dropped (e.g., OHLCV, datetime).
        Defaults to ['datetime', 'open', 'high', 'low', 'close', 'volume']

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        (Cleaned DataFrame, NaN audit report)

    Raises
    ------
    ValueError
        If protected columns have all NaN values (corrupt input data)
        If all rows are dropped after NaN handling
    """
    if protected_columns is None:
        protected_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

    rows_before = len(df)
    cols_before = len(df.columns)

    # Compute NaN statistics per column
    nan_counts = df.isna().sum()
    nan_rates = nan_counts / rows_before if rows_before > 0 else nan_counts

    # Identify problematic columns
    all_nan_cols = nan_rates[nan_rates == 1.0].index.tolist()
    high_nan_cols = nan_rates[
        (nan_rates > nan_threshold) & (nan_rates < 1.0)
    ].index.tolist()
    moderate_nan_cols = nan_rates[
        (nan_rates > 0.5) & (nan_rates <= nan_threshold)
    ].index.tolist()

    # Log NaN audit results
    logger.info(f"\n--- NaN Audit for {symbol} ---")
    logger.info(f"Total columns: {cols_before}, Total rows: {rows_before:,}")

    if all_nan_cols:
        logger.warning(
            f"Columns with 100% NaN ({len(all_nan_cols)}): {all_nan_cols[:10]}"
            + (f"... and {len(all_nan_cols)-10} more" if len(all_nan_cols) > 10 else "")
        )

    if high_nan_cols:
        logger.warning(
            f"Columns with >{nan_threshold*100:.0f}% NaN ({len(high_nan_cols)}): {high_nan_cols[:10]}"
            + (f"... and {len(high_nan_cols)-10} more" if len(high_nan_cols) > 10 else "")
        )

    if moderate_nan_cols:
        logger.info(
            f"Columns with 50-{nan_threshold*100:.0f}% NaN ({len(moderate_nan_cols)}): {moderate_nan_cols[:5]}"
            + (f"... and {len(moderate_nan_cols)-5} more" if len(moderate_nan_cols) > 5 else "")
        )

    # Columns to drop (above threshold, not protected)
    cols_to_drop = [
        col for col in (all_nan_cols + high_nan_cols)
        if col not in protected_columns
    ]

    # Check if any protected columns have all NaN (fatal error)
    protected_all_nan = [col for col in all_nan_cols if col in protected_columns]
    if protected_all_nan:
        raise ValueError(
            f"Protected columns have all NaN values: {protected_all_nan}. "
            f"This indicates corrupt or invalid input data."
        )

    # Drop problematic columns
    if cols_to_drop:
        logger.warning(
            f"Dropping {len(cols_to_drop)} columns exceeding NaN threshold: {cols_to_drop[:10]}"
            + (f"... and {len(cols_to_drop)-10} more" if len(cols_to_drop) > 10 else "")
        )
        df = df.drop(columns=cols_to_drop)

    # Now drop rows with remaining NaN values
    rows_after_col_drop = len(df)
    df = df.dropna()
    rows_after = len(df)
    rows_dropped = rows_after_col_drop - rows_after

    # Validate we have data remaining
    if len(df) == 0:
        raise ValueError(
            f"All rows dropped after NaN handling for {symbol}. "
            f"Columns dropped: {len(cols_to_drop)}, "
            f"Rows before dropna: {rows_after_col_drop:,}, "
            f"Check if input data has sufficient rows for indicator warmup periods (~200+ rows). "
            f"Original rows: {rows_before:,}, Original columns: {cols_before}."
        )

    # Log summary
    logger.info(
        f"NaN cleanup: {cols_before} cols -> {len(df.columns)} cols "
        f"(-{len(cols_to_drop)} dropped)"
    )
    logger.info(
        f"NaN cleanup: {rows_before:,} rows -> {len(df):,} rows "
        f"(-{rows_dropped:,} dropped, {rows_dropped/rows_before*100:.1f}%)"
    )

    if rows_dropped > rows_before * 0.5:
        logger.warning(
            f"High row drop rate ({rows_dropped/rows_before*100:.1f}%). "
            f"This may indicate insufficient data for indicator warmup periods."
        )

    # Build audit report
    audit_report = {
        'rows_before': rows_before,
        'rows_after': len(df),
        'rows_dropped': rows_dropped,
        'row_drop_rate': rows_dropped / rows_before if rows_before > 0 else 0,
        'cols_before': cols_before,
        'cols_after': len(df.columns),
        'cols_dropped': len(cols_to_drop),
        'cols_dropped_names': cols_to_drop,
        'all_nan_cols': all_nan_cols,
        'high_nan_cols': high_nan_cols,
        'nan_threshold': nan_threshold
    }

    return df, audit_report
