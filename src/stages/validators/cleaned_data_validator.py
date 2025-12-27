"""
Cleaned Data Validator.

Validates the output of Stage 2 (Data Cleaning) before passing to Stage 3 (Feature Engineering).

Checks:
- Required OHLCV columns present with correct types
- No duplicate timestamps
- OHLCV relationships valid (high >= low, etc.)
- No NaN/Inf values in price columns
- Reasonable row count maintained after cleaning
- Time gaps within acceptable limits
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

from .stage_boundary import (
    ValidationResult,
    check_dataframe_basics,
    check_required_columns,
    check_nan_values,
    check_infinite_values,
    check_datetime_column,
    check_ohlcv_relationships,
    check_positive_values,
    check_row_drop_threshold,
    DEFAULT_THRESHOLDS,
)

logger = logging.getLogger(__name__)

# Required columns after cleaning stage
CLEANED_DATA_REQUIRED_COLUMNS = [
    'datetime', 'open', 'high', 'low', 'close', 'volume'
]


def validate_cleaned_data(
    df: pd.DataFrame,
    original_row_count: Optional[int] = None,
    symbol: str = "",
    max_row_drop_pct: float = DEFAULT_THRESHOLDS['max_row_drop_pct'],
    max_nan_pct: float = DEFAULT_THRESHOLDS['max_nan_pct'],
    min_rows: int = DEFAULT_THRESHOLDS['min_rows'],
    max_gap_minutes: int = DEFAULT_THRESHOLDS['max_gap_minutes'],
) -> ValidationResult:
    """
    Validate cleaned OHLCV data output from Stage 2.

    Args:
        df: Cleaned DataFrame to validate
        original_row_count: Original row count before cleaning (for drop check)
        symbol: Symbol name for error context
        max_row_drop_pct: Maximum allowed percentage of rows dropped
        max_nan_pct: Maximum allowed percentage of NaN values
        min_rows: Minimum required rows
        max_gap_minutes: Maximum allowed gap between bars in minutes

    Returns:
        ValidationResult with validation status, errors, and metrics
    """
    stage_name = f"cleaned_data_{symbol}" if symbol else "cleaned_data"
    result = ValidationResult(passed=True, stage=stage_name)

    logger.info(f"[{stage_name}] Validating cleaned data output...")

    # Basic DataFrame checks
    check_dataframe_basics(df, result, min_rows=min_rows)
    if not result.passed:
        result.log_summary()
        return result

    # Required columns
    check_required_columns(df, CLEANED_DATA_REQUIRED_COLUMNS, result)

    # Datetime validation
    check_datetime_column(df, result)

    # OHLCV relationships
    check_ohlcv_relationships(df, result)

    # Positive values in price and volume columns
    check_positive_values(df, ['open', 'high', 'low', 'close', 'volume'], result)

    # NaN values (critical in OHLCV columns)
    check_nan_values(df, result, max_nan_pct=max_nan_pct)

    # Infinite values
    check_infinite_values(df, result)

    # Row drop threshold (if original count provided)
    if original_row_count is not None:
        check_row_drop_threshold(
            original_row_count,
            len(df),
            result,
            max_drop_pct=max_row_drop_pct,
            context=f"cleaning {symbol}" if symbol else "cleaning"
        )

    # Time gap analysis
    _check_time_gaps(df, result, max_gap_minutes)

    # Symbol column check (if multi-symbol data)
    if 'symbol' in df.columns:
        symbols = df['symbol'].unique()
        result.metrics['symbols'] = list(symbols)
        result.metrics['symbol_count'] = len(symbols)

    result.log_summary()
    return result


def _check_time_gaps(
    df: pd.DataFrame,
    result: ValidationResult,
    max_gap_minutes: int
) -> None:
    """
    Check for excessive time gaps in the data.

    Args:
        df: DataFrame to validate
        result: ValidationResult to update
        max_gap_minutes: Maximum allowed gap in minutes
    """
    if 'datetime' not in df.columns or len(df) < 2:
        return

    # Sort by datetime and calculate gaps
    df_sorted = df.sort_values('datetime')
    time_diffs = df_sorted['datetime'].diff()

    # Get median gap (expected bar frequency)
    median_gap = time_diffs.median()
    max_gap_threshold = pd.Timedelta(minutes=max_gap_minutes)

    # Count gaps exceeding threshold
    large_gaps = time_diffs[time_diffs > max_gap_threshold]
    n_large_gaps = len(large_gaps)

    result.metrics['median_gap'] = str(median_gap)
    result.metrics['max_gap'] = str(time_diffs.max())
    result.metrics['large_gap_count'] = n_large_gaps

    if n_large_gaps > 0:
        max_gap = time_diffs.max()
        # Warn but don't fail for gaps (could be market hours)
        result.add_warning(
            f"Found {n_large_gaps} gaps exceeding {max_gap_minutes} minutes "
            f"(max gap: {max_gap})"
        )


def validate_cleaned_data_for_features(
    df: pd.DataFrame,
    symbol: str = ""
) -> ValidationResult:
    """
    Validate cleaned data specifically as input for feature engineering.

    This is the START validation for Stage 3 (Feature Engineering).

    Args:
        df: DataFrame to validate
        symbol: Symbol name for error context

    Returns:
        ValidationResult
    """
    stage_name = f"feature_input_{symbol}" if symbol else "feature_input"
    result = ValidationResult(passed=True, stage=stage_name)

    logger.info(f"[{stage_name}] Validating input for feature engineering...")

    # All the standard cleaned data checks
    check_dataframe_basics(df, result, min_rows=500)

    if not result.passed:
        result.log_summary()
        return result

    # Required columns for feature engineering
    required = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    check_required_columns(df, required, result)

    # Must have no NaN in OHLCV for feature calculation
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in ohlcv_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                result.add_error(
                    f"Column '{col}' has {nan_count} NaN values - "
                    f"cannot proceed with feature engineering"
                )

    # Datetime must be sorted for proper feature calculation
    if 'datetime' in df.columns:
        if not df['datetime'].is_monotonic_increasing:
            result.add_error("Datetime column must be sorted for feature engineering")

    check_ohlcv_relationships(df, result)
    check_infinite_values(df, result)

    result.log_summary()
    return result
