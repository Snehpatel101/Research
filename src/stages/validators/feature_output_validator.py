"""
Feature Output Validator.

Validates the output of Stage 3 (Feature Engineering) before passing to Stage 4 (Labeling).

Checks:
- All OHLCV columns preserved
- Minimum feature count generated
- No excessive NaN/Inf in feature columns
- Feature columns have valid ranges (not constant, not extreme)
- ATR column present (required for labeling)
"""
import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from .stage_boundary import (
    ValidationResult,
    check_dataframe_basics,
    check_required_columns,
    check_datetime_column,
    check_ohlcv_relationships,
    check_row_drop_threshold,
    DEFAULT_THRESHOLDS,
)

logger = logging.getLogger(__name__)

# OHLCV columns that must be preserved after feature engineering
OHLCV_COLUMNS = ['datetime', 'open', 'high', 'low', 'close', 'volume']

# Critical columns required for labeling stage
LABELING_REQUIRED_COLUMNS = ['datetime', 'close', 'high', 'low', 'open', 'atr_14']


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify feature columns in a DataFrame.

    Args:
        df: DataFrame to analyze

    Returns:
        List of feature column names
    """
    excluded_prefixes = (
        'label_', 'bars_to_hit_', 'mae_', 'mfe_',
        'quality_', 'sample_weight_'
    )
    excluded_cols = {'datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume'}

    feature_cols = [
        c for c in df.columns
        if c not in excluded_cols
        and not any(c.startswith(p) for p in excluded_prefixes)
    ]
    return feature_cols


def validate_feature_output(
    df: pd.DataFrame,
    original_row_count: Optional[int] = None,
    symbol: str = "",
    min_feature_count: int = DEFAULT_THRESHOLDS['min_feature_count'],
    max_row_drop_pct: float = DEFAULT_THRESHOLDS['max_row_drop_pct'],
    max_nan_pct: float = 5.0,  # Allow more NaN in features (warmup period)
) -> ValidationResult:
    """
    Validate feature-engineered data output from Stage 3.

    Args:
        df: DataFrame with features to validate
        original_row_count: Original row count before feature engineering
        symbol: Symbol name for error context
        min_feature_count: Minimum number of feature columns required
        max_row_drop_pct: Maximum allowed percentage of rows dropped
        max_nan_pct: Maximum allowed percentage of NaN values in features

    Returns:
        ValidationResult with validation status, errors, and metrics
    """
    stage_name = f"feature_output_{symbol}" if symbol else "feature_output"
    result = ValidationResult(passed=True, stage=stage_name)

    logger.info(f"[{stage_name}] Validating feature engineering output...")

    # Basic DataFrame checks
    check_dataframe_basics(df, result, min_rows=500)
    if not result.passed:
        result.log_summary()
        return result

    # Check OHLCV columns preserved
    check_required_columns(df, OHLCV_COLUMNS, result)

    # Check datetime
    check_datetime_column(df, result)

    # Check OHLCV relationships still valid
    check_ohlcv_relationships(df, result)

    # Identify feature columns
    feature_cols = get_feature_columns(df)
    result.metrics['feature_count'] = len(feature_cols)
    result.metrics['feature_columns'] = feature_cols[:20]  # First 20 for reference

    # Check minimum feature count
    if len(feature_cols) < min_feature_count:
        result.add_error(
            f"Only {len(feature_cols)} features generated, "
            f"minimum required is {min_feature_count}"
        )

    # Check for ATR column (required for labeling)
    if 'atr_14' not in df.columns:
        result.add_error("ATR column 'atr_14' not found - required for labeling stage")

    # Check feature column quality
    _check_feature_quality(df, feature_cols, result, max_nan_pct)

    # Row drop threshold
    if original_row_count is not None:
        check_row_drop_threshold(
            original_row_count,
            len(df),
            result,
            max_drop_pct=max_row_drop_pct,
            context=f"feature engineering {symbol}" if symbol else "feature engineering"
        )

    result.log_summary()
    return result


def _check_feature_quality(
    df: pd.DataFrame,
    feature_cols: List[str],
    result: ValidationResult,
    max_nan_pct: float
) -> None:
    """
    Check quality of feature columns.

    Args:
        df: DataFrame to validate
        feature_cols: List of feature column names
        result: ValidationResult to update
        max_nan_pct: Maximum allowed NaN percentage
    """
    high_nan_features = []
    constant_features = []
    inf_features = []

    for col in feature_cols:
        if col not in df.columns:
            continue

        series = df[col]

        # Check NaN percentage
        nan_pct = (series.isna().sum() / len(df)) * 100
        if nan_pct > max_nan_pct:
            high_nan_features.append((col, nan_pct))

        # Check for infinite values
        if series.dtype in [np.float64, np.float32]:
            n_inf = np.isinf(series).sum()
            if n_inf > 0:
                inf_features.append((col, n_inf))

        # Check for constant values (zero variance)
        valid_values = series.replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid_values) > 0 and valid_values.std() == 0:
            constant_features.append(col)

    # Report high NaN features
    if high_nan_features:
        result.metrics['high_nan_features'] = [
            {'column': col, 'nan_pct': pct} for col, pct in high_nan_features
        ]
        if len(high_nan_features) > 5:
            result.add_warning(
                f"{len(high_nan_features)} features have high NaN percentage (>{max_nan_pct}%)"
            )
        for col, pct in high_nan_features[:3]:
            result.add_warning(f"Feature '{col}' has {pct:.1f}% NaN values")

    # Report infinite value features
    if inf_features:
        result.metrics['infinite_value_features'] = [
            {'column': col, 'count': count} for col, count in inf_features
        ]
        for col, count in inf_features:
            result.add_error(f"Feature '{col}' has {count} infinite values")

    # Report constant features (warning only - some may be intentional)
    if constant_features:
        result.metrics['constant_features'] = constant_features
        if len(constant_features) > 3:
            result.add_warning(
                f"{len(constant_features)} features are constant (zero variance)"
            )


def validate_features_for_labeling(
    df: pd.DataFrame,
    symbol: str = ""
) -> ValidationResult:
    """
    Validate feature data specifically as input for labeling stage.

    This is the START validation for Stage 4 (Labeling).

    Args:
        df: DataFrame to validate
        symbol: Symbol name for error context

    Returns:
        ValidationResult
    """
    stage_name = f"labeling_input_{symbol}" if symbol else "labeling_input"
    result = ValidationResult(passed=True, stage=stage_name)

    logger.info(f"[{stage_name}] Validating input for labeling stage...")

    # Basic checks
    check_dataframe_basics(df, result, min_rows=500)

    if not result.passed:
        result.log_summary()
        return result

    # Required columns for labeling
    check_required_columns(df, LABELING_REQUIRED_COLUMNS, result)

    # Datetime must be sorted
    if 'datetime' in df.columns:
        if not df['datetime'].is_monotonic_increasing:
            result.add_error("Datetime column must be sorted for labeling")

    # ATR cannot have NaN (required for barrier calculation)
    if 'atr_14' in df.columns:
        atr_nan = df['atr_14'].isna().sum()
        atr_inf = np.isinf(df['atr_14']).sum()

        if atr_nan > 0:
            result.add_error(f"ATR column has {atr_nan} NaN values - cannot compute barriers")

        if atr_inf > 0:
            result.add_error(f"ATR column has {atr_inf} infinite values")

        # ATR should be positive
        atr_nonpositive = (df['atr_14'] <= 0).sum()
        if atr_nonpositive > 0:
            result.add_error(f"ATR column has {atr_nonpositive} non-positive values")

    # Close prices needed for forward return calculation
    if 'close' in df.columns:
        close_nan = df['close'].isna().sum()
        if close_nan > 0:
            result.add_error(f"Close column has {close_nan} NaN values")

    result.log_summary()
    return result
