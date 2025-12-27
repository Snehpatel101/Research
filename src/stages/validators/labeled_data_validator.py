"""
Labeled Data Validator.

Validates the output of Stage 4/6 (Labeling) before passing to Stage 5/7.

Checks:
- Label columns present with valid values (-1, 0, 1, -99)
- Label distribution not excessively imbalanced
- Invalid label percentage within threshold
- Supporting columns present (bars_to_hit, quality, etc.)
- No data corruption from labeling
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
    check_row_drop_threshold,
    DEFAULT_THRESHOLDS,
)
from .data_contract import VALID_LABELS, INVALID_LABEL_SENTINEL

logger = logging.getLogger(__name__)


def validate_labeled_data(
    df: pd.DataFrame,
    horizons: List[int],
    symbol: str = "",
    max_invalid_label_pct: float = DEFAULT_THRESHOLDS['max_invalid_label_pct'],
    max_class_pct: float = DEFAULT_THRESHOLDS['max_label_class_pct'],
    min_class_pct: float = DEFAULT_THRESHOLDS['min_label_class_pct'],
) -> ValidationResult:
    """
    Validate labeled data output from Stage 4 or Stage 6.

    Args:
        df: DataFrame with labels to validate
        horizons: List of label horizons (e.g., [5, 20])
        symbol: Symbol name for error context
        max_invalid_label_pct: Maximum allowed percentage of invalid labels (-99)
        max_class_pct: Maximum allowed percentage for any single class
        min_class_pct: Minimum required percentage for any single class

    Returns:
        ValidationResult with validation status, errors, and metrics
    """
    stage_name = f"labeled_data_{symbol}" if symbol else "labeled_data"
    result = ValidationResult(passed=True, stage=stage_name)

    logger.info(f"[{stage_name}] Validating labeled data output...")

    # Basic DataFrame checks
    check_dataframe_basics(df, result, min_rows=500)
    if not result.passed:
        result.log_summary()
        return result

    # Check datetime column
    check_datetime_column(df, result)

    # Validate labels for each horizon
    label_stats = {}
    for horizon in horizons:
        horizon_result = _validate_horizon_labels(
            df, horizon, result,
            max_invalid_label_pct, max_class_pct, min_class_pct
        )
        label_stats[f'h{horizon}'] = horizon_result

    result.metrics['label_stats'] = label_stats

    # Check for required supporting columns
    _check_supporting_columns(df, horizons, result)

    result.log_summary()
    return result


def _validate_horizon_labels(
    df: pd.DataFrame,
    horizon: int,
    result: ValidationResult,
    max_invalid_label_pct: float,
    max_class_pct: float,
    min_class_pct: float
) -> dict:
    """
    Validate labels for a specific horizon.

    Args:
        df: DataFrame to validate
        horizon: Label horizon
        result: ValidationResult to update
        max_invalid_label_pct: Maximum invalid label percentage
        max_class_pct: Maximum class percentage
        min_class_pct: Minimum class percentage

    Returns:
        Dictionary with label statistics
    """
    label_col = f'label_h{horizon}'

    if label_col not in df.columns:
        result.add_error(f"Label column '{label_col}' not found")
        return {}

    labels = df[label_col]
    total = len(labels)

    # Check for invalid label values
    valid_with_sentinel = VALID_LABELS | {INVALID_LABEL_SENTINEL}
    unique_vals = set(labels.dropna().unique())
    invalid_vals = unique_vals - valid_with_sentinel

    if invalid_vals:
        result.add_error(f"Invalid label values in '{label_col}': {invalid_vals}")

    # Calculate distribution
    n_long = (labels == 1).sum()
    n_short = (labels == -1).sum()
    n_neutral = (labels == 0).sum()
    n_invalid = (labels == INVALID_LABEL_SENTINEL).sum()

    # Valid labels only (excluding -99)
    n_valid = n_long + n_short + n_neutral

    stats = {
        'total': int(total),
        'long': int(n_long),
        'short': int(n_short),
        'neutral': int(n_neutral),
        'invalid': int(n_invalid),
        'valid': int(n_valid),
    }

    # Calculate percentages based on valid labels
    if n_valid > 0:
        stats['long_pct'] = n_long / n_valid * 100
        stats['short_pct'] = n_short / n_valid * 100
        stats['neutral_pct'] = n_neutral / n_valid * 100
    else:
        stats['long_pct'] = 0
        stats['short_pct'] = 0
        stats['neutral_pct'] = 0

    stats['invalid_pct'] = n_invalid / total * 100

    # Check invalid label threshold
    if stats['invalid_pct'] > max_invalid_label_pct:
        result.add_error(
            f"H{horizon}: {stats['invalid_pct']:.1f}% invalid labels (-99), "
            f"exceeds threshold of {max_invalid_label_pct}%"
        )

    # Check class balance (only if we have valid labels)
    if n_valid > 0:
        for class_name, pct in [('long', stats['long_pct']),
                                 ('short', stats['short_pct']),
                                 ('neutral', stats['neutral_pct'])]:
            if pct > max_class_pct:
                result.add_warning(
                    f"H{horizon}: {class_name} class has {pct:.1f}%, "
                    f"exceeds {max_class_pct}% threshold"
                )
            if pct < min_class_pct:
                result.add_warning(
                    f"H{horizon}: {class_name} class has only {pct:.1f}%, "
                    f"below {min_class_pct}% threshold"
                )

    logger.info(
        f"  H{horizon}: L={stats['long_pct']:.1f}% S={stats['short_pct']:.1f}% "
        f"N={stats['neutral_pct']:.1f}% Invalid={stats['invalid_pct']:.1f}%"
    )

    return stats


def _check_supporting_columns(
    df: pd.DataFrame,
    horizons: List[int],
    result: ValidationResult
) -> None:
    """
    Check for supporting columns (bars_to_hit, quality, etc.).

    Args:
        df: DataFrame to validate
        horizons: List of horizons
        result: ValidationResult to update
    """
    expected_columns = []
    for h in horizons:
        expected_columns.extend([
            f'bars_to_hit_h{h}',
            f'mae_h{h}',
        ])

    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        # Warning only - some intermediate stages may not have all columns
        result.add_warning(f"Missing supporting columns: {missing}")


def validate_labels_for_ga(
    df: pd.DataFrame,
    horizons: List[int],
    symbol: str = ""
) -> ValidationResult:
    """
    Validate labeled data specifically as input for GA optimization.

    This is the START validation for Stage 5 (GA Optimization).

    Args:
        df: DataFrame to validate
        horizons: List of horizons
        symbol: Symbol name for error context

    Returns:
        ValidationResult
    """
    stage_name = f"ga_input_{symbol}" if symbol else "ga_input"
    result = ValidationResult(passed=True, stage=stage_name)

    logger.info(f"[{stage_name}] Validating input for GA optimization...")

    # Basic checks
    check_dataframe_basics(df, result, min_rows=1000)

    if not result.passed:
        result.log_summary()
        return result

    # Need close and ATR for GA fitness evaluation
    required = ['close', 'atr_14']
    check_required_columns(df, required, result)

    # Need initial labels for each horizon
    for horizon in horizons:
        label_col = f'label_h{horizon}'
        if label_col not in df.columns:
            result.add_error(f"Initial label column '{label_col}' required for GA")

    # Datetime for proper evaluation
    check_datetime_column(df, result)

    result.log_summary()
    return result


def validate_labels_for_splits(
    df: pd.DataFrame,
    horizons: List[int],
    symbol: str = ""
) -> ValidationResult:
    """
    Validate final labeled data as input for train/val/test splits.

    This is the START validation for Stage 7 (Create Splits).

    Args:
        df: DataFrame to validate
        horizons: List of horizons
        symbol: Symbol name for error context

    Returns:
        ValidationResult
    """
    stage_name = f"splits_input_{symbol}" if symbol else "splits_input"
    result = ValidationResult(passed=True, stage=stage_name)

    logger.info(f"[{stage_name}] Validating input for data splits...")

    # Basic checks
    check_dataframe_basics(df, result, min_rows=1000)

    if not result.passed:
        result.log_summary()
        return result

    # Datetime must be present and sorted (strict requirement for splits)
    check_datetime_column(df, result)

    # For splits, datetime MUST be monotonically increasing (not just a warning)
    if 'datetime' in df.columns:
        if not df['datetime'].is_monotonic_increasing:
            result.add_error("Datetime column must be sorted for time-series splits")

    # Need final labels and quality scores
    for horizon in horizons:
        label_col = f'label_h{horizon}'
        quality_col = f'quality_h{horizon}'
        weight_col = f'sample_weight_h{horizon}'

        if label_col not in df.columns:
            result.add_error(f"Label column '{label_col}' required for splits")

        if quality_col not in df.columns:
            result.add_warning(f"Quality column '{quality_col}' not found")

        if weight_col not in df.columns:
            result.add_warning(f"Sample weight column '{weight_col}' not found")

    # Validate label distribution for each horizon
    for horizon in horizons:
        _validate_horizon_labels(
            df, horizon, result,
            max_invalid_label_pct=DEFAULT_THRESHOLDS['max_invalid_label_pct'],
            max_class_pct=DEFAULT_THRESHOLDS['max_label_class_pct'],
            min_class_pct=DEFAULT_THRESHOLDS['min_label_class_pct']
        )

    result.log_summary()
    return result
