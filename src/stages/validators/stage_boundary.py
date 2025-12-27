"""
Stage Boundary Validation Module.

Provides ValidationResult dataclass and common validation utilities
for stage input/output validation in the ML pipeline.

These validators are called at stage boundaries to ensure data quality
before passing to the next stage. Fail-fast behavior prevents propagating
invalid data through the pipeline.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Result of a stage boundary validation.

    Attributes:
        passed: Whether all validations passed
        errors: List of critical errors that should fail the pipeline
        warnings: List of non-critical warnings for review
        metrics: Dictionary of validation metrics (row counts, percentages, etc.)
        stage: Name of the stage being validated
    """
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    stage: str = ""

    def __post_init__(self):
        """Update passed status based on errors."""
        if self.errors:
            self.passed = False

    def add_error(self, error: str) -> None:
        """Add an error and mark validation as failed."""
        self.errors.append(error)
        self.passed = False

    def add_warning(self, warning: str) -> None:
        """Add a warning without failing validation."""
        self.warnings.append(warning)

    def raise_if_failed(self) -> None:
        """Raise ValueError if validation failed."""
        if not self.passed:
            error_list = "\n".join(f"  - {e}" for e in self.errors)
            raise ValueError(
                f"Validation failed at stage '{self.stage}':\n{error_list}"
            )

    def log_summary(self) -> None:
        """Log a summary of validation results."""
        if self.passed:
            logger.info(f"[{self.stage}] Validation PASSED")
        else:
            logger.error(f"[{self.stage}] Validation FAILED with {len(self.errors)} errors")
            for error in self.errors:
                logger.error(f"  - {error}")

        if self.warnings:
            for warning in self.warnings:
                logger.warning(f"  - {warning}")


# Default thresholds for validation checks
DEFAULT_THRESHOLDS = {
    # Maximum percentage of rows that can be dropped during cleaning
    'max_row_drop_pct': 5.0,
    # Maximum percentage of NaN values allowed in any column
    'max_nan_pct': 1.0,
    # Minimum number of rows required after processing
    'min_rows': 1000,
    # Maximum percentage of any single label class (avoid extreme imbalance)
    'max_label_class_pct': 70.0,
    # Minimum percentage of any single label class
    'min_label_class_pct': 10.0,
    # Maximum percentage of invalid labels (-99) allowed
    'max_invalid_label_pct': 20.0,
    # Minimum number of features expected
    'min_feature_count': 20,
    # Maximum gap allowed between bars (in minutes)
    'max_gap_minutes': 60,
}


def check_dataframe_basics(
    df: pd.DataFrame,
    result: ValidationResult,
    min_rows: int = 1000
) -> None:
    """
    Check basic DataFrame validity.

    Args:
        df: DataFrame to validate
        result: ValidationResult to update
        min_rows: Minimum required row count
    """
    if df is None:
        result.add_error("DataFrame is None")
        return

    if len(df) == 0:
        result.add_error("DataFrame is empty")
        return

    if len(df) < min_rows:
        result.add_error(
            f"DataFrame has only {len(df):,} rows, "
            f"minimum required is {min_rows:,}"
        )

    result.metrics['row_count'] = len(df)
    result.metrics['column_count'] = len(df.columns)


def check_required_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    result: ValidationResult
) -> None:
    """
    Check that all required columns exist.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        result: ValidationResult to update
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        result.add_error(f"Missing required columns: {sorted(missing)}")


def check_nan_values(
    df: pd.DataFrame,
    result: ValidationResult,
    max_nan_pct: float = 1.0,
    exclude_columns: Optional[List[str]] = None
) -> None:
    """
    Check for NaN values in DataFrame.

    Args:
        df: DataFrame to validate
        result: ValidationResult to update
        max_nan_pct: Maximum allowed percentage of NaN values per column
        exclude_columns: Columns to exclude from NaN check
    """
    exclude_columns = exclude_columns or []
    columns_to_check = [c for c in df.columns if c not in exclude_columns]

    nan_counts = df[columns_to_check].isna().sum()
    nan_pcts = (nan_counts / len(df)) * 100

    high_nan_cols = nan_pcts[nan_pcts > max_nan_pct]

    if len(high_nan_cols) > 0:
        for col, pct in high_nan_cols.items():
            if pct > 50:
                result.add_error(f"Column '{col}' has {pct:.1f}% NaN values")
            else:
                result.add_warning(f"Column '{col}' has {pct:.1f}% NaN values")

    result.metrics['nan_columns'] = dict(high_nan_cols)


def check_infinite_values(
    df: pd.DataFrame,
    result: ValidationResult
) -> None:
    """
    Check for infinite values in numeric columns.

    Args:
        df: DataFrame to validate
        result: ValidationResult to update
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    inf_counts = {}
    for col in numeric_cols:
        n_inf = np.isinf(df[col]).sum()
        if n_inf > 0:
            inf_counts[col] = int(n_inf)
            pct = n_inf / len(df) * 100
            result.add_error(f"Column '{col}' has {n_inf} infinite values ({pct:.2f}%)")

    result.metrics['infinite_value_columns'] = inf_counts


def check_datetime_column(
    df: pd.DataFrame,
    result: ValidationResult,
    column: str = 'datetime'
) -> None:
    """
    Validate datetime column.

    Args:
        df: DataFrame to validate
        result: ValidationResult to update
        column: Name of datetime column
    """
    if column not in df.columns:
        result.add_error(f"Datetime column '{column}' not found")
        return

    # Check type
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        result.add_error(f"Column '{column}' is not datetime type")
        return

    # Check for duplicates
    n_dups = df[column].duplicated().sum()
    if n_dups > 0:
        result.add_error(f"Found {n_dups:,} duplicate timestamps")

    # Check monotonicity
    if not df[column].is_monotonic_increasing:
        result.add_warning("Datetime column is not monotonically increasing")

    # Store date range in metrics
    result.metrics['datetime_min'] = str(df[column].min())
    result.metrics['datetime_max'] = str(df[column].max())
    result.metrics['duplicate_timestamps'] = int(n_dups)


def check_ohlcv_relationships(
    df: pd.DataFrame,
    result: ValidationResult
) -> None:
    """
    Validate OHLCV price relationships.

    Args:
        df: DataFrame to validate
        result: ValidationResult to update
    """
    required = ['open', 'high', 'low', 'close']
    missing = [c for c in required if c not in df.columns]
    if missing:
        return  # Skip if columns missing (already caught elsewhere)

    # high >= low
    violations = (df['high'] < df['low']).sum()
    if violations > 0:
        result.add_error(f"OHLCV violation: high < low in {violations} rows")

    # high >= open and high >= close
    violations = ((df['high'] < df['open']) | (df['high'] < df['close'])).sum()
    if violations > 0:
        result.add_error(f"OHLCV violation: high < open or close in {violations} rows")

    # low <= open and low <= close
    violations = ((df['low'] > df['open']) | (df['low'] > df['close'])).sum()
    if violations > 0:
        result.add_error(f"OHLCV violation: low > open or close in {violations} rows")


def check_positive_values(
    df: pd.DataFrame,
    columns: List[str],
    result: ValidationResult
) -> None:
    """
    Check that specified columns have positive values.

    Args:
        df: DataFrame to validate
        columns: Columns that must have positive values
        result: ValidationResult to update
    """
    for col in columns:
        if col not in df.columns:
            continue
        n_nonpositive = (df[col] <= 0).sum()
        if n_nonpositive > 0:
            pct = n_nonpositive / len(df) * 100
            result.add_error(
                f"Column '{col}' has {n_nonpositive} non-positive values ({pct:.1f}%)"
            )


def check_row_drop_threshold(
    original_count: int,
    current_count: int,
    result: ValidationResult,
    max_drop_pct: float = 5.0,
    context: str = ""
) -> None:
    """
    Check if too many rows were dropped during processing.

    Args:
        original_count: Original row count
        current_count: Current row count after processing
        result: ValidationResult to update
        max_drop_pct: Maximum allowed percentage of rows to drop
        context: Context string for error message
    """
    if original_count == 0:
        result.add_error("Original row count is 0")
        return

    dropped = original_count - current_count
    drop_pct = (dropped / original_count) * 100

    result.metrics['rows_dropped'] = dropped
    result.metrics['drop_percentage'] = drop_pct

    ctx = f" ({context})" if context else ""

    if drop_pct > max_drop_pct:
        result.add_error(
            f"Dropped {dropped:,} rows ({drop_pct:.1f}%){ctx}, "
            f"exceeds threshold of {max_drop_pct}%"
        )
    elif drop_pct > max_drop_pct / 2:
        result.add_warning(
            f"Dropped {dropped:,} rows ({drop_pct:.1f}%){ctx}"
        )
