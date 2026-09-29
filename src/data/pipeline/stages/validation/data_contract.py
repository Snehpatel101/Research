"""
OHLCV Validation Schema for ML Pipeline.

This module provides OHLCV-specific data validation, distinct from the canonical
DataContract (src/core/contracts/data_contract.py) which defines model data requirements.

Point-in-Time Contract:
- Features[t]: computed from data[0:t-1] (excludes current bar)
- Prediction[t]: made at close of bar t
- Entry[t]: assumed at open of bar t+1
- Label[t]: forward return from bar[t+1] open to barrier hit

Invalid Label Sentinel: -99 (excluded from training/evaluation)
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class OHLCVValidationSchema:
    """
    Defines expected schema and constraints for OHLCV pipeline data.

    NOTE: This is NOT the same as DataContract (src/core/contracts/data_contract.py).
    - OHLCVValidationSchema: Validates OHLCV data structure and relationships
    - DataContract: Defines model data requirements with full lineage tracking
    """

    # Required OHLCV columns
    REQUIRED_OHLCV: set[str] | None = None  # Initialized in __post_init__

    # Valid label values (excluding sentinel)
    VALID_LABELS: set[int] | None = None  # Initialized in __post_init__

    INVALID_LABEL_SENTINEL: int = -99

    # Numeric columns that must be positive
    POSITIVE_COLUMNS: set[str] | None = None  # Initialized in __post_init__

    def __post_init__(self):
        """Initialize class-level sets."""
        if self.REQUIRED_OHLCV is None:
            self.REQUIRED_OHLCV = {"datetime", "open", "high", "low", "close", "volume"}
        if self.VALID_LABELS is None:
            self.VALID_LABELS = {-1, 0, 1}
        if self.POSITIVE_COLUMNS is None:
            self.POSITIVE_COLUMNS = {"open", "high", "low", "close", "volume"}


# Module-level constants for direct access without instantiation
REQUIRED_OHLCV = {"datetime", "open", "high", "low", "close", "volume"}
VALID_LABELS = {-1, 0, 1}
INVALID_LABEL_SENTINEL = -99
POSITIVE_COLUMNS = {"open", "high", "low", "close", "volume"}


def validate_labels(df: pd.DataFrame, label_columns: list[str]) -> None:
    """
    Validate label columns meet contract requirements.

    Invalid labels (-99) are allowed but flagged in report.

    Args:
        df: DataFrame containing label columns
        label_columns: List of label column names to validate

    Raises:
        ValueError: If label columns are missing or contain invalid values
    """
    errors = []

    for col in label_columns:
        if col not in df.columns:
            errors.append(f"Label column '{col}' not found")
            continue

        unique_vals = set(df[col].dropna().unique())
        valid_with_sentinel = VALID_LABELS | {INVALID_LABEL_SENTINEL}
        invalid_vals = unique_vals - valid_with_sentinel

        if invalid_vals:
            errors.append(f"Invalid label values in '{col}': {invalid_vals}")

    if errors:
        raise ValueError("Label validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


def filter_invalid_labels(df: pd.DataFrame, label_columns: list[str]) -> pd.DataFrame:
    """
    Remove rows with invalid label sentinel (-99) from any label column.

    Args:
        df: DataFrame to filter
        label_columns: List of label column names to check

    Returns:
        Filtered DataFrame with invalid label rows removed
    """
    mask = pd.Series(True, index=df.index)

    for col in label_columns:
        if col in df.columns:
            mask &= df[col] != INVALID_LABEL_SENTINEL

    return df[mask].copy()
