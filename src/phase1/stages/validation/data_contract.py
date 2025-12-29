"""
Data Contract Validation for ML Pipeline.

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
class DataContract:
    """Defines expected schema and constraints for pipeline data."""

    # Required OHLCV columns
    REQUIRED_OHLCV: set[str] = None  # Initialized in __post_init__

    # Valid label values (excluding sentinel)
    VALID_LABELS: set[int] = None  # Initialized in __post_init__

    INVALID_LABEL_SENTINEL: int = -99

    # Numeric columns that must be positive
    POSITIVE_COLUMNS: set[str] = None  # Initialized in __post_init__

    def __post_init__(self):
        """Initialize class-level sets."""
        if self.REQUIRED_OHLCV is None:
            self.REQUIRED_OHLCV = {'datetime', 'open', 'high', 'low', 'close', 'volume'}
        if self.VALID_LABELS is None:
            self.VALID_LABELS = {-1, 0, 1}
        if self.POSITIVE_COLUMNS is None:
            self.POSITIVE_COLUMNS = {'open', 'high', 'low', 'close', 'volume'}

    @staticmethod
    def validate_ohlc_relationships(df: pd.DataFrame) -> list[str]:
        """Validate high >= low, high >= open/close, low <= open/close."""
        errors = []

        if (df['high'] < df['low']).any():
            n_violations = (df['high'] < df['low']).sum()
            errors.append(f"high < low in {n_violations} rows")

        if (df['high'] < df['open']).any():
            n_violations = (df['high'] < df['open']).sum()
            errors.append(f"high < open in {n_violations} rows")

        if (df['high'] < df['close']).any():
            n_violations = (df['high'] < df['close']).sum()
            errors.append(f"high < close in {n_violations} rows")

        if (df['low'] > df['open']).any():
            n_violations = (df['low'] > df['open']).sum()
            errors.append(f"low > open in {n_violations} rows")

        if (df['low'] > df['close']).any():
            n_violations = (df['low'] > df['close']).sum()
            errors.append(f"low > close in {n_violations} rows")

        return errors


# Module-level constants for direct access without instantiation
REQUIRED_OHLCV = {'datetime', 'open', 'high', 'low', 'close', 'volume'}
VALID_LABELS = {-1, 0, 1}
INVALID_LABEL_SENTINEL = -99
POSITIVE_COLUMNS = {'open', 'high', 'low', 'close', 'volume'}


def validate_ohlcv_schema(df: pd.DataFrame, stage: str = "unknown") -> None:
    """
    Validate OHLCV data meets contract requirements.

    Raises ValueError with specific error messages if validation fails.

    Args:
        df: DataFrame to validate
        stage: Name of the pipeline stage for error context

    Raises:
        ValueError: If any contract violations are found
    """
    errors = []

    # Check required columns
    missing = REQUIRED_OHLCV - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {missing}")

    # Check for empty dataframe
    if len(df) == 0:
        errors.append("DataFrame is empty")

    # Check datetime is datetime type
    if 'datetime' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            errors.append("'datetime' column must be datetime type")

        # Check for duplicates
        if df['datetime'].duplicated().any():
            n_dups = df['datetime'].duplicated().sum()
            errors.append(f"Found {n_dups} duplicate timestamps")

        # Check monotonic increasing
        if not df['datetime'].is_monotonic_increasing:
            errors.append("'datetime' must be monotonically increasing")

    # Check positive values
    for col in POSITIVE_COLUMNS:
        if col in df.columns:
            if (df[col] <= 0).any():
                n_neg = (df[col] <= 0).sum()
                errors.append(f"Non-positive values in '{col}': {n_neg} rows")

    # Check OHLC relationships
    if REQUIRED_OHLCV.issubset(set(df.columns)):
        ohlc_errors = DataContract.validate_ohlc_relationships(df)
        errors.extend(ohlc_errors)

    # Raise all errors at once
    if errors:
        raise ValueError(
            f"Data contract violation at {stage}:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )


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
        raise ValueError(
            "Label validation failed:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )


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
            mask &= (df[col] != INVALID_LABEL_SENTINEL)

    return df[mask].copy()


def get_dataset_fingerprint(df: pd.DataFrame) -> dict:
    """
    Generate fingerprint for data lineage tracking.

    Args:
        df: DataFrame to fingerprint

    Returns:
        Dictionary containing dataset metadata for tracking
    """
    return {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'columns': sorted(df.columns.tolist()),
        'datetime_min': str(df['datetime'].min()) if 'datetime' in df.columns else None,
        'datetime_max': str(df['datetime'].max()) if 'datetime' in df.columns else None,
        'schema_hash': hash(tuple(sorted(df.columns))),
    }


def validate_feature_lookahead(
    df: pd.DataFrame,
    feature_columns: list[str],
    label_columns: list[str]
) -> None:
    """
    Validate that features do not leak future information.

    This is a structural check that ensures feature and label columns
    are properly defined. Actual point-in-time validation requires
    knowing the computation logic of each feature.

    Args:
        df: DataFrame with features and labels
        feature_columns: List of feature column names
        label_columns: List of label column names

    Raises:
        ValueError: If obvious lookahead issues are detected
    """
    errors = []

    # Check that feature and label columns don't overlap
    feature_set = set(feature_columns)
    label_set = set(label_columns)
    overlap = feature_set & label_set

    if overlap:
        errors.append(f"Feature and label columns overlap: {overlap}")

    # Check that all specified columns exist
    missing_features = feature_set - set(df.columns)
    if missing_features:
        errors.append(f"Feature columns not in DataFrame: {missing_features}")

    missing_labels = label_set - set(df.columns)
    if missing_labels:
        errors.append(f"Label columns not in DataFrame: {missing_labels}")

    if errors:
        raise ValueError(
            "Lookahead validation failed:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )


def summarize_label_distribution(
    df: pd.DataFrame,
    label_columns: list[str]
) -> dict[str, dict]:
    """
    Summarize label distribution for each label column.

    Args:
        df: DataFrame with label columns
        label_columns: List of label column names

    Returns:
        Dictionary mapping column names to distribution stats
    """
    summary = {}

    for col in label_columns:
        if col not in df.columns:
            continue

        # Exclude invalid sentinel
        valid_mask = df[col] != INVALID_LABEL_SENTINEL
        valid_labels = df.loc[valid_mask, col]

        value_counts = valid_labels.value_counts().to_dict()
        total = len(valid_labels)

        summary[col] = {
            'total_valid': total,
            'total_invalid': (~valid_mask).sum(),
            'distribution': {
                int(k): int(v) for k, v in value_counts.items()
            },
            'percentages': {
                int(k): round(v / total * 100, 2) if total > 0 else 0
                for k, v in value_counts.items()
            }
        }

    return summary
