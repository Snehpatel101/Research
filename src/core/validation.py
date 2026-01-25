"""
Input validation utilities - Fail fast at boundaries.

PHASE_0: This module provides validation functions that should be called
at system boundaries (data ingestion, model input, config parsing).

Design principle: FAIL FAST
- Validate early, crash with clear error messages
- Don't let invalid data propagate through the pipeline

Functions:
- validate_input_shape(): Validate tensor shapes
- validate_labels(): Validate label arrays
- validate_features(): Validate feature columns
- validate_dataframe(): Validate OHLCV DataFrames
- validate_model_name(): Validate model name exists
- validate_config(): Validate configuration objects
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.core.constants import (
    ALL_MODELS,
    CANONICAL_TIMEFRAMES,
    N_CLASSES,
    OHLCV_COLUMNS,
)
from src.core.exceptions import ValidationError

# =============================================================================
# VALIDATION ERROR
# =============================================================================


# =============================================================================
# ARRAY VALIDATION
# =============================================================================


def validate_input_shape(
    X: np.ndarray,
    expected_rank: int,
    context: str = "input",
    allow_nan: bool = False,
) -> None:
    """
    Validate input tensor shape and content.

    Args:
        X: Input numpy array
        expected_rank: Expected number of dimensions (2, 3, or 4)
        context: Context string for error messages
        allow_nan: Whether to allow NaN values

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(X, np.ndarray):
        raise ValidationError(
            f"{context} must be a numpy array",
            field="X",
            expected="np.ndarray",
            actual=type(X).__name__,
        )

    if X.ndim != expected_rank:
        raise ValidationError(
            f"{context} must be {expected_rank}D",
            field="X.ndim",
            expected=expected_rank,
            actual=f"{X.ndim}D with shape {X.shape}",
        )

    if X.size == 0:
        raise ValidationError(f"{context} is empty", field="X")

    if not allow_nan and np.isnan(X).any():
        nan_count = np.isnan(X).sum()
        nan_pct = nan_count / X.size * 100
        raise ValidationError(
            f"{context} contains {nan_count} NaN values ({nan_pct:.2f}%)",
            field="X",
        )

    if np.isinf(X).any():
        inf_count = np.isinf(X).sum()
        raise ValidationError(
            f"{context} contains {inf_count} infinite values",
            field="X",
        )


def validate_labels(
    y: np.ndarray,
    n_samples: int,
    context: str = "labels",
    n_classes: int = N_CLASSES,
) -> None:
    """
    Validate label array.

    Args:
        y: Label array
        n_samples: Expected number of samples
        context: Context string for error messages
        n_classes: Expected number of classes

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(y, np.ndarray):
        raise ValidationError(
            f"{context} must be a numpy array",
            field="y",
            expected="np.ndarray",
            actual=type(y).__name__,
        )

    if y.ndim != 1:
        raise ValidationError(
            f"{context} must be 1D",
            field="y.ndim",
            expected=1,
            actual=y.ndim,
        )

    if len(y) != n_samples:
        raise ValidationError(
            f"{context} length doesn't match features",
            field="y",
            expected=n_samples,
            actual=len(y),
        )

    unique_labels = np.unique(y[~np.isnan(y)])
    valid_labels = {-1, 0, 1}  # Standard labels
    invalid = set(unique_labels) - valid_labels
    if invalid:
        raise ValidationError(
            f"{context} contains invalid label values: {sorted(invalid)}",
            field="y",
            expected=sorted(valid_labels),
            actual=sorted(unique_labels),
        )


def validate_probabilities(
    probs: np.ndarray,
    n_samples: int,
    n_classes: int = N_CLASSES,
    context: str = "probabilities",
) -> None:
    """
    Validate probability array.

    Args:
        probs: Probability array (n_samples, n_classes)
        n_samples: Expected number of samples
        n_classes: Expected number of classes
        context: Context string for error messages

    Raises:
        ValidationError: If validation fails
    """
    if probs.ndim != 2:
        raise ValidationError(
            f"{context} must be 2D",
            field="probs.ndim",
            expected=2,
            actual=probs.ndim,
        )

    if probs.shape != (n_samples, n_classes):
        raise ValidationError(
            f"{context} shape mismatch",
            field="probs.shape",
            expected=(n_samples, n_classes),
            actual=probs.shape,
        )

    # Check probabilities sum to 1
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        bad_rows = np.where(~np.isclose(row_sums, 1.0, atol=1e-5))[0]
        raise ValidationError(
            f"{context} rows don't sum to 1.0 (first 5 bad rows: {bad_rows[:5]})",
            field="probs",
        )

    # Check probabilities in [0, 1]
    if (probs < 0).any() or (probs > 1).any():
        raise ValidationError(
            f"{context} contains values outside [0, 1]",
            field="probs",
            expected="[0, 1]",
            actual=f"[{probs.min():.4f}, {probs.max():.4f}]",
        )


# =============================================================================
# FEATURE VALIDATION
# =============================================================================


def validate_features(
    feature_columns: list[str],
    min_features: int,
    max_features: int,
    context: str = "features",
) -> None:
    """
    Validate feature column count.

    Args:
        feature_columns: List of feature column names
        min_features: Minimum required features
        max_features: Maximum allowed features
        context: Context string for error messages

    Raises:
        ValidationError: If validation fails
    """
    n = len(feature_columns)

    if n < min_features:
        raise ValidationError(
            f"{context}: too few features",
            field="n_features",
            expected=f">= {min_features}",
            actual=n,
        )

    if n > max_features:
        raise ValidationError(
            f"{context}: too many features",
            field="n_features",
            expected=f"<= {max_features}",
            actual=n,
        )

    # Check for duplicates
    if len(set(feature_columns)) != n:
        duplicates = [f for f in feature_columns if feature_columns.count(f) > 1]
        raise ValidationError(
            f"{context}: contains duplicate feature names: {set(duplicates)}",
            field="feature_columns",
        )


# =============================================================================
# DATAFRAME VALIDATION
# =============================================================================


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
    context: str = "DataFrame",
    check_ohlcv: bool = False,
) -> None:
    """
    Validate DataFrame structure and content.

    Args:
        df: Input DataFrame
        required_columns: List of required column names
        context: Context string for error messages
        check_ohlcv: Whether to validate OHLCV columns specifically

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(
            f"{context} must be a pandas DataFrame",
            field="df",
            expected="pd.DataFrame",
            actual=type(df).__name__,
        )

    if df.empty:
        raise ValidationError(f"{context} is empty", field="df")

    # Check required columns
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValidationError(
                f"{context} missing required columns: {sorted(missing)[:10]}",
                field="columns",
                expected=sorted(required_columns)[:10],
                actual=sorted(df.columns)[:10],
            )

    # OHLCV-specific validation
    if check_ohlcv:
        validate_ohlcv(df, context)


def validate_ohlcv(df: pd.DataFrame, context: str = "OHLCV") -> None:
    """
    Validate OHLCV DataFrame structure.

    Checks:
    - Required columns present (datetime, open, high, low, close, volume)
    - DatetimeIndex or datetime column
    - No negative prices/volume
    - High >= Low, High >= Open/Close, Low <= Open/Close

    Args:
        df: OHLCV DataFrame
        context: Context string for error messages

    Raises:
        ValidationError: If validation fails
    """
    # Check columns
    missing = set(OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise ValidationError(
            f"{context} missing OHLCV columns: {sorted(missing)}",
            field="columns",
            expected=OHLCV_COLUMNS,
        )

    # Check index is datetime
    if not isinstance(df.index, pd.DatetimeIndex) and "datetime" not in df.columns:
        raise ValidationError(
            f"{context} must have DatetimeIndex or 'datetime' column",
            field="index",
        )

    # Check no negative values
    for col in OHLCV_COLUMNS:
        if (df[col] < 0).any():
            raise ValidationError(
                f"{context} contains negative values in '{col}'",
                field=col,
            )

    # Check OHLC relationships
    if (df["high"] < df["low"]).any():
        bad_count = (df["high"] < df["low"]).sum()
        raise ValidationError(
            f"{context} has {bad_count} rows where high < low",
            field="high/low",
        )

    if (df["high"] < df["open"]).any() or (df["high"] < df["close"]).any():
        raise ValidationError(
            f"{context} has rows where high < open or high < close",
            field="high",
        )

    if (df["low"] > df["open"]).any() or (df["low"] > df["close"]).any():
        raise ValidationError(
            f"{context} has rows where low > open or low > close",
            field="low",
        )


# =============================================================================
# MODEL VALIDATION
# =============================================================================


def validate_model_name(model_name: str, context: str = "model") -> None:
    """
    Validate that model name is recognized.

    Args:
        model_name: Model name to validate
        context: Context string for error messages

    Raises:
        ValidationError: If model name is not recognized
    """
    model_lower = model_name.lower()
    if model_lower not in ALL_MODELS:
        raise ValidationError(
            f"{context} '{model_name}' is not a recognized model",
            field="model_name",
            expected=f"one of: {ALL_MODELS}",
            actual=model_name,
        )


def validate_model_list(models: list[str], context: str = "models") -> None:
    """
    Validate a list of model names.

    Args:
        models: List of model names
        context: Context string for error messages

    Raises:
        ValidationError: If any model name is invalid
    """
    if not models:
        raise ValidationError(f"{context} list is empty", field="models")

    for model in models:
        validate_model_name(model, context)


# =============================================================================
# PATH VALIDATION
# =============================================================================


def validate_path_exists(
    path: str | Path,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    context: str = "path",
) -> Path:
    """
    Validate that a path exists.

    Args:
        path: Path to validate
        must_be_file: Path must be a file
        must_be_dir: Path must be a directory
        context: Context string for error messages

    Returns:
        Validated Path object

    Raises:
        ValidationError: If path doesn't exist or isn't correct type
    """
    path = Path(path)

    if not path.exists():
        raise ValidationError(
            f"{context} does not exist: {path}",
            field="path",
        )

    if must_be_file and not path.is_file():
        raise ValidationError(
            f"{context} is not a file: {path}",
            field="path",
        )

    if must_be_dir and not path.is_dir():
        raise ValidationError(
            f"{context} is not a directory: {path}",
            field="path",
        )

    return path


# =============================================================================
# TIMEFRAME VALIDATION
# =============================================================================


def validate_timeframe(timeframe: str, context: str = "timeframe") -> None:
    """
    Validate that timeframe is canonical.

    Args:
        timeframe: Timeframe string to validate
        context: Context string for error messages

    Raises:
        ValidationError: If timeframe is not recognized
    """
    if timeframe not in CANONICAL_TIMEFRAMES:
        raise ValidationError(
            f"{context} '{timeframe}' is not a canonical timeframe",
            field="timeframe",
            expected=f"one of: {CANONICAL_TIMEFRAMES}",
            actual=timeframe,
        )


def validate_timeframe_list(timeframes: list[str], context: str = "timeframes") -> None:
    """
    Validate a list of timeframes.

    Args:
        timeframes: List of timeframe strings
        context: Context string for error messages

    Raises:
        ValidationError: If any timeframe is invalid
    """
    if not timeframes:
        raise ValidationError(f"{context} list is empty", field="timeframes")

    for tf in timeframes:
        validate_timeframe(tf, context)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Exception
    "ValidationError",
    # Array validation
    "validate_input_shape",
    "validate_labels",
    "validate_probabilities",
    # Feature validation
    "validate_features",
    # DataFrame validation
    "validate_dataframe",
    "validate_ohlcv",
    # Model validation
    "validate_model_name",
    "validate_model_list",
    # Path validation
    "validate_path_exists",
    # Timeframe validation
    "validate_timeframe",
    "validate_timeframe_list",
]
