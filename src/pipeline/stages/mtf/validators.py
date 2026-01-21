"""
Validation functions for Multi-Timeframe (MTF) Feature Integration.
"""

import pandas as pd

from .constants import MTF_TIMEFRAMES, REQUIRED_OHLCV_COLS


def validate_ohlcv_dataframe(df: pd.DataFrame) -> None:
    """
    Validate that DataFrame has required OHLCV structure.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate

    Raises
    ------
    ValueError
        If required columns are missing or datetime is not present
    """
    if "datetime" not in df.columns:
        raise ValueError("DataFrame must have 'datetime' column")

    missing_cols = [col for col in REQUIRED_OHLCV_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required OHLCV columns: {missing_cols}")


def validate_timeframe_format(tf: str) -> None:
    """
    Validate timeframe string format.

    Parameters
    ----------
    tf : str
        Timeframe string (e.g., '5min', '15min', '1h')

    Raises
    ------
    ValueError
        If timeframe format is not recognized
    """
    if tf not in MTF_TIMEFRAMES:
        valid_formats = list(MTF_TIMEFRAMES.keys())
        raise ValueError(f"Unknown timeframe format: {tf}. " f"Supported formats: {valid_formats}")


def validate_mtf_timeframes(base_tf: str, mtf_timeframes: list) -> None:
    """
    Validate that MTF timeframes are valid relative to base timeframe.

    Parameters
    ----------
    base_tf : str
        Base timeframe string
    mtf_timeframes : list
        List of MTF timeframe strings

    Raises
    ------
    ValueError
        If any MTF timeframe is invalid
    """
    validate_timeframe_format(base_tf)
    base_minutes = MTF_TIMEFRAMES[base_tf]

    for tf in mtf_timeframes:
        validate_timeframe_format(tf)
        tf_minutes = MTF_TIMEFRAMES[tf]

        if tf_minutes <= base_minutes:
            raise ValueError(
                f"MTF timeframe {tf} ({tf_minutes}min) must be > "
                f"base {base_tf} ({base_minutes}min)"
            )

        if tf_minutes % base_minutes != 0:
            raise ValueError(
                f"MTF timeframe {tf} ({tf_minutes}min) must be an integer "
                f"multiple of base {base_tf} ({base_minutes}min)"
            )
