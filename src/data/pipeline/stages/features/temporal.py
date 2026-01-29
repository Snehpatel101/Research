"""
Temporal features for feature engineering.

This module provides functions to calculate time-based features
including cyclical time encoding and trading session indicators.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_temporal_features(df: pd.DataFrame, feature_metadata: dict[str, str]) -> pd.DataFrame:
    """
    Add temporal features with sin/cos encoding.

    Calculates cyclical encoding for hour, minute, and day of week,
    plus trading session indicators.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'datetime' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with temporal features added
    """
    logger.info("Adding temporal features...")

    # Extract datetime components once
    hour = df["datetime"].dt.hour
    minute = df["datetime"].dt.minute
    dayofweek = df["datetime"].dt.dayofweek

    # Batch compute all features to avoid DataFrame fragmentation
    new_cols = {
        # Hour sin/cos encoding (24-hour cycle)
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        # Minute sin/cos encoding (60-minute cycle)
        "minute_sin": np.sin(2 * np.pi * minute / 60),
        "minute_cos": np.cos(2 * np.pi * minute / 60),
        # Day of week sin/cos encoding (7-day cycle)
        "dayofweek_sin": np.sin(2 * np.pi * dayofweek / 7),
        "dayofweek_cos": np.cos(2 * np.pi * dayofweek / 7),
        # Trading sessions - vectorized (no slow .apply())
        # Asia:   00:00-08:00 UTC (hours 0-7)
        # London: 08:00-16:00 UTC (hours 8-15)
        # NY:     16:00-24:00 UTC (hours 16-23)
        "session_asia": ((hour >= 0) & (hour < 8)).astype(int),
        "session_london": ((hour >= 8) & (hour < 16)).astype(int),
        "session_ny": (hour >= 16).astype(int),
    }

    # Single concat to avoid fragmentation warning
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    feature_metadata["hour_sin"] = "Hour sine encoding"
    feature_metadata["hour_cos"] = "Hour cosine encoding"
    feature_metadata["minute_sin"] = "Minute sine encoding"
    feature_metadata["minute_cos"] = "Minute cosine encoding"
    feature_metadata["dayofweek_sin"] = "Day of week sine encoding"
    feature_metadata["dayofweek_cos"] = "Day of week cosine encoding"
    feature_metadata["session_asia"] = "Asia session (00:00-08:00 UTC)"
    feature_metadata["session_london"] = "London session (08:00-16:00 UTC)"
    feature_metadata["session_ny"] = "New York session (16:00-24:00 UTC)"

    return df


def add_session_features(df: pd.DataFrame, feature_metadata: dict[str, str]) -> pd.DataFrame:
    """
    Add trading session features only.

    Simpler version that only adds session indicators without cyclical encoding.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'datetime' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with session features added
    """
    logger.info("Adding session features...")

    hour = df["datetime"].dt.hour

    # Batch concat to avoid fragmentation
    new_cols = {
        "session_asia": ((hour >= 0) & (hour < 8)).astype(int),
        "session_london": ((hour >= 8) & (hour < 16)).astype(int),
        "session_ny": (hour >= 16).astype(int),
    }
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    feature_metadata["session_asia"] = "Asia session (00:00-08:00 UTC)"
    feature_metadata["session_london"] = "London session (08:00-16:00 UTC)"
    feature_metadata["session_ny"] = "New York session (16:00-24:00 UTC)"

    return df


__all__ = [
    "add_temporal_features",
    "add_session_features",
]
