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

    # Hour sin/cos encoding (24-hour cycle)
    df['hour'] = df['datetime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Minute sin/cos encoding (60-minute cycle)
    df['minute'] = df['datetime'].dt.minute
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

    # Day of week sin/cos encoding (7-day cycle)
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # Trading sessions - 3 equal 8-hour blocks covering 24 hours (UTC)
    # Asia:   00:00-08:00 UTC (hours 0-7)
    # London: 08:00-16:00 UTC (hours 8-15)
    # NY:     16:00-24:00 UTC (hours 16-23)
    def get_session(hour):
        if 0 <= hour < 8:
            return 'asia'
        elif 8 <= hour < 16:
            return 'london'
        else:  # 16 <= hour < 24
            return 'ny'

    df['session'] = df['hour'].apply(get_session)

    # One-hot encode session (3 sessions, 8 hours each)
    for session in ['asia', 'london', 'ny']:
        df[f'session_{session}'] = (df['session'] == session).astype(int)

    df = df.drop(['hour', 'minute', 'dayofweek', 'session'], axis=1)

    feature_metadata['hour_sin'] = "Hour sine encoding"
    feature_metadata['hour_cos'] = "Hour cosine encoding"
    feature_metadata['minute_sin'] = "Minute sine encoding"
    feature_metadata['minute_cos'] = "Minute cosine encoding"
    feature_metadata['dayofweek_sin'] = "Day of week sine encoding"
    feature_metadata['dayofweek_cos'] = "Day of week cosine encoding"
    feature_metadata['session_asia'] = "Asia session (00:00-08:00 UTC)"
    feature_metadata['session_london'] = "London session (08:00-16:00 UTC)"
    feature_metadata['session_ny'] = "New York session (16:00-24:00 UTC)"

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

    hour = df['datetime'].dt.hour

    # Trading sessions
    df['session_asia'] = ((hour >= 0) & (hour < 8)).astype(int)
    df['session_london'] = ((hour >= 8) & (hour < 16)).astype(int)
    df['session_ny'] = (hour >= 16).astype(int)

    feature_metadata['session_asia'] = "Asia session (00:00-08:00 UTC)"
    feature_metadata['session_london'] = "London session (08:00-16:00 UTC)"
    feature_metadata['session_ny'] = "New York session (16:00-24:00 UTC)"

    return df


__all__ = [
    'add_temporal_features',
    'add_session_features',
]
