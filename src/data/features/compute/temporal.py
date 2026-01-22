"""
Temporal feature computation - Time-based cyclical encodings and trading sessions.

PHASE_1 Unified Features: 9 TEMPORAL features.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_datetime_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Extract datetime index from DataFrame."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index
    elif "datetime" in df.columns:
        return pd.to_datetime(df["datetime"])
    else:
        raise ValueError("DataFrame must have DatetimeIndex or 'datetime' column")


def _cyclical_encode(values: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    """
    Cyclical encoding using sine/cosine transformation.

    This preserves the cyclical nature of time features (e.g., hour 23 is close to hour 0).

    Args:
        values: Values to encode
        period: The period of the cycle (24 for hours, 60 for minutes, 7 for days)

    Returns:
        Tuple of (sin_encoded, cos_encoded)
    """
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


# =============================================================================
# HOUR FEATURES
# =============================================================================


def compute_hour_sin(df: pd.DataFrame) -> pd.Series:
    """Sine encoding of hour (0-23)."""
    dt_index = _get_datetime_index(df)
    hours = pd.Series(dt_index.hour, index=df.index)
    sin_val, _ = _cyclical_encode(hours, 24.0)
    return pd.Series(sin_val, index=df.index, name="hour_sin")


def compute_hour_cos(df: pd.DataFrame) -> pd.Series:
    """Cosine encoding of hour (0-23)."""
    dt_index = _get_datetime_index(df)
    hours = pd.Series(dt_index.hour, index=df.index)
    _, cos_val = _cyclical_encode(hours, 24.0)
    return pd.Series(cos_val, index=df.index, name="hour_cos")


# =============================================================================
# MINUTE FEATURES
# =============================================================================


def compute_minute_sin(df: pd.DataFrame) -> pd.Series:
    """Sine encoding of minute (0-59)."""
    dt_index = _get_datetime_index(df)
    minutes = pd.Series(dt_index.minute, index=df.index)
    sin_val, _ = _cyclical_encode(minutes, 60.0)
    return pd.Series(sin_val, index=df.index, name="minute_sin")


def compute_minute_cos(df: pd.DataFrame) -> pd.Series:
    """Cosine encoding of minute (0-59)."""
    dt_index = _get_datetime_index(df)
    minutes = pd.Series(dt_index.minute, index=df.index)
    _, cos_val = _cyclical_encode(minutes, 60.0)
    return pd.Series(cos_val, index=df.index, name="minute_cos")


# =============================================================================
# DAY OF WEEK FEATURES
# =============================================================================


def compute_dayofweek_sin(df: pd.DataFrame) -> pd.Series:
    """Sine encoding of day of week (0=Monday, 6=Sunday)."""
    dt_index = _get_datetime_index(df)
    dayofweek = pd.Series(dt_index.dayofweek, index=df.index)
    sin_val, _ = _cyclical_encode(dayofweek, 7.0)
    return pd.Series(sin_val, index=df.index, name="dayofweek_sin")


def compute_dayofweek_cos(df: pd.DataFrame) -> pd.Series:
    """Cosine encoding of day of week (0=Monday, 6=Sunday)."""
    dt_index = _get_datetime_index(df)
    dayofweek = pd.Series(dt_index.dayofweek, index=df.index)
    _, cos_val = _cyclical_encode(dayofweek, 7.0)
    return pd.Series(cos_val, index=df.index, name="dayofweek_cos")


# =============================================================================
# TRADING SESSION FEATURES
# =============================================================================


def compute_session_asia(df: pd.DataFrame) -> pd.Series:
    """
    Asia/Tokyo trading session flag.

    Approximate hours (UTC): 00:00 - 09:00
    Covers: Tokyo, Hong Kong, Singapore markets
    """
    dt_index = _get_datetime_index(df)
    hours = pd.Series(dt_index.hour, index=df.index)
    # Asia session: 00:00 - 09:00 UTC
    is_asia = (hours >= 0) & (hours < 9)
    return is_asia.astype(float)


def compute_session_london(df: pd.DataFrame) -> pd.Series:
    """
    London/European trading session flag.

    Approximate hours (UTC): 08:00 - 17:00
    Covers: London, Frankfurt, Paris markets
    """
    dt_index = _get_datetime_index(df)
    hours = pd.Series(dt_index.hour, index=df.index)
    # London session: 08:00 - 17:00 UTC
    is_london = (hours >= 8) & (hours < 17)
    return is_london.astype(float)


def compute_session_ny(df: pd.DataFrame) -> pd.Series:
    """
    New York/US trading session flag.

    Approximate hours (UTC): 13:00 - 22:00
    Covers: NYSE, NASDAQ, CME markets
    """
    dt_index = _get_datetime_index(df)
    hours = pd.Series(dt_index.hour, index=df.index)
    # NY session: 13:00 - 22:00 UTC
    is_ny = (hours >= 13) & (hours < 22)
    return is_ny.astype(float)


# =============================================================================
# FEATURE MAP
# =============================================================================

TEMPORAL_FEATURES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    # Hour encodings
    "hour_sin": compute_hour_sin,
    "hour_cos": compute_hour_cos,
    # Minute encodings
    "minute_sin": compute_minute_sin,
    "minute_cos": compute_minute_cos,
    # Day of week encodings
    "dayofweek_sin": compute_dayofweek_sin,
    "dayofweek_cos": compute_dayofweek_cos,
    # Trading sessions
    "session_asia": compute_session_asia,
    "session_london": compute_session_london,
    "session_ny": compute_session_ny,
}

# Feature family metadata
FEATURE_FAMILY = "temporal"
FEATURE_COUNT = 9

__all__ = [
    "TEMPORAL_FEATURES",
    "FEATURE_FAMILY",
    "FEATURE_COUNT",
    "compute_hour_sin",
    "compute_hour_cos",
    "compute_minute_sin",
    "compute_minute_cos",
    "compute_dayofweek_sin",
    "compute_dayofweek_cos",
    "compute_session_asia",
    "compute_session_london",
    "compute_session_ny",
]
