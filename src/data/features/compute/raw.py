"""
Raw feature computation - OHLCV passthrough.

PHASE_1 Unified Features: 5 RAW features.
These simply pass through the OHLCV columns from the input DataFrame.
"""

from typing import Callable, Dict

import pandas as pd

from src.core import OHLCV_COLUMNS


# =============================================================================
# FEATURE COMPUTATION FUNCTIONS
# =============================================================================


def compute_open(df: pd.DataFrame) -> pd.Series:
    """Pass through open price."""
    return df["open"].copy()


def compute_high(df: pd.DataFrame) -> pd.Series:
    """Pass through high price."""
    return df["high"].copy()


def compute_low(df: pd.DataFrame) -> pd.Series:
    """Pass through low price."""
    return df["low"].copy()


def compute_close(df: pd.DataFrame) -> pd.Series:
    """Pass through close price."""
    return df["close"].copy()


def compute_volume(df: pd.DataFrame) -> pd.Series:
    """Pass through volume."""
    return df["volume"].copy()


# =============================================================================
# FEATURE MAP - Maps feature names to compute functions
# =============================================================================

RAW_FEATURES: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "open": compute_open,
    "high": compute_high,
    "low": compute_low,
    "close": compute_close,
    "volume": compute_volume,
}

# Feature family metadata
FEATURE_FAMILY = "raw"
FEATURE_COUNT = 5

__all__ = [
    "RAW_FEATURES",
    "FEATURE_FAMILY",
    "FEATURE_COUNT",
    "compute_open",
    "compute_high",
    "compute_low",
    "compute_close",
    "compute_volume",
]
