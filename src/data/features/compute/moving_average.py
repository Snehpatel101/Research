"""
Moving Average feature computation - SMA, EMA, price ratios, and crossovers.

PHASE_1 Unified Features: 16 MOVING_AVERAGE features.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=window, min_periods=window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, min_periods=span, adjust=False).mean()


def _crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """
    Detect crossover events.

    Returns:
        1.0 if fast crossed above slow (bullish)
        -1.0 if fast crossed below slow (bearish)
        0.0 otherwise
    """
    cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))

    result = pd.Series(0.0, index=fast.index)
    result[cross_up] = 1.0
    result[cross_down] = -1.0

    return result


# =============================================================================
# SMA FEATURES
# =============================================================================


def compute_sma_10(df: pd.DataFrame) -> pd.Series:
    """10-period Simple Moving Average."""
    return _sma(df["close"], window=10)


def compute_sma_20(df: pd.DataFrame) -> pd.Series:
    """20-period Simple Moving Average."""
    return _sma(df["close"], window=20)


def compute_sma_50(df: pd.DataFrame) -> pd.Series:
    """50-period Simple Moving Average."""
    return _sma(df["close"], window=50)


def compute_sma_100(df: pd.DataFrame) -> pd.Series:
    """100-period Simple Moving Average."""
    return _sma(df["close"], window=100)


def compute_sma_200(df: pd.DataFrame) -> pd.Series:
    """200-period Simple Moving Average."""
    return _sma(df["close"], window=200)


# =============================================================================
# EMA FEATURES
# =============================================================================


def compute_ema_9(df: pd.DataFrame) -> pd.Series:
    """9-period Exponential Moving Average."""
    return _ema(df["close"], span=9)


def compute_ema_12(df: pd.DataFrame) -> pd.Series:
    """12-period Exponential Moving Average (MACD fast)."""
    return _ema(df["close"], span=12)


def compute_ema_21(df: pd.DataFrame) -> pd.Series:
    """21-period Exponential Moving Average."""
    return _ema(df["close"], span=21)


def compute_ema_26(df: pd.DataFrame) -> pd.Series:
    """26-period Exponential Moving Average (MACD slow)."""
    return _ema(df["close"], span=26)


def compute_ema_50(df: pd.DataFrame) -> pd.Series:
    """50-period Exponential Moving Average."""
    return _ema(df["close"], span=50)


# =============================================================================
# PRICE RATIO FEATURES
# =============================================================================


def compute_price_to_sma_20(df: pd.DataFrame) -> pd.Series:
    """
    Price to SMA-20 ratio.

    Measures how far price is from the 20-period moving average.
    > 1.0 means price is above the MA (bullish)
    < 1.0 means price is below the MA (bearish)
    """
    sma_20 = _sma(df["close"], window=20)
    # Avoid division by zero
    sma_20 = sma_20.replace(0, np.nan)
    return df["close"] / sma_20


def compute_price_to_sma_50(df: pd.DataFrame) -> pd.Series:
    """
    Price to SMA-50 ratio.

    Measures price position relative to medium-term trend.
    """
    sma_50 = _sma(df["close"], window=50)
    sma_50 = sma_50.replace(0, np.nan)
    return df["close"] / sma_50


def compute_price_to_ema_21(df: pd.DataFrame) -> pd.Series:
    """
    Price to EMA-21 ratio.

    Measures price position relative to short-term exponential trend.
    """
    ema_21 = _ema(df["close"], span=21)
    ema_21 = ema_21.replace(0, np.nan)
    return df["close"] / ema_21


# =============================================================================
# CROSSOVER FEATURES
# =============================================================================


def compute_sma_cross_10_50(df: pd.DataFrame) -> pd.Series:
    """
    SMA 10/50 crossover signal.

    Returns:
        1.0 for bullish crossover (fast crosses above slow)
        -1.0 for bearish crossover (fast crosses below slow)
        0.0 otherwise
    """
    sma_10 = _sma(df["close"], window=10)
    sma_50 = _sma(df["close"], window=50)
    return _crossover(sma_10, sma_50)


def compute_sma_cross_20_200(df: pd.DataFrame) -> pd.Series:
    """
    SMA 20/200 crossover signal (Golden/Death Cross variant).

    Returns:
        1.0 for golden cross (bullish)
        -1.0 for death cross (bearish)
        0.0 otherwise
    """
    sma_20 = _sma(df["close"], window=20)
    sma_200 = _sma(df["close"], window=200)
    return _crossover(sma_20, sma_200)


def compute_ema_cross_9_21(df: pd.DataFrame) -> pd.Series:
    """
    EMA 9/21 crossover signal.

    Fast EMA crossover for short-term momentum.
    """
    ema_9 = _ema(df["close"], span=9)
    ema_21 = _ema(df["close"], span=21)
    return _crossover(ema_9, ema_21)


# =============================================================================
# FEATURE MAP
# =============================================================================

MOVING_AVERAGE_FEATURES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    # SMA
    "sma_10": compute_sma_10,
    "sma_20": compute_sma_20,
    "sma_50": compute_sma_50,
    "sma_100": compute_sma_100,
    "sma_200": compute_sma_200,
    # EMA
    "ema_9": compute_ema_9,
    "ema_12": compute_ema_12,
    "ema_21": compute_ema_21,
    "ema_26": compute_ema_26,
    "ema_50": compute_ema_50,
    # Price ratios
    "price_to_sma_20": compute_price_to_sma_20,
    "price_to_sma_50": compute_price_to_sma_50,
    "price_to_ema_21": compute_price_to_ema_21,
    # Crossovers
    "sma_cross_10_50": compute_sma_cross_10_50,
    "sma_cross_20_200": compute_sma_cross_20_200,
    "ema_cross_9_21": compute_ema_cross_9_21,
}

# Feature family metadata
FEATURE_FAMILY = "moving_average"
FEATURE_COUNT = 16

__all__ = [
    "MOVING_AVERAGE_FEATURES",
    "FEATURE_FAMILY",
    "FEATURE_COUNT",
    # SMA
    "compute_sma_10",
    "compute_sma_20",
    "compute_sma_50",
    "compute_sma_100",
    "compute_sma_200",
    # EMA
    "compute_ema_9",
    "compute_ema_12",
    "compute_ema_21",
    "compute_ema_26",
    "compute_ema_50",
    # Price ratios
    "compute_price_to_sma_20",
    "compute_price_to_sma_50",
    "compute_price_to_ema_21",
    # Crossovers
    "compute_sma_cross_10_50",
    "compute_sma_cross_20_200",
    "compute_ema_cross_9_21",
]
