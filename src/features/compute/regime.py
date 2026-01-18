"""
Regime feature computation - Volatility, trend, and composite regime detection.

PHASE_1 Unified Features: 9 REGIME features.

These features identify market regimes for regime-aware trading strategies.
"""

from typing import Callable, Dict

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


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation."""
    return series.rolling(window=window, min_periods=window).std()


def _log_returns(close: pd.Series) -> pd.Series:
    """Calculate log returns."""
    return np.log(close / close.shift(1))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


# =============================================================================
# VOLATILITY REGIME FEATURES
# =============================================================================


def compute_volatility_regime(df: pd.DataFrame) -> pd.Series:
    """
    Volatility regime indicator.

    Compares short-term volatility to long-term volatility.
    Returns:
        0: Low volatility (short-term vol < long-term vol)
        1: High volatility (short-term vol >= long-term vol)
    """
    log_ret = _log_returns(df["close"])

    # Short-term volatility (20-period)
    short_vol = _rolling_std(log_ret, window=20)

    # Long-term volatility (60-period)
    long_vol = _rolling_std(log_ret, window=60)

    # Regime classification
    regime = (short_vol >= long_vol).astype(float)

    return regime


def compute_regime_low_vol(df: pd.DataFrame) -> pd.Series:
    """Low volatility regime flag."""
    vol_regime = compute_volatility_regime(df)
    return (vol_regime == 0).astype(float)


def compute_regime_high_vol(df: pd.DataFrame) -> pd.Series:
    """High volatility regime flag."""
    vol_regime = compute_volatility_regime(df)
    return (vol_regime == 1).astype(float)


# =============================================================================
# TREND REGIME FEATURES
# =============================================================================


def compute_trend_regime(df: pd.DataFrame) -> pd.Series:
    """
    Trend regime indicator using multiple moving average alignment.

    Returns:
        1: Uptrend (price > SMA20 > SMA50)
        -1: Downtrend (price < SMA20 < SMA50)
        0: Sideways/mixed
    """
    sma_20 = _sma(df["close"], window=20)
    sma_50 = _sma(df["close"], window=50)

    close = df["close"]

    # Uptrend: price > SMA20 > SMA50
    uptrend = (close > sma_20) & (sma_20 > sma_50)

    # Downtrend: price < SMA20 < SMA50
    downtrend = (close < sma_20) & (sma_20 < sma_50)

    regime = pd.Series(0.0, index=df.index)
    regime[uptrend] = 1.0
    regime[downtrend] = -1.0

    return regime


def compute_regime_uptrend(df: pd.DataFrame) -> pd.Series:
    """Uptrend regime flag."""
    trend_regime = compute_trend_regime(df)
    return (trend_regime == 1).astype(float)


def compute_regime_downtrend(df: pd.DataFrame) -> pd.Series:
    """Downtrend regime flag."""
    trend_regime = compute_trend_regime(df)
    return (trend_regime == -1).astype(float)


def compute_regime_sideways(df: pd.DataFrame) -> pd.Series:
    """Sideways/ranging regime flag."""
    trend_regime = compute_trend_regime(df)
    return (trend_regime == 0).astype(float)


# =============================================================================
# STRUCTURE REGIME FEATURES
# =============================================================================


def compute_structure_regime(df: pd.DataFrame) -> pd.Series:
    """
    Market structure regime based on ATR and price patterns.

    Combines volatility expansion/contraction with trend.
    Returns:
        0: Contracting (low vol + ranging)
        1: Expanding (high vol + trending)
        2: Transitioning
    """
    # ATR-based volatility
    atr_14 = _atr(df, period=14)
    atr_50 = _atr(df, period=50)

    # Volatility expansion
    vol_expanding = atr_14 > atr_50

    # Trend strength (using ADX concept simplified)
    # Price momentum as proxy
    momentum = df["close"].pct_change(20)
    strong_trend = momentum.abs() > momentum.abs().rolling(50).median()

    regime = pd.Series(2.0, index=df.index)  # Default: transitioning

    # Contracting: low vol + weak trend
    contracting = (~vol_expanding) & (~strong_trend)
    regime[contracting] = 0.0

    # Expanding: high vol + strong trend
    expanding = vol_expanding & strong_trend
    regime[expanding] = 1.0

    return regime


# =============================================================================
# COMPOSITE REGIME FEATURES
# =============================================================================


def compute_composite_regime(df: pd.DataFrame) -> pd.Series:
    """
    Composite regime combining volatility and trend.

    Creates 9 unique regime states (3 vol states x 3 trend states):
        0: Low vol + Downtrend
        1: Low vol + Sideways
        2: Low vol + Uptrend
        3: High vol + Downtrend
        4: High vol + Sideways
        5: High vol + Uptrend
        6-8: Reserved for transition states

    For simplicity, uses 2 vol states x 3 trend states = 6 states.
    """
    vol_regime = compute_volatility_regime(df)
    trend_regime = compute_trend_regime(df)

    # Encode: vol_regime (0/1) * 3 + (trend_regime + 1)
    # trend_regime: -1, 0, 1 -> 0, 1, 2
    trend_encoded = (trend_regime + 1).astype(int)

    composite = (vol_regime * 3 + trend_encoded).astype(float)

    return composite


# =============================================================================
# FEATURE MAP
# =============================================================================

REGIME_FEATURES: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    # Volatility regimes
    "volatility_regime": compute_volatility_regime,
    "regime_low_vol": compute_regime_low_vol,
    "regime_high_vol": compute_regime_high_vol,
    # Trend regimes
    "trend_regime": compute_trend_regime,
    "regime_uptrend": compute_regime_uptrend,
    "regime_downtrend": compute_regime_downtrend,
    "regime_sideways": compute_regime_sideways,
    # Structure
    "structure_regime": compute_structure_regime,
    # Composite
    "composite_regime": compute_composite_regime,
}

# Feature family metadata
FEATURE_FAMILY = "regime"
FEATURE_COUNT = 9

__all__ = [
    "REGIME_FEATURES",
    "FEATURE_FAMILY",
    "FEATURE_COUNT",
    # Volatility regimes
    "compute_volatility_regime",
    "compute_regime_low_vol",
    "compute_regime_high_vol",
    # Trend regimes
    "compute_trend_regime",
    "compute_regime_uptrend",
    "compute_regime_downtrend",
    "compute_regime_sideways",
    # Structure
    "compute_structure_regime",
    # Composite
    "compute_composite_regime",
]
