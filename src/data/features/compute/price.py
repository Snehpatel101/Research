"""
Price feature computation - Returns, ratios, and autocorrelation.

PHASE_1 Unified Features: 12 PRICE features.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _safe_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """Safe percentage change calculation."""
    return series.pct_change(periods=periods)


def _safe_log_return(series: pd.Series, periods: int = 1) -> pd.Series:
    """Safe log return calculation."""
    return np.log(series / series.shift(periods))


def _rolling_autocorr(series: pd.Series, lag: int, window: int = 20) -> pd.Series:
    """Rolling autocorrelation with specified lag."""

    def autocorr_lag(x: np.ndarray) -> float:
        if len(x) <= lag:
            return np.nan
        x_clean = x[~np.isnan(x)]
        if len(x_clean) <= lag:
            return np.nan
        return np.corrcoef(x_clean[:-lag], x_clean[lag:])[0, 1]

    return series.rolling(window=window, min_periods=window).apply(
        lambda x: autocorr_lag(x.values), raw=False
    )


# =============================================================================
# RETURN FEATURES
# =============================================================================


def compute_return_1(df: pd.DataFrame) -> pd.Series:
    """1-period simple return."""
    return _safe_pct_change(df["close"], periods=1)


def compute_return_5(df: pd.DataFrame) -> pd.Series:
    """5-period simple return."""
    return _safe_pct_change(df["close"], periods=5)


def compute_return_10(df: pd.DataFrame) -> pd.Series:
    """10-period simple return."""
    return _safe_pct_change(df["close"], periods=10)


def compute_return_20(df: pd.DataFrame) -> pd.Series:
    """20-period simple return."""
    return _safe_pct_change(df["close"], periods=20)


def compute_log_return_1(df: pd.DataFrame) -> pd.Series:
    """1-period log return."""
    return _safe_log_return(df["close"], periods=1)


def compute_log_return_5(df: pd.DataFrame) -> pd.Series:
    """5-period log return."""
    return _safe_log_return(df["close"], periods=5)


# =============================================================================
# RATIO FEATURES
# =============================================================================


def compute_hl_ratio(df: pd.DataFrame) -> pd.Series:
    """High/Low ratio - measures intrabar range."""
    return df["high"] / df["low"]


def compute_co_ratio(df: pd.DataFrame) -> pd.Series:
    """Close/Open ratio - measures intrabar direction."""
    return df["close"] / df["open"]


def compute_range_pct(df: pd.DataFrame) -> pd.Series:
    """Range as percentage of close price."""
    return (df["high"] - df["low"]) / df["close"]


def compute_clv(df: pd.DataFrame) -> pd.Series:
    """
    Close Location Value - where close is within high-low range.

    CLV = (close - low) / (high - low) * 2 - 1
    Returns: -1 (close at low) to +1 (close at high)
    """
    hl_range = df["high"] - df["low"]
    # Avoid division by zero
    hl_range = hl_range.replace(0, np.nan)
    clv = ((df["close"] - df["low"]) / hl_range) * 2 - 1
    return clv


# =============================================================================
# AUTOCORRELATION FEATURES
# =============================================================================


def compute_autocorr_lag1(df: pd.DataFrame) -> pd.Series:
    """
    Rolling autocorrelation with lag 1.

    Measures persistence in returns. High positive values indicate trending,
    negative values indicate mean reversion.
    """
    returns = _safe_pct_change(df["close"], periods=1)
    return _rolling_autocorr(returns, lag=1, window=20)


def compute_autocorr_lag5(df: pd.DataFrame) -> pd.Series:
    """
    Rolling autocorrelation with lag 5.

    Measures longer-term return persistence.
    """
    returns = _safe_pct_change(df["close"], periods=1)
    return _rolling_autocorr(returns, lag=5, window=30)


# =============================================================================
# FEATURE MAP
# =============================================================================

PRICE_FEATURES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    # Returns
    "return_1": compute_return_1,
    "return_5": compute_return_5,
    "return_10": compute_return_10,
    "return_20": compute_return_20,
    "log_return_1": compute_log_return_1,
    "log_return_5": compute_log_return_5,
    # Ratios
    "hl_ratio": compute_hl_ratio,
    "co_ratio": compute_co_ratio,
    "range_pct": compute_range_pct,
    "clv": compute_clv,
    # Autocorrelation
    "autocorr_lag1": compute_autocorr_lag1,
    "autocorr_lag5": compute_autocorr_lag5,
}

# Feature family metadata
FEATURE_FAMILY = "price"
FEATURE_COUNT = 12

__all__ = [
    "PRICE_FEATURES",
    "FEATURE_FAMILY",
    "FEATURE_COUNT",
    "compute_return_1",
    "compute_return_5",
    "compute_return_10",
    "compute_return_20",
    "compute_log_return_1",
    "compute_log_return_5",
    "compute_hl_ratio",
    "compute_co_ratio",
    "compute_range_pct",
    "compute_clv",
    "compute_autocorr_lag1",
    "compute_autocorr_lag5",
]
