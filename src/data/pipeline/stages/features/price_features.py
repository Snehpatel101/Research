"""
Price-based features for feature engineering.

This module provides functions to calculate price-based features
including returns and price ratios.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safely divide, returning NaN when denominator is zero."""
    return numerator / denominator.replace(0, np.nan)


def add_returns(df: pd.DataFrame, feature_metadata: dict[str, str]) -> pd.DataFrame:
    """
    Add return features.

    Calculates both simple and log returns over multiple periods.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with return features added
    """
    logger.info("Adding return features...")

    periods = [1, 5, 10, 20, 60]

    for period in periods:
        # Simple returns
        # ANTI-LOOKAHEAD: shift(1) ensures return at bar[t] uses close[t-1] vs close[t-1-period]
        # Without shift, return at bar[t] would use close[t], which is not yet available
        df[f"return_{period}"] = df["close"].pct_change(period).shift(1)

        # Log returns
        # ANTI-LOOKAHEAD: shift(1) ensures log return at bar[t] uses close[t-1] vs close[t-1-period]
        df[f"log_return_{period}"] = np.log(df["close"] / df["close"].shift(period)).shift(1)

        feature_metadata[f"return_{period}"] = f"{period}-period simple return (lagged)"
        feature_metadata[f"log_return_{period}"] = f"{period}-period log return (lagged)"

    return df


def add_price_ratios(df: pd.DataFrame, feature_metadata: dict[str, str]) -> pd.DataFrame:
    """
    Add price ratio features.

    Calculates high/low ratio, close/open ratio, and range as percentage of close.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with price ratio features added
    """
    logger.info("Adding price ratio features...")

    # ANTI-LOOKAHEAD: All price ratios use previous bar's OHLC data
    # Features at bar[t] must only use data available at bar[t-1]

    # High/Low ratio (safe: low could be 0)
    df["hl_ratio"] = _safe_divide(df["high"].shift(1), df["low"].shift(1))

    # Close/Open ratio (safe: open could be 0)
    df["co_ratio"] = _safe_divide(df["close"].shift(1), df["open"].shift(1))

    # Range as percentage of close (safe: close could be 0)
    df["range_pct"] = _safe_divide(df["high"].shift(1) - df["low"].shift(1), df["close"].shift(1))

    feature_metadata["hl_ratio"] = "High to low ratio (previous bar)"
    feature_metadata["co_ratio"] = "Close to open ratio (previous bar)"
    feature_metadata["range_pct"] = "Range as percentage of close (previous bar)"

    return df


def add_autocorrelation(
    df: pd.DataFrame, feature_metadata: dict[str, str], lags: list = None, period: int = 20
) -> pd.DataFrame:
    """
    Add rolling autocorrelation of returns at specified lags.

    Autocorrelation measures serial correlation in returns:
    - Positive autocorr at lag 1 suggests momentum/trend persistence
    - Negative autocorr suggests mean-reversion
    - Higher lags capture longer-term serial dependencies

    Common interpretations:
    - Lag 1-2: Short-term momentum or mean-reversion
    - Lag 5: Intraday patterns (for 5min bars, this is ~25 min)
    - Lag 10: Hour-scale patterns
    - Lag 20: Multi-hour patterns

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    lags : list, optional
        Lags for autocorrelation. Default: [1, 2, 5, 10, 20]
    period : int, default 20
        Rolling window for autocorrelation calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with autocorrelation features added
    """
    if lags is None:
        lags = [1, 2, 5, 10, 20]

    logger.info(f"Adding autocorrelation features with lags: {lags}")

    returns = df["close"].pct_change()

    for lag in lags:
        col = f"return_autocorr_lag{lag}"
        # ANTI-LOOKAHEAD: shift(1) ensures autocorr at bar[t] uses data up to bar[t-1]
        autocorr = (
            returns.rolling(period)
            .apply(lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan, raw=False)
            .shift(1)
        )
        df[col] = autocorr
        feature_metadata[col] = f"Return autocorrelation lag {lag} ({period}-period, lagged)"

    return df


def add_clv(df: pd.DataFrame, feature_metadata: dict[str, str]) -> pd.DataFrame:
    """
    Add Close Location Value (CLV).

    CLV shows where close is relative to high-low range:
    - CLV = +1: close at high
    - CLV = -1: close at low
    - CLV = 0: close at midpoint

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with CLV feature added
    """
    logger.info("Adding Close Location Value...")

    # CLV = ((close - low) - (high - close)) / (high - low)
    # Simplified: CLV = (2 * close - high - low) / (high - low)
    hl_range = df["high"] - df["low"]
    hl_range_safe = hl_range.replace(0, np.nan)

    clv_raw = (2 * df["close"] - df["high"] - df["low"]) / hl_range_safe

    # ANTI-LOOKAHEAD: shift(1) ensures CLV at bar[t] uses data up to bar[t-1]
    df["clv"] = clv_raw.shift(1)

    feature_metadata["clv"] = "Close Location Value [-1 to 1] (lagged)"

    return df


__all__ = [
    "add_returns",
    "add_price_ratios",
    "add_autocorrelation",
    "add_clv",
]
