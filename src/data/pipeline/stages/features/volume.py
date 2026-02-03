"""
Volume-based features for feature engineering.

This module provides functions to calculate volume-based technical
indicators including OBV, VWAP, volume ratios, and volume z-scores.
"""

import logging

import numpy as np
import pandas as pd

from src.core.utils.math_utils import safe_divide

logger = logging.getLogger(__name__)


def add_volume_features(
    df: pd.DataFrame, feature_metadata: dict[str, str], period: int = 20
) -> pd.DataFrame:
    """
    Add volume-based features.

    Calculates OBV, volume SMA, volume ratio, and volume z-score.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' and 'volume' columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    period : int, default 20
        Period for volume SMA and z-score

    Returns
    -------
    pd.DataFrame
        DataFrame with volume features added (if volume data exists)
    """
    if "volume" not in df.columns or df["volume"].sum() == 0:
        logger.info("Skipping volume features (no volume data)")
        return df

    logger.info(f"Adding volume features with period: {period}")

    # ANTI-LOOKAHEAD: All volume features shifted by 1 bar
    # OBV - computed then shifted
    obv_raw = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    df["obv"] = obv_raw.shift(1)

    # OBV SMA - computed on raw, then shifted
    obv_sma_col = f"obv_sma_{period}"
    df[obv_sma_col] = obv_raw.rolling(window=period).mean().shift(1)

    # Volume SMA and ratios - all shifted
    volume_sma_raw = df["volume"].rolling(window=period).mean()
    volume_sma_col = f"volume_sma_{period}"
    df[volume_sma_col] = volume_sma_raw.shift(1)
    df["volume_ratio"] = (df["volume"] / volume_sma_raw).shift(1)

    # Volume z-score (safe: std could be 0 when volume is constant)
    vol_mean = df["volume"].rolling(window=period).mean()
    vol_std = df["volume"].rolling(window=period).std()
    volume_zscore_raw = safe_divide(df["volume"] - vol_mean, vol_std, fill_value=np.nan)
    df["volume_zscore"] = volume_zscore_raw.shift(1)

    feature_metadata["obv"] = "On Balance Volume (lagged)"
    feature_metadata[obv_sma_col] = f"OBV {period}-period SMA (lagged)"
    feature_metadata[volume_sma_col] = f"Volume {period}-period SMA (lagged)"
    feature_metadata["volume_ratio"] = f"Volume ratio to {period}-period SMA (lagged)"
    feature_metadata["volume_zscore"] = f"Volume z-score ({period}, lagged)"

    return df


def add_vwap(df: pd.DataFrame, feature_metadata: dict[str, str]) -> pd.DataFrame:
    """
    Add VWAP (session-based).

    Calculates Volume Weighted Average Price with daily session resets
    and price-to-VWAP ratio.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC, volume, and datetime columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with VWAP features added (if volume data exists)
    """
    if "volume" not in df.columns or df["volume"].sum() == 0:
        logger.info("Skipping VWAP (no volume data)")
        return df

    logger.info("Adding VWAP...")

    # Calculate VWAP per day without breaking index alignment
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["date"] = df["datetime"].dt.date

    # Cumulative volume and price*volume per session (vectorized with transform)
    df["cum_vol"] = df.groupby("date")["volume"].cumsum()
    df["tp_vol"] = df["typical_price"] * df["volume"]
    df["cum_vwap_num"] = df.groupby("date")["tp_vol"].cumsum()

    # Safe division - fallback to typical price if no volume
    vwap_raw = np.where(df["cum_vol"] > 0, df["cum_vwap_num"] / df["cum_vol"], df["typical_price"])

    # ANTI-LOOKAHEAD: shift(1) ensures VWAP at bar[t] uses data up to bar[t-1]
    df["vwap"] = pd.Series(vwap_raw, index=df.index).shift(1)

    # Price to VWAP ratio - compare previous close to previous VWAP
    df["price_to_vwap"] = df["close"].shift(1) / df["vwap"] - 1

    # Cleanup temporary columns
    df = df.drop(columns=["date", "cum_vol", "cum_vwap_num", "typical_price", "tp_vol"])

    feature_metadata["vwap"] = "Volume Weighted Average Price (session, lagged)"
    feature_metadata["price_to_vwap"] = "Price deviation from VWAP (lagged)"

    return df


def add_obv(df: pd.DataFrame, feature_metadata: dict[str, str]) -> pd.DataFrame:
    """
    Add On Balance Volume indicator.

    This is a standalone OBV function if you need just OBV without other volume features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' and 'volume' columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions

    Returns
    -------
    pd.DataFrame
        DataFrame with OBV added (if volume data exists)
    """
    if "volume" not in df.columns or df["volume"].sum() == 0:
        logger.info("Skipping OBV (no volume data)")
        return df

    logger.info("Adding OBV...")

    # ANTI-LOOKAHEAD: shift(1) ensures OBV at bar[t] uses data up to bar[t-1]
    obv_raw = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    df["obv"] = obv_raw.shift(1)

    feature_metadata["obv"] = "On Balance Volume (lagged)"

    return df


def add_dollar_volume(
    df: pd.DataFrame, feature_metadata: dict[str, str], periods: list | None = None
) -> pd.DataFrame:
    """
    Add dollar volume features (price × volume).

    Dollar volume is a better liquidity proxy than raw volume because
    it accounts for price differences.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' and 'volume' columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    periods : list, optional
        Periods for SMA calculation. Default: [10, 20]

    Returns
    -------
    pd.DataFrame
        DataFrame with dollar volume features added
    """
    if "volume" not in df.columns or df["volume"].sum() == 0:
        logger.info("Skipping dollar volume (no volume data)")
        return df

    if periods is None:
        periods = [10, 20]

    logger.info(f"Adding dollar volume features with periods: {periods}")

    # ANTI-LOOKAHEAD: shift(1) ensures dollar_vol at bar[t] uses data up to bar[t-1]
    dollar_vol_raw = df["close"] * df["volume"]
    df["dollar_volume"] = dollar_vol_raw.shift(1)

    for period in periods:
        col = f"dollar_volume_sma_{period}"
        df[col] = dollar_vol_raw.rolling(period).mean().shift(1)
        feature_metadata[col] = f"Dollar volume {period}-period SMA (lagged)"

    # Dollar volume ratio (relative to 20-period mean)
    dv_mean = dollar_vol_raw.rolling(20).mean()
    df["dollar_volume_ratio"] = (dollar_vol_raw / dv_mean.replace(0, np.nan)).shift(1)

    feature_metadata["dollar_volume"] = "Dollar volume (price × volume, lagged)"
    feature_metadata["dollar_volume_ratio"] = "Dollar volume ratio to 20-period SMA (lagged)"

    return df


def add_twap_features(
    df: pd.DataFrame, feature_metadata: dict[str, str], periods: list[int] | None = None
) -> pd.DataFrame:
    """
    Add Time-Weighted Average Price (TWAP) features.

    TWAP is the average of OHLC prices per bar, then optionally smoothed with
    rolling averages. Unlike VWAP which weights by volume, TWAP treats all time
    periods equally, making it useful for:
    - Benchmark price for execution algorithms
    - Fair value estimation independent of volume
    - Detecting volume-independent price trends

    Features added:
    - twap: Raw TWAP for current bar (typical price)
    - twap_{period}: Rolling TWAP over specified periods
    - price_to_twap_{period}: Close price ratio to rolling TWAP
    - twap_slope_{period}: Rate of change of TWAP (momentum)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns (open, high, low, close)
    feature_metadata : dict[str, str]
        Dictionary to store feature descriptions
    periods : list[int], optional
        Periods for rolling TWAP calculation. Default: [10, 20]

    Returns
    -------
    pd.DataFrame
        DataFrame with TWAP features added

    Notes
    -----
    Anti-lookahead: All rolling features use .shift(1) to ensure features
    at bar[t] use data only up to bar[t-1]. The raw twap column is NOT shifted
    as it represents the current bar's typical price (which is known at bar close).
    """
    if periods is None:
        periods = [10, 20]

    logger.info(f"Adding TWAP features with periods: {periods}")

    # TWAP = typical price = (open + high + low + close) / 4
    # This is known at bar close, so raw value does not need shift
    df["twap"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    feature_metadata["twap"] = "Time-Weighted Average Price (typical price)"

    for period in periods:
        # Rolling TWAP - ANTI-LOOKAHEAD: shift(1)
        twap_col = f"twap_{period}"
        df[twap_col] = df["twap"].rolling(window=period).mean().shift(1)
        feature_metadata[twap_col] = f"Rolling TWAP ({period}-period, lagged)"

        # Price to TWAP ratio - compare lagged close to lagged TWAP
        ratio_col = f"price_to_twap_{period}"
        close_lagged = df["close"].shift(1)
        df[ratio_col] = safe_divide(close_lagged, df[twap_col], fill_value=np.nan)
        feature_metadata[ratio_col] = f"Close/TWAP ratio ({period}-period, lagged)"

        # TWAP slope (momentum) - ANTI-LOOKAHEAD: shift(1)
        # Use 5-bar lookback for slope calculation
        slope_lookback = min(5, period // 2) if period > 2 else 1
        slope_col = f"twap_slope_{period}"
        twap_rolling = df["twap"].rolling(window=period).mean()
        df[slope_col] = twap_rolling.pct_change(slope_lookback).shift(1)
        feature_metadata[slope_col] = (
            f"TWAP slope ({period}-period, {slope_lookback}-bar change, lagged)"
        )

    return df


__all__ = [
    "add_volume_features",
    "add_vwap",
    "add_obv",
    "add_dollar_volume",
    "add_twap_features",
]
