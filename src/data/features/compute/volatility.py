"""
Volatility feature computation - ATR, Bollinger Bands, Keltner Channels, Historical Vol, GARCH.

PHASE_1 Unified Features: 25 VOLATILITY features.

Performance: Uses DataFrame-id based caching for ATR computations to avoid
redundant calculations when multiple features use the same ATR period.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.data.features.compute._helpers import log_returns as _log_returns

# =============================================================================
# CACHING INFRASTRUCTURE
# =============================================================================

# Module-level caches keyed by (DataFrame id, period/window)
# Stores computed results to avoid redundant calculations
_atr_cache: dict[tuple[int, int], pd.Series] = {}
_sma_cache: dict[tuple[int, int], pd.Series] = {}
_ema_cache: dict[tuple[int, int], pd.Series] = {}
_std_cache: dict[tuple[int, int], pd.Series] = {}
_cache_df_id: int | None = None


def _clear_cache_if_df_changed(df: pd.DataFrame) -> None:
    """Clear all caches if DataFrame has changed."""
    global _cache_df_id
    df_id = id(df)
    if df_id != _cache_df_id:
        _atr_cache.clear()
        _sma_cache.clear()
        _ema_cache.clear()
        _std_cache.clear()
        _cache_df_id = df_id


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _true_range(df: pd.DataFrame) -> pd.Series:
    """
    True Range calculation.

    TR = max(High - Low, |High - Prev Close|, |Low - Prev Close|)
    """
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range using Wilder's smoothing. Uses caching."""
    _clear_cache_if_df_changed(df)
    df_id = id(df)
    cache_key = (df_id, period)

    if cache_key in _atr_cache:
        return _atr_cache[cache_key]

    tr = _true_range(df)
    result = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    _atr_cache[cache_key] = result
    return result


def _get_sma_cached(df: pd.DataFrame, column: str, window: int) -> pd.Series:
    """Get cached SMA or compute and cache it."""
    cache_key = (_cache_df_id, column, window)
    if cache_key in _sma_cache:
        return _sma_cache[cache_key]
    result = df[column].rolling(window=window, min_periods=window).mean()
    _sma_cache[cache_key] = result
    return result


def _get_ema_cached(df: pd.DataFrame, column: str, span: int) -> pd.Series:
    """Get cached EMA or compute and cache it."""
    cache_key = (_cache_df_id, column, span)
    if cache_key in _ema_cache:
        return _ema_cache[cache_key]
    result = df[column].ewm(span=span, min_periods=span, adjust=False).mean()
    _ema_cache[cache_key] = result
    return result


def _get_std_cached(df: pd.DataFrame, column: str, window: int) -> pd.Series:
    """Get cached rolling std or compute and cache it."""
    cache_key = (_cache_df_id, column, window)
    if cache_key in _std_cache:
        return _std_cache[cache_key]
    result = df[column].rolling(window=window, min_periods=window).std()
    _std_cache[cache_key] = result
    return result


def _sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average (non-cached, for arbitrary series)."""
    return series.rolling(window=window, min_periods=window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (non-cached, for arbitrary series)."""
    return series.ewm(span=span, min_periods=span, adjust=False).mean()


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation (non-cached, for arbitrary series)."""
    return series.rolling(window=window, min_periods=window).std()


# =============================================================================
# ATR FEATURES
# =============================================================================


def compute_atr_7(df: pd.DataFrame) -> pd.Series:
    """7-period Average True Range."""
    return _atr(df, period=7)


def compute_atr_14(df: pd.DataFrame) -> pd.Series:
    """14-period Average True Range (standard)."""
    return _atr(df, period=14)


def compute_atr_21(df: pd.DataFrame) -> pd.Series:
    """21-period Average True Range."""
    return _atr(df, period=21)


def compute_atr_pct_14(df: pd.DataFrame) -> pd.Series:
    """ATR as percentage of close price."""
    atr_14 = _atr(df, period=14)
    # Avoid division by zero
    close = df["close"].replace(0, np.nan)
    return (atr_14 / close) * 100


# =============================================================================
# BOLLINGER BANDS FEATURES
# =============================================================================


def compute_bb_upper(df: pd.DataFrame) -> pd.Series:
    """Bollinger Band upper (SMA + 2 * STD)."""
    _clear_cache_if_df_changed(df)
    sma_20 = _get_sma_cached(df, "close", 20)
    std_20 = _get_std_cached(df, "close", 20)
    return sma_20 + (2 * std_20)


def compute_bb_middle(df: pd.DataFrame) -> pd.Series:
    """Bollinger Band middle (20-period SMA)."""
    _clear_cache_if_df_changed(df)
    return _get_sma_cached(df, "close", 20)


def compute_bb_lower(df: pd.DataFrame) -> pd.Series:
    """Bollinger Band lower (SMA - 2 * STD)."""
    _clear_cache_if_df_changed(df)
    sma_20 = _get_sma_cached(df, "close", 20)
    std_20 = _get_std_cached(df, "close", 20)
    return sma_20 - (2 * std_20)


def compute_bb_width(df: pd.DataFrame) -> pd.Series:
    """Bollinger Band width (upper - lower) / middle."""
    _clear_cache_if_df_changed(df)
    sma_20 = _get_sma_cached(df, "close", 20)
    std_20 = _get_std_cached(df, "close", 20)
    upper = sma_20 + (2 * std_20)
    lower = sma_20 - (2 * std_20)

    # Avoid division by zero
    middle = sma_20.replace(0, np.nan)
    return (upper - lower) / middle


def compute_bb_position(df: pd.DataFrame) -> pd.Series:
    """
    Position within Bollinger Bands (0 to 1).

    %B = (Close - Lower) / (Upper - Lower)
    """
    _clear_cache_if_df_changed(df)
    sma_20 = _get_sma_cached(df, "close", 20)
    std_20 = _get_std_cached(df, "close", 20)
    upper = sma_20 + (2 * std_20)
    lower = sma_20 - (2 * std_20)

    # Avoid division by zero
    band_width = upper - lower
    band_width = band_width.replace(0, np.nan)

    return (df["close"] - lower) / band_width


def compute_close_bb_zscore(df: pd.DataFrame) -> pd.Series:
    """
    Close price z-score relative to Bollinger Bands.

    z = (Close - SMA) / STD
    """
    _clear_cache_if_df_changed(df)
    sma_20 = _get_sma_cached(df, "close", 20)
    std_20 = _get_std_cached(df, "close", 20)

    # Avoid division by zero
    std_safe = std_20.replace(0, np.nan)

    return (df["close"] - sma_20) / std_safe


# =============================================================================
# KELTNER CHANNELS FEATURES
# =============================================================================


def compute_kc_upper(df: pd.DataFrame) -> pd.Series:
    """Keltner Channel upper (EMA + 2 * ATR)."""
    _clear_cache_if_df_changed(df)
    ema_20 = _get_ema_cached(df, "close", 20)
    atr_10 = _atr(df, period=10)
    return ema_20 + (2 * atr_10)


def compute_kc_middle(df: pd.DataFrame) -> pd.Series:
    """Keltner Channel middle (20-period EMA)."""
    _clear_cache_if_df_changed(df)
    return _get_ema_cached(df, "close", 20)


def compute_kc_lower(df: pd.DataFrame) -> pd.Series:
    """Keltner Channel lower (EMA - 2 * ATR)."""
    _clear_cache_if_df_changed(df)
    ema_20 = _get_ema_cached(df, "close", 20)
    atr_10 = _atr(df, period=10)
    return ema_20 - (2 * atr_10)


def compute_kc_position(df: pd.DataFrame) -> pd.Series:
    """
    Position within Keltner Channels (0 to 1).

    Similar to BB position but using Keltner Channels.
    """
    _clear_cache_if_df_changed(df)
    ema_20 = _get_ema_cached(df, "close", 20)
    atr_10 = _atr(df, period=10)
    upper = ema_20 + (2 * atr_10)
    lower = ema_20 - (2 * atr_10)

    # Avoid division by zero
    channel_width = upper - lower
    channel_width = channel_width.replace(0, np.nan)

    return (df["close"] - lower) / channel_width


# =============================================================================
# HISTORICAL VOLATILITY FEATURES
# =============================================================================
# _log_returns imported from _helpers.py (Phase 29 consolidation)


def compute_hvol_10(df: pd.DataFrame) -> pd.Series:
    """10-period historical volatility (annualized)."""
    log_ret = _log_returns(df["close"])
    return log_ret.rolling(window=10, min_periods=10).std() * np.sqrt(252)


def compute_hvol_20(df: pd.DataFrame) -> pd.Series:
    """20-period historical volatility (annualized)."""
    log_ret = _log_returns(df["close"])
    return log_ret.rolling(window=20, min_periods=20).std() * np.sqrt(252)


def compute_hvol_60(df: pd.DataFrame) -> pd.Series:
    """60-period historical volatility (annualized)."""
    log_ret = _log_returns(df["close"])
    return log_ret.rolling(window=60, min_periods=60).std() * np.sqrt(252)


# =============================================================================
# ADVANCED VOLATILITY ESTIMATORS
# =============================================================================


def compute_parkinson_vol(df: pd.DataFrame) -> pd.Series:
    """
    Parkinson volatility estimator using high-low range.

    More efficient than close-to-close volatility for continuous trading.
    """
    log_hl = np.log(df["high"] / df["low"])
    factor = 1.0 / (4 * np.log(2))
    parkinson = np.sqrt(factor * (log_hl**2).rolling(window=20, min_periods=20).mean())
    return parkinson * np.sqrt(252)


def compute_gk_vol(df: pd.DataFrame) -> pd.Series:
    """
    Garman-Klass volatility estimator.

    Uses OHLC data for more efficient volatility estimation.
    """
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])

    gk_var = (
        (0.5 * (log_hl**2) - (2 * np.log(2) - 1) * (log_co**2))
        .rolling(window=20, min_periods=20)
        .mean()
    )

    return np.sqrt(gk_var.clip(lower=0)) * np.sqrt(252)


def compute_rs_vol(df: pd.DataFrame) -> pd.Series:
    """
    Rogers-Satchell volatility estimator.

    Accounts for drift in the price process.
    """
    log_ho = np.log(df["high"] / df["open"])
    log_hc = np.log(df["high"] / df["close"])
    log_lo = np.log(df["low"] / df["open"])
    log_lc = np.log(df["low"] / df["close"])

    rs_var = (log_ho * log_hc + log_lo * log_lc).rolling(window=20, min_periods=20).mean()
    return np.sqrt(rs_var.clip(lower=0)) * np.sqrt(252)


def compute_yz_vol(df: pd.DataFrame) -> pd.Series:
    """
    Yang-Zhang volatility estimator.

    Combines overnight and open-to-close volatility for best efficiency.
    """
    # Overnight volatility (close to open)
    log_oc = np.log(df["open"] / df["close"].shift(1))
    overnight_var = log_oc.rolling(window=20, min_periods=20).var()

    # Open to close volatility
    log_co = np.log(df["close"] / df["open"])
    intraday_var = log_co.rolling(window=20, min_periods=20).var()

    # Rogers-Satchell component
    log_ho = np.log(df["high"] / df["open"])
    log_hc = np.log(df["high"] / df["close"])
    log_lo = np.log(df["low"] / df["open"])
    log_lc = np.log(df["low"] / df["close"])
    rs_var = (log_ho * log_hc + log_lo * log_lc).rolling(window=20, min_periods=20).mean()

    # Yang-Zhang combination
    k = 0.34 / (1.34 + (21) / (20 - 1))
    yz_var = overnight_var + k * intraday_var + (1 - k) * rs_var

    return np.sqrt(yz_var.clip(lower=0)) * np.sqrt(252)


# =============================================================================
# RETURN DISTRIBUTION FEATURES
# =============================================================================


def compute_return_skew_20(df: pd.DataFrame) -> pd.Series:
    """20-period return skewness."""
    log_ret = _log_returns(df["close"])
    return log_ret.rolling(window=20, min_periods=20).skew()


def compute_return_kurt_20(df: pd.DataFrame) -> pd.Series:
    """20-period return kurtosis (excess kurtosis)."""
    log_ret = _log_returns(df["close"])
    return log_ret.rolling(window=20, min_periods=20).kurt()


# =============================================================================
# GARCH FEATURES (STUB)
# =============================================================================


def compute_garch_vol_forecast(df: pd.DataFrame) -> pd.Series:
    """
    GARCH(1,1) volatility forecast.

    NOTE: This is a stub that returns NaN. Full implementation requires
    arch library and is computationally expensive.
    """
    # Stub implementation - return NaN series
    return pd.Series(np.nan, index=df.index, name="garch_vol_forecast")


def compute_garch_vol_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Ratio of GARCH forecast to realized volatility.

    NOTE: This is a stub that returns NaN.
    """
    # Stub implementation - return NaN series
    return pd.Series(np.nan, index=df.index, name="garch_vol_ratio")


# =============================================================================
# FEATURE MAP
# =============================================================================

VOLATILITY_FEATURES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    # ATR
    "atr_7": compute_atr_7,
    "atr_14": compute_atr_14,
    "atr_21": compute_atr_21,
    "atr_pct_14": compute_atr_pct_14,
    # Bollinger Bands
    "bb_upper": compute_bb_upper,
    "bb_middle": compute_bb_middle,
    "bb_lower": compute_bb_lower,
    "bb_width": compute_bb_width,
    "bb_position": compute_bb_position,
    "close_bb_zscore": compute_close_bb_zscore,
    # Keltner Channels
    "kc_upper": compute_kc_upper,
    "kc_middle": compute_kc_middle,
    "kc_lower": compute_kc_lower,
    "kc_position": compute_kc_position,
    # Historical Volatility
    "hvol_10": compute_hvol_10,
    "hvol_20": compute_hvol_20,
    "hvol_60": compute_hvol_60,
    # Advanced Volatility
    "parkinson_vol": compute_parkinson_vol,
    "gk_vol": compute_gk_vol,
    "rs_vol": compute_rs_vol,
    "yz_vol": compute_yz_vol,
    # Return Distribution
    "return_skew_20": compute_return_skew_20,
    "return_kurt_20": compute_return_kurt_20,
    # GARCH (stubs)
    "garch_vol_forecast": compute_garch_vol_forecast,
    "garch_vol_ratio": compute_garch_vol_ratio,
}

# Feature family metadata
FEATURE_FAMILY = "volatility"
FEATURE_COUNT = 25

__all__ = [
    "VOLATILITY_FEATURES",
    "FEATURE_FAMILY",
    "FEATURE_COUNT",
    # ATR
    "compute_atr_7",
    "compute_atr_14",
    "compute_atr_21",
    "compute_atr_pct_14",
    # Bollinger Bands
    "compute_bb_upper",
    "compute_bb_middle",
    "compute_bb_lower",
    "compute_bb_width",
    "compute_bb_position",
    "compute_close_bb_zscore",
    # Keltner Channels
    "compute_kc_upper",
    "compute_kc_middle",
    "compute_kc_lower",
    "compute_kc_position",
    # Historical Volatility
    "compute_hvol_10",
    "compute_hvol_20",
    "compute_hvol_60",
    # Advanced Volatility
    "compute_parkinson_vol",
    "compute_gk_vol",
    "compute_rs_vol",
    "compute_yz_vol",
    # Return Distribution
    "compute_return_skew_20",
    "compute_return_kurt_20",
    # GARCH
    "compute_garch_vol_forecast",
    "compute_garch_vol_ratio",
]
