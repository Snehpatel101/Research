"""
Volatility indicator features for feature engineering.

This module provides functions to calculate volatility-based technical
indicators including ATR, Bollinger Bands, Keltner Channels, and various
historical volatility measures.
"""

import logging

import numpy as np
import pandas as pd

from .constants import get_annualization_factor
from .numba_functions import calculate_atr_numba, calculate_ema_numba

logger = logging.getLogger(__name__)


def add_atr(
    df: pd.DataFrame, feature_metadata: dict[str, str], periods: list[int] | None = None
) -> pd.DataFrame:
    """
    Add Average True Range features.

    Calculates ATR for multiple periods with both absolute and percentage values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    periods : List[int], optional
        List of ATR periods. Default: [7, 14, 21]

    Returns
    -------
    pd.DataFrame
        DataFrame with ATR features added
    """
    if periods is None:
        periods = [7, 14, 21]

    logger.info(f"Adding ATR features with periods: {periods}")

    for period in periods:
        atr = calculate_atr_numba(df["high"].values, df["low"].values, df["close"].values, period)
        # ANTI-LOOKAHEAD: shift(1) ensures ATR at bar[t] uses data up to bar[t-1]
        df[f"atr_{period}"] = pd.Series(atr).shift(1).values

        # ATR as percentage of close (safe division)
        # Use lagged close to match lagged ATR
        close_lagged = df["close"].shift(1).replace(0, np.nan)
        df[f"atr_pct_{period}"] = (df[f"atr_{period}"] / close_lagged) * 100

        feature_metadata[f"atr_{period}"] = f"Average True Range ({period}, lagged)"
        feature_metadata[f"atr_pct_{period}"] = f"ATR as % of price ({period}, lagged)"

    return df


def add_bollinger_bands(
    df: pd.DataFrame, feature_metadata: dict[str, str], period: int = 20, std_mult: float = 2.0
) -> pd.DataFrame:
    """
    Add Bollinger Bands features.

    Calculates Bollinger Bands with configurable period and standard deviation,
    band width (normalized), and price position within the bands.

    All features are made stationary using z-scores and normalized values
    to avoid dependency on absolute price levels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    period : int, default 20
        Bollinger Band period
    std_mult : float, default 2.0
        Standard deviation multiplier

    Returns
    -------
    pd.DataFrame
        DataFrame with Bollinger Band features added
    """
    logger.info(f"Adding Bollinger Bands with period: {period}")

    # ANTI-LOOKAHEAD: shift(1) ensures BB at bar[t] uses data up to bar[t-1]
    bb_middle_raw = df["close"].rolling(window=period).mean()
    bb_std_raw = df["close"].rolling(window=period).std()

    df["bb_middle"] = bb_middle_raw.shift(1)
    bb_std = bb_std_raw.shift(1)

    df["bb_upper"] = df["bb_middle"] + (std_mult * bb_std)
    df["bb_lower"] = df["bb_middle"] - (std_mult * bb_std)

    # Bollinger Band width normalized by std (stationary)
    # This is equivalent to band_width / std, making it scale-invariant
    bb_std_safe = bb_std.replace(0, np.nan)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_std_safe

    # Price position in bands - use lagged close to match lagged bands
    # Use safe division to handle band collapse
    band_range = df["bb_upper"] - df["bb_lower"]
    band_range_safe = band_range.replace(0, np.nan)
    close_lagged = df["close"].shift(1)
    df["bb_position"] = (close_lagged - df["bb_lower"]) / band_range_safe

    # Add close price z-score relative to BB middle (stationary)
    df["close_bb_zscore"] = (close_lagged - df["bb_middle"]) / bb_std_safe

    feature_metadata["bb_middle"] = f"Bollinger Band middle ({period},{std_mult}, lagged)"
    feature_metadata["bb_upper"] = f"Bollinger Band upper ({period},{std_mult}, lagged)"
    feature_metadata["bb_lower"] = f"Bollinger Band lower ({period},{std_mult}, lagged)"
    feature_metadata["bb_width"] = "Bollinger Band width (normalized by std, lagged)"
    feature_metadata["bb_position"] = "Price position in Bollinger Bands [0,1] (lagged)"
    feature_metadata["close_bb_zscore"] = "Close price z-score relative to BB middle (lagged)"

    return df


def add_keltner_channels(
    df: pd.DataFrame, feature_metadata: dict[str, str], period: int = 20, atr_mult: float = 2.0
) -> pd.DataFrame:
    """
    Add Keltner Channels features.

    Calculates Keltner Channels with configurable period and ATR multiplier
    and price position within the channels.

    All features use safe division and stationary representations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    period : int, default 20
        Keltner Channel period
    atr_mult : float, default 2.0
        ATR multiplier

    Returns
    -------
    pd.DataFrame
        DataFrame with Keltner Channel features added
    """
    logger.info(f"Adding Keltner Channels with period: {period}")

    ema_raw = calculate_ema_numba(df["close"].values, period)
    atr_raw = calculate_atr_numba(df["high"].values, df["low"].values, df["close"].values, period)

    # ANTI-LOOKAHEAD: shift(1) ensures KC at bar[t] uses data up to bar[t-1]
    ema = pd.Series(ema_raw).shift(1).values
    atr = pd.Series(atr_raw).shift(1).values

    df["kc_middle"] = ema
    df["kc_upper"] = ema + (atr_mult * atr)
    df["kc_lower"] = ema - (atr_mult * atr)

    # Price position in channels - use lagged close to match lagged channels
    # Use safe division to handle channel collapse
    channel_range = df["kc_upper"] - df["kc_lower"]
    channel_range_safe = channel_range.replace(0, np.nan)
    close_lagged = df["close"].shift(1)
    df["kc_position"] = (close_lagged - df["kc_lower"]) / channel_range_safe

    # Add close deviation from KC middle in ATR units (stationary)
    atr_safe = pd.Series(atr).replace(0, np.nan)
    df["close_kc_atr_dev"] = (close_lagged - df["kc_middle"]) / atr_safe

    feature_metadata["kc_middle"] = f"Keltner Channel middle ({period},{atr_mult}, lagged)"
    feature_metadata["kc_upper"] = f"Keltner Channel upper ({period},{atr_mult}, lagged)"
    feature_metadata["kc_lower"] = f"Keltner Channel lower ({period},{atr_mult}, lagged)"
    feature_metadata["kc_position"] = "Price position in Keltner Channels [0,1] (lagged)"
    feature_metadata["close_kc_atr_dev"] = "Close deviation from KC middle in ATR units (lagged)"

    return df


def add_historical_volatility(
    df: pd.DataFrame,
    feature_metadata: dict[str, str],
    periods: list[int] | None = None,
    timeframe: str = "5min",
) -> pd.DataFrame:
    """
    Add historical volatility features.

    Calculates annualized historical volatility for multiple periods.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    periods : List[int], optional
        List of volatility periods. Default: [10, 20, 60]
    timeframe : str, default '5min'
        Bar timeframe for annualization factor calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with historical volatility features added
    """
    if periods is None:
        periods = [10, 20, 60]

    logger.info(f"Adding historical volatility with periods: {periods}")

    log_returns = np.log(df["close"] / df["close"].shift(1))
    annualization_factor = get_annualization_factor(timeframe)

    for period in periods:
        # ANTI-LOOKAHEAD: shift(1) ensures hvol at bar[t] uses data up to bar[t-1]
        df[f"hvol_{period}"] = (
            log_returns.rolling(window=period).std() * annualization_factor
        ).shift(1)

        feature_metadata[f"hvol_{period}"] = f"Historical volatility ({period}, lagged)"

    return df


def add_parkinson_volatility(
    df: pd.DataFrame, feature_metadata: dict[str, str], period: int = 20, timeframe: str = "5min"
) -> pd.DataFrame:
    """
    Add Parkinson volatility.

    Parkinson volatility uses high-low range to estimate volatility,
    which is more efficient than close-to-close volatility.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'high' and 'low' columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    period : int, default 20
        Parkinson volatility period
    timeframe : str, default '5min'
        Bar timeframe for annualization factor calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with Parkinson volatility added
    """
    logger.info(f"Adding Parkinson volatility with period: {period}")

    hl_ratio = np.log(df["high"] / df["low"])
    annualization_factor = get_annualization_factor(timeframe)
    # ANTI-LOOKAHEAD: shift(1) ensures parkinson_vol at bar[t] uses data up to bar[t-1]
    parkinson_raw = (
        np.sqrt((1 / (4 * np.log(2))) * (hl_ratio**2).rolling(window=period).mean())
        * annualization_factor
    )
    df["parkinson_vol"] = parkinson_raw.shift(1)

    feature_metadata["parkinson_vol"] = f"Parkinson volatility ({period}, lagged)"

    return df


def add_garman_klass_volatility(
    df: pd.DataFrame, feature_metadata: dict[str, str], period: int = 20, timeframe: str = "5min"
) -> pd.DataFrame:
    """
    Add Garman-Klass volatility.

    Garman-Klass volatility uses OHLC data for more efficient volatility estimation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    period : int, default 20
        Garman-Klass volatility period
    timeframe : str, default '5min'
        Bar timeframe for annualization factor calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with Garman-Klass volatility added
    """
    logger.info(f"Adding Garman-Klass volatility with period: {period}")

    hl = np.log(df["high"] / df["low"])
    co = np.log(df["close"] / df["open"])
    annualization_factor = get_annualization_factor(timeframe)

    gk = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
    # ANTI-LOOKAHEAD: shift(1) ensures gk_vol at bar[t] uses data up to bar[t-1]
    df["gk_vol"] = (np.sqrt(gk.rolling(window=period).mean()) * annualization_factor).shift(1)

    feature_metadata["gk_vol"] = f"Garman-Klass volatility ({period}, lagged)"

    return df


def add_higher_moments(
    df: pd.DataFrame, feature_metadata: dict[str, str], periods: list[int] | None = None
) -> pd.DataFrame:
    """
    Add rolling skewness and kurtosis of returns.

    Higher moments measure distribution shape:
    - Skewness: asymmetry (negative = fat left tail, crash risk)
    - Kurtosis: tail fatness (high = more extreme moves)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    periods : List[int], optional
        List of rolling periods. Default: [20, 60]

    Returns
    -------
    pd.DataFrame
        DataFrame with skewness and kurtosis features added
    """
    if periods is None:
        periods = [20, 60]

    logger.info(f"Adding higher moments (skewness/kurtosis) with periods: {periods}")

    returns = df["close"].pct_change()

    for period in periods:
        # Skewness - ANTI-LOOKAHEAD: shift(1)
        skew_col = f"return_skew_{period}"
        df[skew_col] = returns.rolling(period).skew().shift(1)
        feature_metadata[skew_col] = f"Return skewness ({period}-period, lagged)"

        # Kurtosis (excess) - ANTI-LOOKAHEAD: shift(1)
        kurt_col = f"return_kurt_{period}"
        df[kurt_col] = returns.rolling(period).kurt().shift(1)
        feature_metadata[kurt_col] = f"Return excess kurtosis ({period}-period, lagged)"

    return df


def add_rogers_satchell_volatility(
    df: pd.DataFrame, feature_metadata: dict[str, str], period: int = 20, timeframe: str = "5min"
) -> pd.DataFrame:
    """
    Add Rogers-Satchell volatility.

    Rogers-Satchell volatility handles non-zero drift (trending markets) better
    than close-to-close volatility. It uses the relationship between OHLC prices
    and does not assume the mean return is zero.

    Formula:
        RS = sqrt(mean((H-C)(H-O) + (L-C)(L-O)))

    Where H, L, O, C are log prices (high, low, open, close).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    period : int, default 20
        Rolling window period
    timeframe : str, default '5min'
        Bar timeframe for annualization factor calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with Rogers-Satchell volatility added
    """
    logger.info(f"Adding Rogers-Satchell volatility with period: {period}")

    annualization_factor = get_annualization_factor(timeframe)

    # Log prices for the calculation
    log_high = np.log(df["high"])
    log_low = np.log(df["low"])
    log_open = np.log(df["open"])
    log_close = np.log(df["close"])

    # Rogers-Satchell formula: (H-C)(H-O) + (L-C)(L-O)
    rs_component = (log_high - log_close) * (log_high - log_open) + (log_low - log_close) * (
        log_low - log_open
    )

    # Rolling mean and sqrt for volatility
    rs_vol_raw = np.sqrt(rs_component.rolling(window=period).mean()) * annualization_factor

    # ANTI-LOOKAHEAD: shift(1) ensures rs_vol at bar[t] uses data up to bar[t-1]
    df["rs_vol"] = rs_vol_raw.shift(1)

    feature_metadata["rs_vol"] = f"Rogers-Satchell volatility ({period}, lagged)"

    return df


def add_yang_zhang_volatility(
    df: pd.DataFrame, feature_metadata: dict[str, str], period: int = 20, timeframe: str = "5min"
) -> pd.DataFrame:
    """
    Add Yang-Zhang volatility.

    Yang-Zhang volatility is a comprehensive estimator that handles both
    drift (trending markets) and opening jumps (overnight gaps). It combines
    overnight volatility, open-to-close volatility, and Rogers-Satchell volatility.

    Formula:
        YZ = sqrt(vol_overnight^2 + k*vol_open_close^2 + (1-k)*vol_rogers_satchell^2)

    Where k = 0.34 / (1.34 + (n+1)/(n-1)) is an optimal weighting factor.

    This is considered one of the most efficient volatility estimators for
    financial data with overnight gaps.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    period : int, default 20
        Rolling window period
    timeframe : str, default '5min'
        Bar timeframe for annualization factor calculation

    Returns
    -------
    pd.DataFrame
        DataFrame with Yang-Zhang volatility added
    """
    logger.info(f"Adding Yang-Zhang volatility with period: {period}")

    annualization_factor = get_annualization_factor(timeframe)
    n = period

    # Optimal weighting factor k
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    # Log prices
    log_open = np.log(df["open"])
    log_close = np.log(df["close"])
    log_high = np.log(df["high"])
    log_low = np.log(df["low"])

    # Overnight return (close to open gap)
    log_close_prev = log_close.shift(1)
    overnight_return = log_open - log_close_prev

    # Open-to-close return
    open_close_return = log_close - log_open

    # Variance components
    # 1. Overnight variance
    overnight_mean = overnight_return.rolling(window=period).mean()
    overnight_var = ((overnight_return - overnight_mean) ** 2).rolling(window=period).mean()

    # 2. Open-to-close variance
    oc_mean = open_close_return.rolling(window=period).mean()
    open_close_var = ((open_close_return - oc_mean) ** 2).rolling(window=period).mean()

    # 3. Rogers-Satchell variance (drift-independent intraday variance)
    rs_component = (log_high - log_close) * (log_high - log_open) + (log_low - log_close) * (
        log_low - log_open
    )
    rs_var = rs_component.rolling(window=period).mean()

    # Yang-Zhang volatility: combines all three components
    yz_var = overnight_var + k * open_close_var + (1 - k) * rs_var
    yz_vol_raw = np.sqrt(yz_var) * annualization_factor

    # ANTI-LOOKAHEAD: shift(1) ensures yz_vol at bar[t] uses data up to bar[t-1]
    df["yz_vol"] = yz_vol_raw.shift(1)

    feature_metadata["yz_vol"] = f"Yang-Zhang volatility ({period}, lagged)"

    return df


__all__ = [
    "add_atr",
    "add_bollinger_bands",
    "add_keltner_channels",
    "add_historical_volatility",
    "add_parkinson_volatility",
    "add_garman_klass_volatility",
    "add_higher_moments",
    "add_rogers_satchell_volatility",
    "add_yang_zhang_volatility",
]
