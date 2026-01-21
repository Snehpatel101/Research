"""
Momentum indicator features for feature engineering.

This module provides functions to calculate momentum-based technical
indicators including RSI, MACD, Stochastic, Williams %R, ROC, CCI, and MFI.
"""

import logging

import numpy as np
import pandas as pd

from .numba_functions import (
    calculate_ema_numba,
    calculate_rsi_numba,
    calculate_stochastic_numba,
)

logger = logging.getLogger(__name__)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safely divide, returning NaN when denominator is zero."""
    return numerator / denominator.replace(0, np.nan)


def add_rsi(df: pd.DataFrame, feature_metadata: dict[str, str], period: int = 14) -> pd.DataFrame:
    """
    Add RSI features.

    Calculates RSI with overbought/oversold flags.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    period : int, default 14
        RSI period

    Returns
    -------
    pd.DataFrame
        DataFrame with RSI features added
    """
    logger.info(f"Adding RSI features with period: {period}")

    # ANTI-LOOKAHEAD: shift(1) ensures RSI at bar[t] uses data up to bar[t-1]
    col_name = f"rsi_{period}"
    df[col_name] = pd.Series(calculate_rsi_numba(df["close"].values, period)).shift(1).values

    # Overbought/Oversold flags - already shifted via RSI column
    df["rsi_overbought"] = (df[col_name] > 70).astype(int)
    df["rsi_oversold"] = (df[col_name] < 30).astype(int)

    feature_metadata[col_name] = f"{period}-period Relative Strength Index (lagged)"
    feature_metadata["rsi_overbought"] = "RSI overbought flag (>70, lagged)"
    feature_metadata["rsi_oversold"] = "RSI oversold flag (<30, lagged)"

    return df


def add_macd(
    df: pd.DataFrame,
    feature_metadata: dict[str, str],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Add MACD features.

    Calculates MACD line, signal line, histogram, and crossover signals.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    fast_period : int, default 12
        Fast EMA period
    slow_period : int, default 26
        Slow EMA period
    signal_period : int, default 9
        Signal line EMA period

    Returns
    -------
    pd.DataFrame
        DataFrame with MACD features added
    """
    logger.info(f"Adding MACD features ({fast_period},{slow_period},{signal_period})...")

    # ANTI-LOOKAHEAD: All MACD components shifted by 1 bar
    # MACD line
    ema_fast = calculate_ema_numba(df["close"].values, fast_period)
    ema_slow = calculate_ema_numba(df["close"].values, slow_period)
    macd_line_raw = pd.Series(ema_fast - ema_slow)
    df["macd_line"] = macd_line_raw.shift(1).values

    # Signal line (computed on raw, then shifted)
    macd_signal_raw = pd.Series(calculate_ema_numba(macd_line_raw.values, signal_period))
    df["macd_signal"] = macd_signal_raw.shift(1).values

    # MACD histogram
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    # MACD crossovers - compare lagged values (now at t-1 vs t-2 in original terms)
    df["macd_cross_up"] = (
        (df["macd_line"] > df["macd_signal"])
        & (df["macd_line"].shift(1) <= df["macd_signal"].shift(1))
    ).astype(int)
    df["macd_cross_down"] = (
        (df["macd_line"] < df["macd_signal"])
        & (df["macd_line"].shift(1) >= df["macd_signal"].shift(1))
    ).astype(int)

    feature_metadata["macd_line"] = f"MACD line ({fast_period},{slow_period}, lagged)"
    feature_metadata["macd_signal"] = f"MACD signal line ({signal_period}, lagged)"
    feature_metadata["macd_hist"] = "MACD histogram (lagged)"
    feature_metadata["macd_cross_up"] = "MACD bullish crossover (lagged)"
    feature_metadata["macd_cross_down"] = "MACD bearish crossover (lagged)"

    return df


def add_stochastic(
    df: pd.DataFrame, feature_metadata: dict[str, str], k_period: int = 14, d_period: int = 3
) -> pd.DataFrame:
    """
    Add Stochastic Oscillator features.

    Calculates %K and %D with overbought/oversold flags.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    k_period : int, default 14
        %K period
    d_period : int, default 3
        %D smoothing period

    Returns
    -------
    pd.DataFrame
        DataFrame with Stochastic features added
    """
    logger.info(f"Adding Stochastic features ({k_period},{d_period})...")

    k, d = calculate_stochastic_numba(
        df["high"].values,
        df["low"].values,
        df["close"].values,
        k_period=k_period,
        d_period=d_period,
    )

    # ANTI-LOOKAHEAD: shift(1) ensures stochastic at bar[t] uses data up to bar[t-1]
    df["stoch_k"] = pd.Series(k).shift(1).values
    df["stoch_d"] = pd.Series(d).shift(1).values

    # Overbought/Oversold - already shifted via stoch_k
    df["stoch_overbought"] = (df["stoch_k"] > 80).astype(int)
    df["stoch_oversold"] = (df["stoch_k"] < 20).astype(int)

    feature_metadata["stoch_k"] = f"Stochastic %K ({k_period},{d_period}, lagged)"
    feature_metadata["stoch_d"] = f"Stochastic %D ({k_period},{d_period}, lagged)"
    feature_metadata["stoch_overbought"] = "Stochastic overbought flag (>80, lagged)"
    feature_metadata["stoch_oversold"] = "Stochastic oversold flag (<20, lagged)"

    return df


def add_williams_r(
    df: pd.DataFrame, feature_metadata: dict[str, str], period: int = 14
) -> pd.DataFrame:
    """
    Add Williams %R indicator.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    period : int, default 14
        Williams %R period

    Returns
    -------
    pd.DataFrame
        DataFrame with Williams %R added
    """
    logger.info(f"Adding Williams %R with period: {period}")

    high_max = df["high"].rolling(window=period).max()
    low_min = df["low"].rolling(window=period).min()

    # Safe division: (high_max - low_min) could be 0 when price is flat
    # ANTI-LOOKAHEAD: shift(1) ensures Williams %R at bar[t] uses data up to bar[t-1]
    williams_r_raw = -100 * _safe_divide(high_max - df["close"], high_max - low_min)
    df["williams_r"] = williams_r_raw.shift(1)

    feature_metadata["williams_r"] = f"Williams %R ({period}, lagged)"

    return df


def add_roc(
    df: pd.DataFrame, feature_metadata: dict[str, str], periods: list[int] | None = None
) -> pd.DataFrame:
    """
    Add Rate of Change features.

    Calculates ROC for multiple periods.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    periods : List[int], optional
        List of ROC periods. Default: [5, 10, 20]

    Returns
    -------
    pd.DataFrame
        DataFrame with ROC features added
    """
    if periods is None:
        periods = [5, 10, 20]

    logger.info(f"Adding ROC features with periods: {periods}")

    for period in periods:
        # ANTI-LOOKAHEAD: shift(1) ensures ROC at bar[t] uses close up to bar[t-1]
        roc_raw = (df["close"] - df["close"].shift(period)) / df["close"].shift(period) * 100
        df[f"roc_{period}"] = roc_raw.shift(1)

        feature_metadata[f"roc_{period}"] = f"Rate of Change ({period}, lagged)"

    return df


def add_cci(df: pd.DataFrame, feature_metadata: dict[str, str], period: int = 20) -> pd.DataFrame:
    """
    Add Commodity Channel Index.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    period : int, default 20
        CCI period

    Returns
    -------
    pd.DataFrame
        DataFrame with CCI added
    """
    logger.info(f"Adding CCI with period: {period}")

    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())

    # Safe division: mad could be 0 when price is constant
    # ANTI-LOOKAHEAD: shift(1) ensures CCI at bar[t] uses data up to bar[t-1]
    cci_raw = _safe_divide(tp - sma_tp, 0.015 * mad)
    col_name = f"cci_{period}"
    df[col_name] = cci_raw.shift(1)

    feature_metadata[col_name] = f"Commodity Channel Index ({period}, lagged)"

    return df


def add_mfi(df: pd.DataFrame, feature_metadata: dict[str, str], period: int = 14) -> pd.DataFrame:
    """
    Add Money Flow Index (if volume available).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLC and volume columns
    feature_metadata : Dict[str, str]
        Dictionary to store feature descriptions
    period : int, default 14
        MFI period

    Returns
    -------
    pd.DataFrame
        DataFrame with MFI added (if volume data exists)
    """
    if "volume" not in df.columns or df["volume"].sum() == 0:
        logger.info("Skipping MFI (no volume data)")
        return df

    logger.info(f"Adding MFI with period: {period}")

    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]

    mf_pos = mf.where(tp > tp.shift(1), 0).rolling(window=period).sum()
    mf_neg = mf.where(tp < tp.shift(1), 0).rolling(window=period).sum()

    # Safe division: mf_neg could be 0 when no down periods
    # ANTI-LOOKAHEAD: shift(1) ensures MFI at bar[t] uses data up to bar[t-1]
    mfr = _safe_divide(mf_pos, mf_neg)
    mfi_raw = 100 - (100 / (1 + mfr))
    col_name = f"mfi_{period}"
    df[col_name] = mfi_raw.shift(1)

    feature_metadata[col_name] = f"Money Flow Index ({period}, lagged)"

    return df


__all__ = [
    "add_rsi",
    "add_macd",
    "add_stochastic",
    "add_williams_r",
    "add_roc",
    "add_cci",
    "add_mfi",
]
