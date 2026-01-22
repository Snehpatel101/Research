"""
Momentum feature computation - RSI, MACD, Stochastic, Williams %R, ROC, CCI, MFI.

PHASE_1 Unified Features: 23 MOMENTUM features.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _ema(series: pd.Series, span: int, min_periods: int = None) -> pd.Series:
    """Exponential moving average."""
    if min_periods is None:
        min_periods = span
    return series.ewm(span=span, min_periods=min_periods, adjust=False).mean()


def _sma(series: pd.Series, window: int, min_periods: int = None) -> pd.Series:
    """Simple moving average."""
    if min_periods is None:
        min_periods = window
    return series.rolling(window=window, min_periods=min_periods).mean()


def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
    """
    Relative Strength Index calculation.

    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Use EMA for smoothing (Wilder's smoothing)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def _compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator calculation.

    %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = SMA(%K)
    """
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()

    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = hl_range.replace(0, np.nan)

    stoch_k = ((close - lowest_low) / hl_range) * 100
    stoch_d = stoch_k.rolling(window=d_period, min_periods=d_period).mean()

    return stoch_k, stoch_d


def _compute_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD calculation.

    MACD Line = Fast EMA - Slow EMA
    Signal Line = EMA(MACD Line)
    Histogram = MACD Line - Signal Line
    """
    fast_ema = _ema(close, span=fast_period)
    slow_ema = _ema(close, span=slow_period)

    macd_line = fast_ema - slow_ema
    signal_line = _ema(macd_line, span=signal_period)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


# =============================================================================
# RSI FEATURES
# =============================================================================


def compute_rsi_7(df: pd.DataFrame) -> pd.Series:
    """7-period RSI - faster, more sensitive."""
    return _compute_rsi(df["close"], period=7)


def compute_rsi_14(df: pd.DataFrame) -> pd.Series:
    """14-period RSI - standard period."""
    return _compute_rsi(df["close"], period=14)


def compute_rsi_21(df: pd.DataFrame) -> pd.Series:
    """21-period RSI - slower, smoother."""
    return _compute_rsi(df["close"], period=21)


def compute_rsi_overbought(df: pd.DataFrame) -> pd.Series:
    """RSI overbought flag (RSI > 70)."""
    rsi = _compute_rsi(df["close"], period=14)
    return (rsi > 70).astype(float)


def compute_rsi_oversold(df: pd.DataFrame) -> pd.Series:
    """RSI oversold flag (RSI < 30)."""
    rsi = _compute_rsi(df["close"], period=14)
    return (rsi < 30).astype(float)


# =============================================================================
# MACD FEATURES
# =============================================================================


def compute_macd_line(df: pd.DataFrame) -> pd.Series:
    """MACD line (12, 26)."""
    macd_line, _, _ = _compute_macd(df["close"])
    return macd_line


def compute_macd_signal(df: pd.DataFrame) -> pd.Series:
    """MACD signal line (9-period EMA of MACD)."""
    _, signal_line, _ = _compute_macd(df["close"])
    return signal_line


def compute_macd_histogram(df: pd.DataFrame) -> pd.Series:
    """MACD histogram (MACD - Signal)."""
    _, _, histogram = _compute_macd(df["close"])
    return histogram


def compute_macd_cross_up(df: pd.DataFrame) -> pd.Series:
    """MACD bullish crossover (MACD crosses above signal)."""
    macd_line, signal_line, _ = _compute_macd(df["close"])
    cross_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    return cross_up.astype(float)


def compute_macd_cross_down(df: pd.DataFrame) -> pd.Series:
    """MACD bearish crossover (MACD crosses below signal)."""
    macd_line, signal_line, _ = _compute_macd(df["close"])
    cross_down = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
    return cross_down.astype(float)


# =============================================================================
# STOCHASTIC FEATURES
# =============================================================================


def compute_stoch_k(df: pd.DataFrame) -> pd.Series:
    """Stochastic %K (14-period)."""
    stoch_k, _ = _compute_stochastic(df["high"], df["low"], df["close"])
    return stoch_k


def compute_stoch_d(df: pd.DataFrame) -> pd.Series:
    """Stochastic %D (3-period SMA of %K)."""
    _, stoch_d = _compute_stochastic(df["high"], df["low"], df["close"])
    return stoch_d


def compute_stoch_overbought(df: pd.DataFrame) -> pd.Series:
    """Stochastic overbought flag (%K > 80)."""
    stoch_k, _ = _compute_stochastic(df["high"], df["low"], df["close"])
    return (stoch_k > 80).astype(float)


def compute_stoch_oversold(df: pd.DataFrame) -> pd.Series:
    """Stochastic oversold flag (%K < 20)."""
    stoch_k, _ = _compute_stochastic(df["high"], df["low"], df["close"])
    return (stoch_k < 20).astype(float)


# =============================================================================
# WILLIAMS %R
# =============================================================================


def compute_williams_r(df: pd.DataFrame) -> pd.Series:
    """
    Williams %R (14-period).

    %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    Ranges from -100 to 0 (inverted scale).
    """
    period = 14
    highest_high = df["high"].rolling(window=period, min_periods=period).max()
    lowest_low = df["low"].rolling(window=period, min_periods=period).min()

    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = hl_range.replace(0, np.nan)

    williams_r = ((highest_high - df["close"]) / hl_range) * -100
    return williams_r


# =============================================================================
# RATE OF CHANGE (ROC)
# =============================================================================


def compute_roc_5(df: pd.DataFrame) -> pd.Series:
    """5-period Rate of Change."""
    return df["close"].pct_change(periods=5) * 100


def compute_roc_10(df: pd.DataFrame) -> pd.Series:
    """10-period Rate of Change."""
    return df["close"].pct_change(periods=10) * 100


def compute_roc_20(df: pd.DataFrame) -> pd.Series:
    """20-period Rate of Change."""
    return df["close"].pct_change(periods=20) * 100


# =============================================================================
# COMMODITY CHANNEL INDEX (CCI)
# =============================================================================


def _compute_cci(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Commodity Channel Index calculation.

    CCI = (Typical Price - SMA(TP)) / (0.015 * Mean Deviation)
    Typical Price = (High + Low + Close) / 3
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = typical_price.rolling(window=period, min_periods=period).mean()

    # Mean deviation
    mean_dev = typical_price.rolling(window=period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )

    # Avoid division by zero
    mean_dev = mean_dev.replace(0, np.nan)

    cci = (typical_price - sma_tp) / (0.015 * mean_dev)
    return cci


def compute_cci_14(df: pd.DataFrame) -> pd.Series:
    """14-period Commodity Channel Index."""
    return _compute_cci(df, period=14)


def compute_cci_20(df: pd.DataFrame) -> pd.Series:
    """20-period Commodity Channel Index."""
    return _compute_cci(df, period=20)


# =============================================================================
# MONEY FLOW INDEX (MFI)
# =============================================================================


def _compute_mfi(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Money Flow Index calculation.

    MFI = 100 - (100 / (1 + Money Flow Ratio))
    Money Flow Ratio = Positive MF / Negative MF
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    raw_money_flow = typical_price * df["volume"]

    # Determine positive/negative money flow
    tp_diff = typical_price.diff()
    positive_mf = raw_money_flow.where(tp_diff > 0, 0.0)
    negative_mf = raw_money_flow.where(tp_diff < 0, 0.0)

    # Sum over period
    positive_mf_sum = positive_mf.rolling(window=period, min_periods=period).sum()
    negative_mf_sum = negative_mf.rolling(window=period, min_periods=period).sum()

    # Avoid division by zero
    negative_mf_sum = negative_mf_sum.replace(0, np.nan)

    money_flow_ratio = positive_mf_sum / negative_mf_sum
    mfi = 100 - (100 / (1 + money_flow_ratio))

    return mfi


def compute_mfi_14(df: pd.DataFrame) -> pd.Series:
    """14-period Money Flow Index."""
    return _compute_mfi(df, period=14)


def compute_mfi_overbought(df: pd.DataFrame) -> pd.Series:
    """MFI overbought flag (MFI > 80)."""
    mfi = _compute_mfi(df, period=14)
    return (mfi > 80).astype(float)


def compute_mfi_oversold(df: pd.DataFrame) -> pd.Series:
    """MFI oversold flag (MFI < 20)."""
    mfi = _compute_mfi(df, period=14)
    return (mfi < 20).astype(float)


# =============================================================================
# FEATURE MAP
# =============================================================================

MOMENTUM_FEATURES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    # RSI
    "rsi_7": compute_rsi_7,
    "rsi_14": compute_rsi_14,
    "rsi_21": compute_rsi_21,
    "rsi_overbought": compute_rsi_overbought,
    "rsi_oversold": compute_rsi_oversold,
    # MACD
    "macd_line": compute_macd_line,
    "macd_signal": compute_macd_signal,
    "macd_histogram": compute_macd_histogram,
    "macd_cross_up": compute_macd_cross_up,
    "macd_cross_down": compute_macd_cross_down,
    # Stochastic
    "stoch_k": compute_stoch_k,
    "stoch_d": compute_stoch_d,
    "stoch_overbought": compute_stoch_overbought,
    "stoch_oversold": compute_stoch_oversold,
    # Williams %R
    "williams_r": compute_williams_r,
    # ROC
    "roc_5": compute_roc_5,
    "roc_10": compute_roc_10,
    "roc_20": compute_roc_20,
    # CCI
    "cci_14": compute_cci_14,
    "cci_20": compute_cci_20,
    # MFI
    "mfi_14": compute_mfi_14,
    "mfi_overbought": compute_mfi_overbought,
    "mfi_oversold": compute_mfi_oversold,
}

# Feature family metadata
FEATURE_FAMILY = "momentum"
FEATURE_COUNT = 23

__all__ = [
    "MOMENTUM_FEATURES",
    "FEATURE_FAMILY",
    "FEATURE_COUNT",
    # RSI
    "compute_rsi_7",
    "compute_rsi_14",
    "compute_rsi_21",
    "compute_rsi_overbought",
    "compute_rsi_oversold",
    # MACD
    "compute_macd_line",
    "compute_macd_signal",
    "compute_macd_histogram",
    "compute_macd_cross_up",
    "compute_macd_cross_down",
    # Stochastic
    "compute_stoch_k",
    "compute_stoch_d",
    "compute_stoch_overbought",
    "compute_stoch_oversold",
    # Williams %R
    "compute_williams_r",
    # ROC
    "compute_roc_5",
    "compute_roc_10",
    "compute_roc_20",
    # CCI
    "compute_cci_14",
    "compute_cci_20",
    # MFI
    "compute_mfi_14",
    "compute_mfi_overbought",
    "compute_mfi_oversold",
]
