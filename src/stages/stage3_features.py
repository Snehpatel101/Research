"""
Stage 3: Feature Engineering Module
Production-ready feature engineering with 50+ technical indicators.

This module generates:
- Price-based features (returns, ratios)
- Moving averages (SMA, EMA)
- Momentum indicators (RSI, MACD, Stochastic, etc.)
- Volatility indicators (ATR, Bollinger Bands, etc.)
- Volume indicators (OBV, VWAP, etc.)
- Trend indicators (ADX, Supertrend, etc.)
- Temporal features (time encoding)
- Regime indicators (volatility, trend)

All with Numba optimization and proper NaN handling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple
import logging
from datetime import datetime
import json
from numba import jit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# NUMBA OPTIMIZED FUNCTIONS
# ============================================================================

@jit(nopython=True)
def calculate_sma_numba(arr: np.ndarray, period: int) -> np.ndarray:
    """Calculate Simple Moving Average using Numba."""
    n = len(arr)
    result = np.full(n, np.nan)

    for i in range(period - 1, n):
        result[i] = np.mean(arr[i - period + 1:i + 1])

    return result


@jit(nopython=True)
def calculate_ema_numba(arr: np.ndarray, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average using Numba."""
    n = len(arr)
    result = np.full(n, np.nan)

    alpha = 2.0 / (period + 1)

    # Start with SMA
    result[period - 1] = np.mean(arr[:period])

    # Calculate EMA
    for i in range(period, n):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]

    return result


@jit(nopython=True)
def calculate_rsi_numba(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI using Numba."""
    n = len(close)
    rsi = np.full(n, np.nan)

    # Calculate price changes
    deltas = np.diff(close)

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Calculate initial averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    # Calculate RSI for remaining periods
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


@jit(nopython=True)
def calculate_atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Average True Range using Numba."""
    n = len(high)
    tr = np.zeros(n)
    atr = np.full(n, np.nan)

    # Calculate True Range
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # Calculate ATR
    atr[period] = np.mean(tr[1:period + 1])

    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


@jit(nopython=True)
def calculate_stochastic_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Stochastic Oscillator using Numba."""
    n = len(close)
    k = np.full(n, np.nan)
    d = np.full(n, np.nan)

    for i in range(k_period - 1, n):
        high_max = np.max(high[i - k_period + 1:i + 1])
        low_min = np.min(low[i - k_period + 1:i + 1])

        if high_max - low_min != 0:
            k[i] = 100.0 * (close[i] - low_min) / (high_max - low_min)
        else:
            k[i] = 50.0

    # Calculate %D (SMA of %K)
    for i in range(k_period + d_period - 2, n):
        d[i] = np.mean(k[i - d_period + 1:i + 1])

    return k, d


@jit(nopython=True)
def calculate_rolling_correlation_numba(x: np.ndarray, y: np.ndarray, period: int) -> np.ndarray:
    """Calculate rolling correlation using Numba."""
    n = len(x)
    result = np.full(n, np.nan)

    for i in range(period - 1, n):
        x_window = x[i - period + 1:i + 1]
        y_window = y[i - period + 1:i + 1]

        x_mean = np.mean(x_window)
        y_mean = np.mean(y_window)

        x_centered = x_window - x_mean
        y_centered = y_window - y_mean

        numerator = np.sum(x_centered * y_centered)
        denominator = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))

        if denominator > 0:
            result[i] = numerator / denominator
        else:
            result[i] = 0.0

    return result


@jit(nopython=True)
def calculate_rolling_beta_numba(y: np.ndarray, x: np.ndarray, period: int) -> np.ndarray:
    """Calculate rolling beta (regression coefficient) of y on x using Numba."""
    n = len(x)
    result = np.full(n, np.nan)

    for i in range(period - 1, n):
        x_window = x[i - period + 1:i + 1]
        y_window = y[i - period + 1:i + 1]

        x_mean = np.mean(x_window)
        y_mean = np.mean(y_window)

        x_centered = x_window - x_mean
        y_centered = y_window - y_mean

        denominator = np.sum(x_centered ** 2)

        if denominator > 0:
            result[i] = np.sum(x_centered * y_centered) / denominator
        else:
            result[i] = 0.0

    return result


@jit(nopython=True)
def calculate_adx_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate ADX, +DI, -DI using Numba."""
    n = len(close)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    adx = np.full(n, np.nan)

    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]

        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff

    # Smooth TR and DM
    tr_smooth = np.zeros(n)
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)

    tr_smooth[period] = np.sum(tr[1:period + 1])
    plus_dm_smooth[period] = np.sum(plus_dm[1:period + 1])
    minus_dm_smooth[period] = np.sum(minus_dm[1:period + 1])

    for i in range(period + 1, n):
        tr_smooth[i] = tr_smooth[i - 1] - (tr_smooth[i - 1] / period) + tr[i]
        plus_dm_smooth[i] = plus_dm_smooth[i - 1] - (plus_dm_smooth[i - 1] / period) + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i - 1] - (minus_dm_smooth[i - 1] / period) + minus_dm[i]

    # Calculate DI
    for i in range(period, n):
        if tr_smooth[i] != 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]

    # Calculate DX and ADX
    dx = np.full(n, np.nan)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum

    # ADX is EMA of DX
    adx[2 * period - 1] = np.mean(dx[period:2 * period])
    for i in range(2 * period, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx, plus_di, minus_di


# ============================================================================
# FEATURE ENGINEERING CLASS
# ============================================================================

class FeatureEngineer:
    """
    Comprehensive feature engineering for financial time series.
    """

    def __init__(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        timeframe: str = '1min'
    ):
        """
        Initialize feature engineer.

        Parameters:
        -----------
        input_dir : Path to cleaned data directory
        output_dir : Path to output directory
        timeframe : Data timeframe (e.g., '1min', '5min')
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.timeframe = timeframe

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Feature metadata
        self.feature_metadata = {}

        logger.info(f"Initialized FeatureEngineer")
        logger.info(f"Input dir: {self.input_dir}")
        logger.info(f"Output dir: {self.output_dir}")

    # ========================================================================
    # PRICE-BASED FEATURES
    # ========================================================================

    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return features."""
        logger.info("Adding return features...")

        periods = [1, 5, 10, 20, 60]

        for period in periods:
            # Simple returns
            df[f'return_{period}'] = df['close'].pct_change(period)

            # Log returns
            df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))

            self.feature_metadata[f'return_{period}'] = f"{period}-period simple return"
            self.feature_metadata[f'log_return_{period}'] = f"{period}-period log return"

        return df

    def add_price_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price ratio features."""
        logger.info("Adding price ratio features...")

        # High/Low ratio
        df['hl_ratio'] = df['high'] / df['low']

        # Close/Open ratio
        df['co_ratio'] = df['close'] / df['open']

        # Range as percentage of close
        df['range_pct'] = (df['high'] - df['low']) / df['close']

        self.feature_metadata['hl_ratio'] = "High to low ratio"
        self.feature_metadata['co_ratio'] = "Close to open ratio"
        self.feature_metadata['range_pct'] = "Range as percentage of close"

        return df

    # ========================================================================
    # MOVING AVERAGES
    # ========================================================================

    def add_sma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Simple Moving Averages."""
        logger.info("Adding SMA features...")

        periods = [10, 20, 50, 100, 200]

        for period in periods:
            df[f'sma_{period}'] = calculate_sma_numba(df['close'].values, period)

            # Price position relative to SMA
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1

            self.feature_metadata[f'sma_{period}'] = f"{period}-period Simple Moving Average"
            self.feature_metadata[f'price_to_sma_{period}'] = f"Price deviation from SMA-{period}"

        return df

    def add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Exponential Moving Averages."""
        logger.info("Adding EMA features...")

        periods = [9, 12, 21, 26, 50]

        for period in periods:
            df[f'ema_{period}'] = calculate_ema_numba(df['close'].values, period)

            # Price position relative to EMA
            df[f'price_to_ema_{period}'] = df['close'] / df[f'ema_{period}'] - 1

            self.feature_metadata[f'ema_{period}'] = f"{period}-period Exponential Moving Average"
            self.feature_metadata[f'price_to_ema_{period}'] = f"Price deviation from EMA-{period}"

        return df

    # ========================================================================
    # MOMENTUM INDICATORS
    # ========================================================================

    def add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI features."""
        logger.info("Adding RSI features...")

        df['rsi_14'] = calculate_rsi_numba(df['close'].values, 14)

        # Overbought/Oversold flags
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)

        self.feature_metadata['rsi_14'] = "14-period Relative Strength Index"
        self.feature_metadata['rsi_overbought'] = "RSI overbought flag (>70)"
        self.feature_metadata['rsi_oversold'] = "RSI oversold flag (<30)"

        return df

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD features."""
        logger.info("Adding MACD features...")

        # MACD line
        ema_12 = calculate_ema_numba(df['close'].values, 12)
        ema_26 = calculate_ema_numba(df['close'].values, 26)
        df['macd_line'] = ema_12 - ema_26

        # Signal line
        df['macd_signal'] = calculate_ema_numba(df['macd_line'].values, 9)

        # MACD histogram
        df['macd_hist'] = df['macd_line'] - df['macd_signal']

        # MACD crossovers
        df['macd_cross_up'] = ((df['macd_line'] > df['macd_signal']) &
                               (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_cross_down'] = ((df['macd_line'] < df['macd_signal']) &
                                 (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))).astype(int)

        self.feature_metadata['macd_line'] = "MACD line (12,26)"
        self.feature_metadata['macd_signal'] = "MACD signal line (9)"
        self.feature_metadata['macd_hist'] = "MACD histogram"
        self.feature_metadata['macd_cross_up'] = "MACD bullish crossover"
        self.feature_metadata['macd_cross_down'] = "MACD bearish crossover"

        return df

    def add_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Stochastic Oscillator."""
        logger.info("Adding Stochastic features...")

        k, d = calculate_stochastic_numba(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            k_period=14,
            d_period=3
        )

        df['stoch_k'] = k
        df['stoch_d'] = d

        # Overbought/Oversold
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)

        self.feature_metadata['stoch_k'] = "Stochastic %K (14,3)"
        self.feature_metadata['stoch_d'] = "Stochastic %D (14,3)"
        self.feature_metadata['stoch_overbought'] = "Stochastic overbought flag (>80)"
        self.feature_metadata['stoch_oversold'] = "Stochastic oversold flag (<20)"

        return df

    def add_williams_r(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Williams %R."""
        logger.info("Adding Williams %R...")

        period = 14
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()

        df['williams_r'] = -100 * (high_max - df['close']) / (high_max - low_min)

        self.feature_metadata['williams_r'] = "Williams %R (14)"

        return df

    def add_roc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Rate of Change."""
        logger.info("Adding ROC features...")

        periods = [5, 10, 20]

        for period in periods:
            df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) /
                                   df['close'].shift(period) * 100)

            self.feature_metadata[f'roc_{period}'] = f"Rate of Change ({period})"

        return df

    def add_cci(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Commodity Channel Index."""
        logger.info("Adding CCI...")

        period = 20
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())

        df['cci_20'] = (tp - sma_tp) / (0.015 * mad)

        self.feature_metadata['cci_20'] = "Commodity Channel Index (20)"

        return df

    def add_mfi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Money Flow Index (if volume available)."""
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            logger.info("Skipping MFI (no volume data)")
            return df

        logger.info("Adding MFI...")

        period = 14
        tp = (df['high'] + df['low'] + df['close']) / 3
        mf = tp * df['volume']

        mf_pos = mf.where(tp > tp.shift(1), 0).rolling(window=period).sum()
        mf_neg = mf.where(tp < tp.shift(1), 0).rolling(window=period).sum()

        mfr = mf_pos / mf_neg
        df['mfi_14'] = 100 - (100 / (1 + mfr))

        self.feature_metadata['mfi_14'] = "Money Flow Index (14)"

        return df

    # ========================================================================
    # VOLATILITY INDICATORS
    # ========================================================================

    def add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Average True Range."""
        logger.info("Adding ATR features...")

        periods = [7, 14, 21]

        for period in periods:
            atr = calculate_atr_numba(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                period
            )
            df[f'atr_{period}'] = atr

            # ATR as percentage of close
            df[f'atr_pct_{period}'] = (atr / df['close']) * 100

            self.feature_metadata[f'atr_{period}'] = f"Average True Range ({period})"
            self.feature_metadata[f'atr_pct_{period}'] = f"ATR as % of price ({period})"

        return df

    def add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands."""
        logger.info("Adding Bollinger Bands...")

        period = 20
        std_mult = 2

        df['bb_middle'] = df['close'].rolling(window=period).mean()
        bb_std = df['close'].rolling(window=period).std()

        df['bb_upper'] = df['bb_middle'] + (std_mult * bb_std)
        df['bb_lower'] = df['bb_middle'] - (std_mult * bb_std)

        # Bollinger Band width
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # Price position in bands
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        self.feature_metadata['bb_middle'] = "Bollinger Band middle (20,2)"
        self.feature_metadata['bb_upper'] = "Bollinger Band upper (20,2)"
        self.feature_metadata['bb_lower'] = "Bollinger Band lower (20,2)"
        self.feature_metadata['bb_width'] = "Bollinger Band width"
        self.feature_metadata['bb_position'] = "Price position in Bollinger Bands"

        return df

    def add_keltner_channels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Keltner Channels."""
        logger.info("Adding Keltner Channels...")

        period = 20
        atr_mult = 2

        ema = calculate_ema_numba(df['close'].values, period)
        atr = calculate_atr_numba(df['high'].values, df['low'].values, df['close'].values, period)

        df['kc_middle'] = ema
        df['kc_upper'] = ema + (atr_mult * atr)
        df['kc_lower'] = ema - (atr_mult * atr)

        # Price position in channels
        df['kc_position'] = (df['close'] - df['kc_lower']) / (df['kc_upper'] - df['kc_lower'])

        self.feature_metadata['kc_middle'] = "Keltner Channel middle (20,2)"
        self.feature_metadata['kc_upper'] = "Keltner Channel upper (20,2)"
        self.feature_metadata['kc_lower'] = "Keltner Channel lower (20,2)"
        self.feature_metadata['kc_position'] = "Price position in Keltner Channels"

        return df

    def add_historical_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add historical volatility."""
        logger.info("Adding historical volatility...")

        periods = [10, 20, 60]
        log_returns = np.log(df['close'] / df['close'].shift(1))

        for period in periods:
            df[f'hvol_{period}'] = log_returns.rolling(window=period).std() * np.sqrt(252 * 390)  # Annualized

            self.feature_metadata[f'hvol_{period}'] = f"Historical volatility ({period})"

        return df

    def add_parkinson_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Parkinson volatility."""
        logger.info("Adding Parkinson volatility...")

        period = 20
        hl_ratio = np.log(df['high'] / df['low'])
        df['parkinson_vol'] = np.sqrt((1 / (4 * np.log(2))) *
                                      (hl_ratio ** 2).rolling(window=period).mean()) * np.sqrt(252 * 390)

        self.feature_metadata['parkinson_vol'] = "Parkinson volatility (20)"

        return df

    def add_garman_klass_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Garman-Klass volatility."""
        logger.info("Adding Garman-Klass volatility...")

        period = 20
        hl = np.log(df['high'] / df['low'])
        co = np.log(df['close'] / df['open'])

        gk = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
        df['gk_vol'] = np.sqrt(gk.rolling(window=period).mean()) * np.sqrt(252 * 390)

        self.feature_metadata['gk_vol'] = "Garman-Klass volatility (20)"

        return df

    # ========================================================================
    # VOLUME INDICATORS
    # ========================================================================

    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            logger.info("Skipping volume features (no volume data)")
            return df

        logger.info("Adding volume features...")

        # OBV
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv'] = obv

        # OBV SMA
        df['obv_sma_20'] = obv.rolling(window=20).mean()

        # Volume SMA and ratios
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        # Volume z-score
        vol_mean = df['volume'].rolling(window=20).mean()
        vol_std = df['volume'].rolling(window=20).std()
        df['volume_zscore'] = (df['volume'] - vol_mean) / vol_std

        self.feature_metadata['obv'] = "On Balance Volume"
        self.feature_metadata['obv_sma_20'] = "OBV 20-period SMA"
        self.feature_metadata['volume_sma_20'] = "Volume 20-period SMA"
        self.feature_metadata['volume_ratio'] = "Volume ratio to 20-period SMA"
        self.feature_metadata['volume_zscore'] = "Volume z-score (20)"

        return df

    def add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add VWAP (session-based)."""
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            logger.info("Skipping VWAP (no volume data)")
            return df

        logger.info("Adding VWAP...")

        # Typical price
        tp = (df['high'] + df['low'] + df['close']) / 3

        # Session-based VWAP (daily reset)
        df['date'] = df['datetime'].dt.date

        vwap_list = []
        for date, group in df.groupby('date'):
            cum_tp_vol = (tp.loc[group.index] * df.loc[group.index, 'volume']).cumsum()
            cum_vol = df.loc[group.index, 'volume'].cumsum()
            vwap = cum_tp_vol / cum_vol
            vwap_list.append(vwap)

        df['vwap'] = pd.concat(vwap_list)

        # Price to VWAP ratio
        df['price_to_vwap'] = df['close'] / df['vwap'] - 1

        df = df.drop('date', axis=1)

        self.feature_metadata['vwap'] = "Volume Weighted Average Price (session)"
        self.feature_metadata['price_to_vwap'] = "Price deviation from VWAP"

        return df

    # ========================================================================
    # TREND INDICATORS
    # ========================================================================

    def add_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ADX and Directional Indicators."""
        logger.info("Adding ADX features...")

        adx, plus_di, minus_di = calculate_adx_numba(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            14
        )

        df['adx_14'] = adx
        df['plus_di_14'] = plus_di
        df['minus_di_14'] = minus_di

        # Trend strength
        df['adx_strong_trend'] = (df['adx_14'] > 25).astype(int)

        self.feature_metadata['adx_14'] = "Average Directional Index (14)"
        self.feature_metadata['plus_di_14'] = "+DI (14)"
        self.feature_metadata['minus_di_14'] = "-DI (14)"
        self.feature_metadata['adx_strong_trend'] = "ADX strong trend flag (>25)"

        return df

    def add_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Supertrend indicator."""
        logger.info("Adding Supertrend...")

        period = 10
        multiplier = 3

        # Calculate ATR
        atr = calculate_atr_numba(df['high'].values, df['low'].values, df['close'].values, period)

        # Calculate basic bands
        hl_avg = (df['high'] + df['low']) / 2
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)

        # Initialize supertrend
        supertrend = np.full(len(df), np.nan)
        direction = np.ones(len(df))

        for i in range(1, len(df)):
            if not np.isnan(upper_band[i]) and not np.isnan(lower_band[i]):
                # Upper band
                if upper_band[i] < supertrend[i-1] if not np.isnan(supertrend[i-1]) else True:
                    upper_band[i] = supertrend[i-1] if not np.isnan(supertrend[i-1]) else upper_band[i]

                # Lower band
                if lower_band[i] > supertrend[i-1] if not np.isnan(supertrend[i-1]) else True:
                    lower_band[i] = supertrend[i-1] if not np.isnan(supertrend[i-1]) else lower_band[i]

                # Determine supertrend
                if df['close'].iloc[i] <= upper_band[i]:
                    supertrend[i] = upper_band[i]
                    direction[i] = -1
                else:
                    supertrend[i] = lower_band[i]
                    direction[i] = 1

        df['supertrend'] = supertrend
        df['supertrend_direction'] = direction

        self.feature_metadata['supertrend'] = "Supertrend (10,3)"
        self.feature_metadata['supertrend_direction'] = "Supertrend direction (1=up, -1=down)"

        return df

    # ========================================================================
    # TEMPORAL FEATURES
    # ========================================================================

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features with sin/cos encoding."""
        logger.info("Adding temporal features...")

        # Hour sin/cos encoding (24-hour cycle)
        df['hour'] = df['datetime'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Minute sin/cos encoding (60-minute cycle)
        df['minute'] = df['datetime'].dt.minute
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

        # Day of week sin/cos encoding (7-day cycle)
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

        # Trading sessions - 3 equal 8-hour blocks covering 24 hours (UTC)
        # Asia:   00:00-08:00 UTC (hours 0-7)
        # London: 08:00-16:00 UTC (hours 8-15)
        # NY:     16:00-24:00 UTC (hours 16-23)
        def get_session(hour):
            if 0 <= hour < 8:
                return 'asia'
            elif 8 <= hour < 16:
                return 'london'
            else:  # 16 <= hour < 24
                return 'ny'

        df['session'] = df['hour'].apply(get_session)

        # One-hot encode session (3 sessions, 8 hours each)
        for session in ['asia', 'london', 'ny']:
            df[f'session_{session}'] = (df['session'] == session).astype(int)

        df = df.drop(['hour', 'minute', 'dayofweek', 'session'], axis=1)

        self.feature_metadata['hour_sin'] = "Hour sine encoding"
        self.feature_metadata['hour_cos'] = "Hour cosine encoding"
        self.feature_metadata['minute_sin'] = "Minute sine encoding"
        self.feature_metadata['minute_cos'] = "Minute cosine encoding"
        self.feature_metadata['dayofweek_sin'] = "Day of week sine encoding"
        self.feature_metadata['dayofweek_cos'] = "Day of week cosine encoding"
        self.feature_metadata['session_asia'] = "Asia session (00:00-08:00 UTC)"
        self.feature_metadata['session_london'] = "London session (08:00-16:00 UTC)"
        self.feature_metadata['session_ny'] = "New York session (16:00-24:00 UTC)"

        return df

    # ========================================================================
    # REGIME INDICATORS
    # ========================================================================

    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime indicators."""
        logger.info("Adding regime features...")

        # Volatility regime (high/low based on historical volatility)
        if 'hvol_20' in df.columns:
            hvol_median = df['hvol_20'].rolling(window=100).median()
            df['volatility_regime'] = (df['hvol_20'] > hvol_median).astype(int)

            self.feature_metadata['volatility_regime'] = "Volatility regime (1=high, 0=low)"

        # Trend regime based on price vs moving averages
        if 'sma_50' in df.columns and 'sma_200' in df.columns:
            # Uptrend: price > SMA50 > SMA200
            uptrend = (df['close'] > df['sma_50']) & (df['sma_50'] > df['sma_200'])

            # Downtrend: price < SMA50 < SMA200
            downtrend = (df['close'] < df['sma_50']) & (df['sma_50'] < df['sma_200'])

            # Trend regime: 1=up, -1=down, 0=sideways
            df['trend_regime'] = np.where(uptrend, 1, np.where(downtrend, -1, 0))

            self.feature_metadata['trend_regime'] = "Trend regime (1=up, -1=down, 0=sideways)"

        return df

    # ========================================================================
    # CROSS-ASSET FEATURES
    # ========================================================================

    def add_cross_asset_features(
        self,
        df: pd.DataFrame,
        mes_close: Optional[np.ndarray] = None,
        mgc_close: Optional[np.ndarray] = None,
        current_symbol: str = ''
    ) -> pd.DataFrame:
        """
        Add cross-asset features between MES (S&P 500 futures) and MGC (Gold futures).

        These features capture:
        - Correlation: Rolling correlation between MES and MGC returns
        - Spread Z-score: Normalized spread between the two assets
        - Beta: Rolling beta of MES returns vs MGC returns
        - Relative Strength: Momentum divergence (MES return - MGC return)

        Parameters:
        -----------
        df : DataFrame to add features to
        mes_close : MES close prices (aligned with df index)
        mgc_close : MGC close prices (aligned with df index)
        current_symbol : The symbol being processed ('MES' or 'MGC')

        Returns:
        --------
        DataFrame with cross-asset features added
        """
        # Check if both assets are available
        has_both_assets = (mes_close is not None and
                          mgc_close is not None and
                          len(mes_close) == len(df) and
                          len(mgc_close) == len(df))

        if not has_both_assets:
            logger.info("Skipping cross-asset features (requires both MES and MGC data)")
            # Set cross-asset features to NaN when only one symbol is present
            df['mes_mgc_correlation_20'] = np.nan
            df['mes_mgc_spread_zscore'] = np.nan
            df['mes_mgc_beta'] = np.nan
            df['relative_strength'] = np.nan

            self.feature_metadata['mes_mgc_correlation_20'] = "20-bar rolling correlation between MES and MGC returns"
            self.feature_metadata['mes_mgc_spread_zscore'] = "Z-score of spread between normalized MES and MGC prices"
            self.feature_metadata['mes_mgc_beta'] = "Rolling beta of MES returns vs MGC returns"
            self.feature_metadata['relative_strength'] = "MES return minus MGC return (momentum divergence)"

            return df

        logger.info("Adding cross-asset features (MES-MGC)...")

        # Calculate returns for both assets
        mes_returns = np.diff(mes_close, prepend=mes_close[0]) / np.where(
            mes_close > 0, mes_close, 1.0
        )
        # Shift to get proper pct_change equivalent
        mes_returns[0] = 0.0

        mgc_returns = np.diff(mgc_close, prepend=mgc_close[0]) / np.where(
            mgc_close > 0, mgc_close, 1.0
        )
        mgc_returns[0] = 0.0

        # Fix: Calculate proper returns (pct_change style)
        mes_returns = np.zeros(len(mes_close))
        mgc_returns = np.zeros(len(mgc_close))
        mes_returns[1:] = (mes_close[1:] - mes_close[:-1]) / mes_close[:-1]
        mgc_returns[1:] = (mgc_close[1:] - mgc_close[:-1]) / mgc_close[:-1]

        # 1. MES-MGC Correlation (20-bar rolling)
        df['mes_mgc_correlation_20'] = calculate_rolling_correlation_numba(
            mes_returns.astype(np.float64),
            mgc_returns.astype(np.float64),
            period=20
        )

        # 2. MES-MGC Spread Z-score
        # Normalize prices to allow comparison (z-score of each)
        period = 20

        # Calculate rolling mean and std for MES
        mes_rolling_mean = pd.Series(mes_close).rolling(window=period).mean().values
        mes_rolling_std = pd.Series(mes_close).rolling(window=period).std().values

        # Calculate rolling mean and std for MGC
        mgc_rolling_mean = pd.Series(mgc_close).rolling(window=period).mean().values
        mgc_rolling_std = pd.Series(mgc_close).rolling(window=period).std().values

        # Normalized prices (z-scores)
        mes_normalized = np.where(
            mes_rolling_std > 0,
            (mes_close - mes_rolling_mean) / mes_rolling_std,
            0.0
        )
        mgc_normalized = np.where(
            mgc_rolling_std > 0,
            (mgc_close - mgc_rolling_mean) / mgc_rolling_std,
            0.0
        )

        # Spread between normalized prices
        spread = mes_normalized - mgc_normalized

        # Z-score of the spread
        spread_mean = pd.Series(spread).rolling(window=period).mean().values
        spread_std = pd.Series(spread).rolling(window=period).std().values
        df['mes_mgc_spread_zscore'] = np.where(
            spread_std > 0,
            (spread - spread_mean) / spread_std,
            0.0
        )

        # 3. MES-MGC Beta (rolling beta of MES returns vs MGC returns)
        df['mes_mgc_beta'] = calculate_rolling_beta_numba(
            mes_returns.astype(np.float64),
            mgc_returns.astype(np.float64),
            period=20
        )

        # 4. Relative Strength (momentum divergence)
        # 20-bar cumulative returns for each asset
        mes_cum_ret = pd.Series(mes_returns).rolling(window=20).sum().values
        mgc_cum_ret = pd.Series(mgc_returns).rolling(window=20).sum().values
        df['relative_strength'] = mes_cum_ret - mgc_cum_ret

        self.feature_metadata['mes_mgc_correlation_20'] = "20-bar rolling correlation between MES and MGC returns"
        self.feature_metadata['mes_mgc_spread_zscore'] = "Z-score of spread between normalized MES and MGC prices"
        self.feature_metadata['mes_mgc_beta'] = "Rolling beta of MES returns vs MGC returns"
        self.feature_metadata['relative_strength'] = "MES return minus MGC return (momentum divergence)"

        return df

    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================

    def engineer_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        cross_asset_data: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete feature engineering pipeline.

        Parameters:
        -----------
        df : Input DataFrame with cleaned OHLCV data
        symbol : Symbol name
        cross_asset_data : Optional dict with 'mes_close' and 'mgc_close' arrays
                          for cross-asset feature computation. Arrays must be
                          aligned with df index (same length and timestamps).

        Returns:
        --------
        tuple : (DataFrame with features, feature report)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting feature engineering for: {symbol}")
        logger.info(f"{'='*60}\n")

        initial_rows = len(df)
        initial_cols = len(df.columns)

        df = df.copy()

        # Add all features
        df = self.add_returns(df)
        df = self.add_price_ratios(df)
        df = self.add_sma(df)
        df = self.add_ema(df)
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_stochastic(df)
        df = self.add_williams_r(df)
        df = self.add_roc(df)
        df = self.add_cci(df)
        df = self.add_mfi(df)
        df = self.add_atr(df)
        df = self.add_bollinger_bands(df)
        df = self.add_keltner_channels(df)
        df = self.add_historical_volatility(df)
        df = self.add_parkinson_volatility(df)
        df = self.add_garman_klass_volatility(df)
        df = self.add_volume_features(df)
        df = self.add_vwap(df)
        df = self.add_adx(df)
        df = self.add_supertrend(df)
        df = self.add_temporal_features(df)
        df = self.add_regime_features(df)

        # Add cross-asset features (MES-MGC)
        mes_close = None
        mgc_close = None
        if cross_asset_data is not None:
            mes_close = cross_asset_data.get('mes_close')
            mgc_close = cross_asset_data.get('mgc_close')
        df = self.add_cross_asset_features(
            df,
            mes_close=mes_close,
            mgc_close=mgc_close,
            current_symbol=symbol
        )

        # Drop rows with NaN (mainly from initial indicator warmup periods)
        # Note: Cross-asset features are NaN when only one symbol is present
        # We use subset to exclude cross-asset columns from dropna if they are all NaN
        cross_asset_cols = ['mes_mgc_correlation_20', 'mes_mgc_spread_zscore',
                           'mes_mgc_beta', 'relative_strength']
        non_cross_asset_cols = [c for c in df.columns if c not in cross_asset_cols]

        rows_before_dropna = len(df)
        # Only drop NaN based on non-cross-asset columns
        df = df.dropna(subset=non_cross_asset_cols)
        rows_dropped = rows_before_dropna - len(df)

        # Check if cross-asset features were computed
        has_cross_asset = (cross_asset_data is not None and
                          'mes_close' in cross_asset_data and
                          'mgc_close' in cross_asset_data)

        feature_report = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'initial_rows': initial_rows,
            'initial_columns': initial_cols,
            'final_rows': len(df),
            'final_columns': len(df.columns),
            'features_added': len(df.columns) - initial_cols,
            'rows_dropped_for_nan': rows_dropped,
            'cross_asset_features': has_cross_asset,
            'cross_asset_feature_names': cross_asset_cols if has_cross_asset else [],
            'date_range': {
                'start': df['datetime'].min().isoformat(),
                'end': df['datetime'].max().isoformat()
            }
        }

        logger.info(f"\nFeature engineering complete for {symbol}")
        logger.info(f"Columns: {initial_cols} -> {len(df.columns)} (+{feature_report['features_added']} features)")
        logger.info(f"Rows: {initial_rows:,} -> {len(df):,} (-{rows_dropped:,} for NaN)")
        if has_cross_asset:
            logger.info(f"Cross-asset features: {cross_asset_cols}")

        return df, feature_report

    def save_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        feature_report: Dict
    ) -> Tuple[Path, Path]:
        """
        Save features and metadata.

        Parameters:
        -----------
        df : DataFrame with features
        symbol : Symbol name
        feature_report : Feature report dict

        Returns:
        --------
        tuple : (data_path, metadata_path)
        """
        # Save features
        data_path = self.output_dir / f"{symbol}_features.parquet"
        df.to_parquet(
            data_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        logger.info(f"Saved features to: {data_path}")

        # Save metadata
        metadata = {
            'report': feature_report,
            'features': self.feature_metadata
        }

        metadata_path = self.output_dir / f"{symbol}_feature_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata to: {metadata_path}")

        return data_path, metadata_path

    def process_file(self, file_path: Union[str, Path]) -> Dict:
        """
        Process a single file.

        Parameters:
        -----------
        file_path : Path to cleaned data file

        Returns:
        --------
        dict : Feature report
        """
        file_path = Path(file_path)
        symbol = file_path.stem.split('_')[0].upper()

        # Load data
        df = pd.read_parquet(file_path)

        # Engineer features
        df, feature_report = self.engineer_features(df, symbol)

        # Save results
        self.save_features(df, symbol, feature_report)

        return feature_report

    def process_directory(self, pattern: str = "*.parquet") -> Dict[str, Dict]:
        """
        Process all files in directory.

        Parameters:
        -----------
        pattern : File pattern to match

        Returns:
        --------
        dict : Dictionary mapping symbols to feature reports
        """
        files = list(self.input_dir.glob(pattern))

        if not files:
            logger.warning(f"No files found matching pattern: {pattern}")
            return {}

        logger.info(f"Found {len(files)} files to process")

        results = {}

        for file_path in files:
            try:
                feature_report = self.process_file(file_path)
                symbol = feature_report['symbol']
                results[symbol] = feature_report

            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}", exc_info=True)
                continue

        return results

    def process_multi_symbol(
        self,
        symbol_files: Dict[str, Union[str, Path]],
        compute_cross_asset: bool = True
    ) -> Dict[str, Dict]:
        """
        Process multiple symbols with cross-asset feature computation.

        This method loads data for multiple symbols, aligns them by timestamp,
        and computes cross-asset features when both MES and MGC are present.

        Parameters:
        -----------
        symbol_files : Dict mapping symbol names to file paths
                      e.g., {'MES': 'path/to/mes.parquet', 'MGC': 'path/to/mgc.parquet'}
        compute_cross_asset : Whether to compute cross-asset features (default True)

        Returns:
        --------
        dict : Dictionary mapping symbols to feature reports
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {len(symbol_files)} symbols with cross-asset features")
        logger.info(f"{'='*60}\n")

        # Load all data
        symbol_data = {}
        for symbol, file_path in symbol_files.items():
            file_path = Path(file_path)
            if file_path.exists():
                df = pd.read_parquet(file_path)
                symbol_data[symbol.upper()] = df
                logger.info(f"Loaded {symbol.upper()}: {len(df):,} rows")
            else:
                logger.warning(f"File not found for {symbol}: {file_path}")

        # Check if we can compute cross-asset features
        has_mes = 'MES' in symbol_data
        has_mgc = 'MGC' in symbol_data
        can_compute_cross_asset = compute_cross_asset and has_mes and has_mgc

        cross_asset_data = None

        if can_compute_cross_asset:
            logger.info("Aligning MES and MGC data for cross-asset features...")

            mes_df = symbol_data['MES'].copy()
            mgc_df = symbol_data['MGC'].copy()

            # Set datetime as index for alignment
            mes_df = mes_df.set_index('datetime')
            mgc_df = mgc_df.set_index('datetime')

            # Get common timestamps
            common_idx = mes_df.index.intersection(mgc_df.index)
            logger.info(f"Common timestamps: {len(common_idx):,}")

            if len(common_idx) > 0:
                # Align data to common timestamps
                mes_aligned = mes_df.loc[common_idx].reset_index()
                mgc_aligned = mgc_df.loc[common_idx].reset_index()

                # Update symbol_data with aligned data
                symbol_data['MES'] = mes_aligned
                symbol_data['MGC'] = mgc_aligned

                # Prepare cross-asset data
                cross_asset_data = {
                    'mes_close': mes_aligned['close'].values,
                    'mgc_close': mgc_aligned['close'].values
                }
            else:
                logger.warning("No common timestamps found, skipping cross-asset features")
                can_compute_cross_asset = False

        # Process each symbol
        results = {}
        for symbol, df in symbol_data.items():
            try:
                logger.info(f"\nProcessing {symbol}...")

                # Engineer features with cross-asset data
                df_features, feature_report = self.engineer_features(
                    df,
                    symbol,
                    cross_asset_data=cross_asset_data if can_compute_cross_asset else None
                )

                # Save results
                self.save_features(df_features, symbol, feature_report)
                results[symbol] = feature_report

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}", exc_info=True)
                continue

        return results


def main():
    """
    Example usage of FeatureEngineer.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Stage 3: Feature Engineering')
    parser.add_argument('--input-dir', type=str, default='data/clean',
                        help='Input data directory')
    parser.add_argument('--output-dir', type=str, default='data/features',
                        help='Output directory')
    parser.add_argument('--timeframe', type=str, default='1min',
                        help='Data timeframe')
    parser.add_argument('--pattern', type=str, default='*.parquet',
                        help='File pattern to match')
    parser.add_argument('--multi-symbol', action='store_true',
                        help='Process MES and MGC together with cross-asset features')
    parser.add_argument('--symbols', type=str, nargs='+', default=['MES', 'MGC'],
                        help='Symbols to process (for multi-symbol mode)')

    args = parser.parse_args()

    # Initialize feature engineer
    engineer = FeatureEngineer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        timeframe=args.timeframe
    )

    if args.multi_symbol:
        # Build symbol_files dict
        input_dir = Path(args.input_dir)
        symbol_files = {}
        for symbol in args.symbols:
            # Look for files matching the symbol
            matches = list(input_dir.glob(f"{symbol.lower()}*.parquet")) + \
                      list(input_dir.glob(f"{symbol.upper()}*.parquet"))
            if matches:
                symbol_files[symbol.upper()] = matches[0]
                print(f"Found {symbol.upper()}: {matches[0]}")
            else:
                print(f"Warning: No file found for {symbol}")

        if symbol_files:
            results = engineer.process_multi_symbol(symbol_files, compute_cross_asset=True)
        else:
            print("No files found for specified symbols")
            results = {}
    else:
        # Process all files independently
        results = engineer.process_directory(pattern=args.pattern)

    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)
    for symbol, report in results.items():
        print(f"\n{symbol}:")
        print(f"  Features added: {report['features_added']}")
        print(f"  Final columns: {report['final_columns']}")
        print(f"  Final rows: {report['final_rows']:,}")
        print(f"  Cross-asset features: {report.get('cross_asset_features', False)}")
        print(f"  Date range: {report['date_range']['start']} to {report['date_range']['end']}")


if __name__ == '__main__':
    main()
