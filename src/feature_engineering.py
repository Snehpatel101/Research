"""
Feature Engineering Module for Ensemble Trading System
Generates 50+ technical indicators for price prediction

DEPRECATED: This module is deprecated as of 2025-12-21.
Use 'from stages.stage3_features import FeatureEngineer' instead.

This file is maintained for backward compatibility only.
All functionality has been moved to src/stages/features/ (modular implementation)
and is orchestrated through stages.stage3_features.FeatureEngineer.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Optional
import warnings

# Deprecation warning
warnings.warn(
    "feature_engineering.py is DEPRECATED (2025-12-21). "
    "Use 'from stages.stage3_features import FeatureEngineer' instead.",
    DeprecationWarning,
    stacklevel=2
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log and simple returns."""
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['simple_return'] = df['close'].pct_change()
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    # Fix: Protect against division by zero when open price is zero
    df['close_open_range'] = np.where(
        df['open'] != 0,
        (df['close'] - df['open']) / df['open'],
        0.0
    )
    return df


def compute_sma(df: pd.DataFrame, periods: List[int] = [10, 20, 50, 100, 200]) -> pd.DataFrame:
    """Compute Simple Moving Averages."""
    for period in periods:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'close_to_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
    return df


def compute_ema(df: pd.DataFrame, periods: List[int] = [9, 21, 50]) -> pd.DataFrame:
    """Compute Exponential Moving Averages."""
    for period in periods:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'close_to_ema_{period}'] = df['close'] / df[f'ema_{period}'] - 1
    return df


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute Relative Strength Index."""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # Fix: Handle zero loss case properly to avoid division by zero
    # When loss is 0 and gain > 0: RSI = 100 (all gains, no losses)
    # When loss is 0 and gain = 0: RSI = 50 (no movement, neutral)
    rs = np.where(
        loss > 0,
        gain / loss,
        np.where(gain > 0, 100.0, 0.0)  # 100 if only gains, 0 if no movement
    )
    df['rsi'] = 100 - (100 / (1 + rs))

    # RSI zones
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)

    return df


def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Compute MACD indicator."""
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()

    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_crossover'] = np.sign(df['macd_hist']).diff()

    return df


def compute_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """Compute Bollinger Bands."""
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()

    df['bb_upper'] = sma + (std * std_dev)
    df['bb_lower'] = sma - (std * std_dev)
    # Fix: Protect against division by zero when SMA is zero
    df['bb_width'] = np.where(
        sma > 0,
        (df['bb_upper'] - df['bb_lower']) / sma,
        0.0
    )
    # Fix: Protect against division by zero when band width is zero
    bb_range = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = np.where(
        bb_range > 0,
        (df['close'] - df['bb_lower']) / bb_range,
        0.5  # Neutral position when bands are identical
    )

    return df


def compute_atr(df: pd.DataFrame, periods: List[int] = [7, 14, 21]) -> pd.DataFrame:
    """Compute Average True Range for multiple periods."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    for period in periods:
        df[f'atr_{period}'] = true_range.rolling(window=period).mean()
        df[f'atr_{period}_pct'] = df[f'atr_{period}'] / df['close']

    return df


def compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Compute Stochastic Oscillator."""
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()

    # Fix: Protect against division by zero when high_max equals low_min
    range_hl = high_max - low_min
    df['stoch_k'] = np.where(
        range_hl != 0,
        100 * (df['close'] - low_min) / range_hl,
        50.0  # Neutral value when range is zero
    )
    df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()

    return df


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute Average Directional Index."""
    high_diff = df['high'].diff()
    low_diff = -df['low'].diff()

    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

    # Calculate ATR
    tr = pd.concat([
        df['high'] - df['low'],
        np.abs(df['high'] - df['close'].shift(1)),
        np.abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()

    # Fix: Protect against division by zero when ATR is zero
    plus_di = np.where(
        atr > 0,
        100 * pd.Series(plus_dm).rolling(window=period).mean() / atr,
        0.0
    )
    minus_di = np.where(
        atr > 0,
        100 * pd.Series(minus_dm).rolling(window=period).mean() / atr,
        0.0
    )

    # Fix: Protect against division by zero when plus_di + minus_di is zero
    di_sum = plus_di + minus_di
    dx = np.where(
        di_sum > 0,
        100 * np.abs(plus_di - minus_di) / di_sum,
        0.0
    )
    df['adx'] = pd.Series(dx).rolling(window=period).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di

    return df


def compute_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Compute On-Balance Volume."""
    obv = np.where(df['close'] > df['close'].shift(1), df['volume'],
                   np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))
    df['obv'] = np.cumsum(obv)
    df['obv_sma_20'] = pd.Series(df['obv']).rolling(window=20).mean()

    return df


def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Volume Weighted Average Price (session-based approximation)."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    # Fix: Protect against division by zero when volume sum is zero
    volume_sum = df['volume'].rolling(288).sum()
    df['vwap'] = np.where(
        volume_sum > 0,
        (typical_price * df['volume']).rolling(288).sum() / volume_sum,
        df['close']  # Fallback to close price when no volume
    )
    df['close_to_vwap'] = df['close'] / df['vwap'] - 1

    return df


def compute_roc(df: pd.DataFrame, periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """Compute Rate of Change."""
    for period in periods:
        df[f'roc_{period}'] = df['close'].pct_change(periods=period) * 100
    return df


def compute_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute Williams %R."""
    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()

    # Fix: Protect against division by zero when high_max equals low_min
    range_hl = high_max - low_min
    df['williams_r'] = np.where(
        range_hl != 0,
        -100 * (high_max - df['close']) / range_hl,
        -50.0  # Neutral value when range is zero
    )

    return df


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volume-based features."""
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    # Fix: Protect against division by zero when volume SMA is zero
    df['volume_ratio'] = np.where(
        df['volume_sma_20'] > 0,
        df['volume'] / df['volume_sma_20'],
        1.0  # Neutral ratio when no volume
    )

    # Volume z-score
    vol_std = df['volume'].rolling(window=20).std()
    # Fix: Protect against division by zero when volume std is zero
    df['volume_zscore'] = np.where(
        vol_std > 0,
        (df['volume'] - df['volume_sma_20']) / vol_std,
        0.0  # Neutral z-score when no variance
    )

    return df


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute time-based features."""
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['day_of_week'] = df['datetime'].dt.dayofweek

    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)

    # Trading sessions - 3 equal 8-hour blocks (UTC)
    # Asia: 00:00-08:00, London: 08:00-16:00, NY: 16:00-24:00
    df['session_asia'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['session_london'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['session_ny'] = (df['hour'] >= 16).astype(int)

    # Drop raw temporal columns
    df = df.drop(columns=['hour', 'minute', 'day_of_week'])

    return df


def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute market regime indicators."""
    # Volatility regime
    vol_20 = df['log_return'].rolling(20).std()
    vol_60 = df['log_return'].rolling(60).std()
    df['vol_regime'] = np.where(vol_20 > vol_60, 1, -1)

    # Trend regime
    sma_20 = df['close'].rolling(20).mean()
    sma_50 = df['close'].rolling(50).mean()
    df['trend_regime'] = np.where(sma_20 > sma_50, 1, -1)

    return df


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate all features for a dataframe."""
    logger.info(f"Generating features for {len(df):,} rows...")

    # Make a copy to avoid modifying original
    df = df.copy()

    # Apply all feature generators
    df = compute_returns(df)
    df = compute_sma(df)
    df = compute_ema(df)
    df = compute_rsi(df)
    df = compute_macd(df)
    df = compute_bollinger_bands(df)
    df = compute_atr(df)
    df = compute_stochastic(df)
    df = compute_adx(df)
    df = compute_obv(df)
    df = compute_vwap(df)
    df = compute_roc(df)
    df = compute_williams_r(df)
    df = compute_volume_features(df)
    df = compute_temporal_features(df)
    df = compute_regime_features(df)

    # Drop rows with NaN (from rolling calculations)
    initial_rows = len(df)
    df = df.dropna()
    logger.info(f"Dropped {initial_rows - len(df):,} rows with NaN values")

    # Replace infinities
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    feature_cols = [c for c in df.columns if c not in ['datetime', 'open', 'high', 'low', 'close', 'volume', 'symbol']]
    logger.info(f"Generated {len(feature_cols)} features")

    return df


def process_symbol(input_path: Path, output_path: Path, symbol: str) -> pd.DataFrame:
    """Process features for a single symbol."""
    logger.info(f"="*60)
    logger.info(f"Feature engineering for {symbol}")
    logger.info(f"="*60)

    # Load cleaned data
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} rows from {input_path}")

    # Generate features
    df = generate_features(df)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved features to {output_path}")

    return df


def main():
    """Run feature engineering for all symbols."""
    from config import CLEAN_DATA_DIR, FEATURES_DIR, SYMBOLS

    for symbol in SYMBOLS:
        input_path = CLEAN_DATA_DIR / f"{symbol}_5m_clean.parquet"
        output_path = FEATURES_DIR / f"{symbol}_5m_features.parquet"

        if input_path.exists():
            process_symbol(input_path, output_path, symbol)
        else:
            logger.warning(f"No cleaned data found for {symbol}")


if __name__ == "__main__":
    main()
