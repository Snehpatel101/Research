"""
Test data generators for Phase 1 tests.

Provides specialized data generation functions for creating realistic
test data with specific characteristics:
- Market scenarios (trending, ranging, volatile)
- Edge cases (gaps, outliers, missing data)
- Labeled data with quality scores
- Feature matrices with various distributions

All generators are deterministic (use fixed seeds) for reproducibility.
"""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd


def generate_trending_market(
    n: int = 500,
    base_price: float = 4500.0,
    trend: float = 0.0002,
    volatility: float = 0.001,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate OHLCV data with a trending market.

    Args:
        n: Number of bars
        base_price: Starting price
        trend: Daily trend (e.g., 0.0002 = 0.02% daily drift)
        volatility: Daily volatility (std of returns)
        seed: Random seed for reproducibility

    Returns:
        pd.DataFrame: OHLCV data with upward/downward trend
    """
    np.random.seed(seed)

    # Generate price with trend
    drift = np.arange(n) * trend
    noise = np.random.randn(n) * volatility
    log_returns = drift + noise
    close = base_price * np.exp(np.cumsum(log_returns))

    # Generate OHLC
    daily_range = np.abs(np.random.randn(n) * volatility * 2)
    high = close * (1 + daily_range / 2)
    low = close * (1 - daily_range / 2)
    open_ = close * (1 + np.random.randn(n) * volatility * 0.5)

    # Ensure valid OHLC relationships
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = np.random.randint(100, 10000, n)

    start_time = datetime(2024, 1, 1, 9, 30)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n)]

    return pd.DataFrame({
        'datetime': timestamps,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


def generate_ranging_market(
    n: int = 500,
    base_price: float = 4500.0,
    range_pct: float = 0.02,
    volatility: float = 0.001,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate OHLCV data with a ranging (sideways) market.

    Args:
        n: Number of bars
        base_price: Center price of range
        range_pct: Range as percentage of base price (e.g., 0.02 = 2% range)
        volatility: Daily volatility
        seed: Random seed

    Returns:
        pd.DataFrame: OHLCV data oscillating within range
    """
    np.random.seed(seed)

    # Generate mean-reverting price
    price = base_price
    prices = []
    for _ in range(n):
        # Mean reversion: pull toward base_price
        pull_to_mean = (base_price - price) * 0.1
        noise = np.random.randn() * volatility * base_price
        price += pull_to_mean + noise
        # Enforce range boundaries
        price = np.clip(price, base_price * (1 - range_pct), base_price * (1 + range_pct))
        prices.append(price)

    close = np.array(prices)

    # Generate OHLC
    daily_range = np.abs(np.random.randn(n) * volatility * base_price * 2)
    high = close + daily_range / 2
    low = close - daily_range / 2
    open_ = close + np.random.randn(n) * volatility * base_price * 0.5

    # Ensure valid OHLC
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = np.random.randint(100, 10000, n)

    start_time = datetime(2024, 1, 1, 9, 30)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n)]

    return pd.DataFrame({
        'datetime': timestamps,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


def generate_volatile_market(
    n: int = 500,
    base_price: float = 4500.0,
    volatility: float = 0.005,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate OHLCV data with high volatility.

    Args:
        n: Number of bars
        base_price: Starting price
        volatility: High volatility level (e.g., 0.005 = 0.5% daily)
        seed: Random seed

    Returns:
        pd.DataFrame: Highly volatile OHLCV data
    """
    return generate_trending_market(n, base_price, trend=0, volatility=volatility, seed=seed)


def add_gaps(df: pd.DataFrame, gap_indices: list[int]) -> pd.DataFrame:
    """
    Add gaps to OHLCV data by removing specific rows.

    Args:
        df: OHLCV DataFrame
        gap_indices: Indices to remove to create gaps

    Returns:
        pd.DataFrame: OHLCV data with gaps
    """
    return df.drop(gap_indices).reset_index(drop=True)


def add_outliers(
    df: pd.DataFrame,
    outlier_indices: list[int],
    outlier_multiplier: float = 5.0
) -> pd.DataFrame:
    """
    Add outliers to OHLCV data.

    Args:
        df: OHLCV DataFrame
        outlier_indices: Indices where to add outliers
        outlier_multiplier: Multiplier for outlier magnitude

    Returns:
        pd.DataFrame: OHLCV data with outliers
    """
    df = df.copy()
    for idx in outlier_indices:
        if idx < len(df):
            # Add extreme spike to high/low
            typical_range = df.loc[idx, 'high'] - df.loc[idx, 'low']
            df.loc[idx, 'high'] += typical_range * outlier_multiplier
            df.loc[idx, 'low'] -= typical_range * outlier_multiplier

    return df


def add_duplicates(df: pd.DataFrame, duplicate_indices: list[int]) -> pd.DataFrame:
    """
    Add duplicate timestamps to OHLCV data.

    Args:
        df: OHLCV DataFrame
        duplicate_indices: Indices to duplicate

    Returns:
        pd.DataFrame: OHLCV data with duplicates
    """
    df = df.copy()
    duplicates = df.loc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    return df.sort_values('datetime').reset_index(drop=True)


def generate_labeled_data(
    n: int = 1000,
    symbol: str = 'MES',
    label_distribution: Optional[tuple[float, float, float]] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate labeled OHLCV data for testing stages 5-8.

    Args:
        n: Number of bars
        symbol: Symbol name
        label_distribution: Tuple of (neutral_pct, up_pct, down_pct)
        seed: Random seed

    Returns:
        pd.DataFrame: Labeled OHLCV data with label columns
    """
    np.random.seed(seed)

    if label_distribution is None:
        label_distribution = (0.6, 0.2, 0.2)  # Default: 60% neutral, 20% up, 20% down

    # Generate base OHLCV
    base_price = 100.0
    returns = np.random.randn(n) * 0.002
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
        'symbol': symbol,
        'open': prices * (1 + np.random.randn(n) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n)) * 0.002),
        'low': prices * (1 - np.abs(np.random.randn(n)) * 0.002),
        'close': prices,
        'volume': np.random.randint(100, 1000, n).astype(float),
    })

    # Ensure valid OHLC
    df['high'] = df[['open', 'high', 'close']].max(axis=1) * 1.001
    df['low'] = df[['open', 'low', 'close']].min(axis=1) * 0.999

    # Add ATR
    df['atr_14'] = (df['high'] - df['low']).rolling(14).mean().fillna(0.5)

    # Add labels
    labels = np.random.choice([0, 1, -1], size=n, p=label_distribution)
    df['label_H5'] = labels
    df['label_H20'] = np.roll(labels, 5)  # Slightly different for H20

    return df


def generate_feature_matrix(
    n_samples: int = 1000,
    n_features: int = 50,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate a feature matrix for testing feature selection and scaling.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        seed: Random seed

    Returns:
        pd.DataFrame: Feature matrix
    """
    np.random.seed(seed)

    # Mix of feature types
    features = {}

    # Normal features
    for i in range(n_features // 3):
        features[f'feature_{i}'] = np.random.randn(n_samples)

    # Skewed features
    for i in range(n_features // 3, 2 * n_features // 3):
        features[f'feature_{i}'] = np.random.exponential(1.0, n_samples)

    # Correlated features
    base = np.random.randn(n_samples)
    for i in range(2 * n_features // 3, n_features):
        noise = np.random.randn(n_samples) * 0.1
        features[f'feature_{i}'] = base + noise

    df = pd.DataFrame(features)
    df['datetime'] = pd.date_range('2024-01-01', periods=n_samples, freq='5min')

    return df
