"""
Shared fixtures for data quality tests.
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def clean_ohlcv_df() -> pd.DataFrame:
    """Generate clean, valid OHLCV data."""
    np.random.seed(42)
    n_bars = 100

    base_price = 100
    returns = np.random.randn(n_bars) * 0.01

    close = base_price * np.cumprod(1 + returns)
    range_size = np.abs(np.random.randn(n_bars) * 0.5) + 0.1
    high = close + range_size
    low = close - range_size
    open_price = np.roll(close, 1)
    open_price[0] = base_price
    volume = np.random.randint(100, 10000, n_bars)

    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }, index=pd.date_range('2020-01-01', periods=n_bars, freq='5min'))


@pytest.fixture
def corrupted_ohlcv_df() -> pd.DataFrame:
    """Generate OHLCV data with various quality issues."""
    np.random.seed(42)
    n_bars = 100

    df = pd.DataFrame({
        'open': np.random.uniform(99, 101, n_bars),
        'high': np.random.uniform(100, 102, n_bars),
        'low': np.random.uniform(98, 100, n_bars),
        'close': np.random.uniform(99, 101, n_bars),
        'volume': np.random.randint(100, 10000, n_bars),
    }, index=pd.date_range('2020-01-01', periods=n_bars, freq='5min'))

    # Introduce various issues
    df.loc[df.index[10], 'high'] = 95  # high < low violation
    df.loc[df.index[10], 'low'] = 105

    df.loc[df.index[20], 'close'] = np.nan  # NaN value

    df.loc[df.index[30], 'volume'] = -100  # Negative volume

    df.loc[df.index[40], 'open'] = np.inf  # Infinity

    return df
