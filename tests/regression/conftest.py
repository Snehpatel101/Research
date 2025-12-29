"""
Shared fixtures for regression tests.
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlcv_df() -> pd.DataFrame:
    """Generate synthetic OHLCV data for regression tests."""
    np.random.seed(42)
    n_bars = 500

    base_price = 100
    returns = np.random.randn(n_bars) * 0.02

    close = base_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.randn(n_bars) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n_bars) * 0.01))
    open_price = np.roll(close, 1)
    open_price[0] = base_price
    volume = np.random.randint(100, 10000, n_bars)

    # Ensure OHLCV consistency
    high = np.maximum(high, np.maximum(close, open_price))
    low = np.minimum(low, np.minimum(close, open_price))

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'atr_14': np.abs(np.random.randn(n_bars)) + 0.5,
    }, index=pd.date_range('2020-01-01', periods=n_bars, freq='5min'))

    return df


@pytest.fixture
def small_regression_data() -> dict:
    """Small dataset for fast regression tests."""
    np.random.seed(42)
    n_samples = 100

    return {
        'X_train': np.random.randn(n_samples, 10).astype(np.float32),
        'y_train': np.random.choice([-1, 0, 1], n_samples),
        'X_val': np.random.randn(20, 10).astype(np.float32),
        'y_val': np.random.choice([-1, 0, 1], 20),
    }
