"""
Pytest configuration and shared fixtures.

Provides common test fixtures for backtesting tests.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """
    Create sample OHLCV price data for testing.

    Returns:
        DataFrame with 100 bars of synthetic price data.
    """
    np.random.seed(42)
    n_bars = 100

    # Start price and random walk
    start_price = 4500.0
    returns = np.random.normal(0, 0.002, n_bars)
    prices = start_price * np.cumprod(1 + returns)

    # Generate OHLCV
    timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n_bars)]

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices * (1 + np.random.uniform(-0.001, 0.001, n_bars)),
            "high": prices * (1 + np.random.uniform(0, 0.003, n_bars)),
            "low": prices * (1 - np.random.uniform(0, 0.003, n_bars)),
            "close": prices,
            "volume": np.random.randint(1000, 5000, n_bars),
        }
    )

    return df


@pytest.fixture
def sample_predictions(sample_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Create sample predictions aligned with price data.

    Args:
        sample_prices: Price data fixture.

    Returns:
        DataFrame with predictions (-1, 0, 1).
    """
    np.random.seed(42)
    n = len(sample_prices)

    # Generate random signals with some structure
    predictions = np.random.choice([-1, 0, 1], size=n, p=[0.2, 0.6, 0.2])

    df = pd.DataFrame(
        {
            "timestamp": sample_prices["timestamp"],
            "prediction": predictions,
            "confidence": np.random.uniform(0.5, 1.0, n),
        }
    )

    return df


@pytest.fixture
def winning_trade_predictions(sample_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Create predictions that should result in a winning trade.

    Uses simple momentum: long when price went up, short when down.
    """
    n = len(sample_prices)
    prices = sample_prices["close"].values

    # Simple momentum signals
    predictions = np.zeros(n, dtype=int)
    for i in range(1, n - 1):
        if prices[i] > prices[i - 1]:
            predictions[i] = 1  # Long
        elif prices[i] < prices[i - 1]:
            predictions[i] = -1  # Short

    df = pd.DataFrame(
        {
            "timestamp": sample_prices["timestamp"],
            "prediction": predictions,
            "confidence": np.ones(n),
        }
    )

    return df


@pytest.fixture
def losing_trade_predictions(sample_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Create predictions that should result in losing trades.

    Inverse momentum: opposite of price direction.
    """
    n = len(sample_prices)
    prices = sample_prices["close"].values

    # Inverse momentum signals
    predictions = np.zeros(n, dtype=int)
    for i in range(1, n - 1):
        if prices[i] > prices[i - 1]:
            predictions[i] = -1  # Short when price going up
        elif prices[i] < prices[i - 1]:
            predictions[i] = 1  # Long when price going down

    df = pd.DataFrame(
        {
            "timestamp": sample_prices["timestamp"],
            "prediction": predictions,
            "confidence": np.ones(n),
        }
    )

    return df
