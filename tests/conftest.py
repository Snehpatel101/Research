"""
Pytest configuration and shared fixtures.

Layout: ``tests/unit/<area>/`` (fast, isolated, mirrors ``src/``), ``tests/integration/``
(components wired together) and ``tests/e2e/`` (full pipeline runs). The directory sets the
``unit`` / ``integration`` / ``e2e`` marker automatically; tests over 30 s carry ``slow``.
``tests/property/`` (hypothesis property-based tests of leakage / parity invariants) counts as
``unit``. Hypothesis settings profile: ``HYPOTHESIS_PROFILE`` = ``dev`` (default) or ``ci``.
Shared plain helpers live in ``tests/helpers.py``.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, settings

# Directory under tests/ -> layer marker
_LAYERS = {"unit": "unit", "property": "unit", "integration": "integration", "e2e": "e2e"}

# ci: reproducible (derandomized) and bounded; no deadline because the first example
# pays numba JIT compilation. dev: hypothesis defaults (random exploration, 100 examples).
settings.register_profile(
    "ci",
    derandomize=True,
    max_examples=25,
    deadline=None,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
settings.register_profile("dev", settings.get_profile("default"))
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark each test with its layer (first directory under tests/)."""
    tests_dir = Path(__file__).parent
    for item in items:
        path = Path(str(item.path))
        if not path.is_relative_to(tests_dir):
            continue
        parts = path.relative_to(tests_dir).parts
        if parts[0] in _LAYERS:
            item.add_marker(getattr(pytest.mark, _LAYERS[parts[0]]))


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """
    Create sample OHLCV price data for testing.

    Returns:
        DataFrame with 100 bars of synthetic price data.
    """
    rng = np.random.default_rng(42)
    n_bars = 100

    # Start price and random walk
    start_price = 4500.0
    returns = rng.normal(0, 0.002, n_bars)
    prices = start_price * np.cumprod(1 + returns)

    # Generate OHLCV
    timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n_bars)]

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices * (1 + rng.uniform(-0.001, 0.001, n_bars)),
            "high": prices * (1 + rng.uniform(0, 0.003, n_bars)),
            "low": prices * (1 - rng.uniform(0, 0.003, n_bars)),
            "close": prices,
            "volume": rng.integers(1000, 5000, n_bars),
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
    rng = np.random.default_rng(42)
    n = len(sample_prices)

    # Generate random signals with some structure
    predictions = rng.choice([-1, 0, 1], size=n, p=[0.2, 0.6, 0.2])

    df = pd.DataFrame(
        {
            "timestamp": sample_prices["timestamp"],
            "prediction": predictions,
            "confidence": rng.uniform(0.5, 1.0, n),
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
