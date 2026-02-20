"""
Tests for triple-barrier labeling.

Verifies label correctness for strong trends, flat markets, single-bar
barrier hits, invalid-label sentinel for the last ``horizon`` bars, and
config validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.labeling.triple_barrier import (
    TripleBarrierConfig,
    TripleBarrierLabeler,
    _triple_barrier_python,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _make_ohlcv(
    close: np.ndarray,
    spread: float = 0.001,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Build an OHLCV DataFrame from a close array with synthetic high/low."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(close)
    high = close * (1 + rng.uniform(0, spread, n))
    low = close * (1 - rng.uniform(0, spread, n))
    open_ = close * (1 + rng.uniform(-spread / 2, spread / 2, n))
    volume = rng.integers(1000, 5000, n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="5min"),
    )


def _compute_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Simple EMA-based ATR computation."""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    alpha = 2.0 / (period + 1)
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
    return atr


# ---------------------------------------------------------------------------
# Core label correctness
# ---------------------------------------------------------------------------


class TestUptrend:
    """Strong uptrend should produce mostly long (+1) labels."""

    def test_majority_long(self, rng: np.random.Generator) -> None:
        n = 500
        # Strongly trending up: ~0.5% per bar
        close = 4500.0 * np.cumprod(1 + np.full(n, 0.005))
        df = _make_ohlcv(close, spread=0.0005, rng=rng)
        atr = _compute_atr(df)
        labels, _, _, _, _ = _triple_barrier_python(
            close=df["close"].values,
            high=df["high"].values,
            low=df["low"].values,
            atr=atr,
            k_up=1.5,
            k_down=1.5,
            max_bars=20,
        )
        valid = labels[labels != -99]
        long_frac = (valid == 1).sum() / len(valid)
        assert long_frac > 0.5, f"Expected majority long labels, got {long_frac:.2%}"


class TestDowntrend:
    """Strong downtrend should produce mostly short (-1) labels."""

    def test_majority_short(self, rng: np.random.Generator) -> None:
        n = 500
        close = 4500.0 * np.cumprod(1 + np.full(n, -0.005))
        df = _make_ohlcv(close, spread=0.0005, rng=rng)
        atr = _compute_atr(df)
        labels, _, _, _, _ = _triple_barrier_python(
            close=df["close"].values,
            high=df["high"].values,
            low=df["low"].values,
            atr=atr,
            k_up=1.5,
            k_down=1.5,
            max_bars=20,
        )
        valid = labels[labels != -99]
        short_frac = (valid == -1).sum() / len(valid)
        assert short_frac > 0.5, f"Expected majority short labels, got {short_frac:.2%}"


class TestFlatMarket:
    """Flat market with narrow barriers should produce mostly timeout (0)."""

    def test_majority_timeout(self, rng: np.random.Generator) -> None:
        n = 500
        # Very flat: tiny noise
        close = 4500.0 + rng.normal(0, 0.01, n).cumsum()
        df = _make_ohlcv(close, spread=0.0001, rng=rng)
        atr = _compute_atr(df)
        labels, _, _, _, _ = _triple_barrier_python(
            close=df["close"].values,
            high=df["high"].values,
            low=df["low"].values,
            atr=atr,
            k_up=5.0,  # very wide barriers
            k_down=5.0,
            max_bars=5,  # short horizon -> timeout
        )
        valid = labels[labels != -99]
        timeout_frac = (valid == 0).sum() / len(valid)
        assert timeout_frac > 0.4, f"Expected many timeouts, got {timeout_frac:.2%}"


class TestSingleBarHit:
    """If the next bar's high exceeds upper barrier, label should be +1
    with bars_to_hit == 1."""

    def test_known_single_bar(self) -> None:
        # Construct a minimal scenario: entry at bar 0, bar 1 high hits upper.
        n = 50
        close = np.full(n, 100.0)
        high = np.full(n, 100.1)
        low = np.full(n, 99.9)
        atr = np.full(n, 1.0)

        # Bar 1: high spikes far above upper barrier
        high[1] = 105.0  # 100 + 1.0 * 2.0 = 102 -> 105 clearly exceeds

        labels, bars_to_hit, _, _, _ = _triple_barrier_python(
            close=close, high=high, low=low, atr=atr, k_up=2.0, k_down=2.0, max_bars=10
        )
        assert labels[0] == 1
        assert bars_to_hit[0] == 1


class TestInvalidLastBars:
    """The last ``max_bars`` samples must be marked invalid (-99)."""

    def test_sentinel_at_end(self, rng: np.random.Generator) -> None:
        n = 200
        max_bars = 20
        close = 4500.0 * np.cumprod(1 + rng.normal(0, 0.002, n))
        df = _make_ohlcv(close, rng=rng)
        atr = _compute_atr(df)
        labels, _, _, _, _ = _triple_barrier_python(
            close=df["close"].values,
            high=df["high"].values,
            low=df["low"].values,
            atr=atr,
            k_up=2.0,
            k_down=2.0,
            max_bars=max_bars,
        )
        tail = labels[-max_bars:]
        assert np.all(tail == -99), f"Expected all -99 in last {max_bars} bars, got {tail}"


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestTripleBarrierConfigValidation:
    """Invalid config parameters must raise ValueError."""

    def test_upper_mult_zero(self) -> None:
        with pytest.raises(ValueError, match="upper_mult"):
            TripleBarrierConfig(upper_mult=0)

    def test_lower_mult_negative(self) -> None:
        with pytest.raises(ValueError, match="lower_mult"):
            TripleBarrierConfig(lower_mult=-1.0)

    def test_horizon_zero(self) -> None:
        with pytest.raises(ValueError, match="horizon"):
            TripleBarrierConfig(horizon=0)

    def test_atr_period_zero(self) -> None:
        with pytest.raises(ValueError, match="atr_period"):
            TripleBarrierConfig(atr_period=0)

    def test_vol_lookback_zero(self) -> None:
        with pytest.raises(ValueError, match="vol_lookback"):
            TripleBarrierConfig(vol_lookback=0)


class TestLabelerAPI:
    """TripleBarrierLabeler high-level API works correctly."""

    def test_create_labels_returns_series(self, rng: np.random.Generator) -> None:
        n = 300
        close = 4500.0 * np.cumprod(1 + rng.normal(0, 0.002, n))
        df = _make_ohlcv(close, rng=rng)
        config = TripleBarrierConfig(
            upper_mult=2.0,
            lower_mult=2.0,
            horizon=20,
            apply_transaction_costs=False,
            atr_column=None,
        )
        labeler = TripleBarrierLabeler(config)
        labels = labeler.create_labels(df)
        assert isinstance(labels, pd.Series)
        assert len(labels) == len(df)
        # All values should be in {-99, -1, 0, 1}
        unique = set(labels.unique())
        assert unique.issubset({-99, -1, 0, 1})
