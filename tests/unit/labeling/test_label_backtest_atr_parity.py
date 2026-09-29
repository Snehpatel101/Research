"""
D6: ATR parity between labeling and backtest subsystems.

Both the triple-barrier labeler and the backtester compute ATR using
Wilder's EMA (alpha = 1/period). This test verifies they produce
identical results on the same data, catching C-BT-1 (ATR divergence).

Implementation differences (by design):
- Labeling: np.roll for prev_close, manual EMA loop, atr[0]=TR[0]
- Backtest: pd.shift for prev_close, pd.ewm(min_periods=period), NaN warmup

After the warmup window (period bars), both converge because Wilder's
EMA is memoryless — the initial seed washes out exponentially.
"""

import numpy as np
import pandas as pd

from src.data.labeling.triple_barrier import TripleBarrierConfig, TripleBarrierLabeler
from src.inference.backtesting.backtest import Backtester

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create n bars of synthetic OHLCV with realistic high/low spread."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    spread = rng.uniform(0.2, 1.0, size=n)
    high = close + spread
    low = close - spread
    open_ = close + rng.standard_normal(n) * 0.1
    volume = rng.integers(100, 10000, size=n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _labeling_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Compute ATR via the labeling path."""
    config = TripleBarrierConfig(atr_period=period)
    labeler = TripleBarrierLabeler(config)
    return labeler.compute_atr(df)


def _backtest_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ATR via the backtest path (unbound — no self usage)."""
    return Backtester._compute_atr(None, df, period)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestATRParity:
    """Verify ATR computation is identical between labeling and backtest."""

    def test_atr_converges_after_warmup(self) -> None:
        """After `period` bars, both ATR implementations must match."""
        df = _make_ohlcv(100)
        period = 14

        label_atr = _labeling_atr(df, period)
        bt_atr = _backtest_atr(df, period)

        # After warmup, values must be nearly identical.
        # The backtest ewm with min_periods=period produces its first
        # non-NaN value at index period-1. The labeling path always has
        # a value. We compare from index `period` onward where both are
        # fully warmed up and any seed difference has decayed.
        start = period  # safe: both have valid values here
        np.testing.assert_allclose(
            label_atr[start:],
            bt_atr.values[start:],
            atol=1e-10,
            err_msg="ATR divergence between labeling and backtest after warmup",
        )

    def test_atr_parity_period_20(self) -> None:
        """Parity holds with a non-default period."""
        df = _make_ohlcv(200, seed=99)
        period = 20

        label_atr = _labeling_atr(df, period)
        bt_atr = _backtest_atr(df, period)

        start = period
        np.testing.assert_allclose(
            label_atr[start:],
            bt_atr.values[start:],
            atol=1e-10,
            err_msg="ATR divergence with period=20",
        )

    def test_atr_parity_large_dataset(self) -> None:
        """Parity holds on a larger dataset (1000 bars)."""
        df = _make_ohlcv(1000, seed=7)
        period = 14

        label_atr = _labeling_atr(df, period)
        bt_atr = _backtest_atr(df, period)

        start = period
        np.testing.assert_allclose(
            label_atr[start:],
            bt_atr.values[start:],
            atol=1e-10,
            err_msg="ATR divergence on 1000-bar dataset",
        )

    def test_both_use_wilders_ema(self) -> None:
        """Verify both implementations use alpha=1/period (Wilder's EMA)."""
        df = _make_ohlcv(50, seed=123)
        period = 10

        label_atr = _labeling_atr(df, period)
        bt_atr = _backtest_atr(df, period)

        # Manually compute true range the same way the labeling path does
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(np.maximum(high - low, np.abs(high - prev_close)), np.abs(low - prev_close))

        # Manual Wilder's EMA
        alpha = 1.0 / period
        manual_atr = np.zeros_like(tr)
        manual_atr[0] = tr[0]
        for i in range(1, len(tr)):
            manual_atr[i] = alpha * tr[i] + (1 - alpha) * manual_atr[i - 1]

        # Labeling path must match manual exactly
        np.testing.assert_allclose(
            label_atr,
            manual_atr,
            atol=1e-12,
            err_msg="Labeling ATR does not match manual Wilder's EMA",
        )

        # Backtest path must match after warmup
        start = period
        np.testing.assert_allclose(
            bt_atr.values[start:],
            manual_atr[start:],
            atol=1e-10,
            err_msg="Backtest ATR does not match manual Wilder's EMA after warmup",
        )

    def test_constant_bars_produce_constant_atr(self) -> None:
        """When all bars are identical, ATR should converge to high-low."""
        n = 100
        period = 14
        df = pd.DataFrame(
            {
                "open": np.full(n, 100.0),
                "high": np.full(n, 102.0),
                "low": np.full(n, 98.0),
                "close": np.full(n, 100.0),
                "volume": np.full(n, 1000),
            }
        )

        label_atr = _labeling_atr(df, period)
        bt_atr = _backtest_atr(df, period)

        # True range for constant bars = high - low = 4.0 for every bar.
        # Wilder's EMA of a constant = that constant.
        expected = 4.0

        # Labeling: exact after initial seed (which is also 4.0 here)
        np.testing.assert_allclose(label_atr[-1], expected, atol=1e-12)

        # Backtest: after warmup, should also be 4.0
        np.testing.assert_allclose(bt_atr.values[-1], expected, atol=1e-12)

        # Both must agree after warmup
        start = period
        np.testing.assert_allclose(
            label_atr[start:],
            bt_atr.values[start:],
            atol=1e-10,
        )

    def test_shapes_match(self) -> None:
        """Both ATR outputs have the same length as input."""
        df = _make_ohlcv(100)
        label_atr = _labeling_atr(df)
        bt_atr = _backtest_atr(df)

        assert len(label_atr) == len(df)
        assert len(bt_atr) == len(df)
