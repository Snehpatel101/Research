"""
D6: ATR parity between labeling and backtest subsystems.

The triple-barrier labeler and the backtester both use the one canonical
Wilder ATR (``src.core.utils.atr.wilder_atr``): same values, same warm-up.
These tests capture the ATR each subsystem actually uses and require exact
equality, pin the Wilder definition (SMA seed of the first ``period`` true
ranges, then ``(prev * (period - 1) + TR) / period``), and check that bars
without a valid ATR carry the invalid label (-99).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

import src.data.labeling.triple_barrier as tb
from src.core.utils.atr import true_range, wilder_atr
from src.data.labeling.triple_barrier import TripleBarrierConfig, TripleBarrierLabeler
from src.data.pipeline.stages.features.volatility import add_atr
from src.inference.backtesting.backtest import BacktestConfig, Backtester

INVALID = -99


def _make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """n bars of synthetic OHLCV with realistic high/low spread and timestamps."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    spread = rng.uniform(0.2, 1.0, size=n)
    return pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 2) + timedelta(minutes=5 * i) for i in range(n)],
            "open": close + rng.standard_normal(n) * 0.1,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": rng.integers(100, 10000, size=n),
        }
    )


def _labeler_atr(df: pd.DataFrame, period: int, monkeypatch: pytest.MonkeyPatch) -> np.ndarray:
    """The ATR array the labeler hands to its barrier kernel (inline ATR path)."""
    seen: dict[str, np.ndarray] = {}
    kernel = tb.triple_barrier_numba_with_costs

    def spy(
        close: np.ndarray, high: np.ndarray, low: np.ndarray, atr: np.ndarray, *args: Any
    ) -> Any:
        seen["atr"] = np.array(atr, copy=True)
        return kernel(close, high, low, atr, *args)

    monkeypatch.setattr(tb, "triple_barrier_numba_with_costs", spy)
    labeler = TripleBarrierLabeler(
        TripleBarrierConfig(horizon=10, atr_period=period, atr_column=None, symbol="MES")
    )
    labeler.compute_labels(df, horizon=10)
    return seen["atr"]


def _backtester_atr(df: pd.DataFrame, period: int) -> np.ndarray:
    """The ATR column the backtester computes over the full price series."""
    preds = pd.DataFrame(
        {"timestamp": df["timestamp"], "prediction": 0, "confidence": 1.0},
    )
    config = BacktestConfig(
        barrier_k_up=1.0,
        barrier_k_down=1.0,
        atr_period=period,
        enable_market_hours_filter=False,
    )
    bt = Backtester(predictions=preds, prices=df, config=config)
    return bt._align_data()["atr"].to_numpy(dtype=float)


def _manual_wilder(df: pd.DataFrame, period: int) -> np.ndarray:
    high, low, close = (df[c].to_numpy(dtype=float) for c in ("high", "low", "close"))
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )  # tr[k] is bar k + 1's true range
    atr = np.full(len(df), np.nan)
    atr[period] = tr[:period].mean()
    for i in range(period + 1, len(df)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period
    return atr


class TestATRParity:
    """Labeler and backtester use the same ATR, exactly."""

    @pytest.mark.parametrize(("n", "period", "seed"), [(100, 14, 42), (200, 20, 99), (1000, 14, 7)])
    def test_labeler_atr_equals_backtester_atr(
        self, n: int, period: int, seed: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        df = _make_ohlcv(n, seed)
        label_atr = _labeler_atr(df, period, monkeypatch)
        bt_atr = _backtester_atr(df, period)
        np.testing.assert_array_equal(label_atr, bt_atr)
        np.testing.assert_array_equal(label_atr, wilder_atr(df.high, df.low, df.close, period))

    def test_float32_bars_give_the_same_atr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The factory downcasts frames to float32; the ATR is computed in float64 either way."""
        df = _make_ohlcv(300, seed=5)
        cols = ["open", "high", "low", "close"]
        df32 = df.astype(dict.fromkeys(cols, np.float32))
        np.testing.assert_array_equal(
            _labeler_atr(df32, 14, monkeypatch), _backtester_atr(df32, 14)
        )

    def test_feature_atr_is_the_lagged_canonical_atr(self) -> None:
        """atr_{period} features route through the same function (shifted one bar)."""
        df = _make_ohlcv(120, seed=3)
        out = add_atr(df.copy(), {}, periods=[14])
        expected = np.r_[np.nan, wilder_atr(df.high, df.low, df.close, 14)[:-1]]
        np.testing.assert_array_equal(out["atr_14"].to_numpy(dtype=float), expected)


class TestWilderDefinition:
    """wilder_atr is textbook Wilder (TA-Lib ATR)."""

    @pytest.mark.parametrize("period", [1, 5, 14, 20])
    def test_matches_manual_wilder(self, period: int) -> None:
        df = _make_ohlcv(150, seed=period)
        np.testing.assert_allclose(
            wilder_atr(df.high, df.low, df.close, period),
            _manual_wilder(df, period),
            rtol=1e-12,
            equal_nan=True,
        )

    def test_warmup_is_nan_until_period_true_ranges_exist(self) -> None:
        period = 14
        atr = wilder_atr(*(_make_ohlcv(60)[c] for c in ("high", "low", "close")), period)
        assert np.isnan(atr[:period]).all()
        assert np.isfinite(atr[period:]).all()

    def test_too_short_series_is_all_nan(self) -> None:
        df = _make_ohlcv(14)
        assert np.isnan(wilder_atr(df.high, df.low, df.close, 14)).all()

    def test_constant_bars_give_constant_atr(self) -> None:
        n = 100
        atr = wilder_atr(np.full(n, 102.0), np.full(n, 98.0), np.full(n, 100.0), 14)
        np.testing.assert_allclose(atr[14:], 4.0, rtol=0, atol=1e-12)

    def test_causal(self) -> None:
        """Appending bars never changes earlier ATR values."""
        df = _make_ohlcv(200, seed=11)
        full = wilder_atr(df.high, df.low, df.close, 14)
        head = df.iloc[:120]
        np.testing.assert_array_equal(wilder_atr(head.high, head.low, head.close, 14), full[:120])

    def test_rejects_bad_period(self) -> None:
        with pytest.raises(ValueError, match="period"):
            wilder_atr([1.0], [1.0], [1.0], 0)

    def test_true_range_first_bar_is_high_minus_low(self) -> None:
        tr = true_range([10.0, 12.0], [8.0, 11.5], [9.0, 11.0])
        np.testing.assert_array_equal(tr, [2.0, 3.0])


class TestLabelWarmup:
    """Bars without a valid ATR are never labeled."""

    @pytest.mark.parametrize("period", [14, 20])
    def test_first_period_bars_are_invalid(self, period: int) -> None:
        df = _make_ohlcv(300, seed=17)
        result = TripleBarrierLabeler(
            TripleBarrierConfig(horizon=10, atr_period=period, atr_column=None, symbol="MES")
        ).compute_labels(df, horizon=10)
        # Covers the first period - 1 bars the backtester had no ATR for, and bar period - 1
        assert (result.labels[:period] == INVALID).all()
        # From the first valid ATR on, bars away from the end are labeled
        assert (result.labels[period:-10] != INVALID).all()
