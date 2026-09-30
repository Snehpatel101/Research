"""
Phase 116 regression tests: backtest realism.

1. No same-bar lookahead. A prediction at row i is built from bar i's close,
   so it can only fill at close[i] (MARKET_ON_CLOSE, the optimistic limit) or
   at bar i+1 or later (MARKET_ON_OPEN at open[i+1], MIDPOINT inside bar i+1).
   The canary: sign(close_i - open_i) is perfect knowledge of bar i's
   direction; filled at open[i] it made +$71k gross on a random walk.
2. Circuit breakers pause instead of ending the run: the equity curve covers
   every bar and halts are recorded in the result.
3. Label/backtest barrier parity including the transaction-cost term, with the
   cost in PRICE units (ticks * tick_size).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.utils.atr import wilder_atr
from src.data.labeling import TripleBarrierConfig, TripleBarrierLabeler
from src.data.labeling.triple_barrier import (
    barrier_distances,
    compute_cost_in_atr,
    expanding_cost_in_atr,
    transaction_cost_in_price,
)
from src.data.pipeline.config.barriers_config import get_total_trade_cost
from src.inference.backtesting import BacktestConfig, Backtester
from src.inference.backtesting.backtest import ExecutionModel, ExitReason, HaltReason
from src.inference.backtesting.execution import to_eastern

ALL_MODELS = [
    ExecutionModel.MARKET_ON_OPEN,
    ExecutionModel.MARKET_ON_CLOSE,
    ExecutionModel.FILL_AT_SIGNAL,
    ExecutionModel.MIDPOINT,
]

NO_BREAKERS = {
    "enable_market_hours_filter": False,
    "consecutive_loss_limit": 10**9,
    "max_drawdown_threshold": 10.0,
    "daily_loss_threshold": 10.0,
}


def _random_walk(n: int, seed: int = 0, gap_sd: float = 0.3) -> pd.DataFrame:
    """5-min random walk; open = previous close + an independent gap."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-02 14:30", periods=n, freq="5min", tz="UTC")
    close = 4500 + np.cumsum(rng.normal(0, 0.6, n))
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, gap_sd, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.3, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.3, n))
    return pd.DataFrame(
        {"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": 1000}
    )


def _predictions(prices: pd.DataFrame, signal: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {"timestamp": prices["timestamp"], "prediction": signal.astype(int), "confidence": 0.6}
    )


def _gross(result) -> float:
    return float(sum(t.gross_pnl for t in result.trades))


# ---------------------------------------------------------------------------
# (a) No same-bar lookahead under any execution model
# ---------------------------------------------------------------------------


N_WALK = 4000


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return _random_walk(N_WALK, seed=0)


class TestNoSameBarLookahead:

    @pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: m.value)
    def test_same_bar_direction_signal_has_no_edge(self, prices, model):
        """sign(close_i - open_i) is known only at close_i: no model may profit."""
        signal = np.sign(prices["close"] - prices["open"]).to_numpy()
        cfg = BacktestConfig(execution_model=model, **NO_BREAKERS)
        result = Backtester(_predictions(prices, signal), prices, cfg).run()

        # Edge a same-bar (open[i]) fill would harvest: |close - open| per bar
        oracle = float((prices["close"] - prices["open"]).abs().sum()) * cfg.point_value
        assert len(result.trades) > N_WALK // 4
        assert _gross(result) < 0.02 * oracle, (
            f"{model.value}: gross {_gross(result):.0f} vs same-bar oracle {oracle:.0f} "
            "— fills are using the signal bar's own price move"
        )

    def test_harness_detects_a_real_edge(self, prices):
        """Power check: a signal that peeks at the NEXT bar is caught as profitable."""
        nxt = np.sign(prices["close"].shift(-1) - prices["open"].shift(-1)).fillna(0).to_numpy()
        cfg = BacktestConfig(execution_model=ExecutionModel.MARKET_ON_OPEN, **NO_BREAKERS)
        result = Backtester(_predictions(prices, nxt), prices, cfg).run()
        oracle = float((prices["close"] - prices["open"]).abs().sum()) * cfg.point_value
        assert _gross(result) > 0.3 * oracle

    @pytest.mark.parametrize(
        ("model", "expected_bar", "column"),
        [
            (ExecutionModel.MARKET_ON_OPEN, 11, "open"),
            (ExecutionModel.MARKET_ON_CLOSE, 10, "close"),
            (ExecutionModel.FILL_AT_SIGNAL, 10, "close"),
            (ExecutionModel.MIDPOINT, 11, "mid"),
        ],
        ids=lambda v: getattr(v, "value", v),
    )
    def test_fill_bar_and_price(self, model, expected_bar, column):
        prices = _random_walk(40, seed=1)
        signal = np.zeros(40)
        signal[10] = 1
        bt = Backtester(_predictions(prices, signal), prices, BacktestConfig(execution_model=model))
        bt.market_hours_filter.enable_adverse_selection = False
        bt.config.enable_market_hours_filter = False
        bt.market_hours_filter.enable_market_hours_filter = False
        trade = bt.run().trades[0]

        row = prices.iloc[expected_bar]
        price = (row["high"] + row["low"]) / 2 if column == "mid" else row[column]
        assert trade.entry_time == row["timestamp"]
        assert trade.entry_price == pytest.approx(price)

    @pytest.mark.parametrize(
        "model", [ExecutionModel.MARKET_ON_OPEN, ExecutionModel.MIDPOINT], ids=lambda m: m.value
    )
    def test_zero_delay_rejected_for_open_and_midpoint(self, model):
        with pytest.raises(ValueError, match="signal_delay_bars"):
            BacktestConfig(execution_model=model, signal_delay_bars=0)

    def test_explicit_extra_delay(self):
        prices = _random_walk(40, seed=1)
        signal = np.zeros(40)
        signal[10] = 1
        cfg = BacktestConfig(signal_delay_bars=3, **NO_BREAKERS)
        trade = Backtester(_predictions(prices, signal), prices, cfg).run().trades[0]
        assert trade.entry_time == prices["timestamp"].iloc[13]

    def test_barriers_ignore_the_signal_bar_range(self):
        """A stop touched on the signal bar (before the fill) must not trigger."""
        n = 40
        bars = [(100.0, 100.5, 99.5, 100.0)] * n  # ATR -> 1.0
        bars[25] = (100.0, 100.5, 90.0, 100.0)  # signal bar: deep low
        prices = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-02 14:30", periods=n, freq="5min", tz="UTC"),
                "open": [b[0] for b in bars],
                "high": [b[1] for b in bars],
                "low": [b[2] for b in bars],
                "close": [b[3] for b in bars],
            }
        )
        signal = np.zeros(n)
        signal[25:30] = 1
        cfg = BacktestConfig(
            barrier_k_up=2.0, barrier_k_down=2.0, barrier_cost_in_atr=0.0, **NO_BREAKERS
        )
        trades = Backtester(_predictions(prices, signal), prices, cfg).run().trades
        assert trades[0].entry_time == prices["timestamp"].iloc[26]
        assert trades[0].exit_reason != ExitReason.STOP_LOSS.value


# ---------------------------------------------------------------------------
# (b) Circuit breakers pause; the simulation covers the whole period
# ---------------------------------------------------------------------------


class TestScopedCircuitBreakers:
    def test_random_predictions_simulate_full_period(self):
        n = 30_000
        prices = _random_walk(n, seed=0, gap_sd=0.0)
        rng = np.random.default_rng(0)
        preds = _predictions(prices, rng.choice([-1, 0, 1], n))
        result = Backtester(preds, prices, BacktestConfig(enable_market_hours_filter=False)).run()

        curve = result.equity_curve
        assert len(curve.equity_values) == n
        assert curve.timestamps[-1] == prices["timestamp"].iloc[-1]
        assert len(result.trades) > 200  # was 10: the first breaker ended the run
        # Breakers did trip — and trading resumed afterwards
        assert result.stats["n_halts"] == len(result.halts) > 1
        assert result.stats["halted_at"] == str(result.halts[0].timestamp)
        assert all(isinstance(h.reason, HaltReason) for h in result.halts)
        resumed = [h for h in result.halts if h.resume_bar is not None]
        assert resumed and all(h.resume_bar > h.bar for h in resumed)
        last_exit = max(t.exit_time for t in result.trades)
        assert last_exit > prices["timestamp"].iloc[n // 2]

    def test_equity_flat_while_halted(self):
        n = 3000
        prices = _random_walk(n, seed=3, gap_sd=0.0)
        rng = np.random.default_rng(3)
        preds = _predictions(prices, rng.choice([-1, 1], n))
        cfg = BacktestConfig(enable_market_hours_filter=False, consecutive_loss_limit=3)
        result = Backtester(preds, prices, cfg).run()
        equity = np.asarray(result.equity_curve.equity_values)
        halts = [h for h in result.halts if h.resume_bar is not None]
        assert halts
        for h in halts:
            # Flattened at the next open (MARKET_ON_OPEN), then flat until resume
            window = equity[h.bar + 1 : h.resume_bar]
            assert np.ptp(window) == pytest.approx(0.0, abs=1e-9)

    def test_daily_loss_resumes_next_day(self):
        n = 3000
        prices = _random_walk(n, seed=5, gap_sd=0.0)
        rng = np.random.default_rng(5)
        preds = _predictions(prices, rng.choice([-1, 1], n))
        cfg = BacktestConfig(
            enable_market_hours_filter=False,
            daily_loss_threshold=0.0005,
            consecutive_loss_limit=10**9,
            max_drawdown_threshold=10.0,
        )
        result = Backtester(preds, prices, cfg).run()
        daily = [h for h in result.halts if h.reason == HaltReason.DAILY_LOSS]
        assert daily
        et_date = to_eastern(pd.DatetimeIndex(prices["timestamp"])).date
        for h in daily:
            if h.resume_bar is not None:
                assert et_date[h.resume_bar] != et_date[h.bar]
                assert et_date[h.resume_bar - 1] == et_date[h.bar]

    def test_drawdown_cooloff_bars(self):
        n = 3000
        prices = _random_walk(n, seed=7, gap_sd=0.0)
        rng = np.random.default_rng(7)
        preds = _predictions(prices, rng.choice([-1, 1], n))
        cfg = BacktestConfig(
            enable_market_hours_filter=False,
            max_drawdown_threshold=0.002,
            drawdown_cooloff_bars=50,
            daily_loss_threshold=10.0,
            consecutive_loss_limit=10**9,
        )
        result = Backtester(preds, prices, cfg).run()
        dd = [h for h in result.halts if h.reason == HaltReason.MAX_DRAWDOWN]
        assert len(dd) > 1
        assert all(h.resume_bar == h.bar + 50 for h in dd if h.resume_bar is not None)

    def test_permanent_drawdown_halt_still_covers_period(self):
        n = 3000
        prices = _random_walk(n, seed=7, gap_sd=0.0)
        rng = np.random.default_rng(7)
        preds = _predictions(prices, rng.choice([-1, 1], n))
        cfg = BacktestConfig(
            enable_market_hours_filter=False,
            max_drawdown_threshold=0.002,
            drawdown_halt_permanent=True,
            daily_loss_threshold=10.0,
            consecutive_loss_limit=10**9,
        )
        bt = Backtester(preds, prices, cfg)
        result = bt.run()
        assert len(result.halts) == 1 and result.halts[0].resume_bar is None
        assert bt._halt_trading is True
        assert len(result.equity_curve.equity_values) == n
        halt_ts = prices["timestamp"].iloc[result.halts[0].bar]
        # Only the flatten of the open position may happen after the halt
        assert all(t.entry_time <= halt_ts for t in result.trades)
        tail = np.asarray(result.equity_curve.equity_values)[result.halts[0].bar + 1 :]
        assert np.ptp(tail) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# (c) Label / backtest barrier parity including the cost term
# ---------------------------------------------------------------------------


K_UP, K_DOWN, MAX_BARS = 1.5, 1.0, 12


@pytest.fixture(scope="module")
def labeled_walk() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
    prices = _random_walk(600, seed=11, gap_sd=0.0)
    labeler = TripleBarrierLabeler(
        TripleBarrierConfig(
            upper_mult=K_UP,
            lower_mult=K_DOWN,
            horizon=MAX_BARS,
            atr_column=None,
            symbol="MES",
        )
    )
    result = labeler.compute_labels(prices, horizon=MAX_BARS)
    cost_in_atr = float(result.metadata["cost_in_atr"][0])
    atr = wilder_atr(prices["high"], prices["low"], prices["close"], 14)
    return prices, result.labels, atr, cost_in_atr


def _parity_backtester(prices, signal, cost_in_atr, model=ExecutionModel.MARKET_ON_CLOSE):
    cfg = BacktestConfig.for_mes(
        execution_model=model,
        barrier_k_up=K_UP,
        barrier_k_down=K_DOWN,
        barrier_cost_in_atr=cost_in_atr,
        max_holding_period=MAX_BARS,
        min_holding_period=1,
        **NO_BREAKERS,
    )
    bt = Backtester(_predictions(prices, signal), prices, cfg)
    bt.market_hours_filter.enable_adverse_selection = False
    return bt


class TestBarrierParity:
    def test_label_cost_term_is_price_units_over_median_atr(self, labeled_walk):
        prices, _labels, atr, cost_in_atr = labeled_walk
        expected = get_total_trade_cost("MES", "low_vol") * 0.25 / float(np.median(atr[atr > 0]))
        assert cost_in_atr == pytest.approx(expected, rel=1e-12)

    def test_backtest_derives_same_cost_term(self, labeled_walk):
        """barrier_cost_in_atr=None -> the labeler's helpers, causally (expanding median ATR)."""
        prices, _labels, _atr, cost_in_atr = labeled_walk
        bt = _parity_backtester(prices, np.zeros(len(prices)), None)
        result = bt.run()
        derived = result.stats["barrier_cost_in_atr"]
        assert derived > 0
        # Same helper; the final expanding median approximates the global one
        assert derived == pytest.approx(cost_in_atr, rel=0.02)

    @pytest.mark.parametrize("bar", [60, 150, 333])
    def test_stop_and_tp_equal_label_barriers(self, labeled_walk, bar):
        """Close fill at close[i] == the label's entry reference: identical levels."""
        prices, _labels, atr, cost_in_atr = labeled_walk
        signal = np.zeros(len(prices))
        signal[bar] = 1
        bt = _parity_backtester(prices, signal, cost_in_atr)
        seen = {}
        original = bt._open_position

        def spy(*args, **kwargs):
            original(*args, **kwargs)
            seen["pos"] = bt._current_position

        bt._open_position = spy  # type: ignore[method-assign]
        bt.run()
        pos = seen["pos"]
        entry = prices["close"].iloc[bar]
        up, down = barrier_distances(atr[bar], K_UP, K_DOWN, cost_in_atr)
        assert pos.entry_price == pytest.approx(entry)
        assert pos.take_profit == pytest.approx(entry + (K_UP + cost_in_atr) * atr[bar], rel=1e-9)
        assert pos.stop_loss == pytest.approx(entry - (K_DOWN + cost_in_atr) * atr[bar], rel=1e-9)
        assert pos.take_profit - entry == pytest.approx(up)
        assert entry - pos.stop_loss == pytest.approx(down)

    @pytest.mark.parametrize("bar", [30, 150, 333])
    def test_derived_cost_is_the_signal_bars_value(self, labeled_walk, bar):
        """barrier_cost_in_atr=None: each trade's barriers use the cost known at its signal bar."""
        prices, _labels, _atr, _cost = labeled_walk
        signal = np.zeros(len(prices))
        signal[bar] = 1
        bt = _parity_backtester(prices, signal, None)
        # The backtester's ATR: the canonical Wilder ATR the labeler uses
        atr = bt._atr(prices)
        np.testing.assert_array_equal(atr, _atr)
        per_bar = expanding_cost_in_atr(transaction_cost_in_price("MES"), atr)
        cost = per_bar[bar]
        # The final (fully calibrated) cost would place the barriers elsewhere
        assert cost > 0
        assert cost != pytest.approx(per_bar[-1], rel=1e-3)
        seen = {}
        original = bt._open_position

        def spy(*args, **kwargs):
            original(*args, **kwargs)
            seen["pos"] = bt._current_position

        bt._open_position = spy  # type: ignore[method-assign]
        bt.run()
        pos = seen["pos"]
        entry = prices["close"].iloc[bar]
        assert pos.entry_price == pytest.approx(entry)
        assert pos.take_profit == pytest.approx(entry + (K_UP + cost) * atr[bar], rel=1e-9)
        assert pos.stop_loss == pytest.approx(entry - (K_DOWN + cost) * atr[bar], rel=1e-9)

    def test_trade_outcome_matches_label(self, labeled_walk):
        """Long from bar i under the label's rules exits the way label[i] says."""
        prices, labels, atr, cost_in_atr = labeled_walk
        high, low = prices["high"].to_numpy(), prices["low"].to_numpy()
        close = prices["close"].to_numpy()
        expected_exit = {
            1: ExitReason.TAKE_PROFIT,
            -1: ExitReason.STOP_LOSS,
            0: ExitReason.MAX_HOLDING,
        }
        checked = 0
        for i in range(40, len(prices) - MAX_BARS - 1, 7):
            if labels[i] == -99:
                continue
            up, down = barrier_distances(atr[i], K_UP, K_DOWN, cost_in_atr)
            window = range(i + 1, i + MAX_BARS + 1)
            if any(high[j] >= close[i] + up and low[j] <= close[i] - down for j in window):
                continue  # both touched in one bar: label and backtest tie-break differently
            signal = np.zeros(len(prices))
            signal[i:] = 1  # hold: no signal exit, only barriers / time
            trade = _parity_backtester(prices, signal, cost_in_atr).run().trades[0]
            assert trade.exit_reason == expected_exit[int(labels[i])].value, i
            if labels[i] == 0:
                assert trade.exit_time == prices["timestamp"].iloc[i + MAX_BARS]
            checked += 1
        assert checked >= 30

    def test_open_fill_barriers_keep_label_distances(self, labeled_walk):
        """Default open fill: same distances, anchored at the fill (gap documented)."""
        prices, _labels, atr, cost_in_atr = labeled_walk
        bar = 150
        signal = np.zeros(len(prices))
        signal[bar] = -1
        bt = _parity_backtester(prices, signal, cost_in_atr, model=ExecutionModel.MARKET_ON_OPEN)
        seen = {}
        original = bt._open_position

        def spy(*args, **kwargs):
            original(*args, **kwargs)
            seen["pos"] = bt._current_position

        bt._open_position = spy  # type: ignore[method-assign]
        bt.run()
        pos = seen["pos"]
        up, down = barrier_distances(atr[bar], K_UP, K_DOWN, cost_in_atr)
        assert pos.entry_price == pytest.approx(prices["open"].iloc[bar + 1])
        assert pos.stop_loss - pos.entry_price == pytest.approx(up)
        assert pos.entry_price - pos.take_profit == pytest.approx(down)


# ---------------------------------------------------------------------------
# (d) Cost units: ticks * tick_size (price points), not ticks * tick_value ($)
# ---------------------------------------------------------------------------


class TestCostUnits:
    def test_mes_cost_in_price_points(self):
        ticks = get_total_trade_cost("MES", "low_vol")
        assert ticks == pytest.approx(2.43 + 2 * 1.0)
        assert transaction_cost_in_price("MES") == pytest.approx(ticks * 0.25)
        assert transaction_cost_in_price("MES") == pytest.approx(1.1075)

    def test_mgc_and_mnq_use_their_tick_size(self):
        assert transaction_cost_in_price("MGC") == pytest.approx(
            get_total_trade_cost("MGC", "low_vol") * 0.10
        )
        assert transaction_cost_in_price("MNQ") == pytest.approx(
            get_total_trade_cost("MNQ", "low_vol") * 0.25
        )

    def test_cost_in_atr_is_price_cost_over_median_atr(self):
        atr = np.array([np.nan, 0.0, 2.0, 4.0, 6.0])
        assert compute_cost_in_atr("MES", atr) == pytest.approx(1.1075 / 4.0)
        assert compute_cost_in_atr("MES", np.array([np.nan, 0.0])) == 0.0
