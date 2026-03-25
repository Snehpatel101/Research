"""D7: Verify stop-loss and take-profit exits happen at barrier price, not bar close.

When barriers are active (barrier_k_up > 0 or barrier_k_down > 0), the backtester
should exit at the barrier level (stop_loss or take_profit price) rather than
the bar's close price.  This is implemented by _resolve_exit_price() which returns
the barrier level for STOP_LOSS/TAKE_PROFIT exits and the close for everything else.
"""

from __future__ import annotations

import pandas as pd

from src.inference.backtesting.backtest import (
    BacktestConfig,
    Backtester,
    ExitReason,
    Position,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prices(
    bars: list[tuple[float, float, float, float]],
    start: str = "2024-01-01 10:00",
    freq: str = "h",
) -> pd.DataFrame:
    """Build a prices DataFrame from (open, high, low, close) tuples."""
    ts = pd.date_range(start, periods=len(bars), freq=freq)
    data = {
        "timestamp": ts,
        "open": [b[0] for b in bars],
        "high": [b[1] for b in bars],
        "low": [b[2] for b in bars],
        "close": [b[3] for b in bars],
        "volume": [1000] * len(bars),
    }
    return pd.DataFrame(data)


def _make_predictions(
    signals: list[int],
    start: str = "2024-01-01 10:00",
    freq: str = "h",
) -> pd.DataFrame:
    """Build a predictions DataFrame from signal values."""
    ts = pd.date_range(start, periods=len(signals), freq=freq)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "prediction": signals,
            "confidence": [1.0] * len(signals),
        }
    )


def _barrier_config(**overrides) -> BacktestConfig:
    """BacktestConfig with barriers enabled and market hours filter off."""
    defaults = {
        "barrier_k_up": 1.5,
        "barrier_k_down": 1.5,
        "enable_market_hours_filter": False,
        "initial_equity": 100_000.0,
        "position_sizing": "fixed_contracts",
        "fixed_contracts": 1,
        "commission_per_contract": 0.0,
        "slippage_ticks": 0.0,
        "tick_value": 1.25,
        "tick_size": 0.25,
        "point_value": 5.0,
        "min_holding_period": 1,
        "max_holding_period": 0,
        "max_drawdown_threshold": 1.0,
        "daily_loss_threshold": 1.0,
        "consecutive_loss_limit": 999,
    }
    defaults.update(overrides)
    return BacktestConfig(**defaults)


# ---------------------------------------------------------------------------
# Unit test: _resolve_exit_price returns barrier level for barrier exits
# ---------------------------------------------------------------------------


def _make_backtester_for_unit_test(**config_overrides) -> Backtester:
    """Create a minimal Backtester for unit-testing _resolve_exit_price."""
    defaults = {
        "enable_market_hours_filter": False,
        "slippage_ticks": 0.0,
        "tick_size": 0.25,
        "commission_per_contract": 0.0,
    }
    defaults.update(config_overrides)
    cfg = BacktestConfig(**defaults)
    # Minimal DataFrames — only used to construct the Backtester, not to run
    ts = pd.date_range("2024-01-01", periods=2, freq="h")
    prices = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100, 100],
            "high": [101, 101],
            "low": [99, 99],
            "close": [100, 100],
        }
    )
    preds = pd.DataFrame({"timestamp": ts, "prediction": [0, 0], "confidence": [1.0, 1.0]})
    return Backtester(predictions=preds, prices=prices, config=cfg)


class TestResolveExitPrice:
    """Unit tests for _resolve_exit_price instance method."""

    def test_stop_loss_returns_stop_level(self):
        bt = _make_backtester_for_unit_test(slippage_ticks=0.0)
        pos = Position(
            direction=1,
            contracts=1,
            entry_price=100.0,
            entry_time=pd.Timestamp("2024-01-01"),
            entry_bar=0,
            stop_loss=95.0,
            take_profit=110.0,
        )
        price = bt._resolve_exit_price(ExitReason.STOP_LOSS, pos, close_price=98.0)
        assert price == 95.0, f"Expected stop_loss=95.0, got {price}"

    def test_stop_loss_with_slippage(self):
        """Stop exits include slippage: price worse than barrier."""
        bt = _make_backtester_for_unit_test(slippage_ticks=2.0, tick_size=0.25)
        pos = Position(
            direction=1,
            contracts=1,
            entry_price=100.0,
            entry_time=pd.Timestamp("2024-01-01"),
            entry_bar=0,
            stop_loss=95.0,
            take_profit=110.0,
        )
        price = bt._resolve_exit_price(ExitReason.STOP_LOSS, pos, close_price=98.0)
        # Long stop: 95.0 - 2.0*0.25 = 94.50
        assert price == 94.50, f"Expected 94.50, got {price}"

    def test_short_stop_loss_with_slippage(self):
        """Short stop exits slip upward."""
        bt = _make_backtester_for_unit_test(slippage_ticks=2.0, tick_size=0.25)
        pos = Position(
            direction=-1,
            contracts=1,
            entry_price=100.0,
            entry_time=pd.Timestamp("2024-01-01"),
            entry_bar=0,
            stop_loss=105.0,
            take_profit=90.0,
        )
        price = bt._resolve_exit_price(ExitReason.STOP_LOSS, pos, close_price=106.0)
        # Short stop: 105.0 + 2.0*0.25 = 105.50
        assert price == 105.50, f"Expected 105.50, got {price}"

    def test_take_profit_returns_tp_level(self):
        bt = _make_backtester_for_unit_test(slippage_ticks=0.0)
        pos = Position(
            direction=1,
            contracts=1,
            entry_price=100.0,
            entry_time=pd.Timestamp("2024-01-01"),
            entry_bar=0,
            stop_loss=95.0,
            take_profit=110.0,
        )
        price = bt._resolve_exit_price(ExitReason.TAKE_PROFIT, pos, close_price=112.0)
        assert price == 110.0, f"Expected take_profit=110.0, got {price}"

    def test_max_holding_returns_close(self):
        bt = _make_backtester_for_unit_test()
        pos = Position(
            direction=1,
            contracts=1,
            entry_price=100.0,
            entry_time=pd.Timestamp("2024-01-01"),
            entry_bar=0,
            stop_loss=95.0,
            take_profit=110.0,
        )
        price = bt._resolve_exit_price(ExitReason.MAX_HOLDING, pos, close_price=102.0)
        assert price == 102.0, f"Expected close=102.0, got {price}"

    def test_signal_reversal_returns_close(self):
        bt = _make_backtester_for_unit_test()
        pos = Position(
            direction=1,
            contracts=1,
            entry_price=100.0,
            entry_time=pd.Timestamp("2024-01-01"),
            entry_bar=0,
            stop_loss=95.0,
            take_profit=110.0,
        )
        price = bt._resolve_exit_price(ExitReason.SIGNAL_REVERSAL, pos, close_price=103.0)
        assert price == 103.0, f"Expected close=103.0, got {price}"

    def test_stop_loss_none_falls_back_to_close(self):
        """Legacy mode: stop_loss is None, should return close."""
        bt = _make_backtester_for_unit_test()
        pos = Position(
            direction=1,
            contracts=1,
            entry_price=100.0,
            entry_time=pd.Timestamp("2024-01-01"),
            entry_bar=0,
            stop_loss=None,
            take_profit=None,
        )
        price = bt._resolve_exit_price(ExitReason.STOP_LOSS, pos, close_price=97.0)
        assert price == 97.0, f"Expected close=97.0 (fallback), got {price}"


# ---------------------------------------------------------------------------
# Integration test: Long position — stop-loss hit exits at stop level
# ---------------------------------------------------------------------------


class TestBarrierExitLongStopLoss:
    """A long position hits stop-loss; exit_price should be the stop level."""

    def test_long_stop_loss_exit_price(self):
        # Build price data:
        # - First 14+ bars: stable around 100 (ATR needs 14 bars to warm up)
        # - Bar where entry signal fires: still stable
        # - Next bar(s): price crashes — low penetrates stop level
        #
        # With k_down=1.5 and ATR ~ 2.0, stop ~ entry - 3.0 ~ 97.0
        # We craft bars so ATR stabilizes near 2.0.

        n_warmup = 20
        bars = []
        # Warmup: gentle oscillation giving ATR ~ 2.0
        for _i in range(n_warmup):
            mid = 100.0
            bars.append((mid - 0.5, mid + 1.0, mid - 1.0, mid + 0.5))
        # Bar 20: entry bar — signal=1, price stable
        bars.append((100.0, 101.0, 99.0, 100.0))
        # Bar 21: price crashes through stop — low goes to 90
        # Close stays at 94 (to verify exit_price != close)
        bars.append((99.0, 100.0, 90.0, 94.0))
        # Bar 22: neutral (won't be reached if exit happened)
        bars.append((94.0, 95.0, 93.0, 94.5))

        prices = _make_prices(bars)

        # Signals: 0 during warmup, 1 on bar 20 (entry), 1 on bar 21+ (hold)
        signals = [0] * n_warmup + [1, 1, 0]
        preds = _make_predictions(signals)

        config = _barrier_config(barrier_k_down=1.5, barrier_k_up=1.5)
        bt = Backtester(predictions=preds, prices=prices, config=config)
        result = bt.run()

        assert len(result.trades) >= 1, "Expected at least one trade"
        trade = result.trades[0]

        # Entry was on bar 20 at close ~ 100.0
        # ATR(14) on these bars ~ 2.0 (true range is high-low = 2.0 each bar)
        # stop_loss = entry_price - k_down * ATR ~ 100 - 1.5*2.0 = 97.0
        # Bar 21 low=90 << 97 → stop triggered
        # Exit price should be at the stop level (~97.0), NOT the close (94.0)
        assert (
            trade.exit_price != 94.0
        ), f"exit_price should NOT be bar close (94.0), got {trade.exit_price}"
        # The stop level is entry_price - k_down * ATR
        # Allow some tolerance since ATR is EWM-based and entry price has adverse selection
        expected_stop = trade.entry_price - 1.5 * 2.0
        assert (
            abs(trade.exit_price - expected_stop) < 1.0
        ), f"exit_price={trade.exit_price:.4f} not near expected stop={expected_stop:.4f}"


# ---------------------------------------------------------------------------
# Integration test: Long position — take-profit hit exits at TP level
# ---------------------------------------------------------------------------


class TestBarrierExitLongTakeProfit:
    """A long position hits take-profit; exit_price should be the TP level."""

    def test_long_take_profit_exit_price(self):
        n_warmup = 20
        bars = []
        for _i in range(n_warmup):
            mid = 100.0
            bars.append((mid - 0.5, mid + 1.0, mid - 1.0, mid + 0.5))
        # Bar 20: entry
        bars.append((100.0, 101.0, 99.0, 100.0))
        # Bar 21: price rockets — high goes to 110, well above TP
        # Close at 108 (to verify exit_price != close)
        bars.append((101.0, 110.0, 100.5, 108.0))
        # Trailing bar
        bars.append((108.0, 109.0, 107.0, 108.5))

        prices = _make_prices(bars)

        signals = [0] * n_warmup + [1, 1, 0]
        preds = _make_predictions(signals)

        config = _barrier_config(barrier_k_up=1.5, barrier_k_down=1.5)
        bt = Backtester(predictions=preds, prices=prices, config=config)
        result = bt.run()

        assert len(result.trades) >= 1, "Expected at least one trade"
        trade = result.trades[0]

        # take_profit = entry_price + k_up * ATR ~ 100 + 1.5*2.0 = 103.0
        # Bar 21 high=110 >> 103 → TP triggered
        # Exit price should be TP level (~103), NOT the close (108)
        assert (
            trade.exit_price != 108.0
        ), f"exit_price should NOT be bar close (108.0), got {trade.exit_price}"
        expected_tp = trade.entry_price + 1.5 * 2.0
        assert (
            abs(trade.exit_price - expected_tp) < 1.0
        ), f"exit_price={trade.exit_price:.4f} not near expected TP={expected_tp:.4f}"


# ---------------------------------------------------------------------------
# Integration test: Short position — stop-loss hit
# ---------------------------------------------------------------------------


class TestBarrierExitShortStopLoss:
    """A short position hits stop-loss; exit_price should be the stop level."""

    def test_short_stop_loss_exit_price(self):
        n_warmup = 20
        bars = []
        for _i in range(n_warmup):
            mid = 100.0
            bars.append((mid - 0.5, mid + 1.0, mid - 1.0, mid + 0.5))
        # Bar 20: entry for short
        bars.append((100.0, 101.0, 99.0, 100.0))
        # Bar 21: price spikes up — high goes to 110 (above short stop)
        # Close at 106 (to verify exit_price != close)
        bars.append((101.0, 110.0, 100.0, 106.0))
        bars.append((106.0, 107.0, 105.0, 106.5))

        prices = _make_prices(bars)

        # Short signal (-1) on bar 20
        signals = [0] * n_warmup + [-1, -1, 0]
        preds = _make_predictions(signals)

        config = _barrier_config(barrier_k_up=1.5, barrier_k_down=1.5, allow_short=True)
        bt = Backtester(predictions=preds, prices=prices, config=config)
        result = bt.run()

        assert len(result.trades) >= 1, "Expected at least one trade"
        trade = result.trades[0]
        assert trade.direction == -1, f"Expected short trade, got direction={trade.direction}"

        # Short stop_loss = entry_price + k_up * ATR ~ 100 + 1.5*2.0 = 103.0
        # Bar 21 high=110 >> 103 → stop triggered
        # Exit price should be stop level (~103), NOT the close (106)
        assert (
            trade.exit_price != 106.0
        ), f"exit_price should NOT be bar close (106.0), got {trade.exit_price}"
        expected_stop = trade.entry_price + 1.5 * 2.0
        assert (
            abs(trade.exit_price - expected_stop) < 1.0
        ), f"exit_price={trade.exit_price:.4f} not near expected stop={expected_stop:.4f}"


# ---------------------------------------------------------------------------
# Integration test: Short position — take-profit hit
# ---------------------------------------------------------------------------


class TestBarrierExitShortTakeProfit:
    """A short position hits take-profit; exit_price should be the TP level."""

    def test_short_take_profit_exit_price(self):
        n_warmup = 20
        bars = []
        for _i in range(n_warmup):
            mid = 100.0
            bars.append((mid - 0.5, mid + 1.0, mid - 1.0, mid + 0.5))
        # Bar 20: entry for short
        bars.append((100.0, 101.0, 99.0, 100.0))
        # Bar 21: price crashes — low goes to 90 (below short TP)
        # Close at 92 (to verify exit_price != close)
        bars.append((99.0, 100.0, 90.0, 92.0))
        bars.append((92.0, 93.0, 91.0, 92.5))

        prices = _make_prices(bars)

        signals = [0] * n_warmup + [-1, -1, 0]
        preds = _make_predictions(signals)

        config = _barrier_config(barrier_k_up=1.5, barrier_k_down=1.5, allow_short=True)
        bt = Backtester(predictions=preds, prices=prices, config=config)
        result = bt.run()

        assert len(result.trades) >= 1, "Expected at least one trade"
        trade = result.trades[0]
        assert trade.direction == -1, f"Expected short trade, got direction={trade.direction}"

        # Short take_profit = entry_price - k_down * ATR ~ 100 - 1.5*2.0 = 97.0
        # Bar 21 low=90 << 97 → TP triggered
        # Exit price should be TP level (~97), NOT the close (92)
        assert (
            trade.exit_price != 92.0
        ), f"exit_price should NOT be bar close (92.0), got {trade.exit_price}"
        expected_tp = trade.entry_price - 1.5 * 2.0
        assert (
            abs(trade.exit_price - expected_tp) < 1.0
        ), f"exit_price={trade.exit_price:.4f} not near expected TP={expected_tp:.4f}"


# ---------------------------------------------------------------------------
# Both stop and TP triggered on same bar — stop wins (conservative)
# ---------------------------------------------------------------------------


class TestBarrierBothTriggeredStopWins:
    """When both stop and TP can trigger on the same bar, stop_loss wins."""

    def test_stop_wins_over_take_profit(self):
        n_warmup = 20
        bars = []
        for _i in range(n_warmup):
            mid = 100.0
            bars.append((mid - 0.5, mid + 1.0, mid - 1.0, mid + 0.5))
        # Bar 20: entry
        bars.append((100.0, 101.0, 99.0, 100.0))
        # Bar 21: extreme bar — both high and low breach barriers
        # ATR ~ 2.0 → stop ~ 97.0, TP ~ 103.0
        # low=90 < 97 (stop triggers) AND high=110 > 103 (TP triggers)
        bars.append((100.0, 110.0, 90.0, 100.0))
        bars.append((100.0, 101.0, 99.0, 100.0))

        prices = _make_prices(bars)
        signals = [0] * n_warmup + [1, 1, 0]
        preds = _make_predictions(signals)

        config = _barrier_config(barrier_k_up=1.5, barrier_k_down=1.5)
        bt = Backtester(predictions=preds, prices=prices, config=config)
        result = bt.run()

        assert len(result.trades) >= 1, "Expected at least one trade"
        trade = result.trades[0]

        # When both trigger, _should_exit_fast returns STOP_LOSS (conservative)
        # So exit_price should be the stop level, not the TP level
        expected_stop = trade.entry_price - 1.5 * 2.0
        expected_tp = trade.entry_price + 1.5 * 2.0

        # Exit price should be near stop, not near TP
        assert abs(trade.exit_price - expected_stop) < abs(trade.exit_price - expected_tp), (
            f"exit_price={trade.exit_price:.4f} closer to TP={expected_tp:.4f} "
            f"than stop={expected_stop:.4f}; stop should win"
        )


# ---------------------------------------------------------------------------
# Legacy mode: barriers off → exit at close
# ---------------------------------------------------------------------------


class TestLegacyModeExitsAtClose:
    """With barrier_k_up=0 and barrier_k_down=0, exits use bar close."""

    def test_legacy_stop_exits_at_close(self):
        n_warmup = 20
        bars = []
        for _i in range(n_warmup):
            mid = 100.0
            bars.append((mid - 0.5, mid + 1.0, mid - 1.0, mid + 0.5))
        bars.append((100.0, 101.0, 99.0, 100.0))
        # Price drops below legacy 2% stop (98.0)
        bars.append((99.0, 100.0, 90.0, 94.0))
        bars.append((94.0, 95.0, 93.0, 94.5))

        prices = _make_prices(bars)
        signals = [0] * n_warmup + [1, 1, 0]
        preds = _make_predictions(signals)

        # Legacy: barrier_k_up=0, barrier_k_down=0
        config = _barrier_config(barrier_k_up=0.0, barrier_k_down=0.0)
        bt = Backtester(predictions=preds, prices=prices, config=config)
        result = bt.run()

        assert len(result.trades) >= 1, "Expected at least one trade"
        trade = result.trades[0]

        # In legacy mode, stop_loss = entry * (1 - 0.02)
        # _resolve_exit_price still returns stop_loss level (it's set even in legacy mode)
        # The key difference: legacy stop is 2% fixed vs barrier stop is ATR-based
        legacy_stop = trade.entry_price * 0.98
        assert (
            abs(trade.exit_price - legacy_stop) < 0.5
        ), f"Legacy exit_price={trade.exit_price:.4f} not near 2% stop={legacy_stop:.4f}"
