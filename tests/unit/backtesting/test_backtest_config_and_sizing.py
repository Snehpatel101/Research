"""
BacktestConfig defaults and the inputs the backtester feeds its position sizer.

- Production-correct defaults (next-bar open fills, legacy barrier mode, ...).
- MIDPOINT is the execution-model name; VWAP survives only as an alias.
- Kelly sizing uses defaults until ``kelly_min_trades`` trades are complete, then
  the running win rate / average win / average loss.
- A confidence of exactly 0.0 is a real value, not "missing".
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.inference.backtesting.backtest import BacktestConfig, ExecutionModel
from src.inference.backtesting.position_sizing import BasePositionSizer
from tests.helpers import make_minimal_backtester


class TestBacktestConfigDefaults:
    def test_production_defaults(self) -> None:
        config = BacktestConfig()

        assert config.execution_model == ExecutionModel.MARKET_ON_OPEN
        assert config.force_session_close is False
        assert config.kelly_min_trades == 30
        assert config.barrier_k_up == 0.0  # 0.0 = legacy (no barrier) mode
        assert config.barrier_k_down == 0.0
        assert config.alignment_loss_warn_pct == 5.0

    def test_midpoint_is_canonical_and_vwap_is_an_alias(self) -> None:
        assert ExecutionModel.VWAP is ExecutionModel.MIDPOINT
        assert ExecutionModel.MIDPOINT.value == "midpoint"


class _SpySizer(BasePositionSizer):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def calculate_position_size(self, account_equity: float, **kwargs: Any) -> int:
        self.calls.append(kwargs)
        return 1


def _backtester(min_trades: int = 5):
    spy = _SpySizer()
    return make_minimal_backtester(position_sizer=spy, kelly_min_trades=min_trades), spy


def _record_trade(bt, pnl: float) -> None:
    bt._trades.append(SimpleNamespace(net_pnl=pnl))
    bt._update_kelly_stats()


def _sizer_inputs(bt, spy, **kwargs: Any) -> dict[str, Any]:
    bt._calculate_position_size(current_price=4500.0, **kwargs)
    return spy.calls[-1]


class TestKellyWarmUp:
    def test_defaults_are_used_until_min_trades_are_complete(self) -> None:
        bt, spy = _backtester(min_trades=5)
        for pnl in (100.0, 100.0, -50.0, 100.0):  # 4 of 5 required trades
            _record_trade(bt, pnl)

        inputs = _sizer_inputs(bt, spy)
        assert (inputs["win_rate"], inputs["avg_win"], inputs["avg_loss"]) == (0.5, 100.0, 100.0)

    def test_running_stats_take_over_at_min_trades(self) -> None:
        bt, spy = _backtester(min_trades=5)
        for pnl in (200.0, 100.0, -50.0, -70.0, 300.0):
            _record_trade(bt, pnl)

        inputs = _sizer_inputs(bt, spy)
        assert inputs["win_rate"] == pytest.approx(3 / 5)
        assert inputs["avg_win"] == pytest.approx(200.0)
        assert inputs["avg_loss"] == pytest.approx(60.0)

    def test_stats_keep_updating_after_activation(self) -> None:
        bt, spy = _backtester(min_trades=2)
        _record_trade(bt, 100.0)
        _record_trade(bt, -100.0)
        assert _sizer_inputs(bt, spy)["win_rate"] == pytest.approx(0.5)

        _record_trade(bt, 100.0)
        assert _sizer_inputs(bt, spy)["win_rate"] == pytest.approx(2 / 3)


class TestConfidenceInput:
    @pytest.mark.parametrize(("confidence", "expected"), [(0.0, 0.0), (0.8, 0.8), (None, 0.5)])
    def test_confidence_reaches_the_sizer_as_probability(
        self, confidence: float | None, expected: float
    ) -> None:
        bt, spy = _backtester()
        assert _sizer_inputs(bt, spy, confidence=confidence)["probability"] == expected

    def test_default_stop_distance_is_two_percent_of_price(self) -> None:
        bt, spy = _backtester()
        assert _sizer_inputs(bt, spy)["stop_distance"] == pytest.approx(4500.0 * 0.02)
