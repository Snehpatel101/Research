"""
Tests for financial performance metrics.

Verifies Sharpe ratio, Sortino ratio, max drawdown, win rate, and
profit factor against known analytical values and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.inference.backtesting.metrics import (
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_win_rate,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Sharpe ratio
# ---------------------------------------------------------------------------


class TestSharpeRatio:
    """Sharpe ratio edge cases and known values."""

    def test_zero_returns_sharpe_zero(self) -> None:
        """Constant zero returns have zero excess return -> Sharpe = 0."""
        returns = np.zeros(100)
        assert calculate_sharpe_ratio(returns) == pytest.approx(0.0)

    def test_constant_positive_returns(self) -> None:
        """Constant positive returns have std ~0 via ddof=1 for n=1 edge,
        but for n>1 the std with ddof=1 is 0, which returns 0.0."""
        returns = np.full(100, 0.001)
        # std(ddof=1) of constant array = 0 -> function returns 0.0
        assert calculate_sharpe_ratio(returns) == pytest.approx(0.0)

    def test_positive_mean_positive_sharpe(self, rng: np.random.Generator) -> None:
        """Returns with positive mean should yield positive Sharpe."""
        returns = rng.normal(0.001, 0.01, 500)
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        assert sharpe > 0

    def test_empty_returns(self) -> None:
        """Empty returns array should return 0."""
        assert calculate_sharpe_ratio(np.array([])) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Sortino ratio
# ---------------------------------------------------------------------------


class TestSortinoRatio:
    """Sortino ratio correctness."""

    def test_no_downside(self) -> None:
        """All positive returns -> capped Sortino of 10.0."""
        returns = np.full(100, 0.01)
        assert calculate_sortino_ratio(returns) == pytest.approx(10.0)

    def test_negative_mean_negative_sortino(self, rng: np.random.Generator) -> None:
        """Returns with strongly negative mean should yield negative Sortino."""
        returns = rng.normal(-0.005, 0.01, 500)
        sortino = calculate_sortino_ratio(returns)
        assert sortino < 0

    def test_empty_returns(self) -> None:
        assert calculate_sortino_ratio(np.array([])) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    """Max drawdown of known equity curves."""

    def test_simple_known_drawdown(self) -> None:
        """Equity: 100 -> 120 -> 90 -> 110. DD = (90-120)/120 = -25%."""
        equity = np.array([100.0, 120.0, 90.0, 110.0])
        max_dd, peak_idx, trough_idx = calculate_max_drawdown(equity)
        assert max_dd == pytest.approx(-0.25)
        assert peak_idx == 1
        assert trough_idx == 2

    def test_monotonically_increasing(self) -> None:
        """Strictly increasing equity -> 0% drawdown."""
        equity = np.arange(1.0, 101.0)
        max_dd, _, _ = calculate_max_drawdown(equity)
        assert max_dd == pytest.approx(0.0)

    def test_monotonically_decreasing(self) -> None:
        """Equity: 100 -> 50 -> drawdown is (50-100)/100 = -50%."""
        equity = np.linspace(100, 50, 50)
        max_dd, peak_idx, trough_idx = calculate_max_drawdown(equity)
        assert max_dd == pytest.approx(-0.50)
        assert peak_idx == 0
        assert trough_idx == 49

    def test_empty_equity(self) -> None:
        max_dd, _, _ = calculate_max_drawdown(np.array([]))
        assert max_dd == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Win rate
# ---------------------------------------------------------------------------


class TestWinRate:
    """Win rate of known trade sequences."""

    def test_all_winners(self) -> None:
        trades = np.array([10.0, 5.0, 20.0])
        assert calculate_win_rate(trades) == pytest.approx(1.0)

    def test_all_losers(self) -> None:
        trades = np.array([-10.0, -5.0, -20.0])
        assert calculate_win_rate(trades) == pytest.approx(0.0)

    def test_mixed(self) -> None:
        """3 wins, 2 losses -> 60%."""
        trades = np.array([10.0, -5.0, 20.0, -3.0, 15.0])
        assert calculate_win_rate(trades) == pytest.approx(0.6)

    def test_empty(self) -> None:
        assert calculate_win_rate(np.array([])) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Profit factor
# ---------------------------------------------------------------------------


class TestProfitFactor:
    """Profit factor of known trade sequences."""

    def test_known_profit_factor(self) -> None:
        """gross_profit=30, gross_loss=8 -> PF=3.75."""
        trades = np.array([10.0, -5.0, 20.0, -3.0])
        pf = calculate_profit_factor(trades)
        assert pf == pytest.approx(30.0 / 8.0)

    def test_no_losses(self) -> None:
        """All wins -> infinite profit factor."""
        trades = np.array([10.0, 20.0])
        pf = calculate_profit_factor(trades)
        assert pf == float("inf")

    def test_no_wins(self) -> None:
        """All losses -> 0 profit factor."""
        trades = np.array([-10.0, -20.0])
        pf = calculate_profit_factor(trades)
        assert pf == pytest.approx(0.0)

    def test_empty(self) -> None:
        assert calculate_profit_factor(np.array([])) == pytest.approx(0.0)

    def test_breakeven(self) -> None:
        """Equal wins and losses -> PF = 1.0."""
        trades = np.array([10.0, -10.0])
        pf = calculate_profit_factor(trades)
        assert pf == pytest.approx(1.0)
