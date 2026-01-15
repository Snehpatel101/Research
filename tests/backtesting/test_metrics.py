"""
Tests for backtesting metrics module.

Tests cover:
- Sharpe ratio calculation
- Sortino ratio calculation
- Maximum drawdown
- Calmar ratio
- Win rate and profit factor
- VaR and CVaR
- Expectancy metrics
- PerformanceMetrics dataclass
"""

import numpy as np
import pytest

from src.backtesting.metrics import (
    PerformanceMetrics,
    calculate_all_metrics,
    calculate_calmar_ratio,
    calculate_cvar,
    calculate_expectancy,
    calculate_expectancy_ratio,
    calculate_max_drawdown,
    calculate_max_drawdown_duration,
    calculate_payoff_ratio,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_ulcer_index,
    calculate_var,
    calculate_win_rate,
)


class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    def test_positive_sharpe(self):
        """Positive returns should give positive Sharpe."""
        # Consistent positive returns
        returns = np.array([0.01, 0.02, 0.01, 0.015, 0.02])
        sharpe = calculate_sharpe_ratio(returns, periods_per_year=252)

        assert sharpe > 0

    def test_negative_sharpe(self):
        """Consistent negative returns should give negative Sharpe."""
        returns = np.array([-0.01, -0.02, -0.01, -0.015, -0.02])
        sharpe = calculate_sharpe_ratio(returns, periods_per_year=252)

        assert sharpe < 0

    def test_zero_returns(self):
        """Zero returns should give zero Sharpe."""
        returns = np.zeros(100)
        sharpe = calculate_sharpe_ratio(returns)

        assert sharpe == 0.0

    def test_annualization_daily(self):
        """Daily returns should use 252 for annualization."""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.0003

        sharpe_annual = calculate_sharpe_ratio(returns, periods_per_year=252)
        sharpe_no_annual = calculate_sharpe_ratio(returns, periods_per_year=1)

        # Annualized should be sqrt(252) times larger in magnitude
        assert abs(sharpe_annual) > abs(sharpe_no_annual)

    def test_empty_returns(self):
        """Empty returns should return 0."""
        sharpe = calculate_sharpe_ratio(np.array([]))
        assert sharpe == 0.0


class TestSortinoRatio:
    """Tests for Sortino ratio calculation."""

    def test_positive_sortino(self):
        """Positive returns should give positive Sortino."""
        returns = np.array([0.01, 0.02, 0.01, 0.015, 0.02])
        sortino = calculate_sortino_ratio(returns, periods_per_year=252)

        assert sortino > 0

    def test_no_downside_capped(self):
        """All positive returns should give capped Sortino."""
        returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        sortino = calculate_sortino_ratio(returns)

        # With no negative returns, Sortino is capped at 10.0
        assert sortino == 10.0

    def test_sortino_vs_sharpe(self):
        """Sortino should be calculated for skewed returns."""
        # Returns with few large negative returns but mostly positive
        returns = np.array([0.01, 0.02, 0.01, -0.05, 0.02, 0.01, 0.015])

        sharpe = calculate_sharpe_ratio(returns, periods_per_year=1)
        sortino = calculate_sortino_ratio(returns, periods_per_year=1)

        # Both should be valid numbers
        assert isinstance(sortino, float)
        assert isinstance(sharpe, float)


class TestMaxDrawdown:
    """Tests for maximum drawdown calculation."""

    def test_no_drawdown(self):
        """Monotonically increasing equity has 0 drawdown."""
        equity = np.array([100, 101, 102, 103, 104, 105])
        dd, peak_idx, trough_idx = calculate_max_drawdown(equity)

        assert dd == 0.0
        assert peak_idx == 0
        assert trough_idx == 0

    def test_full_drawdown(self):
        """Going to near zero is ~100% drawdown."""
        equity = np.array([100, 50, 25, 1])
        dd, peak_idx, trough_idx = calculate_max_drawdown(equity)

        assert dd < -0.90  # More than 90% drawdown (negative)

    def test_partial_drawdown(self):
        """50% decline should give -50% drawdown."""
        equity = np.array([100, 100, 50, 50])
        dd, peak_idx, trough_idx = calculate_max_drawdown(equity)

        assert dd == pytest.approx(-0.5, abs=0.01)

    def test_recovery_doesnt_reduce_max(self):
        """Recovery after drawdown shouldn't change max drawdown."""
        equity = np.array([100, 80, 60, 80, 100])  # 40% drawdown, then recovery
        dd, peak_idx, trough_idx = calculate_max_drawdown(equity)

        assert dd == pytest.approx(-0.4, abs=0.01)

    def test_empty_equity(self):
        """Empty equity should return (0, 0, 0)."""
        dd, peak_idx, trough_idx = calculate_max_drawdown(np.array([]))
        assert dd == 0.0
        assert peak_idx == 0
        assert trough_idx == 0


class TestMaxDrawdownDuration:
    """Tests for max drawdown duration calculation."""

    def test_no_drawdown_zero_duration(self):
        """No drawdown means zero duration."""
        equity = np.array([100, 101, 102, 103])
        duration, start, end = calculate_max_drawdown_duration(equity)

        assert duration == 0

    def test_underwater_duration(self):
        """Should count bars underwater."""
        equity = np.array([100, 90, 85, 90, 95, 100, 105])
        duration, start, end = calculate_max_drawdown_duration(equity)

        # Underwater from index 1-5 (5 bars), recovered at 6
        # Duration counts consecutive bars in drawdown
        assert duration >= 4


class TestCalmarRatio:
    """Tests for Calmar ratio calculation."""

    def test_positive_calmar(self):
        """Positive return with drawdown gives positive Calmar."""
        returns = np.array([0.01, 0.02, -0.01, 0.02, 0.01])  # Net positive
        calmar = calculate_calmar_ratio(returns, periods_per_year=252)

        assert calmar > 0

    def test_no_drawdown_calmar(self):
        """No drawdown should give capped Calmar."""
        returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        calmar = calculate_calmar_ratio(returns)

        # With no drawdown, Calmar is capped at 10.0
        assert calmar == 10.0


class TestWinRate:
    """Tests for win rate calculation."""

    def test_all_wins(self):
        """All positive returns = 100% win rate."""
        returns = np.array([0.01, 0.02, 0.03, 0.04])
        win_rate = calculate_win_rate(returns)

        assert win_rate == 1.0

    def test_all_losses(self):
        """All negative returns = 0% win rate."""
        returns = np.array([-0.01, -0.02, -0.03, -0.04])
        win_rate = calculate_win_rate(returns)

        assert win_rate == 0.0

    def test_half_wins(self):
        """Half positive = 50% win rate."""
        returns = np.array([0.01, -0.01, 0.02, -0.02])
        win_rate = calculate_win_rate(returns)

        assert win_rate == 0.5

    def test_zeros_included_in_total(self):
        """Zero returns are counted in total trades."""
        returns = np.array([0.01, 0, -0.01, 0])
        win_rate = calculate_win_rate(returns)

        # 1 win out of 4 trades = 25%
        assert win_rate == 0.25


class TestProfitFactor:
    """Tests for profit factor calculation."""

    def test_positive_profit_factor(self):
        """More gross wins than losses = PF > 1."""
        returns = np.array([0.03, 0.02, -0.01, 0.01])
        pf = calculate_profit_factor(returns)

        assert pf > 1.0

    def test_negative_profit_factor(self):
        """More gross losses than wins = PF < 1."""
        returns = np.array([-0.03, -0.02, 0.01, -0.01])
        pf = calculate_profit_factor(returns)

        assert pf < 1.0

    def test_no_losses(self):
        """No losses = infinite PF."""
        returns = np.array([0.01, 0.02, 0.03])
        pf = calculate_profit_factor(returns)

        assert np.isinf(pf) or pf > 1000


class TestExpectancy:
    """Tests for expectancy calculation."""

    def test_positive_expectancy(self):
        """Profitable strategy has positive expectancy."""
        returns = np.array([0.03, 0.02, -0.01, 0.01, -0.02])
        expectancy = calculate_expectancy(returns)

        # Sum = 0.03 + 0.02 - 0.01 + 0.01 - 0.02 = 0.03, avg = 0.006
        assert expectancy > 0

    def test_negative_expectancy(self):
        """Losing strategy has negative expectancy."""
        returns = np.array([-0.03, -0.02, 0.01, -0.01])
        expectancy = calculate_expectancy(returns)

        assert expectancy < 0


class TestExpectancyRatio:
    """Tests for expectancy ratio (edge ratio)."""

    def test_positive_ratio(self):
        """Positive edge has ratio > 0."""
        returns = np.array([0.03, 0.02, -0.01, 0.01])
        ratio = calculate_expectancy_ratio(returns)

        # Should be positive for net winning strategy
        assert ratio > 0


class TestPayoffRatio:
    """Tests for payoff ratio (reward/risk)."""

    def test_equal_wins_losses(self):
        """Equal avg win/loss = payoff of 1."""
        returns = np.array([0.02, -0.02, 0.02, -0.02])
        payoff = calculate_payoff_ratio(returns)

        assert payoff == pytest.approx(1.0, abs=0.01)

    def test_higher_wins(self):
        """Larger wins than losses = payoff > 1."""
        returns = np.array([0.04, -0.02, 0.03, -0.01])
        payoff = calculate_payoff_ratio(returns)

        assert payoff > 1.0


class TestVaR:
    """Tests for Value at Risk calculation."""

    def test_var_95(self):
        """95% VaR should be the 5th percentile."""
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02

        var = calculate_var(returns, confidence_level=0.95)

        # VaR is typically positive (representing loss)
        # For normal returns, 95% VaR ≈ 1.645 * std
        assert var > 0
        assert var < 0.1  # Reasonable bound


class TestCVaR:
    """Tests for Conditional VaR (Expected Shortfall)."""

    def test_cvar_greater_than_var(self):
        """CVaR should be >= VaR."""
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02

        var = calculate_var(returns, confidence_level=0.95)
        cvar = calculate_cvar(returns, confidence_level=0.95)

        assert cvar >= var


class TestUlcerIndex:
    """Tests for Ulcer Index calculation."""

    def test_no_drawdown_zero_ulcer(self):
        """No drawdown means zero Ulcer Index."""
        equity = np.array([100, 101, 102, 103, 104])
        ulcer = calculate_ulcer_index(equity)

        assert ulcer == 0.0

    def test_drawdown_positive_ulcer(self):
        """Drawdown should give positive Ulcer Index."""
        equity = np.array([100, 95, 90, 95, 100])
        ulcer = calculate_ulcer_index(equity)

        assert ulcer > 0


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""

    def test_dataclass_fields(self):
        """PerformanceMetrics should have expected fields."""
        from dataclasses import asdict

        metrics = PerformanceMetrics(
            total_return=0.10,
            cagr=0.12,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=-0.15,  # Negative indicates 15% drawdown
            calmar_ratio=0.8,
            win_rate=0.55,
            profit_factor=1.3,
            total_trades=100,
            expectancy=50.0,
            payoff_ratio=1.2,
        )

        d = asdict(metrics)

        assert d["total_return"] == 0.10
        assert d["sharpe_ratio"] == 1.5
        assert d["win_rate"] == 0.55


class TestCalculateAllMetrics:
    """Tests for calculate_all_metrics function."""

    def test_returns_performance_metrics(self):
        """Should return PerformanceMetrics object."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.02
        equity = 100000 * (1 + returns).cumprod()

        metrics = calculate_all_metrics(returns, equity)

        assert isinstance(metrics, PerformanceMetrics)
        assert hasattr(metrics, "sharpe_ratio")
        assert hasattr(metrics, "win_rate")

    def test_with_equity_curve(self):
        """Should work with equity curve provided."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.02
        equity = 100000 * (1 + returns).cumprod()

        metrics = calculate_all_metrics(returns, equity)

        # Max drawdown is stored as negative (e.g., -0.29 = 29% drawdown)
        assert metrics.max_drawdown <= 0


class TestValidationIntegration:
    """Test that metrics are accessible from validation module."""

    def test_import_from_validation(self):
        """Should be importable from src.validation."""
        from src.validation import (
            PerformanceMetrics,
            calculate_all_metrics,
            calculate_calmar_ratio,
            calculate_max_drawdown,
            calculate_sharpe_ratio,
            calculate_sortino_ratio,
            calculate_win_rate,
        )

        assert PerformanceMetrics is not None
        assert callable(calculate_sharpe_ratio)
        assert callable(calculate_max_drawdown)
