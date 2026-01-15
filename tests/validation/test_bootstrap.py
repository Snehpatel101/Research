"""
Tests for bootstrap confidence intervals.

Tests cover:
- Generic bootstrap_metric function
- Specialized bootstrap functions (Sharpe, drawdown, accuracy, F1, win rate)
- Multi-metric bootstrap
- BCa vs percentile methods
"""

import numpy as np
import pytest

from src.validation.bootstrap import (
    BootstrapResult,
    bootstrap_accuracy,
    bootstrap_f1_score,
    bootstrap_max_drawdown,
    bootstrap_metric,
    bootstrap_multiple_metrics,
    bootstrap_sharpe_ratio,
    bootstrap_win_rate,
)


class TestBootstrapMetric:
    """Tests for generic bootstrap_metric function."""

    def test_basic_mean_estimate(self):
        """Test bootstrap CI for mean."""
        np.random.seed(42)
        data = np.random.randn(100) + 5  # Mean ~5

        result = bootstrap_metric(data, np.mean, n_bootstrap=500, random_state=42)

        assert isinstance(result, BootstrapResult)
        assert 4.5 < result.estimate < 5.5
        assert result.ci_lower < result.estimate < result.ci_upper
        assert result.ci_level == 0.95

    def test_ci_contains_true_value(self):
        """CI should contain true parameter value most of the time."""
        np.random.seed(42)
        true_mean = 10
        data = np.random.randn(200) + true_mean

        result = bootstrap_metric(data, np.mean, n_bootstrap=1000, random_state=42)

        # CI should contain true mean
        assert result.ci_lower < true_mean < result.ci_upper

    def test_std_error_computed(self):
        """Standard error should be computed."""
        np.random.seed(42)
        data = np.random.randn(100)

        result = bootstrap_metric(data, np.mean, n_bootstrap=500, random_state=42)

        assert result.std_error > 0
        assert result.std_error < np.std(data)  # SE < SD for mean

    def test_different_ci_levels(self):
        """Test different confidence levels."""
        np.random.seed(42)
        data = np.random.randn(100)

        result_90 = bootstrap_metric(data, np.mean, ci_level=0.90, n_bootstrap=500, random_state=42)
        result_95 = bootstrap_metric(data, np.mean, ci_level=0.95, n_bootstrap=500, random_state=42)
        result_99 = bootstrap_metric(data, np.mean, ci_level=0.99, n_bootstrap=500, random_state=42)

        # Wider CI for higher confidence
        assert result_90.ci_width < result_95.ci_width < result_99.ci_width

    def test_percentile_method(self):
        """Test percentile CI method."""
        np.random.seed(42)
        data = np.random.randn(100)

        result = bootstrap_metric(
            data, np.mean, method="percentile", n_bootstrap=500, random_state=42
        )

        assert result.method == "percentile"

    def test_bca_method(self):
        """Test BCa CI method."""
        np.random.seed(42)
        data = np.random.randn(100)

        result = bootstrap_metric(data, np.mean, method="bca", n_bootstrap=500, random_state=42)

        assert result.method == "bca"

    def test_return_distribution(self):
        """Test returning full bootstrap distribution."""
        np.random.seed(42)
        data = np.random.randn(50)

        result = bootstrap_metric(
            data, np.mean, n_bootstrap=200, return_distribution=True, random_state=42
        )

        assert result.bootstrap_distribution is not None
        assert len(result.bootstrap_distribution) <= 200

    def test_reproducibility(self):
        """Same seed should give same results."""
        data = np.random.randn(50)

        result1 = bootstrap_metric(data, np.mean, n_bootstrap=100, random_state=123)
        result2 = bootstrap_metric(data, np.mean, n_bootstrap=100, random_state=123)

        assert result1.estimate == result2.estimate
        assert result1.ci_lower == result2.ci_lower
        assert result1.ci_upper == result2.ci_upper

    def test_minimum_data_required(self):
        """Should require at least 2 data points."""
        data = np.array([1.0])

        with pytest.raises(ValueError, match="at least 2"):
            bootstrap_metric(data, np.mean)

    def test_invalid_method_raises(self):
        """Invalid method should raise."""
        data = np.random.randn(50)

        with pytest.raises(ValueError, match="Unknown method"):
            bootstrap_metric(data, np.mean, method="invalid")

    def test_ci_width_property(self):
        """Test ci_width property."""
        np.random.seed(42)
        data = np.random.randn(50)

        result = bootstrap_metric(data, np.mean, n_bootstrap=200, random_state=42)

        assert result.ci_width == result.ci_upper - result.ci_lower

    def test_relative_ci_width(self):
        """Test relative_ci_width property."""
        np.random.seed(42)
        data = np.random.randn(50) + 10  # Non-zero mean

        result = bootstrap_metric(data, np.mean, n_bootstrap=200, random_state=42)

        expected_rel_width = result.ci_width / abs(result.estimate)
        assert abs(result.relative_ci_width - expected_rel_width) < 1e-10

    def test_to_dict(self):
        """Test conversion to dictionary."""
        np.random.seed(42)
        data = np.random.randn(50)

        result = bootstrap_metric(data, np.mean, n_bootstrap=200, random_state=42)
        result_dict = result.to_dict()

        assert "estimate" in result_dict
        assert "ci_lower" in result_dict
        assert "ci_upper" in result_dict
        assert "ci_level" in result_dict
        assert "std_error" in result_dict
        assert "ci_width" in result_dict

    def test_str_representation(self):
        """Test string representation."""
        np.random.seed(42)
        data = np.random.randn(50)

        result = bootstrap_metric(data, np.mean, n_bootstrap=200, random_state=42)
        str_repr = str(result)

        assert "95% CI" in str_repr
        assert "[" in str_repr and "]" in str_repr


class TestBootstrapSharpeRatio:
    """Tests for bootstrap_sharpe_ratio function."""

    def test_positive_sharpe(self):
        """Test with positive Sharpe ratio."""
        np.random.seed(42)
        # Simulate returns with positive expected return
        returns = np.random.randn(252) * 0.01 + 0.0005  # ~12% annual return

        result = bootstrap_sharpe_ratio(returns, n_bootstrap=500, random_state=42)

        assert result.estimate > 0
        assert result.ci_lower < result.estimate < result.ci_upper

    def test_annualization_daily(self):
        """Test daily annualization (252 periods)."""
        np.random.seed(42)
        daily_returns = np.random.randn(252) * 0.01 + 0.0003

        result = bootstrap_sharpe_ratio(
            daily_returns, periods_per_year=252, n_bootstrap=300, random_state=42
        )

        # Manual calculation for comparison
        manual_sharpe = np.mean(daily_returns) / np.std(daily_returns, ddof=1) * np.sqrt(252)
        assert abs(result.estimate - manual_sharpe) < 0.01

    def test_annualization_monthly(self):
        """Test monthly annualization (12 periods)."""
        np.random.seed(42)
        monthly_returns = np.random.randn(36) * 0.03 + 0.005

        result = bootstrap_sharpe_ratio(
            monthly_returns, periods_per_year=12, n_bootstrap=300, random_state=42
        )

        manual_sharpe = np.mean(monthly_returns) / np.std(monthly_returns, ddof=1) * np.sqrt(12)
        assert abs(result.estimate - manual_sharpe) < 0.01

    def test_handles_zero_std(self):
        """Should handle constant returns (zero std)."""
        returns = np.ones(100) * 0.001  # Constant returns

        result = bootstrap_sharpe_ratio(returns, n_bootstrap=100, random_state=42)

        # With constant returns, std=0, Sharpe is either 0 or undefined (inf)
        # The function should handle this gracefully without crashing
        # Constant positive returns with ddof=1 gives std very close to 0 but not exactly
        assert isinstance(result, BootstrapResult)


class TestBootstrapMaxDrawdown:
    """Tests for bootstrap_max_drawdown function."""

    def test_positive_drawdown(self):
        """Drawdown should be positive (as percentage)."""
        np.random.seed(42)
        returns = np.random.randn(200) * 0.02

        result = bootstrap_max_drawdown(returns, n_bootstrap=300, random_state=42)

        assert result.estimate > 0
        assert result.ci_lower > 0

    def test_drawdown_increases_with_volatility(self):
        """Higher volatility should lead to larger drawdowns."""
        np.random.seed(42)
        returns_low_vol = np.random.randn(200) * 0.005
        returns_high_vol = np.random.randn(200) * 0.05

        result_low = bootstrap_max_drawdown(returns_low_vol, n_bootstrap=300, random_state=42)
        result_high = bootstrap_max_drawdown(returns_high_vol, n_bootstrap=300, random_state=42)

        assert result_high.estimate > result_low.estimate


class TestBootstrapAccuracy:
    """Tests for bootstrap_accuracy function."""

    def test_perfect_accuracy(self):
        """Perfect predictions should give 100% accuracy."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = y_true.copy()

        result = bootstrap_accuracy(y_true, y_pred, n_bootstrap=200, random_state=42)

        assert result.estimate == 1.0
        assert result.ci_lower == 1.0

    def test_random_accuracy(self):
        """Random predictions should give ~50% accuracy."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_pred = np.random.randint(0, 2, 200)

        result = bootstrap_accuracy(y_true, y_pred, n_bootstrap=300, random_state=42)

        assert 0.4 < result.estimate < 0.6

    def test_paired_resampling(self):
        """Should use paired resampling (same indices for true and pred)."""
        np.random.seed(42)
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 0, 1, 1])

        result = bootstrap_accuracy(y_true, y_pred, n_bootstrap=200, random_state=42)

        # 3/5 = 0.6 accuracy
        assert abs(result.estimate - 0.6) < 0.01

    def test_length_mismatch_raises(self):
        """Different length arrays should raise."""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1])

        with pytest.raises(ValueError, match="same length"):
            bootstrap_accuracy(y_true, y_pred)


class TestBootstrapF1Score:
    """Tests for bootstrap_f1_score function."""

    def test_perfect_f1(self):
        """Perfect predictions should give F1=1.0."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = y_true.copy()

        result = bootstrap_f1_score(y_true, y_pred, n_bootstrap=200, random_state=42)

        assert result.estimate == 1.0

    def test_different_averaging(self):
        """Test different averaging methods."""
        np.random.seed(42)
        y_true = np.random.randint(0, 3, 100)  # 3 classes
        y_pred = np.random.randint(0, 3, 100)

        result_macro = bootstrap_f1_score(
            y_true, y_pred, average="macro", n_bootstrap=200, random_state=42
        )
        result_weighted = bootstrap_f1_score(
            y_true, y_pred, average="weighted", n_bootstrap=200, random_state=42
        )

        # Both should produce valid results
        assert 0 <= result_macro.estimate <= 1
        assert 0 <= result_weighted.estimate <= 1


class TestBootstrapWinRate:
    """Tests for bootstrap_win_rate function."""

    def test_all_winners(self):
        """All positive returns should give 100% win rate."""
        trade_returns = np.array([0.01, 0.02, 0.005, 0.015, 0.008])

        result = bootstrap_win_rate(trade_returns, n_bootstrap=200, random_state=42)

        assert result.estimate == 100.0

    def test_all_losers(self):
        """All negative returns should give 0% win rate."""
        trade_returns = np.array([-0.01, -0.02, -0.005, -0.015, -0.008])

        result = bootstrap_win_rate(trade_returns, n_bootstrap=200, random_state=42)

        assert result.estimate == 0.0

    def test_mixed_trades(self):
        """Mixed results should give intermediate win rate."""
        np.random.seed(42)
        trade_returns = np.random.randn(100) * 0.02  # ~50% positive

        result = bootstrap_win_rate(trade_returns, n_bootstrap=300, random_state=42)

        assert 30 < result.estimate < 70


class TestBootstrapMultipleMetrics:
    """Tests for bootstrap_multiple_metrics function."""

    def test_multiple_metrics_computed(self):
        """Multiple metrics should be computed efficiently."""
        np.random.seed(42)
        data = np.random.randn(100) + 1

        metrics = {
            "mean": np.mean,
            "std": np.std,
            "median": np.median,
        }

        results = bootstrap_multiple_metrics(data, metrics, n_bootstrap=300, random_state=42)

        assert "mean" in results
        assert "std" in results
        assert "median" in results
        assert all(isinstance(r, BootstrapResult) for r in results.values())

    def test_shared_bootstrap_samples(self):
        """Same bootstrap samples should be used for consistency."""
        np.random.seed(42)
        data = np.random.randn(50)

        # With shared samples, correlated metrics should show correlated CIs
        metrics = {
            "mean": np.mean,
            "double_mean": lambda x: 2 * np.mean(x),
        }

        results = bootstrap_multiple_metrics(data, metrics, n_bootstrap=200, random_state=42)

        # Double mean estimate should be 2x mean estimate
        assert abs(results["double_mean"].estimate - 2 * results["mean"].estimate) < 1e-10

    def test_bca_method_multiple(self):
        """Test BCa method with multiple metrics."""
        np.random.seed(42)
        data = np.random.randn(50)

        metrics = {"mean": np.mean, "std": np.std}

        results = bootstrap_multiple_metrics(
            data, metrics, method="bca", n_bootstrap=200, random_state=42
        )

        assert results["mean"].method == "bca"
        assert results["std"].method == "bca"

    def test_handles_failed_metrics(self):
        """Should handle metrics that fail on some bootstrap samples."""

        def unstable_metric(x):
            if np.std(x) < 0.001:  # Very unlikely with random data
                raise ValueError("Unstable")
            return np.mean(x) / np.std(x)

        np.random.seed(42)
        data = np.random.randn(50)

        metrics = {"mean": np.mean, "unstable": unstable_metric}

        results = bootstrap_multiple_metrics(data, metrics, n_bootstrap=200, random_state=42)

        assert "mean" in results
        # unstable might or might not be present depending on bootstrap samples
