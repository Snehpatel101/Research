"""
Tests for regime-conditional evaluation module.

Tests cover:
- VolatilityRegime, TrendRegime, TimeOfDay enums
- RegimeClassifier (volatility, trend, time-of-day classification)
- RegimeEvaluator (per-regime metrics calculation)
- RegimeEvaluationResult dataclass
- Convenience functions (evaluate_regime_performance, get_regime_summary)
"""

import numpy as np
import pandas as pd
import pytest

from src.models.regime_evaluation import (
    RegimeClassifier,
    RegimeEvaluationResult,
    RegimeEvaluator,
    RegimeMetrics,
    TimeOfDay,
    TrendRegime,
    VolatilityRegime,
    evaluate_regime_performance,
    get_regime_summary,
)


class TestRegimeEnums:
    """Tests for regime enumeration values."""

    def test_volatility_regime_values(self):
        """VolatilityRegime should have expected values."""
        assert VolatilityRegime.LOW.value == "low"
        assert VolatilityRegime.NORMAL.value == "normal"
        assert VolatilityRegime.HIGH.value == "high"

    def test_trend_regime_values(self):
        """TrendRegime should have expected values."""
        assert TrendRegime.TRENDING_UP.value == "trending_up"
        assert TrendRegime.TRENDING_DOWN.value == "trending_down"
        assert TrendRegime.MEAN_REVERTING.value == "mean_reverting"

    def test_time_of_day_values(self):
        """TimeOfDay should have expected values."""
        assert TimeOfDay.ASIA.value == "asia"
        assert TimeOfDay.LONDON.value == "london"
        assert TimeOfDay.US_OPEN.value == "us_open"
        assert TimeOfDay.US_MIDDAY.value == "us_midday"
        assert TimeOfDay.US_CLOSE.value == "us_close"


class TestRegimeMetrics:
    """Tests for RegimeMetrics dataclass."""

    def test_to_dict(self):
        """to_dict() should return all metrics."""
        metrics = RegimeMetrics(
            regime_name="high",
            n_samples=100,
            accuracy=0.65,
            f1_score=0.62,
            sharpe_ratio=1.5,
            win_rate=0.55,
            avg_return=0.001,
            max_drawdown=0.05,
            precision=0.60,
            recall=0.65,
        )

        d = metrics.to_dict()
        assert d["regime_name"] == "high"
        assert d["n_samples"] == 100
        assert d["accuracy"] == 0.65
        assert d["sharpe_ratio"] == 1.5


class TestRegimeClassifier:
    """Tests for RegimeClassifier."""

    @pytest.fixture
    def classifier(self):
        """Create classifier with default settings."""
        return RegimeClassifier()

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns series."""
        np.random.seed(42)
        n = 500
        index = pd.date_range("2024-01-01", periods=n, freq="5min", tz="US/Eastern")
        returns = pd.Series(np.random.randn(n) * 0.001, index=index)
        return returns

    @pytest.fixture
    def sample_ohlc(self):
        """Create sample OHLC data."""
        np.random.seed(42)
        n = 500
        index = pd.date_range("2024-01-01", periods=n, freq="5min", tz="US/Eastern")

        # Generate random walk price
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        open_ = close + np.random.randn(n) * 0.1

        df = pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }, index=index)
        return df

    def test_invalid_vol_window_raises(self):
        """Should raise on invalid vol_window."""
        with pytest.raises(ValueError, match="vol_window must be >= 2"):
            RegimeClassifier(vol_window=1)

    def test_invalid_trend_window_raises(self):
        """Should raise on invalid trend_window."""
        with pytest.raises(ValueError, match="trend_window must be >= 2"):
            RegimeClassifier(trend_window=1)

    def test_invalid_vol_percentiles_raises(self):
        """Should raise on invalid vol_percentiles."""
        with pytest.raises(ValueError, match="vol_percentiles"):
            RegimeClassifier(vol_percentiles=(75, 25))  # Reversed

    def test_classify_volatility_returns_series(self, classifier, sample_returns):
        """classify_volatility should return Series."""
        regimes = classifier.classify_volatility(sample_returns)

        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(sample_returns)

    def test_classify_volatility_values(self, classifier, sample_returns):
        """Volatility regimes should be valid enum values."""
        regimes = classifier.classify_volatility(sample_returns)
        valid_values = {r.value for r in VolatilityRegime}

        # Drop NaN values from comparison
        non_nan_values = set(regimes.dropna().unique())
        assert non_nan_values.issubset(valid_values)

    def test_classify_volatility_insufficient_data(self, classifier):
        """Should handle insufficient data gracefully."""
        short_returns = pd.Series([0.001] * 5)  # Less than vol_window
        regimes = classifier.classify_volatility(short_returns)

        # Should return normal for all when insufficient data
        assert (regimes == VolatilityRegime.NORMAL.value).all()

    def test_classify_trend_returns_series(self, classifier, sample_ohlc):
        """classify_trend should return Series."""
        regimes = classifier.classify_trend(sample_ohlc)

        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(sample_ohlc)

    def test_classify_trend_values(self, classifier, sample_ohlc):
        """Trend regimes should be valid enum values."""
        regimes = classifier.classify_trend(sample_ohlc)
        valid_values = {r.value for r in TrendRegime}

        non_nan_values = set(regimes.dropna().unique())
        assert non_nan_values.issubset(valid_values)

    def test_classify_time_of_day_returns_series(self, classifier):
        """classify_time_of_day should return Series."""
        timestamps = pd.date_range(
            "2024-01-01 09:30", periods=100, freq="5min", tz="US/Eastern"
        )
        regimes = classifier.classify_time_of_day(timestamps)

        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(timestamps)

    def test_classify_time_of_day_us_open(self, classifier):
        """US open session should be classified correctly."""
        # 9:30 AM - 12:00 PM Eastern is US Open
        timestamps = pd.date_range(
            "2024-01-02 09:30", periods=10, freq="5min", tz="US/Eastern"
        )
        regimes = classifier.classify_time_of_day(timestamps)

        assert (regimes == TimeOfDay.US_OPEN.value).all()

    def test_classify_time_of_day_london(self, classifier):
        """London session should be classified correctly."""
        # 3:00 AM - 8:00 AM Eastern is London
        timestamps = pd.date_range(
            "2024-01-02 05:00", periods=10, freq="5min", tz="US/Eastern"
        )
        regimes = classifier.classify_time_of_day(timestamps)

        assert (regimes == TimeOfDay.LONDON.value).all()

    def test_classify_time_of_day_utc_conversion(self, classifier):
        """Should handle UTC timestamps correctly."""
        # 14:30 UTC = 9:30 AM Eastern (US Open)
        timestamps = pd.date_range(
            "2024-01-02 14:30", periods=10, freq="5min", tz="UTC"
        )
        regimes = classifier.classify_time_of_day(timestamps)

        assert (regimes == TimeOfDay.US_OPEN.value).all()


class TestRegimeEvaluator:
    """Tests for RegimeEvaluator."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator with default settings."""
        return RegimeEvaluator(min_samples_per_regime=10)

    @pytest.fixture
    def sample_data(self):
        """Create sample evaluation data."""
        np.random.seed(42)
        n = 500

        timestamps = pd.date_range(
            "2024-01-01 09:30", periods=n, freq="5min", tz="US/Eastern"
        )

        # Ground truth labels (-1, 0, 1)
        y_true = np.random.choice([-1, 0, 1], n)

        # Predictions (slightly correlated with truth)
        noise = np.random.choice([-1, 0, 1], n)
        y_pred = np.where(np.random.rand(n) > 0.3, y_true, noise)

        # Probabilities
        y_proba = np.random.rand(n, 3)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

        # Price data
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        open_ = close + np.random.randn(n) * 0.1

        prices = pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }, index=timestamps)

        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "prices": prices,
            "timestamps": timestamps,
        }

    def test_evaluate_returns_result(self, evaluator, sample_data):
        """evaluate() should return RegimeEvaluationResult."""
        result = evaluator.evaluate(**sample_data)

        assert isinstance(result, RegimeEvaluationResult)

    def test_evaluate_overall_metrics(self, evaluator, sample_data):
        """evaluate() should compute overall metrics."""
        result = evaluator.evaluate(**sample_data)

        assert "accuracy" in result.overall_metrics
        assert "f1_score" in result.overall_metrics
        assert "sharpe_ratio" in result.overall_metrics
        assert "win_rate" in result.overall_metrics

    def test_evaluate_volatility_breakdown(self, evaluator, sample_data):
        """evaluate() should compute volatility breakdown."""
        result = evaluator.evaluate(**sample_data)

        # Should have at least some volatility regimes
        assert len(result.volatility_breakdown) > 0

        for regime, metrics in result.volatility_breakdown.items():
            assert isinstance(metrics, RegimeMetrics)
            assert metrics.n_samples > 0

    def test_evaluate_trend_breakdown(self, evaluator, sample_data):
        """evaluate() should compute trend breakdown."""
        result = evaluator.evaluate(**sample_data)

        # Should have at least some trend regimes
        assert len(result.trend_breakdown) > 0

    def test_evaluate_time_of_day_breakdown(self, evaluator, sample_data):
        """evaluate() should compute time-of-day breakdown."""
        result = evaluator.evaluate(**sample_data)

        # Should have at least some time-of-day regimes
        assert len(result.time_of_day_breakdown) > 0

    def test_evaluate_length_mismatch_raises(self, evaluator, sample_data):
        """Should raise on length mismatch."""
        sample_data["y_pred"] = sample_data["y_pred"][:-10]  # Shorten

        with pytest.raises(ValueError, match="Length mismatch"):
            evaluator.evaluate(**sample_data)

    def test_evaluate_missing_close_raises(self, evaluator, sample_data):
        """Should raise if prices missing 'close' column."""
        sample_data["prices"] = sample_data["prices"].drop("close", axis=1)

        with pytest.raises(ValueError, match="close"):
            evaluator.evaluate(**sample_data)

    def test_identify_weak_regimes(self, evaluator, sample_data):
        """identify_weak_regimes() should find underperforming regimes."""
        result = evaluator.evaluate(**sample_data)
        weak = evaluator.identify_weak_regimes(
            result, accuracy_threshold=0.9, sharpe_threshold=5.0
        )

        # With high thresholds, most regimes should be weak
        assert len(weak) > 0

        for w in weak:
            assert "regime_type" in w
            assert "regime_name" in w
            assert "reason" in w


class TestRegimeEvaluationResult:
    """Tests for RegimeEvaluationResult dataclass."""

    @pytest.fixture
    def sample_result(self):
        """Create sample evaluation result."""
        return RegimeEvaluationResult(
            overall_metrics={
                "accuracy": 0.65,
                "f1_score": 0.62,
                "sharpe_ratio": 1.2,
                "win_rate": 0.55,
            },
            volatility_breakdown={
                "low": RegimeMetrics(
                    regime_name="low",
                    n_samples=100,
                    accuracy=0.70,
                    f1_score=0.68,
                    sharpe_ratio=1.5,
                    win_rate=0.60,
                    avg_return=0.001,
                    max_drawdown=0.03,
                ),
                "high": RegimeMetrics(
                    regime_name="high",
                    n_samples=80,
                    accuracy=0.40,
                    f1_score=0.35,
                    sharpe_ratio=-0.5,
                    win_rate=0.45,
                    avg_return=-0.001,
                    max_drawdown=0.08,
                ),
            },
            trend_breakdown={
                "trending_up": RegimeMetrics(
                    regime_name="trending_up",
                    n_samples=120,
                    accuracy=0.75,
                    f1_score=0.72,
                    sharpe_ratio=2.0,
                    win_rate=0.65,
                    avg_return=0.002,
                    max_drawdown=0.02,
                ),
            },
            time_of_day_breakdown={
                "us_open": RegimeMetrics(
                    regime_name="us_open",
                    n_samples=150,
                    accuracy=0.68,
                    f1_score=0.65,
                    sharpe_ratio=1.8,
                    win_rate=0.58,
                    avg_return=0.0015,
                    max_drawdown=0.04,
                ),
            },
            regime_correlations={
                "volatility_correlation": 0.15,
                "trend_correlation": 0.25,
            },
        )

    def test_to_dict(self, sample_result):
        """to_dict() should return nested dictionary."""
        d = sample_result.to_dict()

        assert "overall_metrics" in d
        assert "volatility_breakdown" in d
        assert "trend_breakdown" in d
        assert "time_of_day_breakdown" in d
        assert "regime_correlations" in d

        # Nested metrics should be dicts
        assert isinstance(d["volatility_breakdown"]["low"], dict)

    def test_to_dataframe(self, sample_result):
        """to_dataframe() should return comparison DataFrame."""
        df = sample_result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "regime_type" in df.columns
        assert "regime_name" in df.columns
        assert "accuracy" in df.columns

        # Should have rows for all regimes
        assert len(df) == 4  # 2 volatility + 1 trend + 1 time_of_day

    def test_get_weak_regimes(self, sample_result):
        """get_weak_regimes() should identify underperforming regimes."""
        weak = sample_result.get_weak_regimes(accuracy_threshold=0.5)

        assert len(weak) > 0
        assert "volatility:high" in weak  # accuracy 0.40 < 0.5


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for evaluation."""
        np.random.seed(42)
        n = 200

        timestamps = pd.date_range(
            "2024-01-01 09:30", periods=n, freq="5min", tz="US/Eastern"
        )
        y_true = np.random.choice([-1, 0, 1], n)
        y_pred = np.random.choice([-1, 0, 1], n)
        y_proba = np.random.rand(n, 3)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        prices = pd.DataFrame({
            "open": close + np.random.randn(n) * 0.1,
            "high": close + np.abs(np.random.randn(n) * 0.3),
            "low": close - np.abs(np.random.randn(n) * 0.3),
            "close": close,
        }, index=timestamps)

        return {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "prices": prices,
            "timestamps": timestamps,
        }

    def test_evaluate_regime_performance(self, sample_data):
        """evaluate_regime_performance() should work."""
        result = evaluate_regime_performance(**sample_data)

        assert isinstance(result, RegimeEvaluationResult)
        assert "accuracy" in result.overall_metrics

    def test_evaluate_regime_performance_with_config(self, sample_data):
        """Should accept classifier config."""
        result = evaluate_regime_performance(
            **sample_data,
            classifier_config={"vol_window": 10, "trend_window": 7},
        )

        assert isinstance(result, RegimeEvaluationResult)

    def test_get_regime_summary(self, sample_data):
        """get_regime_summary() should return formatted string."""
        result = evaluate_regime_performance(**sample_data)
        summary = get_regime_summary(result)

        assert isinstance(summary, str)
        assert "OVERALL PERFORMANCE" in summary
        assert "VOLATILITY REGIME BREAKDOWN" in summary
        assert "TREND REGIME BREAKDOWN" in summary
        assert "TIME-OF-DAY BREAKDOWN" in summary


class TestValidationIntegration:
    """Test that regime evaluation is accessible from validation module."""

    def test_import_from_validation(self):
        """Should be importable from src.validation."""
        from src.validation import (
            RegimeClassifier,
            RegimeEvaluationResult,
            RegimeEvaluator,
            RegimeMetrics,
            TimeOfDay,
            TrendRegime,
            VolatilityRegime,
            evaluate_regime_performance,
            get_regime_summary,
        )

        assert RegimeClassifier is not None
        assert RegimeEvaluator is not None
        assert VolatilityRegime is not None
        assert TrendRegime is not None
        assert TimeOfDay is not None
        assert callable(evaluate_regime_performance)
        assert callable(get_regime_summary)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_predictions(self):
        """Should handle empty data gracefully."""
        evaluator = RegimeEvaluator(min_samples_per_regime=1)

        timestamps = pd.date_range(
            "2024-01-01 09:30", periods=0, freq="5min", tz="US/Eastern"
        )

        with pytest.raises(ValueError):
            evaluator.evaluate(
                y_true=np.array([]),
                y_pred=np.array([]),
                y_proba=np.zeros((0, 3)),
                prices=pd.DataFrame({"close": []}, index=timestamps),
                timestamps=timestamps,
            )

    def test_constant_returns_regime(self):
        """Should handle constant returns (zero volatility)."""
        classifier = RegimeClassifier()
        returns = pd.Series(np.zeros(100))

        regimes = classifier.classify_volatility(returns)
        # Should not crash, may return NaN or normal
        assert len(regimes) == 100

    def test_single_sample_regime(self):
        """Should handle single sample per regime."""
        evaluator = RegimeEvaluator(min_samples_per_regime=1)

        np.random.seed(42)
        n = 50

        timestamps = pd.date_range(
            "2024-01-01 09:30", periods=n, freq="5min", tz="US/Eastern"
        )
        y_true = np.random.choice([-1, 0, 1], n)
        y_pred = np.random.choice([-1, 0, 1], n)
        y_proba = np.random.rand(n, 3)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        prices = pd.DataFrame({
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
        }, index=timestamps)

        # Should not crash even with small sample sizes
        result = evaluator.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            prices=prices,
            timestamps=timestamps,
        )

        assert isinstance(result, RegimeEvaluationResult)
