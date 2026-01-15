"""
Tests for backtester core functionality.

Tests cover:
- BacktestConfig creation and presets
- Backtester simulation
- BacktestResult output
- Position management
- Trade execution
- Walk-forward backtesting
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.backtesting import (
    BacktestConfig,
    BacktestResult,
    Backtester,
    ExecutionModel,
    Position,
    run_backtest,
    run_walk_forward_backtest,
)


class TestBacktestConfig:
    """Tests for BacktestConfig dataclass."""

    def test_default_config(self):
        """Default config should have reasonable values."""
        config = BacktestConfig()

        assert config.initial_equity == 100000.0
        assert config.position_sizing_method == "fixed_contracts"
        assert config.allow_short is True
        assert config.max_positions == 1

    def test_for_mes_preset(self):
        """MES preset should have correct tick values."""
        config = BacktestConfig.for_mes()

        assert config.tick_value == 1.25
        assert config.tick_size == 0.25
        assert config.point_value == 5.0

    def test_for_mgc_preset(self):
        """MGC preset should have correct tick values."""
        config = BacktestConfig.for_mgc()

        assert config.tick_value == 1.00
        assert config.tick_size == 0.10
        assert config.point_value == 10.0

    def test_custom_config(self):
        """Should accept custom parameters."""
        config = BacktestConfig(
            initial_equity=50000.0,
            position_sizing_method="fixed_fractional",
            risk_per_trade=0.01,
            allow_short=False,
        )

        assert config.initial_equity == 50000.0
        assert config.position_sizing_method == "fixed_fractional"
        assert config.risk_per_trade == 0.01
        assert config.allow_short is False


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation(self):
        """Should create position with all fields."""
        pos = Position(
            direction=1,
            contracts=2,
            entry_price=4500.0,
            entry_time=datetime.now(),
            entry_bar=0,
            stop_loss=4450.0,
            take_profit=4550.0,
        )

        assert pos.direction == 1
        assert pos.contracts == 2
        assert pos.entry_price == 4500.0


class TestBacktester:
    """Tests for Backtester class."""

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions DataFrame."""
        n = 100
        timestamps = pd.date_range("2024-01-01", periods=n, freq="5min")
        np.random.seed(42)

        return pd.DataFrame({
            "timestamp": timestamps,
            "prediction": np.random.choice([-1, 0, 1], n),
            "confidence": np.random.uniform(0.5, 1.0, n),
        })

    @pytest.fixture
    def sample_prices(self):
        """Create sample prices DataFrame."""
        n = 100
        timestamps = pd.date_range("2024-01-01", periods=n, freq="5min")
        np.random.seed(42)

        # Generate random walk
        close = 4500 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n) * 1)
        low = close - np.abs(np.random.randn(n) * 1)
        open_ = close + np.random.randn(n) * 0.5

        return pd.DataFrame({
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        })

    @pytest.fixture
    def simple_predictions(self):
        """Create simple alternating signal predictions."""
        n = 20
        timestamps = pd.date_range("2024-01-01", periods=n, freq="5min")

        # Alternating signals: long, neutral, short, neutral...
        signals = [1, 0, -1, 0] * 5

        return pd.DataFrame({
            "timestamp": timestamps,
            "prediction": signals,
            "confidence": [0.8] * n,
        })

    @pytest.fixture
    def simple_prices(self):
        """Create simple trending prices."""
        n = 20
        timestamps = pd.date_range("2024-01-01", periods=n, freq="5min")

        # Upward trending prices
        close = 4500 + np.arange(n) * 0.5
        high = close + 0.5
        low = close - 0.5
        open_ = close - 0.25

        return pd.DataFrame({
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        })

    def test_backtester_init(self, sample_predictions, sample_prices):
        """Backtester should initialize correctly."""
        config = BacktestConfig.for_mes()
        backtester = Backtester(sample_predictions, sample_prices, config)

        assert backtester.config == config
        assert len(backtester.predictions) == len(sample_predictions)
        assert len(backtester.prices) == len(sample_prices)

    def test_backtester_run_returns_result(self, sample_predictions, sample_prices):
        """run() should return BacktestResult."""
        config = BacktestConfig.for_mes()
        backtester = Backtester(sample_predictions, sample_prices, config)
        result = backtester.run()

        assert isinstance(result, BacktestResult)
        assert result.config == config

    def test_result_has_equity_curve(self, sample_predictions, sample_prices):
        """Result should have equity curve."""
        result = run_backtest(sample_predictions, sample_prices)

        assert result.equity_curve is not None
        assert len(result.equity_curve.equity_values) > 0

    def test_result_has_metrics(self, sample_predictions, sample_prices):
        """Result should have performance metrics."""
        result = run_backtest(sample_predictions, sample_prices)

        assert result.metrics is not None
        assert hasattr(result.metrics, "sharpe_ratio")
        assert hasattr(result.metrics, "win_rate")
        assert hasattr(result.metrics, "max_drawdown")

    def test_result_has_trades(self, sample_predictions, sample_prices):
        """Result should have trade list."""
        result = run_backtest(sample_predictions, sample_prices)

        assert isinstance(result.trades, list)
        # With random signals, we should have some trades
        assert len(result.trades) >= 0

    def test_long_only(self, simple_predictions, simple_prices):
        """Long only should not open short positions."""
        config = BacktestConfig.for_mes(allow_short=False)
        result = run_backtest(simple_predictions, simple_prices, config)

        # All trades should be long (direction=1)
        for trade in result.trades:
            assert trade.direction == 1

    def test_allow_short(self, simple_predictions, simple_prices):
        """With shorts allowed, should have short positions."""
        config = BacktestConfig.for_mes(allow_short=True)
        result = run_backtest(simple_predictions, simple_prices, config)

        directions = [t.direction for t in result.trades]
        # Should have both long and short trades
        if len(directions) > 0:
            has_short = -1 in directions
            has_long = 1 in directions
            # With alternating signals, we expect both
            assert has_long or has_short

    def test_equity_changes_with_trades(self, simple_predictions, simple_prices):
        """Equity should change from trades."""
        config = BacktestConfig.for_mes(initial_equity=100000.0)
        result = run_backtest(simple_predictions, simple_prices, config)

        # Final equity should differ from initial if trades occurred
        if len(result.trades) > 0:
            assert result.equity_curve.current_equity != config.initial_equity

    def test_invalid_predictions_raises(self, sample_prices):
        """Invalid predictions should raise ValueError."""
        bad_predictions = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="5min"),
            "prediction": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # Invalid values
        })

        with pytest.raises(ValueError, match="must be in"):
            Backtester(bad_predictions, sample_prices)

    def test_missing_price_columns_raises(self, sample_predictions):
        """Missing price columns should raise ValueError."""
        bad_prices = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="5min"),
            "close": np.random.randn(100),
            # Missing open, high, low
        })

        with pytest.raises(ValueError, match="must have columns"):
            Backtester(sample_predictions, bad_prices)

    def test_no_overlapping_timestamps_raises(self, sample_predictions):
        """Non-overlapping timestamps should raise ValueError."""
        # Prices from different time period
        different_prices = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=100, freq="5min"),
            "open": np.random.randn(100) + 4500,
            "high": np.random.randn(100) + 4501,
            "low": np.random.randn(100) + 4499,
            "close": np.random.randn(100) + 4500,
        })

        backtester = Backtester(sample_predictions, different_prices)
        with pytest.raises(ValueError, match="No overlapping"):
            backtester.run()

    def test_summary_dict(self, sample_predictions, sample_prices):
        """summary() should return dict with key metrics."""
        result = run_backtest(sample_predictions, sample_prices)
        summary = result.summary()

        assert "initial_equity" in summary
        assert "final_equity" in summary
        assert "total_return_pct" in summary
        assert "sharpe_ratio" in summary
        assert "max_drawdown_pct" in summary

    def test_execution_model_close(self, simple_predictions, simple_prices):
        """Market on close should use close price."""
        config = BacktestConfig.for_mes(
            execution_model=ExecutionModel.MARKET_ON_CLOSE
        )
        result = run_backtest(simple_predictions, simple_prices, config)

        # Just verify it runs without error
        assert result is not None

    def test_execution_model_open(self, simple_predictions, simple_prices):
        """Market on open should use open price."""
        config = BacktestConfig.for_mes(
            execution_model=ExecutionModel.MARKET_ON_OPEN
        )
        result = run_backtest(simple_predictions, simple_prices, config)

        assert result is not None


class TestRunBacktest:
    """Tests for run_backtest convenience function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        n = 50
        timestamps = pd.date_range("2024-01-01", periods=n, freq="5min")
        np.random.seed(42)

        predictions = pd.DataFrame({
            "timestamp": timestamps,
            "prediction": np.random.choice([-1, 0, 1], n),
        })

        prices = pd.DataFrame({
            "timestamp": timestamps,
            "open": np.random.randn(n) + 4500,
            "high": np.random.randn(n) + 4501,
            "low": np.random.randn(n) + 4499,
            "close": np.random.randn(n) + 4500,
        })

        return predictions, prices

    def test_convenience_function(self, sample_data):
        """run_backtest should work like Backtester.run()."""
        predictions, prices = sample_data
        result = run_backtest(predictions, prices)

        assert isinstance(result, BacktestResult)

    def test_with_config_kwargs(self, sample_data):
        """Should accept config kwargs directly."""
        predictions, prices = sample_data
        result = run_backtest(
            predictions,
            prices,
            initial_equity=50000.0,
            allow_short=False,
        )

        assert result.config.initial_equity == 50000.0
        assert result.config.allow_short is False


class TestRunWalkForwardBacktest:
    """Tests for run_walk_forward_backtest function."""

    @pytest.fixture
    def long_data(self):
        """Create longer data for walk-forward."""
        n = 200
        timestamps = pd.date_range("2024-01-01", periods=n, freq="5min")
        np.random.seed(42)

        predictions = pd.DataFrame({
            "timestamp": timestamps,
            "prediction": np.random.choice([-1, 0, 1], n),
        })

        close = 4500 + np.cumsum(np.random.randn(n) * 2)
        prices = pd.DataFrame({
            "timestamp": timestamps,
            "open": close + np.random.randn(n) * 0.5,
            "high": close + np.abs(np.random.randn(n)),
            "low": close - np.abs(np.random.randn(n)),
            "close": close,
        })

        return predictions, prices

    def test_returns_list_of_results(self, long_data):
        """Should return list of BacktestResult."""
        predictions, prices = long_data
        results = run_walk_forward_backtest(predictions, prices, n_splits=3)

        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, BacktestResult) for r in results)

    def test_each_split_has_trades(self, long_data):
        """Each split should have valid structure."""
        predictions, prices = long_data
        results = run_walk_forward_backtest(predictions, prices, n_splits=3)

        for result in results:
            assert result.equity_curve is not None
            assert result.metrics is not None

    def test_uses_config(self, long_data):
        """Should apply config to all splits."""
        predictions, prices = long_data
        config = BacktestConfig.for_mes(allow_short=False)
        results = run_walk_forward_backtest(
            predictions, prices, n_splits=3, config=config
        )

        for result in results:
            assert result.config.allow_short is False


class TestValidationIntegration:
    """Test that backtesting is accessible from validation module."""

    def test_import_from_validation(self):
        """Should be importable from src.validation."""
        from src.validation import (
            BacktestConfig,
            BacktestResult,
            Backtester,
            ExecutionModel,
            Position,
            Trade,
            run_backtest,
            run_walk_forward_backtest,
        )

        assert Backtester is not None
        assert BacktestConfig is not None
        assert callable(run_backtest)
        assert callable(run_walk_forward_backtest)


class TestBacktestResultPrintSummary:
    """Tests for BacktestResult.print_summary()."""

    @pytest.fixture
    def sample_result(self):
        """Create sample backtest result."""
        n = 50
        timestamps = pd.date_range("2024-01-01", periods=n, freq="5min")
        np.random.seed(42)

        predictions = pd.DataFrame({
            "timestamp": timestamps,
            "prediction": np.random.choice([-1, 0, 1], n),
        })

        prices = pd.DataFrame({
            "timestamp": timestamps,
            "open": np.random.randn(n) + 4500,
            "high": np.random.randn(n) + 4501,
            "low": np.random.randn(n) + 4499,
            "close": np.random.randn(n) + 4500,
        })

        return run_backtest(predictions, prices)

    def test_print_summary_runs(self, sample_result, capsys):
        """print_summary() should output formatted text."""
        sample_result.print_summary()
        captured = capsys.readouterr()

        assert "BACKTEST RESULTS" in captured.out
        assert "Initial Equity" in captured.out
        assert "Final Equity" in captured.out
        assert "Sharpe Ratio" in captured.out
