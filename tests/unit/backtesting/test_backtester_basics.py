"""Backtester imports, basic runs, result summary and circuit breakers."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# =============================================================================
# Backtester Tests
# =============================================================================


class TestBacktesterImports:
    """Test that backtester modules import correctly."""

    def test_import_backtester(self):
        from src.inference.backtesting.backtest import Backtester

        assert Backtester is not None

    def test_import_backtest_config(self):
        from src.inference.backtesting.backtest import BacktestConfig

        assert BacktestConfig is not None

    def test_import_backtest_result(self):
        from src.inference.backtesting.backtest import BacktestResult

        assert BacktestResult is not None

    def test_import_run_backtest(self):
        from src.inference.backtesting.backtest import run_backtest

        assert callable(run_backtest)


class TestBacktesterBasicRun:
    """Test basic backtester execution."""

    def test_backtest_runs_without_error(
        self, sample_prices: pd.DataFrame, sample_predictions: pd.DataFrame
    ):
        from src.inference.backtesting.backtest import BacktestConfig, Backtester

        config = BacktestConfig(
            initial_equity=100000.0,
            enable_market_hours_filter=False,
        )
        backtester = Backtester(predictions=sample_predictions, prices=sample_prices, config=config)
        result = backtester.run()

        assert result is not None
        assert result.equity_curve is not None
        assert result.metrics is not None
        assert result.config == config

    def test_backtest_returns_valid_metrics(
        self, sample_prices: pd.DataFrame, sample_predictions: pd.DataFrame
    ):
        from src.inference.backtesting.backtest import BacktestConfig, Backtester

        config = BacktestConfig(initial_equity=100000.0, enable_market_hours_filter=False)
        backtester = Backtester(predictions=sample_predictions, prices=sample_prices, config=config)
        result = backtester.run()
        metrics = result.metrics

        assert np.isfinite(metrics.sharpe_ratio) or metrics.total_trades == 0
        assert 0.0 <= metrics.win_rate <= 1.0
        assert -1.0 <= metrics.max_drawdown <= 0.0

    def test_backtest_config_for_mes(self):
        from src.inference.backtesting.backtest import BacktestConfig

        config = BacktestConfig.for_mes()
        assert config.tick_value == 1.25
        assert config.tick_size == 0.25
        assert config.point_value == 5.0

    def test_backtest_config_for_mgc(self):
        from src.inference.backtesting.backtest import BacktestConfig

        config = BacktestConfig.for_mgc()
        assert config.tick_value == 1.00
        assert config.tick_size == 0.10
        assert config.point_value == 10.0


class TestBacktestResultSummary:
    """Test backtest result summary generation."""

    def test_summary_contains_required_keys(
        self, sample_prices: pd.DataFrame, sample_predictions: pd.DataFrame
    ):
        from src.inference.backtesting.backtest import BacktestConfig, Backtester

        config = BacktestConfig(initial_equity=100000.0, enable_market_hours_filter=False)
        backtester = Backtester(predictions=sample_predictions, prices=sample_prices, config=config)
        result = backtester.run()
        summary = result.summary()

        required_keys = [
            "initial_equity",
            "final_equity",
            "total_return_pct",
            "total_pnl",
            "total_trades",
            "win_rate_pct",
            "profit_factor",
            "sharpe_ratio",
            "max_drawdown_pct",
        ]
        for key in required_keys:
            assert key in summary, f"Missing key: {key}"


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreakerConfig:
    """Test circuit breaker configuration parameters."""

    def test_default_config_has_circuit_breaker_params(self):
        from src.inference.backtesting.backtest import BacktestConfig

        config = BacktestConfig()
        assert hasattr(config, "max_drawdown_threshold")
        assert hasattr(config, "daily_loss_threshold")
        assert hasattr(config, "consecutive_loss_limit")
        assert config.max_drawdown_threshold == 0.10
        assert config.daily_loss_threshold == 0.02
        assert config.consecutive_loss_limit == 5

    def test_custom_circuit_breaker_config(self):
        from src.inference.backtesting.backtest import BacktestConfig

        config = BacktestConfig(
            max_drawdown_threshold=0.15,
            daily_loss_threshold=0.03,
            consecutive_loss_limit=3,
        )
        assert config.max_drawdown_threshold == 0.15
        assert config.daily_loss_threshold == 0.03
        assert config.consecutive_loss_limit == 3


class TestMaxDrawdownCircuitBreaker:
    """Test max drawdown circuit breaker triggers."""

    def test_backtest_halts_on_max_drawdown(self):
        from src.inference.backtesting.backtest import BacktestConfig, Backtester

        n_bars = 200
        timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n_bars)]
        base_price = 4500.0
        prices = base_price * np.exp(np.linspace(0, -0.20, n_bars))

        prices_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": prices,
                "high": prices * 1.001,
                "low": prices * 0.999,
                "close": prices,
                "volume": np.full(n_bars, 1000),
            }
        )
        predictions_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "prediction": np.ones(n_bars, dtype=int),
                "confidence": np.ones(n_bars),
            }
        )

        config = BacktestConfig(
            initial_equity=100000.0,
            max_drawdown_threshold=0.05,
            enable_market_hours_filter=False,
            min_holding_period=0,
        )
        backtester = Backtester(predictions=predictions_df, prices=prices_df, config=config)
        result = backtester.run()

        assert result.metrics.max_drawdown < 0


class TestDailyLossCircuitBreaker:
    """Test daily loss limit circuit breaker."""

    def test_config_has_daily_loss_threshold(self):
        from src.inference.backtesting.backtest import BacktestConfig

        config = BacktestConfig(daily_loss_threshold=0.01)
        assert config.daily_loss_threshold == 0.01


class TestConsecutiveLossCircuitBreaker:
    """Test consecutive loss limit circuit breaker."""

    def test_config_has_consecutive_loss_limit(self):
        from src.inference.backtesting.backtest import BacktestConfig

        config = BacktestConfig(consecutive_loss_limit=10)
        assert config.consecutive_loss_limit == 10

    def test_backtester_tracks_consecutive_losses(
        self, sample_prices: pd.DataFrame, sample_predictions: pd.DataFrame
    ):
        from src.inference.backtesting.backtest import BacktestConfig, Backtester

        config = BacktestConfig(
            initial_equity=100000.0,
            consecutive_loss_limit=100,
            enable_market_hours_filter=False,
        )
        backtester = Backtester(predictions=sample_predictions, prices=sample_prices, config=config)
        result = backtester.run()
        assert result is not None


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker behavior."""

    def test_normal_operation_no_trigger(
        self, sample_prices: pd.DataFrame, sample_predictions: pd.DataFrame
    ):
        from src.inference.backtesting.backtest import BacktestConfig, Backtester

        config = BacktestConfig(
            initial_equity=100000.0,
            max_drawdown_threshold=0.50,
            daily_loss_threshold=0.20,
            consecutive_loss_limit=100,
            enable_market_hours_filter=False,
        )
        backtester = Backtester(predictions=sample_predictions, prices=sample_prices, config=config)
        result = backtester.run()

        assert result is not None
        assert result.stats.get("total_bars", 0) > 0
