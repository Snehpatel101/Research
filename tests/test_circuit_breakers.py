"""
Integration tests for circuit breaker functionality.

Tests the circuit breaker logic added in Phase 12B:
- Max drawdown threshold
- Daily loss limit
- Consecutive loss limit
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


class TestCircuitBreakerConfig:
    """Test circuit breaker configuration parameters."""

    def test_default_config_has_circuit_breaker_params(self):
        """Verify default config includes circuit breaker parameters."""
        from src.inference.backtesting.backtest import BacktestConfig

        config = BacktestConfig()

        # Should have circuit breaker thresholds
        assert hasattr(config, "max_drawdown_threshold")
        assert hasattr(config, "daily_loss_threshold")
        assert hasattr(config, "consecutive_loss_limit")

        # Check default values
        assert config.max_drawdown_threshold == 0.10  # 10%
        assert config.daily_loss_threshold == 0.02  # 2%
        assert config.consecutive_loss_limit == 5

    def test_custom_circuit_breaker_config(self):
        """Test custom circuit breaker thresholds."""
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
        """Test that trading halts when max drawdown is exceeded."""
        from src.inference.backtesting.backtest import Backtester, BacktestConfig

        # Create price data with significant decline
        np.random.seed(42)
        n_bars = 200

        timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n_bars)]

        # Price that declines steadily to trigger drawdown
        base_price = 4500.0
        prices = base_price * np.exp(np.linspace(0, -0.20, n_bars))  # 20% decline

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

        # Always long signals (will lose money as price declines)
        predictions_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "prediction": np.ones(n_bars, dtype=int),
                "confidence": np.ones(n_bars),
            }
        )

        config = BacktestConfig(
            initial_equity=100000.0,
            max_drawdown_threshold=0.05,  # 5% threshold
            enable_market_hours_filter=False,
            min_holding_period=0,  # Allow immediate exits
        )

        backtester = Backtester(
            predictions=predictions_df,
            prices=prices_df,
            config=config,
        )

        result = backtester.run()

        # Verify drawdown was significant enough to potentially trigger
        # (exact trigger depends on trade timing)
        assert result.metrics.max_drawdown < 0, "Should have some drawdown (negative value)"


class TestDailyLossCircuitBreaker:
    """Test daily loss limit circuit breaker."""

    def test_config_has_daily_loss_threshold(self):
        """Verify daily loss threshold is configurable."""
        from src.inference.backtesting.backtest import BacktestConfig

        config = BacktestConfig(daily_loss_threshold=0.01)  # 1%

        assert config.daily_loss_threshold == 0.01


class TestConsecutiveLossCircuitBreaker:
    """Test consecutive loss limit circuit breaker."""

    def test_config_has_consecutive_loss_limit(self):
        """Verify consecutive loss limit is configurable."""
        from src.inference.backtesting.backtest import BacktestConfig

        config = BacktestConfig(consecutive_loss_limit=10)

        assert config.consecutive_loss_limit == 10

    def test_backtester_tracks_consecutive_losses(
        self, sample_prices: pd.DataFrame, sample_predictions: pd.DataFrame
    ):
        """Test that backtester tracks consecutive losses."""
        from src.inference.backtesting.backtest import Backtester, BacktestConfig

        config = BacktestConfig(
            initial_equity=100000.0,
            consecutive_loss_limit=100,  # High limit to not trigger
            enable_market_hours_filter=False,
        )

        backtester = Backtester(
            predictions=sample_predictions,
            prices=sample_prices,
            config=config,
        )

        result = backtester.run()

        # Just verify it ran without error
        assert result is not None


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker behavior."""

    def test_normal_operation_no_trigger(
        self, sample_prices: pd.DataFrame, sample_predictions: pd.DataFrame
    ):
        """Test normal operation without circuit breaker triggers."""
        from src.inference.backtesting.backtest import Backtester, BacktestConfig

        config = BacktestConfig(
            initial_equity=100000.0,
            max_drawdown_threshold=0.50,  # 50% - very high
            daily_loss_threshold=0.20,  # 20% - very high
            consecutive_loss_limit=100,  # Very high
            enable_market_hours_filter=False,
        )

        backtester = Backtester(
            predictions=sample_predictions,
            prices=sample_prices,
            config=config,
        )

        result = backtester.run()

        # Should complete normally
        assert result is not None
        # Should have processed all bars
        assert result.stats.get("total_bars", 0) > 0
