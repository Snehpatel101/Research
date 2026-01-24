"""
Smoke tests for Backtester.

Minimal tests to verify the backtester imports and runs without errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestBacktesterImports:
    """Test that backtester modules import correctly."""

    def test_import_backtester(self):
        """Verify Backtester can be imported."""
        from src.inference.backtesting.backtest import Backtester

        assert Backtester is not None

    def test_import_backtest_config(self):
        """Verify BacktestConfig can be imported."""
        from src.inference.backtesting.backtest import BacktestConfig

        assert BacktestConfig is not None

    def test_import_backtest_result(self):
        """Verify BacktestResult can be imported."""
        from src.inference.backtesting.backtest import BacktestResult

        assert BacktestResult is not None

    def test_import_run_backtest(self):
        """Verify run_backtest convenience function imports."""
        from src.inference.backtesting.backtest import run_backtest

        assert callable(run_backtest)


class TestBacktesterBasicRun:
    """Test basic backtester execution."""

    def test_backtest_runs_without_error(
        self, sample_prices: pd.DataFrame, sample_predictions: pd.DataFrame
    ):
        """Verify backtester runs end-to-end without exceptions."""
        from src.inference.backtesting.backtest import Backtester, BacktestConfig

        config = BacktestConfig(
            initial_equity=100000.0,
            enable_market_hours_filter=False,  # Disable for test
        )

        backtester = Backtester(
            predictions=sample_predictions,
            prices=sample_prices,
            config=config,
        )

        result = backtester.run()

        # Basic sanity checks
        assert result is not None
        assert result.equity_curve is not None
        assert result.metrics is not None
        assert result.config == config

    def test_backtest_returns_valid_metrics(
        self, sample_prices: pd.DataFrame, sample_predictions: pd.DataFrame
    ):
        """Verify backtester returns valid performance metrics."""
        from src.inference.backtesting.backtest import Backtester, BacktestConfig

        config = BacktestConfig(
            initial_equity=100000.0,
            enable_market_hours_filter=False,
        )

        backtester = Backtester(
            predictions=sample_predictions,
            prices=sample_prices,
            config=config,
        )

        result = backtester.run()
        metrics = result.metrics

        # Metrics should be finite numbers
        assert np.isfinite(metrics.sharpe_ratio) or metrics.total_trades == 0
        assert metrics.win_rate >= 0.0
        assert metrics.win_rate <= 1.0
        assert metrics.max_drawdown >= 0.0
        assert metrics.max_drawdown <= 1.0

    def test_backtest_config_for_mes(self):
        """Test MES-specific configuration factory."""
        from src.inference.backtesting.backtest import BacktestConfig

        config = BacktestConfig.for_mes()

        assert config.tick_value == 1.25
        assert config.tick_size == 0.25
        assert config.point_value == 5.0

    def test_backtest_config_for_mgc(self):
        """Test MGC-specific configuration factory."""
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
        """Verify summary dict contains all required keys."""
        from src.inference.backtesting.backtest import Backtester, BacktestConfig

        config = BacktestConfig(
            initial_equity=100000.0,
            enable_market_hours_filter=False,
        )

        backtester = Backtester(
            predictions=sample_predictions,
            prices=sample_prices,
            config=config,
        )

        result = backtester.run()
        summary = result.summary()

        # Required keys
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
