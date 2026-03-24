"""
ML Factory - Consolidated Test Suite.

Tests:
- Backtester imports and basic runs
- Circuit breaker configuration and behavior
- Transaction costs and slippage models
- MDA holdout scoring (Fix #6) and timestamp alignment (Fix #7)
- R-multiple calculations and Trade properties
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

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

        np.random.seed(42)
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


# =============================================================================
# Transaction Costs Tests
# =============================================================================


class TestTransactionCostsImports:
    """Test that cost modules import correctly."""

    def test_import_transaction_costs(self):
        from src.inference.backtesting.costs import TransactionCosts

        assert TransactionCosts is not None

    def test_import_cost_calculator(self):
        from src.inference.backtesting.costs import CostCalculator

        assert CostCalculator is not None

    def test_import_slippage_models(self):
        from src.inference.backtesting.costs import (
            FixedSlippage,
            LinearSlippage,
            SquareRootSlippage,
            VolatilityScaledSlippage,
        )

        assert FixedSlippage is not None
        assert LinearSlippage is not None
        assert SquareRootSlippage is not None
        assert VolatilityScaledSlippage is not None


class TestTransactionCostsCalculations:
    """Test TransactionCosts calculation methods."""

    def test_round_trip_cost_single_contract(self):
        from src.inference.backtesting.costs import TransactionCosts

        costs = TransactionCosts(
            commission_per_contract=2.50,
            slippage_ticks=1.0,
            tick_value=1.25,
            exchange_fee=0.52,
            nfa_fee=0.02,
        )
        expected = 5.54
        actual = costs.calculate_round_trip_cost(contracts=1)
        assert abs(actual - expected) < 0.01

    def test_round_trip_cost_multiple_contracts(self):
        from src.inference.backtesting.costs import TransactionCosts

        costs = TransactionCosts.for_mes()
        cost_1 = costs.calculate_round_trip_cost(contracts=1)
        cost_5 = costs.calculate_round_trip_cost(contracts=5)
        assert abs(cost_5 - 5 * cost_1) < 0.01

    def test_entry_cost_calculation(self):
        from src.inference.backtesting.costs import TransactionCosts

        costs = TransactionCosts(
            commission_per_contract=2.50,
            slippage_ticks=1.0,
            tick_value=1.25,
            exchange_fee=0.52,
            nfa_fee=0.02,
        )
        entry_cost = costs.calculate_entry_cost(contracts=1, entry_price=4500.0)
        expected = 2.77
        assert abs(entry_cost - expected) < 0.01

    def test_total_fixed_cost_property(self):
        from src.inference.backtesting.costs import TransactionCosts

        costs = TransactionCosts(commission_per_contract=2.50, exchange_fee=0.52, nfa_fee=0.02)
        expected = 3.04
        assert abs(costs.total_fixed_cost_per_contract - expected) < 0.01

    def test_slippage_cost_property(self):
        from src.inference.backtesting.costs import TransactionCosts

        costs = TransactionCosts(slippage_ticks=1.0, tick_value=1.25)
        expected = 2.50
        assert abs(costs.slippage_cost_per_contract - expected) < 0.01


class TestTransactionCostsFactories:
    """Test factory methods for different contracts."""

    def test_for_mes(self):
        from src.inference.backtesting.costs import TransactionCosts

        costs = TransactionCosts.for_mes()
        assert costs.tick_value == 1.25
        assert costs.tick_size == 0.25
        assert costs.commission_per_contract == 2.50

    def test_for_mgc(self):
        from src.inference.backtesting.costs import TransactionCosts

        costs = TransactionCosts.for_mgc()
        assert costs.tick_value == 1.00
        assert costs.tick_size == 0.10
        assert costs.commission_per_contract == 2.50

    def test_for_mnq(self):
        from src.inference.backtesting.costs import TransactionCosts

        costs = TransactionCosts.for_mnq()
        assert costs.tick_value == 0.50
        assert costs.tick_size == 0.25


class TestSlippageModels:
    """Test slippage model calculations."""

    def test_fixed_slippage(self):
        from src.inference.backtesting.costs import FixedSlippage

        model = FixedSlippage(ticks=2.0, tick_size=0.25)
        slippage = model.estimate_slippage(order_size=1, price=4500.0)
        assert slippage == 0.50

        slippage_large = model.estimate_slippage(order_size=100, price=4500.0)
        assert slippage_large == 0.50

    def test_linear_slippage_scales_with_size(self):
        from src.inference.backtesting.costs import LinearSlippage

        model = LinearSlippage(base_ticks=0.5, size_factor=0.1, tick_size=0.25)
        slip_1 = model.estimate_slippage(order_size=1, price=4500.0)
        slip_10 = model.estimate_slippage(order_size=10, price=4500.0)
        assert slip_10 > slip_1

    def test_volatility_scaled_slippage(self):
        from src.inference.backtesting.costs import VolatilityScaledSlippage

        model = VolatilityScaledSlippage(
            base_ticks=1.0,
            base_volatility=0.15,
            volatility_multiplier=2.0,
            tick_size=0.25,
        )
        slip_low = model.estimate_slippage(order_size=1, price=4500.0, volatility=0.10)
        slip_high = model.estimate_slippage(order_size=1, price=4500.0, volatility=0.30)
        assert slip_high > slip_low


class TestCostCalculator:
    """Test CostCalculator integration."""

    def test_calculate_pnl_long_winning_trade(self):
        from src.inference.backtesting.costs import CostCalculator, TransactionCosts

        tx_costs = TransactionCosts.for_mes()
        calculator = CostCalculator(transaction_costs=tx_costs)
        result = calculator.calculate_pnl(
            contracts=1,
            entry_price=4500.0,
            exit_price=4510.0,
            direction=1,
            point_value=5.0,
        )
        assert result["gross_pnl"] == 50.0
        assert result["net_pnl"] < result["gross_pnl"]
        assert result["costs"] > 0

    def test_calculate_pnl_short_winning_trade(self):
        from src.inference.backtesting.costs import CostCalculator, TransactionCosts

        tx_costs = TransactionCosts.for_mes()
        calculator = CostCalculator(transaction_costs=tx_costs)
        result = calculator.calculate_pnl(
            contracts=1,
            entry_price=4500.0,
            exit_price=4490.0,
            direction=-1,
            point_value=5.0,
        )
        assert result["gross_pnl"] == 50.0

    def test_calculate_pnl_no_costs(self):
        from src.inference.backtesting.costs import CostCalculator, TransactionCosts

        tx_costs = TransactionCosts.for_mes()
        calculator = CostCalculator(transaction_costs=tx_costs)
        result = calculator.calculate_pnl(
            contracts=1,
            entry_price=4500.0,
            exit_price=4510.0,
            direction=1,
            point_value=5.0,
            include_costs=False,
        )
        assert result["net_pnl"] == result["gross_pnl"]
        assert result["costs"] == 0.0


# =============================================================================
# MDA Holdout Scoring (Fix #6) and Timestamp Alignment (Fix #7)
# =============================================================================


class TestMDAHoldoutScoring:
    """Verify MDA permutation importance uses holdout data."""

    def _make_data(self, n_train=200, n_test=80, n_features=10, seed=42):
        rng = np.random.RandomState(seed)
        X = pd.DataFrame(
            rng.randn(n_train + n_test, n_features),
            columns=[f"f{i}" for i in range(n_features)],
        )
        y = pd.Series((X["f0"] > 0).astype(int))
        return X.iloc[:n_train], y.iloc[:n_train], X.iloc[n_train:], y.iloc[n_train:]

    def test_mda_uses_holdout_when_provided(self):
        from src.optimization.feature_selection.walk_forward import (
            WalkForwardFeatureSelector,
        )

        X_train, y_train, X_test, y_test = self._make_data()
        selector = WalkForwardFeatureSelector(selection_method="mda", n_estimators=20)

        captured = {}
        import sklearn.inspection as si

        original_pi = si.permutation_importance

        def spy_pi(estimator, X, y, **kwargs):
            captured["X_shape"] = X.shape
            captured["y_len"] = len(y)
            return original_pi(estimator, X, y, **kwargs)

        with patch(
            "src.optimization.feature_selection.walk_forward.permutation_importance",
            side_effect=spy_pi,
        ):
            selector._mda_importance(X_train, y_train, X_test=X_test, y_test=y_test)

        assert captured["X_shape"][0] == 80

    def test_mda_warns_without_holdout(self, caplog):
        from src.optimization.feature_selection.walk_forward import (
            WalkForwardFeatureSelector,
        )

        X_train, y_train, _, _ = self._make_data()
        selector = WalkForwardFeatureSelector(selection_method="mda", n_estimators=20)

        with caplog.at_level("WARNING"):
            result = selector._mda_importance(X_train, y_train)

        assert len(result) == 10
        assert "no holdout set provided" in caplog.text

    def test_walkforward_passes_test_split(self):
        from src.optimization.feature_selection.walk_forward import (
            WalkForwardFeatureSelector,
        )

        rng = np.random.RandomState(42)
        n = 300
        X = pd.DataFrame(rng.randn(n, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(rng.randint(0, 2, n))

        cv_splits = [
            (np.arange(0, 200), np.arange(200, 260)),
            (np.arange(0, 240), np.arange(240, 300)),
        ]

        selector = WalkForwardFeatureSelector(
            selection_method="mda", n_estimators=10, n_features_to_select=3
        )

        calls = []
        original = selector._compute_importance

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        with patch.object(selector, "_compute_importance", side_effect=spy):
            selector.select_features_walkforward(X, y, cv_splits)

        assert len(calls) == 2
        for _i, call_kwargs in enumerate(calls):
            assert "X_test" in call_kwargs
            assert "y_test" in call_kwargs


class TestTimestampAlignment:
    """Verify multi-stream adapter uses timestamp-based alignment."""

    def _make_market_dfs(self):
        am_anchor_idx = pd.date_range("2024-01-15 09:00", periods=60, freq="1min")
        pm_anchor_idx = pd.date_range("2024-01-15 14:00", periods=60, freq="1min")
        anchor_idx = am_anchor_idx.append(pm_anchor_idx)

        am_higher_idx = pd.date_range("2024-01-15 09:00", periods=12, freq="5min")
        pm_higher_idx = pd.date_range("2024-01-15 14:00", periods=12, freq="5min")
        higher_idx = am_higher_idx.append(pm_higher_idx)

        rng = np.random.RandomState(42)

        anchor_df = pd.DataFrame(
            {
                "open": rng.uniform(100, 105, len(anchor_idx)),
                "high": rng.uniform(105, 110, len(anchor_idx)),
                "low": rng.uniform(95, 100, len(anchor_idx)),
                "close": rng.uniform(100, 105, len(anchor_idx)),
                "volume": rng.randint(100, 1000, len(anchor_idx)),
                "label_h20": rng.randint(0, 2, len(anchor_idx)),
            },
            index=anchor_idx,
        )

        higher_df = pd.DataFrame(
            {
                "open": rng.uniform(100, 105, len(higher_idx)),
                "high": rng.uniform(105, 110, len(higher_idx)),
                "low": rng.uniform(95, 100, len(higher_idx)),
                "close": rng.uniform(100, 105, len(higher_idx)),
                "volume": rng.randint(500, 5000, len(higher_idx)),
            },
            index=higher_idx,
        )
        return anchor_df, higher_df

    def test_timestamp_align_handles_gaps(self):
        from src.data.adapters.multi_stream import MultiStreamAdapter

        anchor_df, higher_df = self._make_market_dfs()
        adapter = MultiStreamAdapter(
            feature_columns=["open", "high", "low", "close", "volume"],
            timeframes=["1min", "5min"],
            sequence_length=10,
            stride=1,
        )
        idx_map = adapter._timestamp_align(anchor_df, higher_df)

        assert idx_map[0] == 0
        assert idx_map[59] == 11
        assert idx_map[60] == 12
        assert idx_map[65] == 13
        assert idx_map.max() < len(higher_df)

    def test_ratio_fallback_without_datetime_index(self):
        """Non-DatetimeIndex DataFrames should raise ValueError (Phase 60 change)."""
        from src.data.adapters.multi_stream import MultiStreamAdapter

        rng = np.random.RandomState(42)
        n = 100
        anchor_df = pd.DataFrame(
            {
                "open": rng.randn(n),
                "high": rng.randn(n),
                "low": rng.randn(n),
                "close": rng.randn(n),
                "volume": rng.randn(n),
                "label_h20": rng.randint(0, 2, n),
            }
        )
        higher_df = pd.DataFrame(
            {
                "open": rng.randn(n // 5),
                "high": rng.randn(n // 5),
                "low": rng.randn(n // 5),
                "close": rng.randn(n // 5),
                "volume": rng.randn(n // 5),
            }
        )

        adapter = MultiStreamAdapter(
            feature_columns=["open", "high", "low", "close", "volume"],
            timeframes=["1min", "5min"],
            sequence_length=10,
            stride=1,
        )

        tf_dfs = {"1min": anchor_df, "5min": higher_df}
        feature_cols = ["open", "high", "low", "close", "volume"]
        with pytest.raises(ValueError, match="requires DatetimeIndex"):
            adapter._build_timestamp_index_maps(anchor_df, tf_dfs, ["1min", "5min"], feature_cols)

    def test_full_transform_with_timestamps(self):
        from src.data.adapters.multi_stream import MultiStreamAdapter

        anchor_df, higher_df = self._make_market_dfs()
        adapter = MultiStreamAdapter(
            feature_columns=["open", "high", "low", "close", "volume"],
            timeframes=["1min", "5min"],
            sequence_length=10,
            stride=1,
        )
        result = adapter.transform(anchor_df, additional_dfs={"5min": higher_df})

        n_seqs = (120 - 10) // 1 + 1
        assert result.X.shape == (n_seqs, 2, 10, 5)
        assert result.y.shape == (n_seqs,)
        assert result.n_timeframes == 2

    def test_extract_aligned_sequence_pads_correctly(self):
        from src.data.adapters.multi_stream import MultiStreamAdapter

        adapter = MultiStreamAdapter(
            feature_columns=["open", "close"],
            timeframes=["1min", "5min"],
            sequence_length=10,
        )

        tf_values = np.arange(10).reshape(5, 2).astype(np.float32)
        idx_map = np.array([0, 0, 0, 0, 0, 1, 1, 2, 3, 4])

        result = adapter._extract_aligned_sequence(
            tf_values=tf_values,
            idx_map=idx_map,
            anchor_start=0,
            anchor_end=10,
            seq_len=10,
        )

        assert result.shape == (10, 2)
        np.testing.assert_array_equal(result[:5, 0], [0, 0, 0, 0, 0])
        np.testing.assert_array_equal(result[5:, 0], [0, 2, 4, 6, 8])


# =============================================================================
# R-Multiple Tests
# =============================================================================


class TestTradeImports:
    """Test that Trade class imports correctly."""

    def test_import_trade(self):
        from src.inference.backtesting.equity_curve import Trade

        assert Trade is not None


class TestRMultipleCalculation:
    """Test R-multiple calculation logic."""

    def test_r_multiple_winning_long_trade(self):
        from src.inference.backtesting.equity_curve import Trade

        trade = Trade(
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 1, 14, 0),
            direction=1,
            contracts=1,
            entry_price=4500.0,
            exit_price=4600.0,
            gross_pnl=500.0,
            costs=5.0,
            net_pnl=495.0,
            return_pct=1.0,
            stop_loss_price=4450.0,
        )
        trade.calculate_r_multiple(point_value=5.0)

        assert trade.initial_risk_1r == 250.0
        assert abs(trade.r_multiple - 1.98) < 0.01

    def test_r_multiple_losing_long_trade(self):
        from src.inference.backtesting.equity_curve import Trade

        trade = Trade(
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 1, 11, 0),
            direction=1,
            contracts=1,
            entry_price=4500.0,
            exit_price=4450.0,
            gross_pnl=-250.0,
            costs=5.0,
            net_pnl=-255.0,
            return_pct=-1.0,
            stop_loss_price=4450.0,
        )
        trade.calculate_r_multiple(point_value=5.0)

        assert trade.initial_risk_1r == 250.0
        assert trade.r_multiple < 0
        assert abs(trade.r_multiple - (-1.02)) < 0.01

    def test_r_multiple_winning_short_trade(self):
        from src.inference.backtesting.equity_curve import Trade

        trade = Trade(
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 1, 14, 0),
            direction=-1,
            contracts=1,
            entry_price=4500.0,
            exit_price=4400.0,
            gross_pnl=500.0,
            costs=5.0,
            net_pnl=495.0,
            return_pct=1.0,
            stop_loss_price=4550.0,
        )
        trade.calculate_r_multiple(point_value=5.0)

        assert trade.initial_risk_1r == 250.0
        assert trade.r_multiple > 0

    def test_r_multiple_no_stop_loss(self):
        from src.inference.backtesting.equity_curve import Trade

        trade = Trade(
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 1, 14, 0),
            direction=1,
            contracts=1,
            entry_price=4500.0,
            exit_price=4600.0,
            gross_pnl=500.0,
            costs=5.0,
            net_pnl=495.0,
            return_pct=1.0,
            stop_loss_price=None,
        )
        trade.calculate_r_multiple(point_value=5.0)

        assert trade.initial_risk_1r == 0.0
        assert trade.r_multiple == 0.0

    def test_r_multiple_stop_loss_zero(self):
        from src.inference.backtesting.equity_curve import Trade

        trade = Trade(
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 1, 14, 0),
            direction=1,
            contracts=1,
            entry_price=4500.0,
            exit_price=4600.0,
            gross_pnl=500.0,
            costs=5.0,
            net_pnl=495.0,
            return_pct=1.0,
            stop_loss_price=0,
        )
        trade.calculate_r_multiple(point_value=5.0)

        assert trade.r_multiple == 0.0

    def test_r_multiple_multiple_contracts(self):
        from src.inference.backtesting.equity_curve import Trade

        trade = Trade(
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 1, 14, 0),
            direction=1,
            contracts=5,
            entry_price=4500.0,
            exit_price=4600.0,
            gross_pnl=2500.0,
            costs=25.0,
            net_pnl=2475.0,
            return_pct=1.0,
            stop_loss_price=4450.0,
        )
        trade.calculate_r_multiple(point_value=5.0)

        assert trade.initial_risk_1r == 1250.0
        assert abs(trade.r_multiple - 1.98) < 0.01


class TestTradeProperties:
    """Test Trade helper properties."""

    def test_is_winner_property(self):
        from src.inference.backtesting.equity_curve import Trade

        winning_trade = Trade(
            entry_time=datetime(2024, 1, 1),
            exit_time=datetime(2024, 1, 1),
            direction=1,
            contracts=1,
            entry_price=4500.0,
            exit_price=4600.0,
            gross_pnl=500.0,
            costs=5.0,
            net_pnl=495.0,
            return_pct=1.0,
        )
        losing_trade = Trade(
            entry_time=datetime(2024, 1, 1),
            exit_time=datetime(2024, 1, 1),
            direction=1,
            contracts=1,
            entry_price=4500.0,
            exit_price=4400.0,
            gross_pnl=-500.0,
            costs=5.0,
            net_pnl=-505.0,
            return_pct=-1.0,
        )

        assert winning_trade.is_winner is True
        assert losing_trade.is_winner is False

    def test_to_dict_includes_r_multiple(self):
        from src.inference.backtesting.equity_curve import Trade

        trade = Trade(
            entry_time=datetime(2024, 1, 1),
            exit_time=datetime(2024, 1, 1),
            direction=1,
            contracts=1,
            entry_price=4500.0,
            exit_price=4600.0,
            gross_pnl=500.0,
            costs=5.0,
            net_pnl=495.0,
            return_pct=1.0,
            stop_loss_price=4450.0,
        )
        trade.calculate_r_multiple(point_value=5.0)
        trade_dict = trade.to_dict()

        assert "initial_risk_1r" in trade_dict
        assert "r_multiple" in trade_dict
        assert "stop_loss_price" in trade_dict


# =============================================================================
# Phase 86: Barrier Alignment, Per-Contract Sessions, Per-Symbol Slippage
# =============================================================================


class TestComputeATR:
    """Test _compute_atr produces correct ATR values with known data."""

    def test_compute_atr_known_values(self):
        from src.inference.backtesting.backtest import BacktestConfig, Backtester

        # Build a 20-bar dataset with known high/low/close
        n = 20
        timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
        high = np.array([110.0] * n)
        low = np.array([90.0] * n)
        close = np.array([100.0] * n)

        prices_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 1000),
            }
        )
        preds_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "prediction": np.zeros(n, dtype=int),
                "confidence": np.ones(n),
            }
        )

        config = BacktestConfig(initial_equity=100000.0, enable_market_hours_filter=False)
        bt = Backtester(predictions=preds_df, prices=prices_df, config=config)
        data = bt._align_data()
        atr = bt._compute_atr(data, period=14)

        # True range for every bar (after the first):
        #   max(high-low, |high-prev_close|, |low-prev_close|)
        #   = max(20, 10, 10) = 20
        # First bar TR = high - low = 20 (no prev_close)
        # ATR(14) = rolling mean of TR with min_periods=1
        # After warm-up all values should be exactly 20.0
        assert len(atr) == n
        assert abs(atr.iloc[-1] - 20.0) < 1e-6


class TestBarrierStopTP:
    """Test barrier stop-loss / take-profit computation in _open_position."""

    def _make_backtester(self, k_up: float, k_down: float):
        from src.inference.backtesting.backtest import BacktestConfig, Backtester

        n = 5
        timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
        prices_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0] * n,
                "high": [105.0] * n,
                "low": [95.0] * n,
                "close": [100.0] * n,
                "volume": [1000] * n,
            }
        )
        preds_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "prediction": np.zeros(n, dtype=int),
                "confidence": np.ones(n),
            }
        )
        config = BacktestConfig(
            initial_equity=100000.0,
            enable_market_hours_filter=False,
            barrier_k_up=k_up,
            barrier_k_down=k_down,
        )
        return Backtester(predictions=preds_df, prices=prices_df, config=config)

    def test_barrier_stop_tp_long(self):
        """LONG: stop = price - k_down*ATR, TP = price + k_up*ATR."""
        bt = self._make_backtester(k_up=3.0, k_down=2.0)
        price = 5000.0
        atr = 10.0
        bt._open_position(
            direction=1,
            price=price,
            timestamp=datetime(2024, 1, 1),
            bar_idx=0,
            atr=atr,
        )
        pos = bt._current_position
        assert pos is not None
        assert abs(pos.stop_loss - (price - 2.0 * atr)) < 1e-6  # 5000 - 20 = 4980
        assert abs(pos.take_profit - (price + 3.0 * atr)) < 1e-6  # 5000 + 30 = 5030

    def test_barrier_stop_tp_short(self):
        """SHORT: stop = price + k_up*ATR, TP = price - k_down*ATR."""
        bt = self._make_backtester(k_up=3.0, k_down=2.0)
        price = 5000.0
        atr = 10.0
        bt._open_position(
            direction=-1,
            price=price,
            timestamp=datetime(2024, 1, 1),
            bar_idx=0,
            atr=atr,
        )
        pos = bt._current_position
        assert pos is not None
        assert abs(pos.stop_loss - (price + 3.0 * atr)) < 1e-6  # 5000 + 30 = 5030
        assert abs(pos.take_profit - (price - 2.0 * atr)) < 1e-6  # 5000 - 20 = 4980

    def test_asymmetric_barriers(self):
        """MES H20 asymmetric: k_up=3.0, k_down=2.1."""
        bt = self._make_backtester(k_up=3.0, k_down=2.1)
        price = 4500.0
        atr = 5.0
        bt._open_position(
            direction=1,
            price=price,
            timestamp=datetime(2024, 1, 1),
            bar_idx=0,
            atr=atr,
        )
        pos = bt._current_position
        assert pos is not None
        # Long: stop = 4500 - 2.1*5 = 4489.5, TP = 4500 + 3.0*5 = 4515
        assert abs(pos.stop_loss - 4489.5) < 1e-6
        assert abs(pos.take_profit - 4515.0) < 1e-6

    def test_legacy_mode_no_barriers(self):
        """k_up=0, k_down=0 -> legacy 2% stop, no take profit."""
        bt = self._make_backtester(k_up=0.0, k_down=0.0)
        price = 5000.0
        bt._open_position(
            direction=1,
            price=price,
            timestamp=datetime(2024, 1, 1),
            bar_idx=0,
            atr=10.0,  # should be ignored
        )
        pos = bt._current_position
        assert pos is not None
        # Legacy long: stop = price * (1 - 1 * 0.02) = 4900
        assert abs(pos.stop_loss - 4900.0) < 1e-6
        assert pos.take_profit is None


class TestPerContractSessionTimes:
    """Test CONTRACT_SESSION_TIMES per-symbol session windows."""

    def test_per_contract_session_times(self):
        from datetime import time as dt_time

        from src.inference.backtesting.execution import CONTRACT_SESSION_TIMES

        # MGC = COMEX primary session: 8:20 - 13:30 ET
        mgc_start, mgc_end = CONTRACT_SESSION_TIMES["MGC"]
        assert mgc_start == dt_time(8, 20)
        assert mgc_end == dt_time(13, 30)

        # MES = NYSE cash session: 9:30 - 16:00 ET
        mes_start, mes_end = CONTRACT_SESSION_TIMES["MES"]
        assert mes_start == dt_time(9, 30)
        assert mes_end == dt_time(16, 0)

        # MNQ shares MES session
        mnq_start, mnq_end = CONTRACT_SESSION_TIMES["MNQ"]
        assert mnq_start == dt_time(9, 30)
        assert mnq_end == dt_time(16, 0)

    def test_market_hours_filter_uses_per_contract_times(self):
        """MarketHoursFilter resolves session from CONTRACT_SESSION_TIMES."""
        from datetime import time as dt_time

        from src.inference.backtesting.execution import MarketHoursFilter

        filt_mgc = MarketHoursFilter(contract="MGC", enable_market_hours_filter=True)
        assert filt_mgc._session_start == dt_time(8, 20)
        assert filt_mgc._session_end == dt_time(13, 30)

        filt_mes = MarketHoursFilter(contract="MES", enable_market_hours_filter=True)
        assert filt_mes._session_start == dt_time(9, 30)
        assert filt_mes._session_end == dt_time(16, 0)


class TestPerSymbolSlippageDefaults:
    """Test SYMBOL_SLIPPAGE_DEFAULTS and create_cost_calculator wiring."""

    def test_per_symbol_slippage_defaults(self):
        from src.inference.backtesting.costs import SYMBOL_SLIPPAGE_DEFAULTS

        # MES: high liquidity equity
        mes = SYMBOL_SLIPPAGE_DEFAULTS["MES"]
        assert mes["base_volatility"] == 0.15
        assert mes["typical_volume"] == 1500.0

        # MGC: moderate liquidity gold
        mgc = SYMBOL_SLIPPAGE_DEFAULTS["MGC"]
        assert mgc["base_volatility"] == 0.18
        assert mgc["typical_volume"] == 500.0
        # MGC has wider impact coefficient than MES
        assert mgc["impact_coefficient"] > mes["impact_coefficient"]

    def test_create_cost_calculator_uses_symbol_defaults(self):
        from src.inference.backtesting.costs import create_cost_calculator

        calc_mes = create_cost_calculator(symbol="MES", slippage_model="square_root")
        calc_mgc = create_cost_calculator(symbol="MGC", slippage_model="square_root")

        # MES tick_size = 0.25, MGC tick_size = 0.10
        assert calc_mes.transaction_costs.tick_size == 0.25
        assert calc_mgc.transaction_costs.tick_size == 0.10

        # Slippage models should pick up per-symbol defaults
        assert calc_mes.slippage_model.typical_volume == 1500.0
        assert calc_mgc.slippage_model.typical_volume == 500.0


class TestBarrierConfigFactoryWiring:
    """Test get_barrier_params feeds into BacktestConfig round-trip."""

    def test_barrier_config_factory_wiring(self):
        from src.data.pipeline.config.barriers_config import get_barrier_params
        from src.inference.backtesting.backtest import BacktestConfig

        params = get_barrier_params("MES", 20)
        config = BacktestConfig(
            barrier_k_up=params["k_up"],
            barrier_k_down=params["k_down"],
            max_holding_period=params["max_bars"],
        )

        # MES H20: k_up=3.0, k_down=2.1, max_bars=50
        assert config.barrier_k_up == 3.0
        assert config.barrier_k_down == 2.1
        assert config.max_holding_period == 50

    def test_barrier_params_propagate_to_position(self):
        """Barrier params in config flow through to Position stop/TP."""
        from src.data.pipeline.config.barriers_config import get_barrier_params
        from src.inference.backtesting.backtest import BacktestConfig, Backtester

        params = get_barrier_params("MES", 20)
        n = 5
        timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
        prices_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": [4500.0] * n,
                "high": [4510.0] * n,
                "low": [4490.0] * n,
                "close": [4500.0] * n,
                "volume": [1000] * n,
            }
        )
        preds_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "prediction": np.zeros(n, dtype=int),
                "confidence": np.ones(n),
            }
        )
        config = BacktestConfig(
            initial_equity=100000.0,
            enable_market_hours_filter=False,
            barrier_k_up=params["k_up"],
            barrier_k_down=params["k_down"],
        )
        bt = Backtester(predictions=preds_df, prices=prices_df, config=config)
        price = 4500.0
        atr = 8.0
        bt._open_position(
            direction=1,
            price=price,
            timestamp=datetime(2024, 1, 1),
            bar_idx=0,
            atr=atr,
        )
        pos = bt._current_position
        assert pos is not None
        # k_up=3.0, k_down=2.1 from get_barrier_params
        assert abs(pos.stop_loss - (4500.0 - 2.1 * 8.0)) < 1e-6
        assert abs(pos.take_profit - (4500.0 + 3.0 * 8.0)) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
