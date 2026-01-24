"""
Unit tests for TransactionCosts and slippage models.

Tests the cost calculation logic added/enhanced in Phase 12.
"""

from __future__ import annotations

import pytest


class TestTransactionCostsImports:
    """Test that cost modules import correctly."""

    def test_import_transaction_costs(self):
        """Verify TransactionCosts can be imported."""
        from src.inference.backtesting.costs import TransactionCosts

        assert TransactionCosts is not None

    def test_import_cost_calculator(self):
        """Verify CostCalculator can be imported."""
        from src.inference.backtesting.costs import CostCalculator

        assert CostCalculator is not None

    def test_import_slippage_models(self):
        """Verify slippage models can be imported."""
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
        """Test round-trip cost for 1 contract."""
        from src.inference.backtesting.costs import TransactionCosts

        costs = TransactionCosts(
            commission_per_contract=2.50,
            slippage_ticks=1.0,
            tick_value=1.25,
            exchange_fee=0.52,
            nfa_fee=0.02,
        )

        # Round-trip cost = commission + exchange + nfa + (2 * slippage)
        # = 2.50 + 0.52 + 0.02 + (2 * 1.0 * 1.25)
        # = 3.04 + 2.50 = 5.54
        expected = 5.54
        actual = costs.calculate_round_trip_cost(contracts=1)

        assert abs(actual - expected) < 0.01, f"Expected {expected}, got {actual}"

    def test_round_trip_cost_multiple_contracts(self):
        """Test round-trip cost scales linearly with contracts."""
        from src.inference.backtesting.costs import TransactionCosts

        costs = TransactionCosts.for_mes()

        cost_1 = costs.calculate_round_trip_cost(contracts=1)
        cost_5 = costs.calculate_round_trip_cost(contracts=5)

        assert abs(cost_5 - 5 * cost_1) < 0.01, "Cost should scale linearly"

    def test_entry_cost_calculation(self):
        """Test entry cost is half of round-trip."""
        from src.inference.backtesting.costs import TransactionCosts

        costs = TransactionCosts(
            commission_per_contract=2.50,
            slippage_ticks=1.0,
            tick_value=1.25,
            exchange_fee=0.52,
            nfa_fee=0.02,
        )

        # Entry cost = (commission + exchange + nfa) / 2 + slippage
        # = (2.50 + 0.52 + 0.02) / 2 + 1.25
        # = 1.52 + 1.25 = 2.77
        entry_cost = costs.calculate_entry_cost(contracts=1, entry_price=4500.0)
        expected = 2.77

        assert abs(entry_cost - expected) < 0.01, f"Expected {expected}, got {entry_cost}"

    def test_total_fixed_cost_property(self):
        """Test total_fixed_cost_per_contract property."""
        from src.inference.backtesting.costs import TransactionCosts

        costs = TransactionCosts(
            commission_per_contract=2.50,
            exchange_fee=0.52,
            nfa_fee=0.02,
        )

        # Total fixed = 2.50 + 0.52 + 0.02 = 3.04
        expected = 3.04
        assert abs(costs.total_fixed_cost_per_contract - expected) < 0.01

    def test_slippage_cost_property(self):
        """Test slippage_cost_per_contract property."""
        from src.inference.backtesting.costs import TransactionCosts

        costs = TransactionCosts(
            slippage_ticks=1.0,
            tick_value=1.25,
        )

        # Slippage (round-trip) = 2 * 1.0 * 1.25 = 2.50
        expected = 2.50
        assert abs(costs.slippage_cost_per_contract - expected) < 0.01


class TestTransactionCostsFactories:
    """Test factory methods for different contracts."""

    def test_for_mes(self):
        """Test MES factory method."""
        from src.inference.backtesting.costs import TransactionCosts

        costs = TransactionCosts.for_mes()

        assert costs.tick_value == 1.25
        assert costs.tick_size == 0.25
        assert costs.commission_per_contract == 2.50

    def test_for_mgc(self):
        """Test MGC factory method."""
        from src.inference.backtesting.costs import TransactionCosts

        costs = TransactionCosts.for_mgc()

        assert costs.tick_value == 1.00
        assert costs.tick_size == 0.10
        assert costs.commission_per_contract == 2.50

    def test_for_mnq(self):
        """Test MNQ factory method."""
        from src.inference.backtesting.costs import TransactionCosts

        costs = TransactionCosts.for_mnq()

        assert costs.tick_value == 0.50
        assert costs.tick_size == 0.25


class TestSlippageModels:
    """Test slippage model calculations."""

    def test_fixed_slippage(self):
        """Test fixed slippage returns constant value."""
        from src.inference.backtesting.costs import FixedSlippage

        model = FixedSlippage(ticks=2.0, tick_size=0.25)

        # Should return 2.0 * 0.25 = 0.50 regardless of inputs
        slippage = model.estimate_slippage(
            order_size=1,
            price=4500.0,
        )

        assert slippage == 0.50

        # Same result for different order size
        slippage_large = model.estimate_slippage(
            order_size=100,
            price=4500.0,
        )

        assert slippage_large == 0.50

    def test_linear_slippage_scales_with_size(self):
        """Test linear slippage increases with order size."""
        from src.inference.backtesting.costs import LinearSlippage

        model = LinearSlippage(
            base_ticks=0.5,
            size_factor=0.1,
            tick_size=0.25,
        )

        slip_1 = model.estimate_slippage(order_size=1, price=4500.0)
        slip_10 = model.estimate_slippage(order_size=10, price=4500.0)

        assert slip_10 > slip_1, "Slippage should increase with order size"

    def test_volatility_scaled_slippage(self):
        """Test volatility-scaled slippage increases with volatility."""
        from src.inference.backtesting.costs import VolatilityScaledSlippage

        model = VolatilityScaledSlippage(
            base_ticks=1.0,
            base_volatility=0.15,
            volatility_multiplier=2.0,
            tick_size=0.25,
        )

        # Low volatility
        slip_low = model.estimate_slippage(
            order_size=1,
            price=4500.0,
            volatility=0.10,
        )

        # High volatility
        slip_high = model.estimate_slippage(
            order_size=1,
            price=4500.0,
            volatility=0.30,
        )

        assert slip_high > slip_low, "Slippage should increase with volatility"


class TestCostCalculator:
    """Test CostCalculator integration."""

    def test_calculate_pnl_long_winning_trade(self):
        """Test P&L calculation for winning long trade."""
        from src.inference.backtesting.costs import CostCalculator, TransactionCosts

        tx_costs = TransactionCosts.for_mes()
        calculator = CostCalculator(transaction_costs=tx_costs)

        result = calculator.calculate_pnl(
            contracts=1,
            entry_price=4500.0,
            exit_price=4510.0,  # 10 point gain
            direction=1,  # Long
            point_value=5.0,
        )

        # Gross P&L = 1 * 10 * 5.0 = 50.0
        assert result["gross_pnl"] == 50.0

        # Net should be less than gross due to costs
        assert result["net_pnl"] < result["gross_pnl"]

        # Costs should be positive
        assert result["costs"] > 0

    def test_calculate_pnl_short_winning_trade(self):
        """Test P&L calculation for winning short trade."""
        from src.inference.backtesting.costs import CostCalculator, TransactionCosts

        tx_costs = TransactionCosts.for_mes()
        calculator = CostCalculator(transaction_costs=tx_costs)

        result = calculator.calculate_pnl(
            contracts=1,
            entry_price=4500.0,
            exit_price=4490.0,  # 10 point drop
            direction=-1,  # Short
            point_value=5.0,
        )

        # Gross P&L = -1 * 1 * -10 * 5.0 = 50.0
        assert result["gross_pnl"] == 50.0

    def test_calculate_pnl_no_costs(self):
        """Test P&L calculation with costs disabled."""
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

        # With no costs, net == gross
        assert result["net_pnl"] == result["gross_pnl"]
        assert result["costs"] == 0.0
