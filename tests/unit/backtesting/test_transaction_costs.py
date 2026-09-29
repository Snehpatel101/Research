"""Transaction costs, slippage models, cost calculator and per-symbol slippage defaults."""

from __future__ import annotations

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
