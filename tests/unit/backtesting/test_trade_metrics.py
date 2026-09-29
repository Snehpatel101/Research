"""Trade dataclass: R-multiple calculations and derived properties."""

from __future__ import annotations

from datetime import datetime

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
