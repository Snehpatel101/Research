"""
Unit tests for R-Multiple calculations.

Tests the Trade.calculate_r_multiple() method added in Phase 12B.
"""

from __future__ import annotations

from datetime import datetime

import pytest


class TestTradeImports:
    """Test that Trade class imports correctly."""

    def test_import_trade(self):
        """Verify Trade can be imported."""
        from src.inference.backtesting.equity_curve import Trade

        assert Trade is not None


class TestRMultipleCalculation:
    """Test R-multiple calculation logic."""

    def test_r_multiple_winning_long_trade(self):
        """Test R-multiple for a winning long trade."""
        from src.inference.backtesting.equity_curve import Trade

        # Long trade: Entry 4500, Stop 4450 (50 point risk), Exit 4600 (100 point gain)
        trade = Trade(
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 1, 14, 0),
            direction=1,  # Long
            contracts=1,
            entry_price=4500.0,
            exit_price=4600.0,
            gross_pnl=500.0,  # 100 points * 5.0 point_value
            costs=5.0,
            net_pnl=495.0,
            return_pct=1.0,
            stop_loss_price=4450.0,  # 50 points below entry
        )

        trade.calculate_r_multiple(point_value=5.0)

        # 1R = |4500 - 4450| * 5.0 * 1 contract = 250
        # R-multiple = 495 / 250 = 1.98
        assert trade.initial_risk_1r == 250.0
        assert abs(trade.r_multiple - 1.98) < 0.01

    def test_r_multiple_losing_long_trade(self):
        """Test R-multiple for a losing long trade (stopped out)."""
        from src.inference.backtesting.equity_curve import Trade

        # Long trade stopped out at stop loss
        trade = Trade(
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 1, 11, 0),
            direction=1,
            contracts=1,
            entry_price=4500.0,
            exit_price=4450.0,  # Hit stop loss
            gross_pnl=-250.0,
            costs=5.0,
            net_pnl=-255.0,
            return_pct=-1.0,
            stop_loss_price=4450.0,
        )

        trade.calculate_r_multiple(point_value=5.0)

        # 1R = 250, loss of -255, so R = -255/250 = -1.02
        assert trade.initial_risk_1r == 250.0
        assert trade.r_multiple < 0  # Should be negative
        assert abs(trade.r_multiple - (-1.02)) < 0.01

    def test_r_multiple_winning_short_trade(self):
        """Test R-multiple for a winning short trade."""
        from src.inference.backtesting.equity_curve import Trade

        # Short trade: Entry 4500, Stop 4550 (50 point risk), Exit 4400 (100 point gain)
        trade = Trade(
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 1, 14, 0),
            direction=-1,  # Short
            contracts=1,
            entry_price=4500.0,
            exit_price=4400.0,
            gross_pnl=500.0,  # 100 points * 5.0
            costs=5.0,
            net_pnl=495.0,
            return_pct=1.0,
            stop_loss_price=4550.0,  # 50 points above entry
        )

        trade.calculate_r_multiple(point_value=5.0)

        # 1R = |4500 - 4550| * 5.0 = 250
        assert trade.initial_risk_1r == 250.0
        assert trade.r_multiple > 0  # Winning trade

    def test_r_multiple_no_stop_loss(self):
        """Test R-multiple when no stop loss is set."""
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
            stop_loss_price=None,  # No stop loss
        )

        trade.calculate_r_multiple(point_value=5.0)

        # Without stop loss, R-multiple should be 0
        assert trade.initial_risk_1r == 0.0
        assert trade.r_multiple == 0.0

    def test_r_multiple_stop_loss_zero(self):
        """Test R-multiple when stop loss is zero."""
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
            stop_loss_price=0,  # Zero stop loss
        )

        trade.calculate_r_multiple(point_value=5.0)

        # Zero stop loss should result in 0 R-multiple
        assert trade.r_multiple == 0.0

    def test_r_multiple_multiple_contracts(self):
        """Test R-multiple scales with contract count."""
        from src.inference.backtesting.equity_curve import Trade

        trade = Trade(
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 1, 14, 0),
            direction=1,
            contracts=5,  # 5 contracts
            entry_price=4500.0,
            exit_price=4600.0,
            gross_pnl=2500.0,  # 100 * 5 * 5 contracts
            costs=25.0,
            net_pnl=2475.0,
            return_pct=1.0,
            stop_loss_price=4450.0,
        )

        trade.calculate_r_multiple(point_value=5.0)

        # 1R = 50 * 5.0 * 5 = 1250
        assert trade.initial_risk_1r == 1250.0
        # R = 2475 / 1250 = 1.98
        assert abs(trade.r_multiple - 1.98) < 0.01


class TestTradeProperties:
    """Test Trade helper properties."""

    def test_is_winner_property(self):
        """Test is_winner property."""
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
            net_pnl=495.0,  # Positive
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
            net_pnl=-505.0,  # Negative
            return_pct=-1.0,
        )

        assert winning_trade.is_winner is True
        assert losing_trade.is_winner is False

    def test_to_dict_includes_r_multiple(self):
        """Test that to_dict includes R-multiple fields."""
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
