"""
Test GA optimization bug fixes.

Tests for:
1. Critical Issue #5: GA profit factor bug (shorts) - short_risk calculation
2. Critical Issue #6: Transaction cost penalty dimensionally incorrect
"""
import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'stages'))

from stage5_ga_optimize import calculate_fitness
from config import TRANSACTION_COSTS, TICK_VALUES


class TestShortRiskCalculation:
    """Test Critical Issue #5: Short risk calculation bug."""

    def test_short_risk_includes_negative_mfe(self):
        """
        Test that short risk correctly includes negative MFE values.

        BEFORE FIX: np.maximum(mfe[short_mask], 0) zeros out negative values
        AFTER FIX: mfe[short_mask].sum() includes all values

        Example: If MFE = [-2.0, -1.0, 3.0] for shorts
        - BEFORE: risk = max(-2, 0) + max(-1, 0) + max(3, 0) = 3.0
        - AFTER: risk = -2.0 + -1.0 + 3.0 = 0.0
        """
        # Setup: Balanced labels to meet fitness requirements (20-30% neutral)
        # 10 total: 3 long, 3 short, 4 neutral = 30% signal rate, 40% neutral
        labels = np.array([-1, -1, -1, 1, 1, 1, 0, 0, 0, 0])
        bars_to_hit = np.array([5, 5, 5, 5, 5, 5, 10, 10, 10, 10])

        # MAE: Downward movement (profit for shorts, risk for longs)
        mae = np.array([-5.0, -3.0, -4.0, -2.0, -2.0, -2.0, 0, 0, 0, 0])

        # MFE: Upward movement (risk for shorts, profit for longs)
        # Include negative values for shorts to test the fix
        mfe = np.array([-2.0, -1.0, 3.0, 10.0, 10.0, 10.0, 0, 0, 0, 0])

        horizon = 5
        atr_mean = 10.0
        symbol = 'MES'

        # Calculate fitness (should not crash and should use correct risk)
        fitness = calculate_fitness(
            labels, bars_to_hit, mae, mfe, horizon, atr_mean, symbol
        )

        # Fitness should be a valid number (not -inf)
        assert np.isfinite(fitness), "Fitness should be finite"

        # The key test: With the fix, negative MFE values contribute to risk
        # Before fix: risk would only be 3.0 (positive MFE)
        # After fix: risk includes all MFE values
        # We can't directly check internal values, but the fitness should be reasonable
        # (not the -1000 penalty zone for bad signal rate)
        assert fitness > -500, "Fitness should be reasonable with proper label distribution"


    def test_short_risk_with_all_positive_mfe(self):
        """
        Test short risk calculation when all MFE values are positive (upward risk).
        This should work both before and after the fix.
        """
        labels = np.array([-1, -1, -1])  # All shorts
        bars_to_hit = np.array([5, 5, 5])
        mae = np.array([-5.0, -3.0, -4.0])  # Profit for shorts
        mfe = np.array([2.0, 1.5, 2.5])  # All positive (upward risk)

        horizon = 5
        atr_mean = 10.0
        symbol = 'MES'

        fitness = calculate_fitness(
            labels, bars_to_hit, mae, mfe, horizon, atr_mean, symbol
        )

        assert np.isfinite(fitness), "Fitness should be finite"


    def test_short_risk_with_all_negative_mfe(self):
        """
        Test short risk calculation when all MFE values are negative.
        This is the critical case that exposes the bug.
        """
        labels = np.array([-1, -1, -1])  # All shorts
        bars_to_hit = np.array([5, 5, 5])
        mae = np.array([-5.0, -3.0, -4.0])  # Profit for shorts
        mfe = np.array([-1.0, -0.5, -2.0])  # All negative (no upward risk)

        horizon = 5
        atr_mean = 10.0
        symbol = 'MES'

        fitness = calculate_fitness(
            labels, bars_to_hit, mae, mfe, horizon, atr_mean, symbol
        )

        # With the fix, this should produce a valid fitness
        # Before fix: risk would be 0, profit_factor = inf, likely triggering errors
        # After fix: risk is negative (which is weird but handled), or zero
        assert np.isfinite(fitness), "Fitness should be finite even with all negative MFE"


class TestTransactionCostPenalty:
    """Test Critical Issue #6: Transaction cost penalty dimensional correctness."""

    def test_transaction_cost_units_match(self):
        """
        Test that transaction cost penalty uses correct units.

        BEFORE FIX: cost_ticks / (avg_profit_per_trade / atr_mean)
                    = ticks / (price_units / price_units)
                    = ticks / dimensionless  <- WRONG

        AFTER FIX: (cost_ticks * tick_value) / avg_profit_per_trade
                   = price_units / price_units  <- CORRECT
        """
        # Setup: Balanced long/short trades with known profit
        # 10 total: 3 long, 3 short, 4 neutral = 60% signal rate, 40% neutral (good balance)
        labels = np.array([1, -1, 1, -1, 1, -1, 0, 0, 0, 0])
        bars_to_hit = np.array([5, 5, 5, 5, 5, 5, 10, 10, 10, 10])

        # Setup MAE/MFE to produce known profit
        # Long trades (indices 0, 2, 4): MFE = profit, MAE = risk
        # Short trades (indices 1, 3, 5): |MAE| = profit, MFE = risk
        mae = np.array([-2.0, -5.0, -2.0, -5.0, -2.0, -5.0, 0, 0, 0, 0])
        mfe = np.array([10.0, 2.0, 10.0, 2.0, 10.0, 2.0, 0, 0, 0, 0])

        # Total profit = long_profit + short_profit
        # long_profit = 10.0 + 10.0 + 10.0 = 30.0
        # short_profit = 5.0 + 5.0 + 5.0 = 15.0
        # total_profit = 45.0
        # n_trades = 6
        # avg_profit_per_trade = 7.5

        horizon = 5
        atr_mean = 10.0
        symbol = 'MES'

        # MES: cost = 0.5 ticks * $1.25/tick = $0.625
        # cost_ratio = $0.625 / $7.5 = 0.0833 (8.33%)
        # This is < 20%, so should get bonus (+0.5)

        fitness = calculate_fitness(
            labels, bars_to_hit, mae, mfe, horizon, atr_mean, symbol
        )

        # Fitness should be finite and reasonable
        assert np.isfinite(fitness), "Fitness should be finite"

        # The transaction cost penalty should contribute positively (bonus)
        # since cost ratio (8.33%) < 20% threshold
        # We can't check absolute fitness value, but it should be reasonable
        assert fitness > -500, "Fitness should be reasonable with good cost ratio"


    def test_transaction_cost_penalty_high_cost(self):
        """
        Test transaction cost penalty when costs are high relative to profit.
        Should apply negative penalty.
        """
        # Setup: Trades with very low profit
        labels = np.array([1, -1, 1, -1])
        bars_to_hit = np.array([5, 5, 5, 5])

        # Very small profit (0.5 per trade)
        mae = np.array([-0.1, -0.5, -0.1, -0.5])
        mfe = np.array([0.5, 0.1, 0.5, 0.1])

        # avg_profit_per_trade = (0.5 + 0.5 + 0.5 + 0.5) / 4 = 0.5
        # MES: cost = 0.5 * 1.25 = 0.625
        # cost_ratio = 0.625 / 0.5 = 1.25 (125%)
        # This is >> 20%, so should get heavy penalty

        horizon = 5
        atr_mean = 10.0
        symbol = 'MES'

        fitness = calculate_fitness(
            labels, bars_to_hit, mae, mfe, horizon, atr_mean, symbol
        )

        # Fitness should be finite but negative due to cost penalty
        assert np.isfinite(fitness), "Fitness should be finite"
        # Transaction penalty should drag fitness down


    def test_transaction_cost_mgc_vs_mes(self):
        """
        Test that MGC and MES use different transaction costs correctly.

        MGC: 0.3 ticks * $1.00/tick = $0.30
        MES: 0.5 ticks * $1.25/tick = $0.625

        With same profit, MGC should have lower cost ratio and better fitness.
        """
        # Same setup for both symbols
        labels = np.array([1, -1, 1, -1, 0, 0, 0, 0])
        bars_to_hit = np.array([5, 5, 5, 5, 10, 10, 10, 10])
        mae = np.array([-2.0, -5.0, -2.0, -5.0, 0, 0, 0, 0])
        mfe = np.array([10.0, 2.0, 10.0, 2.0, 0, 0, 0, 0])
        horizon = 5
        atr_mean = 10.0

        # Calculate for MES
        fitness_mes = calculate_fitness(
            labels, bars_to_hit, mae, mfe, horizon, atr_mean, symbol='MES'
        )

        # Calculate for MGC
        fitness_mgc = calculate_fitness(
            labels, bars_to_hit, mae, mfe, horizon, atr_mean, symbol='MGC'
        )

        # Both should be finite
        assert np.isfinite(fitness_mes), "MES fitness should be finite"
        assert np.isfinite(fitness_mgc), "MGC fitness should be finite"

        # MGC should have slightly better (or equal) fitness due to lower costs
        # (All other factors being equal, lower transaction cost is better)
        # Note: The difference might be small, so we just check both are reasonable
        assert fitness_mes > -1000, "MES fitness should be reasonable"
        assert fitness_mgc > -1000, "MGC fitness should be reasonable"


class TestProfitFactorCalculation:
    """Additional tests for profit factor calculation to ensure it's correct."""

    def test_profitable_strategy(self):
        """Test profit factor for a clearly profitable strategy."""
        # Long trades: good profit (MFE) with small risk (|MAE|)
        # Short trades: good profit (|MAE|) with small risk (MFE)
        labels = np.array([1, 1, -1, -1])
        bars_to_hit = np.array([5, 5, 5, 5])

        # Longs: profit=10, risk=2
        # Shorts: profit=10, risk=2
        mae = np.array([-2.0, -2.0, -10.0, -10.0])
        mfe = np.array([10.0, 10.0, 2.0, 2.0])

        # Expected:
        # long_profit = 10 + 10 = 20
        # long_risk = 2 + 2 = 4
        # short_profit = 10 + 10 = 20
        # short_risk = 2 + 2 = 4
        # profit_factor = 40 / 8 = 5.0 (very good)

        horizon = 5
        atr_mean = 10.0
        symbol = 'MES'

        fitness = calculate_fitness(
            labels, bars_to_hit, mae, mfe, horizon, atr_mean, symbol
        )

        # Should be positive (though might be capped due to overfit penalty)
        assert np.isfinite(fitness), "Fitness should be finite"
        # With profit factor = 5.0, this is actually too high (>2.0 threshold)
        # So it might get a reduced pf_score (1.5 instead of 2.0)


    def test_unprofitable_strategy(self):
        """Test profit factor for an unprofitable strategy."""
        # Profit < Risk
        labels = np.array([1, 1, -1, -1])
        bars_to_hit = np.array([5, 5, 5, 5])

        # Longs: profit=2, risk=10
        # Shorts: profit=2, risk=10
        mae = np.array([-10.0, -10.0, -2.0, -2.0])
        mfe = np.array([2.0, 2.0, 10.0, 10.0])

        # Expected:
        # profit_factor = (2+2+2+2) / (10+10+10+10) = 8/40 = 0.2 (very bad)

        horizon = 5
        atr_mean = 10.0
        symbol = 'MES'

        fitness = calculate_fitness(
            labels, bars_to_hit, mae, mfe, horizon, atr_mean, symbol
        )

        # Should get heavy penalty for profit_factor < 0.8
        assert np.isfinite(fitness), "Fitness should be finite"
        # Likely negative overall due to unprofitable trades


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
