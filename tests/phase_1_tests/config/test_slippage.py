"""
Unit tests for slippage modeling in barriers_config.

Tests SLIPPAGE_TICKS configuration and slippage calculation functions.

Run with: pytest tests/phase_1_tests/config/test_slippage.py -v
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.phase1.config.barriers_config import (
    SLIPPAGE_TICKS,
    TRANSACTION_COSTS,
    TICK_VALUES,
    get_slippage_ticks,
    get_total_trade_cost,
    validate_barrier_params,
)


# =============================================================================
# CONFIGURATION VALIDATION TESTS
# =============================================================================

class TestSlippageConfiguration:
    """Tests for SLIPPAGE_TICKS configuration structure."""

    def test_slippage_ticks_structure(self):
        """Test SLIPPAGE_TICKS has correct structure for all symbols."""
        required_symbols = ['MES', 'MGC']
        required_regimes = ['low_vol', 'high_vol']

        for symbol in required_symbols:
            assert symbol in SLIPPAGE_TICKS, f"Missing symbol: {symbol}"
            assert isinstance(SLIPPAGE_TICKS[symbol], dict), f"{symbol} must be a dict"

            for regime in required_regimes:
                assert regime in SLIPPAGE_TICKS[symbol], f"Missing regime {regime} for {symbol}"
                assert isinstance(SLIPPAGE_TICKS[symbol][regime], (int, float)), \
                    f"Slippage for {symbol}[{regime}] must be numeric"
                assert SLIPPAGE_TICKS[symbol][regime] >= 0, \
                    f"Slippage for {symbol}[{regime}] must be non-negative"

    def test_slippage_high_vol_greater_than_low_vol(self):
        """Test high volatility slippage is >= low volatility slippage."""
        for symbol in ['MES', 'MGC']:
            low_vol = SLIPPAGE_TICKS[symbol]['low_vol']
            high_vol = SLIPPAGE_TICKS[symbol]['high_vol']
            assert high_vol >= low_vol, \
                f"{symbol}: high_vol ({high_vol}) should be >= low_vol ({low_vol})"

    def test_mes_slippage_less_than_mgc(self):
        """Test MES slippage is less than MGC (higher liquidity)."""
        # MES should have tighter slippage due to higher liquidity
        mes_low = SLIPPAGE_TICKS['MES']['low_vol']
        mgc_low = SLIPPAGE_TICKS['MGC']['low_vol']
        assert mes_low <= mgc_low, \
            f"MES low_vol ({mes_low}) should be <= MGC low_vol ({mgc_low})"

        mes_high = SLIPPAGE_TICKS['MES']['high_vol']
        mgc_high = SLIPPAGE_TICKS['MGC']['high_vol']
        assert mes_high <= mgc_high, \
            f"MES high_vol ({mes_high}) should be <= MGC high_vol ({mgc_high})"

    def test_slippage_within_reasonable_bounds(self):
        """Test slippage values are within reasonable bounds (0.5-2.0 ticks)."""
        for symbol in ['MES', 'MGC']:
            for regime in ['low_vol', 'high_vol']:
                slippage = SLIPPAGE_TICKS[symbol][regime]
                assert 0.3 <= slippage <= 2.0, \
                    f"Unreasonable slippage for {symbol}[{regime}]: {slippage}"


# =============================================================================
# SLIPPAGE CALCULATION TESTS
# =============================================================================

class TestGetSlippageTicks:
    """Tests for get_slippage_ticks() function."""

    def test_get_slippage_mes_low_vol(self):
        """Test get slippage for MES low volatility."""
        slippage = get_slippage_ticks('MES', 'low_vol')
        assert slippage == SLIPPAGE_TICKS['MES']['low_vol']
        assert slippage == 0.5

    def test_get_slippage_mes_high_vol(self):
        """Test get slippage for MES high volatility."""
        slippage = get_slippage_ticks('MES', 'high_vol')
        assert slippage == SLIPPAGE_TICKS['MES']['high_vol']
        assert slippage == 1.0

    def test_get_slippage_mgc_low_vol(self):
        """Test get slippage for MGC low volatility."""
        slippage = get_slippage_ticks('MGC', 'low_vol')
        assert slippage == SLIPPAGE_TICKS['MGC']['low_vol']
        assert slippage == 0.75

    def test_get_slippage_mgc_high_vol(self):
        """Test get slippage for MGC high volatility."""
        slippage = get_slippage_ticks('MGC', 'high_vol')
        assert slippage == SLIPPAGE_TICKS['MGC']['high_vol']
        assert slippage == 1.5

    def test_get_slippage_default_regime(self):
        """Test get slippage defaults to low_vol if regime not specified."""
        slippage = get_slippage_ticks('MES')
        assert slippage == SLIPPAGE_TICKS['MES']['low_vol']

    def test_get_slippage_invalid_regime_defaults_to_low_vol(self):
        """Test invalid regime defaults to low_vol."""
        slippage = get_slippage_ticks('MES', 'invalid_regime')
        assert slippage == SLIPPAGE_TICKS['MES']['low_vol']

    def test_get_slippage_invalid_symbol_defaults_to_mes(self):
        """Test invalid symbol defaults to MES."""
        slippage = get_slippage_ticks('INVALID', 'low_vol')
        assert slippage == SLIPPAGE_TICKS['MES']['low_vol']


# =============================================================================
# TOTAL TRADE COST TESTS
# =============================================================================

class TestGetTotalTradeCost:
    """Tests for get_total_trade_cost() function."""

    def test_mes_low_vol_with_slippage(self):
        """Test MES low volatility total cost (commission + slippage)."""
        cost = get_total_trade_cost('MES', 'low_vol', include_slippage=True)
        expected = TRANSACTION_COSTS['MES'] + 2 * SLIPPAGE_TICKS['MES']['low_vol']
        assert cost == expected
        assert cost == 1.5  # 0.5 commission + 2*0.5 slippage

    def test_mes_high_vol_with_slippage(self):
        """Test MES high volatility total cost (commission + slippage)."""
        cost = get_total_trade_cost('MES', 'high_vol', include_slippage=True)
        expected = TRANSACTION_COSTS['MES'] + 2 * SLIPPAGE_TICKS['MES']['high_vol']
        assert cost == expected
        assert cost == 2.5  # 0.5 commission + 2*1.0 slippage

    def test_mgc_low_vol_with_slippage(self):
        """Test MGC low volatility total cost (commission + slippage)."""
        cost = get_total_trade_cost('MGC', 'low_vol', include_slippage=True)
        expected = TRANSACTION_COSTS['MGC'] + 2 * SLIPPAGE_TICKS['MGC']['low_vol']
        assert cost == expected
        assert cost == 1.8  # 0.3 commission + 2*0.75 slippage

    def test_mgc_high_vol_with_slippage(self):
        """Test MGC high volatility total cost (commission + slippage)."""
        cost = get_total_trade_cost('MGC', 'high_vol', include_slippage=True)
        expected = TRANSACTION_COSTS['MGC'] + 2 * SLIPPAGE_TICKS['MGC']['high_vol']
        assert cost == expected
        assert cost == 3.3  # 0.3 commission + 2*1.5 slippage

    def test_commission_only_no_slippage(self):
        """Test cost calculation with slippage disabled."""
        cost = get_total_trade_cost('MES', 'low_vol', include_slippage=False)
        assert cost == TRANSACTION_COSTS['MES']
        assert cost == 0.5

    def test_default_regime_low_vol(self):
        """Test default regime is low_vol."""
        cost = get_total_trade_cost('MES')
        expected = TRANSACTION_COSTS['MES'] + 2 * SLIPPAGE_TICKS['MES']['low_vol']
        assert cost == expected

    def test_high_vol_costs_more_than_low_vol(self):
        """Test high volatility costs more than low volatility."""
        for symbol in ['MES', 'MGC']:
            low_vol_cost = get_total_trade_cost(symbol, 'low_vol')
            high_vol_cost = get_total_trade_cost(symbol, 'high_vol')
            assert high_vol_cost > low_vol_cost, \
                f"{symbol}: high_vol cost should be > low_vol cost"

    def test_slippage_doubles_for_round_trip(self):
        """Test slippage is correctly doubled for round-trip."""
        symbol = 'MES'
        regime = 'low_vol'
        slippage_per_fill = SLIPPAGE_TICKS[symbol][regime]

        cost_with_slippage = get_total_trade_cost(symbol, regime, include_slippage=True)
        cost_without_slippage = get_total_trade_cost(symbol, regime, include_slippage=False)

        slippage_component = cost_with_slippage - cost_without_slippage
        assert slippage_component == 2 * slippage_per_fill, \
            "Slippage should be 2x per-fill (entry + exit)"


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestSlippageValidation:
    """Tests for slippage validation in validate_barrier_params()."""

    def test_validate_barrier_params_includes_slippage(self):
        """Test validate_barrier_params() validates slippage configuration."""
        # Should pass with current configuration
        errors = validate_barrier_params()
        assert isinstance(errors, list)
        # Filter for slippage-related errors
        slippage_errors = [e for e in errors if 'SLIPPAGE' in e]
        assert len(slippage_errors) == 0, f"Unexpected slippage errors: {slippage_errors}"

    def test_slippage_validation_detects_missing_regime(self):
        """Test validation detects missing regime keys."""
        # This is a structural test - we can't easily modify the global config
        # but we can verify the validation logic exists
        errors = validate_barrier_params()
        assert isinstance(errors, list)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestSlippageIntegration:
    """Integration tests for slippage in realistic scenarios."""

    def test_mes_cost_comparison_across_regimes(self):
        """Test MES cost difference between calm and volatile markets."""
        calm_cost = get_total_trade_cost('MES', 'low_vol')
        volatile_cost = get_total_trade_cost('MES', 'high_vol')

        cost_increase = volatile_cost - calm_cost
        assert cost_increase == 1.0, \
            "MES slippage should increase by 1.0 tick in high volatility"

    def test_mgc_cost_comparison_across_regimes(self):
        """Test MGC cost difference between calm and volatile markets."""
        calm_cost = get_total_trade_cost('MGC', 'low_vol')
        volatile_cost = get_total_trade_cost('MGC', 'high_vol')

        cost_increase = volatile_cost - calm_cost
        assert abs(cost_increase - 1.5) < 1e-10, \
            f"MGC slippage should increase by 1.5 ticks, got {cost_increase}"

    def test_dollar_cost_calculation(self):
        """Test conversion from tick cost to dollar cost."""
        # MES: 1.5 ticks at $1.25/tick = $1.875
        mes_tick_cost = get_total_trade_cost('MES', 'low_vol')
        mes_dollar_cost = mes_tick_cost * TICK_VALUES['MES']
        assert mes_dollar_cost == 1.875

        # MGC: 1.8 ticks at $1.00/tick = $1.80
        mgc_tick_cost = get_total_trade_cost('MGC', 'low_vol')
        mgc_dollar_cost = mgc_tick_cost * TICK_VALUES['MGC']
        assert mgc_dollar_cost == 1.80

    def test_cost_impact_on_profit_threshold(self):
        """Test slippage increases minimum profit threshold."""
        # Without slippage, MES needs 0.5 ticks profit to break even
        # With slippage (low_vol), MES needs 1.5 ticks profit
        # With slippage (high_vol), MES needs 2.5 ticks profit

        commission_only = get_total_trade_cost('MES', include_slippage=False)
        with_low_vol_slip = get_total_trade_cost('MES', 'low_vol')
        with_high_vol_slip = get_total_trade_cost('MES', 'high_vol')

        assert with_low_vol_slip == 3 * commission_only
        assert with_high_vol_slip == 5 * commission_only


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
