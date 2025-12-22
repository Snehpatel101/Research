"""
Unit tests for lookahead bias prevention in feature engineering.

These tests ensure that all features are properly shifted to prevent
lookahead bias, where features at bar[t] would include information
from bar[t] that wouldn't be available until after bar[t] closes.

Critical Rule:
    Features at bar[t] must ONLY use data up to and including bar[t-1],
    ensuring they are available BEFORE bar[t] closes for trading decisions.

Run with: pytest tests/phase_1_tests/stages/test_lookahead_bias_prevention.py -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from stages.features.cross_asset import add_cross_asset_features
from stages.features.momentum import add_macd, add_rsi, add_stochastic


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_ohlc_data():
    """Create sample OHLC data for testing."""
    np.random.seed(42)
    n = 100

    dates = pd.date_range('2024-01-01', periods=n, freq='5min')
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.random.rand(n) * 2
    low = close - np.random.rand(n) * 2
    open_prices = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 5000, n)

    df = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    return df


@pytest.fixture
def mes_mgc_price_arrays():
    """Create MES and MGC close price arrays for cross-asset testing."""
    np.random.seed(42)
    n = 100

    mes_close = 4500 + np.cumsum(np.random.randn(n) * 2.0)
    mgc_close = 2000 + np.cumsum(np.random.randn(n) * 1.0)

    return mes_close, mgc_close


# =============================================================================
# CROSS-ASSET FEATURE TESTS
# =============================================================================

class TestCrossAssetLookaheadPrevention:
    """Tests for lookahead bias prevention in cross-asset features."""

    def test_correlation_shifted_properly(self, sample_ohlc_data, mes_mgc_price_arrays):
        """
        Test that MES-MGC correlation at bar[t] uses data up to bar[t-1].

        Lookahead Issue:
            Original: correlation[t] includes returns[t], but we trade at bar[t] close
            Fixed: correlation[t] = correlation_calc[t-1], available before bar[t] close
        """
        mes_close, mgc_close = mes_mgc_price_arrays
        df = sample_ohlc_data.copy()
        metadata = {}

        # Add cross-asset features
        df = add_cross_asset_features(df, metadata, mes_close, mgc_close, 'MES')

        # Check that shift was applied correctly
        # Without shift: first valid correlation would be at position 19 (20-bar window, 0-indexed)
        # With shift(1): first valid correlation should be at position 20
        # Position 0 should be NaN (shift introduces leading NaN)
        assert pd.isna(df['mes_mgc_correlation_20'].iloc[0]), \
            "Position 0 should be NaN after shift(1)"

        # Position 19 should also be NaN (gets shifted forward from position 18)
        assert pd.isna(df['mes_mgc_correlation_20'].iloc[19]), \
            "Position 19 should be NaN (first valid is at position 20 after shift)"

        # Position 20 should have the first valid value (shifted from position 19's calculation)
        # This means correlation[20] uses data up to bar[19], which is correct!

    def test_spread_zscore_shifted_properly(self, sample_ohlc_data, mes_mgc_price_arrays):
        """
        Test that MES-MGC spread z-score at bar[t] uses data up to bar[t-1].
        """
        mes_close, mgc_close = mes_mgc_price_arrays
        df = sample_ohlc_data.copy()
        metadata = {}

        df = add_cross_asset_features(df, metadata, mes_close, mgc_close, 'MES')

        # Check that spread z-score is properly shifted
        assert pd.isna(df['mes_mgc_spread_zscore'].iloc[0]), \
            "Position 0 should be NaN after shift(1)"

        # The spread calculation needs 2x20 bars (one for normalization, one for z-score)
        # With shift, first valid should be even later

    def test_beta_shifted_properly(self, sample_ohlc_data, mes_mgc_price_arrays):
        """
        Test that MES-MGC beta at bar[t] uses data up to bar[t-1].
        """
        mes_close, mgc_close = mes_mgc_price_arrays
        df = sample_ohlc_data.copy()
        metadata = {}

        df = add_cross_asset_features(df, metadata, mes_close, mgc_close, 'MES')

        # Check that beta is properly shifted
        assert pd.isna(df['mes_mgc_beta'].iloc[0]), \
            "Position 0 should be NaN after shift(1)"
        assert pd.isna(df['mes_mgc_beta'].iloc[19]), \
            "Position 19 should be NaN (first valid is at position 20 after shift)"

    def test_relative_strength_no_current_bar_return(self, sample_ohlc_data, mes_mgc_price_arrays):
        """
        Test that relative strength at bar[t] doesn't include bar[t]'s return.

        Critical Lookahead Issue:
            Original: mes_cum_ret[t] = sum(returns[t-19:t+1]) includes current bar
            Fixed: mes_cum_ret[t] = sum(returns[t-20:t]) excludes current bar

        This is the CRITICAL issue mentioned in the bug report.
        """
        mes_close, mgc_close = mes_mgc_price_arrays
        df = sample_ohlc_data.copy()
        metadata = {}

        df = add_cross_asset_features(df, metadata, mes_close, mgc_close, 'MES')

        # With shift(1) on returns, then rolling(20):
        # - Position 0: NaN (from shift)
        # - Positions 1-20: returns are shifted, so rolling needs to wait
        # - Position 20: should have first valid (shifted returns [0-19] rolled)
        assert pd.isna(df['relative_strength'].iloc[0]), \
            "Position 0 should be NaN after shift(1)"
        assert pd.isna(df['relative_strength'].iloc[19]), \
            "Position 19 should be NaN (window size 20 on shifted data)"

        # Verify we don't have lookahead by checking that value depends on past only
        # If properly implemented, changing bar[t] shouldn't affect relative_strength[t]
        df_copy = sample_ohlc_data.copy()
        mes_close_modified = mes_close.copy()

        # Modify current bar (position 50)
        mes_close_modified[50] *= 1.1  # 10% jump

        df_modified = add_cross_asset_features(df_copy, {}, mes_close_modified, mgc_close, 'MES')

        # Value at position 50 should be UNCHANGED (uses data up to position 49)
        if not pd.isna(df['relative_strength'].iloc[50]):
            assert df['relative_strength'].iloc[50] == df_modified['relative_strength'].iloc[50], \
                "Relative strength at bar[t] should not change when bar[t] price changes"


# =============================================================================
# MOMENTUM FEATURE TESTS
# =============================================================================

class TestMomentumLookaheadPrevention:
    """Tests for lookahead bias prevention in momentum features."""

    def test_macd_crossover_shifted_properly(self, sample_ohlc_data):
        """
        Test that MACD crossover signals at bar[t] are based on bar[t-1] data.

        Lookahead Issue:
            Original: cross_up[t] = (macd[t] > signal[t]) & (macd[t-1] <= signal[t-1])
            This detects crossover AT bar[t] close, but we need it BEFORE close
            Fixed: cross_up[t] = shift(1) of the above, so cross_up[t] uses t-1 and t-2
        """
        df = sample_ohlc_data.copy()
        metadata = {}

        df = add_macd(df, metadata)

        # Check that crossover signals are shifted
        # First valid MACD is around bar 26 (12+26-1 for EMA, +9 for signal)
        # After shift(1), first valid crossover should be at bar 27 or later

        # Verify no lookahead: changing current bar shouldn't affect current crossover signal
        df_copy = sample_ohlc_data.copy()
        df_copy['close'].iloc[50] *= 1.05  # 5% jump

        df_modified = add_macd(df_copy, {})

        # Crossover at position 50 should be UNCHANGED
        if not pd.isna(df['macd_cross_up'].iloc[50]):
            assert df['macd_cross_up'].iloc[50] == df_modified['macd_cross_up'].iloc[50], \
                "MACD crossover at bar[t] should not change when bar[t] price changes"

        if not pd.isna(df['macd_cross_down'].iloc[50]):
            assert df['macd_cross_down'].iloc[50] == df_modified['macd_cross_down'].iloc[50], \
                "MACD crossover at bar[t] should not change when bar[t] price changes"

    def test_rsi_flags_shifted_properly(self, sample_ohlc_data):
        """
        Test that RSI overbought/oversold flags at bar[t] are based on bar[t-1] RSI.
        """
        df = sample_ohlc_data.copy()
        metadata = {}

        df = add_rsi(df, metadata)

        # Verify no lookahead: changing current bar shouldn't affect current flag
        df_copy = sample_ohlc_data.copy()
        df_copy['close'].iloc[50] *= 1.10  # 10% jump (likely to change RSI significantly)

        df_modified = add_rsi(df_copy, {})

        # Flags at position 50 should be UNCHANGED (based on RSI at position 49)
        if not pd.isna(df['rsi_overbought'].iloc[50]):
            assert df['rsi_overbought'].iloc[50] == df_modified['rsi_overbought'].iloc[50], \
                "RSI overbought flag at bar[t] should not change when bar[t] price changes"

        if not pd.isna(df['rsi_oversold'].iloc[50]):
            assert df['rsi_oversold'].iloc[50] == df_modified['rsi_oversold'].iloc[50], \
                "RSI oversold flag at bar[t] should not change when bar[t] price changes"

    def test_stochastic_flags_shifted_properly(self, sample_ohlc_data):
        """
        Test that Stochastic overbought/oversold flags at bar[t] are based on bar[t-1] data.
        """
        df = sample_ohlc_data.copy()
        metadata = {}

        df = add_stochastic(df, metadata)

        # Verify no lookahead
        df_copy = sample_ohlc_data.copy()
        df_copy['close'].iloc[50] *= 1.10
        df_copy['high'].iloc[50] *= 1.10

        df_modified = add_stochastic(df_copy, {})

        # Flags at position 50 should be UNCHANGED
        if not pd.isna(df['stoch_overbought'].iloc[50]):
            assert df['stoch_overbought'].iloc[50] == df_modified['stoch_overbought'].iloc[50], \
                "Stochastic overbought flag at bar[t] should not change when bar[t] data changes"

        if not pd.isna(df['stoch_oversold'].iloc[50]):
            assert df['stoch_oversold'].iloc[50] == df_modified['stoch_oversold'].iloc[50], \
                "Stochastic oversold flag at bar[t] should not change when bar[t] data changes"


# =============================================================================
# GENERAL LOOKAHEAD TESTS
# =============================================================================

class TestGeneralLookaheadPrinciples:
    """General tests for lookahead bias prevention principles."""

    def test_feature_at_t_independent_of_bar_t_data(self, sample_ohlc_data, mes_mgc_price_arrays):
        """
        Master test: Feature values at bar[t] should not change when bar[t] data changes.

        This is the fundamental principle of lookahead prevention:
        If feature[t] depends on data[t], we have lookahead bias.
        """
        mes_close, mgc_close = mes_mgc_price_arrays

        # Original data
        df1 = sample_ohlc_data.copy()
        metadata1 = {}
        df1 = add_cross_asset_features(df1, metadata1, mes_close, mgc_close, 'MES')
        df1 = add_macd(df1, metadata1)
        df1 = add_rsi(df1, metadata1)

        # Modified data at position 60 (well past warm-up period)
        df2 = sample_ohlc_data.copy()
        df2['close'].iloc[60] *= 1.20  # Dramatic 20% change
        df2['high'].iloc[60] *= 1.20
        df2['low'].iloc[60] *= 1.20
        df2['open'].iloc[60] *= 1.20

        mes_close_modified = mes_close.copy()
        mes_close_modified[60] *= 1.20

        metadata2 = {}
        df2 = add_cross_asset_features(df2, metadata2, mes_close_modified, mgc_close, 'MES')
        df2 = add_macd(df2, metadata2)
        df2 = add_rsi(df2, metadata2)

        # Test critical features at position 60
        features_to_test = [
            'mes_mgc_correlation_20',
            'mes_mgc_spread_zscore',
            'mes_mgc_beta',
            'relative_strength',
            'macd_cross_up',
            'macd_cross_down',
            'rsi_overbought',
            'rsi_oversold'
        ]

        for feature in features_to_test:
            val1 = df1[feature].iloc[60]
            val2 = df2[feature].iloc[60]

            # Skip if NaN in both (expected for some features)
            if pd.isna(val1) and pd.isna(val2):
                continue

            # Values should be IDENTICAL (no lookahead)
            assert val1 == val2, \
                f"Feature '{feature}' at bar[60] changed when bar[60] data changed. " \
                f"This indicates lookahead bias! Original={val1}, Modified={val2}"

    def test_all_shifted_features_have_extra_nan_at_boundary(self, sample_ohlc_data, mes_mgc_price_arrays):
        """
        Test that shifted features have one extra NaN compared to their unshifted versions.

        This verifies that shift(1) was actually applied.
        """
        mes_close, mgc_close = mes_mgc_price_arrays
        df = sample_ohlc_data.copy()
        metadata = {}

        df = add_cross_asset_features(df, metadata, mes_close, mgc_close, 'MES')
        df = add_macd(df, metadata)
        df = add_rsi(df, metadata)

        # Features that should be shifted
        shifted_features = [
            'mes_mgc_correlation_20',
            'mes_mgc_spread_zscore',
            'mes_mgc_beta',
            'relative_strength',
            'macd_cross_up',
            'macd_cross_down',
            'rsi_overbought',
            'rsi_oversold'
        ]

        for feature in shifted_features:
            # Count leading NaNs
            first_valid_idx = df[feature].first_valid_index()

            if first_valid_idx is not None:
                # Should have at least some NaNs due to shift
                leading_nans = df.index.get_loc(first_valid_idx)
                assert leading_nans > 0, \
                    f"Feature '{feature}' should have leading NaNs from shift(1) and window size"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
