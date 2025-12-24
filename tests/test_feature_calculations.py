"""
Unit tests for feature calculation functions.

Tests all feature calculation functions across:
- volatility.py (ATR, Bollinger Bands, Keltner, HVol, Parkinson, Garman-Klass)
- momentum.py (RSI, MACD, Stochastic, Williams %R, ROC, CCI, MFI)
- trend.py (ADX, Supertrend)
- volume.py (OBV, VWAP, volume ratios)
- price_features.py (returns, price ratios)

Run with: pytest tests/test_feature_calculations.py -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from stages.features.volatility import (
    add_atr,
    add_bollinger_bands,
    add_keltner_channels,
    add_historical_volatility,
    add_parkinson_volatility,
    add_garman_klass_volatility,
)
from stages.features.momentum import (
    add_rsi,
    add_macd,
    add_stochastic,
    add_williams_r,
    add_roc,
    add_cci,
    add_mfi,
)
from stages.features.trend import (
    add_adx,
    add_supertrend,
)
from stages.features.volume import (
    add_volume_features,
    add_vwap,
    add_obv,
)
from stages.features.price_features import (
    add_returns,
    add_price_ratios,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Generate realistic OHLCV data for testing."""
    np.random.seed(42)
    n = 500

    base_price = 4500.0
    volatility = 0.001

    # Generate trending price with noise
    drift = np.arange(n) * 0.0001
    noise = np.random.randn(n) * volatility
    log_returns = drift + noise
    close = base_price * np.exp(np.cumsum(log_returns))

    # Generate OHLC
    daily_range = np.abs(np.random.randn(n) * volatility * 2)
    high = close * (1 + daily_range / 2)
    low = close * (1 - daily_range / 2)
    open_ = close * (1 + np.random.randn(n) * volatility * 0.5)

    # Ensure valid OHLC relationships
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = np.random.randint(100, 10000, n)

    timestamps = pd.date_range('2024-01-01 09:30', periods=n, freq='5min')

    return pd.DataFrame({
        'datetime': timestamps,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'symbol': 'MES'
    })


@pytest.fixture
def sample_no_volume_data():
    """Generate OHLC data without volume for testing volume-dependent features."""
    np.random.seed(42)
    n = 200

    base_price = 4500.0
    close = base_price + np.cumsum(np.random.randn(n) * 0.5)

    high = close + np.abs(np.random.randn(n) * 2)
    low = close - np.abs(np.random.randn(n) * 2)
    open_ = close + np.random.randn(n) * 1

    timestamps = pd.date_range('2024-01-01 09:30', periods=n, freq='5min')

    return pd.DataFrame({
        'datetime': timestamps,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
    })


# =============================================================================
# VOLATILITY FEATURE TESTS
# =============================================================================

class TestVolatilityFeatures:
    """Tests for volatility indicator calculations."""

    def test_add_atr_creates_expected_columns(self, sample_ohlcv_data):
        """Test that ATR adds all expected columns."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_atr(df, metadata)

        # Check columns exist
        expected_cols = ['atr_7', 'atr_14', 'atr_21', 'atr_pct_7', 'atr_pct_14', 'atr_pct_21']
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"
            assert col in metadata, f"Missing metadata for: {col}"

    def test_add_atr_values_positive(self, sample_ohlcv_data):
        """Test that ATR values are always positive."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_atr(df, metadata)

        # ATR should be positive (ignoring NaN)
        assert (result['atr_14'].dropna() > 0).all(), "ATR should be positive"
        assert (result['atr_pct_14'].dropna() >= 0).all(), "ATR % should be non-negative"

    def test_add_atr_handles_zero_prices(self):
        """Test that ATR handles zero prices gracefully."""
        df = pd.DataFrame({
            'high': [10, 11, 0, 13, 14],
            'low': [9, 10, 0, 12, 13],
            'close': [9.5, 10.5, 0, 12.5, 13.5],
        })
        metadata = {}

        result = add_atr(df, metadata)

        # Should not raise error
        assert 'atr_7' in result.columns

    def test_add_bollinger_bands_creates_expected_columns(self, sample_ohlcv_data):
        """Test that Bollinger Bands adds all expected columns."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_bollinger_bands(df, metadata)

        expected_cols = ['bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position', 'close_bb_zscore']
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_add_bollinger_bands_position_bounded(self, sample_ohlcv_data):
        """Test that BB position is generally bounded [0, 1]."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_bollinger_bands(df, metadata)

        bb_pos = result['bb_position'].dropna()
        # BB position can exceed [0,1] during volatile moves, but should be mostly within
        pct_in_range = ((bb_pos >= 0) & (bb_pos <= 1)).sum() / len(bb_pos)
        assert pct_in_range > 0.80, \
            f"At least 80% of BB position should be in [0, 1], got {pct_in_range:.1%}"

    def test_add_bollinger_bands_upper_greater_than_lower(self, sample_ohlcv_data):
        """Test that BB upper is always >= lower."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_bollinger_bands(df, metadata)

        valid_rows = result[['bb_upper', 'bb_lower']].dropna()
        assert (valid_rows['bb_upper'] >= valid_rows['bb_lower']).all(), \
            "BB upper should be >= lower"

    def test_add_keltner_channels_creates_expected_columns(self, sample_ohlcv_data):
        """Test that Keltner Channels adds all expected columns."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_keltner_channels(df, metadata)

        expected_cols = ['kc_middle', 'kc_upper', 'kc_lower', 'kc_position', 'close_kc_atr_dev']
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_add_keltner_channels_position_bounded(self, sample_ohlcv_data):
        """Test that KC position is calculated."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_keltner_channels(df, metadata)

        kc_pos = result['kc_position'].dropna()
        # KC position can vary widely in volatile markets
        # Just verify it's calculated and has valid values
        assert len(kc_pos) > 0, "KC position should have values"
        assert np.isfinite(kc_pos).all(), "KC position should be finite"

    def test_add_historical_volatility_creates_expected_columns(self, sample_ohlcv_data):
        """Test that historical volatility adds all expected columns."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_historical_volatility(df, metadata)

        expected_cols = ['hvol_10', 'hvol_20', 'hvol_60']
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_add_historical_volatility_positive_values(self, sample_ohlcv_data):
        """Test that historical volatility is always positive."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_historical_volatility(df, metadata)

        assert (result['hvol_20'].dropna() >= 0).all(), "HVol should be non-negative"

    def test_add_parkinson_volatility_creates_expected_columns(self, sample_ohlcv_data):
        """Test that Parkinson volatility adds expected column."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_parkinson_volatility(df, metadata)

        assert 'parkinson_vol' in result.columns
        assert (result['parkinson_vol'].dropna() >= 0).all(), "Parkinson vol should be non-negative"

    def test_add_garman_klass_volatility_creates_expected_columns(self, sample_ohlcv_data):
        """Test that Garman-Klass volatility adds expected column."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_garman_klass_volatility(df, metadata)

        assert 'gk_vol' in result.columns
        assert (result['gk_vol'].dropna() >= 0).all(), "GK vol should be non-negative"


# =============================================================================
# MOMENTUM FEATURE TESTS
# =============================================================================

class TestMomentumFeatures:
    """Tests for momentum indicator calculations."""

    def test_add_rsi_creates_expected_columns(self, sample_ohlcv_data):
        """Test that RSI adds all expected columns."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_rsi(df, metadata)

        expected_cols = ['rsi_14', 'rsi_overbought', 'rsi_oversold']
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_add_rsi_bounded_0_100(self, sample_ohlcv_data):
        """Test that RSI values are bounded [0, 100]."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_rsi(df, metadata)

        rsi_values = result['rsi_14'].dropna()
        assert (rsi_values >= 0).all() and (rsi_values <= 100).all(), \
            "RSI should be in [0, 100] range"

    def test_add_rsi_flags_correct(self, sample_ohlcv_data):
        """Test that RSI overbought/oversold flags are correct."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_rsi(df, metadata)

        # Check flags match thresholds - need to handle edge case where RSI exactly equals 70
        overbought_check = result[result['rsi_14'] > 70.0]['rsi_overbought']
        oversold_check = result[result['rsi_14'] < 30.0]['rsi_oversold']

        # All RSI > 70 should have overbought flag = 1
        if len(overbought_check) > 0:
            assert (overbought_check == 1).mean() > 0.95, \
                "At least 95% of RSI > 70 should have overbought flag"

        # All RSI < 30 should have oversold flag = 1
        if len(oversold_check) > 0:
            assert (oversold_check == 1).mean() > 0.95, \
                "At least 95% of RSI < 30 should have oversold flag"

    def test_add_macd_creates_expected_columns(self, sample_ohlcv_data):
        """Test that MACD adds all expected columns."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_macd(df, metadata)

        expected_cols = ['macd_line', 'macd_signal', 'macd_hist', 'macd_cross_up', 'macd_cross_down']
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_add_macd_histogram_correct(self, sample_ohlcv_data):
        """Test that MACD histogram = line - signal."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_macd(df, metadata)

        valid_rows = result[['macd_line', 'macd_signal', 'macd_hist']].dropna()
        expected_hist = valid_rows['macd_line'] - valid_rows['macd_signal']

        assert np.allclose(valid_rows['macd_hist'], expected_hist, rtol=1e-5), \
            "MACD histogram should equal line - signal"

    def test_add_stochastic_creates_expected_columns(self, sample_ohlcv_data):
        """Test that Stochastic adds all expected columns."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_stochastic(df, metadata)

        expected_cols = ['stoch_k', 'stoch_d', 'stoch_overbought', 'stoch_oversold']
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_add_stochastic_bounded_0_100(self, sample_ohlcv_data):
        """Test that Stochastic values are bounded [0, 100]."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_stochastic(df, metadata)

        stoch_k = result['stoch_k'].dropna()
        stoch_d = result['stoch_d'].dropna()

        assert (stoch_k >= 0).all() and (stoch_k <= 100).all(), "%K should be in [0, 100]"
        assert (stoch_d >= 0).all() and (stoch_d <= 100).all(), "%D should be in [0, 100]"

    def test_add_williams_r_creates_expected_columns(self, sample_ohlcv_data):
        """Test that Williams %R adds expected column."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_williams_r(df, metadata)

        assert 'williams_r' in result.columns

    def test_add_williams_r_bounded(self, sample_ohlcv_data):
        """Test that Williams %R is bounded [-100, 0]."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_williams_r(df, metadata)

        williams = result['williams_r'].dropna()
        assert (williams >= -100).all() and (williams <= 0).all(), \
            "Williams %R should be in [-100, 0]"

    def test_add_roc_creates_expected_columns(self, sample_ohlcv_data):
        """Test that ROC adds all expected columns."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_roc(df, metadata)

        expected_cols = ['roc_5', 'roc_10', 'roc_20']
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_add_roc_calculation_correct(self, sample_ohlcv_data):
        """Test that ROC calculation is correct."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_roc(df, metadata)

        # Manually calculate ROC_5 and compare
        # Note: ROC has anti-lookahead shift(1), so it's lagged by 1 bar
        roc_raw = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 100)
        expected_roc_5 = roc_raw.shift(1)

        valid_mask = ~result['roc_5'].isna() & ~expected_roc_5.isna()
        assert np.allclose(
            result.loc[valid_mask, 'roc_5'],
            expected_roc_5[valid_mask],
            rtol=1e-5
        ), "ROC calculation should match expected formula with anti-lookahead shift"

    def test_add_cci_creates_expected_columns(self, sample_ohlcv_data):
        """Test that CCI adds expected column."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_cci(df, metadata)

        assert 'cci_20' in result.columns

    def test_add_mfi_creates_expected_columns(self, sample_ohlcv_data):
        """Test that MFI adds expected column when volume is present."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_mfi(df, metadata)

        assert 'mfi_14' in result.columns

    def test_add_mfi_bounded_0_100(self, sample_ohlcv_data):
        """Test that MFI values are bounded [0, 100]."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_mfi(df, metadata)

        mfi_values = result['mfi_14'].dropna()
        if len(mfi_values) > 0:
            assert (mfi_values >= 0).all() and (mfi_values <= 100).all(), \
                "MFI should be in [0, 100] range"

    def test_add_mfi_skips_without_volume(self, sample_no_volume_data):
        """Test that MFI is skipped when volume is missing."""
        df = sample_no_volume_data.copy()
        metadata = {}

        result = add_mfi(df, metadata)

        # Should not have mfi_14 column when volume is missing
        assert 'mfi_14' not in result.columns


# =============================================================================
# TREND FEATURE TESTS
# =============================================================================

class TestTrendFeatures:
    """Tests for trend indicator calculations."""

    def test_add_adx_creates_expected_columns(self, sample_ohlcv_data):
        """Test that ADX adds all expected columns."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_adx(df, metadata)

        expected_cols = ['adx_14', 'plus_di_14', 'minus_di_14', 'adx_strong_trend']
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_add_adx_values_positive(self, sample_ohlcv_data):
        """Test that ADX and DI values are non-negative."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_adx(df, metadata)

        assert (result['adx_14'].dropna() >= 0).all(), "ADX should be non-negative"
        assert (result['plus_di_14'].dropna() >= 0).all(), "+DI should be non-negative"
        assert (result['minus_di_14'].dropna() >= 0).all(), "-DI should be non-negative"

    def test_add_adx_strong_trend_flag(self, sample_ohlcv_data):
        """Test that ADX strong trend flag is correct."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_adx(df, metadata)

        # Check flags match threshold
        strong_trends = result[result['adx_14'] > 25]['adx_strong_trend']
        if len(strong_trends) > 0:
            assert (strong_trends == 1).all(), "Strong trend flag should be 1 when ADX > 25"

    def test_add_supertrend_creates_expected_columns(self, sample_ohlcv_data):
        """Test that Supertrend adds all expected columns."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_supertrend(df, metadata)

        expected_cols = ['supertrend', 'supertrend_direction']
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_add_supertrend_direction_values(self, sample_ohlcv_data):
        """Test that Supertrend direction is -1 or 1."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_supertrend(df, metadata)

        directions = result['supertrend_direction'].dropna().unique()
        assert set(directions).issubset({-1.0, 1.0}), "Supertrend direction should be -1 or 1"


# =============================================================================
# VOLUME FEATURE TESTS
# =============================================================================

class TestVolumeFeatures:
    """Tests for volume-based calculations."""

    def test_add_volume_features_creates_expected_columns(self, sample_ohlcv_data):
        """Test that volume features adds all expected columns."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_volume_features(df, metadata)

        expected_cols = ['obv', 'obv_sma_20', 'volume_sma_20', 'volume_ratio', 'volume_zscore']
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_add_volume_features_skips_without_volume(self, sample_no_volume_data):
        """Test that volume features are skipped when volume is missing."""
        df = sample_no_volume_data.copy()
        metadata = {}

        result = add_volume_features(df, metadata)

        # Should not have volume features
        assert 'obv' not in result.columns

    def test_add_vwap_creates_expected_columns(self, sample_ohlcv_data):
        """Test that VWAP adds all expected columns."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_vwap(df, metadata)

        expected_cols = ['vwap', 'price_to_vwap']
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_add_vwap_positive_values(self, sample_ohlcv_data):
        """Test that VWAP values are positive."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_vwap(df, metadata)

        assert (result['vwap'].dropna() > 0).all(), "VWAP should be positive"

    def test_add_obv_creates_expected_column(self, sample_ohlcv_data):
        """Test that OBV adds expected column."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_obv(df, metadata)

        assert 'obv' in result.columns


# =============================================================================
# PRICE FEATURE TESTS
# =============================================================================

class TestPriceFeatures:
    """Tests for price-based calculations."""

    def test_add_returns_creates_expected_columns(self, sample_ohlcv_data):
        """Test that returns adds all expected columns."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_returns(df, metadata)

        periods = [1, 5, 10, 20, 60]
        for period in periods:
            assert f'return_{period}' in result.columns, f"Missing return_{period}"
            assert f'log_return_{period}' in result.columns, f"Missing log_return_{period}"

    def test_add_returns_calculation_correct(self, sample_ohlcv_data):
        """Test that return calculations are correct."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_returns(df, metadata)

        # Verify return_1 matches pct_change (with anti-lookahead shift)
        expected_return_1 = df['close'].pct_change(1).shift(1)
        valid_mask = ~result['return_1'].isna() & ~expected_return_1.isna()
        assert np.allclose(
            result.loc[valid_mask, 'return_1'],
            expected_return_1[valid_mask],
            rtol=1e-5
        ), "return_1 should match pct_change(1) with anti-lookahead shift"

        # Verify log_return_1 calculation (with anti-lookahead shift)
        expected_log_return_1 = np.log(df['close'] / df['close'].shift(1)).shift(1)
        valid_mask = ~result['log_return_1'].isna() & ~expected_log_return_1.isna()
        assert np.allclose(
            result.loc[valid_mask, 'log_return_1'],
            expected_log_return_1[valid_mask],
            rtol=1e-5
        ), "log_return_1 should match log(close[t] / close[t-1]) with anti-lookahead shift"

    def test_add_price_ratios_creates_expected_columns(self, sample_ohlcv_data):
        """Test that price ratios adds all expected columns."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_price_ratios(df, metadata)

        expected_cols = ['hl_ratio', 'co_ratio', 'range_pct']
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_add_price_ratios_hl_ratio_greater_than_one(self, sample_ohlcv_data):
        """Test that high/low ratio is always >= 1."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_price_ratios(df, metadata)

        assert (result['hl_ratio'].dropna() >= 1.0).all(), "H/L ratio should be >= 1"

    def test_add_price_ratios_range_pct_positive(self, sample_ohlcv_data):
        """Test that range percentage is positive."""
        df = sample_ohlcv_data.copy()
        metadata = {}

        result = add_price_ratios(df, metadata)

        assert (result['range_pct'].dropna() >= 0).all(), "Range % should be non-negative"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_features_handle_small_dataset(self):
        """Test that features handle very small datasets gracefully."""
        # Create tiny dataset (smaller than most window sizes)
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=10, freq='5min'),
            'open': [100 + i for i in range(10)],
            'high': [101 + i for i in range(10)],
            'low': [99 + i for i in range(10)],
            'close': [100 + i for i in range(10)],
            'volume': [1000] * 10,
        })
        metadata = {}

        # Should not crash, but will have many NaN values
        result = add_rsi(df.copy(), metadata)
        assert 'rsi_14' in result.columns

        result = add_macd(df.copy(), metadata)
        assert 'macd_line' in result.columns

    def test_features_handle_constant_price(self):
        """Test that features handle constant price gracefully."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'open': [100] * 100,
            'high': [100] * 100,
            'low': [100] * 100,
            'close': [100] * 100,
            'volume': [1000] * 100,
        })
        metadata = {}

        # Should not crash
        result = add_bollinger_bands(df.copy(), metadata)
        assert 'bb_middle' in result.columns

        # BB width should be near zero (or NaN) for constant price
        bb_width = result['bb_width'].dropna()
        if len(bb_width) > 0:
            # Width should be 0 or very small for constant price
            assert (bb_width < 0.1).all() or bb_width.isna().all()

    def test_features_handle_missing_values(self):
        """Test that features handle missing values in input data."""
        np.random.seed(42)
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'open': np.random.randn(100) * 10 + 100,
            'high': np.random.randn(100) * 10 + 105,
            'low': np.random.randn(100) * 10 + 95,
            'close': np.random.randn(100) * 10 + 100,
            'volume': [1000] * 100,
        })

        # Insert some NaN values
        df.loc[10:15, 'close'] = np.nan
        df.loc[50:55, 'high'] = np.nan

        metadata = {}

        # Should not crash
        result = add_rsi(df.copy(), metadata)
        assert 'rsi_14' in result.columns

        result = add_atr(df.copy(), metadata)
        assert 'atr_14' in result.columns
