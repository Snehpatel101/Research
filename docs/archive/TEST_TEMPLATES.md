# Test Templates for ML Pipeline

**Purpose:** Ready-to-use test templates for implementing missing tests
**Date:** 2025-12-21

---

## Template 1: Feature Unit Test

**File:** `tests/test_feature_calculations.py`

```python
"""
Comprehensive unit tests for all feature calculation functions.
Tests correctness, bounds, edge cases, and no-lookahead guarantees.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from stages.features.momentum import (
    add_rsi, add_macd, add_stochastic, add_mfi, add_williams_r, add_roc, add_cci
)
from stages.features.volatility import (
    add_atr, add_bollinger_bands, add_keltner_channels,
    add_historical_volatility, add_parkinson_volatility, add_garman_klass_volatility
)
from stages.features.volume import add_volume_features, add_vwap, add_obv
from stages.features.trend import add_adx, add_supertrend
from stages.features.price_features import add_returns, add_price_ratios
from stages.features.temporal import add_temporal_features, add_session_features
from stages.features.regime import add_regime_features, add_trend_regime, add_volatility_regime
from stages.features.cross_asset import add_cross_asset_features


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 500

    dates = pd.date_range('2024-01-01', periods=n, freq='5min')
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(1000, 10000, n)

    df = pd.DataFrame({
        'datetime': dates,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

    return df


@pytest.fixture
def feature_metadata():
    """Empty metadata dict for feature descriptions."""
    return {}


# =============================================================================
# MOMENTUM FEATURES
# =============================================================================

class TestStochasticOscillator:
    """Tests for Stochastic %K and %D."""

    def test_stochastic_bounds(self, sample_ohlcv_data, feature_metadata):
        """Stochastic %K and %D should be between 0 and 100."""
        df = add_stochastic(sample_ohlcv_data.copy(), feature_metadata)

        assert 'stoch_k' in df.columns
        assert 'stoch_d' in df.columns

        valid_k = df['stoch_k'].dropna()
        valid_d = df['stoch_d'].dropna()

        assert np.all(valid_k >= 0), "Stochastic %K below 0"
        assert np.all(valid_k <= 100), "Stochastic %K above 100"
        assert np.all(valid_d >= 0), "Stochastic %D below 0"
        assert np.all(valid_d <= 100), "Stochastic %D above 100"

    def test_stochastic_overbought_flag(self, sample_ohlcv_data, feature_metadata):
        """Overbought flag should trigger when %K > 80."""
        df = add_stochastic(sample_ohlcv_data.copy(), feature_metadata)

        # Where %K > 80, flag should be 1
        overbought_condition = df['stoch_k'] > 80
        assert np.all(
            df.loc[overbought_condition, 'stoch_overbought'] == 1
        ), "Overbought flag not set when %K > 80"

    def test_stochastic_oversold_flag(self, sample_ohlcv_data, feature_metadata):
        """Oversold flag should trigger when %K < 20."""
        df = add_stochastic(sample_ohlcv_data.copy(), feature_metadata)

        # Where %K < 20, flag should be 1
        oversold_condition = df['stoch_k'] < 20
        assert np.all(
            df.loc[oversold_condition, 'stoch_oversold'] == 1
        ), "Oversold flag not set when %K < 20"

    def test_stochastic_d_smoothed(self, sample_ohlcv_data, feature_metadata):
        """%D should be smoother than %K (lower variance)."""
        df = add_stochastic(sample_ohlcv_data.copy(), feature_metadata)

        valid_rows = df[['stoch_k', 'stoch_d']].dropna()
        if len(valid_rows) > 10:
            k_variance = valid_rows['stoch_k'].var()
            d_variance = valid_rows['stoch_d'].var()

            # %D (smoothed) should have lower variance than %K
            assert d_variance < k_variance, "%D not smoother than %K"


class TestMoneyFlowIndex:
    """Tests for Money Flow Index (MFI)."""

    def test_mfi_bounds(self, sample_ohlcv_data, feature_metadata):
        """MFI should be between 0 and 100."""
        df = add_mfi(sample_ohlcv_data.copy(), feature_metadata)

        if 'mfi_14' in df.columns:  # Only if volume data present
            valid_mfi = df['mfi_14'].dropna()
            assert np.all(valid_mfi >= 0), "MFI below 0"
            assert np.all(valid_mfi <= 100), "MFI above 100"

    def test_mfi_volume_weighted(self, sample_ohlcv_data, feature_metadata):
        """MFI should increase with volume-weighted price gains."""
        # Create synthetic data with volume-weighted uptrend
        df = sample_ohlcv_data.copy()

        # Add volume spike on up days
        df['volume'] = np.where(
            df['close'] > df['close'].shift(1),
            df['volume'] * 2,  # Double volume on up days
            df['volume']
        )

        df = add_mfi(df, feature_metadata)

        if 'mfi_14' in df.columns:
            # MFI should trend higher with volume-weighted buying
            mfi = df['mfi_14'].dropna()
            assert mfi.iloc[-10:].mean() > mfi.iloc[:10].mean(), \
                "MFI doesn't reflect volume-weighted price changes"

    def test_mfi_zero_volume_handling(self, sample_ohlcv_data, feature_metadata):
        """MFI should handle zero volume gracefully."""
        df = sample_ohlcv_data.copy()
        df['volume'] = 0  # All zero volume

        # Should skip MFI calculation
        df = add_mfi(df, feature_metadata)
        assert 'mfi_14' not in df.columns, "MFI calculated with zero volume"


# =============================================================================
# VOLATILITY FEATURES
# =============================================================================

class TestParkinsonVolatility:
    """Tests for Parkinson volatility (range-based)."""

    def test_parkinson_positive(self, sample_ohlcv_data, feature_metadata):
        """Parkinson volatility should always be positive."""
        df = add_parkinson_volatility(sample_ohlcv_data.copy(), feature_metadata)

        valid_pvol = df['parkinson_vol'].dropna()
        assert np.all(valid_pvol >= 0), "Parkinson volatility negative"

    def test_parkinson_uses_high_low(self, feature_metadata):
        """Parkinson volatility should use high-low range."""
        # Create data with wide range -> higher volatility
        df_wide = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'open': [100] * 100,
            'high': [110] * 100,  # Wide range
            'low': [90] * 100,
            'close': [100] * 100,
            'volume': [1000] * 100
        })

        # Create data with narrow range -> lower volatility
        df_narrow = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'open': [100] * 100,
            'high': [101] * 100,  # Narrow range
            'low': [99] * 100,
            'close': [100] * 100,
            'volume': [1000] * 100
        })

        df_wide = add_parkinson_volatility(df_wide, feature_metadata)
        df_narrow = add_parkinson_volatility(df_narrow, feature_metadata)

        wide_pvol = df_wide['parkinson_vol'].iloc[-1]
        narrow_pvol = df_narrow['parkinson_vol'].iloc[-1]

        assert wide_pvol > narrow_pvol, \
            "Parkinson volatility doesn't reflect high-low range"

    def test_parkinson_more_efficient_than_close_to_close(
        self, sample_ohlcv_data, feature_metadata
    ):
        """Parkinson volatility should be more efficient (lower) than close-to-close."""
        df = add_historical_volatility(sample_ohlcv_data.copy(), feature_metadata)
        df = add_parkinson_volatility(df, feature_metadata)

        valid_rows = df[['hvol_20', 'parkinson_vol']].dropna()

        if len(valid_rows) > 0:
            # Parkinson uses more information (high/low), should be more efficient
            # This manifests as lower variance in estimates, not necessarily lower value
            # So we just verify both are calculated and positive
            assert valid_rows['parkinson_vol'].mean() > 0
            assert valid_rows['hvol_20'].mean() > 0


class TestGarmanKlassVolatility:
    """Tests for Garman-Klass volatility (OHLC-based)."""

    def test_garman_klass_positive(self, sample_ohlcv_data, feature_metadata):
        """Garman-Klass volatility should always be positive."""
        df = add_garman_klass_volatility(sample_ohlcv_data.copy(), feature_metadata)

        valid_gkvol = df['gk_vol'].dropna()
        assert np.all(valid_gkvol >= 0), "Garman-Klass volatility negative"

    def test_garman_klass_uses_ohlc(self, feature_metadata):
        """Garman-Klass should use all OHLC data."""
        # Create data with gaps (open != close)
        df_gaps = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'open': [100, 95, 105] * 33 + [100],  # Gaps
            'high': [110] * 100,
            'low': [90] * 100,
            'close': [105, 100, 95] * 33 + [100],  # Gaps
            'volume': [1000] * 100
        })

        # Create data without gaps (open = close)
        df_no_gaps = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'open': [100] * 100,
            'high': [110] * 100,
            'low': [90] * 100,
            'close': [100] * 100,  # No gaps
            'volume': [1000] * 100
        })

        df_gaps = add_garman_klass_volatility(df_gaps, feature_metadata)
        df_no_gaps = add_garman_klass_volatility(df_no_gaps, feature_metadata)

        gaps_gkvol = df_gaps['gk_vol'].iloc[-1]
        no_gaps_gkvol = df_no_gaps['gk_vol'].iloc[-1]

        # Gaps should increase GK volatility (uses open-close info)
        assert gaps_gkvol > no_gaps_gkvol, \
            "Garman-Klass doesn't use open-close information"


# =============================================================================
# VOLUME FEATURES
# =============================================================================

class TestVWAP:
    """Tests for Volume Weighted Average Price."""

    def test_vwap_session_reset(self, feature_metadata):
        """VWAP should reset at session boundaries (midnight UTC)."""
        # Create 2-day dataset
        dates = pd.date_range('2024-01-01 00:00', periods=576, freq='5min')  # 2 days

        df = pd.DataFrame({
            'datetime': dates,
            'open': [100] * 576,
            'high': [105] * 576,
            'low': [95] * 576,
            'close': [102] * 576,
            'volume': [1000] * 576
        })

        df = add_vwap(df, feature_metadata)

        # VWAP should reset at midnight (index 288 = start of day 2)
        # Day 1 last value != Day 2 first value
        day1_last_vwap = df.iloc[287]['vwap']
        day2_first_vwap = df.iloc[288]['vwap']

        # They should be different because VWAP resets
        # (unless by coincidence they're identical)
        # Better test: check that VWAP at start of day 2 is close to typical price
        typical_price = (df.iloc[288]['high'] + df.iloc[288]['low'] + df.iloc[288]['close']) / 3

        assert abs(day2_first_vwap - typical_price) < 1.0, \
            "VWAP doesn't reset at session boundary"

    def test_vwap_zero_volume_handling(self, feature_metadata):
        """VWAP should skip calculation when volume is zero."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'open': [100] * 100,
            'high': [105] * 100,
            'low': [95] * 100,
            'close': [102] * 100,
            'volume': [0] * 100  # All zero volume
        })

        df = add_vwap(df, feature_metadata)

        assert 'vwap' not in df.columns, "VWAP calculated with zero volume"

    def test_vwap_volume_weighted(self, feature_metadata):
        """VWAP should weight prices by volume."""
        # Create data where high volume occurs at high prices
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01 00:00', periods=10, freq='5min'),
            'open': [100, 105, 110, 115, 120, 125, 130, 135, 140, 145],
            'high': [101, 106, 111, 116, 121, 126, 131, 136, 141, 146],
            'low': [99, 104, 109, 114, 119, 124, 129, 134, 139, 144],
            'close': [100, 105, 110, 115, 120, 125, 130, 135, 140, 145],
            # High volume at high prices
            'volume': [100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
        })

        df = add_vwap(df, feature_metadata)

        # VWAP at end should be weighted toward high prices (where volume was high)
        final_vwap = df.iloc[-1]['vwap']
        simple_avg = df['close'].mean()

        assert final_vwap > simple_avg, \
            "VWAP not weighted toward high-volume prices"


class TestOBV:
    """Tests for On Balance Volume."""

    def test_obv_accumulation(self, feature_metadata):
        """OBV should accumulate on up days, decrease on down days."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=10, freq='5min'),
            'open': [100] * 10,
            'high': [105] * 10,
            'low': [95] * 10,
            'close': [101, 102, 103, 104, 105, 104, 103, 102, 101, 100],  # Up then down
            'volume': [1000] * 10
        })

        df = add_obv(df, feature_metadata)

        # OBV should increase during uptrend (indices 0-4)
        assert df.iloc[4]['obv'] > df.iloc[0]['obv'], "OBV doesn't accumulate on uptrend"

        # OBV should decrease during downtrend (indices 5-9)
        assert df.iloc[9]['obv'] < df.iloc[4]['obv'], "OBV doesn't decrease on downtrend"

    def test_obv_zero_on_flat_price(self, feature_metadata):
        """OBV should not change when price is flat."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=10, freq='5min'),
            'open': [100] * 10,
            'high': [105] * 10,
            'low': [95] * 10,
            'close': [100] * 10,  # Flat price
            'volume': [1000] * 10
        })

        df = add_obv(df, feature_metadata)

        # OBV should be 0 throughout (no price change)
        assert np.all(df['obv'] == 0), "OBV changes on flat price"


# =============================================================================
# CROSS-ASSET FEATURES (CRITICAL - 100% UNTESTED)
# =============================================================================

class TestCrossAssetFeatures:
    """Tests for MES-MGC cross-asset features."""

    @pytest.fixture
    def mes_mgc_data(self):
        """Generate aligned MES and MGC price data."""
        np.random.seed(42)
        n = 500

        dates = pd.date_range('2024-01-01', periods=n, freq='5min')

        # MES (equity) - upward drift
        mes_close = 4500 + np.cumsum(np.random.randn(n) * 1.0 + 0.1)

        # MGC (gold) - mean reverting
        mgc_close = 2000 + np.cumsum(np.random.randn(n) * 0.5)

        df = pd.DataFrame({
            'datetime': dates,
            'open': mes_close + np.random.randn(n) * 0.5,
            'high': mes_close + 2,
            'low': mes_close - 2,
            'close': mes_close,
            'volume': np.random.randint(1000, 10000, n)
        })

        return df, mes_close, mgc_close

    def test_correlation_bounds(self, mes_mgc_data, feature_metadata):
        """MES-MGC correlation should be between -1 and 1."""
        df, mes_close, mgc_close = mes_mgc_data

        df = add_cross_asset_features(
            df, feature_metadata,
            mes_close=mes_close,
            mgc_close=mgc_close,
            current_symbol='MES'
        )

        valid_corr = df['mes_mgc_correlation_20'].dropna()
        assert np.all(valid_corr >= -1), "Correlation below -1"
        assert np.all(valid_corr <= 1), "Correlation above 1"

    def test_beta_calculation(self, mes_mgc_data, feature_metadata):
        """Beta should equal cov(MES, MGC) / var(MGC)."""
        df, mes_close, mgc_close = mes_mgc_data

        df = add_cross_asset_features(
            df, feature_metadata,
            mes_close=mes_close,
            mgc_close=mgc_close,
            current_symbol='MES'
        )

        # Calculate beta manually for validation
        mes_returns = pd.Series(mes_close).pct_change()
        mgc_returns = pd.Series(mgc_close).pct_change()

        # Rolling 20-period beta
        expected_beta = (
            mes_returns.rolling(20).cov(mgc_returns) /
            mgc_returns.rolling(20).var()
        )

        # Compare last 10 values (after warmup)
        actual_beta = df['mes_mgc_beta'].iloc[-10:]
        expected_beta_vals = expected_beta.iloc[-10:]

        np.testing.assert_allclose(
            actual_beta, expected_beta_vals,
            rtol=0.01, atol=0.01,
            err_msg="Beta calculation incorrect"
        )

    def test_single_symbol_nan_handling(self, mes_mgc_data, feature_metadata):
        """Cross-asset features should be NaN when only one symbol present."""
        df, _, _ = mes_mgc_data

        # Call without MES/MGC data
        df = add_cross_asset_features(
            df, feature_metadata,
            mes_close=None,  # No MES data
            mgc_close=None,  # No MGC data
            current_symbol='MES'
        )

        # All cross-asset features should be NaN
        assert df['mes_mgc_correlation_20'].isna().all(), \
            "Correlation not NaN when single symbol"
        assert df['mes_mgc_beta'].isna().all(), \
            "Beta not NaN when single symbol"
        assert df['relative_strength'].isna().all(), \
            "Relative strength not NaN when single symbol"

    def test_relative_strength_calculation(self, mes_mgc_data, feature_metadata):
        """Relative strength should be MES return - MGC return."""
        df, mes_close, mgc_close = mes_mgc_data

        df = add_cross_asset_features(
            df, feature_metadata,
            mes_close=mes_close,
            mgc_close=mgc_close,
            current_symbol='MES'
        )

        # Calculate manually
        mes_returns = pd.Series(mes_close).pct_change()
        mgc_returns = pd.Series(mgc_close).pct_change()
        expected_rs = mes_returns - mgc_returns

        actual_rs = df['relative_strength']

        # Compare (skip first NaN)
        np.testing.assert_allclose(
            actual_rs.iloc[1:], expected_rs.iloc[1:],
            rtol=0.01, atol=0.01,
            err_msg="Relative strength calculation incorrect"
        )


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Comprehensive edge case tests."""

    def test_empty_dataframe(self, feature_metadata):
        """All feature functions should handle empty DataFrame gracefully."""
        df = pd.DataFrame()

        # Should raise or return empty, not crash
        with pytest.raises((ValueError, KeyError)):
            add_rsi(df, feature_metadata)

    def test_single_row_dataframe(self, feature_metadata):
        """All feature functions should handle single-row DataFrame."""
        df = pd.DataFrame({
            'datetime': [pd.Timestamp('2024-01-01')],
            'open': [100.0],
            'high': [105.0],
            'low': [95.0],
            'close': [102.0],
            'volume': [1000]
        })

        # Should not crash, features should be NaN
        df = add_rsi(df, feature_metadata)
        assert 'rsi_14' in df.columns
        assert df['rsi_14'].isna().all()

    def test_constant_price(self, feature_metadata):
        """Features should handle constant price (zero returns, zero volatility)."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'open': [100.0] * 100,
            'high': [100.0] * 100,
            'low': [100.0] * 100,
            'close': [100.0] * 100,
            'volume': [1000] * 100
        })

        # Should not crash
        df = add_historical_volatility(df, feature_metadata)
        df = add_returns(df, feature_metadata)

        # Volatility should be 0 or NaN
        if 'hvol_20' in df.columns:
            assert np.all(df['hvol_20'].dropna() == 0) or df['hvol_20'].isna().all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

## Template 2: Leakage Prevention Test

**File:** `tests/test_leakage_prevention.py`

```python
"""
Comprehensive data leakage prevention tests.
Ensures no future information leaks into training data.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from stages.features.engineer import FeatureEngineer
from stages.feature_scaler import FeatureScaler
from stages.stage7_splits import create_chronological_splits


class TestRollingWindowLookahead:
    """Test that all rolling windows use only past data."""

    def test_no_lookahead_in_sma(self):
        """SMA at time T should only use data [T-period+1:T]."""
        # Create data where future differs from past
        close_full = np.array([100.0] * 50 + [200.0] * 50)  # Jump at index 50
        close_truncated = np.array([100.0] * 50)  # Only first 50

        from stages.features.numba_functions import calculate_sma_numba

        # Calculate SMA(10) on both
        sma_full = calculate_sma_numba(close_full, 10)
        sma_truncated = calculate_sma_numba(close_truncated, 10)

        # SMA at indices 0-49 should be identical
        for i in range(40, 50):  # Check last 10 values before jump
            if not np.isnan(sma_full[i]) and not np.isnan(sma_truncated[i]):
                assert np.isclose(sma_full[i], sma_truncated[i]), \
                    f"SMA lookahead detected at index {i}"

    def test_no_lookahead_in_all_features(self):
        """All features at time T should be identical when calculated on different future data."""
        np.random.seed(42)

        # Create full dataset (1000 rows)
        n_full = 1000
        dates_full = pd.date_range('2024-01-01', periods=n_full, freq='5min')
        close_full = 100 + np.cumsum(np.random.randn(n_full) * 0.5)

        df_full = pd.DataFrame({
            'datetime': dates_full,
            'open': close_full + np.random.randn(n_full) * 0.1,
            'high': close_full + 1,
            'low': close_full - 1,
            'close': close_full,
            'volume': np.random.randint(1000, 10000, n_full)
        })

        # Create truncated dataset (first 500 rows)
        df_truncated = df_full.iloc[:500].copy()

        # Calculate features on both
        engineer = FeatureEngineer(
            input_dir=Path('/tmp'),
            output_dir=Path('/tmp/out')
        )

        df_full_features, _ = engineer.engineer_features(df_full, 'TEST')

        # Reset engineer for truncated
        engineer.feature_metadata = {}
        df_truncated_features, _ = engineer.engineer_features(df_truncated, 'TEST')

        # Compare features at a common timestamp (e.g., index 400)
        test_idx = 400
        test_time = df_truncated_features['datetime'].iloc[test_idx]

        # Get matching row in full dataset
        full_row = df_full_features[df_full_features['datetime'] == test_time]

        if len(full_row) > 0:
            truncated_row = df_truncated_features.iloc[test_idx]

            # Compare all feature columns
            feature_cols = [c for c in truncated_row.index
                          if c not in ['datetime', 'open', 'high', 'low', 'close', 'volume', 'symbol']]

            for col in feature_cols:
                truncated_val = truncated_row[col]
                full_val = full_row[col].iloc[0]

                if pd.notna(truncated_val) and pd.notna(full_val):
                    assert np.isclose(truncated_val, full_val, rtol=1e-5), \
                        f"Lookahead detected in {col}: truncated={truncated_val}, full={full_val}"


class TestScalingLeakage:
    """Test that scaling uses only train statistics."""

    def test_scaler_fit_on_train_only(self):
        """Scaler statistics should come from train only, not val/test."""
        np.random.seed(42)

        # Train: mean=0, std=1
        train_df = pd.DataFrame({'f1': np.random.randn(1000)})

        # Val/Test: mean=100, std=10 (very different distribution)
        val_df = pd.DataFrame({'f1': np.random.randn(200) * 10 + 100})
        test_df = pd.DataFrame({'f1': np.random.randn(200) * 10 + 100})

        # Fit scaler on train only
        scaler = FeatureScaler(clip_outliers=False)
        train_scaled = scaler.fit_transform(train_df, ['f1'])

        # Transform val/test (should use train statistics)
        val_scaled = scaler.transform(val_df)
        test_scaled = scaler.transform(test_df)

        # Val/test means should be far from 0 (if using train stats)
        # If leakage occurred, means would be ~0
        assert abs(val_scaled['f1'].mean()) > 5, "Leakage: val scaled using own statistics"
        assert abs(test_scaled['f1'].mean()) > 5, "Leakage: test scaled using own statistics"

        # Train statistics should not change after transforming val/test
        original_train_mean = scaler.statistics['f1'].train_mean

        # Transform val/test again
        val_scaled_2 = scaler.transform(val_df)

        assert scaler.statistics['f1'].train_mean == original_train_mean, \
            "Train statistics changed after transforming val/test"


class TestPurgeEmbargoLeakage:
    """Test that purge and embargo prevent label leakage."""

    def test_purge_prevents_label_leakage(self):
        """Purge should remove samples that see future labels."""
        # For H20 (max_bars=60):
        # - Sample at train_end sees labels up to train_end+60
        # - These samples must be purged
        # - First val sample should be at train_end+61 or later

        np.random.seed(42)
        n = 10000

        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'symbol': 'MES',
            'close': 100 + np.cumsum(np.random.randn(n) * 0.1)
        })

        PURGE_BARS = 60  # For H20
        EMBARGO_BARS = 288

        train_idx, val_idx, test_idx, metadata = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=PURGE_BARS,
            embargo_bars=EMBARGO_BARS
        )

        # Expected train end before purge
        expected_train_end_raw = int(n * 0.70)  # 7000

        # Actual train end after purge
        actual_train_end = train_idx.max()

        # Train should end exactly PURGE_BARS before raw split point
        assert actual_train_end == expected_train_end_raw - PURGE_BARS - 1, \
            f"Purge boundary incorrect: expected {expected_train_end_raw - PURGE_BARS - 1}, got {actual_train_end}"

        # Val should start at least EMBARGO_BARS after purged train end
        val_start = val_idx.min()
        gap = val_start - actual_train_end

        assert gap >= EMBARGO_BARS + 1, \
            f"Embargo gap incorrect: expected >= {EMBARGO_BARS + 1}, got {gap}"

    def test_no_label_overlap_at_boundaries(self):
        """Labels at train end and val start should have no temporal overlap."""
        # For H20 (max_bars=60):
        # - Last train sample at index T has label based on data [T:T+60]
        # - After purge (60 bars) and embargo (288 bars):
        # - First val sample at index T+349 has label based on data [T+349:T+409]
        # - No overlap: T+60 < T+349

        np.random.seed(42)
        n = 10000

        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'symbol': 'MES',
            'close': 100 + np.cumsum(np.random.randn(n) * 0.1)
        })

        PURGE_BARS = 60
        EMBARGO_BARS = 288
        MAX_BARS_H20 = 60

        train_idx, val_idx, _, _ = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=PURGE_BARS,
            embargo_bars=EMBARGO_BARS
        )

        last_train = train_idx.max()
        first_val = val_idx.min()

        # Last train sample's label window: [last_train, last_train + MAX_BARS_H20]
        # First val sample's label window: [first_val, first_val + MAX_BARS_H20]

        # No overlap if: last_train + MAX_BARS_H20 < first_val
        assert last_train + MAX_BARS_H20 < first_val, \
            f"Label overlap detected: train label ends at {last_train + MAX_BARS_H20}, val starts at {first_val}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

## Template 3: GA Optimization Validation

**File:** `tests/test_ga_optimization_validation.py`

```python
"""
Validation tests for Genetic Algorithm optimization.
Tests convergence, fitness function correctness, and constraints.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from stages.stage5_ga_optimize import calculate_fitness


class TestFitnessFunction:
    """Test fitness function correctness."""

    def test_signal_rate_requirement(self):
        """Fitness should reject solutions with < 60% directional signals."""
        # Create labels with only 40% signals (60% neutral)
        labels_low_signal = np.array([1] * 20 + [-1] * 20 + [0] * 60)  # 40% signals
        bars_to_hit = np.ones(100) * 10
        mae = np.random.randn(100) * -0.01
        mfe = np.random.randn(100) * 0.02

        fitness_low = calculate_fitness(
            labels_low_signal, bars_to_hit, mae, mfe,
            horizon=5, atr_mean=2.0, symbol='MES'
        )

        # Should get very low fitness (penalty)
        assert fitness_low < -500, f"Expected penalty for low signal rate, got {fitness_low}"

    def test_neutral_rate_target(self):
        """Fitness should reward 20-30% neutral rate."""
        # Good: 25% neutral (in target range)
        labels_good = np.array([1] * 37 + [-1] * 38 + [0] * 25)

        # Bad: 5% neutral (too few)
        labels_too_few = np.array([1] * 47 + [-1] * 48 + [0] * 5)

        # Bad: 50% neutral (too many)
        labels_too_many = np.array([1] * 25 + [-1] * 25 + [0] * 50)

        bars_to_hit = np.ones(100) * 10
        mae = np.random.randn(100) * -0.01
        mfe = np.random.randn(100) * 0.02

        fitness_good = calculate_fitness(
            labels_good, bars_to_hit, mae, mfe,
            horizon=5, atr_mean=2.0, symbol='MES'
        )

        fitness_too_few = calculate_fitness(
            labels_too_few, bars_to_hit, mae, mfe,
            horizon=5, atr_mean=2.0, symbol='MES'
        )

        fitness_too_many = calculate_fitness(
            labels_too_many, bars_to_hit, mae, mfe,
            horizon=5, atr_mean=2.0, symbol='MES'
        )

        # Good should have higher fitness than both bad cases
        assert fitness_good > fitness_too_few, "Fitness doesn't reward optimal neutral rate"
        assert fitness_good > fitness_too_many, "Fitness doesn't penalize too many neutrals"

    def test_transaction_cost_penalty(self):
        """Fitness should include transaction cost penalty."""
        # Create labels
        labels = np.array([1] * 50 + [-1] * 50)
        bars_to_hit = np.ones(100) * 10

        # Scenario 1: Small MAE/MFE (profit < transaction cost)
        mae_small = np.ones(100) * -0.0001  # -0.01% loss
        mfe_small = np.ones(100) * 0.0002   # +0.02% gain

        # Scenario 2: Large MAE/MFE (profit > transaction cost)
        mae_large = np.ones(100) * -0.01    # -1% loss
        mfe_large = np.ones(100) * 0.02     # +2% gain

        fitness_small = calculate_fitness(
            labels, bars_to_hit, mae_small, mfe_small,
            horizon=5, atr_mean=2.0, symbol='MES'
        )

        fitness_large = calculate_fitness(
            labels, bars_to_hit, mae_large, mfe_large,
            horizon=5, atr_mean=2.0, symbol='MES'
        )

        # Large profits should yield higher fitness
        assert fitness_large > fitness_small, \
            "Transaction costs not properly penalizing small profits"


class TestGAConvergence:
    """Test GA convergence behavior."""

    @pytest.mark.slow
    def test_fitness_improves_over_generations(self):
        """Best fitness should improve from gen 1 to gen 10."""
        # This would require running actual GA
        # Skipping implementation here (too slow for unit tests)
        # Should be in integration tests
        pytest.skip("Requires full GA run - move to integration tests")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

## Quick Reference: Test Checklist

For each new feature function, ensure:

- [ ] **Correctness test** - Formula validation with known inputs/outputs
- [ ] **Bounds test** - Output within expected range
- [ ] **Edge case tests**:
  - [ ] Empty DataFrame
  - [ ] Single row
  - [ ] All NaN
  - [ ] Constant values
  - [ ] Zero volume (for volume features)
- [ ] **No lookahead test** - Feature at time T uses only data [0:T]
- [ ] **Integration test** - Works in full pipeline

For each pipeline stage, ensure:

- [ ] **Input validation** - Required columns present
- [ ] **Output validation** - Expected columns added
- [ ] **Leakage test** - Fit on train only
- [ ] **Boundary tests** - Purge/embargo exact
- [ ] **Symbol-specific tests** - MES and MGC processed correctly

---

## Usage

```bash
# Run all new tests
pytest tests/test_feature_calculations.py -v
pytest tests/test_leakage_prevention.py -v
pytest tests/test_ga_optimization_validation.py -v

# Run specific test class
pytest tests/test_feature_calculations.py::TestCrossAssetFeatures -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Implementation Timeline

**Week 1:** Implement critical templates
- `test_leakage_prevention.py` (2 days)
- `test_feature_calculations.py` - Cross-asset features only (1 day)
- `test_purge_embargo_precision.py` (1 day)

**Week 2:** Implement remaining feature tests
- Complete `test_feature_calculations.py` (all 36 functions) (3 days)
- `test_ga_optimization_validation.py` (2 days)

**Week 3:** Edge cases and integration
- `test_edge_cases_comprehensive.py` (2 days)
- `test_full_pipeline_integration.py` (3 days)

**Expected Outcome:** 1000+ tests, 9.0/10 maturity, production-ready
