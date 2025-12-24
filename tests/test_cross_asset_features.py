"""
Tests for cross-asset feature calculations.

Tests MES-MGC (S&P 500 - Gold) features:
- Rolling correlation between returns
- Spread z-score calculations
- Beta calculations
- Relative strength (momentum divergence)
- Timestamp alignment
- Missing data handling

Run with: pytest tests/test_cross_asset_features.py -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.phase1.stages.features.cross_asset import add_cross_asset_features


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def aligned_mes_mgc_data():
    """
    Generate aligned MES and MGC data with same timestamps.

    Creates realistic price movements with some correlation.
    """
    np.random.seed(42)
    n = 500

    timestamps = pd.date_range('2024-01-01 09:30', periods=n, freq='5min')

    # MES: S&P 500 futures around 4500
    mes_base = 4500.0
    mes_returns = np.random.randn(n) * 0.001  # 0.1% volatility
    mes_close = mes_base * np.exp(np.cumsum(mes_returns))

    # MGC: Gold futures around 2000, partially correlated with MES
    mgc_base = 2000.0
    # 30% correlation with MES, 70% independent
    mgc_returns = 0.3 * mes_returns + 0.7 * np.random.randn(n) * 0.001
    mgc_close = mgc_base * np.exp(np.cumsum(mgc_returns))

    df_mes = pd.DataFrame({
        'datetime': timestamps,
        'close': mes_close,
        'symbol': 'MES'
    })

    df_mgc = pd.DataFrame({
        'datetime': timestamps,
        'close': mgc_close,
        'symbol': 'MGC'
    })

    return df_mes, df_mgc


@pytest.fixture
def misaligned_timestamps_data():
    """
    Generate MES and MGC data with slightly different timestamps.

    Tests handling of misalignment.
    """
    np.random.seed(42)
    n = 200

    # MES timestamps
    mes_timestamps = pd.date_range('2024-01-01 09:30', periods=n, freq='5min')
    mes_close = 4500.0 + np.cumsum(np.random.randn(n) * 2)

    # MGC timestamps offset by 1 minute
    mgc_timestamps = pd.date_range('2024-01-01 09:31', periods=n, freq='5min')
    mgc_close = 2000.0 + np.cumsum(np.random.randn(n) * 1)

    df_mes = pd.DataFrame({
        'datetime': mes_timestamps,
        'close': mes_close,
        'symbol': 'MES'
    })

    df_mgc = pd.DataFrame({
        'datetime': mgc_timestamps,
        'close': mgc_close,
        'symbol': 'MGC'
    })

    return df_mes, df_mgc


@pytest.fixture
def single_asset_data():
    """Generate data for single asset (no cross-asset features possible)."""
    np.random.seed(42)
    n = 200

    timestamps = pd.date_range('2024-01-01 09:30', periods=n, freq='5min')
    close = 4500.0 + np.cumsum(np.random.randn(n) * 2)

    df = pd.DataFrame({
        'datetime': timestamps,
        'close': close,
        'symbol': 'MES'
    })

    return df


# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================

class TestCrossAssetBasics:
    """Tests for basic cross-asset feature functionality."""

    def test_adds_expected_columns_with_both_assets(self, aligned_mes_mgc_data):
        """Test that all cross-asset columns are added when both assets present."""
        df_mes, df_mgc = aligned_mes_mgc_data

        # Prepare test data
        df = df_mes.copy()
        metadata = {}

        result = add_cross_asset_features(
            df,
            metadata,
            mes_close=df_mes['close'].values,
            mgc_close=df_mgc['close'].values,
            current_symbol='MES'
        )

        expected_cols = [
            'mes_mgc_correlation_20',
            'mes_mgc_spread_zscore',
            'mes_mgc_beta',
            'relative_strength'
        ]

        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"
            assert col in metadata, f"Missing metadata for: {col}"

    def test_sets_nan_with_single_asset(self, single_asset_data):
        """Test that cross-asset features are NaN when only one asset present."""
        df = single_asset_data.copy()
        metadata = {}

        result = add_cross_asset_features(
            df,
            metadata,
            mes_close=None,
            mgc_close=None,
            current_symbol='MES'
        )

        # Should have columns but all NaN
        expected_cols = [
            'mes_mgc_correlation_20',
            'mes_mgc_spread_zscore',
            'mes_mgc_beta',
            'relative_strength'
        ]

        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"
            assert result[col].isna().all(), f"{col} should be all NaN when only one asset"

    def test_handles_mismatched_lengths(self):
        """Test that function handles mismatched array lengths gracefully."""
        n = 200
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'close': np.random.randn(n) * 10 + 100,
        })

        mes_close = np.random.randn(n) * 10 + 4500
        mgc_close = np.random.randn(n - 10) * 5 + 2000  # Different length

        metadata = {}

        # Should set to NaN due to length mismatch
        result = add_cross_asset_features(
            df,
            metadata,
            mes_close=mes_close,
            mgc_close=mgc_close,
            current_symbol='MES'
        )

        assert result['mes_mgc_correlation_20'].isna().all(), \
            "Should be NaN when array lengths don't match"


# =============================================================================
# CORRELATION TESTS
# =============================================================================

class TestCorrelationCalculation:
    """Tests for MES-MGC correlation calculation."""

    def test_correlation_bounded_minus1_to_1(self, aligned_mes_mgc_data):
        """Test that correlation values are bounded [-1, 1]."""
        df_mes, df_mgc = aligned_mes_mgc_data

        df = df_mes.copy()
        metadata = {}

        result = add_cross_asset_features(
            df,
            metadata,
            mes_close=df_mes['close'].values,
            mgc_close=df_mgc['close'].values,
            current_symbol='MES'
        )

        corr = result['mes_mgc_correlation_20'].dropna()

        assert (corr >= -1.0).all() and (corr <= 1.0).all(), \
            "Correlation should be in [-1, 1] range"

    def test_correlation_perfect_positive(self):
        """Test correlation with perfectly correlated assets."""
        n = 200
        timestamps = pd.date_range('2024-01-01', periods=n, freq='5min')

        # Create perfectly correlated prices (same returns)
        returns = np.random.randn(n) * 0.01
        mes_close = 4500 * np.exp(np.cumsum(returns))
        mgc_close = 2000 * np.exp(np.cumsum(returns))  # Same returns, different scale

        df = pd.DataFrame({
            'datetime': timestamps,
            'close': mes_close,
        })

        metadata = {}

        result = add_cross_asset_features(
            df,
            metadata,
            mes_close=mes_close,
            mgc_close=mgc_close,
            current_symbol='MES'
        )

        # Correlation should be very close to 1.0
        corr = result['mes_mgc_correlation_20'].dropna()
        assert (corr > 0.95).mean() > 0.9, \
            "Correlation should be near 1.0 for perfectly correlated assets"

    def test_correlation_perfect_negative(self):
        """Test correlation with perfectly negatively correlated assets."""
        n = 200
        timestamps = pd.date_range('2024-01-01', periods=n, freq='5min')

        returns = np.random.randn(n) * 0.01
        mes_close = 4500 * np.exp(np.cumsum(returns))
        mgc_close = 2000 * np.exp(np.cumsum(-returns))  # Opposite returns

        df = pd.DataFrame({
            'datetime': timestamps,
            'close': mes_close,
        })

        metadata = {}

        result = add_cross_asset_features(
            df,
            metadata,
            mes_close=mes_close,
            mgc_close=mgc_close,
            current_symbol='MES'
        )

        # Correlation should be very close to -1.0
        corr = result['mes_mgc_correlation_20'].dropna()
        assert (corr < -0.95).mean() > 0.9, \
            "Correlation should be near -1.0 for perfectly negatively correlated assets"

    def test_correlation_uncorrelated(self):
        """Test correlation with uncorrelated assets."""
        np.random.seed(42)
        n = 500

        timestamps = pd.date_range('2024-01-01', periods=n, freq='5min')

        # Independent random walks
        mes_close = 4500 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
        mgc_close = 2000 * np.exp(np.cumsum(np.random.randn(n) * 0.01))

        df = pd.DataFrame({
            'datetime': timestamps,
            'close': mes_close,
        })

        metadata = {}

        result = add_cross_asset_features(
            df,
            metadata,
            mes_close=mes_close,
            mgc_close=mgc_close,
            current_symbol='MES'
        )

        # Correlation should be near 0 on average
        corr = result['mes_mgc_correlation_20'].dropna()
        mean_corr = corr.mean()

        assert abs(mean_corr) < 0.3, \
            f"Mean correlation should be near 0 for uncorrelated assets, got {mean_corr}"


# =============================================================================
# BETA TESTS
# =============================================================================

class TestBetaCalculation:
    """Tests for MES-MGC beta calculation."""

    def test_beta_calculation_exists(self, aligned_mes_mgc_data):
        """Test that beta is calculated and has reasonable values."""
        df_mes, df_mgc = aligned_mes_mgc_data

        df = df_mes.copy()
        metadata = {}

        result = add_cross_asset_features(
            df,
            metadata,
            mes_close=df_mes['close'].values,
            mgc_close=df_mgc['close'].values,
            current_symbol='MES'
        )

        beta = result['mes_mgc_beta'].dropna()

        # Beta can be any real number, but should exist
        assert len(beta) > 0, "Beta should have values"

        # For typical financial assets, beta usually in range [-5, 5]
        assert (beta >= -10).all() and (beta <= 10).all(), \
            "Beta should be in reasonable range for financial assets"

    def test_beta_with_zero_variance(self):
        """Test beta calculation when one asset has zero variance."""
        n = 200
        timestamps = pd.date_range('2024-01-01', periods=n, freq='5min')

        mes_close = np.full(n, 4500.0)  # Constant price (zero variance)
        mgc_close = 2000.0 + np.cumsum(np.random.randn(n))

        df = pd.DataFrame({
            'datetime': timestamps,
            'close': mes_close,
        })

        metadata = {}

        result = add_cross_asset_features(
            df,
            metadata,
            mes_close=mes_close,
            mgc_close=mgc_close,
            current_symbol='MES'
        )

        # Beta should be 0 or NaN when MES has zero variance
        beta = result['mes_mgc_beta'].dropna()
        if len(beta) > 0:
            assert (beta == 0).all() or beta.isna().all(), \
                "Beta should be 0 or NaN when asset has zero variance"


# =============================================================================
# SPREAD Z-SCORE TESTS
# =============================================================================

class TestSpreadZScore:
    """Tests for MES-MGC spread z-score calculation."""

    def test_spread_zscore_exists(self, aligned_mes_mgc_data):
        """Test that spread z-score is calculated."""
        df_mes, df_mgc = aligned_mes_mgc_data

        df = df_mes.copy()
        metadata = {}

        result = add_cross_asset_features(
            df,
            metadata,
            mes_close=df_mes['close'].values,
            mgc_close=df_mgc['close'].values,
            current_symbol='MES'
        )

        spread_zscore = result['mes_mgc_spread_zscore'].dropna()
        assert len(spread_zscore) > 0, "Spread z-score should have values"

    def test_spread_zscore_distribution(self, aligned_mes_mgc_data):
        """Test that spread z-score has approximately normal distribution."""
        df_mes, df_mgc = aligned_mes_mgc_data

        df = df_mes.copy()
        metadata = {}

        result = add_cross_asset_features(
            df,
            metadata,
            mes_close=df_mes['close'].values,
            mgc_close=df_mgc['close'].values,
            current_symbol='MES'
        )

        spread_zscore = result['mes_mgc_spread_zscore'].dropna()

        # Z-score should have mean near 0, std near 1
        mean_zscore = spread_zscore.mean()
        std_zscore = spread_zscore.std()

        assert abs(mean_zscore) < 0.5, \
            f"Spread z-score mean should be near 0, got {mean_zscore}"
        assert 0.5 < std_zscore < 2.0, \
            f"Spread z-score std should be near 1, got {std_zscore}"


# =============================================================================
# RELATIVE STRENGTH TESTS
# =============================================================================

class TestRelativeStrength:
    """Tests for relative strength (momentum divergence) calculation."""

    def test_relative_strength_exists(self, aligned_mes_mgc_data):
        """Test that relative strength is calculated."""
        df_mes, df_mgc = aligned_mes_mgc_data

        df = df_mes.copy()
        metadata = {}

        result = add_cross_asset_features(
            df,
            metadata,
            mes_close=df_mes['close'].values,
            mgc_close=df_mgc['close'].values,
            current_symbol='MES'
        )

        rel_strength = result['relative_strength'].dropna()
        assert len(rel_strength) > 0, "Relative strength should have values"

    def test_relative_strength_positive_when_mes_stronger(self):
        """Test that relative strength is positive when MES outperforms MGC."""
        n = 200
        timestamps = pd.date_range('2024-01-01', periods=n, freq='5min')

        # MES strong uptrend, MGC flat
        mes_returns = np.full(n, 0.001)  # Consistent positive returns
        mgc_returns = np.zeros(n)  # Flat

        mes_close = 4500 * np.exp(np.cumsum(mes_returns))
        mgc_close = 2000 * np.exp(np.cumsum(mgc_returns))

        df = pd.DataFrame({
            'datetime': timestamps,
            'close': mes_close,
        })

        metadata = {}

        result = add_cross_asset_features(
            df,
            metadata,
            mes_close=mes_close,
            mgc_close=mgc_close,
            current_symbol='MES'
        )

        rel_strength = result['relative_strength'].dropna()

        # Should be mostly positive when MES outperforms
        assert (rel_strength > 0).mean() > 0.9, \
            "Relative strength should be positive when MES outperforms MGC"


# =============================================================================
# TIMESTAMP ALIGNMENT TESTS
# =============================================================================

class TestTimestampAlignment:
    """Tests for timestamp alignment between MES and MGC."""

    def test_features_computed_with_aligned_timestamps(self, aligned_mes_mgc_data):
        """Test that features are correctly computed when timestamps align."""
        df_mes, df_mgc = aligned_mes_mgc_data

        # Verify timestamps align
        assert (df_mes['datetime'] == df_mgc['datetime']).all(), \
            "Test fixture should have aligned timestamps"

        df = df_mes.copy()
        metadata = {}

        result = add_cross_asset_features(
            df,
            metadata,
            mes_close=df_mes['close'].values,
            mgc_close=df_mgc['close'].values,
            current_symbol='MES'
        )

        # All features should have valid (non-NaN) values after warmup period
        warmup = 20  # Correlation window
        valid_data = result.iloc[warmup:]

        for col in ['mes_mgc_correlation_20', 'mes_mgc_beta', 'relative_strength']:
            assert not valid_data[col].isna().all(), \
                f"{col} should have valid values after warmup"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_small_dataset(self):
        """Test cross-asset features with small dataset."""
        n = 50  # Smaller than correlation window (20)
        timestamps = pd.date_range('2024-01-01', periods=n, freq='5min')

        mes_close = 4500.0 + np.cumsum(np.random.randn(n))
        mgc_close = 2000.0 + np.cumsum(np.random.randn(n))

        df = pd.DataFrame({
            'datetime': timestamps,
            'close': mes_close,
        })

        metadata = {}

        # Should not crash
        result = add_cross_asset_features(
            df,
            metadata,
            mes_close=mes_close,
            mgc_close=mgc_close,
            current_symbol='MES'
        )

        # Features exist but may be mostly NaN
        assert 'mes_mgc_correlation_20' in result.columns

    def test_handles_constant_prices(self):
        """Test cross-asset features when prices are constant."""
        n = 200
        timestamps = pd.date_range('2024-01-01', periods=n, freq='5min')

        mes_close = np.full(n, 4500.0)
        mgc_close = np.full(n, 2000.0)

        df = pd.DataFrame({
            'datetime': timestamps,
            'close': mes_close,
        })

        metadata = {}

        # Should not crash
        result = add_cross_asset_features(
            df,
            metadata,
            mes_close=mes_close,
            mgc_close=mgc_close,
            current_symbol='MES'
        )

        # Correlation should be NaN or 0 for constant prices
        corr = result['mes_mgc_correlation_20'].dropna()
        if len(corr) > 0:
            assert corr.isna().all() or (corr == 0).all(), \
                "Correlation should be NaN or 0 for constant prices"

    def test_handles_extreme_values(self):
        """Test cross-asset features with extreme price values."""
        n = 200
        timestamps = pd.date_range('2024-01-01', periods=n, freq='5min')

        # Very large prices
        mes_close = 1e6 + np.cumsum(np.random.randn(n) * 1000)
        mgc_close = 1e5 + np.cumsum(np.random.randn(n) * 100)

        df = pd.DataFrame({
            'datetime': timestamps,
            'close': mes_close,
        })

        metadata = {}

        # Should handle large numbers gracefully
        result = add_cross_asset_features(
            df,
            metadata,
            mes_close=mes_close,
            mgc_close=mgc_close,
            current_symbol='MES'
        )

        # Features should exist and be finite
        assert 'mes_mgc_correlation_20' in result.columns
        corr = result['mes_mgc_correlation_20'].dropna()
        if len(corr) > 0:
            assert np.isfinite(corr).all(), "Correlation should be finite"

    def test_handles_missing_values_in_prices(self):
        """Test cross-asset features when price arrays contain NaN."""
        n = 200
        timestamps = pd.date_range('2024-01-01', periods=n, freq='5min')

        mes_close = 4500.0 + np.cumsum(np.random.randn(n))
        mgc_close = 2000.0 + np.cumsum(np.random.randn(n))

        # Insert some NaN values
        mes_close[50:55] = np.nan
        mgc_close[100:105] = np.nan

        df = pd.DataFrame({
            'datetime': timestamps,
            'close': mes_close,
        })

        metadata = {}

        # Should handle NaN gracefully
        result = add_cross_asset_features(
            df,
            metadata,
            mes_close=mes_close,
            mgc_close=mgc_close,
            current_symbol='MES'
        )

        # Should have features but some may be NaN
        assert 'mes_mgc_correlation_20' in result.columns
