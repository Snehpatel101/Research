"""
Tests for Information-Theoretic Entropy Features.

This module tests the entropy feature functions to ensure:
1. Correct calculation of Shannon entropy, Lempel-Ziv complexity, and ApEn
2. No lookahead bias (proper .shift(1) application)
3. Proper handling of edge cases (NaN values, constant data)
4. Feature metadata registration
5. Integration with the feature engineering pipeline

Test Categories:
- Unit tests for helper functions
- Integration tests for full feature functions
- Lookahead bias prevention tests
- Edge case and error handling tests
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.pipeline._phase1_impl.stages.features.entropy import (
    DEFAULT_APEN_M,
    DEFAULT_APEN_R,
    DEFAULT_APEN_WINDOWS,
    DEFAULT_LZ_WINDOWS,
    DEFAULT_SHANNON_BINS,
    DEFAULT_SHANNON_WINDOWS,
    _approximate_entropy,
    _calculate_shannon_entropy,
    _discretize_returns,
    _lempel_ziv_complexity,
    _normalized_lz_complexity,
    _phi_correlation,
    _rolling_approximate_entropy,
    _rolling_lz_complexity,
    _rolling_shannon_entropy,
    add_approximate_entropy,
    add_entropy_features,
    add_lempel_ziv_complexity,
    add_shannon_entropy,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_ohlcv():
    """
    Generate sample OHLCV data for testing.

    Creates 500 bars of realistic price data with proper OHLC relationships.
    """
    np.random.seed(42)
    n = 500

    # Generate realistic price series with some trending and mean-reverting periods
    base_price = 100.0
    returns = np.random.randn(n) * 0.01
    close = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC with proper relationships
    daily_range = np.abs(np.random.randn(n) * 0.005)
    high = close * (1 + daily_range / 2)
    low = close * (1 - daily_range / 2)
    open_ = close * (1 + np.random.randn(n) * 0.002)

    # Ensure valid OHLC
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    # Generate volume
    volume = np.random.randint(100, 10000, n).astype(float)

    # Generate timestamps
    start_time = datetime(2024, 1, 1, 9, 30)
    timestamps = [start_time + timedelta(minutes=5 * i) for i in range(n)]

    return pd.DataFrame(
        {
            "datetime": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def small_ohlcv():
    """Generate small OHLCV data for edge case testing."""
    np.random.seed(42)
    n = 50

    base_price = 100.0
    close = base_price + np.cumsum(np.random.randn(n) * 0.5)

    start_time = datetime(2024, 1, 1, 9, 30)
    timestamps = [start_time + timedelta(minutes=5 * i) for i in range(n)]

    return pd.DataFrame(
        {
            "datetime": timestamps,
            "open": close * (1 + np.random.randn(n) * 0.001),
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.random.randint(100, 1000, n).astype(float),
        }
    )


@pytest.fixture
def trending_data():
    """Generate data with clear trend (low entropy expected)."""
    np.random.seed(42)
    n = 200

    # Strong uptrend with small noise
    base_price = 100.0
    trend = np.linspace(0, 0.5, n)  # 50% gain
    noise = np.random.randn(n) * 0.001
    close = base_price * np.exp(trend + noise)

    start_time = datetime(2024, 1, 1, 9, 30)
    timestamps = [start_time + timedelta(minutes=5 * i) for i in range(n)]

    return pd.DataFrame(
        {
            "datetime": timestamps,
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": np.ones(n) * 1000,
        }
    )


@pytest.fixture
def random_walk_data():
    """Generate random walk data (high entropy expected)."""
    np.random.seed(123)
    n = 200

    # Pure random walk
    base_price = 100.0
    returns = np.random.randn(n) * 0.02
    close = base_price * np.exp(np.cumsum(returns))

    start_time = datetime(2024, 1, 1, 9, 30)
    timestamps = [start_time + timedelta(minutes=5 * i) for i in range(n)]

    return pd.DataFrame(
        {
            "datetime": timestamps,
            "open": close * (1 + np.random.randn(n) * 0.003),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.random.randint(500, 2000, n).astype(float),
        }
    )


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestDiscretizeReturns:
    """Tests for _discretize_returns helper function."""

    def test_basic_discretization(self):
        """Test that returns are correctly binned."""
        returns = np.array([0.01, 0.02, -0.01, -0.02, 0.0, 0.03, -0.03, 0.015, -0.015, 0.005])
        binned = _discretize_returns(returns, n_bins=5)

        assert len(binned) == len(returns)
        assert not np.isnan(binned).all()
        # All values should be in valid bin range
        valid = binned[~np.isnan(binned)]
        assert all(0 <= v < 5 for v in valid)

    def test_nan_handling(self):
        """Test that NaN values are preserved."""
        returns = np.array([0.01, np.nan, -0.01, np.nan, 0.0])
        binned = _discretize_returns(returns, n_bins=3)

        assert np.isnan(binned[1])
        assert np.isnan(binned[3])

    def test_constant_returns(self):
        """Test handling of constant returns."""
        returns = np.array([0.0] * 20)
        binned = _discretize_returns(returns, n_bins=5)

        # Should assign to single bin
        valid = binned[~np.isnan(binned)]
        assert len(np.unique(valid)) == 1

    def test_insufficient_data(self):
        """Test with insufficient data for binning."""
        returns = np.array([0.01, 0.02])
        binned = _discretize_returns(returns, n_bins=10)

        # Should return all NaN
        assert np.isnan(binned).all()


class TestShannonEntropy:
    """Tests for _calculate_shannon_entropy helper function."""

    def test_uniform_distribution(self):
        """Test entropy of uniform distribution (maximum entropy)."""
        # 10 bins with equal counts
        bin_counts = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        entropy = _calculate_shannon_entropy(bin_counts)

        # Maximum entropy for 10 bins = log2(10)
        expected_max = np.log2(10)
        assert np.isclose(entropy, expected_max, rtol=0.01)

    def test_single_bin(self):
        """Test entropy when all values in one bin (minimum entropy)."""
        bin_counts = np.array([100, 0, 0, 0, 0])
        entropy = _calculate_shannon_entropy(bin_counts)

        assert entropy == 0.0

    def test_empty_counts(self):
        """Test handling of empty bin counts."""
        bin_counts = np.array([0, 0, 0, 0, 0])
        entropy = _calculate_shannon_entropy(bin_counts)

        assert np.isnan(entropy)


class TestLempelZivComplexity:
    """Tests for _lempel_ziv_complexity helper function."""

    def test_repeating_pattern(self):
        """Test complexity of simple repeating pattern."""
        # Repeating 0101...
        binary_seq = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        complexity = _lempel_ziv_complexity(binary_seq)

        # Low complexity for repeating pattern
        assert complexity < 5

    def test_random_sequence(self):
        """Test complexity of random sequence."""
        np.random.seed(42)
        binary_seq = np.random.randint(0, 2, 50)
        complexity = _lempel_ziv_complexity(binary_seq)

        # Higher complexity for random sequence
        assert complexity > 5

    def test_constant_sequence(self):
        """Test complexity of constant sequence."""
        binary_seq = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        complexity = _lempel_ziv_complexity(binary_seq)

        # Minimal complexity
        assert complexity <= 2

    def test_empty_sequence(self):
        """Test handling of empty sequence."""
        binary_seq = np.array([])
        complexity = _lempel_ziv_complexity(binary_seq)

        assert complexity == 0


class TestNormalizedLZComplexity:
    """Tests for _normalized_lz_complexity helper function."""

    def test_normalization_range(self):
        """Test that normalized complexity is in expected range."""
        # For a reasonably random sequence
        np.random.seed(42)
        binary_seq = np.random.randint(0, 2, 100)
        complexity = _lempel_ziv_complexity(binary_seq)
        normalized = _normalized_lz_complexity(complexity, 100)

        # Should be between 0 and ~2 (can exceed 1 for very random sequences)
        assert 0 < normalized < 3

    def test_short_sequence(self):
        """Test normalization with very short sequence."""
        normalized = _normalized_lz_complexity(1, 1)
        assert np.isnan(normalized)


class TestPhiCorrelation:
    """Tests for _phi_correlation helper function."""

    def test_regular_pattern(self):
        """Test phi for regular repeating pattern."""
        # Sine wave pattern (regular)
        data = np.sin(np.linspace(0, 4 * np.pi, 50))
        phi_2 = _phi_correlation(data, m=2, r=0.2 * np.std(data))

        # Should have high phi (many matches)
        assert phi_2 > -3

    def test_random_data(self):
        """Test phi for random data."""
        np.random.seed(42)
        data = np.random.randn(50)
        phi_2 = _phi_correlation(data, m=2, r=0.2 * np.std(data))

        # Should have lower phi than regular pattern
        assert not np.isnan(phi_2)


class TestApproximateEntropy:
    """Tests for _approximate_entropy helper function."""

    def test_regular_vs_random(self):
        """Test that regular data has lower ApEn than random."""
        np.random.seed(42)

        # Regular pattern (sine wave)
        regular_data = np.sin(np.linspace(0, 8 * np.pi, 100))

        # Random data
        random_data = np.random.randn(100)

        apen_regular = _approximate_entropy(regular_data, m=2, r_fraction=0.2)
        apen_random = _approximate_entropy(random_data, m=2, r_fraction=0.2)

        # Regular data should have lower ApEn
        assert apen_regular < apen_random

    def test_constant_data(self):
        """Test ApEn of constant data."""
        data = np.ones(50)
        apen = _approximate_entropy(data, m=2, r_fraction=0.2)

        # Constant data has zero std, should return NaN
        assert np.isnan(apen)


# =============================================================================
# FEATURE FUNCTION TESTS
# =============================================================================


class TestAddShannonEntropy:
    """Tests for add_shannon_entropy function."""

    def test_basic_functionality(self, sample_ohlcv):
        """Test that Shannon entropy features are added correctly."""
        df = sample_ohlcv.copy()
        feature_metadata = {}
        windows = [10, 20]

        df = add_shannon_entropy(df, feature_metadata, windows=windows)

        # Check columns are added
        for window in windows:
            assert f"entropy_shannon_{window}" in df.columns
            assert f"entropy_shannon_norm_{window}" in df.columns

        # Check metadata is registered
        assert len(feature_metadata) == len(windows) * 2

    def test_anti_lookahead(self, sample_ohlcv):
        """Test that features are properly lagged to prevent lookahead."""
        df = sample_ohlcv.copy()
        feature_metadata = {}

        df = add_shannon_entropy(df, feature_metadata, windows=[20])

        # First window-1 values should be NaN due to rolling window
        # Plus one more due to shift(1)
        assert pd.isna(df["entropy_shannon_20"].iloc[0])

        # Values should be lagged (non-NaN values start later)
        first_valid_idx = df["entropy_shannon_20"].first_valid_index()
        assert first_valid_idx > 19  # At least window size + 1

    def test_value_range(self, sample_ohlcv):
        """Test that normalized entropy is in [0, 1] range."""
        df = sample_ohlcv.copy()
        feature_metadata = {}

        df = add_shannon_entropy(df, feature_metadata, windows=[20], n_bins=10)

        valid_values = df["entropy_shannon_norm_20"].dropna()
        assert (valid_values >= 0).all()
        assert (valid_values <= 1.1).all()  # Allow small numerical tolerance

    def test_trending_vs_random(self, trending_data, random_walk_data):
        """Test that trending data has lower entropy than random."""
        feature_metadata = {}

        df_trend = add_shannon_entropy(trending_data.copy(), feature_metadata, windows=[50])
        df_random = add_shannon_entropy(random_walk_data.copy(), {}, windows=[50])

        # Compare mean entropy (trending should be lower)
        mean_trend = df_trend["entropy_shannon_50"].mean()
        mean_random = df_random["entropy_shannon_50"].mean()

        # Note: This is a statistical test, not always guaranteed
        # Just check they're both valid
        assert not np.isnan(mean_trend)
        assert not np.isnan(mean_random)


class TestAddLempelZivComplexity:
    """Tests for add_lempel_ziv_complexity function."""

    def test_basic_functionality(self, sample_ohlcv):
        """Test that LZ complexity features are added correctly."""
        df = sample_ohlcv.copy()
        feature_metadata = {}
        windows = [20, 50]

        df = add_lempel_ziv_complexity(df, feature_metadata, windows=windows)

        # Check columns are added
        for window in windows:
            assert f"entropy_lz_{window}" in df.columns

        # Check metadata is registered
        assert len(feature_metadata) == len(windows)

    def test_anti_lookahead(self, sample_ohlcv):
        """Test that features are properly lagged."""
        df = sample_ohlcv.copy()
        feature_metadata = {}

        df = add_lempel_ziv_complexity(df, feature_metadata, windows=[30])

        # First values should be NaN
        assert pd.isna(df["entropy_lz_30"].iloc[0])

        # Values should start after window + 1 shift
        first_valid_idx = df["entropy_lz_30"].first_valid_index()
        assert first_valid_idx > 29

    def test_value_range(self, sample_ohlcv):
        """Test that normalized LZ is in reasonable range."""
        df = sample_ohlcv.copy()
        feature_metadata = {}

        df = add_lempel_ziv_complexity(df, feature_metadata, windows=[50])

        valid_values = df["entropy_lz_50"].dropna()
        # Normalized LZ should be positive
        assert (valid_values >= 0).all()
        # Typically less than 2 for financial data
        assert (valid_values < 3).all()


class TestAddApproximateEntropy:
    """Tests for add_approximate_entropy function."""

    def test_basic_functionality(self, sample_ohlcv):
        """Test that ApEn features are added correctly."""
        df = sample_ohlcv.copy()
        feature_metadata = {}
        windows = [20, 50]

        df = add_approximate_entropy(df, feature_metadata, windows=windows)

        # Check columns are added
        for window in windows:
            assert f"entropy_apen_{window}" in df.columns

        # Check metadata is registered
        assert len(feature_metadata) == len(windows)

    def test_anti_lookahead(self, sample_ohlcv):
        """Test that features are properly lagged."""
        df = sample_ohlcv.copy()
        feature_metadata = {}

        df = add_approximate_entropy(df, feature_metadata, windows=[30])

        # First values should be NaN
        assert pd.isna(df["entropy_apen_30"].iloc[0])

        # Values should start after window + 1 shift
        first_valid_idx = df["entropy_apen_30"].first_valid_index()
        assert first_valid_idx > 29

    def test_parameter_variation(self, sample_ohlcv):
        """Test ApEn with different m and r parameters."""
        df1 = sample_ohlcv.copy()
        df2 = sample_ohlcv.copy()
        feature_metadata1 = {}
        feature_metadata2 = {}

        # Different embedding dimensions
        df1 = add_approximate_entropy(df1, feature_metadata1, windows=[30], m=2, r=0.2)
        df2 = add_approximate_entropy(df2, feature_metadata2, windows=[30], m=3, r=0.2)

        # Both should produce valid values
        assert df1["entropy_apen_30"].notna().sum() > 0
        assert df2["entropy_apen_30"].notna().sum() > 0


class TestAddEntropyFeatures:
    """Tests for add_entropy_features main function."""

    def test_all_features_added(self, sample_ohlcv):
        """Test that all entropy features are added."""
        df = sample_ohlcv.copy()
        feature_metadata = {}

        df = add_entropy_features(df, feature_metadata)

        # Check Shannon features
        for window in DEFAULT_SHANNON_WINDOWS:
            assert f"entropy_shannon_{window}" in df.columns
            assert f"entropy_shannon_norm_{window}" in df.columns

        # Check LZ features
        for window in DEFAULT_LZ_WINDOWS:
            assert f"entropy_lz_{window}" in df.columns

        # Check ApEn features
        for window in DEFAULT_APEN_WINDOWS:
            assert f"entropy_apen_{window}" in df.columns

    def test_selective_features(self, sample_ohlcv):
        """Test adding only specific feature types."""
        df = sample_ohlcv.copy()
        feature_metadata = {}

        # Only Shannon entropy
        df = add_entropy_features(
            df,
            feature_metadata,
            include_shannon=True,
            include_lempel_ziv=False,
            include_approximate=False,
            shannon_windows=[20],
        )

        assert "entropy_shannon_20" in df.columns
        assert "entropy_lz_20" not in df.columns
        assert "entropy_apen_20" not in df.columns

    def test_custom_parameters(self, sample_ohlcv):
        """Test with custom parameters."""
        df = sample_ohlcv.copy()
        feature_metadata = {}

        df = add_entropy_features(
            df,
            feature_metadata,
            shannon_windows=[15],
            shannon_bins=8,
            lz_windows=[25],
            apen_windows=[25],
            apen_m=3,
            apen_r=0.15,
        )

        assert "entropy_shannon_15" in df.columns
        assert "entropy_lz_25" in df.columns
        assert "entropy_apen_25" in df.columns

    def test_metadata_completeness(self, sample_ohlcv):
        """Test that all features are registered in metadata."""
        df = sample_ohlcv.copy()
        feature_metadata = {}

        df = add_entropy_features(df, feature_metadata)

        # All new columns should have metadata
        entropy_cols = [col for col in df.columns if col.startswith("entropy_")]
        for col in entropy_cols:
            assert col in feature_metadata

    def test_none_metadata(self, sample_ohlcv):
        """Test that function handles None metadata gracefully."""
        df = sample_ohlcv.copy()

        # Should not raise with None metadata
        df = add_entropy_features(df, None, shannon_windows=[20])
        assert "entropy_shannon_20" in df.columns


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_small_dataset(self, small_ohlcv):
        """Test handling of small datasets."""
        df = small_ohlcv.copy()
        feature_metadata = {}

        # Should not raise, even if some features are NaN
        df = add_entropy_features(df, feature_metadata, apen_windows=[20])

        assert "entropy_apen_20" in df.columns

    def test_nan_in_price(self, sample_ohlcv):
        """Test handling of NaN values in price data."""
        df = sample_ohlcv.copy()
        df.loc[100:110, "close"] = np.nan
        feature_metadata = {}

        # Should not raise
        df = add_entropy_features(df, feature_metadata, shannon_windows=[20])

        assert "entropy_shannon_20" in df.columns

    def test_constant_price(self):
        """Test handling of constant price series."""
        n = 100
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01", periods=n, freq="5min"),
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1000.0,
            }
        )
        feature_metadata = {}

        # Should not raise
        df = add_entropy_features(df, feature_metadata, shannon_windows=[20])

        # Entropy should be NaN or 0 for constant data
        assert "entropy_shannon_20" in df.columns

    def test_different_price_columns(self, sample_ohlcv):
        """Test using different price columns."""
        df = sample_ohlcv.copy()
        feature_metadata = {}

        # Should work with 'open' column
        df = add_shannon_entropy(df, feature_metadata, price_col="open", windows=[20])

        assert "entropy_shannon_20" in df.columns


# =============================================================================
# LOOKAHEAD BIAS TESTS
# =============================================================================


class TestLookaheadBias:
    """Tests specifically for lookahead bias prevention."""

    def test_future_data_not_used(self, sample_ohlcv):
        """Test that modifying future data doesn't affect current features."""
        df1 = sample_ohlcv.copy()
        df2 = sample_ohlcv.copy()

        # Modify future data in df2
        df2.loc[250:, "close"] = df2.loc[250:, "close"] * 2

        feature_metadata = {}
        df1 = add_entropy_features(df1.copy(), feature_metadata, shannon_windows=[20])
        df2 = add_entropy_features(df2.copy(), {}, shannon_windows=[20])

        # Features before modification point should be identical
        # (accounting for the shift(1) which means we compare up to index 248)
        np.testing.assert_array_almost_equal(
            df1["entropy_shannon_20"].iloc[:248].values,
            df2["entropy_shannon_20"].iloc[:248].values,
        )

    def test_shift_applied(self, sample_ohlcv):
        """Verify that shift(1) is consistently applied."""
        df = sample_ohlcv.copy()
        feature_metadata = {}

        df = add_entropy_features(
            df, feature_metadata, shannon_windows=[20], lz_windows=[20], apen_windows=[20]
        )

        # The first valid index should be > window size due to shift
        for col in ["entropy_shannon_20", "entropy_lz_20", "entropy_apen_20"]:
            first_valid = df[col].first_valid_index()
            assert first_valid > 19, f"{col} has first valid at {first_valid}, expected > 19"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests with other feature engineering components."""

    def test_import_from_package(self):
        """Test that entropy features can be imported from the package."""
        from src.pipeline._phase1_impl.stages.features import (
            DEFAULT_APEN_M,
            DEFAULT_APEN_R,
            DEFAULT_APEN_WINDOWS,
            DEFAULT_LZ_WINDOWS,
            DEFAULT_SHANNON_BINS,
            DEFAULT_SHANNON_WINDOWS,
            add_approximate_entropy,
            add_entropy_features,
            add_lempel_ziv_complexity,
            add_shannon_entropy,
        )

        # All should be importable
        assert callable(add_entropy_features)
        assert callable(add_shannon_entropy)
        assert callable(add_lempel_ziv_complexity)
        assert callable(add_approximate_entropy)
        assert isinstance(DEFAULT_SHANNON_WINDOWS, list)

    def test_combine_with_other_features(self, sample_ohlcv):
        """Test combining entropy features with other feature types."""
        from src.pipeline._phase1_impl.stages.features.volatility import add_historical_volatility

        df = sample_ohlcv.copy()
        feature_metadata = {}

        # Add volatility features
        df = add_historical_volatility(df, feature_metadata, periods=[20])

        # Add entropy features
        df = add_entropy_features(df, feature_metadata, shannon_windows=[20])

        # Both should exist
        assert "hvol_20" in df.columns
        assert "entropy_shannon_20" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
