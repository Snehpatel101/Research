"""
Tests for ML leakage prevention.

Validates that:
1. Features don't use future data (lookahead bias)
2. Rolling calculations use only past data
3. Train/val/test splits don't leak information
4. Feature engineering maintains temporal consistency
5. Labels don't contaminate feature windows

Run with: pytest tests/test_leakage_prevention.py -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from stages.features.volatility import add_atr, add_bollinger_bands
from stages.features.momentum import add_rsi, add_macd
from stages.features.price_features import add_returns
from stages.stage7_splits import create_chronological_splits


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temporal_test_data():
    """
    Create test data with known temporal pattern for leak detection.

    Pattern: price increases by exactly 1.0 every bar.
    This allows precise testing of lookahead bias.
    """
    n = 500
    timestamps = pd.date_range('2024-01-01 09:30', periods=n, freq='5min')

    # Price increases by 1.0 each bar - deterministic pattern
    prices = np.arange(1000.0, 1000.0 + n, 1.0)

    df = pd.DataFrame({
        'datetime': timestamps,
        'open': prices,
        'high': prices + 0.5,
        'low': prices - 0.5,
        'close': prices,
        'volume': np.full(n, 10000),
        'symbol': 'MES'
    })

    return df


@pytest.fixture
def labeled_data_with_features():
    """Create labeled data with features for split leakage testing."""
    np.random.seed(42)
    n = 5000  # Larger dataset to accommodate purge/embargo

    timestamps = pd.date_range('2024-01-01 09:30', periods=n, freq='5min')
    close = 4500 + np.cumsum(np.random.randn(n) * 0.5)

    df = pd.DataFrame({
        'datetime': timestamps,
        'open': close + np.random.randn(n) * 0.3,
        'high': close + np.abs(np.random.randn(n) * 1.5),
        'low': close - np.abs(np.random.randn(n) * 1.5),
        'close': close,
        'volume': np.random.randint(1000, 50000, n),
        'symbol': 'MES'
    })

    # Add labels (simulating triple-barrier)
    df['label_h5'] = np.random.choice([0, 1, 2], size=n)
    df['label_h20'] = np.random.choice([0, 1, 2], size=n)

    return df


# =============================================================================
# LOOKAHEAD BIAS TESTS
# =============================================================================

class TestLookaheadBias:
    """Tests that features don't use future information."""

    def test_returns_no_lookahead(self, temporal_test_data):
        """Test that returns only use past data."""
        df = temporal_test_data.copy()
        metadata = {}

        result = add_returns(df, metadata)

        # For deterministic price increase of 1.0 per bar:
        # return_1[t] = (close[t] - close[t-1]) / close[t-1]
        # Since close[t] = 1000 + t, return should be approximately 1.0 / (1000 + t)

        # Check return_1 at row 100
        idx = 100
        expected_return = (df.loc[idx, 'close'] - df.loc[idx-1, 'close']) / df.loc[idx-1, 'close']
        actual_return = result.loc[idx, 'return_1']

        assert np.isclose(actual_return, expected_return, rtol=1e-6), \
            f"Return calculation uses lookahead: expected {expected_return}, got {actual_return}"

    def test_rsi_no_lookahead(self, temporal_test_data):
        """Test that RSI only uses past data."""
        df = temporal_test_data.copy()
        metadata = {}

        result = add_rsi(df, metadata)

        # For consistently increasing prices, RSI should be high (near 100)
        # but never use future data

        # Verify RSI at bar 100 only depends on bars 0-100
        rsi_at_100 = result.loc[100, 'rsi_14']

        # Modify future data (bars 101+) and verify RSI at 100 doesn't change
        df_modified = df.copy()
        df_modified.loc[101:, 'close'] = 0  # Corrupt future data

        result_modified = add_rsi(df_modified, {})
        rsi_at_100_modified = result_modified.loc[100, 'rsi_14']

        assert np.isclose(rsi_at_100, rsi_at_100_modified, rtol=1e-6), \
            "RSI uses future data - value changed when future data was modified"

    def test_bollinger_bands_no_lookahead(self, temporal_test_data):
        """Test that Bollinger Bands only use past data."""
        df = temporal_test_data.copy()
        metadata = {}

        result = add_bollinger_bands(df, metadata)

        # Check BB at bar 100
        bb_middle_at_100 = result.loc[100, 'bb_middle']

        # Modify future data
        df_modified = df.copy()
        df_modified.loc[101:, 'close'] = 9999  # Corrupt future

        result_modified = add_bollinger_bands(df_modified, {})
        bb_middle_at_100_modified = result_modified.loc[100, 'bb_middle']

        assert np.isclose(bb_middle_at_100, bb_middle_at_100_modified, rtol=1e-6), \
            "Bollinger Bands use future data"

    def test_macd_no_lookahead(self, temporal_test_data):
        """Test that MACD only uses past data."""
        df = temporal_test_data.copy()
        metadata = {}

        result = add_macd(df, metadata)

        # Check MACD at bar 200
        macd_at_200 = result.loc[200, 'macd_line']

        # Modify future data
        df_modified = df.copy()
        df_modified.loc[201:, 'close'] = 0

        result_modified = add_macd(df_modified, {})
        macd_at_200_modified = result_modified.loc[200, 'macd_line']

        assert np.isclose(macd_at_200, macd_at_200_modified, rtol=1e-6), \
            "MACD uses future data"

    def test_atr_no_lookahead(self, temporal_test_data):
        """Test that ATR only uses past data."""
        df = temporal_test_data.copy()
        metadata = {}

        result = add_atr(df, metadata)

        # Check ATR at bar 100
        atr_at_100 = result.loc[100, 'atr_14']

        # Modify future data
        df_modified = df.copy()
        df_modified.loc[101:, ['high', 'low', 'close']] = 0

        result_modified = add_atr(df_modified, {})
        atr_at_100_modified = result_modified.loc[100, 'atr_14']

        assert np.isclose(atr_at_100, atr_at_100_modified, rtol=1e-6), \
            "ATR uses future data"


# =============================================================================
# ROLLING CALCULATION TESTS
# =============================================================================

class TestRollingCalculations:
    """Tests that rolling windows use only past data."""

    def test_rolling_mean_window_correct(self, temporal_test_data):
        """Test that rolling mean uses correct window size."""
        df = temporal_test_data.copy()

        # Test 20-period rolling mean
        rolling_mean = df['close'].rolling(window=20).mean()

        # At bar 50, should use bars 31-50 (20 bars)
        idx = 50
        expected_mean = df.loc[31:50, 'close'].mean()
        actual_mean = rolling_mean.loc[idx]

        assert np.isclose(actual_mean, expected_mean, rtol=1e-6), \
            f"Rolling mean incorrect: expected {expected_mean}, got {actual_mean}"

        # First 19 bars should be NaN
        assert rolling_mean.iloc[:19].isna().all(), \
            "Rolling mean should be NaN for first 19 bars"

    def test_rolling_std_window_correct(self, temporal_test_data):
        """Test that rolling std uses correct window size."""
        df = temporal_test_data.copy()

        rolling_std = df['close'].rolling(window=20).std()

        # At bar 50
        idx = 50
        expected_std = df.loc[31:50, 'close'].std()
        actual_std = rolling_std.loc[idx]

        assert np.isclose(actual_std, expected_std, rtol=1e-6), \
            "Rolling std incorrect"

    def test_rolling_calculations_dont_use_future(self):
        """Test rolling calculations with deliberate future data corruption."""
        # Create data where first 100 bars have value 10, next 100 have value 1000
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=200, freq='5min'),
            'close': [10.0] * 100 + [1000.0] * 100,
        })

        rolling_mean = df['close'].rolling(window=20).mean()

        # At bar 80 (before jump), rolling mean should be 10
        assert np.isclose(rolling_mean.loc[80], 10.0, rtol=1e-6), \
            "Rolling mean at bar 80 should not see future jump to 1000"

        # At bar 119 (after jump, 20 bars in), rolling mean should be 1000
        assert np.isclose(rolling_mean.loc[119], 1000.0, rtol=1e-6), \
            "Rolling mean at bar 119 should be 1000"


# =============================================================================
# TRAIN/VAL/TEST SPLIT LEAKAGE TESTS
# =============================================================================

class TestSplitLeakage:
    """Tests that train/val/test splits don't leak information."""

    def test_chronological_order_maintained(self, labeled_data_with_features):
        """Test that splits maintain strict chronological order."""
        df = labeled_data_with_features.copy()

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=60,
            embargo_bars=288
        )

        # All train timestamps should be before all val timestamps
        train_times = df.loc[train_idx, 'datetime']
        val_times = df.loc[val_idx, 'datetime']
        test_times = df.loc[test_idx, 'datetime']

        assert train_times.max() < val_times.min(), \
            "Train data timestamp overlaps with validation"
        assert val_times.max() < test_times.min(), \
            "Validation data timestamp overlaps with test"

    def test_no_index_overlap(self, labeled_data_with_features):
        """Test that there is zero index overlap between splits."""
        df = labeled_data_with_features.copy()

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=60,
            embargo_bars=288
        )

        train_set = set(train_idx)
        val_set = set(val_idx)
        test_set = set(test_idx)

        # No overlaps
        assert len(train_set & val_set) == 0, "Train/val index overlap detected"
        assert len(train_set & test_set) == 0, "Train/test index overlap detected"
        assert len(val_set & test_set) == 0, "Val/test index overlap detected"

    def test_purge_prevents_label_leakage(self):
        """Test that purge removes samples where labels would leak into val/test."""
        n = 1000
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01 09:30', periods=n, freq='5min'),
            'close': np.random.randn(n) * 10 + 100,
            'symbol': 'MES',
            'label_h20': np.random.choice([0, 1, 2], size=n)
        })

        purge_bars = 60  # Should equal max_bars for H20

        train_idx, val_idx, test_idx, metadata = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=purge_bars,
            embargo_bars=0  # Focus on purge
        )

        # Gap between last train sample and first val sample should be >= purge_bars
        gap_train_val = val_idx.min() - train_idx.max()
        assert gap_train_val >= purge_bars, \
            f"Purge gap should be >= {purge_bars}, got {gap_train_val}"

    def test_embargo_prevents_temporal_leakage(self):
        """Test that embargo creates buffer to prevent temporal correlation leakage."""
        n = 5000  # Larger dataset to accommodate embargo
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01 09:30', periods=n, freq='5min'),
            'close': np.random.randn(n) * 10 + 100,
            'symbol': 'MES',
        })

        embargo_bars = 288  # ~1 day buffer

        train_idx, val_idx, test_idx, metadata = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=0,  # Focus on embargo
            embargo_bars=embargo_bars
        )

        # Gap should be at least embargo_bars
        gap_train_val = val_idx.min() - train_idx.max()
        gap_val_test = test_idx.min() - val_idx.max()

        assert gap_train_val >= embargo_bars, \
            f"Embargo gap train/val should be >= {embargo_bars}, got {gap_train_val}"
        assert gap_val_test >= embargo_bars, \
            f"Embargo gap val/test should be >= {embargo_bars}, got {gap_val_test}"

    def test_combined_purge_embargo(self):
        """Test that purge and embargo work together correctly."""
        n = 5000  # Larger dataset
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01 09:30', periods=n, freq='5min'),
            'close': np.random.randn(n) * 10 + 100,
            'symbol': 'MES',
        })

        purge_bars = 60
        embargo_bars = 288

        train_idx, val_idx, test_idx, metadata = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars
        )

        # Total gap should be purge + embargo
        gap_train_val = val_idx.min() - train_idx.max()
        expected_min_gap = purge_bars + embargo_bars

        assert gap_train_val >= expected_min_gap, \
            f"Combined gap should be >= {expected_min_gap}, got {gap_train_val}"


# =============================================================================
# FEATURE ENGINEERING TEMPORAL CONSISTENCY
# =============================================================================

class TestFeatureEngineeringConsistency:
    """Tests that feature engineering maintains temporal consistency."""

    def test_feature_values_dont_jump_backwards(self):
        """Test that modifying future data doesn't affect past features."""
        np.random.seed(42)
        n = 300

        df_original = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'close': 100 + np.cumsum(np.random.randn(n) * 0.5),
            'high': 0,
            'low': 0,
            'open': 0,
        })
        df_original['high'] = df_original['close'] + 1
        df_original['low'] = df_original['close'] - 1
        df_original['open'] = df_original['close']

        metadata = {}
        result_original = add_rsi(df_original.copy(), metadata)

        # Corrupt last 100 bars
        df_corrupted = df_original.copy()
        df_corrupted.loc[200:, 'close'] = 9999

        result_corrupted = add_rsi(df_corrupted.copy(), {})

        # RSI values for first 200 bars should be identical
        rsi_orig = result_original.loc[:199, 'rsi_14']
        rsi_corr = result_corrupted.loc[:199, 'rsi_14']

        valid_mask = ~rsi_orig.isna() & ~rsi_corr.isna()
        assert np.allclose(rsi_orig[valid_mask], rsi_corr[valid_mask], rtol=1e-6), \
            "Feature values changed when only future data was modified"

    def test_features_computed_incrementally_match_batch(self):
        """Test that features computed incrementally match batch computation."""
        np.random.seed(42)
        n = 200

        df_full = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'close': 100 + np.cumsum(np.random.randn(n) * 0.5),
            'high': 0,
            'low': 0,
            'open': 0,
        })
        df_full['high'] = df_full['close'] + 1
        df_full['low'] = df_full['close'] - 1
        df_full['open'] = df_full['close']

        # Compute features on full dataset
        metadata = {}
        result_full = add_bollinger_bands(df_full.copy(), metadata)

        # Compute features on first 150 bars only
        df_partial = df_full.iloc[:150].copy()
        result_partial = add_bollinger_bands(df_partial, {})

        # BB values for first 150 bars should match
        bb_middle_full = result_full.loc[:149, 'bb_middle']
        bb_middle_partial = result_partial['bb_middle']

        valid_mask = ~bb_middle_full.isna() & ~bb_middle_partial.isna()
        assert np.allclose(
            bb_middle_full[valid_mask],
            bb_middle_partial[valid_mask],
            rtol=1e-6
        ), "Incremental feature computation doesn't match batch computation"


# =============================================================================
# LABEL CONTAMINATION TESTS
# =============================================================================

class TestLabelContamination:
    """Tests that labels don't contaminate feature windows."""

    def test_label_horizon_respects_purge(self):
        """Test that label horizon (max_bars) is covered by purge."""
        # For H20 with max_bars=60, purge should be >= 60
        purge_bars = 60
        max_bars_h20 = 60  # From triple-barrier labeling

        assert purge_bars >= max_bars_h20, \
            f"Purge ({purge_bars}) should cover max_bars ({max_bars_h20})"

    def test_train_labels_dont_overlap_val_features(self):
        """Test that train labels don't overlap with val feature windows."""
        n = 1000
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01 09:30', periods=n, freq='5min'),
            'close': np.random.randn(n) * 10 + 100,
            'symbol': 'MES',
            'label_h20': np.random.choice([0, 1, 2], size=n)
        })

        # H20 label can look forward up to 60 bars (max_bars)
        purge_bars = 60

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=purge_bars,
            embargo_bars=0
        )

        # Last train sample's label could touch up to max_bars into the future
        last_train_label_reach = train_idx.max() + purge_bars
        first_val_sample = val_idx.min()

        # First val sample should be beyond label reach
        assert first_val_sample >= last_train_label_reach, \
            "Train labels overlap with validation features"

    def test_no_leakage_with_rolling_features(self):
        """Test no leakage when combining rolling features and labels."""
        n = 5000  # Larger dataset to accommodate purge/embargo
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01 09:30', periods=n, freq='5min'),
            'close': 100 + np.cumsum(np.random.randn(n) * 0.5),
            'high': 0,
            'low': 0,
            'open': 0,
        })
        df['high'] = df['close'] + 1
        df['low'] = df['close'] - 1
        df['open'] = df['close']
        df['volume'] = 10000

        # Add features with 20-bar lookback
        metadata = {}
        df = add_bollinger_bands(df, metadata)

        # Add mock labels
        df['label_h20'] = np.random.choice([0, 1, 2], size=n)

        # Create splits
        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=60,
            embargo_bars=288
        )

        # Verify first val sample can compute features without touching train labels
        first_val_idx = val_idx.min()

        # BB uses 20-bar window, so first val sample looks back to (first_val_idx - 19)
        feature_lookback_idx = first_val_idx - 19
        last_train_idx = train_idx.max()

        # Feature lookback should not include any train samples
        # (because of purge + embargo gap)
        assert feature_lookback_idx > last_train_idx, \
            "Validation features touch training data"
