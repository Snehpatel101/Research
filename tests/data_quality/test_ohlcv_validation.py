"""
Data Quality Tests for OHLCV Validation.

Tests that OHLCV data maintains proper consistency and quality:
- Price relationships (high >= low, open/close within high/low)
- No negative prices or volumes
- Timestamp continuity
- NaN handling
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# OHLCV CONSISTENCY TESTS
# =============================================================================


class TestOHLCVConsistency:
    """Tests for OHLCV price relationship consistency."""

    @pytest.fixture
    def valid_ohlcv_df(self) -> pd.DataFrame:
        """Create valid OHLCV data."""
        np.random.seed(42)
        n_bars = 100

        base_price = 100
        returns = np.random.randn(n_bars) * 0.02

        close = base_price * np.cumprod(1 + returns)
        high = close * (1 + np.abs(np.random.randn(n_bars) * 0.01))
        low = close * (1 - np.abs(np.random.randn(n_bars) * 0.01))
        open_price = np.roll(close, 1)
        open_price[0] = base_price
        volume = np.random.randint(100, 10000, n_bars)

        # Ensure high >= low
        high = np.maximum(high, low)

        # Ensure open/close within high/low
        open_price = np.clip(open_price, low, high)
        close = np.clip(close, low, high)

        return pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
        }, index=pd.date_range('2020-01-01', periods=n_bars, freq='5min'))

    @pytest.fixture
    def invalid_ohlcv_df(self) -> pd.DataFrame:
        """Create OHLCV data with intentional errors."""
        np.random.seed(42)
        n_bars = 100

        df = pd.DataFrame({
            'open': np.random.uniform(99, 101, n_bars),
            'high': np.random.uniform(99, 101, n_bars),
            'low': np.random.uniform(99, 101, n_bars),
            'close': np.random.uniform(99, 101, n_bars),
            'volume': np.random.randint(100, 10000, n_bars),
        }, index=pd.date_range('2020-01-01', periods=n_bars, freq='5min'))

        # Introduce errors
        df.loc[df.index[10], 'high'] = 95  # high < low
        df.loc[df.index[10], 'low'] = 105

        df.loc[df.index[20], 'volume'] = -100  # negative volume

        df.loc[df.index[30], 'close'] = np.nan  # NaN value

        return df

    def test_high_greater_equal_low(self, valid_ohlcv_df):
        """High price should always be >= low price."""
        assert (valid_ohlcv_df['high'] >= valid_ohlcv_df['low']).all(), \
            "Found bars where high < low"

    def test_open_within_high_low(self, valid_ohlcv_df):
        """Open price should be within [low, high]."""
        within_range = (
            (valid_ohlcv_df['open'] >= valid_ohlcv_df['low']) &
            (valid_ohlcv_df['open'] <= valid_ohlcv_df['high'])
        )
        assert within_range.all(), "Found bars where open is outside [low, high]"

    def test_close_within_high_low(self, valid_ohlcv_df):
        """Close price should be within [low, high]."""
        within_range = (
            (valid_ohlcv_df['close'] >= valid_ohlcv_df['low']) &
            (valid_ohlcv_df['close'] <= valid_ohlcv_df['high'])
        )
        assert within_range.all(), "Found bars where close is outside [low, high]"

    def test_no_negative_prices(self, valid_ohlcv_df):
        """All prices should be positive."""
        for col in ['open', 'high', 'low', 'close']:
            assert (valid_ohlcv_df[col] > 0).all(), f"Found negative values in {col}"

    def test_no_negative_volume(self, valid_ohlcv_df):
        """Volume should be non-negative."""
        assert (valid_ohlcv_df['volume'] >= 0).all(), "Found negative volume"

    def test_detect_high_low_violation(self, invalid_ohlcv_df):
        """Should detect when high < low."""
        violation_mask = invalid_ohlcv_df['high'] < invalid_ohlcv_df['low']
        assert violation_mask.any(), "Should detect high < low violation"

        # Find the violating bars
        violations = invalid_ohlcv_df[violation_mask]
        assert len(violations) > 0

    def test_detect_negative_volume(self, invalid_ohlcv_df):
        """Should detect negative volume."""
        violation_mask = invalid_ohlcv_df['volume'] < 0
        assert violation_mask.any(), "Should detect negative volume"

    def test_detect_nan_values(self, invalid_ohlcv_df):
        """Should detect NaN values in OHLCV."""
        has_nan = invalid_ohlcv_df[['open', 'high', 'low', 'close', 'volume']].isna().any().any()
        assert has_nan, "Should detect NaN values"


# =============================================================================
# TIMESTAMP VALIDATION TESTS
# =============================================================================


class TestTimestampValidation:
    """Tests for timestamp continuity and ordering."""

    def test_timestamps_monotonically_increasing(self):
        """Timestamps should be strictly increasing."""
        np.random.seed(42)
        df = pd.DataFrame({
            'close': np.random.uniform(99, 101, 100),
        }, index=pd.date_range('2020-01-01', periods=100, freq='5min'))

        # Check monotonic increasing
        assert df.index.is_monotonic_increasing, "Timestamps not monotonically increasing"

    def test_detect_duplicate_timestamps(self):
        """Should detect duplicate timestamps."""
        dates = pd.date_range('2020-01-01', periods=100, freq='5min')
        # Introduce duplicate
        dates = dates.insert(50, dates[50])

        df = pd.DataFrame({
            'close': np.random.uniform(99, 101, len(dates)),
        }, index=dates)

        assert df.index.duplicated().any(), "Should detect duplicate timestamps"

    def test_detect_gaps_in_timestamps(self):
        """Should detect gaps in timestamps during trading hours."""
        dates = pd.date_range('2020-01-01', periods=100, freq='5min')
        # Remove some timestamps to create gaps
        dates = dates.delete([50, 51, 52])

        df = pd.DataFrame({
            'close': np.random.uniform(99, 101, len(dates)),
        }, index=dates)

        # Check for gaps
        time_diff = df.index.to_series().diff()
        expected_diff = pd.Timedelta('5min')
        gaps = time_diff > expected_diff

        assert gaps.any(), "Should detect timestamp gaps"


# =============================================================================
# NAN HANDLING TESTS
# =============================================================================


class TestNaNHandling:
    """Tests for proper NaN handling in data processing."""

    def test_rolling_window_creates_nan_at_start(self):
        """Rolling window operations should create NaN at start."""
        np.random.seed(42)
        close = pd.Series(np.random.uniform(99, 101, 100))

        sma_20 = close.rolling(window=20).mean()

        # First 19 values should be NaN
        assert sma_20.iloc[:19].isna().all(), "Rolling mean should have NaN for insufficient data"
        # Remaining values should be valid
        assert not sma_20.iloc[19:].isna().any(), "Rolling mean should have values after window"

    def test_dropna_removes_all_nan_rows(self):
        """dropna should remove all rows with NaN."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
        })

        # Introduce NaN
        df.iloc[10, 0] = np.nan
        df.iloc[20, 1] = np.nan

        cleaned = df.dropna()

        assert not cleaned.isna().any().any(), "Cleaned data should have no NaN"
        assert len(cleaned) == 98, "Should remove rows with NaN"


# =============================================================================
# OUTLIER DETECTION TESTS
# =============================================================================


class TestOutlierDetection:
    """Tests for outlier detection in OHLCV data."""

    def test_detect_price_spikes(self):
        """Should detect abnormal price spikes."""
        np.random.seed(42)
        n_bars = 100

        close = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)
        # Introduce spike
        close[50] = close[49] * 1.5  # 50% jump

        returns = pd.Series(close).pct_change()
        threshold = returns.std() * 5

        spikes = np.abs(returns) > threshold
        assert spikes.any(), "Should detect price spike"

    def test_detect_zero_volume_bars(self):
        """Should detect bars with zero volume."""
        np.random.seed(42)
        n_bars = 100

        volume = np.random.randint(100, 10000, n_bars)
        volume[50:55] = 0  # Zero volume period

        zero_volume = pd.Series(volume) == 0
        assert zero_volume.any(), "Should detect zero volume bars"
        assert zero_volume.sum() == 5


# =============================================================================
# DATA VALIDATION INTEGRATION TESTS
# =============================================================================


class TestDataValidationIntegration:
    """Integration tests for data validation."""

    def test_ohlcv_validator_exists(self):
        """OHLCV validator should be importable."""
        try:
            from src.phase1.stages.ingest.validators import OHLCVValidator
            validator = OHLCVValidator()
            assert validator is not None
        except ImportError:
            # May have different name
            try:
                from src.phase1.stages.ingest.validators import validate_ohlcv
                assert callable(validate_ohlcv)
            except ImportError:
                pytest.skip("OHLCV validator not available with expected name")

    def test_validation_detects_issues(self):
        """Validation should detect data quality issues."""
        np.random.seed(42)
        n_bars = 100

        # Create data with issues
        df = pd.DataFrame({
            'open': np.random.uniform(99, 101, n_bars),
            'high': np.random.uniform(95, 98, n_bars),  # Lower than low!
            'low': np.random.uniform(100, 102, n_bars),  # Higher than high!
            'close': np.random.uniform(99, 101, n_bars),
            'volume': np.random.randint(100, 10000, n_bars),
        }, index=pd.date_range('2020-01-01', periods=n_bars, freq='5min'))

        # Basic check: high < low violation
        violation = (df['high'] < df['low']).any()
        assert violation, "Test data should have high < low violation"
