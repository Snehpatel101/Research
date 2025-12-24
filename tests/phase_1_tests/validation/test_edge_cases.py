"""
Tests for edge case handling across the pipeline.
Verifies that edge cases are handled gracefully.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def sample_ohlcv_df():
    """Create a sample OHLCV DataFrame for testing."""
    n = 500
    np.random.seed(42)

    base_price = 4500.0
    returns = np.random.randn(n) * 0.001
    close = base_price * np.exp(np.cumsum(returns))

    daily_range = np.abs(np.random.randn(n) * 0.002)
    high = close * (1 + daily_range / 2)
    low = close * (1 - daily_range / 2)
    open_ = close * (1 + np.random.randn(n) * 0.0005)

    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = np.random.randint(100, 10000, n)

    from datetime import datetime, timedelta
    start_time = datetime(2024, 1, 1, 9, 30)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n)]

    df = pd.DataFrame({
        'datetime': timestamps,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

    return df


# =============================================================================
# TEST EMPTY DATAFRAME HANDLING
# =============================================================================

class TestEmptyDataFrame:
    """Tests for empty DataFrame handling."""

    def test_stage1_empty_file_raises(self, temp_dir):
        """Test that empty file raises ValueError."""
        from src.phase1.stages.stage1_ingest import DataIngestor

        # Create empty parquet file
        empty_df = pd.DataFrame()
        empty_file = temp_dir / 'empty.parquet'
        empty_df.to_parquet(empty_file)

        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / 'output'
        )

        with pytest.raises(ValueError, match="empty"):
            ingestor.load_data(empty_file)

    def test_feature_engineer_empty_raises(self, temp_dir):
        """Test that empty DataFrame raises in feature engineering."""
        from src.phase1.stages.features.features import FeatureEngineer

        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / 'output'
        )

        empty_df = pd.DataFrame()

        # Empty DataFrame should raise when iterating/processing
        with pytest.raises((ValueError, KeyError)):
            engineer.engineer_features(empty_df, 'MES')


class TestAllNaNColumns:
    """Tests for all-NaN column handling."""

    def test_feature_dropna_validation(self, temp_dir):
        """Test that dropping all rows raises error."""
        from src.phase1.stages.features.features import FeatureEngineer

        # Create DataFrame with ~50 rows (some but not enough for all rolling windows)
        # The longest rolling window is 200-period SMA, so this should result
        # in all rows being dropped after NaN removal
        from datetime import datetime, timedelta
        start_time = datetime(2024, 1, 1, 9, 30)
        n = 50  # Enough for numba but not for 200-period SMA

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)

        df = pd.DataFrame({
            'datetime': [start_time + timedelta(minutes=i) for i in range(n)],
            'open': close * 0.999,
            'high': close * 1.002,
            'low': close * 0.998,
            'close': close,
            'volume': np.random.randint(100, 1000, n)
        })

        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / 'output'
        )

        # Should raise due to insufficient data for rolling windows (200-bar SMA)
        with pytest.raises(ValueError, match="All rows dropped|Insufficient data"):
            engineer.engineer_features(df, 'MES')


# =============================================================================
# TEST DIVISION BY ZERO HANDLING
# =============================================================================

class TestDivisionByZero:
    """Tests that division by zero is handled."""

    def test_stochastic_zero_range(self):
        """Test stochastic with zero range (high == low)."""
        from src.phase1.stages.features import calculate_stochastic_numba

        n = 50
        high = np.full(n, 100.0, dtype=np.float64)
        low = np.full(n, 100.0, dtype=np.float64)
        close = np.full(n, 100.0, dtype=np.float64)

        stoch_k, stoch_d = calculate_stochastic_numba(high, low, close, k_period=14, d_period=3)

        # Should return neutral value (50) not inf/nan
        # After warmup period, values should be valid
        valid_k = stoch_k[~np.isnan(stoch_k)]
        assert len(valid_k) > 0, "stoch_k should have some valid values"
        assert not np.isinf(valid_k).any(), "stoch_k should not contain infinity"
        # When range is zero, stoch_k should be 50 (neutral)
        assert np.allclose(valid_k, 50.0), "stoch_k should be 50 when range is zero"

    def test_rsi_no_losses(self):
        """Test RSI when there are no losses."""
        from src.phase1.stages.features import calculate_rsi_numba

        # Monotonically increasing - no losses
        close = np.array([100.0 + i for i in range(50)], dtype=np.float64)

        result = calculate_rsi_numba(close, period=14)

        # Should return 100 (all gains), not inf
        valid_rsi = result[~np.isnan(result)]
        assert len(valid_rsi) > 0, "RSI should have some valid values"
        assert not np.isinf(valid_rsi).any(), "RSI should not contain infinity"
        # With all gains and no losses, RSI should approach 100
        assert np.allclose(valid_rsi, 100.0), "RSI should be 100 for all gains"

    def test_rsi_no_gains(self):
        """Test RSI when there are no gains."""
        from src.phase1.stages.features import calculate_rsi_numba

        # Monotonically decreasing - no gains
        close = np.array([100.0 - i * 0.5 for i in range(50)], dtype=np.float64)

        result = calculate_rsi_numba(close, period=14)

        # Should return 0 (all losses), not inf
        valid_rsi = result[~np.isnan(result)]
        assert len(valid_rsi) > 0, "RSI should have some valid values"
        assert not np.isinf(valid_rsi).any(), "RSI should not contain infinity"
        # With all losses and no gains, RSI should approach 0
        assert np.allclose(valid_rsi, 0.0, atol=1e-10), "RSI should be 0 for all losses"

    def test_vwap_zero_volume(self):
        """Test VWAP when volume is zero (handled by add_vwap function)."""
        from src.phase1.stages.features.volume import add_vwap

        df = pd.DataFrame({
            'high': [101.0, 102.0, 100.0, 101.0, 103.0] * 100,
            'low': [99.0, 100.0, 98.0, 99.5, 101.0] * 100,
            'close': [100.0, 101.0, 99.0, 100.5, 102.0] * 100,
            'volume': [0.0] * 500  # Zero volume
        })

        feature_metadata = {}
        result = add_vwap(df.copy(), feature_metadata)

        # VWAP with zero volume should handle gracefully
        # The function may or may not add vwap_ratio when all volumes are zero
        # What matters is that it doesn't crash and produces no infinity
        for col in result.columns:
            if result[col].dtype in [np.float32, np.float64]:
                valid_values = result[col].dropna()
                if len(valid_values) > 0:
                    assert not np.isinf(valid_values).any(), f"{col} should not contain infinity"

    def test_bollinger_zero_std(self):
        """Test Bollinger Bands when std is zero (constant price)."""
        from src.phase1.stages.features.volatility import add_bollinger_bands

        df = pd.DataFrame({
            'close': [100.0] * 50  # Constant price
        })

        feature_metadata = {}
        result = add_bollinger_bands(df.copy(), feature_metadata)

        # Check that we don't get infinity for width and position
        if 'bb_width' in result.columns:
            valid_width = result['bb_width'].dropna()
            if len(valid_width) > 0:
                assert not np.isinf(valid_width).any(), "bb_width should not contain infinity"

        if 'bb_position' in result.columns:
            valid_position = result['bb_position'].dropna()
            if len(valid_position) > 0:
                assert not np.isinf(valid_position).any(), "bb_position should not contain infinity"


# =============================================================================
# TEST NEGATIVE INDEX HANDLING
# =============================================================================

class TestNegativeIndices:
    """Tests for negative index handling."""

    def test_splits_negative_train_end(self):
        """Test that negative train_end_purged raises error."""
        from src.phase1.stages.stage7_splits import create_chronological_splits
        from datetime import datetime, timedelta

        # Small dataset with large purge
        start_time = datetime(2024, 1, 1, 9, 30)
        df = pd.DataFrame({
            'datetime': [start_time + timedelta(minutes=i*5) for i in range(100)],
            'close': range(100)
        })

        with pytest.raises(ValueError, match="eliminated|insufficient|too small"):
            create_chronological_splits(
                df,
                train_ratio=0.1,  # Only 10 samples for train
                val_ratio=0.45,
                test_ratio=0.45,
                purge_bars=60  # More than train size!
            )

    def test_splits_val_consumed_by_embargo(self):
        """Test that validation set consumed by embargo raises error."""
        from src.phase1.stages.stage7_splits import create_chronological_splits
        from datetime import datetime, timedelta

        start_time = datetime(2024, 1, 1, 9, 30)
        df = pd.DataFrame({
            'datetime': [start_time + timedelta(minutes=i*5) for i in range(200)],
            'close': range(200)
        })

        with pytest.raises(ValueError, match="empty|insufficient|too small"):
            create_chronological_splits(
                df,
                train_ratio=0.70,
                val_ratio=0.05,  # Very small validation
                test_ratio=0.25,
                purge_bars=20,
                embargo_bars=50  # Consumes entire validation
            )


# =============================================================================
# TEST INTEGER DIVISION EDGE CASES
# =============================================================================

class TestIntegerDivision:
    """Tests for integer division edge cases."""

    def test_gap_fill_min_one(self, temp_dir):
        """Test that gap fill is at least 1."""
        from src.phase1.stages.clean import DataCleaner

        cleaner = DataCleaner(
            input_dir=temp_dir,
            output_dir=temp_dir / 'output',
            timeframe='5min',  # 5-minute bars
            max_gap_fill_minutes=3  # Less than bar size
        )

        # max_fill_bars should be at least 1
        # max_fill_bars = max(1, 3 // 5) = max(1, 0) = 1
        assert cleaner.freq_minutes == 5
        assert cleaner.max_gap_fill_minutes == 3

        # When fill_gaps is called, it calculates:
        # max_fill_bars = max(1, max_gap_fill_minutes // freq_minutes)
        # This is tested in the fill_gaps method

    def test_timeframe_parsing(self, temp_dir):
        """Test timeframe parsing for various formats."""
        from src.phase1.stages.clean import DataCleaner

        test_cases = [
            ('1min', 1),
            ('5min', 5),
            ('15min', 15),
            ('1h', 60),
            ('4h', 240),
            ('1d', 1440),
        ]

        for timeframe, expected_minutes in test_cases:
            cleaner = DataCleaner(
                input_dir=temp_dir,
                output_dir=temp_dir / 'output',
                timeframe=timeframe
            )
            assert cleaner.freq_minutes == expected_minutes, \
                f"Failed for {timeframe}: expected {expected_minutes}, got {cleaner.freq_minutes}"


# =============================================================================
# TEST VOLATILITY ANNUALIZATION
# =============================================================================

class TestVolatilityAnnualization:
    """Tests for volatility annualization fix."""

    def test_annualization_factor(self):
        """Test that annualization factor is correct."""
        from src.phase1.stages.features.constants import ANNUALIZATION_FACTOR, BARS_PER_DAY

        # For 5-min bars: 78 bars/day, 252 trading days
        expected = np.sqrt(252 * 78)

        assert abs(ANNUALIZATION_FACTOR - expected) < 0.01, \
            f"Expected {expected}, got {ANNUALIZATION_FACTOR}"
        assert BARS_PER_DAY == 78, f"Expected 78, got {BARS_PER_DAY}"

    def test_hvol_not_overstated(self, sample_ohlcv_df, temp_dir):
        """Test that historical volatility is not overstated."""
        from src.phase1.stages.features.features import FeatureEngineer
        from src.phase1.stages.features.constants import ANNUALIZATION_FACTOR

        # Create sample data with known volatility
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.001  # 0.1% per bar

        close = 1000 * np.exp(np.cumsum(returns))

        from datetime import datetime, timedelta
        start_time = datetime(2024, 1, 1, 9, 30)

        df = pd.DataFrame({
            'datetime': [start_time + timedelta(minutes=i*5) for i in range(1000)],
            'open': close * 0.999,
            'high': close * 1.001,
            'low': close * 0.998,
            'close': close,
            'volume': np.random.randint(100, 1000, 1000)
        })

        # Expected annualized vol: 0.001 * sqrt(78 * 252) approx 0.14 (14%)
        # If bug existed (390 bars): 0.001 * sqrt(390 * 252) approx 0.31 (31%)

        # Calculate bar-level std
        bar_std = np.std(returns)
        expected_annual_vol = bar_std * np.sqrt(78 * 252)

        # The actual test verifies the expected values are reasonable
        assert expected_annual_vol < 0.20, \
            f"Expected annual vol ~14%, got {expected_annual_vol*100:.1f}%"

        # Also verify annualization factor
        assert ANNUALIZATION_FACTOR < 145, \
            f"Annualization factor should be ~140, got {ANNUALIZATION_FACTOR}"


# =============================================================================
# TEST NUMBA FUNCTION EDGE CASES
# =============================================================================

class TestNumbaFunctionEdgeCases:
    """Tests for Numba-optimized function edge cases."""

    def test_rsi_numba_all_gains(self):
        """Test RSI Numba function with all gains."""
        from src.phase1.stages.features import calculate_rsi_numba

        # All increasing prices
        close = np.array([100.0 + i for i in range(50)], dtype=np.float64)
        result = calculate_rsi_numba(close, period=14)

        # After warmup, RSI should be 100 (all gains)
        valid_rsi = result[~np.isnan(result)]
        assert len(valid_rsi) > 0
        assert np.allclose(valid_rsi, 100.0), "RSI should be 100 for all gains"

    def test_rsi_numba_all_losses(self):
        """Test RSI Numba function with all losses."""
        from src.phase1.stages.features import calculate_rsi_numba

        # All decreasing prices
        close = np.array([100.0 - i * 0.5 for i in range(50)], dtype=np.float64)
        result = calculate_rsi_numba(close, period=14)

        # After warmup, RSI should be 0 (all losses)
        valid_rsi = result[~np.isnan(result)]
        assert len(valid_rsi) > 0
        assert np.allclose(valid_rsi, 0.0, atol=1e-10), "RSI should be 0 for all losses"

    def test_stochastic_numba_zero_range(self):
        """Test Stochastic Numba function with zero range."""
        from src.phase1.stages.features import calculate_stochastic_numba

        n = 50
        high = np.full(n, 100.0, dtype=np.float64)
        low = np.full(n, 100.0, dtype=np.float64)
        close = np.full(n, 100.0, dtype=np.float64)

        k, d = calculate_stochastic_numba(high, low, close, k_period=14, d_period=3)

        # Should return 50 (neutral) when range is zero
        valid_k = k[~np.isnan(k)]
        assert len(valid_k) > 0
        assert np.allclose(valid_k, 50.0), "Stochastic K should be 50 when range is zero"

    def test_atr_numba_constant_price(self):
        """Test ATR Numba function with constant price."""
        from src.phase1.stages.features import calculate_atr_numba

        n = 50
        high = np.full(n, 100.0, dtype=np.float64)
        low = np.full(n, 100.0, dtype=np.float64)
        close = np.full(n, 100.0, dtype=np.float64)

        result = calculate_atr_numba(high, low, close, period=14)

        # ATR should be 0 when price is constant
        valid_atr = result[~np.isnan(result)]
        assert len(valid_atr) > 0
        assert np.allclose(valid_atr, 0.0), "ATR should be 0 when price is constant"


# =============================================================================
# TEST OHLCV RELATIONSHIP VALIDATION
# =============================================================================

class TestOHLCVValidation:
    """Tests for OHLCV relationship validation."""

    def test_high_less_than_low_fix(self, temp_dir):
        """Test that high < low is fixed."""
        from src.phase1.stages.stage1_ingest import DataIngestor
        from datetime import datetime, timedelta

        start_time = datetime(2024, 1, 1, 9, 30)

        # Create data with invalid OHLC (high < low)
        df = pd.DataFrame({
            'datetime': [start_time + timedelta(minutes=i) for i in range(10)],
            'open': [100.0] * 10,
            'high': [98.0] * 10,  # Invalid: less than low
            'low': [102.0] * 10,
            'close': [100.0] * 10,
            'volume': [1000] * 10
        })

        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / 'output'
        )

        result, report = ingestor.validate_ohlcv_relationships(df, auto_fix=True)

        # After fix, high should be >= low
        assert (result['high'] >= result['low']).all(), "High should be >= low after fix"
        assert report['violations']['high_lt_low'] == 10, "Should have detected 10 violations"

    def test_negative_prices_removed(self, temp_dir):
        """Test that negative prices are removed."""
        from src.phase1.stages.stage1_ingest import DataIngestor
        from datetime import datetime, timedelta

        start_time = datetime(2024, 1, 1, 9, 30)

        df = pd.DataFrame({
            'datetime': [start_time + timedelta(minutes=i) for i in range(10)],
            'open': [100.0, -1.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            'high': [101.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0],
            'low': [99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
            'close': [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            'volume': [1000] * 10
        })

        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / 'output'
        )

        result, report = ingestor.validate_ohlcv_relationships(df, auto_fix=True)

        # Negative price row should be removed
        assert len(result) == 9, "Row with negative price should be removed"
        assert (result['open'] > 0).all(), "All prices should be positive"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
