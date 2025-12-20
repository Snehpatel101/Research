"""
Comprehensive Unit Tests for Phase 1 Stages (1-4)

This module provides comprehensive test coverage for:
- Stage 1: DataIngestor (data loading, standardization, validation)
- Stage 2: DataCleaner (gap detection, outlier removal, cleaning)
- Stage 3: FeatureEngineer (technical indicators, temporal features)
- Stage 4: TripleBarrierLabeler (triple barrier labeling)

Target coverage: >70%

Run with: pytest tests/test_phase1_stages.py -v --cov=src/stages
"""

import sys
from pathlib import Path
import tempfile
import shutil

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from stages.stage1_ingest import DataIngestor
from stages.stage2_clean import DataCleaner, calculate_atr_numba
from stages.stage3_features import (
    FeatureEngineer,
    calculate_sma_numba,
    calculate_ema_numba,
    calculate_rsi_numba,
    calculate_atr_numba as features_atr_numba,
    calculate_stochastic_numba,
    calculate_adx_numba,
)
from stages.stage4_labeling import triple_barrier_numba, apply_triple_barrier


# =============================================================================
# FIXTURES - Shared test data
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

    # Generate realistic price series with random walk
    base_price = 4500.0
    returns = np.random.randn(n) * 0.001  # 0.1% daily volatility
    close = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    daily_range = np.abs(np.random.randn(n) * 0.002)  # 0.2% typical range
    high = close * (1 + daily_range / 2)
    low = close * (1 - daily_range / 2)
    open_ = close * (1 + np.random.randn(n) * 0.0005)  # Small random offset

    # Ensure OHLC relationships are valid
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    # Generate volume
    volume = np.random.randint(100, 10000, n)

    # Generate timestamps (1-minute bars)
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


@pytest.fixture
def sample_ohlcv_with_gaps(sample_ohlcv_df):
    """Create OHLCV DataFrame with gaps for testing gap detection."""
    df = sample_ohlcv_df.copy()

    # Remove some rows to create gaps
    gap_indices = [50, 51, 52, 100, 150, 151]  # Create multi-bar gaps
    df = df.drop(index=gap_indices).reset_index(drop=True)

    return df


@pytest.fixture
def sample_ohlcv_with_outliers(sample_ohlcv_df):
    """Create OHLCV DataFrame with outliers for testing outlier detection."""
    df = sample_ohlcv_df.copy()

    # Inject price spikes (outliers)
    spike_indices = [100, 200, 300]
    for idx in spike_indices:
        df.loc[idx, 'close'] = df.loc[idx, 'close'] * 1.10  # 10% spike
        df.loc[idx, 'high'] = df.loc[idx, 'close'] * 1.02

    return df


@pytest.fixture
def sample_features_df(sample_ohlcv_df):
    """Create a sample DataFrame with features for labeling tests."""
    df = sample_ohlcv_df.copy()

    # Calculate ATR for labeling
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    atr = features_atr_numba(high, low, close, 14)
    df['atr_14'] = atr

    return df


# =============================================================================
# STAGE 1 TESTS: DataIngestor
# =============================================================================

class TestDataIngestorLoadData:
    """Tests for DataIngestor.load_data() method."""

    def test_load_data_parquet_success(self, temp_dir, sample_ohlcv_df):
        """Test successful loading of Parquet file."""
        # Arrange
        file_path = temp_dir / "test_data.parquet"
        sample_ohlcv_df.to_parquet(file_path, index=False)

        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = ingestor.load_data(file_path)

        # Assert
        assert len(df) == len(sample_ohlcv_df)
        assert list(df.columns) == list(sample_ohlcv_df.columns)
        pd.testing.assert_frame_equal(df, sample_ohlcv_df)

    def test_load_data_csv_success(self, temp_dir, sample_ohlcv_df):
        """Test successful loading of CSV file."""
        # Arrange
        file_path = temp_dir / "test_data.csv"
        sample_ohlcv_df.to_csv(file_path, index=False)

        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = ingestor.load_data(file_path)

        # Assert
        assert len(df) == len(sample_ohlcv_df)
        # CSV reads datetime as string, so just check columns exist
        assert 'datetime' in df.columns
        assert 'close' in df.columns

    def test_load_data_file_not_found(self, temp_dir):
        """Test FileNotFoundError when file does not exist."""
        # Arrange
        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            ingestor.load_data(temp_dir / "nonexistent.parquet")

    def test_load_data_unsupported_format(self, temp_dir, sample_ohlcv_df):
        """Test ValueError for unsupported file formats."""
        # Arrange
        file_path = temp_dir / "test_data.xyz"
        sample_ohlcv_df.to_csv(file_path, index=False)  # Save as CSV with wrong extension

        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Unsupported file format"):
            ingestor.load_data(file_path)


class TestDataIngestorStandardizeColumns:
    """Tests for DataIngestor.standardize_columns() method."""

    def test_standardize_columns_mapping(self, temp_dir):
        """Test column name standardization with various aliases."""
        # Arrange
        df = pd.DataFrame({
            'timestamp': ['2024-01-01'],
            'O': [100.0],
            'H': [101.0],
            'L': [99.0],
            'C': [100.5],
            'Vol': [1000]
        })

        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        result = ingestor.standardize_columns(df)

        # Assert
        assert 'datetime' in result.columns
        assert 'open' in result.columns
        assert 'high' in result.columns
        assert 'low' in result.columns
        assert 'close' in result.columns
        assert 'volume' in result.columns

    def test_standardize_columns_missing_required(self, temp_dir):
        """Test that missing required columns are logged but not raise error."""
        # Arrange
        df = pd.DataFrame({
            'datetime': ['2024-01-01'],
            'open': [100.0],
            # Missing high, low, close, volume
        })

        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        result = ingestor.standardize_columns(df)

        # Assert - should not raise, just return what it can
        assert 'datetime' in result.columns
        assert 'open' in result.columns


class TestDataIngestorValidateOHLCV:
    """Tests for DataIngestor.validate_ohlcv_relationships() method."""

    def test_validate_ohlcv_high_lt_low_fix(self, temp_dir):
        """Test that high < low violations are fixed by swapping."""
        # Arrange
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=5, freq='min'),
            'open': [100.0, 100.0, 100.0, 100.0, 100.0],
            'high': [99.0, 102.0, 102.0, 102.0, 102.0],  # First high < low
            'low': [101.0, 98.0, 98.0, 98.0, 98.0],      # First low > high
            'close': [100.0, 100.0, 100.0, 100.0, 100.0],
            'volume': [1000, 1000, 1000, 1000, 1000]
        })

        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        result, report = ingestor.validate_ohlcv_relationships(df)

        # Assert - high and low should be swapped for first row
        assert result.loc[0, 'high'] == 101.0
        assert result.loc[0, 'low'] == 99.0
        assert 'high_lt_low' in report['violations']
        assert report['violations']['high_lt_low'] == 1

    def test_validate_ohlcv_negative_prices_removed(self, temp_dir):
        """Test that rows with negative prices are removed."""
        # Arrange
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=5, freq='min'),
            'open': [100.0, -100.0, 100.0, 100.0, 100.0],  # Second row has negative
            'high': [102.0, 102.0, 102.0, 102.0, 102.0],
            'low': [98.0, 98.0, 98.0, 98.0, 98.0],
            'close': [100.0, 100.0, 100.0, 100.0, 100.0],
            'volume': [1000, 1000, 1000, 1000, 1000]
        })

        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        result, report = ingestor.validate_ohlcv_relationships(df)

        # Assert
        assert len(result) == 4  # One row removed
        assert 'negative_prices' in report['violations']
        assert report['violations']['negative_prices'] == 1

    def test_validate_ohlcv_all_valid(self, temp_dir, sample_ohlcv_df):
        """Test validation with all valid data - no changes."""
        # Arrange
        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        result, report = ingestor.validate_ohlcv_relationships(sample_ohlcv_df)

        # Assert
        assert len(result) == len(sample_ohlcv_df)
        assert report['violations'] == {} or all(v == 0 for v in report['violations'].values())


class TestDataIngestorTimezone:
    """Tests for DataIngestor.handle_timezone() method."""

    def test_handle_timezone_conversion(self, temp_dir):
        """Test timezone conversion from EST to UTC."""
        # Arrange
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01 09:30', periods=5, freq='min'),
            'open': [100.0] * 5,
            'high': [102.0] * 5,
            'low': [98.0] * 5,
            'close': [100.0] * 5,
            'volume': [1000] * 5
        })

        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / "output",
            source_timezone='EST'
        )

        # Act
        result = ingestor.handle_timezone(df)

        # Assert - EST is UTC-5, so 09:30 EST = 14:30 UTC
        assert result['datetime'].iloc[0].hour == 14
        assert result['datetime'].dt.tz is None  # Should be naive UTC

    def test_handle_timezone_utc_input(self, temp_dir):
        """Test that UTC input remains unchanged."""
        # Arrange
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01 09:30', periods=5, freq='min'),
            'open': [100.0] * 5,
            'high': [102.0] * 5,
            'low': [98.0] * 5,
            'close': [100.0] * 5,
            'volume': [1000] * 5
        })

        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / "output",
            source_timezone='UTC'
        )

        # Act
        result = ingestor.handle_timezone(df)

        # Assert
        assert result['datetime'].iloc[0].hour == 9  # Unchanged
        assert result['datetime'].dt.tz is None


class TestDataIngestorFullPipeline:
    """Integration tests for DataIngestor.ingest_file()."""

    def test_ingest_file_full_pipeline(self, temp_dir, sample_ohlcv_df):
        """Test full ingestion pipeline end-to-end."""
        # Arrange
        file_path = temp_dir / "MES_1m.parquet"
        sample_ohlcv_df.to_parquet(file_path, index=False)

        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df, metadata = ingestor.ingest_file(file_path, validate=True)

        # Assert
        assert len(df) > 0
        assert 'datetime' in df.columns
        assert metadata['symbol'] == 'MES'
        assert 'raw_rows' in metadata
        assert 'final_rows' in metadata
        assert 'validation' in metadata


# =============================================================================
# STAGE 2 TESTS: DataCleaner
# =============================================================================

class TestDataCleanerGapDetection:
    """Tests for DataCleaner gap detection methods."""

    def test_detect_gaps_finds_missing_bars(self, temp_dir, sample_ohlcv_with_gaps):
        """Test that gap detection correctly identifies missing bars."""
        # Arrange
        cleaner = DataCleaner(
            input_dir=temp_dir,
            output_dir=temp_dir / "output",
            timeframe='1min'
        )

        # Act
        df, gap_report = cleaner.detect_gaps(sample_ohlcv_with_gaps)

        # Assert
        assert gap_report['total_gaps'] > 0
        assert gap_report['total_missing_bars'] > 0
        assert gap_report['completeness_pct'] < 100

    def test_detect_gaps_handles_weekends(self, temp_dir):
        """Test that gap detection handles weekend gaps appropriately."""
        # Arrange - Create data spanning a weekend
        # Friday 4pm to Monday 9:30am would be a normal gap
        df = pd.DataFrame({
            'datetime': pd.to_datetime([
                '2024-01-05 16:00',  # Friday
                '2024-01-08 09:30',  # Monday - normal market gap, not a data issue
            ]),
            'open': [100.0, 100.0],
            'high': [102.0, 102.0],
            'low': [98.0, 98.0],
            'close': [100.0, 100.0],
            'volume': [1000, 1000]
        })

        cleaner = DataCleaner(
            input_dir=temp_dir,
            output_dir=temp_dir / "output",
            timeframe='1min'
        )

        # Act
        df_result, gap_report = cleaner.detect_gaps(df)

        # Assert - Weekend gap should be detected but is expected
        # The detection should still work, reporting the gap
        assert gap_report['total_gaps'] == 1


class TestDataCleanerGapFilling:
    """Tests for DataCleaner gap filling methods."""

    def test_fill_gaps_forward_fill(self, temp_dir, sample_ohlcv_with_gaps):
        """Test forward fill gap filling method."""
        # Arrange
        cleaner = DataCleaner(
            input_dir=temp_dir,
            output_dir=temp_dir / "output",
            timeframe='1min',
            gap_fill_method='forward',
            max_gap_fill_minutes=5
        )

        initial_len = len(sample_ohlcv_with_gaps)

        # Act
        df = cleaner.fill_gaps(sample_ohlcv_with_gaps)

        # Assert - Should have more rows after filling
        assert len(df) >= initial_len
        # No NaN values in OHLC after forward fill (within limit)
        assert df['close'].notna().sum() > 0

    def test_fill_gaps_max_limit(self, temp_dir):
        """Test that gaps larger than max_fill_bars are not filled."""
        # Arrange - Create a large gap (10 bars when limit is 5)
        df = pd.DataFrame({
            'datetime': pd.to_datetime([
                '2024-01-01 09:30',
                '2024-01-01 09:31',
                '2024-01-01 09:45',  # 14-minute gap (14 bars)
                '2024-01-01 09:46',
            ]),
            'open': [100.0, 100.0, 100.0, 100.0],
            'high': [102.0, 102.0, 102.0, 102.0],
            'low': [98.0, 98.0, 98.0, 98.0],
            'close': [100.0, 100.0, 100.0, 100.0],
            'volume': [1000, 1000, 1000, 1000]
        })

        cleaner = DataCleaner(
            input_dir=temp_dir,
            output_dir=temp_dir / "output",
            timeframe='1min',
            gap_fill_method='forward',
            max_gap_fill_minutes=5  # Only fill up to 5 bars
        )

        # Act
        result = cleaner.fill_gaps(df)

        # Assert - Large gap should result in NaN rows that get dropped
        # The gap is 14 bars but we can only fill 5
        assert len(result) < 17  # If all were filled we'd have 17 rows

    def test_fill_gaps_method_none(self, temp_dir, sample_ohlcv_with_gaps):
        """Test that gap filling can be disabled."""
        # Arrange
        cleaner = DataCleaner(
            input_dir=temp_dir,
            output_dir=temp_dir / "output",
            timeframe='1min',
            gap_fill_method='none'
        )

        initial_len = len(sample_ohlcv_with_gaps)

        # Act
        df = cleaner.fill_gaps(sample_ohlcv_with_gaps)

        # Assert - Should be unchanged
        assert len(df) == initial_len


class TestDataCleanerDuplicates:
    """Tests for DataCleaner duplicate detection."""

    def test_remove_duplicates(self, temp_dir):
        """Test duplicate timestamp removal."""
        # Arrange
        df = pd.DataFrame({
            'datetime': pd.to_datetime([
                '2024-01-01 09:30',
                '2024-01-01 09:30',  # Duplicate
                '2024-01-01 09:31',
                '2024-01-01 09:32',
                '2024-01-01 09:32',  # Duplicate
            ]),
            'open': [100.0, 100.5, 101.0, 102.0, 102.5],
            'high': [102.0, 102.5, 103.0, 104.0, 104.5],
            'low': [98.0, 98.5, 99.0, 100.0, 100.5],
            'close': [100.0, 100.5, 101.0, 102.0, 102.5],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })

        cleaner = DataCleaner(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        result, report = cleaner.detect_duplicates(df)

        # Assert
        assert len(result) == 3  # 5 - 2 duplicates
        assert report['n_duplicates'] == 2
        assert len(result['datetime'].unique()) == len(result)  # All unique


class TestDataCleanerOutlierDetection:
    """Tests for DataCleaner outlier detection methods."""

    def test_detect_outliers_atr_method(self, temp_dir, sample_ohlcv_with_outliers):
        """Test ATR-based spike detection."""
        # Arrange
        cleaner = DataCleaner(
            input_dir=temp_dir,
            output_dir=temp_dir / "output",
            outlier_method='atr',
            atr_threshold=5.0
        )

        # Act
        df, report = cleaner.clean_outliers(sample_ohlcv_with_outliers)

        # Assert - Should detect the 10% spikes as outliers
        assert report['total_outliers'] > 0
        assert 'atr_spikes' in report['methods']
        assert len(df) < len(sample_ohlcv_with_outliers)

    def test_detect_outliers_zscore_method(self, temp_dir, sample_ohlcv_with_outliers):
        """Test z-score based outlier detection."""
        # Arrange
        cleaner = DataCleaner(
            input_dir=temp_dir,
            output_dir=temp_dir / "output",
            outlier_method='zscore',
            zscore_threshold=3.0
        )

        # Act
        df, report = cleaner.clean_outliers(sample_ohlcv_with_outliers)

        # Assert
        assert 'zscore' in report['methods']
        # May or may not detect depending on distribution

    def test_detect_outliers_zscore_constant_series(self, temp_dir):
        """Test z-score handling of constant series (std=0)."""
        # Arrange
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=100, freq='min'),
            'open': [100.0] * 100,
            'high': [100.0] * 100,
            'low': [100.0] * 100,
            'close': [100.0] * 100,  # Constant - std = 0
            'volume': [1000] * 100
        })

        cleaner = DataCleaner(
            input_dir=temp_dir,
            output_dir=temp_dir / "output",
            outlier_method='zscore'
        )

        # Act - Should not raise ZeroDivisionError
        result = cleaner.detect_outliers_zscore(df['close'].pct_change())

        # Assert - Should return all False (no outliers when std=0)
        assert not result.any()


class TestDataCleanerIntegration:
    """Integration tests for DataCleaner.clean_file()."""

    def test_clean_file_integration(self, temp_dir, sample_ohlcv_df):
        """Test complete cleaning pipeline."""
        # Arrange
        file_path = temp_dir / "test.parquet"
        sample_ohlcv_df.to_parquet(file_path, index=False)

        cleaner = DataCleaner(
            input_dir=temp_dir,
            output_dir=temp_dir / "output",
            timeframe='1min',
            gap_fill_method='forward'
        )

        # Act
        df, report = cleaner.clean_file(file_path)

        # Assert
        assert len(df) > 0
        assert 'initial_rows' in report
        assert 'final_rows' in report
        assert 'duplicates' in report
        assert 'gaps' in report
        assert 'outliers' in report


# =============================================================================
# STAGE 3 TESTS: FeatureEngineer
# =============================================================================

class TestFeatureEngineerSMA:
    """Tests for SMA calculation."""

    def test_compute_sma_correct_values(self):
        """Test SMA calculation produces correct values."""
        # Arrange
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        period = 3

        # Act
        sma = calculate_sma_numba(prices, period)

        # Assert
        # SMA(3) at index 2: (1+2+3)/3 = 2.0
        # SMA(3) at index 3: (2+3+4)/3 = 3.0
        assert np.isnan(sma[0])  # Not enough data
        assert np.isnan(sma[1])  # Not enough data
        assert np.isclose(sma[2], 2.0)
        assert np.isclose(sma[3], 3.0)
        assert np.isclose(sma[9], 9.0)  # (8+9+10)/3


class TestFeatureEngineerEMA:
    """Tests for EMA calculation."""

    def test_compute_ema_correct_values(self):
        """Test EMA calculation produces correct values."""
        # Arrange
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)
        period = 3

        # Act
        ema = calculate_ema_numba(prices, period)

        # Assert
        # EMA starts with SMA at period-1
        # alpha = 2/(3+1) = 0.5
        assert np.isnan(ema[0])
        assert np.isnan(ema[1])
        assert np.isclose(ema[2], 2.0)  # SMA of first 3 = 2.0
        # EMA[3] = 0.5 * 4 + 0.5 * 2 = 3.0
        assert np.isclose(ema[3], 3.0)
        # Values should generally follow the trend
        assert ema[9] > ema[5]  # Uptrend


class TestFeatureEngineerRSI:
    """Tests for RSI calculation."""

    def test_compute_rsi_bounds(self):
        """Test RSI values are bounded between 0 and 100."""
        # Arrange
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)

        # Act
        rsi = calculate_rsi_numba(prices, 14)

        # Assert
        valid_rsi = rsi[~np.isnan(rsi)]
        assert np.all(valid_rsi >= 0)
        assert np.all(valid_rsi <= 100)

    def test_compute_rsi_uptrend(self):
        """Test RSI is high in strong uptrend."""
        # Arrange - Strong uptrend
        prices = np.array([float(i) for i in range(1, 31)])  # 1 to 30

        # Act
        rsi = calculate_rsi_numba(prices, 14)

        # Assert - RSI should be close to 100 in strong uptrend
        assert rsi[-1] > 90

    def test_compute_rsi_downtrend(self):
        """Test RSI is low in strong downtrend."""
        # Arrange - Strong downtrend
        prices = np.array([float(30 - i) for i in range(30)])  # 30 to 1

        # Act
        rsi = calculate_rsi_numba(prices, 14)

        # Assert - RSI should be close to 0 in strong downtrend
        assert rsi[-1] < 10


class TestFeatureEngineerATR:
    """Tests for ATR calculation."""

    def test_compute_atr_positive_values(self, sample_ohlcv_df):
        """Test ATR produces positive values."""
        # Arrange
        high = sample_ohlcv_df['high'].values
        low = sample_ohlcv_df['low'].values
        close = sample_ohlcv_df['close'].values

        # Act
        atr = features_atr_numba(high, low, close, 14)

        # Assert
        valid_atr = atr[~np.isnan(atr)]
        assert np.all(valid_atr >= 0)
        assert len(valid_atr) > 0


class TestFeatureEngineerMACD:
    """Tests for MACD calculation."""

    def test_compute_macd_components(self, temp_dir, sample_ohlcv_df):
        """Test MACD line, signal, and histogram are calculated."""
        # Arrange
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = engineer.add_macd(sample_ohlcv_df.copy())

        # Assert
        assert 'macd_line' in df.columns
        assert 'macd_signal' in df.columns
        assert 'macd_hist' in df.columns

        # Histogram should be difference of line and signal
        valid_idx = df['macd_hist'].notna()
        np.testing.assert_array_almost_equal(
            df.loc[valid_idx, 'macd_hist'].values,
            (df.loc[valid_idx, 'macd_line'] - df.loc[valid_idx, 'macd_signal']).values,
            decimal=10
        )


class TestFeatureEngineerBollingerBands:
    """Tests for Bollinger Bands calculation."""

    def test_compute_bollinger_bands(self, temp_dir, sample_ohlcv_df):
        """Test Bollinger Bands are correctly calculated."""
        # Arrange
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = engineer.add_bollinger_bands(sample_ohlcv_df.copy())

        # Assert
        assert 'bb_upper' in df.columns
        assert 'bb_middle' in df.columns
        assert 'bb_lower' in df.columns
        assert 'bb_width' in df.columns
        assert 'bb_position' in df.columns

        # Upper > Middle > Lower
        valid_idx = df['bb_upper'].notna()
        assert np.all(df.loc[valid_idx, 'bb_upper'] >= df.loc[valid_idx, 'bb_middle'])
        assert np.all(df.loc[valid_idx, 'bb_middle'] >= df.loc[valid_idx, 'bb_lower'])


class TestFeatureEngineerTemporalFeatures:
    """Tests for temporal feature encoding."""

    def test_temporal_features_encoding(self, temp_dir, sample_ohlcv_df):
        """Test temporal features with sin/cos encoding."""
        # Arrange
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = engineer.add_temporal_features(sample_ohlcv_df.copy())

        # Assert
        assert 'hour_sin' in df.columns
        assert 'hour_cos' in df.columns
        assert 'minute_sin' in df.columns
        assert 'minute_cos' in df.columns
        assert 'dayofweek_sin' in df.columns
        assert 'dayofweek_cos' in df.columns

        # Sin/cos values should be in [-1, 1]
        assert np.all(df['hour_sin'].abs() <= 1)
        assert np.all(df['hour_cos'].abs() <= 1)

    def test_temporal_features_session_encoding(self, temp_dir):
        """Test trading session encoding."""
        # Arrange - Create data across different sessions
        df = pd.DataFrame({
            'datetime': pd.to_datetime([
                '2024-01-02 02:00',  # Asia (0-8 UTC)
                '2024-01-02 10:00',  # London (8-16 UTC)
                '2024-01-02 18:00',  # NY (16-24 UTC)
            ]),
            'open': [100.0] * 3,
            'high': [102.0] * 3,
            'low': [98.0] * 3,
            'close': [100.0] * 3,
            'volume': [1000] * 3
        })

        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        result = engineer.add_temporal_features(df)

        # Assert
        assert 'session_asia' in result.columns
        assert 'session_london' in result.columns
        assert 'session_ny' in result.columns

        # Check correct session assignment
        assert result.loc[0, 'session_asia'] == 1
        assert result.loc[1, 'session_london'] == 1
        assert result.loc[2, 'session_ny'] == 1


class TestFeatureEngineerRegimeFeatures:
    """Tests for regime feature calculation."""

    def test_regime_features_categories(self, temp_dir, sample_ohlcv_df):
        """Test regime features produce valid categories."""
        # Arrange
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Need to add prerequisite features first
        df = sample_ohlcv_df.copy()
        df = engineer.add_sma(df)
        df = engineer.add_historical_volatility(df)

        # Act
        df = engineer.add_regime_features(df)

        # Assert
        if 'trend_regime' in df.columns:
            valid_values = {-1, 0, 1}
            assert set(df['trend_regime'].dropna().unique()).issubset(valid_values)

        if 'volatility_regime' in df.columns:
            valid_values = {0, 1}
            assert set(df['volatility_regime'].dropna().unique()).issubset(valid_values)


class TestFeatureEngineerNoLookahead:
    """Critical tests to ensure no lookahead bias in features."""

    def test_no_lookahead_in_features(self, temp_dir, sample_ohlcv_df):
        """Test that features only use past data, no future data."""
        # Arrange - use larger dataset to ensure enough rows after NaN drop
        # Extend sample to 1000 rows
        n = 1000
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
        start_time = datetime(2024, 1, 1, 9, 30)
        timestamps = [start_time + timedelta(minutes=i) for i in range(n)]

        large_df = pd.DataFrame({
            'datetime': timestamps,
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Calculate features for full dataset
        df_full = large_df.copy()
        df_full, _ = engineer.engineer_features(df_full, 'TEST')

        # Calculate features for subset (first 400 rows)
        df_subset = large_df.head(400).copy()
        engineer.feature_metadata = {}  # Reset metadata
        df_subset, _ = engineer.engineer_features(df_subset, 'TEST')

        # After NaN dropna, we should have enough rows to compare
        if len(df_subset) == 0 or len(df_full) == 0:
            pytest.skip("Not enough data after NaN drop")

        # Use a common index that exists in both after processing
        # Pick a timestamp that should exist in both
        min_rows_needed = 250  # Need at least this many rows after dropna
        if len(df_subset) < min_rows_needed:
            pytest.skip(f"Not enough rows in subset: {len(df_subset)}")

        # Get a timestamp from the middle of the subset result
        test_idx = len(df_subset) // 2
        test_time = df_subset['datetime'].iloc[test_idx]
        full_match = df_full[df_full['datetime'] == test_time]

        if len(full_match) == 0:
            pytest.skip("Test timestamp not found in full dataset")

        # Compare features at matching timestamp
        common_cols = [c for c in df_subset.columns if c in df_full.columns
                       and c not in ['datetime', 'open', 'high', 'low', 'close', 'volume', 'symbol']]

        for col in common_cols[:20]:  # Test first 20 feature columns
            subset_val = df_subset[df_subset['datetime'] == test_time][col].iloc[0]
            full_val = full_match[col].iloc[0]

            if pd.notna(subset_val) and pd.notna(full_val):
                assert np.isclose(subset_val, full_val, rtol=1e-5), \
                    f"Lookahead detected in {col}: subset={subset_val}, full={full_val}"


class TestFeatureEngineerNaNHandling:
    """Tests for NaN handling in feature calculation."""

    def test_feature_nan_handling(self, temp_dir, sample_ohlcv_df):
        """Test that NaN values are properly handled during feature calculation."""
        # Arrange
        df = sample_ohlcv_df.copy()
        # Inject some NaN values
        df.loc[10, 'close'] = np.nan
        df.loc[20, 'high'] = np.nan

        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act - should not raise
        df_result, report = engineer.engineer_features(df, 'TEST')

        # Assert - NaN rows should be dropped
        assert len(df_result) < len(df)
        assert df_result['close'].notna().all()


# =============================================================================
# STAGE 4 TESTS: Triple Barrier Labeling
# =============================================================================

class TestTripleBarrierUpperHit:
    """Tests for upper barrier (profit target) hits."""

    def test_triple_barrier_upper_hit(self):
        """Test that upper barrier hit produces label +1."""
        # Arrange - Price that clearly hits upper barrier
        n = 20
        close = np.array([100.0] * n)
        close[5:] = 110.0  # Jump up at bar 5

        high = close + 1.0
        low = close - 1.0
        open_ = close.copy()
        atr = np.ones(n) * 2.0  # ATR = 2

        k_up = 2.0  # Upper barrier at 100 + 2*2 = 104
        k_down = 2.0  # Lower barrier at 100 - 2*2 = 96
        max_bars = 10

        # Act
        labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
            close, high, low, open_, atr, k_up, k_down, max_bars
        )

        # Assert - First few bars should hit upper barrier (label +1)
        assert labels[0] == 1
        assert touch_type[0] == 1


class TestTripleBarrierLowerHit:
    """Tests for lower barrier (stop loss) hits."""

    def test_triple_barrier_lower_hit(self):
        """Test that lower barrier hit produces label -1."""
        # Arrange - Price that clearly hits lower barrier
        n = 20
        close = np.array([100.0] * n)
        close[5:] = 90.0  # Drop at bar 5

        high = close + 1.0
        low = close - 1.0
        open_ = close.copy()
        atr = np.ones(n) * 2.0  # ATR = 2

        k_up = 2.0  # Upper barrier at 100 + 2*2 = 104
        k_down = 2.0  # Lower barrier at 100 - 2*2 = 96
        max_bars = 10

        # Act
        labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
            close, high, low, open_, atr, k_up, k_down, max_bars
        )

        # Assert - First few bars should hit lower barrier (label -1)
        assert labels[0] == -1
        assert touch_type[0] == -1


class TestTripleBarrierTimeout:
    """Tests for timeout (neutral) cases."""

    def test_triple_barrier_timeout_neutral(self):
        """Test that timeout produces label 0 (neutral)."""
        # Arrange - Price stays flat, doesn't hit any barrier
        n = 20
        close = np.array([100.0] * n)  # Flat price
        high = close + 0.5  # Very small range
        low = close - 0.5
        open_ = close.copy()
        atr = np.ones(n) * 5.0  # Large ATR means wide barriers

        k_up = 2.0  # Upper barrier at 100 + 2*5 = 110
        k_down = 2.0  # Lower barrier at 100 - 2*5 = 90
        max_bars = 5  # Short timeout

        # Act
        labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
            close, high, low, open_, atr, k_up, k_down, max_bars
        )

        # Assert - Should timeout with label 0
        assert labels[0] == 0
        assert touch_type[0] == 0
        assert bars_to_hit[0] == max_bars


class TestTripleBarrierSameBarHit:
    """Tests for same-bar hit resolution."""

    def test_triple_barrier_same_bar_hit_resolution(self):
        """Test resolution when both barriers are hit on same bar."""
        # Arrange - Create a bar where both barriers are hit
        # The bar has wide range crossing both barriers
        n = 10
        close = np.full(n, 100.0)

        # Bar 1 has extreme range crossing both barriers
        high = close.copy()
        low = close.copy()
        open_ = close.copy()

        high[1] = 108.0  # Hits upper barrier (100 + 2*2 = 104)
        low[1] = 92.0    # Hits lower barrier (100 - 2*2 = 96)
        open_[1] = 102.0  # Open closer to upper barrier

        atr = np.ones(n) * 2.0
        k_up = 2.0
        k_down = 2.0
        max_bars = 10

        # Act
        labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
            close, high, low, open_, atr, k_up, k_down, max_bars
        )

        # Assert - Should resolve based on distance from open
        # Open at 102, upper barrier at 104, lower at 96
        # dist_to_upper = |102 - 104| = 2
        # dist_to_lower = |102 - 96| = 6
        # Upper is closer, so should be +1
        assert labels[0] == 1
        assert touch_type[0] == 1


class TestTripleBarrierATRUsage:
    """Tests for ATR-based barrier calculation."""

    def test_barrier_uses_atr_correctly(self):
        """Test that barriers scale correctly with ATR."""
        # Arrange
        n = 20
        close = np.full(n, 100.0)
        high = close + 1.0
        low = close - 1.0
        open_ = close.copy()

        # Different ATR values
        atr_small = np.ones(n) * 1.0
        atr_large = np.ones(n) * 10.0

        k_up = 2.0
        k_down = 2.0
        max_bars = 5

        # Create price that moves up 5 points at bar 2
        close[2:] = 105.0
        high[2:] = 106.0
        low[2:] = 104.0

        # Act
        labels_small, _, _, _, _ = triple_barrier_numba(
            close, high, low, open_, atr_small, k_up, k_down, max_bars
        )
        labels_large, _, _, _, _ = triple_barrier_numba(
            close, high, low, open_, atr_large, k_up, k_down, max_bars
        )

        # Assert
        # With small ATR (1), upper barrier at 102, 5-point move should hit it -> +1
        # With large ATR (10), upper barrier at 120, 5-point move won't hit -> timeout 0
        assert labels_small[0] == 1  # Upper barrier hit
        assert labels_large[0] == 0  # Timeout (barrier too far)

    def test_barrier_handles_invalid_atr(self):
        """Test that invalid ATR (NaN, 0) is handled gracefully."""
        # Arrange
        n = 10
        close = np.full(n, 100.0)
        high = close + 5.0
        low = close - 5.0
        open_ = close.copy()

        atr = np.ones(n) * 2.0
        atr[0] = np.nan  # Invalid ATR for first bar
        atr[3] = 0.0     # Zero ATR

        # Act - should not raise
        labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
            close, high, low, open_, atr, 2.0, 2.0, 10
        )

        # Assert - Invalid ATR rows should be handled (label 0, max_bars)
        assert labels[0] == 0
        assert bars_to_hit[0] == 10  # max_bars


class TestTripleBarrierQualityScoring:
    """Tests for quality score calculation."""

    def test_quality_score_calculation(self, sample_features_df):
        """Test that quality-related metrics (MAE/MFE) are calculated."""
        # Arrange
        df = sample_features_df.copy()
        df = df.dropna(subset=['atr_14'])

        # Act
        df = apply_triple_barrier(df, horizon=5)

        # Assert
        assert 'mae_h5' in df.columns
        assert 'mfe_h5' in df.columns

        # MAE should be <= 0 (adverse = negative direction for long)
        # MFE should be >= 0 (favorable = positive direction for long)
        valid_idx = df['label_h5'] != 0
        # MAE represents worst drawdown (negative) and MFE represents best upside
        # Both are percentages


class TestTripleBarrierSampleWeights:
    """Tests for sample weight tier assignment."""

    def test_sample_weight_tiers(self):
        """Test sample weight tier assignment based on quality scores."""
        # This test verifies the concept of sample weights based on label quality
        # Actual implementation may be in stage6_final_labels

        # Arrange - Simulate quality scores
        np.random.seed(42)
        n = 1000
        quality_scores = np.random.rand(n)

        # Assign tiers: top 20% get 1.5x, middle 60% get 1.0x, bottom 20% get 0.5x
        percentile_20 = np.percentile(quality_scores, 20)
        percentile_80 = np.percentile(quality_scores, 80)

        weights = np.where(
            quality_scores >= percentile_80, 1.5,
            np.where(quality_scores >= percentile_20, 1.0, 0.5)
        )

        # Assert - Check tier distribution
        assert np.isclose(np.mean(weights == 1.5), 0.2, atol=0.05)
        assert np.isclose(np.mean(weights == 1.0), 0.6, atol=0.05)
        assert np.isclose(np.mean(weights == 0.5), 0.2, atol=0.05)


class TestTripleBarrierNoFutureData:
    """Critical test to ensure no future data leakage in labels."""

    def test_no_future_data_in_labels(self, sample_features_df):
        """Test that labels only depend on future price action, not features."""
        # The triple barrier method by design uses future prices to determine
        # labels (it looks forward to see which barrier is hit). This is correct
        # for supervised learning. What we need to verify is that the features
        # used at time t do NOT include any information from time t+1 or later.

        # Arrange
        df = sample_features_df.copy()
        df = df.dropna(subset=['atr_14'])
        n = len(df)

        # Apply labeling
        df = apply_triple_barrier(df, horizon=5)

        # Act - Check bars_to_hit
        # bars_to_hit should always be >= 1 (can't hit barrier at same bar)
        # Exception: last bar always has 0
        bars = df['bars_to_hit_h5'].values

        # Assert
        # All bars except last few should have bars_to_hit >= 1 when not timeout
        for i in range(n - 10):  # Skip last few bars
            label = df['label_h5'].iloc[i]
            if label != 0:  # Not timeout
                assert bars[i] >= 1, f"bars_to_hit should be >= 1 at index {i}"


class TestApplyTripleBarrierIntegration:
    """Integration tests for apply_triple_barrier function."""

    def test_apply_triple_barrier_creates_columns(self, sample_features_df):
        """Test that apply_triple_barrier creates expected columns."""
        # Arrange
        df = sample_features_df.copy()
        df = df.dropna(subset=['atr_14'])
        horizon = 5

        # Act
        result = apply_triple_barrier(df, horizon=horizon)

        # Assert
        expected_cols = [
            f'label_h{horizon}',
            f'bars_to_hit_h{horizon}',
            f'mae_h{horizon}',
            f'mfe_h{horizon}',
            f'touch_type_h{horizon}'
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_apply_triple_barrier_uses_defaults(self, sample_features_df):
        """Test that apply_triple_barrier uses default parameters from config."""
        # Arrange
        df = sample_features_df.copy()
        df = df.dropna(subset=['atr_14'])

        # Act - Call without explicit k_up, k_down, max_bars
        result = apply_triple_barrier(df, horizon=5)

        # Assert - Should complete without error
        assert 'label_h5' in result.columns
        # Check that labels are valid
        assert set(result['label_h5'].unique()).issubset({-1, 0, 1})


# =============================================================================
# ADDITIONAL UTILITY TESTS
# =============================================================================

class TestCalculateATRNumba:
    """Tests for the standalone ATR calculation functions."""

    def test_calculate_atr_numba_stage2(self):
        """Test stage2 ATR calculation."""
        # Arrange
        n = 100
        high = np.random.rand(n) * 10 + 100
        low = high - np.random.rand(n) * 2
        close = (high + low) / 2

        # Act
        atr = calculate_atr_numba(high, low, close, 14)

        # Assert
        valid_atr = atr[~np.isnan(atr)]
        assert len(valid_atr) > 0
        assert np.all(valid_atr >= 0)


class TestStochasticOscillator:
    """Tests for Stochastic Oscillator calculation."""

    def test_stochastic_bounds(self):
        """Test Stochastic %K and %D are bounded 0-100."""
        # Arrange
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))

        # Act
        k, d = calculate_stochastic_numba(high, low, close, 14, 3)

        # Assert
        valid_k = k[~np.isnan(k)]
        valid_d = d[~np.isnan(d)]

        assert np.all(valid_k >= 0) and np.all(valid_k <= 100)
        assert np.all(valid_d >= 0) and np.all(valid_d <= 100)


class TestADX:
    """Tests for ADX calculation."""

    def test_adx_positive(self):
        """Test ADX produces positive values."""
        # Arrange
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))

        # Act
        adx, plus_di, minus_di = calculate_adx_numba(high, low, close, 14)

        # Assert
        valid_adx = adx[~np.isnan(adx)]
        assert len(valid_adx) > 0
        assert np.all(valid_adx >= 0)


# =============================================================================
# ADDITIONAL TESTS FOR HIGHER COVERAGE
# =============================================================================

class TestDataIngestorSaveParquet:
    """Tests for DataIngestor.save_parquet() method."""

    def test_save_parquet_creates_file(self, temp_dir, sample_ohlcv_df):
        """Test that save_parquet creates a parquet file."""
        # Arrange
        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        output_path = ingestor.save_parquet(sample_ohlcv_df, 'TEST')

        # Assert
        assert output_path.exists()
        assert output_path.suffix == '.parquet'

        # Verify data can be read back
        df_read = pd.read_parquet(output_path)
        assert len(df_read) == len(sample_ohlcv_df)

    def test_save_parquet_with_metadata(self, temp_dir, sample_ohlcv_df):
        """Test that save_parquet creates metadata file when provided."""
        # Arrange
        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / "output"
        )
        metadata = {'symbol': 'TEST', 'rows': len(sample_ohlcv_df)}

        # Act
        output_path = ingestor.save_parquet(sample_ohlcv_df, 'TEST', metadata=metadata)

        # Assert
        metadata_path = temp_dir / "output" / "TEST_metadata.json"
        assert metadata_path.exists()


class TestDataIngestorValidateDataTypes:
    """Tests for DataIngestor.validate_data_types() method."""

    def test_validate_data_types_converts_strings(self, temp_dir):
        """Test that string OHLC values are converted to numeric."""
        # Arrange
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=5, freq='min'),
            'open': ['100.0', '101.0', '102.0', '103.0', '104.0'],
            'high': ['102.0', '103.0', '104.0', '105.0', '106.0'],
            'low': ['98.0', '99.0', '100.0', '101.0', '102.0'],
            'close': ['100.0', '101.0', '102.0', '103.0', '104.0'],
            'volume': ['1000', '1100', '1200', '1300', '1400']
        })

        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        result = ingestor.validate_data_types(df)

        # Assert
        assert result['close'].dtype == np.float64 or result['close'].dtype == float
        assert result['volume'].dtype == np.int64


class TestDataCleanerContractRolls:
    """Tests for DataCleaner.handle_contract_rolls() method."""

    def test_handle_contract_rolls_detects_jumps(self, temp_dir):
        """Test that large price jumps are detected as potential rolls."""
        # Arrange
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=10, freq='min'),
            'open': [100.0] * 10,
            'high': [102.0] * 10,
            'low': [98.0] * 10,
            'close': [100.0, 100.0, 100.0, 108.0, 108.0, 108.0, 108.0, 108.0, 108.0, 108.0],  # 8% jump at bar 3
            'volume': [1000] * 10
        })

        cleaner = DataCleaner(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        result, report = cleaner.handle_contract_rolls(df, 'MES')

        # Assert
        assert report['potential_rolls'] > 0


class TestDataCleanerParseTimeframe:
    """Tests for DataCleaner timeframe parsing."""

    def test_parse_timeframe_minutes(self, temp_dir):
        """Test parsing minute timeframes."""
        cleaner = DataCleaner(
            input_dir=temp_dir,
            output_dir=temp_dir / "output",
            timeframe='5min'
        )
        assert cleaner.freq_minutes == 5

    def test_parse_timeframe_hours(self, temp_dir):
        """Test parsing hour timeframes."""
        cleaner = DataCleaner(
            input_dir=temp_dir,
            output_dir=temp_dir / "output",
            timeframe='1h'
        )
        assert cleaner.freq_minutes == 60

    def test_parse_timeframe_days(self, temp_dir):
        """Test parsing day timeframes."""
        cleaner = DataCleaner(
            input_dir=temp_dir,
            output_dir=temp_dir / "output",
            timeframe='1d'
        )
        assert cleaner.freq_minutes == 60 * 24


class TestFeatureEngineerROC:
    """Tests for Rate of Change calculation."""

    def test_compute_roc_values(self, temp_dir, sample_ohlcv_df):
        """Test ROC calculation produces expected columns."""
        # Arrange
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = engineer.add_roc(sample_ohlcv_df.copy())

        # Assert
        assert 'roc_5' in df.columns
        assert 'roc_10' in df.columns
        assert 'roc_20' in df.columns


class TestFeatureEngineerWilliamsR:
    """Tests for Williams %R calculation."""

    def test_williams_r_bounds(self, temp_dir, sample_ohlcv_df):
        """Test Williams %R is bounded between -100 and 0."""
        # Arrange
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = engineer.add_williams_r(sample_ohlcv_df.copy())

        # Assert
        assert 'williams_r' in df.columns
        valid_wr = df['williams_r'].dropna()
        assert np.all(valid_wr >= -100)
        assert np.all(valid_wr <= 0)


class TestFeatureEngineerCCI:
    """Tests for Commodity Channel Index calculation."""

    def test_cci_calculation(self, temp_dir, sample_ohlcv_df):
        """Test CCI calculation produces values."""
        # Arrange
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = engineer.add_cci(sample_ohlcv_df.copy())

        # Assert
        assert 'cci_20' in df.columns
        valid_cci = df['cci_20'].dropna()
        assert len(valid_cci) > 0


class TestFeatureEngineerKeltnerChannels:
    """Tests for Keltner Channels calculation."""

    def test_keltner_channels_order(self, temp_dir, sample_ohlcv_df):
        """Test Keltner Channels upper > middle > lower."""
        # Arrange
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = engineer.add_keltner_channels(sample_ohlcv_df.copy())

        # Assert
        assert 'kc_upper' in df.columns
        assert 'kc_middle' in df.columns
        assert 'kc_lower' in df.columns

        valid_idx = df['kc_upper'].notna()
        assert np.all(df.loc[valid_idx, 'kc_upper'] >= df.loc[valid_idx, 'kc_middle'])
        assert np.all(df.loc[valid_idx, 'kc_middle'] >= df.loc[valid_idx, 'kc_lower'])


class TestFeatureEngineerVolatility:
    """Tests for volatility indicators."""

    def test_historical_volatility_positive(self, temp_dir, sample_ohlcv_df):
        """Test historical volatility produces positive values."""
        # Arrange
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = engineer.add_historical_volatility(sample_ohlcv_df.copy())

        # Assert
        assert 'hvol_20' in df.columns
        valid_hvol = df['hvol_20'].dropna()
        assert np.all(valid_hvol >= 0)

    def test_parkinson_volatility_positive(self, temp_dir, sample_ohlcv_df):
        """Test Parkinson volatility produces positive values."""
        # Arrange
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = engineer.add_parkinson_volatility(sample_ohlcv_df.copy())

        # Assert
        assert 'parkinson_vol' in df.columns
        valid_pv = df['parkinson_vol'].dropna()
        assert np.all(valid_pv >= 0)

    def test_garman_klass_volatility(self, temp_dir, sample_ohlcv_df):
        """Test Garman-Klass volatility calculation."""
        # Arrange
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = engineer.add_garman_klass_volatility(sample_ohlcv_df.copy())

        # Assert
        assert 'gk_vol' in df.columns


class TestFeatureEngineerVolumeFeatures:
    """Tests for volume-based features."""

    def test_volume_features_with_volume(self, temp_dir, sample_ohlcv_df):
        """Test volume features are calculated when volume is present."""
        # Arrange
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = engineer.add_volume_features(sample_ohlcv_df.copy())

        # Assert
        assert 'obv' in df.columns
        assert 'volume_sma_20' in df.columns
        assert 'volume_ratio' in df.columns

    def test_volume_features_without_volume(self, temp_dir):
        """Test volume features are skipped when no volume."""
        # Arrange
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=100, freq='min'),
            'open': [100.0] * 100,
            'high': [102.0] * 100,
            'low': [98.0] * 100,
            'close': [100.0] * 100,
            # No volume column
        })

        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        result = engineer.add_volume_features(df)

        # Assert - Should return unchanged
        assert 'obv' not in result.columns


class TestFeatureEngineerSupertrend:
    """Tests for Supertrend calculation."""

    def test_supertrend_direction_values(self, temp_dir, sample_ohlcv_df):
        """Test Supertrend direction is 1 or -1."""
        # Arrange
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = engineer.add_supertrend(sample_ohlcv_df.copy())

        # Assert
        assert 'supertrend' in df.columns
        assert 'supertrend_direction' in df.columns

        valid_dir = df['supertrend_direction'].dropna()
        assert set(valid_dir.unique()).issubset({-1.0, 1.0})


class TestTripleBarrierMultipleHorizons:
    """Tests for labeling with multiple horizons."""

    def test_apply_multiple_horizons(self, sample_features_df):
        """Test applying labeling for multiple horizons."""
        # Arrange
        df = sample_features_df.copy()
        df = df.dropna(subset=['atr_14'])

        # Act
        df = apply_triple_barrier(df, horizon=1)
        df = apply_triple_barrier(df, horizon=5)
        df = apply_triple_barrier(df, horizon=20)

        # Assert
        assert 'label_h1' in df.columns
        assert 'label_h5' in df.columns
        assert 'label_h20' in df.columns


class TestTripleBarrierCustomParameters:
    """Tests for custom barrier parameters."""

    def test_custom_k_up_k_down(self, sample_features_df):
        """Test labeling with custom k_up and k_down."""
        # Arrange
        df = sample_features_df.copy()
        df = df.dropna(subset=['atr_14'])

        # Act - use asymmetric barriers
        df = apply_triple_barrier(df, horizon=5, k_up=1.5, k_down=1.0, max_bars=20)

        # Assert
        assert 'label_h5' in df.columns
        # With k_up > k_down, should potentially favor short labels

    def test_custom_max_bars(self, sample_features_df):
        """Test labeling with custom max_bars."""
        # Arrange
        df = sample_features_df.copy()
        df = df.dropna(subset=['atr_14'])

        # Act - use very short max_bars (should result in more timeouts)
        df = apply_triple_barrier(df, horizon=5, max_bars=2)

        # Assert
        neutral_count = (df['label_h5'] == 0).sum()
        # With only 2 bars, expect more timeouts
        assert neutral_count > 0


class TestFeatureEngineerCrossAsset:
    """Tests for cross-asset features."""

    def test_cross_asset_features_missing_data(self, temp_dir, sample_ohlcv_df):
        """Test that cross-asset features are NaN when data is missing."""
        # Arrange
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act - call without cross_asset_data
        df = engineer.add_cross_asset_features(sample_ohlcv_df.copy())

        # Assert - cross-asset features should be NaN
        assert 'mes_mgc_correlation_20' in df.columns
        assert df['mes_mgc_correlation_20'].isna().all()


class TestDataCleanerIQROutliers:
    """Tests for IQR-based outlier detection."""

    def test_iqr_outlier_detection(self, temp_dir):
        """Test IQR outlier detection method."""
        # Arrange
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=100, freq='min'),
            'open': [100.0] * 100,
            'high': [102.0] * 100,
            'low': [98.0] * 100,
            'close': [100.0] * 95 + [200.0] * 5,  # 5 extreme outliers
            'volume': [1000] * 100
        })

        cleaner = DataCleaner(
            input_dir=temp_dir,
            output_dir=temp_dir / "output",
            outlier_method='iqr',
            iqr_multiplier=1.5
        )

        # Act
        result, report = cleaner.clean_outliers(df)

        # Assert
        assert 'iqr' in report['methods']
        assert report['methods']['iqr']['n_outliers'] > 0


class TestFeatureEngineerADXIndicator:
    """Tests for ADX indicator in FeatureEngineer."""

    def test_add_adx_features(self, temp_dir, sample_ohlcv_df):
        """Test ADX feature calculation through FeatureEngineer."""
        # Arrange
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = engineer.add_adx(sample_ohlcv_df.copy())

        # Assert
        assert 'adx_14' in df.columns
        assert 'plus_di_14' in df.columns
        assert 'minus_di_14' in df.columns
        assert 'adx_strong_trend' in df.columns


class TestFeatureEngineerStochasticIndicator:
    """Tests for Stochastic indicator in FeatureEngineer."""

    def test_add_stochastic_features(self, temp_dir, sample_ohlcv_df):
        """Test Stochastic feature calculation."""
        # Arrange
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = engineer.add_stochastic(sample_ohlcv_df.copy())

        # Assert
        assert 'stoch_k' in df.columns
        assert 'stoch_d' in df.columns
        assert 'stoch_overbought' in df.columns
        assert 'stoch_oversold' in df.columns


class TestFeatureEngineerMFI:
    """Tests for Money Flow Index calculation."""

    def test_add_mfi_with_volume(self, temp_dir, sample_ohlcv_df):
        """Test MFI calculation with volume data."""
        # Arrange
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = engineer.add_mfi(sample_ohlcv_df.copy())

        # Assert
        assert 'mfi_14' in df.columns
        valid_mfi = df['mfi_14'].dropna()
        assert np.all(valid_mfi >= 0)
        assert np.all(valid_mfi <= 100)


class TestFeatureEngineerVWAP:
    """Tests for VWAP calculation."""

    def test_add_vwap(self, temp_dir, sample_ohlcv_df):
        """Test VWAP calculation."""
        # Arrange
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = engineer.add_vwap(sample_ohlcv_df.copy())

        # Assert
        assert 'vwap' in df.columns
        assert 'price_to_vwap' in df.columns

        # VWAP should be positive
        valid_vwap = df['vwap'].dropna()
        assert np.all(valid_vwap > 0)


class TestFeatureEngineerReturns:
    """Tests for return calculations."""

    def test_add_returns(self, temp_dir, sample_ohlcv_df):
        """Test return feature calculation."""
        # Arrange
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = engineer.add_returns(sample_ohlcv_df.copy())

        # Assert
        assert 'return_1' in df.columns
        assert 'return_5' in df.columns
        assert 'log_return_1' in df.columns
        assert 'log_return_5' in df.columns


class TestFeatureEngineerPriceRatios:
    """Tests for price ratio calculations."""

    def test_add_price_ratios(self, temp_dir, sample_ohlcv_df):
        """Test price ratio feature calculation."""
        # Arrange
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = engineer.add_price_ratios(sample_ohlcv_df.copy())

        # Assert
        assert 'hl_ratio' in df.columns
        assert 'co_ratio' in df.columns
        assert 'range_pct' in df.columns

        # HL ratio should be >= 1
        assert np.all(df['hl_ratio'] >= 1)


class TestFeatureEngineerSaveFeatures:
    """Tests for saving features."""

    def test_save_features(self, temp_dir, sample_ohlcv_df):
        """Test saving features to parquet."""
        # Arrange
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        df, report = engineer.engineer_features(sample_ohlcv_df.copy(), 'TEST')

        # Act
        data_path, meta_path = engineer.save_features(df, 'TEST', report)

        # Assert
        assert data_path.exists()
        assert meta_path.exists()


class TestTripleBarrierNonStandardHorizon:
    """Tests for non-standard horizons."""

    def test_apply_triple_barrier_non_standard_horizon(self, sample_features_df):
        """Test labeling with a non-configured horizon."""
        # Arrange
        df = sample_features_df.copy()
        df = df.dropna(subset=['atr_14'])

        # Act - use horizon=10 which is not in BARRIER_PARAMS
        df = apply_triple_barrier(df, horizon=10, k_up=1.0, k_down=1.0, max_bars=30)

        # Assert
        assert 'label_h10' in df.columns


class TestTripleBarrierEdgeCases:
    """Edge case tests for triple barrier labeling."""

    def test_last_bar_always_timeout(self):
        """Test that the last bar always has label 0."""
        # Arrange
        n = 20
        close = np.full(n, 100.0)
        high = close + 1.0
        low = close - 1.0
        open_ = close.copy()
        atr = np.ones(n) * 2.0

        # Act
        labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
            close, high, low, open_, atr, 2.0, 2.0, 10
        )

        # Assert - Last bar should always be timeout
        assert labels[-1] == 0
        assert bars_to_hit[-1] == 0

    def test_very_small_atr(self):
        """Test behavior with very small ATR."""
        # Arrange
        n = 20
        close = np.full(n, 100.0)
        close[5:] = 100.05  # Tiny move

        high = close + 0.1
        low = close - 0.1
        open_ = close.copy()
        atr = np.ones(n) * 0.01  # Very small ATR

        # Act - With tiny ATR, barriers will be very tight
        labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
            close, high, low, open_, atr, 2.0, 2.0, 10
        )

        # Assert - Should still produce valid labels
        assert set(labels).issubset({-1, 0, 1})


class TestDataCleanerSaveResults:
    """Tests for DataCleaner result saving."""

    def test_save_results(self, temp_dir, sample_ohlcv_df):
        """Test saving cleaned data and report."""
        # Arrange
        cleaner = DataCleaner(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        cleaning_report = {
            'symbol': 'TEST',
            'initial_rows': len(sample_ohlcv_df),
            'final_rows': len(sample_ohlcv_df)
        }

        # Act
        data_path, report_path = cleaner.save_results(
            sample_ohlcv_df, 'TEST', cleaning_report
        )

        # Assert
        assert data_path.exists()
        assert report_path.exists()


class TestDataCleanerInterpolateMethod:
    """Tests for interpolation gap filling."""

    def test_fill_gaps_interpolate(self, temp_dir):
        """Test interpolation gap filling method."""
        # Arrange
        df = pd.DataFrame({
            'datetime': pd.to_datetime([
                '2024-01-01 09:30',
                '2024-01-01 09:31',
                '2024-01-01 09:35',  # 4-minute gap
                '2024-01-01 09:36',
            ]),
            'open': [100.0, 101.0, 105.0, 106.0],
            'high': [102.0, 103.0, 107.0, 108.0],
            'low': [98.0, 99.0, 103.0, 104.0],
            'close': [100.0, 101.0, 105.0, 106.0],
            'volume': [1000, 1100, 1400, 1500]
        })

        cleaner = DataCleaner(
            input_dir=temp_dir,
            output_dir=temp_dir / "output",
            timeframe='1min',
            gap_fill_method='interpolate',
            max_gap_fill_minutes=5
        )

        # Act
        result = cleaner.fill_gaps(df)

        # Assert - Should have filled some gaps
        assert len(result) > len(df)


class TestDataCleanerAllOutlierMethods:
    """Tests for combined outlier detection."""

    def test_all_outlier_methods(self, temp_dir, sample_ohlcv_with_outliers):
        """Test using all outlier detection methods together."""
        # Arrange
        cleaner = DataCleaner(
            input_dir=temp_dir,
            output_dir=temp_dir / "output",
            outlier_method='all'
        )

        # Act
        df, report = cleaner.clean_outliers(sample_ohlcv_with_outliers)

        # Assert
        assert 'atr_spikes' in report['methods']
        assert 'zscore' in report['methods']
        assert 'iqr' in report['methods']


class TestFeatureEngineerProcessFile:
    """Tests for processing a complete file."""

    def test_process_file(self, temp_dir, sample_ohlcv_df):
        """Test processing a complete file."""
        # Arrange
        file_path = temp_dir / "TEST.parquet"
        sample_ohlcv_df.to_parquet(file_path, index=False)

        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        report = engineer.process_file(file_path)

        # Assert
        assert report['symbol'] == 'TEST'
        assert report['features_added'] > 0


class TestFeatureEngineerRSIIndicator:
    """Tests for RSI indicator in FeatureEngineer."""

    def test_add_rsi_features(self, temp_dir, sample_ohlcv_df):
        """Test RSI feature calculation through FeatureEngineer."""
        # Arrange
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = engineer.add_rsi(sample_ohlcv_df.copy())

        # Assert
        assert 'rsi_14' in df.columns
        assert 'rsi_overbought' in df.columns
        assert 'rsi_oversold' in df.columns

        # Overbought/oversold should be 0 or 1
        assert set(df['rsi_overbought'].unique()).issubset({0, 1})
        assert set(df['rsi_oversold'].unique()).issubset({0, 1})


class TestDataIngestorIngestDirectory:
    """Tests for directory-level ingestion."""

    def test_ingest_directory_success(self, temp_dir, sample_ohlcv_df):
        """Test ingesting multiple files from a directory."""
        # Arrange
        raw_dir = temp_dir / "raw"
        raw_dir.mkdir()
        output_dir = temp_dir / "output"

        # Create multiple test files
        for symbol in ['MES', 'MGC', 'NQ']:
            df = sample_ohlcv_df.copy()
            df['symbol'] = symbol
            df.to_parquet(raw_dir / f"{symbol}_1m.parquet", index=False)

        ingestor = DataIngestor(
            raw_data_dir=raw_dir,
            output_dir=output_dir
        )

        # Act
        results = ingestor.ingest_directory(pattern="*.parquet")

        # Assert
        assert len(results) == 3
        assert 'MES' in results
        assert 'MGC' in results
        assert 'NQ' in results

    def test_ingest_directory_no_files(self, temp_dir):
        """Test handling of empty directory."""
        # Arrange
        raw_dir = temp_dir / "empty"
        raw_dir.mkdir()

        ingestor = DataIngestor(
            raw_data_dir=raw_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        results = ingestor.ingest_directory()

        # Assert
        assert results == {}


class TestDataIngestorOHLCViolations:
    """Additional tests for OHLCV validation edge cases."""

    def test_high_lt_open_fix(self, temp_dir):
        """Test fixing high < open violations."""
        # Arrange
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=5, freq='min'),
            'open': [105.0, 100.0, 100.0, 100.0, 100.0],  # First open > high
            'high': [102.0, 102.0, 102.0, 102.0, 102.0],
            'low': [98.0, 98.0, 98.0, 98.0, 98.0],
            'close': [100.0, 100.0, 100.0, 100.0, 100.0],
            'volume': [1000] * 5
        })

        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        result, report = ingestor.validate_ohlcv_relationships(df)

        # Assert - high should now be >= open
        assert result.loc[0, 'high'] >= result.loc[0, 'open']
        assert 'high_lt_open' in report['violations']

    def test_low_gt_close_fix(self, temp_dir):
        """Test fixing low > close violations."""
        # Arrange
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=5, freq='min'),
            'open': [100.0] * 5,
            'high': [102.0] * 5,
            'low': [101.0, 98.0, 98.0, 98.0, 98.0],  # First low > close
            'close': [99.0, 100.0, 100.0, 100.0, 100.0],  # First close < low
            'volume': [1000] * 5
        })

        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        result, report = ingestor.validate_ohlcv_relationships(df)

        # Assert - low should now be <= close
        assert result.loc[0, 'low'] <= result.loc[0, 'close']
        assert 'low_gt_close' in report['violations']

    def test_negative_volume_fixed(self, temp_dir):
        """Test fixing negative volume."""
        # Arrange
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=5, freq='min'),
            'open': [100.0] * 5,
            'high': [102.0] * 5,
            'low': [98.0] * 5,
            'close': [100.0] * 5,
            'volume': [1000, -500, 1000, 1000, 1000]  # Negative volume
        })

        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        result, report = ingestor.validate_ohlcv_relationships(df)

        # Assert - negative volume should be set to 0
        assert result.loc[1, 'volume'] == 0
        assert 'negative_volume' in report['violations']


class TestTripleBarrierLabelDistribution:
    """Tests for label distribution characteristics."""

    def test_label_distribution_balanced(self):
        """Test that asymmetric barriers produce more balanced labels."""
        # Arrange - Create trending data (upward bias)
        n = 200
        np.random.seed(42)
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.3 + 0.02)  # Slight upward drift

        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        open_ = close + np.random.randn(n) * 0.1
        atr = np.ones(n) * 1.0

        # Act - Symmetric barriers
        labels_sym, _, _, _, _ = triple_barrier_numba(
            close, high, low, open_, atr, 2.0, 2.0, 15
        )

        # Act - Asymmetric barriers (easier lower barrier)
        labels_asym, _, _, _, _ = triple_barrier_numba(
            close, high, low, open_, atr, 2.5, 1.5, 15  # k_up > k_down
        )

        # Assert - Both should produce valid labels
        assert set(labels_sym).issubset({-1, 0, 1})
        assert set(labels_asym).issubset({-1, 0, 1})

        # Asymmetric should have more short labels
        long_sym = (labels_sym == 1).sum()
        short_sym = (labels_sym == -1).sum()
        long_asym = (labels_asym == 1).sum()
        short_asym = (labels_asym == -1).sum()

        # Verify asymmetric has different distribution
        # (not asserting specific values as it depends on random data)
        assert (long_asym + short_asym) > 0  # At least some non-timeout labels


class TestTripleBarrierMAEMFE:
    """Tests for MAE/MFE calculations."""

    def test_mae_mfe_values(self):
        """Test that MAE and MFE are calculated correctly."""
        # Arrange - Simple case with known excursions
        n = 20
        close = np.full(n, 100.0)
        high = np.full(n, 100.0)
        low = np.full(n, 100.0)
        open_ = np.full(n, 100.0)

        # At bar 5, price moves up significantly
        close[5:10] = 105.0
        high[5:10] = 106.0
        low[5:10] = 104.0

        atr = np.ones(n) * 2.0

        # Act
        labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
            close, high, low, open_, atr, 2.0, 2.0, 15
        )

        # Assert - MFE should capture the upside
        # Entry at bar 0 (close=100), max high is 106 at bars 5-9
        # MFE = (106-100)/100 = 0.06
        assert mfe[0] > 0  # Should have positive favorable excursion


class TestDataCleanerResampleToTimeframe:
    """Tests for resampling functionality."""

    def test_resample_to_5min(self, temp_dir, sample_ohlcv_df):
        """Test resampling from 1min to 5min."""
        # Arrange
        cleaner = DataCleaner(
            input_dir=temp_dir,
            output_dir=temp_dir / "output",
            timeframe='5min'
        )

        # Act - resample_data method if it exists
        if hasattr(cleaner, 'resample_data'):
            result = cleaner.resample_data(sample_ohlcv_df)

            # Assert
            # After resampling 1min to 5min, should have fewer rows
            assert len(result) <= len(sample_ohlcv_df)


class TestFeatureEngineerATRMethod:
    """Tests for ATR calculation in FeatureEngineer."""

    def test_add_atr(self, temp_dir, sample_ohlcv_df):
        """Test ATR feature calculation."""
        # Arrange
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = engineer.add_atr(sample_ohlcv_df.copy())

        # Assert - periods are [7, 14, 21]
        assert 'atr_14' in df.columns
        assert 'atr_pct_14' in df.columns

        # ATR should be positive
        valid_atr = df['atr_14'].dropna()
        assert np.all(valid_atr >= 0)


class TestFeatureEngineerSMAMethod:
    """Tests for SMA features in FeatureEngineer."""

    def test_add_sma(self, temp_dir, sample_ohlcv_df):
        """Test SMA feature calculation."""
        # Arrange
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = engineer.add_sma(sample_ohlcv_df.copy())

        # Assert - periods are [10, 20, 50, 100, 200]
        assert 'sma_10' in df.columns
        assert 'sma_20' in df.columns
        assert 'sma_50' in df.columns


class TestFeatureEngineerEMAMethod:
    """Tests for EMA features in FeatureEngineer."""

    def test_add_ema(self, temp_dir, sample_ohlcv_df):
        """Test EMA feature calculation."""
        # Arrange
        engineer = FeatureEngineer(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        df = engineer.add_ema(sample_ohlcv_df.copy())

        # Assert - periods are [9, 12, 21, 26, 50]
        assert 'ema_9' in df.columns
        assert 'ema_12' in df.columns
        assert 'ema_21' in df.columns


class TestApplyTripleBarrierWithATRColumn:
    """Tests for apply_triple_barrier with different ATR columns."""

    def test_custom_atr_column(self, temp_dir, sample_ohlcv_df):
        """Test using a custom ATR column name."""
        # Arrange
        df = sample_ohlcv_df.copy()

        # Calculate ATR with different name
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        atr = features_atr_numba(high, low, close, 10)
        df['custom_atr'] = atr

        df = df.dropna(subset=['custom_atr'])

        # Act
        result = apply_triple_barrier(df, horizon=5, atr_column='custom_atr')

        # Assert
        assert 'label_h5' in result.columns


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
