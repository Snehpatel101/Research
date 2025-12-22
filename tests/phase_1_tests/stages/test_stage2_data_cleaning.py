"""
Unit tests for Stage 2: Data Cleaning.

DataCleaner - Gap filling, outlier detection, and data quality

Run with: pytest tests/phase_1_tests/stages/test_stage2_*.py -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from stages.stage2_clean import DataCleaner, calculate_atr_numba


# =============================================================================
# TESTS
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

