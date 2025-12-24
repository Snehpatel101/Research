"""
Unit tests for Stage 1: Data Ingestion (DataIngestor).

Tests cover:
- Data loading from Parquet and CSV files
- Column name standardization and mapping
- OHLCV relationship validation and fixing
- Timezone handling and conversion
- Data type validation
- Directory ingestion
- Full pipeline integration

Run with: pytest tests/phase_1_tests/stages/test_stage1_ingest.py -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.phase1.stages.ingest import DataIngestor


# =============================================================================
# DATA LOADING TESTS
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
        sample_ohlcv_df.to_csv(file_path, index=False)

        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Unsupported file format"):
            ingestor.load_data(file_path)


# =============================================================================
# COLUMN STANDARDIZATION TESTS
# =============================================================================

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

    def test_standardize_columns_lowercase_already(self, temp_dir, sample_ohlcv_df):
        """Test standardization when columns are already correct."""
        # Arrange
        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        result = ingestor.standardize_columns(sample_ohlcv_df)

        # Assert
        pd.testing.assert_frame_equal(result, sample_ohlcv_df)

    def test_standardize_columns_mixed_case(self, temp_dir):
        """Test standardization with mixed case column names."""
        # Arrange
        df = pd.DataFrame({
            'DateTime': pd.date_range('2024-01-01', periods=3, freq='min'),
            'Open': [100.0, 100.5, 101.0],
            'High': [102.0, 102.5, 103.0],
            'Low': [98.0, 98.5, 99.0],
            'Close': [100.5, 101.0, 101.5],
            'Volume': [1000, 1100, 1200]
        })

        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        result = ingestor.standardize_columns(df)

        # Assert
        expected_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        assert all(col in result.columns for col in expected_cols)


# =============================================================================
# OHLCV VALIDATION TESTS
# =============================================================================

class TestDataIngestorValidateOHLCV:
    """Tests for DataIngestor.validate_ohlcv_relationships() method."""

    def test_validate_ohlcv_high_lt_low_fix(self, temp_dir):
        """Test that high < low violations are fixed by swapping."""
        # Arrange
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=5, freq='min'),
            'open': [100.0, 100.0, 100.0, 100.0, 100.0],
            'high': [99.0, 102.0, 102.0, 102.0, 102.0],
            'low': [101.0, 98.0, 98.0, 98.0, 98.0],
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
        assert result.loc[0, 'high'] == 101.0
        assert result.loc[0, 'low'] == 99.0
        assert 'high_lt_low' in report['violations']
        assert report['violations']['high_lt_low'] == 1

    def test_validate_ohlcv_negative_prices_removed(self, temp_dir):
        """Test that rows with negative prices are removed."""
        # Arrange
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=5, freq='min'),
            'open': [100.0, -100.0, 100.0, 100.0, 100.0],
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
        assert len(result) == 4
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

    def test_validate_ohlcv_close_outside_range(self, temp_dir):
        """Test that close outside [low, high] is corrected."""
        # Arrange
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=5, freq='min'),
            'open': [100.0, 100.0, 100.0, 100.0, 100.0],
            'high': [102.0, 102.0, 102.0, 102.0, 102.0],
            'low': [98.0, 98.0, 98.0, 98.0, 98.0],
            'close': [105.0, 100.0, 100.0, 100.0, 95.0],  # First too high, last too low
            'volume': [1000, 1000, 1000, 1000, 1000]
        })

        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act
        result, report = ingestor.validate_ohlcv_relationships(df)

        # Assert
        # Close should be clipped to [low, high]
        assert result.loc[0, 'close'] <= result.loc[0, 'high']
        assert result.loc[4, 'close'] >= result.loc[4, 'low']
        # Implementation uses specific violation keys:
        # 'high_lt_close' (high < close) and 'low_gt_close' (low > close)
        assert 'high_lt_close' in report['violations'] or 'low_gt_close' in report['violations']


# =============================================================================
# TIMEZONE HANDLING TESTS
# =============================================================================

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
        assert result['datetime'].dt.tz is None

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
        assert result['datetime'].iloc[0].hour == 9
        assert result['datetime'].dt.tz is None


# =============================================================================
# DATA TYPE VALIDATION TESTS
# =============================================================================

class TestDataIngestorValidateDataTypes:
    """Tests for DataIngestor data type validation."""

    def test_validate_data_types_correct(self, temp_dir, sample_ohlcv_df):
        """Test that correct data types pass validation."""
        # Arrange
        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act - should not raise
        ingestor.validate_data_types(sample_ohlcv_df)

    def test_validate_data_types_numeric_columns(self, temp_dir):
        """Test that price and volume columns are coerced to numeric."""
        # Arrange
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=5, freq='min'),
            'open': ['100.0', '100.0', '100.0', '100.0', '100.0'],  # String instead of float
            'high': [102.0] * 5,
            'low': [98.0] * 5,
            'close': [100.0] * 5,
            'volume': [1000] * 5
        })

        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        # Act - implementation coerces strings to numeric
        result = ingestor.validate_data_types(df)

        # Assert - 'open' column should now be numeric
        assert pd.api.types.is_numeric_dtype(result['open'])


# =============================================================================
# DIRECTORY INGESTION TESTS
# =============================================================================

class TestDataIngestorIngestDirectory:
    """Tests for DataIngestor.ingest_directory() method."""

    def test_ingest_directory_multiple_files(self, temp_dir, sample_ohlcv_df):
        """Test ingesting multiple files from a directory."""
        # Arrange
        raw_dir = temp_dir / "raw"
        raw_dir.mkdir()
        output_dir = temp_dir / "output"

        # Create multiple parquet files
        for symbol in ['MES', 'MGC', 'MNQ']:
            df = sample_ohlcv_df.copy()
            df['symbol'] = symbol
            df.to_parquet(raw_dir / f"{symbol}_data.parquet", index=False)

        ingestor = DataIngestor(
            raw_data_dir=raw_dir,
            output_dir=output_dir
        )

        # Act
        results = ingestor.ingest_directory()

        # Assert - results is Dict[str, Dict] mapping symbol -> metadata
        assert len(results) == 3
        assert all(s in results for s in ['MES', 'MGC', 'MNQ'])
        # Verify output files were created (filename format: {symbol}.parquet)
        assert all((output_dir / f"{s}.parquet").exists() for s in ['MES', 'MGC', 'MNQ'])

    def test_ingest_directory_empty(self, temp_dir):
        """Test ingesting from empty directory."""
        # Arrange
        raw_dir = temp_dir / "raw"
        raw_dir.mkdir()
        output_dir = temp_dir / "output"

        ingestor = DataIngestor(
            raw_data_dir=raw_dir,
            output_dir=output_dir
        )

        # Act
        results = ingestor.ingest_directory()

        # Assert
        assert len(results) == 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestDataIngestorFullPipeline:
    """Integration tests for DataIngestor full pipeline."""

    def test_ingest_file_full_pipeline(self, temp_dir, sample_ohlcv_df):
        """Test complete ingestion pipeline from file to output."""
        # Arrange
        input_file = temp_dir / "input.parquet"
        sample_ohlcv_df.to_parquet(input_file, index=False)

        output_dir = temp_dir / "output"
        output_dir.mkdir()

        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=output_dir
        )

        # Act - ingest_file returns (DataFrame, metadata)
        result_df, metadata = ingestor.ingest_file(input_file, symbol='MES')

        # Assert
        assert len(result_df) == len(sample_ohlcv_df)
        assert 'symbol' in result_df.columns
        assert result_df['symbol'].iloc[0] == 'MES'
        assert metadata['symbol'] == 'MES'

    def test_ingest_file_with_cleaning(self, temp_dir):
        """Test ingestion with data that needs cleaning."""
        # Arrange
        df = pd.DataFrame({
            'DateTime': pd.date_range('2024-01-01', periods=10, freq='min'),
            'Open': [100.0] * 10,
            'High': [102.0] * 10,
            'Low': [98.0] * 10,
            'Close': [100.0] * 10,
            'Volume': [1000] * 10
        })
        # Add a problematic row
        df.loc[5, 'High'] = 97.0  # High < Low violation

        input_file = temp_dir / "input.parquet"
        df.to_parquet(input_file, index=False)

        output_dir = temp_dir / "output"
        output_dir.mkdir()

        ingestor = DataIngestor(
            raw_data_dir=temp_dir,
            output_dir=output_dir
        )

        # Act - ingest_file returns (DataFrame, metadata)
        result_df, metadata = ingestor.ingest_file(input_file, symbol='MES')

        # Assert - Violation should be fixed
        assert (result_df['high'] >= result_df['low']).all()
        assert (result_df['high'] >= result_df['close']).all()
        assert (result_df['low'] <= result_df['close']).all()
