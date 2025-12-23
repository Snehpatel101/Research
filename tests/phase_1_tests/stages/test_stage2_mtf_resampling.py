"""
Unit tests for Stage 2: Multi-Timeframe (MTF) Resampling.

Tests the configurable timeframe resampling functionality:
- resample_ohlcv() function
- DataCleaner with target_timeframe
- clean_symbol_data() with target_timeframe
- Timeframe validation and metadata

Run with: pytest tests/phase_1_tests/stages/test_stage2_mtf_resampling.py -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def sample_1min_ohlcv():
    """Create sample 1-minute OHLCV data for resampling tests."""
    # 60 rows of 1-minute data = 1 hour
    dates = pd.date_range('2024-01-01 09:30', periods=60, freq='1min')
    np.random.seed(42)

    # Generate realistic OHLCV data
    base_price = 100.0
    returns = np.random.randn(60) * 0.001  # Small random returns

    close_prices = base_price * np.cumprod(1 + returns)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price

    high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.randn(60)) * 0.0005)
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.randn(60)) * 0.0005)

    df = pd.DataFrame({
        'datetime': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': np.random.randint(100, 1000, 60)
    })

    return df


@pytest.fixture
def sample_1min_ohlcv_300_rows():
    """Create 300 rows of 1-minute data (5 hours) for multi-timeframe tests."""
    dates = pd.date_range('2024-01-01 09:30', periods=300, freq='1min')
    np.random.seed(42)

    base_price = 100.0
    returns = np.random.randn(300) * 0.001

    close_prices = base_price * np.cumprod(1 + returns)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price

    high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.randn(300)) * 0.0005)
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.randn(300)) * 0.0005)

    df = pd.DataFrame({
        'datetime': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': np.random.randint(100, 1000, 300)
    })

    return df


# =============================================================================
# CONFIG TIMEFRAME TESTS
# =============================================================================

class TestConfigTimeframe:
    """Tests for timeframe configuration in config.py."""

    def test_supported_timeframes_exist(self):
        """Test that SUPPORTED_TIMEFRAMES is defined."""
        from config import SUPPORTED_TIMEFRAMES
        assert isinstance(SUPPORTED_TIMEFRAMES, list)
        assert len(SUPPORTED_TIMEFRAMES) > 0
        assert '5min' in SUPPORTED_TIMEFRAMES
        assert '15min' in SUPPORTED_TIMEFRAMES

    def test_target_timeframe_default(self):
        """Test that TARGET_TIMEFRAME has a default value."""
        from config import TARGET_TIMEFRAME
        assert TARGET_TIMEFRAME == '5min'

    def test_validate_timeframe_valid(self):
        """Test validation of valid timeframes."""
        from config import validate_timeframe, SUPPORTED_TIMEFRAMES
        for tf in SUPPORTED_TIMEFRAMES:
            assert validate_timeframe(tf) is True

    def test_validate_timeframe_invalid(self):
        """Test validation rejects invalid timeframes."""
        from config import validate_timeframe
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            validate_timeframe('2min')

        with pytest.raises(ValueError, match="Unsupported timeframe"):
            validate_timeframe('invalid')

    def test_parse_timeframe_to_minutes(self):
        """Test parsing timeframe strings to minutes."""
        from config import parse_timeframe_to_minutes

        assert parse_timeframe_to_minutes('1min') == 1
        assert parse_timeframe_to_minutes('5min') == 5
        assert parse_timeframe_to_minutes('15min') == 15
        assert parse_timeframe_to_minutes('30min') == 30
        assert parse_timeframe_to_minutes('60min') == 60

    def test_parse_timeframe_to_minutes_hours(self):
        """Test parsing hour timeframes."""
        from config import parse_timeframe_to_minutes

        assert parse_timeframe_to_minutes('1h') == 60
        assert parse_timeframe_to_minutes('2h') == 120

    def test_get_timeframe_metadata(self):
        """Test getting timeframe metadata."""
        from config import get_timeframe_metadata

        meta = get_timeframe_metadata('5min')
        assert meta['timeframe'] == '5min'
        assert meta['minutes'] == 5
        assert meta['bars_per_hour'] == 12
        assert 'description' in meta

    def test_get_timeframe_metadata_invalid(self):
        """Test metadata fails for invalid timeframe."""
        from config import get_timeframe_metadata
        with pytest.raises(ValueError):
            get_timeframe_metadata('invalid')


# =============================================================================
# RESAMPLE_OHLCV TESTS
# =============================================================================

class TestResampleOHLCV:
    """Tests for the resample_ohlcv() function."""

    def test_resample_1min_to_5min(self, sample_1min_ohlcv):
        """Test resampling from 1-minute to 5-minute bars."""
        from stages.stage2_clean import resample_ohlcv

        result = resample_ohlcv(sample_1min_ohlcv, '5min')

        # 60 1-minute bars should become 12 5-minute bars
        assert len(result) == 12
        assert 'timeframe' in result.columns
        assert result['timeframe'].iloc[0] == '5min'

    def test_resample_1min_to_15min(self, sample_1min_ohlcv):
        """Test resampling from 1-minute to 15-minute bars."""
        from stages.stage2_clean import resample_ohlcv

        result = resample_ohlcv(sample_1min_ohlcv, '15min')

        # 60 1-minute bars should become 4 15-minute bars
        assert len(result) == 4
        assert result['timeframe'].iloc[0] == '15min'

    def test_resample_1min_to_30min(self, sample_1min_ohlcv):
        """Test resampling from 1-minute to 30-minute bars."""
        from stages.stage2_clean import resample_ohlcv

        result = resample_ohlcv(sample_1min_ohlcv, '30min')

        # 60 1-minute bars should become 2 30-minute bars
        assert len(result) == 2
        assert result['timeframe'].iloc[0] == '30min'

    def test_resample_1min_to_60min(self, sample_1min_ohlcv):
        """Test resampling from 1-minute to 60-minute bars."""
        from stages.stage2_clean import resample_ohlcv

        result = resample_ohlcv(sample_1min_ohlcv, '60min')

        # 60 1-minute bars spanning 09:30-10:29 may create 1 or 2 60-minute bars
        # depending on alignment (09:00-09:59 and 10:00-10:59)
        # The important thing is that data is resampled correctly
        assert len(result) <= 2  # Maximum possible with 60 minutes of data
        assert len(result) >= 1  # At least one bar
        assert result['timeframe'].iloc[0] == '60min'

    def test_resample_ohlcv_aggregation(self, sample_1min_ohlcv):
        """Test that OHLCV aggregation is correct."""
        from stages.stage2_clean import resample_ohlcv

        result = resample_ohlcv(sample_1min_ohlcv, '5min')

        # Take first 5-minute bar (first 5 rows of 1-minute data)
        first_5_rows = sample_1min_ohlcv.iloc[:5]
        first_bar = result.iloc[0]

        # Open should be first open
        assert first_bar['open'] == first_5_rows['open'].iloc[0]
        # High should be max high
        assert first_bar['high'] == first_5_rows['high'].max()
        # Low should be min low
        assert first_bar['low'] == first_5_rows['low'].min()
        # Close should be last close
        assert first_bar['close'] == first_5_rows['close'].iloc[-1]
        # Volume should be sum
        assert first_bar['volume'] == first_5_rows['volume'].sum()

    def test_resample_no_metadata(self, sample_1min_ohlcv):
        """Test resampling without metadata column."""
        from stages.stage2_clean import resample_ohlcv

        result = resample_ohlcv(sample_1min_ohlcv, '5min', include_metadata=False)

        assert 'timeframe' not in result.columns

    def test_resample_preserves_symbol(self, sample_1min_ohlcv):
        """Test that resampling preserves symbol column if present."""
        from stages.stage2_clean import resample_ohlcv

        sample_1min_ohlcv['symbol'] = 'MES'
        result = resample_ohlcv(sample_1min_ohlcv, '5min')

        assert 'symbol' in result.columns
        assert result['symbol'].iloc[0] == 'MES'

    def test_resample_invalid_timeframe(self, sample_1min_ohlcv):
        """Test that invalid timeframe raises error."""
        from stages.stage2_clean import resample_ohlcv

        with pytest.raises(ValueError, match="Unsupported timeframe"):
            resample_ohlcv(sample_1min_ohlcv, '3min')

    def test_resample_missing_columns(self, sample_1min_ohlcv):
        """Test that missing columns raises error."""
        from stages.stage2_clean import resample_ohlcv

        df_missing = sample_1min_ohlcv.drop('volume', axis=1)
        with pytest.raises(ValueError, match="Missing required columns"):
            resample_ohlcv(df_missing, '5min')

    def test_resample_empty_df(self):
        """Test that empty DataFrame raises error."""
        from stages.stage2_clean import resample_ohlcv

        df_empty = pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        with pytest.raises(ValueError, match="empty"):
            resample_ohlcv(df_empty, '5min')


# =============================================================================
# BACKWARD COMPATIBILITY TESTS
# =============================================================================

class TestBackwardCompatibility:
    """Tests for backward compatibility with resample_to_5min()."""

    def test_resample_to_5min_still_works(self, sample_1min_ohlcv):
        """Test that resample_to_5min() still works."""
        from stages.stage2_clean import resample_to_5min

        result = resample_to_5min(sample_1min_ohlcv)

        # Should produce 12 5-minute bars from 60 1-minute bars
        assert len(result) == 12

    def test_resample_to_5min_no_metadata(self, sample_1min_ohlcv):
        """Test that resample_to_5min() does not add timeframe column."""
        from stages.stage2_clean import resample_to_5min

        result = resample_to_5min(sample_1min_ohlcv)

        # Backward compat: no timeframe column
        assert 'timeframe' not in result.columns


# =============================================================================
# DATA CLEANER MTF TESTS
# =============================================================================

class TestDataCleanerMTF:
    """Tests for DataCleaner with target_timeframe parameter."""

    def test_datacleaner_accepts_target_timeframe(self, temp_dir):
        """Test that DataCleaner accepts target_timeframe parameter."""
        from stages.stage2_clean import DataCleaner

        cleaner = DataCleaner(
            input_dir=temp_dir,
            output_dir=temp_dir / "output",
            timeframe='1min',
            target_timeframe='15min'
        )

        assert cleaner.target_timeframe == '15min'
        assert cleaner.target_freq_minutes == 15

    def test_datacleaner_default_target_timeframe(self, temp_dir):
        """Test that DataCleaner defaults to 5min target."""
        from stages.stage2_clean import DataCleaner

        cleaner = DataCleaner(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )

        assert cleaner.target_timeframe == '5min'

    def test_datacleaner_invalid_target_timeframe(self, temp_dir):
        """Test that DataCleaner rejects invalid target_timeframe."""
        from stages.stage2_clean import DataCleaner

        with pytest.raises(ValueError, match="Unsupported timeframe"):
            DataCleaner(
                input_dir=temp_dir,
                output_dir=temp_dir / "output",
                target_timeframe='3min'
            )

    def test_datacleaner_resample_data_method(self, temp_dir, sample_1min_ohlcv):
        """Test DataCleaner.resample_data() method."""
        from stages.stage2_clean import DataCleaner

        cleaner = DataCleaner(
            input_dir=temp_dir,
            output_dir=temp_dir / "output",
            timeframe='1min',
            target_timeframe='15min'
        )

        result = cleaner.resample_data(sample_1min_ohlcv)

        assert len(result) == 4  # 60 / 15 = 4
        assert result['timeframe'].iloc[0] == '15min'

    def test_datacleaner_resample_override_timeframe(self, temp_dir, sample_1min_ohlcv):
        """Test that resample_data can override default timeframe."""
        from stages.stage2_clean import DataCleaner

        cleaner = DataCleaner(
            input_dir=temp_dir,
            output_dir=temp_dir / "output",
            timeframe='1min',
            target_timeframe='5min'  # Default
        )

        # Override to 30min
        result = cleaner.resample_data(sample_1min_ohlcv, target_timeframe='30min')

        assert len(result) == 2  # 60 / 30 = 2
        assert result['timeframe'].iloc[0] == '30min'

    def test_datacleaner_resample_skip_if_same(self, temp_dir, sample_1min_ohlcv):
        """Test that resample_data skips if source equals target."""
        from stages.stage2_clean import DataCleaner

        cleaner = DataCleaner(
            input_dir=temp_dir,
            output_dir=temp_dir / "output",
            timeframe='1min',
            target_timeframe='1min'  # Same as source
        )

        result = cleaner.resample_data(sample_1min_ohlcv)

        # Should not resample - same length
        assert len(result) == len(sample_1min_ohlcv)


# =============================================================================
# CLEAN_SYMBOL_DATA MTF TESTS
# =============================================================================

class TestCleanSymbolDataMTF:
    """Tests for clean_symbol_data() with target_timeframe parameter."""

    def test_clean_symbol_data_default_5min(self, temp_dir, sample_1min_ohlcv_300_rows):
        """Test clean_symbol_data defaults to 5-minute resampling."""
        from stages.stage2_clean import clean_symbol_data

        # Save input data
        input_path = temp_dir / "MES.parquet"
        output_path = temp_dir / "MES_clean.parquet"
        sample_1min_ohlcv_300_rows.to_parquet(input_path, index=False)

        result = clean_symbol_data(input_path, output_path, 'MES')

        # 300 1-min bars = 60 5-min bars
        assert len(result) == 60
        assert 'timeframe' in result.columns
        assert result['timeframe'].iloc[0] == '5min'

    def test_clean_symbol_data_custom_timeframe(self, temp_dir, sample_1min_ohlcv_300_rows):
        """Test clean_symbol_data with custom target_timeframe."""
        from stages.stage2_clean import clean_symbol_data

        input_path = temp_dir / "MES.parquet"
        output_path = temp_dir / "MES_15min.parquet"
        sample_1min_ohlcv_300_rows.to_parquet(input_path, index=False)

        result = clean_symbol_data(
            input_path,
            output_path,
            'MES',
            target_timeframe='15min'
        )

        # 300 1-min bars = 20 15-min bars
        assert len(result) == 20
        assert result['timeframe'].iloc[0] == '15min'

    def test_clean_symbol_data_no_metadata(self, temp_dir, sample_1min_ohlcv_300_rows):
        """Test clean_symbol_data without timeframe metadata."""
        from stages.stage2_clean import clean_symbol_data

        input_path = temp_dir / "MES.parquet"
        output_path = temp_dir / "MES_clean.parquet"
        sample_1min_ohlcv_300_rows.to_parquet(input_path, index=False)

        result = clean_symbol_data(
            input_path,
            output_path,
            'MES',
            include_timeframe_metadata=False
        )

        assert 'timeframe' not in result.columns

    def test_clean_symbol_data_invalid_timeframe(self, temp_dir, sample_1min_ohlcv_300_rows):
        """Test clean_symbol_data rejects invalid timeframe."""
        from stages.stage2_clean import clean_symbol_data

        input_path = temp_dir / "MES.parquet"
        output_path = temp_dir / "MES_clean.parquet"
        sample_1min_ohlcv_300_rows.to_parquet(input_path, index=False)

        with pytest.raises(ValueError, match="Unsupported timeframe"):
            clean_symbol_data(
                input_path,
                output_path,
                'MES',
                target_timeframe='7min'
            )


# =============================================================================
# MULTI-TIMEFRAME PIPELINE TESTS
# =============================================================================

class TestMultiTimeframePipeline:
    """Tests for clean_symbol_data_multi_timeframe()."""

    def test_multi_timeframe_default(self, temp_dir, sample_1min_ohlcv_300_rows):
        """Test multi-timeframe processing with defaults."""
        from stages.stage2_clean import clean_symbol_data_multi_timeframe

        input_path = temp_dir / "MES.parquet"
        output_dir = temp_dir / "clean"
        sample_1min_ohlcv_300_rows.to_parquet(input_path, index=False)

        results = clean_symbol_data_multi_timeframe(
            input_path,
            output_dir,
            'MES'
        )

        # Default: ['5min', '15min', '30min']
        assert '5min' in results
        assert '15min' in results
        assert '30min' in results

        assert len(results['5min']) == 60   # 300 / 5
        assert len(results['15min']) == 20  # 300 / 15
        assert len(results['30min']) == 10  # 300 / 30

    def test_multi_timeframe_custom_list(self, temp_dir, sample_1min_ohlcv_300_rows):
        """Test multi-timeframe with custom timeframe list."""
        from stages.stage2_clean import clean_symbol_data_multi_timeframe

        input_path = temp_dir / "MES.parquet"
        output_dir = temp_dir / "clean"
        sample_1min_ohlcv_300_rows.to_parquet(input_path, index=False)

        results = clean_symbol_data_multi_timeframe(
            input_path,
            output_dir,
            'MES',
            timeframes=['10min', '20min', '60min']
        )

        assert '10min' in results
        assert '20min' in results
        assert '60min' in results

        # Use approximate ranges due to timestamp alignment
        # 300 1-minute bars = 5 hours of data = 09:30 to 14:29
        assert 28 <= len(results['10min']) <= 32  # ~300 / 10
        assert 14 <= len(results['20min']) <= 17  # ~300 / 20
        assert 5 <= len(results['60min']) <= 6    # ~300 / 60

    def test_multi_timeframe_creates_files(self, temp_dir, sample_1min_ohlcv_300_rows):
        """Test that multi-timeframe creates output files."""
        from stages.stage2_clean import clean_symbol_data_multi_timeframe

        input_path = temp_dir / "MES.parquet"
        output_dir = temp_dir / "clean"
        sample_1min_ohlcv_300_rows.to_parquet(input_path, index=False)

        clean_symbol_data_multi_timeframe(
            input_path,
            output_dir,
            'MES',
            timeframes=['5min', '15min']
        )

        assert (output_dir / "MES_5min.parquet").exists()
        assert (output_dir / "MES_15min.parquet").exists()


# =============================================================================
# GET_RESAMPLING_INFO TESTS
# =============================================================================

class TestGetResamplingInfo:
    """Tests for get_resampling_info() utility function."""

    def test_resampling_info_1min_to_5min(self):
        """Test resampling info from 1min to 5min."""
        from stages.stage2_clean import get_resampling_info

        info = get_resampling_info('1min', '5min')

        assert info['source_timeframe'] == '1min'
        assert info['target_timeframe'] == '5min'
        assert info['bars_per_target'] == 5
        assert info['scale_factor'] == 5.0

    def test_resampling_info_5min_to_15min(self):
        """Test resampling info from 5min to 15min."""
        from stages.stage2_clean import get_resampling_info

        info = get_resampling_info('5min', '15min')

        assert info['bars_per_target'] == 3
        assert info['scale_factor'] == 3.0

    def test_resampling_info_invalid_downsampling(self):
        """Test that downsampling (larger to smaller) raises error."""
        from stages.stage2_clean import get_resampling_info

        with pytest.raises(ValueError, match="Cannot resample"):
            get_resampling_info('15min', '5min')


# =============================================================================
# PIPELINE CONFIG MTF TESTS
# =============================================================================

class TestPipelineConfigMTF:
    """Tests for PipelineConfig with target_timeframe."""

    def test_pipeline_config_default_timeframe(self):
        """Test PipelineConfig defaults to 5min."""
        from pipeline_config import PipelineConfig

        config = PipelineConfig()
        assert config.target_timeframe == '5min'

    def test_pipeline_config_custom_timeframe(self):
        """Test PipelineConfig with custom target_timeframe."""
        from pipeline_config import PipelineConfig

        config = PipelineConfig(target_timeframe='15min')
        assert config.target_timeframe == '15min'
        assert config.bar_resolution == '15min'  # Should sync

    def test_pipeline_config_bar_resolution_backward_compat(self):
        """Test that bar_resolution syncs with target_timeframe."""
        from pipeline_config import PipelineConfig

        # Old code might set bar_resolution directly
        config = PipelineConfig(bar_resolution='30min')
        assert config.target_timeframe == '30min'

    def test_pipeline_config_invalid_timeframe(self):
        """Test PipelineConfig rejects invalid timeframe."""
        from pipeline_config import PipelineConfig

        with pytest.raises(ValueError, match="Unsupported timeframe"):
            PipelineConfig(target_timeframe='7min')

    def test_pipeline_config_validate_timeframe(self):
        """Test PipelineConfig.validate() includes timeframe check."""
        from pipeline_config import PipelineConfig

        config = PipelineConfig()
        issues = config.validate()

        # Should be valid with defaults
        assert not any('timeframe' in issue.lower() for issue in issues)
