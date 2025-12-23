"""
Tests for Multi-Timeframe (MTF) Feature Integration.

This module tests the MTFFeatureGenerator class and related functions
to ensure:
1. Correct resampling from base to higher timeframes
2. No lookahead bias in alignment
3. Proper handling of edge cases
4. Integration with the feature engineering pipeline

Test Categories:
- Unit tests for individual methods
- Integration tests for the full pipeline
- Lookahead bias prevention tests
- Edge case and error handling tests
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import pytest
import numpy as np
import pandas as pd

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from stages.mtf_features import (
    MTFFeatureGenerator,
    add_mtf_features,
    validate_mtf_alignment,
    validate_ohlcv_dataframe,
    validate_timeframe_format,
    MTF_TIMEFRAMES,
    REQUIRED_OHLCV_COLS,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_5min_ohlcv():
    """
    Generate sample 5-minute OHLCV data for testing.

    Creates 3000 bars of realistic price data with proper OHLC relationships.
    This size is sufficient for feature engineering with MTF features after
    NaN removal from rolling windows (SMA-200 needs 200 warmup bars, etc.).
    """
    np.random.seed(42)
    n = 3000

    # Generate realistic price series
    base_price = 4500.0
    returns = np.random.randn(n) * 0.001
    close = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC with proper relationships
    daily_range = np.abs(np.random.randn(n) * 0.002)
    high = close * (1 + daily_range / 2)
    low = close * (1 - daily_range / 2)
    open_ = close * (1 + np.random.randn(n) * 0.0005)

    # Ensure valid OHLC
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    # Generate volume
    volume = np.random.randint(100, 10000, n).astype(float)

    # Generate 5-minute timestamps
    start_time = datetime(2024, 1, 1, 9, 30)
    timestamps = [start_time + timedelta(minutes=5*i) for i in range(n)]

    return pd.DataFrame({
        'datetime': timestamps,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


@pytest.fixture
def small_ohlcv():
    """Generate small OHLCV data for edge case testing."""
    np.random.seed(42)
    n = 100

    base_price = 100.0
    close = base_price + np.cumsum(np.random.randn(n) * 0.5)

    start_time = datetime(2024, 1, 1, 9, 30)
    timestamps = [start_time + timedelta(minutes=5*i) for i in range(n)]

    return pd.DataFrame({
        'datetime': timestamps,
        'open': close * 0.999,
        'high': close * 1.002,
        'low': close * 0.998,
        'close': close,
        'volume': np.random.randint(100, 1000, n).astype(float)
    })


@pytest.fixture
def mtf_generator():
    """Create default MTFFeatureGenerator instance."""
    return MTFFeatureGenerator(
        base_timeframe='5min',
        mtf_timeframes=['15min', '60min'],
        include_ohlcv=True,
        include_indicators=True
    )


# =============================================================================
# VALIDATION FUNCTION TESTS
# =============================================================================

class TestValidationFunctions:
    """Tests for validation functions."""

    def test_validate_ohlcv_dataframe_valid(self, sample_5min_ohlcv):
        """Test validation passes for valid DataFrame."""
        # Should not raise
        validate_ohlcv_dataframe(sample_5min_ohlcv)

    def test_validate_ohlcv_dataframe_missing_datetime(self):
        """Test validation fails when datetime is missing."""
        df = pd.DataFrame({
            'open': [100], 'high': [101], 'low': [99],
            'close': [100], 'volume': [1000]
        })

        with pytest.raises(ValueError, match="must have 'datetime' column"):
            validate_ohlcv_dataframe(df)

    def test_validate_ohlcv_dataframe_missing_ohlcv(self):
        """Test validation fails when OHLCV columns are missing."""
        df = pd.DataFrame({
            'datetime': [datetime.now()],
            'open': [100], 'high': [101], 'low': [99]
            # Missing 'close' and 'volume'
        })

        with pytest.raises(ValueError, match="missing required OHLCV columns"):
            validate_ohlcv_dataframe(df)

    def test_validate_timeframe_format_valid(self):
        """Test timeframe validation for valid formats."""
        for tf in ['5min', '15min', '30min', '60min', '1h']:
            validate_timeframe_format(tf)  # Should not raise

    def test_validate_timeframe_format_invalid(self):
        """Test timeframe validation fails for invalid formats."""
        with pytest.raises(ValueError, match="Unknown timeframe format"):
            validate_timeframe_format('10min')

        with pytest.raises(ValueError, match="Unknown timeframe format"):
            validate_timeframe_format('2h')


# =============================================================================
# MTF FEATURE GENERATOR INITIALIZATION TESTS
# =============================================================================

class TestMTFFeatureGeneratorInit:
    """Tests for MTFFeatureGenerator initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        generator = MTFFeatureGenerator()

        assert generator.base_timeframe == '5min'
        assert generator.mtf_timeframes == ['15min', '60min']
        assert generator.include_ohlcv is True
        assert generator.include_indicators is True

    def test_custom_initialization(self):
        """Test custom initialization."""
        generator = MTFFeatureGenerator(
            base_timeframe='5min',
            mtf_timeframes=['30min'],
            include_ohlcv=False,
            include_indicators=True
        )

        assert generator.mtf_timeframes == ['30min']
        assert generator.include_ohlcv is False

    def test_invalid_mtf_timeframe_too_small(self):
        """Test that MTF timeframe must be greater than base."""
        with pytest.raises(ValueError, match="must be >"):
            MTFFeatureGenerator(
                base_timeframe='15min',
                mtf_timeframes=['5min']
            )

    def test_invalid_mtf_timeframe_not_multiple(self):
        """Test that MTF timeframe must be integer multiple of base."""
        # Note: 10min is not in our supported list, so this should fail validation
        with pytest.raises(ValueError):
            MTFFeatureGenerator(
                base_timeframe='5min',
                mtf_timeframes=['10min']
            )


# =============================================================================
# RESAMPLING TESTS
# =============================================================================

class TestResampling:
    """Tests for OHLCV resampling."""

    def test_resample_to_15min(self, sample_5min_ohlcv, mtf_generator):
        """Test resampling from 5min to 15min."""
        df_15min = mtf_generator.resample_to_tf(sample_5min_ohlcv, '15min')

        # Should have 1/3 the rows (3 5min bars = 1 15min bar)
        expected_rows = len(sample_5min_ohlcv) // 3
        assert len(df_15min) >= expected_rows - 1  # Allow for edge effects

        # Should have datetime column
        assert 'datetime' in df_15min.columns

        # Should have OHLCV columns
        for col in REQUIRED_OHLCV_COLS:
            assert col in df_15min.columns

    def test_resample_to_60min(self, sample_5min_ohlcv, mtf_generator):
        """Test resampling from 5min to 60min."""
        df_60min = mtf_generator.resample_to_tf(sample_5min_ohlcv, '60min')

        # Should have 1/12 the rows (12 5min bars = 1 60min bar)
        expected_rows = len(sample_5min_ohlcv) // 12
        assert len(df_60min) >= expected_rows - 1

    def test_resample_ohlc_aggregation(self, sample_5min_ohlcv, mtf_generator):
        """Test that OHLC aggregation is correct."""
        df_15min = mtf_generator.resample_to_tf(sample_5min_ohlcv, '15min')

        # For the first 15min bar, check aggregation
        first_3_bars = sample_5min_ohlcv.iloc[:3]
        first_15min = df_15min.iloc[0]

        # Open should be first bar's open
        assert np.isclose(first_15min['open'], first_3_bars['open'].iloc[0])

        # High should be max of all highs
        assert np.isclose(first_15min['high'], first_3_bars['high'].max())

        # Low should be min of all lows
        assert np.isclose(first_15min['low'], first_3_bars['low'].min())

        # Close should be last bar's close
        assert np.isclose(first_15min['close'], first_3_bars['close'].iloc[-1])

        # Volume should be sum of all volumes
        assert np.isclose(first_15min['volume'], first_3_bars['volume'].sum())


# =============================================================================
# INDICATOR COMPUTATION TESTS
# =============================================================================

class TestIndicatorComputation:
    """Tests for indicator computation on higher TFs."""

    def test_compute_mtf_indicators(self, sample_5min_ohlcv, mtf_generator):
        """Test that indicators are computed correctly."""
        df_15min = mtf_generator.resample_to_tf(sample_5min_ohlcv, '15min')
        df_with_indicators = mtf_generator.compute_mtf_indicators(df_15min, '15min')

        # Check expected indicator columns exist
        expected_cols = [
            'sma_20_15m', 'sma_50_15m',
            'ema_9_15m', 'ema_21_15m',
            'rsi_14_15m', 'atr_14_15m',
            'bb_position_15m', 'macd_hist_15m',
            'close_sma20_ratio_15m'
        ]

        for col in expected_cols:
            assert col in df_with_indicators.columns, f"Missing column: {col}"

    def test_indicator_values_in_range(self, sample_5min_ohlcv, mtf_generator):
        """Test that indicator values are in expected ranges."""
        df_15min = mtf_generator.resample_to_tf(sample_5min_ohlcv, '15min')
        df_with_indicators = mtf_generator.compute_mtf_indicators(df_15min, '15min')

        # RSI should be in [0, 100]
        rsi_valid = df_with_indicators['rsi_14_15m'].dropna()
        assert (rsi_valid >= 0).all() and (rsi_valid <= 100).all()

        # BB position should be roughly in [-0.5, 1.5] (can exceed [0, 1])
        bb_valid = df_with_indicators['bb_position_15m'].dropna()
        assert (bb_valid >= -1).all() and (bb_valid <= 2).all()


# =============================================================================
# ALIGNMENT TESTS - CRITICAL FOR LOOKAHEAD PREVENTION
# =============================================================================

class TestAlignment:
    """Tests for MTF alignment to base timeframe."""

    def test_align_to_base_tf_structure(self, sample_5min_ohlcv, mtf_generator):
        """Test that alignment produces correct structure."""
        # Resample and compute
        df_15min = mtf_generator.resample_to_tf(sample_5min_ohlcv, '15min')
        df_15min['test_col_15m'] = df_15min['close']

        # Align back to base
        aligned = mtf_generator.align_to_base_tf(
            sample_5min_ohlcv,
            df_15min,
            ['test_col_15m']
        )

        # Should have same length as base
        assert len(aligned) == len(sample_5min_ohlcv)

        # Should have the MTF column
        assert 'test_col_15m' in aligned.columns

    def test_no_lookahead_shift_applied(self, sample_5min_ohlcv, mtf_generator):
        """Test that shift(1) is applied to prevent lookahead."""
        df_15min = mtf_generator.resample_to_tf(sample_5min_ohlcv, '15min')
        df_15min['test_col_15m'] = np.arange(len(df_15min))  # Sequential values

        aligned = mtf_generator.align_to_base_tf(
            sample_5min_ohlcv,
            df_15min,
            ['test_col_15m']
        )

        # First few rows should be NaN due to shift(1)
        assert aligned['test_col_15m'].iloc[0] != aligned['test_col_15m'].iloc[0]  # isnan

    def test_forward_fill_behavior(self, sample_5min_ohlcv, mtf_generator):
        """Test that forward-fill works correctly."""
        df_15min = mtf_generator.resample_to_tf(sample_5min_ohlcv, '15min')
        df_15min['test_col_15m'] = np.arange(len(df_15min))

        aligned = mtf_generator.align_to_base_tf(
            sample_5min_ohlcv,
            df_15min,
            ['test_col_15m']
        )

        # After initial NaN, values should repeat (forward-filled)
        # Get first non-NaN value
        first_valid_idx = aligned['test_col_15m'].first_valid_index()
        if first_valid_idx is not None:
            # Next few values should be the same (forward-filled)
            first_val = aligned.loc[first_valid_idx, 'test_col_15m']
            next_val = aligned.loc[first_valid_idx + 1, 'test_col_15m']
            assert first_val == next_val


# =============================================================================
# LOOKAHEAD BIAS PREVENTION TESTS
# =============================================================================

class TestLookaheadPrevention:
    """Critical tests for lookahead bias prevention."""

    def test_mtf_features_have_initial_nans(self, sample_5min_ohlcv, mtf_generator):
        """Test that MTF features have NaN at start due to shift."""
        df_with_mtf = mtf_generator.generate_mtf_features(sample_5min_ohlcv)

        # Get MTF columns
        mtf_cols = [c for c in df_with_mtf.columns if '_15m' in c or '_1h' in c]

        for col in mtf_cols:
            # First row should be NaN (due to shift + warmup)
            assert pd.isna(df_with_mtf[col].iloc[0]), f"Column {col} has value at index 0"

    def test_validate_no_lookahead_passes(self, sample_5min_ohlcv, mtf_generator):
        """Test that validate_no_lookahead passes for correct data."""
        df_with_mtf = mtf_generator.generate_mtf_features(sample_5min_ohlcv)

        # Should not raise
        result = mtf_generator.validate_no_lookahead(df_with_mtf)
        assert result is True

    def test_lookahead_detection(self, sample_5min_ohlcv, mtf_generator):
        """Test that lookahead is detected when present."""
        df_with_mtf = mtf_generator.generate_mtf_features(sample_5min_ohlcv)

        # Artificially remove NaN at start (simulate lookahead bug)
        mtf_cols = [c for c in df_with_mtf.columns if '_15m' in c]
        if mtf_cols:
            df_with_mtf[mtf_cols[0]].iloc[0] = 100.0

            with pytest.raises(ValueError, match="Potential lookahead"):
                mtf_generator.validate_no_lookahead(df_with_mtf)

    def test_mtf_value_at_boundary(self, sample_5min_ohlcv, mtf_generator):
        """Test MTF values at TF boundaries are from PREVIOUS completed bar."""
        # This is a critical test for lookahead prevention

        # At minute 15 (3rd 5min bar), we should have data from minute 0-15 bar
        # But with shift(1), we should see data from the bar BEFORE that

        df_15min = mtf_generator.resample_to_tf(sample_5min_ohlcv, '15min')
        df_15min['close_15m'] = df_15min['close'].copy()

        # Get the value at the first 15min bar
        first_15m_close = df_15min.iloc[0]['close']

        # Align to base
        aligned = mtf_generator.align_to_base_tf(
            sample_5min_ohlcv,
            df_15min,
            ['close_15m']
        )

        # CRITICAL TEST: First row should ALWAYS be NaN due to shift(1)
        # This ensures we never use the current incomplete bar
        first_bar_mtf = aligned.loc[0, 'close_15m']
        assert pd.isna(first_bar_mtf), "First row should be NaN due to shift(1) anti-lookahead"

        # The first 15min bar value should only appear AFTER the shift
        # Find where the first value appears
        first_valid_idx = aligned['close_15m'].first_valid_index()
        assert first_valid_idx > 0, "MTF values should not appear at index 0"

        # The first valid value should match the FIRST 15min close
        # (due to shift(1), first valid at base index 3 uses MTF index 0)
        if first_valid_idx is not None:
            first_mtf_value = aligned.loc[first_valid_idx, 'close_15m']
            assert np.isclose(first_mtf_value, first_15m_close), \
                "First MTF value should match first resampled bar's close"


# =============================================================================
# FULL PIPELINE TESTS
# =============================================================================

class TestFullPipeline:
    """Tests for the complete MTF feature generation pipeline."""

    def test_generate_mtf_features_all_columns(self, sample_5min_ohlcv, mtf_generator):
        """Test that all expected columns are generated."""
        df_result = mtf_generator.generate_mtf_features(sample_5min_ohlcv)

        # Should have more columns than input
        assert len(df_result.columns) > len(sample_5min_ohlcv.columns)

        # Check for 15min columns
        assert any('_15m' in c for c in df_result.columns)

        # Check for 1h columns
        assert any('_1h' in c for c in df_result.columns)

    def test_generate_mtf_features_ohlcv_only(self, sample_5min_ohlcv):
        """Test generation with OHLCV only (no indicators)."""
        generator = MTFFeatureGenerator(
            base_timeframe='5min',
            mtf_timeframes=['15min'],
            include_ohlcv=True,
            include_indicators=False
        )

        df_result = generator.generate_mtf_features(sample_5min_ohlcv)

        # Should have OHLCV columns
        for col in ['open_15m', 'high_15m', 'low_15m', 'close_15m', 'volume_15m']:
            assert col in df_result.columns

        # Should NOT have indicator columns
        assert 'rsi_14_15m' not in df_result.columns

    def test_generate_mtf_features_indicators_only(self, sample_5min_ohlcv):
        """Test generation with indicators only (no OHLCV)."""
        generator = MTFFeatureGenerator(
            base_timeframe='5min',
            mtf_timeframes=['15min'],
            include_ohlcv=False,
            include_indicators=True
        )

        df_result = generator.generate_mtf_features(sample_5min_ohlcv)

        # Should NOT have raw OHLCV columns
        assert 'open_15m' not in df_result.columns

        # Should have indicator columns
        assert 'rsi_14_15m' in df_result.columns

    def test_convenience_function_add_mtf_features(self, sample_5min_ohlcv):
        """Test the add_mtf_features convenience function."""
        metadata = {}
        df_result = add_mtf_features(
            sample_5min_ohlcv,
            feature_metadata=metadata,
            base_timeframe='5min',
            mtf_timeframes=['15min']
        )

        # Should have MTF columns
        assert any('_15m' in c for c in df_result.columns)

        # Metadata should be populated
        assert len(metadata) > 0


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        # Create very small dataset that triggers the 100-row minimum
        np.random.seed(42)
        n = 50  # Less than minimum of 100

        start_time = datetime(2024, 1, 1, 9, 30)
        timestamps = [start_time + timedelta(minutes=5*i) for i in range(n)]

        df = pd.DataFrame({
            'datetime': timestamps,
            'open': np.ones(n) * 100,
            'high': np.ones(n) * 101,
            'low': np.ones(n) * 99,
            'close': np.ones(n) * 100,
            'volume': np.ones(n) * 1000
        })

        generator = MTFFeatureGenerator()
        with pytest.raises(ValueError, match="Insufficient data"):
            generator.generate_mtf_features(df)

    def test_missing_datetime_column(self, mtf_generator):
        """Test error when datetime column is missing."""
        df = pd.DataFrame({
            'open': [100], 'high': [101], 'low': [99],
            'close': [100], 'volume': [1000]
        })

        with pytest.raises(ValueError, match="datetime"):
            mtf_generator.generate_mtf_features(df)

    def test_missing_ohlcv_columns(self, mtf_generator):
        """Test error when OHLCV columns are missing."""
        df = pd.DataFrame({
            'datetime': [datetime.now()],
            'open': [100], 'high': [101]
            # Missing low, close, volume
        })

        with pytest.raises(ValueError):
            mtf_generator.generate_mtf_features(df)

    def test_get_mtf_column_names(self, mtf_generator):
        """Test getting expected column names."""
        col_names = mtf_generator.get_mtf_column_names()

        assert '15min' in col_names
        assert '60min' in col_names

        # Check 15min columns include expected names
        assert 'close_15m' in col_names['15min']
        assert 'rsi_14_15m' in col_names['15min']


# =============================================================================
# INTEGRATION WITH FEATURE ENGINEER
# =============================================================================

class TestFeatureEngineerIntegration:
    """Tests for integration with FeatureEngineer class."""

    def test_feature_engineer_with_mtf(self, sample_5min_ohlcv, tmp_path):
        """Test FeatureEngineer with MTF features enabled."""
        from stages.features import FeatureEngineer

        input_dir = tmp_path / 'input'
        output_dir = tmp_path / 'output'
        input_dir.mkdir()
        output_dir.mkdir()

        # Save test data
        sample_5min_ohlcv.to_parquet(input_dir / 'MES_clean.parquet')

        engineer = FeatureEngineer(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            timeframe='5min',
            enable_mtf=True,
            mtf_timeframes=['15min']
        )

        df, report = engineer.engineer_features(sample_5min_ohlcv, 'MES')

        # Should have MTF features
        assert report.get('mtf_features') is True
        assert report.get('mtf_feature_count', 0) > 0

        # Should have MTF columns
        mtf_cols = [c for c in df.columns if '_15m' in c]
        assert len(mtf_cols) > 0

    def test_feature_engineer_mtf_disabled(self, sample_5min_ohlcv, tmp_path):
        """Test FeatureEngineer with MTF features disabled."""
        from stages.features import FeatureEngineer

        input_dir = tmp_path / 'input'
        output_dir = tmp_path / 'output'
        input_dir.mkdir()
        output_dir.mkdir()

        engineer = FeatureEngineer(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            timeframe='5min',
            enable_mtf=False
        )

        df, report = engineer.engineer_features(sample_5min_ohlcv, 'MES')

        # Should NOT have MTF features
        assert report.get('mtf_features') is False
        assert report.get('mtf_feature_count', 0) == 0


# =============================================================================
# VALIDATION FUNCTION TESTS
# =============================================================================

class TestValidateMTFAlignment:
    """Tests for validate_mtf_alignment function."""

    def test_valid_alignment(self, sample_5min_ohlcv):
        """Test validation passes for valid alignment."""
        df_base = sample_5min_ohlcv
        df_mtf = sample_5min_ohlcv.copy()  # Same timestamps

        is_valid, issues = validate_mtf_alignment(df_base, df_mtf)
        assert is_valid is True
        assert len(issues) == 0

    def test_missing_datetime(self):
        """Test validation fails when datetime is missing."""
        df_base = pd.DataFrame({'close': [100]})
        df_mtf = pd.DataFrame({'datetime': [datetime.now()]})

        is_valid, issues = validate_mtf_alignment(df_base, df_mtf)
        assert is_valid is False
        assert any('datetime' in issue for issue in issues)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
