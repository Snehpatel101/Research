"""
Tests for stage boundary validators.

Tests the validation logic for data passing between pipeline stages 2-7.
Ensures fail-fast behavior when data quality issues are detected.
"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from stages.validators import (
    # Core validation result
    ValidationResult,
    DEFAULT_THRESHOLDS,
    # Stage boundary utilities
    check_dataframe_basics,
    check_required_columns,
    check_nan_values,
    check_infinite_values,
    check_datetime_column,
    check_ohlcv_relationships,
    check_positive_values,
    check_row_drop_threshold,
    # Stage 2 output / Stage 3 input
    validate_cleaned_data,
    validate_cleaned_data_for_features,
    # Stage 3 output / Stage 4 input
    validate_feature_output,
    validate_features_for_labeling,
    get_feature_columns,
    # Stage 4/6 output / Stage 5/7 input
    validate_labeled_data,
    validate_labels_for_ga,
    validate_labels_for_splits,
)


# --- Fixtures ---

@pytest.fixture
def valid_ohlcv_df():
    """Create a valid OHLCV DataFrame for testing."""
    # Use 1100 rows to account for warmup period NaN drops
    n = 1100
    dates = pd.date_range(start='2024-01-01', periods=n, freq='5min')
    np.random.seed(42)

    base_price = 5000.0
    prices = base_price + np.cumsum(np.random.randn(n) * 0.5)

    df = pd.DataFrame({
        'datetime': dates,
        'open': prices,
        'high': prices + np.abs(np.random.randn(n) * 0.3),
        'low': prices - np.abs(np.random.randn(n) * 0.3),
        'close': prices + np.random.randn(n) * 0.1,
        'volume': np.random.randint(100, 10000, n),
    })

    # Ensure OHLC relationships are valid
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)

    return df


@pytest.fixture
def valid_features_df(valid_ohlcv_df):
    """Create a valid features DataFrame for testing."""
    df = valid_ohlcv_df.copy()

    # Add technical features
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['ema_10'] = df['close'].ewm(span=10).mean()
    df['rsi_14'] = 50 + np.random.randn(len(df)) * 10
    df['macd'] = np.random.randn(len(df)) * 0.1
    df['atr_14'] = np.abs(df['high'] - df['low']).rolling(14).mean()
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['return_1'] = df['close'].pct_change()
    df['return_5'] = df['close'].pct_change(5)
    df['volatility_20'] = df['return_1'].rolling(20).std()

    # Add more features to meet minimum count
    for i in range(15):
        df[f'feature_{i}'] = np.random.randn(len(df))

    # Drop warmup NaN rows
    df = df.dropna().reset_index(drop=True)

    return df


@pytest.fixture
def valid_labeled_df(valid_features_df):
    """Create a valid labeled DataFrame for testing."""
    df = valid_features_df.copy()
    n = len(df)

    # Add labels for horizons 5 and 20
    for horizon in [5, 20]:
        # Random labels: -1, 0, 1
        labels = np.random.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3])
        # Mark last few rows as invalid (-99)
        labels[-horizon*3:] = -99

        df[f'label_h{horizon}'] = labels
        df[f'bars_to_hit_h{horizon}'] = np.random.randint(1, horizon, n)
        df[f'mae_h{horizon}'] = np.random.rand(n) * 0.01
        df[f'mfe_h{horizon}'] = np.random.rand(n) * 0.01
        df[f'quality_h{horizon}'] = np.random.rand(n)
        df[f'sample_weight_h{horizon}'] = 0.5 + np.random.rand(n)

    return df


# --- ValidationResult Tests ---

class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_passed_result(self):
        """Test creating a passed validation result."""
        result = ValidationResult(passed=True, stage="test")
        assert result.passed is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_add_error_fails_validation(self):
        """Test that adding an error marks validation as failed."""
        result = ValidationResult(passed=True, stage="test")
        result.add_error("Test error")
        assert result.passed is False
        assert "Test error" in result.errors

    def test_add_warning_preserves_passed(self):
        """Test that warnings don't fail validation."""
        result = ValidationResult(passed=True, stage="test")
        result.add_warning("Test warning")
        assert result.passed is True
        assert "Test warning" in result.warnings

    def test_raise_if_failed(self):
        """Test that raise_if_failed raises ValueError on failure."""
        result = ValidationResult(passed=True, stage="test")
        result.add_error("Critical error")

        with pytest.raises(ValueError, match="Critical error"):
            result.raise_if_failed()

    def test_raise_if_failed_does_nothing_on_pass(self):
        """Test that raise_if_failed does nothing when passed."""
        result = ValidationResult(passed=True, stage="test")
        result.raise_if_failed()  # Should not raise


# --- Check Functions Tests ---

class TestCheckFunctions:
    """Tests for individual validation check functions."""

    def test_check_dataframe_basics_valid(self, valid_ohlcv_df):
        """Test basic DataFrame checks with valid data."""
        result = ValidationResult(passed=True, stage="test")
        check_dataframe_basics(valid_ohlcv_df, result, min_rows=500)
        assert result.passed is True
        assert result.metrics['row_count'] == len(valid_ohlcv_df)

    def test_check_dataframe_basics_empty(self):
        """Test basic checks fail with empty DataFrame."""
        result = ValidationResult(passed=True, stage="test")
        check_dataframe_basics(pd.DataFrame(), result)
        assert result.passed is False
        assert any("empty" in e.lower() for e in result.errors)

    def test_check_dataframe_basics_too_few_rows(self, valid_ohlcv_df):
        """Test basic checks fail with insufficient rows."""
        result = ValidationResult(passed=True, stage="test")
        small_df = valid_ohlcv_df.head(100)
        check_dataframe_basics(small_df, result, min_rows=500)
        assert result.passed is False
        assert any("rows" in e.lower() for e in result.errors)

    def test_check_required_columns_present(self, valid_ohlcv_df):
        """Test required columns check passes with all columns."""
        result = ValidationResult(passed=True, stage="test")
        required = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        check_required_columns(valid_ohlcv_df, required, result)
        assert result.passed is True

    def test_check_required_columns_missing(self, valid_ohlcv_df):
        """Test required columns check fails with missing columns."""
        result = ValidationResult(passed=True, stage="test")
        df = valid_ohlcv_df.drop(columns=['volume'])
        check_required_columns(df, ['datetime', 'volume'], result)
        assert result.passed is False
        assert any("volume" in e for e in result.errors)

    def test_check_nan_values_clean(self, valid_ohlcv_df):
        """Test NaN check passes with clean data."""
        result = ValidationResult(passed=True, stage="test")
        check_nan_values(valid_ohlcv_df, result)
        assert result.passed is True

    def test_check_nan_values_high_nan(self, valid_ohlcv_df):
        """Test NaN check warns/fails with high NaN percentage."""
        result = ValidationResult(passed=True, stage="test")
        df = valid_ohlcv_df.copy()
        df.loc[:100, 'close'] = np.nan  # 10% NaN
        check_nan_values(df, result, max_nan_pct=5.0)
        # Should have warnings for high NaN
        assert 'nan_columns' in result.metrics

    def test_check_infinite_values_clean(self, valid_ohlcv_df):
        """Test infinite values check passes with clean data."""
        result = ValidationResult(passed=True, stage="test")
        check_infinite_values(valid_ohlcv_df, result)
        assert result.passed is True
        assert result.metrics['infinite_value_columns'] == {}

    def test_check_infinite_values_with_inf(self, valid_ohlcv_df):
        """Test infinite values check fails with inf values."""
        result = ValidationResult(passed=True, stage="test")
        df = valid_ohlcv_df.copy()
        df.loc[0, 'close'] = np.inf
        check_infinite_values(df, result)
        assert result.passed is False
        assert 'close' in result.metrics['infinite_value_columns']

    def test_check_datetime_column_valid(self, valid_ohlcv_df):
        """Test datetime check passes with valid timestamps."""
        result = ValidationResult(passed=True, stage="test")
        check_datetime_column(valid_ohlcv_df, result)
        assert result.passed is True
        assert 'datetime_min' in result.metrics
        assert 'datetime_max' in result.metrics

    def test_check_datetime_column_duplicates(self, valid_ohlcv_df):
        """Test datetime check fails with duplicate timestamps."""
        result = ValidationResult(passed=True, stage="test")
        df = valid_ohlcv_df.copy()
        df.loc[1, 'datetime'] = df.loc[0, 'datetime']  # Duplicate
        check_datetime_column(df, result)
        assert result.passed is False
        assert any("duplicate" in e.lower() for e in result.errors)

    def test_check_ohlcv_relationships_valid(self, valid_ohlcv_df):
        """Test OHLCV relationships check passes with valid data."""
        result = ValidationResult(passed=True, stage="test")
        check_ohlcv_relationships(valid_ohlcv_df, result)
        assert result.passed is True

    def test_check_ohlcv_relationships_high_less_than_low(self, valid_ohlcv_df):
        """Test OHLCV check fails when high < low."""
        result = ValidationResult(passed=True, stage="test")
        df = valid_ohlcv_df.copy()
        df.loc[0, 'high'] = df.loc[0, 'low'] - 1  # Invalid
        check_ohlcv_relationships(df, result)
        assert result.passed is False
        assert any("high < low" in e for e in result.errors)

    def test_check_row_drop_threshold_within_limit(self):
        """Test row drop check passes within threshold."""
        result = ValidationResult(passed=True, stage="test")
        check_row_drop_threshold(1000, 980, result, max_drop_pct=5.0)
        assert result.passed is True
        assert result.metrics['drop_percentage'] == 2.0

    def test_check_row_drop_threshold_exceeds_limit(self):
        """Test row drop check fails when exceeding threshold."""
        result = ValidationResult(passed=True, stage="test")
        check_row_drop_threshold(1000, 900, result, max_drop_pct=5.0)
        assert result.passed is False
        assert result.metrics['drop_percentage'] == 10.0


# --- Cleaned Data Validator Tests ---

class TestCleanedDataValidator:
    """Tests for Stage 2 output / Stage 3 input validation."""

    def test_validate_cleaned_data_valid(self, valid_ohlcv_df):
        """Test validation passes with valid cleaned data."""
        result = validate_cleaned_data(valid_ohlcv_df, symbol="TEST")
        assert result.passed is True
        assert result.metrics['row_count'] == len(valid_ohlcv_df)

    def test_validate_cleaned_data_missing_columns(self):
        """Test validation fails with missing OHLCV columns."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=1000, freq='5min'),
            'close': np.random.randn(1000) + 5000,
        })
        result = validate_cleaned_data(df, symbol="TEST")
        assert result.passed is False
        assert any("Missing" in e for e in result.errors)

    def test_validate_cleaned_data_for_features_sorted(self, valid_ohlcv_df):
        """Test features input validation requires sorted datetime."""
        result = validate_cleaned_data_for_features(valid_ohlcv_df, symbol="TEST")
        assert result.passed is True

    def test_validate_cleaned_data_for_features_unsorted(self, valid_ohlcv_df):
        """Test features input validation fails with unsorted datetime."""
        df = valid_ohlcv_df.sample(frac=1).reset_index(drop=True)  # Shuffle
        result = validate_cleaned_data_for_features(df, symbol="TEST")
        assert result.passed is False
        assert any("sorted" in e.lower() for e in result.errors)


# --- Feature Output Validator Tests ---

class TestFeatureOutputValidator:
    """Tests for Stage 3 output / Stage 4 input validation."""

    def test_validate_feature_output_valid(self, valid_features_df):
        """Test validation passes with valid features data."""
        result = validate_feature_output(valid_features_df, symbol="TEST")
        assert result.passed is True
        assert result.metrics['feature_count'] >= 20

    def test_validate_feature_output_missing_atr(self, valid_ohlcv_df):
        """Test validation fails without ATR column."""
        df = valid_ohlcv_df.copy()
        # Add some features but no ATR
        for i in range(25):
            df[f'feature_{i}'] = np.random.randn(len(df))

        result = validate_feature_output(df, symbol="TEST")
        assert result.passed is False
        assert any("atr" in e.lower() for e in result.errors)

    def test_validate_feature_output_insufficient_features(self, valid_ohlcv_df):
        """Test validation fails with too few features."""
        df = valid_ohlcv_df.copy()
        df['feature_1'] = np.random.randn(len(df))
        df['atr_14'] = np.abs(df['high'] - df['low'])

        result = validate_feature_output(df, symbol="TEST", min_feature_count=20)
        assert result.passed is False
        assert any("features" in e.lower() for e in result.errors)

    def test_validate_features_for_labeling_valid(self, valid_features_df):
        """Test labeling input validation passes with valid features."""
        result = validate_features_for_labeling(valid_features_df, symbol="TEST")
        assert result.passed is True

    def test_validate_features_for_labeling_nan_atr(self, valid_features_df):
        """Test labeling input validation fails with NaN ATR."""
        df = valid_features_df.copy()
        df.loc[:10, 'atr_14'] = np.nan
        result = validate_features_for_labeling(df, symbol="TEST")
        assert result.passed is False
        assert any("atr" in e.lower() for e in result.errors)

    def test_get_feature_columns(self, valid_labeled_df):
        """Test feature column identification excludes labels and metadata."""
        feature_cols = get_feature_columns(valid_labeled_df)

        # Should not include OHLCV
        assert 'open' not in feature_cols
        assert 'close' not in feature_cols

        # Should not include labels
        assert 'label_h5' not in feature_cols
        assert 'label_h20' not in feature_cols

        # Should not include supporting columns
        assert 'bars_to_hit_h5' not in feature_cols
        assert 'quality_h5' not in feature_cols


# --- Labeled Data Validator Tests ---

class TestLabeledDataValidator:
    """Tests for Stage 4/6 output / Stage 5/7 input validation."""

    def test_validate_labeled_data_valid(self, valid_labeled_df):
        """Test validation passes with valid labeled data."""
        result = validate_labeled_data(
            valid_labeled_df,
            horizons=[5, 20],
            symbol="TEST"
        )
        assert result.passed is True
        assert 'label_stats' in result.metrics

    def test_validate_labeled_data_missing_labels(self, valid_features_df):
        """Test validation fails when label columns missing."""
        result = validate_labeled_data(
            valid_features_df,
            horizons=[5, 20],
            symbol="TEST"
        )
        assert result.passed is False
        assert any("label_h5" in e for e in result.errors)

    def test_validate_labeled_data_invalid_values(self, valid_labeled_df):
        """Test validation fails with invalid label values."""
        df = valid_labeled_df.copy()
        df.loc[0, 'label_h5'] = 999  # Invalid label value

        result = validate_labeled_data(df, horizons=[5, 20], symbol="TEST")
        assert result.passed is False
        assert any("Invalid" in e for e in result.errors)

    def test_validate_labeled_data_excessive_invalid(self, valid_labeled_df):
        """Test validation fails with too many invalid labels (-99)."""
        df = valid_labeled_df.copy()
        # Make 50% of labels invalid
        df.loc[:len(df)//2, 'label_h5'] = -99

        result = validate_labeled_data(
            df,
            horizons=[5, 20],
            symbol="TEST",
            max_invalid_label_pct=20.0
        )
        assert result.passed is False
        assert any("invalid" in e.lower() for e in result.errors)

    def test_validate_labels_for_ga_valid(self, valid_labeled_df):
        """Test GA input validation passes with valid data."""
        result = validate_labels_for_ga(
            valid_labeled_df,
            horizons=[5, 20],
            symbol="TEST"
        )
        assert result.passed is True

    def test_validate_labels_for_ga_missing_initial_labels(self, valid_features_df):
        """Test GA input validation fails without initial labels."""
        result = validate_labels_for_ga(
            valid_features_df,
            horizons=[5, 20],
            symbol="TEST"
        )
        assert result.passed is False

    def test_validate_labels_for_splits_valid(self, valid_labeled_df):
        """Test splits input validation passes with valid data."""
        result = validate_labels_for_splits(
            valid_labeled_df,
            horizons=[5, 20],
            symbol="TEST"
        )
        assert result.passed is True

    def test_validate_labels_for_splits_unsorted(self, valid_labeled_df):
        """Test splits input validation fails with unsorted datetime."""
        df = valid_labeled_df.sample(frac=1).reset_index(drop=True)  # Shuffle
        result = validate_labels_for_splits(df, horizons=[5, 20], symbol="TEST")
        assert result.passed is False


# --- Integration Tests ---

class TestValidatorIntegration:
    """Integration tests for validator chain."""

    def test_validation_chain_passes(self, valid_ohlcv_df):
        """Test that valid data passes through the complete validation chain."""
        # Stage 2 output validation
        result1 = validate_cleaned_data(valid_ohlcv_df, symbol="TEST")
        assert result1.passed is True

        # Stage 3 input validation
        result2 = validate_cleaned_data_for_features(valid_ohlcv_df, symbol="TEST")
        assert result2.passed is True

    def test_features_to_labels_chain(self, valid_features_df):
        """Test features -> labeling validation chain."""
        # Stage 3 output
        result1 = validate_feature_output(valid_features_df, symbol="TEST")
        assert result1.passed is True

        # Stage 4 input
        result2 = validate_features_for_labeling(valid_features_df, symbol="TEST")
        assert result2.passed is True

    def test_labels_to_splits_chain(self, valid_labeled_df):
        """Test labels -> splits validation chain."""
        horizons = [5, 20]

        # Stage 6 output
        result1 = validate_labeled_data(valid_labeled_df, horizons=horizons, symbol="TEST")
        assert result1.passed is True

        # Stage 7 input
        result2 = validate_labels_for_splits(valid_labeled_df, horizons=horizons, symbol="TEST")
        assert result2.passed is True

    def test_fail_fast_propagation(self, valid_ohlcv_df):
        """Test that validation failures properly propagate with clear errors."""
        # Create invalid data
        df = valid_ohlcv_df.copy()
        df.loc[0, 'high'] = df.loc[0, 'low'] - 100  # OHLC violation

        result = validate_cleaned_data(df, symbol="TEST")
        assert result.passed is False

        # Should raise with descriptive error when called
        with pytest.raises(ValueError) as exc_info:
            result.raise_if_failed()

        assert "high < low" in str(exc_info.value)
        assert "TEST" in str(exc_info.value)
