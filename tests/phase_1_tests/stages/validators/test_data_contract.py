"""Tests for data contract validation."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.phase1.stages.validation.data_contract import (
    validate_ohlcv_schema,
    validate_labels,
    filter_invalid_labels,
    get_dataset_fingerprint,
    validate_feature_lookahead,
    summarize_label_distribution,
    DataContract,
    REQUIRED_OHLCV,
    VALID_LABELS,
    INVALID_LABEL_SENTINEL,
)


class TestOHLCVValidation:
    """Tests for OHLCV schema validation."""

    def test_valid_ohlcv_passes(self):
        """Valid OHLCV data should pass validation."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=10, freq='5min'),
            'open': [100.0] * 10,
            'high': [102.0] * 10,
            'low': [98.0] * 10,
            'close': [101.0] * 10,
            'volume': [1000] * 10
        })

        # Should not raise
        validate_ohlcv_schema(df, stage="test")

    def test_missing_columns_fails(self):
        """Missing required columns should fail."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=10, freq='5min'),
            'open': [100.0] * 10,
            # Missing high, low, close, volume
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_ohlcv_schema(df, stage="test")

    def test_high_less_than_low_fails(self):
        """High < low should fail validation."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=10, freq='5min'),
            'open': [100.0] * 10,
            'high': [98.0] * 10,  # Wrong: lower than low
            'low': [102.0] * 10,
            'close': [101.0] * 10,
            'volume': [1000] * 10
        })

        with pytest.raises(ValueError, match="high < low"):
            validate_ohlcv_schema(df, stage="test")

    def test_empty_dataframe_fails(self):
        """Empty DataFrame should fail validation."""
        df = pd.DataFrame({
            'datetime': pd.Series([], dtype='datetime64[ns]'),
            'open': pd.Series([], dtype='float64'),
            'high': pd.Series([], dtype='float64'),
            'low': pd.Series([], dtype='float64'),
            'close': pd.Series([], dtype='float64'),
            'volume': pd.Series([], dtype='float64')
        })

        with pytest.raises(ValueError, match="DataFrame is empty"):
            validate_ohlcv_schema(df, stage="test")

    def test_duplicate_timestamps_fail(self):
        """Duplicate timestamps should fail validation."""
        dates = pd.date_range('2024-01-01', periods=10, freq='5min')
        # Create duplicate by repeating first timestamp
        dates_with_dup = dates.tolist()
        dates_with_dup[1] = dates_with_dup[0]

        df = pd.DataFrame({
            'datetime': dates_with_dup,
            'open': [100.0] * 10,
            'high': [102.0] * 10,
            'low': [98.0] * 10,
            'close': [101.0] * 10,
            'volume': [1000] * 10
        })

        with pytest.raises(ValueError, match="duplicate timestamps"):
            validate_ohlcv_schema(df, stage="test")

    def test_non_monotonic_timestamps_fail(self):
        """Non-monotonic timestamps should fail validation."""
        dates = pd.date_range('2024-01-01', periods=10, freq='5min').tolist()
        # Swap two dates to break monotonicity
        dates[3], dates[4] = dates[4], dates[3]

        df = pd.DataFrame({
            'datetime': dates,
            'open': [100.0] * 10,
            'high': [102.0] * 10,
            'low': [98.0] * 10,
            'close': [101.0] * 10,
            'volume': [1000] * 10
        })

        with pytest.raises(ValueError, match="monotonically increasing"):
            validate_ohlcv_schema(df, stage="test")

    def test_non_positive_volume_fails(self):
        """Zero or negative volume should fail validation."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=10, freq='5min'),
            'open': [100.0] * 10,
            'high': [102.0] * 10,
            'low': [98.0] * 10,
            'close': [101.0] * 10,
            'volume': [1000, 0, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
        })

        with pytest.raises(ValueError, match="Non-positive values in 'volume'"):
            validate_ohlcv_schema(df, stage="test")

    def test_high_less_than_open_fails(self):
        """High < open should fail validation."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=10, freq='5min'),
            'open': [105.0] * 10,  # Higher than high
            'high': [102.0] * 10,
            'low': [98.0] * 10,
            'close': [101.0] * 10,
            'volume': [1000] * 10
        })

        with pytest.raises(ValueError, match="high < open"):
            validate_ohlcv_schema(df, stage="test")

    def test_low_greater_than_close_fails(self):
        """Low > close should fail validation."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=10, freq='5min'),
            'open': [100.0] * 10,
            'high': [102.0] * 10,
            'low': [101.5] * 10,  # Higher than close
            'close': [101.0] * 10,
            'volume': [1000] * 10
        })

        with pytest.raises(ValueError, match="low > close"):
            validate_ohlcv_schema(df, stage="test")


class TestLabelValidation:
    """Tests for label validation."""

    def test_valid_labels_pass(self):
        """Valid labels (-1, 0, 1) should pass."""
        df = pd.DataFrame({
            'label_h5': [-1, 0, 1, 1, -1],
            'label_h20': [1, 1, 0, -1, 0]
        })

        validate_labels(df, ['label_h5', 'label_h20'])

    def test_invalid_label_sentinel_allowed(self):
        """The -99 sentinel should be allowed but flagged."""
        df = pd.DataFrame({
            'label_h5': [-1, 0, 1, -99, -1],
        })

        # Should not raise - sentinel is allowed
        validate_labels(df, ['label_h5'])

    def test_invalid_label_values_fail(self):
        """Invalid label values (not -1, 0, 1, -99) should fail."""
        df = pd.DataFrame({
            'label_h5': [-1, 0, 1, 5, -1],  # 5 is invalid
        })

        with pytest.raises(ValueError, match="Invalid label values"):
            validate_labels(df, ['label_h5'])

    def test_missing_label_column_fails(self):
        """Missing label column should fail."""
        df = pd.DataFrame({
            'label_h5': [-1, 0, 1, 1, -1],
        })

        with pytest.raises(ValueError, match="Label column 'label_h20' not found"):
            validate_labels(df, ['label_h5', 'label_h20'])

    def test_float_labels_converted(self):
        """Float labels that are valid integers should work."""
        df = pd.DataFrame({
            'label_h5': [-1.0, 0.0, 1.0, -99.0, -1.0],
        })

        # Should not raise
        validate_labels(df, ['label_h5'])

    def test_nan_labels_ignored(self):
        """NaN labels should be ignored in validation."""
        df = pd.DataFrame({
            'label_h5': [-1, 0, np.nan, 1, -1],
        })

        # Should not raise - NaN is dropped before validation
        validate_labels(df, ['label_h5'])


class TestFilterInvalidLabels:
    """Tests for filtering -99 invalid labels."""

    def test_filters_invalid_labels(self):
        """Should remove rows with -99 in any label column."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=5, freq='5min'),
            'close': [100.0] * 5,
            'label_h5': [1, -99, 0, -1, 1],
            'label_h20': [0, 1, -99, 1, -1]
        })

        result = filter_invalid_labels(df, ['label_h5', 'label_h20'])

        # Rows 1 and 2 should be removed (indices 1 and 2 have -99)
        assert len(result) == 3
        assert -99 not in result['label_h5'].values
        assert -99 not in result['label_h20'].values

    def test_no_invalid_labels_unchanged(self):
        """DataFrame with no -99 values should be unchanged."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=5, freq='5min'),
            'label_h5': [1, 0, 0, -1, 1],
            'label_h20': [0, 1, -1, 1, -1]
        })

        result = filter_invalid_labels(df, ['label_h5', 'label_h20'])

        assert len(result) == 5

    def test_all_invalid_labels_empty_result(self):
        """All -99 values should result in empty DataFrame."""
        df = pd.DataFrame({
            'label_h5': [-99, -99, -99],
        })

        result = filter_invalid_labels(df, ['label_h5'])

        assert len(result) == 0

    def test_missing_column_ignored(self):
        """Missing label columns should be ignored."""
        df = pd.DataFrame({
            'label_h5': [1, -99, 0],
        })

        # Should not raise even though label_h20 doesn't exist
        result = filter_invalid_labels(df, ['label_h5', 'label_h20'])

        assert len(result) == 2


class TestDatasetFingerprint:
    """Tests for dataset fingerprinting."""

    def test_fingerprint_contains_expected_keys(self):
        """Fingerprint should contain all expected metadata."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=10, freq='5min'),
            'close': [100.0] * 10,
            'volume': [1000] * 10
        })

        fingerprint = get_dataset_fingerprint(df)

        assert fingerprint['n_rows'] == 10
        assert fingerprint['n_cols'] == 3
        assert 'columns' in fingerprint
        assert 'datetime_min' in fingerprint
        assert 'datetime_max' in fingerprint
        assert 'schema_hash' in fingerprint

    def test_fingerprint_columns_sorted(self):
        """Column list in fingerprint should be sorted."""
        df = pd.DataFrame({
            'z_col': [1],
            'a_col': [2],
            'm_col': [3]
        })

        fingerprint = get_dataset_fingerprint(df)

        assert fingerprint['columns'] == ['a_col', 'm_col', 'z_col']

    def test_fingerprint_no_datetime_column(self):
        """Fingerprint should handle missing datetime column."""
        df = pd.DataFrame({
            'close': [100.0] * 10,
        })

        fingerprint = get_dataset_fingerprint(df)

        assert fingerprint['datetime_min'] is None
        assert fingerprint['datetime_max'] is None


class TestFeatureLookahead:
    """Tests for lookahead validation."""

    def test_no_overlap_passes(self):
        """Non-overlapping feature and label columns should pass."""
        df = pd.DataFrame({
            'feature_a': [1.0] * 5,
            'feature_b': [2.0] * 5,
            'label_h5': [1, 0, -1, 1, 0]
        })

        # Should not raise
        validate_feature_lookahead(df, ['feature_a', 'feature_b'], ['label_h5'])

    def test_overlapping_columns_fail(self):
        """Overlapping feature and label columns should fail."""
        df = pd.DataFrame({
            'feature_a': [1.0] * 5,
            'label_h5': [1, 0, -1, 1, 0]
        })

        with pytest.raises(ValueError, match="overlap"):
            validate_feature_lookahead(df, ['feature_a', 'label_h5'], ['label_h5'])

    def test_missing_feature_columns_fail(self):
        """Missing feature columns should fail."""
        df = pd.DataFrame({
            'feature_a': [1.0] * 5,
            'label_h5': [1, 0, -1, 1, 0]
        })

        with pytest.raises(ValueError, match="Feature columns not in DataFrame"):
            validate_feature_lookahead(df, ['feature_a', 'feature_b'], ['label_h5'])


class TestSummarizeLabelDistribution:
    """Tests for label distribution summary."""

    def test_basic_distribution(self):
        """Should correctly summarize label distribution."""
        df = pd.DataFrame({
            'label_h5': [1, 1, 0, -1, -1],  # 2 long, 1 neutral, 2 short
        })

        summary = summarize_label_distribution(df, ['label_h5'])

        assert 'label_h5' in summary
        assert summary['label_h5']['total_valid'] == 5
        assert summary['label_h5']['total_invalid'] == 0
        assert summary['label_h5']['distribution'][1] == 2
        assert summary['label_h5']['distribution'][0] == 1
        assert summary['label_h5']['distribution'][-1] == 2

    def test_excludes_invalid_sentinel(self):
        """Should exclude -99 from valid count."""
        df = pd.DataFrame({
            'label_h5': [1, -99, 0, -99, -1],  # 3 valid, 2 invalid
        })

        summary = summarize_label_distribution(df, ['label_h5'])

        assert summary['label_h5']['total_valid'] == 3
        assert summary['label_h5']['total_invalid'] == 2

    def test_missing_column_skipped(self):
        """Missing columns should be skipped."""
        df = pd.DataFrame({
            'label_h5': [1, 0, -1],
        })

        summary = summarize_label_distribution(df, ['label_h5', 'label_h20'])

        assert 'label_h5' in summary
        assert 'label_h20' not in summary


class TestDataContractConstants:
    """Tests for module-level constants."""

    def test_required_ohlcv_columns(self):
        """Required OHLCV columns should be defined."""
        assert 'datetime' in REQUIRED_OHLCV
        assert 'open' in REQUIRED_OHLCV
        assert 'high' in REQUIRED_OHLCV
        assert 'low' in REQUIRED_OHLCV
        assert 'close' in REQUIRED_OHLCV
        assert 'volume' in REQUIRED_OHLCV

    def test_valid_labels(self):
        """Valid labels should be -1, 0, 1."""
        assert VALID_LABELS == {-1, 0, 1}

    def test_invalid_sentinel(self):
        """Invalid label sentinel should be -99."""
        assert INVALID_LABEL_SENTINEL == -99

    def test_datacontract_class_attributes(self):
        """DataContract class should have consistent values."""
        contract = DataContract()
        assert contract.INVALID_LABEL_SENTINEL == INVALID_LABEL_SENTINEL
