"""
Unit tests for Data Integrity Validation.

Tests the data integrity validation module which checks:
- Duplicate timestamps
- NaN values
- Infinite values
- Time gaps
- Date range verification

Run with: pytest tests/phase_1_tests/validators/test_integrity_validator.py -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from stages.validators.integrity import (
    check_duplicate_timestamps,
    check_nan_values,
    check_infinite_values,
    analyze_time_gaps,
    verify_date_range,
    check_data_integrity,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def clean_ohlcv_df():
    """Create a clean OHLCV DataFrame with no issues."""
    np.random.seed(42)
    n = 200

    df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01 09:30', periods=n, freq='5min'),
        'symbol': 'MES',
        'open': 100 + np.random.randn(n) * 0.5,
        'high': 101 + np.random.randn(n) * 0.5,
        'low': 99 + np.random.randn(n) * 0.5,
        'close': 100 + np.random.randn(n) * 0.5,
        'volume': np.random.randint(100, 1000, n),
    })

    return df


@pytest.fixture
def multi_symbol_df():
    """Create a multi-symbol OHLCV DataFrame."""
    np.random.seed(42)
    n = 100

    dfs = []
    for symbol in ['MES', 'MGC']:
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01 09:30', periods=n, freq='5min'),
            'symbol': symbol,
            'open': 100 + np.random.randn(n) * 0.5,
            'high': 101 + np.random.randn(n) * 0.5,
            'low': 99 + np.random.randn(n) * 0.5,
            'close': 100 + np.random.randn(n) * 0.5,
            'volume': np.random.randint(100, 1000, n),
        })
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


@pytest.fixture
def df_with_duplicates(clean_ohlcv_df):
    """Create DataFrame with duplicate timestamps."""
    df = clean_ohlcv_df.copy()

    # Add duplicate rows
    dup_rows = df.iloc[0:3].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


@pytest.fixture
def df_with_nans(clean_ohlcv_df):
    """Create DataFrame with NaN values."""
    df = clean_ohlcv_df.copy()

    # Add NaN values
    df.loc[0, 'close'] = np.nan
    df.loc[1, 'close'] = np.nan
    df.loc[5, 'volume'] = np.nan

    return df


@pytest.fixture
def df_with_infs(clean_ohlcv_df):
    """Create DataFrame with infinite values."""
    df = clean_ohlcv_df.copy()

    # Add infinite values
    df.loc[0, 'close'] = np.inf
    df.loc[1, 'close'] = -np.inf
    df.loc[5, 'volume'] = np.inf

    return df


@pytest.fixture
def df_with_gaps(clean_ohlcv_df):
    """Create DataFrame with time gaps."""
    df = clean_ohlcv_df.copy()

    # Remove rows to create gaps
    df = df.drop(df.index[50:60]).reset_index(drop=True)  # 10 bars = 50min gap

    return df


# =============================================================================
# CHECK DUPLICATE TIMESTAMPS TESTS
# =============================================================================

class TestCheckDuplicateTimestamps:
    """Tests for check_duplicate_timestamps function."""

    def test_detects_duplicates_single_symbol(self, df_with_duplicates):
        """Test duplicate detection for single symbol."""
        issues_found = []

        result = check_duplicate_timestamps(df_with_duplicates, issues_found)

        assert 'MES' in result
        assert result['MES'] == 3  # We added 3 duplicate rows
        assert len(issues_found) == 1
        assert '3 duplicate' in issues_found[0].lower()

    def test_no_duplicates_clean_data(self, clean_ohlcv_df):
        """Test no false positives with clean data."""
        issues_found = []

        result = check_duplicate_timestamps(clean_ohlcv_df, issues_found)

        assert 'MES' in result
        assert result['MES'] == 0
        assert len(issues_found) == 0

    def test_multi_symbol_duplicate_detection(self, multi_symbol_df):
        """Test duplicate detection across multiple symbols."""
        # Add duplicates for one symbol only
        df = multi_symbol_df.copy()
        mes_rows = df[df['symbol'] == 'MES'].iloc[0:2].copy()
        df = pd.concat([df, mes_rows], ignore_index=True)

        issues_found = []
        result = check_duplicate_timestamps(df, issues_found)

        assert 'MES' in result
        assert 'MGC' in result
        assert result['MES'] == 2  # Duplicates added
        assert result['MGC'] == 0  # No duplicates

    def test_handles_no_symbol_column(self):
        """Test handling when symbol column is absent."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=10, freq='5min'),
            'close': [100] * 10,
        })

        # Add duplicate
        df = pd.concat([df, df.iloc[0:1]], ignore_index=True)

        issues_found = []
        result = check_duplicate_timestamps(df, issues_found)

        assert 'total' in result
        assert result['total'] == 1


# =============================================================================
# CHECK NAN VALUES TESTS
# =============================================================================

class TestCheckNanValues:
    """Tests for check_nan_values function."""

    def test_detects_nan_values(self, df_with_nans):
        """Test NaN detection in DataFrame."""
        issues_found = []

        result = check_nan_values(df_with_nans, issues_found)

        assert 'close' in result
        assert result['close'] == 2  # 2 NaN values in close
        assert 'volume' in result
        assert result['volume'] == 1
        assert len(issues_found) == 2

    def test_no_nans_clean_data(self, clean_ohlcv_df):
        """Test no false positives with clean data."""
        issues_found = []

        result = check_nan_values(clean_ohlcv_df, issues_found)

        assert len(result) == 0
        assert len(issues_found) == 0

    def test_reports_percentage(self, df_with_nans):
        """Test that percentage is included in issues."""
        issues_found = []

        check_nan_values(df_with_nans, issues_found)

        # Issues should contain percentage
        for issue in issues_found:
            assert '%' in issue

    def test_handles_all_nan_column(self, clean_ohlcv_df):
        """Test handling of column with all NaN values."""
        df = clean_ohlcv_df.copy()
        df['all_nan_col'] = np.nan

        issues_found = []
        result = check_nan_values(df, issues_found)

        assert 'all_nan_col' in result
        assert result['all_nan_col'] == len(df)


# =============================================================================
# CHECK INFINITE VALUES TESTS
# =============================================================================

class TestCheckInfiniteValues:
    """Tests for check_infinite_values function."""

    def test_detects_positive_inf(self, clean_ohlcv_df):
        """Test detection of positive infinity."""
        df = clean_ohlcv_df.copy()
        df.loc[0, 'close'] = np.inf

        issues_found = []
        result = check_infinite_values(df, issues_found)

        assert 'close' in result
        assert result['close'] == 1
        assert len(issues_found) == 1

    def test_detects_negative_inf(self, clean_ohlcv_df):
        """Test detection of negative infinity."""
        df = clean_ohlcv_df.copy()
        df.loc[0, 'close'] = -np.inf

        issues_found = []
        result = check_infinite_values(df, issues_found)

        assert 'close' in result
        assert result['close'] == 1

    def test_detects_mixed_inf(self, df_with_infs):
        """Test detection of mixed +inf and -inf."""
        issues_found = []

        result = check_infinite_values(df_with_infs, issues_found)

        assert 'close' in result
        assert result['close'] == 2  # 1 +inf, 1 -inf
        assert 'volume' in result
        assert result['volume'] == 1

    def test_no_infs_clean_data(self, clean_ohlcv_df):
        """Test no false positives with clean data."""
        issues_found = []

        result = check_infinite_values(clean_ohlcv_df, issues_found)

        assert len(result) == 0
        assert len(issues_found) == 0

    def test_ignores_non_numeric_columns(self, clean_ohlcv_df):
        """Test that non-numeric columns are ignored."""
        df = clean_ohlcv_df.copy()
        df['text_col'] = 'text'

        issues_found = []
        result = check_infinite_values(df, issues_found)

        # Should not fail on text column
        assert len(result) == 0


# =============================================================================
# ANALYZE TIME GAPS TESTS
# =============================================================================

class TestAnalyzeTimeGaps:
    """Tests for analyze_time_gaps function."""

    def test_detects_large_gaps(self, df_with_gaps):
        """Test detection of large time gaps."""
        result = analyze_time_gaps(df_with_gaps)

        assert len(result) > 0

        # Check structure
        gap_info = result[0]
        assert 'count' in gap_info
        assert 'median_gap' in gap_info
        assert 'max_gap' in gap_info

    def test_no_gaps_clean_data(self, clean_ohlcv_df):
        """Test with continuous data (no large gaps)."""
        result = analyze_time_gaps(clean_ohlcv_df)

        # May still report gaps due to 3x median threshold
        # but gap count should be reasonable
        if len(result) > 0:
            # All gaps should be small
            pass

    def test_multi_symbol_gap_analysis(self, multi_symbol_df):
        """Test gap analysis per symbol."""
        # Create gap in one symbol
        df = multi_symbol_df.copy()
        mes_mask = df['symbol'] == 'MES'
        mes_indices = df[mes_mask].index[50:60].tolist()
        df = df.drop(mes_indices).reset_index(drop=True)

        result = analyze_time_gaps(df)

        # Should have results per symbol
        symbols_with_gaps = [g.get('symbol') for g in result if 'symbol' in g]
        assert 'MES' in symbols_with_gaps

    def test_handles_unsorted_data(self, clean_ohlcv_df):
        """Test handling of unsorted data."""
        df = clean_ohlcv_df.copy()
        df = df.sample(frac=1, random_state=42)  # Shuffle

        # Should still work (function sorts internally)
        result = analyze_time_gaps(df)
        assert isinstance(result, list)


# =============================================================================
# VERIFY DATE RANGE TESTS
# =============================================================================

class TestVerifyDateRange:
    """Tests for verify_date_range function."""

    def test_returns_correct_range(self, clean_ohlcv_df):
        """Test correct date range calculation."""
        result = verify_date_range(clean_ohlcv_df)

        assert 'start' in result
        assert 'end' in result
        assert 'duration_days' in result
        assert 'total_bars' in result

        assert result['total_bars'] == len(clean_ohlcv_df)
        assert '2024-01-01' in result['start']

    def test_duration_calculation(self):
        """Test duration calculation in days."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=2880, freq='5min'),
            'close': [100] * 2880,
        })

        result = verify_date_range(df)

        # 2880 5-min bars = 10 days
        assert result['duration_days'] >= 9  # Allow for partial day

    def test_single_row(self):
        """Test with single row."""
        df = pd.DataFrame({
            'datetime': [pd.Timestamp('2024-01-01 09:30')],
            'close': [100],
        })

        result = verify_date_range(df)

        assert result['total_bars'] == 1
        assert result['duration_days'] == 0.0


# =============================================================================
# CHECK DATA INTEGRITY TESTS
# =============================================================================

class TestCheckDataIntegrity:
    """Tests for check_data_integrity function."""

    def test_returns_all_checks(self, clean_ohlcv_df):
        """Test that all check results are included."""
        issues_found = []

        result = check_data_integrity(clean_ohlcv_df, issues_found)

        assert 'duplicate_timestamps' in result
        assert 'nan_values' in result
        assert 'infinite_values' in result
        assert 'gaps' in result
        assert 'date_range' in result

    def test_clean_data_passes(self, clean_ohlcv_df):
        """Test that clean data has no issues."""
        issues_found = []

        check_data_integrity(clean_ohlcv_df, issues_found)

        assert len(issues_found) == 0

    def test_multiple_issues_detected(self, df_with_nans):
        """Test detection of multiple issues."""
        # Add more issues
        df = df_with_nans.copy()
        df.loc[10, 'close'] = np.inf

        # Add duplicate
        dup = df.iloc[0:1].copy()
        df = pd.concat([df, dup], ignore_index=True)

        issues_found = []
        check_data_integrity(df, issues_found)

        # Should have multiple issues
        assert len(issues_found) >= 3  # NaN + inf + duplicate

    def test_mutates_issues_list(self, df_with_nans):
        """Test that issues_found list is mutated."""
        issues_found = ['existing_issue']

        check_data_integrity(df_with_nans, issues_found)

        # Should append to existing list
        assert len(issues_found) > 1
        assert 'existing_issue' in issues_found


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestIntegrityValidatorEdgeCases:
    """Edge case tests for integrity validator."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({
            'datetime': pd.Series([], dtype='datetime64[ns]'),
            'symbol': pd.Series([], dtype='str'),
            'close': pd.Series([], dtype='float64'),
        })

        issues_found = []

        # Should not raise
        result = check_data_integrity(df, issues_found)
        assert result is not None

    def test_single_row_dataframe(self):
        """Test handling of single row DataFrame."""
        df = pd.DataFrame({
            'datetime': [pd.Timestamp('2024-01-01 09:30')],
            'symbol': ['MES'],
            'close': [100.0],
            'volume': [1000],
        })

        issues_found = []
        result = check_data_integrity(df, issues_found)

        assert result['date_range']['total_bars'] == 1
        assert len(issues_found) == 0

    def test_datetime_as_index(self):
        """Test handling when datetime is index."""
        df = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200],
        }, index=pd.date_range('2024-01-01', periods=3, freq='5min'))
        df.index.name = 'datetime'
        df = df.reset_index()

        issues_found = []
        result = check_data_integrity(df, issues_found)

        assert result['date_range']['total_bars'] == 3

    def test_mixed_dtypes(self):
        """Test handling of mixed data types."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=10, freq='5min'),
            'symbol': ['MES'] * 10,
            'close': [100.0] * 10,
            'volume': [1000] * 10,
            'flag': [True, False] * 5,  # Boolean
            'category': ['A', 'B'] * 5,  # String
        })

        issues_found = []
        result = check_data_integrity(df, issues_found)

        # Should handle mixed types without error
        assert result is not None

    def test_very_large_values(self):
        """Test handling of very large (but finite) values."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=10, freq='5min'),
            'symbol': ['MES'] * 10,
            'close': [1e15] * 10,  # Large but finite
            'volume': [1e12] * 10,
        })

        issues_found = []
        result = check_data_integrity(df, issues_found)

        # Large values are not issues by themselves
        inf_issues = [i for i in issues_found if 'inf' in i.lower()]
        assert len(inf_issues) == 0

    def test_negative_volume(self):
        """Test that negative values in volume are not flagged as integrity issues."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=10, freq='5min'),
            'symbol': ['MES'] * 10,
            'close': [100.0] * 10,
            'volume': [-100, 200, -300, 400, 500, 600, 700, 800, 900, 1000],
        })

        issues_found = []
        result = check_data_integrity(df, issues_found)

        # Negative values are not NaN or inf
        assert result['nan_values'] == {}
        assert result['infinite_values'] == {}
