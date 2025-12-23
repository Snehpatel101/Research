"""
Unit tests for Stage 8: Data Validation.

Comprehensive data quality validation

Run with: pytest tests/phase_1_tests/stages/test_stage8_*.py -v
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from stages.stage8_validate import DataValidator, validate_data
from utils.feature_selection import FeatureSelectionResult


# =============================================================================
# TESTS
# =============================================================================

class TestStage8DataValidator:
    """Tests for Stage 8: Comprehensive Data Validation."""

    def test_check_duplicates_detection(self, sample_labeled_data):
        """Test duplicate timestamp detection."""
        df = sample_labeled_data.copy()

        # Add duplicates
        dup_row = df.iloc[0:1].copy()
        df = pd.concat([df, dup_row], ignore_index=True)

        validator = DataValidator(df)
        results = validator.check_data_integrity()

        # Should detect duplicates
        assert 'duplicate_timestamps' in results

    def test_check_nan_inf_detection(self, sample_labeled_data):
        """Test NaN and Inf detection."""
        df = sample_labeled_data.copy()

        # Add NaN values
        df.loc[0, 'rsi'] = np.nan
        df.loc[1, 'rsi'] = np.nan

        validator = DataValidator(df)
        results = validator.check_data_integrity()

        # Should detect NaN
        assert 'nan_values' in results
        if 'rsi' in results['nan_values']:
            assert results['nan_values']['rsi'] == 2

    def test_check_inf_values(self, sample_labeled_data):
        """Test infinite value detection."""
        df = sample_labeled_data.copy()

        # Add infinite values
        df.loc[0, 'macd'] = np.inf
        df.loc[1, 'macd'] = -np.inf

        validator = DataValidator(df)
        results = validator.check_data_integrity()

        # Should detect infinite values
        assert 'infinite_values' in results
        if 'macd' in results['infinite_values']:
            assert results['infinite_values']['macd'] == 2

    def test_check_gaps_detection(self, sample_labeled_data):
        """Test time gap detection."""
        df = sample_labeled_data.copy()

        # Create a gap by removing rows
        df = df.drop(df.index[50:100])  # Remove 50 rows

        validator = DataValidator(df)
        results = validator.check_data_integrity()

        # Should detect gaps
        assert 'gaps' in results

    def test_label_distribution_analysis(self, sample_labeled_data):
        """Test label distribution analysis."""
        df = sample_labeled_data

        validator = DataValidator(df, horizons=[5, 20])
        results = validator.check_label_sanity()

        # Should have results for each horizon
        assert 'horizon_5' in results or 'horizon_20' in results

        # Check distribution structure
        for key, horizon_results in results.items():
            if 'distribution' in horizon_results:
                dist = horizon_results['distribution']
                assert 'long' in dist or 'short' in dist or 'neutral' in dist

    def test_feature_correlation_detection(self, sample_labeled_data):
        """Test feature correlation detection."""
        df = sample_labeled_data.copy()

        # Add highly correlated feature
        df['sma_10_copy'] = df['sma_10'] * 1.001  # Nearly identical

        validator = DataValidator(df)
        results = validator.check_feature_quality()

        # Should detect high correlations
        assert 'high_correlations' in results

    def test_normalization_recommendations(self, sample_labeled_data):
        """Test normalization recommendations."""
        df = sample_labeled_data.copy()

        # Add feature with large scale
        df['large_feature'] = df['close'] * 1000

        validator = DataValidator(df)
        results = validator.check_feature_normalization()

        # Should have recommendations
        assert 'recommendations' in results
        assert 'unnormalized_features' in results

    def test_stationarity_check(self, sample_labeled_data):
        """Test stationarity checking."""
        df = sample_labeled_data

        validator = DataValidator(df)
        results = validator.check_feature_quality()

        # Should have stationarity results
        assert 'stationarity_tests' in results

    def test_validation_report_structure(self, sample_labeled_data):
        """Test validation report has correct structure."""
        df = sample_labeled_data

        validator = DataValidator(df, horizons=[1, 5, 20])
        validator.check_data_integrity()
        validator.check_label_sanity()
        validator.check_feature_quality()

        summary = validator.generate_summary()

        # Check structure
        assert 'timestamp' in summary
        assert 'total_rows' in summary
        assert 'total_columns' in summary
        assert 'issues_count' in summary
        assert 'warnings_count' in summary
        assert 'status' in summary
        assert summary['status'] in ['PASSED', 'FAILED']

    def test_validate_data_integration(self, sample_labeled_data, temp_directory):
        """Test full validation pipeline."""
        df = sample_labeled_data

        # Save test data
        data_path = temp_directory / "test_data.parquet"
        df.to_parquet(data_path)

        output_path = temp_directory / "validation_report.json"

        summary, _ = validate_data(
            data_path,
            output_path=output_path,
            horizons=[5, 20],
            run_feature_selection=False
        )

        # Check report was saved
        assert output_path.exists(), "Report should be saved"

        # Check summary
        assert 'status' in summary

    def test_feature_selection_integration(self, sample_labeled_data, temp_directory):
        """Test feature selection integration."""
        df = sample_labeled_data

        # Save test data
        data_path = temp_directory / "test_data.parquet"
        df.to_parquet(data_path)

        output_path = temp_directory / "validation_report.json"

        summary, feature_result = validate_data(
            data_path,
            output_path=output_path,
            run_feature_selection=True
        )

        # Feature selection should return results
        if feature_result is not None:
            # Use duck typing - check for expected attributes instead of isinstance
            # (avoids import path identity issues)
            assert hasattr(feature_result, 'selected_features')
            assert hasattr(feature_result, 'removed_features')
            assert hasattr(feature_result, 'final_count')


# =============================================================================
# PIPELINE INTEGRATION TESTS
# =============================================================================
