"""
Tests for src.stages.datasets.validators module.

Validates model-ready validation for TimeSeriesDataContainer.
"""
import numpy as np
import pandas as pd
import pytest

from src.stages.datasets import TimeSeriesDataContainer, validate_model_ready, ValidationResult


def _create_test_df(n_rows=100, horizon=20, has_issues=False):
    """Create test DataFrame with optional issues for validation testing."""
    np.random.seed(42)
    df = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n_rows, freq="5min"),
        "symbol": ["MES"] * n_rows,
        "open": np.random.randn(n_rows).clip(-5, 5),
        "high": np.random.randn(n_rows).clip(-5, 5),
        "low": np.random.randn(n_rows).clip(-5, 5),
        "close": np.random.randn(n_rows).clip(-5, 5),
        "volume": np.abs(np.random.randn(n_rows)) * 100,
        "feature_a": np.random.randn(n_rows).clip(-5, 5),
        "feature_b": np.random.randn(n_rows).clip(-5, 5),
        "feature_c": np.zeros(n_rows) if has_issues else np.random.randn(n_rows).clip(-5, 5),
        f"label_h{horizon}": np.random.choice([-1, 0, 1], n_rows),
        f"sample_weight_h{horizon}": np.random.uniform(0.5, 1.5, n_rows),
    })
    if has_issues:
        df.loc[0, "feature_a"] = np.nan
        df.loc[1, "feature_b"] = np.inf
    return df


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_empty_result_is_valid(self):
        result = ValidationResult()
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_add_error_makes_invalid(self):
        result = ValidationResult()
        result.add_error("test error")
        assert result.is_valid is False
        assert "test error" in result.errors

    def test_add_warning_keeps_valid(self):
        result = ValidationResult()
        result.add_warning("test warning")
        assert result.is_valid is True
        assert "test warning" in result.warnings

    def test_to_dict(self):
        result = ValidationResult()
        result.add_error("err")
        result.add_warning("warn")
        result.metadata["key"] = "value"
        d = result.to_dict()
        assert d["is_valid"] is False
        assert d["error_count"] == 1
        assert d["warning_count"] == 1
        assert "err" in d["errors"]
        assert "warn" in d["warnings"]
        assert d["metadata"]["key"] == "value"

    def test_repr(self):
        result = ValidationResult()
        assert "VALID" in repr(result)
        result.add_error("err")
        assert "INVALID" in repr(result)


class TestValidateModelReady:
    """Tests for validate_model_ready function."""

    def test_valid_container(self):
        train_df = _create_test_df(n_rows=200, horizon=20)
        val_df = _create_test_df(n_rows=50, horizon=20)
        test_df = _create_test_df(n_rows=50, horizon=20)

        container = TimeSeriesDataContainer.from_dataframes(
            train_df=train_df, val_df=val_df, test_df=test_df, horizon=20
        )
        result = validate_model_ready(container, seq_len=10)

        # Should be valid (no blocking errors)
        # May have warnings for class imbalance, etc.
        assert isinstance(result, ValidationResult)
        assert "horizon" in result.metadata
        assert "n_features" in result.metadata

    def test_detects_nan_in_features(self):
        train_df = _create_test_df(n_rows=100, horizon=20, has_issues=True)
        container = TimeSeriesDataContainer.from_dataframes(train_df=train_df, horizon=20)
        result = validate_model_ready(container)
        # Should detect NaN error
        assert any("NaN" in e for e in result.errors)

    def test_detects_inf_in_features(self):
        train_df = _create_test_df(n_rows=100, horizon=20, has_issues=True)
        container = TimeSeriesDataContainer.from_dataframes(train_df=train_df, horizon=20)
        result = validate_model_ready(container)
        # Should detect Inf error
        assert any("Inf" in e for e in result.errors)

    def test_detects_constant_features(self):
        train_df = _create_test_df(n_rows=100, horizon=20, has_issues=True)
        container = TimeSeriesDataContainer.from_dataframes(train_df=train_df, horizon=20)
        result = validate_model_ready(container)
        # Should warn about constant feature_c
        assert any("constant" in w.lower() for w in result.warnings)

    def test_detects_class_imbalance(self):
        train_df = _create_test_df(n_rows=100, horizon=20)
        # Make extreme class imbalance
        train_df["label_h20"] = [1] * 95 + [-1] * 3 + [0] * 2
        container = TimeSeriesDataContainer.from_dataframes(train_df=train_df, horizon=20)
        result = validate_model_ready(container)
        # Should warn about class imbalance
        assert any("class" in w.lower() for w in result.warnings)

    def test_detects_invalid_labels(self):
        train_df = _create_test_df(n_rows=100, horizon=20)
        train_df.loc[0:4, "label_h20"] = -99  # Invalid labels
        container = TimeSeriesDataContainer.from_dataframes(
            train_df=train_df, horizon=20, exclude_invalid_labels=False
        )
        result = validate_model_ready(container)
        # Should warn about invalid labels
        assert any("invalid" in w.lower() for w in result.warnings)

    def test_detects_unexpected_label_values(self):
        train_df = _create_test_df(n_rows=100, horizon=20)
        train_df.loc[0, "label_h20"] = 5  # Invalid label value
        container = TimeSeriesDataContainer.from_dataframes(train_df=train_df, horizon=20)
        result = validate_model_ready(container)
        # Should error on unexpected label value
        assert any("Unexpected" in e for e in result.errors)

    def test_detects_feature_mismatch(self):
        # Create DataFrames with different feature sets
        train_df = _create_test_df(n_rows=100, horizon=20)
        val_df = _create_test_df(n_rows=50, horizon=20)
        # Add extra feature to val_df that doesn't exist in train
        val_df["extra_feature"] = np.random.randn(len(val_df))

        # Create containers separately with their own feature detection
        # Then merge manually to simulate a mismatch scenario
        container = TimeSeriesDataContainer.from_dataframes(
            train_df=train_df, val_df=val_df, horizon=20
        )

        # Manually update val's feature list to include the extra (simulating configuration error)
        # but the DataFrame already has it, so feature check will pass
        container.splits["val"].feature_columns = list(container.splits["train"].feature_columns) + ["extra_feature"]

        result = validate_model_ready(container)
        # Should error on extra features in val
        assert any("Extra" in e for e in result.errors), f"Errors: {result.errors}"

    def test_metadata_populated(self):
        train_df = _create_test_df(n_rows=100, horizon=20)
        container = TimeSeriesDataContainer.from_dataframes(train_df=train_df, horizon=20)
        result = validate_model_ready(container, seq_len=10)

        assert "validated_at" in result.metadata
        assert result.metadata["horizon"] == 20
        assert "split_sizes" in result.metadata
        assert result.metadata["split_sizes"]["train"] == 100


class TestSequenceValidation:
    """Tests for sequence-related validation."""

    def test_detects_datetime_gaps(self):
        train_df = _create_test_df(n_rows=100, horizon=20)
        # Introduce gap by modifying datetime
        train_df.loc[50:, "datetime"] = pd.date_range("2024-01-02", periods=50, freq="5min")
        container = TimeSeriesDataContainer.from_dataframes(train_df=train_df, horizon=20)
        result = validate_model_ready(container, seq_len=10)
        # Should warn about datetime gaps
        assert any("gap" in w.lower() for w in result.warnings)

    def test_warns_on_few_sequences(self):
        # Create very small dataset
        train_df = _create_test_df(n_rows=20, horizon=20)
        container = TimeSeriesDataContainer.from_dataframes(train_df=train_df, horizon=20)
        # With seq_len=60, no valid sequences from 20 rows
        result = validate_model_ready(container, seq_len=60)
        # Should warn about insufficient sequences
        assert any("sequence" in w.lower() for w in result.warnings)


class TestWeightValidation:
    """Tests for sample weight validation."""

    def test_detects_weight_out_of_range(self):
        # Need enough rows to avoid sequence warning dominating
        train_df = _create_test_df(n_rows=200, horizon=20)
        train_df["sample_weight_h20"] = 0.1  # Below valid range (0.4 - 1.6)
        container = TimeSeriesDataContainer.from_dataframes(train_df=train_df, horizon=20)
        result = validate_model_ready(container, seq_len=10)  # Small seq_len to avoid sequence warning
        # Should warn about weight below range
        # Check for either "weight" or "Weight" in warnings
        assert any("Weight" in w or "weight" in w for w in result.warnings), f"Warnings: {result.warnings}"
