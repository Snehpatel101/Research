"""
Tests for labeling preconditions and prerequisites (TEST-003).

Tests validate_labeling_prerequisites() function which checks:
- ATR column dependency (required for dynamic barrier calculation)
- OHLC column presence
- Clear error messages with context
"""

import pytest
import pandas as pd
import numpy as np
from typing import Any

from src.pipeline._phase1_impl.stages.labeling.run import (
    validate_labeling_prerequisites,
    REQUIRED_LABELING_COLUMNS,
    REQUIRED_OHLC_COLUMNS,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def valid_df() -> pd.DataFrame:
    """DataFrame with all required columns for labeling."""
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=100, freq="min"),
            "open": np.random.uniform(100, 101, 100),
            "high": np.random.uniform(101, 102, 100),
            "low": np.random.uniform(99, 100, 100),
            "close": np.random.uniform(100, 101, 100),
            "volume": np.random.randint(100, 1000, 100),
            "atr_14": np.random.uniform(0.5, 1.5, 100),
            # Additional feature columns
            "rsi_14": np.random.uniform(30, 70, 100),
            "macd_hist": np.random.uniform(-0.5, 0.5, 100),
        }
    )


@pytest.fixture
def df_missing_atr() -> pd.DataFrame:
    """DataFrame missing the ATR column."""
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=100, freq="min"),
            "open": np.random.uniform(100, 101, 100),
            "high": np.random.uniform(101, 102, 100),
            "low": np.random.uniform(99, 100, 100),
            "close": np.random.uniform(100, 101, 100),
            "volume": np.random.randint(100, 1000, 100),
            # No atr_14 column
            "rsi_14": np.random.uniform(30, 70, 100),
        }
    )


@pytest.fixture
def df_missing_ohlc() -> pd.DataFrame:
    """DataFrame missing OHLC columns."""
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=100, freq="min"),
            # Missing: open, high
            "low": np.random.uniform(99, 100, 100),
            "close": np.random.uniform(100, 101, 100),
            "volume": np.random.randint(100, 1000, 100),
            "atr_14": np.random.uniform(0.5, 1.5, 100),
        }
    )


@pytest.fixture
def df_missing_all() -> pd.DataFrame:
    """DataFrame missing all required columns."""
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=100, freq="min"),
            "volume": np.random.randint(100, 1000, 100),
            # Missing: open, high, low, close, atr_14
            "rsi_14": np.random.uniform(30, 70, 100),
        }
    )


# =============================================================================
# BASIC VALIDATION TESTS
# =============================================================================


class TestValidateLabelingPrerequisites:
    """Test validate_labeling_prerequisites() function."""

    def test_valid_df_passes(self, valid_df: pd.DataFrame) -> None:
        """DataFrame with all required columns should pass validation."""
        # Should not raise
        validate_labeling_prerequisites(valid_df)

    def test_missing_atr_raises_valueerror(self, df_missing_atr: pd.DataFrame) -> None:
        """Missing ATR column should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_labeling_prerequisites(df_missing_atr)

        error_msg = str(exc_info.value)
        assert "atr_14" in error_msg
        assert "Missing required columns" in error_msg

    def test_missing_ohlc_raises_valueerror(self, df_missing_ohlc: pd.DataFrame) -> None:
        """Missing OHLC columns should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_labeling_prerequisites(df_missing_ohlc)

        error_msg = str(exc_info.value)
        # Should mention at least one missing OHLC column
        assert "open" in error_msg or "high" in error_msg
        assert "Missing required columns" in error_msg

    def test_missing_all_raises_valueerror(self, df_missing_all: pd.DataFrame) -> None:
        """Missing all required columns should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_labeling_prerequisites(df_missing_all)

        error_msg = str(exc_info.value)
        assert "Missing required columns" in error_msg


# =============================================================================
# ERROR MESSAGE TESTS
# =============================================================================


class TestErrorMessageContent:
    """Test error message content and context."""

    def test_error_includes_atr_explanation(
        self, df_missing_atr: pd.DataFrame
    ) -> None:
        """Error message should explain ATR dependency."""
        with pytest.raises(ValueError) as exc_info:
            validate_labeling_prerequisites(df_missing_atr)

        error_msg = str(exc_info.value)
        # Should mention ATR is required for barriers
        assert "ATR" in error_msg or "atr" in error_msg
        assert "barrier" in error_msg.lower() or "Stage 3" in error_msg

    def test_error_includes_symbol_context(
        self, df_missing_atr: pd.DataFrame
    ) -> None:
        """Error message should include symbol context when provided."""
        with pytest.raises(ValueError) as exc_info:
            validate_labeling_prerequisites(
                df_missing_atr, symbol="MES", timeframe="5min"
            )

        error_msg = str(exc_info.value)
        assert "MES" in error_msg
        assert "5min" in error_msg

    def test_error_includes_only_symbol(
        self, df_missing_atr: pd.DataFrame
    ) -> None:
        """Error message should handle symbol-only context."""
        with pytest.raises(ValueError) as exc_info:
            validate_labeling_prerequisites(df_missing_atr, symbol="MGC")

        error_msg = str(exc_info.value)
        assert "MGC" in error_msg

    def test_error_includes_only_timeframe(
        self, df_missing_atr: pd.DataFrame
    ) -> None:
        """Error message should handle timeframe-only context."""
        with pytest.raises(ValueError) as exc_info:
            validate_labeling_prerequisites(df_missing_atr, timeframe="15min")

        error_msg = str(exc_info.value)
        assert "15min" in error_msg

    def test_error_lists_all_missing_columns(
        self, df_missing_all: pd.DataFrame
    ) -> None:
        """Error message should list all missing columns."""
        with pytest.raises(ValueError) as exc_info:
            validate_labeling_prerequisites(df_missing_all)

        error_msg = str(exc_info.value)
        # Should contain a list of missing columns
        assert "[" in error_msg  # List formatting


# =============================================================================
# CUSTOM REQUIRED COLUMNS TESTS
# =============================================================================


class TestCustomRequiredColumns:
    """Test with custom required_cols parameter."""

    def test_custom_columns_validated(self, valid_df: pd.DataFrame) -> None:
        """Should validate custom required columns."""
        # Add a custom column to df
        valid_df["custom_indicator"] = 1.0

        # Should pass with custom columns
        validate_labeling_prerequisites(
            valid_df, required_cols=["close", "custom_indicator"]
        )

    def test_missing_custom_column_raises(self, valid_df: pd.DataFrame) -> None:
        """Missing custom column should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_labeling_prerequisites(
                valid_df, required_cols=["close", "nonexistent_column"]
            )

        error_msg = str(exc_info.value)
        assert "nonexistent_column" in error_msg

    def test_empty_custom_columns_passes(self, valid_df: pd.DataFrame) -> None:
        """Empty required_cols should pass for any DataFrame."""
        # Should not raise even for a minimal df
        validate_labeling_prerequisites(valid_df, required_cols=[])

    def test_default_columns_are_correct(self) -> None:
        """Default required columns should match constants."""
        expected_atr = ["atr_14"]
        expected_ohlc = ["open", "high", "low", "close"]

        assert REQUIRED_LABELING_COLUMNS == expected_atr
        assert set(REQUIRED_OHLC_COLUMNS) == set(expected_ohlc)


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dataframe_with_columns(self) -> None:
        """Empty DataFrame with correct columns should pass."""
        df = pd.DataFrame(
            columns=["open", "high", "low", "close", "atr_14"]
        )

        # Should not raise - columns exist even if df is empty
        validate_labeling_prerequisites(df)

    def test_case_sensitive_column_names(self, valid_df: pd.DataFrame) -> None:
        """Column name matching should be case-sensitive."""
        # Rename to wrong case
        df = valid_df.rename(columns={"atr_14": "ATR_14"})

        # Should fail - case matters
        with pytest.raises(ValueError) as exc_info:
            validate_labeling_prerequisites(df)

        assert "atr_14" in str(exc_info.value)

    def test_extra_columns_ignored(self, valid_df: pd.DataFrame) -> None:
        """Extra columns should not affect validation."""
        # Add many extra columns
        valid_df["extra_1"] = 1
        valid_df["extra_2"] = 2
        valid_df["extra_3"] = 3

        # Should still pass
        validate_labeling_prerequisites(valid_df)

    def test_column_with_nan_values_passes(self, valid_df: pd.DataFrame) -> None:
        """Columns with NaN values should pass validation (presence check only)."""
        # Set some NaN values
        valid_df.loc[0:10, "atr_14"] = np.nan

        # Should pass - we only check column presence, not values
        validate_labeling_prerequisites(valid_df)

    def test_column_order_irrelevant(self, valid_df: pd.DataFrame) -> None:
        """Column order should not affect validation."""
        # Reverse column order
        df = valid_df[valid_df.columns[::-1]]

        # Should still pass
        validate_labeling_prerequisites(df)


# =============================================================================
# INTEGRATION WITH LABELING WORKFLOW
# =============================================================================


class TestLabelingWorkflowIntegration:
    """Test integration with the labeling workflow."""

    def test_stage3_output_should_pass(self) -> None:
        """Simulated Stage 3 output should pass validation."""
        # Create DataFrame similar to Stage 3 (feature engineering) output
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01", periods=100, freq="min"),
                "open": np.random.uniform(100, 101, 100),
                "high": np.random.uniform(101, 102, 100),
                "low": np.random.uniform(99, 100, 100),
                "close": np.random.uniform(100, 101, 100),
                "volume": np.random.randint(100, 1000, 100),
                # Stage 3 should produce ATR
                "atr_14": np.random.uniform(0.5, 1.5, 100),
                # And other features
                "rsi_14": np.random.uniform(30, 70, 100),
                "macd_line": np.random.uniform(-1, 1, 100),
                "bb_position": np.random.uniform(0, 1, 100),
                "volume_ratio": np.random.uniform(0.8, 1.2, 100),
            }
        )

        # Should pass
        validate_labeling_prerequisites(df)

    def test_raw_ohlcv_should_fail(self) -> None:
        """Raw OHLCV data (before Stage 3) should fail validation."""
        # Raw data without ATR
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01", periods=100, freq="min"),
                "open": np.random.uniform(100, 101, 100),
                "high": np.random.uniform(101, 102, 100),
                "low": np.random.uniform(99, 100, 100),
                "close": np.random.uniform(100, 101, 100),
                "volume": np.random.randint(100, 1000, 100),
            }
        )

        # Should fail - no ATR
        with pytest.raises(ValueError) as exc_info:
            validate_labeling_prerequisites(df)

        error_msg = str(exc_info.value)
        assert "atr_14" in error_msg
        assert "Stage 3" in error_msg or "feature engineering" in error_msg.lower()


# =============================================================================
# REQUIRED COLUMNS CONSTANTS TESTS
# =============================================================================


class TestRequiredColumnsConstants:
    """Test that required column constants are correct."""

    def test_required_labeling_columns_contains_atr(self) -> None:
        """REQUIRED_LABELING_COLUMNS should contain atr_14."""
        assert "atr_14" in REQUIRED_LABELING_COLUMNS

    def test_required_ohlc_columns_complete(self) -> None:
        """REQUIRED_OHLC_COLUMNS should contain all OHLC columns."""
        expected = {"open", "high", "low", "close"}
        assert set(REQUIRED_OHLC_COLUMNS) == expected

    def test_combined_requirements(self) -> None:
        """Combined requirements should include ATR and OHLC."""
        combined = set(REQUIRED_LABELING_COLUMNS) | set(REQUIRED_OHLC_COLUMNS)
        expected = {"atr_14", "open", "high", "low", "close"}
        assert combined == expected
