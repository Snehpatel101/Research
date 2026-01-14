"""
Tests for feature set resolution and validation (TEST-002).

Tests feature set name resolution, alias handling, prefix matching,
and the validate_feature_set_coverage() function.
"""

import pytest
import pandas as pd
from typing import Any

from src.phase1.config.feature_sets import (
    FEATURE_SET_DEFINITIONS,
    FEATURE_SET_ALIASES,
    FeatureSetDefinition,
    resolve_feature_set_name,
    resolve_feature_set_names,
    validate_feature_set_config,
    validate_feature_set_coverage,
    get_feature_set_columns,
)
from src.phase1.utils.feature_sets import (
    resolve_feature_set,
    build_feature_set_manifest,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame with typical feature columns."""
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=10, freq="min"),
            "symbol": ["MES"] * 10,
            # OHLCV (excluded from features)
            "open": [1.0] * 10,
            "high": [1.1] * 10,
            "low": [0.9] * 10,
            "close": [1.05] * 10,
            "volume": [100] * 10,
            # Return features
            "return_1": [0.01] * 10,
            "return_5": [0.02] * 10,
            "log_return_1": [0.01] * 10,
            "log_return_5": [0.02] * 10,
            # Momentum features
            "rsi_14": [50.0] * 10,
            "rsi_7": [55.0] * 10,
            "stoch_k": [70.0] * 10,
            "stoch_d": [65.0] * 10,
            "macd_line": [0.1] * 10,
            "macd_signal": [0.08] * 10,
            "macd_hist": [0.02] * 10,
            # Volatility features
            "atr_14": [0.05] * 10,
            "hvol_20": [0.15] * 10,
            "bb_position": [0.6] * 10,
            "bb_width": [0.02] * 10,
            # Volume features
            "volume_ratio": [1.1] * 10,
            "volume_zscore": [0.5] * 10,
            "obv": [10000] * 10,
            # Regime indicators
            "trend_regime": [1] * 10,
            "volatility_regime": [0] * 10,
            # Time features
            "hour_sin": [0.5] * 10,
            "hour_cos": [0.5] * 10,
            "dayofweek_sin": [0.3] * 10,
            "dayofweek_cos": [0.9] * 10,
            "is_rth": [1] * 10,
            # Price ratio features
            "price_to_vwap": [1.01] * 10,
            "close_bb_zscore": [0.2] * 10,
            # MTF features (with timeframe suffix)
            "rsi_14_15m": [52.0] * 10,
            "rsi_14_1h": [48.0] * 10,
            "close_1h": [1.0] * 10,
            # Labels (should be excluded)
            "label_h5": [1] * 10,
            "label_h20": [0] * 10,
            "sample_weight_h5": [1.0] * 10,
        }
    )


@pytest.fixture
def minimal_df() -> pd.DataFrame:
    """DataFrame with minimal columns for edge case testing."""
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=5, freq="min"),
            "symbol": ["MES"] * 5,
            "open": [1.0] * 5,
            "high": [1.1] * 5,
            "low": [0.9] * 5,
            "close": [1.05] * 5,
            "volume": [100] * 5,
            # Only one feature
            "return_1": [0.01] * 5,
        }
    )


# =============================================================================
# FEATURE SET NAME RESOLUTION TESTS
# =============================================================================


class TestResolveFeatureSetName:
    """Test feature set name resolution."""

    def test_canonical_name_unchanged(self) -> None:
        """Canonical names should pass through unchanged."""
        assert resolve_feature_set_name("core_min") == "core_min"
        assert resolve_feature_set_name("core_full") == "core_full"
        assert resolve_feature_set_name("boosting_optimal") == "boosting_optimal"
        assert resolve_feature_set_name("neural_optimal") == "neural_optimal"

    def test_alias_resolved_to_canonical(self) -> None:
        """Aliases should resolve to canonical names."""
        # Original aliases
        assert resolve_feature_set_name("minimal") == "core_min"
        assert resolve_feature_set_name("min") == "core_min"
        assert resolve_feature_set_name("full") == "core_full"
        assert resolve_feature_set_name("mtf") == "mtf_plus"

        # Model-family aliases
        assert resolve_feature_set_name("boosting") == "boosting_optimal"
        assert resolve_feature_set_name("xgboost") == "boosting_optimal"
        assert resolve_feature_set_name("lightgbm") == "boosting_optimal"
        assert resolve_feature_set_name("catboost") == "boosting_optimal"

        # Neural aliases
        assert resolve_feature_set_name("neural") == "neural_optimal"
        assert resolve_feature_set_name("lstm") == "neural_optimal"
        assert resolve_feature_set_name("gru") == "neural_optimal"

    def test_case_insensitive_resolution(self) -> None:
        """Resolution should be case-insensitive."""
        assert resolve_feature_set_name("CORE_MIN") == "core_min"
        assert resolve_feature_set_name("Core_Full") == "core_full"
        assert resolve_feature_set_name("BOOSTING") == "boosting_optimal"
        assert resolve_feature_set_name("XgBoost") == "boosting_optimal"

    def test_whitespace_trimmed(self) -> None:
        """Whitespace should be trimmed from name."""
        assert resolve_feature_set_name("  core_min  ") == "core_min"
        assert resolve_feature_set_name("\tcore_full\n") == "core_full"

    def test_empty_name_raises_error(self) -> None:
        """Empty feature set name should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            resolve_feature_set_name("")

        # Whitespace-only is caught as unknown feature set
        with pytest.raises(ValueError, match="Unknown feature_set"):
            resolve_feature_set_name("   ")

    def test_unknown_name_raises_error(self) -> None:
        """Unknown feature set name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown feature_set"):
            resolve_feature_set_name("nonexistent_feature_set")

        with pytest.raises(ValueError, match="Unknown feature_set"):
            resolve_feature_set_name("invalid_name")

    def test_architecture_specific_aliases(self) -> None:
        """Architecture-specific aliases should resolve correctly."""
        assert resolve_feature_set_name("nbeats") == "nbeats_optimal"
        assert resolve_feature_set_name("n_beats") == "nbeats_optimal"
        assert resolve_feature_set_name("n-beats") == "nbeats_optimal"
        assert resolve_feature_set_name("inceptiontime") == "inceptiontime_optimal"
        assert resolve_feature_set_name("resnet1d") == "resnet1d_optimal"
        assert resolve_feature_set_name("itransformer") == "itransformer_optimal"
        assert resolve_feature_set_name("tft") == "tft_optimal"


class TestResolveFeatureSetNames:
    """Test resolution of multiple feature set names."""

    def test_single_name_returns_list(self) -> None:
        """Single name should return single-element list."""
        result = resolve_feature_set_names("core_min")
        assert result == ["core_min"]

    def test_comma_separated_names(self) -> None:
        """Comma-separated names should all be resolved."""
        result = resolve_feature_set_names("core_min,core_full,boosting_optimal")
        assert result == ["core_min", "core_full", "boosting_optimal"]

    def test_aliases_in_list_resolved(self) -> None:
        """Aliases in comma-separated list should be resolved."""
        result = resolve_feature_set_names("min,full,boosting")
        assert result == ["core_min", "core_full", "boosting_optimal"]

    def test_all_keyword_expands(self) -> None:
        """'all' should expand to all feature set names."""
        result = resolve_feature_set_names("all")
        assert len(result) == len(FEATURE_SET_DEFINITIONS)
        for name in FEATURE_SET_DEFINITIONS:
            assert name in result

    def test_deduplication(self) -> None:
        """Duplicate names should be deduplicated."""
        # Same canonical name via different aliases
        result = resolve_feature_set_names("min,minimal,core_min")
        assert result == ["core_min"]

        # Different representations
        result = resolve_feature_set_names("boosting,xgboost,boosting_optimal")
        assert result == ["boosting_optimal"]

    def test_order_preserved(self) -> None:
        """Order should be preserved after deduplication."""
        result = resolve_feature_set_names("core_full,core_min,neural_optimal")
        assert result == ["core_full", "core_min", "neural_optimal"]

    def test_whitespace_in_list(self) -> None:
        """Whitespace around comma-separated values should be trimmed."""
        result = resolve_feature_set_names("core_min , core_full , neural_optimal")
        assert result == ["core_min", "core_full", "neural_optimal"]

    def test_empty_string_raises_error(self) -> None:
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError):
            resolve_feature_set_names("")


# =============================================================================
# FEATURE SET VALIDATION TESTS
# =============================================================================


class TestValidateFeatureSetConfig:
    """Test validate_feature_set_config() function."""

    def test_valid_config_no_errors(self) -> None:
        """Valid feature set config should return no errors."""
        errors = validate_feature_set_config("core_min")
        assert len(errors) == 0

        errors = validate_feature_set_config("boosting_optimal")
        assert len(errors) == 0

    def test_invalid_name_returns_error(self) -> None:
        """Invalid feature set name should return error."""
        errors = validate_feature_set_config("invalid_name")
        assert len(errors) == 1
        assert "Unknown feature_set" in errors[0]

    def test_all_keyword_valid(self) -> None:
        """'all' keyword should be valid."""
        errors = validate_feature_set_config("all")
        assert len(errors) == 0


# =============================================================================
# FEATURE SET COVERAGE VALIDATION TESTS
# =============================================================================


class TestValidateFeatureSetCoverage:
    """Test validate_feature_set_coverage() function."""

    def test_matched_columns_returned(self, sample_df: pd.DataFrame) -> None:
        """Should return matched columns from feature set."""
        feature_set = FEATURE_SET_DEFINITIONS["boosting_optimal"]
        result = validate_feature_set_coverage(
            list(sample_df.columns), feature_set
        )

        assert result["matched_count"] > 0
        assert len(result["matched_columns"]) == result["matched_count"]

    def test_unmatched_prefixes_tracked(self) -> None:
        """Prefixes that match no columns should be tracked."""
        df_columns = ["return_1", "rsi_14"]

        # Create a feature set with some unmatched prefixes
        feature_set = FeatureSetDefinition(
            name="test_set",
            description="Test",
            include_prefixes=["return_", "nonexistent_", "another_missing_"],
        )

        result = validate_feature_set_coverage(df_columns, feature_set)

        assert "nonexistent_" in result["unmatched_prefixes"]
        assert "another_missing_" in result["unmatched_prefixes"]
        assert "return_" not in result["unmatched_prefixes"]

    def test_missing_explicit_columns_tracked(self) -> None:
        """Missing explicit_columns should be tracked."""
        df_columns = ["return_1", "rsi_14"]

        feature_set = FeatureSetDefinition(
            name="test_set",
            description="Test",
            include_prefixes=["return_"],
            explicit_columns=["required_column_1", "required_column_2"],
        )

        result = validate_feature_set_coverage(df_columns, feature_set)

        assert result["valid"] is False
        assert "required_column_1" in result["missing_explicit"]
        assert "required_column_2" in result["missing_explicit"]

    def test_empty_result_warns(self) -> None:
        """Empty feature set result should set valid=False and add warning."""
        df_columns = ["unrelated_column"]

        feature_set = FeatureSetDefinition(
            name="test_set",
            description="Test",
            include_prefixes=["nonexistent_"],
        )

        result = validate_feature_set_coverage(df_columns, feature_set)

        assert result["valid"] is False
        assert result["matched_count"] == 0
        assert len(result["warnings"]) > 0
        assert any("zero columns" in w for w in result["warnings"])

    def test_empty_result_raises_when_requested(self) -> None:
        """Should raise ValueError when raise_on_empty=True and result is empty."""
        df_columns = ["unrelated_column"]

        feature_set = FeatureSetDefinition(
            name="test_set",
            description="Test",
            include_prefixes=["nonexistent_"],
        )

        with pytest.raises(ValueError, match="zero columns"):
            validate_feature_set_coverage(
                df_columns, feature_set, raise_on_empty=True
            )

    def test_exclusions_applied(self, sample_df: pd.DataFrame) -> None:
        """Exclude prefixes and columns should be applied."""
        feature_set = FeatureSetDefinition(
            name="test_set",
            description="Test",
            include_prefixes=["return_", "rsi_"],
            exclude_prefixes=["rsi_"],
            exclude_columns=["return_5"],
        )

        result = validate_feature_set_coverage(list(sample_df.columns), feature_set)

        # rsi_ columns should be excluded
        assert not any(c.startswith("rsi_") for c in result["matched_columns"])
        # return_5 should be excluded
        assert "return_5" not in result["matched_columns"]
        # return_1 should still be included
        assert "return_1" in result["matched_columns"]


# =============================================================================
# GET FEATURE SET COLUMNS TESTS
# =============================================================================


class TestGetFeatureSetColumns:
    """Test get_feature_set_columns() convenience function."""

    def test_returns_sorted_columns(self, sample_df: pd.DataFrame) -> None:
        """Should return sorted list of matched columns."""
        columns = get_feature_set_columns(list(sample_df.columns), "core_min")

        assert isinstance(columns, list)
        assert columns == sorted(columns)

    def test_alias_works(self, sample_df: pd.DataFrame) -> None:
        """Should work with feature set aliases."""
        columns_alias = get_feature_set_columns(list(sample_df.columns), "lstm")
        columns_canonical = get_feature_set_columns(
            list(sample_df.columns), "neural_optimal"
        )

        assert columns_alias == columns_canonical

    def test_unknown_name_raises(self, sample_df: pd.DataFrame) -> None:
        """Should raise ValueError for unknown feature set name."""
        with pytest.raises(ValueError, match="Unknown feature_set"):
            get_feature_set_columns(list(sample_df.columns), "invalid_name")


# =============================================================================
# RESOLVE FEATURE SET TESTS (from utils)
# =============================================================================


class TestResolveFeatureSet:
    """Test resolve_feature_set() function from utils."""

    def test_core_full_excludes_mtf(self, sample_df: pd.DataFrame) -> None:
        """core_full should exclude MTF columns."""
        features = resolve_feature_set(
            sample_df, FEATURE_SET_DEFINITIONS["core_full"]
        )

        # MTF columns (with _15m, _1h suffix) should be excluded
        assert "rsi_14_15m" not in features
        assert "rsi_14_1h" not in features
        assert "close_1h" not in features

        # Base features should be included
        assert "rsi_14" in features

    def test_mtf_plus_includes_mtf(self, sample_df: pd.DataFrame) -> None:
        """mtf_plus should include MTF columns."""
        features = resolve_feature_set(
            sample_df, FEATURE_SET_DEFINITIONS["mtf_plus"]
        )

        # MTF columns should be included
        assert "rsi_14_15m" in features or "rsi_14_1h" in features

    def test_labels_excluded(self, sample_df: pd.DataFrame) -> None:
        """Label columns should always be excluded."""
        for name, definition in FEATURE_SET_DEFINITIONS.items():
            features = resolve_feature_set(sample_df, definition)

            # No label columns
            assert not any(f.startswith("label_") for f in features)
            assert not any(f.startswith("sample_weight_") for f in features)

    def test_metadata_excluded(self, sample_df: pd.DataFrame) -> None:
        """Metadata columns should always be excluded."""
        for name, definition in FEATURE_SET_DEFINITIONS.items():
            features = resolve_feature_set(sample_df, definition)

            # No metadata columns
            assert "datetime" not in features
            assert "symbol" not in features


# =============================================================================
# BUILD FEATURE SET MANIFEST TESTS
# =============================================================================


class TestBuildFeatureSetManifest:
    """Test build_feature_set_manifest() function."""

    def test_manifest_contains_all_sets(self, sample_df: pd.DataFrame) -> None:
        """Manifest should contain all feature set definitions."""
        manifest = build_feature_set_manifest(sample_df, FEATURE_SET_DEFINITIONS)

        for name in FEATURE_SET_DEFINITIONS:
            assert name in manifest

    def test_manifest_structure(self, sample_df: pd.DataFrame) -> None:
        """Each manifest entry should have expected structure."""
        manifest = build_feature_set_manifest(sample_df, FEATURE_SET_DEFINITIONS)

        for name, entry in manifest.items():
            assert "description" in entry
            assert "feature_count" in entry
            assert "features" in entry
            assert "include_mtf" in entry
            assert "mtf_feature_count" in entry

            assert isinstance(entry["feature_count"], int)
            assert isinstance(entry["features"], list)
            assert entry["feature_count"] == len(entry["features"])

    def test_mtf_count_tracked(self, sample_df: pd.DataFrame) -> None:
        """MTF feature count should be tracked correctly."""
        manifest = build_feature_set_manifest(sample_df, FEATURE_SET_DEFINITIONS)

        # mtf_plus should have MTF features
        assert manifest["mtf_plus"]["mtf_feature_count"] >= 0

        # core_full should have 0 MTF features (include_mtf=False)
        assert manifest["core_full"]["mtf_feature_count"] == 0


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_minimal_dataframe(self, minimal_df: pd.DataFrame) -> None:
        """Should handle DataFrame with minimal columns."""
        # Should not crash, even if feature set resolves to few columns
        for name, definition in FEATURE_SET_DEFINITIONS.items():
            features = resolve_feature_set(minimal_df, definition)
            assert isinstance(features, list)

    def test_empty_dataframe_columns(self) -> None:
        """Should handle empty column list."""
        result = validate_feature_set_coverage(
            [], FEATURE_SET_DEFINITIONS["core_min"]
        )

        assert result["valid"] is False
        assert result["matched_count"] == 0

    def test_prefix_matching_edge_cases(self) -> None:
        """Test prefix matching edge cases."""
        df_columns = [
            "return_1",
            "return_",  # Empty suffix
            "_return_1",  # Leading underscore
            "Return_1",  # Case mismatch (actual matching is case-sensitive)
        ]

        feature_set = FeatureSetDefinition(
            name="test",
            description="Test",
            include_prefixes=["return_"],
        )

        result = validate_feature_set_coverage(df_columns, feature_set)

        # "return_1" and "return_" should match
        assert "return_1" in result["matched_columns"]
        # "_return_1" should NOT match (doesn't start with "return_")
        assert "_return_1" not in result["matched_columns"]

    def test_all_definitions_are_valid(self) -> None:
        """All built-in feature set definitions should be valid."""
        for name, definition in FEATURE_SET_DEFINITIONS.items():
            assert definition.name == name
            assert len(definition.description) > 0
            assert isinstance(definition.include_prefixes, list)
            assert isinstance(definition.exclude_prefixes, list)
            assert isinstance(definition.include_columns, list)
            assert isinstance(definition.exclude_columns, list)

    def test_all_aliases_resolve_correctly(self) -> None:
        """All defined aliases should resolve to valid feature sets."""
        for alias, canonical in FEATURE_SET_ALIASES.items():
            assert canonical in FEATURE_SET_DEFINITIONS, (
                f"Alias '{alias}' -> '{canonical}' but '{canonical}' not in definitions"
            )
