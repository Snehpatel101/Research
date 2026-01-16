"""
Tests for main FeatureStore class.

Tests cover:
- FeatureStore initialization
- Feature storage and retrieval
- Point-in-time queries
- Version management integration
- Lineage tracking integration
- Statistics and validation
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from src.feature_store import (
    FeatureStore,
    FeatureStoreError,
    FeatureNotFoundError,
    FeatureIntegrityError,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_store_dir(tmp_path):
    """Create a temporary store directory."""
    store_dir = tmp_path / "feature_store"
    store_dir.mkdir()
    return store_dir


@pytest.fixture
def sample_features():
    """Create sample feature DataFrame."""
    np.random.seed(42)
    n_rows = 100

    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n_rows, freq="5min"),
        "feature_1": np.random.randn(n_rows),
        "feature_2": np.random.randn(n_rows),
        "feature_3": np.random.randn(n_rows),
        "close": 4500 + np.cumsum(np.random.randn(n_rows) * 0.5),
    })


@pytest.fixture
def feature_store(temp_store_dir):
    """Create a FeatureStore instance."""
    return FeatureStore(cache_dir=temp_store_dir)


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestFeatureStoreInit:
    """Tests for FeatureStore initialization."""

    def test_create_store(self, temp_store_dir):
        """Test creating a feature store."""
        store = FeatureStore(cache_dir=temp_store_dir)

        assert store.cache_dir == temp_store_dir
        assert store.validate_on_read is True

    def test_creates_directories(self, tmp_path):
        """Test that store creates necessary directories."""
        store_dir = tmp_path / "new_store"
        store = FeatureStore(cache_dir=store_dir)

        assert store_dir.exists()
        assert (store_dir / "lineage").exists()
        assert (store_dir / "versions").exists()

    def test_custom_lineage_dir(self, temp_store_dir, tmp_path):
        """Test using custom lineage directory."""
        lineage_dir = tmp_path / "custom_lineage"
        store = FeatureStore(
            cache_dir=temp_store_dir,
            lineage_dir=lineage_dir,
        )

        assert store.lineage_dir == lineage_dir
        assert lineage_dir.exists()

    def test_disable_validation(self, temp_store_dir):
        """Test disabling validation on read."""
        store = FeatureStore(
            cache_dir=temp_store_dir,
            validate_on_read=False,
        )

        assert store.validate_on_read is False


# =============================================================================
# FEATURE STORAGE TESTS
# =============================================================================


class TestFeatureStorage:
    """Tests for storing features."""

    def test_put_features(self, feature_store, sample_features):
        """Test storing features."""
        version = feature_store.put_features(
            df=sample_features,
            symbol="MES",
            feature_set="core_features",
        )

        assert version == "1.0.0"
        assert feature_store.has_features(symbol="MES", feature_set="core_features")

    def test_put_features_with_lineage(self, feature_store, sample_features):
        """Test storing features with lineage information."""
        version = feature_store.put_features(
            df=sample_features,
            symbol="MES",
            feature_set="core_features",
            lineage={
                "raw_path": "/data/raw/MES_1m.parquet",
                "config": {"enable_wavelets": True},
                "transformations": [
                    {
                        "type": "feature_engineering",
                        "name": "core_indicators",
                    }
                ],
            },
        )

        assert version is not None

    def test_put_features_custom_version(self, feature_store, sample_features):
        """Test storing features with custom version."""
        version = feature_store.put_features(
            df=sample_features,
            symbol="MES",
            feature_set="core_features",
            version="2.5.0",
        )

        assert version == "2.5.0"

    def test_put_empty_dataframe_raises(self, feature_store):
        """Test that storing empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Cannot store empty"):
            feature_store.put_features(
                df=empty_df,
                symbol="MES",
                feature_set="core_features",
            )

    def test_version_auto_increment(self, feature_store, sample_features):
        """Test automatic version increment."""
        v1 = feature_store.put_features(
            df=sample_features,
            symbol="MES",
            feature_set="core",
        )

        # Same data = patch bump
        v2 = feature_store.put_features(
            df=sample_features,
            symbol="MES",
            feature_set="core",
        )

        assert v1 == "1.0.0"
        assert v2 == "1.0.1"


# =============================================================================
# FEATURE RETRIEVAL TESTS
# =============================================================================


class TestFeatureRetrieval:
    """Tests for retrieving features."""

    def test_get_features(self, feature_store, sample_features):
        """Test getting stored features."""
        feature_store.put_features(
            df=sample_features,
            symbol="MES",
            feature_set="core",
        )

        df = feature_store.get_features(symbol="MES", feature_set="core")

        assert len(df) == len(sample_features)
        assert list(df.columns) == list(sample_features.columns)

    def test_get_features_specific_version(self, feature_store, sample_features):
        """Test getting specific version."""
        feature_store.put_features(
            df=sample_features,
            symbol="MES",
            feature_set="core",
            version="1.0.0",
        )
        feature_store.put_features(
            df=sample_features,
            symbol="MES",
            feature_set="core",
            version="2.0.0",
        )

        df = feature_store.get_features(
            symbol="MES",
            feature_set="core",
            version="1.0.0",
        )

        assert df is not None

    def test_get_features_latest(self, feature_store, sample_features):
        """Test getting latest version by default."""
        feature_store.put_features(
            df=sample_features,
            symbol="MES",
            feature_set="core",
            version="1.0.0",
        )
        feature_store.put_features(
            df=sample_features,
            symbol="MES",
            feature_set="core",
            version="2.0.0",
        )

        # Default is latest
        df = feature_store.get_features(symbol="MES", feature_set="core")
        assert df is not None

    def test_get_features_not_found(self, feature_store):
        """Test getting nonexistent features raises error."""
        with pytest.raises(FeatureNotFoundError):
            feature_store.get_features(
                symbol="NONEXISTENT",
                feature_set="core",
            )

    def test_get_features_specific_columns(self, feature_store, sample_features):
        """Test getting specific columns."""
        feature_store.put_features(
            df=sample_features,
            symbol="MES",
            feature_set="core",
        )

        df = feature_store.get_features(
            symbol="MES",
            feature_set="core",
            columns=["datetime", "feature_1"],
        )

        assert list(df.columns) == ["datetime", "feature_1"]

    def test_get_features_with_metadata(self, feature_store, sample_features):
        """Test getting features with metadata."""
        feature_store.put_features(
            df=sample_features,
            symbol="MES",
            feature_set="core",
        )

        df, metadata = feature_store.get_features_with_metadata(
            symbol="MES",
            feature_set="core",
        )

        assert df is not None
        assert metadata.row_count == len(sample_features)


# =============================================================================
# POINT-IN-TIME TESTS
# =============================================================================


class TestPointInTime:
    """Tests for point-in-time queries."""

    def test_get_features_as_of(self, feature_store, sample_features):
        """Test point-in-time feature retrieval."""
        feature_store.put_features(
            df=sample_features,
            symbol="MES",
            feature_set="core",
        )

        # Get data as of midpoint
        midpoint = sample_features["datetime"].iloc[len(sample_features) // 2]

        df = feature_store.get_features_as_of(
            symbol="MES",
            feature_set="core",
            as_of_date=midpoint,
        )

        # Should only have data before midpoint
        assert len(df) < len(sample_features)
        assert df["datetime"].max() < midpoint

    def test_get_features_as_of_no_datetime_column(self, feature_store):
        """Test that missing datetime column raises error."""
        df_no_datetime = pd.DataFrame({
            "feature_1": [1, 2, 3],
            "feature_2": [4, 5, 6],
        })

        feature_store.put_features(
            df=df_no_datetime,
            symbol="MES",
            feature_set="no_datetime",
        )

        with pytest.raises(ValueError, match="No datetime column"):
            feature_store.get_features_as_of(
                symbol="MES",
                feature_set="no_datetime",
                as_of_date=datetime.now(),
            )


# =============================================================================
# VERSION MANAGEMENT TESTS
# =============================================================================


class TestVersionManagement:
    """Tests for version management integration."""

    def test_list_versions(self, feature_store, sample_features):
        """Test listing versions."""
        feature_store.put_features(
            df=sample_features,
            symbol="MES",
            feature_set="core",
            version="1.0.0",
        )
        feature_store.put_features(
            df=sample_features,
            symbol="MES",
            feature_set="core",
            version="1.1.0",
        )

        versions = feature_store.list_versions("core")

        assert "1.0.0" in versions
        assert "1.1.0" in versions

    def test_get_version_info(self, feature_store, sample_features):
        """Test getting version info."""
        feature_store.put_features(
            df=sample_features,
            symbol="MES",
            feature_set="core",
            version="1.0.0",
            description="Initial release",
        )

        info = feature_store.get_version_info("core", "1.0.0")

        assert info is not None
        assert str(info.version) == "1.0.0"

    def test_list_feature_sets(self, feature_store, sample_features):
        """Test listing feature sets."""
        feature_store.put_features(
            df=sample_features,
            symbol="MES",
            feature_set="core",
        )
        feature_store.put_features(
            df=sample_features,
            symbol="MES",
            feature_set="mtf",
        )

        feature_sets = feature_store.list_feature_sets()

        assert "core" in feature_sets
        assert "mtf" in feature_sets

    def test_list_symbols(self, feature_store, sample_features):
        """Test listing symbols."""
        feature_store.put_features(
            df=sample_features,
            symbol="MES",
            feature_set="core",
        )
        feature_store.put_features(
            df=sample_features,
            symbol="MGC",
            feature_set="core",
        )

        symbols = feature_store.list_symbols("core")

        assert "MES" in symbols
        assert "MGC" in symbols


# =============================================================================
# LINEAGE TESTS
# =============================================================================


class TestLineageIntegration:
    """Tests for lineage tracking integration."""

    def test_get_lineage(self, feature_store, sample_features):
        """Test getting lineage for stored features."""
        feature_store.put_features(
            df=sample_features,
            symbol="MES",
            feature_set="core",
            version="1.0.0",
            lineage={
                "raw_path": "/data/raw/MES_1m.parquet",
                "config": {"indicators": ["rsi", "macd"]},
            },
        )

        lineage = feature_store.get_lineage(
            symbol="MES",
            feature_set="core",
            version="1.0.0",
        )

        # Lineage should be tracked
        assert lineage is not None


# =============================================================================
# INVALIDATION TESTS
# =============================================================================


class TestInvalidation:
    """Tests for cache invalidation."""

    def test_invalidate_specific_version(self, feature_store, sample_features):
        """Test invalidating specific version."""
        feature_store.put_features(
            df=sample_features,
            symbol="MES",
            feature_set="core",
            version="1.0.0",
        )

        assert feature_store.has_features(
            symbol="MES", feature_set="core", version="1.0.0"
        )

        result = feature_store.invalidate(
            symbol="MES", feature_set="core", version="1.0.0"
        )

        assert result is True
        assert not feature_store.has_features(
            symbol="MES", feature_set="core", version="1.0.0"
        )

    def test_invalidate_all_versions(self, feature_store, sample_features):
        """Test invalidating all versions for a symbol."""
        feature_store.put_features(
            df=sample_features,
            symbol="MES",
            feature_set="core",
            version="1.0.0",
        )
        feature_store.put_features(
            df=sample_features,
            symbol="MES",
            feature_set="core",
            version="2.0.0",
        )

        result = feature_store.invalidate(symbol="MES", feature_set="core")

        assert result is True


# =============================================================================
# STATISTICS AND VALIDATION TESTS
# =============================================================================


class TestStatisticsAndValidation:
    """Tests for statistics and validation."""

    def test_get_stats(self, feature_store, sample_features):
        """Test getting store statistics."""
        feature_store.put_features(
            df=sample_features,
            symbol="MES",
            feature_set="core",
        )

        stats = feature_store.get_stats()

        assert "cache" in stats
        assert "versions" in stats
        assert "feature_sets" in stats

    def test_validate_integrity(self, feature_store, sample_features):
        """Test validating feature integrity."""
        feature_store.put_features(
            df=sample_features,
            symbol="MES",
            feature_set="core",
        )

        result = feature_store.validate_integrity()

        assert "valid" in result
        assert "invalid" in result
        assert len(result["valid"]) >= 1

    def test_cleanup(self, feature_store, sample_features):
        """Test cleanup of old entries."""
        feature_store.put_features(
            df=sample_features,
            symbol="MES",
            feature_set="core",
        )

        # Cleanup with future max_age (should keep everything)
        count = feature_store.cleanup(max_age_days=365)

        # Newly created entries should not be cleaned up
        assert count == 0


# =============================================================================
# VALIDATION MODULE INTEGRATION TESTS
# =============================================================================


class TestValidationIntegration:
    """Test that feature store is accessible from validation module."""

    def test_import_from_validation(self):
        """Test importing feature store from validation module."""
        from src.validation import (
            FeatureStore,
            FeatureCache,
            FeatureNotFoundError,
            FeatureIntegrityError,
            FeatureStoreError,
            CacheMetadata,
            SemanticVersion,
            VersionInfo,
            VersionManager,
            LineageTracker,
            FeatureLineage,
            DataSource,
            Transformation,
            TransformationType,
            compute_file_checksum,
            compute_dataframe_checksum,
            compute_schema_hash,
            compute_config_hash,
        )

        assert FeatureStore is not None
        assert FeatureCache is not None
        assert SemanticVersion is not None
        assert callable(compute_schema_hash)

    def test_import_cpcv_from_validation(self):
        """Test importing CPCV from validation module."""
        from src.validation import (
            CombinatorialPurgedCV,
            CPCVConfig,
            CPCVResult,
            CPCVPathResult,
            create_cpcv,
        )

        assert CombinatorialPurgedCV is not None
        assert callable(create_cpcv)

    def test_import_pbo_from_validation(self):
        """Test importing PBO from validation module."""
        from src.validation import (
            PBOConfig,
            PBOResult,
            compute_pbo,
            compute_pbo_from_returns,
            pbo_gate,
            analyze_overfitting_risk,
        )

        assert PBOConfig is not None
        assert callable(compute_pbo)
        assert callable(pbo_gate)
