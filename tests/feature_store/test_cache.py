"""
Tests for feature store caching module.

Tests cover:
- CacheMetadata creation and serialization
- FeatureCache operations (put, get, has, invalidate)
- Cache key computation
- Checksum validation
- Cache cleanup
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from src.feature_store.cache import (
    CacheMetadata,
    FeatureCache,
    compute_dataframe_checksum,
    compute_file_checksum,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "feature_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n_rows = 100

    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="5min"),
        "feature_1": np.random.randn(n_rows),
        "feature_2": np.random.randn(n_rows),
        "feature_3": np.random.randn(n_rows),
    })


@pytest.fixture
def feature_cache(temp_cache_dir):
    """Create a FeatureCache instance."""
    return FeatureCache(cache_dir=temp_cache_dir)


# =============================================================================
# CACHE METADATA TESTS
# =============================================================================


class TestCacheMetadata:
    """Tests for CacheMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating cache metadata."""
        metadata = CacheMetadata(
            cache_key="abc123",
            feature_set="core_features",
            version="1.0.0",
            symbol="MES",
            input_checksum="input_hash",
            config_hash="config_hash",
            row_count=1000,
            column_count=50,
            columns=["col1", "col2"],
            date_start="2024-01-01",
            date_end="2024-06-30",
        )

        assert metadata.cache_key == "abc123"
        assert metadata.feature_set == "core_features"
        assert metadata.row_count == 1000

    def test_to_dict(self):
        """Test serialization to dict."""
        metadata = CacheMetadata(
            cache_key="abc123",
            feature_set="core",
            version="1.0.0",
            symbol="MES",
            input_checksum="input",
            config_hash="config",
            row_count=100,
            column_count=10,
            columns=["a", "b"],
            date_start="2024-01-01",
            date_end="2024-01-31",
            parquet_checksum="parquet_hash",
        )

        d = metadata.to_dict()

        assert d["cache_key"] == "abc123"
        assert d["columns"] == ["a", "b"]
        assert "created_at" in d

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "cache_key": "xyz789",
            "feature_set": "mtf_features",
            "version": "2.0.0",
            "symbol": "MGC",
            "input_checksum": "input",
            "config_hash": "config",
            "row_count": 500,
            "column_count": 25,
            "columns": ["x", "y", "z"],
            "date_start": "2024-01-01",
            "date_end": "2024-03-31",
            "created_at": "2024-01-15T10:30:00",
            "parquet_checksum": "parquet",
            "compression": "snappy",
            "metadata": {"extra": "data"},
        }

        metadata = CacheMetadata.from_dict(data)

        assert metadata.cache_key == "xyz789"
        assert metadata.symbol == "MGC"
        assert metadata.columns == ["x", "y", "z"]

    def test_roundtrip_serialization(self):
        """Test to_dict/from_dict roundtrip."""
        original = CacheMetadata(
            cache_key="test",
            feature_set="core",
            version="1.0.0",
            symbol="MES",
            input_checksum="input",
            config_hash="config",
            row_count=100,
            column_count=10,
            columns=["a"],
            date_start="2024-01-01",
            date_end="2024-01-31",
            metadata={"key": "value"},
        )

        restored = CacheMetadata.from_dict(original.to_dict())

        assert restored.cache_key == original.cache_key
        assert restored.metadata == original.metadata


# =============================================================================
# FEATURE CACHE TESTS
# =============================================================================


class TestFeatureCache:
    """Tests for FeatureCache class."""

    def test_create_cache(self, temp_cache_dir):
        """Test creating a feature cache."""
        cache = FeatureCache(cache_dir=temp_cache_dir)

        assert cache.cache_dir == temp_cache_dir
        assert cache.compression == "snappy"
        assert cache.validate_on_read is True

    def test_compute_cache_key(self, feature_cache):
        """Test cache key computation."""
        key1 = feature_cache.compute_cache_key(
            input_checksum="input1",
            config_hash="config1",
            symbol="MES",
            version="1.0.0",
        )

        assert isinstance(key1, str)
        assert len(key1) == 32  # SHA256 truncated to 32 chars

    def test_cache_key_deterministic(self, feature_cache):
        """Test that cache key is deterministic."""
        key1 = feature_cache.compute_cache_key(
            input_checksum="input",
            config_hash="config",
            symbol="MES",
            version="1.0.0",
        )
        key2 = feature_cache.compute_cache_key(
            input_checksum="input",
            config_hash="config",
            symbol="MES",
            version="1.0.0",
        )

        assert key1 == key2

    def test_cache_key_changes_with_input(self, feature_cache):
        """Test that cache key changes when inputs change."""
        key1 = feature_cache.compute_cache_key(
            input_checksum="input1",
            config_hash="config",
            symbol="MES",
            version="1.0.0",
        )
        key2 = feature_cache.compute_cache_key(
            input_checksum="input2",
            config_hash="config",
            symbol="MES",
            version="1.0.0",
        )

        assert key1 != key2

    def test_has_returns_false_for_missing(self, feature_cache):
        """Test has() returns False for missing entries."""
        assert not feature_cache.has("core", "nonexistent_key")

    def test_put_and_has(self, feature_cache, sample_dataframe):
        """Test putting data and checking existence."""
        feature_cache.put(
            df=sample_dataframe,
            feature_set="test_features",
            version="1.0.0",
            symbol="MES",
            input_checksum="input_hash",
            config_hash="config_hash",
            date_start=datetime(2024, 1, 1),
            date_end=datetime(2024, 1, 31),
            cache_key="test_key",
        )

        assert feature_cache.has("test_features", "test_key")

    def test_put_and_get(self, feature_cache, sample_dataframe):
        """Test putting and getting data."""
        feature_cache.put(
            df=sample_dataframe,
            feature_set="core",
            version="1.0.0",
            symbol="MES",
            input_checksum="input",
            config_hash="config",
            date_start="2024-01-01",
            date_end="2024-01-31",
            cache_key="my_key",
        )

        result = feature_cache.get("core", "my_key")

        assert result is not None
        df, metadata = result
        assert len(df) == len(sample_dataframe)
        assert list(df.columns) == list(sample_dataframe.columns)
        assert metadata.row_count == len(sample_dataframe)

    def test_get_specific_columns(self, feature_cache, sample_dataframe):
        """Test getting specific columns only."""
        feature_cache.put(
            df=sample_dataframe,
            feature_set="core",
            version="1.0.0",
            symbol="MES",
            input_checksum="input",
            config_hash="config",
            date_start="2024-01-01",
            date_end="2024-01-31",
            cache_key="columns_test",
        )

        result = feature_cache.get("core", "columns_test", columns=["feature_1"])

        assert result is not None
        df, _ = result
        assert list(df.columns) == ["feature_1"]

    def test_get_missing_returns_none(self, feature_cache):
        """Test get() returns None for missing entries."""
        result = feature_cache.get("missing", "missing_key")
        assert result is None

    def test_invalidate(self, feature_cache, sample_dataframe):
        """Test invalidating a cache entry."""
        feature_cache.put(
            df=sample_dataframe,
            feature_set="core",
            version="1.0.0",
            symbol="MES",
            input_checksum="input",
            config_hash="config",
            date_start="2024-01-01",
            date_end="2024-01-31",
            cache_key="to_invalidate",
        )

        assert feature_cache.has("core", "to_invalidate")

        result = feature_cache.invalidate("core", "to_invalidate")

        assert result is True
        assert not feature_cache.has("core", "to_invalidate")

    def test_invalidate_missing_returns_false(self, feature_cache):
        """Test invalidating missing entry returns False."""
        result = feature_cache.invalidate("missing", "missing_key")
        assert result is False

    def test_list_entries(self, feature_cache, sample_dataframe):
        """Test listing cache entries."""
        # Add multiple entries
        for i in range(3):
            feature_cache.put(
                df=sample_dataframe,
                feature_set="core",
                version=f"1.0.{i}",
                symbol="MES",
                input_checksum=f"input_{i}",
                config_hash="config",
                date_start="2024-01-01",
                date_end="2024-01-31",
                cache_key=f"key_{i}",
            )

        entries = feature_cache.list_entries("core")

        assert len(entries) == 3
        assert all(isinstance(e, CacheMetadata) for e in entries)

    def test_list_entries_filter_by_feature_set(self, feature_cache, sample_dataframe):
        """Test listing entries filtered by feature set."""
        feature_cache.put(
            df=sample_dataframe,
            feature_set="core",
            version="1.0.0",
            symbol="MES",
            input_checksum="input",
            config_hash="config",
            date_start="2024-01-01",
            date_end="2024-01-31",
            cache_key="core_key",
        )
        feature_cache.put(
            df=sample_dataframe,
            feature_set="mtf",
            version="1.0.0",
            symbol="MES",
            input_checksum="input",
            config_hash="config",
            date_start="2024-01-01",
            date_end="2024-01-31",
            cache_key="mtf_key",
        )

        core_entries = feature_cache.list_entries("core")
        mtf_entries = feature_cache.list_entries("mtf")

        assert len(core_entries) == 1
        assert len(mtf_entries) == 1

    def test_get_cache_stats(self, feature_cache, sample_dataframe):
        """Test getting cache statistics."""
        feature_cache.put(
            df=sample_dataframe,
            feature_set="core",
            version="1.0.0",
            symbol="MES",
            input_checksum="input",
            config_hash="config",
            date_start="2024-01-01",
            date_end="2024-01-31",
            cache_key="stats_key",
        )

        stats = feature_cache.get_cache_stats()

        assert "total_entries" in stats
        assert "total_size_bytes" in stats
        assert "feature_sets" in stats
        assert stats["total_entries"] >= 1

    def test_invalidate_feature_set(self, feature_cache, sample_dataframe):
        """Test invalidating all entries for a feature set."""
        for i in range(3):
            feature_cache.put(
                df=sample_dataframe,
                feature_set="to_delete",
                version=f"1.0.{i}",
                symbol="MES",
                input_checksum=f"input_{i}",
                config_hash="config",
                date_start="2024-01-01",
                date_end="2024-01-31",
                cache_key=f"delete_key_{i}",
            )

        assert len(feature_cache.list_entries("to_delete")) == 3

        count = feature_cache.invalidate_feature_set("to_delete")

        assert count == 3
        assert len(feature_cache.list_entries("to_delete")) == 0


# =============================================================================
# CHECKSUM TESTS
# =============================================================================


class TestChecksumComputation:
    """Tests for checksum computation utilities."""

    def test_compute_dataframe_checksum(self, sample_dataframe):
        """Test DataFrame checksum computation."""
        checksum = compute_dataframe_checksum(sample_dataframe)

        assert isinstance(checksum, str)
        assert len(checksum) == 32

    def test_dataframe_checksum_deterministic(self, sample_dataframe):
        """Test that checksum is deterministic."""
        checksum1 = compute_dataframe_checksum(sample_dataframe)
        checksum2 = compute_dataframe_checksum(sample_dataframe)

        assert checksum1 == checksum2

    def test_dataframe_checksum_changes_with_data(self, sample_dataframe):
        """Test that checksum changes when data changes."""
        checksum1 = compute_dataframe_checksum(sample_dataframe)

        modified = sample_dataframe.copy()
        modified.iloc[0, 1] = 999.0

        checksum2 = compute_dataframe_checksum(modified)

        assert checksum1 != checksum2

    def test_compute_file_checksum(self, tmp_path):
        """Test file checksum computation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        checksum = compute_file_checksum(test_file)

        assert isinstance(checksum, str)
        assert len(checksum) == 64  # Full SHA256

    def test_file_checksum_deterministic(self, tmp_path):
        """Test that file checksum is deterministic."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        checksum1 = compute_file_checksum(test_file)
        checksum2 = compute_file_checksum(test_file)

        assert checksum1 == checksum2

    def test_file_checksum_missing_file_raises(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            compute_file_checksum("/nonexistent/path/file.txt")
