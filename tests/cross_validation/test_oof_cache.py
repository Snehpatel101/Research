"""
Tests for OOF prediction caching.

Tests the OOFCache class that enables caching of OOF predictions
to avoid expensive recomputation during stacking ensemble training.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cross_validation.oof_cache import (
    OOFCache,
    OOFCacheEntry,
    compute_data_hash,
)


class TestOOFCache:
    """Tests for OOFCache class."""

    @pytest.fixture
    def cache_dir(self) -> Path:
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_predictions(self) -> pd.DataFrame:
        """Create sample OOF predictions DataFrame."""
        n_samples = 100
        return pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=n_samples, freq="5min"),
            "xgboost_prob_short": np.random.rand(n_samples),
            "xgboost_prob_neutral": np.random.rand(n_samples),
            "xgboost_prob_long": np.random.rand(n_samples),
            "xgboost_pred": np.random.choice([-1, 0, 1], n_samples),
            "xgboost_confidence": np.random.rand(n_samples),
        })

    @pytest.fixture
    def sample_fold_info(self) -> list[dict]:
        """Create sample fold info."""
        return [
            {"fold": 0, "train_size": 80, "val_size": 20, "val_accuracy": 0.65},
            {"fold": 1, "train_size": 80, "val_size": 20, "val_accuracy": 0.67},
            {"fold": 2, "train_size": 80, "val_size": 20, "val_accuracy": 0.64},
        ]

    def test_compute_cache_key_deterministic(self, cache_dir: Path) -> None:
        """Cache key should be deterministic for same inputs."""
        cache = OOFCache(cache_dir)

        key1 = cache.compute_cache_key(
            model_name="xgboost",
            model_config={"max_depth": 6, "n_estimators": 100},
            data_hash="abc123",
            cv_config={"n_splits": 5, "purge_bars": 60},
        )

        key2 = cache.compute_cache_key(
            model_name="xgboost",
            model_config={"max_depth": 6, "n_estimators": 100},
            data_hash="abc123",
            cv_config={"n_splits": 5, "purge_bars": 60},
        )

        assert key1 == key2
        assert len(key1) == 16  # Truncated hash

    def test_compute_cache_key_changes_with_config(self, cache_dir: Path) -> None:
        """Cache key should change when config changes."""
        cache = OOFCache(cache_dir)

        key1 = cache.compute_cache_key(
            model_name="xgboost",
            model_config={"max_depth": 6},
            data_hash="abc123",
            cv_config={"n_splits": 5},
        )

        key2 = cache.compute_cache_key(
            model_name="xgboost",
            model_config={"max_depth": 8},  # Changed
            data_hash="abc123",
            cv_config={"n_splits": 5},
        )

        assert key1 != key2

    def test_has_oof_empty_cache(self, cache_dir: Path) -> None:
        """has_oof should return False for empty cache."""
        cache = OOFCache(cache_dir)
        assert not cache.has_oof("nonexistent_key")

    def test_put_and_get_oof(
        self,
        cache_dir: Path,
        sample_predictions: pd.DataFrame,
        sample_fold_info: list[dict],
    ) -> None:
        """Should be able to put and get OOF predictions."""
        cache = OOFCache(cache_dir)

        cache_key = cache.compute_cache_key(
            model_name="xgboost",
            model_config={"max_depth": 6},
            data_hash="abc123",
            cv_config={"n_splits": 5},
        )

        # Put OOF predictions
        cache.put_oof(
            cache_key=cache_key,
            predictions=sample_predictions,
            fold_info=sample_fold_info,
            model_name="xgboost",
            model_config={"max_depth": 6},
            data_hash="abc123",
            cv_config={"n_splits": 5},
            coverage=0.95,
        )

        # Verify it exists
        assert cache.has_oof(cache_key)

        # Get OOF predictions
        cached = cache.get_oof(cache_key)

        assert "predictions" in cached
        assert "fold_info" in cached
        assert "metadata" in cached

        # Verify data integrity
        pd.testing.assert_frame_equal(cached["predictions"], sample_predictions)
        assert cached["fold_info"] == sample_fold_info
        assert cached["metadata"].model_name == "xgboost"
        assert cached["metadata"].coverage == 0.95

    def test_invalidate(
        self,
        cache_dir: Path,
        sample_predictions: pd.DataFrame,
        sample_fold_info: list[dict],
    ) -> None:
        """Should be able to invalidate cache entries."""
        cache = OOFCache(cache_dir)

        cache_key = cache.compute_cache_key(
            model_name="xgboost",
            model_config={"max_depth": 6},
            data_hash="abc123",
            cv_config={"n_splits": 5},
        )

        cache.put_oof(
            cache_key=cache_key,
            predictions=sample_predictions,
            fold_info=sample_fold_info,
            model_name="xgboost",
            model_config={"max_depth": 6},
            data_hash="abc123",
            cv_config={"n_splits": 5},
            coverage=0.95,
        )

        assert cache.has_oof(cache_key)

        # Invalidate
        result = cache.invalidate(cache_key)
        assert result is True
        assert not cache.has_oof(cache_key)

        # Invalidate non-existent
        result = cache.invalidate("nonexistent")
        assert result is False

    def test_invalidate_model(
        self,
        cache_dir: Path,
        sample_predictions: pd.DataFrame,
        sample_fold_info: list[dict],
    ) -> None:
        """Should be able to invalidate all entries for a model."""
        cache = OOFCache(cache_dir)

        # Add entries for two models
        for model_name in ["xgboost", "lightgbm", "xgboost"]:
            cache_key = cache.compute_cache_key(
                model_name=model_name,
                model_config={"seed": np.random.randint(1000)},
                data_hash="abc123",
                cv_config={"n_splits": 5},
            )
            cache.put_oof(
                cache_key=cache_key,
                predictions=sample_predictions,
                fold_info=sample_fold_info,
                model_name=model_name,
                model_config={},
                data_hash="abc123",
                cv_config={"n_splits": 5},
                coverage=0.95,
            )

        # Should have 3 entries
        assert len(cache.list_entries()) == 3

        # Invalidate xgboost
        count = cache.invalidate_model("xgboost")
        assert count == 2

        # Should have 1 entry left
        assert len(cache.list_entries()) == 1
        assert cache.list_entries()[0].model_name == "lightgbm"

    def test_clear(
        self,
        cache_dir: Path,
        sample_predictions: pd.DataFrame,
        sample_fold_info: list[dict],
    ) -> None:
        """Should be able to clear all cache entries."""
        cache = OOFCache(cache_dir)

        # Add multiple entries
        for i in range(5):
            cache_key = cache.compute_cache_key(
                model_name="xgboost",
                model_config={"seed": i},
                data_hash="abc123",
                cv_config={"n_splits": 5},
            )
            cache.put_oof(
                cache_key=cache_key,
                predictions=sample_predictions,
                fold_info=sample_fold_info,
                model_name="xgboost",
                model_config={"seed": i},
                data_hash="abc123",
                cv_config={"n_splits": 5},
                coverage=0.95,
            )

        assert len(cache.list_entries()) == 5

        count = cache.clear()
        assert count == 5
        assert len(cache.list_entries()) == 0

    def test_max_entries_cleanup(
        self,
        cache_dir: Path,
        sample_predictions: pd.DataFrame,
        sample_fold_info: list[dict],
    ) -> None:
        """Should cleanup oldest entries when max_entries is reached."""
        cache = OOFCache(cache_dir, max_entries=5)

        # Add 6 entries (exceeds max_entries=5)
        for i in range(6):
            cache_key = cache.compute_cache_key(
                model_name="xgboost",
                model_config={"seed": i},
                data_hash="abc123",
                cv_config={"n_splits": 5},
            )
            cache.put_oof(
                cache_key=cache_key,
                predictions=sample_predictions,
                fold_info=sample_fold_info,
                model_name="xgboost",
                model_config={"seed": i},
                data_hash="abc123",
                cv_config={"n_splits": 5},
                coverage=0.95,
            )

        # Should have max_entries (some were cleaned up)
        assert len(cache.list_entries()) <= 5

    def test_get_stats(
        self,
        cache_dir: Path,
        sample_predictions: pd.DataFrame,
        sample_fold_info: list[dict],
    ) -> None:
        """Should return cache statistics."""
        cache = OOFCache(cache_dir)

        cache_key = cache.compute_cache_key(
            model_name="xgboost",
            model_config={"max_depth": 6},
            data_hash="abc123",
            cv_config={"n_splits": 5},
        )

        cache.put_oof(
            cache_key=cache_key,
            predictions=sample_predictions,
            fold_info=sample_fold_info,
            model_name="xgboost",
            model_config={"max_depth": 6},
            data_hash="abc123",
            cv_config={"n_splits": 5},
            coverage=0.95,
        )

        stats = cache.get_stats()

        assert stats["n_entries"] == 1
        assert stats["max_entries"] == 100
        assert stats["total_size_mb"] > 0
        assert "xgboost" in stats["models"]

    def test_persistence(
        self,
        cache_dir: Path,
        sample_predictions: pd.DataFrame,
        sample_fold_info: list[dict],
    ) -> None:
        """Cache should persist across instances."""
        cache_key = None

        # First instance - add entry
        cache1 = OOFCache(cache_dir)
        cache_key = cache1.compute_cache_key(
            model_name="xgboost",
            model_config={"max_depth": 6},
            data_hash="abc123",
            cv_config={"n_splits": 5},
        )
        cache1.put_oof(
            cache_key=cache_key,
            predictions=sample_predictions,
            fold_info=sample_fold_info,
            model_name="xgboost",
            model_config={"max_depth": 6},
            data_hash="abc123",
            cv_config={"n_splits": 5},
            coverage=0.95,
        )

        # Second instance - should find entry
        cache2 = OOFCache(cache_dir)
        assert cache2.has_oof(cache_key)

        cached = cache2.get_oof(cache_key)
        pd.testing.assert_frame_equal(cached["predictions"], sample_predictions)


class TestComputeDataHash:
    """Tests for compute_data_hash function."""

    def test_deterministic_hash(self) -> None:
        """Hash should be deterministic for same data."""
        X = pd.DataFrame(np.random.rand(100, 10))
        y = pd.Series(np.random.choice([0, 1, 2], 100))

        hash1 = compute_data_hash(X, y)
        hash2 = compute_data_hash(X, y)

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_different_data_different_hash(self) -> None:
        """Different data should produce different hashes."""
        X1 = pd.DataFrame(np.random.rand(100, 10))
        y1 = pd.Series(np.random.choice([0, 1, 2], 100))

        X2 = pd.DataFrame(np.random.rand(100, 10))
        y2 = pd.Series(np.random.choice([0, 1, 2], 100))

        hash1 = compute_data_hash(X1, y1)
        hash2 = compute_data_hash(X2, y2)

        # Very likely different (could theoretically collide but extremely unlikely)
        assert hash1 != hash2

    def test_numpy_arrays(self) -> None:
        """Should work with numpy arrays."""
        X = np.random.rand(100, 10)
        y = np.random.choice([0, 1, 2], 100)

        hash1 = compute_data_hash(X, y)
        assert len(hash1) == 16

    def test_large_data_sampling(self) -> None:
        """Should efficiently handle large data via sampling."""
        X = pd.DataFrame(np.random.rand(100000, 50))
        y = pd.Series(np.random.choice([0, 1, 2], 100000))

        # Should complete quickly due to sampling
        hash1 = compute_data_hash(X, y, sample_size=1000)
        assert len(hash1) == 16


class TestOOFCacheEntry:
    """Tests for OOFCacheEntry dataclass."""

    def test_to_dict(self) -> None:
        """Should convert to dictionary."""
        entry = OOFCacheEntry(
            cache_key="abc123",
            model_name="xgboost",
            created_at="2024-01-01T00:00:00",
            data_hash="def456",
            cv_config={"n_splits": 5},
            model_config={"max_depth": 6},
            n_samples=1000,
            coverage=0.95,
        )

        d = entry.to_dict()

        assert d["cache_key"] == "abc123"
        assert d["model_name"] == "xgboost"
        assert d["coverage"] == 0.95

    def test_from_dict(self) -> None:
        """Should create from dictionary."""
        d = {
            "cache_key": "abc123",
            "model_name": "xgboost",
            "created_at": "2024-01-01T00:00:00",
            "data_hash": "def456",
            "cv_config": {"n_splits": 5},
            "model_config": {"max_depth": 6},
            "n_samples": 1000,
            "coverage": 0.95,
            "extra_metadata": {},
        }

        entry = OOFCacheEntry.from_dict(d)

        assert entry.cache_key == "abc123"
        assert entry.model_name == "xgboost"
        assert entry.coverage == 0.95

    def test_roundtrip(self) -> None:
        """Should survive roundtrip through dict."""
        entry = OOFCacheEntry(
            cache_key="abc123",
            model_name="xgboost",
            created_at="2024-01-01T00:00:00",
            data_hash="def456",
            cv_config={"n_splits": 5},
            model_config={"max_depth": 6},
            n_samples=1000,
            coverage=0.95,
            extra_metadata={"horizon": 20},
        )

        d = entry.to_dict()
        entry2 = OOFCacheEntry.from_dict(d)

        assert entry == entry2
