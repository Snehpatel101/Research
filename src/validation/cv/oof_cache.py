"""
OOF prediction caching to avoid recomputation.

Provides disk-based caching for OOF predictions generated during
stacking ensemble training. Cache invalidation is automatic when
model configuration, data hash, or CV parameters change.

Usage:
    >>> cache = OOFCache(cache_dir=Path("cache/oof"))
    >>> cache_key = cache.compute_cache_key(
    ...     model_name="xgboost",
    ...     model_config={"max_depth": 6},
    ...     data_hash="abc123",
    ...     cv_config={"n_splits": 5, "purge_bars": 60}
    ... )
    >>> if cache.has_oof(cache_key):
    ...     oof_data = cache.get_oof(cache_key)
    ... else:
    ...     oof_data = generate_oof(...)  # expensive computation
    ...     cache.put_oof(cache_key, oof_data, metadata={"model": "xgboost"})
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OOFCacheEntry:
    """Metadata for a cached OOF prediction."""

    cache_key: str
    model_name: str
    created_at: str
    data_hash: str
    cv_config: dict[str, Any]
    model_config: dict[str, Any]
    n_samples: int
    coverage: float
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OOFCacheEntry:
        """Create from dictionary."""
        return cls(**data)


class OOFCache:
    """
    Cache OOF predictions for stacking ensembles.

    Uses content-based cache keys derived from model config, data hash,
    and CV parameters. This ensures automatic invalidation when any
    relevant parameter changes.

    Cache structure:
        cache_dir/
            {cache_key}/
                predictions.parquet  (OOF predictions DataFrame)
                fold_info.json       (per-fold training info)
                metadata.json        (cache entry metadata)

    Thread Safety:
        This implementation is NOT thread-safe. For concurrent access,
        use external synchronization or file locking.
    """

    def __init__(self, cache_dir: Path, max_entries: int = 100) -> None:
        """
        Initialize OOFCache.

        Args:
            cache_dir: Directory for storing cached predictions
            max_entries: Maximum number of cache entries before cleanup
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self._index_path = self.cache_dir / "cache_index.json"
        self._index: dict[str, OOFCacheEntry] = self._load_index()

    def _load_index(self) -> dict[str, OOFCacheEntry]:
        """Load cache index from disk."""
        if not self._index_path.exists():
            return {}

        try:
            with open(self._index_path) as f:
                data = json.load(f)
            return {k: OOFCacheEntry.from_dict(v) for k, v in data.items()}
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load cache index: {e}. Starting fresh.")
            return {}

    def _save_index(self) -> None:
        """Save cache index to disk."""
        data = {k: v.to_dict() for k, v in self._index.items()}
        with open(self._index_path, "w") as f:
            json.dump(data, f, indent=2)

    def compute_cache_key(
        self,
        model_name: str,
        model_config: dict[str, Any],
        data_hash: str,
        cv_config: dict[str, Any],
    ) -> str:
        """
        Generate unique cache key from model+data+cv configuration.

        The key is a SHA256 hash of the JSON-serialized configuration,
        truncated to 16 characters for filesystem friendliness.

        Args:
            model_name: Name of the model (e.g., "xgboost", "lstm")
            model_config: Model hyperparameters
            data_hash: Hash of the training data
            cv_config: Cross-validation parameters

        Returns:
            16-character hex string cache key
        """
        # Normalize config for consistent hashing
        key_dict = {
            "model": model_name,
            "config": self._normalize_config(model_config),
            "data_hash": data_hash,
            "cv": self._normalize_config(cv_config),
        }
        key_str = json.dumps(key_dict, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def _normalize_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize configuration for consistent hashing.

        Converts non-JSON-serializable types and sorts keys.
        """
        normalized = {}
        for key, value in sorted(config.items()):
            if isinstance(value, dict):
                normalized[key] = self._normalize_config(value)
            elif isinstance(value, (list, tuple)):
                normalized[key] = [
                    self._normalize_config(v) if isinstance(v, dict) else v for v in value
                ]
            elif isinstance(value, np.ndarray):
                normalized[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                normalized[key] = value.item()
            elif isinstance(value, Path):
                normalized[key] = str(value)
            else:
                normalized[key] = value
        return normalized

    def has_oof(self, cache_key: str) -> bool:
        """
        Check if OOF predictions exist in cache.

        Also validates that the cache files exist on disk.

        Args:
            cache_key: Cache key from compute_cache_key()

        Returns:
            True if valid cache entry exists
        """
        if cache_key not in self._index:
            return False

        # Verify files exist
        entry_dir = self.cache_dir / cache_key
        required_files = ["predictions.parquet", "fold_info.json", "metadata.json"]
        for filename in required_files:
            if not (entry_dir / filename).exists():
                # Index is stale, remove entry
                logger.debug(f"Removing stale cache entry: {cache_key}")
                del self._index[cache_key]
                self._save_index()
                return False

        return True

    def get_oof(self, cache_key: str) -> dict[str, Any]:
        """
        Load cached OOF predictions.

        Args:
            cache_key: Cache key from compute_cache_key()

        Returns:
            Dict containing:
            - predictions: pd.DataFrame with OOF predictions
            - fold_info: List of per-fold training info
            - metadata: OOFCacheEntry metadata

        Raises:
            KeyError: If cache key not found
            FileNotFoundError: If cache files missing
        """
        if not self.has_oof(cache_key):
            raise KeyError(f"Cache key not found: {cache_key}")

        entry_dir = self.cache_dir / cache_key

        # Load predictions
        predictions = pd.read_parquet(entry_dir / "predictions.parquet")

        # Load fold info
        with open(entry_dir / "fold_info.json") as f:
            fold_info = json.load(f)

        # Get metadata from index
        metadata = self._index[cache_key]

        logger.info(f"Loaded OOF predictions from cache: {cache_key} ({metadata.model_name})")

        return {
            "predictions": predictions,
            "fold_info": fold_info,
            "metadata": metadata,
        }

    def put_oof(
        self,
        cache_key: str,
        predictions: pd.DataFrame,
        fold_info: list[dict[str, Any]],
        model_name: str,
        model_config: dict[str, Any],
        data_hash: str,
        cv_config: dict[str, Any],
        coverage: float,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Save OOF predictions to cache.

        Args:
            cache_key: Cache key from compute_cache_key()
            predictions: DataFrame with OOF predictions
            fold_info: List of per-fold training info
            model_name: Name of the model
            model_config: Model hyperparameters
            data_hash: Hash of the training data
            cv_config: Cross-validation parameters
            coverage: OOF prediction coverage (0-1)
            extra_metadata: Additional metadata to store
        """
        # Enforce max entries
        if len(self._index) >= self.max_entries:
            self._cleanup_oldest()

        entry_dir = self.cache_dir / cache_key
        entry_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions
        predictions.to_parquet(entry_dir / "predictions.parquet", index=False)

        # Save fold info
        with open(entry_dir / "fold_info.json", "w") as f:
            json.dump(fold_info, f, indent=2)

        # Create metadata entry
        entry = OOFCacheEntry(
            cache_key=cache_key,
            model_name=model_name,
            created_at=datetime.now().isoformat(),
            data_hash=data_hash,
            cv_config=cv_config,
            model_config=model_config,
            n_samples=len(predictions),
            coverage=coverage,
            extra_metadata=extra_metadata or {},
        )

        # Save metadata
        with open(entry_dir / "metadata.json", "w") as f:
            json.dump(entry.to_dict(), f, indent=2)

        # Update index
        self._index[cache_key] = entry
        self._save_index()

        logger.info(
            f"Cached OOF predictions: {cache_key} ({model_name}, "
            f"{len(predictions)} samples, coverage={coverage:.2%})"
        )

    def invalidate(self, cache_key: str) -> bool:
        """
        Remove cached entry.

        Args:
            cache_key: Cache key to invalidate

        Returns:
            True if entry was removed, False if not found
        """
        if cache_key not in self._index:
            return False

        # Remove files
        entry_dir = self.cache_dir / cache_key
        if entry_dir.exists():
            import shutil

            shutil.rmtree(entry_dir)

        # Update index
        del self._index[cache_key]
        self._save_index()

        logger.info(f"Invalidated cache entry: {cache_key}")
        return True

    def invalidate_model(self, model_name: str) -> int:
        """
        Remove all cached entries for a model.

        Args:
            model_name: Model name to invalidate

        Returns:
            Number of entries removed
        """
        keys_to_remove = [k for k, v in self._index.items() if v.model_name == model_name]
        for key in keys_to_remove:
            self.invalidate(key)
        return len(keys_to_remove)

    def clear(self) -> int:
        """
        Clear all cached entries.

        Returns:
            Number of entries removed
        """
        count = len(self._index)
        for key in list(self._index.keys()):
            self.invalidate(key)
        return count

    def _cleanup_oldest(self, n_remove: int = 10) -> None:
        """Remove oldest cache entries to make room for new ones."""
        # Sort by creation time
        sorted_entries = sorted(self._index.items(), key=lambda x: x[1].created_at)

        # Remove oldest entries
        for key, _ in sorted_entries[:n_remove]:
            self.invalidate(key)

        logger.debug(f"Cleaned up {n_remove} oldest cache entries")

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats
        """
        total_size = 0
        for cache_key in self._index:
            entry_dir = self.cache_dir / cache_key
            if entry_dir.exists():
                for f in entry_dir.iterdir():
                    total_size += f.stat().st_size

        return {
            "n_entries": len(self._index),
            "max_entries": self.max_entries,
            "total_size_mb": total_size / (1024 * 1024),
            "models": list({v.model_name for v in self._index.values()}),
            "cache_dir": str(self.cache_dir),
        }

    def list_entries(self) -> list[OOFCacheEntry]:
        """
        List all cache entries.

        Returns:
            List of OOFCacheEntry objects
        """
        return list(self._index.values())


def compute_data_hash(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    sample_size: int = 1000,
) -> str:
    """
    Compute hash of training data for cache keying.

    Uses a sample of the data for efficiency with large datasets.
    Hash includes shape, dtype, and sampled values.

    Args:
        X: Feature matrix
        y: Labels
        sample_size: Number of rows to sample for hashing

    Returns:
        16-character hex hash
    """
    # Get shapes
    X_shape = X.shape if hasattr(X, "shape") else (len(X),)
    y_shape = y.shape if hasattr(y, "shape") else (len(y),)

    # Sample indices (deterministic based on size)
    n_samples = X_shape[0]
    if n_samples > sample_size:
        np.random.seed(42)  # Deterministic sampling
        sample_idx = np.random.choice(n_samples, sample_size, replace=False)
        sample_idx.sort()
    else:
        sample_idx = np.arange(n_samples)

    # Extract sample values
    if isinstance(X, pd.DataFrame):
        X_sample = X.iloc[sample_idx].values
    else:
        X_sample = np.asarray(X)[sample_idx]

    if isinstance(y, pd.Series):
        y_sample = y.iloc[sample_idx].values
    else:
        y_sample = np.asarray(y)[sample_idx]

    # Build hash input
    hash_dict = {
        "X_shape": X_shape,
        "y_shape": y_shape,
        "X_dtype": str(X_sample.dtype),
        "y_dtype": str(y_sample.dtype),
        "X_sample": X_sample.tolist(),
        "y_sample": y_sample.tolist(),
    }

    hash_str = json.dumps(hash_dict, sort_keys=True, default=str)
    return hashlib.sha256(hash_str.encode()).hexdigest()[:16]


__all__ = [
    "OOFCache",
    "OOFCacheEntry",
    "compute_data_hash",
]
