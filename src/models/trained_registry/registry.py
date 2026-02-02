"""
Trained Model Registry.

Provides centralized tracking and querying of all trained models
with persistent index storage.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .query import ModelQuery

logger = logging.getLogger(__name__)


@dataclass
class TrainedModelEntry:
    """
    Entry for a trained model in the registry.

    Stores metadata and performance metrics for a single trained model.
    """

    run_id: str
    model_name: str
    model_family: str
    horizon: int
    run_dir: Path
    train_date: datetime
    feature_set: str = "full"
    metrics: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    checksum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "model_family": self.model_family,
            "horizon": self.horizon,
            "run_dir": str(self.run_dir),
            "train_date": self.train_date.isoformat(),
            "feature_set": self.feature_set,
            "metrics": self.metrics,
            "config": self.config,
            "tags": self.tags,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainedModelEntry:
        """Create entry from dictionary."""
        return cls(
            run_id=data["run_id"],
            model_name=data["model_name"],
            model_family=data.get("model_family", "unknown"),
            horizon=data["horizon"],
            run_dir=Path(data["run_dir"]),
            train_date=datetime.fromisoformat(data["train_date"]),
            feature_set=data.get("feature_set", "full"),
            metrics=data.get("metrics", {}),
            config=data.get("config", {}),
            tags=data.get("tags", {}),
            checksum=data.get("checksum"),
        )

    @property
    def model_path(self) -> Path:
        """Get path to model checkpoint."""
        return self.run_dir / "checkpoints"

    @property
    def config_path(self) -> Path:
        """Get path to config directory."""
        return self.run_dir / "config"

    def is_valid(self) -> bool:
        """Check if model files still exist."""
        return self.run_dir.exists() and self.model_path.exists()


class TrainedModelRegistry:
    """
    Registry for tracking trained models.

    Maintains a JSON index of all trained models with their metadata
    and performance metrics. Supports querying, filtering, and
    finding the best model for a given configuration.

    The registry uses a local JSON file for persistence, making it
    suitable for development and small-scale production use.

    Example:
        registry = TrainedModelRegistry("experiments/model_index.json")

        # Register a new model
        entry = registry.register(Path("experiments/runs/xgboost_h20_..."))

        # Query models
        results = registry.query(
            ModelQuery()
            .model_name("xgboost")
            .horizon(20)
            .min_metric("val_f1", 0.5)
        )

        # Get best model
        best = registry.get_best("xgboost", horizon=20, metric="val_f1")
    """

    def __init__(
        self,
        index_path: Path | str = "experiments/model_index.json",
        auto_load: bool = True,
    ) -> None:
        """
        Initialize registry.

        Args:
            index_path: Path to the index JSON file
            auto_load: Whether to load existing index on init
        """
        self.index_path = Path(index_path)
        self._entries: dict[str, TrainedModelEntry] = {}

        if auto_load and self.index_path.exists():
            self._load_index()

    @property
    def count(self) -> int:
        """Get number of registered models."""
        return len(self._entries)

    def register(self, run_dir: Path | str) -> TrainedModelEntry:
        """
        Register a trained model from its run directory.

        Reads model metadata and metrics from the run directory
        and adds it to the registry.

        Args:
            run_dir: Path to the model's run directory

        Returns:
            The created TrainedModelEntry

        Raises:
            FileNotFoundError: If run directory doesn't exist
            ValueError: If required files are missing
        """
        run_dir = Path(run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        # Load trainer config
        config_path = run_dir / "config" / "trainer_config.json"
        if not config_path.exists():
            raise ValueError(f"Trainer config not found: {config_path}")

        with open(config_path) as f:
            config = json.load(f)

        # Load metrics
        metrics = {}
        metrics_path = run_dir / "metrics" / "evaluation_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                eval_metrics = json.load(f)
                # Flatten metrics for easier querying
                metrics = self._flatten_metrics(eval_metrics)

        # Load checksum if available
        checksum = None
        checksum_path = run_dir / "checksums.json"
        if checksum_path.exists():
            with open(checksum_path) as f:
                checksums = json.load(f)
                checksum = checksums.get("model", {}).get("sha256")

        # Extract run_id from directory name
        run_id = run_dir.name

        # Get model family from registry
        model_name = config.get("model_name", "unknown")
        model_family = self._get_model_family(model_name)

        # Create entry
        entry = TrainedModelEntry(
            run_id=run_id,
            model_name=model_name,
            model_family=model_family,
            horizon=config.get("horizon", 20),
            run_dir=run_dir,
            train_date=datetime.now(),  # Could be read from file timestamp
            feature_set=config.get("feature_set", "full"),
            metrics=metrics,
            config=config,
            tags={},
            checksum=checksum,
        )

        # Add to registry
        self._entries[run_id] = entry
        self._save_index()

        logger.info(f"Registered model: {run_id} (val_f1={metrics.get('val_f1', 0):.4f})")
        return entry

    def unregister(self, run_id: str) -> bool:
        """
        Remove a model from the registry.

        Args:
            run_id: The run ID to remove

        Returns:
            True if removed, False if not found
        """
        if run_id in self._entries:
            del self._entries[run_id]
            self._save_index()
            logger.info(f"Unregistered model: {run_id}")
            return True
        return False

    def get(self, run_id: str) -> TrainedModelEntry | None:
        """
        Get a specific model entry by run ID.

        Args:
            run_id: The run ID to look up

        Returns:
            TrainedModelEntry or None if not found
        """
        return self._entries.get(run_id)

    def query(self, query: ModelQuery | None = None) -> list[TrainedModelEntry]:
        """
        Query models with optional filters.

        Args:
            query: ModelQuery with filter criteria, or None for all models

        Returns:
            List of matching TrainedModelEntry objects
        """
        if query is None:
            results = list(self._entries.values())
        else:
            results = [entry for entry in self._entries.values() if query.matches(entry.to_dict())]

        # Sort if specified
        if query and query.sort_metric:
            results.sort(
                key=lambda e: e.metrics.get(query.sort_metric or "", float("-inf")),
                reverse=query.sort_descending,
            )

        # Limit if specified
        if query and query.result_limit:
            results = results[: query.result_limit]

        return results

    def get_best(
        self,
        model_name: str,
        horizon: int,
        metric: str = "val_f1",
    ) -> TrainedModelEntry | None:
        """
        Get the best model for a given configuration.

        Args:
            model_name: Model name to filter by
            horizon: Prediction horizon
            metric: Metric to optimize (higher is better)

        Returns:
            Best TrainedModelEntry or None if none found
        """
        query = (
            ModelQuery()
            .model_name(model_name)
            .horizon(horizon)
            .sort_by(metric, descending=True)
            .limit(1)
        )
        results = self.query(query)
        return results[0] if results else None

    def list_all(self) -> list[TrainedModelEntry]:
        """Get all registered models."""
        return list(self._entries.values())

    def list_by_model(self, model_name: str) -> list[TrainedModelEntry]:
        """Get all models with given name."""
        return [e for e in self._entries.values() if e.model_name == model_name]

    def list_by_horizon(self, horizon: int) -> list[TrainedModelEntry]:
        """Get all models with given horizon."""
        return [e for e in self._entries.values() if e.horizon == horizon]

    def rebuild_index(self, runs_dir: Path | str) -> int:
        """
        Rebuild index by scanning run directories.

        Useful for recovering from corrupted index or indexing
        existing models not yet in the registry.

        Args:
            runs_dir: Path to experiments/runs directory

        Returns:
            Number of models indexed
        """
        runs_dir = Path(runs_dir)
        if not runs_dir.exists():
            logger.warning(f"Runs directory not found: {runs_dir}")
            return 0

        count = 0
        for run_path in runs_dir.iterdir():
            if not run_path.is_dir():
                continue

            # Check if it looks like a valid run directory
            config_path = run_path / "config" / "trainer_config.json"
            if not config_path.exists():
                continue

            # Skip if already registered
            if run_path.name in self._entries:
                continue

            try:
                self.register(run_path)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to index {run_path}: {e}")

        logger.info(f"Rebuilt index: {count} new models indexed")
        return count

    def validate_entries(self) -> dict[str, bool]:
        """
        Validate that all registered models still exist.

        Returns:
            Dict mapping run_id to validity status
        """
        results = {}
        for run_id, entry in self._entries.items():
            results[run_id] = entry.is_valid()
        return results

    def prune_invalid(self) -> int:
        """
        Remove entries for models that no longer exist.

        Returns:
            Number of entries removed
        """
        invalid_ids = [run_id for run_id, entry in self._entries.items() if not entry.is_valid()]
        for run_id in invalid_ids:
            del self._entries[run_id]

        if invalid_ids:
            self._save_index()
            logger.info(f"Pruned {len(invalid_ids)} invalid entries")

        return len(invalid_ids)

    def get_summary(self) -> dict[str, Any]:
        """
        Get summary statistics for the registry.

        Returns:
            Dict with summary statistics
        """
        if not self._entries:
            return {"count": 0}

        model_counts: dict[str, int] = {}
        horizon_counts: dict[int, int] = {}
        metrics_ranges: dict[str, dict[str, float]] = {}

        for entry in self._entries.values():
            # Count by model
            model_counts[entry.model_name] = model_counts.get(entry.model_name, 0) + 1

            # Count by horizon
            horizon_counts[entry.horizon] = horizon_counts.get(entry.horizon, 0) + 1

            # Track metric ranges
            for metric, value in entry.metrics.items():
                if metric not in metrics_ranges:
                    metrics_ranges[metric] = {"min": value, "max": value}
                else:
                    metrics_ranges[metric]["min"] = min(metrics_ranges[metric]["min"], value)
                    metrics_ranges[metric]["max"] = max(metrics_ranges[metric]["max"], value)

        return {
            "count": len(self._entries),
            "models": model_counts,
            "horizons": horizon_counts,
            "metrics_ranges": metrics_ranges,
        }

    def _flatten_metrics(self, metrics: dict[str, Any]) -> dict[str, float]:
        """Flatten nested metrics dict for easier querying."""
        flat = {}

        # Top-level metrics
        for key in ["accuracy", "macro_f1", "precision", "recall"]:
            if key in metrics:
                flat[f"val_{key}"] = float(metrics[key])

        # Trading metrics
        if "trading" in metrics:
            for key, value in metrics["trading"].items():
                if isinstance(value, (int, float)):
                    flat[f"val_{key}"] = float(value)

        # Calibration metrics
        if "calibration" in metrics:
            for key, value in metrics["calibration"].items():
                if isinstance(value, (int, float)):
                    flat[f"cal_{key}"] = float(value)

        # Feature selection info
        if "feature_selection" in metrics:
            fs = metrics["feature_selection"]
            if "n_features_selected" in fs:
                flat["n_features"] = float(fs["n_features_selected"])
            if "reduction_ratio" in fs:
                flat["feature_reduction"] = float(fs["reduction_ratio"])

        return flat

    def _get_model_family(self, model_name: str) -> str:
        """Get model family from registry."""
        try:
            from ..registry import ModelRegistry

            info = ModelRegistry.get_model_info(model_name)
            if info:
                return str(info.get("family", "unknown"))
        except Exception as e:
            logger.warning(f"Failed to get model family from registry for {model_name}: {e}. Using fallback.")
            pass

        # Fallback based on name
        if model_name in ("xgboost", "lightgbm", "catboost"):
            return "boosting"
        if model_name in ("random_forest", "logistic", "svm"):
            return "classical"
        if model_name in ("lstm", "gru", "tcn", "transformer", "patchtst"):
            return "neural"
        if model_name in ("voting", "stacking", "blending"):
            return "ensemble"
        return "unknown"

    def _load_index(self) -> None:
        """Load index from disk."""
        try:
            with open(self.index_path) as f:
                data = json.load(f)

            self._entries = {}
            for entry_data in data.get("entries", []):
                entry = TrainedModelEntry.from_dict(entry_data)
                self._entries[entry.run_id] = entry

            logger.debug(f"Loaded {len(self._entries)} entries from index")
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
            self._entries = {}

    def _save_index(self) -> None:
        """Save index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "updated": datetime.now().isoformat(),
            "count": len(self._entries),
            "entries": [e.to_dict() for e in self._entries.values()],
        }

        with open(self.index_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved index with {len(self._entries)} entries")


__all__ = ["TrainedModelRegistry", "TrainedModelEntry"]
