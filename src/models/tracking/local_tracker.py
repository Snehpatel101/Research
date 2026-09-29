"""
Local JSON-based experiment tracker.

Provides offline experiment tracking without external dependencies.
Useful for development and environments without MLflow.
"""

from __future__ import annotations

import json
import logging
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from .base import ExperimentTracker, TrackerConfig

logger = logging.getLogger(__name__)

# Metrics are flushed to disk every this many log_metrics calls (and at end_run)
_METRICS_FLUSH_EVERY = 10


class LocalTracker(ExperimentTracker):
    """
    Local JSON-based experiment tracker.

    Stores experiment data in a directory structure under
    ``config.local_root`` (``tracking_uri``, else ``output_dir``):

        <root>/
        ├── {experiment_name}/
        │   ├── {run_id}/
        │   │   ├── run_info.json    # name, status, times, parent_run_id
        │   │   ├── params.json
        │   │   ├── metrics.json     # [{timestamp, step, metrics}]
        │   │   ├── tags.json
        │   │   └── artifacts.json   # [{path, type}] references

    Child runs (``config.parent_run_id``) live next to their parent in the same
    experiment directory and record the parent's ID in ``run_info.json``.
    Artifacts are recorded by reference, not copied: they already live in the
    run's output directory, and model checkpoints can be gigabytes.

    Usage:
        config = TrackerConfig(backend="local", tracking_uri="experiments/tracking")
        tracker = LocalTracker(config)
        tracker.start_run("my_run")
        tracker.log_params({"lr": 0.001})
        tracker.log_metrics({"loss": 0.5}, step=1)
        tracker.end_run()
    """

    def __init__(self, config: TrackerConfig) -> None:
        """Initialize local tracker."""
        super().__init__(config)
        self._run_dir: Path | None = None
        self._params: dict[str, Any] = {}
        self._metrics: list[dict[str, Any]] = []
        self._tags: dict[str, str] = {}
        self._artifacts: list[dict[str, Any]] = []

    @property
    def run_dir(self) -> Path | None:
        """Get current run directory."""
        return self._run_dir

    def start_run(
        self,
        run_name: str | None = None,
        run_id: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """
        Start a new local tracking run.

        Args:
            run_name: Optional name for the run
            run_id: Optional ID to resume existing run
            tags: Optional tags to add to run

        Returns:
            Run ID
        """
        if self._is_active:
            logger.warning("Run already active, ending previous run")
            self.end_run(status="INTERRUPTED")

        # Unique even for runs started in the same second (walk-forward
        # windows, regime models)
        if run_id:
            self._run_id = run_id
        else:
            name_part = run_name.replace(" ", "_")[:40] if run_name else "run"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._run_id = f"{name_part}_{timestamp}_{secrets.token_hex(3)}"

        experiment_name = self.config.experiment_name or "default"
        self._run_dir = self.config.local_root / experiment_name / self._run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

        self._params = {}
        self._metrics = []
        self._artifacts = []
        self._tags = dict(self.config.tags)
        if tags:
            self._tags.update(tags)
        self._is_active = True

        run_info = {
            "run_id": self._run_id,
            "run_name": run_name,
            "experiment_name": experiment_name,
            "parent_run_id": self.config.parent_run_id,
            "start_time": datetime.now().isoformat(),
            "status": "RUNNING",
        }
        self._save_json(self._run_dir / "run_info.json", run_info)
        if self._tags:
            self._save_json(self._run_dir / "tags.json", self._tags)

        logger.info(f"Started local tracking run: {self._run_id}")
        return self._run_id

    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current run.

        Args:
            status: Final status ("FINISHED", "FAILED", "KILLED", "INTERRUPTED")
        """
        if not self._is_active or not self._run_dir:
            logger.warning("No active run to end")
            return

        run_info_path = self._run_dir / "run_info.json"
        if run_info_path.exists():
            run_info = self._load_json(run_info_path)
            run_info["end_time"] = datetime.now().isoformat()
            run_info["status"] = status
            self._save_json(run_info_path, run_info)

        if self._params:
            self._save_json(self._run_dir / "params.json", self._params)
        if self._metrics:
            self._save_json(self._run_dir / "metrics.json", self._metrics)
        if self._tags:
            self._save_json(self._run_dir / "tags.json", self._tags)
        if self._artifacts:
            self._save_json(self._run_dir / "artifacts.json", self._artifacts)

        logger.info(f"Ended local tracking run: {self._run_id} with status {status}")

        self._is_active = False
        self._run_dir = None

    def log_params(self, params: dict[str, Any]) -> None:
        """
        Log parameters.

        Args:
            params: Dictionary of parameter name -> value
        """
        if not self._is_active:
            logger.warning("No active run, params not logged")
            return

        for key, value in params.items():
            if isinstance(value, Path):
                self._params[key] = str(value)
            elif hasattr(value, "tolist"):  # numpy arrays / scalars
                self._params[key] = value.tolist()
            elif not isinstance(value, str | int | float | bool | list | dict | type(None)):
                self._params[key] = str(value)
            else:
                self._params[key] = value

        if self._run_dir:
            self._save_json(self._run_dir / "params.json", self._params)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """
        Log metrics.

        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number (epoch, iteration)
        """
        if not self._is_active:
            logger.warning("No active run, metrics not logged")
            return

        entry: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "metrics": {},
        }
        for key, value in metrics.items():
            # float() also unwraps numpy scalars and single-element tensors
            entry["metrics"][key] = float(value) if value is not None else None

        self._metrics.append(entry)
        if self._run_dir and len(self._metrics) % _METRICS_FLUSH_EVERY == 0:
            self._save_json(self._run_dir / "metrics.json", self._metrics)

    def log_artifact(
        self,
        path: Path | str,
        artifact_type: str | None = None,
    ) -> None:
        """
        Record an artifact by reference (resolved path + type).

        Args:
            path: Path to artifact
            artifact_type: Optional type label ("model", "data", "plot")
        """
        if not self._is_active or not self._run_dir:
            logger.warning("No active run, artifact not logged")
            return

        if not self.config.log_artifacts:
            logger.debug("Artifact logging disabled")
            return

        path = Path(path)
        if not path.exists():
            logger.warning(f"Artifact path does not exist: {path}")
            return

        self._artifacts.append({"path": str(path.resolve()), "type": artifact_type})
        self._save_json(self._run_dir / "artifacts.json", self._artifacts)
        logger.debug(f"Logged artifact reference: {path}")

    def set_tags(self, tags: dict[str, str]) -> None:
        """Add or overwrite tags on the current run."""
        if not self._is_active or not self._run_dir:
            logger.warning("No active run, tags not set")
            return
        self._tags.update({k: str(v) for k, v in tags.items()})
        self._save_json(self._run_dir / "tags.json", self._tags)

    @staticmethod
    def _save_json(path: Path, data: Any) -> None:
        """Save data to JSON file."""
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def _load_json(path: Path) -> Any:
        """Load data from JSON file."""
        with open(path) as f:
            return json.load(f)

    @classmethod
    def list_runs(
        cls,
        output_dir: Path,
        experiment_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List all runs in output directory.

        Args:
            output_dir: Base output directory
            experiment_name: Optional filter by experiment name

        Returns:
            List of run info dictionaries
        """
        runs = []

        if experiment_name:
            exp_dirs = [output_dir / experiment_name]
        else:
            exp_dirs = [d for d in output_dir.iterdir() if d.is_dir()]

        for exp_dir in exp_dirs:
            if not exp_dir.exists():
                continue
            for run_dir in exp_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                run_info_path = run_dir / "run_info.json"
                if run_info_path.exists():
                    run_info = cls._load_json(run_info_path)
                    run_info["run_dir"] = str(run_dir)
                    runs.append(run_info)

        # Sort by start time (newest first)
        runs.sort(key=lambda x: x.get("start_time", ""), reverse=True)
        return runs


__all__ = ["LocalTracker"]
