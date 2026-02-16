"""
Local JSON-based experiment tracker.

Provides offline experiment tracking without external dependencies.
Useful for development and environments without MLflow.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .base import ExperimentTracker, TrackerConfig

logger = logging.getLogger(__name__)


class LocalTracker(ExperimentTracker):
    """
    Local JSON-based experiment tracker.

    Stores experiment data in a directory structure:
        output_dir/
        ├── {experiment_name}/
        │   ├── {run_id}/
        │   │   ├── params.json
        │   │   ├── metrics.json
        │   │   ├── tags.json
        │   │   ├── run_info.json
        │   │   └── artifacts/
        │   │       └── ...

    Usage:
        config = TrackerConfig(backend="local", output_dir=Path("experiments"))
        tracker = LocalTracker(config)
        with tracker.run_context("my_run") as run_id:
            tracker.log_params({"lr": 0.001})
            tracker.log_metrics({"loss": 0.5}, step=1)
    """

    def __init__(self, config: TrackerConfig) -> None:
        """Initialize local tracker."""
        super().__init__(config)
        self._run_dir: Path | None = None
        self._params: dict[str, Any] = {}
        self._metrics: list[dict[str, Any]] = []
        self._tags: dict[str, str] = {}

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

        # Generate run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_id:
            self._run_id = run_id
        else:
            name_part = run_name.replace(" ", "_")[:20] if run_name else "run"
            self._run_id = f"{name_part}_{timestamp}"

        # Create run directory
        experiment_name = self.config.experiment_name or "default"
        self._run_dir = self.config.output_dir / experiment_name / self._run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize run state
        self._params = {}
        self._metrics = []
        self._tags = dict(self.config.tags)  # Start with config tags
        if tags:
            self._tags.update(tags)
        self._is_active = True

        # Save run info
        run_info = {
            "run_id": self._run_id,
            "run_name": run_name,
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "status": "RUNNING",
        }
        self._save_json(self._run_dir / "run_info.json", run_info)

        # Save initial tags
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

        # Update run info with end status
        run_info_path = self._run_dir / "run_info.json"
        if run_info_path.exists():
            run_info = self._load_json(run_info_path)
            run_info["end_time"] = datetime.now().isoformat()
            run_info["status"] = status
            self._save_json(run_info_path, run_info)

        # Save final params and metrics
        if self._params:
            self._save_json(self._run_dir / "params.json", self._params)
        if self._metrics:
            self._save_json(self._run_dir / "metrics.json", self._metrics)
        if self._tags:
            self._save_json(self._run_dir / "tags.json", self._tags)

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

        # Convert non-serializable values to strings
        for key, value in params.items():
            if isinstance(value, Path):
                self._params[key] = str(value)
            elif hasattr(value, "tolist"):  # numpy arrays
                self._params[key] = value.tolist()
            elif not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                self._params[key] = str(value)
            else:
                self._params[key] = value

        # Save immediately for persistence
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

        # Create metrics entry
        entry: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "metrics": {},
        }

        # Handle potential non-float values
        for key, value in metrics.items():
            if hasattr(value, "item"):  # torch tensors, numpy scalars
                entry["metrics"][key] = float(value.item())
            elif isinstance(value, (int, float)):
                entry["metrics"][key] = float(value)
            else:
                entry["metrics"][key] = float(value) if value is not None else None

        self._metrics.append(entry)

        # Save periodically (every N entries or at certain steps)
        if self._run_dir and len(self._metrics) % 10 == 0:
            self._save_json(self._run_dir / "metrics.json", self._metrics)

    def log_artifact(
        self,
        path: Path | str,
        artifact_type: str | None = None,
    ) -> None:
        """
        Log an artifact (copy file/directory to run artifacts).

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

        # Create artifacts directory
        artifacts_dir = self._run_dir / "artifacts"
        if artifact_type:
            artifacts_dir = artifacts_dir / artifact_type
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Copy artifact
        dest = artifacts_dir / path.name
        if path.is_file():
            shutil.copy2(path, dest)
        elif path.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(path, dest)

        logger.debug(f"Logged artifact: {path} -> {dest}")

    def set_tags(self, tags: dict[str, str]) -> None:
        """
        Set tags on the current run.

        Args:
            tags: Dictionary of tag name -> value
        """
        if not self._is_active:
            logger.warning("No active run, tags not set")
            return

        self._tags.update(tags)

        if self._run_dir:
            self._save_json(self._run_dir / "tags.json", self._tags)

    def get_run_metrics(self) -> list[dict[str, Any]]:
        """Get all logged metrics for current run."""
        return list(self._metrics)

    def get_run_params(self) -> dict[str, Any]:
        """Get all logged params for current run."""
        return dict(self._params)

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
    def load_run(cls, run_dir: Path) -> dict[str, Any]:
        """
        Load a completed run's data.

        Args:
            run_dir: Path to run directory

        Returns:
            Dictionary with run_info, params, metrics, tags
        """
        result = {}

        for name in ["run_info", "params", "metrics", "tags"]:
            path = run_dir / f"{name}.json"
            if path.exists():
                result[name] = cls._load_json(path)
            else:
                result[name] = {} if name != "metrics" else []

        return result

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
