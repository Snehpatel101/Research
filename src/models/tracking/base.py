"""
Base experiment tracking protocol and utilities.

Defines the ExperimentTracker protocol that all tracking backends implement,
the backend-neutral ``TrackerConfig`` and ``get_tracker`` factory, and
``flatten_params`` (nested config dicts -> dotted parameter names).

Backends (``TRACKING_BACKENDS``): "none" (no-op), "local" (JSON files) and
"mlflow" (optional dependency: ``pip install '.[mlflow]'``). Runs nest: a
tracker whose config names a ``parent_run_id`` opens a child run of it.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.config import TRACKING_BACKENDS

logger = logging.getLogger(__name__)


@dataclass
class TrackerConfig:
    """
    Configuration for experiment tracking.

    Attributes:
        backend: "none", "local" or "mlflow"
        experiment_name: Experiment the runs belong to (default "default")
        tracking_uri: local = root directory of the run files; mlflow =
            tracking server URI or store (None = MLflow's default)
        output_dir: Local root used when ``tracking_uri`` is unset
        parent_run_id: When set, ``start_run`` opens a child run of this run
        log_artifacts: Whether ``log_artifact`` records artifacts at all
        tags: Tags added to every run this tracker starts
    """

    backend: str = "none"
    experiment_name: str | None = None
    tracking_uri: str | None = None
    output_dir: Path = field(default_factory=lambda: Path("experiments/tracking"))
    parent_run_id: str | None = None
    log_artifacts: bool = True
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if self.backend not in TRACKING_BACKENDS:
            raise ValueError(
                f"Unknown tracking backend {self.backend!r}; expected one of {TRACKING_BACKENDS}"
            )

    @property
    def local_root(self) -> Path:
        """Root directory of a local tracker's files."""
        return Path(self.tracking_uri) if self.tracking_uri else self.output_dir


def flatten_params(params: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """
    Flatten nested dicts into dotted keys: ``{"a": {"b": 1}}`` -> ``{"a.b": 1}``.

    Trackers store parameters as a flat key -> scalar map; lists stay whole
    values (a list of model names is one parameter).
    """
    flat: dict[str, Any] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            flat.update(flatten_params(value, prefix=f"{name}."))
        else:
            flat[name] = value
    return flat


class ExperimentTracker(ABC):
    """
    Abstract base class for experiment tracking.

    All tracking backends (MLflow, local, etc.) implement this interface.

    Usage:
        tracker = get_tracker(TrackerConfig(backend="mlflow"))
        tracker.start_run("my_run")
        tracker.log_params({"lr": 0.001, "batch_size": 64})
        for epoch in range(epochs):
            tracker.log_metrics(train_epoch(...), step=epoch)
        tracker.log_artifact(model_path, "model")
        tracker.end_run()
    """

    def __init__(self, config: TrackerConfig) -> None:
        """Initialize tracker with configuration."""
        self.config = config
        self._run_id: str | None = None
        self._is_active: bool = False

    @property
    def run_id(self) -> str | None:
        """Get current run ID."""
        return self._run_id

    @property
    def is_active(self) -> bool:
        """Whether a run is open."""
        return self._is_active

    @abstractmethod
    def start_run(
        self,
        run_name: str | None = None,
        run_id: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """
        Start a new tracking run (a child run when ``config.parent_run_id`` is set).

        Args:
            run_name: Optional name for the run
            run_id: Optional ID to resume existing run
            tags: Optional tags to add to run

        Returns:
            Run ID
        """

    @abstractmethod
    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current run.

        Args:
            status: Final status ("FINISHED", "FAILED", "KILLED")
        """

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """
        Log parameters (hyperparameters, config values).

        Args:
            params: Dictionary of parameter name -> value
        """

    @abstractmethod
    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """
        Log metrics (loss, accuracy, etc.).

        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number (epoch, iteration)
        """

    @abstractmethod
    def log_artifact(
        self,
        path: Path | str,
        artifact_type: str | None = None,
    ) -> None:
        """
        Log an artifact (file or directory).

        Args:
            path: Path to artifact
            artifact_type: Optional type label ("model", "data", "plot")
        """

    @abstractmethod
    def set_tags(self, tags: dict[str, str]) -> None:
        """
        Add or overwrite tags on the current run.

        Args:
            tags: Dictionary of tag name -> value
        """


class DisabledTracker(ExperimentTracker):
    """No-op tracker for backend "none"."""

    def start_run(
        self,
        run_name: str | None = None,
        run_id: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Start a no-op run."""
        self._run_id = run_id or f"disabled_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._is_active = True
        return self._run_id

    def end_run(self, status: str = "FINISHED") -> None:
        """End the no-op run."""
        self._is_active = False

    def log_params(self, params: dict[str, Any]) -> None:
        """No-op parameter logging."""

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """No-op metrics logging."""

    def log_artifact(self, path: Path | str, artifact_type: str | None = None) -> None:
        """No-op artifact logging."""

    def set_tags(self, tags: dict[str, str]) -> None:
        """No-op tagging."""


def get_tracker(config: TrackerConfig | None = None) -> ExperimentTracker:
    """
    Get the experiment tracker for ``config.backend``.

    Args:
        config: Tracker configuration. None = backend "none" (no-op tracker).

    Returns:
        ExperimentTracker instance

    Raises:
        ImportError: backend "mlflow" without MLflow installed. A run that asked
            for MLflow never silently falls back to another backend.
    """
    if config is None or config.backend == "none":
        return DisabledTracker(config or TrackerConfig())

    if config.backend == "local":
        from .local_tracker import LocalTracker

        return LocalTracker(config)

    from .mlflow_tracker import MLflowTracker

    return MLflowTracker(config)


__all__ = [
    "TrackerConfig",
    "ExperimentTracker",
    "DisabledTracker",
    "flatten_params",
    "get_tracker",
]
