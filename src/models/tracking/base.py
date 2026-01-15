"""
Base experiment tracking protocol and utilities.

Defines the ExperimentTracker protocol that all tracking backends implement.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class TrackerConfig:
    """Configuration for experiment tracking."""

    enabled: bool = True
    backend: str = "local"  # "local", "mlflow", "disabled"
    experiment_name: str | None = None
    run_name: str | None = None
    tracking_uri: str | None = None  # For MLflow
    output_dir: Path = field(default_factory=lambda: Path("experiments/tracking"))
    log_artifacts: bool = True
    log_metrics_every_n_epochs: int = 1
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if self.backend not in ("local", "mlflow", "disabled"):
            raise ValueError(f"Unknown backend: {self.backend}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "tracking_uri": self.tracking_uri,
            "output_dir": str(self.output_dir),
            "log_artifacts": self.log_artifacts,
            "log_metrics_every_n_epochs": self.log_metrics_every_n_epochs,
            "tags": self.tags,
        }


class ExperimentTracker(ABC):
    """
    Abstract base class for experiment tracking.

    All tracking backends (MLflow, local, etc.) implement this interface.

    Usage:
        tracker = get_tracker("mlflow", config)
        with tracker.start_run("my_run"):
            tracker.log_params({"lr": 0.001, "batch_size": 64})
            for epoch in range(epochs):
                metrics = train_epoch(...)
                tracker.log_metrics(metrics, step=epoch)
            tracker.log_artifact(model_path, "model")
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
        """Check if a run is currently active."""
        return self._is_active

    @abstractmethod
    def start_run(
        self,
        run_name: str | None = None,
        run_id: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """
        Start a new tracking run.

        Args:
            run_name: Optional name for the run
            run_id: Optional ID to resume existing run
            tags: Optional tags to add to run

        Returns:
            Run ID
        """
        pass

    @abstractmethod
    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current run.

        Args:
            status: Final status ("FINISHED", "FAILED", "KILLED")
        """
        pass

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """
        Log parameters (hyperparameters, config values).

        Args:
            params: Dictionary of parameter name -> value
        """
        pass

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
        pass

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
        pass

    def log_model(
        self,
        model_path: Path | str,
        model_name: str | None = None,
    ) -> None:
        """
        Log a trained model.

        Default implementation calls log_artifact.
        Backends may override for model-specific logging.

        Args:
            model_path: Path to model file or directory
            model_name: Optional model name
        """
        self.log_artifact(model_path, artifact_type="model")

    @abstractmethod
    def set_tags(self, tags: dict[str, str]) -> None:
        """
        Set tags on the current run.

        Args:
            tags: Dictionary of tag name -> value
        """
        pass

    @contextmanager
    def run_context(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Iterator[str]:
        """
        Context manager for tracking runs.

        Automatically starts and ends the run, handling exceptions.

        Args:
            run_name: Optional name for the run
            tags: Optional tags to add

        Yields:
            Run ID

        Example:
            with tracker.run_context("training_run") as run_id:
                tracker.log_params(config)
                # ... training code ...
                tracker.log_metrics(final_metrics)
        """
        run_id = self.start_run(run_name=run_name, tags=tags)
        try:
            yield run_id
            self.end_run(status="FINISHED")
        except Exception as e:
            logger.error(f"Run failed with error: {e}")
            self.end_run(status="FAILED")
            raise


class DisabledTracker(ExperimentTracker):
    """No-op tracker for when tracking is disabled."""

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
        pass

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """No-op metrics logging."""
        pass

    def log_artifact(self, path: Path | str, artifact_type: str | None = None) -> None:
        """No-op artifact logging."""
        pass

    def set_tags(self, tags: dict[str, str]) -> None:
        """No-op tag setting."""
        pass


def get_tracker(backend: str | None = None, config: TrackerConfig | None = None) -> ExperimentTracker:
    """
    Get an experiment tracker for the specified backend.

    Args:
        backend: Backend name ("local", "mlflow", "disabled"). Defaults to config.backend.
        config: Tracker configuration. Uses defaults if None.

    Returns:
        ExperimentTracker instance

    Raises:
        ValueError: If unknown backend specified
        ImportError: If MLflow requested but not installed
    """
    if config is None:
        config = TrackerConfig()

    backend = backend or config.backend

    if backend == "disabled" or not config.enabled:
        return DisabledTracker(config)

    if backend == "local":
        from .local_tracker import LocalTracker
        return LocalTracker(config)

    if backend == "mlflow":
        try:
            from .mlflow_tracker import MLflowTracker
            return MLflowTracker(config)
        except ImportError as e:
            logger.warning(f"MLflow not installed, falling back to local tracker: {e}")
            from .local_tracker import LocalTracker
            return LocalTracker(config)

    raise ValueError(f"Unknown tracking backend: {backend}")


__all__ = [
    "TrackerConfig",
    "ExperimentTracker",
    "DisabledTracker",
    "get_tracker",
]
