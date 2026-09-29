"""
MLflow experiment tracker.

Optional dependency: ``pip install '.[mlflow]'``. MLflow is imported when a
tracker is created, never at module import, so the rest of the package works
without it.

Uses ``MlflowClient`` with explicit run IDs rather than MLflow's fluent
(``mlflow.start_run``) API: no process-global "active run" is touched, so
trackers are safe in worker processes and threads, never interfere with a
caller's own MLflow runs, and child runs attach to their parent through the
``mlflow.parentRunId`` tag (how the MLflow UI nests runs).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .base import ExperimentTracker, TrackerConfig

logger = logging.getLogger(__name__)

MLFLOW_INSTALL_HINT = "pip install '.[mlflow]'"
# MLflow tag that nests a run under its parent in the UI
PARENT_RUN_TAG = "mlflow.parentRunId"
# Server-side limits (MLflow >= 2.x): param value length and batch sizes
_MAX_PARAM_VALUE_LENGTH = 6000
_PARAMS_PER_BATCH = 100
_METRICS_PER_BATCH = 1000
_STATUS_MAP = {
    "FINISHED": "FINISHED",
    "FAILED": "FAILED",
    "KILLED": "KILLED",
    "INTERRUPTED": "KILLED",
}


def _load_mlflow() -> tuple[Any, Any, Any, Any]:
    """(MlflowClient, Metric, Param, RunTag), or ImportError with the install hint."""
    try:
        from mlflow.entities import Metric, Param, RunTag  # type: ignore[import-not-found]
        from mlflow.tracking import MlflowClient  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            f"Tracking backend 'mlflow' needs MLflow, which is not installed: "
            f"{MLFLOW_INSTALL_HINT}"
        ) from e
    return MlflowClient, Metric, Param, RunTag


class MLflowTracker(ExperimentTracker):
    """
    MLflow-based experiment tracker.

    Usage:
        config = TrackerConfig(
            backend="mlflow",
            tracking_uri="http://localhost:5000",
            experiment_name="my_experiment",
        )
        tracker = MLflowTracker(config)
        tracker.start_run("training_run")
        tracker.log_params({"lr": 0.001})
        tracker.log_metrics({"loss": 0.5}, step=1)
        tracker.log_artifact(model_path, "model")
        tracker.end_run()
    """

    def __init__(self, config: TrackerConfig) -> None:
        """
        Initialize MLflow tracker (resolves or creates the experiment).

        Raises:
            ImportError: If MLflow is not installed
        """
        client_cls, self._metric_cls, self._param_cls, self._tag_cls = _load_mlflow()
        super().__init__(config)
        self._client = client_cls(tracking_uri=config.tracking_uri)
        self._experiment_id = self._resolve_experiment(config.experiment_name or "default")

    def _resolve_experiment(self, name: str) -> str:
        """ID of experiment ``name``, created when missing (race-safe across workers)."""
        experiment = self._client.get_experiment_by_name(name)
        if experiment is not None:
            return str(experiment.experiment_id)
        try:
            return str(self._client.create_experiment(name))
        except Exception:
            # Another worker created it between the lookup and the create
            experiment = self._client.get_experiment_by_name(name)
            if experiment is None:
                raise
            return str(experiment.experiment_id)

    @property
    def experiment_id(self) -> str:
        """Get experiment ID."""
        return self._experiment_id

    def start_run(
        self,
        run_name: str | None = None,
        run_id: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """
        Start a new MLflow run (a child of ``config.parent_run_id`` when set).

        Args:
            run_name: Optional name for the run
            run_id: Optional ID of an existing run to continue
            tags: Optional tags to add to run

        Returns:
            Run ID
        """
        if self._is_active:
            logger.warning("Run already active, ending previous run")
            self.end_run(status="INTERRUPTED")

        all_tags = {**self.config.tags, **(tags or {})}
        if self.config.parent_run_id:
            all_tags[PARENT_RUN_TAG] = self.config.parent_run_id

        if run_id:
            self._run_id = run_id
            if all_tags:
                self._is_active = True
                self.set_tags(all_tags)
        else:
            run = self._client.create_run(
                experiment_id=self._experiment_id,
                tags={k: str(v) for k, v in all_tags.items()},
                run_name=run_name,
            )
            self._run_id = str(run.info.run_id)

        self._is_active = True
        logger.info(f"Started MLflow run: {self._run_id}")
        return self._run_id

    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current MLflow run.

        Args:
            status: Final status ("FINISHED", "FAILED", "KILLED", "INTERRUPTED")
        """
        if not self._is_active or self._run_id is None:
            logger.warning("No active run to end")
            return
        mlflow_status = _STATUS_MAP.get(status, "FINISHED")
        self._client.set_terminated(self._run_id, status=mlflow_status)
        logger.info(f"Ended MLflow run: {self._run_id} with status {mlflow_status}")
        self._is_active = False

    def log_params(self, params: dict[str, Any]) -> None:
        """
        Log parameters (values stringified, truncated to MLflow's length limit).

        Args:
            params: Dictionary of parameter name -> value
        """
        if not self._is_active or self._run_id is None:
            logger.warning("No active run, params not logged")
            return

        entries = []
        for key, value in params.items():
            if hasattr(value, "tolist"):  # numpy arrays / scalars
                value = value.tolist()
            entries.append(self._param_cls(key, str(value)[:_MAX_PARAM_VALUE_LENGTH]))
        for start in range(0, len(entries), _PARAMS_PER_BATCH):
            self._client.log_batch(self._run_id, params=entries[start : start + _PARAMS_PER_BATCH])

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """
        Log numeric metrics (None and non-numeric values are skipped).

        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number (epoch, iteration)
        """
        if not self._is_active or self._run_id is None:
            logger.warning("No active run, metrics not logged")
            return

        timestamp_ms = int(time.time() * 1000)
        entries = []
        for key, value in metrics.items():
            if value is None:
                continue
            try:
                # float() also unwraps numpy scalars and single-element tensors
                number = float(value)
            except (TypeError, ValueError):
                logger.debug(f"Skipping non-numeric metric: {key}={value}")
                continue
            entries.append(self._metric_cls(key, number, timestamp_ms, step or 0))
        for start in range(0, len(entries), _METRICS_PER_BATCH):
            self._client.log_batch(
                self._run_id, metrics=entries[start : start + _METRICS_PER_BATCH]
            )

    def log_artifact(
        self,
        path: Path | str,
        artifact_type: str | None = None,
    ) -> None:
        """
        Upload a file or directory to the run's artifact store.

        Args:
            path: Path to artifact
            artifact_type: Optional type label used as artifact subdirectory
        """
        if not self._is_active or self._run_id is None:
            logger.warning("No active run, artifact not logged")
            return

        if not self.config.log_artifacts:
            logger.debug("Artifact logging disabled")
            return

        path = Path(path)
        if not path.exists():
            logger.warning(f"Artifact path does not exist: {path}")
            return

        try:
            if path.is_dir():
                self._client.log_artifacts(self._run_id, str(path), artifact_path=artifact_type)
            else:
                self._client.log_artifact(self._run_id, str(path), artifact_path=artifact_type)
            logger.debug(f"Logged artifact: {path}")
        except Exception as e:
            # An unreachable artifact store must not fail the training run
            logger.warning(f"Failed to log artifact {path}: {e}")

    def set_tags(self, tags: dict[str, str]) -> None:
        """Add or overwrite tags on the current run."""
        if not self._is_active or self._run_id is None:
            logger.warning("No active run, tags not set")
            return
        self._client.log_batch(
            self._run_id, tags=[self._tag_cls(k, str(v)) for k, v in tags.items()]
        )


__all__ = ["MLflowTracker", "MLFLOW_INSTALL_HINT", "PARENT_RUN_TAG"]
