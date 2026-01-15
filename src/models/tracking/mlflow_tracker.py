"""
MLflow experiment tracker.

Provides production-grade experiment tracking using MLflow.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .base import ExperimentTracker, TrackerConfig

logger = logging.getLogger(__name__)

# Lazy import MLflow to make it optional
try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None


class MLflowTracker(ExperimentTracker):
    """
    MLflow-based experiment tracker.

    Provides full MLOps capabilities including:
    - Experiment and run tracking
    - Parameter and metric logging
    - Artifact storage
    - Model registry integration

    Usage:
        config = TrackerConfig(
            backend="mlflow",
            tracking_uri="http://localhost:5000",
            experiment_name="my_experiment"
        )
        tracker = MLflowTracker(config)
        with tracker.run_context("training_run") as run_id:
            tracker.log_params({"lr": 0.001})
            tracker.log_metrics({"loss": 0.5}, step=1)
            tracker.log_artifact(model_path, "model")
    """

    def __init__(self, config: TrackerConfig) -> None:
        """
        Initialize MLflow tracker.

        Args:
            config: Tracker configuration

        Raises:
            ImportError: If MLflow is not installed
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError(
                "MLflow is not installed. Install with: pip install mlflow>=2.0"
            )

        super().__init__(config)

        # Set tracking URI
        if config.tracking_uri:
            mlflow.set_tracking_uri(config.tracking_uri)

        # Get or create experiment
        experiment_name = config.experiment_name or "default"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self._experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=str(config.output_dir / "mlruns" / experiment_name),
            )
        else:
            self._experiment_id = experiment.experiment_id

        self._client = MlflowClient()

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
        Start a new MLflow run.

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

        # Merge config tags with provided tags
        all_tags = dict(self.config.tags)
        if tags:
            all_tags.update(tags)

        # Start MLflow run
        if run_id:
            # Resume existing run
            run = mlflow.start_run(
                run_id=run_id,
                experiment_id=self._experiment_id,
            )
        else:
            # Start new run
            run = mlflow.start_run(
                run_name=run_name,
                experiment_id=self._experiment_id,
                tags=all_tags,
            )

        self._run_id = run.info.run_id
        self._is_active = True

        logger.info(f"Started MLflow run: {self._run_id}")
        return self._run_id

    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current MLflow run.

        Args:
            status: Final status ("FINISHED", "FAILED", "KILLED")
        """
        if not self._is_active:
            logger.warning("No active run to end")
            return

        # Map status to MLflow status
        status_map = {
            "FINISHED": "FINISHED",
            "FAILED": "FAILED",
            "KILLED": "KILLED",
            "INTERRUPTED": "KILLED",
        }
        mlflow_status = status_map.get(status, "FINISHED")

        mlflow.end_run(status=mlflow_status)
        logger.info(f"Ended MLflow run: {self._run_id} with status {mlflow_status}")

        self._is_active = False

    def log_params(self, params: dict[str, Any]) -> None:
        """
        Log parameters to MLflow.

        Args:
            params: Dictionary of parameter name -> value
        """
        if not self._is_active:
            logger.warning("No active run, params not logged")
            return

        # MLflow has limits on param value length (500 chars)
        processed_params = {}
        for key, value in params.items():
            if isinstance(value, Path):
                str_value = str(value)
            elif hasattr(value, "tolist"):  # numpy arrays
                str_value = str(value.tolist())[:500]
            elif isinstance(value, dict):
                str_value = str(value)[:500]
            elif isinstance(value, list):
                str_value = str(value)[:500]
            else:
                str_value = str(value)[:500] if value is not None else "None"

            processed_params[key] = str_value

        try:
            mlflow.log_params(processed_params)
        except Exception as e:
            logger.warning(f"Failed to log some params: {e}")
            # Try logging one by one
            for key, value in processed_params.items():
                try:
                    mlflow.log_param(key, value)
                except Exception as e2:
                    logger.debug(f"Failed to log param {key}: {e2}")

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """
        Log metrics to MLflow.

        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number (epoch, iteration)
        """
        if not self._is_active:
            logger.warning("No active run, metrics not logged")
            return

        # Process metrics to ensure they're floats
        processed_metrics = {}
        for key, value in metrics.items():
            if value is None:
                continue
            if hasattr(value, "item"):  # torch tensors, numpy scalars
                processed_metrics[key] = float(value.item())
            else:
                try:
                    processed_metrics[key] = float(value)
                except (TypeError, ValueError):
                    logger.debug(f"Skipping non-numeric metric: {key}={value}")

        if processed_metrics:
            mlflow.log_metrics(processed_metrics, step=step)

    def log_artifact(
        self,
        path: Path | str,
        artifact_type: str | None = None,
    ) -> None:
        """
        Log an artifact to MLflow.

        Args:
            path: Path to artifact
            artifact_type: Optional type label used as artifact subdirectory
        """
        if not self._is_active:
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
            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path=artifact_type)
            elif path.is_dir():
                mlflow.log_artifacts(str(path), artifact_path=artifact_type)
            logger.debug(f"Logged artifact: {path}")
        except Exception as e:
            logger.warning(f"Failed to log artifact {path}: {e}")

    def log_model(
        self,
        model_path: Path | str,
        model_name: str | None = None,
    ) -> None:
        """
        Log a trained model to MLflow.

        Args:
            model_path: Path to model file or directory
            model_name: Optional model name for registry
        """
        if not self._is_active:
            logger.warning("No active run, model not logged")
            return

        model_path = Path(model_path)
        if not model_path.exists():
            logger.warning(f"Model path does not exist: {model_path}")
            return

        # Log as artifact with "model" type
        self.log_artifact(model_path, artifact_type="model")

        # Optionally register in model registry
        if model_name:
            try:
                artifact_uri = f"runs:/{self._run_id}/model"
                mlflow.register_model(artifact_uri, model_name)
                logger.info(f"Registered model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to register model {model_name}: {e}")

    def set_tags(self, tags: dict[str, str]) -> None:
        """
        Set tags on the current run.

        Args:
            tags: Dictionary of tag name -> value
        """
        if not self._is_active:
            logger.warning("No active run, tags not set")
            return

        for key, value in tags.items():
            try:
                mlflow.set_tag(key, str(value))
            except Exception as e:
                logger.debug(f"Failed to set tag {key}: {e}")

    def log_figure(self, figure: Any, filename: str) -> None:
        """
        Log a matplotlib/plotly figure.

        Args:
            figure: Figure object (matplotlib or plotly)
            filename: Filename for the figure
        """
        if not self._is_active:
            logger.warning("No active run, figure not logged")
            return

        try:
            mlflow.log_figure(figure, filename)
        except Exception as e:
            logger.warning(f"Failed to log figure {filename}: {e}")

    def get_run_url(self) -> str | None:
        """Get URL to current run in MLflow UI."""
        if not self._run_id:
            return None

        tracking_uri = mlflow.get_tracking_uri()
        return f"{tracking_uri}/#/experiments/{self._experiment_id}/runs/{self._run_id}"


__all__ = ["MLflowTracker", "MLFLOW_AVAILABLE"]
