"""
Experiment tracking module.

Provides a unified interface for experiment tracking with multiple backends:
- "none": no-op
- "local": JSON files (offline/development, no dependencies)
- "mlflow": MLflow tracking server or store (optional: ``pip install '.[mlflow]'``;
  imported only when an MLflow tracker is created)

``MLFactory.run`` opens one parent run per experiment run and every trained
model logs a child run under it (see ``ExperimentConfig.tracking``).
"""

from .base import DisabledTracker, ExperimentTracker, TrackerConfig, flatten_params, get_tracker
from .local_tracker import LocalTracker
from .mlflow_tracker import MLflowTracker

__all__ = [
    "DisabledTracker",
    "ExperimentTracker",
    "TrackerConfig",
    "LocalTracker",
    "MLflowTracker",
    "flatten_params",
    "get_tracker",
]
