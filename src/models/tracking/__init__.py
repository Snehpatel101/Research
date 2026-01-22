"""
Experiment tracking module.

Provides a unified interface for experiment tracking with multiple backends:
- MLflow (for full MLOps workflows)
- Local JSON (for offline/development)
"""

from .base import ExperimentTracker, TrackerConfig, get_tracker
from .local_tracker import LocalTracker

__all__ = [
    "ExperimentTracker",
    "TrackerConfig",
    "LocalTracker",
    "get_tracker",
]

# Optional MLflow import
try:
    from .mlflow_tracker import MLflowTracker

    __all__.append("MLflowTracker")
except ImportError:
    pass
