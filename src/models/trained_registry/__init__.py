"""
Trained Model Registry module.

Provides centralized tracking and querying of all trained models.
"""

from .query import ModelQuery
from .registry import TrainedModelEntry, TrainedModelRegistry

__all__ = [
    "TrainedModelRegistry",
    "TrainedModelEntry",
    "ModelQuery",
]
