"""
Training package - Modular trainer implementation.

This package provides the Trainer class and related mixins for model training:

- Trainer: Main orchestrator for model training workflow
- TrainerFeaturesMixin: Feature selection and feature set resolution
- TrainerEvaluationMixin: Test set evaluation functionality
- TrainerArtifactsMixin: Artifact saving (configs, metrics, models)

Example:
    >>> from src.models.training import Trainer
    >>> from src.models.config import TrainerConfig
    >>> config = TrainerConfig(model_name="xgboost", horizon=20)
    >>> trainer = Trainer(config)
    >>> results = trainer.run(container)
"""

from .artifacts import TrainerArtifactsMixin
from .evaluation import INVALID_LABEL_SENTINEL, TrainerEvaluationMixin, _validate_labels
from .features import TrainerFeaturesMixin
from .trainer import Trainer

__all__ = [
    "Trainer",
    "TrainerFeaturesMixin",
    "TrainerEvaluationMixin",
    "TrainerArtifactsMixin",
    "INVALID_LABEL_SENTINEL",
    "_validate_labels",
]
