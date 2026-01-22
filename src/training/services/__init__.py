"""
Training services package.

Provides modular services extracted from UnifiedTrainingOrchestrator:
- ArtifactManager: Save and load training artifacts
- HyperparameterTuningService: Optuna-based hyperparameter optimization
- ModelTrainingService: Train individual models
- OOFGenerationService: Generate out-of-fold predictions
"""

from .artifact_persistence import ArtifactManager, ArtifactSaveRequest
from .hyperparameter_tuning import (
    HyperparameterTuningService,
    TuningRequest,
    TuningResult,
)
from .model_training import (
    ModelTrainingRequest,
    ModelTrainingResult,
    ModelTrainingService,
)
from .oof_generation import OOFGenerationService, OOFRequest

__all__ = [
    # Artifact persistence
    "ArtifactManager",
    "ArtifactSaveRequest",
    # Hyperparameter tuning
    "HyperparameterTuningService",
    "TuningRequest",
    "TuningResult",
    # Model training
    "ModelTrainingRequest",
    "ModelTrainingResult",
    "ModelTrainingService",
    # OOF generation
    "OOFGenerationService",
    "OOFRequest",
]
