"""
Training package - Modular trainer implementation.

This package provides the Trainer class and related mixins for model training:

- Trainer: Main orchestrator for model training workflow
- TrainerFeaturesMixin: Feature selection and feature set resolution
- TrainerEvaluationMixin: Test set evaluation functionality
- TrainerArtifactsMixin: Artifact saving (configs, metrics, models)

Additionally, re-exports from src/training for the new module structure:
- UnifiedTrainingOrchestrator: THE single entry point (PHASE_3)
- train_pipeline: Convenience function for full pipeline training

Example:
    >>> from src.models.training import Trainer
    >>> from src.models.config import TrainerConfig
    >>> config = TrainerConfig(model_name="xgboost", horizon=20)
    >>> trainer = Trainer(config)
    >>> results = trainer.run(container)

New import path:
    from src.models.training import UnifiedTrainingOrchestrator, train_pipeline

Legacy import path (still works):
    from src.training import UnifiedTrainingOrchestrator, train_pipeline
"""

from .artifacts import TrainerArtifactsMixin
from .checksums import ArtifactChecksum, ArtifactIntegrityManager, compute_file_checksum
from .evaluation import INVALID_LABEL_SENTINEL, TrainerEvaluationMixin, _validate_labels
from .features import TrainerFeaturesMixin
from .trainer import Trainer

# Re-exports from src.training are loaded lazily to avoid circular imports
_TRAINING_EXPORTS = [
    "UnifiedTrainingOrchestrator",
    "TrainingRunResult",
    "ModelTrainingResult",
    "train_pipeline",
    "train_meta_labeling",
    "RegimeDetector",
    "RegimeDetectorConfig",
    "RegimeResult",
    "RegimeDetectionMethod",
    "detect_regimes",
    "RegimeAwareTrainer",
    "RegimeTrainingResult",
    "RegimeModelResult",
    "BetSizingStrategy",
    "BetSizingConfig",
    "compute_bet_sizes",
    "predict_with_sizing",
    "get_strategy_description",
    "ModelTrainer",
    "TrainedModelArtifact",
    "train_models",
    "TrainingOrchestrator",
    "ExperimentConfig",
    "ModelConfig",
    "ConfigLoader",
    "load_config_from_params",
    "load_config_from_yaml",
]


def __getattr__(name):
    """Lazy import for src.training exports to avoid circular imports."""
    if name in _TRAINING_EXPORTS:
        import src.training as training_module
        return getattr(training_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Local trainer classes
    "Trainer",
    "TrainerFeaturesMixin",
    "TrainerEvaluationMixin",
    "TrainerArtifactsMixin",
    "INVALID_LABEL_SENTINEL",
    "_validate_labels",
    # Checksums
    "ArtifactChecksum",
    "ArtifactIntegrityManager",
    "compute_file_checksum",
    # Re-exports from src.training (lazy loaded)
] + _TRAINING_EXPORTS
