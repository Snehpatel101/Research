"""
DEPRECATED: Import from 'src.models.training' instead.

This module will be removed in v2.0.0.

Migration:
    # Old (deprecated)
    from src.training import UnifiedTrainingOrchestrator, train_pipeline

    # New (preferred)
    from src.models.training import UnifiedTrainingOrchestrator, train_pipeline
"""

import warnings

_TRAINING_EXPORTS = [
    # PHASE_3: Unified Training Orchestrator (THE entry point)
    "UnifiedTrainingOrchestrator",
    "TrainingRunResult",
    "ModelTrainingResult",
    "train_pipeline",
    "train_meta_labeling",
    # PHASE_3: Regime-aware training
    "RegimeDetector",
    "RegimeDetectorConfig",
    "RegimeResult",
    "RegimeDetectionMethod",
    "detect_regimes",
    "RegimeAwareTrainer",
    "RegimeTrainingResult",
    "RegimeModelResult",
    # PHASE_3: Meta-labeling components
    "BetSizingStrategy",
    "BetSizingConfig",
    "compute_bet_sizes",
    "predict_with_sizing",
    "get_strategy_description",
    # PHASE_3: Unified model trainer
    "ModelTrainer",
    "TrainedModelArtifact",
    "train_models",
    # Legacy orchestrator
    "TrainingOrchestrator",
    "ExperimentConfig",
    "ModelConfig",
    "ConfigLoader",
    "load_config_from_params",
    "load_config_from_yaml",
]


def __getattr__(name: str):
    """Lazy import to avoid circular imports."""
    if name in _TRAINING_EXPORTS:
        warnings.warn(
            f"Importing '{name}' from 'src.training' is deprecated. "
            f"Use 'from src.models.training import {name}' instead. "
            "This will be removed in v2.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        from src.models import training as training_module

        return getattr(training_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return _TRAINING_EXPORTS
