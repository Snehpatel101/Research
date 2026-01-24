"""
Training package - Unified training orchestration system.

This package provides a configuration-driven interface for training
any combination of models, features, optimization, and ensembles.

Main Components:
- UnifiedTrainingOrchestrator: THE single entry point (PHASE_3)
- ModelTrainer: PHASE_3 unified model trainer with adapter integration
- Trainer: Main orchestrator for model training workflow
- TrainerFeaturesMixin: Feature selection and feature set resolution
- TrainerEvaluationMixin: Test set evaluation functionality
- TrainerArtifactsMixin: Artifact saving (configs, metrics, models)

Usage:
    from src.models.training import UnifiedTrainingOrchestrator, train_pipeline

    config = PipelineConfig(
        symbol="MES",
        data_path="./data/mes_1min.parquet",
        output_dir="./experiments/exp_001",
        models=["xgboost", "lightgbm", "lstm"],
        build_ensemble=True,
    )

    result = train_pipeline(config, df)
"""

# Local trainer classes
from .artifacts import TrainerArtifactsMixin
from .checksums import ArtifactChecksum, ArtifactIntegrityManager, compute_file_checksum

# Config and loading
from .config import ExperimentConfig, ModelConfig
from .config_loader import ConfigLoader, load_config_from_params, load_config_from_yaml
from .evaluation import INVALID_LABEL_SENTINEL, TrainerEvaluationMixin, _validate_labels
from .features import TrainerFeaturesMixin

# PHASE_3: Meta-labeling components
from .meta_labeling import (
    BetSizingConfig,
    BetSizingStrategy,
    compute_bet_sizes,
    get_strategy_description,
    predict_with_sizing,
)

# PHASE_3: Unified model trainer with adapter integration
from .model_trainer import ModelTrainer, TrainedModelArtifact, train_models

# PHASE_3: Regime-aware training components
from .regime_detector import (
    RegimeDetectionMethod,
    RegimeDetector,
    RegimeDetectorConfig,
    RegimeResult,
    detect_regimes,
)
from .regime_trainer import (
    RegimeAwareTrainer,
    RegimeModelResult,
    RegimeTrainingResult,
)
from .trainer import Trainer

# PHASE_3: Unified Training Orchestrator - THE single entry point
from .unified_orchestrator import (
    ModelTrainingResult,
    PreTrainingValidationError,
    TrainingRunResult,
    UnifiedTrainingOrchestrator,
    train_meta_labeling,
    train_pipeline,
)

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
    # PHASE_3: Unified Training Orchestrator (THE entry point)
    "UnifiedTrainingOrchestrator",
    "PreTrainingValidationError",
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
    # Config and loading (used by legacy code, but TrainingOrchestrator removed)
    "ExperimentConfig",
    "ModelConfig",
    "ConfigLoader",
    "load_config_from_params",
    "load_config_from_yaml",
]
