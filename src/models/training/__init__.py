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
from .evaluation import INVALID_LABEL_SENTINEL, TrainerEvaluationMixin, _validate_labels
from .features import TrainerFeaturesMixin
from .trainer import Trainer

# Config and loading
from .config import ExperimentConfig, ModelConfig
from .config_loader import ConfigLoader, load_config_from_params, load_config_from_yaml

# Legacy orchestrator
from .orchestrator import TrainingOrchestrator

# PHASE_3: Unified model trainer with adapter integration
from .model_trainer import ModelTrainer, TrainedModelArtifact, train_models

# PHASE_3: Unified Training Orchestrator - THE single entry point
from .unified_orchestrator import (
    UnifiedTrainingOrchestrator,
    TrainingRunResult,
    ModelTrainingResult,
    train_pipeline,
    train_meta_labeling,
)

# PHASE_3: Regime-aware training components
from .regime_detector import (
    RegimeDetector,
    RegimeDetectorConfig,
    RegimeResult,
    RegimeDetectionMethod,
    detect_regimes,
)
from .regime_trainer import (
    RegimeAwareTrainer,
    RegimeTrainingResult,
    RegimeModelResult,
)

# PHASE_3: Meta-labeling components
from .meta_labeling import (
    BetSizingStrategy,
    BetSizingConfig,
    compute_bet_sizes,
    predict_with_sizing,
    get_strategy_description,
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
