"""
Unified training orchestration system.

This package provides a configuration-driven interface for training
any combination of models, features, optimization, and ensembles.

Main Components:
- UnifiedTrainingOrchestrator: THE single entry point (PHASE_3)
- ModelTrainer: PHASE_3 unified model trainer with adapter integration
- TrainingOrchestrator: Master controller (legacy)
- ConfigLoader: Config loading and validation
- FeatureSelector: Feature mode selection
- ModelFactory: Model instantiation
- OptunaOptimizer: Feature and hyperparameter optimization
- EnsembleBuilder: Ensemble creation

PHASE_3 Usage (RECOMMENDED - UnifiedTrainingOrchestrator):
    from src.core import PipelineConfig
    from src.training import UnifiedTrainingOrchestrator, train_pipeline

    config = PipelineConfig(
        symbol="MES",
        data_path="./data/mes_1min.parquet",
        output_dir="./experiments/exp_001",
        models=["xgboost", "lightgbm", "lstm"],
        build_ensemble=True,
    )

    # Option 1: Use UnifiedTrainingOrchestrator class
    orchestrator = UnifiedTrainingOrchestrator(config)
    result = orchestrator.train(df)

    # Option 2: Use convenience function
    result = train_pipeline(config, df)

PHASE_3 Usage (ModelTrainer - per-model training):
    from src.core import PipelineConfig
    from src.training import ModelTrainer, train_models

    config = PipelineConfig(...)

    # Option 1: Use ModelTrainer class
    trainer = ModelTrainer(config)
    artifact = trainer.train_model("xgboost", df, horizon=20)
    artifacts = trainer.train_all(df)

    # Option 2: Use convenience function
    artifacts = train_models(config, df)

Legacy Usage:
    from src.training import TrainingOrchestrator

    config = {
        'data': {'symbol': 'MES', 'horizons': [20]},
        'models': {'model_list': [{'name': 'xgboost'}]},
        'features': {'mode': 'full'}
    }

    orchestrator = TrainingOrchestrator(config)
    results = orchestrator.run()
"""

from .config import ExperimentConfig, ModelConfig
from .config_loader import ConfigLoader, load_config_from_params, load_config_from_yaml
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

__all__ = [
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
