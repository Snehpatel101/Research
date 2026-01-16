"""
Unified training orchestration system.

This package provides a configuration-driven interface for training
any combination of models, features, optimization, and ensembles.

Main Components:
- TrainingOrchestrator: Master controller
- ConfigLoader: Config loading and validation
- FeatureSelector: Feature mode selection
- ModelFactory: Model instantiation
- OptunaOptimizer: Feature and hyperparameter optimization
- EnsembleBuilder: Ensemble creation

Usage:
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

__all__ = [
    "TrainingOrchestrator",
    "ExperimentConfig",
    "ModelConfig",
    "ConfigLoader",
    "load_config_from_params",
    "load_config_from_yaml",
]
