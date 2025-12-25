"""
Model Factory - Plugin-based model training system.

This package provides a unified interface for training ML models
on OHLCV time series data. It supports multiple model families
through a plugin architecture.

Supported Model Families:
- Boosting: XGBoost, LightGBM, CatBoost
- Neural: LSTM, GRU, TCN, MLP
- Transformer: PatchTST, Informer
- Classical: Random Forest, Logistic Regression, SVM

Quick Start:
-----------
    # Register a new model
    from src.models import BaseModel, register

    @register("my_model", family="boosting")
    class MyModel(BaseModel):
        ...

    # Create and train a model
    from src.models import ModelRegistry, Trainer, TrainerConfig

    config = TrainerConfig(model_name="xgboost", horizon=20)
    trainer = Trainer(config)
    results = trainer.run(container)

    # Or use the convenience function
    from src.models import train_model
    results = train_model("xgboost", container, horizon=20)

Architecture:
------------
    BaseModel: Abstract interface for all models
    ModelRegistry: Plugin system for model registration
    Trainer: Training orchestration
    TrainerConfig: Training configuration

    config/models/*.yaml: Model-specific configurations
"""
from __future__ import annotations

# Core classes
from .base import (
    BaseModel,
    PredictionOutput,
    TrainingMetrics,
)

from .registry import (
    ModelRegistry,
    register,
)

from .config import (
    TrainerConfig,
    CONFIG_DIR,
    load_yaml_config,
    load_model_config,
    build_config,
    create_trainer_config,
    merge_configs,
    validate_config,
    save_config,
)

from .trainer import (
    Trainer,
    train_model,
    evaluate_model,
    compute_classification_metrics,
)

from .device import (
    # Environment detection
    is_colab,
    is_kaggle,
    is_notebook,
    get_environment_info,
    setup_colab,
    # GPU detection
    GPUInfo,
    GPU_PROFILES,
    detect_cuda_available,
    get_gpu_count,
    get_gpu_info,
    get_best_gpu,
    get_device,
    # Memory estimation
    estimate_memory_requirements,
    get_optimal_batch_size,
    # Mixed precision
    get_amp_dtype,
    get_mixed_precision_config,
    # Optimal settings
    get_optimal_gpu_settings,
    # Convenience
    get_training_device_config,
    print_gpu_info,
    # Device manager
    DeviceManager,
)

# Auto-import model implementations to trigger registration
# These imports are necessary to register models with the ModelRegistry
from . import boosting  # XGBoost, LightGBM, CatBoost
from . import neural    # LSTM, GRU
from . import classical  # RandomForest, Logistic, SVM
from . import ensemble  # VotingEnsemble, StackingEnsemble, BlendingEnsemble

# Version
__version__ = "0.1.0"


# Public API
__all__ = [
    # Version
    "__version__",
    # Base classes
    "BaseModel",
    "PredictionOutput",
    "TrainingMetrics",
    # Registry
    "ModelRegistry",
    "register",
    # Configuration
    "TrainerConfig",
    "CONFIG_DIR",
    "load_yaml_config",
    "load_model_config",
    "build_config",
    "create_trainer_config",
    "merge_configs",
    "validate_config",
    "save_config",
    # Training
    "Trainer",
    "train_model",
    "evaluate_model",
    "compute_classification_metrics",
    # Environment detection
    "is_colab",
    "is_kaggle",
    "is_notebook",
    "get_environment_info",
    "setup_colab",
    # Device utilities
    "GPUInfo",
    "GPU_PROFILES",
    "detect_cuda_available",
    "get_gpu_count",
    "get_gpu_info",
    "get_best_gpu",
    "get_device",
    "estimate_memory_requirements",
    "get_optimal_batch_size",
    "get_amp_dtype",
    "get_mixed_precision_config",
    "get_optimal_gpu_settings",
    "get_training_device_config",
    "print_gpu_info",
    "DeviceManager",
]
