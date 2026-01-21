"""
Trainer - Orchestrates model training workflow.

This module re-exports the Trainer class from the training package for backward compatibility.

The Trainer class handles the complete training pipeline:
1. Load and prepare data from TimeSeriesDataContainer
2. Apply per-model feature selection (for tabular/classical models)
3. Apply model-specific preprocessing
4. Train model with early stopping
5. Evaluate on validation set
6. Save artifacts (model, metrics, predictions, feature selection)

Example:
    >>> from src.models.trainer import Trainer
    >>> from src.models.config import TrainerConfig
    >>> from src.core.container import TimeSeriesDataContainer
    ...
    >>> config = TrainerConfig(model_name="xgboost", horizon=20)
    >>> container = TimeSeriesDataContainer.from_parquet_dir(
    ...     "data/splits/scaled", horizon=20
    ... )
    ...
    >>> trainer = Trainer(config)
    >>> results = trainer.run(container)
    >>> print(results["evaluation_metrics"]["val_f1"])
"""

# Re-export from training package
from .training import (
    INVALID_LABEL_SENTINEL,
    Trainer,
    TrainerArtifactsMixin,
    TrainerEvaluationMixin,
    TrainerFeaturesMixin,
    _validate_labels,
)

# Re-export config for convenience
from .config import TrainerConfig

# Re-export metrics for convenience
from .metrics import compute_classification_metrics, compute_trading_metrics

# Re-export training utilities
from .training_utils import evaluate_model, train_model

__all__ = [
    # Core trainer
    "Trainer",
    "TrainerConfig",
    # Mixins (for extension)
    "TrainerFeaturesMixin",
    "TrainerEvaluationMixin",
    "TrainerArtifactsMixin",
    # Utilities
    "INVALID_LABEL_SENTINEL",
    "_validate_labels",
    "compute_classification_metrics",
    "compute_trading_metrics",
    "train_model",
    "evaluate_model",
]
