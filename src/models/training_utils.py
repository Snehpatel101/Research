"""
Convenience utilities for model training and evaluation.

This module provides high-level functions for training and evaluating models
without directly instantiating the Trainer class.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BaseModel
from .config import TrainerConfig
from .metrics import compute_classification_metrics

if TYPE_CHECKING:
    from src.phase1.stages.datasets.container import TimeSeriesDataContainer


def train_model(
    model_name: str,
    container: TimeSeriesDataContainer,
    horizon: int = 20,
    config_overrides: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Convenience function to train a model.

    Args:
        model_name: Name of model to train
        container: TimeSeriesDataContainer with data
        horizon: Label horizon
        config_overrides: Optional config overrides
        output_dir: Output directory (default: experiments/runs)

    Returns:
        Training results dict

    Example:
        >>> results = train_model(
        ...     "xgboost",
        ...     container,
        ...     horizon=20,
        ...     config_overrides={"max_depth": 8}
        ... )
    """
    # Import here to avoid circular dependency
    from .trainer import Trainer

    config_kwargs = {
        "model_name": model_name,
        "horizon": horizon,
    }

    if output_dir:
        config_kwargs["output_dir"] = output_dir

    if config_overrides:
        config_kwargs["model_config"] = config_overrides

    config = TrainerConfig(**config_kwargs)
    trainer = Trainer(config)
    return trainer.run(container)


def evaluate_model(
    model: BaseModel,
    container: TimeSeriesDataContainer,
    split: str = "test",
) -> dict[str, Any]:
    """
    Evaluate a trained model on a data split.

    Args:
        model: Trained model
        container: TimeSeriesDataContainer with data
        split: Data split to evaluate on ("val" or "test")

    Returns:
        Evaluation metrics dict

    Example:
        >>> model.load("experiments/runs/xgboost_h20_xxx/checkpoints/best_model")
        >>> metrics = evaluate_model(model, container, split="test")
    """
    if model.requires_sequences:
        dataset = container.get_pytorch_sequences(
            split, seq_len=60, symbol_isolated=True
        )
        # Convert to arrays (simplified - in practice use DataLoader)
        X_list, y_list = [], []
        for i in range(len(dataset)):
            X_i, y_i, _ = dataset[i]
            X_list.append(X_i)
            y_list.append(y_i)
        X = np.stack(X_list)
        y = np.array(y_list)
    else:
        X, y, _ = container.get_sklearn_arrays(split)

    predictions = model.predict(X)

    return compute_classification_metrics(
        y_true=y,
        y_pred=predictions.class_predictions,
        y_proba=predictions.class_probabilities,
    )


__all__ = [
    "train_model",
    "evaluate_model",
]
