"""
Convenience utilities for model training and evaluation.

This module provides high-level functions for training and evaluating models
without directly instantiating the Trainer class.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import BaseModel
from .config import TrainerConfig
from .metrics import compute_classification_metrics

if TYPE_CHECKING:
    from src.core.container import TimeSeriesDataContainer


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

    config_kwargs: dict[str, Any] = {
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
    if getattr(model, "requires_4d", False):
        from src.core.contracts import FeatureMode, get_model_contract
        from src.models.data_preparation import dataset_to_arrays

        model_name = getattr(model, "_get_model_type", lambda: "unknown")()
        contract = get_model_contract(model_name)

        timeframes = [contract.primary_timeframe, *list(contract.mtf_timeframes)]
        features_per_timeframe = None
        if contract.feature_mode == FeatureMode.RAW:
            features_per_timeframe = ["open", "high", "low", "close", "volume"]

        model_config = getattr(model, "_config", {})
        seq_len = (
            int(model_config.get("sequence_length", 60)) if isinstance(model_config, dict) else 60
        )
        dataset = container.get_multi_resolution_4d(
            split,
            seq_len=seq_len,
            timeframes=timeframes,
            features_per_timeframe=features_per_timeframe,
            include_base_features=True,
            symbol_isolated=True,
        )
        X, y, _ = dataset_to_arrays(dataset)  # type: ignore[arg-type]

    elif model.requires_sequences:
        dataset = container.get_pytorch_sequences(split, seq_len=60, symbol_isolated=True)
        # Use memory-efficient conversion (pre-allocate instead of list accumulation)
        from src.models.data_preparation import dataset_to_arrays

        X, y, _ = dataset_to_arrays(dataset)  # type: ignore[arg-type]
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
