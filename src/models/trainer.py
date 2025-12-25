"""
Trainer - Orchestrates model training workflow.

The Trainer class handles the complete training pipeline:
1. Load and prepare data from TimeSeriesDataContainer
2. Apply model-specific preprocessing
3. Train model with early stopping
4. Evaluate on validation set
5. Save artifacts (model, metrics, predictions)

Example:
    >>> from src.models.trainer import Trainer
    >>> from src.models.config import TrainerConfig
    >>> from src.phase1.stages.datasets.container import TimeSeriesDataContainer
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
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import BaseModel, PredictionOutput, TrainingMetrics
from .config import TrainerConfig, save_config_json
from .registry import ModelRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities

    Returns:
        Dict with accuracy, F1 scores, confusion matrix, etc.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
        classification_report,
    )

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Per-class F1
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    per_class_f1 = f1_score(
        y_true, y_pred, average=None, labels=classes, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Class names for readability
    class_names = {0: "short", 1: "neutral", 2: "long"}

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "precision": float(precision),
        "recall": float(recall),
        "per_class_f1": {
            class_names.get(c, str(c)): float(f1)
            for c, f1 in zip(classes, per_class_f1)
        },
        "confusion_matrix": cm.tolist(),
        "n_samples": len(y_true),
    }


# =============================================================================
# TRAINER CLASS
# =============================================================================

class Trainer:
    """
    Orchestrates model training and evaluation.

    Handles the complete training workflow including data preparation,
    model training, evaluation, and artifact saving.

    Attributes:
        config: TrainerConfig with training settings
        model: Instantiated model from registry
        run_id: Unique identifier for this training run
        output_path: Path to output directory

    Example:
        >>> config = TrainerConfig(model_name="xgboost", horizon=20)
        >>> trainer = Trainer(config)
        >>> results = trainer.run(container)
    """

    def __init__(self, config: TrainerConfig) -> None:
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.run_id = self._generate_run_id()
        self.output_path = config.output_dir / self.run_id

        # Create model from registry
        self.model = ModelRegistry.create(
            config.model_name,
            config=config.model_config,
        )

        logger.info(
            f"Initialized Trainer: model={config.model_name}, "
            f"horizon={config.horizon}, run_id={self.run_id}"
        )

    def _generate_run_id(self) -> str:
        """Generate unique run identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.config.model_name}_h{self.config.horizon}_{timestamp}"

    def _setup_output_dir(self) -> None:
        """Create output directory structure."""
        dirs = [
            self.output_path / "config",
            self.output_path / "checkpoints",
            self.output_path / "predictions",
            self.output_path / "metrics",
            self.output_path / "plots",
            self.output_path / "logs",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Created output directory: {self.output_path}")

    def run(
        self,
        container: "TimeSeriesDataContainer",
        skip_save: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute complete training pipeline.

        Workflow:
        1. Setup output directories
        2. Load and prepare data
        3. Apply model-specific preprocessing
        4. Train model
        5. Evaluate on validation set
        6. Save artifacts

        Args:
            container: TimeSeriesDataContainer with train/val data
            skip_save: If True, skip saving artifacts (for testing)

        Returns:
            Dict with training results including:
            - run_id: Run identifier
            - training_metrics: TrainingMetrics from training
            - evaluation_metrics: Validation set metrics
            - output_path: Path to outputs
        """
        start_time = time.time()

        # Setup
        self._setup_output_dir()
        self._save_config()

        # Load data
        logger.info("Loading data from container...")
        X_train, y_train, w_train, X_val, y_val = self._prepare_data(container)

        # Log data shapes
        logger.info(
            f"Data shapes: "
            f"X_train={X_train.shape}, y_train={y_train.shape}, "
            f"X_val={X_val.shape}, y_val={y_val.shape}"
        )

        # Train
        logger.info(f"Training {self.config.model_name}...")
        training_metrics = self.model.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            sample_weights=w_train,
            config=self.config.model_config,
        )

        # Evaluate
        logger.info("Evaluating on validation set...")
        val_predictions = self.model.predict(X_val)
        eval_metrics = compute_classification_metrics(
            y_true=y_val,
            y_pred=val_predictions.class_predictions,
            y_proba=val_predictions.class_probabilities,
        )

        # Add trading metrics placeholder
        eval_metrics["trading"] = self._compute_trading_metrics(
            y_val, val_predictions
        )

        # Save artifacts
        if not skip_save:
            self._save_artifacts(training_metrics, eval_metrics, val_predictions)
            self._save_model()

        total_time = time.time() - start_time

        results = {
            "run_id": self.run_id,
            "model_name": self.config.model_name,
            "horizon": self.config.horizon,
            "training_metrics": training_metrics.to_dict(),
            "evaluation_metrics": eval_metrics,
            "output_path": str(self.output_path),
            "total_time_seconds": total_time,
        }

        logger.info(
            f"Training complete: "
            f"val_f1={eval_metrics['macro_f1']:.4f}, "
            f"val_accuracy={eval_metrics['accuracy']:.4f}, "
            f"time={total_time:.1f}s"
        )

        return results

    def _prepare_data(
        self,
        container: "TimeSeriesDataContainer",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training based on model requirements.

        Args:
            container: TimeSeriesDataContainer with data

        Returns:
            Tuple of (X_train, y_train, w_train, X_val, y_val)
        """
        if self.model.requires_sequences:
            # Get sequence data for sequential models
            train_dataset = container.get_pytorch_sequences(
                "train",
                seq_len=self.config.sequence_length,
                symbol_isolated=True,
            )
            val_dataset = container.get_pytorch_sequences(
                "val",
                seq_len=self.config.sequence_length,
                symbol_isolated=True,
            )

            # Convert to numpy arrays
            X_train, y_train, w_train = self._dataset_to_arrays(train_dataset)
            X_val, y_val, _ = self._dataset_to_arrays(val_dataset)

        else:
            # Get tabular data for non-sequential models
            X_train, y_train, w_train = container.get_sklearn_arrays("train")
            X_val, y_val, _ = container.get_sklearn_arrays("val")

        return X_train, y_train, w_train, X_val, y_val

    def _dataset_to_arrays(
        self,
        dataset: "Dataset",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert PyTorch dataset to numpy arrays.

        Args:
            dataset: SequenceDataset from container

        Returns:
            Tuple of (X, y, weights) numpy arrays
        """
        # Get all data from dataset
        X_list = []
        y_list = []
        w_list = []

        for i in range(len(dataset)):
            X_i, y_i, w_i = dataset[i]
            X_list.append(X_i)
            y_list.append(y_i)
            w_list.append(w_i)

        X = np.stack(X_list)
        y = np.array(y_list)
        w = np.array(w_list)

        return X, y, w

    def _compute_trading_metrics(
        self,
        y_true: np.ndarray,
        predictions: PredictionOutput,
    ) -> Dict[str, Any]:
        """
        Compute basic trading metrics.

        This is a placeholder for more sophisticated backtesting.
        Full trading metrics will be computed in Phase 3.

        Args:
            y_true: True labels
            predictions: Model predictions

        Returns:
            Dict with basic trading statistics
        """
        y_pred = predictions.class_predictions

        # Simple trading stats (placeholder)
        # 0=short, 1=neutral, 2=long
        long_signals = (y_pred == 2).sum()
        short_signals = (y_pred == 0).sum()
        neutral_signals = (y_pred == 1).sum()

        # Win rate when taking positions
        position_mask = y_pred != 1  # Not neutral
        if position_mask.sum() > 0:
            correct_positions = (y_pred[position_mask] == y_true[position_mask]).sum()
            position_win_rate = correct_positions / position_mask.sum()
        else:
            position_win_rate = 0.0

        return {
            "long_signals": int(long_signals),
            "short_signals": int(short_signals),
            "neutral_signals": int(neutral_signals),
            "position_win_rate": float(position_win_rate),
            "total_positions": int(long_signals + short_signals),
            "note": "Basic stats only. Full backtest in Phase 3.",
        }

    def _save_config(self) -> None:
        """Save training configuration."""
        config_path = self.output_path / "config" / "training_config.json"
        save_config_json(self.config.to_dict(), config_path)

        model_config_path = self.output_path / "config" / "model_config.json"
        save_config_json(self.config.model_config, model_config_path)

    def _save_artifacts(
        self,
        training_metrics: TrainingMetrics,
        eval_metrics: Dict[str, Any],
        predictions: PredictionOutput,
    ) -> None:
        """Save training artifacts."""
        # Training metrics
        metrics_path = self.output_path / "metrics" / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(training_metrics.to_dict(), f, indent=2)

        # Evaluation metrics
        eval_path = self.output_path / "metrics" / "evaluation_metrics.json"
        with open(eval_path, "w") as f:
            json.dump(eval_metrics, f, indent=2)

        # Predictions
        pred_path = self.output_path / "predictions" / "val_predictions.npz"
        np.savez(
            pred_path,
            class_predictions=predictions.class_predictions,
            class_probabilities=predictions.class_probabilities,
            confidence=predictions.confidence,
        )

        # Feature importance (if available)
        importance = self.model.get_feature_importance()
        if importance:
            importance_path = self.output_path / "metrics" / "feature_importance.json"
            with open(importance_path, "w") as f:
                json.dump(importance, f, indent=2)

        logger.debug(f"Saved artifacts to {self.output_path}")

    def _save_model(self) -> None:
        """Save trained model."""
        model_path = self.output_path / "checkpoints" / "best_model"
        self.model.save(model_path)
        logger.info(f"Saved model to {model_path}")


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def train_model(
    model_name: str,
    container: "TimeSeriesDataContainer",
    horizon: int = 20,
    config_overrides: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
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
    container: "TimeSeriesDataContainer",
    split: str = "test",
) -> Dict[str, Any]:
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
    "Trainer",
    "TrainerConfig",
    "compute_classification_metrics",
    "train_model",
    "evaluate_model",
]
