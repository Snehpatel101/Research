"""
Cross-Validation Data Classes.

Contains dataclasses for CV fold metrics and results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FoldMetrics:
    """Metrics from a single CV fold."""

    fold: int
    train_size: int
    val_size: int
    accuracy: float
    f1: float
    precision: float
    recall: float
    training_time: float
    val_loss: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "training_time": self.training_time,
            "val_loss": self.val_loss,
        }


@dataclass
class CVResult:
    """
    Results from cross-validation run.

    Attributes:
        model_name: Name of the model
        horizon: Label horizon
        fold_metrics: List of per-fold metrics
        oos_predictions: DataFrame with OOF predictions
        feature_importance: Feature importance DataFrame
        tuned_params: Best hyperparameters from tuning
        selected_features: Features selected by walk-forward selection
        total_time: Total CV time in seconds
    """

    model_name: str
    horizon: int
    fold_metrics: list[FoldMetrics]
    oos_predictions: pd.DataFrame
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    tuned_params: dict[str, Any] = field(default_factory=dict)
    selected_features: list[str] = field(default_factory=list)
    total_time: float = 0.0

    @property
    def n_folds(self) -> int:
        return len(self.fold_metrics)

    @property
    def mean_accuracy(self) -> float:
        return np.mean([m.accuracy for m in self.fold_metrics])

    @property
    def mean_f1(self) -> float:
        return np.mean([m.f1 for m in self.fold_metrics])

    @property
    def std_f1(self) -> float:
        return np.std([m.f1 for m in self.fold_metrics])

    def get_stability_score(self) -> float:
        """Coefficient of variation for F1 score (lower = more stable)."""
        mean = self.mean_f1
        std = self.std_f1
        return std / mean if mean > 0 else float("inf")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "n_folds": self.n_folds,
            "mean_accuracy": self.mean_accuracy,
            "mean_f1": self.mean_f1,
            "std_f1": self.std_f1,
            "stability_score": self.get_stability_score(),
            "tuned_params": self.tuned_params,
            "n_selected_features": len(self.selected_features),
            "total_time": self.total_time,
            "fold_metrics": [m.to_dict() for m in self.fold_metrics],
        }


__all__ = ["FoldMetrics", "CVResult"]
