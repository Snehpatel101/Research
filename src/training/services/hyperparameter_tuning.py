# src/training/services/hyperparameter_tuning.py
"""Service for hyperparameter optimization."""

from dataclasses import dataclass
from typing import Any
import logging

import numpy as np

from src.adapters import PreparedData
from src.validation.cv import TimeSeriesOptunaTuner

logger = logging.getLogger(__name__)


@dataclass
class TuningRequest:
    """Request for hyperparameter tuning."""
    model_name: str
    horizon: int
    prepared_data: PreparedData
    n_splits: int = 5
    n_trials: int = 100


@dataclass
class TuningResult:
    """Result from hyperparameter tuning."""
    best_params: dict[str, Any]
    best_score: float
    n_trials_completed: int


class HyperparameterTuningService:
    """
    Service for hyperparameter optimization.

    Responsibilities:
    - Run Optuna trials
    - Return best hyperparameters

    Does NOT:
    - Create or train models (handled by ModelTrainingService)
    """

    def optimize(self, request: TuningRequest) -> TuningResult:
        """
        Run hyperparameter optimization.

        Args:
            request: TuningRequest containing model config and data

        Returns:
            TuningResult with best parameters and score
        """
        logger.info("  Running hyperparameter optimization...")

        tuner = TimeSeriesOptunaTuner(
            model_name=request.model_name,
            horizon=request.horizon,
            n_splits=request.n_splits,
        )

        # Flatten to 2D for tuning
        X_train = request.prepared_data.X_train
        X_train_2d = (
            X_train.reshape(X_train.shape[0], -1)
            if X_train.ndim > 2
            else X_train
        )

        best_params = tuner.optimize(
            X=X_train_2d,
            y=request.prepared_data.y_train,
            n_trials=request.n_trials,
        )

        logger.info(f"  Best params: {best_params}")

        # Get the best score from the tuner's study if available
        best_score = float("nan")
        if hasattr(tuner, "study") and tuner.study is not None:
            best_score = tuner.study.best_value

        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            n_trials_completed=request.n_trials,
        )
