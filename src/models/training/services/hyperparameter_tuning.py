# src/training/services/hyperparameter_tuning.py
"""Service for hyperparameter optimization."""

import logging
from dataclasses import dataclass
from typing import Any

from src.data.adapters import PreparedData
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
    scoring: str = "f1_weighted"  # Optimization metric (from PipelineConfig.optuna_metric)


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
        import pandas as pd

        from src.validation.cv import PurgedKFold, PurgedKFoldConfig

        logger.info("  Running hyperparameter optimization...")

        # Create purged K-fold CV
        cv_config = PurgedKFoldConfig(
            n_splits=request.n_splits,
            embargo_bars=request.horizon * 2,  # Use horizon for embargo
        )
        cv = PurgedKFold(cv_config)

        tuner = TimeSeriesOptunaTuner(
            model_name=request.model_name,
            cv=cv,
            n_trials=request.n_trials,
            metric=request.scoring,
        )

        # Flatten to 2D for tuning
        X_train = request.prepared_data.X_train
        X_train_2d = X_train.reshape(X_train.shape[0], -1) if X_train.ndim > 2 else X_train

        # Convert to pandas for tuner API
        X_df = pd.DataFrame(X_train_2d)
        y_series = pd.Series(request.prepared_data.y_train)

        # CRITICAL: Filter invalid labels (-99) before tuning
        # The sentinel value -99 marks invalid/ambiguous samples that should be excluded
        INVALID_LABEL = -99
        valid_mask = y_series != INVALID_LABEL
        n_invalid = (~valid_mask).sum()
        if n_invalid > 0:
            logger.warning(
                f"  Filtering {n_invalid} invalid labels (-99) from tuning data "
                f"({n_invalid / len(y_series) * 100:.2f}% of samples)"
            )
            X_df = X_df.loc[valid_mask].reset_index(drop=True)
            y_series = y_series.loc[valid_mask].reset_index(drop=True)

        result = tuner.tune(
            X=X_df,
            y=y_series,
        )

        logger.info(f"  Best params: {result.get('best_params', {})}")

        # Get the best score from the result
        best_score = result.get("best_value", float("nan"))
        if best_score is None:
            best_score = float("nan")

        return TuningResult(
            best_params=result.get("best_params", {}),
            best_score=float(best_score) if best_score is not None else float("nan"),
            n_trials_completed=request.n_trials,
        )
