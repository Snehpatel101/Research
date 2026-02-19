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

        Note:
            Supports all data ranks (2D, 3D, 4D). For 3D/4D data, native
            numpy arrays are passed to the tuner which indexes by sample
            axis (axis 0). MedianPruner is used to early-stop bad trials.
        """
        import pandas as pd

        from src.validation.cv import PurgedKFold, PurgedKFoldConfig

        data_rank = request.prepared_data.data_rank

        logger.info(
            f"  Running hyperparameter optimization "
            f"({data_rank}D data, model: {request.model_name})..."
        )

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

        X_train = request.prepared_data.X_train
        y_train = request.prepared_data.y_train

        # Prepare data based on rank
        if data_rank == 2:
            # 2D: convert to pandas for tuner API
            X_input = pd.DataFrame(X_train)
            y_input = pd.Series(y_train)
        else:
            # 3D/4D: pass numpy arrays directly (tuner handles rank-aware indexing)
            X_input = X_train  # np.ndarray shape (n, seq, feat) or (n, tf, seq, feat)
            y_input = y_train  # np.ndarray shape (n,)

        # CRITICAL: Filter invalid labels (-99) before tuning
        # The sentinel value -99 marks invalid/ambiguous samples that should be excluded
        INVALID_LABEL = -99
        if isinstance(y_input, pd.Series):
            valid_mask = y_input != INVALID_LABEL
            n_invalid = (~valid_mask).sum()
            if n_invalid > 0:
                logger.warning(
                    f"  Filtering {n_invalid} invalid labels (-99) from tuning data "
                    f"({n_invalid / len(y_input) * 100:.2f}% of samples)"
                )
                X_input = X_input.loc[valid_mask].reset_index(drop=True)
                y_input = y_input.loc[valid_mask].reset_index(drop=True)
        else:
            valid_mask = y_input != INVALID_LABEL
            n_invalid = int((~valid_mask).sum())
            if n_invalid > 0:
                logger.warning(
                    f"  Filtering {n_invalid} invalid labels (-99) from tuning data "
                    f"({n_invalid / len(y_input) * 100:.2f}% of samples)"
                )
                X_input = X_input[valid_mask]
                y_input = y_input[valid_mask]

        result = tuner.tune(
            X=X_input,
            y=y_input,
            data_rank=data_rank,
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
