"""Service for generating out-of-fold predictions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging

import numpy as np
import pandas as pd

from src.validation.cv import OOFGenerator, OOFPrediction, PurgedKFold, PurgedKFoldConfig
from src.data.adapters import PreparedData

logger = logging.getLogger(__name__)


@dataclass
class OOFRequest:
    """Request to generate OOF predictions."""

    model_name: str
    horizon: int
    prepared_data: PreparedData
    n_splits: int = 5
    purge_bars: int = 10
    embargo_bars: int = 5


class OOFGenerationService:
    """
    Service for generating out-of-fold predictions.

    Responsibilities:
    - Create CV strategy
    - Generate OOF predictions via cross-validation
    - Return OOFPrediction object

    Does NOT:
    - Train the final model (handled by ModelTrainingService)
    - Align OOF predictions (handled by ensemble module)
    """

    def __init__(self, cache_dir: Path | None = None):
        """
        Initialize OOF generation service.

        Args:
            cache_dir: Optional directory for caching OOF predictions
        """
        self._cache_dir = cache_dir
        self._oof_generator: OOFGenerator | None = None

    def _ensure_generator(self, request: OOFRequest) -> OOFGenerator:
        """
        Ensure OOF generator is initialized with appropriate CV strategy.

        Args:
            request: OOF generation request

        Returns:
            Configured OOFGenerator instance
        """
        if self._oof_generator is None:
            cv_config = PurgedKFoldConfig(
                n_splits=request.n_splits,
                purge_bars=request.purge_bars,
                embargo_bars=request.embargo_bars,
            )
            cv = PurgedKFold(cv_config)
            self._oof_generator = OOFGenerator(cv, cache_dir=self._cache_dir)
        return self._oof_generator

    def _flatten_to_2d(self, X: np.ndarray, data_rank: int) -> np.ndarray:
        """
        Flatten multi-dimensional data to 2D for OOF generation.

        Args:
            X: Input array of any dimensionality
            data_rank: Rank of the data (2, 3, or 4)

        Returns:
            2D array of shape (n_samples, n_features)
        """
        if data_rank > 2:
            return X.reshape(X.shape[0], -1)
        return X

    def generate_oof(self, request: OOFRequest) -> OOFPrediction | None:
        """
        Generate out-of-fold predictions for a model.

        Args:
            request: OOF generation request containing model name, horizon,
                    prepared data, and CV configuration

        Returns:
            OOFPrediction object or None if generation fails
        """
        try:
            prepared = request.prepared_data
            model_name = request.model_name

            # Flatten to 2D for OOF generation
            X_train_2d = self._flatten_to_2d(prepared.X_train, prepared.data_rank)

            X_train_df = pd.DataFrame(
                X_train_2d,
                columns=[f"f{i}" for i in range(X_train_2d.shape[1])],
            )
            y_train = pd.Series(prepared.y_train)

            # Ensure generator is initialized
            oof_generator = self._ensure_generator(request)

            oof_predictions = oof_generator.generate_oof_predictions(
                X=X_train_df,
                y=y_train,
                model_configs={model_name: {}},
                use_cache=True,
            )

            return oof_predictions.get(model_name)

        except Exception as e:
            logger.warning(f"Failed to generate OOF for {request.model_name}: {e}")
            return None

    def set_cv_strategy(self, cv: PurgedKFold) -> None:
        """
        Set a custom CV strategy for OOF generation.

        Args:
            cv: PurgedKFold instance to use for cross-validation
        """
        self._oof_generator = OOFGenerator(cv, cache_dir=self._cache_dir)
