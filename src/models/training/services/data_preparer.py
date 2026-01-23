# src/models/training/services/data_preparer.py
"""
Service for preparing data for training.

Wraps UnifiedDataPreparation to provide a simpler interface for the orchestrator.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.data.adapters import PreparedData, UnifiedDataPreparation

if TYPE_CHECKING:
    import pandas as pd

    from src.core import PipelineConfig

logger = logging.getLogger(__name__)


class DataPreparer:
    """
    Service for preparing data for training.

    Wraps UnifiedDataPreparation and provides a consistent interface
    for the orchestrator to use.

    Responsibilities:
    - Initialize data preparation with config
    - Prepare data for specific models
    - Handle multi-timeframe data preparation

    Does NOT:
    - Train models (handled by ModelTrainingService)
    - Generate OOF (handled by OOFGenerationService)
    """

    def __init__(self, config: PipelineConfig) -> None:
        """
        Initialize DataPreparer with PipelineConfig.

        Args:
            config: PipelineConfig from src/core
        """
        self.config = config
        self._data_prep = UnifiedDataPreparation(config)

    def prepare(
        self,
        df: pd.DataFrame,
        model_name: str,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> PreparedData:
        """
        Prepare data for a specific model.

        Args:
            df: Raw OHLCV DataFrame
            model_name: Name of the model to prepare data for
            additional_dfs: Optional additional timeframe DataFrames

        Returns:
            PreparedData ready for training
        """
        logger.debug(f"Preparing data for model: {model_name}")

        prepared = self._data_prep.prepare(
            df=df,
            model_name=model_name,
            additional_dfs=additional_dfs,
        )

        logger.debug(f"  Data prepared: {prepared.summary()}")
        return prepared

    def get_feature_names(self, model_name: str, df: pd.DataFrame) -> list[str]:
        """
        Get feature names for a specific model by preparing data.

        Note: This prepares the data to get the feature names. For production
        use, prefer calling prepare() and accessing feature_names from the result.

        Args:
            model_name: Name of the model
            df: DataFrame to extract features from

        Returns:
            List of feature names
        """
        prepared = self._data_prep.prepare(df=df, model_name=model_name)
        return prepared.feature_names

    @property
    def data_prep(self) -> UnifiedDataPreparation:
        """Get underlying UnifiedDataPreparation instance."""
        return self._data_prep


__all__ = ["DataPreparer"]
