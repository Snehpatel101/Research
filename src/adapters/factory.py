"""
AdapterFactory - Creates appropriate adapters based on model and config.

PHASE_2: Single entry point for adapter creation. ALL data prep goes through adapters.

This factory integrates the existing adapter infrastructure with PipelineConfig
to provide a unified interface for creating model-appropriate data adapters.

Usage:
    from src.adapters.factory import AdapterFactory, create_adapter_factory
    from src.core.config import PipelineConfig

    config = PipelineConfig(
        symbol="MES",
        data_path="./data/mes_1min.parquet",
        output_dir="./experiments/exp_001",
        models=["xgboost", "lstm", "patchtst"],
    )

    factory = AdapterFactory(config)

    # Single model preparation
    result = factory.prepare_data("xgboost", df)
    # result.X.shape = (n_samples, n_features)

    # Heterogeneous ensemble preparation
    results = factory.prepare_heterogeneous(
        models=["xgboost", "lstm", "patchtst"],
        df=df_1min,
        additional_dfs={"5min": df_5min, "15min": df_15min},
    )
    # results["xgboost"].X.shape = (n_samples, n_features)
    # results["lstm"].X.shape = (n_sequences, 60, n_features)
    # results["patchtst"].X.shape = (n_sequences, 3, 60, 5)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

import pandas as pd

from src.core.constants import MODEL_ADAPTER_MAP, MODEL_DATA_RANKS
from src.core.config import PipelineConfig
from .registry import AdapterRegistry
from .base import AdapterResult, BaseAdapter

if TYPE_CHECKING:
    from src.core.interfaces import DataContract

logger = logging.getLogger(__name__)


class AdapterFactory:
    """
    Factory for creating model-appropriate adapters.

    Uses MODEL_ADAPTER_MAP from src/core/constants to determine
    which adapter type each model needs. Integrates with PipelineConfig
    to extract sequence_length, mtf_timeframes, and other parameters.

    The factory ensures that ALL data preparation goes through the
    adapter system - no bypassing allowed. This guarantees consistent
    data transformation across all models and training phases.

    Attributes:
        config: PipelineConfig with pipeline-wide settings
        _adapters: Cache of created adapter instances

    Example:
        factory = AdapterFactory(config)
        adapter = factory.get_adapter("xgboost")
        result = adapter.transform(df)

        # Or use prepare_data for convenience
        result = factory.prepare_data("xgboost", df)
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize factory with pipeline config.

        Args:
            config: PipelineConfig containing sequence_length, mtf_timeframes,
                and other parameters needed by different adapter types.
        """
        self.config = config
        self._adapters: Dict[str, BaseAdapter] = {}
        logger.debug(
            f"AdapterFactory initialized with config: "
            f"seq_len={config.sequence_length}, "
            f"mtf_timeframes={config.mtf_timeframes}"
        )

    def get_adapter(
        self,
        model_name: str,
        feature_columns: Optional[List[str]] = None,
        label_column: str = "label_h20",
        weight_column: Optional[str] = "sample_weight_h20",
        use_cache: bool = False,
    ) -> BaseAdapter:
        """
        Get or create adapter for a model.

        Determines the appropriate adapter type from MODEL_ADAPTER_MAP
        and creates an instance with parameters from PipelineConfig.

        Args:
            model_name: Name of the model (e.g., "xgboost", "lstm", "patchtst").
                Must be a valid model name from MODEL_ADAPTER_MAP.
            feature_columns: Explicit feature columns to use. If None,
                adapter will auto-detect features from the DataFrame.
            label_column: Name of the label column. Default "label_h20".
            weight_column: Name of the weight column. None for uniform weights.
            use_cache: If True, cache and reuse adapter instances by model name.
                Default False for thread safety.

        Returns:
            BaseAdapter instance appropriate for the model type:
                - TabularAdapter for boosting/classical/ensemble models
                - SequenceAdapter for RNN/TCN models
                - MultiStreamAdapter for PatchTST/iTransformer

        Raises:
            ValueError: If model_name is not in MODEL_ADAPTER_MAP.

        Example:
            adapter = factory.get_adapter("lstm", label_column="label_h10")
            result = adapter.transform(df)
        """
        model_key = model_name.lower()

        # Check cache if enabled
        if use_cache and model_key in self._adapters:
            logger.debug(f"Returning cached adapter for model: {model_name}")
            return self._adapters[model_key]

        # Determine adapter type from MODEL_ADAPTER_MAP
        adapter_type = MODEL_ADAPTER_MAP.get(model_key)
        if adapter_type is None:
            available = sorted(MODEL_ADAPTER_MAP.keys())
            raise ValueError(
                f"No adapter mapping for model: {model_name}. "
                f"Available models: {available[:10]}... ({len(available)} total)"
            )

        # Build kwargs based on adapter type
        kwargs: Dict[str, any] = {
            "feature_columns": feature_columns,
            "label_column": label_column,
            "weight_column": weight_column,
        }

        if adapter_type == "sequence":
            kwargs["sequence_length"] = self.config.sequence_length
            kwargs["stride"] = 1  # Default stride
            logger.debug(
                f"Creating sequence adapter for {model_name} with "
                f"sequence_length={self.config.sequence_length}"
            )

        elif adapter_type == "multi_stream":
            kwargs["sequence_length"] = self.config.sequence_length
            kwargs["timeframes"] = self.config.mtf_timeframes
            kwargs["stride"] = 1  # Default stride
            logger.debug(
                f"Creating multi_stream adapter for {model_name} with "
                f"sequence_length={self.config.sequence_length}, "
                f"timeframes={self.config.mtf_timeframes}"
            )

        # Create adapter instance via registry
        adapter = AdapterRegistry.create(adapter_type, **kwargs)

        # Cache if enabled
        if use_cache:
            self._adapters[model_key] = adapter

        logger.debug(
            f"Created {adapter_type} adapter for model: {model_name}"
        )

        return adapter

    def prepare_data(
        self,
        model_name: str,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        label_column: str = "label_h20",
        weight_column: Optional[str] = "sample_weight_h20",
        additional_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> AdapterResult:
        """
        Prepare data for a model - NO BYPASS.

        ALL data preparation goes through adapters. This method is the
        primary entry point for transforming DataFrames into model-ready
        numpy arrays.

        Args:
            model_name: Name of the model to prepare data for.
            df: Source DataFrame with features, labels, and optional weights.
            feature_columns: Explicit feature columns (None = auto-detect).
            label_column: Name of the label column.
            weight_column: Name of the weight column (None = uniform).
            additional_dfs: For multi-stream models, dictionary mapping
                timeframe strings to DataFrames (e.g., {"5min": df_5min}).

        Returns:
            AdapterResult containing:
                - X: Transformed feature array (2D, 3D, or 4D)
                - y: Label array
                - weights: Sample weights or None
                - Metadata about the transformation

        Raises:
            ValueError: If model_name is invalid or data validation fails.

        Example:
            # For tabular model
            result = factory.prepare_data("xgboost", df)
            model.fit(result.X, result.y, sample_weight=result.weights)

            # For sequence model
            result = factory.prepare_data("lstm", df)
            model.fit(result.X, result.y)  # X is (n, seq_len, features)

            # For multi-stream model
            result = factory.prepare_data(
                "patchtst", df,
                additional_dfs={"5min": df_5min, "15min": df_15min}
            )
        """
        adapter = self.get_adapter(
            model_name,
            feature_columns=feature_columns,
            label_column=label_column,
            weight_column=weight_column,
        )

        # Multi-stream adapters need additional_dfs parameter
        adapter_type = MODEL_ADAPTER_MAP.get(model_name.lower())
        if adapter_type == "multi_stream":
            result = adapter.transform(df, additional_dfs=additional_dfs)
        else:
            result = adapter.transform(df)

        logger.debug(
            f"Prepared data for {model_name}: X.shape={result.X.shape}, "
            f"y.shape={result.y.shape}, rank={result.data_rank}"
        )

        return result

    def prepare_heterogeneous(
        self,
        models: List[str],
        df: pd.DataFrame,
        additional_dfs: Optional[Dict[str, pd.DataFrame]] = None,
        feature_columns: Optional[List[str]] = None,
        label_column: str = "label_h20",
        weight_column: Optional[str] = "sample_weight_h20",
    ) -> Dict[str, AdapterResult]:
        """
        Prepare data for heterogeneous ensemble (different adapters per model).

        This method handles the common case of preparing data for multiple
        models with different input requirements (2D vs 3D vs 4D).

        Args:
            models: List of model names to prepare data for.
            df: Source DataFrame (used as-is for tabular, as anchor for sequences).
            additional_dfs: For multi-stream models, dictionary mapping
                timeframe strings to DataFrames.
            feature_columns: Explicit feature columns (None = auto-detect).
            label_column: Name of the label column.
            weight_column: Name of the weight column (None = uniform).

        Returns:
            Dictionary mapping model_name -> AdapterResult.

        Raises:
            ValueError: If any model name is invalid or data validation fails.

        Example:
            results = factory.prepare_heterogeneous(
                models=["xgboost", "lstm", "patchtst"],
                df=df_1min,
                additional_dfs={"5min": df_5min, "15min": df_15min},
            )

            for model_name, result in results.items():
                print(f"{model_name}: {result.X.shape}, rank={result.data_rank.value}")
            # xgboost: (10000, 162), rank=2
            # lstm: (9940, 60, 162), rank=3
            # patchtst: (9940, 3, 60, 5), rank=4
        """
        results: Dict[str, AdapterResult] = {}

        for model_name in models:
            adapter = self.get_adapter(
                model_name,
                feature_columns=feature_columns,
                label_column=label_column,
                weight_column=weight_column,
            )

            # Multi-stream adapters need additional_dfs
            adapter_type = MODEL_ADAPTER_MAP.get(model_name.lower())
            if adapter_type == "multi_stream":
                result = adapter.transform(df, additional_dfs=additional_dfs)
            else:
                result = adapter.transform(df)

            results[model_name] = result

            logger.debug(
                f"Heterogeneous prep - {model_name}: "
                f"X.shape={result.X.shape}, rank={result.data_rank}"
            )

        logger.info(
            f"Prepared heterogeneous data for {len(models)} models: "
            f"{list(results.keys())}"
        )

        return results

    def get_model_info(self, model_name: str) -> Dict[str, any]:
        """
        Get adapter info for a model without creating an adapter.

        Args:
            model_name: Name of the model.

        Returns:
            Dictionary with adapter_type, data_rank, and config parameters.

        Raises:
            ValueError: If model_name is not valid.
        """
        adapter_type = self.get_adapter_type(model_name)
        data_rank = self.get_data_rank(model_name)

        info = {
            "model_name": model_name,
            "adapter_type": adapter_type,
            "data_rank": data_rank,
        }

        if adapter_type == "sequence":
            info["sequence_length"] = self.config.sequence_length

        elif adapter_type == "multi_stream":
            info["sequence_length"] = self.config.sequence_length
            info["timeframes"] = self.config.mtf_timeframes
            info["n_timeframes"] = len(self.config.mtf_timeframes)

        return info

    def clear_cache(self) -> None:
        """Clear the adapter cache."""
        self._adapters.clear()
        logger.debug("Adapter cache cleared")

    @staticmethod
    def get_adapter_type(model_name: str) -> str:
        """
        Get adapter type for a model.

        Args:
            model_name: Name of the model.

        Returns:
            Adapter type string ("tabular", "sequence", or "multi_stream").

        Raises:
            ValueError: If model_name is not in MODEL_ADAPTER_MAP.
        """
        adapter_type = MODEL_ADAPTER_MAP.get(model_name.lower())
        if adapter_type is None:
            raise ValueError(f"Unknown model: {model_name}")
        return adapter_type

    @staticmethod
    def get_data_rank(model_name: str) -> int:
        """
        Get data rank (2/3/4) for a model.

        Args:
            model_name: Name of the model.

        Returns:
            Data rank integer (2 for tabular, 3 for sequence, 4 for multi-stream).

        Raises:
            ValueError: If model_name is not in MODEL_DATA_RANKS.
        """
        rank = MODEL_DATA_RANKS.get(model_name.lower())
        if rank is None:
            raise ValueError(f"Unknown model: {model_name}")
        return rank

    @staticmethod
    def list_models_by_adapter() -> Dict[str, List[str]]:
        """
        List all models grouped by their adapter type.

        Returns:
            Dictionary mapping adapter_type -> list of model names.

        Example:
            >>> AdapterFactory.list_models_by_adapter()
            {
                "tabular": ["xgboost", "lightgbm", "catboost", ...],
                "sequence": ["lstm", "gru", "tcn", ...],
                "multi_stream": ["patchtst", "itransformer"],
            }
        """
        grouped: Dict[str, List[str]] = {}
        for model, adapter in MODEL_ADAPTER_MAP.items():
            if adapter not in grouped:
                grouped[adapter] = []
            grouped[adapter].append(model)

        # Sort model lists for consistent output
        for adapter in grouped:
            grouped[adapter] = sorted(grouped[adapter])

        return grouped

    @staticmethod
    def list_models_by_rank() -> Dict[int, List[str]]:
        """
        List all models grouped by their data rank.

        Returns:
            Dictionary mapping rank -> list of model names.

        Example:
            >>> AdapterFactory.list_models_by_rank()
            {
                2: ["xgboost", "lightgbm", ...],
                3: ["lstm", "gru", "tcn", ...],
                4: ["patchtst", "itransformer"],
            }
        """
        grouped: Dict[int, List[str]] = {}
        for model, rank in MODEL_DATA_RANKS.items():
            if rank not in grouped:
                grouped[rank] = []
            grouped[rank].append(model)

        # Sort model lists for consistent output
        for rank in grouped:
            grouped[rank] = sorted(grouped[rank])

        return grouped


def create_adapter_factory(config: PipelineConfig) -> AdapterFactory:
    """
    Create an AdapterFactory from PipelineConfig.

    Convenience function for quick factory creation.

    Args:
        config: PipelineConfig with pipeline settings.

    Returns:
        AdapterFactory instance.

    Example:
        from src.core.config import PipelineConfig
        from src.adapters.factory import create_adapter_factory

        config = PipelineConfig(
            symbol="MES",
            data_path="./data/mes.parquet",
            output_dir="./experiments/exp_001",
        )
        factory = create_adapter_factory(config)
        result = factory.prepare_data("xgboost", df)
    """
    return AdapterFactory(config)


__all__ = [
    "AdapterFactory",
    "create_adapter_factory",
]
