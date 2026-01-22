"""
Unified Data Preparation - ALL data goes through adapters.

PHASE_2: No bypass paths. Every model gets data via its adapter.

This module provides the UnifiedDataPreparation class which is THE entry
point for preparing data for any model. It enforces:
- All data goes through appropriate adapters (no bypass)
- Proper chronological splitting with purge/embargo gaps
- Fit-on-train scaling (prevents data leakage)
- Consistent data format per model type

Design Principles:
- Single entry point for all data preparation
- Adapter-first: model requirements determine data format
- Leakage prevention: chronological splits with gaps
- Reproducibility: scalers saved for inference
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import logging

import numpy as np
import pandas as pd

from src.core.config import PipelineConfig
from src.core.constants import (
    MODEL_ADAPTER_MAP,
    MODEL_DATA_RANKS,
    DEFAULT_SEQUENCE_LENGTH,
)
from .registry import AdapterRegistry, get_adapter
from .scaling import AdapterScaler, ScalerConfig
from .base import AdapterResult


logger = logging.getLogger(__name__)


@dataclass
class PreparedData:
    """
    Prepared data for a single model.

    Contains scaled, adapter-transformed data ready for training.
    All data has been:
    1. Split chronologically (train/val/test)
    2. Transformed through the appropriate adapter
    3. Scaled (fit on train only)

    Attributes:
        X_train: Training features (2D, 3D, or 4D depending on model)
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features (optional)
        y_test: Test labels (optional)
        model_name: Name of the model this data is prepared for
        adapter_type: Type of adapter used ("tabular", "sequence", "multi_stream")
        data_rank: Dimensionality of the data (2, 3, or 4)
        feature_names: List of feature column names
        train_indices: Original DataFrame indices for training samples
        val_indices: Original DataFrame indices for validation samples
        test_indices: Original DataFrame indices for test samples
        scaler: The fitted scaler (for inference)
        sequence_length: Sequence length if applicable (3D/4D)
        n_timeframes: Number of timeframes if multi-stream (4D)
    """
    # Data arrays - required
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray

    # Data arrays - optional (test set)
    X_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None

    # Weights (optional)
    train_weights: Optional[np.ndarray] = None
    val_weights: Optional[np.ndarray] = None
    test_weights: Optional[np.ndarray] = None

    # Metadata
    model_name: str = ""
    adapter_type: str = ""
    data_rank: int = 2
    feature_names: List[str] = field(default_factory=list)

    # Index tracking (for mapping predictions back to source)
    train_indices: Optional[np.ndarray] = None
    val_indices: Optional[np.ndarray] = None
    test_indices: Optional[np.ndarray] = None

    # Scaler for inference
    scaler: Optional[AdapterScaler] = None

    # Sequence/multi-stream metadata
    sequence_length: Optional[int] = None
    n_timeframes: Optional[int] = None
    timeframe_names: List[str] = field(default_factory=list)

    @property
    def n_train(self) -> int:
        """Number of training samples."""
        return self.X_train.shape[0]

    @property
    def n_val(self) -> int:
        """Number of validation samples."""
        return self.X_val.shape[0]

    @property
    def n_test(self) -> int:
        """Number of test samples (0 if no test set)."""
        return self.X_test.shape[0] if self.X_test is not None else 0

    @property
    def n_features(self) -> int:
        """Number of features (last dimension of X)."""
        return self.X_train.shape[-1]

    @property
    def has_test(self) -> bool:
        """Check if test set is available."""
        return self.X_test is not None and self.y_test is not None

    @property
    def has_weights(self) -> bool:
        """Check if sample weights are available."""
        return self.train_weights is not None

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate prepared data consistency.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues: List[str] = []

        # Check X/y length match
        if self.X_train.shape[0] != self.y_train.shape[0]:
            issues.append(
                f"Train X/y length mismatch: X={self.X_train.shape[0]}, y={len(self.y_train)}"
            )
        if self.X_val.shape[0] != self.y_val.shape[0]:
            issues.append(
                f"Val X/y length mismatch: X={self.X_val.shape[0]}, y={len(self.y_val)}"
            )
        if self.has_test and self.X_test.shape[0] != self.y_test.shape[0]:
            issues.append(
                f"Test X/y length mismatch: X={self.X_test.shape[0]}, y={len(self.y_test)}"
            )

        # Check weights length if present
        if self.train_weights is not None:
            if len(self.train_weights) != self.n_train:
                issues.append(
                    f"Train weights length mismatch: {len(self.train_weights)} vs {self.n_train}"
                )

        # Check for NaN/Inf
        for name, arr in [("X_train", self.X_train), ("X_val", self.X_val)]:
            if np.isnan(arr).any():
                issues.append(f"{name} contains NaN values")
            if np.isinf(arr).any():
                issues.append(f"{name} contains Inf values")

        if self.has_test:
            if np.isnan(self.X_test).any():
                issues.append("X_test contains NaN values")
            if np.isinf(self.X_test).any():
                issues.append("X_test contains Inf values")

        # Check data rank matches array dimensions
        if self.X_train.ndim != self.data_rank:
            issues.append(
                f"Data rank mismatch: declared={self.data_rank}, actual={self.X_train.ndim}"
            )

        return len(issues) == 0, issues

    def summary(self) -> str:
        """Generate a summary string for logging."""
        test_info = f", test={self.n_test}" if self.has_test else ""
        return (
            f"PreparedData({self.model_name}): "
            f"train={self.n_train}, val={self.n_val}{test_info}, "
            f"features={self.n_features}, rank={self.data_rank}D, "
            f"adapter={self.adapter_type}"
        )


class UnifiedDataPreparation:
    """
    Unified data preparation - ALL data goes through adapters.

    This is THE entry point for data preparation. No bypass paths.
    Every model gets its data in the format it needs via the appropriate
    adapter (tabular, sequence, or multi_stream).

    Pipeline:
    1. Split data chronologically (train/val/test) with purge/embargo
    2. Apply adapter transformation (2D/3D/4D based on model)
    3. Fit scaler on train only, transform all splits
    4. Return PreparedData with all metadata

    Example:
        >>> from src.core.config import PipelineConfig
        >>> config = PipelineConfig(
        ...     symbol="MES",
        ...     data_path="./data/mes.parquet",
        ...     output_dir="./output",
        ... )
        >>> prep = UnifiedDataPreparation(config)
        >>> data = prep.prepare(df, model_name="xgboost")
        >>> print(data.X_train.shape)  # (n_train, n_features)
        >>>
        >>> # For LSTM (sequence model)
        >>> data = prep.prepare(df, model_name="lstm")
        >>> print(data.X_train.shape)  # (n_train, seq_len, n_features)
    """

    def __init__(
        self,
        config: PipelineConfig,
        scaler_config: Optional[ScalerConfig] = None,
    ):
        """
        Initialize unified data preparation.

        Args:
            config: Pipeline configuration with split ratios, purge/embargo
                    settings, and sequence length for neural models.
            scaler_config: Optional custom scaler configuration. Defaults to
                          RobustScaler with clip_value=5.0.
        """
        self.config = config
        self.scaler_config = scaler_config or ScalerConfig()
        self._scalers: Dict[str, AdapterScaler] = {}

    def prepare(
        self,
        df: pd.DataFrame,
        model_name: str,
        feature_columns: Optional[List[str]] = None,
        label_column: str = "label",
        weight_column: Optional[str] = None,
        additional_dfs: Optional[Dict[str, pd.DataFrame]] = None,
        apply_scaling: bool = True,
    ) -> PreparedData:
        """
        Prepare data for a single model - NO BYPASS.

        ALL data preparation goes through the appropriate adapter based on
        the model's requirements. There are no special paths or bypasses.

        Args:
            df: Source DataFrame with features, labels, and optionally weights.
                Must be sorted chronologically (oldest first).
            model_name: Name of the model to prepare data for. This determines
                       which adapter is used (tabular, sequence, multi_stream).
            feature_columns: Optional list of feature columns. If None, columns
                            are auto-detected (excluding metadata and labels).
            label_column: Name of the label column.
            weight_column: Optional name of the sample weight column.
            additional_dfs: For multi-stream models, dictionary mapping timeframe
                           strings to DataFrames (e.g., {"5min": df_5min}).
            apply_scaling: Whether to apply scaling. Default True.

        Returns:
            PreparedData containing scaled, adapter-transformed data ready
            for training.

        Raises:
            ValueError: If model_name is unknown or required data is missing.
        """
        model_key = model_name.lower()

        # 1. Determine adapter type from model name
        adapter_type = MODEL_ADAPTER_MAP.get(model_key)
        if adapter_type is None:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {list(MODEL_ADAPTER_MAP.keys())}"
            )

        data_rank = MODEL_DATA_RANKS.get(model_key, 2)

        logger.info(
            f"Preparing data for {model_name}: "
            f"adapter={adapter_type}, rank={data_rank}D"
        )

        # 2. Split data chronologically with purge/embargo
        train_df, val_df, test_df = self._split_with_purge_embargo(df)

        logger.debug(
            f"Split sizes: train={len(train_df)}, val={len(val_df)}, "
            f"test={len(test_df) if test_df is not None else 0}"
        )

        # 3. Get adapter and transform each split
        # Build adapter kwargs based on adapter type
        adapter_kwargs = self._build_adapter_kwargs(
            adapter_type=adapter_type,
            feature_columns=feature_columns,
            label_column=label_column,
            weight_column=weight_column,
        )

        adapter = get_adapter(adapter_id=adapter_type, **adapter_kwargs)

        # Transform each split through the adapter
        if adapter_type == "multi_stream" and additional_dfs:
            # Multi-stream needs additional timeframe DataFrames
            train_additional = self._split_additional_dfs(additional_dfs, "train")
            val_additional = self._split_additional_dfs(additional_dfs, "val")
            test_additional = self._split_additional_dfs(additional_dfs, "test") if test_df is not None else None

            train_result = adapter.transform(train_df, additional_dfs=train_additional)
            val_result = adapter.transform(val_df, additional_dfs=val_additional)
            test_result = adapter.transform(test_df, additional_dfs=test_additional) if test_df is not None else None
        else:
            train_result = adapter.transform(train_df)
            val_result = adapter.transform(val_df)
            test_result = adapter.transform(test_df) if test_df is not None else None

        # 4. Scale data (fit on train only)
        if apply_scaling:
            scaler = AdapterScaler(self.scaler_config)
            X_train_scaled = scaler.fit_transform(train_result.X)
            X_val_scaled = scaler.transform(val_result.X)
            X_test_scaled = scaler.transform(test_result.X) if test_result else None
            self._scalers[model_name] = scaler
        else:
            X_train_scaled = train_result.X
            X_val_scaled = val_result.X
            X_test_scaled = test_result.X if test_result else None
            scaler = None

        # 5. Build and return PreparedData
        prepared = PreparedData(
            X_train=X_train_scaled,
            y_train=train_result.y,
            X_val=X_val_scaled,
            y_val=val_result.y,
            X_test=X_test_scaled,
            y_test=test_result.y if test_result else None,
            train_weights=train_result.weights,
            val_weights=val_result.weights,
            test_weights=test_result.weights if test_result else None,
            model_name=model_name,
            adapter_type=adapter_type,
            data_rank=data_rank,
            feature_names=train_result.feature_columns,
            train_indices=train_result.original_indices,
            val_indices=val_result.original_indices,
            test_indices=test_result.original_indices if test_result else None,
            scaler=scaler,
            sequence_length=train_result.sequence_length,
            n_timeframes=train_result.n_timeframes,
            timeframe_names=train_result.timeframe_names,
        )

        # Validate the prepared data
        is_valid, issues = prepared.validate()
        if not is_valid:
            logger.warning(f"PreparedData validation issues: {issues}")

        logger.info(prepared.summary())
        return prepared

    def prepare_heterogeneous(
        self,
        df: pd.DataFrame,
        models: List[str],
        feature_columns: Optional[List[str]] = None,
        label_column: str = "label",
        weight_column: Optional[str] = None,
        additional_dfs: Optional[Dict[str, pd.DataFrame]] = None,
        apply_scaling: bool = True,
    ) -> Dict[str, PreparedData]:
        """
        Prepare data for heterogeneous ensemble (multiple model types).

        Calls prepare() for each model, returning a dictionary of PreparedData
        objects. Each model gets data in its required format.

        Args:
            df: Source DataFrame.
            models: List of model names to prepare data for.
            feature_columns: Optional feature columns (shared across models).
            label_column: Label column name.
            weight_column: Optional weight column name.
            additional_dfs: For multi-stream models.
            apply_scaling: Whether to apply scaling.

        Returns:
            Dictionary mapping model name -> PreparedData.

        Example:
            >>> results = prep.prepare_heterogeneous(
            ...     df,
            ...     models=["xgboost", "lightgbm", "lstm"],
            ... )
            >>> xgb_data = results["xgboost"]  # 2D
            >>> lstm_data = results["lstm"]    # 3D
        """
        results: Dict[str, PreparedData] = {}

        for model_name in models:
            logger.info(f"Preparing data for model: {model_name}")
            results[model_name] = self.prepare(
                df=df,
                model_name=model_name,
                feature_columns=feature_columns,
                label_column=label_column,
                weight_column=weight_column,
                additional_dfs=additional_dfs,
                apply_scaling=apply_scaling,
            )

        return results

    def _split_with_purge_embargo(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Split data chronologically with purge and embargo gaps.

        Creates gaps between train/val and val/test to prevent
        information leakage from overlapping prediction horizons.

        Args:
            df: Source DataFrame (must be chronologically sorted).

        Returns:
            Tuple of (train_df, val_df, test_df).
            test_df may be None if no samples remain after embargo.
        """
        n = len(df)
        purge = self.config.purge_bars
        embargo = self.config.embargo_bars

        # Calculate split indices
        train_end = int(n * self.config.train_ratio)
        val_start = train_end + purge  # Gap after train
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        test_start = val_end + embargo  # Gap after val

        # Extract splits
        train_df = df.iloc[:train_end].copy()

        # Validate val_start is within bounds
        if val_start >= n:
            raise ValueError(
                f"Purge gap ({purge}) too large for dataset size ({n}). "
                f"train_end={train_end}, val_start={val_start}"
            )

        val_df = df.iloc[val_start:val_end].copy()

        # Test set may be empty or None
        if test_start < n:
            test_df = df.iloc[test_start:].copy()
            if len(test_df) == 0:
                test_df = None
        else:
            test_df = None
            logger.warning(
                f"No test samples: embargo ({embargo}) pushes test_start ({test_start}) "
                f"beyond data length ({n})"
            )

        logger.debug(
            f"Split with purge/embargo: "
            f"train=[0:{train_end}], "
            f"val=[{val_start}:{val_end}], "
            f"test=[{test_start}:{n if test_df is not None else 'None'}], "
            f"purge={purge}, embargo={embargo}"
        )

        return train_df, val_df, test_df

    def _split_additional_dfs(
        self,
        additional_dfs: Dict[str, pd.DataFrame],
        split: str,
    ) -> Dict[str, pd.DataFrame]:
        """
        Split additional DataFrames for multi-stream adapter.

        Applies the same chronological split logic to each additional
        timeframe DataFrame.

        Args:
            additional_dfs: Dictionary of timeframe -> DataFrame.
            split: Which split to return ("train", "val", or "test").

        Returns:
            Dictionary with the requested split for each timeframe.
        """
        result: Dict[str, pd.DataFrame] = {}

        for tf, tf_df in additional_dfs.items():
            train_df, val_df, test_df = self._split_with_purge_embargo(tf_df)

            if split == "train":
                result[tf] = train_df
            elif split == "val":
                result[tf] = val_df
            elif split == "test":
                if test_df is not None:
                    result[tf] = test_df
                else:
                    # Return empty DataFrame with same columns
                    result[tf] = tf_df.iloc[:0].copy()
            else:
                raise ValueError(f"Unknown split: {split}")

        return result

    def _build_adapter_kwargs(
        self,
        adapter_type: str,
        feature_columns: Optional[List[str]],
        label_column: str,
        weight_column: Optional[str],
    ) -> Dict[str, Any]:
        """
        Build keyword arguments for adapter constructor.

        Args:
            adapter_type: Type of adapter ("tabular", "sequence", "multi_stream").
            feature_columns: Optional feature columns.
            label_column: Label column name.
            weight_column: Optional weight column name.

        Returns:
            Dictionary of kwargs for adapter constructor.
        """
        kwargs: Dict[str, Any] = {
            "feature_columns": feature_columns,
            "label_column": label_column,
            "weight_column": weight_column,
        }

        # Add sequence-specific kwargs
        if adapter_type in ("sequence", "multi_stream"):
            kwargs["sequence_length"] = self.config.sequence_length

        # Add multi-stream specific kwargs
        if adapter_type == "multi_stream":
            kwargs["timeframes"] = self.config.mtf_timeframes

        return kwargs

    def get_scaler(self, model_name: str) -> Optional[AdapterScaler]:
        """
        Get the fitted scaler for a model.

        Args:
            model_name: Name of the model.

        Returns:
            Fitted AdapterScaler or None if not found/not applied.
        """
        return self._scalers.get(model_name)

    def get_all_scalers(self) -> Dict[str, AdapterScaler]:
        """
        Get all fitted scalers.

        Returns:
            Dictionary mapping model name -> AdapterScaler.
        """
        return dict(self._scalers)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def prepare_for_model(
    df: pd.DataFrame,
    model_name: str,
    config: PipelineConfig,
    feature_columns: Optional[List[str]] = None,
    label_column: str = "label",
    **kwargs: Any,
) -> PreparedData:
    """
    Convenience function to prepare data for a single model.

    Args:
        df: Source DataFrame.
        model_name: Model name.
        config: Pipeline configuration.
        feature_columns: Optional feature columns.
        label_column: Label column name.
        **kwargs: Additional arguments passed to prepare().

    Returns:
        PreparedData for the model.

    Example:
        >>> from src.core.config import PipelineConfig
        >>> config = PipelineConfig(...)
        >>> data = prepare_for_model(df, "xgboost", config)
    """
    prep = UnifiedDataPreparation(config)
    return prep.prepare(
        df=df,
        model_name=model_name,
        feature_columns=feature_columns,
        label_column=label_column,
        **kwargs,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "PreparedData",
    "UnifiedDataPreparation",
    "prepare_for_model",
]
