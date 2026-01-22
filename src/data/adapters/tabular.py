"""
Tabular Adapter - Converts DataFrame to 2D arrays for tabular models.

This adapter handles boosting and classical models:
- XGBoost, LightGBM, CatBoost (boosting family)
- RandomForest, Logistic, SVM (classical family)

Output shape: (n_samples, n_features)

Phase 2 SNwH Implementation.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .base import AdapterResult, BaseAdapter
from .registry import AdapterRegistry
from src.core.contracts import DataContract, DataRank

if TYPE_CHECKING:
    from src.core.contracts import ModelContract

logger = logging.getLogger(__name__)


@AdapterRegistry.register("tabular")
class TabularAdapter(BaseAdapter):
    """
    Adapter for tabular (2D) models.

    Converts a canonical DataFrame with features, labels, and weights
    into 2D numpy arrays suitable for boosting and classical models.

    Attributes:
        adapter_id: Unique identifier "tabular"
        output_rank: DataRank.TABULAR_2D (2D output)

    Example:
        >>> adapter = TabularAdapter(label_column="label_h20")
        >>> result = adapter.transform(df)
        >>> print(result.X.shape)  # (n_samples, n_features)
        >>> print(result.y.shape)  # (n_samples,)
    """

    adapter_id: str = "tabular"
    output_rank: DataRank = DataRank.TABULAR_2D

    def transform(
        self,
        df: pd.DataFrame,
        model_contract: "ModelContract | None" = None,
    ) -> AdapterResult:
        """
        Transform DataFrame to 2D arrays for tabular models.

        Args:
            df: Source DataFrame with features, labels, and optional weights.
                Must contain the label column and either explicit feature
                columns or columns that can be auto-detected as features.
            model_contract: Optional ModelContract for validation of feature
                count bounds (min_features, max_features).

        Returns:
            AdapterResult containing:
                - X: float32 array of shape (n_samples, n_features)
                - y: int64 array of shape (n_samples,)
                - weights: Optional float32 array of shape (n_samples,)
                - data_contract: DataContract with metadata
                - feature_columns: List of feature column names

        Raises:
            ValueError: If input validation fails (missing label column,
                missing feature columns, or feature count out of bounds).
        """
        # Validate input DataFrame
        is_valid, issues = self.validate_input(df)
        if not is_valid:
            raise ValueError(f"Input validation failed: {issues}")

        # Get feature columns (explicit or auto-detected)
        feature_cols = self._get_feature_columns(df)
        n_features = len(feature_cols)

        # Validate feature count against contract bounds
        if model_contract is not None:
            if n_features < model_contract.min_features:
                raise ValueError(
                    f"Feature count {n_features} below minimum "
                    f"{model_contract.min_features} required by {model_contract.model_name}"
                )
            if n_features > model_contract.max_features:
                raise ValueError(
                    f"Feature count {n_features} exceeds maximum "
                    f"{model_contract.max_features} allowed by {model_contract.model_name}"
                )

        # Extract X as float32 - shape (n_samples, n_features)
        X = df[feature_cols].values.astype(np.float32)

        # Extract y as int64
        y = df[self.label_column].values.astype(np.int64)

        # Get sample weights if available
        weights = self._get_weights(df)

        # Create DataContract with metadata extracted from DataFrame
        data_contract = self._create_data_contract(
            df=df,
            X=X,
            feature_cols=feature_cols,
        )

        # Build the adapter result
        result = AdapterResult(
            X=X,
            y=y,
            weights=weights,
            n_samples=X.shape[0],
            n_features=n_features,
            data_rank=DataRank.TABULAR_2D,
            feature_columns=feature_cols,
            data_contract=data_contract,
            adapter_name=self.adapter_id,
        )

        # Validate the result
        is_valid, issues = result.validate()
        if not is_valid:
            raise ValueError(f"Adapter result validation failed: {issues}")

        logger.debug(
            f"TabularAdapter transformed {X.shape[0]} samples, "
            f"{n_features} features -> {X.shape}"
        )

        return result

    def _create_data_contract(
        self,
        df: pd.DataFrame,
        X: np.ndarray,
        feature_cols: list[str],
    ) -> DataContract:
        """
        Create DataContract with metadata extracted from DataFrame.

        Args:
            df: Source DataFrame
            X: Transformed feature array
            feature_cols: List of feature column names

        Returns:
            DataContract with appropriate metadata
        """
        # Extract symbol from DataFrame if available
        symbol = self._get_metadata_value(df, "symbol", default="unknown")

        # Extract timeframe from DataFrame if available
        timeframe = self._get_metadata_value(df, "timeframe", default="5min")

        # Parse horizon from label column name (e.g., "label_h20" -> 20)
        horizon = self._parse_horizon_from_label_column(self.label_column)

        # Determine split if available
        split = self._get_metadata_value(df, "split", default="unknown")

        return DataContract(
            symbol=symbol,
            timeframe=timeframe,
            horizon=horizon,
            split=split,
            n_samples=X.shape[0],
            n_features=X.shape[1],
            data_rank=DataRank.TABULAR_2D,
            feature_columns=feature_cols,
            label_column=self.label_column,
            weight_column=self.weight_column or "",
        )

    def _get_metadata_value(
        self,
        df: pd.DataFrame,
        column: str,
        default: str,
    ) -> str:
        """
        Extract a single metadata value from DataFrame column.

        If the column exists and has a single unique non-null value,
        return that value. Otherwise return the default.

        Args:
            df: DataFrame to extract from
            column: Column name to look for
            default: Default value if column missing or ambiguous

        Returns:
            Extracted metadata value or default
        """
        if column not in df.columns:
            return default

        unique_values = df[column].dropna().unique()
        if len(unique_values) == 1:
            return str(unique_values[0])
        elif len(unique_values) == 0:
            return default
        else:
            # Multiple values - log warning and use first
            logger.warning(
                f"Multiple values for metadata column '{column}': {unique_values[:5]}. "
                f"Using first value."
            )
            return str(unique_values[0])

    def _parse_horizon_from_label_column(self, label_column: str) -> int:
        """
        Parse horizon from label column name.

        Examples:
            "label_h20" -> 20
            "label_h5" -> 5
            "label_h10" -> 10
            "label" -> 20 (default)

        Args:
            label_column: Label column name

        Returns:
            Extracted horizon or default of 20
        """
        # Pattern: label_h{number}
        match = re.search(r"label_h(\d+)", label_column)
        if match:
            return int(match.group(1))

        # Fallback: check for just a number at the end
        match = re.search(r"_(\d+)$", label_column)
        if match:
            return int(match.group(1))

        # Default horizon
        return 20


__all__ = [
    "TabularAdapter",
]
