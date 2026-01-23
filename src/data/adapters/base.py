"""
Base adapter class and result dataclass.

Phase 2 SNwH Implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.core.contracts import DataContract, ModelContract

from src.core.contracts import DataRank


@dataclass
class AdapterResult:
    """
    Result from adapter transformation.

    Contains the transformed data arrays plus metadata
    for validation and debugging.
    """

    # Data arrays
    X: np.ndarray
    y: np.ndarray
    weights: np.ndarray | None = None

    # Shape info
    n_samples: int = 0
    n_features: int = 0
    data_rank: DataRank = DataRank.TABULAR_2D

    # For sequences
    sequence_length: int | None = None
    original_indices: np.ndarray | None = None  # Maps back to source DataFrame

    # For multi-stream
    n_timeframes: int | None = None
    timeframe_names: list[str] = field(default_factory=list)

    # Metadata
    feature_columns: list[str] = field(default_factory=list)
    data_contract: DataContract | None = None
    adapter_name: str = ""

    def __post_init__(self) -> None:
        """Compute derived fields."""
        if self.n_samples == 0:
            self.n_samples = self.X.shape[0]
        if self.n_features == 0:
            if self.X.ndim == 2:
                self.n_features = self.X.shape[1]
            elif self.X.ndim == 3:
                self.n_features = self.X.shape[2]
            elif self.X.ndim == 4:
                self.n_features = self.X.shape[3]

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate adapter result.

        Returns:
            (is_valid, list_of_issues)
        """
        issues: list[str] = []

        # Check X shape matches declared rank
        if self.X.ndim != self.data_rank.value:
            issues.append(f"X rank mismatch: expected {self.data_rank.value}D, got {self.X.ndim}D")

        # Check y length
        if self.y.shape[0] != self.X.shape[0]:
            issues.append(
                f"y length mismatch: X has {self.X.shape[0]} samples, " f"y has {self.y.shape[0]}"
            )

        # Check weights if present
        if self.weights is not None and self.weights.shape[0] != self.X.shape[0]:
            issues.append(
                f"weights length mismatch: X has {self.X.shape[0]} samples, "
                f"weights has {self.weights.shape[0]}"
            )

        # Check for NaN/Inf
        if np.isnan(self.X).any():
            n_nan = int(np.isnan(self.X).sum())
            issues.append(f"X contains {n_nan} NaN values")
        if np.isinf(self.X).any():
            n_inf = int(np.isinf(self.X).sum())
            issues.append(f"X contains {n_inf} Inf values")

        return len(issues) == 0, issues

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to dictionary (excludes arrays)."""
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "data_rank": self.data_rank.value,
            "sequence_length": self.sequence_length,
            "n_timeframes": self.n_timeframes,
            "timeframe_names": self.timeframe_names,
            "feature_columns": self.feature_columns,
            "adapter_name": self.adapter_name,
            "X_shape": list(self.X.shape),
            "y_shape": list(self.y.shape),
            "has_weights": self.weights is not None,
        }


class BaseAdapter(ABC):
    """
    Base class for data adapters.

    Adapters transform canonical DataFrames into model-specific
    numpy array formats (2D, 3D, or 4D).
    """

    # Adapter identity
    adapter_id: str = "base"
    output_rank: DataRank = DataRank.TABULAR_2D

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        label_column: str = "label_h20",
        weight_column: str | None = "sample_weight_h20",
    ):
        """
        Initialize adapter.

        Args:
            feature_columns: Feature columns to use (None = auto-detect)
            label_column: Label column name
            weight_column: Weight column name (None = uniform weights)
        """
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.weight_column = weight_column

    @abstractmethod
    def transform(
        self,
        df: pd.DataFrame,
        model_contract: ModelContract | None = None,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> AdapterResult:
        """
        Transform DataFrame to model-specific format.

        Args:
            df: Source DataFrame with features, labels, weights
            model_contract: Optional contract for validation
            additional_dfs: Optional additional DataFrames for multi-stream adapters

        Returns:
            AdapterResult with transformed arrays
        """
        pass

    def validate_input(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """
        Validate input DataFrame.

        Args:
            df: DataFrame to validate

        Returns:
            (is_valid, list_of_issues)
        """
        issues: list[str] = []

        # Check label column
        if self.label_column not in df.columns:
            issues.append(f"Missing label column: {self.label_column}")

        # Check feature columns if specified
        if self.feature_columns:
            missing = set(self.feature_columns) - set(df.columns)
            if missing:
                issues.append(f"Missing feature columns: {sorted(missing)[:10]}")

        return len(issues) == 0, issues

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Get feature columns (explicit or auto-detected)."""
        if self.feature_columns:
            return self.feature_columns

        # Auto-detect: exclude metadata and labels
        exclude_prefixes = (
            "label_",
            "sample_weight_",
            "regime_",
            "timestamp",
            "datetime",
            "symbol",
            "timeframe",
            "split",
        )
        exclude_exact = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "bar_index",
            "session_id",
        }

        def is_feature_col(col: str) -> bool:
            col_lower = col.lower()
            if col_lower in exclude_exact:
                return False
            for prefix in exclude_prefixes:
                if col_lower.startswith(prefix):
                    return False
            return True

        return [col for col in df.columns if is_feature_col(col)]

    def _get_weights(self, df: pd.DataFrame) -> np.ndarray | None:
        """Get sample weights from DataFrame."""
        if self.weight_column and self.weight_column in df.columns:
            return np.asarray(df[self.weight_column].values.astype(np.float32))
        return None


__all__ = [
    "AdapterResult",
    "BaseAdapter",
]
