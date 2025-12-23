"""
Base classes and types for labeling strategies.

This module defines the abstract base class for all labeling strategies
and the enum for labeling types. All labeling implementations must inherit
from LabelingStrategy and implement the required abstract methods.

Engineering Rules Applied:
- Clear contracts with abstract methods
- Input validation at boundaries
- Fail fast with explicit errors
- Type hints throughout
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class LabelingType(Enum):
    """Enumeration of available labeling strategies."""

    TRIPLE_BARRIER = 'triple_barrier'
    ADAPTIVE_TRIPLE_BARRIER = 'adaptive_triple_barrier'
    DIRECTIONAL = 'directional'
    THRESHOLD = 'threshold'
    REGRESSION = 'regression'
    META = 'meta'


@dataclass
class LabelingResult:
    """
    Container for labeling results with quality metrics.

    Attributes:
        labels: Array of labels (-1, 0, 1 for classification; float for regression)
        horizon: The horizon identifier used for labeling
        metadata: Dictionary containing additional metrics (e.g., bars_to_hit, mae, mfe)
        quality_metrics: Dictionary containing quality assessment metrics
    """
    labels: np.ndarray
    horizon: int
    metadata: dict[str, np.ndarray] = field(default_factory=dict)
    quality_metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the labeling result after initialization."""
        if not isinstance(self.labels, np.ndarray):
            raise TypeError(f"labels must be np.ndarray, got {type(self.labels).__name__}")
        if not isinstance(self.horizon, int) or self.horizon <= 0:
            raise ValueError(f"horizon must be a positive integer, got {self.horizon}")


class LabelingStrategy(ABC):
    """
    Abstract base class for all labeling strategies.

    Each labeling strategy must implement:
    - compute_labels: Generate labels for the given data and horizon
    - validate_inputs: Validate that required columns exist in the DataFrame

    Subclasses may optionally override:
    - get_quality_metrics: Compute quality assessment metrics for labels
    """

    @property
    @abstractmethod
    def labeling_type(self) -> LabelingType:
        """Return the type of this labeling strategy."""
        pass

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """Return list of required DataFrame columns for this strategy."""
        pass

    @abstractmethod
    def compute_labels(
        self,
        df: pd.DataFrame,
        horizon: int,
        **kwargs: Any
    ) -> LabelingResult:
        """
        Compute labels for the given horizon.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLCV data and any required features
        horizon : int
            Horizon identifier (e.g., 1, 5, 20 bars)
        **kwargs : Any
            Strategy-specific parameters

        Returns
        -------
        LabelingResult
            Container with labels, metadata, and quality metrics

        Raises
        ------
        ValueError
            If DataFrame is empty or parameters are invalid
        KeyError
            If required columns are missing
        """
        pass

    def validate_inputs(self, df: pd.DataFrame) -> None:
        """
        Validate that required columns exist in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate

        Raises
        ------
        ValueError
            If DataFrame is empty
        KeyError
            If required columns are missing
        """
        if df.empty:
            raise ValueError("DataFrame is empty - cannot compute labels")

        missing = set(self.required_columns) - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")

    def get_quality_metrics(self, result: LabelingResult) -> dict[str, float]:
        """
        Compute quality metrics for the labeling result.

        Default implementation computes basic distribution metrics.
        Subclasses may override for strategy-specific metrics.

        Parameters
        ----------
        result : LabelingResult
            The labeling result to analyze

        Returns
        -------
        dict[str, float]
            Dictionary of quality metric names to values
        """
        labels = result.labels

        # Filter out invalid labels (e.g., -99 sentinel)
        valid_mask = labels != -99
        valid_labels = labels[valid_mask]

        if len(valid_labels) == 0:
            return {
                'total_samples': 0,
                'valid_samples': 0,
                'invalid_samples': len(labels),
            }

        metrics: dict[str, float] = {
            'total_samples': float(len(labels)),
            'valid_samples': float(len(valid_labels)),
            'invalid_samples': float((~valid_mask).sum()),
        }

        # Classification-specific metrics (for labels in {-1, 0, 1})
        unique_labels = set(valid_labels)
        if unique_labels.issubset({-1, 0, 1}):
            for label_val in [-1, 0, 1]:
                count = (valid_labels == label_val).sum()
                pct = count / len(valid_labels) * 100
                label_name = {-1: 'short', 0: 'neutral', 1: 'long'}[label_val]
                metrics[f'{label_name}_count'] = float(count)
                metrics[f'{label_name}_pct'] = pct

            # Label imbalance ratio
            long_count = metrics.get('long_count', 0)
            short_count = metrics.get('short_count', 0)
            if short_count > 0:
                metrics['long_short_ratio'] = long_count / short_count
            else:
                metrics['long_short_ratio'] = float('inf') if long_count > 0 else 0.0
        else:
            # Regression targets - compute basic statistics
            metrics['mean'] = float(np.mean(valid_labels))
            metrics['std'] = float(np.std(valid_labels))
            metrics['min'] = float(np.min(valid_labels))
            metrics['max'] = float(np.max(valid_labels))

        return metrics

    def add_labels_to_dataframe(
        self,
        df: pd.DataFrame,
        result: LabelingResult,
        prefix: str = 'label'
    ) -> pd.DataFrame:
        """
        Add labeling results to the DataFrame as new columns.

        Parameters
        ----------
        df : pd.DataFrame
            Original DataFrame
        result : LabelingResult
            Labeling result to add
        prefix : str
            Column name prefix (default: 'label')

        Returns
        -------
        pd.DataFrame
            DataFrame with new label columns added
        """
        horizon = result.horizon
        df = df.copy()

        # Add main label column
        df[f'{prefix}_h{horizon}'] = result.labels

        # Add metadata columns
        for key, values in result.metadata.items():
            df[f'{key}_h{horizon}'] = values

        return df
