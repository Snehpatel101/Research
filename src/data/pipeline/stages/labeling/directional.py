"""
Directional Return Labeling Strategy.

Labels based on the sign of the return over the next N bars.
Simple but effective baseline for directional prediction.

Label mapping:
- +1: Positive return (price goes up)
- -1: Negative return (price goes down)
- 0: No significant change (within threshold)
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from .base import LabelingResult, LabelingStrategy, LabelingType

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DirectionalLabeler(LabelingStrategy):
    """
    Directional return labeling strategy.

    Labels each bar based on the sign of the return over the next N bars.
    Optionally applies a threshold to filter out noise.

    Parameters
    ----------
    threshold : float
        Minimum absolute return to be considered directional (default: 0.0)
        Returns within [-threshold, +threshold] are labeled as neutral (0)
    use_log_returns : bool
        Whether to use log returns instead of simple returns (default: False)
    """

    def __init__(self, threshold: float = 0.0, use_log_returns: bool = False):
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")

        self._threshold = threshold
        self._use_log_returns = use_log_returns

    @property
    def labeling_type(self) -> LabelingType:
        """Return the type of this labeling strategy."""
        return LabelingType.DIRECTIONAL

    @property
    def required_columns(self) -> list[str]:
        """Return list of required DataFrame columns."""
        return ["close"]

    def compute_labels(
        self, df: pd.DataFrame, horizon: int, threshold: float | None = None, **kwargs: Any
    ) -> LabelingResult:
        """
        Compute directional labels for the given horizon.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with close prices
        horizon : int
            Number of bars to look ahead for return calculation
        threshold : float, optional
            Override for minimum return threshold
        **kwargs : Any
            Additional parameters (ignored)

        Returns
        -------
        LabelingResult
            Container with labels and quality metrics
        """
        # Validate inputs
        self.validate_inputs(df)

        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(f"horizon must be a positive integer, got {horizon}")

        threshold = threshold if threshold is not None else self._threshold

        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")

        logger.info(f"Computing directional labels for horizon {horizon}")
        logger.info(f"  threshold={threshold:.6f}, log_returns={self._use_log_returns}")

        close = df["close"].values
        n = len(close)

        # Compute forward returns
        if self._use_log_returns:
            # Log returns: log(P_t+h / P_t)
            returns = np.zeros(n, dtype=np.float64)
            returns[:-horizon] = np.log(close[horizon:] / close[:-horizon])
        else:
            # Simple returns: (P_t+h - P_t) / P_t
            returns = np.zeros(n, dtype=np.float64)
            returns[:-horizon] = (close[horizon:] - close[:-horizon]) / close[:-horizon]

        # Compute labels based on return sign and threshold
        labels = np.zeros(n, dtype=np.int8)
        labels[returns > threshold] = 1
        labels[returns < -threshold] = -1
        # Returns within threshold remain 0 (neutral)

        # Mark last horizon bars as invalid (no future data available)
        labels[-horizon:] = -99

        # Store returns as metadata
        result = LabelingResult(
            labels=labels, horizon=horizon, metadata={"forward_return": returns.astype(np.float32)}
        )

        # Compute quality metrics
        result.quality_metrics = self.get_quality_metrics(result)

        # Log statistics
        self._log_label_statistics(result, horizon, threshold)

        return result

    def _log_label_statistics(self, result: LabelingResult, horizon: int, threshold: float) -> None:
        """Log label distribution statistics."""
        labels = result.labels
        valid_mask = labels != -99
        valid_labels = labels[valid_mask]

        if len(valid_labels) == 0:
            logger.warning(f"No valid labels for horizon {horizon}")
            return

        label_counts = pd.Series(valid_labels).value_counts().sort_index()
        total = len(valid_labels)

        logger.info(f"Label distribution for horizon {horizon} (threshold={threshold:.4f}):")
        for label_val in [-1, 0, 1]:
            count = label_counts.get(label_val, 0)
            pct = count / total * 100
            label_name = {-1: "Down", 0: "Neutral", 1: "Up"}[label_val]
            logger.info(f"  {label_name:10s}: {count:6d} ({pct:5.1f}%)")

        # Log return statistics
        returns = result.metadata.get("forward_return", np.array([]))
        if len(returns) > 0:
            valid_returns = returns[valid_mask]
            if len(valid_returns) > 0:
                mean_ret = np.mean(valid_returns) * 100
                std_ret = np.std(valid_returns) * 100
                logger.info(f"  Avg return: {mean_ret:.4f}%, Std: {std_ret:.4f}%")

    def get_quality_metrics(self, result: LabelingResult) -> dict[str, float]:
        """
        Compute quality metrics for directional labeling.

        Includes return distribution statistics.
        """
        metrics = super().get_quality_metrics(result)

        # Add return-based metrics
        returns = result.metadata.get("forward_return", np.array([]))
        labels = result.labels
        valid_mask = labels != -99

        if len(returns) > 0:
            valid_returns = returns[valid_mask]
            if len(valid_returns) > 0:
                metrics["avg_return"] = float(np.mean(valid_returns))
                metrics["std_return"] = float(np.std(valid_returns))
                metrics["skew_return"] = float(
                    ((valid_returns - np.mean(valid_returns)) ** 3).mean()
                    / (np.std(valid_returns) ** 3 + 1e-10)
                )

                # Return by label
                for label_val in [-1, 0, 1]:
                    label_mask = (labels == label_val) & valid_mask
                    label_returns = returns[label_mask]
                    if len(label_returns) > 0:
                        label_name = {-1: "down", 0: "neutral", 1: "up"}[label_val]
                        metrics[f"avg_return_{label_name}"] = float(np.mean(label_returns))

        return metrics
