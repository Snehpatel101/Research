"""
Regression Target Labeling Strategy.

Generates continuous return values as targets for regression models.
Instead of classifying into discrete labels, this provides the actual
return value for direct return prediction.

Label values:
- Continuous float: The actual forward return over the horizon
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from .base import LabelingResult, LabelingStrategy, LabelingType

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class RegressionLabeler(LabelingStrategy):
    """
    Regression target labeling strategy.

    Computes the actual forward return as a continuous target for regression
    models. This is useful when you want to predict the magnitude of returns,
    not just the direction.

    Parameters
    ----------
    use_log_returns : bool
        Whether to use log returns instead of simple returns (default: False)
    winsorize_pct : float
        Percentile for winsorization to handle outliers (default: 0.0 = no winsorization)
        If set to 0.01, values below 1st and above 99th percentile are clipped.
    scale_factor : float
        Scaling factor for returns (default: 1.0)
        Use 100.0 to convert to percentage returns.
    """

    def __init__(
        self,
        use_log_returns: bool = False,
        winsorize_pct: float = 0.0,
        scale_factor: float = 1.0
    ):
        if winsorize_pct < 0 or winsorize_pct >= 0.5:
            raise ValueError(
                f"winsorize_pct must be in [0, 0.5), got {winsorize_pct}"
            )
        if scale_factor <= 0:
            raise ValueError(f"scale_factor must be positive, got {scale_factor}")

        self._use_log_returns = use_log_returns
        self._winsorize_pct = winsorize_pct
        self._scale_factor = scale_factor

    @property
    def labeling_type(self) -> LabelingType:
        """Return the type of this labeling strategy."""
        return LabelingType.REGRESSION

    @property
    def required_columns(self) -> list[str]:
        """Return list of required DataFrame columns."""
        return ['close']

    def compute_labels(
        self,
        df: pd.DataFrame,
        horizon: int,
        use_log_returns: bool | None = None,
        winsorize_pct: float | None = None,
        scale_factor: float | None = None,
        **kwargs: Any
    ) -> LabelingResult:
        """
        Compute regression targets for the given horizon.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with close prices
        horizon : int
            Number of bars to look ahead for return calculation
        use_log_returns : bool, optional
            Override for log returns setting
        winsorize_pct : float, optional
            Override for winsorization percentile
        scale_factor : float, optional
            Override for scaling factor
        **kwargs : Any
            Additional parameters (ignored)

        Returns
        -------
        LabelingResult
            Container with continuous labels and quality metrics
        """
        # Validate inputs
        self.validate_inputs(df)

        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(f"horizon must be a positive integer, got {horizon}")

        # Resolve parameters
        use_log_returns = use_log_returns if use_log_returns is not None else self._use_log_returns
        winsorize_pct = winsorize_pct if winsorize_pct is not None else self._winsorize_pct
        scale_factor = scale_factor if scale_factor is not None else self._scale_factor

        logger.info(f"Computing regression targets for horizon {horizon}")
        logger.info(
            f"  log_returns={use_log_returns}, winsorize={winsorize_pct}, scale={scale_factor}"
        )

        close = df['close'].values
        n = len(close)

        # Compute forward returns
        returns = np.full(n, np.nan, dtype=np.float64)

        if use_log_returns:
            # Log returns: log(P_t+h / P_t)
            returns[:-horizon] = np.log(close[horizon:] / close[:-horizon])
        else:
            # Simple returns: (P_t+h - P_t) / P_t
            returns[:-horizon] = (close[horizon:] - close[:-horizon]) / close[:-horizon]

        # Apply winsorization if requested
        if winsorize_pct > 0:
            valid_returns = returns[:-horizon]
            lower_bound = np.percentile(valid_returns, winsorize_pct * 100)
            upper_bound = np.percentile(valid_returns, (1 - winsorize_pct) * 100)
            returns = np.clip(returns, lower_bound, upper_bound)
            logger.info(f"  Winsorized to [{lower_bound:.6f}, {upper_bound:.6f}]")

        # Apply scaling
        returns = returns * scale_factor

        # Create labels array (use float32 for efficiency)
        labels = returns.astype(np.float32)

        # Mark last horizon bars as invalid using NaN (regression uses float)
        labels[-horizon:] = np.nan

        # For LabelingResult compatibility, we need to handle NaN differently
        # Use a sentinel-based approach similar to classification
        # Store raw returns in metadata, use -99 as sentinel in int labels for compatibility
        int_labels = np.zeros(n, dtype=np.int8)
        int_labels[-horizon:] = -99  # Mark invalid

        result = LabelingResult(
            labels=int_labels,  # For compatibility with base class validation
            horizon=horizon,
            metadata={
                'regression_target': labels,
                'raw_return': (returns / scale_factor).astype(np.float32)
            }
        )

        # Compute quality metrics
        result.quality_metrics = self._compute_regression_metrics(labels, horizon)

        # Log statistics
        self._log_label_statistics(labels, horizon)

        return result

    def _compute_regression_metrics(
        self,
        labels: np.ndarray,
        horizon: int
    ) -> dict[str, float]:
        """Compute regression-specific quality metrics."""
        # Filter out NaN values
        valid_labels = labels[~np.isnan(labels)]

        if len(valid_labels) == 0:
            return {
                'total_samples': float(len(labels)),
                'valid_samples': 0.0,
                'invalid_samples': float(len(labels)),
            }

        metrics: dict[str, float] = {
            'total_samples': float(len(labels)),
            'valid_samples': float(len(valid_labels)),
            'invalid_samples': float(np.isnan(labels).sum()),
            'mean': float(np.mean(valid_labels)),
            'std': float(np.std(valid_labels)),
            'min': float(np.min(valid_labels)),
            'max': float(np.max(valid_labels)),
            'median': float(np.median(valid_labels)),
        }

        # Percentiles
        for pct in [5, 25, 75, 95]:
            metrics[f'percentile_{pct}'] = float(np.percentile(valid_labels, pct))

        # Skewness and kurtosis
        if metrics['std'] > 0:
            centered = valid_labels - metrics['mean']
            metrics['skewness'] = float(np.mean(centered ** 3) / (metrics['std'] ** 3))
            metrics['kurtosis'] = float(np.mean(centered ** 4) / (metrics['std'] ** 4) - 3)
        else:
            metrics['skewness'] = 0.0
            metrics['kurtosis'] = 0.0

        # Positive/negative split
        positive_count = (valid_labels > 0).sum()
        negative_count = (valid_labels < 0).sum()
        metrics['positive_pct'] = float(positive_count / len(valid_labels) * 100)
        metrics['negative_pct'] = float(negative_count / len(valid_labels) * 100)

        return metrics

    def _log_label_statistics(self, labels: np.ndarray, horizon: int) -> None:
        """Log regression target statistics."""
        valid_labels = labels[~np.isnan(labels)]

        if len(valid_labels) == 0:
            logger.warning(f"No valid labels for horizon {horizon}")
            return

        logger.info(f"Regression target statistics for horizon {horizon}:")
        logger.info(f"  Valid samples: {len(valid_labels):,}")
        logger.info(f"  Mean: {np.mean(valid_labels):.6f}")
        logger.info(f"  Std:  {np.std(valid_labels):.6f}")
        logger.info(f"  Min:  {np.min(valid_labels):.6f}")
        logger.info(f"  Max:  {np.max(valid_labels):.6f}")

        positive_pct = (valid_labels > 0).sum() / len(valid_labels) * 100
        logger.info(f"  Positive: {positive_pct:.1f}%, Negative: {100-positive_pct:.1f}%")

    def add_labels_to_dataframe(
        self,
        df: pd.DataFrame,
        result: LabelingResult,
        prefix: str = 'target'
    ) -> pd.DataFrame:
        """
        Add regression targets to the DataFrame as new columns.

        Overrides base method to add regression-specific columns.
        """
        horizon = result.horizon
        df = df.copy()

        # Add regression target column
        regression_target = result.metadata.get('regression_target', np.array([]))
        if len(regression_target) > 0:
            df[f'{prefix}_h{horizon}'] = regression_target

        # Add raw return if available
        raw_return = result.metadata.get('raw_return', np.array([]))
        if len(raw_return) > 0:
            df[f'return_h{horizon}'] = raw_return

        return df
