"""
Walk-Forward Evaluator for Time Series.

Implements walk-forward analysis where the training window expands (or rolls)
forward through time, providing a realistic simulation of model deployment.

Walk-forward is more realistic than k-fold for trading:
- Models only see past data (no future leakage)
- Tests on truly out-of-sample future periods
- Captures regime changes and model degradation

Reference: Pardo (2008) "The Evaluation and Optimization of Trading Strategies"
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class WalkForwardConfig:
    """
    Configuration for walk-forward evaluation.

    Attributes:
        n_windows: Number of walk-forward windows (test periods)
        window_type: "expanding" (growing train) or "rolling" (fixed train size)
        min_train_pct: Minimum training data percentage (for first window)
        test_pct: Percentage of data per test window
        embargo_bars: Bars to skip after each test period (serial correlation)
        gap_bars: Bars to skip between train and test (label leakage)

    Example:
        >>> config = WalkForwardConfig(n_windows=5, window_type="expanding")
        >>> wf = WalkForwardEvaluator(config)
    """
    n_windows: int = 5
    window_type: str = "expanding"  # "expanding" or "rolling"
    min_train_pct: float = 0.4      # First window uses at least 40% for training
    test_pct: float = 0.1           # Each test window is ~10% of data
    embargo_bars: int = 0           # Post-test embargo (for serial correlation)
    gap_bars: int = 0               # Gap between train and test (for label leakage)

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.n_windows < 1:
            raise ValueError(f"n_windows must be >= 1, got {self.n_windows}")

        if self.window_type not in ("expanding", "rolling"):
            raise ValueError(
                f"window_type must be 'expanding' or 'rolling', got '{self.window_type}'"
            )

        if not 0 < self.min_train_pct < 1:
            raise ValueError(
                f"min_train_pct must be in (0, 1), got {self.min_train_pct}"
            )

        if not 0 < self.test_pct < 1:
            raise ValueError(f"test_pct must be in (0, 1), got {self.test_pct}")

        # Validate total doesn't exceed 100%
        if self.min_train_pct + self.n_windows * self.test_pct > 1.0:
            raise ValueError(
                f"min_train_pct ({self.min_train_pct}) + n_windows ({self.n_windows}) * "
                f"test_pct ({self.test_pct}) exceeds 1.0. Reduce n_windows or test_pct."
            )

        if self.embargo_bars < 0:
            raise ValueError(f"embargo_bars must be >= 0, got {self.embargo_bars}")

        if self.gap_bars < 0:
            raise ValueError(f"gap_bars must be >= 0, got {self.gap_bars}")


# =============================================================================
# WALK-FORWARD RESULT
# =============================================================================

@dataclass
class WindowMetrics:
    """Metrics for a single walk-forward window."""
    window: int
    train_size: int
    test_size: int
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    train_start_time: pd.Timestamp | None = None
    train_end_time: pd.Timestamp | None = None
    test_start_time: pd.Timestamp | None = None
    test_end_time: pd.Timestamp | None = None
    accuracy: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    training_time: float = 0.0


@dataclass
class WalkForwardResult:
    """
    Results from walk-forward evaluation.

    Attributes:
        model_name: Name of the evaluated model
        horizon: Label horizon
        window_metrics: Per-window performance metrics
        predictions: DataFrame with all out-of-sample predictions
        config: WalkForwardConfig used
        total_time: Total evaluation time in seconds
    """
    model_name: str
    horizon: int
    window_metrics: list[WindowMetrics]
    predictions: pd.DataFrame
    config: WalkForwardConfig
    total_time: float = 0.0
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_windows(self) -> int:
        """Number of windows evaluated."""
        return len(self.window_metrics)

    @property
    def mean_accuracy(self) -> float:
        """Mean accuracy across all windows."""
        if not self.window_metrics:
            return 0.0
        return float(np.mean([m.accuracy for m in self.window_metrics]))

    @property
    def mean_f1(self) -> float:
        """Mean F1 score across all windows."""
        if not self.window_metrics:
            return 0.0
        return float(np.mean([m.f1 for m in self.window_metrics]))

    @property
    def std_accuracy(self) -> float:
        """Standard deviation of accuracy across windows."""
        if len(self.window_metrics) < 2:
            return 0.0
        return float(np.std([m.accuracy for m in self.window_metrics]))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "n_windows": self.n_windows,
            "mean_accuracy": self.mean_accuracy,
            "std_accuracy": self.std_accuracy,
            "mean_f1": self.mean_f1,
            "total_time": self.total_time,
            "window_type": self.config.window_type,
            "windows": [
                {
                    "window": m.window,
                    "train_size": m.train_size,
                    "test_size": m.test_size,
                    "accuracy": m.accuracy,
                    "f1": m.f1,
                }
                for m in self.window_metrics
            ],
        }


# =============================================================================
# WALK-FORWARD EVALUATOR
# =============================================================================

class WalkForwardEvaluator:
    """
    Walk-forward analysis for time series models.

    Generates expanding or rolling window splits for realistic backtesting.
    Unlike k-fold, walk-forward respects temporal ordering and simulates
    how models would be deployed in production.

    Walk-forward windows:
        Expanding: |---Train---|--Test--|  |----Train-----|--Test--|  |------Train------|--Test--|
        Rolling:   |---Train---|--Test--|    |---Train---|--Test--|      |---Train---|--Test--|

    Example:
        >>> config = WalkForwardConfig(n_windows=5)
        >>> wf = WalkForwardEvaluator(config)
        >>> for train_idx, test_idx in wf.split(X, y):
        ...     model.fit(X.iloc[train_idx], y.iloc[train_idx])
        ...     predictions = model.predict(X.iloc[test_idx])
    """

    def __init__(self, config: WalkForwardConfig) -> None:
        """
        Initialize WalkForwardEvaluator.

        Args:
            config: WalkForwardConfig with window parameters
        """
        self.config = config

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        groups: pd.Series | None = None,
        label_end_times: pd.Series | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for each walk-forward window.

        Args:
            X: Features DataFrame with DatetimeIndex or integer index
            y: Labels (optional, unused but kept for sklearn API compatibility)
            groups: Groups (optional, unused)
            label_end_times: When each label's outcome is known (optional)
                If provided, enables proper purging for overlapping labels

        Yields:
            Tuple of (train_indices, test_indices) for each window

        Raises:
            ValueError: If data is too small for requested configuration
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate window sizes
        test_size = max(1, int(n_samples * self.config.test_pct))
        min_train_size = int(n_samples * self.config.min_train_pct)

        # Validate we have enough data
        total_test = test_size * self.config.n_windows
        if min_train_size + total_test > n_samples:
            raise ValueError(
                f"Insufficient data: need {min_train_size + total_test} samples "
                f"but only have {n_samples}. Reduce n_windows or test_pct."
            )

        # Get timestamps if available (for label-aware purging)
        has_datetime_index = isinstance(X.index, pd.DatetimeIndex)

        # Generate windows
        for window_idx in range(self.config.n_windows):
            # Test window boundaries
            test_start = min_train_size + window_idx * test_size
            test_end = min(test_start + test_size, n_samples)

            # Apply gap between train and test
            train_end = max(0, test_start - self.config.gap_bars)

            # Training window boundaries
            if self.config.window_type == "expanding":
                train_start = 0
            else:  # rolling
                # Rolling window: fixed size, slides forward
                rolling_train_size = min_train_size
                train_start = max(0, train_end - rolling_train_size)

            # Build train mask
            train_mask = np.zeros(n_samples, dtype=bool)
            train_mask[train_start:train_end] = True

            # Apply embargo: exclude samples in embargo zones after ALL previous test periods
            # This breaks autocorrelation between training and nearby tested samples
            # For each previous test period [test_start_i, test_end_i], we exclude
            # the embargo zone [test_end_i, test_end_i + embargo_bars] from training
            embargo_excluded = 0
            if self.config.embargo_bars > 0 and window_idx > 0:
                # Check ALL previous test periods
                for prev_window in range(window_idx):
                    prev_test_start = min_train_size + prev_window * test_size
                    prev_test_end = prev_test_start + test_size
                    embargo_zone_end = min(prev_test_end + self.config.embargo_bars, train_end)

                    # Remove embargo zone from training (samples right after this test period)
                    for i in range(prev_test_end, embargo_zone_end):
                        if train_mask[i]:
                            train_mask[i] = False
                            embargo_excluded += 1

                if embargo_excluded > 0:
                    logger.debug(
                        f"Window {window_idx}: Excluded {embargo_excluded} samples "
                        f"in embargo zones after {window_idx} previous test periods"
                    )

            # Apply label-aware purging if label_end_times provided
            if label_end_times is not None and has_datetime_index:
                test_start_time = X.index[test_start]
                for i in range(train_start, train_end):
                    if train_mask[i] and label_end_times.iloc[i] >= test_start_time:
                        train_mask[i] = False

            train_indices = indices[train_mask]
            test_indices = indices[test_start:test_end]

            if len(train_indices) == 0:
                raise ValueError(
                    f"Window {window_idx}: Empty training set after purging/embargo. "
                    "Increase min_train_pct or reduce gap_bars/embargo_bars."
                )

            yield train_indices, test_indices

    def get_n_splits(
        self,
        X: pd.DataFrame | None = None,
        y: pd.Series | None = None,
        groups: pd.Series | None = None,
    ) -> int:
        """Return number of windows (sklearn API compatibility)."""
        return self.config.n_windows

    def get_window_info(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        label_end_times: pd.Series | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get detailed information about each walk-forward window.

        Args:
            X: Features DataFrame (needed for timestamp info)
            y: Labels (optional)
            label_end_times: Label end times for purging info

        Returns:
            List of dicts with window information including sizes and time ranges
        """
        info = []
        has_datetime_index = isinstance(X.index, pd.DatetimeIndex)

        for window_idx, (train_idx, test_idx) in enumerate(
            self.split(X, y, label_end_times=label_end_times)
        ):
            window_info = {
                "window": window_idx,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "train_start_idx": int(train_idx[0]),
                "train_end_idx": int(train_idx[-1]),
                "test_start_idx": int(test_idx[0]),
                "test_end_idx": int(test_idx[-1]),
                "window_type": self.config.window_type,
                "embargo_bars": self.config.embargo_bars,
                "gap_bars": self.config.gap_bars,
            }

            if has_datetime_index:
                window_info.update({
                    "train_start_time": X.index[train_idx[0]],
                    "train_end_time": X.index[train_idx[-1]],
                    "test_start_time": X.index[test_idx[0]],
                    "test_end_time": X.index[test_idx[-1]],
                })

            info.append(window_info)

        return info

    def validate_coverage(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        label_end_times: pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Validate that walk-forward covers expected samples.

        Args:
            X: Features DataFrame
            y: Labels (optional)
            label_end_times: Label end times (optional)

        Returns:
            Dict with coverage statistics
        """
        n_samples = len(X)
        test_coverage = np.zeros(n_samples, dtype=int)
        train_coverage = np.zeros(n_samples, dtype=int)

        for train_idx, test_idx in self.split(X, y, label_end_times=label_end_times):
            test_coverage[test_idx] += 1
            train_coverage[train_idx] += 1

        # Calculate test range (samples that should be in test)
        test_size = max(1, int(n_samples * self.config.test_pct))
        min_train = int(n_samples * self.config.min_train_pct)
        expected_test_start = min_train
        expected_test_end = min(min_train + test_size * self.config.n_windows, n_samples)

        return {
            "total_samples": n_samples,
            "samples_in_test": int((test_coverage > 0).sum()),
            "test_coverage_fraction": float((test_coverage > 0).sum() / n_samples),
            "expected_test_range": (expected_test_start, expected_test_end),
            "samples_in_multiple_tests": int((test_coverage > 1).sum()),
            "avg_train_size": float(train_coverage[train_coverage > 0].mean()),
            "window_type": self.config.window_type,
        }

    def __repr__(self) -> str:
        parts = [
            f"WalkForwardEvaluator(n_windows={self.config.n_windows}",
            f"type={self.config.window_type}",
            f"min_train={self.config.min_train_pct:.0%}",
            f"test={self.config.test_pct:.0%}",
        ]
        if self.config.embargo_bars > 0:
            parts.append(f"embargo={self.config.embargo_bars}")
        if self.config.gap_bars > 0:
            parts.append(f"gap={self.config.gap_bars}")
        return ", ".join(parts) + ")"


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_walk_forward_evaluator(
    n_windows: int = 5,
    window_type: str = "expanding",
    min_train_pct: float = 0.4,
    test_pct: float = 0.1,
    embargo_bars: int = 0,
    gap_bars: int = 0,
) -> WalkForwardEvaluator:
    """
    Factory function to create a WalkForwardEvaluator.

    Args:
        n_windows: Number of walk-forward windows
        window_type: "expanding" or "rolling"
        min_train_pct: Minimum training data percentage
        test_pct: Percentage per test window
        embargo_bars: Post-test embargo bars
        gap_bars: Gap between train and test

    Returns:
        Configured WalkForwardEvaluator instance
    """
    config = WalkForwardConfig(
        n_windows=n_windows,
        window_type=window_type,
        min_train_pct=min_train_pct,
        test_pct=test_pct,
        embargo_bars=embargo_bars,
        gap_bars=gap_bars,
    )
    return WalkForwardEvaluator(config)


__all__ = [
    "WalkForwardConfig",
    "WalkForwardEvaluator",
    "WalkForwardResult",
    "WindowMetrics",
    "create_walk_forward_evaluator",
]
