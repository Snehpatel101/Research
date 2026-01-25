"""
Purged K-Fold Cross-Validation for Time Series.

Implements time-series aware cross-validation with purging and embargo
to prevent information leakage from overlapping labels and serial correlation.

Reference: Lopez de Prado (2018) "Advances in Financial Machine Learning"

Key concepts:
- Purge: Remove samples whose labels overlap with test set start time
- Embargo: Buffer period after test set to break serial correlation
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class PurgedKFoldConfig:
    """
    Configuration for purged k-fold cross-validation.

    Attributes:
        n_splits: Number of CV folds (default 5 for boosting, 3 for neural)
        purge_bars: Number of bars to remove before test set (default 60 = 3x max horizon).
            Should be at least max(horizons) * 3 to prevent label leakage.
        embargo_bars: Number of bars to skip after test set (default 1440 = 5 days at 5min)
        min_train_size: Minimum training set fraction (raises error if violated)
        timeframe: Optional timeframe for auto-calculating embargo_bars from calendar time.
            When provided, embargo_bars is computed as EMBARGO_TIME_MINUTES / timeframe_minutes.
            This ensures consistent ~5 day buffer regardless of bar resolution.

    Example:
        >>> # Legacy mode (assumes 5-min bars)
        >>> config = PurgedKFoldConfig(n_splits=5, purge_bars=60, embargo_bars=1440)
        >>> cv = PurgedKFold(config)

        >>> # Timeframe-aware mode (recommended)
        >>> config = PurgedKFoldConfig.from_timeframe(n_splits=5, purge_bars=60, timeframe='15min')
        >>> config.embargo_bars  # 480 bars (5 days at 15min)

        >>> # Horizon-aware mode (recommended for dynamic purge calculation)
        >>> config = PurgedKFoldConfig.from_horizons([5, 10, 20, 60, 120])
        >>> config.purge_bars  # 360 bars (max(horizons) * 3 = 120 * 3)
    """

    n_splits: int = 5
    purge_bars: int = 60
    embargo_bars: int = 1440
    min_train_size: float = 0.3
    timeframe: str | None = None  # For documentation/tracking purposes

    # Purge calculation multiplier: purge_bars = max(horizons) * PURGE_MULTIPLIER
    PURGE_MULTIPLIER: int = 3

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {self.n_splits}")
        if self.purge_bars < 0:
            raise ValueError(f"purge_bars must be >= 0, got {self.purge_bars}")
        if self.embargo_bars < 0:
            raise ValueError(f"embargo_bars must be >= 0, got {self.embargo_bars}")
        if not 0 < self.min_train_size < 1:
            raise ValueError(f"min_train_size must be in (0, 1), got {self.min_train_size}")

    @classmethod
    def from_horizons(
        cls,
        horizons: list[int],
        n_splits: int = 5,
        embargo_bars: int = 1440,
        min_train_size: float = 0.3,
        timeframe: str | None = None,
        purge_multiplier: int | None = None,
    ) -> PurgedKFoldConfig:
        """
        Create config with dynamically computed purge_bars based on prediction horizons.

        The purge period must be large enough to prevent label leakage. Since labels
        are computed over a horizon period (e.g., 120 bars forward), we need to purge
        at least that many bars before the test set to ensure no training sample's
        label depends on test data.

        Formula: purge_bars = max(horizons) * multiplier

        The default multiplier of 3 provides a safety margin:
        - 1x covers the label horizon itself
        - Additional 2x accounts for any autocorrelation in the data

        Args:
            horizons: List of prediction horizons in bars (e.g., [5, 10, 20, 60, 120])
            n_splits: Number of CV folds
            embargo_bars: Number of bars to skip after test set
            min_train_size: Minimum training set fraction
            timeframe: Optional timeframe for documentation
            purge_multiplier: Override the default multiplier (default: 3)

        Returns:
            PurgedKFoldConfig with purge_bars = max(horizons) * multiplier

        Examples:
            >>> config = PurgedKFoldConfig.from_horizons([5, 10, 20, 60, 120])
            >>> config.purge_bars
            360  # max(horizons) * 3 = 120 * 3

            >>> config = PurgedKFoldConfig.from_horizons([5, 10, 20], purge_multiplier=2)
            >>> config.purge_bars
            40   # max(horizons) * 2 = 20 * 2

        Raises:
            ValueError: If horizons list is empty or contains non-positive values
        """
        if not horizons:
            raise ValueError("horizons list cannot be empty")

        if any(h <= 0 for h in horizons):
            raise ValueError(f"All horizons must be positive, got {horizons}")

        multiplier = purge_multiplier if purge_multiplier is not None else cls.PURGE_MULTIPLIER
        max_horizon = max(horizons)
        computed_purge_bars = max_horizon * multiplier

        logger.info(
            f"Dynamic purge calculation: max(horizons)={max_horizon} * {multiplier} = {computed_purge_bars} bars"
        )

        return cls(
            n_splits=n_splits,
            purge_bars=computed_purge_bars,
            embargo_bars=embargo_bars,
            min_train_size=min_train_size,
            timeframe=timeframe,
        )

    def validate_purge_for_horizons(self, horizons: list[int]) -> list[str]:
        """
        Validate that purge_bars is sufficient for the given horizons.

        This method checks if the current purge_bars setting is large enough
        to prevent label leakage for the specified prediction horizons.

        Args:
            horizons: List of prediction horizons in bars

        Returns:
            List of warning messages (empty if validation passes)

        Example:
            >>> config = PurgedKFoldConfig(purge_bars=60)
            >>> warnings = config.validate_purge_for_horizons([5, 10, 20, 60, 120])
            >>> # Returns warning because purge_bars=60 < max(horizons)*3=360
        """
        warnings = []
        if not horizons:
            return warnings

        max_horizon = max(horizons)
        recommended_purge = max_horizon * self.PURGE_MULTIPLIER

        if self.purge_bars < recommended_purge:
            warnings.append(
                f"purge_bars={self.purge_bars} may be insufficient for horizons={horizons}. "
                f"Recommended: purge_bars >= max(horizons) * {self.PURGE_MULTIPLIER} = {recommended_purge}. "
                f"This could lead to label leakage in cross-validation."
            )
            logger.warning(warnings[-1])

        return warnings

    @classmethod
    def from_timeframe(
        cls,
        timeframe: str,
        n_splits: int = 5,
        purge_bars: int = 60,
        min_train_size: float = 0.3,
        embargo_time_minutes: int | None = None,
    ) -> PurgedKFoldConfig:
        """
        Create config with timeframe-aware embargo calculation.

        This factory method computes embargo_bars based on calendar time
        (default 5 days = 7200 minutes) to ensure consistent decorrelation
        periods regardless of bar resolution.

        Args:
            timeframe: Bar timeframe (e.g., '5min', '15min', '1h')
            n_splits: Number of CV folds
            purge_bars: Number of bars to purge before test set
            min_train_size: Minimum training set fraction
            embargo_time_minutes: Embargo duration in minutes (default: 7200 = 5 days)

        Returns:
            PurgedKFoldConfig with embargo_bars computed for the given timeframe

        Examples:
            >>> config = PurgedKFoldConfig.from_timeframe('5min')
            >>> config.embargo_bars
            1440  # 5 days at 5-min bars

            >>> config = PurgedKFoldConfig.from_timeframe('15min')
            >>> config.embargo_bars
            480   # 5 days at 15-min bars

            >>> config = PurgedKFoldConfig.from_timeframe('1h')
            >>> config.embargo_bars
            120   # 5 days at 1-hour bars
        """
        from src.core.common.horizon_config import compute_embargo_bars

        embargo_bars = compute_embargo_bars(
            timeframe=timeframe,
            embargo_time_minutes=embargo_time_minutes,
        )

        return cls(
            n_splits=n_splits,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            min_train_size=min_train_size,
            timeframe=timeframe,
        )

    @classmethod
    def from_horizons_and_timeframe(
        cls,
        horizons: list[int],
        timeframe: str,
        n_splits: int = 5,
        min_train_size: float = 0.3,
        purge_multiplier: int | None = None,
        embargo_time_minutes: int | None = None,
    ) -> PurgedKFoldConfig:
        """
        Create config with both dynamic purge (from horizons) and embargo (from timeframe).

        This is the recommended factory method when you have both prediction horizons
        and timeframe information available. It computes:
        - purge_bars = max(horizons) * multiplier (default 3)
        - embargo_bars = embargo_time_minutes / timeframe_minutes (default 5 days)

        Args:
            horizons: List of prediction horizons in bars (e.g., [5, 10, 20, 60, 120])
            timeframe: Bar timeframe (e.g., '5min', '15min', '1h')
            n_splits: Number of CV folds
            min_train_size: Minimum training set fraction
            purge_multiplier: Override purge multiplier (default: 3)
            embargo_time_minutes: Embargo duration in minutes (default: 7200 = 5 days)

        Returns:
            PurgedKFoldConfig with both purge_bars and embargo_bars computed dynamically

        Examples:
            >>> config = PurgedKFoldConfig.from_horizons_and_timeframe(
            ...     horizons=[5, 10, 20, 60, 120],
            ...     timeframe='5min'
            ... )
            >>> config.purge_bars   # 360 (max * 3)
            >>> config.embargo_bars  # 1440 (5 days at 5min)

        Raises:
            ValueError: If horizons list is empty or contains non-positive values
        """
        from src.core.common.horizon_config import compute_embargo_bars

        if not horizons:
            raise ValueError("horizons list cannot be empty")

        if any(h <= 0 for h in horizons):
            raise ValueError(f"All horizons must be positive, got {horizons}")

        multiplier = purge_multiplier if purge_multiplier is not None else cls.PURGE_MULTIPLIER
        max_horizon = max(horizons)
        computed_purge_bars = max_horizon * multiplier

        embargo_bars = compute_embargo_bars(
            timeframe=timeframe,
            embargo_time_minutes=embargo_time_minutes,
        )

        logger.info(
            f"Dynamic CV config: purge_bars={computed_purge_bars} "
            f"(max(horizons)={max_horizon} * {multiplier}), "
            f"embargo_bars={embargo_bars} (timeframe={timeframe})"
        )

        return cls(
            n_splits=n_splits,
            purge_bars=computed_purge_bars,
            embargo_bars=embargo_bars,
            min_train_size=min_train_size,
            timeframe=timeframe,
        )


# =============================================================================
# CV STRATEGIES BY MODEL FAMILY
# =============================================================================

CV_STRATEGIES: dict[str, dict] = {
    "boosting": {
        "n_splits": 5,
        "tuning_trials": 100,
        "description": "Full purged k-fold, fast retraining per fold",
    },
    "neural": {
        "n_splits": 3,
        "tuning_trials": 50,
        "description": "Fewer folds, early stopping within each fold",
    },
    "transformer": {
        "n_splits": 3,
        "tuning_trials": 30,
        "description": "Minimal folds, transfer learning between folds",
    },
    "classical": {
        "n_splits": 5,
        "tuning_trials": 50,
        "description": "Standard k-fold for classical models",
    },
}


def get_cv_config_for_family(family: str, base_config: PurgedKFoldConfig) -> PurgedKFoldConfig:
    """
    Get CV configuration adapted for model family.

    Args:
        family: Model family (boosting, neural, transformer, classical)
        base_config: Base configuration to modify

    Returns:
        PurgedKFoldConfig with family-appropriate n_splits
    """
    family_lower = family.lower()
    if family_lower not in CV_STRATEGIES:
        logger.warning(f"Unknown family '{family}', using default CV config")
        return base_config

    strategy = CV_STRATEGIES[family_lower]
    return PurgedKFoldConfig(
        n_splits=strategy["n_splits"],
        purge_bars=base_config.purge_bars,
        embargo_bars=base_config.embargo_bars,
        min_train_size=base_config.min_train_size,
    )


# =============================================================================
# PURGED K-FOLD IMPLEMENTATION
# =============================================================================


class PurgedKFold:
    """
    Time-series cross-validation with purging and embargo.

    Implements purged k-fold CV from Lopez de Prado (2018) which prevents
    information leakage in overlapping labels by:
    1. Purging samples before test set whose labels depend on test data
    2. Adding embargo period after test set to break serial correlation

    Fold structure:
        |----Train----|PURGE|--Test--|EMBARGO|----Train----|

    Attributes:
        config: PurgedKFoldConfig with CV parameters

    Example:
        >>> config = PurgedKFoldConfig(n_splits=5, purge_bars=60, embargo_bars=1440)
        >>> cv = PurgedKFold(config)
        >>> for train_idx, test_idx in cv.split(X, y):
        ...     model.fit(X.iloc[train_idx], y.iloc[train_idx])
        ...     predictions = model.predict(X.iloc[test_idx])
    """

    def __init__(self, config: PurgedKFoldConfig) -> None:
        """
        Initialize PurgedKFold.

        Args:
            config: PurgedKFoldConfig with CV parameters
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
        Generate train/test indices for each fold.

        Args:
            X: Features DataFrame with DatetimeIndex or integer index
            y: Labels (optional, unused but kept for sklearn API compatibility)
            groups: Symbol groups for symbol isolation (optional)
            label_end_times: When each label's outcome is known (optional)
                If provided, enables proper purging for overlapping labels

        Yields:
            Tuple of (train_indices, test_indices) for each fold

        Raises:
            ValueError: If training set becomes too small after purge/embargo or
                       if n_splits is too large relative to data size
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Validate n_splits is reasonable for data size
        # In k-fold CV, the worst-case training size occurs for middle folds where
        # both purge (before test) and embargo (after test) reduce training data
        # Correct formula: train_size = n_samples - purge - test_size - embargo
        test_size = n_samples // self.config.n_splits
        min_train = int(n_samples * self.config.min_train_size)
        worst_case_train = n_samples - self.config.purge_bars - test_size - self.config.embargo_bars

        if worst_case_train < min_train:
            # Calculate maximum reasonable n_splits given constraints
            # Derived from: n_samples - purge - (n_samples/n_splits) - embargo >= min_train
            # => n_samples/n_splits <= n_samples - purge - embargo - min_train
            # => n_splits <= n_samples / (n_samples - purge - embargo - min_train) [invalid]
            # Simpler: just report what the problem is
            raise ValueError(
                f"n_splits={self.config.n_splits} is too large for {n_samples} samples. "
                f"Worst-case training size ({worst_case_train}) would be below minimum ({min_train}). "
                f"With purge_bars={self.config.purge_bars} and embargo_bars={self.config.embargo_bars}, "
                f"consider reducing n_splits or increasing data size."
            )

        # Get timestamps if available (for label-aware purging)
        has_datetime_index = isinstance(X.index, pd.DatetimeIndex)

        # Calculate fold boundaries (test_size already calculated in validation above)
        fold_size = test_size

        for fold_idx in range(self.config.n_splits):
            # Test fold boundaries
            test_start = fold_idx * fold_size
            if fold_idx == self.config.n_splits - 1:
                test_end = n_samples  # Last fold gets remaining samples
            else:
                test_end = (fold_idx + 1) * fold_size

            test_indices = indices[test_start:test_end]

            # Training indices: everything except test + purge + embargo
            train_mask = np.ones(n_samples, dtype=bool)

            # Remove test period
            train_mask[test_start:test_end] = False

            # Apply purge before test
            purge_start = max(0, test_start - self.config.purge_bars)
            train_mask[purge_start:test_start] = False

            # Apply embargo after test
            embargo_end = min(n_samples, test_end + self.config.embargo_bars)
            train_mask[test_end:embargo_end] = False

            # Additional purge for overlapping labels (if label_end_times provided)
            # BUG FIX: Check ALL training samples, not just those before purge_start
            # Training data can exist on BOTH sides of the test set in k-fold CV
            # Any sample whose label extends into test period must be excluded
            if label_end_times is not None and has_datetime_index:
                test_start_time = X.index[test_start]
                test_end_time = X.index[test_end - 1]

                # Check every potential training sample for label overlap
                for i in range(n_samples):
                    if train_mask[i]:  # Only check samples still in training set
                        label_end = label_end_times.iloc[i]
                        # Skip samples with missing label_end_times (NaT/None)
                        # These are typically invalid labels that should have been filtered
                        if pd.isna(label_end):
                            continue
                        # Remove if label's outcome period overlaps with test period
                        # This handles:
                        # 1. Samples before test whose labels extend into test period
                        # 2. Samples after embargo whose labels started during test period
                        if label_end >= test_start_time and X.index[i] <= test_end_time:
                            train_mask[i] = False

            train_indices = indices[train_mask]

            # Validate minimum training size
            if len(train_indices) < min_train:
                raise ValueError(
                    f"Fold {fold_idx}: Training set too small after purge/embargo "
                    f"({len(train_indices)} < {min_train}). Consider reducing "
                    f"n_splits, purge_bars, or embargo_bars."
                )

            yield train_indices, test_indices

    def get_n_splits(
        self,
        X: pd.DataFrame | None = None,
        y: pd.Series | None = None,
        groups: pd.Series | None = None,
    ) -> int:
        """Return number of splits (sklearn API compatibility)."""
        return self.config.n_splits

    def get_fold_info(self, X: pd.DataFrame) -> list[dict]:
        """
        Get detailed information about each fold.

        Args:
            X: Features DataFrame (needed for timestamp info)

        Returns:
            List of dicts with fold information including sizes and time ranges
        """
        info = []
        has_datetime_index = isinstance(X.index, pd.DatetimeIndex)

        for fold_idx, (train_idx, test_idx) in enumerate(self.split(X)):
            fold_info = {
                "fold": fold_idx,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "purge_bars": self.config.purge_bars,
                "embargo_bars": self.config.embargo_bars,
            }

            if has_datetime_index:
                fold_info.update(
                    {
                        "train_start": X.index[train_idx[0]],
                        "train_end": X.index[train_idx[-1]],
                        "test_start": X.index[test_idx[0]],
                        "test_end": X.index[test_idx[-1]],
                    }
                )
            else:
                fold_info.update(
                    {
                        "train_start_idx": int(train_idx[0]),
                        "train_end_idx": int(train_idx[-1]),
                        "test_start_idx": int(test_idx[0]),
                        "test_end_idx": int(test_idx[-1]),
                    }
                )

            info.append(fold_info)

        return info

    def validate_coverage(self, X: pd.DataFrame) -> dict:
        """
        Validate that CV covers all samples at least once.

        Args:
            X: Features DataFrame

        Returns:
            Dict with coverage statistics
        """
        n_samples = len(X)
        test_coverage = np.zeros(n_samples, dtype=int)

        for _, test_idx in self.split(X):
            test_coverage[test_idx] += 1

        return {
            "total_samples": n_samples,
            "samples_in_test": int((test_coverage > 0).sum()),
            "coverage_fraction": float((test_coverage > 0).mean()),
            "samples_in_multiple_folds": int((test_coverage > 1).sum()),
            "uncovered_samples": int((test_coverage == 0).sum()),
        }

    def __repr__(self) -> str:
        return (
            f"PurgedKFold(n_splits={self.config.n_splits}, "
            f"purge={self.config.purge_bars}, embargo={self.config.embargo_bars})"
        )


# =============================================================================
# MODEL-AWARE CV WRAPPER
# =============================================================================


class ModelAwareCV:
    """
    Cross-validation strategy adapted to model training costs.

    Different model families require different CV strategies:
    - Boosting: Fast training allows more folds (5)
    - Neural: Moderate training time, fewer folds (3)
    - Transformer: Expensive training, minimal folds (3)

    Example:
        >>> base_cv = PurgedKFold(PurgedKFoldConfig())
        >>> model_cv = ModelAwareCV("neural", base_cv)
        >>> for train_idx, test_idx in model_cv.get_cv_splits(X):
        ...     # Train neural model
    """

    def __init__(self, model_family: str, base_cv: PurgedKFold) -> None:
        """
        Initialize ModelAwareCV.

        Args:
            model_family: Model family (boosting, neural, transformer, classical)
            base_cv: Base PurgedKFold instance
        """
        self.model_family = model_family.lower()
        self.base_cv = base_cv

        if self.model_family in CV_STRATEGIES:
            self.strategy = CV_STRATEGIES[self.model_family]
        else:
            logger.warning(f"Unknown family '{model_family}', using default strategy")
            self.strategy = CV_STRATEGIES["boosting"]

    def get_cv_splits(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        label_end_times: pd.Series | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Return appropriate number of splits for model family.

        Args:
            X: Features DataFrame
            y: Labels (optional)
            label_end_times: When labels are resolved (optional)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_splits = self.strategy["n_splits"]

        # Adjust base CV if needed
        if n_splits != self.base_cv.config.n_splits:
            adjusted_config = PurgedKFoldConfig(
                n_splits=n_splits,
                purge_bars=self.base_cv.config.purge_bars,
                embargo_bars=self.base_cv.config.embargo_bars,
                min_train_size=self.base_cv.config.min_train_size,
            )
            cv = PurgedKFold(adjusted_config)
        else:
            cv = self.base_cv

        yield from cv.split(X, y, label_end_times=label_end_times)

    def get_tuning_trials(self) -> int:
        """Return appropriate number of Optuna trials for model family."""
        return int(self.strategy["tuning_trials"])

    def get_n_splits(self) -> int:
        """Return number of CV splits for this model family."""
        return int(self.strategy["n_splits"])


__all__ = [
    "PurgedKFoldConfig",
    "PurgedKFold",
    "ModelAwareCV",
    "CV_STRATEGIES",
    "get_cv_config_for_family",
]
