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

from src.core.label_spans import LabelSpans

logger = logging.getLogger(__name__)


def resolve_label_spans(
    X: pd.DataFrame | np.ndarray,
    n_samples: int,
    label_end_times: pd.Series | None,
    label_spans: LabelSpans | None,
) -> LabelSpans | None:
    """Normalize the two ways of passing label ends to one ``LabelSpans``.

    Raises instead of silently ignoring label ends that cannot be used.
    """
    if label_spans is not None and label_end_times is not None:
        raise ValueError("Pass either label_spans or label_end_times, not both")
    if label_end_times is not None:
        index = getattr(X, "index", None)
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError(
                "label_end_times needs X with a DatetimeIndex to locate label ends; "
                "pass label_spans (integer bar positions) for other index types"
            )
        label_spans = LabelSpans.from_end_times(index, label_end_times)
    if label_spans is not None and len(label_spans) != n_samples:
        raise ValueError(
            f"label_spans has {len(label_spans)} samples but X has {n_samples}; "
            "label ends must be subset together with the samples"
        )
    return label_spans


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class PurgedKFoldConfig:
    """
    Configuration for purged k-fold cross-validation.

    Attributes:
        n_splits: Number of CV folds (default 5 for boosting, 3 for neural)
        purge_bars: Samples removed immediately before each test block. A floor
            that applies even without label spans; with spans the purge follows
            each label's actual resolution bar (see ``PurgedKFold.split``).
            Should be at least the longest label span (triple-barrier max_bars).
        embargo_bars: Samples skipped after each test block (serial correlation).
        min_train_size: Minimum training set fraction (raises error if violated)
        timeframe: Optional bar timeframe, for documentation/tracking only.

    ``ExperimentConfig.resolve_cv_gaps`` derives purge (longest label span)
    and embargo (one trading day of bars) for factory runs.

    Example:
        >>> config = PurgedKFoldConfig(n_splits=5, purge_bars=12, embargo_bars=288)
        >>> cv = PurgedKFold(config)
    """

    n_splits: int = 5
    purge_bars: int = 60
    embargo_bars: int = 1440
    min_train_size: float = 0.3
    timeframe: str | None = None  # For documentation/tracking purposes

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
        label_spans: LabelSpans | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for each fold.

        Purging always drops ``purge_bars`` samples before each test block and
        ``embargo_bars`` after it. When label spans are known, every training
        sample whose label span overlaps the test block's span is dropped as
        well, wherever it sits (the fixed purge is then only a floor).

        Args:
            X: Features (only its length and, with ``label_end_times``, its
                DatetimeIndex are used)
            y: Labels (optional, unused but kept for sklearn API compatibility)
            groups: Symbol groups (optional, unused)
            label_end_times: Per-row label resolution timestamps. Requires a
                sorted DatetimeIndex on X; converted to bar-position spans.
            label_spans: Per-sample label spans in bar positions
                (index-type independent). Preferred over ``label_end_times``.

        Yields:
            Tuple of (train_indices, test_indices) for each fold

        Raises:
            ValueError: If training set becomes too small after purge/embargo,
                if n_splits is too large relative to data size, or if label
                ends are given but cannot be used (length mismatch, or
                ``label_end_times`` without a DatetimeIndex).
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        spans = resolve_label_spans(X, n_samples, label_end_times, label_spans)

        # Validate n_splits is reasonable for data size
        # In k-fold CV, the worst-case training size occurs for middle folds where
        # both purge (before test) and embargo (after test) reduce training data
        # Correct formula: train_size = n_samples - purge - test_size - embargo
        test_size = n_samples // self.config.n_splits
        min_train = int(n_samples * self.config.min_train_size)
        worst_case_train = n_samples - self.config.purge_bars - test_size - self.config.embargo_bars

        if worst_case_train < min_train:
            raise ValueError(
                f"n_splits={self.config.n_splits} is too large for {n_samples} samples. "
                f"Worst-case training size ({worst_case_train}) would be below minimum ({min_train}). "
                f"With purge_bars={self.config.purge_bars} and embargo_bars={self.config.embargo_bars}, "
                f"consider reducing n_splits or increasing data size."
            )

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
            train_mask[test_start:test_end] = False

            # Fixed purge floor before test
            purge_start = max(0, test_start - self.config.purge_bars)
            train_mask[purge_start:test_start] = False

            # Embargo after test
            embargo_end = min(n_samples, test_end + self.config.embargo_bars)
            train_mask[test_end:embargo_end] = False

            # Label-overlap purge on both sides of the test block
            if spans is not None:
                train_mask &= ~spans.overlap_mask(test_start, test_end)

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
        label_spans: LabelSpans | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Return appropriate number of splits for model family.

        Args:
            X: Features DataFrame
            y: Labels (optional)
            label_end_times: When labels are resolved (optional)
            label_spans: Label spans in bar positions (optional)

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

        yield from cv.split(X, y, label_end_times=label_end_times, label_spans=label_spans)

    def get_tuning_trials(self) -> int:
        """Return appropriate number of Optuna trials for model family."""
        return int(self.strategy["tuning_trials"])

    def get_n_splits(self) -> int:
        """Return number of CV splits for this model family."""
        return int(self.strategy["n_splits"])


__all__ = [
    "resolve_label_spans",
    "PurgedKFoldConfig",
    "PurgedKFold",
    "ModelAwareCV",
    "CV_STRATEGIES",
    "get_cv_config_for_family",
]
