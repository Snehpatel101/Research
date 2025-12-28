"""
Combinatorially Purged Cross-Validation (CPCV).

CPCV generates multiple train/test paths by combining different fold combinations,
providing robust validation through combinatorial splitting with purging.

Reference: Lopez de Prado (2018) "Advances in Financial Machine Learning", Chapter 12

Key benefits:
- Tests model robustness across many different train/test configurations
- Generates paths for PBO (Probability of Backtest Overfitting) computation
- More thorough validation than standard k-fold

Example:
    >>> config = CPCVConfig(n_groups=6, n_test_groups=2)
    >>> cpcv = CombinatorialPurgedCV(config)
    >>> for train_idx, test_idx, path_id in cpcv.split(X, y):
    ...     model.fit(X.iloc[train_idx], y.iloc[train_idx])
    ...     predictions = model.predict(X.iloc[test_idx])
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CPCVConfig:
    """
    Configuration for Combinatorially Purged Cross-Validation.

    Attributes:
        n_groups: Number of sequential time groups to partition data into
        n_test_groups: Number of groups to hold out as test in each combination
        max_combinations: Maximum number of combinations to evaluate (limits compute)
        purge_pct: Percentage of each group to purge (prevents label leakage)
        embargo_pct: Percentage of data to embargo after test (serial correlation)

    Example:
        n_groups=6, n_test_groups=2 → C(6,2) = 15 combinations
        n_groups=10, n_test_groups=2 → C(10,2) = 45 combinations
    """
    n_groups: int = 6
    n_test_groups: int = 2
    max_combinations: int = 20
    purge_pct: float = 0.01  # 1% of total data
    embargo_pct: float = 0.01  # 1% of total data

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.n_groups < 2:
            raise ValueError(f"n_groups must be >= 2, got {self.n_groups}")

        if self.n_test_groups < 1:
            raise ValueError(f"n_test_groups must be >= 1, got {self.n_test_groups}")

        if self.n_test_groups >= self.n_groups:
            raise ValueError(
                f"n_test_groups ({self.n_test_groups}) must be < n_groups ({self.n_groups})"
            )

        if self.max_combinations < 1:
            raise ValueError(f"max_combinations must be >= 1, got {self.max_combinations}")

        if not 0 <= self.purge_pct < 0.5:
            raise ValueError(f"purge_pct must be in [0, 0.5), got {self.purge_pct}")

        if not 0 <= self.embargo_pct < 0.5:
            raise ValueError(f"embargo_pct must be in [0, 0.5), got {self.embargo_pct}")

    @property
    def n_train_groups(self) -> int:
        """Number of groups used for training in each combination."""
        return self.n_groups - self.n_test_groups

    @property
    def total_combinations(self) -> int:
        """Total possible combinations C(n_groups, n_test_groups)."""
        from math import comb
        return comb(self.n_groups, self.n_test_groups)


# =============================================================================
# CPCV RESULT
# =============================================================================

@dataclass
class CPCVPathResult:
    """Results from a single CPCV path (combination)."""
    path_id: int
    test_groups: Tuple[int, ...]
    train_size: int
    test_size: int
    train_groups: Tuple[int, ...]
    accuracy: float = 0.0
    f1: float = 0.0
    sharpe: float = 0.0
    returns: Optional[np.ndarray] = None


@dataclass
class CPCVResult:
    """
    Aggregated results from CPCV evaluation.

    Attributes:
        config: CPCVConfig used
        path_results: Results from each combination path
        model_name: Name of evaluated model
        horizon: Label horizon
    """
    config: CPCVConfig
    path_results: List[CPCVPathResult]
    model_name: str = ""
    horizon: int = 0

    @property
    def n_paths(self) -> int:
        """Number of paths evaluated."""
        return len(self.path_results)

    @property
    def mean_accuracy(self) -> float:
        """Mean accuracy across all paths."""
        if not self.path_results:
            return 0.0
        return float(np.mean([p.accuracy for p in self.path_results]))

    @property
    def std_accuracy(self) -> float:
        """Standard deviation of accuracy across paths."""
        if len(self.path_results) < 2:
            return 0.0
        return float(np.std([p.accuracy for p in self.path_results]))

    @property
    def mean_sharpe(self) -> float:
        """Mean Sharpe ratio across all paths."""
        if not self.path_results:
            return 0.0
        return float(np.mean([p.sharpe for p in self.path_results]))

    def get_oos_matrix(self) -> np.ndarray:
        """
        Get out-of-sample returns matrix for PBO computation.

        Returns:
            Matrix of shape (n_samples, n_paths) with OOS returns
        """
        if not self.path_results or self.path_results[0].returns is None:
            return np.array([])

        # Stack returns from all paths
        returns_list = [p.returns for p in self.path_results if p.returns is not None]
        if not returns_list:
            return np.array([])

        # Pad to same length if needed
        max_len = max(len(r) for r in returns_list)
        padded = []
        for r in returns_list:
            if len(r) < max_len:
                padded.append(np.pad(r, (0, max_len - len(r)), constant_values=np.nan))
            else:
                padded.append(r)

        return np.column_stack(padded)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "n_paths": self.n_paths,
            "n_groups": self.config.n_groups,
            "n_test_groups": self.config.n_test_groups,
            "mean_accuracy": self.mean_accuracy,
            "std_accuracy": self.std_accuracy,
            "mean_sharpe": self.mean_sharpe,
            "paths": [
                {
                    "path_id": p.path_id,
                    "test_groups": p.test_groups,
                    "train_size": p.train_size,
                    "test_size": p.test_size,
                    "accuracy": p.accuracy,
                    "f1": p.f1,
                    "sharpe": p.sharpe,
                }
                for p in self.path_results
            ],
        }


# =============================================================================
# COMBINATORIAL PURGED CV
# =============================================================================

class CombinatorialPurgedCV:
    """
    Combinatorially Purged Cross-Validation (CPCV).

    Generates multiple train/test paths by:
    1. Dividing data into N sequential time groups
    2. Creating all combinations of k test groups
    3. Applying purging and embargo for each combination

    This enables:
    - Robust model validation across many configurations
    - PBO (Probability of Backtest Overfitting) computation
    - Better estimation of generalization error

    Example:
        >>> config = CPCVConfig(n_groups=6, n_test_groups=2)
        >>> cpcv = CombinatorialPurgedCV(config)
        >>> for train_idx, test_idx, path_id in cpcv.split(X, y):
        ...     # Train and evaluate model
        ...     pass
    """

    def __init__(self, config: CPCVConfig) -> None:
        """
        Initialize CombinatorialPurgedCV.

        Args:
            config: CPCVConfig with CPCV parameters
        """
        self.config = config
        self._group_boundaries: Optional[List[Tuple[int, int]]] = None

    def _compute_group_boundaries(self, n_samples: int) -> List[Tuple[int, int]]:
        """Compute start/end indices for each group."""
        group_size = n_samples // self.config.n_groups
        boundaries = []

        for i in range(self.config.n_groups):
            start = i * group_size
            if i == self.config.n_groups - 1:
                end = n_samples  # Last group gets remaining samples
            else:
                end = (i + 1) * group_size
            boundaries.append((start, end))

        return boundaries

    def _get_purge_embargo_sizes(self, n_samples: int) -> Tuple[int, int]:
        """Compute purge and embargo sizes in bars."""
        purge_size = max(1, int(n_samples * self.config.purge_pct))
        embargo_size = max(1, int(n_samples * self.config.embargo_pct))
        return purge_size, embargo_size

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
        label_end_times: Optional[pd.Series] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, int]]:
        """
        Generate train/test indices for each CPCV combination.

        Args:
            X: Features DataFrame
            y: Labels (optional)
            groups: Groups (optional, unused)
            label_end_times: Label end times for overlap purging (optional)

        Yields:
            Tuple of (train_indices, test_indices, path_id) for each combination

        Example:
            >>> for train_idx, test_idx, path_id in cpcv.split(X, y):
            ...     print(f"Path {path_id}: train={len(train_idx)}, test={len(test_idx)}")
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Compute group boundaries
        self._group_boundaries = self._compute_group_boundaries(n_samples)
        purge_size, embargo_size = self._get_purge_embargo_sizes(n_samples)

        # Get datetime index if available
        has_datetime_index = isinstance(X.index, pd.DatetimeIndex)

        # Generate all test group combinations
        all_test_combos = list(combinations(range(self.config.n_groups), self.config.n_test_groups))

        # Limit combinations if needed
        if len(all_test_combos) > self.config.max_combinations:
            logger.info(
                f"Limiting combinations from {len(all_test_combos)} to {self.config.max_combinations}"
            )
            # Sample evenly across combinations
            step = len(all_test_combos) / self.config.max_combinations
            selected_indices = [int(i * step) for i in range(self.config.max_combinations)]
            all_test_combos = [all_test_combos[i] for i in selected_indices]

        for path_id, test_groups in enumerate(all_test_combos):
            train_groups = tuple(g for g in range(self.config.n_groups) if g not in test_groups)

            # Build test mask
            test_mask = np.zeros(n_samples, dtype=bool)
            for g in test_groups:
                start, end = self._group_boundaries[g]
                test_mask[start:end] = True

            # Build train mask (exclude test groups + purge + embargo)
            train_mask = np.zeros(n_samples, dtype=bool)
            for g in train_groups:
                start, end = self._group_boundaries[g]
                train_mask[start:end] = True

            # Apply purging: remove samples near test boundaries
            for g in test_groups:
                test_start, test_end = self._group_boundaries[g]

                # Purge before test
                purge_start = max(0, test_start - purge_size)
                train_mask[purge_start:test_start] = False

                # Embargo after test
                embargo_end = min(n_samples, test_end + embargo_size)
                train_mask[test_end:embargo_end] = False

            # Apply label-aware purging if label_end_times provided
            if label_end_times is not None and has_datetime_index:
                for g in test_groups:
                    test_start, _ = self._group_boundaries[g]
                    test_start_time = X.index[test_start]

                    # Remove training samples whose labels extend into test
                    for i in np.where(train_mask)[0]:
                        if i < test_start:
                            # Validate and convert label_end_time
                            label_end = label_end_times.iloc[i]
                            if pd.isna(label_end):
                                continue
                            # Ensure timestamp comparison
                            label_end = pd.Timestamp(label_end)
                            if label_end >= test_start_time:
                                train_mask[i] = False

            train_indices = indices[train_mask]
            test_indices = indices[test_mask]

            if len(train_indices) == 0:
                logger.warning(f"Path {path_id}: Empty training set, skipping")
                continue

            if len(test_indices) == 0:
                logger.warning(f"Path {path_id}: Empty test set, skipping")
                continue

            yield train_indices, test_indices, path_id

    def get_n_splits(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> int:
        """Return number of splits (combinations)."""
        return min(self.config.total_combinations, self.config.max_combinations)

    def get_path_info(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Get detailed information about each CPCV path.

        Args:
            X: Features DataFrame

        Returns:
            List of dicts with path information
        """
        info = []
        has_datetime = isinstance(X.index, pd.DatetimeIndex)

        for train_idx, test_idx, path_id in self.split(X):
            path_info = {
                "path_id": path_id,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "train_start_idx": int(train_idx[0]) if len(train_idx) > 0 else None,
                "train_end_idx": int(train_idx[-1]) if len(train_idx) > 0 else None,
                "test_start_idx": int(test_idx[0]) if len(test_idx) > 0 else None,
                "test_end_idx": int(test_idx[-1]) if len(test_idx) > 0 else None,
            }

            if has_datetime and len(train_idx) > 0:
                path_info.update({
                    "train_start_time": X.index[train_idx[0]],
                    "train_end_time": X.index[train_idx[-1]],
                    "test_start_time": X.index[test_idx[0]],
                    "test_end_time": X.index[test_idx[-1]],
                })

            info.append(path_info)

        return info

    def validate_coverage(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate CPCV coverage statistics.

        Args:
            X: Features DataFrame

        Returns:
            Dict with coverage statistics
        """
        n_samples = len(X)
        test_coverage = np.zeros(n_samples, dtype=int)
        train_coverage = np.zeros(n_samples, dtype=int)

        path_count = 0
        for train_idx, test_idx, _ in self.split(X):
            test_coverage[test_idx] += 1
            train_coverage[train_idx] += 1
            path_count += 1

        return {
            "total_samples": n_samples,
            "n_paths": path_count,
            "n_groups": self.config.n_groups,
            "n_test_groups": self.config.n_test_groups,
            "samples_never_in_test": int((test_coverage == 0).sum()),
            "samples_never_in_train": int((train_coverage == 0).sum()),
            "avg_test_appearances": float(test_coverage.mean()),
            "avg_train_appearances": float(train_coverage.mean()),
            "max_test_appearances": int(test_coverage.max()),
        }

    def __repr__(self) -> str:
        return (
            f"CombinatorialPurgedCV(n_groups={self.config.n_groups}, "
            f"n_test_groups={self.config.n_test_groups}, "
            f"paths={self.get_n_splits()})"
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_cpcv(
    n_groups: int = 6,
    n_test_groups: int = 2,
    max_combinations: int = 20,
    purge_pct: float = 0.01,
    embargo_pct: float = 0.01,
) -> CombinatorialPurgedCV:
    """
    Factory function to create CombinatorialPurgedCV.

    Args:
        n_groups: Number of sequential time groups
        n_test_groups: Groups held out as test per combination
        max_combinations: Maximum combinations to evaluate
        purge_pct: Percentage to purge before test
        embargo_pct: Percentage to embargo after test

    Returns:
        Configured CombinatorialPurgedCV instance
    """
    config = CPCVConfig(
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        max_combinations=max_combinations,
        purge_pct=purge_pct,
        embargo_pct=embargo_pct,
    )
    return CombinatorialPurgedCV(config)


__all__ = [
    "CPCVConfig",
    "CombinatorialPurgedCV",
    "CPCVResult",
    "CPCVPathResult",
    "create_cpcv",
]
