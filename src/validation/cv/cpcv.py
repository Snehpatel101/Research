"""
Combinatorial Purged Cross-Validation (CPCV).

The T observations are partitioned into N contiguous groups. Every one of the
C(N, k) combinations of k test groups defines one split (train on the other
N - k groups, purged and embargoed around each test group). The out-of-sample
predictions of the splits are then re-assembled into

    phi = (k / N) * C(N, k) = C(N - 1, k - 1)

complete backtest paths, each of which covers every group exactly once.

Path assembly (Lopez de Prado 2018, AFML Sec. 12.4): splits are enumerated in
lexicographic order of their test-group tuples. Group g is a test group in
exactly phi splits; the j-th of those splits (in that order) supplies the
predictions of group g on path j.

Purging (AFML Sec. 7.4): a training observation i whose label spans
[i, i + purge_bars] overlaps a test group [s, e) when
i in [s - purge_bars, e + purge_bars). Those rows are dropped, and an embargo
of ``embargo_bars`` further rows after each test group is applied on top.
When label spans are given (``label_spans`` in bar positions, or
``label_end_times`` on DatetimeIndex data) any training row whose actual label
interval overlaps a test group's label interval is also dropped.

Example:
    >>> config = CPCVConfig(n_groups=6, n_test_groups=2, purge_bars=20, embargo_bars=10)
    >>> cpcv = CombinatorialPurgedCV(config)
    >>> preds = {}
    >>> for train_idx, test_idx, split_id in cpcv.split(X):
    ...     model.fit(X.iloc[train_idx], y.iloc[train_idx])
    ...     preds[split_id] = model.predict(X.iloc[test_idx])
    >>> paths = cpcv.assemble_paths(preds, n_samples=len(X))   # (phi, T)
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.core.label_spans import LabelSpans

from .purged_kfold import resolve_label_spans

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class CPCVConfig:
    """
    Configuration for Combinatorial Purged Cross-Validation.

    Attributes:
        n_groups: Number of contiguous time groups N.
        n_test_groups: Test groups per split k (1 <= k < N).
        purge_bars: Label span in bars. Training rows whose labels could
            overlap a test group (within ``purge_bars`` before it or after it)
            are removed.
        embargo_bars: Additional rows removed after each test group (on top of
            the purge) to break serial correlation.

    Example:
        n_groups=6, n_test_groups=2 -> C(6,2) = 15 splits, phi = 5 paths
        n_groups=10, n_test_groups=2 -> 45 splits, phi = 9 paths
    """

    n_groups: int = 6
    n_test_groups: int = 2
    purge_bars: int = 60
    embargo_bars: int = 0

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
        if self.purge_bars < 0:
            raise ValueError(f"purge_bars must be >= 0, got {self.purge_bars}")
        if self.embargo_bars < 0:
            raise ValueError(f"embargo_bars must be >= 0, got {self.embargo_bars}")

    @property
    def n_train_groups(self) -> int:
        """Number of groups used for training in each split."""
        return self.n_groups - self.n_test_groups

    @property
    def total_combinations(self) -> int:
        """Number of splits C(N, k)."""
        return comb(self.n_groups, self.n_test_groups)

    @property
    def n_paths(self) -> int:
        """Number of backtest paths phi = (k / N) * C(N, k) = C(N-1, k-1)."""
        return comb(self.n_groups - 1, self.n_test_groups - 1)


# =============================================================================
# CPCV RESULT
# =============================================================================


@dataclass
class CPCVPathResult:
    """
    Metrics of one assembled CPCV backtest path.

    Attributes:
        path_id: Path index in [0, phi).
        split_ids: For each group g, the split whose test predictions fill
            group g on this path.
        n_samples: Number of observations on the path (= T).
        accuracy: Classification accuracy over the path.
        f1: Weighted F1 over the path.
        sharpe: Per-period (non-annualized) Sharpe of the path's returns.
        returns: Per-period strategy returns along the path (length T).
    """

    path_id: int
    split_ids: tuple[int, ...]
    n_samples: int
    accuracy: float = 0.0
    f1: float = 0.0
    sharpe: float = 0.0
    returns: np.ndarray | None = None


@dataclass
class CPCVResult:
    """
    Aggregated results from a CPCV evaluation.

    Attributes:
        config: CPCVConfig used
        path_results: One entry per assembled backtest path (phi entries)
        model_name: Name of evaluated model
        horizon: Label horizon
    """

    config: CPCVConfig
    path_results: list[CPCVPathResult]
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
        """Mean per-period Sharpe ratio across all paths."""
        if not self.path_results:
            return 0.0
        return float(np.mean([p.sharpe for p in self.path_results]))

    def get_oos_matrix(self) -> np.ndarray:
        """
        Out-of-sample returns of every path.

        Returns:
            Array of shape (T, phi); every path covers all T observations.
            Empty array if returns were not recorded.
        """
        returns_list = [p.returns for p in self.path_results if p.returns is not None]
        if not returns_list:
            return np.array([])
        lengths = {len(r) for r in returns_list}
        if len(lengths) != 1:
            raise ValueError(f"CPCV paths have inconsistent lengths {sorted(lengths)}")
        return np.column_stack(returns_list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "n_paths": self.n_paths,
            "n_groups": self.config.n_groups,
            "n_test_groups": self.config.n_test_groups,
            "purge_bars": self.config.purge_bars,
            "embargo_bars": self.config.embargo_bars,
            "mean_accuracy": self.mean_accuracy,
            "std_accuracy": self.std_accuracy,
            "mean_sharpe": self.mean_sharpe,
            "paths": [
                {
                    "path_id": p.path_id,
                    "split_ids": list(p.split_ids),
                    "n_samples": p.n_samples,
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
    Combinatorial Purged Cross-Validation (CPCV).

    ``split`` yields every one of the C(N, k) splits as
    ``(train_indices, test_indices, split_id)``; ``assemble_paths`` turns the
    per-split test predictions into phi complete backtest paths.
    """

    def __init__(self, config: CPCVConfig) -> None:
        """
        Initialize CombinatorialPurgedCV.

        Args:
            config: CPCVConfig with CPCV parameters
        """
        self.config = config
        self._test_combinations: list[tuple[int, ...]] = list(
            combinations(range(config.n_groups), config.n_test_groups)
        )

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    @property
    def test_combinations(self) -> list[tuple[int, ...]]:
        """Test-group tuple of each split, indexed by split_id (lexicographic order)."""
        return list(self._test_combinations)

    @property
    def n_paths(self) -> int:
        """Number of backtest paths phi."""
        return self.config.n_paths

    def group_boundaries(self, n_samples: int) -> list[tuple[int, int]]:
        """Half-open [start, end) row range of each group (sizes differ by at most 1)."""
        if n_samples < self.config.n_groups:
            raise ValueError(
                f"n_samples ({n_samples}) must be >= n_groups ({self.config.n_groups})"
            )
        edges = np.linspace(0, n_samples, self.config.n_groups + 1).astype(int)
        return [(int(edges[g]), int(edges[g + 1])) for g in range(self.config.n_groups)]

    def get_path_assignments(self) -> np.ndarray:
        """
        Split assignment of every (path, group) cell.

        Returns:
            Integer array of shape (phi, N); entry [p, g] is the split_id whose
            test predictions fill group g on path p. Each group g appears as a
            test group in exactly phi splits, taken in split_id order.
        """
        n_groups = self.config.n_groups
        assignments = np.full((self.n_paths, n_groups), -1, dtype=int)
        filled = np.zeros(n_groups, dtype=int)
        for split_id, test_groups in enumerate(self._test_combinations):
            for g in test_groups:
                assignments[filled[g], g] = split_id
                filled[g] += 1
        if not np.all(filled == self.n_paths):
            raise RuntimeError(f"CPCV path assembly inconsistent: group counts {filled}")
        return assignments

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def _train_mask(
        self,
        n_samples: int,
        test_groups: tuple[int, ...],
        boundaries: list[tuple[int, int]],
        spans: LabelSpans | None,
    ) -> np.ndarray:
        """Boolean training mask for one split (test groups, purge and embargo removed)."""
        purge = self.config.purge_bars
        after = purge + self.config.embargo_bars
        mask = np.ones(n_samples, dtype=bool)
        for g in test_groups:
            start, end = boundaries[g]
            mask[max(0, start - purge) : min(n_samples, end + after)] = False
            if spans is not None:
                # Train label span [start_i, end_i] overlaps the test group's span
                mask &= ~spans.overlap_mask(start, end)
        return mask

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        groups: pd.Series | np.ndarray | None = None,
        label_end_times: pd.Series | None = None,
        label_spans: LabelSpans | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
        """
        Generate train/test indices for every CPCV split.

        Args:
            X: Features (only its length and, with ``label_end_times``, its
                DatetimeIndex are used)
            y: Unused (sklearn API compatibility)
            groups: Unused (sklearn API compatibility)
            label_end_times: Per-row label resolution timestamps (requires a
                DatetimeIndex on X). Converted to bar-position spans.
            label_spans: Per-sample label spans in bar positions (any index type).

        Yields:
            (train_indices, test_indices, split_id). Every split in
            ``test_combinations`` is yielded, so ``assemble_paths`` can build
            all phi paths. Raises instead of skipping if a split has no
            training rows left, or if label ends are given but unusable.
        """
        n_samples = len(X)
        boundaries = self.group_boundaries(n_samples)
        indices = np.arange(n_samples)
        spans = resolve_label_spans(X, n_samples, label_end_times, label_spans)

        for split_id, test_groups in enumerate(self._test_combinations):
            train_mask = self._train_mask(n_samples, test_groups, boundaries, spans)
            test_indices = np.concatenate([indices[slice(*boundaries[g])] for g in test_groups])
            train_indices = indices[train_mask]
            if len(train_indices) == 0:
                raise ValueError(
                    f"CPCV split {split_id} (test groups {test_groups}) has no training rows "
                    f"after purge={self.config.purge_bars} / embargo={self.config.embargo_bars}"
                )
            yield train_indices, test_indices, split_id

    def assemble_paths(
        self,
        split_values: Mapping[int, np.ndarray] | Sequence[np.ndarray],
        n_samples: int,
    ) -> np.ndarray:
        """
        Assemble per-split out-of-sample values into phi full backtest paths.

        Args:
            split_values: For each split_id, the values (e.g. predictions)
                aligned with that split's ``test_indices`` as yielded by
                ``split`` (test groups concatenated in ascending order).
            n_samples: T, the length of the data passed to ``split``.

        Returns:
            Array of shape (phi, T) (plus any trailing dims of the values);
            row p is backtest path p.
        """
        boundaries = self.group_boundaries(n_samples)
        assignments = self.get_path_assignments()

        # Offset of every group within each split's concatenated test block
        group_slices: dict[int, dict[int, slice]] = {}
        for split_id, test_groups in enumerate(self._test_combinations):
            offset = 0
            slices: dict[int, slice] = {}
            for g in test_groups:
                size = boundaries[g][1] - boundaries[g][0]
                slices[g] = slice(offset, offset + size)
                offset += size
            group_slices[split_id] = slices

        first = np.asarray(split_values[0])
        paths = np.empty((self.n_paths, n_samples) + first.shape[1:], dtype=np.result_type(first))
        for p in range(self.n_paths):
            for g, (start, end) in enumerate(boundaries):
                split_id = int(assignments[p, g])
                values = np.asarray(split_values[split_id])
                expected = sum(
                    boundaries[h][1] - boundaries[h][0] for h in self._test_combinations[split_id]
                )
                if len(values) != expected:
                    raise ValueError(
                        f"split {split_id}: expected {expected} test values, got {len(values)}"
                    )
                paths[p, start:end] = values[group_slices[split_id][g]]
        return paths

    def get_n_splits(
        self,
        X: pd.DataFrame | np.ndarray | None = None,
        y: pd.Series | np.ndarray | None = None,
        groups: pd.Series | np.ndarray | None = None,
    ) -> int:
        """Return number of splits C(N, k)."""
        return self.config.total_combinations

    def get_path_info(self, X: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Train/test sizes and boundaries of every split.

        Args:
            X: Features DataFrame

        Returns:
            List of dicts with split information
        """
        info = []
        has_datetime = isinstance(X.index, pd.DatetimeIndex)

        for train_idx, test_idx, split_id in self.split(X):
            split_info: dict[str, Any] = {
                "split_id": split_id,
                "test_groups": self._test_combinations[split_id],
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "train_start_idx": int(train_idx[0]),
                "train_end_idx": int(train_idx[-1]),
                "test_start_idx": int(test_idx[0]),
                "test_end_idx": int(test_idx[-1]),
            }
            if has_datetime:
                split_info.update(
                    {
                        "train_start_time": X.index[train_idx[0]],
                        "train_end_time": X.index[train_idx[-1]],
                        "test_start_time": X.index[test_idx[0]],
                        "test_end_time": X.index[test_idx[-1]],
                    }
                )
            info.append(split_info)

        return info

    def validate_coverage(self, X: pd.DataFrame | np.ndarray) -> dict[str, Any]:
        """
        CPCV coverage statistics.

        Args:
            X: Features

        Returns:
            Dict with coverage statistics
        """
        n_samples = len(X)
        test_coverage = np.zeros(n_samples, dtype=int)
        train_coverage = np.zeros(n_samples, dtype=int)

        n_splits = 0
        for train_idx, test_idx, _ in self.split(X):
            test_coverage[test_idx] += 1
            train_coverage[train_idx] += 1
            n_splits += 1

        return {
            "total_samples": n_samples,
            "n_splits": n_splits,
            "n_paths": self.n_paths,
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
            f"splits={self.get_n_splits()}, paths={self.n_paths}, "
            f"purge={self.config.purge_bars}, embargo={self.config.embargo_bars})"
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_cpcv(
    n_groups: int = 6,
    n_test_groups: int = 2,
    purge_bars: int = 60,
    embargo_bars: int = 0,
) -> CombinatorialPurgedCV:
    """
    Factory function to create CombinatorialPurgedCV.

    Args:
        n_groups: Number of contiguous time groups N
        n_test_groups: Test groups per split k
        purge_bars: Label span in bars (purge window around each test group)
        embargo_bars: Extra rows embargoed after each test group

    Returns:
        Configured CombinatorialPurgedCV instance
    """
    return CombinatorialPurgedCV(
        CPCVConfig(
            n_groups=n_groups,
            n_test_groups=n_test_groups,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
        )
    )


__all__ = [
    "CPCVConfig",
    "CombinatorialPurgedCV",
    "CPCVResult",
    "CPCVPathResult",
    "create_cpcv",
]
