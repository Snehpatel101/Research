"""Block-subsample feature stability (stability selection for time series).

Stability selection (Meinshausen & Buhlmann 2010) re-runs a base selector on
many random subsamples and keeps the variables that are picked in a large share
of them. An i.i.d. bootstrap is wrong for bars: rows are autocorrelated and
overlapping labels share outcomes, so resampling single rows leaks duplicates
across the train/test halves of any CV inside the replicate. Replicates here are
random CONTIGUOUS blocks (a fixed fraction of the rows, random start), so each
one is a plausible alternative history; the base ranking (purged-CV
out-of-sample permutation importance, computed by the caller through
``rank_fn``) is what enforces purge/embargo inside the block.

The class only does the bookkeeping: draw blocks, rank, count how often each
feature lands in the top-K. It never sees data, so it cannot leak anything.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# rank_fn(start, stop) -> importance per feature on rows [start, stop), or None
# when the block cannot be ranked (too few rows, single class, CV failure).
RankFn = Callable[[int, int], "pd.Series | None"]


@dataclass
class BootstrapStabilityResult:
    """Stability of a single feature across block subsamples."""

    feature_name: str
    selection_frequency: float  # share of usable blocks with the feature in the top-K
    mean_rank: float  # average rank (1 = most important)
    rank_std: float  # standard deviation of the rank
    is_stable: bool  # selection_frequency >= stability_threshold


class BootstrapFeatureStability:
    """Selection frequency of each feature over random contiguous blocks.

    Args:
        n_bootstrap: Number of blocks to draw.
        top_k: A feature is "selected" in a block when it ranks within the top-K.
        stability_threshold: Minimum selection frequency to call a feature stable
            (Meinshausen & Buhlmann suggest 0.6-0.9).
        window_fraction: Block length as a share of the rows (0 < f <= 1).
        min_window_rows: Blocks shorter than this are not drawn (the data is too
            small to subsample meaningfully).
        random_state: Seed for the block starts (own generator; touches no global RNG).
    """

    def __init__(
        self,
        n_bootstrap: int = 8,
        top_k: int = 30,
        stability_threshold: float = 0.6,
        window_fraction: float = 0.5,
        min_window_rows: int = 300,
        random_state: int = 42,
    ) -> None:
        if n_bootstrap < 1:
            raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
        if not 0.0 < window_fraction <= 1.0:
            raise ValueError(f"window_fraction must be in (0, 1], got {window_fraction}")
        if not 0.0 < stability_threshold <= 1.0:
            raise ValueError(f"stability_threshold must be in (0, 1], got {stability_threshold}")
        self.n_bootstrap = n_bootstrap
        self.top_k = top_k
        self.stability_threshold = stability_threshold
        self.window_fraction = window_fraction
        self.min_window_rows = min_window_rows
        self.random_state = random_state

    def draw_windows(self, n_rows: int) -> list[tuple[int, int]]:
        """Random contiguous ``[start, stop)`` blocks covering ``window_fraction`` of the rows."""
        length = int(n_rows * self.window_fraction)
        if length < self.min_window_rows or length > n_rows:
            return []
        rng = np.random.default_rng(self.random_state)
        starts = rng.integers(0, n_rows - length + 1, size=self.n_bootstrap)
        return [(int(s), int(s) + length) for s in starts]

    def evaluate(
        self,
        feature_names: Sequence[str],
        n_rows: int,
        rank_fn: RankFn,
    ) -> list[BootstrapStabilityResult]:
        """Rank features on each block and summarise how consistently they place.

        Args:
            feature_names: Candidate features (the ranking must cover them).
            n_rows: Rows available (blocks are drawn inside ``[0, n_rows)``).
            rank_fn: Importance per feature on rows ``[start, stop)`` (or None).

        Returns:
            Results sorted by selection frequency (desc) then mean rank; empty
            when no block could be ranked.
        """
        names = list(feature_names)
        if not names:
            return []
        top_k = min(self.top_k, len(names))

        ranks_by_feature: dict[str, list[float]] = {f: [] for f in names}
        n_usable = 0
        for i, (start, stop) in enumerate(self.draw_windows(n_rows)):
            importance = rank_fn(start, stop)
            if importance is None or importance.empty:
                logger.info("  Stability block %d [%d:%d) skipped (no ranking)", i + 1, start, stop)
                continue
            n_usable += 1
            # Features missing from the ranking count as least important; ties share
            # their average rank so an all-zero ranking cannot put everything in the top-K
            ranks = (
                importance.reindex(names).fillna(-np.inf).rank(ascending=False, method="average")
            )
            for f in names:
                ranks_by_feature[f].append(float(ranks[f]))

        if n_usable == 0:
            logger.warning("Feature stability: no block could be ranked; no stability scores")
            return []

        results = []
        for f in names:
            ranks = np.asarray(ranks_by_feature[f], dtype=float)
            freq = float(np.mean(ranks <= top_k))
            results.append(
                BootstrapStabilityResult(
                    feature_name=f,
                    selection_frequency=freq,
                    mean_rank=float(ranks.mean()),
                    rank_std=float(ranks.std()),
                    is_stable=freq >= self.stability_threshold,
                )
            )
        results.sort(key=lambda r: (-r.selection_frequency, r.mean_rank))
        logger.info(
            "Feature stability: %d/%d features stable over %d blocks (top-%d, threshold %.2f)",
            sum(r.is_stable for r in results),
            len(results),
            n_usable,
            top_k,
            self.stability_threshold,
        )
        return results


__all__ = ["BootstrapFeatureStability", "BootstrapStabilityResult", "RankFn"]
