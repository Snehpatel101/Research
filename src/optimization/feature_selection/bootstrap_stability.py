"""Block-subsample feature stability (stability selection for time series).

Stability selection (Meinshausen & Buhlmann 2010) re-runs a base selector on
many random subsamples and keeps the variables that are picked in a large share
of them. An i.i.d. bootstrap is wrong for bars: rows are autocorrelated and
overlapping labels share outcomes, so resampling single rows leaks duplicates
across the train/test halves of any CV inside the replicate. Replicates here are
random CONTIGUOUS blocks (a fixed fraction of the rows, random start), so each
one is a plausible alternative history.

The base selector is the caller's: ``block_fn(start, stop)`` replays the ACTUAL
selection on that block (purged-CV MDA ranking, budget, filters, decorrelation,
per-model cut) and returns the features each model would keep. Stability is then
"how often does the real pipeline keep this feature", not "how often is it
top-ranked" -- decorrelation deliberately keeps lower-ranked cluster
representatives, which a raw top-K test would misreport as unstable.

The class only does the bookkeeping, so it cannot leak anything.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# block_fn(start, stop) -> {model: features it keeps on rows [start, stop)}, or None
# when the block cannot be ranked (too few rows, single class, CV failure).
BlockFn = Callable[[int, int], "Mapping[str, Sequence[str]] | None"]


@dataclass
class BootstrapStabilityResult:
    """Stability of a single feature across block subsamples."""

    feature_name: str
    selection_frequency: float  # share of usable blocks where ANY model keeps it
    group_frequency: dict[str, float] = field(default_factory=dict)  # per model
    is_stable: bool = False  # selection_frequency >= stability_threshold
    group_stable: dict[str, bool] = field(default_factory=dict)  # per model


@dataclass
class StabilitySummary:
    """Per-feature results plus how many blocks actually contributed."""

    results: list[BootstrapStabilityResult]
    n_blocks_drawn: int
    n_blocks_used: int


class BootstrapFeatureStability:
    """Selection frequency of each feature over random contiguous blocks.

    Args:
        n_bootstrap: Number of blocks to draw.
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
        block_fn: BlockFn,
    ) -> StabilitySummary:
        """Replay the selection on each block and count how often each feature is kept.

        Args:
            feature_names: Candidate features.
            n_rows: Rows available (blocks are drawn inside ``[0, n_rows)``).
            block_fn: Features each model keeps on rows ``[start, stop)`` (or None).

        Returns:
            A summary sorted by union selection frequency (desc); no results when
            no block could be ranked.
        """
        names = list(feature_names)
        windows = self.draw_windows(n_rows)
        if not names:
            return StabilitySummary([], len(windows), 0)

        union_hits = dict.fromkeys(names, 0)
        group_hits: dict[str, dict[str, int]] = {}
        n_used = 0
        for i, (start, stop) in enumerate(windows):
            kept = block_fn(start, stop)
            if not kept:
                logger.info("  Stability block %d [%d:%d) skipped (no ranking)", i + 1, start, stop)
                continue
            n_used += 1
            in_block: set[str] = set()
            for group, features in kept.items():
                hits = group_hits.setdefault(group, dict.fromkeys(names, 0))
                for f in features:
                    if f in hits:
                        hits[f] += 1
                        in_block.add(f)
            for f in in_block:
                union_hits[f] += 1

        if n_used == 0:
            logger.warning("Feature stability: no block could be ranked; no stability scores")
            return StabilitySummary([], len(windows), 0)

        thr = self.stability_threshold
        results = []
        for f in names:
            freq = union_hits[f] / n_used
            group_freq = {g: hits[f] / n_used for g, hits in group_hits.items()}
            results.append(
                BootstrapStabilityResult(
                    feature_name=f,
                    selection_frequency=freq,
                    group_frequency=group_freq,
                    is_stable=freq >= thr,
                    group_stable={g: v >= thr for g, v in group_freq.items()},
                )
            )
        results.sort(key=lambda r: -r.selection_frequency)
        logger.info(
            "Feature stability: %d/%d features kept in >= %.0f%% of %d blocks",
            sum(r.is_stable for r in results),
            len(results),
            thr * 100,
            n_used,
        )
        return StabilitySummary(results, len(windows), n_used)


__all__ = [
    "BlockFn",
    "BootstrapFeatureStability",
    "BootstrapStabilityResult",
    "StabilitySummary",
]
