"""
Leak-free early-stopping sets for fold models.

A fold model that early-stops (boosting rounds, neural best-epoch restore)
selects its stopping point on whatever is passed as ``X_val``. If that is
the held-out fold it then predicts, the out-of-fold predictions are
optimistic: the held-out rows chose the model.

``carve_early_stopping_split`` takes the fold's TRAIN indices and carves a
contiguous tail off one of its contiguous runs as the early-stopping set,
dropping ``purge_bars`` rows between it and the remaining fit rows so no
fit-row label overlaps the early-stopping labels. The held-out fold is
never touched.

When the fold is too small to spare a separate set, the split falls back to
an early-stopping set drawn from the fit rows themselves. In-sample
validation loss keeps improving, so the model effectively trains for its
full fixed budget — no held-out information is used either way.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Fraction of a fold's training rows used for early stopping.
DEFAULT_EARLY_STOPPING_FRACTION = 0.15
# Smallest early-stopping set worth stopping on (fewer rows = noisy stopping point).
MIN_EARLY_STOPPING_SAMPLES = 20
# Smallest fit set left after carving; below this the carve is not worth it.
MIN_FIT_SAMPLES = 50


@dataclass(frozen=True)
class EarlyStoppingSplit:
    """
    Partition of a fold's training indices.

    Attributes:
        fit_idx: Rows the fold model is fit on.
        es_idx: Rows passed as X_val/y_val (early stopping / best-epoch).
        purged_idx: Rows dropped between fit and early-stopping rows.
        held_out: True when es_idx is disjoint from fit_idx. False on the
            small-fold fallback, where es_idx is a tail of the fit rows and
            the fit is effectively fixed-length.
    """

    fit_idx: np.ndarray
    es_idx: np.ndarray
    purged_idx: np.ndarray
    held_out: bool


def _contiguous_runs(sorted_idx: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) positions (end exclusive) of consecutive-integer runs."""
    if len(sorted_idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(sorted_idx) != 1) + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(sorted_idx)]])
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def carve_early_stopping_split(
    train_idx: np.ndarray,
    purge_bars: int,
    fraction: float = DEFAULT_EARLY_STOPPING_FRACTION,
    min_es_samples: int = MIN_EARLY_STOPPING_SAMPLES,
    min_fit_samples: int = MIN_FIT_SAMPLES,
) -> EarlyStoppingSplit:
    """
    Carve a purged, contiguous early-stopping tail from a fold's train rows.

    The tail is taken from the latest contiguous run of ``train_idx`` long
    enough to hold it plus the purge gap (for a middle PurgedKFold fold the
    train rows are two runs, before and after the held-out fold).

    Args:
        train_idx: The fold's training row positions.
        purge_bars: Rows dropped between the fit rows and the tail.
        fraction: Share of the training rows used for early stopping.
        min_es_samples: Minimum early-stopping set size.
        min_fit_samples: Minimum fit rows that must remain.

    Returns:
        EarlyStoppingSplit (``held_out=False`` on the small-fold fallback).
    """
    train_idx = np.sort(np.asarray(train_idx, dtype=np.int64))
    n_train = len(train_idx)
    purge_bars = max(int(purge_bars), 0)
    n_es = max(int(round(fraction * n_train)), min_es_samples)

    if n_train - n_es - purge_bars >= min_fit_samples:
        for start, end in reversed(_contiguous_runs(train_idx)):
            if end - start < n_es + purge_bars:
                continue
            es_pos = np.arange(end - n_es, end)
            purged_pos = np.arange(end - n_es - purge_bars, end - n_es)
            keep = np.ones(n_train, dtype=bool)
            keep[es_pos] = False
            keep[purged_pos] = False
            return EarlyStoppingSplit(
                fit_idx=train_idx[keep],
                es_idx=train_idx[es_pos],
                purged_idx=train_idx[purged_pos],
                held_out=True,
            )

    n_tail = min(n_es, n_train)
    logger.warning(
        "Fold too small for a held-out early-stopping set (%d train rows, need %d "
        "tail + %d purge + %d fit); fitting fixed-length with an in-sample "
        "validation tail of %d rows.",
        n_train,
        n_es,
        purge_bars,
        min_fit_samples,
        n_tail,
    )
    return EarlyStoppingSplit(
        fit_idx=train_idx,
        es_idx=train_idx[n_train - n_tail :],
        purged_idx=np.empty(0, dtype=np.int64),
        held_out=False,
    )


__all__ = [
    "DEFAULT_EARLY_STOPPING_FRACTION",
    "MIN_EARLY_STOPPING_SAMPLES",
    "MIN_FIT_SAMPLES",
    "EarlyStoppingSplit",
    "carve_early_stopping_split",
]
