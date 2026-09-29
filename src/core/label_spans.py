"""
Label spans: purging overlapping labels and sample-uniqueness weights.

A triple-barrier label decided at bar ``i`` is only known once a barrier is
touched, ``bars_to_hit`` bars later. The sample's *label span* is the closed
bar interval ``[i, i + bars_to_hit]``. Two consequences (Lopez de Prado,
AFML ch. 4 and 7):

- Cross-validation must drop every training sample whose span overlaps the
  span of the test block, or the model trains on outcomes that resolve inside
  the test period (purging).
- Samples with overlapping spans share outcome information, so they are not
  independent draws; weighting each by its *average uniqueness* keeps
  heavily-overlapping stretches from dominating the fit.

Spans are integer BAR POSITIONS in one coordinate system shared by every
sample (the rows of the labeled DataFrame), so they survive row filtering
(invalid labels, split slicing, windowing, strided subsampling) and do not
depend on the DataFrame index type.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

# Label value marking rows without a valid label (end of data, missing ATR)
INVALID_LABEL = -99
# Label-end position marking "no label end" (invalid label)
NO_LABEL_END = -1


def label_end_column(label_column: str) -> str:
    """Name of the label-end column paired with ``label_column``.

    ``label`` -> ``label_end``, ``label_h20`` -> ``label_end_h20``.
    """
    if label_column == "label":
        return "label_end"
    if label_column.startswith("label_h"):
        return "label_end_h" + label_column[len("label_h") :]
    return f"{label_column}_end"


def label_end_positions(labels: np.ndarray, bars_to_hit: np.ndarray) -> np.ndarray:
    """Bar position at which each label resolves: ``i + bars_to_hit[i]``.

    Invalid labels (-99) get ``NO_LABEL_END`` (-1).
    """
    labels = np.asarray(labels)
    bars = np.asarray(bars_to_hit, dtype=np.int64)
    if labels.shape != bars.shape:
        raise ValueError(f"labels {labels.shape} and bars_to_hit {bars.shape} differ")
    ends = np.arange(len(labels), dtype=np.int64) + bars
    return np.where(labels == INVALID_LABEL, NO_LABEL_END, ends).astype(np.int64)


def remap_label_ends(ends: np.ndarray, kept_rows: np.ndarray) -> np.ndarray:
    """Re-express label-end positions after rows were dropped.

    Args:
        ends: Label-end positions in the ORIGINAL row coordinates, one per
            kept row (``NO_LABEL_END`` where invalid).
        kept_rows: Original (strictly increasing) row position of every kept row.

    Returns:
        Label ends in the new row coordinates: the last kept row at or before
        the original end. The set of kept rows a label's span covers is
        unchanged, so purging and uniqueness are preserved exactly.
    """
    kept_rows = np.asarray(kept_rows, dtype=np.int64)
    if len(kept_rows) and np.any(np.diff(kept_rows) <= 0):
        raise ValueError("kept_rows must be strictly increasing (rows reordered?)")
    ends = np.asarray(ends, dtype=np.int64)
    valid = ends >= 0
    out = np.full(len(ends), NO_LABEL_END, dtype=np.int64)
    out[valid] = np.searchsorted(kept_rows, ends[valid], side="right") - 1
    return out


@dataclass(frozen=True)
class LabelSpans:
    """Per-sample label spans ``[starts[k], ends[k]]`` in bar positions.

    Attributes:
        starts: Bar position of each sample's label (the decision bar; for a
            windowed 3D/4D sample, the bar its window ends on).
        ends: Bar position at which the label resolves (``>= starts``), or a
            negative value when unknown.
    """

    starts: np.ndarray
    ends: np.ndarray

    def __post_init__(self) -> None:
        starts = np.asarray(self.starts, dtype=np.int64)
        ends = np.asarray(self.ends, dtype=np.int64)
        if starts.ndim != 1 or starts.shape != ends.shape:
            raise ValueError(
                f"LabelSpans needs two 1D arrays of equal length, got "
                f"{starts.shape} and {ends.shape}"
            )
        known = ends >= 0
        if np.any(ends[known] < starts[known]):
            raise ValueError("LabelSpans: a label cannot resolve before its own bar")
        object.__setattr__(self, "starts", starts)
        object.__setattr__(self, "ends", ends)

    def __len__(self) -> int:
        return len(self.starts)

    @classmethod
    def from_rows(cls, rows: np.ndarray, label_ends: np.ndarray) -> LabelSpans:
        """Spans of the samples at source ``rows`` given per-row label ends."""
        rows = np.asarray(rows, dtype=np.int64)
        return cls(starts=rows, ends=np.asarray(label_ends, dtype=np.int64)[rows])

    @classmethod
    def from_end_times(cls, index: pd.DatetimeIndex, label_end_times: pd.Series) -> LabelSpans:
        """Spans from per-row label resolution timestamps on a sorted DatetimeIndex.

        A label ending at time t covers every row stamped at or before t.
        """
        if not index.is_monotonic_increasing:
            raise ValueError("label_end_times purging needs a sorted DatetimeIndex")
        if len(label_end_times) != len(index):
            raise ValueError(f"label_end_times has {len(label_end_times)} rows, X has {len(index)}")
        end_times = pd.to_datetime(pd.Series(label_end_times).reset_index(drop=True))
        known = end_times.notna().to_numpy()
        ends = np.full(len(index), NO_LABEL_END, dtype=np.int64)
        ends[known] = (
            index.searchsorted(pd.DatetimeIndex(end_times[known]), side="right").astype(np.int64)
            - 1
        )
        starts = np.arange(len(index), dtype=np.int64)
        # Timestamps before the row itself (bad data) cannot shrink the span
        ends[known] = np.maximum(ends[known], starts[known])
        return cls(starts=starts, ends=ends)

    def subset(self, idx: np.ndarray) -> LabelSpans:
        """Spans of the samples selected by ``idx`` (positions or boolean mask)."""
        return LabelSpans(starts=self.starts[idx], ends=self.ends[idx])

    def resolved_ends(self) -> np.ndarray:
        """Ends with unknown values replaced conservatively (start + longest known span)."""
        known = self.ends >= 0
        if known.all():
            return self.ends
        longest = int((self.ends[known] - self.starts[known]).max()) if known.any() else 0
        return np.where(known, self.ends, self.starts + longest)

    def overlap_mask(self, test_start: int, test_end: int) -> np.ndarray:
        """Samples whose span overlaps the span of the test block.

        The test block is samples ``[test_start, test_end)`` (sample
        positions); its span runs from the first test bar to the latest
        test label end. Every sample (including the test block itself) whose
        ``[start, end]`` intersects it is flagged.
        """
        ends = self.resolved_ends()
        block_lo = int(self.starts[test_start:test_end].min())
        block_hi = int(max(self.starts[test_start:test_end].max(), ends[test_start:test_end].max()))
        return (self.starts <= block_hi) & (ends >= block_lo)

    def fingerprint(self) -> str:
        """Short stable hash of the spans (for cache keys)."""
        import hashlib

        digest = hashlib.sha256(self.starts.tobytes() + self.ends.tobytes())
        return digest.hexdigest()[:16]


def average_uniqueness(spans: LabelSpans) -> np.ndarray:
    """AFML 4.4 average uniqueness of each label.

    Concurrency ``c_t`` = number of the given labels whose span contains bar t.
    A label's uniqueness is the mean of ``1 / c_t`` over its own span: 1.0 for
    a label that overlaps nothing, ``1/k`` for k labels on identical spans.
    Only the given samples count toward concurrency, so pass the training
    split's spans to keep evaluation labels out of the weights.

    Samples with an unknown end get uniqueness 1.0 (they are excluded from
    training anyway and do not count toward concurrency).
    """
    n = len(spans)
    out = np.ones(n, dtype=np.float64)
    known = spans.ends >= 0
    if not known.any():
        return out
    starts = spans.starts[known]
    ends = spans.ends[known]
    lo = int(starts.min())
    hi = int(ends.max())
    size = hi - lo + 2
    # Concurrency by difference array: +1 at start, -1 after end
    diff = np.zeros(size, dtype=np.int64)
    np.add.at(diff, starts - lo, 1)
    np.add.at(diff, ends - lo + 1, -1)
    concurrency = np.cumsum(diff)[: size - 1]
    inv = np.zeros(size - 1, dtype=np.float64)
    covered = concurrency > 0
    inv[covered] = 1.0 / concurrency[covered]
    prefix = np.concatenate([[0.0], np.cumsum(inv)])
    total = prefix[ends - lo + 1] - prefix[starts - lo]
    out[known] = total / (ends - starts + 1)
    return out


def uniqueness_sample_weights(spans: LabelSpans) -> np.ndarray:
    """Average-uniqueness training weights, rescaled to mean 1 (float32).

    The mean-1 scale keeps loss magnitudes and weight-sum based regularisers
    (e.g. XGBoost ``min_child_weight``) comparable to unweighted training;
    only the relative weighting between samples carries the AFML correction.
    The mean is taken over labels with a known end (invalid labels keep 1.0
    and are dropped before training).
    """
    u = average_uniqueness(spans)
    known = spans.ends >= 0
    if known.any():
        u[known] /= float(u[known].mean())
    return u.astype(np.float32)


__all__ = [
    "INVALID_LABEL",
    "NO_LABEL_END",
    "LabelSpans",
    "average_uniqueness",
    "label_end_column",
    "label_end_positions",
    "remap_label_ends",
    "uniqueness_sample_weights",
]
