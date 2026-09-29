"""Properties 2 and 3: purged / combinatorial CV and early-stopping carves never leak.

Independent brute-force oracles (pairwise span overlap, index distances) check the
splitters, rather than re-deriving the splitters' own masks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from src.core.label_spans import LabelSpans
from src.validation.cv.cpcv import CombinatorialPurgedCV, CPCVConfig
from src.validation.cv.early_stopping_split import carve_early_stopping_split
from src.validation.cv.purged_kfold import PurgedKFold, PurgedKFoldConfig

_CV_SETTINGS = {"max_examples": 60, "deadline": None}


@st.composite
def label_spans(draw: st.DrawFn, n: int) -> LabelSpans:
    """Per-bar spans [i, i + length] with end >= start, clipped to the data."""
    max_len = draw(st.integers(0, 25))
    lengths = draw(st.lists(st.integers(0, max_len), min_size=n, max_size=n))
    starts = np.arange(n)
    ends = np.minimum(starts + np.asarray(lengths), n - 1)
    return LabelSpans(starts=starts, ends=ends)


def _frame(n: int) -> pd.DataFrame:
    return pd.DataFrame({"x": np.zeros(n)})


def _overlapping_pairs(train: np.ndarray, test: np.ndarray, spans: LabelSpans) -> int:
    """Number of (train row, test row) pairs whose closed spans intersect (brute force)."""
    ts, te = spans.starts[train][:, None], spans.ends[train][:, None]
    vs, ve = spans.starts[test][None, :], spans.ends[test][None, :]
    return int(np.count_nonzero((ts <= ve) & (vs <= te)))


def _contiguous_blocks(test: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) blocks of consecutive test rows."""
    cuts = np.flatnonzero(np.diff(test) != 1) + 1
    return [(int(b[0]), int(b[-1]) + 1) for b in np.split(np.sort(test), cuts)]


# =============================================================================
# Property 2: PurgedKFold / CombinatorialPurgedCV
# =============================================================================


@settings(suppress_health_check=[HealthCheck.filter_too_much], **_CV_SETTINGS)
@given(
    data=st.data(),
    n=st.integers(100, 500),
    n_splits=st.integers(2, 8),
    purge=st.integers(0, 15),
    embargo=st.integers(0, 30),
)
def test_purged_kfold_train_never_overlaps_test_spans_or_embargo(
    data: st.DataObject, n: int, n_splits: int, purge: int, embargo: int
) -> None:
    """No train row's label span touches a test row's span; nothing trains inside the embargo."""
    spans = data.draw(label_spans(n))
    cv = PurgedKFold(
        PurgedKFoldConfig(
            n_splits=n_splits, purge_bars=purge, embargo_bars=embargo, min_train_size=0.01
        )
    )
    try:
        folds = list(cv.split(_frame(n), label_spans=spans))
    except ValueError:  # too little data left after purge/embargo for this draw
        assume(False)
        return

    assert len(folds) == n_splits
    for train, test in folds:
        assert np.intersect1d(train, test).size == 0
        assert _overlapping_pairs(train, test, spans) == 0, "train label span overlaps a test span"
        for start, end in _contiguous_blocks(test):
            in_embargo = np.arange(end, min(n, end + embargo))
            in_purge = np.arange(max(0, start - purge), start)
            assert np.intersect1d(train, in_embargo).size == 0, "train row inside the embargo"
            assert np.intersect1d(train, in_purge).size == 0, "train row inside the purge floor"


@settings(suppress_health_check=[HealthCheck.filter_too_much], **_CV_SETTINGS)
@given(
    data=st.data(),
    n=st.integers(120, 500),
    n_groups=st.integers(3, 7),
    n_test_groups=st.integers(1, 3),
    purge=st.integers(0, 15),
    embargo=st.integers(0, 20),
)
def test_cpcv_train_never_overlaps_test_spans_or_embargo(
    data: st.DataObject, n: int, n_groups: int, n_test_groups: int, purge: int, embargo: int
) -> None:
    """Same guarantee for every C(N, k) combinatorial split (embargo follows the purge)."""
    assume(n_test_groups < n_groups)
    spans = data.draw(label_spans(n))
    cv = CombinatorialPurgedCV(
        CPCVConfig(
            n_groups=n_groups,
            n_test_groups=n_test_groups,
            purge_bars=purge,
            embargo_bars=embargo,
        )
    )
    try:
        splits = list(cv.split(_frame(n), label_spans=spans))
    except ValueError:  # a split left no training rows
        assume(False)
        return

    assert len(splits) == cv.get_n_splits()
    for train, test, _split_id in splits:
        assert np.intersect1d(train, test).size == 0
        assert _overlapping_pairs(train, test, spans) == 0, "train label span overlaps a test span"
        for start, end in _contiguous_blocks(test):
            guarded = np.arange(end, min(n, end + purge + embargo))
            before = np.arange(max(0, start - purge), start)
            assert np.intersect1d(train, guarded).size == 0, "train row inside purge+embargo"
            assert np.intersect1d(train, before).size == 0, "train row inside the purge floor"


# =============================================================================
# Property 3: early-stopping carve
# =============================================================================


@st.composite
def train_runs(draw: st.DrawFn, min_gap: int = 1) -> np.ndarray:
    """Sorted train positions made of 1-3 contiguous runs (like a PurgedKFold middle fold).

    Runs are separated by at least ``min_gap`` missing rows: in a real fold that gap is the
    held-out block plus purge and embargo, so it is never narrower than the purge.
    """
    n_runs = draw(st.integers(1, 3))
    rows: list[np.ndarray] = []
    cursor = draw(st.integers(0, 50))
    for _ in range(n_runs):
        length = draw(st.integers(1, 400))
        rows.append(np.arange(cursor, cursor + length))
        cursor += length + draw(st.integers(min_gap, min_gap + 200))
    return np.concatenate(rows)


@settings(max_examples=200, deadline=None)
@given(
    data=st.data(),
    purge=st.integers(0, 40),
    fraction=st.floats(0.05, 0.4),
    min_es=st.integers(1, 40),
    min_fit=st.integers(1, 100),
)
def test_early_stopping_carve_is_leak_free(
    data: st.DataObject, purge: int, fraction: float, min_es: int, min_fit: int
) -> None:
    """ES rows come from train, are disjoint from fit rows, contiguous and purge-separated."""
    train_idx = data.draw(train_runs(min_gap=max(1, purge)))
    split = carve_early_stopping_split(
        train_idx, purge, fraction=fraction, min_es_samples=min_es, min_fit_samples=min_fit
    )
    train = set(train_idx.tolist())

    assert set(split.es_idx.tolist()) <= train
    assert set(split.fit_idx.tolist()) <= train
    assert len(split.es_idx) > 0

    if not split.held_out:
        # Documented small-fold fallback: in-sample validation tail, nothing dropped
        assert set(split.es_idx.tolist()) <= set(split.fit_idx.tolist())
        assert len(split.purged_idx) == 0
        return

    es, fit = split.es_idx, split.fit_idx
    assert np.intersect1d(es, fit).size == 0
    assert np.intersect1d(es, split.purged_idx).size == 0
    assert np.intersect1d(fit, split.purged_idx).size == 0
    # Every train row is fit, purged or early-stopping
    assert set(fit.tolist()) | set(split.purged_idx.tolist()) | set(es.tolist()) == train
    # ES rows are one contiguous block
    assert np.all(np.diff(es) == 1)
    # Fit rows before the ES block sit at least purge_bars rows away from it
    before = fit[fit < es.min()]
    if len(before):
        assert es.min() - before.max() - 1 >= purge
    # ... and no fit row lies inside the purge window on either side
    assert not np.any((fit > es.min() - purge - 1) & (fit <= es.max() + purge))


@pytest.mark.parametrize("purge", [0, 5, 20])
def test_early_stopping_carve_never_reads_rows_outside_train(purge: int) -> None:
    """Rows absent from train_idx (e.g. the held-out fold) never appear in any output set."""
    train_idx = np.concatenate([np.arange(0, 300), np.arange(500, 800)])
    split = carve_early_stopping_split(train_idx, purge)
    used = np.concatenate([split.fit_idx, split.es_idx, split.purged_idx])
    assert np.all(np.isin(used, train_idx))
