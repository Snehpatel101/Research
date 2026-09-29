"""Property 4: triple-barrier labels depend only on prices.

Relabeling the time index (time zone, shifted dates, integer or string index, reversed
timestamps) must not change a single label, and every label's resolution bar is at or after
the bar it was decided on and inside the data.
"""

from __future__ import annotations

from datetime import timedelta, timezone

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from src.core.label_spans import INVALID_LABEL, NO_LABEL_END
from src.data.labeling.triple_barrier import TripleBarrierConfig, TripleBarrierLabeler
from tests.property.strategies import build_ohlcv


def _labeler(
    horizon: int, k_up: float, k_down: float, costs: bool, symbol: str
) -> TripleBarrierLabeler:
    return TripleBarrierLabeler(
        TripleBarrierConfig(
            upper_mult=k_up,
            lower_mult=k_down,
            horizon=horizon,
            atr_column=None,  # Wilder ATR computed inline from the bars
            apply_transaction_costs=costs,
            symbol=symbol,
        )
    )


def _relabeled_indexes(n: int, style: str) -> pd.Index:
    if style == "utc":
        return pd.date_range("2019-06-03 00:00", periods=n, freq="7min", tz="UTC")
    if style == "offset_tz":
        return pd.date_range(
            "2030-12-31 23:00", periods=n, freq="1h", tz=timezone(timedelta(hours=5, minutes=30))
        )
    if style == "reversed_time":  # timestamps decrease with row order
        return pd.date_range("2024-01-01", periods=n, freq="5min")[::-1]
    if style == "range":
        return pd.RangeIndex(1000, 1000 + n)
    return pd.Index([f"bar-{i}" for i in range(n)])


config_strategy = {
    "seed": st.integers(0, 2**32 - 1),
    "n": st.integers(30, 300),
    "vol": st.sampled_from([2e-4, 1e-3, 5e-3]),
    "horizon": st.integers(1, 30),
    "k_up": st.floats(0.25, 4.0),
    "k_down": st.floats(0.25, 4.0),
    "costs": st.booleans(),
    "symbol": st.sampled_from(["MES", "MGC", "MNQ"]),
    "style": st.sampled_from(["utc", "offset_tz", "reversed_time", "range", "string"]),
}


@settings(max_examples=60, deadline=None)
@given(**config_strategy)
def test_labels_and_ends_ignore_the_time_index(
    seed: int,
    n: int,
    vol: float,
    horizon: int,
    k_up: float,
    k_down: float,
    costs: bool,
    symbol: str,
    style: str,
) -> None:
    """Same prices under any index give identical labels and label-end positions."""
    bars = build_ohlcv(n, seed=seed, vol=vol)
    labeler = _labeler(horizon, k_up, k_down, costs, symbol)
    labels, ends = labeler.create_labels_with_ends(bars)

    relabeled = bars.copy()
    relabeled.index = _relabeled_indexes(n, style)
    labels2, ends2 = labeler.create_labels_with_ends(relabeled)

    np.testing.assert_array_equal(labels.to_numpy(), labels2.to_numpy())
    np.testing.assert_array_equal(ends, ends2)
    assert labels2.index.equals(relabeled.index)  # labels stay aligned to the caller's index


@settings(max_examples=60, deadline=None)
@given(**{k: v for k, v in config_strategy.items() if k != "style"})
def test_label_end_is_not_before_the_row_and_within_data(
    seed: int,
    n: int,
    vol: float,
    horizon: int,
    k_up: float,
    k_down: float,
    costs: bool,
    symbol: str,
) -> None:
    """Valid labels resolve at row >= own row (within the horizon and the data); invalid get -1."""
    bars = build_ohlcv(n, seed=seed, vol=vol)
    labels, ends = _labeler(horizon, k_up, k_down, costs, symbol).create_labels_with_ends(bars)
    positions = np.arange(n)
    valid = labels.to_numpy() != INVALID_LABEL

    assert set(np.unique(labels.to_numpy()[valid])) <= {-1, 0, 1}
    assert np.all(ends[valid] >= positions[valid])
    assert np.all(ends[valid] - positions[valid] <= horizon)
    assert np.all(ends[valid] < n)
    assert np.all(ends[~valid] == NO_LABEL_END)


@settings(max_examples=40, deadline=None)
@given(
    seed=st.integers(0, 2**32 - 1),
    n=st.integers(60, 250),
    horizon=st.integers(1, 20),
    cut_frac=st.floats(0.3, 0.9),
)
def test_labels_before_the_last_horizon_bars_ignore_appended_bars(
    seed: int, n: int, horizon: int, cut_frac: float
) -> None:
    """A label resolved within the data is final: appending later bars cannot change it.

    Labels decided in the last ``horizon`` bars of the shorter series may legitimately
    change (their barrier walk was cut off); earlier ones may not. Costs are off because
    the cost calibration uses a median ATR over the whole series.
    """
    bars = build_ohlcv(n, seed=seed, vol=1e-3)
    cut = max(horizon + 5, int(cut_frac * n))
    labeler = _labeler(horizon, 2.0, 2.0, costs=False, symbol="MES")

    short_labels, short_ends = labeler.create_labels_with_ends(bars.iloc[:cut])
    full_labels, _ = labeler.create_labels_with_ends(bars)

    settled = short_labels.to_numpy() != INVALID_LABEL
    settled &= np.arange(cut) + horizon < cut
    np.testing.assert_array_equal(
        short_labels.to_numpy()[settled], full_labels.to_numpy()[:cut][settled]
    )
    assert np.all(short_ends[settled] < cut)
