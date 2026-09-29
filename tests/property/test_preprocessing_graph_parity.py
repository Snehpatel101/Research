"""Property 6: the serving path (PreprocessingGraph) is causal and matches batch computation.

Transforming a prefix of raw bars must give exactly the rows the full-series transform gives
for the same timestamps: serving on a live, growing bar stream sees the same features the model
was trained on, with no dependence on bars that have not happened yet.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.core.constants import OHLCV_COLUMNS
from src.data.pipeline.stages.features.engineer import FeatureEngineer
from src.inference.preprocessing_graph import PreprocessingGraph
from tests.property.strategies import budget, build_ohlcv

RTOL = 1e-6
ATOL = 1e-9
_SLOW_OK = {"deadline": None, "suppress_health_check": [HealthCheck.too_slow]}


@cache
def _graph(enable_mtf: bool) -> PreprocessingGraph:
    """Graph over every feature the engineer produces (like a model trained on all of them)."""
    engineer = FeatureEngineer(enable_mtf=enable_mtf, mtf_min_rows=200)
    warmup_frame = build_ohlcv(1200 if enable_mtf else 400, seed=0, vol=1e-3).reset_index()
    features, _ = engineer.compute_features(warmup_frame)
    columns = [c for c in features.columns if c not in ("datetime", *OHLCV_COLUMNS)]
    return PreprocessingGraph.from_feature_pipeline(
        {"bar_timeframe": "5min", "engineer": engineer.to_spec()}, feature_columns=columns
    )


def _as_raw(bars: pd.DataFrame, datetime_column: bool, shuffle_seed: int | None) -> pd.DataFrame:
    """Raw serving input: index or ``datetime`` column, optionally in arbitrary row order."""
    raw = bars.copy()
    if shuffle_seed is not None:
        raw = raw.iloc[np.random.default_rng(shuffle_seed).permutation(len(raw))]
    return raw.reset_index() if datetime_column else raw


def _assert_prefix_matches_full(
    graph: PreprocessingGraph,
    bars: pd.DataFrame,
    t: int,
    datetime_column: bool,
    shuffle: int | None,
) -> None:
    full = graph.transform(_as_raw(bars, datetime_column, None), skip_scaling=True)
    prefix = graph.transform(
        _as_raw(bars.iloc[: t + 1], datetime_column, shuffle), skip_scaling=True
    )

    # Every full-series row up to bar t, not just a leading slice of the full output:
    # a feature reading future bars would be NaN (dropped) on the prefix's last rows,
    # so a shorter prefix must fail here rather than be compared on fewer rows.
    expected = full.loc[full.index <= bars.index[t]]

    assert len(expected) > 0
    assert list(prefix.columns) == list(full.columns)
    assert prefix.index.equals(expected.index), "prefix rows differ from the full rows up to t"
    np.testing.assert_allclose(
        prefix.to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        rtol=RTOL,
        atol=ATOL,
        err_msg="serving features on a prefix differ from the full-series features (lookahead)",
    )


@settings(budget(8, heavy=True), **_SLOW_OK)
@given(
    seed=st.integers(0, 2**32 - 1),
    vol=st.sampled_from([5e-4, 2e-3, 1e-2]),
    t_frac=st.floats(0.55, 0.95),
    datetime_column=st.booleans(),
    shuffle=st.one_of(st.none(), st.integers(0, 2**32 - 1)),
)
def test_transform_on_prefix_equals_rows_of_full_transform(
    seed: int, vol: float, t_frac: float, datetime_column: bool, shuffle: int | None
) -> None:
    """transform(bars[:t+1]) == transform(bars)[rows up to t] (after warmup)."""
    bars = build_ohlcv(400, seed=seed, vol=vol)
    t = int(t_frac * len(bars))

    _assert_prefix_matches_full(_graph(False), bars, t, datetime_column, shuffle)


@settings(budget(3, heavy=True), **_SLOW_OK)
@given(
    seed=st.integers(0, 2**32 - 1),
    t_frac=st.floats(0.6, 0.95),
)
def test_transform_prefix_parity_with_multi_timeframe_features(seed: int, t_frac: float) -> None:
    """Same parity with 15min / 60min MTF features enabled."""
    bars = build_ohlcv(1200, seed=seed, vol=5e-4)
    t = int(t_frac * len(bars))

    _assert_prefix_matches_full(_graph(True), bars, t, datetime_column=False, shuffle=None)
