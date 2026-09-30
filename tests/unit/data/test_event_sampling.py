"""CUSUM event sampling (AFML ch. 2): events on known jumps, train-only threshold, causality.

The factory-level tests build an ``MLFactory`` only to call its resolution
helpers; no model is trained.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config.experiment import ExperimentConfig
from src.core.label_spans import INVALID_LABEL, NO_LABEL_END
from src.data.features.cusum_filter import (
    auto_cusum_threshold,
    log_returns,
)
from src.data.labeling.event_sampling import (
    EventSamplingSpec,
    apply_event_mask,
    resolve_event_sampling,
)
from src.factory import MLFactory
from tests.helpers import make_intraday_ohlcv


def cusum_event_mask(close: pd.Series, threshold: float) -> np.ndarray:
    return EventSamplingSpec("cusum", threshold).mask(close)


def _price_with_jumps(n: int, jumps: dict[int, float]) -> pd.Series:
    """Flat log price with instantaneous log-return jumps at the given bars."""
    log_price = np.zeros(n)
    for pos, size in jumps.items():
        log_price[pos:] += size
    return pd.Series(100.0 * np.exp(log_price))


# ---------------------------------------------------------------------------
# Events on synthetic series with known jumps
# ---------------------------------------------------------------------------


def test_events_fire_exactly_on_known_jumps() -> None:
    close = _price_with_jumps(300, {50: 0.03, 120: -0.03, 200: 0.03})
    mask = cusum_event_mask(close, threshold=0.02)
    assert np.flatnonzero(mask).tolist() == [50, 120, 200]


def test_no_events_on_a_flat_series() -> None:
    assert not cusum_event_mask(pd.Series(np.full(200, 100.0)), threshold=0.01).any()


def test_slow_drift_fires_at_the_predicted_bar_and_resets() -> None:
    # +0.3% per bar: three returns sum to 0.9% (below), four to 1.2% (above the 1%
    # threshold), so an event fires every 4th bar and the sum resets each time
    close = pd.Series(100.0 * np.exp(0.003 * np.arange(60)))
    events = np.flatnonzero(cusum_event_mask(close, threshold=0.01))
    assert events[:4].tolist() == [4, 8, 12, 16]


def test_small_moves_below_threshold_never_fire() -> None:
    rng = np.random.default_rng(0)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 1e-4, 500))))
    assert not cusum_event_mask(close, threshold=0.5).any()


def test_mask_is_causal_prefix_property() -> None:
    """The mask of a prefix equals the prefix of the mask: no bar sees the future."""
    close = make_intraday_ohlcv(3000, seed=3)["close"].reset_index(drop=True)
    threshold = 1.5 * float(log_returns(close).std())
    full = cusum_event_mask(close, threshold)
    for cut in (500, 1234, 2500):
        np.testing.assert_array_equal(cusum_event_mask(close.iloc[:cut], threshold), full[:cut])
    assert full.sum() > 20


def test_perturbing_future_bars_does_not_change_past_events() -> None:
    close = make_intraday_ohlcv(2000, seed=4)["close"].reset_index(drop=True)
    threshold = 1.5 * float(log_returns(close).std())
    base = cusum_event_mask(close, threshold)
    perturbed = close.copy()
    perturbed.iloc[1500:] *= 1.05
    np.testing.assert_array_equal(cusum_event_mask(perturbed, threshold)[:1500], base[:1500])


# ---------------------------------------------------------------------------
# Auto threshold: TRAIN rows only
# ---------------------------------------------------------------------------


def test_auto_threshold_is_a_multiple_of_train_volatility() -> None:
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0, 0.002, 5000))
    assert auto_cusum_threshold(returns, vol_multiple=3.0) == pytest.approx(
        3.0 * float(returns.std(ddof=1))
    )


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_auto_threshold_rejects_non_positive_multiple(bad: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        auto_cusum_threshold(pd.Series(np.random.default_rng(0).normal(size=100)), bad)


def test_auto_threshold_rejects_too_few_or_constant_returns() -> None:
    with pytest.raises(ValueError, match="at least 30"):
        auto_cusum_threshold(pd.Series([0.001] * 10))
    with pytest.raises(ValueError, match="no variance"):
        auto_cusum_threshold(pd.Series([0.0] * 100))


def test_resolve_event_sampling_none_and_explicit_and_unknown() -> None:
    close = make_intraday_ohlcv(500)["close"]
    assert resolve_event_sampling("none", "auto", 3.0, close) is None
    spec = resolve_event_sampling("cusum", 0.004, 3.0, close)
    assert spec == EventSamplingSpec("cusum", 0.004)
    with pytest.raises(ValueError, match="Unknown event sampling"):
        resolve_event_sampling("bogus", "auto", 3.0, close)
    with pytest.raises(ValueError, match="number or 'auto'"):
        resolve_event_sampling("cusum", "sometimes", 3.0, close)


def test_spec_round_trips_and_validates() -> None:
    spec = EventSamplingSpec("cusum", 0.0123)
    assert EventSamplingSpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(ValueError):
        EventSamplingSpec("cusum", 0.0)
    with pytest.raises(ValueError):
        EventSamplingSpec("nope", 0.01)


def _factory(tmp_path: Path, **labeling: object) -> MLFactory:
    cfg = ExperimentConfig()
    cfg.verbose = 0
    cfg.output_dir = tmp_path / "out" / cfg.run_id
    cfg.data.symbol = "MES"
    cfg.data.labeling.event_sampling = "cusum"
    for key, value in labeling.items():
        setattr(cfg.data.labeling, key, value)
    return MLFactory(cfg, verbose=0, enable_checkpoints=False)


def test_auto_threshold_uses_train_rows_only(tmp_path: Path) -> None:
    """Changing validation/test rows does not change the threshold; changing train rows does."""
    raw = make_intraday_ohlcv(4000, seed=5)
    factory = _factory(tmp_path)
    train_end = int(len(raw) * factory.config.data.splits.train_ratio)

    base = factory._resolve_event_sampling(raw)
    assert base is not None

    later = raw.copy()
    rng = np.random.default_rng(9)
    later.iloc[train_end:, later.columns.get_loc("close")] *= np.exp(
        np.cumsum(rng.normal(0, 0.05, len(raw) - train_end))  # wildly different val/test period
    )
    assert factory._resolve_event_sampling(later) == base

    earlier = raw.copy()
    earlier.iloc[: train_end // 2, earlier.columns.get_loc("close")] *= np.exp(
        np.cumsum(rng.normal(0, 0.01, train_end // 2))
    )
    assert factory._resolve_event_sampling(earlier) != base


def test_explicit_threshold_is_used_verbatim(tmp_path: Path) -> None:
    factory = _factory(tmp_path, cusum_threshold=0.0042)
    spec = factory._resolve_event_sampling(make_intraday_ohlcv(1000))
    assert spec == EventSamplingSpec("cusum", 0.0042)


def test_no_event_sampling_by_default(tmp_path: Path) -> None:
    factory = _factory(tmp_path, event_sampling="none")
    assert factory._resolve_event_sampling(make_intraday_ohlcv(1000)) is None


# ---------------------------------------------------------------------------
# Label invalidation keeps spans in bar coordinates
# ---------------------------------------------------------------------------


def test_apply_event_mask_invalidates_labels_and_ends_only_off_events() -> None:
    labels = np.array([1, -1, 0, 1, -1, 0], dtype=np.int8)
    ends = np.array([2, 3, 5, 5, 8, 9], dtype=np.int64)
    events = np.array([True, False, True, False, False, True])
    out_labels, out_ends = apply_event_mask(labels, ends, events, INVALID_LABEL)
    assert out_labels.tolist() == [1, INVALID_LABEL, 0, INVALID_LABEL, INVALID_LABEL, 0]
    assert out_ends.tolist() == [2, NO_LABEL_END, 5, NO_LABEL_END, NO_LABEL_END, 9]
    assert out_labels.dtype == labels.dtype
    # inputs untouched
    assert labels.tolist() == [1, -1, 0, 1, -1, 0]


def test_uniqueness_weights_follow_only_the_event_spans() -> None:
    """Non-event rows carry no span, so they do not dilute the events' uniqueness."""
    from src.core.label_spans import LabelSpans, uniqueness_sample_weights

    starts = np.arange(10, dtype=np.int64)
    dense_ends = starts + 4
    events = np.zeros(10, dtype=bool)
    events[[0, 9]] = True
    _labels, sparse_ends = apply_event_mask(
        np.zeros(10, dtype=np.int8), dense_ends, events, INVALID_LABEL
    )
    dense = uniqueness_sample_weights(LabelSpans(starts, dense_ends))
    sparse = uniqueness_sample_weights(LabelSpans(starts, sparse_ends))
    # Two far-apart events overlap nothing: equal weights, both 1.0 after mean-1 scaling
    assert sparse[0] == pytest.approx(sparse[9]) == pytest.approx(1.0)
    # Dense labels overlap heavily: the edge labels are the most unique
    assert dense[0] > dense[5]


# ---------------------------------------------------------------------------
# Serving: the frozen event definition travels with the preprocessing graph
# ---------------------------------------------------------------------------


def _graph(event_sampling: dict | None):
    from src.data.pipeline.stages.features.engineer import FeatureEngineer
    from src.inference.preprocessing_graph import PreprocessingGraph

    engineer = FeatureEngineer(timeframe="5min", enable_mtf=False, enable_wavelets=False)
    pipeline: dict = engineer.pipeline_record("5min")
    if event_sampling is not None:
        pipeline["event_sampling"] = event_sampling
    return PreprocessingGraph.from_feature_pipeline(pipeline, feature_columns=["rsi_14"])


def test_graph_without_event_sampling_keeps_its_serialized_form_and_flags_nothing() -> None:
    graph = _graph(None)
    assert "event_sampling" not in graph.to_dict()  # same keys (and hash) as before
    raw = make_intraday_ohlcv(300)
    assert graph.event_flags(raw, raw.index) is None


def test_graph_round_trips_event_sampling_and_replays_the_training_mask(tmp_path: Path) -> None:
    from src.inference.preprocessing_graph import PreprocessingGraph

    raw = make_intraday_ohlcv(1500, seed=12)
    threshold = 1.5 * float(log_returns(raw["close"]).std())
    graph = _graph({"method": "cusum", "threshold": threshold})
    path = tmp_path / "graph.json"
    graph.save(path)
    loaded = PreprocessingGraph.load(path)
    assert loaded.config.event_sampling == {"method": "cusum", "threshold": threshold}
    assert loaded.config.config_hash == loaded.config.compute_hash()  # hash covers the spec

    expected = EventSamplingSpec("cusum", threshold).mask(raw["close"])
    flags = loaded.event_flags(raw, raw.index)
    assert flags is not None and flags.dtype == bool
    np.testing.assert_array_equal(flags, expected)
    # timestamps outside the supplied history are not events
    stamps = raw.index[[10, 20]].append(pd.DatetimeIndex(["2030-01-01"]))
    outside = loaded.event_flags(raw, stamps)
    assert outside is not None and not outside[-1]
