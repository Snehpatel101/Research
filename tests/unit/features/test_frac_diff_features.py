"""Fractional differentiation (AFML ch. 5) features: causality, frozen d, train/serve replay."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config.experiment import ExperimentConfig
from src.data.features.frac_diff import (
    ffd_weights,
    find_min_d,
    frac_diff_ffd,
    resolve_frac_diff_d,
)
from src.data.pipeline.stages.features.engineer import FeatureEngineer
from src.data.pipeline.stages.features.frac_diff_features import (
    add_frac_diff_features,
    frac_diff_feature_name,
)
from src.factory import MLFactory
from src.inference.preprocessing_graph import PreprocessingGraph
from tests.helpers import make_intraday_ohlcv

WINDOW = 40


def _log_price(n: int = 600, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(np.cumsum(rng.normal(0, 0.001, n)) + np.log(5000.0))


# ---------------------------------------------------------------------------
# The FFD primitive: fixed window, causal
# ---------------------------------------------------------------------------


def test_weights_depend_only_on_d_threshold_and_window() -> None:
    w = ffd_weights(0.4, threshold=1e-5, max_window=WINDOW)
    assert len(w) == WINDOW
    assert w[0] == 1.0 and w[1] == pytest.approx(-0.4)
    np.testing.assert_array_equal(w, ffd_weights(0.4, threshold=1e-5, max_window=WINDOW))


def test_fixed_window_value_does_not_depend_on_how_much_history_is_supplied() -> None:
    """A value from a long history equals the value from just its last `window` bars."""
    series = _log_price(600)
    full = frac_diff_ffd(series, d=0.4, threshold=1e-5, max_window=WINDOW)
    for t in (100, 333, 599):
        window_only = series.iloc[t - WINDOW + 1 : t + 1].reset_index(drop=True)
        short = frac_diff_ffd(window_only, d=0.4, threshold=1e-5, max_window=WINDOW)
        assert short.iloc[-1] == pytest.approx(full.iloc[t], rel=1e-12, abs=1e-12)


def test_ffd_is_causal_perturbing_future_bars_leaves_the_past_unchanged() -> None:
    series = _log_price(600)
    base = frac_diff_ffd(series, d=0.35, max_window=WINDOW)
    perturbed = series.copy()
    perturbed.iloc[400:] += 0.5
    changed = frac_diff_ffd(perturbed, d=0.35, max_window=WINDOW)
    np.testing.assert_array_equal(changed.iloc[:400].to_numpy(), base.iloc[:400].to_numpy())
    assert not np.allclose(changed.iloc[400:].dropna(), base.iloc[400:].dropna())


def test_first_window_minus_one_bars_are_warmup_nan() -> None:
    out = frac_diff_ffd(_log_price(200), d=0.3, max_window=WINDOW)
    assert out.iloc[: WINDOW - 1].isna().all()
    assert out.iloc[WINDOW - 1 :].notna().all()


# ---------------------------------------------------------------------------
# The feature: lagged one bar like every other feature
# ---------------------------------------------------------------------------


def _ohlc_frame(n: int = 400) -> pd.DataFrame:
    df = make_intraday_ohlcv(n, seed=2).reset_index()
    return df


def test_feature_at_bar_t_ignores_bar_t_and_everything_after() -> None:
    df = _ohlc_frame()
    args = {"columns": ["close", "open"], "d": 0.4, "max_window": WINDOW, "threshold": 1e-5}
    base = add_frac_diff_features(df.copy(), {}, **args)

    t = 300
    tampered = df.copy()
    tampered.loc[t:, ["open", "high", "low", "close"]] *= 1.2
    out = add_frac_diff_features(tampered, {}, **args)

    for column in ("close", "open"):
        name = frac_diff_feature_name(column)
        np.testing.assert_array_equal(
            out[name].iloc[: t + 1].to_numpy(), base[name].iloc[: t + 1].to_numpy()
        )
        assert out[name].iloc[t + 1 :].notna().any()
        assert not np.allclose(out[name].iloc[t + 1 :].dropna(), base[name].iloc[t + 1 :].dropna())


def test_feature_rejects_non_price_columns() -> None:
    with pytest.raises(ValueError, match="frac_diff columns"):
        add_frac_diff_features(_ohlc_frame(), {}, ["volume"], 0.4, WINDOW, 1e-5)


# ---------------------------------------------------------------------------
# FeatureEngineer: opt-in, d frozen in the spec, replayed by from_spec
# ---------------------------------------------------------------------------


def _engineer(**kwargs: object) -> FeatureEngineer:
    return FeatureEngineer(
        timeframe="5min",
        enable_mtf=False,
        enable_wavelets=False,
        frac_diff_window=WINDOW,
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.fixture(scope="module")
def raw_bars() -> pd.DataFrame:
    return make_intraday_ohlcv(700, seed=8)


def test_disabled_by_default_adds_no_columns_and_spec_says_so(raw_bars: pd.DataFrame) -> None:
    engineer = _engineer()
    assert engineer.to_spec()["frac_diff_columns"] == []
    features, _ = engineer.compute_features(raw_bars.reset_index())
    assert not [c for c in features.columns if c.startswith("ffd_")]


def test_enabled_adds_ffd_columns_and_records_d_in_the_spec(raw_bars: pd.DataFrame) -> None:
    engineer = _engineer(frac_diff_columns=["close", "high"], frac_diff_d=0.37)
    spec = engineer.to_spec()
    assert spec["frac_diff_d"] == 0.37
    assert spec["frac_diff_columns"] == ["close", "high"]
    assert spec["frac_diff_window"] == WINDOW

    features, _ = engineer.compute_features(raw_bars.reset_index())
    assert {"ffd_log_close", "ffd_log_high"} <= set(features.columns)
    assert "ffd_log_open" not in features.columns


def test_from_spec_replays_the_same_d_and_reproduces_the_values(raw_bars: pd.DataFrame) -> None:
    engineer = _engineer(frac_diff_columns=["close"], frac_diff_d=0.37)
    trained, _ = engineer.compute_features(raw_bars.reset_index())

    replayed_engineer = FeatureEngineer.from_spec(engineer.to_spec())
    assert replayed_engineer.frac_diff_d == 0.37
    assert replayed_engineer.to_spec() == engineer.to_spec()
    replayed, _ = replayed_engineer.compute_features(raw_bars.reset_index())
    np.testing.assert_array_equal(
        replayed["ffd_log_close"].to_numpy(), trained["ffd_log_close"].to_numpy()
    )


def test_spec_from_before_frac_diff_still_loads() -> None:
    spec = _engineer().to_spec()
    for key in [k for k in spec if k.startswith("frac_diff")]:
        del spec[key]
    assert FeatureEngineer.from_spec(spec).frac_diff_columns == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frac_diff_columns": ["close"]}, "resolved frac_diff_d"),
        ({"frac_diff_columns": ["close"], "frac_diff_d": 0.0}, "resolved frac_diff_d"),
        ({"frac_diff_columns": ["close"], "frac_diff_d": 1.5}, "resolved frac_diff_d"),
        ({"frac_diff_columns": ["volume"], "frac_diff_d": 0.4}, "must be among"),
    ],
)
def test_engineer_rejects_incomplete_frac_diff_settings(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _engineer(**kwargs)


def test_preprocessing_graph_replays_the_recorded_d_at_inference(
    raw_bars: pd.DataFrame, tmp_path: Path
) -> None:
    """Train/serve parity: the graph rebuilds the engineer from the recorded spec."""
    engineer = _engineer(frac_diff_columns=["close", "low"], frac_diff_d=0.42)
    trained, _ = engineer.compute_features(raw_bars.reset_index())
    trained = trained.set_index("datetime")
    columns = ["ffd_log_close", "ffd_log_low"]

    graph = PreprocessingGraph.from_feature_pipeline(
        {"bar_timeframe": "5min", "engineer": engineer.to_spec()}, feature_columns=columns
    )
    path = tmp_path / "graph.json"
    graph.save(path)
    assert '"frac_diff_d": 0.42' in path.read_text()

    served = PreprocessingGraph.load(path).transform(raw_bars, skip_scaling=True)
    common = served.index.intersection(trained.index)
    assert len(common) > 500
    np.testing.assert_allclose(
        served.loc[common, columns].to_numpy(float),
        trained.loc[common, columns].to_numpy(float),
        rtol=1e-9,
        atol=1e-12,
    )


def test_serving_window_shorter_than_training_history_gives_the_same_values(
    raw_bars: pd.DataFrame,
) -> None:
    """Only the last `window` bars matter, so a serving tail matches full-history training."""
    engineer = _engineer(frac_diff_columns=["close"], frac_diff_d=0.4)
    full, _ = engineer.compute_features(raw_bars.reset_index())
    tail_start = 450
    tail, _ = engineer.compute_features(raw_bars.iloc[tail_start:].reset_index())
    # first WINDOW rows of the tail are its warmup
    np.testing.assert_allclose(
        tail["ffd_log_close"].iloc[WINDOW + 5 :].to_numpy(),
        full["ffd_log_close"].iloc[tail_start + WINDOW + 5 :].to_numpy(),
        rtol=1e-9,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# d resolution: 'auto' fitted on TRAIN rows only, frozen; clear error without statsmodels
# ---------------------------------------------------------------------------


def _factory(tmp_path: Path, **frac_diff: object) -> MLFactory:
    cfg = ExperimentConfig()
    cfg.verbose = 0
    cfg.output_dir = tmp_path / "out" / cfg.run_id
    cfg.data.symbol = "MES"
    cfg.data.features.frac_diff.enabled = True
    cfg.data.features.frac_diff.window = WINDOW
    for key, value in frac_diff.items():
        setattr(cfg.data.features.frac_diff, key, value)
    return MLFactory(cfg, verbose=0, enable_checkpoints=False)


def test_explicit_d_is_frozen_verbatim(tmp_path: Path) -> None:
    kwargs = _factory(tmp_path, d=0.33)._resolve_frac_diff(make_intraday_ohlcv(1000))
    assert kwargs["frac_diff_d"] == 0.33
    assert kwargs["frac_diff_columns"] == ["close", "open", "high", "low"]
    assert kwargs["frac_diff_window"] == WINDOW


def test_disabled_resolves_to_no_engineer_settings(tmp_path: Path) -> None:
    factory = _factory(tmp_path, enabled=False)
    assert factory._resolve_frac_diff(make_intraday_ohlcv(1000)) == {}


def test_auto_d_uses_train_rows_only(tmp_path: Path) -> None:
    """Changing validation/test rows leaves d untouched; changing train rows can move it."""
    pytest.importorskip("statsmodels")
    raw = make_intraday_ohlcv(4000, seed=6)
    factory = _factory(tmp_path, d="auto")
    train_end = int(len(raw) * factory.config.data.splits.train_ratio)
    close = raw.columns.get_loc("close")

    base = factory._resolve_frac_diff(raw)["frac_diff_d"]
    assert 0.0 < base <= 1.0

    later = raw.copy()
    rng = np.random.default_rng(3)
    # a strongly mean-reverting (stationary) val/test period would change any d fitted on it
    later.iloc[train_end:, close] = 5000.0 + rng.normal(0, 3.0, len(raw) - train_end)
    assert factory._resolve_frac_diff(later)["frac_diff_d"] == base

    stationary_train = raw.copy()
    stationary_train.iloc[:train_end, close] = 5000.0 + rng.normal(0, 3.0, train_end)
    assert factory._resolve_frac_diff(stationary_train)["frac_diff_d"] < base


def test_find_min_d_matches_between_direct_call_and_resolution() -> None:
    pytest.importorskip("statsmodels")
    series = _log_price(1500, seed=4)
    direct = find_min_d(series, threshold=1e-5, max_window=WINDOW)
    assert resolve_frac_diff_d("auto", series, threshold=1e-5, max_window=WINDOW) == direct


def test_auto_without_statsmodels_raises_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "statsmodels.tsa.stattools", None)
    with pytest.raises(ImportError, match=r"statsmodels.*\[stats\]"):
        resolve_frac_diff_d("auto", _log_price(300), max_window=WINDOW)


def test_explicit_d_needs_no_statsmodels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "statsmodels.tsa.stattools", None)
    assert resolve_frac_diff_d(0.4, _log_price(300)) == 0.4


@pytest.mark.parametrize("bad", [0.0, 1.2, -0.1, "sometimes"])
def test_resolve_rejects_bad_d(bad: object) -> None:
    with pytest.raises(ValueError, match="frac_diff d"):
        resolve_frac_diff_d(bad, _log_price(300))  # type: ignore[arg-type]
