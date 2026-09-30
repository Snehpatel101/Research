"""
Multi-stream (4D) higher-timeframe data must be lagged one bar.

MLFactory._generate_additional_dfs resamples the raw 1-minute bars to higher
timeframes for transformer models. A higher-TF row stamped T must only hold data
from bars that were COMPLETE before T, never the still-forming bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config.experiment import ExperimentConfig
from src.factory import MLFactory


def _raw_minutes(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="1min", name="datetime")
    close = 100.0 + np.arange(n, dtype=float)  # strictly increasing: close encodes its own time
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 0.2,
            "low": close - 0.7,
            "close": close,
            "volume": np.full(n, 100.0),
        },
        index=idx,
    )


def _factory(tmp_path, models: list[str], timeframes: list[str]) -> MLFactory:
    cfg = ExperimentConfig(output_dir=str(tmp_path / "run"))
    cfg.training.models = models
    cfg.data.mtf.timeframes = timeframes
    return MLFactory(cfg, verbose=0, enable_checkpoints=False)


def test_higher_timeframe_bar_only_contains_completed_data(tmp_path) -> None:
    raw = _raw_minutes()
    dfs = _factory(tmp_path, ["patchtst"], ["15min"])._generate_additional_dfs(raw)

    assert dfs is not None and list(dfs) == ["15min"]
    htf = dfs["15min"]
    assert len(htf) > 0

    for ts, row in htf.iterrows():
        # the last 1-min bar strictly before T closed at T - 1min: nothing at/after T leaks in
        last_completed_close = raw["close"].asof(ts - pd.Timedelta(minutes=1))
        assert row["close"] == last_completed_close, f"bar {ts} holds data from the forming bar"
        assert row["high"] < raw.loc[ts, "high"]


def test_first_higher_timeframe_bar_is_dropped_by_the_lag(tmp_path) -> None:
    raw = _raw_minutes(120)  # 8 complete 15-min bars
    htf = _factory(tmp_path, ["patchtst"], ["15min"])._generate_additional_dfs(raw)["15min"]  # type: ignore[index]

    assert len(htf) == 7
    assert htf.index[0] == raw.index[0] + pd.Timedelta(minutes=15)


def test_no_multi_stream_models_means_no_additional_data(tmp_path) -> None:
    factory = _factory(tmp_path, ["xgboost"], ["15min"])
    assert factory._generate_additional_dfs(_raw_minutes()) is None


@pytest.mark.parametrize("timeframe", ["1h", "60min"])
def test_timeframe_keys_are_normalized(tmp_path, timeframe: str) -> None:
    raw = _raw_minutes(240)
    dfs = _factory(tmp_path, ["patchtst"], [timeframe])._generate_additional_dfs(raw)
    assert dfs is not None and list(dfs) == ["60min"]


def _bars(n: int, freq: str) -> pd.DataFrame:
    raw = _raw_minutes(n * int(pd.Timedelta(freq).total_seconds() // 60))
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return raw.resample(freq, closed="left", label="left").agg(agg).dropna()


def test_streams_not_coarser_than_the_bars_are_dropped(tmp_path, caplog) -> None:
    factory = _factory(tmp_path, ["patchtst"], ["5min", "15min", "60min"])
    bars = _bars(64, "15min")

    with caplog.at_level("WARNING"):
        dfs = factory._generate_additional_dfs(bars, "15min")

    assert dfs is not None and list(dfs) == ["60min"]
    assert factory._multi_stream_timeframes("15min") == ["15min", "60min"]
    assert "['5min']" in caplog.text  # finer than the bars: dropped with a warning
    # A timeframe equal to the bars is the anchor stream itself
    same = _factory(tmp_path, ["patchtst"], ["5min", "15min"])
    assert same._multi_stream_timeframes("5min") == ["5min", "15min"]


def test_serving_builds_the_training_streams_from_the_anchor_bars(tmp_path) -> None:
    from types import SimpleNamespace

    from src.inference.bundle import ModelBundle

    factory = _factory(tmp_path, ["patchtst"], ["5min", "60min"])
    bars = _bars(64, "15min")  # training bars, resampled from finer raw input
    trained = factory._generate_additional_dfs(bars, "15min")
    assert trained is not None

    stub = SimpleNamespace(
        metadata=SimpleNamespace(model_name="patchtst", extra={"mtf_timeframes": list(trained)})
    )
    served = ModelBundle._generate_mtf_dataframes(stub, bars)  # type: ignore[arg-type]
    assert list(served) == list(trained)
    for key, frame in trained.items():
        pd.testing.assert_frame_equal(served[key], frame)
