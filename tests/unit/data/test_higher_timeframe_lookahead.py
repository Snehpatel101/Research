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
        last_completed_close = raw.loc[ts - pd.Timedelta(minutes=1), "close"]
        assert row["close"] == last_completed_close
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
