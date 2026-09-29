"""Raw-bar sanitizing: one canonical step shared by training and inference.

MLFactory._load_raw_bars is the only ingest since the runner's cleaning stage was
removed, so duplicated timestamps, shuffled rows, NaN/negative prices, high < low
and tz-aware timestamps must be handled there - and identically at serving time
(PreprocessingGraph), or train and serve diverge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config.experiment import ExperimentConfig
from src.data.pipeline.stages.clean.sanitize import sanitize_bars, to_naive_utc
from src.factory import MLFactory
from src.inference.preprocessing_graph import PreprocessingGraph

N = 200


def _clean(n: int = N, tz: str | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-03-04 14:30", periods=n, freq="5min", tz=tz)
    close = 100 + np.cumsum(rng.normal(0, 0.2, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 0.1,
            "low": np.minimum(open_, close) - 0.1,
            "close": close,
            "volume": rng.integers(10, 100, n).astype(float),
        },
        index=pd.DatetimeIndex(idx, name="datetime"),
    )


def _messy() -> pd.DataFrame:
    df = _clean()
    df.loc[df.index[10], "close"] = np.nan  # NaN price
    df.loc[df.index[20], "open"] = np.inf  # inf price
    df.loc[df.index[30], "close"] = -5.0  # negative price
    df.loc[df.index[40], "low"] = df.loc[df.index[40], "high"] + 1.0  # low above high
    df.loc[df.index[50], "volume"] = -7.0  # negative volume
    dup = df.iloc[[60]].copy()
    dup["close"] = df.iloc[60]["close"] + 0.05  # a later, corrected print of the same bar
    dup["high"] = dup["high"] + 0.5
    df = pd.concat([df, dup])
    return df.sample(frac=1.0, random_state=1)  # shuffled


class TestSanitizeBars:
    def test_clean_frame_passes_through_unchanged(self) -> None:
        df = _clean()
        out, report = sanitize_bars(df)
        assert not report.changed
        pd.testing.assert_frame_equal(out, df, check_freq=False)

    def test_messy_frame(self) -> None:
        raw = _messy()
        out, report = sanitize_bars(raw)

        assert out.index.is_monotonic_increasing and out.index.is_unique
        assert out.index.name == "datetime"
        assert report.reordered
        assert report.non_finite_prices == 2  # NaN and inf rows
        assert report.non_positive_prices == 1
        assert report.duplicate_timestamps == 1
        assert report.high_low_fixed >= 1
        assert report.bad_volume == 1
        assert report.rows_out == N - 3  # three price rows dropped, duplicate collapsed
        assert np.isfinite(out.to_numpy()).all()
        assert (out[["open", "high", "low", "close"]] > 0).all().all()
        assert (out["high"] >= out[["open", "close", "low"]].max(axis=1)).all()
        assert (out["low"] <= out[["open", "close", "high"]].min(axis=1)).all()
        assert (out["volume"] >= 0).all()
        # the row that comes last in the input wins the duplicated timestamp
        ts = raw.index[(raw.index.duplicated(keep=False))][0]
        last_close = raw.loc[ts, "close"].iloc[-1]
        assert out.loc[ts, "close"] == pytest.approx(last_close)

    def test_tz_aware_index_becomes_naive_utc(self) -> None:
        df = _clean(tz="America/New_York")
        out, report = sanitize_bars(df)
        assert report.tz_converted
        assert out.index.tz is None
        assert out.index[0] == pd.Timestamp("2024-03-04 19:30")  # 14:30 EST = 19:30 UTC

    def test_all_bars_invalid_raises(self) -> None:
        df = _clean(20)
        df["close"] = -1.0
        with pytest.raises(ValueError, match="No valid bars"):
            sanitize_bars(df)

    def test_missing_columns_raise(self) -> None:
        with pytest.raises(ValueError, match="Missing required OHLCV"):
            sanitize_bars(_clean().drop(columns=["volume"]))

    def test_to_naive_utc(self) -> None:
        assert to_naive_utc("2024-03-04 09:30-05:00") == pd.Timestamp("2024-03-04 14:30")
        assert to_naive_utc("2024-03-04") == pd.Timestamp("2024-03-04")


class TestTrainServeParity:
    def _factory(self, path: Path, tmp_path: Path, **data) -> MLFactory:
        cfg = ExperimentConfig(output_dir=tmp_path / "out")
        cfg.data.data_path = path
        for key, value in data.items():
            setattr(cfg.data, key, value)
        return MLFactory(cfg, enable_checkpoints=False)

    def test_messy_parquet_loads_clean(self, tmp_path: Path) -> None:
        path = tmp_path / "messy.parquet"
        _messy().to_parquet(path)
        raw, bar_tf = self._factory(path, tmp_path)._load_raw_bars()
        assert bar_tf == "5min"
        assert raw.index.is_monotonic_increasing and raw.index.is_unique
        assert np.isfinite(raw.to_numpy()).all()

    def test_tz_aware_with_start_date(self, tmp_path: Path) -> None:
        path = tmp_path / "tz.parquet"
        _clean(tz="America/New_York").to_parquet(path)
        # 14:30 EST first bar = 19:30 UTC; the start date is read as UTC too
        factory = self._factory(path, tmp_path, start_date="2024-03-04 20:00")
        raw, _ = factory._load_raw_bars()
        assert raw.index.tz is None
        assert raw.index[0] >= pd.Timestamp("2024-03-04 20:00")
        assert len(raw) < N

        aware_start = self._factory(path, tmp_path, start_date="2024-03-04 15:00-05:00")
        raw2, _ = aware_start._load_raw_bars()  # 20:00 UTC, same instant
        assert raw2.index[0] == raw.index[0]

    def test_inference_graph_sanitizes_like_training(self, tmp_path: Path) -> None:
        messy = _messy()
        path = tmp_path / "messy.parquet"
        messy.to_parquet(path)
        train_raw, _ = self._factory(path, tmp_path)._load_raw_bars()

        served = PreprocessingGraph._to_datetime_column(messy)
        served = served.set_index("datetime")[list(train_raw.columns)]
        pd.testing.assert_frame_equal(served, train_raw, check_freq=False)

    def test_inference_graph_accepts_tz_aware_bars(self) -> None:
        aware = _clean(tz="America/New_York")
        served = PreprocessingGraph._to_datetime_column(aware)
        assert served["datetime"].dt.tz is None
        assert served["datetime"].iloc[0] == pd.Timestamp("2024-03-04 19:30")
