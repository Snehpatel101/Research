"""Label-span purging on legacy container (RangeIndex) paths.

Phase-1 parquet splits are written with ``index=False`` and the container
resets the index after dropping invalid labels, so every container frame has
a RangeIndex while carrying ``label_end_time_h{h}`` columns. Passing those end
times as ``label_end_times=`` to PurgedKFold/walk-forward/CPCV raised
"label_end_times needs X with a DatetimeIndex", so ``ml walk-forward`` and
``ml cpcv-pbo`` failed every model (and still exited 0), and feature selection
in ``ml train`` crashed. The container now converts the end times into
``LabelSpans`` (row positions) via ``get_label_spans``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from src.core.container import TimeSeriesDataContainer
from src.core.label_spans import LabelSpans
from src.validation.cv.purged_kfold import PurgedKFold, PurgedKFoldConfig

HORIZON = 5
SPAN = 12  # every label resolves SPAN bars after its own bar


def _split_frame(n: int, seed: int = 0, symbol: str = "MES") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt = pd.date_range("2024-01-02 09:30", periods=n, freq="5min")
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.1, n))
    df = pd.DataFrame(
        {
            "datetime": dt,
            "symbol": symbol,
            "close": close,
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "f3": rng.normal(size=n),
            f"label_h{HORIZON}": rng.choice([-1, 0, 1], size=n),
            f"sample_weight_h{HORIZON}": 1.0,
        }
    )
    ends = dt + pd.Timedelta(minutes=5 * SPAN)
    df[f"label_end_time_h{HORIZON}"] = ends
    return df


def _write_phase1_dir(path: Path, df: pd.DataFrame) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    # Phase-1 writes splits without the index
    df.to_parquet(path / "train_scaled.parquet", index=False)
    return path


class TestContainerLabelSpans:
    def test_rangeindex_container_label_end_times_were_unusable(self, tmp_path: Path) -> None:
        """The old call pattern (label_end_times on a RangeIndex X) cannot purge."""
        c = TimeSeriesDataContainer.from_parquet_dir(
            _write_phase1_dir(tmp_path / "d", _split_frame(300)), horizon=HORIZON
        )
        X, y, _ = c.get_sklearn_arrays("train", return_df=True)
        assert isinstance(X.index, pd.RangeIndex)
        cv = PurgedKFold(PurgedKFoldConfig(n_splits=3, purge_bars=0, embargo_bars=0))
        with pytest.raises(ValueError, match="DatetimeIndex"):
            list(cv.split(X, y, label_end_times=c.get_label_end_times("train")))

    def test_get_label_spans_rows_and_purge(self, tmp_path: Path) -> None:
        df = _split_frame(300)
        df.loc[[7, 50], f"label_h{HORIZON}"] = -99  # dropped by the container
        c = TimeSeriesDataContainer.from_parquet_dir(
            _write_phase1_dir(tmp_path / "d", df), horizon=HORIZON
        )
        X, y, _ = c.get_sklearn_arrays("train", return_df=True)
        spans = c.get_label_spans("train")
        assert isinstance(spans, LabelSpans)
        assert len(spans) == len(X) == 298
        np.testing.assert_array_equal(spans.starts, np.arange(len(X)))
        # The label at kept row 0 ends SPAN bars later in the original frame;
        # row 7 was dropped before that, so the end shifts one row back.
        assert spans.ends[0] == SPAN - 1
        # Labels near the end resolve past the last row: clamped to it
        assert spans.ends[-1] == len(X) - 1

        cv = PurgedKFold(PurgedKFoldConfig(n_splits=3, purge_bars=0, embargo_bars=0))
        folds = list(cv.split(X, y, label_spans=spans))
        assert len(folds) == 3
        train_idx, test_idx = folds[1]
        test_lo = int(test_idx.min())
        # Without a purge floor, only the label spans remove rows before the
        # test block: every training label ends before the first test bar.
        before = train_idx[train_idx < test_lo]
        assert spans.ends[before].max() < test_lo
        assert test_lo - 1 not in set(before)

    def test_no_end_time_column_returns_none(self) -> None:
        df = _split_frame(100).drop(columns=[f"label_end_time_h{HORIZON}"])
        c = TimeSeriesDataContainer.from_dataframes(train_df=df, horizon=HORIZON)
        assert c.get_label_spans("train") is None

    def test_stacked_multi_symbol_spans_stay_within_symbol(self) -> None:
        a = _split_frame(60, seed=1, symbol="MES")
        b = _split_frame(60, seed=2, symbol="MGC")
        df = pd.concat([a, b], ignore_index=True)  # datetime restarts at row 60
        c = TimeSeriesDataContainer.from_dataframes(train_df=df, horizon=HORIZON)
        spans = c.get_label_spans("train")
        assert spans is not None
        assert spans.ends[0] == SPAN
        assert spans.ends[59] == 59  # MES label ends are clamped to MES rows
        assert spans.ends[60] == 60 + SPAN
        assert spans.ends[-1] == 119


class TestLegacyCliExitCodes:
    """walk-forward / cpcv-pbo run on containers and fail loudly when nothing ran."""

    @pytest.fixture
    def data_dir(self, tmp_path: Path) -> Path:
        return _write_phase1_dir(tmp_path / "scaled", _split_frame(600))

    def test_walk_forward_runs_on_rangeindex_container(
        self, data_dir: Path, tmp_path: Path
    ) -> None:
        from src.cli.unified_cli import app

        out = tmp_path / "wf"
        result = CliRunner().invoke(
            app,
            [
                "walk-forward",
                "--models",
                "logistic",
                "--horizons",
                str(HORIZON),
                "--n-windows",
                "2",
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (out / f"wf_logistic_h{HORIZON}.json").exists()

    def test_walk_forward_exits_1_when_every_model_fails(
        self, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.cli.commands.evaluate as evaluate
        from src.cli.unified_cli import app

        def _boom(**_kwargs: object) -> None:
            raise RuntimeError("model failed")

        monkeypatch.setattr(evaluate, "_run_walk_forward_for_model", _boom)
        result = CliRunner().invoke(
            app,
            [
                "walk-forward",
                "--models",
                "logistic",
                "--horizons",
                str(HORIZON),
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(tmp_path / "wf"),
            ],
        )
        assert result.exit_code == 1, result.output

    def test_cpcv_pbo_runs_on_rangeindex_container(self, data_dir: Path, tmp_path: Path) -> None:
        from src.cli.unified_cli import app

        out = tmp_path / "cpcv"
        result = CliRunner().invoke(
            app,
            [
                "cpcv-pbo",
                "--models",
                "logistic",
                "--horizons",
                str(HORIZON),
                "--n-groups",
                "4",
                "--n-test-groups",
                "1",
                "--purge-bars",
                "5",
                "--embargo-bars",
                "5",
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (out / f"cpcv_logistic_h{HORIZON}.json").exists()

    def test_cpcv_pbo_exits_1_when_every_model_fails(
        self, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.cli.commands.evaluate as evaluate
        from src.cli.unified_cli import app

        def _boom(**_kwargs: object) -> None:
            raise RuntimeError("model failed")

        monkeypatch.setattr(evaluate, "_run_cpcv_for_model", _boom)
        result = CliRunner().invoke(
            app,
            [
                "cpcv-pbo",
                "--models",
                "logistic",
                "--horizons",
                str(HORIZON),
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(tmp_path / "cpcv"),
            ],
        )
        assert result.exit_code == 1, result.output
