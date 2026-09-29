"""Label spans on the evaluation container and the standalone evaluation commands.

The container carries ``label_end_h{h}`` (row positions at which each label
resolves, produced by MLFactory's labeling step). ``get_label_spans`` turns them
into ``LabelSpans`` so purged k-fold, walk-forward and CPCV purge every training
sample whose label overlaps the test block, on frames with a plain RangeIndex.

The CLI tests run ``ml walk-forward`` / ``ml cpcv-pbo`` on a synthetic container
(the MLFactory data step is replaced by a stub): they pin that each command runs
its evaluator, writes results, and exits 1 when every model fails.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from src.core.container import TimeSeriesDataContainer
from src.core.label_spans import LabelSpans, label_end_column
from src.validation.cv.purged_kfold import PurgedKFold, PurgedKFoldConfig

HORIZON = 5
SPAN = 12  # every label resolves SPAN bars after its own bar
END_COL = label_end_column(f"label_h{HORIZON}")


def _split_frame(n: int, seed: int = 0, symbol: str = "MES") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.1, n))
    return pd.DataFrame(
        {
            "symbol": symbol,
            "close": close,
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "f3": rng.normal(size=n),
            f"label_h{HORIZON}": rng.choice([-1, 0, 1], size=n),
            f"sample_weight_h{HORIZON}": 1.0,
            END_COL: np.minimum(np.arange(n) + SPAN, n - 1),
        }
    )


class TestContainerLabelSpans:
    def test_get_label_spans_rows_and_purge(self) -> None:
        df = _split_frame(300)
        df.loc[[7, 50], f"label_h{HORIZON}"] = -99  # dropped by the container
        c = TimeSeriesDataContainer.from_dataframes(train_df=df, horizon=HORIZON)
        X, y, _ = c.get_sklearn_arrays("train", return_df=True)
        assert isinstance(X.index, pd.RangeIndex)
        spans = c.get_label_spans("train")
        assert isinstance(spans, LabelSpans)
        assert len(spans) == len(X) == 298
        np.testing.assert_array_equal(spans.starts, np.arange(len(X)))
        # The label at kept row 0 ends SPAN bars later in the original frame;
        # row 7 was dropped before that, so the end shifts one row back.
        assert spans.ends[0] == SPAN - 1
        # Labels near the end resolve at the last row
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

    def test_no_label_end_column_returns_none(self) -> None:
        df = _split_frame(100).drop(columns=[END_COL])
        c = TimeSeriesDataContainer.from_dataframes(train_df=df, horizon=HORIZON)
        assert c.get_label_spans("train") is None

    def test_label_end_column_is_not_a_feature(self) -> None:
        c = TimeSeriesDataContainer.from_dataframes(train_df=_split_frame(50), horizon=HORIZON)
        assert c.feature_columns == ["f1", "f2", "f3"]


class TestEvaluationContainer:
    """MLFactory-prepared tabular data -> container the evaluators read."""

    def test_train_split_aligned_with_source_rows(self) -> None:
        from src.data.adapters import PreparedData
        from src.validation.cv.evaluation_data import build_evaluation_container

        n = 200
        rng = np.random.default_rng(3)
        close = pd.Series(100.0 + np.arange(n, dtype=float))
        rows = np.arange(10, 150)  # train rows of the source frame (first 10 warm-up rows gone)
        label_ends = np.minimum(np.arange(n) + SPAN, n - 1)
        prepared = PreparedData(
            X_train=rng.normal(size=(len(rows), 3)),
            y_train=rng.choice([-1, 0, 1], size=len(rows)),
            X_val=rng.normal(size=(20, 3)),
            y_val=rng.choice([-1, 0, 1], size=20),
            model_name="xgboost",
            data_rank=2,
            feature_names=["f1", "f2", "f3"],
            train_indices=rows,
            label_end_positions=label_ends,
        )
        c = build_evaluation_container(prepared, close, "MES", HORIZON)

        split = c.get_split("train")
        np.testing.assert_array_equal(split.df["close"].to_numpy(), close.to_numpy()[rows])
        assert (split.df["symbol"] == "MES").all()
        assert c.feature_columns == ["f1", "f2", "f3"]
        spans = c.get_label_spans("train")
        assert spans is not None
        # Sample k is source row rows[k]; its label ends at source row rows[k]+SPAN,
        # i.e. sample k+SPAN (rows are contiguous), clamped to the last train sample
        expected = np.minimum(np.arange(len(rows)) + SPAN, len(rows) - 1)
        np.testing.assert_array_equal(spans.ends, expected)

    def test_rejects_sequence_data(self) -> None:
        from src.data.adapters import PreparedData
        from src.validation.cv.evaluation_data import build_evaluation_container

        prepared = PreparedData(
            X_train=np.zeros((5, 4, 3)),
            y_train=np.zeros(5),
            X_val=np.zeros((2, 4, 3)),
            y_val=np.zeros(2),
            model_name="lstm",
            data_rank=3,
        )
        with pytest.raises(ValueError, match="tabular"):
            build_evaluation_container(prepared, pd.Series(np.arange(10.0)), "MES", HORIZON)


class TestEvaluationCliExitCodes:
    """walk-forward / cpcv-pbo run on containers and fail loudly when nothing ran."""

    @pytest.fixture
    def stub_data(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """Replace the MLFactory data step by a synthetic container."""
        import src.cli.commands.evaluate as evaluate

        container = TimeSeriesDataContainer.from_dataframes(
            train_df=_split_frame(600), horizon=HORIZON
        )

        def _load(**kwargs):
            results = tmp_path / kwargs["command"]
            results.mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(cv_gaps=(SPAN, 5)), {HORIZON: container}, results

        monkeypatch.setattr(evaluate, "_load_evaluation_data", _load)
        return tmp_path

    def _args(self, command: str, tmp_path: Path, *extra: str) -> list[str]:
        return [
            command,
            "-d",
            str(tmp_path / "unused.parquet"),
            "--models",
            "logistic",
            "--horizons",
            str(HORIZON),
            *extra,
        ]

    def test_walk_forward_runs_on_rangeindex_container(self, stub_data: Path) -> None:
        from src.cli.unified_cli import app

        result = CliRunner().invoke(app, self._args("walk-forward", stub_data, "--n-windows", "2"))
        assert result.exit_code == 0, result.output
        assert (stub_data / "walk-forward" / f"wf_logistic_h{HORIZON}.json").exists()

    def test_walk_forward_exits_1_when_every_model_fails(
        self, stub_data: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.cli.commands.evaluate as evaluate
        from src.cli.unified_cli import app

        def _boom(**_kwargs: object) -> None:
            raise RuntimeError("model failed")

        monkeypatch.setattr(evaluate, "_run_walk_forward_for_model", _boom)
        result = CliRunner().invoke(app, self._args("walk-forward", stub_data))
        assert result.exit_code == 1, result.output

    def test_cpcv_pbo_runs_on_rangeindex_container(self, stub_data: Path) -> None:
        from src.cli.unified_cli import app

        result = CliRunner().invoke(
            app,
            self._args("cpcv-pbo", stub_data, "--n-groups", "4", "--n-test-groups", "1"),
        )
        assert result.exit_code == 0, result.output
        assert (stub_data / "cpcv-pbo" / f"cpcv_logistic_h{HORIZON}.json").exists()

    def test_cpcv_pbo_exits_1_when_every_model_fails(
        self, stub_data: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.cli.commands.evaluate as evaluate
        from src.cli.unified_cli import app

        def _boom(**_kwargs: object) -> None:
            raise RuntimeError("model failed")

        monkeypatch.setattr(evaluate, "_run_cpcv_for_model", _boom)
        result = CliRunner().invoke(app, self._args("cpcv-pbo", stub_data))
        assert result.exit_code == 1, result.output
