"""End-to-end CLI tests: every `ml` command on MLFactory, on tiny synthetic bars.

One pipeline sits behind the CLI (raw OHLCV -> FeatureEngineer -> triple-barrier
labels with label spans -> models). Each test runs a command in-process on
~3000 synthetic 5-minute bars with the fastest settings (xgboost / logistic,
no Optuna, no MTF) and asserts exit code 0 and the outputs it promises.

Run with ``pytest -m slow tests/test_cli_e2e.py`` (each command re-computes the
features, so the module takes a few minutes).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from src.cli.unified_cli import app

pytestmark = pytest.mark.slow

N_ROWS = 3000
HORIZON = 5

# Small-data settings shared by every command: 2 CV folds, gaps sized for 3000 bars
FAST = ["--no-mtf"]
FAST_RUN = [*FAST, "--n-splits", "2", "--purge-bars", "15", "--embargo-bars", "30"]


def _make_synthetic_ohlcv(path: Path, n_rows: int = N_ROWS, seed: int = 7) -> None:
    """Synthetic 5-min OHLCV random walk (same generator as scripts/mix_match.py)."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-02 09:30", periods=n_rows, freq="5min")
    close = 5000.0 + np.cumsum(rng.normal(0, 2.0, n_rows))
    open_ = np.roll(close, 1) + rng.normal(0, 0.5, n_rows)
    open_[0] = close[0]
    eps = np.abs(rng.normal(0, 0.5, n_rows)) + 0.25
    df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + eps,
            "low": np.minimum(open_, close) - eps,
            "close": close,
            "volume": rng.randint(100, 5000, n_rows).astype(float),
        },
        index=idx,
    )
    df.index.name = "datetime"
    df.to_parquet(path)


@pytest.fixture(scope="module")
def data_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("cli_e2e_data") / "mes_5min.parquet"
    _make_synthetic_ohlcv(path)
    return path


def _invoke(*args: str):
    result = CliRunner().invoke(app, [str(a) for a in args])
    assert result.exit_code == 0, f"`ml {' '.join(args)}` failed:\n{result.output}"
    return result


def _only_run_dir(output_dir: Path) -> Path:
    runs = [p for p in output_dir.iterdir() if p.is_dir()]
    assert len(runs) == 1, f"expected exactly one run directory in {output_dir}, got {runs}"
    return runs[0]


class TestData:
    def test_data_writes_features_and_labels(self, data_path: Path, tmp_path: Path) -> None:
        _invoke("data", "-d", data_path, "--horizons", HORIZON, *FAST, "-o", tmp_path)

        run_dir = _only_run_dir(tmp_path)
        frame = pd.read_parquet(run_dir / "features_labels.parquet")
        assert isinstance(frame.index, pd.DatetimeIndex)
        assert f"label_h{HORIZON}" in frame.columns
        assert f"label_end_h{HORIZON}" in frame.columns  # label spans for purging
        assert frame.shape[1] > 100  # engineered features, not just OHLCV
        assert set(frame[f"label_h{HORIZON}"].unique()) <= {-99, -1, 0, 1}


class TestRun:
    def test_run_single_model_with_backtest_and_deploy(
        self, data_path: Path, tmp_path: Path
    ) -> None:
        result = _invoke(
            "run",
            "-d",
            data_path,
            "--horizons",
            HORIZON,
            *FAST_RUN,
            "--backtest",
            "-o",
            tmp_path,
        )
        assert "PIPELINE COMPLETED SUCCESSFULLY" in result.output
        assert "Best model: xgboost_h5" in result.output

        run_dir = _only_run_dir(tmp_path)
        assert (run_dir / "experiment_config.yaml").exists()
        assert (run_dir / "bundles" / f"xgboost_h{HORIZON}").exists()
        manifest = json.loads((run_dir / "deploy" / "manifest.json").read_text())
        assert manifest  # deploy artifact written
        backtest = json.loads((run_dir / "cache" / "evaluation.json").read_text())
        assert backtest["total_trades"] > 0  # the barrier-aligned backtest ran

    def test_status_then_resume(self, data_path: Path, tmp_path: Path) -> None:
        _invoke(
            "run", "-d", data_path, "--horizons", HORIZON, *FAST_RUN, "--no-deploy", "-o", tmp_path
        )
        run_dir = _only_run_dir(tmp_path)

        status = _invoke("status", run_dir)
        for stage in ("data_pipeline", "training", "evaluation", "bundling"):
            assert stage in status.output
        assert "All stages completed" in status.output

        resumed = _invoke("run", "--resume", run_dir)
        assert "PIPELINE COMPLETED SUCCESSFULLY" in resumed.output

    def test_run_ensemble(self, data_path: Path, tmp_path: Path) -> None:
        result = _invoke(
            "run",
            "-d",
            data_path,
            "-m",
            "xgboost,logistic",
            "--build-ensemble",
            "--horizons",
            HORIZON,
            *FAST_RUN,
            "--no-deploy",
            "-o",
            tmp_path,
        )
        assert "PIPELINE COMPLETED SUCCESSFULLY" in result.output
        assert "diversity_score" in result.output  # stacked ensemble was built and scored
        oofs = sorted(p.name for p in _only_run_dir(tmp_path).glob("run_standard_*/oof/*.parquet"))
        assert oofs == [f"logistic_h{HORIZON}_oof.parquet", f"xgboost_h{HORIZON}_oof.parquet"]

    def test_run_walk_forward_training_mode(self, data_path: Path, tmp_path: Path) -> None:
        result = _invoke(
            "run",
            "-d",
            data_path,
            "--horizons",
            HORIZON,
            *FAST_RUN,
            "--training-mode",
            "walk_forward",
            "--no-deploy",
            "-o",
            tmp_path,
        )
        assert "PIPELINE COMPLETED SUCCESSFULLY" in result.output


class TestEvaluation:
    def test_cv_writes_results_and_stacking_dataset(self, data_path: Path, tmp_path: Path) -> None:
        _invoke(
            "cv",
            "-d",
            data_path,
            "-m",
            "xgboost",
            "--horizons",
            HORIZON,
            *FAST,
            "--n-splits",
            "2",
            "--no-feature-selection",
            "-o",
            tmp_path,
        )
        cv_dir = _only_run_dir(tmp_path) / "cv"
        assert (cv_dir / "stacking" / f"stacking_dataset_h{HORIZON}.parquet").exists()
        assert any(cv_dir.glob("*.json"))

    def test_walk_forward_writes_windows_and_predictions(
        self, data_path: Path, tmp_path: Path
    ) -> None:
        _invoke(
            "walk-forward",
            "-d",
            data_path,
            "-m",
            "logistic",
            "--horizons",
            HORIZON,
            *FAST,
            "--n-windows",
            "3",
            "-o",
            tmp_path,
        )
        wf_dir = _only_run_dir(tmp_path) / "walk-forward"
        result = json.loads((wf_dir / f"wf_logistic_h{HORIZON}.json").read_text())
        assert result["n_windows"] == 3
        assert (wf_dir / f"wf_preds_logistic_h{HORIZON}.parquet").exists()
        assert (wf_dir / "walk_forward_summary.csv").exists()

    def test_cpcv_pbo_writes_paths_and_pbo(self, data_path: Path, tmp_path: Path) -> None:
        _invoke(
            "cpcv-pbo",
            "-d",
            data_path,
            "-m",
            "xgboost,logistic",
            "--horizons",
            HORIZON,
            *FAST,
            "--n-groups",
            "4",
            "--pbo-block",
            "1.0",
            "-o",
            tmp_path,
        )
        out = _only_run_dir(tmp_path) / "cpcv-pbo"
        for model in ("xgboost", "logistic"):
            payload = json.loads((out / f"cpcv_{model}_h{HORIZON}.json").read_text())
            assert payload["n_paths"] == 3  # C(4,2)*2/4 CPCV backtest paths
        assert 0.0 <= json.loads((out / f"pbo_h{HORIZON}.json").read_text())["pbo"] <= 1.0
        assert (out / "cpcv_pbo_summary.csv").exists()
