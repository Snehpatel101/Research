"""
End-to-end: every horizon trains, is scored and ships on its OWN labels.

Before the fix every horizon's model was prepared with the default ``label``
column (a copy of the first horizon's labels), so ``xgboost_h20`` was an h5
model shipped as h20. This run uses two horizons whose triple-barrier labels
differ and checks, per horizon:

- the PreparedData the final model trained on has ``y == label_h{h}`` on its
  rows, and label spans from ``label_end_h{h}``;
- the OOF predictions carry ``label_h{h}`` as ``y_true``;
- the bundle and the deploy manifest entry carry that horizon;
- the backtest replays the first horizon (with its barriers) and says so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.config.experiment import ExperimentConfig
from tests.helpers import make_intraday_ohlcv

pytestmark = pytest.mark.slow

HORIZONS = [5, 20]


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from src.factory import MLFactory
    from src.models.training.training_ops import TrainingOpsMixin

    base = tmp_path_factory.mktemp("multi_horizon")
    data_path = base / "mes_5min.parquet"
    make_intraday_ohlcv(2500, seed=11).to_parquet(data_path)

    cfg = ExperimentConfig(name="multi_horizon_e2e", random_seed=42, verbose=0)
    cfg.output_dir = base / cfg.run_id
    cfg.data.symbol = "MES"
    cfg.data.data_path = data_path
    cfg.data.mtf.enabled = False
    cfg.training.models = ["xgboost"]
    cfg.training.horizons = list(HORIZONS)
    cfg.training.training_mode = "standard"
    cfg.training.n_splits = 2
    cfg.training.purge_bars = 50  # >= max_bars of the H20 label (MES table)
    cfg.training.embargo_bars = 30
    cfg.training.build_ensemble = False
    cfg.training.optuna.n_trials = 0
    cfg.evaluation.run_backtest = True
    cfg.bundling.create_bundle = True
    cfg.bundling.deploy_artifact = True

    trained: list[tuple[str, Any, int]] = []
    original = TrainingOpsMixin._train_single_model

    def spy(self: Any, model_name: str, prepared: Any, horizon: int) -> Any:
        trained.append((model_name, prepared, horizon))
        return original(self, model_name, prepared, horizon)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(TrainingOpsMixin, "_train_single_model", spy)
        result = MLFactory(cfg, verbose=0).run()
    assert result.success
    assert result.output_dir is not None
    df = pd.read_parquet(result.output_dir / "cache" / "data_pipeline.parquet")
    return {"result": result, "trained": trained, "df": df}


def test_horizon_labels_differ(run: dict[str, Any]) -> None:
    df = run["df"]
    assert (df["label_h5"] != df["label_h20"]).any(), "test needs distinct labels"


def test_each_model_trains_on_its_horizons_labels(run: dict[str, Any]) -> None:
    df = run["df"]
    by_horizon = {h: prepared for _name, prepared, h in run["trained"]}
    assert sorted(by_horizon) == HORIZONS
    for horizon, prepared in by_horizon.items():
        labels = df[f"label_h{horizon}"].to_numpy()
        ends = df[f"label_end_h{horizon}"].to_numpy()
        np.testing.assert_array_equal(prepared.y_train, labels[prepared.train_indices])
        np.testing.assert_array_equal(prepared.y_val, labels[prepared.val_indices])
        spans = prepared.label_spans("train")
        assert spans is not None
        np.testing.assert_array_equal(spans.ends, ends[prepared.train_indices])
        other = HORIZONS[1 - HORIZONS.index(horizon)]
        assert (prepared.y_train != df[f"label_h{other}"].to_numpy()[prepared.train_indices]).any()


def test_oof_targets_are_the_horizons_labels(run: dict[str, Any]) -> None:
    df = run["df"]
    results = run["result"].training_result.model_results
    for horizon in HORIZONS:
        mr = results[f"xgboost_h{horizon}"]
        assert mr.horizon == horizon
        oof = mr.oof_prediction
        assert oof is not None
        rows = oof.original_indices
        np.testing.assert_array_equal(
            oof.predictions["y_true"].to_numpy()[rows], df[f"label_h{horizon}"].to_numpy()[rows]
        )


def test_bundles_and_manifest_are_per_horizon(run: dict[str, Any]) -> None:
    result = run["result"]
    out = Path(result.output_dir)
    for horizon in HORIZONS:
        with open(out / "bundles" / f"xgboost_h{horizon}" / "metadata.json") as f:
            assert json.load(f)["horizon"] == horizon
    manifest = json.loads((out / "deploy" / "manifest.json").read_text())
    horizons = {int(h): entry for h, entry in manifest["horizons"].items()}
    assert sorted(horizons) == HORIZONS
    for horizon, entry in horizons.items():
        paths = [e["bundle_path"] for e in entry["entries"]]
        assert paths and all(p.endswith(f"_h{horizon}") for p in paths)


def test_backtest_is_the_primary_horizon(run: dict[str, Any]) -> None:
    metrics = run["result"].backtest_metrics
    assert metrics["horizon"] == HORIZONS[0]
    assert metrics["strategy"] == f"oof:xgboost_h{HORIZONS[0]}"
    assert run["result"].best_model == f"xgboost_h{HORIZONS[0]}"
