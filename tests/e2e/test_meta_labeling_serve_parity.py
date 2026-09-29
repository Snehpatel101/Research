"""
Meta-labeling train/serve parity on held-out bars.

Trains a meta-labeling system (xgboost primary + logistic bet filter), then checks
the deployed MetaLabelingBundle against what training computed:

1. Served on the training history, P(bet pays off) on every validation bar equals
   the trained meta-model applied to the validation meta features training built
   (``X_meta_val``), and the traded bars are the same.
2. Served on only the last 1000 bars (as the docs example serves), bars after the
   training range get (nearly) the same P(win) and trade decisions as with full
   history. With OBV as a running sum from the first bar, the short window shifted
   the filter's logit by ~+1.5 and it traded 891 of 917 bars at P(win) ~0.9.

The primary is the model refit on all training rows, so on training bars its
probabilities are in-sample; only held-out bars are compared.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import src.inference.meta_labeling_bundle as meta_labeling_bundle
from src.config.experiment import ExperimentConfig
from src.factory import MLFactory
from src.inference import load_deploy_artifact
from src.inference.meta_labeling_bundle import MetaLabelingBundle, primary_sides
from src.models.training.training_ops import TrainingOpsMixin
from src.models.training.unified_orchestrator import UnifiedTrainingOrchestrator
from tests.helpers import make_intraday_ohlcv

pytestmark = pytest.mark.slow

N_ROWS = 4000
SERVE_WINDOW = 1000
# Features with long warmups (SMA-200 regimes, expanding wavelet z-scores) still
# depend on the window start for the first few hundred bars; compare after them.
WINDOW_WARMUP = 300


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Train + deploy once, capturing training's validation meta features and rows."""
    tmp = tmp_path_factory.mktemp("meta_parity")
    data_path = tmp / "bars_5min.parquet"
    make_intraday_ohlcv(N_ROWS, seed=7).to_parquet(data_path)

    cfg = ExperimentConfig(name="meta_parity", output_dir=tmp / "run")
    cfg.random_seed = 42
    cfg.data.symbol = "MES"
    cfg.data.data_path = data_path
    cfg.data.mtf.enabled = False
    cfg.training.models = ["xgboost"]
    cfg.training.horizons = [5]
    cfg.training.n_splits = 3
    cfg.training.optuna.n_trials = 0
    cfg.training.training_mode = "meta_labeling"
    cfg.training.meta_labeling.meta_model = "logistic"
    cfg.training.meta_labeling.threshold = 0.5
    cfg.evaluation.run_backtest = False

    captured: dict[str, Any] = {"meta_features": []}
    build = meta_labeling_bundle.build_meta_features
    prepare = UnifiedTrainingOrchestrator._prepare_for_horizon
    model_input = TrainingOpsMixin._primary_model_input

    def recording_build(inputs: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        out = build(inputs, probabilities)
        captured["meta_features"].append(out)
        return out

    def recording_prepare(self: Any, df: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
        captured["df"] = df
        return prepare(self, df, *args, **kwargs)

    def recording_input(trainer: Any, prepared: Any, X: np.ndarray) -> np.ndarray:
        out = model_input(trainer, prepared, X)
        if X is prepared.X_val:
            captured.update(trainer=trainer, prepared=prepared, val_input=out)
        return out

    with pytest.MonkeyPatch.context() as mp:
        # Training imports build_meta_features from the module at call time
        mp.setattr(meta_labeling_bundle, "build_meta_features", recording_build)
        mp.setattr(UnifiedTrainingOrchestrator, "_prepare_for_horizon", recording_prepare)
        mp.setattr(TrainingOpsMixin, "_primary_model_input", staticmethod(recording_input))
        result = MLFactory(cfg, verbose=0).run()

    assert result.deploy_path is not None and result.training_result is not None
    (model_result,) = result.training_result.model_results.values()
    prepared = captured["prepared"]
    bundle = load_deploy_artifact(result.deploy_path, horizon=5)
    assert isinstance(bundle, MetaLabelingBundle)
    return {
        "bundle": bundle,
        "raw": pd.read_parquet(data_path),
        "meta_model": model_result.mode_artifacts["meta_model"],
        "threshold": model_result.mode_artifacts["threshold"],
        # Stage 3 builds train features, then validation features
        "X_meta_val": captured["meta_features"][-1],
        "val_directions": captured["trainer"]
        .model.predict(captured["val_input"])
        .class_predictions,
        "val_rows": pd.DatetimeIndex(captured["df"].index[prepared.val_indices]),
    }


def _served(bundle: MetaLabelingBundle, raw: pd.DataFrame) -> pd.DataFrame:
    meta = bundle.predict_meta(raw, calibrate=False)
    return pd.DataFrame(
        {"p_win": meta.meta_probabilities, "direction": meta.directions, "trade": meta.trade_mask},
        index=meta.timestamps,
    )


def test_served_meta_probability_equals_training_on_validation_bars(
    trained: dict[str, Any],
) -> None:
    bundle, raw, rows = trained["bundle"], trained["raw"], trained["val_rows"]
    assert len(trained["X_meta_val"]) == len(rows) > 100

    expected_p = trained["meta_model"].predict_proba(trained["X_meta_val"])[:, 1]
    expected_trade = primary_sides(trained["val_directions"]) & (expected_p >= trained["threshold"])
    assert bundle.threshold == trained["threshold"]

    served = _served(bundle, raw).reindex(rows)
    assert served.notna().all().all(), "validation bars missing from the served output"
    np.testing.assert_allclose(served["p_win"].to_numpy(), expected_p, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(served["direction"].to_numpy(), trained["val_directions"])
    np.testing.assert_array_equal(served["trade"].to_numpy(dtype=bool), expected_trade)

    # The default serving path (calibrated primary) takes exactly the same trades
    pred = bundle.predict_from_raw(raw)
    traded = pd.Series(pred.metadata["trade_mask"], index=pred.metadata["timestamps"])
    np.testing.assert_array_equal(traded.reindex(rows).to_numpy(dtype=bool), expected_trade)


def test_short_serving_window_matches_full_history_on_held_out_bars(
    trained: dict[str, Any],
) -> None:
    bundle, raw = trained["bundle"], trained["raw"]
    full = _served(bundle, raw)
    window = _served(bundle, raw.iloc[-SERVE_WINDOW:])
    # Held out: after the validation split, past the window's feature warmup
    bars = window.index[WINDOW_WARMUP:]
    bars = bars[bars > trained["val_rows"][-1]]
    assert len(bars) > 200

    dp = np.abs(window.loc[bars, "p_win"].to_numpy() - full.loc[bars, "p_win"].to_numpy())
    same_trade = window.loc[bars, "trade"].to_numpy() == full.loc[bars, "trade"].to_numpy()
    assert dp.mean() < 0.03, f"mean |dP(win)| {dp.mean():.3f} (max {dp.max():.3f})"
    assert same_trade.mean() > 0.9, f"trade decisions agree on {same_trade.mean():.0%}"
