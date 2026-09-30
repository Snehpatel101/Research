"""Serving guards on a real ModelBundle (tiny xgboost on engineered features).

1. Warmup: predict_from_raw scores only bars the training warmup rule keeps, so
   a short window never emits rows whose features depend on where it starts.
2. Engine version: a bundle whose features came from another feature engine
   version (or recorded none) is refused at load unless allow_engine_mismatch.
3. Too little history raises a ValueError naming the raw bars needed (after
   bar-timeframe resampling), not a model error on an empty input.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from src.data.pipeline.stages.features.engineer import FEATURE_ENGINE_VERSION, FeatureEngineer
from src.inference.bundle import ModelBundle
from src.inference.deploy import load_bundle
from src.inference.preprocessing_graph import PREPROCESSING_GRAPH_FILE, PreprocessingGraph
from src.models.registry import ModelRegistry
from tests.helpers import make_intraday_ohlcv

COLUMNS = ["rsi_14", "adx_14", "atr_14", "obv", "price_to_vwap"]
MODEL_CONFIG = {
    "n_estimators": 20,
    "max_depth": 3,
    "early_stopping_rounds": 5,
    "use_gpu": False,
    "n_jobs": 1,
    "random_state": 0,
    "verbosity": 0,
}


def _engineer() -> FeatureEngineer:
    return FeatureEngineer(timeframe="5min", enable_mtf=False, enable_wavelets=False)


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return make_intraday_ohlcv(1500, seed=4)


@pytest.fixture(scope="module")
def bundle_dir(raw: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> Path:
    engineer = _engineer()
    features, _ = engineer.compute_features(raw.reset_index())
    features = features[engineer.warmup_mask(features["datetime"])]
    X = features[COLUMNS].to_numpy(np.float32)
    future = features["close"].shift(-5) / features["close"] - 1
    y = np.digitize(future.fillna(0).to_numpy(), [-0.0005, 0.0005]) - 1
    scaler = RobustScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)
    split = int(len(Xs) * 0.8)
    model = ModelRegistry.create("xgboost", config=MODEL_CONFIG)
    model.fit(Xs[:split], y[:split], Xs[split:], y[split:])
    graph = PreprocessingGraph.from_feature_pipeline(
        engineer.pipeline_record("5min"), feature_columns=COLUMNS, horizon=5
    )
    bundle = ModelBundle.from_training(
        model=model,
        scaler=scaler,
        feature_columns=COLUMNS,
        horizon=5,
        preprocessing_graph=graph,
        model_name="xgboost",
    )
    return bundle.save(tmp_path_factory.mktemp("bundle") / "xgboost_h5")


def _set_engine_version(bundle_dir: Path, version: int | None) -> Path:
    graph_path = bundle_dir / PREPROCESSING_GRAPH_FILE
    data = json.loads(graph_path.read_text())
    if version is None:
        data.pop("feature_engine_version")
    else:
        data["feature_engine_version"] = version
    graph_path.write_text(json.dumps(data))
    return bundle_dir


@pytest.fixture
def stale_bundle(bundle_dir: Path, tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    import shutil

    copy = Path(shutil.copytree(bundle_dir, tmp_path / "stale"))
    return _set_engine_version(copy, request.param)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


def test_graph_records_engine_version_and_warmup(bundle_dir: Path) -> None:
    graph = PreprocessingGraph.load(bundle_dir / PREPROCESSING_GRAPH_FILE)
    assert graph.config.feature_engine_version == FEATURE_ENGINE_VERSION
    assert graph.config.warmup_bars == _engineer().warmup_bars()


def test_served_rows_follow_the_training_warmup_rule(bundle_dir: Path, raw: pd.DataFrame) -> None:
    bundle = ModelBundle.load(bundle_dir)
    window = raw.iloc[-700:]
    served = pd.DatetimeIndex(bundle.predict_from_raw(window).metadata["timestamps"])
    kept = _engineer().warmup_mask(pd.Series(window.index))
    assert list(served) == list(window.index[kept])

    # ...and those rows score exactly as with the full history
    full = bundle.predict_from_raw(raw)
    full_probs = pd.DataFrame(full.class_probabilities, index=full.metadata["timestamps"])
    short = bundle.predict_from_raw(window)
    np.testing.assert_allclose(
        short.class_probabilities, full_probs.loc[served].to_numpy(), rtol=0, atol=1e-5
    )


# ---------------------------------------------------------------------------
# Too little history
# ---------------------------------------------------------------------------


def test_short_input_names_the_raw_bars_needed(bundle_dir: Path) -> None:
    bundle = ModelBundle.load(bundle_dir)
    needed = bundle.preprocessing_graph.config.warmup_bars + 1
    # 1-minute input resampled to the 5-minute training bars: 5 raw bars per bar
    with pytest.raises(ValueError, match=f"at least {needed * 5} raw 1min bars"):
        bundle.predict_from_raw(make_intraday_ohlcv(400, freq="1min"))
    with pytest.raises(ValueError, match=f"at least {needed} raw 5min bars"):
        bundle.predict_from_raw(make_intraday_ohlcv(needed - 1))


# ---------------------------------------------------------------------------
# Engine version
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stale_bundle", "message"),
    [(FEATURE_ENGINE_VERSION - 1, "feature engine version"), (None, "unversioned")],
    indirect=["stale_bundle"],
    ids=["older_version", "no_version"],
)
def test_bundle_from_another_engine_is_refused(stale_bundle: Path, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ModelBundle.load(stale_bundle)
    with pytest.raises(ValueError, match="allow_engine_mismatch=True"):
        load_bundle(stale_bundle)


@pytest.mark.parametrize("stale_bundle", [FEATURE_ENGINE_VERSION - 1], indirect=True)
def test_engine_mismatch_can_be_allowed_explicitly(
    stale_bundle: Path, raw: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR, logger="src.inference.preprocessing_graph"):
        bundle = load_bundle(stale_bundle, allow_engine_mismatch=True)
    assert "FEATURE ENGINE MISMATCH" in caplog.text
    assert len(bundle.predict_from_raw(raw).class_predictions) > 0
