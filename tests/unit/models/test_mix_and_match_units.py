"""
Unit tests for the mix-and-match plumbing: every meta-learner round-trips
through disk, the soft-voting combiner, bar-timeframe detection, the shared
FeatureEngineer spec, train/serve feature parity of PreprocessingGraph, and
CPU-safe DataLoader defaults.

All tests are fast (no neural training, no MLFactory runs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.common.timeframes import detect_timeframe
from src.core.constants import MODEL_FAMILIES
from src.data.pipeline.stages.features import FeatureEngineer
from src.inference.preprocessing_graph import PreprocessingGraph
from src.models.ensemble import VotingMetaLearner, get_meta_learner

N_CLASSES = 3


def _stacking_data(
    n: int, n_models: int = 3, seed: int = 0, n_classes: int = N_CLASSES
) -> tuple[np.ndarray, np.ndarray]:
    """OOFAligner-shaped stacking features: n_models*n_classes probability cols + 3 derived."""
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(n_classes), size=(n, n_models)).reshape(n, -1)
    derived = rng.random((n, 3))
    # Labels: {-1, 0, 1} for 3 classes, {0, 1} for binary
    y = rng.integers(-1, 2, size=n) if n_classes == 3 else rng.integers(0, 2, size=n)
    return np.hstack([probs, derived]).astype(np.float64), y


def _ohlcv(n: int, freq: str, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq=freq)
    close = 5000.0 + np.cumsum(rng.normal(0, 2.0, n))
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, 0.5, n)
    eps = np.abs(rng.normal(0, 0.5, n)) + 0.25
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + eps,
            "low": np.minimum(open_, close) - eps,
            "close": close,
            "volume": rng.integers(100, 5000, n).astype(float),
        },
        index=pd.DatetimeIndex(idx, name="datetime"),
    )


# ---------------------------------------------------------------------------
# Meta-learners
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_classes", [3, 2], ids=["3class", "binary"])
@pytest.mark.parametrize("name", MODEL_FAMILIES["meta_learner"])
def test_every_meta_learner_fits_saves_loads_and_predicts_identically(
    name, n_classes, tmp_path
) -> None:
    X_train, y_train = _stacking_data(300, seed=1, n_classes=n_classes)
    X_val, y_val = _stacking_data(80, seed=2, n_classes=n_classes)

    meta = get_meta_learner(name, n_classes=n_classes)
    meta.fit(X_train, y_train, X_val, y_val)
    before = meta.predict(X_val)
    assert before.class_probabilities.shape == (len(X_val), n_classes)
    assert set(np.unique(before.class_predictions)) <= set(np.unique(y_train))
    np.testing.assert_allclose(before.class_probabilities.sum(axis=1), 1.0, atol=1e-6)

    meta.save(tmp_path / name)
    reloaded = get_meta_learner(name)
    reloaded.load(tmp_path / name)
    after = reloaded.predict(X_val)
    np.testing.assert_allclose(after.class_probabilities, before.class_probabilities)


def test_voting_meta_is_mean_of_base_probabilities() -> None:
    X, y = _stacking_data(50, n_models=2, seed=4)
    meta = VotingMetaLearner(config={"n_classes": N_CLASSES})
    meta.fit(X, y, X, y)

    expected = X[:, :6].reshape(50, 2, 3).mean(axis=1)
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(meta.predict(X).class_probabilities, expected)


def test_voting_meta_rejects_non_stacking_layout() -> None:
    meta = VotingMetaLearner(config={"n_classes": N_CLASSES})
    X = np.zeros((10, 7))  # 7 - 3 derived = 4, not a multiple of 3 classes
    with pytest.raises(ValueError, match="probability"):
        meta.fit(X, np.zeros(10), X, np.zeros(10))


# ---------------------------------------------------------------------------
# Bar timeframe detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("freq", "expected"),
    [("1min", "1min"), ("5min", "5min"), ("15min", "15min"), ("1h", "60min")],
)
def test_detect_timeframe(freq: str, expected: str) -> None:
    assert detect_timeframe(_ohlcv(50, freq)) == expected


def test_detect_timeframe_is_robust_to_session_gaps() -> None:
    df = pd.concat([_ohlcv(100, "5min"), _ohlcv(100, "5min").shift(1, freq="1D")])
    assert detect_timeframe(df) == "5min"


def test_detect_timeframe_rejects_sub_minute_bars() -> None:
    assert detect_timeframe(_ohlcv(50, "30s")) is None


# ---------------------------------------------------------------------------
# FeatureEngineer spec + PreprocessingGraph train/serve parity
# ---------------------------------------------------------------------------


def test_feature_engineer_spec_round_trips() -> None:
    engineer = FeatureEngineer(timeframe="1min", enable_mtf=False, wavelet_window=32)
    rebuilt = FeatureEngineer.from_spec(engineer.to_spec())
    assert rebuilt.to_spec() == engineer.to_spec()
    assert rebuilt.period_config == engineer.period_config


# Default engineer (MTF on 15min/60min): ~1900 bars of warmup, then scored bars
N_BARS = 2500


@pytest.fixture(scope="module")
def trained_features(tmp_path_factory: pytest.TempPathFactory) -> tuple[pd.DataFrame, dict]:
    raw = _ohlcv(N_BARS, "5min")
    engineer = FeatureEngineer(output_dir=tmp_path_factory.mktemp("fe"), timeframe="5min")
    featured, _report = engineer.engineer_features(raw.reset_index(), symbol="TEST")
    return featured.set_index("datetime"), engineer.pipeline_record("5min")


def test_graph_reproduces_training_features_exactly(trained_features) -> None:
    featured, pipeline = trained_features
    columns = [c for c in featured.columns if c not in ("open", "high", "low", "close", "volume")]
    graph = PreprocessingGraph.from_feature_pipeline(pipeline, feature_columns=columns)

    served = graph.transform(_ohlcv(N_BARS, "5min"), skip_scaling=True)

    assert list(served.columns) == columns
    common = served.index.intersection(featured.index)
    assert len(common) == len(featured)
    np.testing.assert_allclose(
        served.loc[common].to_numpy(float),
        featured.loc[common, columns].to_numpy(float),
        rtol=1e-6,
        atol=1e-8,
    )


def test_graph_resamples_finer_bars_to_training_timeframe(trained_features) -> None:
    _featured, pipeline = trained_features
    graph = PreprocessingGraph.from_feature_pipeline(pipeline, feature_columns=["rsi_14"])
    served = graph.transform(_ohlcv(5 * N_BARS, "1min"), skip_scaling=True)
    assert detect_timeframe(pd.DataFrame(index=served.index)) == "5min"


def test_graph_rejects_bars_coarser_than_training(trained_features) -> None:
    _featured, pipeline = trained_features
    graph = PreprocessingGraph.from_feature_pipeline(pipeline, feature_columns=["rsi_14"])
    with pytest.raises(ValueError, match="trained on 5min"):
        graph.transform(_ohlcv(600, "15min"))


def test_graph_round_trips_through_disk(trained_features, tmp_path) -> None:
    _featured, pipeline = trained_features
    graph = PreprocessingGraph.from_feature_pipeline(pipeline, feature_columns=["rsi_14"])
    graph.save(tmp_path / "preprocessing_graph.json")
    loaded = PreprocessingGraph.load(tmp_path / "preprocessing_graph.json")
    assert loaded.to_dict() == graph.to_dict()
    assert loaded.validate()["valid"]


def test_legacy_graph_without_spec_refuses_to_transform() -> None:
    graph = PreprocessingGraph.from_dict({"version": "1.0.0", "feature_columns": ["rsi_14"]})
    with pytest.raises(ValueError, match="predates train/serve parity"):
        graph.transform(_ohlcv(200, "5min"))


# ---------------------------------------------------------------------------
# DataLoader defaults
# ---------------------------------------------------------------------------


def test_cpu_dataloader_uses_no_worker_processes_by_default() -> None:
    """Forked workers duplicate the parent's memory; on CPU they only cost RAM."""
    from src.models.config.trainer_config import TrainerConfig
    from src.models.registry import ModelRegistry

    trainer_cfg = TrainerConfig(model_name="lstm")
    assert trainer_cfg.num_workers is None
    assert trainer_cfg.pin_memory is None

    model = ModelRegistry.create("lstm", config={"device": "cpu"})
    model._device = __import__("torch").device("cpu")
    loader = model._create_dataloader(
        np.zeros((8, 4, 3), dtype=np.float32),
        np.zeros(8),
        None,
        {"num_workers": trainer_cfg.num_workers, "pin_memory": trainer_cfg.pin_memory},
        shuffle=False,
    )
    assert loader.num_workers == 0
    assert loader.pin_memory is False
