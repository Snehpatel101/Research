"""
End-to-end: CUSUM event sampling + fractional differentiation + probability bet sizing.

One factory run with all three opt-in options on tiny synthetic 5-minute bars
(xgboost, no Optuna): the run trains on event bars only, adds ``ffd_log_*``
features with a train-fitted ``d`` frozen into the bundle, backtests with AFML
probability sizing, deploys, and the deployed bundle reproduces the trained
model (prediction parity) and the training features (train/serve parity).
The same options run through the mix-and-match harness for a cross-rank stack.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config.experiment import ExperimentConfig
from src.core.label_spans import INVALID_LABEL, NO_LABEL_END
from src.data.pipeline.stages.features.engineer import FeatureEngineer
from src.factory import ExperimentResult, MLFactory
from src.inference.bundle import ModelBundle
from src.inference.preprocessing_graph import PreprocessingGraph
from tests.helpers import REPO_ROOT, make_intraday_ohlcv

pytestmark = pytest.mark.slow

N_ROWS = 6000
OVERRIDES = {
    "data.labeling.event_sampling": "cusum",
    "data.labeling.cusum_vol_multiple": 1.5,  # ~28% of bars are events on this data
    "data.features.frac_diff.enabled": True,
    "data.features.frac_diff.d": "auto",
    "evaluation.position_sizing": "probability",
}


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "mix_match", REPO_ROOT / "scripts" / "mix_match.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harness = _load_harness()


@pytest.fixture(scope="module")
def raw_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("statsmodels")  # d="auto" needs the ADF test
    path = tmp_path_factory.mktemp("data") / "mes_5min.parquet"
    make_intraday_ohlcv(N_ROWS, seed=7).to_parquet(path)
    return path


@pytest.fixture(scope="module")
def run(
    raw_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[ExperimentConfig, ExperimentResult]:
    cfg = ExperimentConfig()
    cfg.name = "event_frac_bet_e2e"
    cfg.random_seed = 42
    cfg.verbose = 0
    cfg.output_dir = tmp_path_factory.mktemp("run") / cfg.run_id
    cfg.data.symbol = "MES"
    cfg.data.data_path = raw_path
    cfg.data.mtf.enabled = False
    cfg.training.models = ["xgboost"]
    cfg.training.horizons = [5]
    cfg.training.n_splits = 2
    cfg.training.purge_bars = 15
    cfg.training.embargo_bars = 30
    cfg.training.build_ensemble = False
    cfg.training.optuna.n_trials = 0
    cfg.evaluation.run_backtest = True
    cfg.evaluation.bet_max_contracts = 6
    cfg.bundling.create_bundle = True
    cfg.bundling.deploy_artifact = True
    for dotted, value in OVERRIDES.items():
        harness.apply_override(cfg, dotted, value)
    result = MLFactory(cfg, verbose=0, enable_checkpoints=True).run()
    return cfg, result


@pytest.fixture(scope="module")
def trained_frame(run: tuple[ExperimentConfig, ExperimentResult]) -> pd.DataFrame:
    _cfg, result = run
    return pd.read_parquet(Path(result.output_dir) / "cache" / "data_pipeline.parquet")


# ---------------------------------------------------------------------------
# Event sampling: labels only at events, spans in bar coordinates
# ---------------------------------------------------------------------------


def test_run_succeeds_with_a_backtest(run: tuple[ExperimentConfig, ExperimentResult]) -> None:
    _cfg, result = run
    assert result.success and result.n_models == 1
    assert result.backtest_metrics.get("total_trades", 0) > 0


def test_only_event_bars_carry_labels(trained_frame: pd.DataFrame) -> None:
    labels = trained_frame["label_h5"].to_numpy()
    ends = trained_frame["label_end_h5"].to_numpy()
    valid = labels != INVALID_LABEL
    assert 0.15 < valid.mean() < 0.45  # events, not every bar
    # every invalid label has no label end, every valid one resolves at or after its own bar
    assert (ends[~valid] == NO_LABEL_END).all()
    rows = np.flatnonzero(valid)
    assert (ends[rows] >= rows).all()
    # spans stay in bar coordinates of the FULL series: they reach beyond the next event
    assert (ends[rows] - rows).max() <= 12  # max_bars of the H5 barriers
    assert ends[valid].max() < len(trained_frame)


def test_features_are_computed_on_every_bar(trained_frame: pd.DataFrame) -> None:
    ffd = trained_frame["ffd_log_close"]
    assert ffd.notna().all()  # warmup rows were dropped once, no gaps at non-event bars
    # a lagged momentum feature exists on non-event bars too
    non_event = (trained_frame["label_h5"] == INVALID_LABEL).to_numpy()
    assert np.isfinite(trained_frame["rsi_14"].to_numpy()[non_event]).mean() > 0.99


def test_evaluation_counts_event_rows_only(run: tuple[ExperimentConfig, ExperimentResult]) -> None:
    _cfg, result = run
    metrics = next(iter(result.metrics.values()))
    # validation samples are the events inside the 15% validation split (900 of 6000 bars)
    assert 100 < metrics["n_samples"] < 0.15 * N_ROWS


# ---------------------------------------------------------------------------
# Frozen options in the recorded pipeline and the bundle
# ---------------------------------------------------------------------------


def test_resolved_options_are_recorded_in_pipeline_and_bundle(
    run: tuple[ExperimentConfig, ExperimentResult],
) -> None:
    _cfg, result = run
    out = Path(result.output_dir)
    recorded = json.loads((out / "cache" / "feature_pipeline.json").read_text())
    threshold = recorded["event_sampling"]["threshold"]
    d = recorded["engineer"]["frac_diff_d"]
    assert recorded["event_sampling"]["method"] == "cusum" and threshold > 0
    assert 0.0 < d <= 1.0
    assert recorded["engineer"]["frac_diff_columns"] == ["close", "open", "high", "low"]

    bundle = ModelBundle.load(out / "bundles" / "xgboost_h5")
    assert bundle.preprocessing_graph is not None
    graph_config = bundle.preprocessing_graph.config
    assert graph_config.event_sampling == {"method": "cusum", "threshold": threshold}
    assert graph_config.feature_engineering["frac_diff_d"] == d


def test_frac_diff_features_replay_exactly_from_the_bundle_spec(
    run: tuple[ExperimentConfig, ExperimentResult], raw_path: Path, trained_frame: pd.DataFrame
) -> None:
    _cfg, result = run
    bundle = ModelBundle.load(Path(result.output_dir) / "bundles" / "xgboost_h5")
    assert bundle.preprocessing_graph is not None
    engineer = FeatureEngineer.from_spec(bundle.preprocessing_graph.config.feature_engineering)
    bars = PreprocessingGraph._to_datetime_column(pd.read_parquet(raw_path))
    served, _ = engineer.compute_features(bars)
    served = served.set_index("datetime")
    common = served.index.intersection(trained_frame.index)
    assert len(common) == len(trained_frame)
    for name in ("ffd_log_close", "ffd_log_open", "ffd_log_high", "ffd_log_low"):
        np.testing.assert_allclose(
            served.loc[common, name].to_numpy(float),
            trained_frame.loc[common, name].to_numpy(float),
            rtol=1e-4,
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Serving: every bar predicted, event bars flagged; parity with the trained model
# ---------------------------------------------------------------------------


def test_predict_from_raw_flags_the_bars_the_model_was_trained_on(
    run: tuple[ExperimentConfig, ExperimentResult], raw_path: Path, trained_frame: pd.DataFrame
) -> None:
    _cfg, result = run
    bundle = ModelBundle.load(Path(result.output_dir) / "bundles" / "xgboost_h5")
    served = bundle.predict_from_raw(pd.read_parquet(raw_path), calibrate=False)

    stamps = pd.DatetimeIndex(served.metadata["timestamps"])
    flags = served.metadata["is_event"]
    assert flags.dtype == bool and len(flags) == len(stamps) == len(served.class_predictions)
    assert 0.15 < flags.mean() < 0.45  # predictions for every bar, only some are events

    labeled = trained_frame.index[trained_frame["label_h5"] != INVALID_LABEL]
    covered = labeled.intersection(stamps)
    assert len(covered) > 0.9 * len(labeled)
    assert flags[stamps.get_indexer(covered)].all()  # every training label sits on a flagged bar


def test_served_probabilities_match_the_trained_model_on_event_rows(
    run: tuple[ExperimentConfig, ExperimentResult], raw_path: Path, trained_frame: pd.DataFrame
) -> None:
    """Explicit prediction parity (the harness check below could skip silently)."""
    from src.core.constants import OHLCV_COLUMNS
    from src.data.adapters.preparation import UnifiedDataPreparation

    _cfg, result = run
    assert result.training_result is not None
    model_result = next(iter(result.training_result.model_results.values()))
    bundle = ModelBundle.load(Path(result.output_dir) / "bundles" / "xgboost_h5")
    keep = [
        c
        for c in trained_frame.columns
        if c in bundle.feature_columns or c in OHLCV_COLUMNS or c.startswith(("label", "sample_"))
    ]
    prepared = (
        UnifiedDataPreparation(result.training_result.config)
        .prepare(trained_frame[keep], model_name="xgboost", label_column="label_h5")
        .filter_invalid_labels()
    )
    assert prepared.val_indices is not None and len(prepared.val_indices) > 100
    assert model_result.trainer is not None
    expected = model_result.trainer.model.predict(prepared.X_val).class_probabilities

    served = bundle.predict_from_raw(pd.read_parquet(raw_path), calibrate=False)
    got = pd.DataFrame(
        served.class_probabilities, index=pd.DatetimeIndex(served.metadata["timestamps"])
    ).reindex(trained_frame.index[prepared.val_indices])
    covered = got.notna().all(axis=1).to_numpy()
    assert covered.mean() > 0.95
    np.testing.assert_allclose(got.to_numpy()[covered], expected[covered], atol=1e-3)


def test_deployed_bundle_reproduces_the_trained_model(
    run: tuple[ExperimentConfig, ExperimentResult], raw_path: Path
) -> None:
    _cfg, result = run
    assert harness._check_prediction_parity(result, raw_path) == []
    assert harness._check_deploy_predict(Path(result.deploy_path), raw_path) == []
    assert harness._check_universal_pipeline(result, raw_path) == []


# ---------------------------------------------------------------------------
# Backtest: event bars only, probability-sized positions
# ---------------------------------------------------------------------------


def test_backtest_acts_on_event_bars_with_probability_sized_positions(
    run: tuple[ExperimentConfig, ExperimentResult], trained_frame: pd.DataFrame
) -> None:
    from src.inference.backtesting import afml_bet_size

    _cfg, result = run
    trades = result.backtest_trades
    assert trades
    # entries are decided on event bars: the signal bar (one before the fill) is labeled
    labeled = set(trained_frame.index[trained_frame["label_h5"] != INVALID_LABEL])
    stamps = list(trained_frame.index)
    position = {t: i for i, t in enumerate(stamps)}
    for trade in trades:
        signal_bar = stamps[position[pd.Timestamp(trade.entry_time)] - 1]
        assert signal_bar in labeled
    # every trade is sized from its confidence by the AFML formula (at most 6 contracts)
    for trade in trades:
        expected = int(np.floor(afml_bet_size(trade.confidence) * 6 + 0.5))
        assert trade.contracts == expected >= 1
    assert len({t.contracts for t in trades}) > 1


# ---------------------------------------------------------------------------
# Same options through the mix-and-match harness: a cross-rank stack (2D + 3D)
# ---------------------------------------------------------------------------


def test_harness_cross_rank_stack_with_all_three_options(raw_path: Path, tmp_path: Path) -> None:
    spec = {
        "name": "event_frac_bet_stack",
        "models": ["xgboost", "lstm"],
        "meta": "ridge_meta",
        "mode": "standard",
        "overrides": OVERRIDES,
    }
    record = harness.run_one(spec, raw_path, tmp_path)
    assert "exception" not in record, record.get("traceback")
    assert record["ok"], record["problems"]


# ---------------------------------------------------------------------------
# Review regression: d="auto" when the train prefix already passes ADF at d=0
# ---------------------------------------------------------------------------


def test_prepare_data_with_auto_d_on_an_already_stationary_prefix(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """make_intraday_ohlcv(3000, seed=14): the raw ADF finding is d=0, which the spec rejects."""
    pytest.importorskip("statsmodels")
    data = tmp_path_factory.mktemp("seed14") / "mes.parquet"
    make_intraday_ohlcv(3000, seed=14).to_parquet(data)
    cfg = ExperimentConfig()
    cfg.verbose = 0
    cfg.output_dir = tmp_path_factory.mktemp("seed14_run") / cfg.run_id
    cfg.data.symbol = "MES"
    cfg.data.data_path = data
    cfg.data.mtf.enabled = False
    cfg.data.features.frac_diff.enabled = True  # d="auto"
    factory = MLFactory(cfg, verbose=0, enable_checkpoints=False)

    df, _ = factory.prepare_data()

    assert factory._feature_pipeline is not None
    assert factory._feature_pipeline["engineer"]["frac_diff_d"] >= 0.05
    assert "ffd_log_close" in df.columns
