"""Regression tests for the Phase 117 adversarial-review findings.

- embargo units under event sampling (bar embargo for the split, sample embargo
  covering it for the CV)
- fit prefix for the CUSUM threshold / auto d (walk-forward, mid/tail row drops)
- regime-aware minimum-sample guard counts labeled rows
- YAML scientific notation, mixed-case position_sizing
- probability sizing edge cases (missing confidence, skipped-trade count, K)
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.config.experiment import ExperimentConfig
from src.core.label_spans import INVALID_LABEL, LabelSpans
from src.data.adapters.preparation import UnifiedDataPreparation
from src.factory import MLFactory
from src.validation.cv.purged_kfold import PurgedKFold, PurgedKFoldConfig
from tests.helpers import make_intraday_ohlcv, tiny_prepared_data

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _factory(tmp_path: Path, **overrides: object) -> MLFactory:
    cfg = ExperimentConfig()
    cfg.verbose = 0
    cfg.output_dir = tmp_path / "out" / cfg.run_id
    cfg.data.symbol = "MES"
    cfg.data.labeling.event_sampling = "cusum"
    for dotted, value in overrides.items():
        target = cfg
        *parents, leaf = dotted.split(".")
        for part in parents:
            target = getattr(target, part)
        setattr(target, leaf, value)
    factory = MLFactory(cfg, verbose=0, enable_checkpoints=False)
    factory._feature_pipeline = {"bar_timeframe": "5min", "engineer": {}}
    return factory


def _labeled_frame(n_bars: int, event_bars: np.ndarray) -> pd.DataFrame:
    labels = np.full(n_bars, INVALID_LABEL)
    labels[event_bars] = 1
    return pd.DataFrame({"label": labels})


def _clustered_events(n_bars: int, cluster: int = 120, gap: int = 380) -> np.ndarray:
    """Every bar of a `cluster`-bar burst is an event, then `gap` quiet bars."""
    bars = [np.arange(s, min(s + cluster, n_bars)) for s in range(0, n_bars, cluster + gap)]
    return np.concatenate(bars)


# ---------------------------------------------------------------------------
# 2. Embargo units
# ---------------------------------------------------------------------------


def test_split_embargo_stays_in_bars_of_the_full_frame(tmp_path: Path) -> None:
    """The val->test embargo is derived from the bars, not shrunk by the event count."""
    n_bars = 20000
    rng = np.random.default_rng(0)
    events = np.sort(rng.choice(n_bars, size=2200, replace=False))
    df = _labeled_frame(n_bars, events)
    factory = _factory(tmp_path)

    pipeline = factory._pipeline_config(df=df)

    bar_purge, bar_embargo = factory.config.resolve_cv_gaps("5min", n_bars)
    assert bar_embargo == 288  # the reviewer's example: 2200 samples would have capped it at 77
    assert pipeline.split_embargo_bars == bar_embargo
    prep = UnifiedDataPreparation(pipeline)
    _train_end, _val_start, val_end, test_start = prep._split_bounds(n_bars)
    assert test_start - val_end == max(bar_purge, bar_embargo)


def test_split_embargo_defaults_to_the_cv_embargo_without_event_sampling(tmp_path: Path) -> None:
    factory = _factory(tmp_path, **{"data.labeling.event_sampling": "none"})
    df = _labeled_frame(5000, np.arange(5000))
    pipeline = factory._pipeline_config(df=df)
    assert pipeline.split_embargo_bars is None
    prep = UnifiedDataPreparation(pipeline)
    assert prep._split_embargo_bars() == pipeline.embargo_bars


def _bar_embargo_violations(events: np.ndarray, n_splits: int, sample_embargo: int) -> int:
    """Folds where a training sample sits within 288 bars after the test block."""
    spans = LabelSpans(events, events + 5)
    cv = PurgedKFold(
        PurgedKFoldConfig(n_splits=n_splits, purge_bars=12, embargo_bars=sample_embargo)
    )
    violations = 0
    for train_idx, test_idx in cv.split(pd.DataFrame(index=range(len(events))), label_spans=spans):
        last_test_bar = events[test_idx.max()]
        later_train = events[train_idx][events[train_idx] > last_test_bar]
        if len(later_train) and later_train.min() - last_test_bar <= 288:
            violations += 1
    return violations


def test_cv_embargo_in_samples_covers_the_bar_embargo_even_when_events_cluster(
    tmp_path: Path,
) -> None:
    """After every test block PurgedKFold leaves out at least `bar_embargo` bars of samples."""
    n_bars, bar_embargo, n_splits = 12000, 288, 5
    events = _clustered_events(n_bars, cluster=150, gap=350)
    df = _labeled_frame(n_bars, events)
    factory = _factory(
        tmp_path,
        **{
            "training.embargo_bars": bar_embargo,
            "training.n_splits": n_splits,
            "training.purge_bars": 12,
        },
    )
    pipeline = factory._pipeline_config(df=df)
    assert pipeline.split_embargo_bars == bar_embargo

    # A sample embargo sized from the event COUNT (25% of a fold, 77 here) lets
    # clustered events back into training within the 288-bar embargo ...
    assert _bar_embargo_violations(events, n_splits, 77) > 0
    # ... the converted one covers it in every fold
    assert pipeline.embargo_bars > 77
    assert _bar_embargo_violations(events, n_splits, pipeline.embargo_bars) == 0


def test_sample_embargo_is_at_most_the_bar_embargo(tmp_path: Path) -> None:
    """Each bar holds at most one event, so a sample embargo never exceeds the bar value."""
    events = np.arange(3000)  # every bar an event: densest possible
    factory = _factory(tmp_path, **{"training.embargo_bars": 100})
    assert factory._sample_embargo(_labeled_frame(3000, events), 100) == 100


def test_capped_derived_embargo_is_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    events = np.arange(600)  # tiny frame: 25% of a fold caps the derived embargo
    factory = _factory(tmp_path)
    with caplog.at_level(logging.WARNING, logger="src.factory"):
        capped = factory._sample_embargo(_labeled_frame(600, events), 288)
    assert capped < 288
    assert "capped" in caplog.text


# ---------------------------------------------------------------------------
# 3. Fit prefix: walk-forward, and the guard against dropped rows
# ---------------------------------------------------------------------------


def test_fit_fraction_is_the_train_ratio_in_standard_mode(tmp_path: Path) -> None:
    assert _factory(tmp_path)._fit_fraction(4000) == pytest.approx(0.7)


def test_fit_fraction_stops_at_the_first_walk_forward_test_window(tmp_path: Path) -> None:
    factory = _factory(tmp_path, **{"training.training_mode": "walk_forward"})
    span = factory.config.label_span_bars()
    assert factory._fit_fraction(4000) == pytest.approx(0.4 - span / 4000)
    assert factory._fit_fraction(4000) < 0.7


def test_walk_forward_threshold_ignores_bars_between_the_first_test_window_and_the_split(
    tmp_path: Path,
) -> None:
    raw = make_intraday_ohlcv(4000, seed=5)
    factory = _factory(tmp_path, **{"training.training_mode": "walk_forward"})
    base = factory._resolve_event_sampling(raw)
    close = raw.columns.get_loc("close")

    inside_train_split = raw.copy()  # bars 45%..70%: tested by walk-forward windows
    rng = np.random.default_rng(2)
    inside_train_split.iloc[1800:2800, close] *= np.exp(np.cumsum(rng.normal(0, 0.05, 1000)))
    assert factory._resolve_event_sampling(inside_train_split) == base

    before_first_window = raw.copy()
    before_first_window.iloc[:800, close] *= np.exp(np.cumsum(rng.normal(0, 0.01, 800)))
    assert factory._resolve_event_sampling(before_first_window) != base


def test_labeler_cost_calibration_uses_the_same_walk_forward_prefix(tmp_path: Path) -> None:
    wf = _factory(tmp_path, **{"training.training_mode": "walk_forward"})
    assert wf._fit_fraction(3000) < 0.4


def _features_frame(raw: pd.DataFrame, keep: np.ndarray) -> pd.DataFrame:
    return raw.iloc[keep].reset_index().rename(columns={"index": "datetime"})


def test_fit_prefix_guard_accepts_warmup_dropped_at_the_head(tmp_path: Path) -> None:
    raw = make_intraday_ohlcv(1000, seed=1)
    factory = _factory(tmp_path)
    factory._check_fit_prefix(raw, _features_frame(raw, np.arange(200, 1000)))


def test_fit_prefix_guard_rejects_rows_dropped_at_the_tail(tmp_path: Path) -> None:
    """A tail drop moves the training split earlier than the bars used for the fit."""
    raw = make_intraday_ohlcv(1000, seed=1)
    factory = _factory(tmp_path)
    with pytest.raises(ValueError, match="reach past the training split"):
        factory._check_fit_prefix(raw, _features_frame(raw, np.arange(0, 700)))


# ---------------------------------------------------------------------------
# 4. Regime-aware minimum samples count labeled rows
# ---------------------------------------------------------------------------


def _regime_setup(tmp_path: Path):
    from src.core import PipelineConfig
    from src.models.training.regime_detector import RegimeResult
    from src.models.training.regime_trainer import RegimeAwareTrainer

    prepared = tiny_prepared_data(n_train=400, n_val=40)
    y = prepared.y_train.copy()
    y[:300] = INVALID_LABEL  # regime "sparse": 300 rows, all invalid but 0
    y[:20] = 1  # 20 labeled rows
    prepared.y_train = y
    regimes = pd.Series(["sparse"] * 300 + ["dense"] * 100)
    config = PipelineConfig(
        symbol="MES", data_path="x", output_dir=tmp_path, regime_min_samples=50, models=["xgboost"]
    )
    return RegimeAwareTrainer(config), prepared, RegimeResult(regimes=regimes)


def test_regime_minimum_counts_labeled_rows_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.models.training.services as services

    trained_on: list[int] = []

    class _FakeService:
        def train_model(self, request):
            trained_on.append(len(request.prepared_data.y_train))
            trainer = SimpleNamespace(model=SimpleNamespace(save=lambda path: None))
            return SimpleNamespace(trainer=trainer, metrics={}, training_time_seconds=0.0)

    monkeypatch.setattr(services, "ModelTrainingService", _FakeService)
    trainer, prepared, regime_result = _regime_setup(tmp_path)

    results = trainer._train_separate_models(
        prepared, horizon=5, models=["xgboost"], regime_result=regime_result, save_models=False
    )

    # "sparse" has 300 rows but only 20 labeled (< 50): skipped. Before the fix its
    # 300 rows (incl. -99) passed the guard and a model was trained on 20 labels.
    assert set(results) == {("xgboost", "dense")}
    assert trained_on == [100]
    dense = results[("xgboost", "dense")]
    labeled_total = int((prepared.y_train != INVALID_LABEL).sum())
    assert dense.n_samples == int((prepared.y_train[300:] != INVALID_LABEL).sum())
    assert dense.sample_fraction == pytest.approx(dense.n_samples / labeled_total)


# ---------------------------------------------------------------------------
# 5. YAML scientific notation
# ---------------------------------------------------------------------------


def test_yaml_scientific_notation_without_a_dot_is_numeric(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "data:\n"
        "  labeling:\n"
        "    event_sampling: cusum\n"
        "    cusum_threshold: 2e-3\n"
        "  features:\n"
        "    frac_diff:\n"
        "      enabled: true\n"
        "      d: 4e-1\n"
        "      threshold: 1e-5\n"
        "evaluation:\n"
        "  bet_step_size: 1e-1\n"
    )
    cfg = ExperimentConfig.from_yaml(path)

    assert cfg.data.labeling.cusum_threshold == pytest.approx(2e-3)
    assert cfg.data.features.frac_diff.threshold == pytest.approx(1e-5)
    assert cfg.data.features.frac_diff.d == pytest.approx(0.4)
    assert cfg.evaluation.bet_step_size == pytest.approx(0.1)
    assert cfg.validate() == []


def test_auto_and_other_strings_are_not_coerced(tmp_path: Path) -> None:
    cfg = ExperimentConfig.from_dict(
        {
            "data": {
                "labeling": {"cusum_threshold": "auto"},
                "features": {"frac_diff": {"d": "auto"}},
            }
        }
    )
    assert cfg.data.labeling.cusum_threshold == "auto"
    assert cfg.data.features.frac_diff.d == "auto"
    cfg.data.features.frac_diff.threshold = "1e-5"  # type: ignore[assignment]
    assert any("frac_diff.threshold" in i for i in cfg.validate())  # reported, not a TypeError


# ---------------------------------------------------------------------------
# 6. position_sizing is case-insensitive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["Kelly", "FIXED", "Probability"])
def test_mixed_case_position_sizing_validates(value: str) -> None:
    cfg = ExperimentConfig()
    cfg.evaluation.position_sizing = value
    assert cfg.validate() == []


# ---------------------------------------------------------------------------
# 7. Probability sizing edge cases
# ---------------------------------------------------------------------------


def _bars(n: int = 120) -> pd.DataFrame:
    ts = pd.date_range("2024-01-02 09:30", periods=n, freq="5min")
    close = 5000.0 + np.arange(n) * 0.25
    return pd.DataFrame(
        {"timestamp": ts, "open": close, "high": close + 1, "low": close - 1, "close": close}
    )


def _run(signals: pd.DataFrame, **config: object):
    from src.inference.backtesting import BacktestConfig, Backtester

    prices = _bars()
    cfg = BacktestConfig(
        position_sizing="probability",
        enable_market_hours_filter=False,
        max_holding_period=6,
        **config,  # type: ignore[arg-type]
    )
    return Backtester(predictions=signals, prices=prices, config=cfg).run()


def _signals(confidences: list[float | None], with_column: bool = True) -> pd.DataFrame:
    ts = _bars()["timestamp"]
    rows = {"timestamp": [ts[5 + 12 * i] for i in range(len(confidences))], "prediction": 1}
    if with_column:
        rows["confidence"] = confidences
    frame = pd.DataFrame(rows)
    closing = pd.DataFrame({"timestamp": [ts.iloc[-1]], "prediction": [0]})
    if with_column:
        closing["confidence"] = 0.5
    return pd.concat([frame, closing], ignore_index=True)


def test_missing_confidence_column_means_no_bet_not_full_size(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="src.inference.backtesting.backtest"):
        result = _run(_signals([0.9, 0.9], with_column=False))
    assert result.trades == []
    assert "needs a 'confidence'" in caplog.text


def test_other_sizers_still_default_a_missing_confidence_to_one() -> None:
    from src.inference.backtesting import BacktestConfig, Backtester

    cfg = BacktestConfig(position_sizing="fixed_contracts", enable_market_hours_filter=False)
    bt = Backtester(predictions=_signals([1.0], with_column=False), prices=_bars(), config=cfg)
    assert (bt.predictions["confidence"] == 1.0).all()


def test_nan_confidence_opens_no_position() -> None:
    result = _run(_signals([float("nan"), 0.9]), bet_max_contracts=5)
    assert [t.confidence for t in result.trades] == [0.9]


def test_zero_size_entries_are_counted_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="src.inference.backtesting.backtest"):
        result = _run(_signals([0.30, 0.34, 0.9]), bet_max_contracts=5)
    assert result.summary()["zero_size_signals"] == 2
    assert len(result.trades) == 1
    assert "2 entry signals opened no position" in caplog.text


def test_factory_wires_the_class_count_into_the_sizer(tmp_path: Path) -> None:
    from src.inference.backtesting import BacktestConfig

    captured: dict[str, object] = {}

    class _Spy:
        def __init__(self, predictions, prices, config):
            captured["config"] = config

        def run(self):
            raise RuntimeError("stop here")

    import src.inference.backtesting as backtesting

    factory = _factory(
        tmp_path,
        **{
            "evaluation.run_backtest": True,
            "evaluation.position_sizing": "probability",
            "evaluation.bet_max_contracts": 7,
        },
    )
    df = make_intraday_ohlcv(300)
    df["label"] = 1
    preds = pd.DataFrame({"datetime": df.index[10:20], "prediction": 1, "confidence": 0.9})
    original = backtesting.Backtester
    backtesting.Backtester = _Spy  # type: ignore[misc]
    try:
        factory._extract_predictions = lambda d, t: (preds, "spy")  # type: ignore[method-assign]
        factory._run_evaluation(df, SimpleNamespace())
    finally:
        backtesting.Backtester = original
    config: BacktestConfig = captured["config"]  # type: ignore[assignment]
    assert (config.bet_max_contracts, config.bet_n_classes) == (7, 3)
    assert config.position_sizing == "probability"
