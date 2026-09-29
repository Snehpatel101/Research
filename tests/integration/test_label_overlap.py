"""
Label-overlap handling (Phase 116).

A. Purging on integer label spans, independent of the index type: PurgedKFold,
   CPCV and walk-forward drop every training sample whose label resolves
   inside the test block; label ends that cannot be used raise.
   The spans flow labeler -> factory frame -> PreparedData -> OOF / tuner.
B. Derived purge (longest label span = triple-barrier max_bars) and embargo
   (one trading day at the bar timeframe, capped at 25% of a CV fold).
C. AFML average-uniqueness sample weights (default training weights).
D. AFML meta-labeling: the meta-model trains only on bars where the primary
   takes a side, and the deployed bundle builds the same meta features.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.config.experiment import ExperimentConfig
from src.core.label_spans import (
    NO_LABEL_END,
    LabelSpans,
    average_uniqueness,
    label_end_column,
    label_end_positions,
    remap_label_ends,
    uniqueness_sample_weights,
)
from src.validation.cv import PurgedKFold, PurgedKFoldConfig
from src.validation.cv.cpcv import CombinatorialPurgedCV, CPCVConfig
from src.validation.cv.walk_forward import WalkForwardConfig, WalkForwardEvaluator
from tests.helpers import REPO_ROOT, make_intraday_ohlcv

LABEL_LAG = 100  # labels resolve 100 bars after their bar (t5 audit reproduction)


def _lagged_spans(n: int, lag: int = LABEL_LAG) -> LabelSpans:
    starts = np.arange(n)
    return LabelSpans(starts=starts, ends=np.minimum(starts + lag, n - 1))


def _overlapping_train_rows(train: np.ndarray, test: np.ndarray, spans: LabelSpans) -> int:
    """Training samples whose [start, end] intersects the test block's span."""
    block_lo = spans.starts[test].min()
    block_hi = max(spans.starts[test].max(), spans.ends[test].max())
    return int(((spans.starts[train] <= block_hi) & (spans.ends[train] >= block_lo)).sum())


# =============================================================================
# A. PURGING ON POSITIONS
# =============================================================================


class TestPurgedKFoldLabelSpans:
    def test_range_index_purges_long_labels_beyond_fixed_purge(self) -> None:
        """t5: labels resolve 100 bars later, purge_bars=10 -> no overlapping train rows."""
        n = 2000
        spans = _lagged_spans(n)
        cv = PurgedKFold(PurgedKFoldConfig(n_splits=5, purge_bars=10, embargo_bars=10))
        X = pd.DataFrame(index=range(n))
        for train, test in cv.split(X, label_spans=spans):
            assert _overlapping_train_rows(train, test, spans) == 0
            # Both sides: nothing within LABEL_LAG before the block, nor after it
            assert not ((train < test.min()) & (train + LABEL_LAG >= test.min())).any()

    def test_without_spans_the_fixed_purge_leaks(self) -> None:
        """Control: the fixed purge alone leaves overlapping rows (what the audit found)."""
        n = 2000
        spans = _lagged_spans(n)
        cv = PurgedKFold(PurgedKFoldConfig(n_splits=5, purge_bars=10, embargo_bars=10))
        train, test = list(cv.split(pd.DataFrame(index=range(n))))[2]
        assert _overlapping_train_rows(train, test, spans) > 0

    def test_datetime_end_times_give_the_same_folds_as_positions(self) -> None:
        n = 1500
        idx = pd.date_range("2024-01-01", periods=n, freq="5min")
        spans = _lagged_spans(n)
        end_times = pd.Series(idx[spans.ends], index=idx)
        cv = PurgedKFold(PurgedKFoldConfig(n_splits=5, purge_bars=10, embargo_bars=10))
        by_time = list(cv.split(pd.DataFrame(index=idx), label_end_times=end_times))
        by_pos = list(cv.split(pd.DataFrame(index=range(n)), label_spans=spans))
        for (tr_a, te_a), (tr_b, te_b) in zip(by_time, by_pos, strict=True):
            np.testing.assert_array_equal(tr_a, tr_b)
            np.testing.assert_array_equal(te_a, te_b)

    def test_end_times_without_datetime_index_raise(self) -> None:
        n = 500
        idx = pd.date_range("2024-01-01", periods=n, freq="5min")
        cv = PurgedKFold(PurgedKFoldConfig(n_splits=5, purge_bars=5, embargo_bars=5))
        with pytest.raises(ValueError, match="DatetimeIndex"):
            list(cv.split(pd.DataFrame(index=range(n)), label_end_times=pd.Series(idx)))

    def test_span_length_mismatch_raises(self) -> None:
        cv = PurgedKFold(PurgedKFoldConfig(n_splits=5, purge_bars=5, embargo_bars=5))
        with pytest.raises(ValueError, match="label_spans"):
            list(cv.split(pd.DataFrame(index=range(500)), label_spans=_lagged_spans(400)))

    def test_spans_survive_row_filtering(self) -> None:
        """Spans are bar positions: after dropping rows the purge stays exact."""
        n = 2000
        spans = _lagged_spans(n)
        keep = np.arange(n) % 3 != 0  # e.g. invalid labels removed
        sub = spans.subset(keep)
        cv = PurgedKFold(PurgedKFoldConfig(n_splits=4, purge_bars=5, embargo_bars=5))
        for train, test in cv.split(pd.DataFrame(index=range(len(sub))), label_spans=sub):
            assert _overlapping_train_rows(train, test, sub) == 0


class TestOtherSplittersLabelSpans:
    def test_cpcv_purges_overlapping_labels(self) -> None:
        n = 1800
        spans = _lagged_spans(n)
        cpcv = CombinatorialPurgedCV(
            CPCVConfig(n_groups=6, n_test_groups=2, purge_bars=10, embargo_bars=5)
        )
        bounds = cpcv.group_boundaries(n)
        for train, _test, split_id in cpcv.split(np.zeros((n, 1)), label_spans=spans):
            for g in cpcv._test_combinations[split_id]:
                group = np.arange(*bounds[g])
                assert _overlapping_train_rows(train, group, spans) == 0

    def test_walk_forward_drops_labels_resolving_in_test(self) -> None:
        n = 3000
        spans = _lagged_spans(n)
        wf = WalkForwardEvaluator(
            WalkForwardConfig(n_windows=3, min_train_pct=0.4, test_pct=0.1, gap_bars=10)
        )
        for train, test in wf.split(pd.DataFrame(index=range(n)), label_spans=spans):
            assert (spans.ends[train] < spans.starts[test[0]]).all()


class TestLabelEnds:
    def test_labeler_exposes_label_end_positions(self) -> None:
        from src.data.labeling import TripleBarrierConfig, TripleBarrierLabeler

        df = make_intraday_ohlcv(1200, index_name=None)
        max_bars = 12
        labeler = TripleBarrierLabeler(
            TripleBarrierConfig(
                horizon=max_bars, upper_mult=1.0, lower_mult=1.0, atr_column=None, symbol="MES"
            )
        )
        labels_series, ends = labeler.create_labels_with_ends(df)
        labels = labels_series.to_numpy()
        valid = labels != -99
        rows = np.arange(len(df))
        assert valid.sum() > 0 and (~valid).sum() >= max_bars
        assert (ends[~valid] == NO_LABEL_END).all()
        assert (ends[valid] >= rows[valid] + 1).all()
        assert (ends[valid] <= rows[valid] + max_bars).all()
        # Timeouts resolve at the vertical barrier
        timeout = valid & (labels == 0)
        assert (ends[timeout] == rows[timeout] + max_bars).all()

    def test_label_end_positions_and_remap(self) -> None:
        labels = np.array([1, 0, -99, -1])
        ends = label_end_positions(labels, np.array([2, 1, 5, 1]))
        np.testing.assert_array_equal(ends, [2, 2, NO_LABEL_END, 4])
        # Drop row 1: row 0's label (end 2) and later rows shift left by one
        kept = np.array([0, 2, 3])
        np.testing.assert_array_equal(remap_label_ends(ends[kept], kept), [1, NO_LABEL_END, 2])

    def test_label_end_column_names(self) -> None:
        assert label_end_column("label") == "label_end"
        assert label_end_column("label_h20") == "label_end_h20"


# =============================================================================
# B. DERIVED PURGE / EMBARGO
# =============================================================================


def _config(**training: Any) -> ExperimentConfig:
    cfg = ExperimentConfig(run_id="fixed")
    cfg.data.symbol = "MES"
    cfg.training.horizons = [5, 20]
    for key, value in training.items():
        setattr(cfg.training, key, value)
    return cfg


class TestDerivedGaps:
    def test_defaults_are_derived(self) -> None:
        cfg = _config()
        assert cfg.training.purge_bars is None and cfg.training.embargo_bars is None

    def test_purge_equals_longest_label_span(self) -> None:
        cfg = _config()
        spans = [cfg.resolve_barrier_params(h)[2] for h in cfg.training.horizons]
        purge, _ = cfg.resolve_cv_gaps("5min")
        assert purge == max(spans) == cfg.label_span_bars()
        assert purge > max(cfg.training.horizons)  # labels outlive the horizon

    def test_purge_follows_max_holding_override(self) -> None:
        cfg = _config()
        cfg.data.labeling.max_holding_bars = 77
        assert cfg.resolve_cv_gaps("5min")[0] == 77

    def test_short_explicit_purge_is_raised_to_span(self, caplog) -> None:
        cfg = _config(purge_bars=5)
        with caplog.at_level(logging.WARNING):
            purge, _ = cfg.resolve_cv_gaps("5min")
        assert purge == cfg.label_span_bars()
        assert "shorter than the longest label span" in caplog.text

    def test_long_explicit_purge_is_respected(self) -> None:
        assert _config(purge_bars=500).resolve_cv_gaps("5min")[0] == 500

    @pytest.mark.parametrize(("timeframe", "bars"), [("1min", 1440), ("5min", 288), ("1h", 24)])
    def test_embargo_is_one_trading_day_of_bars(self, timeframe: str, bars: int) -> None:
        assert _config().resolve_cv_gaps(timeframe)[1] == bars

    def test_embargo_capped_at_quarter_fold(self) -> None:
        cfg = _config(n_splits=5)
        n_rows = 4000  # train 2800 -> fold 560 -> cap 140
        assert cfg.resolve_cv_gaps("1min", n_rows=n_rows)[1] == 140
        assert cfg.resolve_cv_gaps("1h", n_rows=n_rows)[1] == 24  # under the cap

    def test_explicit_embargo_is_respected(self) -> None:
        assert _config(embargo_bars=7).resolve_cv_gaps("1min", n_rows=4000)[1] == 7

    def test_pipeline_config_gets_resolved_gaps_and_weighting(self) -> None:
        cfg = _config(sample_weighting="none")
        cfg.data.bar_timeframe = "15min"
        pipeline = cfg.to_pipeline_config()
        assert pipeline.purge_bars == cfg.label_span_bars()
        assert pipeline.embargo_bars == 96
        assert pipeline.sample_weighting == "none"

    def test_round_trip_keeps_derived_and_explicit_values(self, tmp_path: Path) -> None:
        for cfg in (_config(), _config(purge_bars=70, embargo_bars=9, sample_weighting="none")):
            path = tmp_path / "cfg.yaml"
            cfg.save_yaml(path)
            loaded = ExperimentConfig.from_yaml(path)
            assert loaded.to_dict() == cfg.to_dict()

    def test_invalid_sample_weighting_raises(self) -> None:
        cfg = _config()
        cfg.training.sample_weighting = "bogus"
        with pytest.raises(ValueError, match="sample_weighting"):
            ExperimentConfig.from_dict(cfg.to_dict())


# =============================================================================
# C. SAMPLE UNIQUENESS
# =============================================================================


class TestUniqueness:
    def test_non_overlapping_labels_are_fully_unique(self) -> None:
        spans = LabelSpans(starts=np.array([0, 5, 10]), ends=np.array([4, 9, 14]))
        np.testing.assert_allclose(average_uniqueness(spans), 1.0)

    @pytest.mark.parametrize("k", [2, 3, 7])
    def test_fully_overlapping_labels_share_uniqueness(self, k: int) -> None:
        spans = LabelSpans(starts=np.zeros(k, dtype=int), ends=np.full(k, 4))
        np.testing.assert_allclose(average_uniqueness(spans), 1.0 / k)

    def test_partial_overlap(self) -> None:
        # [0,1] and [1,2]: concurrency 1,2,1 -> each label (1 + 1/2) / 2
        spans = LabelSpans(starts=np.array([0, 1]), ends=np.array([1, 2]))
        np.testing.assert_allclose(average_uniqueness(spans), [0.75, 0.75])

    def test_unknown_ends_do_not_count(self) -> None:
        spans = LabelSpans(starts=np.array([0, 0, 0]), ends=np.array([4, 4, NO_LABEL_END]))
        np.testing.assert_allclose(average_uniqueness(spans), [0.5, 0.5, 1.0])

    def test_training_weights_have_mean_one(self) -> None:
        w = uniqueness_sample_weights(_lagged_spans(500, lag=20))
        assert w.dtype == np.float32
        assert w.mean() == pytest.approx(1.0, rel=1e-5)
        assert w.min() < w.max()  # tail labels overlap less


# =============================================================================
# PREPARED DATA / OOF / TUNER WIRING
# =============================================================================


def _labeled_frame(n: int = 900, lag: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 4)).astype(np.float32), columns=list("abcd"))
    labels = rng.integers(-1, 2, size=n)
    labels[-lag:] = -99
    ends = label_end_positions(labels, np.full(n, lag))
    df["label"] = labels
    df["label_end"] = ends
    return df


def _pipeline_config(**overrides: Any) -> Any:
    from src.core import PipelineConfig

    values: dict[str, Any] = {
        "symbol": "MES",
        "data_path": "dummy.parquet",
        "output_dir": "unused",
        "models": ["xgboost"],
        "horizons": [5],
        "purge_bars": 30,
        "embargo_bars": 10,
        "sequence_length": 16,
    }
    values.update(overrides)
    return PipelineConfig(**values)


class TestPreparedDataSpans:
    @pytest.mark.parametrize("model", ["xgboost", "lstm"])
    def test_spans_and_uniqueness_weights(self, model: str) -> None:
        from src.data.adapters.preparation import UnifiedDataPreparation

        df = _labeled_frame()
        prepared = UnifiedDataPreparation(_pipeline_config()).prepare(df, model_name=model)
        spans = prepared.label_spans("train")
        assert spans is not None and len(spans) == prepared.n_train
        # A windowed sample's span starts at its label bar
        np.testing.assert_array_equal(spans.starts, prepared.train_indices)
        np.testing.assert_array_equal(spans.ends, df["label_end"].to_numpy()[spans.starts])
        np.testing.assert_allclose(prepared.train_weights, uniqueness_sample_weights(spans))
        filtered_spans = prepared.filter_invalid_labels().label_spans("train")
        assert filtered_spans is not None
        assert (filtered_spans.ends >= 0).all()

    def test_weighting_none_leaves_weights_unset(self) -> None:
        from src.data.adapters.preparation import UnifiedDataPreparation

        prep = UnifiedDataPreparation(_pipeline_config(sample_weighting="none"))
        assert prep.prepare(_labeled_frame(), model_name="xgboost").train_weights is None

    def test_label_ends_before_their_row_raise(self) -> None:
        from src.data.adapters.preparation import UnifiedDataPreparation

        df = _labeled_frame()
        df.loc[100, "label_end"] = 50
        with pytest.raises(ValueError, match="before their own row"):
            UnifiedDataPreparation(_pipeline_config()).prepare(df, model_name="xgboost")


class _RecordingCV(PurgedKFold):
    """PurgedKFold that records the spans and folds it produced."""

    calls: list[tuple[LabelSpans | None, list[tuple[np.ndarray, np.ndarray]]]] = []

    def split(self, X, y=None, groups=None, label_end_times=None, label_spans=None):  # noqa: ANN001
        folds = list(super().split(X, y, groups, label_end_times, label_spans))
        _RecordingCV.calls.append((label_spans, folds))
        yield from folds


class _ConstantModel:
    def fit(self, *args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(val_accuracy=0.0, val_f1=0.0)

    def predict(self, X: np.ndarray) -> Any:
        from src.models.base import PredictionResult

        probs = np.full((len(X), 3), 1 / 3)
        return PredictionResult(
            class_predictions=np.zeros(len(X), dtype=int),
            class_probabilities=probs,
            confidence=probs.max(axis=1),
        )


class TestOOFAndTunerUseSpans:
    @pytest.mark.parametrize("model", ["xgboost", "lstm"])
    def test_oof_folds_are_purged_on_label_spans(self, model, monkeypatch) -> None:  # noqa: ANN001
        from src.data.adapters.preparation import UnifiedDataPreparation
        from src.models.training.services import oof_generation
        from src.validation.cv import oof_core, oof_generator

        fake_registry = SimpleNamespace(create=lambda *a, **k: _ConstantModel())
        monkeypatch.setattr(oof_core, "ModelRegistry", fake_registry)
        monkeypatch.setattr(oof_generation, "ModelRegistry", fake_registry)
        monkeypatch.setattr(oof_generation, "PurgedKFold", _RecordingCV)
        monkeypatch.setattr(oof_generator.ModelRegistry, "get_model_info", lambda name: {})
        _RecordingCV.calls = []

        prepared = (
            UnifiedDataPreparation(_pipeline_config(purge_bars=2, embargo_bars=2))
            .prepare(_labeled_frame(n=1500), model_name=model)
            .filter_invalid_labels()
        )
        request = oof_generation.OOFRequest(
            model_name=model,
            horizon=5,
            prepared_data=prepared,
            n_splits=3,
            purge_bars=2,
            embargo_bars=2,
        )
        assert oof_generation.OOFGenerationService().generate_oof(request) is not None
        assert _RecordingCV.calls, "OOF generation never split"
        for spans, folds in _RecordingCV.calls:
            assert spans is not None, "OOF CV ran without label spans"
            for train, test in folds:
                assert _overlapping_train_rows(train, test, spans) == 0

    def test_tuner_receives_spans_and_weights(self, monkeypatch) -> None:  # noqa: ANN001
        from src.data.adapters.preparation import UnifiedDataPreparation
        from src.models.training.services import hyperparameter_tuning as ht

        captured: dict[str, Any] = {}

        class _Tuner:
            def __init__(self, **kwargs: Any) -> None:
                captured["cv"] = kwargs["cv"]

            def tune(
                self, X, y, sample_weights=None, param_space=None, data_rank=2, label_spans=None
            ):  # noqa: ANN001,E501
                captured.update(n=len(y), weights=sample_weights, spans=label_spans)
                return {"best_params": {}, "best_value": 0.0}

        monkeypatch.setattr(ht, "TimeSeriesOptunaTuner", _Tuner)
        prepared = UnifiedDataPreparation(_pipeline_config()).prepare(
            _labeled_frame(), model_name="xgboost"
        )
        ht.HyperparameterTuningService().optimize(
            ht.TuningRequest(
                model_name="xgboost",
                horizon=5,
                prepared_data=prepared,
                n_splits=3,
                embargo_bars=10,
                purge_bars=30,
            )
        )
        assert captured["cv"].config.purge_bars == 30
        assert captured["spans"] is not None and len(captured["spans"]) == captured["n"]
        assert captured["weights"] is not None and len(captured["weights"]) == captured["n"]


# =============================================================================
# FACTORY INTEGRATION (derived gaps, label-end columns, meta-labeling)
# =============================================================================


def _factory_config(data_path: Path, out: Path, mode: str) -> ExperimentConfig:
    cfg = ExperimentConfig(name=f"label_overlap_{mode}", random_seed=42, verbose=0)
    cfg.output_dir = out / cfg.run_id
    cfg.data.symbol = "MES"
    cfg.data.data_path = data_path
    cfg.data.mtf.enabled = False
    cfg.training.models = ["xgboost"]
    cfg.training.horizons = [5]
    cfg.training.training_mode = mode
    cfg.training.n_splits = 3
    cfg.training.build_ensemble = False
    cfg.training.optuna.n_trials = 0
    cfg.evaluation.run_backtest = False
    cfg.bundling.create_bundle = mode == "meta_labeling"
    cfg.bundling.deploy_artifact = False
    return cfg  # purge/embargo left derived


@pytest.fixture(scope="module")
def data_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("label_overlap") / "mes_5min.parquet"
    make_intraday_ohlcv(3000, index_name=None).to_parquet(path)
    return path


@pytest.mark.slow
def test_factory_derives_gaps_and_writes_label_ends(data_path: Path, tmp_path: Path) -> None:
    from src.factory import MLFactory

    cfg = _factory_config(data_path, tmp_path, "standard")
    factory = MLFactory(cfg, verbose=0)
    result = factory.run()
    assert result.success
    assert result.output_dir is not None and result.training_result is not None

    span = cfg.label_span_bars()
    df = pd.read_parquet(result.output_dir / "cache" / "data_pipeline.parquet")
    purge, embargo = factory._cv_gaps  # type: ignore[misc]
    assert purge == span
    fold = int(len(df) * cfg.data.splits.train_ratio) // cfg.training.n_splits
    assert embargo == min(288, int(fold * 0.25))
    assert result.training_result.config.purge_bars == purge

    for col in ("label_end_h5", "label_end"):
        ends = df[col].to_numpy()
        rows = np.arange(len(df))
        valid = df["label"].to_numpy() != -99
        assert (ends[~valid] == NO_LABEL_END).all()
        assert ((ends[valid] > rows[valid]) & (ends[valid] <= rows[valid] + span)).all()
    # label-end columns never become model features
    trainer = result.training_result.model_results["xgboost_h5"].trainer
    assert trainer is not None
    assert not any(c.startswith("label") for c in trainer.feature_columns)


def _load_harness() -> Any:
    spec = importlib.util.spec_from_file_location(
        "mix_match", REPO_ROOT / "scripts" / "mix_match.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_meta_labeling_trains_on_sided_bars_and_serves_identically(
    data_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.factory import MLFactory
    from src.inference.meta_labeling_bundle import MetaLabelingBundle
    from src.models.training.training_ops import TrainingOpsMixin

    primary_oofs: list[Any] = []
    meta_fits: list[tuple[int, int]] = []
    original_oof = TrainingOpsMixin._generate_oof
    original_meta = TrainingOpsMixin._create_meta_model

    def spy_oof(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN001
        oof = original_oof(self, *args, **kwargs)
        primary_oofs.append((oof, args[1], kwargs.get("model_config")))  # (OOF, prepared, cfg)
        return oof

    def spy_meta(self, name: str) -> Any:  # noqa: ANN001
        model = original_meta(self, name)
        fit = model.fit

        def recording_fit(X: np.ndarray, y: np.ndarray) -> Any:
            meta_fits.append((X.shape[0], X.shape[1]))
            del model.fit  # back to the class method so the model stays picklable
            return fit(X, y)

        model.fit = recording_fit
        return model

    monkeypatch.setattr(TrainingOpsMixin, "_generate_oof", spy_oof)
    monkeypatch.setattr(TrainingOpsMixin, "_create_meta_model", spy_meta)

    cfg = _factory_config(data_path, tmp_path, "meta_labeling")
    result = MLFactory(cfg, verbose=0).run()
    assert result.success
    assert result.output_dir is not None and result.training_result is not None
    mr = next(iter(result.training_result.model_results.values()))

    # The meta-model trains only on bars where the primary OOF takes a side
    oof, prepared, oof_model_config = primary_oofs[0]
    # OOF fold models share the deployed primary's configuration (no
    # train/serve skew between what the meta-model learns from and what ships)
    assert mr.trainer is not None
    assert oof_model_config == mr.trainer.model.config
    oof_classes = oof.get_class_predictions()[prepared.train_indices]
    sided = ~np.isnan(oof_classes) & (np.nan_to_num(oof_classes) != 0)
    n_sided = int(sided.sum())
    assert 0 < n_sided < len(oof_classes)
    assert mr.metrics["meta_train_samples"] == n_sided
    final_rows, final_width = meta_fits[-1]
    assert final_rows == n_sided
    # meta features = model input + primary class probabilities + confidence
    assert final_width == prepared.n_features + 3 + 1
    for key in ("primary_precision", "meta_precision", "meta_net_per_trade", "trades_taken"):
        assert key in mr.metrics

    # Train/serve parity of the deployed bundle (same check the e2e harness runs)
    harness = _load_harness()
    assert harness._check_meta_labeling_parity(result, data_path) == []
    bundle = MetaLabelingBundle.load(
        result.output_dir / "bundles" / f"{mr.model_name}_h{mr.horizon}"
    )
    served = bundle.predict_meta(pd.read_parquet(data_path))
    assert not served.trade_mask[served.directions == 0].any()
