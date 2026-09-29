"""
Regression tests for the adversarial ML-integrity review fixes.

1. Every horizon trains on its OWN labels (``label_h{h}`` and its label-end
   column); the PreparedData cache is keyed by horizon; ``best_model`` ranks
   within the primary horizon.
2. Walk-forward windows use the deployed model's configuration and receive
   unscaled features (non-finite values filled from the window's training rows).
3. The backtest replays the deployed strategy: the stacking meta-learner's
   purged-holdout signals, never a per-row majority vote.
4. The tuner early-stops on a purged tail of each fold's train rows, never on
   the fold it scores.
5. MDA ranking drops invalid (-99) labels and purges by label spans.
6. The val/test gap is max(purge, embargo).
7. The backtest keeps bars between prediction segments (holding counted in
   bars, ATR from the full price series).
8. The labeler's cost term is calibrated on the training rows only.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.core.label_spans import INVALID_LABEL, label_end_positions

# =============================================================================
# Helpers
# =============================================================================


def _pipeline_config(output_dir: Path, **overrides: Any) -> Any:
    from src.core import PipelineConfig

    values: dict[str, Any] = {
        "symbol": "MES",
        "data_path": "dummy.parquet",
        "output_dir": str(output_dir),
        "models": ["xgboost"],
        "horizons": [5, 20],
        "purge_bars": 30,
        "embargo_bars": 10,
    }
    values.update(overrides)
    return PipelineConfig(**values)


def _two_horizon_frame(n: int = 900, seed: int = 0) -> pd.DataFrame:
    """Features plus DIFFERENT labels / label ends for horizons 5 and 20."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 4)).astype(np.float32), columns=list("abcd"))
    for horizon, lag in ((5, 8), (20, 30)):
        labels = rng.integers(-1, 2, size=n)
        labels[-lag:] = INVALID_LABEL
        df[f"label_h{horizon}"] = labels
        df[f"label_end_h{horizon}"] = label_end_positions(labels, np.full(n, lag))
    df["label"] = df["label_h5"]
    df["label_end"] = df["label_end_h5"]
    return df


# =============================================================================
# 1. Per-horizon labels
# =============================================================================


class TestPerHorizonLabels:
    @pytest.mark.parametrize("model", ["xgboost", "lstm"])
    def test_each_horizon_prepares_its_own_labels_and_spans(
        self, tmp_path: Path, model: str
    ) -> None:
        from src.models.training import UnifiedTrainingOrchestrator

        df = _two_horizon_frame()
        orch = UnifiedTrainingOrchestrator(_pipeline_config(tmp_path, sequence_length=16))
        for horizon in (5, 20):
            prepared = orch._prepare_for_horizon(df, model, horizon)
            assert prepared.train_indices is not None
            np.testing.assert_array_equal(
                prepared.y_train, df[f"label_h{horizon}"].to_numpy()[prepared.train_indices]
            )
            spans = prepared.label_spans("train")
            assert spans is not None
            np.testing.assert_array_equal(
                spans.ends, df[f"label_end_h{horizon}"].to_numpy()[prepared.train_indices]
            )
        # The two horizons' targets really differ (else the test proves nothing)
        h5 = orch._prepare_for_horizon(df, model, 5).y_train
        h20 = orch._prepare_for_horizon(df, model, 20).y_train
        assert (h5 != h20).any()

    def test_cache_key_includes_horizon(self, tmp_path: Path) -> None:
        from src.models.training import UnifiedTrainingOrchestrator

        orch = UnifiedTrainingOrchestrator(_pipeline_config(tmp_path))
        assert orch._prepared_cache_key("xgboost", 5) != orch._prepared_cache_key("xgboost", 20)
        df = _two_horizon_frame()
        a = orch._prepare_with_cache(df, "xgboost", 5)
        b = orch._prepare_with_cache(df, "xgboost", 20)
        assert a is not b
        assert orch._prepare_with_cache(df, "xgboost", 5) is a

    def test_missing_horizon_column_is_an_error_not_a_silent_fallback(self, tmp_path: Path) -> None:
        from src.models.training import UnifiedTrainingOrchestrator

        orch = UnifiedTrainingOrchestrator(_pipeline_config(tmp_path))
        df = _two_horizon_frame().drop(columns=["label_h20", "label_end_h20"])
        # The primary horizon may use the factory's 'label' copy ...
        assert orch._label_column(df.drop(columns=["label_h5"]), 5) == "label"
        # ... a later horizon never trains on the first horizon's labels
        with pytest.raises(ValueError, match="label_h20"):
            orch._label_column(df, 20)

    def test_unscaled_preparation(self, tmp_path: Path) -> None:
        from src.models.training import UnifiedTrainingOrchestrator

        df = _two_horizon_frame()
        orch = UnifiedTrainingOrchestrator(_pipeline_config(tmp_path))
        raw = orch._prepare_for_horizon(df, "xgboost", 5, apply_scaling=False)
        assert raw.scaler is None
        np.testing.assert_allclose(
            raw.X_train, df[list("abcd")].to_numpy()[raw.train_indices], rtol=0, atol=0
        )

    def test_best_model_ranks_within_primary_horizon(self) -> None:
        from src.models.training.unified_orchestrator import (
            ModelTrainingResult,
            TrainingRunResult,
        )

        results = {
            "xgboost_h5": ModelTrainingResult("xgboost", 5, metrics={"val_f1": 0.40}),
            "lstm_h5": ModelTrainingResult("lstm", 5, metrics={"val_f1": 0.45}),
            # Higher score, but a different target: not comparable
            "xgboost_h20": ModelTrainingResult("xgboost", 20, metrics={"val_f1": 0.90}),
        }
        run = TrainingRunResult(
            run_id="r", config=SimpleNamespace(horizons=[5, 20]), model_results=results  # type: ignore[arg-type]
        )
        assert run.best_model == "lstm_h5"
        assert run.best_model_for(20) == "xgboost_h20"


# =============================================================================
# 2. Walk-forward: deployed model config, unscaled windows
# =============================================================================


class _RecordingModel:
    configs: list[dict[str, Any]] = []

    def __init__(self, config: dict[str, Any]) -> None:
        _RecordingModel.configs.append(dict(config))

    def fit(self, X_train, y_train, X_val, y_val, sample_weights=None, config=None):  # noqa: ANN001
        assert np.isfinite(X_train).all() and np.isfinite(X_val).all()

    def predict(self, X):  # noqa: ANN001
        assert np.isfinite(X).all()
        n = len(X)
        probs = np.full((n, 3), 1 / 3)
        return SimpleNamespace(
            class_predictions=np.zeros(n, dtype=int),
            class_probabilities=probs,
            confidence=probs.max(axis=1),
        )


class TestWalkForwardWindows:
    def test_fill_non_finite_uses_training_medians(self) -> None:
        from src.models.training.modes.walk_forward import _fill_non_finite

        X_train = np.array([[1.0, np.nan], [3.0, np.nan], [np.inf, np.nan]], dtype=np.float32)
        X_held = np.array([[np.nan, 5.0]], dtype=np.float32)
        _fill_non_finite(X_train, X_held)
        np.testing.assert_array_equal(X_train, [[1, 0], [3, 0], [2, 0]])
        np.testing.assert_array_equal(X_held, [[2, 5]])

    def test_windows_use_deployed_model_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.models.training.modes.walk_forward as wf
        from src.core.container import TimeSeriesDataContainer
        from src.models.training.config import _ModeConfig

        monkeypatch.setattr(
            wf,
            "ModelRegistry",
            SimpleNamespace(create=lambda name, config: _RecordingModel(config)),
        )
        _RecordingModel.configs = []

        rng = np.random.default_rng(1)
        n = 600
        frame = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
        frame.loc[:40, "f2"] = np.nan  # e.g. a stream that has not started yet
        frame["label_h5"] = rng.integers(-1, 2, size=n)
        frame["sample_weight_h5"] = 1.0
        container = TimeSeriesDataContainer.from_dataframes(
            train_df=frame, horizon=5, feature_columns=["f0", "f1", "f2"]
        )
        deployed = {"n_estimators": 7, "max_depth": 2, "max_epochs": 3}
        trainer = wf.WalkForwardTrainer(
            _ModeConfig(symbol="MES", horizons=[5], models=["xgboost"], output_dir=tmp_path),
            wf.WalkForwardTrainerConfig(n_windows=2, min_train_pct=0.5, test_pct=0.2, gap_bars=5),
            pipeline_config=SimpleNamespace(n_classes=3, sample_weighting="none"),
            model_config=deployed,
        )
        trainer.run(container, save_models=False, save_predictions=False)

        assert len(_RecordingModel.configs) == 2
        for config in _RecordingModel.configs:
            assert {k: config[k] for k in deployed} == deployed
            assert config["n_classes"] == 3


# =============================================================================
# 3. Backtest replays the deployed strategy
# =============================================================================


def _bare_factory() -> Any:
    from src.config.experiment import ExperimentConfig
    from src.factory import MLFactory

    factory = MLFactory.__new__(MLFactory)
    factory.config = ExperimentConfig()
    return factory


class TestDeployedStrategySignals:
    def test_ensemble_backtests_meta_learner_holdout(self) -> None:
        index = pd.date_range("2024-01-02", periods=50, freq="5min")
        df = pd.DataFrame({"close": np.arange(50.0)}, index=index)
        holdout = pd.DataFrame(
            {"row": [40, 41, 45], "prediction": [1, -1, 0], "confidence": [0.7, 0.6, 0.5]}
        )
        training_result = SimpleNamespace(
            ensemble_result=SimpleNamespace(
                trainer=object(), metadata={"holdout_predictions": holdout}
            ),
            best_model="xgboost_h5",
            model_results={},
        )
        preds, strategy = _bare_factory()._extract_predictions(df, training_result)
        assert strategy == "stacking_holdout"
        assert preds is not None
        np.testing.assert_array_equal(preds["datetime"], index[[40, 41, 45]].values)
        np.testing.assert_array_equal(preds["prediction"], [1, -1, 0])

    def test_without_ensemble_backtests_primary_horizon_model(self) -> None:
        from src.models.training.unified_orchestrator import (
            ModelTrainingResult,
            TrainingRunResult,
        )
        from src.validation.cv import OOFPrediction

        index = pd.date_range("2024-01-02", periods=6, freq="5min")
        df = pd.DataFrame({"close": np.arange(6.0)}, index=index)

        def oof(pred: float) -> OOFPrediction:
            frame = pd.DataFrame(
                {
                    "m_pred": [np.nan, pred, pred, pred, pred, np.nan],
                    "m_prob_short": 0.2,
                    "m_prob_neutral": 0.3,
                    "m_prob_long": 0.5,
                }
            )
            return OOFPrediction(
                model_name="m",
                predictions=frame,
                fold_info=[],
                coverage=4 / 6,
                original_indices=np.arange(1, 5),
                n_total_samples=6,
            )

        results = {
            "m_h5": ModelTrainingResult("m", 5, metrics={"val_f1": 0.3}, oof_prediction=oof(1)),
            "m_h20": ModelTrainingResult("m", 20, metrics={"val_f1": 0.9}, oof_prediction=oof(-1)),
        }
        run = TrainingRunResult(
            run_id="r", config=SimpleNamespace(horizons=[5, 20]), model_results=results  # type: ignore[arg-type]
        )
        preds, strategy = _bare_factory()._extract_predictions(df, run)
        assert strategy == "oof:m_h5"  # the backtest's barriers are h5's
        assert preds is not None
        np.testing.assert_array_equal(preds["prediction"], [1, 1, 1, 1])
        np.testing.assert_array_equal(preds["datetime"], index[1:5].values)


# =============================================================================
# 4. Tuner: early stopping never on the scored fold
# =============================================================================


class _TunerStub:
    calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def __init__(self) -> None:
        self._fit: tuple[np.ndarray, np.ndarray] | None = None

    def fit(self, X_train, y_train, X_val, y_val, sample_weights=None, config=None):  # noqa: ANN001
        self._fit = (X_train[:, 0].astype(int), X_val[:, 0].astype(int))

    def predict(self, X):  # noqa: ANN001
        assert self._fit is not None
        _TunerStub.calls.append((self._fit[0], self._fit[1], X[:, 0].astype(int)))
        return SimpleNamespace(class_predictions=np.zeros(len(X), dtype=int))


class TestTunerEarlyStopping:
    def test_early_stopping_rows_are_purged_train_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.validation.cv.cv_tuner as cv_tuner
        from src.validation.cv import PurgedKFold, PurgedKFoldConfig, TimeSeriesOptunaTuner

        monkeypatch.setattr(
            cv_tuner, "ModelRegistry", SimpleNamespace(create=lambda name, config: _TunerStub())
        )
        _TunerStub.calls = []
        n, purge = 600, 12
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"row": np.arange(n, dtype=float), "noise": rng.normal(size=n)})
        y = pd.Series(rng.integers(-1, 2, size=n))
        cv = PurgedKFold(PurgedKFoldConfig(n_splits=3, purge_bars=purge, embargo_bars=5))
        tuner = TimeSeriesOptunaTuner("xgboost", cv, n_trials=1, metric="f1_weighted")
        tuner.tune(X, y)

        splits = list(cv.split(X, y))
        assert len(_TunerStub.calls) == len(splits)
        for (fit_rows, es_rows, scored_rows), (train_idx, val_idx) in zip(
            _TunerStub.calls, splits, strict=True
        ):
            np.testing.assert_array_equal(scored_rows, val_idx)
            assert not np.intersect1d(es_rows, val_idx).size, "early stopping on scored fold"
            assert np.isin(es_rows, train_idx).all() and np.isin(fit_rows, train_idx).all()
            assert not np.intersect1d(fit_rows, es_rows).size
            # purge gap between the fit rows and the early-stopping tail
            before = fit_rows[fit_rows < es_rows.min()]
            assert es_rows.min() - before.max() > purge


# =============================================================================
# 5. MDA ranking: invalid labels dropped, label-span purging
# =============================================================================


class TestMdaSplits:
    def test_mda_cv_gets_spans_and_no_invalid_labels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.models.training.feature_selection as fs

        captured: dict[str, Any] = {}

        class _SpyKFold:
            def __init__(self, config: Any) -> None:
                pass

            def split(self, X, y, label_spans=None):  # noqa: ANN001
                captured.update(X=X, y=y, spans=label_spans)
                raise RuntimeError("captured")  # the ranking falls back; we only inspect

        monkeypatch.setattr(fs, "PurgedKFold", _SpyKFold)

        df = _two_horizon_frame(n=600)
        df.loc[100:119, "label_h5"] = INVALID_LABEL  # mid-series invalid labels
        df.loc[100:119, "label_end_h5"] = -1
        host = fs.FeatureSelectionMixin()
        host.config = SimpleNamespace(horizons=[5, 20], purge_bars=10, embargo_bars=5)  # type: ignore[assignment]
        assert host._compute_mda_ranking(df, list("abcd")) is None  # spy aborted the CV

        y = captured["y"]
        assert (y != INVALID_LABEL).all()
        spans = captured["spans"]
        assert spans is not None and len(spans) == len(y)
        kept = np.flatnonzero(df["label_h5"].to_numpy() != INVALID_LABEL)
        np.testing.assert_array_equal(spans.starts, kept)
        np.testing.assert_array_equal(spans.ends, df["label_end_h5"].to_numpy()[kept])


# =============================================================================
# 6. Split gap after validation
# =============================================================================


def test_val_test_gap_covers_purge_and_embargo(tmp_path: Path) -> None:
    from src.data.adapters.preparation import UnifiedDataPreparation

    for purge, embargo in ((30, 10), (10, 30)):
        prep = UnifiedDataPreparation(
            _pipeline_config(tmp_path, purge_bars=purge, embargo_bars=embargo)
        )
        _train_end, _val_start, val_end, test_start = prep._split_bounds(1000)
        assert test_start - val_end == max(purge, embargo)


# =============================================================================
# 7. Backtest over gaps between prediction segments
# =============================================================================


def _flat_prices(n: int) -> pd.DataFrame:
    ts = pd.date_range("2024-01-02 14:30", periods=n, freq="5min", tz="UTC")
    close = np.full(n, 4500.0) + np.sin(np.arange(n)) * 0.25
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000,
        }
    )


NO_BREAKERS = {
    "enable_market_hours_filter": False,
    "consecutive_loss_limit": 10**9,
    "max_drawdown_threshold": 10.0,
    "daily_loss_threshold": 10.0,
}


class TestBacktestPredictionGaps:
    def test_holding_is_counted_in_bars_across_a_gap(self) -> None:
        from src.inference.backtesting import BacktestConfig, Backtester
        from src.inference.backtesting.backtest import ExitReason

        prices = _flat_prices(80)
        covered = np.r_[np.arange(0, 21), np.arange(40, 61)]  # no predictions for 21..39
        signal = np.zeros(len(covered), dtype=int)
        signal[covered == 20] = 1  # long decided at bar 20
        preds = pd.DataFrame(
            {"timestamp": prices["timestamp"].iloc[covered].to_list(), "prediction": signal}
        )
        cfg = BacktestConfig(max_holding_period=5, **NO_BREAKERS)
        result = Backtester(preds, prices, cfg).run()

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.exit_reason == ExitReason.MAX_HOLDING.value
        # 5 BARS after the signal bar (bar 25, inside the gap) — not 5 rows
        assert pd.Timestamp(trade.exit_time) == prices["timestamp"].iloc[25]
        # every bar of the predictions' span is simulated
        assert len(result.equity_curve.timestamps) == 61

    def test_atr_uses_bars_before_the_first_prediction(self) -> None:
        from src.inference.backtesting import BacktestConfig, Backtester

        prices = _flat_prices(80)
        preds = pd.DataFrame(
            {"timestamp": prices["timestamp"].iloc[30:].to_list(), "prediction": 0}
        )
        data = Backtester(preds, prices, BacktestConfig(**NO_BREAKERS))._align_data()
        assert data["timestamp"].iloc[0] == prices["timestamp"].iloc[30]
        assert np.isfinite(data["atr"].iloc[0])  # warm-up came from bars 0..29
        assert data["has_signal"].all()


# =============================================================================
# 8. Labeler cost calibrated on the training rows
# =============================================================================


def test_label_cost_uses_training_rows_only() -> None:
    from src.data.labeling import TripleBarrierConfig, TripleBarrierLabeler
    from src.data.labeling.triple_barrier import compute_cost_in_atr

    n = 1000
    rng = np.random.default_rng(3)
    # Calm first 70%, then a volatility explosion (val/test period)
    scale = np.r_[np.full(700, 0.5), np.full(300, 5.0)]
    close = 4500 + np.cumsum(rng.normal(0, 1, n) * scale)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + scale,
            "low": close - scale,
            "close": close,
            "volume": 1000.0,
        }
    )

    def cost(fraction: float) -> float:
        config = TripleBarrierConfig(
            horizon=12, atr_column=None, symbol="MES", cost_calibration_fraction=fraction
        )
        labeler = TripleBarrierLabeler(config)
        return float(labeler.compute_labels(df, horizon=12).metadata["cost_in_atr"][0])

    labeler = TripleBarrierLabeler(TripleBarrierConfig(atr_column=None))
    atr = np.asarray(labeler.compute_atr(df), dtype=float)
    assert cost(0.7) == pytest.approx(compute_cost_in_atr("MES", atr[:700]))
    assert cost(0.7) > cost(1.0)  # later volatility no longer shrinks the cost term
