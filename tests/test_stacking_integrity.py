"""
Stacking integrity (Phase 116).

(a) OOF fold models never early-stop on the held-out fold they predict:
    the early-stopping set is a purged tail of the fold's train rows
    (tabular, 3D sequence, and windowed 3D/4D paths).
(b) mlp_meta validation metrics are out-of-sample (it no longer trains on X_val).
(c) ridge_meta outputs real probabilities (confident when the signal is easy).
(d) Binary and label-subset calibration / conformal use canonical label mapping.
(e) EnsembleBundle._stack_predictions refuses unequal-length base outputs.
Plus the meta-learner holdout (purge gap, refit, uniform base-model metrics)
and the vectorized vote-agreement feature.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.core.interfaces import PredictionResult
from src.data.adapters.alignment import compute_vote_agreement
from src.inference.ensemble_bundle import EnsembleBundle, EnsembleBundleMetadata
from src.models.calibration import (
    CalibrationConfig,
    ConformalConfig,
    ConformalPredictor,
    ProbabilityCalibrator,
)
from src.models.ensemble import get_meta_learner
from src.models.training.services.ensemble_service import EnsembleService, split_temporal_tail
from src.models.training.services.oof_generation import OOFGenerationService, OOFRequest
from src.validation.cv import PurgedKFold, PurgedKFoldConfig, StackingDataset
from src.validation.cv.early_stopping_split import carve_early_stopping_split
from src.validation.cv.oof_core import CoreOOFGenerator
from src.validation.cv.oof_sequence import SequenceOOFGenerator

N_CLASSES = 3
PURGE = 10
EMBARGO = 5


# ---------------------------------------------------------------------------
# Recording fold model
# ---------------------------------------------------------------------------


class _RecordingModel:
    """Stands in for any registry model; records what fit/predict receive."""

    instances: list[_RecordingModel] = []

    def __init__(self, *_: Any, **__: Any) -> None:
        self.fit_X: np.ndarray | None = None
        self.es_X: np.ndarray | None = None
        self.pred_X: list[np.ndarray] = []
        _RecordingModel.instances.append(self)

    def fit(self, X_train, y_train, X_val, y_val, sample_weights=None, config=None):  # noqa: ANN001
        self.fit_X, self.es_X = np.asarray(X_train), np.asarray(X_val)
        return SimpleNamespace(val_accuracy=0.0, val_f1=0.0)

    def predict(self, X: np.ndarray) -> PredictionResult:
        self.pred_X.append(np.asarray(X))
        n = len(X)
        probs = np.full((n, N_CLASSES), 1.0 / N_CLASSES)
        return PredictionResult(
            class_predictions=np.zeros(n, dtype=int),
            class_probabilities=probs,
            confidence=probs.max(axis=1),
        )


@pytest.fixture
def recording_registry(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingModel]:
    _RecordingModel.instances = []
    fake = SimpleNamespace(create=lambda *a, **k: _RecordingModel())
    monkeypatch.setattr("src.validation.cv.oof_core.ModelRegistry", fake)
    monkeypatch.setattr("src.validation.cv.oof_sequence.ModelRegistry", fake)
    monkeypatch.setattr("src.models.training.services.oof_generation.ModelRegistry", fake)
    return _RecordingModel


def _row_ids(values: np.ndarray, ref_values: np.ndarray, ref_ids: np.ndarray) -> np.ndarray:
    """Invert the fold's affine scaling of the row-id feature (fit on known rows)."""
    slope, intercept = np.polyfit(ref_values, ref_ids, 1)
    return np.rint(slope * values + intercept).astype(int)


def _assert_leak_free(fit_ids, es_ids, val_ids, train_ids) -> None:  # noqa: ANN001
    fit_ids, es_ids, val_ids = set(fit_ids), set(es_ids), set(val_ids)
    assert not es_ids & val_ids, "early-stopping set overlaps the held-out fold"
    assert not fit_ids & val_ids, "fit set overlaps the held-out fold"
    assert not fit_ids & es_ids, "fit set overlaps the early-stopping set"
    assert es_ids <= set(train_ids), "early-stopping rows must come from the fold's train rows"
    # Purge: no fit row within PURGE bars before the early-stopping block
    es_start = min(es_ids)
    assert not any(es_start - PURGE <= r < es_start for r in fit_ids)
    # The tail is contiguous
    assert max(es_ids) - es_start + 1 == len(es_ids)


def _cv() -> PurgedKFold:
    return PurgedKFold(PurgedKFoldConfig(n_splits=3, purge_bars=PURGE, embargo_bars=EMBARGO))


def _labels(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(-1, 2, n)


# ---------------------------------------------------------------------------
# (a) OOF early stopping never sees the held-out fold
# ---------------------------------------------------------------------------


class TestOOFEarlyStoppingIsolation:
    def test_carve_split_is_purged_contiguous_tail(self) -> None:
        train = np.r_[np.arange(0, 190), np.arange(215, 600)]  # middle fold held out
        split = carve_early_stopping_split(train, purge_bars=PURGE)
        assert split.held_out
        _assert_leak_free(split.fit_idx, split.es_idx, np.arange(200, 210), train)
        assert split.es_idx[-1] == 599
        assert len(split.fit_idx) + len(split.es_idx) + len(split.purged_idx) == len(train)

    def test_carve_split_small_fold_falls_back_to_fixed_length(self) -> None:
        split = carve_early_stopping_split(np.arange(40), purge_bars=PURGE)
        assert not split.held_out
        assert set(split.es_idx) <= set(split.fit_idx)
        assert len(split.fit_idx) == 40

    def test_tabular_oof(self, recording_registry: type[_RecordingModel]) -> None:
        n = 600
        X = pd.DataFrame({"row": np.arange(n, dtype=float), "noise": np.random.rand(n)})
        y = pd.Series(_labels(n))
        CoreOOFGenerator(_cv()).generate_tabular_oof(X, y, "logistic", config={})

        folds = list(_cv().split(X, y))
        assert len(recording_registry.instances) == len(folds)
        for model, (train_idx, val_idx) in zip(recording_registry.instances, folds, strict=True):
            pred = np.concatenate(model.pred_X)[:, 0]
            assert model.fit_X is not None and model.es_X is not None
            fit_ids = _row_ids(model.fit_X[:, 0], pred, val_idx)
            es_ids = _row_ids(model.es_X[:, 0], pred, val_idx)
            assert np.array_equal(_row_ids(pred, pred, val_idx), val_idx)
            _assert_leak_free(fit_ids, es_ids, val_idx, train_idx)

    def test_sequence_oof(self, recording_registry: type[_RecordingModel]) -> None:
        n, seq_len = 600, 8
        idx = pd.date_range("2024-01-02", periods=n, freq="5min")
        X = pd.DataFrame({"row": np.arange(n, dtype=float), "noise": np.random.rand(n)}, index=idx)
        y = pd.Series(_labels(n), index=idx)
        SequenceOOFGenerator(_cv()).generate_sequence_oof(
            X, y, "lstm", config={}, seq_len=seq_len, strict_validation=False
        )

        folds = list(_cv().split(X, y))
        for model, (train_idx, val_idx) in zip(recording_registry.instances, folds, strict=True):
            pred = np.concatenate(model.pred_X)[:, -1, 0]  # window's last bar = target
            pred_ids = val_idx[val_idx >= seq_len - 1]
            assert model.fit_X is not None and model.es_X is not None
            fit_ids = _row_ids(model.fit_X[:, -1, 0], pred, pred_ids)
            es_ids = _row_ids(model.es_X[:, -1, 0], pred, pred_ids)
            _assert_leak_free(fit_ids, es_ids, val_idx, train_idx)

    @pytest.mark.parametrize("rank", [3, 4])
    def test_windowed_oof(self, recording_registry: type[_RecordingModel], rank: int) -> None:
        n, seq_len = 600, 6
        ids = np.arange(n, dtype=np.float32)
        shape = (n, seq_len, 2) if rank == 3 else (n, 2, seq_len, 2)
        X = np.random.rand(*shape).astype(np.float32)
        X[..., 0] = ids.reshape((n,) + (1,) * (rank - 2))  # row id in feature 0
        prepared = SimpleNamespace(
            X_train=X,
            y_train=_labels(n),
            train_weights=None,
            data_rank=rank,
            label_spans=lambda split="train": None,  # fixed purge only
        )
        request = OOFRequest(
            model_name="lstm" if rank == 3 else "patchtst",
            horizon=5,
            prepared_data=prepared,  # type: ignore[arg-type]
            n_splits=3,
            purge_bars=PURGE,
            embargo_bars=EMBARGO,
        )
        OOFGenerationService()._generate_windowed_oof(request)

        dummy = pd.DataFrame({"d": np.zeros(n)})
        folds = list(_cv().split(dummy, pd.Series(prepared.y_train)))
        for model, (train_idx, val_idx) in zip(recording_registry.instances, folds, strict=True):
            pred = np.concatenate(model.pred_X).reshape(len(val_idx), -1)[:, 0]
            assert model.fit_X is not None and model.es_X is not None
            fit_ids = _row_ids(model.fit_X.reshape(len(model.fit_X), -1)[:, 0], pred, val_idx)
            es_ids = _row_ids(model.es_X.reshape(len(model.es_X), -1)[:, 0], pred, val_idx)
            _assert_leak_free(fit_ids, es_ids, val_idx, train_idx)


# ---------------------------------------------------------------------------
# (b) mlp_meta validation metrics are out-of-sample
# ---------------------------------------------------------------------------


class TestMLPMetaOutOfSample:
    def test_memorizing_noise_does_not_inflate_val_metrics(self) -> None:
        rng = np.random.default_rng(0)
        X_train, X_val = rng.normal(size=(300, 12)), rng.normal(size=(200, 12))
        y_train, y_val = rng.integers(-1, 2, 300), rng.integers(-1, 2, 200)
        meta = get_meta_learner(
            "mlp_meta",
            hidden_layer_sizes=(128, 128),
            alpha=1e-6,
            learning_rate_init=0.01,
            max_iter=300,
            early_stopping=False,
        )
        metrics = meta.fit(X_train, y_train, X_val, y_val)

        # It memorizes the training noise ...
        assert metrics.train_accuracy > 0.9
        # ... but X_val was never trained on, so val stays near chance (1/3).
        assert metrics.val_accuracy < 0.55
        pred = meta.predict(X_val).class_predictions
        assert metrics.val_accuracy == pytest.approx(float((pred == y_val).mean()))

    def test_temporal_early_stopping_restores_best_epoch(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.normal(size=(400, 6))
        y = np.sign(X[:, 0]).astype(int)
        meta = get_meta_learner("mlp_meta", max_iter=60, n_iter_no_change=5)
        metrics = meta.fit(X[:300], y[:300], X[300:], y[300:])
        curve = metrics.history["val_loss"]
        assert metrics.best_epoch == int(np.argmin(curve))
        assert metrics.val_loss == pytest.approx(min(curve), rel=1e-9)
        assert meta.refit_config() == {"early_stopping": False, "max_iter": metrics.best_epoch + 1}


# ---------------------------------------------------------------------------
# (c) ridge_meta probabilities
# ---------------------------------------------------------------------------


class TestRidgeMetaProbabilities:
    @pytest.mark.parametrize("n_classes", [3, 2], ids=["3class", "binary"])
    def test_easy_signal_gives_confident_probabilities(self, n_classes: int) -> None:
        rng = np.random.default_rng(2)
        n = 600
        cls = rng.integers(0, n_classes, n)
        X = np.eye(n_classes)[cls] * 3.0 + rng.normal(scale=0.2, size=(n, n_classes))
        y = cls - 1 if n_classes == 3 else cls
        meta = get_meta_learner("ridge_meta", n_classes=n_classes)
        meta.fit(X[:400], y[:400], X[400:], y[400:])

        out = meta.predict(X[400:])
        assert out.class_probabilities.shape == (200, n_classes)
        np.testing.assert_allclose(out.class_probabilities.sum(axis=1), 1.0)
        assert (out.class_predictions == y[400:]).mean() > 0.98
        assert out.class_probabilities.max(axis=1).mean() > 0.9

    def test_class_weight_defaults_to_priors(self) -> None:
        meta = get_meta_learner("ridge_meta")
        assert meta._config["class_weight"] is None
        assert get_meta_learner("xgboost_meta")._config["class_weight"] is None


# ---------------------------------------------------------------------------
# (d) Canonical label mapping in calibration + conformal
# ---------------------------------------------------------------------------


def _informative_probs(y_idx: np.ndarray, n_classes: int, seed: int = 3) -> np.ndarray:
    """Probabilities that put most mass on the true class index."""
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(n_classes), size=len(y_idx)) * 0.2
    probs[np.arange(len(y_idx)), y_idx] += 0.8
    result: np.ndarray = probs / probs.sum(axis=1, keepdims=True)
    return result


class TestCanonicalLabelMapping:
    def test_binary_conformal_fit(self) -> None:
        y = np.random.default_rng(4).integers(0, 2, 500)
        probs = _informative_probs(y, 2)
        conformal = ConformalPredictor(ConformalConfig(confidence_level=0.9))
        metrics = conformal.fit(y, probs)
        assert conformal.n_classes == 2
        assert metrics.empirical_coverage >= 0.9
        assert metrics.threshold < 0.5  # informative scores, small threshold

    def test_binary_calibrator(self) -> None:
        y = np.random.default_rng(5).integers(0, 2, 400)
        cal = ProbabilityCalibrator(CalibrationConfig())
        cal.fit(y, _informative_probs(y, 2))
        assert all(c is not None for c in cal._calibrators.values())
        out = cal.calibrate(_informative_probs(y, 2, seed=6))
        np.testing.assert_allclose(out.sum(axis=1), 1.0)

    def test_three_class_validation_without_shorts(self) -> None:
        """Labels {0,1} in 3-class mode are neutral/long = class indices 1 and 2."""
        y = np.random.default_rng(7).integers(0, 2, 500)  # no -1
        probs = _informative_probs(y + 1, 3)

        conformal = ConformalPredictor(ConformalConfig(confidence_level=0.9))
        metrics = conformal.fit(y, probs)
        assert metrics.empirical_coverage >= 0.9
        assert metrics.threshold < 0.5  # a wrong mapping scores the wrong column

        cal = ProbabilityCalibrator(CalibrationConfig())
        cal.fit(y, probs)
        assert cal._calibrators[0] is None
        assert cal._calibrators[1] is not None and cal._calibrators[2] is not None

    def test_conformal_threshold_is_finite_sample_order_statistic(self) -> None:
        y = np.random.default_rng(8).integers(-1, 2, 99)
        probs = _informative_probs(y + 1, 3)
        conformal = ConformalPredictor(ConformalConfig(confidence_level=0.9, method="lac"))
        conformal.fit(y, probs)
        scores = np.sort(1 - probs[np.arange(99), y + 1])
        k = int(np.ceil((99 + 1) * 0.9))  # finite-sample rank
        # Never below the k-th smallest score (the coverage guarantee), and
        # exactly the "higher" empirical quantile at level k / n.
        assert conformal.threshold >= scores[k - 1]
        assert conformal.threshold == pytest.approx(np.quantile(scores, k / 99, method="higher"))

    def test_auto_isotonic_needs_1000_per_class(self) -> None:
        y = np.random.default_rng(9).integers(-1, 2, 1500)
        cal = ProbabilityCalibrator(CalibrationConfig())
        cal.fit(y, _informative_probs(y + 1, 3))
        assert cal._method_used == "sigmoid"


# ---------------------------------------------------------------------------
# (e) _stack_predictions refuses unequal lengths
# ---------------------------------------------------------------------------


def _bundle(n_classes: int = 3) -> EnsembleBundle:
    from src.inference.ensemble_bundle import AlignmentConfig

    return EnsembleBundle(
        meta_learner=None,
        metadata=EnsembleBundleMetadata(
            version="1",
            created_at="now",
            meta_learner_name="ridge_meta",
            base_model_names=["a", "b"],
            horizon=5,
            n_base_models=2,
            n_stacking_features=2 * n_classes + 3,
        ),
        alignment_config=AlignmentConfig(n_classes=n_classes),
    )


class TestStackPredictions:
    def test_unequal_lengths_raise(self) -> None:
        probs = np.random.default_rng(10).dirichlet(np.ones(3), size=50)
        with pytest.raises(ValueError, match="different numbers of rows"):
            _bundle()._stack_predictions({"a": probs, "b": probs[5:]})

    @pytest.mark.parametrize("n_classes", [3, 2])
    def test_equal_lengths_stack_with_derived_features(self, n_classes: int) -> None:
        rng = np.random.default_rng(11)
        a, b = rng.dirichlet(np.ones(n_classes), size=(2, 40))
        stacked = _bundle(n_classes)._stack_predictions({"a": a, "b": b})
        assert stacked.shape == (40, 2 * n_classes + 3)
        agree = (a.argmax(1) == b.argmax(1)).astype(float) * 0.5 + 0.5
        np.testing.assert_allclose(stacked[:, 2 * n_classes + 1], agree)


def test_vote_agreement_matches_loop() -> None:
    rng = np.random.default_rng(12)
    votes = rng.integers(-1, 2, size=(200, 4))
    votes[rng.random((200, 4)) < 0.2] = -999
    votes[0] = -999
    got = compute_vote_agreement(votes, missing=-999).ravel()
    for i, row in enumerate(votes):
        valid = row[row != -999]
        if len(valid) == 0:
            assert np.isnan(got[i])
        else:
            assert got[i] == pytest.approx(
                np.unique(valid, return_counts=True)[1].max() / len(valid)
            )


# ---------------------------------------------------------------------------
# Meta-learner holdout: purge gap, refit, uniform base-model metrics
# ---------------------------------------------------------------------------


class TestMetaLearnerHoldout:
    def test_split_temporal_tail_purges_by_bar(self) -> None:
        rows = np.arange(100, 400)
        head, tail, purge = split_temporal_tail(rows, 0.2, purge_bars=15)
        assert purge == 15
        assert rows[head].max() < rows[tail].min() - 15
        assert len(tail) == 60

    @pytest.mark.parametrize("meta", ["ridge_meta", "xgboost_meta", "mlp_meta"])
    def test_holdout_metrics_and_refit(self, meta: str) -> None:
        rng = np.random.default_rng(13)
        n = 400
        y = rng.integers(-1, 2, n)
        good = _informative_probs(y + 1, 3, seed=14)
        noise = rng.dirichlet(np.ones(3), size=n)
        derived = rng.random((n, 3))
        cols = [f"{m}_prob_{c}" for m in ("good", "noise") for c in ("short", "neutral", "long")]
        data = pd.DataFrame(np.hstack([good, noise, derived]), columns=cols + ["a", "b", "c"])
        data["y_true"] = y
        dataset = StackingDataset(
            data=data,
            model_names=["good", "noise"],
            horizon=5,
            metadata={"row_indices": np.arange(n)},
        )
        config = SimpleNamespace(n_classes=3, purge_bars=12, meta_learner=meta)
        deployed, metrics, base, holdout = EnsembleService()._train_meta_learner(
            dataset, config  # type: ignore[arg-type]
        )

        assert deployed is not None, metrics
        for key in ("val_f1", "val_accuracy", "macro_f1", "log_loss", "brier", "precision_long"):
            assert key in metrics
        assert metrics["val_f1"] == metrics["macro_f1"]
        assert metrics["n_holdout"] == 80
        assert metrics["holdout_purge_bars"] == 12
        assert metrics["n_refit"] == n
        # Base models scored on the same rows with the same code
        assert set(base) == {"good", "noise"}
        assert base["good"]["macro_f1"] > base["noise"]["macro_f1"]
        assert base["good"]["n_samples"] == 80
        # Out-of-sample signals of the evaluation fit on the holdout rows (the
        # backtest of the deployed ensemble replays these)
        assert holdout is not None
        np.testing.assert_array_equal(holdout["row"], np.arange(n - 80, n))
        assert set(holdout["prediction"]) <= {-1, 0, 1}
        assert ((holdout["confidence"] > 0) & (holdout["confidence"] <= 1)).all()
