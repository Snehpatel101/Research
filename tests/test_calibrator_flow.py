"""
Regression tests for the calibration flow.

Covers three claims:

1. ProbabilityCalibrator single-class pass-through (Phase 69): fitting with
   validation labels that are all one class must not crash — the per-class
   calibrators are skipped (None = pass-through) and calibrate() leaves the
   probabilities effectively unchanged.

2. Calibrator storage regression (fixed this session):
   TrainingOpsMixin._calibrate_model must ALWAYS store the fitted calibrator
   on the result object, even when the result already has a ``calibrator``
   attribute (the canonical dataclass field defaults to None — the old
   ``if not hasattr(...)`` guard silently dropped the fitted calibrator).

3. Parallel conversion copy: the canonical
   unified_orchestrator.ModelTrainingResult dataclass accepts a
   ``calibrator=`` kwarg, and the conversion pattern
   ``getattr(service_result, "calibrator", None)`` carries a dynamically
   attached calibrator from the service-flavor result onto the canonical one.

All tests use tiny in-memory numpy arrays; no models are trained and no data
files are loaded.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.models.calibration import CalibrationConfig, CalibrationMetrics, ProbabilityCalibrator
from src.models.training.services.model_training import (
    ModelTrainingResult as ServiceModelTrainingResult,
)
from src.models.training.training_ops import TrainingOpsMixin
from src.models.training.unified_orchestrator import ModelTrainingResult

RNG = np.random.default_rng(42)


def _make_probs(n: int, n_classes: int = 3) -> np.ndarray:
    """Random probability rows that sum to 1 and stay well above clip epsilon."""
    raw = RNG.random((n, n_classes)) + 0.05
    result: np.ndarray = raw / raw.sum(axis=1, keepdims=True)
    return result


# ---------------------------------------------------------------------------
# 1. ProbabilityCalibrator single-class pass-through (Phase 69)
# ---------------------------------------------------------------------------


class TestSingleClassPassThrough:
    """Phase 69: all-one-class validation labels must not crash calibration."""

    @pytest.mark.parametrize("label", [-1, 0, 1])
    def test_fit_all_one_class_does_not_crash(self, label: int) -> None:
        n = 200
        y_val = np.full(n, label)
        probs = _make_probs(n)

        calibrator = ProbabilityCalibrator(CalibrationConfig())
        metrics = calibrator.fit(y_val, probs)

        assert isinstance(metrics, CalibrationMetrics)
        assert calibrator.is_fitted

    @pytest.mark.parametrize("label", [-1, 1])
    def test_single_class_skips_all_per_class_calibrators(self, label: int) -> None:
        """Every one-vs-rest target is constant, so every class is pass-through."""
        n = 150
        y_val = np.full(n, label)
        probs = _make_probs(n)

        calibrator = ProbabilityCalibrator(CalibrationConfig())
        calibrator.fit(y_val, probs)

        assert set(calibrator._calibrators.keys()) == {0, 1, 2}
        assert all(c is None for c in calibrator._calibrators.values())

    def test_single_class_predict_path_is_identity(self) -> None:
        """calibrate() after a single-class fit must leave probabilities unchanged."""
        n = 150
        y_val = np.full(n, 1)
        probs_val = _make_probs(n)
        probs_test = _make_probs(80)

        calibrator = ProbabilityCalibrator(CalibrationConfig())
        calibrator.fit(y_val, probs_val)
        calibrated = calibrator.calibrate(probs_test)

        assert calibrated.shape == probs_test.shape
        # Pass-through + renormalize + epsilon clip => effectively identity.
        np.testing.assert_allclose(calibrated, probs_test, atol=1e-5)

    def test_partial_single_class_skips_only_missing_class(self) -> None:
        """With labels {0,1} out of 3 classes, only class 2 is pass-through."""
        n = 200
        y_val = np.array([0, 1] * (n // 2))
        probs = _make_probs(n)

        calibrator = ProbabilityCalibrator(CalibrationConfig())
        calibrator.fit(y_val, probs)

        assert calibrator._calibrators[0] is not None
        assert calibrator._calibrators[1] is not None
        assert calibrator._calibrators[2] is None  # pass-through

        calibrated = calibrator.calibrate(_make_probs(60))
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-9)
        assert np.all(calibrated > 0.0)
        assert np.all(calibrated < 1.0)

    def test_calibrate_before_fit_raises(self) -> None:
        calibrator = ProbabilityCalibrator(CalibrationConfig())
        with pytest.raises(RuntimeError, match="not fitted"):
            calibrator.calibrate(_make_probs(10))


# ---------------------------------------------------------------------------
# 2. TrainingOpsMixin._calibrate_model storage regression
# ---------------------------------------------------------------------------


class _Host(TrainingOpsMixin):
    """Minimal orchestrator-flavored host exposing only what _calibrate_model needs."""

    def __init__(self, min_samples: int = 10, method: str = "auto") -> None:
        self.config = SimpleNamespace(  # type: ignore[assignment]
            calibration_min_samples=min_samples,
            calibration_method=method,
        )


class _FakeTrainer:
    """Fake trainer whose predict_proba returns a fixed-shape probability array."""

    def __init__(self, probas: np.ndarray) -> None:
        self._probas = probas

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._probas


def _prepared_stub(n: int, y_val: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(X_val=RNG.random((n, 4)), y_val=y_val)


class TestCalibrateModelStoresCalibrator:
    """_calibrate_model must always store the fitted calibrator on the result."""

    def test_stores_calibrator_when_field_preexists_as_none(self) -> None:
        """Regression: a result that ALREADY has calibrator=None (the canonical
        dataclass default) must still receive the fitted calibrator. The old
        ``if not hasattr(result, "calibrator")`` guard silently skipped it.

        Note: predict_proba here returns shape (n, 1) because _calibrate_model
        collapses any (n, >1) output to 1D, which ProbabilityCalibrator.fit
        rejects (see TestCalibrateModelGuards for that documented behavior).
        """
        n = 60
        trainer = _FakeTrainer(np.full((n, 1), 0.6))
        result = ModelTrainingResult(model_name="fake", horizon=5, trainer=trainer)
        assert result.calibrator is None  # dataclass field preexists as None

        host = _Host(min_samples=10)
        host._calibrate_model(result, _prepared_stub(n, np.zeros(n, dtype=int)), "fake")

        assert result.calibrator is not None
        assert isinstance(result.calibrator, ProbabilityCalibrator)
        assert result.calibrator.is_fitted
        assert isinstance(result.calibration_metrics, CalibrationMetrics)

    def test_stores_calibrator_on_service_flavor_result(self) -> None:
        """Service-flavor results declare calibrator fields (default None)."""
        n = 60
        trainer = _FakeTrainer(np.full((n, 1), 0.55))
        service_result = ServiceModelTrainingResult(model_name="fake", horizon=5, trainer=trainer)
        assert service_result.calibrator is None  # declared field, not dynamic

        host = _Host(min_samples=10)
        host._calibrate_model(service_result, _prepared_stub(n, np.zeros(n, dtype=int)), "fake")

        assert service_result.calibrator is not None
        assert service_result.calibrator.is_fitted


class TestCalibrateModelGuards:
    """Guard paths leave result.calibrator as None and never raise."""

    def test_skips_when_trainer_lacks_predict_proba(self) -> None:
        result = ModelTrainingResult(model_name="fake", horizon=5, trainer=object())
        host = _Host(min_samples=10)
        host._calibrate_model(result, _prepared_stub(60, np.zeros(60, dtype=int)), "fake")
        assert result.calibrator is None

    def test_skips_when_validation_data_missing(self) -> None:
        result = ModelTrainingResult(
            model_name="fake", horizon=5, trainer=_FakeTrainer(np.full((60, 1), 0.5))
        )
        host = _Host(min_samples=10)
        prepared = SimpleNamespace(X_val=None, y_val=None)
        host._calibrate_model(result, prepared, "fake")
        assert result.calibrator is None

    def test_skips_when_too_few_validation_samples(self) -> None:
        n = 5
        result = ModelTrainingResult(
            model_name="fake", horizon=5, trainer=_FakeTrainer(np.full((n, 1), 0.5))
        )
        host = _Host(min_samples=10)
        host._calibrate_model(result, _prepared_stub(n, np.zeros(n, dtype=int)), "fake")
        assert result.calibrator is None

    def test_multiclass_probas_are_calibrated(self) -> None:
        """Multiclass (n, n_classes) predict_proba output IS calibrated.

        Regression: _calibrate_model used to collapse the matrix to 1D via
        ``val_probas[:, 1]``; ProbabilityCalibrator.fit requires 2D, raised
        ValueError, and the broad except silently dropped calibration for
        every multiclass model.
        """
        n = 60
        rng = np.random.default_rng(0)
        trainer = _FakeTrainer(_make_probs(n, n_classes=3))
        result = ModelTrainingResult(model_name="fake", horizon=5, trainer=trainer)
        host = _Host(min_samples=10)

        y_val = rng.choice([-1, 0, 1], size=n)
        host._calibrate_model(result, _prepared_stub(n, y_val), "fake")

        assert result.calibrator is not None


# ---------------------------------------------------------------------------
# 3. Parallel conversion copy: calibrator kwarg + getattr pattern
# ---------------------------------------------------------------------------


class TestParallelConversionCalibratorCopy:
    """Canonical ModelTrainingResult carries the service result's calibrator."""

    def test_canonical_dataclass_accepts_calibrator_kwarg(self) -> None:
        sentinel = object()
        result = ModelTrainingResult(model_name="xgboost", horizon=5, calibrator=sentinel)
        assert result.calibrator is sentinel

    def test_canonical_dataclass_calibrator_defaults_to_none(self) -> None:
        result = ModelTrainingResult(model_name="xgboost", horizon=5)
        assert result.calibrator is None

    def test_conversion_pattern_copies_dynamic_calibrator(self) -> None:
        """Mirror of the parallel-branch conversion in training_ops.py:
        calibrator=getattr(service_result, "calibrator", None)."""
        sentinel = ProbabilityCalibrator(CalibrationConfig())
        service_result = ServiceModelTrainingResult(model_name="xgboost", horizon=5, trainer=None)
        service_result.calibrator = sentinel  # dynamic attr, as set by _calibrate_model

        converted = ModelTrainingResult(
            model_name=service_result.model_name,
            horizon=service_result.horizon,
            metrics=service_result.metrics,
            trainer=service_result.trainer,
            training_time_seconds=service_result.training_time_seconds,
            n_features=service_result.n_features,
            data_rank=service_result.data_rank,
            calibrator=getattr(service_result, "calibrator", None),
        )

        assert converted.calibrator is sentinel

    def test_conversion_pattern_defaults_none_when_not_calibrated(self) -> None:
        service_result = ServiceModelTrainingResult(model_name="xgboost", horizon=5, trainer=None)
        assert service_result.calibrator is None  # declared field, defaults None

        converted = ModelTrainingResult(
            model_name=service_result.model_name,
            horizon=service_result.horizon,
            calibrator=getattr(service_result, "calibrator", None),
        )

        assert converted.calibrator is None
