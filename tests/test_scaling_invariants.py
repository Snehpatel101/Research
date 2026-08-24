"""Pin the invariant that makes double-scaling harmless (Stage 5).

Phase 0 (Agent 4) reported "global scaler fit precedes CV folds" as HIGH
severity preprocessing leakage. Stage 5 REFUTED that for the CV/OOF path —
see docs/program/evidence/F4_double_scaling.py.

The refutation is real but CONDITIONAL. `PreparedData.X_train` genuinely is
pre-scaled on the whole train split before the CV machinery (which calls it
"raw, unscaled") scales it again per fold. That is harmless only because:

    every reachable scaler is AFFINE

Affine pre-transform + affine fold-scaling fit on fold-train data cancels
exactly:

    pre  : z = (x - m) / s                    [global m, s]
    fold : w = (z - med(z_tr)) / IQR(z_tr)
         =  (x - med(x_tr)) / IQR(x_tr)       [m and s cancel]

`ScalerType` also declares QUANTILE, which is a rank transform and is NOT
affine. If it ever becomes reachable as the pre-scaler, the cancellation
breaks and the contamination becomes genuine leakage.

These tests exist so that stops being an unstated assumption. If someone
widens the scaler set, these fail and point at this file.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.validation.cv.fold_scaling import FoldAwareScaler

CUT = 800
N = 1000


@pytest.fixture
def regime_shift_data() -> np.ndarray:
    """Train rows whose later portion has 10x the volatility.

    Chosen deliberately: if pre-scaling contamination could matter anywhere,
    it would matter here, where fold-validation looks nothing like fold-train.
    """
    rng = np.random.default_rng(0)
    early = rng.normal(0.0, 1.0, (CUT, 1))
    late = rng.normal(0.0, 10.0, (N - CUT, 1))
    return np.vstack([early, late]).astype(np.float64)


def _affine_prescale(X: np.ndarray) -> np.ndarray:
    """What preparation.py does: robust scaling fit on the WHOLE split."""
    med = np.median(X, axis=0)
    q75, q25 = np.percentile(X, 75, axis=0), np.percentile(X, 25, axis=0)
    iqr = np.where((q75 - q25) > 1e-8, q75 - q25, 1.0)
    return (X - med) / iqr


class TestAffineCancellation:
    """The property the no-leakage conclusion rests on."""

    @pytest.mark.parametrize("method", ["robust", "standard"])
    def test_prescaling_cancels_exactly(self, regime_shift_data, method):
        X = regime_shift_data

        direct = FoldAwareScaler(method=method).fit_transform_fold(X[:CUT], X[CUT:])
        Xp = _affine_prescale(X)
        doubled = FoldAwareScaler(method=method).fit_transform_fold(Xp[:CUT], Xp[CUT:])

        np.testing.assert_allclose(
            direct.X_train_scaled,
            doubled.X_train_scaled,
            rtol=1e-9,
            atol=1e-9,
            err_msg=(
                f"Affine pre-scaling did NOT cancel for method={method!r}. "
                f"If this fails, the double-scaling in oof_generation.py has "
                f"become genuine preprocessing leakage — see this module's "
                f"docstring."
            ),
        )

    @pytest.mark.parametrize("method", ["robust", "standard"])
    def test_cancellation_holds_for_validation_rows_too(self, regime_shift_data, method):
        X = regime_shift_data
        direct = FoldAwareScaler(method=method).fit_transform_fold(X[:CUT], X[CUT:])
        Xp = _affine_prescale(X)
        doubled = FoldAwareScaler(method=method).fit_transform_fold(Xp[:CUT], Xp[CUT:])
        np.testing.assert_allclose(direct.X_val_scaled, doubled.X_val_scaled, rtol=1e-9, atol=1e-9)


class TestScalerSetStaysAffine:
    """Guard the *reachability* half of the invariant."""

    def test_fold_scaler_rejects_non_affine_methods(self):
        """FoldAwareScaler must not silently accept a rank transform.

        Both scalers refuse at CONSTRUCTION rather than at fit time, which is
        stronger than this test originally assumed — an invalid scaler cannot
        even be built, let alone reach data.
        """
        with pytest.raises(ValueError, match="Unknown scaling method"):
            FoldAwareScaler(method="quantile")

    def test_adapter_scaler_rejects_non_affine_methods(self):
        """AdapterScaler is the PRE-scaler; a non-affine one breaks cancellation."""
        from src.data.adapters.scaling import AdapterScaler, ScalerConfig

        with pytest.raises(ValueError, match="Invalid scaling method"):
            AdapterScaler(ScalerConfig(method="quantile"))

    def test_known_affine_methods_are_accepted(self):
        """The complement: the affine set must keep working."""
        from src.data.adapters.scaling import AdapterScaler, ScalerConfig

        X = np.random.default_rng(0).normal(size=(100, 3))
        for method in ("robust", "standard", "minmax"):
            out = AdapterScaler(ScalerConfig(method=method)).fit_transform(X)
            assert out.shape == X.shape, f"{method} changed shape"


class TestNonAffineWouldBreakIt:
    """Demonstrate the failure mode we are guarding against.

    Not a test of repo code — a test of the REASONING. If a rank transform
    were used as the pre-scaler, cancellation would fail and the
    contamination would become real. This makes the guard tests meaningful
    rather than arbitrary.
    """

    def test_rank_prescaling_does_not_cancel(self, regime_shift_data):
        from sklearn.preprocessing import QuantileTransformer

        X = regime_shift_data
        direct = FoldAwareScaler(method="robust").fit_transform_fold(X[:CUT], X[CUT:])

        qt = QuantileTransformer(n_quantiles=100, output_distribution="normal")
        Xq = qt.fit_transform(X)  # non-affine, fit on the WHOLE split
        doubled = FoldAwareScaler(method="robust").fit_transform_fold(Xq[:CUT], Xq[CUT:])

        max_diff = float(np.abs(direct.X_train_scaled - doubled.X_train_scaled).max())
        assert max_diff > 1e-3, (
            "A rank transform is expected NOT to cancel. If it now does, the "
            "reasoning behind TestAffineCancellation needs revisiting."
        )
