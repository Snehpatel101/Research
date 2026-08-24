"""Validate the control instrument itself, and pin the metric ruling.

If these fail, every statistical claim made later in the program is
unsupported — so these run first and are never skipped.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from tests.controls import (
    featurize,
    majority_baseline,
    make_noise,
    make_signal,
    temporal_split,
)


def _fit_predict(X_tr, y_tr, X_te, y_te):
    """Train the repo's own xgboost through the real registry."""
    import src.models  # noqa: F401  (registration side effect)
    from src.models.registry import ModelRegistry

    model = ModelRegistry.create("xgboost")
    model.fit(X_tr, y_tr, X_te, y_te)
    return model.predict(X_te).class_predictions


@pytest.fixture(scope="module")
def noise_result():
    X, y, _ = featurize(make_noise())
    X_tr, X_te, y_tr, y_te = temporal_split(X, y)
    return y_te, _fit_predict(X_tr, y_tr, X_te, y_te), majority_baseline(y_tr, y_te)


@pytest.fixture(scope="module")
def signal_result():
    X, y, _ = featurize(make_signal())
    X_tr, X_te, y_tr, y_te = temporal_split(X, y)
    return y_te, _fit_predict(X_tr, y_tr, X_te, y_te), majority_baseline(y_tr, y_te)


class TestNoiseControl:
    """On unpredictable data, a model must NOT appear skilful."""

    def test_model_does_not_beat_baseline_accuracy(self, noise_result):
        y_te, pred, base = noise_result
        acc_model = accuracy_score(y_te, pred)
        acc_base = accuracy_score(y_te, base)
        assert acc_model <= acc_base + 0.02, (
            f"Model beat the majority baseline on PURE NOISE "
            f"({acc_model:.4f} > {acc_base:.4f}). That indicates leakage, "
            f"not skill."
        )

    def test_mcc_is_approximately_zero(self, noise_result):
        y_te, pred, _ = noise_result
        mcc = matthews_corrcoef(y_te, pred)
        assert abs(mcc) < 0.10, f"MCC={mcc:.4f} on pure noise implies leakage."


class TestSignalControl:
    """On genuinely learnable data, a model MUST demonstrate skill."""

    def test_model_beats_baseline_accuracy(self, signal_result):
        y_te, pred, base = signal_result
        acc_model = accuracy_score(y_te, pred)
        acc_base = accuracy_score(y_te, base)
        assert acc_model > acc_base + 0.05, (
            f"Model failed to beat the baseline on LEARNABLE data "
            f"({acc_model:.4f} vs {acc_base:.4f}). The pipeline is broken."
        )

    def test_mcc_shows_real_skill(self, signal_result):
        y_te, pred, _ = signal_result
        mcc = matthews_corrcoef(y_te, pred)
        assert mcc > 0.10, f"MCC={mcc:.4f} — no skill on learnable data."

    def test_shuffling_labels_destroys_skill(self):
        """Negative control: break the X->y relationship, skill must vanish."""
        X, y, _ = featurize(make_signal())
        X_tr, X_te, y_tr, y_te = temporal_split(X, y)
        shuffled = np.random.default_rng(7).permutation(y_tr)
        pred = _fit_predict(X_tr, shuffled, X_te, y_te)
        mcc = matthews_corrcoef(y_te, pred)
        assert abs(mcc) < 0.10, (
            f"MCC={mcc:.4f} after shuffling training labels. Skill that "
            f"survives label destruction is leakage."
        )


class TestMetricValidity:
    """Pins the Phase 0 ruling on which metrics may carry a claim."""

    def test_macro_f1_credits_a_no_skill_model_on_noise(self, noise_result):
        """Documents WHY macro-F1 may not be a headline metric here.

        This asserts the defect exists rather than that it is absent: the
        rule "use accuracy-vs-majority and MCC" is only justified while this
        remains true. If a future change makes macro-F1 trustworthy, this
        test fails loudly and the ruling gets revisited deliberately.
        """
        y_te, pred, base = noise_result
        gain = f1_score(y_te, pred, average="macro") - f1_score(y_te, base, average="macro")
        assert gain > 0.01, (
            "macro-F1 no longer rewards a no-skill model on pure noise. "
            "Revisit the Phase 0 metric ruling in docs/program/STAGE_PLAN.md."
        )

    def test_mcc_does_separate_noise_from_signal(self, noise_result, signal_result):
        """The metric we DO rely on must actually discriminate."""
        mcc_noise = matthews_corrcoef(noise_result[0], noise_result[1])
        mcc_signal = matthews_corrcoef(signal_result[0], signal_result[1])
        assert mcc_signal > mcc_noise + 0.10, (
            f"MCC failed to separate signal ({mcc_signal:.4f}) from noise "
            f"({mcc_noise:.4f}); it cannot carry a go/no-go claim."
        )
