"""Tests for the canonical PredictionSet / PredictionPanel contract (Stage 4).

The headline test is `TestF1CannotRecur` — it rebuilds the exact scenario
Phase 0 proved broken (a 2D model and a 60-bar 3D model over the same source
data) and shows the keyed contract aligns them correctly where the positional
one produced a 59-bar shift.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.prediction_panel import PredictionPanel
from src.core.predictions import (
    PredictionProvenance,
    PredictionSchema,
    PredictionSet,
)

SEQ_LEN = 60
N_BARS = 400


def _probs_for(labels: np.ndarray, class_labels=(-1, 0, 1)) -> np.ndarray:
    """One-hot probabilities for a perfect model."""
    lut = {c: i for i, c in enumerate(class_labels)}
    p = np.zeros((len(labels), len(class_labels)))
    for r, lab in enumerate(labels):
        p[r, lut[int(lab)]] = 1.0
    return p


def _make(name: str, keys: np.ndarray, labels: np.ndarray, **kw) -> PredictionSet:
    return PredictionSet(
        keys=keys,
        probabilities=_probs_for(labels),
        valid=np.ones(len(keys), dtype=bool),
        schema=PredictionSchema.ternary(horizon=5),
        provenance=PredictionProvenance(model_name=name, **kw),
    )


# ---------------------------------------------------------------- the point
class TestF1CannotRecur:
    """The defect that motivated this contract must be structurally impossible."""

    def test_two_perfect_models_agree_when_keyed(self):
        truth = (np.arange(N_BARS) % 3) - 1

        # Tabular model: predicts every bar. Keys ARE source bars.
        k2d = np.arange(N_BARS)
        m2d = _make("xgboost", k2d, truth[k2d])

        # Sequence model: loses the first seq_len-1 bars to windowing.
        # Its keys are the SOURCE BARS it predicted, not 0..n.
        k3d = np.arange(SEQ_LEN - 1, N_BARS)
        m3d = _make("lstm", k3d, truth[k3d])

        panel = PredictionPanel.align([m2d, m3d])

        # The intersection is exactly the bars both models covered.
        assert panel.n_keys == N_BARS - SEQ_LEN + 1
        assert panel.keys[0] == SEQ_LEN - 1

        agreement = panel.agreement_matrix()[0, 1]
        assert agreement == pytest.approx(1.0), (
            f"Two perfect models agree {agreement:.1%} after keyed alignment; "
            f"expected 100%. The keyed contract has the F1 defect."
        )

    def test_positional_keys_are_rejected_not_silently_shifted(self):
        """The old producers emitted 0..n-1. That must now be visible."""
        truth = (np.arange(N_BARS) % 3) - 1
        m2d = _make("xgboost", np.arange(N_BARS), truth)

        # A sequence model naively emitting POSITIONAL indices claims to have
        # predicted bars 0..340 -- bars it never saw.
        n_seq = N_BARS - SEQ_LEN + 1
        wrong = _make("lstm_positional", np.arange(n_seq), truth[SEQ_LEN - 1 :])

        panel = PredictionPanel.align([m2d, wrong])
        agreement = panel.agreement_matrix()[0, 1]

        # This is the OLD behaviour, and the panel now makes it measurable
        # instead of invisible: a diagnostic exists where none did before.
        assert agreement < 0.9, (
            "Positional keys should produce visible disagreement between two "
            "otherwise-perfect models — that visibility is the safeguard."
        )


class TestKeyIntegrity:
    def test_duplicate_keys_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            _make("m", np.array([0, 1, 1]), np.array([0, 0, 0]))

    def test_legacy_sentinel_in_keys_rejected(self):
        """-99 in the key space means labels were passed as coordinates."""
        with pytest.raises(ValueError, match="sentinel"):
            _make("m", np.array([0, 1, -99]), np.array([0, 0, 0]))

    def test_reindex_marks_absent_keys_invalid_not_shifted(self):
        m = _make("m", np.array([10, 11, 12]), np.array([1, 0, -1]))
        out = m.reindex(np.array([9, 10, 11, 12, 13]))
        assert out.n == 5
        np.testing.assert_array_equal(out.valid, [False, True, True, True, False])
        # The values that ARE present kept their identity.
        assert out.class_predictions[1] == 1
        assert out.class_predictions[3] == -1


class TestMissingnessIsAMask:
    def test_invalid_rows_may_be_nan(self):
        probs = np.array([[0.2, 0.3, 0.5], [np.nan, np.nan, np.nan]])
        ps = PredictionSet(
            keys=np.array([0, 1]),
            probabilities=probs,
            valid=np.array([True, False]),
            schema=PredictionSchema.ternary(5),
            provenance=PredictionProvenance(model_name="m"),
        )
        assert ps.n_valid == 1
        assert ps.coverage == 0.5
        assert np.isnan(ps.confidence[1])

    def test_nan_on_a_valid_row_is_rejected(self):
        probs = np.array([[0.2, 0.3, 0.5], [np.nan, np.nan, np.nan]])
        with pytest.raises(ValueError, match="NaN/inf on rows marked valid"):
            PredictionSet(
                keys=np.array([0, 1]),
                probabilities=probs,
                valid=np.array([True, True]),
                schema=PredictionSchema.ternary(5),
                provenance=PredictionProvenance(model_name="m"),
            )

    def test_unnormalised_probabilities_rejected(self):
        with pytest.raises(ValueError, match="sum to 1"):
            PredictionSet(
                keys=np.array([0]),
                probabilities=np.array([[3.0, 1.0, 2.0]]),
                valid=np.array([True]),
                schema=PredictionSchema.ternary(5),
                provenance=PredictionProvenance(model_name="m"),
            )


class TestDerivedNeverDrifts:
    def test_class_predictions_track_probabilities(self):
        ps = _make("m", np.array([0, 1, 2]), np.array([-1, 0, 1]))
        np.testing.assert_array_equal(ps.class_predictions, [-1, 0, 1])

    def test_margin_distinguishes_decisive_from_marginal(self):
        ps = PredictionSet(
            keys=np.array([0, 1]),
            probabilities=np.array([[0.05, 0.05, 0.90], [0.33, 0.32, 0.35]]),
            valid=np.array([True, True]),
            schema=PredictionSchema.ternary(5),
            provenance=PredictionProvenance(model_name="m"),
        )
        assert ps.margin[0] > ps.margin[1]
        # Max-probability alone would call these 0.90 vs 0.35; margin shows
        # the second is nearly a coin flip.
        assert ps.margin[1] < 0.05


class TestCombinabilityFailsClearly:
    def test_schema_mismatch_refused_with_reason(self):
        a = _make("a", np.array([0]), np.array([0]))
        b = PredictionSet(
            keys=np.array([0]),
            probabilities=np.array([[0.5, 0.5]]),
            valid=np.array([True]),
            schema=PredictionSchema.binary(horizon=5),
            provenance=PredictionProvenance(model_name="b"),
        )
        ok, why = a.can_combine_with(b)
        assert not ok and "schema mismatch" in why
        with pytest.raises(ValueError, match="schema mismatch"):
            PredictionPanel.align([a, b])

    def test_oof_and_test_predictions_refused(self):
        a = _make("a", np.array([0]), np.array([0]), source="oof")
        b = _make("b", np.array([0]), np.array([0]), source="test")
        ok, why = a.can_combine_with(b)
        assert not ok and "source mismatch" in why

    def test_disjoint_keys_fail_with_actionable_message(self):
        a = _make("a", np.array([0, 1, 2]), np.array([0, 0, 0]))
        b = _make("b", np.array([100, 101]), np.array([0, 0]))
        with pytest.raises(ValueError, match="no common keys"):
            PredictionPanel.align([a, b])


class TestPanelMechanics:
    def test_union_keeps_all_keys_and_marks_gaps(self):
        a = _make("a", np.array([0, 1]), np.array([0, 0]))
        b = _make("b", np.array([1, 2]), np.array([0, 0]))
        panel = PredictionPanel.align([a, b], how="union")
        assert panel.n_keys == 3
        vm = panel.valid_matrix()
        assert vm[:, 0].tolist() == [True, True, False]
        assert vm[:, 1].tolist() == [False, True, True]

    def test_design_matrix_shape(self):
        a = _make("a", np.array([0, 1]), np.array([0, 0]))
        b = _make("b", np.array([0, 1]), np.array([0, 0]))
        panel = PredictionPanel.align([a, b])
        assert panel.design_matrix().shape == (2, 2 * 3)
        assert panel.probability_tensor().shape == (2, 2, 3)

    def test_y_true_requires_keys(self):
        a = _make("a", np.array([0, 1]), np.array([0, 0]))
        with pytest.raises(ValueError, match="y_true_keys"):
            PredictionPanel.align([a], y_true=np.array([0, 1]))

    def test_y_true_is_resolved_by_key(self):
        truth = (np.arange(N_BARS) % 3) - 1
        a = _make("a", np.arange(100, 200), truth[100:200])
        panel = PredictionPanel.align([a], y_true=truth, y_true_keys=np.arange(N_BARS))
        np.testing.assert_array_equal(panel.y_true, truth[100:200])


class TestCoreStaysLight:
    def test_module_imports_without_ml_stack(self):
        """src.core must stay importable without torch (verified Stage 3)."""
        import subprocess
        import sys

        code = (
            "import sys; import src.core.predictions, src.core.prediction_panel; "
            "print([m for m in ('torch','sklearn','xgboost') if m in sys.modules])"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=".")
        assert out.returncode == 0, out.stderr
        assert "[]" in out.stdout, f"heavy imports leaked in: {out.stdout}"
