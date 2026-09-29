"""
Regression tests for the OOFPrediction schema contract.

A walk-forward producer once emitted compact prediction frames (only the rows
that had predictions) while the ensemble consumer assumed full-length
NaN-padded frames indexed by ``original_indices``. The consumer
(``EnsembleService._convert_to_oof_results``) now rejects compact frames with
a ValueError instead of silently mis-indexing them.

Covered here:
1. Compact frame (len < n_total_samples) -> ValueError mentioning 'schema'.
2. Full-length NaN-padded frame + original_indices -> conversion succeeds and
   per-model arrays have len == len(original_indices).
3. ``get_probabilities`` is column-dynamic: canonical short/neutral/long order
   for 3-class frames, prob_0/prob_1 for binary frames, KeyError when no
   probability columns exist.

The integration variant (driving the real walk-forward OOF path through
TrainingOpsMixin) is intentionally skipped: there is no existing light harness
for it in this test suite and standing one up would train real models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.training.services.ensemble_service import EnsembleService
from src.validation.cv import OOFPrediction

# =============================================================================
# HELPERS
# =============================================================================


def _prob_cols(model_name: str, n_classes: int) -> list[str]:
    """Probability column names matching the producer convention."""
    if n_classes == 3:
        return [f"{model_name}_prob_{c}" for c in ("short", "neutral", "long")]
    return [f"{model_name}_prob_{i}" for i in range(n_classes)]


def _make_probs(n: int, n_classes: int, rng: np.random.Generator) -> np.ndarray:
    """Random valid probability rows (each row sums to 1)."""
    raw = rng.uniform(0.05, 1.0, size=(n, n_classes))
    return raw / raw.sum(axis=1, keepdims=True)


def _full_length_frame(
    model_name: str,
    n_total: int,
    valid_indices: np.ndarray,
    n_classes: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a full-length NaN-padded OOF frame (the schema contract shape).

    Rows NOT in ``valid_indices`` carry NaN probabilities/predictions and
    fold_id == -1, mirroring what walk-forward and sequence producers emit.
    """
    rng = np.random.default_rng(seed)
    n_valid = len(valid_indices)

    probs_full = np.full((n_total, n_classes), np.nan)
    preds_full = np.full(n_total, np.nan)
    conf_full = np.full(n_total, np.nan)
    fold_full = np.full(n_total, -1, dtype=int)

    probs_full[valid_indices] = _make_probs(n_valid, n_classes, rng)
    preds_full[valid_indices] = rng.choice([-1, 0, 1], size=n_valid)
    conf_full[valid_indices] = np.nanmax(probs_full[valid_indices], axis=1)
    fold_full[valid_indices] = np.arange(n_valid) % 2

    data: dict[str, np.ndarray] = {"y_true": rng.choice([-1, 0, 1], size=n_total)}
    for i, col in enumerate(_prob_cols(model_name, n_classes)):
        data[col] = probs_full[:, i]
    data[f"{model_name}_pred"] = preds_full
    data[f"{model_name}_confidence"] = conf_full
    data["fold_id"] = fold_full
    return pd.DataFrame(data)


def _compact_frame(model_name: str, n_rows: int, n_classes: int = 3, seed: int = 1) -> pd.DataFrame:
    """Build a COMPACT frame (only predicted rows) — the broken producer shape."""
    rng = np.random.default_rng(seed)
    probs = _make_probs(n_rows, n_classes, rng)
    data: dict[str, np.ndarray] = {"y_true": rng.choice([-1, 0, 1], size=n_rows)}
    for i, col in enumerate(_prob_cols(model_name, n_classes)):
        data[col] = probs[:, i]
    data[f"{model_name}_pred"] = rng.choice([-1, 0, 1], size=n_rows)
    data[f"{model_name}_confidence"] = probs.max(axis=1)
    data["fold_id"] = np.arange(n_rows) % 2
    return pd.DataFrame(data)


# =============================================================================
# 1. COMPACT FRAME REJECTION
# =============================================================================


class TestCompactFrameRejection:
    """Compact frames are rejected at the ROOT — OOFPrediction construction.

    The contract is enforced in OOFPrediction.__init__ so a compact frame
    fails at the offending producer (with the model name in the error),
    before any consumer can mis-index it. EnsembleService keeps a
    defense-in-depth check for objects built before the contract existed.
    """

    def test_compact_frame_raises_valueerror_mentioning_schema(self) -> None:
        n_total = 100
        valid_indices = np.arange(60, 100)  # 40 valid rows
        compact = _compact_frame("xgboost", n_rows=len(valid_indices))

        with pytest.raises(ValueError, match="schema"):
            OOFPrediction(
                model_name="xgboost",
                predictions=compact,  # 40 rows — violates full-length contract
                fold_info=[],
                coverage=len(valid_indices) / n_total,
                original_indices=valid_indices,
                n_total_samples=n_total,
            )

    def test_error_message_names_offending_model(self) -> None:
        compact = _compact_frame("lightgbm", n_rows=20)
        with pytest.raises(ValueError, match="lightgbm"):
            OOFPrediction(
                model_name="lightgbm",
                predictions=compact,
                fold_info=[],
                original_indices=np.arange(30, 50),
                n_total_samples=50,
            )

    def test_legacy_frame_without_n_total_samples_is_not_rejected(self) -> None:
        """Legacy producers (n_total_samples=None) predate the contract.

        Their frames are treated as full-length by definition, so a short
        frame with no metadata passes through — documents current behavior.
        """
        compact = _compact_frame("xgboost", n_rows=30)
        oof = OOFPrediction(
            model_name="xgboost",
            predictions=compact,
            fold_info=[],
        )
        service = EnsembleService()
        results = service._convert_to_oof_results({"xgboost": oof})
        assert len(results) == 1
        assert results[0].n_samples == 30


# =============================================================================
# 2. FULL-LENGTH NAN-PADDED FRAME CONVERSION
# =============================================================================


class TestFullLengthFrameConversion:
    """Contract-conforming frames convert cleanly to OOFResult."""

    def test_full_length_frame_converts_to_valid_row_arrays(self) -> None:
        n_total = 100
        valid_indices = np.arange(20, 100)  # first 20 rows lost to windowing
        frame = _full_length_frame("xgboost", n_total, valid_indices)

        oof = OOFPrediction(
            model_name="xgboost",
            predictions=frame,
            fold_info=[],
            coverage=len(valid_indices) / n_total,
            original_indices=valid_indices,
            n_total_samples=n_total,
        )

        service = EnsembleService()
        results = service._convert_to_oof_results({"xgboost": oof})

        assert len(results) == 1
        result = results[0]
        assert result.model_name == "xgboost"

        # Per-model arrays are filtered down to the valid rows only
        n_valid = len(valid_indices)
        assert len(result.predictions) == n_valid
        assert len(result.probabilities) == n_valid
        assert len(result.fold_ids) == n_valid
        assert np.array_equal(result.indices, valid_indices)

        # NaN padding must not leak into the converted arrays
        assert not np.any(np.isnan(result.probabilities))
        assert result.fold_ids.min() >= 0
        assert set(np.unique(result.predictions)).issubset({-1, 0, 1})

    def test_two_models_convert_to_two_results(self) -> None:
        n_total = 80
        idx_a = np.arange(10, 80)
        idx_b = np.arange(0, 80)
        oof_a = OOFPrediction(
            model_name="xgboost",
            predictions=_full_length_frame("xgboost", n_total, idx_a, seed=2),
            fold_info=[],
            original_indices=idx_a,
            n_total_samples=n_total,
        )
        oof_b = OOFPrediction(
            model_name="lightgbm",
            predictions=_full_length_frame("lightgbm", n_total, idx_b, seed=3),
            fold_info=[],
            original_indices=idx_b,
            n_total_samples=n_total,
        )

        service = EnsembleService()
        results = service._convert_to_oof_results({"xgboost": oof_a, "lightgbm": oof_b})

        assert [r.model_name for r in results] == ["xgboost", "lightgbm"]
        assert results[0].n_samples == len(idx_a)
        assert results[1].n_samples == len(idx_b)

    def test_binary_full_length_frame_converts(self) -> None:
        n_total = 60
        valid_indices = np.arange(5, 60)
        frame = _full_length_frame("xgboost", n_total, valid_indices, n_classes=2, seed=4)
        oof = OOFPrediction(
            model_name="xgboost",
            predictions=frame,
            fold_info=[],
            original_indices=valid_indices,
            n_total_samples=n_total,
        )
        service = EnsembleService()
        results = service._convert_to_oof_results({"xgboost": oof})
        assert results[0].probabilities.shape == (len(valid_indices), 2)


# =============================================================================
# 3. get_probabilities COLUMN-DYNAMIC BEHAVIOR
# =============================================================================


class TestGetProbabilitiesColumnDynamic:
    """get_probabilities adapts to 3-class, binary, and missing columns."""

    def test_three_class_returns_canonical_short_neutral_long_order(self) -> None:
        rng = np.random.default_rng(5)
        n = 25
        probs = _make_probs(n, 3, rng)
        # Deliberately scrambled column order: long, short, neutral
        frame = pd.DataFrame(
            {
                "m_prob_long": probs[:, 2],
                "m_prob_short": probs[:, 0],
                "m_prob_neutral": probs[:, 1],
                "m_pred": rng.choice([-1, 0, 1], size=n),
            }
        )
        oof = OOFPrediction(model_name="m", predictions=frame, fold_info=[])

        result = oof.get_probabilities()

        assert result.shape == (n, 3)
        # Canonical order wins regardless of frame column order
        np.testing.assert_array_equal(result[:, 0], probs[:, 0])  # short
        np.testing.assert_array_equal(result[:, 1], probs[:, 1])  # neutral
        np.testing.assert_array_equal(result[:, 2], probs[:, 2])  # long

    def test_binary_prob_0_prob_1_returns_n_by_2(self) -> None:
        rng = np.random.default_rng(6)
        n = 30
        probs = _make_probs(n, 2, rng)
        frame = pd.DataFrame(
            {
                "m_prob_0": probs[:, 0],
                "m_prob_1": probs[:, 1],
                "m_pred": rng.choice([0, 1], size=n),
            }
        )
        oof = OOFPrediction(model_name="m", predictions=frame, fold_info=[])

        result = oof.get_probabilities()

        assert result.shape == (n, 2)
        np.testing.assert_array_equal(result[:, 0], probs[:, 0])
        np.testing.assert_array_equal(result[:, 1], probs[:, 1])

    def test_no_prob_columns_raises_keyerror(self) -> None:
        frame = pd.DataFrame({"m_pred": [0, 1, -1], "y_true": [0, 0, 1]})
        oof = OOFPrediction(model_name="m", predictions=frame, fold_info=[])

        with pytest.raises(KeyError, match="No probability columns"):
            oof.get_probabilities()
