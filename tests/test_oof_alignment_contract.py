"""Regression pin for Phase 0 finding F1 — heterogeneous OOF misalignment.

The defect (proven in docs/program/evidence/F1_oof_misalignment_repro.py):

  * adapters compute CORRECT source-bar coordinates
    (``sequence.py`` -> ``original_indices = df_indices[label_positions]``,
    starting at ``seq_len - 1``),
  * but OOF producers discard them and emit POSITIONAL indices into each
    model's own windowed array (``oof_generation.py:383``),
  * and ``OOFAligner`` set-intersects those positional integers as if they
    were a shared key.

Result: a 2D model's row i is aligned with a 3D model's row i, which is
actually source bar ``i + seq_len - 1``. Two *perfect* models then agree 0%
of the time. Silent — no crash, no warning, plausible val_f1.

These tests are marked ``xfail(strict=True)``. They fail today by design.
When Stage 17 lands key-based alignment they will XPASS, which strict mode
turns into a suite failure — forcing the marker to be removed deliberately
rather than quietly outliving the bug.

DO NOT relax these assertions to make the suite green.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

SEQ_LEN = 60
N_BARS = 400


def _build(seq_len: int = SEQ_LEN, n: int = N_BARS):
    """Two PERFECT models over the same bars, one tabular and one windowed."""
    from src.data.adapters.alignment import OOFResult
    from src.data.adapters.sequence import SequenceAdapter

    rng = np.random.default_rng(0)
    # Label is a pure function of bar index, so misalignment is detectable
    # by value rather than by inspection.
    df = pd.DataFrame(
        {
            "f0": rng.normal(size=n),
            "f1": rng.normal(size=n),
            "label": (np.arange(n) % 3) - 1,
        }
    )

    adapter = SequenceAdapter(
        sequence_length=seq_len, feature_columns=["f0", "f1"], label_column="label"
    )
    source_bars = adapter.transform(df).original_indices

    def one_hot(labels):
        p = np.zeros((len(labels), 3), dtype=np.float32)
        p[np.arange(len(labels)), labels + 1] = 1.0
        return p

    # What the producers actually emit today: positional indices.
    pos_2d = np.arange(n)
    pos_3d = np.arange(len(source_bars))

    y_2d = df["label"].to_numpy()[pos_2d]
    y_3d = df["label"].to_numpy()[source_bars]

    oof_2d = OOFResult(
        model_name="xgboost",
        predictions=y_2d,
        probabilities=one_hot(y_2d),
        indices=pos_2d,
        fold_ids=np.zeros(len(y_2d), dtype=int),
    )
    oof_3d = OOFResult(
        model_name="lstm",
        predictions=y_3d,
        probabilities=one_hot(y_3d),
        indices=pos_3d,
        fold_ids=np.zeros(len(y_3d), dtype=int),
    )
    return df, source_bars, oof_2d, oof_3d


def test_adapter_emits_source_bar_coordinates():
    """The adapter half is CORRECT — this must keep passing (guards the fix)."""
    _, source_bars, _, _ = _build()
    assert source_bars[0] == SEQ_LEN - 1, (
        "SequenceAdapter must map window 0 to source bar seq_len-1; "
        "the alignment fix depends on this being true."
    )
    assert len(source_bars) == N_BARS - SEQ_LEN + 1
    np.testing.assert_array_equal(source_bars, np.arange(SEQ_LEN - 1, N_BARS))


@pytest.mark.xfail(
    strict=True,
    reason="Phase 0 F1: OOF producers emit positional indices, so the aligner "
    "intersects incompatible coordinate systems. Fixes at Stage 17.",
)
def test_two_perfect_models_agree_after_alignment():
    """The core contract: aligned rows must describe the SAME bar."""
    from src.data.adapters.alignment import OOFAligner

    _, _, oof_2d, oof_3d = _build()
    aligned = OOFAligner(n_classes=3).align([oof_2d, oof_3d])

    agreement = (aligned.predictions[:, 0] == aligned.predictions[:, 1]).mean()
    assert agreement == pytest.approx(1.0), (
        f"Two PERFECT models agree only {agreement:.1%} of the time after "
        f"alignment. Aligned rows refer to different source bars."
    )


@pytest.mark.xfail(
    strict=True,
    reason="Phase 0 F1: OOF indices are positional, not source-bar. " "Fixes at Stage 17.",
)
def test_aligned_indices_are_source_bar_coordinates():
    """Alignment keys must be a SHARED coordinate system, not per-model offsets."""
    from src.data.adapters.alignment import OOFAligner

    _, source_bars, oof_2d, oof_3d = _build()
    aligned = OOFAligner(n_classes=3).align([oof_2d, oof_3d])

    # Only bars both models actually predicted can be common.
    expected_first = source_bars[0]
    assert aligned.common_indices[0] == expected_first, (
        f"First aligned key is {aligned.common_indices[0]}, expected source "
        f"bar {expected_first}. Keys are positional, not shared coordinates."
    )
