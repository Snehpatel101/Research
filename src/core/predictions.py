"""Canonical prediction contract — keyed, never positional.

WHY THIS EXISTS
---------------
Phase 0 proved (docs/program/evidence/F1_oof_misalignment_repro.py) that
heterogeneous ensembles in this repo silently align predictions from
different models by *position in each model's own array*. A tabular model's
row 0 is source bar 0; a 60-bar sequence model's row 0 is source bar 59.
Intersecting those integers as if they were a shared key produced a
systematic 59-bar shift: two DELIBERATELY PERFECT models agreed 0% of the
time, with no crash and a plausible-looking val_f1.

The fix is structural, not a patch. A prediction is meaningless without the
identity of the thing it predicts, so `keys` is mandatory here and carries
SOURCE-BAR coordinates — the one coordinate system every model shares.

The second defect this closes: missingness was signalled four different ways
(`-99`, `-999`, `-1`, and NaN), and `INVALID_LABEL = -99` is independently
redefined in FIVE modules. A sentinel is a value that must never collide with
real data, which is a promise no numeric code can keep — `-1` is a genuine
class label in this codebase. `valid` is a boolean mask and is the ONLY
missingness signal.

DESIGN RULES
------------
1. `keys` are source-bar indices. Never positional. Never per-model.
2. `valid` is the sole missingness signal. No sentinels.
3. `class_predictions`, `confidence` and `margin` are DERIVED, never stored —
   a stored copy can drift from the probabilities it claims to summarise.
4. Frozen. Alignment bugs love mutable shared state.
5. `src.core` must import in ~0.4s with no ML stack (verified Stage 3), so
   this module imports numpy only — no torch, no sklearn, no src.data,
   no src.models.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

# Missingness is a mask, not a magic number. Kept here only to document the
# legacy values this contract replaces, so migrations can find them.
LEGACY_MISSING_SENTINELS: tuple[int, ...] = (-99, -999)


@dataclass(frozen=True)
class PredictionSchema:
    """What the columns of a probability matrix MEAN.

    Two prediction sets may only be combined if their schemas are equal —
    that is the typed replacement for the old `k.endswith("_h5")` string
    check that decided combinability.
    """

    n_classes: int
    class_labels: tuple[int, ...]
    horizon: int
    task: str = "classification"

    def __post_init__(self) -> None:
        if self.n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {self.n_classes}")
        if len(self.class_labels) != self.n_classes:
            raise ValueError(
                f"class_labels has {len(self.class_labels)} entries but "
                f"n_classes={self.n_classes}"
            )
        if len(set(self.class_labels)) != len(self.class_labels):
            raise ValueError(f"class_labels contains duplicates: {self.class_labels}")

    @classmethod
    def ternary(cls, horizon: int) -> PredictionSchema:
        """The repo's default short/neutral/long labelling."""
        return cls(n_classes=3, class_labels=(-1, 0, 1), horizon=horizon)

    @classmethod
    def binary(cls, horizon: int) -> PredictionSchema:
        """Binary mode: no-move vs significant-move."""
        return cls(n_classes=2, class_labels=(0, 1), horizon=horizon)


@dataclass(frozen=True)
class PredictionProvenance:
    """Where a prediction came from.

    Carried so that an ensemble can refuse to combine things that should not
    be combined (e.g. OOF with test predictions), and so results remain
    explicable after the fact.
    """

    model_name: str
    family: str = ""
    data_rank: str = ""
    source: str = "oof"  # "oof" | "val" | "test" | "live"
    calibrated: bool = False
    calibration_method: str | None = None
    feature_fingerprint: str = ""
    cv_fingerprint: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionSet:
    """One model's predictions, keyed by source-bar index.

    Args:
        keys: (N,) int64 SOURCE-BAR indices. The shared coordinate system.
        probabilities: (N, C) float. Rows where ``valid`` is False are ignored
            and may be NaN.
        valid: (N,) bool. The only missingness signal.
        schema: what the C columns mean.
        provenance: where these came from.
        timestamps: (N,) datetime64, optional. Preferred key when available —
            it survives re-indexing, which integer positions do not.
        fold_id: (N,) int, optional. -1 means "not from a CV fold".
    """

    keys: np.ndarray
    probabilities: np.ndarray
    valid: np.ndarray
    schema: PredictionSchema
    provenance: PredictionProvenance
    timestamps: np.ndarray | None = None
    fold_id: np.ndarray | None = None

    # ---------------------------------------------------------------- checks
    def __post_init__(self) -> None:
        keys = np.asarray(self.keys)
        probs = np.asarray(self.probabilities, dtype=np.float64)
        valid = np.asarray(self.valid, dtype=bool)

        if keys.ndim != 1:
            raise ValueError(f"keys must be 1-D, got shape {keys.shape}")
        if probs.ndim != 2:
            raise ValueError(f"probabilities must be 2-D (N, C), got shape {probs.shape}")
        n = len(keys)
        if probs.shape[0] != n:
            raise ValueError(f"probabilities has {probs.shape[0]} rows but {n} keys")
        if probs.shape[1] != self.schema.n_classes:
            raise ValueError(
                f"probabilities has {probs.shape[1]} columns but schema declares "
                f"n_classes={self.schema.n_classes}"
            )
        if valid.shape != (n,):
            raise ValueError(f"valid must have shape ({n},), got {valid.shape}")

        # Keys must be unique: they are a coordinate, and a duplicated
        # coordinate makes alignment ambiguous rather than merely wrong.
        if len(np.unique(keys)) != n:
            uniq, counts = np.unique(keys, return_counts=True)
            dupes = [int(k) for k, c in zip(uniq, counts, strict=True) if c > 1]
            raise ValueError(f"keys must be unique; duplicates: {dupes[:5]}")

        # A legacy sentinel appearing in `keys` means someone passed labels or
        # a padded array as a coordinate. Catch it at the boundary.
        for sentinel in LEGACY_MISSING_SENTINELS:
            if np.any(keys == sentinel):
                raise ValueError(
                    f"keys contain the legacy missing-sentinel {sentinel}. "
                    f"Missingness belongs in `valid`, not in the key space."
                )

        if self.timestamps is not None and len(np.asarray(self.timestamps)) != n:
            raise ValueError("timestamps length must match keys")
        if self.fold_id is not None and len(np.asarray(self.fold_id)) != n:
            raise ValueError("fold_id length must match keys")

        # Valid rows must carry a usable distribution. Invalid rows are free
        # to be NaN — that is what the mask is for.
        if valid.any():
            vp = probs[valid]
            if not np.isfinite(vp).all():
                raise ValueError(
                    "probabilities contain NaN/inf on rows marked valid. "
                    "Mark them invalid instead of smuggling in a sentinel."
                )
            sums = vp.sum(axis=1)
            if not np.allclose(sums, 1.0, atol=1e-3):
                worst = float(np.max(np.abs(sums - 1.0)))
                raise ValueError(
                    f"probabilities on valid rows must sum to 1 (max deviation "
                    f"{worst:.4g}). Un-normalised scores are not probabilities."
                )

    # ------------------------------------------------------------- derived
    @property
    def n(self) -> int:
        """Total rows, valid or not."""
        return len(self.keys)

    @property
    def n_valid(self) -> int:
        return int(np.asarray(self.valid).sum())

    @property
    def coverage(self) -> float:
        """Fraction of rows carrying a real prediction."""
        return self.n_valid / self.n if self.n else 0.0

    @property
    def class_predictions(self) -> np.ndarray:
        """Argmax mapped through ``schema.class_labels``.

        Derived on access so it cannot drift from `probabilities`. Invalid
        rows are NOT given a fake class — use ``valid`` to filter, or
        ``valid_view()``.
        """
        labels = np.asarray(self.schema.class_labels)
        probs = np.asarray(self.probabilities, dtype=np.float64)
        idx = np.full(self.n, 0, dtype=np.int64)
        v = np.asarray(self.valid)
        if v.any():
            idx[v] = np.nanargmax(probs[v], axis=1)
        out = labels[idx]
        return out

    @property
    def confidence(self) -> np.ndarray:
        """Max class probability; NaN on invalid rows."""
        probs = np.asarray(self.probabilities, dtype=np.float64)
        out = np.full(self.n, np.nan)
        v = np.asarray(self.valid)
        if v.any():
            out[v] = probs[v].max(axis=1)
        return out

    @property
    def margin(self) -> np.ndarray:
        """Top-1 minus top-2 probability; NaN on invalid rows.

        A better confidence signal than max-probability alone: 0.4 vs 0.35 is
        far less decisive than 0.4 vs 0.05.
        """
        probs = np.asarray(self.probabilities, dtype=np.float64)
        out = np.full(self.n, np.nan)
        v = np.asarray(self.valid)
        if v.any():
            part = np.sort(probs[v], axis=1)
            out[v] = part[:, -1] - part[:, -2]
        return out

    # -------------------------------------------------------------- helpers
    def valid_view(self) -> PredictionSet:
        """Drop invalid rows. Keys are preserved, so alignment still works."""
        v = np.asarray(self.valid)
        return replace(
            self,
            keys=np.asarray(self.keys)[v],
            probabilities=np.asarray(self.probabilities)[v],
            valid=np.ones(int(v.sum()), dtype=bool),
            timestamps=None if self.timestamps is None else np.asarray(self.timestamps)[v],
            fold_id=None if self.fold_id is None else np.asarray(self.fold_id)[v],
        )

    def reindex(self, target_keys: np.ndarray) -> PredictionSet:
        """Project onto ``target_keys``, marking absent keys invalid.

        This is the operation the old code did by positional slicing, which is
        exactly how the 59-bar shift arose. Here a key that this model never
        predicted becomes an explicitly invalid row instead of silently
        borrowing its neighbour's value.
        """
        target = np.asarray(target_keys)
        pos = {int(k): i for i, k in enumerate(np.asarray(self.keys))}
        take = np.array([pos.get(int(k), -1) for k in target], dtype=np.int64)
        hit = take >= 0

        probs = np.full((len(target), self.schema.n_classes), np.nan)
        valid = np.zeros(len(target), dtype=bool)
        if hit.any():
            src = np.asarray(self.probabilities, dtype=np.float64)[take[hit]]
            probs[hit] = src
            valid[hit] = np.asarray(self.valid)[take[hit]]

        folds = None
        if self.fold_id is not None:
            folds = np.full(len(target), -1, dtype=np.int64)
            folds[hit] = np.asarray(self.fold_id)[take[hit]]

        stamps = None
        if self.timestamps is not None:
            stamps = np.full(len(target), np.datetime64("NaT"))
            stamps[hit] = np.asarray(self.timestamps)[take[hit]]

        return replace(
            self,
            keys=target,
            probabilities=probs,
            valid=valid,
            fold_id=folds,
            timestamps=stamps,
        )

    def can_combine_with(self, other: PredictionSet) -> tuple[bool, str]:
        """Whether two sets may legally be combined, and why not if not.

        Returns (ok, reason). Deliberately returns a reason rather than a
        bare bool: 'incompatible combinations must fail clearly and
        intentionally' is a stated requirement, and a caller needs something
        actionable to print.
        """
        if self.schema != other.schema:
            return False, (
                f"schema mismatch: {self.provenance.model_name} has {self.schema}, "
                f"{other.provenance.model_name} has {other.schema}"
            )
        if self.provenance.source != other.provenance.source:
            return False, (
                f"source mismatch: {self.provenance.model_name} is "
                f"'{self.provenance.source}' but {other.provenance.model_name} is "
                f"'{other.provenance.source}'. Combining OOF with test "
                f"predictions inflates the result."
            )
        return True, ""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"PredictionSet(model={self.provenance.model_name!r}, n={self.n}, "
            f"valid={self.n_valid}, coverage={self.coverage:.1%}, "
            f"keys=[{self.keys[0] if self.n else '-'}..{self.keys[-1] if self.n else '-'}], "
            f"schema={self.schema.n_classes}-class h{self.schema.horizon})"
        )
