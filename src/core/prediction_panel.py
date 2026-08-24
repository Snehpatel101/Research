"""PredictionPanel — align many models onto ONE key set.

This replaces the positional intersection in
``src/data/adapters/alignment.py`` that Phase 0 proved wrong (finding F1):
it set-intersected each model's *own array positions* as though they were a
shared coordinate, silently pairing a tabular model's bar 0 with a 60-bar
sequence model's bar 59.

Here alignment is a join on source-bar keys. A model that never predicted a
given bar contributes an explicitly invalid row rather than its neighbour's
value, so rank differences become *absent keys* — a representable, inspectable
fact — instead of a silent shift.

Imports numpy only: `src.core` must stay light (Stage 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce

import numpy as np

from src.core.predictions import PredictionSchema, PredictionSet


@dataclass(frozen=True)
class PredictionPanel:
    """Several models' predictions on one shared key axis.

    Attributes:
        keys: (K,) the shared source-bar coordinates, sorted ascending.
        members: the models, reindexed onto ``keys`` and in column order.
        y_true: (K,) ground truth resolved ONCE from a reference frame, or
            None. Resolving it once (rather than per-model) is deliberate:
            per-model truth vectors are how misalignment hides.
    """

    keys: np.ndarray
    members: tuple[PredictionSet, ...]
    y_true: np.ndarray | None = None

    # ------------------------------------------------------------ builders
    @classmethod
    def align(
        cls,
        sets: list[PredictionSet] | tuple[PredictionSet, ...],
        *,
        how: str = "intersection",
        y_true: np.ndarray | None = None,
        y_true_keys: np.ndarray | None = None,
        require_combinable: bool = True,
    ) -> PredictionPanel:
        """Align prediction sets onto a common key axis.

        Args:
            sets: the models to align. Must be non-empty.
            how: "intersection" (keys every model predicted — the safe
                default for stacking) or "union" (every key any model
                predicted; absent ones become invalid rows).
            y_true: ground truth, paired with ``y_true_keys``.
            y_true_keys: the source-bar keys ``y_true`` is indexed by.
            require_combinable: enforce schema/source compatibility. Leave on
                unless you are deliberately inspecting a mismatch.

        Raises:
            ValueError: on empty input, incompatible members, or an empty
                intersection — each with an actionable message.
        """
        members = tuple(sets)
        if not members:
            raise ValueError("PredictionPanel.align requires at least one PredictionSet")

        if require_combinable:
            first = members[0]
            for other in members[1:]:
                ok, why = first.can_combine_with(other)
                if not ok:
                    raise ValueError(f"cannot align prediction sets: {why}")

        key_sets = [set(np.asarray(m.keys).tolist()) for m in members]
        if how == "intersection":
            common = reduce(lambda a, b: a & b, key_sets)
        elif how == "union":
            common = reduce(lambda a, b: a | b, key_sets)
        else:
            raise ValueError(f"how must be 'intersection' or 'union', got {how!r}")

        if not common:
            spans = ", ".join(
                f"{m.provenance.model_name}=[{int(np.min(m.keys))}..{int(np.max(m.keys))}]"
                for m in members
                if m.n
            )
            raise ValueError(
                "no common keys across models. Key spans: "
                f"{spans}. If these should overlap, the producers are emitting "
                "positional indices rather than source-bar keys (Phase 0 F1)."
            )

        keys = np.array(sorted(common), dtype=np.int64)
        aligned = tuple(m.reindex(keys) for m in members)

        truth = None
        if y_true is not None:
            if y_true_keys is None:
                raise ValueError("y_true requires y_true_keys — unkeyed truth is how F1 happened")
            pos = {int(k): i for i, k in enumerate(np.asarray(y_true_keys))}
            take = np.array([pos.get(int(k), -1) for k in keys], dtype=np.int64)
            if np.any(take < 0):
                missing = int((take < 0).sum())
                raise ValueError(
                    f"y_true is missing {missing} of {len(keys)} aligned keys; "
                    f"truth must cover the panel"
                )
            truth = np.asarray(y_true)[take]

        return cls(keys=keys, members=aligned, y_true=truth)

    # -------------------------------------------------------------- views
    @property
    def n_keys(self) -> int:
        return len(self.keys)

    @property
    def n_models(self) -> int:
        return len(self.members)

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(m.provenance.model_name for m in self.members)

    @property
    def schema(self) -> PredictionSchema:
        return self.members[0].schema

    def valid_matrix(self) -> np.ndarray:
        """(K, M) bool — which model has a real prediction on which key."""
        return np.column_stack([np.asarray(m.valid) for m in self.members])

    def probability_tensor(self) -> np.ndarray:
        """(K, M, C) probabilities; NaN where a model has no prediction."""
        return np.stack(
            [np.asarray(m.probabilities, dtype=np.float64) for m in self.members], axis=1
        )

    def design_matrix(self) -> np.ndarray:
        """(K, M*C) stacked probabilities — the meta-learner's input."""
        return np.column_stack(
            [np.asarray(m.probabilities, dtype=np.float64) for m in self.members]
        )

    def complete_rows(self) -> np.ndarray:
        """(K,) bool — rows where EVERY model has a valid prediction."""
        return self.valid_matrix().all(axis=1)

    def coverage_by_model(self) -> dict[str, float]:
        vm = self.valid_matrix()
        return {name: float(vm[:, i].mean()) for i, name in enumerate(self.model_names)}

    def agreement_matrix(self) -> np.ndarray:
        """(M, M) pairwise agreement on rows where both models are valid.

        The diagnostic that would have caught F1 immediately: two models
        trained on the same bars should not disagree at chance level.
        """
        preds = np.column_stack([m.class_predictions for m in self.members])
        vm = self.valid_matrix()
        m = self.n_models
        out = np.full((m, m), np.nan)
        for i in range(m):
            for j in range(m):
                both = vm[:, i] & vm[:, j]
                if both.any():
                    out[i, j] = float((preds[both, i] == preds[both, j]).mean())
        return out

    def summary(self) -> str:
        cov = self.coverage_by_model()
        lines = [
            f"PredictionPanel: {self.n_models} models x {self.n_keys} keys "
            f"(bars {int(self.keys[0])}..{int(self.keys[-1])})",
            f"  complete rows: {int(self.complete_rows().sum())} / {self.n_keys}",
        ]
        for name, c in cov.items():
            lines.append(f"  {name:22s} coverage {c:6.1%}")
        return "\n".join(lines)
