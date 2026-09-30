"""Label-perturbation robustness of feature importance.

A feature whose importance rank collapses when the triple-barrier widths change
by a modest amount is describing one particular labeling artifact rather than a
durable relationship. This module compares importance rankings computed under
perturbed label variants (built by the caller with the same labeler and purged
CV as a x1.0 CONTROL relabel) and flags features whose rank moves more than a
tolerance. The control is relabeled exactly like the variants, so rank movement
that the relabeling procedure itself causes (a truncated tail, a re-calibrated
cost) is measured against the live ranking and never blamed on the perturbation:
a feature is fragile only when its shift exceeds the control's own shift too.

Only bookkeeping happens here: the importances come in, per-feature rank shifts
and a rank-correlation summary come out.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PerturbationResult:
    """Rank movement of one feature across label variants."""

    feature_name: str
    baseline_rank: int  # rank under the reference labels (the control when given)
    perturbed_ranks: list[int]
    max_rank_change: int  # largest |variant rank - reference rank|
    control_shift: int  # |control rank - live baseline rank| (0 without a control)
    mean_rank: float
    is_robust: bool


@dataclass
class PerturbationSummary:
    """Per-feature results plus how similar each variant's ranking is overall."""

    results: list[PerturbationResult]
    rank_correlation: dict[str, float]  # variant name -> Spearman vs the reference
    variants_used: list[str]
    control_rank_correlation: float | None = None  # control vs the live baseline


class LabelPerturbationTester:
    """Compare importance rankings under baseline vs perturbed labels.

    Args:
        rank_change_fraction: A feature is robust when its largest rank move is at
            most this share of the feature count (relative, so the tolerance is
            meaningful for 10 features and for 200).
        min_rank_change: Floor on the absolute tolerance (ranks are noisy for
            near-zero importances).
    """

    def __init__(self, rank_change_fraction: float = 0.15, min_rank_change: int = 3) -> None:
        if not 0.0 < rank_change_fraction <= 1.0:
            raise ValueError(f"rank_change_fraction must be in (0, 1], got {rank_change_fraction}")
        self.rank_change_fraction = rank_change_fraction
        self.min_rank_change = min_rank_change

    def evaluate(
        self,
        baseline: pd.Series,
        variants: dict[str, pd.Series | None],
        control: pd.Series | None = None,
    ) -> PerturbationSummary:
        """Rank shifts of every baseline feature under each usable variant.

        Args:
            baseline: Importance per feature under the live labels.
            variants: Importance per feature under each perturbed label variant
                (None = the variant could not be ranked; it is skipped).
            control: Importance under a x1.0 relabel computed like the variants.
                When given, shifts are measured against it and a feature is
                flagged only if its shift also exceeds ``|control - baseline|``.
        """
        names = list(baseline.index)
        threshold = max(self.min_rank_change, int(round(self.rank_change_fraction * len(names))))

        def _ranks(imp: pd.Series) -> pd.Series:
            return imp.reindex(names).fillna(-np.inf).rank(ascending=False, method="average")

        base_ranks = _ranks(baseline)
        control_ranks = _ranks(control) if control is not None and len(control) else None
        reference = control_ranks if control_ranks is not None else base_ranks
        used: dict[str, pd.Series] = {
            name: _ranks(imp) for name, imp in variants.items() if imp is not None and len(imp)
        }

        def _spearman(a: pd.Series, b: pd.Series) -> float:
            return float(a.corr(b, method="spearman")) if len(names) > 1 else 1.0

        correlation = {name: _spearman(reference, r) for name, r in used.items()}
        results = []
        for f in names:
            ref = float(reference[f])
            perturbed = [float(r[f]) for r in used.values()]
            max_change = max((abs(p - ref) for p in perturbed), default=0.0)
            own = abs(ref - float(base_ranks[f])) if control_ranks is not None else 0.0
            results.append(
                PerturbationResult(
                    feature_name=f,
                    baseline_rank=int(round(ref)),
                    perturbed_ranks=[int(round(p)) for p in perturbed],
                    max_rank_change=int(round(max_change)),
                    control_shift=int(round(own)),
                    mean_rank=round(float(np.mean([ref, *perturbed])), 2),
                    is_robust=bool(used) and max_change <= max(threshold, own),
                )
            )
        results.sort(key=lambda r: r.max_rank_change)
        logger.info(
            "Label perturbation: %d/%d features robust over %d variants (rank move <= %d)",
            sum(r.is_robust for r in results),
            len(results),
            len(used),
            threshold,
        )
        return PerturbationSummary(
            results=results,
            rank_correlation=correlation,
            variants_used=list(used),
            control_rank_correlation=(
                _spearman(base_ranks, control_ranks) if control_ranks is not None else None
            ),
        )


__all__ = ["LabelPerturbationTester", "PerturbationResult", "PerturbationSummary"]
