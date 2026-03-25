"""
Label perturbation testing for feature importance robustness.

Tests whether feature importance rankings are stable when labeling parameters
change (e.g., different barrier widths). Features whose rank shifts drastically
under small label perturbations are likely overfitting to labeling artifacts
rather than capturing genuine signal.

Approach:
1. Compute MDA (permutation importance) ranking under each label variant.
2. Compare ranks against the baseline (first variant).
3. Features with max rank change <= threshold are deemed "robust".

Reference: improvements_sneh.md Section 16.3 item 3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


@dataclass
class PerturbationResult:
    """Result of label perturbation testing for a single feature."""

    feature_name: str
    baseline_rank: int
    perturbed_ranks: list[int]
    max_rank_change: int
    mean_rank: float
    is_robust: bool


class LabelPerturbationTester:
    """Test feature importance stability across different label definitions.

    Ranks features by MDA importance under each label variant and flags
    features whose rank shifts beyond a configurable threshold.
    """

    def __init__(
        self,
        rank_change_threshold: int = 20,
        n_estimators: int = 50,
        n_repeats: int = 3,
        max_samples: int = 50_000,
        random_state: int = 42,
    ) -> None:
        self.rank_change_threshold = rank_change_threshold
        self.n_estimators = n_estimators
        self.n_repeats = n_repeats
        self.max_samples = max_samples
        self.random_state = random_state

    def _compute_ranking(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Compute MDA importance ranking for features.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Label vector aligned with X.

        Returns:
            Series indexed by feature name with integer ranks (1 = most important).
        """
        # Subsample if dataset is large
        if len(X) > self.max_samples:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(len(X), size=self.max_samples, replace=False)
            X = X.iloc[idx]
            y = y.iloc[idx]

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=6,
            n_jobs=-1,
            random_state=self.random_state,
        )

        # Fit on full data for permutation importance
        rf.fit(X, y)

        result = permutation_importance(
            rf,
            X,
            y,
            n_repeats=self.n_repeats,
            n_jobs=-1,
            random_state=self.random_state,
            scoring="accuracy",
        )

        importance = pd.Series(result.importances_mean, index=X.columns)
        # Rank descending: highest importance = rank 1
        ranks = importance.rank(ascending=False, method="min").astype(int)
        return ranks

    def evaluate(
        self,
        X: pd.DataFrame,
        y_variants: dict[str, pd.Series],
    ) -> list[PerturbationResult]:
        """Evaluate feature ranking stability across label variants.

        Args:
            X: Feature matrix (n_samples, n_features).
            y_variants: Mapping of variant name to label vector. The first
                entry is treated as the baseline. All variants must have the
                same length as X.

        Returns:
            List of PerturbationResult sorted by max_rank_change ascending
            (most robust features first).

        Raises:
            ValueError: If y_variants is empty or lengths mismatch.
        """
        if not y_variants:
            raise ValueError("y_variants must contain at least one label variant")

        variant_names = list(y_variants.keys())
        n_samples = len(X)

        for name, y in y_variants.items():
            if len(y) != n_samples:
                raise ValueError(
                    f"Label variant '{name}' has {len(y)} samples, "
                    f"expected {n_samples} to match X"
                )

        logger.info(
            f"Label perturbation test: {len(variant_names)} variants, "
            f"{X.shape[1]} features, {n_samples} samples"
        )

        # Compute rankings for each variant
        rankings: dict[str, pd.Series] = {}
        for name, y in y_variants.items():
            logger.info(f"  Computing MDA ranking for variant '{name}'...")
            rankings[name] = self._compute_ranking(X, y)

        baseline_name = variant_names[0]
        baseline_ranks = rankings[baseline_name]
        perturbed_names = variant_names[1:]

        results: list[PerturbationResult] = []
        for feature in X.columns:
            b_rank = int(baseline_ranks[feature])

            p_ranks = [int(rankings[name][feature]) for name in perturbed_names]

            if p_ranks:
                max_change = max(abs(r - b_rank) for r in p_ranks)
                all_ranks = [b_rank] + p_ranks
            else:
                # Single variant: no perturbation, always robust
                max_change = 0
                all_ranks = [b_rank]

            mean_r = float(np.mean(all_ranks))

            results.append(
                PerturbationResult(
                    feature_name=feature,
                    baseline_rank=b_rank,
                    perturbed_ranks=p_ranks,
                    max_rank_change=max_change,
                    mean_rank=round(mean_r, 2),
                    is_robust=max_change <= self.rank_change_threshold,
                )
            )

        results.sort(key=lambda r: r.max_rank_change)

        n_robust = sum(1 for r in results if r.is_robust)
        logger.info(
            f"Label perturbation complete: {n_robust}/{len(results)} features robust "
            f"(threshold={self.rank_change_threshold})"
        )
        return results

    def get_robust_features(self, results: list[PerturbationResult]) -> list[str]:
        """Return sorted list of robust feature names.

        Args:
            results: Output from evaluate().

        Returns:
            Alphabetically sorted list of feature names where is_robust is True.
        """
        return sorted(r.feature_name for r in results if r.is_robust)


__all__ = ["LabelPerturbationTester", "PerturbationResult"]
