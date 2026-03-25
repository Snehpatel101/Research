"""Bootstrap feature stability testing.

Runs N bootstrap resamples of the data, computes MDA (permutation importance)
on each, and measures how consistently each feature ranks in the top-K.
Stable features consistently rank high regardless of sample variation.

Reference: Section 16.3 of improvements_sneh.md
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
class BootstrapStabilityResult:
    """Result for a single feature's bootstrap stability evaluation."""

    feature_name: str
    selection_frequency: float  # fraction of bootstrap samples in top-K
    mean_rank: float  # average rank across bootstrap samples
    rank_std: float  # standard deviation of rank
    is_stable: bool  # True if selection_frequency >= stability_threshold


class BootstrapFeatureStability:
    """Evaluate feature stability via bootstrap resampling of MDA importance.

    For each bootstrap sample, trains a RandomForest and computes permutation
    importance. Features that consistently appear in the top-K across samples
    are considered stable.
    """

    def __init__(
        self,
        n_bootstrap: int = 20,
        top_k: int = 30,
        stability_threshold: float = 0.6,
        n_estimators: int = 50,
        n_repeats: int = 3,
        max_samples: int = 50_000,
        random_state: int = 42,
    ) -> None:
        self.n_bootstrap = n_bootstrap
        self.top_k = top_k
        self.stability_threshold = stability_threshold
        self.n_estimators = n_estimators
        self.n_repeats = n_repeats
        self.max_samples = max_samples
        self.random_state = random_state

    def _compute_importance(self, X: pd.DataFrame, y: pd.Series, seed: int) -> pd.Series:
        """Train RF and compute permutation importance on a single sample.

        Args:
            X: Feature matrix.
            y: Target labels.
            seed: Random seed for reproducibility.

        Returns:
            Series of importance scores indexed by feature name.
        """
        # Subsample if needed
        if len(X) > self.max_samples:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(X), size=self.max_samples, replace=False)
            X = X.iloc[idx]
            y = y.iloc[idx]

        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=8,
            n_jobs=-1,
            random_state=seed,
        )
        clf.fit(X, y)

        result = permutation_importance(
            clf,
            X,
            y,
            n_repeats=self.n_repeats,
            random_state=seed,
            n_jobs=-1,
        )
        return pd.Series(result.importances_mean, index=X.columns)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> list[BootstrapStabilityResult]:
        """Run bootstrap stability evaluation.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target labels (n_samples,).

        Returns:
            List of BootstrapStabilityResult sorted by selection_frequency
            descending.
        """
        n_features = X.shape[1]
        if n_features == 0 or len(X) == 0:
            logger.warning("Empty data passed to bootstrap stability — returning empty results")
            return []

        # Check for single-class target
        n_classes = y.nunique()
        if n_classes < 2:
            logger.warning(
                "Only %d class(es) in target — cannot compute MDA, returning empty results",
                n_classes,
            )
            return []

        effective_top_k = min(self.top_k, n_features)

        # Track ranks per feature across bootstrap samples
        all_feature_names = list(X.columns)
        rank_records: dict[str, list[int]] = {f: [] for f in all_feature_names}
        top_k_counts: dict[str, int] = dict.fromkeys(all_feature_names, 0)

        for i in range(self.n_bootstrap):
            logger.info("Bootstrap sample %d/%d", i + 1, self.n_bootstrap)

            # Resample with replacement
            rng = np.random.RandomState(self.random_state + i)
            idx = rng.choice(len(X), size=len(X), replace=True)
            X_boot = X.iloc[idx]
            y_boot = y.iloc[idx]

            # Skip if resampled target has only one class
            if y_boot.nunique() < 2:
                logger.warning("Bootstrap sample %d has single class — skipping", i + 1)
                continue

            importance = self._compute_importance(X_boot, y_boot, seed=self.random_state + i)

            # Rank features (1 = most important)
            ranks = importance.rank(ascending=False, method="min").astype(int)

            for feat in all_feature_names:
                rank = int(ranks.get(feat, n_features))
                rank_records[feat].append(rank)
                if rank <= effective_top_k:
                    top_k_counts[feat] += 1

        # Compute summary statistics
        n_samples_run = len(next(iter(rank_records.values()))) if rank_records else 0
        n_samples_run = max(n_samples_run, 1)

        results: list[BootstrapStabilityResult] = []
        for feat in all_feature_names:
            ranks_list = rank_records[feat]
            if ranks_list:
                freq = top_k_counts[feat] / n_samples_run
                mean_r = float(np.mean(ranks_list))
                std_r = float(np.std(ranks_list))
            else:
                freq = 0.0
                mean_r = float(n_features)
                std_r = 0.0

            results.append(
                BootstrapStabilityResult(
                    feature_name=feat,
                    selection_frequency=freq,
                    mean_rank=mean_r,
                    rank_std=std_r,
                    is_stable=freq >= self.stability_threshold,
                )
            )

        results.sort(key=lambda r: (-r.selection_frequency, r.mean_rank))

        n_stable = sum(1 for r in results if r.is_stable)
        logger.info(
            "Bootstrap stability: %d/%d features stable (threshold=%.2f, top_k=%d)",
            n_stable,
            len(results),
            self.stability_threshold,
            effective_top_k,
        )
        return results

    def get_stable_features(self, results: list[BootstrapStabilityResult]) -> list[str]:
        """Return sorted list of feature names that are stable.

        Args:
            results: Output from evaluate().

        Returns:
            Feature names where is_stable=True, sorted alphabetically.
        """
        return sorted(r.feature_name for r in results if r.is_stable)


__all__ = ["BootstrapFeatureStability", "BootstrapStabilityResult"]
