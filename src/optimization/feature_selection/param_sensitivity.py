"""Parameter sensitivity testing for feature selection.

Checks whether feature importance is stable across different parameter choices.
Features important only under very specific parameters are fragile.

Provide multiple DataFrames (one per parameter variant) with identical columns.
Permutation importance is computed on each, and features with low coefficient
of variation (CV) across variants are deemed stable.

Reference: improvements_sneh.md Section 16.3 item 4.
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
class SensitivityResult:
    """Sensitivity result for a single feature across parameter variants."""

    feature_name: str
    importance_values: list[float]
    mean_importance: float
    std_importance: float
    cv: float
    is_stable: bool


class ParameterSensitivityTester:
    """Test whether feature importance is robust to parameter variation.

    For each variant DataFrame, computes permutation importance using a
    lightweight RandomForest proxy. Then for each feature, CV = std / mean
    across variants measures stability. Low CV = parameter-robust.
    """

    def __init__(
        self,
        cv_threshold: float = 0.5,
        n_estimators: int = 50,
        n_repeats: int = 3,
        max_samples: int = 50_000,
        random_state: int = 42,
    ) -> None:
        self.cv_threshold = cv_threshold
        self.n_estimators = n_estimators
        self.n_repeats = n_repeats
        self.max_samples = max_samples
        self.random_state = random_state

    def _compute_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Compute permutation importance for all features.

        Returns a Series indexed by feature name with mean importance values.
        Returns zeros if the data is empty or has fewer than 2 classes.
        """
        if X.shape[0] == 0 or X.shape[1] == 0:
            return pd.Series(0.0, index=X.columns)

        if y.nunique() < 2:
            logger.warning("Single class in target — returning zero importance")
            return pd.Series(0.0, index=X.columns)

        # Subsample large datasets
        if X.shape[0] > self.max_samples:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(X.shape[0], self.max_samples, replace=False)
            X = X.iloc[idx].reset_index(drop=True)
            y = y.iloc[idx].reset_index(drop=True)

        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        clf.fit(X, y)

        result = permutation_importance(
            clf,
            X,
            y,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            n_jobs=-1,
        )
        return pd.Series(result.importances_mean, index=X.columns)

    def evaluate(
        self,
        X_variants: dict[str, pd.DataFrame],
        y: pd.Series,
    ) -> list[SensitivityResult]:
        """Evaluate feature importance stability across parameter variants.

        X_variants maps variant name (e.g. "rsi_12") to feature matrices.
        All must have the same columns. Returns results sorted by CV ascending.
        """
        if not X_variants:
            logger.warning("No variants provided — returning empty results")
            return []

        variant_names = list(X_variants.keys())
        first_df = X_variants[variant_names[0]]
        feature_names = list(first_df.columns)

        if not feature_names:
            logger.warning("No features in variant DataFrames — returning empty results")
            return []

        # Compute importance for each variant
        importance_map: dict[str, pd.Series] = {}
        for name, X_var in X_variants.items():
            logger.info("Computing importance for variant '%s' (%d rows)", name, len(X_var))
            importance_map[name] = self._compute_importance(X_var, y)

        # Aggregate per feature
        results: list[SensitivityResult] = []
        for feat in feature_names:
            values = [float(importance_map[v].get(feat, 0.0)) for v in variant_names]
            mean_imp = float(np.mean(values))
            std_imp = float(np.std(values))
            cv = std_imp / mean_imp if mean_imp > 0 else 0.0
            results.append(
                SensitivityResult(
                    feature_name=feat,
                    importance_values=values,
                    mean_importance=mean_imp,
                    std_importance=std_imp,
                    cv=cv,
                    is_stable=cv <= self.cv_threshold,
                )
            )

        results.sort(key=lambda r: r.cv)

        n_stable = sum(1 for r in results if r.is_stable)
        logger.info(
            "Sensitivity analysis: %d/%d features stable (cv <= %.2f) across %d variants",
            n_stable,
            len(results),
            self.cv_threshold,
            len(variant_names),
        )
        return results

    def get_stable_features(self, results: list[SensitivityResult]) -> list[str]:
        """Return alphabetically sorted feature names where is_stable is True."""
        return sorted(r.feature_name for r in results if r.is_stable)


__all__ = ["ParameterSensitivityTester", "SensitivityResult"]
