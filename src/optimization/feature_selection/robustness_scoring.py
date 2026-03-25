"""
Feature robustness scoring across stability, predictive power, and regime dimensions.

Scores each feature on three dimensions:
1. Stability (0-1): How often a feature is selected across CV folds
2. Predictive Power (0-1): Normalized MDA importance score
3. Regime Robustness (0-1): Normalized max importance across regimes

The composite score is a weighted combination of all three dimensions,
defaulting to 0.4 * stability + 0.4 * predictive + 0.2 * regime_robustness.

Reference: Phase E3 of Feature Governance roadmap.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class RobustnessScorer:
    """Score features on stability, predictive power, and regime robustness."""

    def __init__(
        self,
        stability_weight: float = 0.4,
        predictive_weight: float = 0.4,
        regime_weight: float = 0.2,
    ) -> None:
        weights_sum = stability_weight + predictive_weight + regime_weight
        if abs(weights_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {weights_sum:.6f} "
                f"(stability={stability_weight}, predictive={predictive_weight}, "
                f"regime={regime_weight})"
            )
        if stability_weight < 0 or predictive_weight < 0 or regime_weight < 0:
            raise ValueError("All weights must be non-negative")

        self.stability_weight = stability_weight
        self.predictive_weight = predictive_weight
        self.regime_weight = regime_weight

    def score_features(
        self,
        feature_names: list[str],
        fold_counts: dict[str, int] | None = None,
        n_folds: int = 5,
        mda_importance: pd.Series | None = None,
        regime_importance: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Score features and return DataFrame sorted by composite score.

        Args:
            feature_names: List of feature names to score.
            fold_counts: Dict mapping feature name to number of folds it was
                selected in. If None, stability scores are 0.
            n_folds: Total number of CV folds (used to normalize fold_counts).
            mda_importance: Series of MDA importance scores indexed by feature
                name. If None, predictive power scores are 0.
            regime_importance: Series of regime importance scores (e.g. max
                across regimes) indexed by feature name. If None, regime
                robustness scores are 0.

        Returns:
            DataFrame with columns: feature, stability, predictive_power,
            regime_robustness, composite_score. Sorted by composite_score
            descending.
        """
        if not feature_names:
            return pd.DataFrame(
                columns=[
                    "feature",
                    "stability",
                    "predictive_power",
                    "regime_robustness",
                    "composite_score",
                ]
            )

        n_folds = max(n_folds, 1)

        # Stability: fold frequency, capped at 1.0
        stability = {}
        for f in feature_names:
            if fold_counts is not None and f in fold_counts:
                stability[f] = min(fold_counts[f] / n_folds, 1.0)
            else:
                stability[f] = 0.0

        # Predictive power: normalized MDA importance
        predictive = _normalize_series(mda_importance, feature_names)

        # Regime robustness: normalized regime importance
        regime = _normalize_series(regime_importance, feature_names)

        # Build result DataFrame
        rows = []
        for f in feature_names:
            s = stability[f]
            p = predictive.get(f, 0.0)
            r = regime.get(f, 0.0)
            composite = (
                self.stability_weight * s + self.predictive_weight * p + self.regime_weight * r
            )
            rows.append(
                {
                    "feature": f,
                    "stability": s,
                    "predictive_power": p,
                    "regime_robustness": r,
                    "composite_score": composite,
                }
            )

        df = pd.DataFrame(rows)
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)

        logger.info(
            f"Robustness scoring: {len(df)} features scored, "
            f"top composite={df['composite_score'].iloc[0]:.4f}"
        )
        return df

    def select_top_features(self, scores_df: pd.DataFrame, n_features: int) -> list[str]:
        """Return top N features by composite score.

        Args:
            scores_df: DataFrame from score_features().
            n_features: Number of features to select.

        Returns:
            List of feature names, ordered by composite score descending.
        """
        n_features = max(n_features, 0)
        return scores_df["feature"].head(n_features).tolist()


def _normalize_series(series: pd.Series | None, feature_names: list[str]) -> dict[str, float]:
    """Normalize a Series to [0, 1] by dividing by max value.

    Args:
        series: Importance series indexed by feature name. May be None.
        feature_names: Full list of features to score.

    Returns:
        Dict mapping feature name to normalized score in [0, 1].
    """
    if series is None or series.empty:
        return dict.fromkeys(feature_names, 0.0)

    max_val = series.max()
    if max_val <= 0:
        return dict.fromkeys(feature_names, 0.0)

    result = {}
    for f in feature_names:
        if f in series.index:
            val = series[f]
            result[f] = max(val / max_val, 0.0)
        else:
            result[f] = 0.0
    return result


__all__ = ["RobustnessScorer"]
