"""Economic value scoring for feature selection via marginal Sharpe contribution.

This is the "slow path" complement to the fast-path robustness scorer (E3).
For each feature, we estimate its marginal contribution to portfolio Sharpe ratio
using a leave-one-out approach: train a proxy model with all features, then
retrain without each feature individually. The drop in Sharpe indicates that
feature's economic value.

Reference: Phase E9 of Feature Governance roadmap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class FeatureValueScore:
    """Economic value score for a single feature."""

    feature_name: str
    baseline_sharpe: float
    without_sharpe: float
    marginal_sharpe: float
    rank: int = 0


class EconomicValueScorer:
    """Estimate marginal Sharpe ratio contribution of each feature.

    Uses a leave-one-out approach with a lightweight RandomForest proxy:
    1. Train on all features -> baseline Sharpe
    2. For each feature, train without it -> without_sharpe
    3. marginal_sharpe = baseline - without (positive = feature adds value)

    This is intentionally slow (N+1 model fits) and designed for the final
    feature selection stage where precision matters more than speed.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the proxy RandomForest.
    test_size : float
        Fraction of data reserved for Sharpe evaluation.
    max_samples : int
        Cap on rows to prevent excessive training time on large datasets.
    annualization_factor : float
        Multiplier for Sharpe annualization (e.g. 252 for daily).
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        test_size: float = 0.2,
        max_samples: int = 50_000,
        annualization_factor: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.test_size = test_size
        self.max_samples = max_samples
        self.annualization_factor = annualization_factor
        self.random_state = random_state

    def _compute_sharpe(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Sharpe-like metric from signed strategy returns.

        strategy_returns = y_true * y_pred (directional agreement).
        Sharpe = mean(returns) / std(returns) * sqrt(annualization_factor).

        Returns 0.0 if standard deviation is near-zero.
        """
        strategy_returns = y_true * y_pred
        std = np.std(strategy_returns)
        if std < 1e-10:
            return 0.0
        sharpe = np.mean(strategy_returns) / std * np.sqrt(self.annualization_factor)
        return float(sharpe)

    def _fit_and_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
    ) -> float:
        """Train proxy model and return Sharpe on test set."""
        if X_train.shape[1] == 0:
            return 0.0

        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return self._compute_sharpe(y_test, y_pred)

    def score_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """Score each feature by its marginal Sharpe contribution.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target labels (signed: -1, 0, +1 or similar).
        feature_names : list[str] | None
            Features to evaluate. If None, uses all columns of X.

        Returns
        -------
        pd.DataFrame
            Columns: feature_name, baseline_sharpe, without_sharpe,
            marginal_sharpe, rank. Sorted by marginal_sharpe descending.
        """
        if feature_names is None:
            feature_names = list(X.columns)

        # Edge case: empty feature set
        if len(feature_names) == 0 or X.shape[0] == 0:
            logger.warning("Empty feature set or data — returning empty scores")
            return pd.DataFrame(
                columns=[
                    "feature_name",
                    "baseline_sharpe",
                    "without_sharpe",
                    "marginal_sharpe",
                    "rank",
                ]
            )

        # Subsample if dataset is large
        if X.shape[0] > self.max_samples:
            logger.info(
                "Subsampling %d -> %d rows for economic scoring",
                X.shape[0],
                self.max_samples,
            )
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(X.shape[0], self.max_samples, replace=False)
            X = X.iloc[idx].reset_index(drop=True)
            y = y.iloc[idx].reset_index(drop=True)

        # Edge case: single class in y
        if y.nunique() < 2:
            logger.warning(
                "Only %d unique class(es) in y — all Sharpe values will be 0.0",
                y.nunique(),
            )
            scores = [
                FeatureValueScore(
                    feature_name=f,
                    baseline_sharpe=0.0,
                    without_sharpe=0.0,
                    marginal_sharpe=0.0,
                    rank=i + 1,
                )
                for i, f in enumerate(feature_names)
            ]
            return self._scores_to_df(scores)

        # Train/test split
        X_sub = X[feature_names]
        X_train, X_test, y_train, y_test = train_test_split(
            X_sub,
            y.values,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y.values,
        )

        # Baseline: all features
        baseline_sharpe = self._fit_and_evaluate(X_train, y_train, X_test, y_test)
        logger.info(
            "Baseline Sharpe (all %d features): %.4f",
            len(feature_names),
            baseline_sharpe,
        )

        # Leave-one-out scoring
        scores: list[FeatureValueScore] = []
        n_features = len(feature_names)

        for i, feat in enumerate(feature_names):
            remaining = [f for f in feature_names if f != feat]

            # Edge case: single feature (removing it leaves nothing)
            if len(remaining) == 0:
                without_sharpe = 0.0
            else:
                without_sharpe = self._fit_and_evaluate(
                    X_train[remaining],
                    y_train,
                    X_test[remaining],
                    y_test,
                )

            marginal = baseline_sharpe - without_sharpe

            scores.append(
                FeatureValueScore(
                    feature_name=feat,
                    baseline_sharpe=baseline_sharpe,
                    without_sharpe=without_sharpe,
                    marginal_sharpe=marginal,
                )
            )

            if (i + 1) % 10 == 0 or (i + 1) == n_features:
                logger.info(
                    "Economic scoring progress: %d/%d features",
                    i + 1,
                    n_features,
                )

        # Sort by marginal Sharpe descending and assign ranks
        scores.sort(key=lambda s: s.marginal_sharpe, reverse=True)
        for rank, score in enumerate(scores, start=1):
            score.rank = rank

        return self._scores_to_df(scores)

    def select_valuable_features(
        self, scores_df: pd.DataFrame, min_marginal: float = 0.0
    ) -> list[str]:
        """Return features with marginal Sharpe above threshold.

        Parameters
        ----------
        scores_df : pd.DataFrame
            Output from score_features().
        min_marginal : float
            Minimum marginal Sharpe contribution to keep a feature.

        Returns
        -------
        list[str]
            Feature names sorted by marginal_sharpe descending.
        """
        filtered = scores_df[scores_df["marginal_sharpe"] > min_marginal]
        filtered = filtered.sort_values("marginal_sharpe", ascending=False)
        return list(filtered["feature_name"])

    @staticmethod
    def _scores_to_df(scores: list[FeatureValueScore]) -> pd.DataFrame:
        """Convert list of FeatureValueScore to DataFrame."""
        if not scores:
            return pd.DataFrame(
                columns=[
                    "feature_name",
                    "baseline_sharpe",
                    "without_sharpe",
                    "marginal_sharpe",
                    "rank",
                ]
            )
        return pd.DataFrame(
            [
                {
                    "feature_name": s.feature_name,
                    "baseline_sharpe": s.baseline_sharpe,
                    "without_sharpe": s.without_sharpe,
                    "marginal_sharpe": s.marginal_sharpe,
                    "rank": s.rank,
                }
                for s in scores
            ]
        )


__all__ = ["EconomicValueScorer", "FeatureValueScore"]
