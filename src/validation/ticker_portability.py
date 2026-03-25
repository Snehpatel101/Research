"""Ticker Portability Testing — test whether features generalize across symbols.

Compares feature importance between a source symbol (e.g., MES) and a target
symbol (e.g., MGC) to identify which features are portable vs. symbol-specific.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

MAX_SUBSAMPLE_ROWS = 50_000


@dataclass
class PortabilityScore:
    """Importance comparison for a single feature across two symbols."""

    feature_name: str
    source_importance: float
    target_importance: float
    portability_ratio: float
    is_portable: bool


@dataclass
class PortabilityReport:
    """Full portability comparison between two symbols."""

    source_symbol: str
    target_symbol: str
    scores: list[PortabilityScore]
    portable_features: list[str] = field(default_factory=list)
    non_portable_features: list[str] = field(default_factory=list)
    overall_portability: float = 0.0


class TickerPortabilityTester:
    """Test whether features generalize across different trading symbols.

    Uses permutation importance on a RandomForest to measure how important
    each feature is on a source symbol vs. a target symbol. Features whose
    importance ratio (target/source) exceeds the threshold are considered
    portable.

    Parameters
    ----------
    portability_threshold : float
        Minimum target/source importance ratio to consider a feature portable.
    n_estimators : int
        Number of trees in the RandomForest used for importance.
    n_repeats : int
        Number of permutation importance repeats.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        portability_threshold: float = 0.5,
        n_estimators: int = 50,
        n_repeats: int = 3,
        random_state: int = 42,
    ) -> None:
        self.portability_threshold = portability_threshold
        self.n_estimators = n_estimators
        self.n_repeats = n_repeats
        self.random_state = random_state

    def compute_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Compute permutation importance for all features.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target labels.

        Returns
        -------
        pd.Series
            Importance values indexed by feature name. Returns zeros on failure.
        """
        if X.empty or len(y) == 0:
            logger.warning("Empty data provided — returning zero importance.")
            return pd.Series(0.0, index=X.columns, dtype=np.float64)

        unique_classes = y.nunique()
        if unique_classes < 2:
            logger.warning("Only %d class in target — cannot compute importance.", unique_classes)
            return pd.Series(0.0, index=X.columns, dtype=np.float64)

        # Subsample large datasets
        if len(X) > MAX_SUBSAMPLE_ROWS:
            logger.info(
                "Subsampling from %d to %d rows for importance.",
                len(X),
                MAX_SUBSAMPLE_ROWS,
            )
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(len(X), MAX_SUBSAMPLE_ROWS, replace=False)
            X = X.iloc[idx]
            y = y.iloc[idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )

        try:
            clf.fit(X_train, y_train)
            result = permutation_importance(
                clf,
                X_test,
                y_test,
                n_repeats=self.n_repeats,
                random_state=self.random_state,
                n_jobs=-1,
            )
            importances = pd.Series(result.importances_mean, index=X.columns, dtype=np.float64)
            # Clip negatives to zero
            importances = importances.clip(lower=0.0)
            return importances
        except Exception:
            logger.exception("Permutation importance failed — returning zeros.")
            return pd.Series(0.0, index=X.columns, dtype=np.float64)

    def test_portability(
        self,
        X_source: pd.DataFrame,
        y_source: pd.Series,
        X_target: pd.DataFrame,
        y_target: pd.Series,
        source_symbol: str = "source",
        target_symbol: str = "target",
    ) -> PortabilityReport:
        """Compare feature importance between source and target symbols.

        Parameters
        ----------
        X_source, y_source : pd.DataFrame, pd.Series
            Feature matrix and labels for the source symbol.
        X_target, y_target : pd.DataFrame, pd.Series
            Feature matrix and labels for the target symbol.
        source_symbol, target_symbol : str
            Names used in the report.

        Returns
        -------
        PortabilityReport
            Detailed comparison of feature portability.
        """
        logger.info(
            "Testing portability: %s → %s (%d / %d features)",
            source_symbol,
            target_symbol,
            len(X_source.columns),
            len(X_target.columns),
        )

        source_imp = self.compute_importance(X_source, y_source)
        target_imp = self.compute_importance(X_target, y_target)

        all_features = sorted(set(source_imp.index) | set(target_imp.index))

        scores: list[PortabilityScore] = []
        portable: list[str] = []
        non_portable: list[str] = []

        for feat in all_features:
            s_imp = float(source_imp.get(feat, 0.0))
            t_imp = float(target_imp.get(feat, 0.0))

            if s_imp > 0.0:
                ratio = np.clip(t_imp / s_imp, 0.0, 2.0)
            elif t_imp > 0.0:
                # Source has zero importance but target has some — treat as portable
                ratio = 2.0
            else:
                # Both zero — not informative on either symbol
                ratio = 0.0

            is_port = ratio >= self.portability_threshold

            scores.append(
                PortabilityScore(
                    feature_name=feat,
                    source_importance=s_imp,
                    target_importance=t_imp,
                    portability_ratio=float(ratio),
                    is_portable=is_port,
                )
            )

            if is_port:
                portable.append(feat)
            else:
                non_portable.append(feat)

        overall = len(portable) / len(all_features) if all_features else 0.0

        logger.info(
            "Portability %s → %s: %.1f%% (%d/%d features portable)",
            source_symbol,
            target_symbol,
            overall * 100,
            len(portable),
            len(all_features),
        )

        return PortabilityReport(
            source_symbol=source_symbol,
            target_symbol=target_symbol,
            scores=scores,
            portable_features=sorted(portable),
            non_portable_features=sorted(non_portable),
            overall_portability=overall,
        )

    def get_portable_features(self, report: PortabilityReport) -> list[str]:
        """Return sorted list of portable feature names from a report."""
        return sorted(report.portable_features)


__all__ = [
    "PortabilityScore",
    "PortabilityReport",
    "TickerPortabilityTester",
]
