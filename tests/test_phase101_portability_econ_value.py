"""
Tests for Phase 101: Ticker Portability (E8) + Economic Value Scoring (E9).

Covers:
- E8: TickerPortabilityTester (importance computation, cross-symbol comparison)
- E9: EconomicValueScorer (marginal Sharpe, leave-one-out)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.optimization.feature_selection.economic_value import (
    EconomicValueScorer,
    FeatureValueScore,
)
from src.validation.ticker_portability import (
    PortabilityReport,
    PortabilityScore,
    TickerPortabilityTester,
)

# ---------------------------------------------------------------------------
# Helper to generate synthetic classification data
# ---------------------------------------------------------------------------


def _make_data(
    n_samples: int = 500,
    n_features: int = 10,
    seed: int = 42,
    informative: int = 3,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic data with some informative features."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    # Create label from first `informative` features
    signal = X.iloc[:, :informative].sum(axis=1)
    y = pd.Series((signal > 0).astype(int), name="label")
    return X, y


# ---------------------------------------------------------------------------
# E8: Ticker Portability
# ---------------------------------------------------------------------------


class TestTickerPortability:
    """E8: Cross-symbol feature portability testing."""

    def test_compute_importance_returns_series(self) -> None:
        X, y = _make_data(n_samples=200, n_features=5)
        tester = TickerPortabilityTester(n_estimators=10, n_repeats=2)
        imp = tester.compute_importance(X, y)
        assert isinstance(imp, pd.Series)
        assert len(imp) == 5
        assert (imp >= 0).all()

    def test_importance_informative_features_higher(self) -> None:
        X, y = _make_data(n_samples=500, n_features=5, informative=2)
        tester = TickerPortabilityTester(n_estimators=20, n_repeats=3)
        imp = tester.compute_importance(X, y)
        # Informative features (feat_0, feat_1) should have higher importance
        top_features = imp.nlargest(2).index.tolist()
        assert "feat_0" in top_features or "feat_1" in top_features

    def test_portability_same_distribution(self) -> None:
        """Same distribution on both symbols → high portability."""
        X1, y1 = _make_data(n_samples=300, n_features=5, seed=42)
        X2, y2 = _make_data(n_samples=300, n_features=5, seed=43)
        tester = TickerPortabilityTester(n_estimators=10, n_repeats=2)
        report = tester.test_portability(X1, y1, X2, y2, "MES", "MGC")
        assert isinstance(report, PortabilityReport)
        assert report.source_symbol == "MES"
        assert report.target_symbol == "MGC"
        assert len(report.scores) == 5

    def test_portability_report_fields(self) -> None:
        X1, y1 = _make_data(n_samples=200, n_features=3, seed=42)
        X2, y2 = _make_data(n_samples=200, n_features=3, seed=43)
        tester = TickerPortabilityTester(n_estimators=10, n_repeats=2)
        report = tester.test_portability(X1, y1, X2, y2)
        assert 0.0 <= report.overall_portability <= 1.0
        assert len(report.portable_features) + len(report.non_portable_features) == len(
            report.scores
        )

    def test_portability_score_dataclass(self) -> None:
        ps = PortabilityScore(
            feature_name="rsi_14",
            source_importance=0.8,
            target_importance=0.6,
            portability_ratio=0.75,
            is_portable=True,
        )
        assert ps.feature_name == "rsi_14"
        assert ps.is_portable is True

    def test_empty_data_returns_zeros(self) -> None:
        X = pd.DataFrame(columns=["a", "b"])
        y = pd.Series(dtype=int)
        tester = TickerPortabilityTester()
        imp = tester.compute_importance(X, y)
        assert (imp == 0.0).all()

    def test_single_class_returns_zeros(self) -> None:
        X, _ = _make_data(n_samples=100, n_features=3)
        y = pd.Series([1] * 100)
        tester = TickerPortabilityTester()
        imp = tester.compute_importance(X, y)
        assert (imp == 0.0).all()

    def test_get_portable_features(self) -> None:
        report = PortabilityReport(
            source_symbol="A",
            target_symbol="B",
            scores=[],
            portable_features=["feat_1", "feat_3"],
            non_portable_features=["feat_2"],
            overall_portability=0.67,
        )
        tester = TickerPortabilityTester()
        result = tester.get_portable_features(report)
        assert result == ["feat_1", "feat_3"]

    def test_mismatched_features(self) -> None:
        """Source and target have different feature sets."""
        rng = np.random.RandomState(42)
        X1 = pd.DataFrame(rng.randn(200, 3), columns=["a", "b", "c"])
        y1 = pd.Series((X1["a"] > 0).astype(int))
        X2 = pd.DataFrame(rng.randn(200, 3), columns=["b", "c", "d"])
        y2 = pd.Series((X2["b"] > 0).astype(int))
        tester = TickerPortabilityTester(n_estimators=10, n_repeats=2)
        report = tester.test_portability(X1, y1, X2, y2)
        all_names = {s.feature_name for s in report.scores}
        assert "a" in all_names  # Only in source
        assert "d" in all_names  # Only in target
        assert "b" in all_names  # In both


# ---------------------------------------------------------------------------
# E9: Economic Value Scoring
# ---------------------------------------------------------------------------


class TestEconomicValueScorer:
    """E9: Leave-one-out marginal Sharpe scoring."""

    def test_score_features_returns_dataframe(self) -> None:
        X, y = _make_data(n_samples=300, n_features=5)
        scorer = EconomicValueScorer(n_estimators=10)
        df = scorer.score_features(X, y)
        assert isinstance(df, pd.DataFrame)
        expected_cols = {
            "feature_name",
            "baseline_sharpe",
            "without_sharpe",
            "marginal_sharpe",
            "rank",
        }
        assert set(df.columns) == expected_cols
        assert len(df) == 5

    def test_ranks_are_sequential(self) -> None:
        X, y = _make_data(n_samples=300, n_features=5)
        scorer = EconomicValueScorer(n_estimators=10)
        df = scorer.score_features(X, y)
        assert list(df["rank"]) == [1, 2, 3, 4, 5]

    def test_marginal_sharpe_computation(self) -> None:
        X, y = _make_data(n_samples=300, n_features=5)
        scorer = EconomicValueScorer(n_estimators=10)
        df = scorer.score_features(X, y)
        # marginal = baseline - without
        for _, row in df.iterrows():
            expected = row["baseline_sharpe"] - row["without_sharpe"]
            assert abs(row["marginal_sharpe"] - expected) < 1e-10

    def test_sorted_by_marginal_descending(self) -> None:
        X, y = _make_data(n_samples=300, n_features=5)
        scorer = EconomicValueScorer(n_estimators=10)
        df = scorer.score_features(X, y)
        marginals = df["marginal_sharpe"].tolist()
        assert marginals == sorted(marginals, reverse=True)

    def test_select_valuable_features(self) -> None:
        X, y = _make_data(n_samples=300, n_features=5)
        scorer = EconomicValueScorer(n_estimators=10)
        df = scorer.score_features(X, y)
        # Select features with marginal > 0
        valuable = scorer.select_valuable_features(df, min_marginal=0.0)
        assert isinstance(valuable, list)
        # All returned features should have positive marginal
        for feat in valuable:
            row = df[df["feature_name"] == feat]
            assert row["marginal_sharpe"].values[0] > 0.0

    def test_empty_features_returns_empty(self) -> None:
        X = pd.DataFrame()
        y = pd.Series(dtype=int)
        scorer = EconomicValueScorer()
        df = scorer.score_features(X, y, feature_names=[])
        assert len(df) == 0

    def test_single_class_returns_zeros(self) -> None:
        X, _ = _make_data(n_samples=100, n_features=3)
        y = pd.Series([1] * 100)
        scorer = EconomicValueScorer(n_estimators=10)
        df = scorer.score_features(X, y)
        assert (df["marginal_sharpe"] == 0.0).all()

    def test_feature_subset(self) -> None:
        """Score only a subset of features."""
        X, y = _make_data(n_samples=300, n_features=5)
        scorer = EconomicValueScorer(n_estimators=10)
        df = scorer.score_features(X, y, feature_names=["feat_0", "feat_1"])
        assert len(df) == 2

    def test_sharpe_computation(self) -> None:
        scorer = EconomicValueScorer(annualization_factor=252.0)
        y_true = np.array([1, -1, 1, 1, -1, 1, 1, -1])
        y_pred = np.array([1, -1, 1, 1, -1, 1, -1, -1])
        sharpe = scorer._compute_sharpe(y_true, y_pred)
        assert isinstance(sharpe, float)
        assert sharpe > 0  # More correct than not

    def test_sharpe_zero_std(self) -> None:
        scorer = EconomicValueScorer()
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        sharpe = scorer._compute_sharpe(y_true, y_pred)
        assert sharpe == 0.0

    def test_feature_value_score_dataclass(self) -> None:
        fvs = FeatureValueScore(
            feature_name="rsi_14",
            baseline_sharpe=1.5,
            without_sharpe=1.2,
            marginal_sharpe=0.3,
            rank=1,
        )
        assert fvs.feature_name == "rsi_14"
        assert fvs.marginal_sharpe == 0.3
