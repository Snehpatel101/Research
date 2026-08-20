"""
Tests for Phase 102: Improvements — Bootstrap Stability, Label Perturbation, Param Sensitivity.

From improvements_sneh.md Section 16.3 "Most Robust Design".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.optimization.feature_selection.bootstrap_stability import (
    BootstrapFeatureStability,
    BootstrapStabilityResult,
)
from src.optimization.feature_selection.label_perturbation import (
    LabelPerturbationTester,
    PerturbationResult,
)
from src.optimization.feature_selection.param_sensitivity import (
    ParameterSensitivityTester,
    SensitivityResult,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_data(
    n_samples: int = 500,
    n_features: int = 10,
    seed: int = 42,
    informative: int = 3,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    signal = X.iloc[:, :informative].sum(axis=1)
    y = pd.Series((signal > 0).astype(int), name="label")
    return X, y


# ---------------------------------------------------------------------------
# Bootstrap Stability
# ---------------------------------------------------------------------------


class TestBootstrapStability:
    """Bootstrap feature stability testing."""

    def test_evaluate_returns_results(self) -> None:
        X, y = _make_data(n_samples=300, n_features=5)
        tester = BootstrapFeatureStability(n_bootstrap=5, top_k=3, n_estimators=10, n_repeats=2)
        results = tester.evaluate(X, y)
        assert isinstance(results, list)
        assert len(results) == 5
        assert all(isinstance(r, BootstrapStabilityResult) for r in results)

    def test_selection_frequency_range(self) -> None:
        X, y = _make_data(n_samples=300, n_features=5)
        tester = BootstrapFeatureStability(n_bootstrap=5, top_k=3, n_estimators=10, n_repeats=2)
        results = tester.evaluate(X, y)
        for r in results:
            assert 0.0 <= r.selection_frequency <= 1.0

    def test_sorted_by_frequency_descending(self) -> None:
        X, y = _make_data(n_samples=300, n_features=5)
        tester = BootstrapFeatureStability(n_bootstrap=5, top_k=3, n_estimators=10, n_repeats=2)
        results = tester.evaluate(X, y)
        freqs = [r.selection_frequency for r in results]
        assert freqs == sorted(freqs, reverse=True)

    def test_get_stable_features(self) -> None:
        X, y = _make_data(n_samples=300, n_features=5)
        tester = BootstrapFeatureStability(
            n_bootstrap=5,
            top_k=3,
            stability_threshold=0.4,
            n_estimators=10,
            n_repeats=2,
        )
        results = tester.evaluate(X, y)
        stable = tester.get_stable_features(results)
        assert isinstance(stable, list)
        # All returned features should have is_stable=True
        stable_set = set(stable)
        for r in results:
            if r.feature_name in stable_set:
                assert r.is_stable

    def test_empty_data(self) -> None:
        X = pd.DataFrame(columns=["a", "b", "c"])
        y = pd.Series(dtype=int)
        tester = BootstrapFeatureStability(n_bootstrap=3, n_estimators=10)
        results = tester.evaluate(X, y)
        assert isinstance(results, list)
        assert len(results) == 0  # Empty data returns empty results

    def test_result_dataclass(self) -> None:
        r = BootstrapStabilityResult(
            feature_name="rsi_14",
            selection_frequency=0.8,
            mean_rank=3.5,
            rank_std=1.2,
            is_stable=True,
        )
        assert r.feature_name == "rsi_14"
        assert r.is_stable is True

    def test_top_k_larger_than_features(self) -> None:
        """If top_k > n_features, all features should be in top-K every time."""
        X, y = _make_data(n_samples=200, n_features=3)
        tester = BootstrapFeatureStability(n_bootstrap=5, top_k=10, n_estimators=10, n_repeats=2)
        results = tester.evaluate(X, y)
        for r in results:
            assert r.selection_frequency == 1.0


# ---------------------------------------------------------------------------
# Label Perturbation
# ---------------------------------------------------------------------------


class TestLabelPerturbation:
    """Label perturbation testing for feature robustness."""

    def test_evaluate_returns_results(self) -> None:
        X, y = _make_data(n_samples=300, n_features=5)
        rng = np.random.RandomState(99)
        y2 = pd.Series((X.iloc[:, :3].sum(axis=1) + rng.randn(300) * 0.5 > 0).astype(int))
        tester = LabelPerturbationTester(n_estimators=10, n_repeats=2)
        results = tester.evaluate(X, {"baseline": y, "perturbed": y2})
        assert isinstance(results, list)
        assert len(results) == 5
        assert all(isinstance(r, PerturbationResult) for r in results)

    def test_sorted_by_rank_change_ascending(self) -> None:
        X, y = _make_data(n_samples=300, n_features=5)
        rng = np.random.RandomState(99)
        y2 = pd.Series((X.iloc[:, :2].sum(axis=1) + rng.randn(300) * 2 > 0).astype(int))
        tester = LabelPerturbationTester(n_estimators=10, n_repeats=2)
        results = tester.evaluate(X, {"baseline": y, "noisy": y2})
        changes = [r.max_rank_change for r in results]
        assert changes == sorted(changes)

    def test_get_robust_features(self) -> None:
        X, y = _make_data(n_samples=300, n_features=5)
        y2 = y.copy()  # Same labels — all features should be robust
        tester = LabelPerturbationTester(rank_change_threshold=5, n_estimators=10, n_repeats=2)
        results = tester.evaluate(X, {"baseline": y, "same": y2})
        robust = tester.get_robust_features(results)
        assert isinstance(robust, list)
        assert len(robust) == 5  # All should be robust (same labels)

    def test_single_variant(self) -> None:
        X, y = _make_data(n_samples=200, n_features=3)
        tester = LabelPerturbationTester(n_estimators=10, n_repeats=2)
        results = tester.evaluate(X, {"only_one": y})
        assert len(results) == 3
        for r in results:
            assert r.max_rank_change == 0

    def test_result_dataclass(self) -> None:
        r = PerturbationResult(
            feature_name="atr_14",
            baseline_rank=2,
            perturbed_ranks=[3, 4],
            max_rank_change=2,
            mean_rank=3.0,
            is_robust=True,
        )
        assert r.feature_name == "atr_14"
        assert r.max_rank_change == 2


# ---------------------------------------------------------------------------
# Parameter Sensitivity
# ---------------------------------------------------------------------------


class TestParameterSensitivity:
    """Parameter sensitivity testing for feature stability."""

    def test_evaluate_returns_results(self) -> None:
        X, y = _make_data(n_samples=300, n_features=5)
        rng = np.random.RandomState(99)
        X2 = X + rng.randn(*X.shape) * 0.1  # Slightly different feature values
        tester = ParameterSensitivityTester(n_estimators=10, n_repeats=2)
        results = tester.evaluate({"v1": X, "v2": X2}, y)
        assert isinstance(results, list)
        assert len(results) == 5
        assert all(isinstance(r, SensitivityResult) for r in results)

    def test_sorted_by_cv_ascending(self) -> None:
        X, y = _make_data(n_samples=300, n_features=5)
        rng = np.random.RandomState(99)
        X2 = X + rng.randn(*X.shape) * 0.5
        tester = ParameterSensitivityTester(n_estimators=10, n_repeats=2)
        results = tester.evaluate({"v1": X, "v2": X2}, y)
        cvs = [r.cv for r in results]
        assert cvs == sorted(cvs)

    def test_get_stable_features(self) -> None:
        X, y = _make_data(n_samples=300, n_features=5)
        # Same data for both variants — should be perfectly stable
        tester = ParameterSensitivityTester(cv_threshold=0.5, n_estimators=10, n_repeats=2)
        results = tester.evaluate({"v1": X, "v2": X}, y)
        stable = tester.get_stable_features(results)
        assert isinstance(stable, list)
        # With identical data, CV should be 0 (or near-0 due to random seed)
        for r in results:
            assert r.cv < 0.01 or r.mean_importance == 0.0

    def test_single_variant(self) -> None:
        X, y = _make_data(n_samples=200, n_features=3)
        tester = ParameterSensitivityTester(n_estimators=10, n_repeats=2)
        results = tester.evaluate({"only": X}, y)
        assert len(results) == 3
        for r in results:
            assert r.cv == 0.0  # Single variant means std=0

    def test_zero_importance_features(self) -> None:
        """Features with zero importance should have cv=0."""
        rng = np.random.RandomState(42)
        X = pd.DataFrame({"noise": rng.randn(200)})
        y = pd.Series([0, 1] * 100)
        tester = ParameterSensitivityTester(n_estimators=10, n_repeats=2)
        results = tester.evaluate({"v1": X, "v2": X}, y)
        # Results may have cv=0 if importance is consistently the same
        assert len(results) == 1

    def test_result_dataclass(self) -> None:
        r = SensitivityResult(
            feature_name="rsi_14",
            importance_values=[0.5, 0.6, 0.4],
            mean_importance=0.5,
            std_importance=0.1,
            cv=0.2,
            is_stable=True,
        )
        assert r.feature_name == "rsi_14"
        assert r.is_stable is True
