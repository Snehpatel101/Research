"""
Tests for Phase 99: Advanced Feature Governance.

Covers:
- E3: Robustness scoring (stability + predictive + regime composite)
- CUSUM filter (event-driven labeling)
- Fractional differentiation (FFD with memory preservation)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.features.cusum_filter import (
    cusum_events_mask,
    cusum_filter,
    get_cusum_threshold,
)
from src.data.features.frac_diff import (
    find_min_d,
    frac_diff_ffd,
)
from src.optimization.feature_selection.robustness_scoring import RobustnessScorer

# ---------------------------------------------------------------------------
# E3: Robustness Scoring
# ---------------------------------------------------------------------------


class TestRobustnessScoring:
    """E3: Feature robustness scoring on 3 dimensions."""

    def test_composite_score_range(self) -> None:
        scorer = RobustnessScorer()
        features = ["a", "b", "c"]
        mda = pd.Series({"a": 0.8, "b": 0.4, "c": 0.1})
        df = scorer.score_features(features, mda_importance=mda)
        assert (df["composite_score"] >= 0).all()
        assert (df["composite_score"] <= 1).all()

    def test_stable_features_score_higher(self) -> None:
        scorer = RobustnessScorer()
        features = ["stable", "unstable"]
        fold_counts = {"stable": 5, "unstable": 1}
        df = scorer.score_features(features, fold_counts=fold_counts, n_folds=5)
        stable_score = df.loc[df["feature"] == "stable", "composite_score"].values[0]
        unstable_score = df.loc[df["feature"] == "unstable", "composite_score"].values[0]
        assert stable_score > unstable_score

    def test_all_zero_importance_returns_zero(self) -> None:
        scorer = RobustnessScorer()
        features = ["a", "b"]
        mda = pd.Series({"a": 0.0, "b": 0.0})
        df = scorer.score_features(features, mda_importance=mda)
        assert (df["composite_score"] == 0.0).all()

    def test_missing_regime_data_still_works(self) -> None:
        scorer = RobustnessScorer()
        features = ["a", "b"]
        mda = pd.Series({"a": 0.5, "b": 0.3})
        df = scorer.score_features(features, mda_importance=mda, regime_importance=None)
        assert len(df) == 2
        assert (df["regime_robustness"] == 0.0).all()

    def test_sort_order_descending(self) -> None:
        scorer = RobustnessScorer()
        features = ["low", "mid", "high"]
        mda = pd.Series({"low": 0.1, "mid": 0.5, "high": 0.9})
        df = scorer.score_features(features, mda_importance=mda)
        scores = df["composite_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_score_features_returns_dataframe(self) -> None:
        scorer = RobustnessScorer()
        df = scorer.score_features(["a", "b"])
        assert isinstance(df, pd.DataFrame)
        expected_cols = {
            "feature",
            "stability",
            "predictive_power",
            "regime_robustness",
            "composite_score",
        }
        assert set(df.columns) == expected_cols

    def test_select_top_features(self) -> None:
        scorer = RobustnessScorer()
        mda = pd.Series({"a": 0.9, "b": 0.5, "c": 0.1})
        df = scorer.score_features(["a", "b", "c"], mda_importance=mda)
        top = scorer.select_top_features(df, 2)
        assert len(top) == 2
        assert top[0] == "a"

    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            RobustnessScorer(stability_weight=0.5, predictive_weight=0.5, regime_weight=0.5)

    def test_empty_features(self) -> None:
        scorer = RobustnessScorer()
        df = scorer.score_features([])
        assert len(df) == 0


# ---------------------------------------------------------------------------
# CUSUM Filter
# ---------------------------------------------------------------------------


class TestCUSUMFilter:
    """CUSUM filter for event-driven labeling."""

    def test_cusum_detects_large_moves(self) -> None:
        np.random.seed(42)
        returns = pd.Series(np.random.randn(1000) * 0.01)
        events = cusum_filter(returns, threshold=0.02)
        assert len(events) > 0
        assert len(events) < len(returns)

    def test_cusum_no_events_below_threshold(self) -> None:
        returns = pd.Series([0.001] * 100)
        events = cusum_filter(returns, threshold=1.0)
        assert len(events) == 0

    def test_cusum_resets_after_event(self) -> None:
        # Cumulative sum hits threshold, resets, then needs to accumulate again
        returns = pd.Series([0.01] * 5 + [0.0] * 5 + [0.01] * 5)
        events = cusum_filter(returns, threshold=0.04)
        assert len(events) >= 2  # Should trigger at least twice

    def test_cusum_events_mask_boolean(self) -> None:
        np.random.seed(42)
        returns = pd.Series(np.random.randn(500) * 0.01)
        mask = cusum_events_mask(returns, threshold=0.02)
        assert isinstance(mask, pd.Series)
        assert mask.dtype == bool
        assert mask.sum() > 0
        assert len(mask) == len(returns)

    def test_cusum_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            cusum_filter(pd.Series([0.01]), threshold=-0.01)

    def test_cusum_nan_handling(self) -> None:
        returns = pd.Series([0.01, np.nan, 0.02, 0.03, np.nan, 0.01])
        events = cusum_filter(returns, threshold=0.04)
        # Should work without crashing, events reference original positions
        assert isinstance(events, np.ndarray)

    def test_get_cusum_threshold_calibration(self) -> None:
        np.random.seed(42)
        returns = pd.Series(np.random.randn(7800) * 0.005)  # 100 days of 78 bars
        threshold = get_cusum_threshold(returns, target_events_per_day=5.0, bars_per_day=78)
        assert threshold > 0
        # Verify it's roughly right
        events = cusum_filter(returns, threshold)
        events_per_day = len(events) / 100.0
        assert 2.0 < events_per_day < 10.0  # Loose bound


# ---------------------------------------------------------------------------
# Fractional Differentiation
# ---------------------------------------------------------------------------


class TestFractionalDiff:
    """Fractional differentiation (FFD) for memory-preserving stationarity."""

    def test_frac_diff_d0_is_original(self) -> None:
        series = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        result = frac_diff_ffd(series, d=0.0)
        pd.testing.assert_series_equal(result, series)

    def test_frac_diff_preserves_memory(self) -> None:
        """d=0.3 should keep high correlation with original."""
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(500) * 0.5))
        diffed = frac_diff_ffd(prices, d=0.3)
        # Drop NaN warmup period
        valid = diffed.dropna()
        assert len(valid) > 50  # Should have enough non-NaN values
        prices_aligned = prices.loc[valid.index]
        # Use numpy for correlation to handle edge cases
        corr = np.corrcoef(prices_aligned.values, valid.values)[0, 1]
        assert corr > 0.5  # Should maintain substantial correlation

    def test_frac_diff_d1_approximates_diff(self) -> None:
        """d=1.0 should approximate first differences."""
        prices = pd.Series([100.0, 101.0, 103.0, 106.0, 110.0])
        diffed = frac_diff_ffd(prices, d=1.0)
        valid = diffed.dropna()
        # First diff of these prices: 1, 2, 3, 4
        expected_diffs = np.diff(prices.values)
        # The FFD d=1 should closely match np.diff
        assert len(valid) > 0
        # Check last few values are close
        np.testing.assert_allclose(valid.values[-len(expected_diffs) :], expected_diffs, atol=0.1)

    def test_frac_diff_nan_handling(self) -> None:
        series = pd.Series([100.0, np.nan, 102.0, 103.0, np.nan, 105.0])
        result = frac_diff_ffd(series, d=0.3)
        assert isinstance(result, pd.Series)
        assert len(result) == len(series)

    def test_frac_diff_returns_aligned_index(self) -> None:
        idx = pd.date_range("2024-01-01", periods=100, freq="h")
        series = pd.Series(np.random.randn(100).cumsum() + 100, index=idx)
        result = frac_diff_ffd(series, d=0.3)
        assert result.index.equals(series.index)

    def test_find_min_d_returns_valid_range(self) -> None:
        pytest.importorskip("statsmodels")
        np.random.seed(42)
        # Random walk (non-stationary) — should need d > 0
        prices = pd.Series(100 + np.cumsum(np.random.randn(500)))
        d = find_min_d(prices, p_value_threshold=0.05, d_step=0.1)
        assert 0.0 <= d <= 1.0
