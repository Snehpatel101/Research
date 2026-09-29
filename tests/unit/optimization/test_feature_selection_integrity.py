"""
Phase 116 regression tests: feature-selection ranking integrity.

Covers the three defects found by the audit of the live per-model ranking:
  (a) clustered "MDA" fit a forest on the cluster mean, so importance was
      target-blind (always 1.0 / cluster size) and pure noise could outrank signal;
  (b) the MDA row subsample shuffled rows before PurgedKFold, so positional
      purge/embargo was meaningless;
  (c) permutation importance was scored with accuracy instead of log-loss.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

import src.models.training.feature_selection as fs_module
from src.models.training.feature_selection import (
    FeatureSelectionMixin,
    _temporal_stride_subsample,
)
from src.optimization.feature_selection import walk_forward as wf_module
from src.optimization.feature_selection.walk_forward import (
    WalkForwardFeatureSelector,
    cluster_features,
)
from src.validation.cv import PurgedKFold, PurgedKFoldConfig


def _signal_dup_noise_data(n: int = 3000, n_noise: int = 30, seed: int = 0):
    """One real signal, a near-duplicate of it, and independent noise columns."""
    rng = np.random.default_rng(seed)
    sig = rng.normal(size=n)
    y = pd.Series((sig + 0.7 * rng.normal(size=n) > 0).astype(int))
    cols = {"signal": sig, "signal_dup": sig + 0.05 * rng.normal(size=n)}
    for i in range(n_noise):
        cols[f"noise_{i}"] = rng.normal(size=n)
    return pd.DataFrame(cols), y


def _splits(X: pd.DataFrame, y: pd.Series):
    cv = PurgedKFold(PurgedKFoldConfig(n_splits=3, purge_bars=5, embargo_bars=5))
    return list(cv.split(X, y))


def _mean_importance(result) -> pd.Series:
    return pd.DataFrame([h["importance"] for h in result.importance_history]).mean()


# ---------------------------------------------------------------------------
# (a) clustered MDA is target-aware
# ---------------------------------------------------------------------------
class TestClusteredMDAIsTargetAware:
    def test_signal_and_duplicate_outrank_every_noise_feature(self):
        X, y = _signal_dup_noise_data()
        selector = WalkForwardFeatureSelector(
            n_features_to_select=X.shape[1],
            n_estimators=50,
            min_feature_frequency=0.01,
            use_clustered_importance=True,
            max_clusters=X.shape[1],
        )
        result = selector.select_features_walkforward(X, y, _splits(X, y))
        imp = _mean_importance(result)

        noise = imp[[c for c in imp.index if c.startswith("noise_")]]
        assert imp["signal"] > noise.max()
        assert imp["signal_dup"] > noise.max()
        ranked = imp.sort_values(ascending=False).index.tolist()
        assert set(ranked[:2]) == {"signal", "signal_dup"}

    def test_noise_importance_is_not_a_function_of_cluster_size(self):
        """Old code gave noise 1/cluster_size (up to 1.0); noise must be ~0 here."""
        X, y = _signal_dup_noise_data(seed=1)
        rng = np.random.default_rng(5)
        base = rng.normal(size=len(X))
        for i in range(6):  # a large block of mutually correlated pure noise
            X[f"noiseblock_{i}"] = base + 0.1 * rng.normal(size=len(X))

        selector = WalkForwardFeatureSelector(
            n_features_to_select=X.shape[1],
            n_estimators=50,
            min_feature_frequency=0.01,
            use_clustered_importance=True,
            max_clusters=X.shape[1],
        )
        imp = _mean_importance(selector.select_features_walkforward(X, y, _splits(X, y)))

        noise = imp[[c for c in imp.index if c.startswith("noise")]]
        assert noise.abs().max() < 0.05 * imp["signal"]
        block = imp[[c for c in imp.index if c.startswith("noiseblock_")]]
        assert block.max() < imp["signal"]

    def test_importance_depends_on_the_target(self):
        """Permuting y must change the ranking: importance is not target-blind."""
        X, y = _signal_dup_noise_data(seed=2)
        rng = np.random.default_rng(9)
        y_shuffled = pd.Series(rng.permutation(y.to_numpy()))
        selector = WalkForwardFeatureSelector(
            n_features_to_select=X.shape[1],
            n_estimators=50,
            min_feature_frequency=0.01,
            use_clustered_importance=True,
            max_clusters=X.shape[1],
        )
        real = _mean_importance(selector.select_features_walkforward(X, y, _splits(X, y)))
        shuf = _mean_importance(
            selector.select_features_walkforward(X, y_shuffled, _splits(X, y_shuffled))
        )
        assert real["signal"] > 10 * max(abs(shuf["signal"]), 1e-4)

    def test_sample_weights_are_accepted(self):
        X, y = _signal_dup_noise_data(seed=3)
        w = pd.Series(np.random.default_rng(0).uniform(0.5, 1.5, len(X)))
        selector = WalkForwardFeatureSelector(
            n_features_to_select=X.shape[1],
            n_estimators=30,
            min_feature_frequency=0.01,
            use_clustered_importance=True,
            max_clusters=X.shape[1],
        )
        result = selector.select_features_walkforward(X, y, _splits(X, y), sample_weights=w)
        imp = _mean_importance(result)
        assert imp.idxmax() in {"signal", "signal_dup"}

    def test_single_fold_call_without_holdout_uses_tail_holdout(self):
        """manager.py calls _compute_importance with no holdout; must still work."""
        X, y = _signal_dup_noise_data(seed=4)
        selector = WalkForwardFeatureSelector(
            n_features_to_select=5,
            n_estimators=30,
            use_clustered_importance=True,
            max_clusters=X.shape[1],
        )
        imp = selector._compute_importance(X, y)
        assert imp.idxmax() in {"signal", "signal_dup"}

    def test_plain_holdout_mda_path_still_works(self):
        X, y = _signal_dup_noise_data(seed=5)
        selector = WalkForwardFeatureSelector(
            n_features_to_select=X.shape[1],
            n_estimators=50,
            min_feature_frequency=0.01,
            use_clustered_importance=False,
        )
        imp = _mean_importance(selector.select_features_walkforward(X, y, _splits(X, y)))
        assert imp.idxmax() in {"signal", "signal_dup"}


class TestFeatureClustering:
    def test_correlated_features_merge_and_independent_stay_apart(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=2000)
        X = pd.DataFrame(
            {"a": a, "a2": a + 0.1 * rng.normal(size=2000), "b": rng.normal(size=2000)}
        )
        c = cluster_features(X, max_clusters=10, distance_threshold=0.5)
        assert c["a"] == c["a2"]
        assert c["b"] != c["a"]

    def test_anticorrelated_features_are_not_merged(self):
        """Signed-correlation distance: rho=-1 is maximally far (|rho| merged them)."""
        rng = np.random.default_rng(1)
        a = rng.normal(size=2000)
        X = pd.DataFrame({"a": a, "neg_a": -a + 0.05 * rng.normal(size=2000)})
        c = cluster_features(X, max_clusters=10, distance_threshold=0.5)
        assert c["a"] != c["neg_a"]

    def test_max_clusters_caps_cluster_count(self):
        rng = np.random.default_rng(2)
        X = pd.DataFrame(rng.normal(size=(500, 40)), columns=[f"f{i}" for i in range(40)])
        c = cluster_features(X, max_clusters=5, distance_threshold=0.1)
        assert c.nunique() <= 5


# ---------------------------------------------------------------------------
# (b) temporal, stride-scaled subsample before purged CV
# ---------------------------------------------------------------------------
class _Harness(FeatureSelectionMixin):
    def __init__(self, purge_bars: int, embargo_bars: int) -> None:
        self.config = SimpleNamespace(  # type: ignore[assignment]
            horizons=[5], purge_bars=purge_bars, embargo_bars=embargo_bars
        )


class TestTemporalSubsample:
    def test_helper_is_strided_ordered_and_capped(self):
        df = pd.DataFrame({"a": np.arange(10_000)})
        out, stride = _temporal_stride_subsample(df, 3000)
        assert stride == 4
        assert len(out) <= 3000
        assert out.index.is_monotonic_increasing
        assert np.all(np.diff(out.index.to_numpy()) == stride)

    def test_helper_is_noop_below_cap(self):
        df = pd.DataFrame({"a": np.arange(100)})
        out, stride = _temporal_stride_subsample(df, 3000)
        assert stride == 1 and out is df

    def test_cv_receives_ordered_subsample_with_scaled_purge_and_embargo(self, monkeypatch):
        n, cap = 6000, 1500  # stride = 4
        monkeypatch.setattr(fs_module, "MDA_MAX_ROWS", cap)
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
        df["label_h5"] = rng.integers(0, 2, n)

        captured: dict[str, Any] = {}

        class SpyCV(PurgedKFold):
            def __init__(self, config):
                captured["config"] = config
                super().__init__(config)

            def split(self, X, y=None, *args, **kwargs):
                captured["X_index"] = X.index.to_numpy()
                return super().split(X, y, *args, **kwargs)

        monkeypatch.setattr(fs_module, "PurgedKFold", SpyCV)
        harness = _Harness(purge_bars=10, embargo_bars=90)
        result = harness._compute_mda_ranking(df, list("abcd"))

        assert result is not None
        idx = captured["X_index"]
        assert len(idx) <= cap
        assert np.all(np.diff(idx) > 0), "subsample must stay in temporal order"
        assert np.all(np.diff(idx) == 4), "subsample must be strided, not shuffled"
        assert captured["config"].purge_bars == 3  # ceil(10 / 4)
        assert captured["config"].embargo_bars == 23  # ceil(90 / 4), under the 15% cap


# ---------------------------------------------------------------------------
# (c) log-loss scoring
# ---------------------------------------------------------------------------
class TestLogLossScoring:
    def test_permutation_importance_uses_neg_log_loss_scorer(self, monkeypatch):
        X, y = _signal_dup_noise_data(n=600)
        captured: dict[str, Any] = {}
        original = wf_module.permutation_importance

        def spy(estimator, Xs, ys, **kwargs):
            captured.update(kwargs)
            captured["estimator"] = estimator
            captured["X"], captured["y"] = Xs, ys
            return original(estimator, Xs, ys, **kwargs)

        monkeypatch.setattr(wf_module, "permutation_importance", spy)
        selector = WalkForwardFeatureSelector(selection_method="mda", n_estimators=20)
        selector._mda_importance(
            X.iloc[:400], y.iloc[:400], X_test=X.iloc[400:], y_test=y.iloc[400:]
        )

        scorer = captured["scoring"]
        assert scorer is not None and not isinstance(scorer, str)
        est = captured["estimator"]
        expected = -log_loss(captured["y"], est.predict_proba(captured["X"]), labels=est.classes_)
        assert scorer(est, captured["X"], captured["y"]) == pytest.approx(expected)

    def test_scorer_respects_sample_weights(self):
        X, y = _signal_dup_noise_data(n=600)
        rf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=0).fit(X, y)
        w = np.random.default_rng(0).uniform(0.1, 3.0, len(X))
        scorer = wf_module._neg_log_loss_scorer(rf.classes_)
        expected = -log_loss(y, rf.predict_proba(X), sample_weight=w, labels=rf.classes_)
        assert scorer(rf, X, y, sample_weight=w) == pytest.approx(expected)

    @pytest.mark.parametrize("clustered", [False, True])
    def test_feature_that_sharpens_probabilities_but_not_argmax_is_detected(self, clustered):
        """Rare positives: P(y=1) rises 2% -> 30% where the feature is high, yet the
        arg-max is always class 0 so accuracy-based MDA sees exactly zero."""
        rng = np.random.default_rng(0)
        n = 6000
        x_sig = rng.uniform(size=n)
        p = np.where(x_sig > 0.8, 0.30, 0.02)
        y = pd.Series((rng.uniform(size=n) < p).astype(int))
        X = pd.DataFrame({"x_sig": x_sig})
        for i in range(5):
            X[f"noise_{i}"] = rng.normal(size=n)

        selector = WalkForwardFeatureSelector(
            selection_method="mda",
            n_estimators=50,
            use_clustered_importance=clustered,
            max_clusters=X.shape[1],
        )
        cut = 4000
        imp = selector._compute_importance(
            X.iloc[:cut], y.iloc[:cut], X_test=X.iloc[cut:], y_test=y.iloc[cut:]
        )
        assert imp["x_sig"] > 0.005
        assert imp["x_sig"] > 5 * imp.drop("x_sig").abs().max()


def test_rank_ordered_decorrelation_moves_past_correlated_clusters() -> None:
    """Top of the ranking is one correlated cluster: selection must reach the next ones."""
    from src.optimization.feature_selection.filtering import select_decorrelated_by_rank

    rng = np.random.default_rng(7)
    n = 2000
    base = rng.normal(size=n)
    cols = {f"cluster_{i}": base + rng.normal(scale=0.05, size=n) for i in range(10)}
    cols.update({f"indep_{i}": rng.normal(size=n) for i in range(20)})
    df = pd.DataFrame(cols)
    ranked = [f"cluster_{i}" for i in range(10)] + [f"indep_{i}" for i in range(20)]

    selected = select_decorrelated_by_rank(df, ranked, n_target=15, n_min=15)

    assert len(selected) == 15
    assert selected[0] == "cluster_0"
    assert sum(f.startswith("cluster_") for f in selected) == 1
    assert selected[1:] == [f"indep_{i}" for i in range(14)]


def test_rank_ordered_decorrelation_tops_up_to_minimum() -> None:
    from src.optimization.feature_selection.filtering import select_decorrelated_by_rank

    rng = np.random.default_rng(8)
    base = rng.normal(size=1000)
    df = pd.DataFrame({f"c{i}": base + rng.normal(scale=0.01, size=1000) for i in range(8)})
    selected = select_decorrelated_by_rank(df, list(df.columns), n_target=8, n_min=5)
    assert selected == ["c0", "c1", "c2", "c3", "c4"]
