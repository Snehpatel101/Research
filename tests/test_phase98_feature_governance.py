"""
Tests for Phase 98: Feature Governance (E2 MDA, E4 Timeframe Budget, E5 Regime Selection).

Covers:
- E2: MDA stabilization (n_estimators=50, configurable mda_n_repeats)
- E4: Timeframe competition (per-timeframe feature budget)
- E5: Regime-conditional feature selection
- Config serialization round-trip for new fields
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization.feature_selection.config import (
    FeatureSelectionConfig,
    FeatureSelectorConfig,
)
from src.optimization.feature_selection.regime_selection import (
    _detect_regimes,
    compute_regime_importance,
)
from src.optimization.feature_selection.timeframe_budget import (
    MTF_SUFFIXES,
    apply_timeframe_budget,
)
from src.optimization.feature_selection.walk_forward import WalkForwardFeatureSelector

# ---------------------------------------------------------------------------
# E2: MDA Stabilization
# ---------------------------------------------------------------------------


class TestMDAStabilization:
    """E2: MDA configurable n_repeats and n_estimators defaults."""

    def test_feature_selection_config_has_mda_n_repeats(self) -> None:
        cfg = FeatureSelectionConfig()
        assert cfg.mda_n_repeats == 5

    def test_feature_selector_config_has_mda_n_repeats(self) -> None:
        cfg = FeatureSelectorConfig()
        assert cfg.mda_n_repeats == 5

    def test_mda_n_repeats_validation_positive(self) -> None:
        with pytest.raises(ValueError, match="mda_n_repeats must be > 0"):
            FeatureSelectionConfig(mda_n_repeats=0)

    def test_feature_selector_mda_n_repeats_validation(self) -> None:
        with pytest.raises(ValueError, match="mda_n_repeats must be > 0"):
            FeatureSelectorConfig(mda_n_repeats=-1)

    def test_walk_forward_selector_default_n_estimators(self) -> None:
        selector = WalkForwardFeatureSelector()
        assert selector.config.n_estimators == 50

    def test_walk_forward_selector_mda_n_repeats(self) -> None:
        selector = WalkForwardFeatureSelector(mda_n_repeats=10)
        assert selector.config.mda_n_repeats == 10


# ---------------------------------------------------------------------------
# E4: Timeframe Competition
# ---------------------------------------------------------------------------


class TestTimeframeBudget:
    """E4: Per-timeframe feature budget to prevent MTF redundancy."""

    def test_mtf_suffixes_defined(self) -> None:
        assert "_5min" in MTF_SUFFIXES
        assert "_60min" in MTF_SUFFIXES
        assert len(MTF_SUFFIXES) == 5

    def test_base_features_kept_unconditionally(self) -> None:
        """Base features (no timeframe suffix) are always kept."""
        ranking = pd.Series({"rsi_14": 0.9, "adx_14": 0.8, "log_return": 0.7})
        result = apply_timeframe_budget(ranking, list(ranking.index), max_per_timeframe=1)
        assert set(result) == {"rsi_14", "adx_14", "log_return"}

    def test_mtf_features_budgeted(self) -> None:
        """Only top N per timeframe survive."""
        features = [
            "rsi_14_5min",
            "macd_line_5min",
            "atr_14_5min",
            "rsi_14_15min",
            "macd_line_15min",
            "base_feat",
        ]
        ranking = pd.Series(
            {
                "rsi_14_5min": 0.9,
                "macd_line_5min": 0.6,
                "atr_14_5min": 0.3,
                "rsi_14_15min": 0.8,
                "macd_line_15min": 0.5,
                "base_feat": 0.7,
            }
        )
        result = apply_timeframe_budget(ranking, features, max_per_timeframe=2)
        # 5min: top 2 = rsi_14_5min, macd_line_5min (atr dropped)
        assert "rsi_14_5min" in result
        assert "macd_line_5min" in result
        assert "atr_14_5min" not in result
        # 15min: top 2 = both kept
        assert "rsi_14_15min" in result
        assert "macd_line_15min" in result
        # base: always kept
        assert "base_feat" in result

    def test_no_mtf_features_passthrough(self) -> None:
        """If no MTF features, returns all features unchanged."""
        features = ["rsi_14", "adx_14", "log_return"]
        ranking = pd.Series({"rsi_14": 0.9, "adx_14": 0.8, "log_return": 0.7})
        result = apply_timeframe_budget(ranking, features, max_per_timeframe=2)
        assert len(result) == 3

    def test_empty_ranking_passthrough(self) -> None:
        """Empty ranking returns features unchanged."""
        features = ["a_5min", "b_5min"]
        result = apply_timeframe_budget(pd.Series(dtype=float), features, max_per_timeframe=1)
        assert len(result) == 2

    def test_budget_respects_max_per_timeframe(self) -> None:
        """Exactly max_per_timeframe features kept per group."""
        features = [f"feat_{i}_5min" for i in range(20)]
        ranking = pd.Series({f: 1.0 / (i + 1) for i, f in enumerate(features)})
        result = apply_timeframe_budget(ranking, features, max_per_timeframe=5)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# E5: Regime-Conditional Selection
# ---------------------------------------------------------------------------


class TestRegimeSelection:
    """E5: Regime-conditional feature importance."""

    @pytest.fixture
    def regime_df(self) -> pd.DataFrame:
        """Create a DataFrame with close prices and features for regime testing."""
        np.random.seed(42)
        n = 1000
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame(
            {
                "close": close,
                "feat_a": np.random.randn(n),
                "feat_b": np.random.randn(n),
                "feat_c": np.random.randn(n),
                "label_h1": np.random.choice([0, 1, 2], size=n),
            }
        )
        return df

    def test_detect_regimes_returns_series(self, regime_df: pd.DataFrame) -> None:
        result = _detect_regimes(regime_df, n_regimes=2)
        assert result is not None
        assert isinstance(result, pd.Series)
        assert len(result) == len(regime_df)
        # Should have 0 and 1 (plus NaN for warmup)
        unique = result.dropna().unique()
        assert set(unique.astype(int)) == {0, 1}

    def test_detect_regimes_3_regimes(self, regime_df: pd.DataFrame) -> None:
        result = _detect_regimes(regime_df, n_regimes=3)
        assert result is not None
        unique = result.dropna().unique()
        assert set(unique.astype(int)) == {0, 1, 2}

    def test_detect_regimes_no_close_col(self) -> None:
        df = pd.DataFrame({"feat": [1, 2, 3]})
        result = _detect_regimes(df)
        assert result is None

    def test_compute_regime_importance_returns_tuple(self, regime_df: pd.DataFrame) -> None:
        features = ["feat_a", "feat_b", "feat_c"]
        result = compute_regime_importance(
            regime_df,
            features,
            label_col="label_h1",
            min_samples_per_regime=50,
            n_estimators=10,
            n_repeats=2,
        )
        assert result is not None
        combined, per_regime = result
        assert isinstance(combined, pd.Series)
        assert isinstance(per_regime, dict)
        assert len(combined) == 3
        # At least 1 regime should have been analyzed
        assert len(per_regime) >= 1

    def test_compute_regime_importance_insufficient_data(self) -> None:
        """Too few samples should return None."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "close": [100, 101, 102, 103, 104],
                "feat_a": [1, 2, 3, 4, 5],
                "label_h1": [0, 1, 0, 1, 0],
            }
        )
        result = compute_regime_importance(
            df,
            ["feat_a"],
            label_col="label_h1",
            min_samples_per_regime=100,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Config Serialization
# ---------------------------------------------------------------------------


class TestConfigSerialization:
    """Config round-trip for new Phase 98 fields."""

    def test_to_dict_has_new_fields(self) -> None:
        cfg = FeatureSelectionConfig(mtf_max_per_timeframe=12, regime_conditional=True)
        d = cfg.to_dict()
        assert d["mtf_max_per_timeframe"] == 12
        assert d["regime_conditional"] is True
        assert d["mda_n_repeats"] == 5

    def test_from_dict_round_trip(self) -> None:
        cfg = FeatureSelectionConfig(
            mtf_max_per_timeframe=6,
            regime_conditional=True,
            mda_n_repeats=10,
        )
        d = cfg.to_dict()
        cfg2 = FeatureSelectionConfig.from_dict(d)
        assert cfg2.mtf_max_per_timeframe == 6
        assert cfg2.regime_conditional is True
        assert cfg2.mda_n_repeats == 10

    def test_from_dict_backward_compat(self) -> None:
        """Old configs without new fields should still load with defaults."""
        d = {
            "enabled": True,
            "n_features": 50,
            "method": "mda",
            "min_feature_frequency": 0.6,
            "n_estimators": 50,
            "mda_n_repeats": 5,
            "use_clustered_importance": False,
            "max_clusters": 20,
            "random_state": 42,
            "model_family": None,
        }
        cfg = FeatureSelectionConfig.from_dict(d)
        assert cfg.mtf_max_per_timeframe == 8  # default
        assert cfg.regime_conditional is False  # default
