"""
Tests for Phase 98 feature governance: MDA stabilization (E2) and the MTF
timeframe budget (E4, ``data.features.mtf_max_per_timeframe``).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.optimization.feature_selection.config import FeatureSelectorConfig
from src.optimization.feature_selection.timeframe_budget import apply_timeframe_budget
from src.optimization.feature_selection.walk_forward import WalkForwardFeatureSelector

# ---------------------------------------------------------------------------
# E2: MDA Stabilization
# ---------------------------------------------------------------------------


class TestMDAStabilization:
    """E2: MDA configurable n_repeats and n_estimators defaults."""

    def test_feature_selector_config_has_mda_n_repeats(self) -> None:
        cfg = FeatureSelectorConfig()
        assert cfg.mda_n_repeats == 5

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
# E4: Timeframe budget
# ---------------------------------------------------------------------------


class TestTimeframeBudget:
    """Per-timeframe cap on MTF features, keyed by the generator's column suffixes."""

    def test_budget_keeps_top_ranked_per_timeframe(self) -> None:
        ranking = pd.Series(
            {
                "rsi_14_15m": 0.9,
                "base_feat": 0.85,
                "rsi_14_1h": 0.8,
                "ema_9_15m": 0.6,
                "sma_20_1h": 0.5,
                "atr_14_15m": 0.3,
                "close_1h": 0.1,
            }
        )
        features = sorted(ranking.index)
        result = apply_timeframe_budget(ranking, features, ["15min", "60min"], 2)
        assert result == [
            "base_feat",
            "ema_9_15m",
            "rsi_14_15m",
            "rsi_14_1h",
            "sma_20_1h",
        ]  # candidate order kept; atr_14_15m and close_1h over budget

    def test_base_and_unbudgeted_timeframes_untouched(self) -> None:
        """Only the listed timeframes are budgeted; base features always stay."""
        features = ["a", "b", "x_15m", "y_15m", "z_15m", "u_4h", "v_4h"]
        ranking = pd.Series({f: 1.0 / (i + 1) for i, f in enumerate(features)})
        result = apply_timeframe_budget(ranking, features, ["15min"], 1)
        assert result == ["a", "b", "x_15m", "u_4h", "v_4h"]

    def test_legacy_minute_suffixes_are_not_mtf_columns(self) -> None:
        """The generator writes ``_15m`` / ``_1h``; ``_15min`` names are base features."""
        features = ["rsi_14_15min", "macd_15min", "rsi_14_15m", "macd_15m"]
        ranking = pd.Series({f: 1.0 / (i + 1) for i, f in enumerate(features)})
        result = apply_timeframe_budget(ranking, features, ["15min"], 1)
        assert result == ["rsi_14_15min", "macd_15min", "rsi_14_15m"]

    def test_suffixes_match_the_mtf_generator(self) -> None:
        from src.data.pipeline.stages.mtf.generator import MTFFeatureGenerator

        gen = MTFFeatureGenerator(base_timeframe="5min", mtf_timeframes=["15min", "60min"])
        features = [f"close{gen._get_tf_suffix(tf)}" for tf in ("15min", "60min")]
        features += [f"rsi_14{gen._get_tf_suffix(tf)}" for tf in ("15min", "60min")]
        ranking = pd.Series({f: 1.0 / (i + 1) for i, f in enumerate(features)})
        result = apply_timeframe_budget(ranking, features, ["15min", "60min"], 1)
        assert result == ["close_15m", "close_1h"]

    def test_passthrough_without_timeframes_or_ranking(self) -> None:
        features = ["a_15m", "b_15m"]
        ranking = pd.Series({"a_15m": 1.0, "b_15m": 0.5})
        assert apply_timeframe_budget(ranking, features, [], 1) == features
        assert apply_timeframe_budget(pd.Series(dtype=float), features, ["15min"], 1) == features

    def test_unranked_features_are_kept(self) -> None:
        features = ["a_15m", "b_15m", "c_15m"]
        ranking = pd.Series({"a_15m": 1.0, "b_15m": 0.5})
        assert apply_timeframe_budget(ranking, features, ["15min"], 1) == ["a_15m", "c_15m"]

    def test_budget_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_per_timeframe must be >= 1"):
            apply_timeframe_budget(pd.Series({"a_15m": 1.0}), ["a_15m"], ["15min"], 0)
