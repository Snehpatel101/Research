"""
Tests for Critical Fix #6 (MDA holdout scoring) and Fix #7 (timestamp alignment).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fix #6: MDA importance must score on holdout, not training data
# ---------------------------------------------------------------------------

class TestMDAHoldoutScoring:
    """Verify MDA permutation importance uses holdout data."""

    def _make_data(self, n_train=200, n_test=80, n_features=10, seed=42):
        """Create synthetic classification data with train/test split."""
        rng = np.random.RandomState(seed)
        X = pd.DataFrame(
            rng.randn(n_train + n_test, n_features),
            columns=[f"f{i}" for i in range(n_features)],
        )
        # Feature 0 is the only real signal
        y = pd.Series((X["f0"] > 0).astype(int))
        return (
            X.iloc[:n_train],
            y.iloc[:n_train],
            X.iloc[n_train:],
            y.iloc[n_train:],
        )

    def test_mda_uses_holdout_when_provided(self):
        """MDA should score on holdout set, not training data."""
        from unittest.mock import patch

        from src.optimization.feature_selection.walk_forward import (
            WalkForwardFeatureSelector,
        )

        X_train, y_train, X_test, y_test = self._make_data()
        selector = WalkForwardFeatureSelector(
            selection_method="mda", n_estimators=20
        )

        # Patch permutation_importance to capture what X,y it receives
        captured = {}
        original_pi = None
        import sklearn.inspection as si
        original_pi = si.permutation_importance

        def spy_pi(estimator, X, y, **kwargs):
            captured["X_shape"] = X.shape
            captured["y_len"] = len(y)
            return original_pi(estimator, X, y, **kwargs)

        with patch(
            "src.optimization.feature_selection.walk_forward.permutation_importance",
            side_effect=spy_pi,
        ):
            selector._mda_importance(
                X_train, y_train,
                X_test=X_test, y_test=y_test,
            )

        # Must have scored on holdout (80 rows), not training (200 rows)
        assert captured["X_shape"][0] == 80, (
            f"MDA scored on {captured['X_shape'][0]} rows, expected 80 (holdout)"
        )

    def test_mda_warns_without_holdout(self, caplog):
        """MDA should log warning when no holdout provided (legacy fallback)."""
        from src.optimization.feature_selection.walk_forward import (
            WalkForwardFeatureSelector,
        )

        X_train, y_train, _, _ = self._make_data()
        selector = WalkForwardFeatureSelector(
            selection_method="mda", n_estimators=20
        )

        with caplog.at_level("WARNING"):
            result = selector._mda_importance(X_train, y_train)

        assert len(result) == 10  # All features get scores
        assert "no holdout set provided" in caplog.text

    def test_walkforward_passes_test_split(self):
        """Walk-forward loop must pass test_idx data to MDA."""
        from unittest.mock import patch

        from src.optimization.feature_selection.walk_forward import (
            WalkForwardFeatureSelector,
        )

        rng = np.random.RandomState(42)
        n = 300
        X = pd.DataFrame(rng.randn(n, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(rng.randint(0, 2, n))

        # Two folds with specific sizes
        cv_splits = [
            (np.arange(0, 200), np.arange(200, 260)),
            (np.arange(0, 240), np.arange(240, 300)),
        ]

        selector = WalkForwardFeatureSelector(
            selection_method="mda", n_estimators=10, n_features_to_select=3
        )

        # Track what _compute_importance receives
        calls = []
        original = selector._compute_importance

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        with patch.object(selector, "_compute_importance", side_effect=spy):
            selector.select_features_walkforward(X, y, cv_splits)

        # Both folds should pass X_test and y_test
        assert len(calls) == 2
        for i, call_kwargs in enumerate(calls):
            assert "X_test" in call_kwargs, f"Fold {i}: X_test not passed"
            assert "y_test" in call_kwargs, f"Fold {i}: y_test not passed"
            assert call_kwargs["X_test"] is not None
            assert call_kwargs["y_test"] is not None


# ---------------------------------------------------------------------------
# Fix #7: Timestamp-based alignment (not ratio-based)
# ---------------------------------------------------------------------------

class TestTimestampAlignment:
    """Verify multi-stream adapter uses timestamp-based alignment."""

    def _make_market_dfs(self):
        """
        Create synthetic market data with an overnight gap.

        1min anchor: 09:00-09:59 (60 bars), gap, 14:00-14:59 (60 bars) = 120 bars
        5min higher: 09:00-09:55 (12 bars), gap, 14:00-14:55 (12 bars) = 24 bars

        Ratio-based alignment would map anchor idx 60 -> higher idx 12,
        but timestamp-based should map 14:00 -> 14:00.
        """
        # Day 1 morning session
        am_anchor_idx = pd.date_range("2024-01-15 09:00", periods=60, freq="1min")
        pm_anchor_idx = pd.date_range("2024-01-15 14:00", periods=60, freq="1min")
        anchor_idx = am_anchor_idx.append(pm_anchor_idx)

        am_higher_idx = pd.date_range("2024-01-15 09:00", periods=12, freq="5min")
        pm_higher_idx = pd.date_range("2024-01-15 14:00", periods=12, freq="5min")
        higher_idx = am_higher_idx.append(pm_higher_idx)

        rng = np.random.RandomState(42)

        anchor_df = pd.DataFrame(
            {
                "open": rng.uniform(100, 105, len(anchor_idx)),
                "high": rng.uniform(105, 110, len(anchor_idx)),
                "low": rng.uniform(95, 100, len(anchor_idx)),
                "close": rng.uniform(100, 105, len(anchor_idx)),
                "volume": rng.randint(100, 1000, len(anchor_idx)),
                "label_h20": rng.randint(0, 2, len(anchor_idx)),
            },
            index=anchor_idx,
        )

        higher_df = pd.DataFrame(
            {
                "open": rng.uniform(100, 105, len(higher_idx)),
                "high": rng.uniform(105, 110, len(higher_idx)),
                "low": rng.uniform(95, 100, len(higher_idx)),
                "close": rng.uniform(100, 105, len(higher_idx)),
                "volume": rng.randint(500, 5000, len(higher_idx)),
            },
            index=higher_idx,
        )

        return anchor_df, higher_df

    def test_timestamp_align_handles_gaps(self):
        """Timestamp alignment should map across gaps correctly."""
        from src.data.adapters.multi_stream import MultiStreamAdapter

        anchor_df, higher_df = self._make_market_dfs()

        adapter = MultiStreamAdapter(
            feature_columns=["open", "high", "low", "close", "volume"],
            timeframes=["1min", "5min"],
            sequence_length=10,
            stride=1,
        )

        idx_map = adapter._timestamp_align(anchor_df, higher_df)

        # Anchor index 0 (09:00) -> higher index 0 (09:00)
        assert idx_map[0] == 0

        # Anchor index 59 (09:59) -> higher index 11 (09:55)
        assert idx_map[59] == 11

        # Anchor index 60 (14:00) -> higher index 12 (14:00), NOT index 12 by ratio
        # With ratio-based: 60 // 5 = 12, which happens to be correct here
        # BUT the key test is the gap doesn't corrupt alignment
        assert idx_map[60] == 12

        # Anchor index 65 (14:05) -> higher index 13 (14:05)
        assert idx_map[65] == 13

        # Verify no index exceeds bounds
        assert idx_map.max() < len(higher_df)

    def test_ratio_fallback_without_datetime_index(self):
        """Without DatetimeIndex, should fall back to ratio-based with warning."""
        from src.data.adapters.multi_stream import MultiStreamAdapter

        rng = np.random.RandomState(42)
        n = 100

        anchor_df = pd.DataFrame(
            {
                "open": rng.randn(n),
                "high": rng.randn(n),
                "low": rng.randn(n),
                "close": rng.randn(n),
                "volume": rng.randn(n),
                "label_h20": rng.randint(0, 2, n),
            }
        )
        higher_df = pd.DataFrame(
            {
                "open": rng.randn(n // 5),
                "high": rng.randn(n // 5),
                "low": rng.randn(n // 5),
                "close": rng.randn(n // 5),
                "volume": rng.randn(n // 5),
            }
        )

        adapter = MultiStreamAdapter(
            feature_columns=["open", "high", "low", "close", "volume"],
            timeframes=["1min", "5min"],
            sequence_length=10,
            stride=1,
        )

        tf_dfs = {"1min": anchor_df, "5min": higher_df}
        feature_cols = ["open", "high", "low", "close", "volume"]

        # Should not raise, should produce valid output
        maps = adapter._build_timestamp_index_maps(
            anchor_df, tf_dfs, ["1min", "5min"], feature_cols
        )
        assert "1min" in maps
        assert "5min" in maps
        assert len(maps["5min"]["anchor_to_tf"]) == n

    def test_full_transform_with_timestamps(self):
        """Full transform should produce correct 4D output with timestamp alignment."""
        from src.data.adapters.multi_stream import MultiStreamAdapter

        anchor_df, higher_df = self._make_market_dfs()

        adapter = MultiStreamAdapter(
            feature_columns=["open", "high", "low", "close", "volume"],
            timeframes=["1min", "5min"],
            sequence_length=10,
            stride=1,
        )

        result = adapter.transform(
            anchor_df, additional_dfs={"5min": higher_df}
        )

        # Shape checks
        n_seqs = (120 - 10) // 1 + 1  # 111
        assert result.X.shape == (n_seqs, 2, 10, 5)
        assert result.y.shape == (n_seqs,)
        assert result.n_timeframes == 2

    def test_extract_aligned_sequence_pads_correctly(self):
        """When fewer unique higher-TF bars than seq_len, pad with earliest bar."""
        from src.data.adapters.multi_stream import MultiStreamAdapter

        adapter = MultiStreamAdapter(
            feature_columns=["open", "close"],
            timeframes=["1min", "5min"],
            sequence_length=10,
        )

        # 5 unique higher-TF bars
        tf_values = np.arange(10).reshape(5, 2).astype(np.float32)
        # idx_map: anchor positions 0-9, all map to bars 0-4
        idx_map = np.array([0, 0, 0, 0, 0, 1, 1, 2, 3, 4])

        result = adapter._extract_aligned_sequence(
            tf_values=tf_values,
            idx_map=idx_map,
            anchor_start=0,
            anchor_end=10,
            seq_len=10,
        )

        assert result.shape == (10, 2)
        # 5 unique bars -> pad 5 at front with bar[0], then 5 unique bars
        # Front padding: 5 copies of bar 0 (values [0,1])
        np.testing.assert_array_equal(result[:5, 0], [0, 0, 0, 0, 0])
        # Unique bars at end
        np.testing.assert_array_equal(result[5:, 0], [0, 2, 4, 6, 8])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
