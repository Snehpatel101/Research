"""Sequence/timestamp alignment: gaps, ratio fallback, padding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestTimestampAlignment:
    """Verify multi-stream adapter uses timestamp-based alignment."""

    def _make_market_dfs(self):
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
        from src.data.adapters.multi_stream import MultiStreamAdapter

        anchor_df, higher_df = self._make_market_dfs()
        adapter = MultiStreamAdapter(
            feature_columns=["open", "high", "low", "close", "volume"],
            timeframes=["1min", "5min"],
            sequence_length=10,
            stride=1,
        )
        idx_map = adapter._timestamp_align(anchor_df, higher_df)

        assert idx_map[0] == 0
        assert idx_map[59] == 11
        assert idx_map[60] == 12
        assert idx_map[65] == 13
        assert idx_map.max() < len(higher_df)

    def test_ratio_fallback_without_datetime_index(self):
        """Non-DatetimeIndex DataFrames should raise ValueError (Phase 60 change)."""
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
        with pytest.raises(ValueError, match="requires DatetimeIndex"):
            adapter._build_timestamp_index_maps(anchor_df, tf_dfs, ["1min", "5min"], feature_cols)

    def test_full_transform_with_timestamps(self):
        from src.data.adapters.multi_stream import MultiStreamAdapter

        anchor_df, higher_df = self._make_market_dfs()
        adapter = MultiStreamAdapter(
            feature_columns=["open", "high", "low", "close", "volume"],
            timeframes=["1min", "5min"],
            sequence_length=10,
            stride=1,
        )
        result = adapter.transform(anchor_df, additional_dfs={"5min": higher_df})

        n_seqs = (120 - 10) // 1 + 1
        assert result.X.shape == (n_seqs, 2, 10, 5)
        assert result.y.shape == (n_seqs,)
        assert result.n_timeframes == 2

    def test_extract_aligned_sequence_pads_correctly(self):
        from src.data.adapters.multi_stream import MultiStreamAdapter

        adapter = MultiStreamAdapter(
            feature_columns=["open", "close"],
            timeframes=["1min", "5min"],
            sequence_length=10,
        )

        tf_values = np.arange(10).reshape(5, 2).astype(np.float32)
        idx_map = np.array([0, 0, 0, 0, 0, 1, 1, 2, 3, 4])

        result = adapter._extract_aligned_sequence(
            tf_values=tf_values,
            idx_map=idx_map,
            anchor_start=0,
            anchor_end=10,
            seq_len=10,
        )

        assert result.shape == (10, 2)
        np.testing.assert_array_equal(result[:5, 0], [0, 0, 0, 0, 0])
        np.testing.assert_array_equal(result[5:, 0], [0, 2, 4, 6, 8])
