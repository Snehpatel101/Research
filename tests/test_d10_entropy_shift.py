"""
D10: Verify entropy features use shift(1) — no lookahead.

Every entropy feature computes on df["close"].shift(1), so the feature at bar N
only sees close prices up to bar N-1. A spike at bar N must not affect bar N's
feature value.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.features.compute.entropy import (
    compute_entropy_shannon_10,
    compute_entropy_shannon_20,
    compute_hurst_50,
)


def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create synthetic OHLCV data."""
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.randn(n) * 0.3
    volume = rng.randint(100, 10000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


class TestEntropyShiftNoLookahead:
    """Entropy features at bar N must be immune to changes at bar N's close."""

    def test_shannon_10_last_bar_immune_to_spike(self):
        """compute_entropy_shannon_10: spike at last bar does not change last bar's value."""
        df_orig = _make_ohlcv()
        df_mod = df_orig.copy()
        df_mod.iloc[-1, df_mod.columns.get_loc("close")] *= 2.0

        feat_orig = compute_entropy_shannon_10(df_orig)
        feat_mod = compute_entropy_shannon_10(df_mod)

        # Last bar should be identical (shift(1) excludes it)
        assert feat_orig.iloc[-1] == pytest.approx(
            feat_mod.iloc[-1]
        ), "Shannon-10 at last bar changed — lookahead detected"

        # Second-to-last bar sanity check (neither dataset changed there)
        assert feat_orig.iloc[-2] == pytest.approx(
            feat_mod.iloc[-2]
        ), "Shannon-10 at second-to-last bar should be identical"

    def test_shannon_20_last_bar_immune_to_spike(self):
        """compute_entropy_shannon_20: spike at last bar does not change last bar's value."""
        df_orig = _make_ohlcv()
        df_mod = df_orig.copy()
        df_mod.iloc[-1, df_mod.columns.get_loc("close")] *= 2.0

        feat_orig = compute_entropy_shannon_20(df_orig)
        feat_mod = compute_entropy_shannon_20(df_mod)

        assert feat_orig.iloc[-1] == pytest.approx(
            feat_mod.iloc[-1]
        ), "Shannon-20 at last bar changed — lookahead detected"

        assert feat_orig.iloc[-2] == pytest.approx(
            feat_mod.iloc[-2]
        ), "Shannon-20 at second-to-last bar should be identical"

    def test_hurst_50_last_bar_immune_to_spike(self):
        """compute_hurst_50: spike at last bar does not change last bar's value."""
        df_orig = _make_ohlcv()
        df_mod = df_orig.copy()
        df_mod.iloc[-1, df_mod.columns.get_loc("close")] *= 2.0

        feat_orig = compute_hurst_50(df_orig)
        feat_mod = compute_hurst_50(df_mod)

        assert feat_orig.iloc[-1] == pytest.approx(
            feat_mod.iloc[-1]
        ), "Hurst-50 at last bar changed — lookahead detected"

        assert feat_orig.iloc[-2] == pytest.approx(
            feat_mod.iloc[-2]
        ), "Hurst-50 at second-to-last bar should be identical"

    def test_spike_does_affect_next_bar(self):
        """Sanity: the spike SHOULD affect the bar AFTER it (if one existed)."""
        df_orig = _make_ohlcv()
        df_mod = df_orig.copy()
        # Spike at bar -2 so bar -1 can see it via shift(1)
        df_mod.iloc[-2, df_mod.columns.get_loc("close")] *= 2.0

        feat_orig = compute_entropy_shannon_10(df_orig)
        feat_mod = compute_entropy_shannon_10(df_mod)

        # Bar -1 should differ because shift(1) lets it see bar -2's spike
        assert feat_orig.iloc[-1] != pytest.approx(
            feat_mod.iloc[-1]
        ), "Shannon-10 should change at bar after spike (shift(1) includes it)"
