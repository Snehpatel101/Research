"""
Regime features: anti-lookahead (shift(1)), the unified label entry point, and
min-bars hysteresis.

A spike injected at bar N must never change any regime value at or before bar N;
it may only show up from bar N+1 onward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.pipeline.stages.features.regime import (
    add_structure_regime,
    add_trend_regime,
    add_volatility_regime,
)
from src.data.pipeline.stages.features.volatility import add_historical_volatility
from src.data.pipeline.stages.regime import (
    MarketStructureDetector,
    TrendRegimeDetector,
    VolatilityRegimeDetector,
)
from src.data.pipeline.stages.regime.composite import CompositeRegimeDetector
from src.data.pipeline.stages.regime.unified import get_regime_labels


def _make_regime_ohlcv(n: int, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV data for regime shift tests."""
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    high = close + np.abs(rng.randn(n) * 0.3)
    low = close - np.abs(rng.randn(n) * 0.3)
    volume = rng.randint(100, 1000, size=n).astype(float)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": volume})


class TestRegimeFeaturesAreLagged:
    def test_volatility_regime_is_lagged_one_bar(self) -> None:
        """A close spike at bar N must not affect bar N's volatility regime.

        The live add_volatility_regime consumes hvol_20, which the live
        add_historical_volatility produces already lagged by 1 bar.
        """

        def pipeline(df: pd.DataFrame) -> pd.DataFrame:
            out = add_historical_volatility(df.copy(), {}, periods=[20], timeframe="5min")
            return add_volatility_regime(out, {})

        df = _make_regime_ohlcv(260)
        spike_bar = 200
        base = pipeline(df)

        df_spiked = df.copy()
        df_spiked.loc[spike_bar, "close"] *= 2.0  # drives hvol through the roof
        spiked = pipeline(df_spiked)

        pd.testing.assert_series_equal(
            base["volatility_regime"].iloc[: spike_bar + 1],
            spiked["volatility_regime"].iloc[: spike_bar + 1],
        )
        # Sanity: the spike DOES propagate into the lagged hvol at bar N+1
        assert base["hvol_20"].iloc[spike_bar + 1] != pytest.approx(
            spiked["hvol_20"].iloc[spike_bar + 1]
        )

    def test_trend_regime_is_lagged_one_bar(self) -> None:
        """A crash at bar N is invisible to bar N's trend regime and seen at N+1."""

        def pipeline(close: pd.Series) -> pd.Series:
            df = pd.DataFrame({"close": close})
            # SMA inputs are pre-lagged, matching the live engine contract
            df["sma_50"] = close.rolling(50).mean().shift(1)
            df["sma_200"] = close.rolling(200).mean().shift(1)
            return add_trend_regime(df, {})["trend_regime"]

        n, spike_bar = 260, 250
        close = pd.Series(100.0 + 0.1 * np.arange(n))  # steady uptrend
        base = pipeline(close)
        assert base.iloc[spike_bar] == 1
        assert base.iloc[spike_bar + 1] == 1

        crashed = close.copy()
        crashed.iloc[spike_bar] *= 0.5
        spiked = pipeline(crashed)

        assert spiked.iloc[spike_bar] == base.iloc[spike_bar], "bar N saw its own crash"
        assert spiked.iloc[spike_bar + 1] == 0, "bar N+1 should leave the uptrend"

    def test_structure_regime_is_the_raw_hurst_regime_lagged_one_bar(self) -> None:
        """Exact: bar N's structure regime is the raw detector's value at bar N-1.

        (A spike test is not enough here: the Hurst category rarely flips at the spiked
        bar, so a missing lag would go unnoticed.)
        """
        df = _make_regime_ohlcv(200)
        raw = MarketStructureDetector(lookback=100).detect(df)
        assert not raw.iloc[1:].equals(raw.shift(1).iloc[1:]), "raw regime is constant: no signal"

        out = add_structure_regime(df.copy(), {}, lookback=100)["structure_regime"]

        assert pd.isna(out.iloc[0])
        pd.testing.assert_series_equal(out, raw.shift(1), check_names=False)


class TestUnifiedRegimeLabels:
    def test_composite_regime_columns_are_raw_detector_output_lagged_one_bar(self) -> None:
        """With hysteresis off, each column is exactly its detector's raw series shifted by 1."""
        df = _make_regime_ohlcv(400)
        detectors = {
            "volatility_regime": VolatilityRegimeDetector(),
            "trend_regime": TrendRegimeDetector(),
            "structure_regime": MarketStructureDetector(),
        }
        composite = CompositeRegimeDetector(
            volatility_detector=detectors["volatility_regime"],
            trend_detector=detectors["trend_regime"],
            structure_detector=detectors["structure_regime"],
            min_regime_bars=1,
        )

        regimes = composite.detect_all(df).regimes

        for column, detector in detectors.items():
            raw = detector.detect(df)
            assert not raw.iloc[1:].equals(raw.shift(1).iloc[1:]), f"{column}: raw is constant"
            pd.testing.assert_series_equal(regimes[column], raw.shift(1), check_names=False)

    def test_labels_align_with_input(self) -> None:
        df = _make_regime_ohlcv(400)
        labels = get_regime_labels(df)

        assert isinstance(labels, pd.Series)
        assert labels.index.equals(df.index)
        assert pd.isna(labels.iloc[0]), "first bar is NaN because every regime is shifted"
        assert labels.iloc[100:].notna().all()


class TestRegimeHysteresis:
    RAW = pd.Series([0, 0, 0, 1, 0, 0, 1, 1, 1, 0])

    def test_short_flip_is_ignored_and_persistent_flip_is_accepted_late(self) -> None:
        """A 1-bar blip never switches the regime; a run of min_regime_bars does."""
        smoothed = CompositeRegimeDetector(min_regime_bars=3)._apply_hysteresis(self.RAW)

        # blip at bar 3 is suppressed; the 1,1,1 run confirms only on its 3rd bar
        assert smoothed.tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]

    def test_single_bar_threshold_disables_smoothing(self) -> None:
        detector = CompositeRegimeDetector(min_regime_bars=1)
        assert detector._apply_hysteresis(self.RAW).tolist() == self.RAW.tolist()

    def test_non_positive_threshold_is_clamped_to_one(self) -> None:
        assert CompositeRegimeDetector(min_regime_bars=0).min_regime_bars == 1

    def test_nan_bars_pass_through(self) -> None:
        raw = pd.Series([0.0, np.nan, 0.0, 1.0, 1.0])
        out = CompositeRegimeDetector(min_regime_bars=2)._apply_hysteresis(raw)
        assert np.isnan(out.iloc[1])
        assert out.tolist()[-2:] == [0.0, 1.0]
