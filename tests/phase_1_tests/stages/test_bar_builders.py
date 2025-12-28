"""
Tests for Alternative Bar Builders.

Tests volume bars, dollar bars, and time bars.
Uses deterministic synthetic data for reproducibility.
"""
import numpy as np
import pandas as pd
import pytest


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_ohlcv_1min():
    """Generate 1-minute OHLCV data."""
    np.random.seed(42)
    n_bars = 1000

    # Generate realistic price series
    returns = np.random.randn(n_bars) * 0.001
    close = 100 * np.exp(np.cumsum(returns))

    open_prices = np.roll(close, 1)
    open_prices[0] = 100

    # High and low must respect OHLC constraints
    high = np.maximum(open_prices, close) * (1 + np.abs(np.random.randn(n_bars) * 0.001))
    low = np.minimum(open_prices, close) * (1 - np.abs(np.random.randn(n_bars) * 0.001))

    # Variable volume
    volume = np.random.randint(100, 1000, n_bars).astype(float)
    # Add volume spikes
    volume[200:250] *= 5
    volume[500:550] *= 3

    df = pd.DataFrame({
        "datetime": pd.date_range("2023-01-01 09:30", periods=n_bars, freq="1min"),
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })

    return df


@pytest.fixture
def minimal_ohlcv():
    """Minimal OHLCV data for edge case testing."""
    return pd.DataFrame({
        "datetime": pd.date_range("2023-01-01", periods=10, freq="1min"),
        "open": [100] * 10,
        "high": [101] * 10,
        "low": [99] * 10,
        "close": [100] * 10,
        "volume": [100] * 10,
    })


# =============================================================================
# BAR BUILDER REGISTRY TESTS
# =============================================================================

class TestBarBuilderRegistry:
    """Test bar builder registry functionality."""

    def test_list_all(self):
        """Test listing all registered builders."""
        from src.phase1.stages.clean.bar_builders import BarBuilderRegistry

        builders = BarBuilderRegistry.list_all()

        assert "time" in builders
        assert "volume" in builders
        assert "dollar" in builders

    def test_get_registered_builder(self):
        """Test getting a registered builder."""
        from src.phase1.stages.clean.bar_builders import BarBuilderRegistry, VolumeBarBuilder

        builder_cls = BarBuilderRegistry.get("volume")
        assert builder_cls == VolumeBarBuilder

    def test_get_unregistered_raises(self):
        """Test getting unregistered builder raises error."""
        from src.phase1.stages.clean.bar_builders import BarBuilderRegistry

        with pytest.raises(ValueError, match="Unknown bar type"):
            BarBuilderRegistry.get("unknown_type")

    def test_is_registered(self):
        """Test checking if builder is registered."""
        from src.phase1.stages.clean.bar_builders import BarBuilderRegistry

        assert BarBuilderRegistry.is_registered("volume")
        assert BarBuilderRegistry.is_registered("dollar")
        assert not BarBuilderRegistry.is_registered("unknown")


# =============================================================================
# VOLUME BAR BUILDER TESTS
# =============================================================================

class TestVolumeBarBuilder:
    """Test volume bar builder."""

    def test_builder_creation(self):
        """Test builder can be created."""
        from src.phase1.stages.clean.bar_builders import VolumeBarBuilder

        builder = VolumeBarBuilder(volume_threshold=10000)
        assert builder.volume_threshold == 10000
        assert builder.bar_type == "volume"

    def test_invalid_threshold_raises(self):
        """Test invalid threshold raises error."""
        from src.phase1.stages.clean.bar_builders import VolumeBarBuilder

        with pytest.raises(ValueError, match="must be > 0"):
            VolumeBarBuilder(volume_threshold=0)

        with pytest.raises(ValueError, match="must be > 0"):
            VolumeBarBuilder(volume_threshold=-100)

    def test_build_volume_bars(self, sample_ohlcv_1min):
        """Test building volume bars."""
        from src.phase1.stages.clean.bar_builders import VolumeBarBuilder

        builder = VolumeBarBuilder(volume_threshold=5000)
        bars = builder.build(sample_ohlcv_1min)

        # Should have fewer bars than input
        assert len(bars) < len(sample_ohlcv_1min)
        assert len(bars) > 0

        # Check columns
        assert "datetime" in bars.columns
        assert "open" in bars.columns
        assert "high" in bars.columns
        assert "low" in bars.columns
        assert "close" in bars.columns
        assert "volume" in bars.columns
        assert "bar_type" in bars.columns

    def test_ohlcv_aggregation_correct(self, sample_ohlcv_1min):
        """Test OHLCV is aggregated correctly."""
        from src.phase1.stages.clean.bar_builders import VolumeBarBuilder

        builder = VolumeBarBuilder(volume_threshold=5000)
        bars = builder.build(sample_ohlcv_1min)

        # Each bar's high should be >= open, close
        assert np.all(bars["high"] >= bars["open"])
        assert np.all(bars["high"] >= bars["close"])

        # Each bar's low should be <= open, close
        assert np.all(bars["low"] <= bars["open"])
        assert np.all(bars["low"] <= bars["close"])

        # Volume should be positive
        assert np.all(bars["volume"] > 0)

    def test_volume_bars_compression(self, sample_ohlcv_1min):
        """Test volume bars compress during high volume periods."""
        from src.phase1.stages.clean.bar_builders import VolumeBarBuilder

        builder = VolumeBarBuilder(volume_threshold=5000)
        bars = builder.build(sample_ohlcv_1min)

        compression = len(sample_ohlcv_1min) / len(bars)
        assert compression > 1, "Should compress input bars"

    def test_symbol_column(self, sample_ohlcv_1min):
        """Test symbol column is added."""
        from src.phase1.stages.clean.bar_builders import VolumeBarBuilder

        builder = VolumeBarBuilder(volume_threshold=5000)
        bars = builder.build(sample_ohlcv_1min, symbol="MES")

        assert "symbol" in bars.columns
        assert bars["symbol"].iloc[0] == "MES"

    def test_missing_columns_raises(self):
        """Test missing columns raises error."""
        from src.phase1.stages.clean.bar_builders import VolumeBarBuilder

        builder = VolumeBarBuilder(volume_threshold=5000)
        bad_df = pd.DataFrame({"datetime": [1, 2], "close": [100, 101]})

        with pytest.raises(ValueError, match="Missing required columns"):
            builder.build(bad_df)


# =============================================================================
# DOLLAR BAR BUILDER TESTS
# =============================================================================

class TestDollarBarBuilder:
    """Test dollar bar builder."""

    def test_builder_creation(self):
        """Test builder can be created."""
        from src.phase1.stages.clean.bar_builders import DollarBarBuilder

        builder = DollarBarBuilder(dollar_threshold=1000000)
        assert builder.dollar_threshold == 1000000
        assert builder.bar_type == "dollar"

    def test_invalid_threshold_raises(self):
        """Test invalid threshold raises error."""
        from src.phase1.stages.clean.bar_builders import DollarBarBuilder

        with pytest.raises(ValueError, match="must be > 0"):
            DollarBarBuilder(dollar_threshold=0)

    def test_build_dollar_bars(self, sample_ohlcv_1min):
        """Test building dollar bars."""
        from src.phase1.stages.clean.bar_builders import DollarBarBuilder

        builder = DollarBarBuilder(dollar_threshold=500000)
        bars = builder.build(sample_ohlcv_1min)

        assert len(bars) < len(sample_ohlcv_1min)
        assert len(bars) > 0

        # Check dollar_value column
        assert "dollar_value" in bars.columns
        assert np.all(bars["dollar_value"] > 0)

    def test_dollar_bars_use_vwap(self, sample_ohlcv_1min):
        """Test dollar bars with VWAP option."""
        from src.phase1.stages.clean.bar_builders import DollarBarBuilder

        builder_close = DollarBarBuilder(dollar_threshold=500000, use_vwap=False)
        builder_vwap = DollarBarBuilder(dollar_threshold=500000, use_vwap=True)

        bars_close = builder_close.build(sample_ohlcv_1min)
        bars_vwap = builder_vwap.build(sample_ohlcv_1min)

        # Different calculation methods may produce different bar counts
        # Just verify both work
        assert len(bars_close) > 0
        assert len(bars_vwap) > 0


# =============================================================================
# TIME BAR BUILDER TESTS
# =============================================================================

class TestTimeBarBuilder:
    """Test time bar builder."""

    def test_builder_creation(self):
        """Test builder can be created."""
        from src.phase1.stages.clean.bar_builders import TimeBarBuilder

        builder = TimeBarBuilder(target_timeframe="5min")
        assert builder.target_timeframe == "5min"
        assert builder.bar_type == "time"

    def test_invalid_timeframe_raises(self):
        """Test invalid timeframe raises error."""
        from src.phase1.stages.clean.bar_builders import TimeBarBuilder

        with pytest.raises(ValueError, match="Unsupported timeframe"):
            TimeBarBuilder(target_timeframe="3min")

    def test_build_5min_bars(self, sample_ohlcv_1min):
        """Test building 5-minute bars."""
        from src.phase1.stages.clean.bar_builders import TimeBarBuilder

        builder = TimeBarBuilder(target_timeframe="5min")
        bars = builder.build(sample_ohlcv_1min)

        # Should compress ~5x
        expected_bars = len(sample_ohlcv_1min) // 5
        assert len(bars) == pytest.approx(expected_bars, abs=5)

        assert "timeframe" in bars.columns
        assert bars["timeframe"].iloc[0] == "5min"

    def test_build_15min_bars(self, sample_ohlcv_1min):
        """Test building 15-minute bars."""
        from src.phase1.stages.clean.bar_builders import TimeBarBuilder

        builder = TimeBarBuilder(target_timeframe="15min")
        bars = builder.build(sample_ohlcv_1min)

        # Should compress ~15x
        expected_bars = len(sample_ohlcv_1min) // 15
        assert len(bars) == pytest.approx(expected_bars, abs=5)

    def test_supported_timeframes(self):
        """Test all supported timeframes."""
        from src.phase1.stages.clean.bar_builders import TimeBarBuilder

        timeframes = ["1min", "5min", "10min", "15min", "20min", "30min", "45min", "60min"]

        for tf in timeframes:
            builder = TimeBarBuilder(target_timeframe=tf)
            assert builder.target_timeframe == tf


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestBuildBarsFactory:
    """Test the build_bars factory function."""

    def test_build_volume_bars(self, sample_ohlcv_1min):
        """Test building volume bars via factory."""
        from src.phase1.stages.clean.bar_builders import build_bars

        bars = build_bars(sample_ohlcv_1min, bar_type="volume", volume_threshold=5000)

        assert len(bars) > 0
        assert "bar_type" in bars.columns
        assert bars["bar_type"].iloc[0] == "volume"

    def test_build_dollar_bars(self, sample_ohlcv_1min):
        """Test building dollar bars via factory."""
        from src.phase1.stages.clean.bar_builders import build_bars

        bars = build_bars(sample_ohlcv_1min, bar_type="dollar", dollar_threshold=500000)

        assert len(bars) > 0
        assert "bar_type" in bars.columns
        assert bars["bar_type"].iloc[0] == "dollar"

    def test_build_time_bars(self, sample_ohlcv_1min):
        """Test building time bars via factory."""
        from src.phase1.stages.clean.bar_builders import build_bars

        bars = build_bars(sample_ohlcv_1min, bar_type="time", target_timeframe="5min")

        assert len(bars) > 0
        assert "bar_type" in bars.columns
        assert bars["bar_type"].iloc[0] == "time"

    def test_build_from_config(self, sample_ohlcv_1min):
        """Test building bars from BarConfig."""
        from src.phase1.stages.clean.bar_builders import build_bars, BarConfig

        config = BarConfig(bar_type="volume", volume_threshold=5000)
        bars = build_bars(sample_ohlcv_1min, config=config)

        assert len(bars) > 0
        assert bars["threshold"].iloc[0] == 5000

    def test_unknown_bar_type_raises(self, sample_ohlcv_1min):
        """Test unknown bar type raises error."""
        from src.phase1.stages.clean.bar_builders import build_bars

        with pytest.raises(ValueError, match="Unknown bar type"):
            build_bars(sample_ohlcv_1min, bar_type="unknown")


# =============================================================================
# BAR CONFIG TESTS
# =============================================================================

class TestBarConfig:
    """Test BarConfig dataclass."""

    def test_default_config(self):
        """Test default config values."""
        from src.phase1.stages.clean.bar_builders import BarConfig

        config = BarConfig()
        assert config.bar_type == "time"
        assert config.target_timeframe == "5min"
        assert config.volume_threshold == 100_000

    def test_get_builder_kwargs_time(self):
        """Test getting builder kwargs for time bars."""
        from src.phase1.stages.clean.bar_builders import BarConfig

        config = BarConfig(bar_type="time", target_timeframe="15min")
        kwargs = config.get_builder_kwargs()

        assert kwargs["target_timeframe"] == "15min"

    def test_get_builder_kwargs_volume(self):
        """Test getting builder kwargs for volume bars."""
        from src.phase1.stages.clean.bar_builders import BarConfig

        config = BarConfig(bar_type="volume", volume_threshold=50000)
        kwargs = config.get_builder_kwargs()

        assert kwargs["volume_threshold"] == 50000

    def test_get_builder_kwargs_dollar(self):
        """Test getting builder kwargs for dollar bars."""
        from src.phase1.stages.clean.bar_builders import BarConfig

        config = BarConfig(bar_type="dollar", dollar_threshold=1000000, use_vwap=True)
        kwargs = config.get_builder_kwargs()

        assert kwargs["dollar_threshold"] == 1000000
        assert kwargs["use_vwap"] is True


# =============================================================================
# ESTIMATE BAR COUNT TESTS
# =============================================================================

class TestEstimateBarCount:
    """Test bar count estimation."""

    def test_estimate_time_bars(self, sample_ohlcv_1min):
        """Test estimating time bar count."""
        from src.phase1.stages.clean.bar_builders import estimate_bar_count

        estimate = estimate_bar_count(
            sample_ohlcv_1min,
            bar_type="time",
            target_timeframe="5min",
        )

        expected = len(sample_ohlcv_1min) // 5
        assert estimate == expected

    def test_estimate_volume_bars(self, sample_ohlcv_1min):
        """Test estimating volume bar count."""
        from src.phase1.stages.clean.bar_builders import estimate_bar_count

        total_volume = sample_ohlcv_1min["volume"].sum()
        threshold = 5000

        estimate = estimate_bar_count(
            sample_ohlcv_1min,
            bar_type="volume",
            threshold=threshold,
        )

        expected = int(total_volume / threshold)
        assert estimate == pytest.approx(expected, rel=0.1)

    def test_estimate_dollar_bars(self, sample_ohlcv_1min):
        """Test estimating dollar bar count."""
        from src.phase1.stages.clean.bar_builders import estimate_bar_count

        total_dollars = (sample_ohlcv_1min["volume"] * sample_ohlcv_1min["close"]).sum()
        threshold = 500000

        estimate = estimate_bar_count(
            sample_ohlcv_1min,
            bar_type="dollar",
            threshold=threshold,
        )

        expected = int(total_dollars / threshold)
        assert estimate == pytest.approx(expected, rel=0.1)


# =============================================================================
# EDGE CASES
# =============================================================================

class TestBarBuilderEdgeCases:
    """Test edge cases for bar builders."""

    def test_empty_input_raises(self):
        """Test empty input raises error."""
        from src.phase1.stages.clean.bar_builders import VolumeBarBuilder

        builder = VolumeBarBuilder(volume_threshold=1000)
        empty_df = pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

        with pytest.raises(ValueError, match="empty"):
            builder.build(empty_df)

    def test_single_bar_output(self, minimal_ohlcv):
        """Test when threshold results in single output bar."""
        from src.phase1.stages.clean.bar_builders import VolumeBarBuilder

        # Large threshold - should result in single bar
        builder = VolumeBarBuilder(volume_threshold=10000)
        bars = builder.build(minimal_ohlcv)

        assert len(bars) == 1

    def test_unsorted_input(self, sample_ohlcv_1min):
        """Test builder handles unsorted input."""
        from src.phase1.stages.clean.bar_builders import VolumeBarBuilder

        # Shuffle the data
        shuffled = sample_ohlcv_1min.sample(frac=1, random_state=42)

        builder = VolumeBarBuilder(volume_threshold=5000)
        bars = builder.build(shuffled)

        # Should sort and process correctly
        assert len(bars) > 0
        # Timestamps should be in order
        assert bars["datetime"].is_monotonic_increasing
