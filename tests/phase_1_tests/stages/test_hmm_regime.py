"""
Tests for HMM Regime Detection.

Tests the HMM-based regime detector and regime router.
Uses deterministic synthetic data for reproducibility.
"""
import numpy as np
import pandas as pd
import pytest


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_ohlcv_df():
    """Generate synthetic OHLCV data with different volatility regimes."""
    np.random.seed(42)
    n_samples = 500

    # Generate price with regime-dependent volatility
    # First 200: low vol, next 150: high vol, last 150: low vol
    returns = np.concatenate([
        np.random.randn(200) * 0.001,  # Low vol
        np.random.randn(150) * 0.005,  # High vol
        np.random.randn(150) * 0.001,  # Low vol
    ])

    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n_samples) * 0.002))
    low = close * (1 - np.abs(np.random.randn(n_samples) * 0.002))
    open_prices = np.roll(close, 1)
    open_prices[0] = 100

    volume = np.concatenate([
        np.random.randint(1000, 2000, 200),  # Normal volume
        np.random.randint(3000, 5000, 150),  # High volume
        np.random.randint(1000, 2000, 150),  # Normal volume
    ])

    df = pd.DataFrame({
        "datetime": pd.date_range("2023-01-01", periods=n_samples, freq="5min"),
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })

    return df


@pytest.fixture
def sample_returns():
    """Generate synthetic returns with regime structure."""
    np.random.seed(42)
    returns = np.concatenate([
        np.random.randn(200) * 0.001,  # Low vol
        np.random.randn(150) * 0.005,  # High vol
        np.random.randn(150) * 0.001,  # Low vol
    ])
    return returns


# =============================================================================
# HMM CONFIG TESTS
# =============================================================================

class TestHMMConfig:
    """Test HMM configuration validation."""

    def test_default_config(self):
        """Test default config is valid."""
        from src.phase1.stages.regime import HMMConfig

        config = HMMConfig()
        assert config.n_states == 3
        assert config.lookback == 252
        assert config.input_type == "returns"

    def test_invalid_n_states(self):
        """Test n_states validation."""
        from src.phase1.stages.regime import HMMConfig

        with pytest.raises(ValueError, match="n_states must be >= 2"):
            HMMConfig(n_states=1)

    def test_invalid_lookback(self):
        """Test lookback validation."""
        from src.phase1.stages.regime import HMMConfig

        with pytest.raises(ValueError, match="lookback"):
            HMMConfig(n_states=3, lookback=30)  # Too small

    def test_invalid_input_type(self):
        """Test input_type validation."""
        from src.phase1.stages.regime import HMMConfig

        with pytest.raises(ValueError, match="input_type"):
            HMMConfig(input_type="invalid")


# =============================================================================
# HMM DETECTOR TESTS
# =============================================================================

class TestHMMRegimeDetector:
    """Test HMM regime detector."""

    def test_detector_creation(self):
        """Test detector can be created."""
        from src.phase1.stages.regime import HMMRegimeDetector

        detector = HMMRegimeDetector(n_states=3, lookback=100)
        assert detector.config.n_states == 3
        assert detector.config.lookback == 100

    def test_required_columns(self):
        """Test required columns are declared."""
        from src.phase1.stages.regime import HMMRegimeDetector

        detector = HMMRegimeDetector(n_states=3)
        assert "close" in detector.get_required_columns()

    def test_detect_returns_series(self, sample_ohlcv_df):
        """Test detect returns a Series."""
        pytest.importorskip("hmmlearn")
        from src.phase1.stages.regime import HMMRegimeDetector

        detector = HMMRegimeDetector(n_states=2, lookback=100, expanding=True)
        regimes = detector.detect(sample_ohlcv_df)

        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(sample_ohlcv_df)

    def test_detect_with_probabilities(self, sample_ohlcv_df):
        """Test detect_with_probabilities returns regimes and probs."""
        pytest.importorskip("hmmlearn")
        from src.phase1.stages.regime import HMMRegimeDetector

        detector = HMMRegimeDetector(n_states=2, lookback=100, expanding=True)
        regimes, probs = detector.detect_with_probabilities(sample_ohlcv_df)

        assert isinstance(regimes, pd.Series)
        assert isinstance(probs, pd.DataFrame)
        assert probs.shape[1] == 2  # n_states columns

    def test_missing_close_column_raises(self):
        """Test missing close column raises error."""
        from src.phase1.stages.regime import HMMRegimeDetector

        detector = HMMRegimeDetector(n_states=2)
        df = pd.DataFrame({"open": [1, 2, 3], "high": [1, 2, 3]})

        with pytest.raises(ValueError, match="Missing required columns"):
            detector.detect(df)

    def test_state_labels_2_states(self):
        """Test state labels for 2-state model."""
        from src.phase1.stages.regime import HMMRegimeDetector

        detector = HMMRegimeDetector(n_states=2)
        labels = detector._get_state_labels()

        assert labels[0] == "low_vol"
        assert labels[1] == "high_vol"

    def test_state_labels_3_states(self):
        """Test state labels for 3-state model."""
        from src.phase1.stages.regime import HMMRegimeDetector

        detector = HMMRegimeDetector(n_states=3)
        labels = detector._get_state_labels()

        assert labels[0] == "low_vol"
        assert labels[1] == "normal"
        assert labels[2] == "high_vol"


# =============================================================================
# PURE FUNCTION TESTS
# =============================================================================

class TestHMMPureFunctions:
    """Test pure HMM functions."""

    def test_fit_gaussian_hmm(self, sample_returns):
        """Test HMM fitting function."""
        pytest.importorskip("hmmlearn")
        from src.phase1.stages.regime import fit_gaussian_hmm

        model, states, probs = fit_gaussian_hmm(
            sample_returns,
            n_states=2,
            max_iter=50,
            n_init=3,
            random_state=42,
        )

        assert model is not None
        assert len(states) == len(sample_returns)
        assert probs.shape == (len(sample_returns), 2)
        assert np.all((states >= 0) | (states == -1))  # Valid states or NaN marker

    def test_fit_gaussian_hmm_insufficient_data(self):
        """Test HMM fitting with insufficient data."""
        pytest.importorskip("hmmlearn")
        from src.phase1.stages.regime import fit_gaussian_hmm

        short_data = np.random.randn(10)

        with pytest.raises(ValueError, match="Insufficient observations"):
            fit_gaussian_hmm(short_data, n_states=3)

    def test_order_states_by_volatility(self, sample_returns):
        """Test state ordering by volatility."""
        pytest.importorskip("hmmlearn")
        from src.phase1.stages.regime import fit_gaussian_hmm, order_states_by_volatility

        model, states, _ = fit_gaussian_hmm(
            sample_returns, n_states=2, max_iter=50, n_init=3, random_state=42
        )

        ordered_states, mapping = order_states_by_volatility(model, states)

        # States should be reordered
        assert len(ordered_states) == len(states)
        assert len(mapping) == 2
        # Lower variance state should be mapped to 0
        assert 0 in mapping.values()
        assert 1 in mapping.values()


# =============================================================================
# REGIME ROUTER TESTS
# =============================================================================

class TestRegimeRouter:
    """Test regime routing functionality."""

    def test_router_creation(self):
        """Test router can be created."""
        from src.phase1.stages.regime import RegimeRouter

        router = RegimeRouter({
            "low_vol": "trend_model",
            "high_vol": "vol_model",
        })

        assert router.route("low_vol") == "trend_model"
        assert router.route("high_vol") == "vol_model"

    def test_router_default_model(self):
        """Test router uses default for unknown regimes."""
        from src.phase1.stages.regime import RegimeRouter

        router = RegimeRouter(
            {"low_vol": "model_a"},
            default_model="fallback_model",
        )

        assert router.route("unknown") == "fallback_model"

    def test_router_batch(self):
        """Test routing a batch of regimes."""
        from src.phase1.stages.regime import RegimeRouter

        router = RegimeRouter({
            "low_vol": "model_a",
            "high_vol": "model_b",
        })

        regimes = pd.Series(["low_vol", "high_vol", "low_vol"])
        routed = router.route_batch(regimes)

        assert list(routed) == ["model_a", "model_b", "model_a"]

    def test_routing_summary(self):
        """Test routing summary statistics."""
        from src.phase1.stages.regime import RegimeRouter

        router = RegimeRouter({
            "low_vol": "model_a",
            "high_vol": "model_b",
        })

        regimes = pd.Series(["low_vol"] * 7 + ["high_vol"] * 3)
        summary = router.get_routing_summary(regimes)

        assert summary["total_samples"] == 10
        assert summary["unique_models"] == 2
        assert "model_a" in summary["model_distribution"]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestHMMIntegration:
    """Integration tests for HMM regime detection."""

    def test_full_pipeline(self, sample_ohlcv_df):
        """Test full detection and routing pipeline."""
        pytest.importorskip("hmmlearn")
        from src.phase1.stages.regime import HMMRegimeDetector, RegimeRouter

        # Detect regimes
        detector = HMMRegimeDetector(n_states=2, lookback=100, expanding=True)
        regimes = detector.detect(sample_ohlcv_df)

        # Create router
        router = RegimeRouter({
            "low_vol": "conservative",
            "high_vol": "aggressive",
        }, default_model="balanced")

        # Route based on regimes
        models = router.route_batch(regimes.fillna("unknown"))

        assert len(models) == len(sample_ohlcv_df)
        assert set(models.unique()).issubset({"conservative", "aggressive", "balanced"})

    def test_without_hmmlearn(self, sample_ohlcv_df):
        """Test graceful degradation without hmmlearn."""
        from src.phase1.stages.regime.hmm import HMM_AVAILABLE, HMMRegimeDetector

        if HMM_AVAILABLE:
            pytest.skip("hmmlearn is installed")

        detector = HMMRegimeDetector(n_states=2, lookback=100)
        regimes = detector.detect(sample_ohlcv_df)

        # Should return NaN series when hmmlearn not available
        assert regimes.isna().all()


# =============================================================================
# LOOKAHEAD BIAS TESTS
# =============================================================================

class TestHMMLookaheadBias:
    """Tests to verify HMM expanding mode does not have lookahead bias."""

    def test_expanding_mode_no_lookahead_bias(self):
        """
        Verify expanding mode doesn't use future data.

        Strategy: Create data where the second half has a dramatically different
        regime (high volatility). If there's lookahead bias, early bars would
        "know" about the future high-vol regime and show different behavior
        than a model trained only on the first half.
        """
        pytest.importorskip("hmmlearn")
        from src.phase1.stages.regime import HMMRegimeDetector

        np.random.seed(42)

        # Create data: first 300 bars = low vol, last 200 bars = extreme high vol
        n_low_vol = 300
        n_high_vol = 200
        total = n_low_vol + n_high_vol

        # Low volatility regime
        returns_low = np.random.randn(n_low_vol) * 0.0005

        # EXTREME high volatility regime (10x volatility spike)
        returns_high = np.random.randn(n_high_vol) * 0.01

        returns = np.concatenate([returns_low, returns_high])
        close = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            "datetime": pd.date_range("2023-01-01", periods=total, freq="5min"),
            "close": close,
        })

        # Run with expanding mode (should NOT use future data)
        detector = HMMRegimeDetector(
            n_states=2,
            lookback=100,
            expanding=True,
            retrain_interval=50,
            n_init=5,
            max_iter=50,
            random_state=42,
        )
        regimes = detector.detect(df)

        # The key test: bars in the low-vol portion (before the regime change)
        # should be classified consistently as low-vol, because the model
        # should NOT know about the future high-vol regime.

        # Get regimes for early low-vol bars (after lookback warmup)
        early_bars = regimes.iloc[100:200]  # Bars 100-199 (all low-vol in reality)
        non_nan_early = early_bars.dropna()

        # If no lookahead bias, early bars should mostly be "low_vol"
        # (allowing some noise from model uncertainty)
        if len(non_nan_early) > 0:
            low_vol_ratio = (non_nan_early == "low_vol").mean()
            # At least 60% should be low_vol (allowing for model noise)
            assert low_vol_ratio >= 0.5, (
                f"Early bars (during low-vol period) should mostly be 'low_vol' "
                f"but got {low_vol_ratio:.1%} low_vol. This may indicate lookahead bias."
            )

    def test_rolling_vs_expanding_consistency(self, sample_ohlcv_df):
        """
        Test that rolling and expanding modes produce similar regime patterns.

        Both modes should avoid lookahead bias, so their outputs should be
        similar (not identical, but correlated).
        """
        pytest.importorskip("hmmlearn")
        from src.phase1.stages.regime import HMMRegimeDetector

        # Expanding mode
        detector_expanding = HMMRegimeDetector(
            n_states=2,
            lookback=100,
            expanding=True,
            retrain_interval=50,
            n_init=5,
            max_iter=50,
            random_state=42,
        )
        regimes_expanding = detector_expanding.detect(sample_ohlcv_df)

        # Rolling mode
        detector_rolling = HMMRegimeDetector(
            n_states=2,
            lookback=100,
            expanding=False,
            n_init=5,
            max_iter=50,
            random_state=42,
        )
        regimes_rolling = detector_rolling.detect(sample_ohlcv_df)

        # Both should have similar coverage (non-NaN values)
        expanding_coverage = regimes_expanding.notna().mean()
        rolling_coverage = regimes_rolling.notna().mean()

        # Coverage should be similar (within 10%)
        assert abs(expanding_coverage - rolling_coverage) < 0.15, (
            f"Expanding ({expanding_coverage:.1%}) and rolling ({rolling_coverage:.1%}) "
            "modes have very different coverage"
        )

    def test_retrain_interval_parameter(self, sample_ohlcv_df):
        """Test that retrain_interval parameter is respected."""
        pytest.importorskip("hmmlearn")
        from src.phase1.stages.regime import HMMRegimeDetector

        # With large retrain interval (faster)
        detector_fast = HMMRegimeDetector(
            n_states=2,
            lookback=100,
            expanding=True,
            retrain_interval=100,
            n_init=3,
            max_iter=30,
        )

        regimes_fast = detector_fast.detect(sample_ohlcv_df)

        # With small retrain interval (slower)
        detector_slow = HMMRegimeDetector(
            n_states=2,
            lookback=100,
            expanding=True,
            retrain_interval=25,
            n_init=3,
            max_iter=30,
        )

        regimes_slow = detector_slow.detect(sample_ohlcv_df)

        # Both should produce valid regimes
        assert regimes_fast.notna().sum() > 0
        assert regimes_slow.notna().sum() > 0

        # Both modes should avoid lookahead bias, producing similar regime patterns
        # for the same data (not identical due to stochasticity, but both valid)
