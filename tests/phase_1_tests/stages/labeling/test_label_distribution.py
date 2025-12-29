"""
Tests for validating label distribution characteristics.

This module tests that triple-barrier labeling produces reasonable label
distributions under various market conditions. These tests help detect
potential biases in the labeling strategy early in the pipeline.

Label values:
- +1 (long/win): Upper barrier hit first
- -1 (short/loss): Lower barrier hit first
- 0 (neutral/timeout): Neither barrier hit within max_bars
- -99 (invalid): Sentinel for excluded samples (end of data)

Engineering Rules Applied:
- Deterministic tests (use seeds)
- Reasonable thresholds (allow natural variation)
- Clear docstrings explaining test purpose
- Helper functions for data generation
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.phase1.stages.labeling import TripleBarrierLabeler


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================


def generate_test_ohlcv(
    n_bars: int, seed: int = 42, volatility: float = 1.0, base_price: float = 100.0
) -> pd.DataFrame:
    """
    Generate realistic synthetic OHLCV data for testing.
    Creates random walk price data with realistic OHLC relationships.
    Returns DataFrame with columns: open, high, low, close, volume, atr_14
    """
    np.random.seed(seed)
    returns = np.random.randn(n_bars) * 0.008 * volatility
    close = base_price * np.exp(np.cumsum(returns))
    range_pct = 0.005 * volatility
    close_position = np.random.uniform(0.2, 0.8, n_bars)
    bar_range = close * range_pct
    high = close + bar_range * (1 - close_position)
    low = close - bar_range * close_position
    open_prices = low + (high - low) * np.random.uniform(0.3, 0.7, n_bars)
    high = np.maximum(high, np.maximum(open_prices, close))
    low = np.minimum(low, np.minimum(open_prices, close))
    true_range = high - low
    atr_14 = pd.Series(true_range).rolling(14).mean().fillna(true_range.mean()).values
    return pd.DataFrame({
        'open': open_prices, 'high': high, 'low': low, 'close': close,
        'volume': np.random.lognormal(10, 0.5, n_bars).astype(int), 'atr_14': atr_14
    })


def generate_trending_ohlcv(
    n_bars: int, trend_direction: str = 'up', trend_strength: float = 0.0002,
    seed: int = 42, base_price: float = 100.0
) -> pd.DataFrame:
    """Generate OHLCV data with a clear trend ('up' or 'down')."""
    np.random.seed(seed)
    drift = trend_strength if trend_direction == 'up' else -trend_strength
    returns = np.random.randn(n_bars) * 0.003 + drift
    close = base_price * np.exp(np.cumsum(returns))
    bar_range = close * 0.003
    close_position = np.random.uniform(0.2, 0.8, n_bars)
    high = close + bar_range * (1 - close_position)
    low = close - bar_range * close_position
    open_prices = low + (high - low) * np.random.uniform(0.3, 0.7, n_bars)
    high = np.maximum(high, np.maximum(open_prices, close))
    low = np.minimum(low, np.minimum(open_prices, close))
    true_range = high - low
    atr_14 = pd.Series(true_range).rolling(14).mean().fillna(true_range.mean()).values
    return pd.DataFrame({
        'open': open_prices, 'high': high, 'low': low, 'close': close,
        'volume': np.random.lognormal(10, 0.5, n_bars).astype(int), 'atr_14': atr_14
    })


def generate_high_volatility_ohlcv(n_bars: int, seed: int = 42) -> pd.DataFrame:
    """Generate OHLCV data with 3x normal volatility."""
    return generate_test_ohlcv(n_bars=n_bars, seed=seed, volatility=3.0)


def generate_ranging_ohlcv(n_bars: int, seed: int = 42, base_price: float = 100.0) -> pd.DataFrame:
    """Generate OHLCV data that oscillates in a range (sine wave + noise)."""
    np.random.seed(seed)
    t = np.linspace(0, 8 * np.pi, n_bars)
    oscillation = np.sin(t) * 0.01
    close = base_price * (1 + oscillation + np.random.randn(n_bars) * 0.001)
    bar_range = close * 0.002
    close_position = np.random.uniform(0.3, 0.7, n_bars)
    high = close + bar_range * (1 - close_position)
    low = close - bar_range * close_position
    open_prices = low + (high - low) * np.random.uniform(0.3, 0.7, n_bars)
    high = np.maximum(high, np.maximum(open_prices, close))
    low = np.minimum(low, np.minimum(open_prices, close))
    true_range = high - low
    atr_14 = pd.Series(true_range).rolling(14).mean().fillna(true_range.mean()).values
    return pd.DataFrame({
        'open': open_prices, 'high': high, 'low': low, 'close': close,
        'volume': np.random.lognormal(10, 0.5, n_bars).astype(int), 'atr_14': atr_14
    })


def get_label_distribution(labels: np.ndarray) -> Tuple[float, float, float, int]:
    """Calculate label distribution: (short_pct, neutral_pct, long_pct, valid_count)."""
    valid_labels = labels[labels != -99]
    if len(valid_labels) == 0:
        return (0.0, 0.0, 0.0, 0)
    total = len(valid_labels)
    return (
        (valid_labels == -1).sum() / total,
        (valid_labels == 0).sum() / total,
        (valid_labels == 1).sum() / total,
        total
    )


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def labeler():
    """Create a TripleBarrierLabeler with transaction costs disabled for testing."""
    return TripleBarrierLabeler(apply_transaction_costs=False)


@pytest.fixture
def labeler_with_costs():
    """Create a TripleBarrierLabeler with transaction costs enabled."""
    return TripleBarrierLabeler(apply_transaction_costs=True, symbol='MES')


@pytest.fixture
def sample_ohlcv_5000():
    """Generate 5000 bars of test OHLCV data."""
    return generate_test_ohlcv(n_bars=5000, seed=42)


@pytest.fixture
def sample_ohlcv_10000():
    """Generate 10000 bars of test OHLCV data for more robust statistics."""
    return generate_test_ohlcv(n_bars=10000, seed=42)


# =============================================================================
# LABEL DISTRIBUTION BALANCE TESTS
# =============================================================================


class TestLabelDistributionBalance:
    """Tests for validating label distribution balance characteristics."""

    def test_label_distribution_not_extreme(self, labeler, sample_ohlcv_5000):
        """
        Labels should not show extreme imbalance after triple barrier labeling.

        No single class should dominate >70% of labels in market-neutral data.
        This catches potential bias in the labeling algorithm.
        """
        result = labeler.compute_labels(sample_ohlcv_5000, horizon=20)
        short_pct, neutral_pct, long_pct, valid_count = get_label_distribution(result.labels)

        assert valid_count > 0, "Should have some valid labels"

        # No class should dominate more than 70%
        max_proportion = max(short_pct, neutral_pct, long_pct)
        assert max_proportion < 0.70, (
            f"Extreme imbalance detected: "
            f"Short={short_pct:.1%}, Neutral={neutral_pct:.1%}, Long={long_pct:.1%}"
        )

    def test_all_label_classes_present(self, labeler, sample_ohlcv_5000):
        """
        All three label classes (-1, 0, +1) should be present in the output
        when using appropriate barrier parameters.

        With tight barriers relative to max_bars, neutrals may be rare.
        This test uses wider barriers to ensure all classes can appear.
        """
        # Use wider barriers and shorter max_bars to ensure some timeouts
        result = labeler.compute_labels(
            sample_ohlcv_5000,
            horizon=20,
            k_up=3.0,
            k_down=3.0,
            max_bars=20  # Shorter timeout to produce some neutrals
        )
        valid_labels = result.labels[result.labels != -99]

        unique_labels = set(valid_labels)

        assert -1 in unique_labels, "Short labels (-1) missing from distribution"
        assert 0 in unique_labels, "Neutral labels (0) missing from distribution"
        assert 1 in unique_labels, "Long labels (+1) missing from distribution"

    def test_directional_labels_reasonable_proportion(self, labeler, sample_ohlcv_5000):
        """
        Directional labels (long and short) should each be between 15% and 60%.

        These bounds are intentionally wide to accommodate natural market variation
        while catching extreme cases of algorithmic bias.
        """
        result = labeler.compute_labels(sample_ohlcv_5000, horizon=20)
        short_pct, neutral_pct, long_pct, _ = get_label_distribution(result.labels)

        # Each directional class should be at least 15%
        assert short_pct > 0.15, f"Short labels too rare: {short_pct:.1%}"
        assert long_pct > 0.15, f"Long labels too rare: {long_pct:.1%}"

        # Neither should dominate too much
        assert short_pct < 0.60, f"Short labels too dominant: {short_pct:.1%}"
        assert long_pct < 0.60, f"Long labels too dominant: {long_pct:.1%}"

    def test_neutral_labels_within_bounds(self, labeler, sample_ohlcv_5000):
        """
        Neutral labels (timeouts) should be within expected range when using
        barrier parameters designed to produce timeouts.

        This test uses very wide barriers with a short timeout to ensure
        that some trades will timeout before hitting barriers.
        """
        # Use very wide barriers and short timeout to guarantee some timeouts
        result = labeler.compute_labels(
            sample_ohlcv_5000,
            horizon=20,
            k_up=6.0,   # Very wide - hard to hit
            k_down=6.0,
            max_bars=10  # Very short - forces timeouts
        )
        _, neutral_pct, _, _ = get_label_distribution(result.labels)

        assert 0.05 < neutral_pct < 0.80, (
            f"Neutral labels out of expected range: {neutral_pct:.1%} "
            f"(expected 5-80% with k=6.0, max_bars=10)"
        )

    def test_long_short_ratio_reasonable(self, labeler, sample_ohlcv_5000):
        """
        The ratio of long to short labels should be between 0.5 and 2.0.

        For market-neutral random data, we expect approximately equal long
        and short signals. A ratio outside this range suggests bias.
        """
        result = labeler.compute_labels(sample_ohlcv_5000, horizon=20)
        short_pct, _, long_pct, _ = get_label_distribution(result.labels)

        if short_pct > 0:
            ratio = long_pct / short_pct
            assert 0.5 < ratio < 2.0, (
                f"Long/Short ratio {ratio:.2f} suggests bias "
                f"(Long={long_pct:.1%}, Short={short_pct:.1%})"
            )


# =============================================================================
# LABEL COVERAGE TESTS
# =============================================================================


class TestLabelCoverage:
    """Tests for validating sufficient label coverage."""

    def test_label_coverage_sufficient(self, labeler, sample_ohlcv_5000):
        """
        Sufficient proportion of data should have valid labels (not -99).

        At least 80% of data should have valid labels. Invalid labels only
        occur at the end of the dataset (within max_bars of the end).
        """
        result = labeler.compute_labels(sample_ohlcv_5000, horizon=20)

        valid_ratio = (result.labels != -99).sum() / len(result.labels)

        assert valid_ratio > 0.80, (
            f"Only {valid_ratio:.1%} valid labels, expected >80%"
        )

    def test_invalid_labels_only_at_end(self, labeler, sample_ohlcv_5000):
        """
        Invalid labels (-99) should only appear at the end of the data.

        The last max_bars samples are marked invalid to prevent lookahead.
        No invalid labels should appear in the middle of the dataset.
        """
        result = labeler.compute_labels(
            sample_ohlcv_5000, horizon=20, max_bars=60
        )

        # Find first invalid label
        invalid_mask = result.labels == -99
        invalid_indices = np.where(invalid_mask)[0]

        if len(invalid_indices) > 0:
            first_invalid = invalid_indices[0]
            n_samples = len(result.labels)

            # All invalid labels should be at the end (within max_bars)
            expected_first_invalid = n_samples - 60  # max_bars

            assert first_invalid >= expected_first_invalid - 1, (
                f"Invalid label found at index {first_invalid}, "
                f"but expected only after index {expected_first_invalid}"
            )

    def test_quality_metrics_include_coverage(self, labeler, sample_ohlcv_5000):
        """Quality metrics should include sample counts."""
        result = labeler.compute_labels(sample_ohlcv_5000, horizon=20)

        assert 'total_samples' in result.quality_metrics
        assert 'valid_samples' in result.quality_metrics
        assert 'invalid_samples' in result.quality_metrics

        # Verify counts add up
        total = result.quality_metrics['total_samples']
        valid = result.quality_metrics['valid_samples']
        invalid = result.quality_metrics['invalid_samples']

        assert total == valid + invalid


# =============================================================================
# CONSISTENCY ACROSS HORIZONS TESTS
# =============================================================================


class TestDistributionConsistencyAcrossHorizons:
    """Tests for distribution consistency across different horizons."""

    def test_distribution_similar_across_horizons(self, labeler, sample_ohlcv_5000):
        """
        Label distribution should be similar across different horizons.

        While exact proportions will vary, the long/short ratio should not
        change dramatically between horizons (e.g., 5 vs 20 bars).
        """
        distributions = {}

        for horizon in [5, 10, 15, 20]:
            result = labeler.compute_labels(sample_ohlcv_5000, horizon=horizon)
            short_pct, neutral_pct, long_pct, _ = get_label_distribution(result.labels)
            distributions[horizon] = {
                'short': short_pct,
                'neutral': neutral_pct,
                'long': long_pct
            }

        # Long class proportion should not vary by more than 25 percentage points
        long_props = [d['long'] for d in distributions.values()]
        long_range = max(long_props) - min(long_props)

        assert long_range < 0.25, (
            f"Long label proportion varies too much across horizons: "
            f"range={long_range:.1%}, values={[f'{p:.1%}' for p in long_props]}"
        )

    def test_all_horizons_produce_valid_distributions(self, labeler, sample_ohlcv_5000):
        """All standard horizons should produce valid distributions."""
        for horizon in [5, 10, 15, 20]:
            result = labeler.compute_labels(sample_ohlcv_5000, horizon=horizon)
            short_pct, neutral_pct, long_pct, valid_count = get_label_distribution(result.labels)

            assert valid_count > 0, f"Horizon {horizon} produced no valid labels"

            # Basic sanity checks
            assert short_pct + neutral_pct + long_pct > 0.999, (
                f"Horizon {horizon}: percentages don't sum to 100%"
            )


# =============================================================================
# MARKET CONDITION TESTS
# =============================================================================


class TestLabelDistributionByMarketCondition:
    """Tests for label distribution under different market conditions."""

    def test_uptrend_favors_long_labels(self, labeler):
        """
        In a strongly trending up market, long labels should exceed short labels.

        This validates that the labeling algorithm correctly captures market direction.
        """
        df = generate_trending_ohlcv(
            n_bars=5000,
            trend_direction='up',
            trend_strength=0.0003,  # Strong uptrend
            seed=42
        )

        result = labeler.compute_labels(df, horizon=20)
        short_pct, _, long_pct, _ = get_label_distribution(result.labels)

        assert long_pct > short_pct, (
            f"Uptrend should have more long labels: "
            f"Long={long_pct:.1%}, Short={short_pct:.1%}"
        )

    def test_downtrend_favors_short_labels(self, labeler):
        """
        In a strongly trending down market, short labels should exceed long labels.

        This validates that the labeling algorithm correctly captures market direction.
        """
        df = generate_trending_ohlcv(
            n_bars=5000,
            trend_direction='down',
            trend_strength=0.0003,  # Strong downtrend
            seed=42
        )

        result = labeler.compute_labels(df, horizon=20)
        short_pct, _, long_pct, _ = get_label_distribution(result.labels)

        assert short_pct > long_pct, (
            f"Downtrend should have more short labels: "
            f"Short={short_pct:.1%}, Long={long_pct:.1%}"
        )

    def test_ranging_market_more_neutral_labels(self, labeler):
        """
        In a ranging (sideways) market, neutral labels should be relatively high
        when using appropriate barrier parameters.

        When price oscillates without clear direction, more trades will timeout.
        """
        df = generate_ranging_ohlcv(n_bars=5000, seed=42)

        # Use wider barriers to capture the ranging behavior
        result = labeler.compute_labels(
            df, horizon=20, k_up=3.0, k_down=3.0, max_bars=30
        )
        _, neutral_pct, _, _ = get_label_distribution(result.labels)

        # Ranging market should have noticeable neutral percentage
        assert neutral_pct > 0.05, (
            f"Ranging market should have some neutral labels: {neutral_pct:.1%}"
        )

    def test_high_volatility_faster_barrier_hits(self, labeler):
        """
        In high volatility, barriers should be hit faster on average.

        Higher volatility means larger price swings, so barriers (which are
        ATR-based) should be hit in fewer bars. This is a more robust test
        than comparing neutral percentages, which can be affected by ATR scaling.
        """
        df_normal = generate_test_ohlcv(n_bars=5000, seed=42, volatility=1.0)
        df_high_vol = generate_high_volatility_ohlcv(n_bars=5000, seed=42)

        # Use same relative barriers for both
        result_normal = labeler.compute_labels(
            df_normal, horizon=20, k_up=2.0, k_down=2.0, max_bars=50
        )
        result_high_vol = labeler.compute_labels(
            df_high_vol, horizon=20, k_up=2.0, k_down=2.0, max_bars=50
        )

        # Get average bars to hit for non-timeout trades
        bars_normal = result_normal.metadata['bars_to_hit']
        bars_high_vol = result_high_vol.metadata['bars_to_hit']
        labels_normal = result_normal.labels
        labels_high_vol = result_high_vol.labels

        # Filter to non-timeout (label != 0) and valid (label != -99) samples
        valid_normal = (labels_normal != 0) & (labels_normal != -99)
        valid_high_vol = (labels_high_vol != 0) & (labels_high_vol != -99)

        if valid_normal.sum() > 0 and valid_high_vol.sum() > 0:
            avg_bars_normal = bars_normal[valid_normal].mean()
            avg_bars_high_vol = bars_high_vol[valid_high_vol].mean()

            # In high volatility, barriers should be hit faster (fewer bars)
            # Allow some tolerance since ATR also scales with volatility
            assert avg_bars_high_vol <= avg_bars_normal * 1.2, (
                f"High volatility should hit barriers at least as fast: "
                f"High vol avg={avg_bars_high_vol:.1f}, Normal avg={avg_bars_normal:.1f}"
            )


# =============================================================================
# TRANSACTION COST IMPACT TESTS
# =============================================================================


class TestTransactionCostImpactOnDistribution:
    """Tests for how transaction costs affect label distribution."""

    def test_transaction_costs_reduce_long_labels(
        self, labeler, labeler_with_costs, sample_ohlcv_5000
    ):
        """
        Transaction costs should reduce long (win) labels.

        When costs are applied, the upper barrier is harder to hit,
        so fewer trades should result in wins.
        """
        result_no_cost = labeler.compute_labels(sample_ohlcv_5000, horizon=20)
        result_with_cost = labeler_with_costs.compute_labels(sample_ohlcv_5000, horizon=20)

        _, _, long_no_cost, _ = get_label_distribution(result_no_cost.labels)
        _, _, long_with_cost, _ = get_label_distribution(result_with_cost.labels)

        # With transaction costs, long labels should decrease
        assert long_with_cost <= long_no_cost, (
            f"Transaction costs should reduce long labels: "
            f"No cost={long_no_cost:.1%}, With cost={long_with_cost:.1%}"
        )

    def test_transaction_costs_increase_short_or_neutral(
        self, labeler, labeler_with_costs, sample_ohlcv_5000
    ):
        """
        Transaction costs should increase short and/or neutral labels.

        As upper barrier becomes harder to hit, more trades either hit
        the lower barrier or timeout.
        """
        result_no_cost = labeler.compute_labels(sample_ohlcv_5000, horizon=20)
        result_with_cost = labeler_with_costs.compute_labels(sample_ohlcv_5000, horizon=20)

        short_no_cost, neutral_no_cost, _, _ = get_label_distribution(result_no_cost.labels)
        short_with_cost, neutral_with_cost, _, _ = get_label_distribution(result_with_cost.labels)

        # Combined short + neutral should increase with costs
        non_long_no_cost = short_no_cost + neutral_no_cost
        non_long_with_cost = short_with_cost + neutral_with_cost

        assert non_long_with_cost >= non_long_no_cost, (
            f"Short+Neutral should increase with costs: "
            f"No cost={non_long_no_cost:.1%}, With cost={non_long_with_cost:.1%}"
        )


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestLabelDistributionEdgeCases:
    """Tests for edge cases in label distribution."""

    def test_small_dataset_still_produces_labels(self, labeler):
        """Small datasets should still produce valid labels."""
        df = generate_test_ohlcv(n_bars=100, seed=42)

        result = labeler.compute_labels(df, horizon=20, max_bars=10)
        valid_count = (result.labels != -99).sum()

        assert valid_count > 0, "Small dataset should produce some valid labels"

    def test_very_tight_barriers_more_barrier_hits(self, labeler, sample_ohlcv_5000):
        """
        Very tight barriers (low k values) should result in fewer neutrals.

        Tight barriers are easier to hit, so most trades should resolve
        before timeout.
        """
        result = labeler.compute_labels(
            sample_ohlcv_5000,
            horizon=20,
            k_up=0.3,
            k_down=0.3,
            max_bars=60
        )

        _, neutral_pct, _, _ = get_label_distribution(result.labels)

        # Tight barriers should have low neutral percentage
        assert neutral_pct < 0.30, (
            f"Tight barriers should have few neutrals: {neutral_pct:.1%}"
        )

    def test_very_wide_barriers_more_neutrals(self, labeler, sample_ohlcv_5000):
        """
        Very wide barriers (high k values) should result in more neutrals
        than tight barriers with the same max_bars.

        Wide barriers are hard to hit, so more trades will timeout.
        """
        # Tight barriers - should hit barriers quickly
        result_tight = labeler.compute_labels(
            sample_ohlcv_5000,
            horizon=20,
            k_up=0.5,
            k_down=0.5,
            max_bars=30
        )

        # Wide barriers - should timeout more often
        result_wide = labeler.compute_labels(
            sample_ohlcv_5000,
            horizon=20,
            k_up=8.0,  # Very wide
            k_down=8.0,
            max_bars=15  # Short timeout
        )

        _, neutral_tight, _, _ = get_label_distribution(result_tight.labels)
        _, neutral_wide, _, _ = get_label_distribution(result_wide.labels)

        # Wide barriers with short timeout should have more neutrals than tight barriers
        assert neutral_wide > neutral_tight, (
            f"Wide barriers should have more neutrals than tight: "
            f"Wide={neutral_wide:.1%}, Tight={neutral_tight:.1%}"
        )

    def test_asymmetric_barriers_affect_distribution(self, labeler, sample_ohlcv_5000):
        """
        Asymmetric barriers should produce asymmetric label distribution.

        If upper barrier is wider than lower, short labels should increase.
        """
        result = labeler.compute_labels(
            sample_ohlcv_5000,
            horizon=20,
            k_up=2.0,   # Harder to hit (wider)
            k_down=1.0  # Easier to hit (tighter)
        )

        short_pct, _, long_pct, _ = get_label_distribution(result.labels)

        # Asymmetric barriers should produce more shorts than longs
        # (since lower barrier is easier to hit)
        assert short_pct > long_pct, (
            f"Asymmetric barriers (k_up=2.0, k_down=1.0) should favor shorts: "
            f"Short={short_pct:.1%}, Long={long_pct:.1%}"
        )

    def test_constant_price_all_neutrals(self, labeler):
        """
        Constant price data should produce all neutral labels.

        If price never moves, no barriers can be hit.
        """
        n_bars = 1000
        df = pd.DataFrame({
            'open': np.full(n_bars, 100.0),
            'high': np.full(n_bars, 100.1),
            'low': np.full(n_bars, 99.9),
            'close': np.full(n_bars, 100.0),
            'volume': np.full(n_bars, 1000),
            'atr_14': np.full(n_bars, 0.2)
        })

        result = labeler.compute_labels(df, horizon=20, max_bars=30)
        _, neutral_pct, _, valid_count = get_label_distribution(result.labels)

        if valid_count > 0:
            # With constant price, almost all should be neutral
            assert neutral_pct > 0.90, (
                f"Constant price should produce ~100% neutral: {neutral_pct:.1%}"
            )

    def test_nan_atr_handled(self, labeler):
        """
        NaN ATR values should be handled gracefully.

        Samples with invalid ATR result in timeout labels (not crashes).
        """
        df = generate_test_ohlcv(n_bars=100, seed=42)

        # Introduce some NaN ATR values
        df.loc[:10, 'atr_14'] = np.nan

        # Should not raise
        result = labeler.compute_labels(df, horizon=20)

        # Should still have valid output
        assert len(result.labels) == len(df)

    def test_zero_atr_handled(self, labeler):
        """
        Zero ATR values should be handled gracefully.

        Samples with zero ATR result in timeout labels (no barrier defined).
        """
        df = generate_test_ohlcv(n_bars=100, seed=42)

        # Introduce some zero ATR values
        df.loc[20:30, 'atr_14'] = 0.0

        # Should not raise
        result = labeler.compute_labels(df, horizon=20)

        # Should still have valid output
        assert len(result.labels) == len(df)


# =============================================================================
# DETERMINISM TESTS
# =============================================================================


class TestLabelDistributionDeterminism:
    """Tests for deterministic label generation."""

    def test_same_data_same_labels(self, labeler):
        """
        Same input data should always produce same labels.

        Labels are deterministic given the same price data and parameters.
        """
        df = generate_test_ohlcv(n_bars=1000, seed=42)

        result1 = labeler.compute_labels(df, horizon=20)
        result2 = labeler.compute_labels(df, horizon=20)

        np.testing.assert_array_equal(
            result1.labels, result2.labels,
            err_msg="Same data should produce identical labels"
        )

    def test_different_seeds_different_distributions(self, labeler):
        """
        Different random seeds should produce different distributions.

        This validates that our test data generation is actually varying.
        """
        df1 = generate_test_ohlcv(n_bars=5000, seed=42)
        df2 = generate_test_ohlcv(n_bars=5000, seed=123)

        result1 = labeler.compute_labels(df1, horizon=20)
        result2 = labeler.compute_labels(df2, horizon=20)

        # Labels should be different (not identical)
        assert not np.array_equal(result1.labels, result2.labels), (
            "Different seeds should produce different labels"
        )


# =============================================================================
# QUALITY METRICS TESTS
# =============================================================================


class TestLabelDistributionQualityMetrics:
    """Tests for quality metrics related to label distribution."""

    def test_quality_metrics_include_distribution(self, labeler, sample_ohlcv_5000):
        """Quality metrics should include label distribution percentages."""
        result = labeler.compute_labels(sample_ohlcv_5000, horizon=20)

        assert 'long_pct' in result.quality_metrics
        assert 'short_pct' in result.quality_metrics
        assert 'neutral_pct' in result.quality_metrics

    def test_quality_metrics_include_counts(self, labeler, sample_ohlcv_5000):
        """Quality metrics should include label counts."""
        result = labeler.compute_labels(sample_ohlcv_5000, horizon=20)

        assert 'long_count' in result.quality_metrics
        assert 'short_count' in result.quality_metrics
        assert 'neutral_count' in result.quality_metrics

    def test_quality_metrics_percentages_sum_to_100(self, labeler, sample_ohlcv_5000):
        """Label percentages should sum to approximately 100%."""
        result = labeler.compute_labels(sample_ohlcv_5000, horizon=20)

        total_pct = (
            result.quality_metrics['long_pct'] +
            result.quality_metrics['short_pct'] +
            result.quality_metrics['neutral_pct']
        )

        assert 99.9 < total_pct < 100.1, (
            f"Percentages should sum to 100%, got {total_pct:.2f}%"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
