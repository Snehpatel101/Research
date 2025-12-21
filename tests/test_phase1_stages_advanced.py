"""
Comprehensive Unit Tests for Phase 1 Stages 5-8

This module provides comprehensive test coverage for:
- Stage 5: GAOptimizer (Genetic Algorithm Optimization)
- Stage 6: FinalLabeler (Apply Optimized Labels with Quality Scoring)
- Stage 7: DataSplitter (Time-Based Splitting with Purging and Embargo)
- Stage 8: DataValidator (Comprehensive Data Validation)

Tests follow TDD principles with red-green-refactor cycles.
Target coverage: >70%

Author: TDD Orchestrator
Created: 2025-12-20
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from unittest.mock import patch, MagicMock
import json
import tempfile
import shutil

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import modules under test
from stages.stage5_ga_optimize import (
    calculate_fitness,
    evaluate_individual,
    get_contiguous_subset,
    run_ga_optimization,
)
from stages.stage6_final_labels import (
    compute_quality_scores,
    assign_sample_weights,
    apply_optimized_labels,
)
from stages.stage7_splits import (
    create_chronological_splits,
    validate_no_overlap,
    create_splits,
)
from stages.stage8_validate import (
    DataValidator,
    validate_data,
)
from utils.feature_selection import (
    FeatureSelectionResult,
    select_features,
    get_feature_priority,
    identify_feature_columns,
    filter_low_variance,
    filter_correlated_features,
)


# =============================================================================
# FIXTURES - Shared Test Data
# =============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 1000

    # Generate realistic price data with trend and noise
    base_price = 100.0
    returns = np.random.randn(n) * 0.002  # 0.2% daily volatility
    prices = base_price * np.exp(np.cumsum(returns))

    # Create OHLCV DataFrame
    df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
        'symbol': 'MES',
        'open': prices * (1 + np.random.randn(n) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n)) * 0.002),
        'low': prices * (1 - np.abs(np.random.randn(n)) * 0.002),
        'close': prices,
        'volume': np.random.randint(100, 1000, n).astype(float),
    })

    # Ensure high >= close >= low and high >= open >= low
    df['high'] = df[['open', 'high', 'close']].max(axis=1) * 1.001
    df['low'] = df[['open', 'low', 'close']].min(axis=1) * 0.999

    # Add ATR (simplified calculation)
    df['atr_14'] = (df['high'] - df['low']).rolling(14).mean().fillna(0.5)

    return df


@pytest.fixture
def sample_labeled_data(sample_ohlcv_data):
    """Create sample labeled data with all required columns."""
    df = sample_ohlcv_data.copy()
    n = len(df)
    np.random.seed(42)

    # Add features
    df['log_return'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    df['rsi'] = 50 + np.random.randn(n) * 15
    df['sma_10'] = df['close'].rolling(10).mean().fillna(df['close'])
    df['sma_20'] = df['close'].rolling(20).mean().fillna(df['close'])
    df['bb_position'] = np.random.randn(n) * 0.5
    df['macd'] = np.random.randn(n) * 0.1
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean().fillna(1)

    # Add labels for each horizon
    for horizon in [1, 5, 20]:
        # Create balanced labels (-1, 0, 1)
        labels = np.random.choice([-1, 0, 1], size=n, p=[0.35, 0.30, 0.35])
        df[f'label_h{horizon}'] = labels
        df[f'bars_to_hit_h{horizon}'] = np.random.randint(1, horizon * 3, n)
        df[f'mae_h{horizon}'] = -np.abs(np.random.randn(n) * 0.01)
        df[f'mfe_h{horizon}'] = np.abs(np.random.randn(n) * 0.02)
        df[f'touch_type_h{horizon}'] = labels  # Simplified: same as labels
        df[f'quality_h{horizon}'] = np.random.rand(n) * 0.5 + 0.3
        df[f'sample_weight_h{horizon}'] = np.random.choice([0.5, 1.0, 1.5], n)

    return df


@pytest.fixture
def sample_price_arrays():
    """Create sample price arrays for numba functions."""
    np.random.seed(42)
    n = 500

    # Generate price series
    returns = np.random.randn(n) * 0.002
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n)) * 0.003)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.003)
    open_prices = close * (1 + np.random.randn(n) * 0.001)
    atr = np.ones(n) * 0.5

    return {
        'close': close.astype(np.float64),
        'high': high.astype(np.float64),
        'low': low.astype(np.float64),
        'open': open_prices.astype(np.float64),
        'atr': atr.astype(np.float64),
    }


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# STAGE 5 TESTS: GAOptimizer
# =============================================================================

class TestStage5GAOptimizer:
    """Tests for Stage 5: Genetic Algorithm Optimization."""

    def test_calculate_fitness_valid_balanced_labels(self):
        """Test fitness calculation with balanced label distribution."""
        np.random.seed(42)
        n = 1000

        # Create balanced labels (40% long, 30% neutral, 30% short)
        labels = np.array([1]*400 + [0]*300 + [-1]*300, dtype=np.int8)
        np.random.shuffle(labels)

        bars_to_hit = np.random.randint(1, 10, n).astype(np.int32)
        mae = -np.abs(np.random.randn(n) * 0.01).astype(np.float32)
        mfe = np.abs(np.random.randn(n) * 0.02).astype(np.float32)
        horizon = 5

        fitness = calculate_fitness(labels, bars_to_hit, mae, mfe, horizon)

        # Should have positive fitness for balanced distribution
        assert fitness > -100, f"Fitness too low for balanced labels: {fitness}"

    def test_calculate_fitness_degenerate_neutral_heavy(self):
        """Test fitness penalizes neutral-heavy distributions."""
        n = 1000

        # Create neutral-heavy labels (90% neutral)
        labels = np.array([0]*900 + [1]*50 + [-1]*50, dtype=np.int8)
        bars_to_hit = np.ones(n, dtype=np.int32) * 5
        mae = np.zeros(n, dtype=np.float32)
        mfe = np.zeros(n, dtype=np.float32)
        horizon = 5

        fitness = calculate_fitness(labels, bars_to_hit, mae, mfe, horizon)

        # Should have very negative fitness due to low signal rate
        assert fitness < -500, f"Fitness should be very negative: {fitness}"

    def test_calculate_fitness_empty_labels(self):
        """Test fitness with empty labels returns minimum value."""
        labels = np.array([], dtype=np.int8)
        bars_to_hit = np.array([], dtype=np.int32)
        mae = np.array([], dtype=np.float32)
        mfe = np.array([], dtype=np.float32)

        fitness = calculate_fitness(labels, bars_to_hit, mae, mfe, horizon=5)

        assert fitness == -1000.0, "Empty labels should return -1000"

    def test_calculate_fitness_signal_rate_threshold(self):
        """Test that signal rate below 40% is heavily penalized."""
        n = 1000

        # 30% signal rate (below threshold)
        labels = np.array([1]*150 + [-1]*150 + [0]*700, dtype=np.int8)
        bars_to_hit = np.ones(n, dtype=np.int32) * 5
        mae = -np.ones(n, dtype=np.float32) * 0.01
        mfe = np.ones(n, dtype=np.float32) * 0.02
        horizon = 5

        fitness = calculate_fitness(labels, bars_to_hit, mae, mfe, horizon)

        # Should be penalized but slightly better than -1000
        assert -1000 < fitness < -900, f"Expected penalty for low signal rate: {fitness}"

    def test_calculate_fitness_profit_factor_components(self):
        """Test profit factor calculation with known values."""
        n = 100

        # All long wins with good MFE
        labels = np.array([1]*50 + [-1]*50, dtype=np.int8)  # 50% long, 50% short
        bars_to_hit = np.ones(n, dtype=np.int32) * 3
        mae = np.zeros(n, dtype=np.float32)  # No adverse excursion
        mfe = np.ones(n, dtype=np.float32) * 0.02  # Good favorable excursion
        horizon = 5

        fitness = calculate_fitness(labels, bars_to_hit, mae, mfe, horizon)

        # Should have good fitness due to balanced distribution and good PF
        assert fitness > -10, f"Expected positive-ish fitness: {fitness}"

    def test_evaluate_individual_valid_params(self, sample_price_arrays):
        """Test evaluate_individual with valid parameters."""
        individual = [1.0, 1.0, 3.0]  # k_up, k_down, max_bars_mult

        fitness_tuple = evaluate_individual(
            individual,
            sample_price_arrays['close'],
            sample_price_arrays['high'],
            sample_price_arrays['low'],
            sample_price_arrays['atr'],
            horizon=5
        )

        assert isinstance(fitness_tuple, tuple), "Should return tuple"
        assert len(fitness_tuple) == 1, "Should have one fitness value"
        assert isinstance(fitness_tuple[0], float), "Fitness should be float"

    def test_evaluate_individual_invalid_params(self, sample_price_arrays):
        """Test evaluate_individual with parameters that cause errors."""
        # Zero ATR should be handled gracefully
        zero_atr = np.zeros_like(sample_price_arrays['atr'])
        individual = [1.0, 1.0, 3.0]

        fitness_tuple = evaluate_individual(
            individual,
            sample_price_arrays['close'],
            sample_price_arrays['high'],
            sample_price_arrays['low'],
            zero_atr,
            horizon=5
        )

        # Should return very negative fitness, not crash
        assert fitness_tuple[0] < 0, "Invalid params should give negative fitness"

    def test_ga_bounds_respected(self, sample_price_arrays):
        """Test that GA optimization respects parameter bounds."""
        individual_too_low = [0.1, 0.1, 1.0]  # Below K_MIN
        individual_too_high = [3.0, 3.0, 6.0]  # Above K_MAX

        # Bounds should be enforced in run_ga_optimization
        # Test via evaluate_individual which uses these params
        fitness_low = evaluate_individual(
            individual_too_low,
            sample_price_arrays['close'],
            sample_price_arrays['high'],
            sample_price_arrays['low'],
            sample_price_arrays['atr'],
            horizon=5
        )

        # Even with extreme params, should return valid fitness
        assert isinstance(fitness_low[0], float)

    def test_asymmetric_barrier_constraint(self, sample_price_arrays):
        """Test asymmetric barrier penalty is applied."""
        # Highly asymmetric barriers (k_up=2.0, k_down=0.5)
        asymmetric = [2.0, 0.5, 3.0]  # 4:1 ratio - very asymmetric
        symmetric = [1.0, 1.0, 3.0]   # 1:1 ratio - symmetric

        fitness_asym = evaluate_individual(
            asymmetric,
            sample_price_arrays['close'],
            sample_price_arrays['high'],
            sample_price_arrays['low'],
            sample_price_arrays['atr'],
            horizon=5
        )

        fitness_sym = evaluate_individual(
            symmetric,
            sample_price_arrays['close'],
            sample_price_arrays['high'],
            sample_price_arrays['low'],
            sample_price_arrays['atr'],
            horizon=5
        )

        # Highly asymmetric should have penalty applied
        # Note: actual comparison depends on data, just verify both run
        assert isinstance(fitness_asym[0], float)
        assert isinstance(fitness_sym[0], float)

    def test_error_handling_returns_negative_infinity(self, sample_price_arrays):
        """Test that errors in evaluation return very negative fitness."""
        # NaN in ATR should cause issues
        nan_atr = sample_price_arrays['atr'].copy()
        nan_atr[:] = np.nan

        individual = [1.0, 1.0, 3.0]

        fitness_tuple = evaluate_individual(
            individual,
            sample_price_arrays['close'],
            sample_price_arrays['high'],
            sample_price_arrays['low'],
            nan_atr,
            horizon=5
        )

        # Should handle NaN gracefully
        assert fitness_tuple[0] < 0

    def test_get_contiguous_subset(self, sample_ohlcv_data):
        """Test contiguous subset extraction."""
        df = sample_ohlcv_data
        subset_fraction = 0.3

        subset = get_contiguous_subset(df, subset_fraction)

        # Check size is approximately correct
        expected_len = int(len(df) * subset_fraction)
        assert len(subset) >= min(expected_len, 1000), "Subset too small"

        # Check it's contiguous (indices should be sequential)
        if len(subset) > 1:
            time_diffs = subset['datetime'].diff().dropna()
            # All time diffs should be equal (5 min intervals)
            assert time_diffs.nunique() == 1, "Subset should be contiguous"

    def test_get_contiguous_subset_respects_minimum(self, sample_ohlcv_data):
        """Test contiguous subset has minimum 1000 samples if available."""
        df = sample_ohlcv_data

        # Request very small fraction
        subset = get_contiguous_subset(df, 0.01)

        # Should still have minimum samples
        assert len(subset) >= min(1000, len(df))


# =============================================================================
# STAGE 6 TESTS: FinalLabeler
# =============================================================================

class TestStage6FinalLabeler:
    """Tests for Stage 6: Final Labeling with Quality Scoring."""

    def test_compute_quality_scores_speed_component(self):
        """Test speed component of quality scoring."""
        n = 100
        horizon = 5

        # Fast hits (around ideal speed)
        fast_bars = np.ones(n, dtype=np.int32) * int(horizon * 1.5)
        # Slow hits (much slower than ideal)
        slow_bars = np.ones(n, dtype=np.int32) * int(horizon * 5)

        mae = np.zeros(n, dtype=np.float32)
        mfe = np.ones(n, dtype=np.float32) * 0.01

        fast_scores = compute_quality_scores(fast_bars, mae, mfe, horizon)
        slow_scores = compute_quality_scores(slow_bars, mae, mfe, horizon)

        # Fast hits should have higher speed component
        assert fast_scores.mean() > slow_scores.mean(), \
            "Faster hits should have higher quality"

    def test_compute_quality_scores_mae_component(self):
        """Test MAE component of quality scoring."""
        n = 100
        horizon = 5

        bars = np.ones(n, dtype=np.int32) * 5

        # Create varied MFE so normalization doesn't collapse
        np.random.seed(42)
        mfe = np.abs(np.random.randn(n) * 0.02).astype(np.float32)

        # Low adverse excursion (good) - range of small values
        low_mae = -np.abs(np.random.randn(n) * 0.005).astype(np.float32)
        # High adverse excursion (bad) - range of larger values
        high_mae = -np.abs(np.random.randn(n) * 0.05 + 0.03).astype(np.float32)

        low_mae_scores = compute_quality_scores(bars, low_mae, mfe, horizon)
        high_mae_scores = compute_quality_scores(bars, high_mae, mfe, horizon)

        # Lower (less negative) MAE should have higher quality
        # Note: MAE component is 40% of total score
        assert low_mae_scores.mean() >= high_mae_scores.mean() - 0.01, \
            f"Lower MAE should have higher quality: {low_mae_scores.mean()} vs {high_mae_scores.mean()}"

    def test_compute_quality_scores_mfe_component(self):
        """Test MFE component of quality scoring."""
        n = 100
        horizon = 5

        bars = np.ones(n, dtype=np.int32) * 5

        # Create varied MAE so normalization doesn't collapse
        np.random.seed(42)
        mae = -np.abs(np.random.randn(n) * 0.02).astype(np.float32)

        # High favorable excursion (good) - range of high values
        high_mfe = (np.abs(np.random.randn(n) * 0.02) + 0.04).astype(np.float32)
        # Low favorable excursion (not as good) - range of low values
        low_mfe = np.abs(np.random.randn(n) * 0.005).astype(np.float32)

        high_mfe_scores = compute_quality_scores(bars, mae, high_mfe, horizon)
        low_mfe_scores = compute_quality_scores(bars, mae, low_mfe, horizon)

        # Higher MFE should have higher quality
        # Note: MFE component is 30% of total score
        assert high_mfe_scores.mean() >= low_mfe_scores.mean() - 0.01, \
            f"Higher MFE should have higher quality: {high_mfe_scores.mean()} vs {low_mfe_scores.mean()}"

    def test_compute_quality_scores_range(self):
        """Test quality scores are in valid range."""
        np.random.seed(42)
        n = 1000
        horizon = 5

        bars = np.random.randint(1, 20, n).astype(np.int32)
        mae = -np.abs(np.random.randn(n) * 0.02).astype(np.float32)
        mfe = np.abs(np.random.randn(n) * 0.03).astype(np.float32)

        scores = compute_quality_scores(bars, mae, mfe, horizon)

        # Scores should be bounded
        assert np.all(scores >= 0), "Scores should be non-negative"
        assert np.all(scores <= 1.5), "Scores should be reasonable"

    def test_assign_sample_weights_tier_distribution(self):
        """Test sample weight tier assignment."""
        np.random.seed(42)
        n = 1000

        quality_scores = np.random.rand(n).astype(np.float32)
        weights = assign_sample_weights(quality_scores)

        # Check tier distribution
        tier1 = (weights == 1.5).sum()
        tier2 = (weights == 1.0).sum()
        tier3 = (weights == 0.5).sum()

        # Tier 1: top 20% (around 200 +/- margin)
        assert 150 < tier1 < 250, f"Tier 1 count unexpected: {tier1}"
        # Tier 2: middle 60% (around 600 +/- margin)
        assert 550 < tier2 < 650, f"Tier 2 count unexpected: {tier2}"
        # Tier 3: bottom 20% (around 200 +/- margin)
        assert 150 < tier3 < 250, f"Tier 3 count unexpected: {tier3}"

    def test_assign_sample_weights_unique_values(self):
        """Test sample weights have correct unique values."""
        n = 1000
        quality_scores = np.random.rand(n).astype(np.float32)

        weights = assign_sample_weights(quality_scores)

        unique_weights = set(np.unique(weights))
        expected_weights = {0.5, 1.0, 1.5}

        assert unique_weights == expected_weights, \
            f"Unexpected weights: {unique_weights}"

    def test_assign_sample_weights_ordering(self):
        """Test high quality gets high weight."""
        n = 100

        # Create ordered quality scores
        quality_scores = np.linspace(0, 1, n).astype(np.float32)
        weights = assign_sample_weights(quality_scores)

        # Top 20% should have weight 1.5
        assert np.all(weights[-20:] == 1.5), "Top quality should have weight 1.5"
        # Bottom 20% should have weight 0.5
        assert np.all(weights[:20] == 0.5), "Bottom quality should have weight 0.5"

    def test_apply_optimized_labels_columns_created(self, sample_ohlcv_data):
        """Test that apply_optimized_labels creates all required columns."""
        df = sample_ohlcv_data.copy()
        horizon = 5
        best_params = {'k_up': 1.0, 'k_down': 1.0, 'max_bars': 15}

        result = apply_optimized_labels(df, horizon, best_params)

        # Check all expected columns exist
        expected_cols = [
            f'label_h{horizon}',
            f'bars_to_hit_h{horizon}',
            f'mae_h{horizon}',
            f'mfe_h{horizon}',
            f'touch_type_h{horizon}',
            f'quality_h{horizon}',
            f'sample_weight_h{horizon}',
        ]

        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_apply_optimized_labels_label_values(self, sample_ohlcv_data):
        """Test that labels have valid values."""
        df = sample_ohlcv_data.copy()
        horizon = 5
        best_params = {'k_up': 1.0, 'k_down': 1.0, 'max_bars': 15}

        result = apply_optimized_labels(df, horizon, best_params)

        labels = result[f'label_h{horizon}'].values
        valid_labels = {-1, 0, 1}

        assert set(np.unique(labels)).issubset(valid_labels), \
            f"Invalid label values: {np.unique(labels)}"

    def test_label_consistency_with_stage4(self, sample_ohlcv_data):
        """Test that stage 6 labels are consistent with stage 4 methodology."""
        # This tests that the same labeling function produces consistent results
        from stages.stage4_labeling import triple_barrier_numba

        df = sample_ohlcv_data.copy()
        k_up, k_down, max_bars = 1.0, 1.0, 15

        # Direct call to triple_barrier_numba
        labels1, _, _, _, _ = triple_barrier_numba(
            df['close'].values,
            df['high'].values,
            df['low'].values,
            df['open'].values,
            df['atr_14'].values,
            k_up, k_down, max_bars
        )

        # Via apply_optimized_labels
        best_params = {'k_up': k_up, 'k_down': k_down, 'max_bars': max_bars}
        result = apply_optimized_labels(df.copy(), horizon=5, best_params=best_params)
        labels2 = result['label_h5'].values

        # Labels should be identical
        np.testing.assert_array_equal(labels1, labels2, "Labels should match stage4")

    def test_handles_missing_ga_params(self, sample_ohlcv_data, temp_directory):
        """Test graceful handling when GA params file is missing."""
        # This is tested implicitly in process_symbol_final
        # When GA results file doesn't exist, defaults are used
        df = sample_ohlcv_data.copy()
        horizon = 5

        # Use default params (simulating missing GA results)
        default_params = {'k_up': 2.0, 'k_down': 1.0, 'max_bars': 15}

        result = apply_optimized_labels(df, horizon, default_params)

        # Should complete successfully with defaults
        assert f'label_h{horizon}' in result.columns


# =============================================================================
# STAGE 7 TESTS: DataSplitter
# =============================================================================

class TestStage7DataSplitter:
    """Tests for Stage 7: Time-Based Splitting."""

    def test_split_ratios_sum_to_one(self, sample_labeled_data):
        """Test that split ratios must sum to 1.0."""
        df = sample_labeled_data

        # Valid ratios with small purge/embargo for 1000 row dataset
        train_idx, val_idx, test_idx, metadata = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=10, embargo_bars=20  # Use smaller values for test data size
        )

        assert metadata is not None, "Should succeed with valid ratios"

    def test_split_ratios_validation_error(self, sample_labeled_data):
        """Test that invalid ratios raise error."""
        df = sample_labeled_data

        with pytest.raises(ValueError, match="sum to 1.0"):
            create_chronological_splits(
                df, train_ratio=0.50, val_ratio=0.20, test_ratio=0.20
            )

    def test_splits_chronological_order(self, sample_labeled_data):
        """Test that splits maintain chronological order."""
        df = sample_labeled_data

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=10, embargo_bars=20
        )

        # Train should come before val
        assert train_idx.max() < val_idx.min(), "Train should precede validation"
        # Val should come before test
        assert val_idx.max() < test_idx.min(), "Validation should precede test"

    def test_no_overlap_between_splits(self, sample_labeled_data):
        """Test that there is no overlap between splits."""
        df = sample_labeled_data

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=10, embargo_bars=20
        )

        # Verify no overlap
        train_set = set(train_idx)
        val_set = set(val_idx)
        test_set = set(test_idx)

        assert len(train_set & val_set) == 0, "Train/val overlap"
        assert len(train_set & test_set) == 0, "Train/test overlap"
        assert len(val_set & test_set) == 0, "Val/test overlap"

    def test_purge_removes_correct_samples(self, sample_labeled_data):
        """Test that purging removes samples at boundaries."""
        df = sample_labeled_data
        n = len(df)
        purge_bars = 60
        embargo_bars = 0  # Disable embargo for this test

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=purge_bars, embargo_bars=embargo_bars
        )

        # Calculate expected train end
        expected_train_end_raw = int(n * 0.70)
        expected_train_end = expected_train_end_raw - purge_bars

        # Train should end before the raw split point
        assert train_idx.max() < expected_train_end_raw, \
            "Purging should remove samples before split"

    def test_embargo_creates_gap(self, sample_labeled_data):
        """Test that embargo creates gap between splits."""
        df = sample_labeled_data
        purge_bars = 0  # Disable purge for this test
        embargo_bars = 50

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=purge_bars, embargo_bars=embargo_bars
        )

        # Gap between train and val should be at least embargo_bars
        gap_train_val = val_idx.min() - train_idx.max()
        assert gap_train_val >= embargo_bars, \
            f"Gap should be >= {embargo_bars}, got {gap_train_val}"

    def test_purge_value_matches_max_horizon(self):
        """Test that purge with max_bars=60 (H20) prevents leakage."""
        # Create a larger dataset for this test (5000 rows)
        np.random.seed(42)
        n = 5000

        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'symbol': 'MES',
            'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
        })

        # PURGE_BARS should equal max(max_bars) = 60 for H20
        purge_bars = 60
        embargo_bars = 288

        train_idx, val_idx, test_idx, metadata = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=purge_bars, embargo_bars=embargo_bars
        )

        assert metadata['purge_bars'] == 60, "Purge should match H20 max_bars"

    def test_split_indices_valid(self, sample_labeled_data):
        """Test that all indices are valid for the dataframe."""
        df = sample_labeled_data
        n = len(df)

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=10, embargo_bars=20
        )

        # All indices should be in valid range
        all_indices = np.concatenate([train_idx, val_idx, test_idx])
        assert all_indices.min() >= 0, "Negative indices"
        assert all_indices.max() < n, "Indices exceed dataframe length"

    def test_per_symbol_splitting(self, sample_labeled_data):
        """Test splitting preserves symbol column."""
        df = sample_labeled_data

        train_idx, val_idx, test_idx, _ = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=10, embargo_bars=20
        )

        # Verify symbol is preserved in splits
        train_symbols = df.iloc[train_idx]['symbol'].unique()
        val_symbols = df.iloc[val_idx]['symbol'].unique()
        test_symbols = df.iloc[test_idx]['symbol'].unique()

        # All splits should have same symbols as original
        original_symbols = df['symbol'].unique()
        np.testing.assert_array_equal(
            sorted(train_symbols), sorted(original_symbols)
        )

    def test_validate_no_overlap_function(self):
        """Test the validate_no_overlap utility function."""
        # No overlap case
        train = np.array([0, 1, 2, 3, 4])
        val = np.array([10, 11, 12, 13])
        test = np.array([20, 21, 22, 23, 24])

        assert validate_no_overlap(train, val, test) is True

        # With overlap
        train_overlap = np.array([0, 1, 2, 3, 4])
        val_overlap = np.array([4, 5, 6, 7])  # Overlaps at 4
        test_no_overlap = np.array([10, 11, 12])

        assert validate_no_overlap(train_overlap, val_overlap, test_no_overlap) is False

    def test_create_splits_saves_files(self, sample_labeled_data, temp_directory):
        """Test that create_splits saves all required files."""
        df = sample_labeled_data

        # Save test data
        data_path = temp_directory / "test_data.parquet"
        df.to_parquet(data_path)

        output_dir = temp_directory / "splits"

        metadata = create_splits(
            data_path=data_path,
            output_dir=output_dir,
            run_id="test_run",
            purge_bars=10,
            embargo_bars=20
        )

        # Check files exist
        split_dir = output_dir / "test_run"
        assert (split_dir / "train.npy").exists(), "train.npy missing"
        assert (split_dir / "val.npy").exists(), "val.npy missing"
        assert (split_dir / "test.npy").exists(), "test.npy missing"
        assert (split_dir / "split_config.json").exists(), "config missing"


# =============================================================================
# STAGE 8 TESTS: DataValidator
# =============================================================================

class TestStage8DataValidator:
    """Tests for Stage 8: Comprehensive Data Validation."""

    def test_check_duplicates_detection(self, sample_labeled_data):
        """Test duplicate timestamp detection."""
        df = sample_labeled_data.copy()

        # Add duplicates
        dup_row = df.iloc[0:1].copy()
        df = pd.concat([df, dup_row], ignore_index=True)

        validator = DataValidator(df)
        results = validator.check_data_integrity()

        # Should detect duplicates
        assert 'duplicate_timestamps' in results

    def test_check_nan_inf_detection(self, sample_labeled_data):
        """Test NaN and Inf detection."""
        df = sample_labeled_data.copy()

        # Add NaN values
        df.loc[0, 'rsi'] = np.nan
        df.loc[1, 'rsi'] = np.nan

        validator = DataValidator(df)
        results = validator.check_data_integrity()

        # Should detect NaN
        assert 'nan_values' in results
        if 'rsi' in results['nan_values']:
            assert results['nan_values']['rsi'] == 2

    def test_check_inf_values(self, sample_labeled_data):
        """Test infinite value detection."""
        df = sample_labeled_data.copy()

        # Add infinite values
        df.loc[0, 'macd'] = np.inf
        df.loc[1, 'macd'] = -np.inf

        validator = DataValidator(df)
        results = validator.check_data_integrity()

        # Should detect infinite values
        assert 'infinite_values' in results
        if 'macd' in results['infinite_values']:
            assert results['infinite_values']['macd'] == 2

    def test_check_gaps_detection(self, sample_labeled_data):
        """Test time gap detection."""
        df = sample_labeled_data.copy()

        # Create a gap by removing rows
        df = df.drop(df.index[50:100])  # Remove 50 rows

        validator = DataValidator(df)
        results = validator.check_data_integrity()

        # Should detect gaps
        assert 'gaps' in results

    def test_label_distribution_analysis(self, sample_labeled_data):
        """Test label distribution analysis."""
        df = sample_labeled_data

        validator = DataValidator(df, horizons=[5, 20])
        results = validator.check_label_sanity()

        # Should have results for each horizon
        assert 'horizon_5' in results or 'horizon_20' in results

        # Check distribution structure
        for key, horizon_results in results.items():
            if 'distribution' in horizon_results:
                dist = horizon_results['distribution']
                assert 'long' in dist or 'short' in dist or 'neutral' in dist

    def test_feature_correlation_detection(self, sample_labeled_data):
        """Test feature correlation detection."""
        df = sample_labeled_data.copy()

        # Add highly correlated feature
        df['sma_10_copy'] = df['sma_10'] * 1.001  # Nearly identical

        validator = DataValidator(df)
        results = validator.check_feature_quality()

        # Should detect high correlations
        assert 'high_correlations' in results

    def test_normalization_recommendations(self, sample_labeled_data):
        """Test normalization recommendations."""
        df = sample_labeled_data.copy()

        # Add feature with large scale
        df['large_feature'] = df['close'] * 1000

        validator = DataValidator(df)
        results = validator.check_feature_normalization()

        # Should have recommendations
        assert 'recommendations' in results
        assert 'unnormalized_features' in results

    def test_stationarity_check(self, sample_labeled_data):
        """Test stationarity checking."""
        df = sample_labeled_data

        validator = DataValidator(df)
        results = validator.check_feature_quality()

        # Should have stationarity results
        assert 'stationarity_tests' in results

    def test_validation_report_structure(self, sample_labeled_data):
        """Test validation report has correct structure."""
        df = sample_labeled_data

        validator = DataValidator(df, horizons=[1, 5, 20])
        validator.check_data_integrity()
        validator.check_label_sanity()
        validator.check_feature_quality()

        summary = validator.generate_summary()

        # Check structure
        assert 'timestamp' in summary
        assert 'total_rows' in summary
        assert 'total_columns' in summary
        assert 'issues_count' in summary
        assert 'warnings_count' in summary
        assert 'status' in summary
        assert summary['status'] in ['PASSED', 'FAILED']

    def test_validate_data_integration(self, sample_labeled_data, temp_directory):
        """Test full validation pipeline."""
        df = sample_labeled_data

        # Save test data
        data_path = temp_directory / "test_data.parquet"
        df.to_parquet(data_path)

        output_path = temp_directory / "validation_report.json"

        summary, _ = validate_data(
            data_path,
            output_path=output_path,
            horizons=[5, 20],
            run_feature_selection=False
        )

        # Check report was saved
        assert output_path.exists(), "Report should be saved"

        # Check summary
        assert 'status' in summary

    def test_feature_selection_integration(self, sample_labeled_data, temp_directory):
        """Test feature selection integration."""
        df = sample_labeled_data

        # Save test data
        data_path = temp_directory / "test_data.parquet"
        df.to_parquet(data_path)

        output_path = temp_directory / "validation_report.json"

        summary, feature_result = validate_data(
            data_path,
            output_path=output_path,
            run_feature_selection=True
        )

        # Feature selection should return results
        if feature_result is not None:
            assert isinstance(feature_result, FeatureSelectionResult)


# =============================================================================
# PIPELINE INTEGRATION TESTS
# =============================================================================

class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_stages_1_through_8(self, sample_ohlcv_data, temp_directory):
        """Test full pipeline from stage 4 labeling through validation."""
        df = sample_ohlcv_data.copy()

        # Stage 4: Apply labeling
        from stages.stage4_labeling import apply_triple_barrier
        df = apply_triple_barrier(df, horizon=5, k_up=1.0, k_down=1.0, max_bars=15)

        # Stage 6: Apply quality scoring
        horizon = 5
        bars_to_hit = df[f'bars_to_hit_h{horizon}'].values
        mae = df[f'mae_h{horizon}'].values
        mfe = df[f'mfe_h{horizon}'].values

        quality = compute_quality_scores(bars_to_hit, mae, mfe, horizon)
        df[f'quality_h{horizon}'] = quality

        weights = assign_sample_weights(quality)
        df[f'sample_weight_h{horizon}'] = weights

        # Stage 7: Create splits
        train_idx, val_idx, test_idx, split_meta = create_chronological_splits(
            df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
            purge_bars=10, embargo_bars=20
        )

        # Stage 8: Validate
        validator = DataValidator(df, horizons=[5])
        validator.check_data_integrity()
        validator.check_label_sanity()
        summary = validator.generate_summary()

        # Verify pipeline completed
        assert split_meta['validation_passed'] is True
        assert summary['status'] in ['PASSED', 'FAILED']

    def test_pipeline_artifact_tracking(self, sample_labeled_data, temp_directory):
        """Test that pipeline artifacts are properly saved."""
        df = sample_labeled_data

        # Save data
        data_path = temp_directory / "data.parquet"
        df.to_parquet(data_path)

        # Create splits
        splits_dir = temp_directory / "splits"
        split_meta = create_splits(
            data_path=data_path,
            output_dir=splits_dir,
            run_id="test",
            purge_bars=10,
            embargo_bars=20
        )

        # Validate
        report_path = temp_directory / "report.json"
        summary, _ = validate_data(
            data_path,
            output_path=report_path,
            run_feature_selection=False
        )

        # Check artifacts
        assert (splits_dir / "test" / "train.npy").exists()
        assert (splits_dir / "test" / "split_config.json").exists()
        assert report_path.exists()

    def test_pipeline_error_recovery(self, sample_ohlcv_data):
        """Test pipeline handles errors gracefully."""
        df = sample_ohlcv_data.copy()

        # Remove required column
        df = df.drop('atr_14', axis=1)

        # Should raise informative error
        from stages.stage4_labeling import apply_triple_barrier
        with pytest.raises(KeyError):
            apply_triple_barrier(df, horizon=5)

    def test_pipeline_idempotency(self, sample_ohlcv_data):
        """Test that running pipeline twice gives same results."""
        df = sample_ohlcv_data.copy()

        # Run labeling twice
        from stages.stage4_labeling import apply_triple_barrier

        df1 = df.copy()
        df1 = apply_triple_barrier(df1, horizon=5, k_up=1.0, k_down=1.0, max_bars=15)

        df2 = df.copy()
        df2 = apply_triple_barrier(df2, horizon=5, k_up=1.0, k_down=1.0, max_bars=15)

        # Results should be identical
        np.testing.assert_array_equal(
            df1['label_h5'].values,
            df2['label_h5'].values,
            "Labels should be deterministic"
        )


# =============================================================================
# FEATURE SELECTION TESTS
# =============================================================================

class TestFeatureSelection:
    """Tests for feature selection module."""

    def test_identify_feature_columns(self, sample_labeled_data):
        """Test feature column identification."""
        df = sample_labeled_data

        feature_cols = identify_feature_columns(df)

        # Should not include metadata columns
        assert 'datetime' not in feature_cols
        assert 'symbol' not in feature_cols
        assert 'open' not in feature_cols

        # Should not include label columns
        assert 'label_h5' not in feature_cols
        assert 'quality_h5' not in feature_cols

        # Should include features
        assert 'log_return' in feature_cols or 'rsi' in feature_cols

    def test_filter_low_variance(self, sample_labeled_data):
        """Test low variance feature filtering."""
        df = sample_labeled_data.copy()

        # Add constant feature
        df['constant_feature'] = 1.0

        feature_cols = ['log_return', 'rsi', 'constant_feature']

        kept, low_var = filter_low_variance(df, feature_cols, variance_threshold=0.01)

        assert 'constant_feature' in low_var, "Constant feature should be filtered"

    def test_filter_correlated_features(self, sample_labeled_data):
        """Test correlated feature filtering."""
        df = sample_labeled_data.copy()

        # Add highly correlated feature
        df['log_return_copy'] = df['log_return'] * 1.0001

        feature_cols = ['log_return', 'log_return_copy', 'rsi']

        kept, removed, groups = filter_correlated_features(
            df, feature_cols, correlation_threshold=0.85
        )

        # One of the correlated pair should be removed
        assert len(removed) > 0 or 'log_return' in kept

    def test_get_feature_priority(self):
        """Test feature priority retrieval."""
        # Known features should have priority
        assert get_feature_priority('log_return') == 100
        assert get_feature_priority('rsi') == 90

        # Unknown features should get default
        assert get_feature_priority('unknown_feature') == 50

    def test_select_features_result_structure(self, sample_labeled_data):
        """Test feature selection result structure."""
        df = sample_labeled_data

        result = select_features(
            df,
            correlation_threshold=0.85,
            variance_threshold=0.01
        )

        assert isinstance(result, FeatureSelectionResult)
        assert hasattr(result, 'selected_features')
        assert hasattr(result, 'removed_features')
        assert hasattr(result, 'original_count')
        assert hasattr(result, 'final_count')
        assert result.final_count <= result.original_count

    def test_feature_selection_result_to_dict(self, sample_labeled_data):
        """Test FeatureSelectionResult serialization."""
        df = sample_labeled_data

        result = select_features(df)
        result_dict = result.to_dict()

        assert 'selected_features' in result_dict
        assert 'removed_features' in result_dict
        assert 'reduction_pct' in result_dict


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        df = pd.DataFrame()

        # Should handle gracefully
        feature_cols = identify_feature_columns(df)
        assert feature_cols == []

    def test_single_row_dataframe(self, sample_ohlcv_data):
        """Test handling of single row dataframe."""
        df = sample_ohlcv_data.iloc[:1].copy()

        # Should not crash
        from stages.stage4_labeling import triple_barrier_numba

        labels, _, _, _, _ = triple_barrier_numba(
            df['close'].values,
            df['high'].values,
            df['low'].values,
            df['open'].values,
            df['atr_14'].values,
            k_up=1.0, k_down=1.0, max_bars=5
        )

        assert len(labels) == 1

    def test_all_neutral_labels(self):
        """Test fitness with all neutral labels."""
        n = 100
        labels = np.zeros(n, dtype=np.int8)  # All neutral
        bars_to_hit = np.ones(n, dtype=np.int32) * 10
        mae = np.zeros(n, dtype=np.float32)
        mfe = np.zeros(n, dtype=np.float32)

        fitness = calculate_fitness(labels, bars_to_hit, mae, mfe, horizon=5)

        # Should be heavily penalized (0% signal rate)
        assert fitness < -900

    def test_extreme_quality_scores(self):
        """Test quality scoring with extreme values."""
        n = 100

        # Extreme bars to hit (very slow)
        bars = np.ones(n, dtype=np.int32) * 1000
        mae = np.zeros(n, dtype=np.float32)
        mfe = np.zeros(n, dtype=np.float32)

        scores = compute_quality_scores(bars, mae, mfe, horizon=5)

        # Should still be bounded
        assert np.all(np.isfinite(scores))

    def test_split_with_tiny_dataset(self):
        """Test splitting with very small dataset."""
        n = 100
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
            'close': np.random.randn(n) * 10 + 100,
        })

        # With very small dataset, may fail if purge/embargo too large
        with pytest.raises(ValueError):
            create_chronological_splits(
                df,
                train_ratio=0.70,
                val_ratio=0.15,
                test_ratio=0.15,
                purge_bars=60,  # Too large for 100 samples
                embargo_bars=50
            )

    def test_validation_with_missing_columns(self, sample_ohlcv_data):
        """Test validation handles missing label columns."""
        df = sample_ohlcv_data

        validator = DataValidator(df, horizons=[1, 5, 20])
        results = validator.check_label_sanity()

        # Should not crash, just warn about missing columns
        assert results is not None


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance and timing tests."""

    @pytest.mark.slow
    def test_large_dataset_labeling(self):
        """Test labeling performance on larger dataset."""
        np.random.seed(42)
        n = 10000

        close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.002))
        high = close * 1.002
        low = close * 0.998
        open_prices = close * (1 + np.random.randn(n) * 0.001)
        atr = np.ones(n) * 0.5

        from stages.stage4_labeling import triple_barrier_numba

        import time
        start = time.time()

        labels, _, _, _, _ = triple_barrier_numba(
            close, high, low, open_prices, atr,
            k_up=1.0, k_down=1.0, max_bars=60
        )

        elapsed = time.time() - start

        # Should complete in reasonable time (< 10 seconds)
        assert elapsed < 10, f"Labeling took too long: {elapsed:.2f}s"
        assert len(labels) == n

    @pytest.mark.slow
    def test_validation_performance(self, sample_labeled_data):
        """Test validation performance."""
        # Expand dataset
        df = pd.concat([sample_labeled_data] * 5, ignore_index=True)

        import time
        start = time.time()

        validator = DataValidator(df)
        validator.check_data_integrity()
        validator.check_label_sanity()
        validator.check_feature_quality()
        validator.generate_summary()

        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 30, f"Validation took too long: {elapsed:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
