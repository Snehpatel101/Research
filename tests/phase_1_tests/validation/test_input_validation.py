"""
Tests for parameter validation across the pipeline using modern Python 3.12+ patterns.
Verifies that validation catches invalid inputs early at boundaries.
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Final

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Test configuration matching pipeline defaults
TEST_SYMBOLS: Final[list[str]] = ['MES', 'MGC']
TEST_HORIZONS: Final[list[int]] = [5, 20]  # H1 excluded (transaction costs > profit)
PURGE_BARS: Final[int] = 60  # = max_bars for H20 (prevents label leakage)
EMBARGO_BARS: Final[int] = 288  # ~1 day for 5-min data


class TestLabelingValidation:
    """Tests for labeling parameter validation at boundaries."""

    def test_negative_horizon_raises(self) -> None:
        """Test that negative horizon raises ValueError."""
        from stages.stage4_labeling import apply_triple_barrier

        df: pd.DataFrame = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=100, freq='5min'),
            'open': [100]*100,
            'high': [101]*100,
            'low': [99]*100,
            'close': [100.5]*100,
            'atr_14': [1.0]*100
        })

        with pytest.raises(ValueError, match="positive"):
            apply_triple_barrier(df, horizon=-5)

    def test_zero_horizon_raises(self):
        """Test that zero horizon raises ValueError."""
        from stages.stage4_labeling import apply_triple_barrier

        df = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=100, freq='5min'),
            'open': [100]*100,
            'high': [101]*100,
            'low': [99]*100,
            'close': [100.5]*100,
            'atr_14': [1.0]*100
        })

        with pytest.raises(ValueError, match="positive"):
            apply_triple_barrier(df, horizon=0)

    def test_negative_k_up_raises(self):
        """Test that negative k_up raises ValueError."""
        from stages.stage4_labeling import apply_triple_barrier

        df = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=100, freq='5min'),
            'open': [100]*100,
            'high': [101]*100,
            'low': [99]*100,
            'close': [100.5]*100,
            'atr_14': [1.0]*100
        })

        with pytest.raises(ValueError, match="positive"):
            apply_triple_barrier(df, horizon=5, k_up=-1.0)

    def test_zero_k_up_raises(self):
        """Test that zero k_up raises ValueError."""
        from stages.stage4_labeling import apply_triple_barrier

        df = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=100, freq='5min'),
            'open': [100]*100,
            'high': [101]*100,
            'low': [99]*100,
            'close': [100.5]*100,
            'atr_14': [1.0]*100
        })

        with pytest.raises(ValueError, match="positive"):
            apply_triple_barrier(df, horizon=5, k_up=0.0)

    def test_negative_k_down_raises(self):
        """Test that negative k_down raises ValueError."""
        from stages.stage4_labeling import apply_triple_barrier

        df = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=100, freq='5min'),
            'open': [100]*100,
            'high': [101]*100,
            'low': [99]*100,
            'close': [100.5]*100,
            'atr_14': [1.0]*100
        })

        with pytest.raises(ValueError, match="positive"):
            apply_triple_barrier(df, horizon=5, k_down=-1.0)

    def test_negative_max_bars_raises(self):
        """Test that negative max_bars raises ValueError."""
        from stages.stage4_labeling import apply_triple_barrier

        df = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=100, freq='5min'),
            'open': [100]*100,
            'high': [101]*100,
            'low': [99]*100,
            'close': [100.5]*100,
            'atr_14': [1.0]*100
        })

        with pytest.raises(ValueError, match="positive"):
            apply_triple_barrier(df, horizon=5, max_bars=-10)

    def test_missing_atr_column_raises(self):
        """Test that missing ATR column raises KeyError."""
        from stages.stage4_labeling import apply_triple_barrier

        df = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=100, freq='5min'),
            'open': [100]*100,
            'high': [101]*100,
            'low': [99]*100,
            'close': [100.5]*100
            # No atr_14 column
        })

        with pytest.raises(KeyError, match="atr"):
            apply_triple_barrier(df, horizon=5)

    def test_missing_ohlc_columns_raises(self):
        """Test that missing OHLC columns raises KeyError."""
        from stages.stage4_labeling import apply_triple_barrier

        df = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=100, freq='5min'),
            'close': [100.5]*100,  # Missing open, high, low
            'atr_14': [1.0]*100
        })

        with pytest.raises(KeyError, match="open|high|low"):
            apply_triple_barrier(df, horizon=5)

    def test_empty_dataframe_raises(self):
        """Test that empty DataFrame raises ValueError."""
        from stages.stage4_labeling import apply_triple_barrier

        df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty"):
            apply_triple_barrier(df, horizon=5)

    def test_valid_parameters_succeed(self) -> None:
        """Test that valid parameters do not raise."""
        from stages.stage4_labeling import apply_triple_barrier

        df: pd.DataFrame = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=100, freq='5min'),
            'open': [100]*100,
            'high': [101]*100,
            'low': [99]*100,
            'close': [100.5]*100,
            'atr_14': [1.0]*100
        })

        # Should not raise - using active horizons from TEST_HORIZONS
        result: pd.DataFrame = apply_triple_barrier(
            df, horizon=5, k_up=2.0, k_down=1.0, max_bars=15
        )
        assert 'label_h5' in result.columns


class TestConfigValidation:
    """Tests for config validation at module level."""

    def test_purge_bars_vs_max_bars(self) -> None:
        """Test that purge_bars >= max_bars is validated."""
        from config import PURGE_BARS, BARRIER_PARAMS, BARRIER_PARAMS_DEFAULT

        # Get max_bars across all configs
        max_max_bars: int = 0

        # Check symbol-specific barrier params
        for symbol, horizons in BARRIER_PARAMS.items():
            for horizon, params in horizons.items():
                mb: int = params.get('max_bars', 0)
                if mb > max_max_bars:
                    max_max_bars = mb

        # Check default barrier params
        for horizon, params in BARRIER_PARAMS_DEFAULT.items():
            mb: int = params.get('max_bars', 0)
            if mb > max_max_bars:
                max_max_bars = mb

        # PURGE_BARS should be >= max_max_bars (critical for preventing label leakage)
        assert PURGE_BARS >= max_max_bars, \
            f"PURGE_BARS ({PURGE_BARS}) < max_bars ({max_max_bars}) - LABEL LEAKAGE RISK!"

    def test_split_ratios_sum_to_one(self) -> None:
        """Test that split ratios sum to 1.0."""
        from config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO

        total: float = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
        assert abs(total - 1.0) < 0.001, f"Ratios sum to {total}, not 1.0"

    def test_validate_config_runs_without_error(self):
        """Test that validate_config succeeds for current config."""
        from config import validate_config

        # Should not raise - already validated at import
        # But we can call it explicitly to ensure it's working
        validate_config()

    def test_barrier_params_have_required_keys(self):
        """Test that all barrier params have required keys."""
        from config import BARRIER_PARAMS, BARRIER_PARAMS_DEFAULT

        required_keys = {'k_up', 'k_down', 'max_bars'}

        for symbol, horizons in BARRIER_PARAMS.items():
            for horizon, params in horizons.items():
                missing = required_keys - set(params.keys())
                assert not missing, f"BARRIER_PARAMS['{symbol}'][{horizon}] missing: {missing}"

        for horizon, params in BARRIER_PARAMS_DEFAULT.items():
            missing = required_keys - set(params.keys())
            assert not missing, f"BARRIER_PARAMS_DEFAULT[{horizon}] missing: {missing}"

    def test_transaction_costs_non_negative(self):
        """Test that transaction costs are non-negative."""
        from config import TRANSACTION_COSTS

        for symbol, cost in TRANSACTION_COSTS.items():
            assert cost >= 0, f"TRANSACTION_COSTS['{symbol}'] is negative: {cost}"

    def test_tick_values_positive(self):
        """Test that tick values are positive."""
        from config import TICK_VALUES

        for symbol, value in TICK_VALUES.items():
            assert value > 0, f"TICK_VALUES['{symbol}'] is not positive: {value}"


class TestSplitsValidation:
    """Tests for splits validation."""

    def test_empty_dataframe_raises(self):
        """Test that empty DataFrame raises ValueError."""
        from stages.stage7_splits import create_chronological_splits

        df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty"):
            create_chronological_splits(df)

    def test_train_set_eliminated_by_purge_raises(self):
        """Test validation catches train set eliminated by purging."""
        from stages.stage7_splits import create_chronological_splits

        # Create small dataset where purge would eliminate train set
        df = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=100, freq='5min'),
            'close': range(100)
        })

        # With train_ratio=0.2 (20 samples) and purge_bars=30, train_end becomes negative
        with pytest.raises(ValueError, match="eliminated|insufficient|small"):
            create_chronological_splits(
                df,
                train_ratio=0.2,  # 20 samples
                val_ratio=0.4,
                test_ratio=0.4,
                purge_bars=30,  # More than train
                embargo_bars=10
            )

    def test_ratios_must_be_positive(self):
        """Test that negative ratios are rejected."""
        from stages.stage7_splits import create_chronological_splits

        df = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=1000, freq='5min'),
            'close': range(1000)
        })

        with pytest.raises(ValueError, match="positive"):
            create_chronological_splits(
                df,
                train_ratio=-0.1,
                val_ratio=0.5,
                test_ratio=0.6
            )

    def test_val_ratio_must_be_positive(self):
        """Test that negative val_ratio is rejected."""
        from stages.stage7_splits import create_chronological_splits

        df = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=1000, freq='5min'),
            'close': range(1000)
        })

        with pytest.raises(ValueError, match="positive"):
            create_chronological_splits(
                df,
                train_ratio=0.7,
                val_ratio=-0.1,
                test_ratio=0.4
            )

    def test_ratios_must_sum_to_one(self):
        """Test that ratios must sum to 1.0."""
        from stages.stage7_splits import create_chronological_splits

        df = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=1000, freq='5min'),
            'close': range(1000)
        })

        with pytest.raises(ValueError, match="sum"):
            create_chronological_splits(
                df,
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3  # Sum = 1.1
            )

    def test_missing_datetime_column_raises(self):
        """Test that missing datetime column raises KeyError."""
        from stages.stage7_splits import create_chronological_splits

        df = pd.DataFrame({
            'close': range(1000)
            # No datetime column
        })

        with pytest.raises(KeyError, match="datetime"):
            create_chronological_splits(df)

    def test_negative_purge_bars_raises(self):
        """Test that negative purge_bars raises ValueError."""
        from stages.stage7_splits import create_chronological_splits

        df = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=1000, freq='5min'),
            'close': range(1000)
        })

        with pytest.raises(ValueError, match="non-negative|positive"):
            create_chronological_splits(
                df,
                purge_bars=-10
            )

    def test_negative_embargo_bars_raises(self):
        """Test that negative embargo_bars raises ValueError."""
        from stages.stage7_splits import create_chronological_splits

        df = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=1000, freq='5min'),
            'close': range(1000)
        })

        with pytest.raises(ValueError, match="non-negative|positive"):
            create_chronological_splits(
                df,
                embargo_bars=-10
            )

    def test_valid_parameters_succeed(self):
        """Test that valid parameters produce correct splits."""
        from stages.stage7_splits import create_chronological_splits

        df = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=1000, freq='5min'),
            'close': range(1000)
        })

        train_idx, val_idx, test_idx, metadata = create_chronological_splits(
            df,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            purge_bars=10,
            embargo_bars=20
        )

        # Verify no overlap
        assert len(set(train_idx) & set(val_idx)) == 0
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(set(val_idx) & set(test_idx)) == 0

        # Verify chronological order
        assert train_idx.max() < val_idx.min()
        assert val_idx.max() < test_idx.min()


class TestPerSymbolValidation:
    """Tests for per-symbol validation in splits."""

    def test_symbol_underrepresentation_raises(self):
        """Test that underrepresented symbol raises error."""
        from stages.stage7_splits import validate_per_symbol_distribution

        # Create imbalanced data
        df = pd.DataFrame({
            'symbol': ['MES']*90 + ['MGC']*10,  # 90% MES, 10% MGC
            'close': range(100)
        })

        train_idx = np.arange(0, 70)
        val_idx = np.arange(70, 85)
        test_idx = np.arange(85, 100)

        with pytest.raises(ValueError, match="underrepresented"):
            validate_per_symbol_distribution(
                df, train_idx, val_idx, test_idx,
                min_symbol_pct=20.0  # MGC is only ~14% in each split
            )

    def test_balanced_distribution_succeeds(self):
        """Test that balanced symbol distribution passes validation."""
        from stages.stage7_splits import validate_per_symbol_distribution

        # Create balanced data
        symbols = ['MES', 'MGC'] * 50
        df = pd.DataFrame({
            'symbol': symbols,
            'close': range(100)
        })

        train_idx = np.arange(0, 70)
        val_idx = np.arange(70, 85)
        test_idx = np.arange(85, 100)

        # Should not raise - each symbol is ~50%
        validate_per_symbol_distribution(
            df, train_idx, val_idx, test_idx,
            min_symbol_pct=20.0
        )


class TestNoOverlapValidation:
    """Tests for split overlap validation."""

    def test_validate_no_overlap_passes_for_disjoint(self):
        """Test that validate_no_overlap passes for disjoint sets."""
        from stages.stage7_splits import validate_no_overlap

        train_idx = np.array([0, 1, 2, 3, 4])
        val_idx = np.array([10, 11, 12])
        test_idx = np.array([20, 21, 22, 23])

        assert validate_no_overlap(train_idx, val_idx, test_idx) is True

    def test_validate_no_overlap_fails_for_train_val_overlap(self):
        """Test that validate_no_overlap fails when train and val overlap."""
        from stages.stage7_splits import validate_no_overlap

        train_idx = np.array([0, 1, 2, 3, 4])
        val_idx = np.array([4, 5, 6])  # 4 overlaps with train
        test_idx = np.array([10, 11, 12])

        assert validate_no_overlap(train_idx, val_idx, test_idx) is False

    def test_validate_no_overlap_fails_for_train_test_overlap(self):
        """Test that validate_no_overlap fails when train and test overlap."""
        from stages.stage7_splits import validate_no_overlap

        train_idx = np.array([0, 1, 2, 3, 4])
        val_idx = np.array([10, 11, 12])
        test_idx = np.array([2, 20, 21])  # 2 overlaps with train

        assert validate_no_overlap(train_idx, val_idx, test_idx) is False

    def test_validate_no_overlap_fails_for_val_test_overlap(self):
        """Test that validate_no_overlap fails when val and test overlap."""
        from stages.stage7_splits import validate_no_overlap

        train_idx = np.array([0, 1, 2, 3, 4])
        val_idx = np.array([10, 11, 12])
        test_idx = np.array([12, 20, 21])  # 12 overlaps with val

        assert validate_no_overlap(train_idx, val_idx, test_idx) is False


class TestRandomSeedReproducibility:
    """Tests for random seed reproducibility."""

    def test_global_seed_function(self):
        """Test that set_global_seeds works."""
        from config import set_global_seeds
        import random
        import numpy as np

        # Set seeds
        set_global_seeds(42)

        # Generate some random numbers
        r1 = random.random()
        n1 = np.random.rand()

        # Reset seeds
        set_global_seeds(42)

        # Should get same numbers
        r2 = random.random()
        n2 = np.random.rand()

        assert r1 == r2, f"Python random not reproducible: {r1} != {r2}"
        assert n1 == n2, f"NumPy random not reproducible: {n1} != {n2}"

    def test_numpy_reproducibility_with_different_seeds(self):
        """Test that different seeds produce different results."""
        from config import set_global_seeds
        import numpy as np

        set_global_seeds(42)
        n1 = np.random.rand()

        set_global_seeds(123)
        n2 = np.random.rand()

        assert n1 != n2, "Different seeds should produce different results"

    def test_reproducible_splits(self):
        """Test that splits are reproducible with same seed."""
        from config import set_global_seeds
        from stages.stage7_splits import create_chronological_splits

        df = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=1000, freq='5min'),
            'close': range(1000)
        })

        # First run
        set_global_seeds(42)
        train1, val1, test1, _ = create_chronological_splits(
            df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            purge_bars=10, embargo_bars=20
        )

        # Second run with same seed
        set_global_seeds(42)
        train2, val2, test2, _ = create_chronological_splits(
            df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            purge_bars=10, embargo_bars=20
        )

        # Should be identical (splits are deterministic)
        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(val1, val2)
        np.testing.assert_array_equal(test1, test2)


class TestGetBarrierParams:
    """Tests for get_barrier_params helper function."""

    def test_symbol_specific_params_returned(self) -> None:
        """Test that symbol-specific params are returned when available."""
        from config import get_barrier_params, BARRIER_PARAMS

        # Test with active horizon from TEST_HORIZONS
        params: dict[str, float | int] = get_barrier_params('MES', 5)
        expected: dict[str, float | int] = BARRIER_PARAMS['MES'][5]

        assert params['k_up'] == expected['k_up']
        assert params['k_down'] == expected['k_down']
        assert params['max_bars'] == expected['max_bars']

    def test_default_params_fallback(self):
        """Test that default params are used for unknown symbols."""
        from config import get_barrier_params, BARRIER_PARAMS_DEFAULT

        params = get_barrier_params('UNKNOWN', 5)
        expected = BARRIER_PARAMS_DEFAULT[5]

        assert params['k_up'] == expected['k_up']
        assert params['k_down'] == expected['k_down']
        assert params['max_bars'] == expected['max_bars']

    def test_ultimate_fallback_for_unknown_horizon(self):
        """Test that ultimate fallback works for unknown horizon."""
        from config import get_barrier_params

        params = get_barrier_params('UNKNOWN', 999)

        # Ultimate fallback: symmetric barriers with k scaled by horizon
        # k_base = 1.0 + (horizon / 20.0) * 1.5
        expected_k = round(1.0 + (999 / 20.0) * 1.5, 2)  # ≈ 75.93
        assert params['k_up'] == expected_k
        assert params['k_down'] == expected_k
        assert params['max_bars'] == 999 * 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
