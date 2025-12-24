"""
Tests for labeling strategy factory.

Tests cover:
- get_labeler function
- get_available_strategies function
- register_strategy function
- create_multi_labeler function
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.phase1.stages.labeling import (
    DirectionalLabeler,
    LabelingResult,
    LabelingStrategy,
    LabelingType,
    RegressionLabeler,
    ThresholdLabeler,
    TripleBarrierLabeler,
    get_labeler,
)
from src.phase1.stages.labeling.factory import (
    create_multi_labeler,
    get_available_strategies,
    register_strategy,
)


class TestGetLabeler:
    """Tests for get_labeler factory function."""

    def test_get_triple_barrier_by_enum(self):
        """Test getting TripleBarrierLabeler by enum."""
        labeler = get_labeler(LabelingType.TRIPLE_BARRIER)
        assert isinstance(labeler, TripleBarrierLabeler)

    def test_get_triple_barrier_by_string(self):
        """Test getting TripleBarrierLabeler by string."""
        labeler = get_labeler('triple_barrier')
        assert isinstance(labeler, TripleBarrierLabeler)

    def test_get_directional_labeler(self):
        """Test getting DirectionalLabeler."""
        labeler = get_labeler(LabelingType.DIRECTIONAL)
        assert isinstance(labeler, DirectionalLabeler)

    def test_get_threshold_labeler(self):
        """Test getting ThresholdLabeler."""
        labeler = get_labeler(LabelingType.THRESHOLD)
        assert isinstance(labeler, ThresholdLabeler)

    def test_get_regression_labeler(self):
        """Test getting RegressionLabeler."""
        labeler = get_labeler(LabelingType.REGRESSION)
        assert isinstance(labeler, RegressionLabeler)

    def test_with_config_parameters(self):
        """Test passing configuration parameters."""
        labeler = get_labeler(
            LabelingType.TRIPLE_BARRIER,
            k_up=1.5,
            k_down=1.0,
            atr_column='atr_20'
        )
        assert labeler._k_up == 1.5
        assert labeler._k_down == 1.0
        assert labeler._atr_column == 'atr_20'

    def test_directional_with_config(self):
        """Test DirectionalLabeler with config."""
        labeler = get_labeler(
            LabelingType.DIRECTIONAL,
            threshold=0.001,
            use_log_returns=True
        )
        assert labeler._threshold == 0.001
        assert labeler._use_log_returns is True

    def test_threshold_with_config(self):
        """Test ThresholdLabeler with config."""
        labeler = get_labeler(
            LabelingType.THRESHOLD,
            pct_up=0.02,
            pct_down=0.015,
            max_bars=30
        )
        assert labeler._pct_up == 0.02
        assert labeler._pct_down == 0.015
        assert labeler._max_bars == 30

    def test_invalid_strategy_string(self):
        """Test that invalid strategy string raises error."""
        with pytest.raises(ValueError, match="Unknown"):
            get_labeler('invalid_strategy')

    def test_case_sensitive_string(self):
        """Test that string matching is case-sensitive."""
        # Should work with correct case
        labeler = get_labeler('triple_barrier')
        assert isinstance(labeler, TripleBarrierLabeler)

        # Should fail with wrong case
        with pytest.raises(ValueError):
            get_labeler('TRIPLE_BARRIER')


class TestGetAvailableStrategies:
    """Tests for get_available_strategies function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        strategies = get_available_strategies()
        assert isinstance(strategies, list)

    def test_contains_expected_strategies(self):
        """Test that expected strategies are present."""
        strategies = get_available_strategies()

        assert LabelingType.TRIPLE_BARRIER in strategies
        assert LabelingType.DIRECTIONAL in strategies
        assert LabelingType.THRESHOLD in strategies
        assert LabelingType.REGRESSION in strategies

    def test_strategies_are_labeling_types(self):
        """Test that all items are LabelingType."""
        strategies = get_available_strategies()

        for strategy in strategies:
            assert isinstance(strategy, LabelingType)


class TestRegisterStrategy:
    """Tests for register_strategy function."""

    def test_register_custom_strategy(self):
        """Test registering a custom strategy."""
        # Import the registry to save/restore original
        from stages.labeling.factory import _STRATEGY_REGISTRY
        from stages.labeling.meta import MetaLabeler

        # Save original MetaLabeler registration
        original_meta = _STRATEGY_REGISTRY.get(LabelingType.META)

        try:

            class CustomLabeler(LabelingStrategy):
                @property
                def labeling_type(self):
                    return LabelingType.META  # Using META for testing

                @property
                def required_columns(self):
                    return ['close']

                def compute_labels(self, df, horizon, **kwargs):
                    labels = np.zeros(len(df), dtype=np.int8)
                    return LabelingResult(labels=labels, horizon=horizon)

            # Register the custom strategy
            register_strategy(LabelingType.META, CustomLabeler)

            # Should be able to get it now
            labeler = get_labeler(LabelingType.META)
            assert isinstance(labeler, CustomLabeler)

        finally:
            # Restore original MetaLabeler to avoid polluting other tests
            if original_meta is not None:
                _STRATEGY_REGISTRY[LabelingType.META] = original_meta

    def test_register_non_strategy_raises(self):
        """Test that registering non-strategy class raises error."""

        class NotAStrategy:
            pass

        with pytest.raises(TypeError, match="subclass"):
            register_strategy(LabelingType.META, NotAStrategy)


class TestCreateMultiLabeler:
    """Tests for create_multi_labeler function."""

    def test_create_multiple_labelers(self):
        """Test creating multiple labelers."""
        strategies = [
            {'type': LabelingType.TRIPLE_BARRIER, 'atr_column': 'atr_14'},
            {'type': LabelingType.DIRECTIONAL, 'threshold': 0.001},
        ]

        labelers = create_multi_labeler(strategies)

        assert len(labelers) == 2
        assert isinstance(labelers[0], TripleBarrierLabeler)
        assert isinstance(labelers[1], DirectionalLabeler)

    def test_create_with_string_types(self):
        """Test creating with string type names."""
        strategies = [
            {'type': 'triple_barrier'},
            {'type': 'directional'},
            {'type': 'threshold'},
        ]

        labelers = create_multi_labeler(strategies)

        assert len(labelers) == 3
        assert isinstance(labelers[0], TripleBarrierLabeler)
        assert isinstance(labelers[1], DirectionalLabeler)
        assert isinstance(labelers[2], ThresholdLabeler)

    def test_missing_type_raises(self):
        """Test that missing type key raises error."""
        strategies = [
            {'atr_column': 'atr_14'},  # Missing 'type'
        ]

        with pytest.raises(ValueError, match="type"):
            create_multi_labeler(strategies)

    def test_config_parameters_passed(self):
        """Test that config parameters are passed to labelers."""
        strategies = [
            {
                'type': LabelingType.THRESHOLD,
                'pct_up': 0.03,
                'pct_down': 0.02,
                'max_bars': 25
            },
        ]

        labelers = create_multi_labeler(strategies)

        assert labelers[0]._pct_up == 0.03
        assert labelers[0]._pct_down == 0.02
        assert labelers[0]._max_bars == 25

    def test_empty_list_returns_empty(self):
        """Test that empty list returns empty list."""
        labelers = create_multi_labeler([])
        assert labelers == []


class TestFactoryIntegration:
    """Integration tests for factory with actual labeling."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        n = 100
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        open_ = close + np.random.randn(n) * 0.1

        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))

        return pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'atr_14': np.ones(n) * 2.0
        })

    def test_factory_labeler_works(self, sample_df):
        """Test that factory-created labeler works correctly."""
        labeler = get_labeler(LabelingType.TRIPLE_BARRIER)
        result = labeler.compute_labels(sample_df, horizon=5)

        assert len(result.labels) == len(sample_df)
        assert result.horizon == 5

    def test_multi_labeler_all_work(self, sample_df):
        """Test that all multi-labelers work correctly."""
        strategies = [
            {'type': 'triple_barrier'},
            {'type': 'directional'},
            {'type': 'threshold'},
            {'type': 'regression'},
        ]

        labelers = create_multi_labeler(strategies)

        for labeler in labelers:
            result = labeler.compute_labels(sample_df, horizon=5)
            assert len(result.labels) == len(sample_df)
