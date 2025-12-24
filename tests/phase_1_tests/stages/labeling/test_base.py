"""
Tests for labeling base classes and types.

Tests:
- LabelingType enum values
- LabelingResult validation
- LabelingStrategy ABC interface
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.phase1.stages.labeling.base import LabelingResult, LabelingStrategy, LabelingType


class TestLabelingType:
    """Tests for LabelingType enum."""

    def test_labeling_type_values(self):
        """Test that all expected labeling types exist."""
        assert LabelingType.TRIPLE_BARRIER.value == 'triple_barrier'
        assert LabelingType.DIRECTIONAL.value == 'directional'
        assert LabelingType.THRESHOLD.value == 'threshold'
        assert LabelingType.REGRESSION.value == 'regression'
        assert LabelingType.META.value == 'meta'

    def test_labeling_type_from_string(self):
        """Test creating LabelingType from string."""
        assert LabelingType('triple_barrier') == LabelingType.TRIPLE_BARRIER
        assert LabelingType('directional') == LabelingType.DIRECTIONAL

    def test_labeling_type_invalid_value(self):
        """Test that invalid string raises ValueError."""
        with pytest.raises(ValueError):
            LabelingType('invalid_type')


class TestLabelingResult:
    """Tests for LabelingResult dataclass."""

    def test_labeling_result_creation(self):
        """Test creating a valid LabelingResult."""
        labels = np.array([1, -1, 0, 1, -1], dtype=np.int8)
        result = LabelingResult(labels=labels, horizon=5)

        assert np.array_equal(result.labels, labels)
        assert result.horizon == 5
        assert result.metadata == {}
        assert result.quality_metrics == {}

    def test_labeling_result_with_metadata(self):
        """Test LabelingResult with metadata."""
        labels = np.array([1, -1, 0], dtype=np.int8)
        metadata = {'bars_to_hit': np.array([3, 5, 10])}

        result = LabelingResult(
            labels=labels,
            horizon=5,
            metadata=metadata
        )

        assert 'bars_to_hit' in result.metadata
        assert np.array_equal(result.metadata['bars_to_hit'], np.array([3, 5, 10]))

    def test_labeling_result_invalid_labels_type(self):
        """Test that non-array labels raises TypeError."""
        with pytest.raises(TypeError):
            LabelingResult(labels=[1, -1, 0], horizon=5)

    def test_labeling_result_invalid_horizon_type(self):
        """Test that non-int horizon raises ValueError."""
        labels = np.array([1, -1, 0], dtype=np.int8)

        with pytest.raises(ValueError):
            LabelingResult(labels=labels, horizon=5.5)

    def test_labeling_result_negative_horizon(self):
        """Test that negative horizon raises ValueError."""
        labels = np.array([1, -1, 0], dtype=np.int8)

        with pytest.raises(ValueError):
            LabelingResult(labels=labels, horizon=-5)

    def test_labeling_result_zero_horizon(self):
        """Test that zero horizon raises ValueError."""
        labels = np.array([1, -1, 0], dtype=np.int8)

        with pytest.raises(ValueError):
            LabelingResult(labels=labels, horizon=0)


class TestLabelingStrategyABC:
    """Tests for LabelingStrategy abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Test that LabelingStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LabelingStrategy()

    def test_concrete_implementation_required(self):
        """Test that concrete implementations must override abstract methods."""

        # Create a partial implementation
        class PartialLabeler(LabelingStrategy):
            @property
            def labeling_type(self):
                return LabelingType.DIRECTIONAL

            # Missing: required_columns, compute_labels

        with pytest.raises(TypeError):
            PartialLabeler()

    def test_valid_concrete_implementation(self):
        """Test that a valid concrete implementation can be instantiated."""

        class TestLabeler(LabelingStrategy):
            @property
            def labeling_type(self):
                return LabelingType.DIRECTIONAL

            @property
            def required_columns(self):
                return ['close']

            def compute_labels(self, df, horizon, **kwargs):
                labels = np.zeros(len(df), dtype=np.int8)
                return LabelingResult(labels=labels, horizon=horizon)

        labeler = TestLabeler()
        assert labeler.labeling_type == LabelingType.DIRECTIONAL
        assert labeler.required_columns == ['close']

    def test_validate_inputs_empty_df(self):
        """Test validate_inputs raises on empty DataFrame."""

        class TestLabeler(LabelingStrategy):
            @property
            def labeling_type(self):
                return LabelingType.DIRECTIONAL

            @property
            def required_columns(self):
                return ['close']

            def compute_labels(self, df, horizon, **kwargs):
                self.validate_inputs(df)
                labels = np.zeros(len(df), dtype=np.int8)
                return LabelingResult(labels=labels, horizon=horizon)

        labeler = TestLabeler()
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty"):
            labeler.validate_inputs(empty_df)

    def test_validate_inputs_missing_columns(self):
        """Test validate_inputs raises on missing columns."""

        class TestLabeler(LabelingStrategy):
            @property
            def labeling_type(self):
                return LabelingType.DIRECTIONAL

            @property
            def required_columns(self):
                return ['close', 'volume']

            def compute_labels(self, df, horizon, **kwargs):
                self.validate_inputs(df)
                labels = np.zeros(len(df), dtype=np.int8)
                return LabelingResult(labels=labels, horizon=horizon)

        labeler = TestLabeler()
        df = pd.DataFrame({'close': [100, 101, 102]})

        with pytest.raises(KeyError, match="volume"):
            labeler.validate_inputs(df)

    def test_get_quality_metrics_classification(self):
        """Test quality metrics for classification labels."""

        class TestLabeler(LabelingStrategy):
            @property
            def labeling_type(self):
                return LabelingType.DIRECTIONAL

            @property
            def required_columns(self):
                return ['close']

            def compute_labels(self, df, horizon, **kwargs):
                labels = np.array([1, -1, 0, 1, 1, -99], dtype=np.int8)
                return LabelingResult(labels=labels, horizon=horizon)

        labeler = TestLabeler()
        result = LabelingResult(
            labels=np.array([1, -1, 0, 1, 1, -99], dtype=np.int8),
            horizon=5
        )

        metrics = labeler.get_quality_metrics(result)

        assert metrics['total_samples'] == 6
        assert metrics['valid_samples'] == 5
        assert metrics['invalid_samples'] == 1
        assert metrics['long_count'] == 3
        assert metrics['short_count'] == 1
        assert metrics['neutral_count'] == 1
        assert metrics['long_pct'] == 60.0
        assert metrics['short_pct'] == 20.0
        assert metrics['neutral_pct'] == 20.0

    def test_add_labels_to_dataframe(self):
        """Test adding labels to DataFrame."""

        class TestLabeler(LabelingStrategy):
            @property
            def labeling_type(self):
                return LabelingType.DIRECTIONAL

            @property
            def required_columns(self):
                return ['close']

            def compute_labels(self, df, horizon, **kwargs):
                labels = np.ones(len(df), dtype=np.int8)
                return LabelingResult(
                    labels=labels,
                    horizon=horizon,
                    metadata={'extra': np.zeros(len(df))}
                )

        labeler = TestLabeler()
        df = pd.DataFrame({'close': [100, 101, 102]})
        result = labeler.compute_labels(df, horizon=5)

        df_labeled = labeler.add_labels_to_dataframe(df, result)

        assert 'label_h5' in df_labeled.columns
        assert 'extra_h5' in df_labeled.columns
        assert df_labeled['label_h5'].tolist() == [1, 1, 1]
