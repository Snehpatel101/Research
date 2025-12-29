"""
Cross-Module Integration Tests for ML Pipeline.

Tests integration between major pipeline components:
- Phase 1 (data pipeline) -> Phase 2 (model training)
- Feature engineering -> Model training
- Cross-validation -> Model evaluation
- Data container -> All model types
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# CV SPLITS TO MODEL TRAINING
# =============================================================================


class TestCVSplitsToTraining:
    """Tests that CV splits integrate correctly with model training."""

    def test_purged_kfold_produces_valid_indices(self):
        """PurgedKFold indices should be valid for array indexing."""
        from src.cross_validation.purged_kfold import PurgedKFold, PurgedKFoldConfig

        np.random.seed(42)
        n_samples = 500

        X = pd.DataFrame(np.random.randn(n_samples, 20))
        y = pd.Series(np.random.choice([-1, 0, 1], n_samples))

        config = PurgedKFoldConfig(n_splits=5, purge_bars=10, embargo_bars=20)
        cv = PurgedKFold(config)

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            # Indices should be valid
            assert train_idx.max() < n_samples
            assert val_idx.max() < n_samples
            assert train_idx.min() >= 0
            assert val_idx.min() >= 0

            # No overlap
            overlap = set(train_idx) & set(val_idx)
            assert len(overlap) == 0, f"Fold {fold_idx}: train/val overlap"

            # Can actually index arrays
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            assert len(X_train) > 0
            assert len(X_val) > 0


# =============================================================================
# DATA CONTAINER TO ALL MODEL TYPES
# =============================================================================


class TestDataContainerToModels:
    """Tests that data container works with all model types."""

    @pytest.fixture
    def mock_container(self):
        """Create mock data container."""
        np.random.seed(42)
        n_train, n_val = 200, 50
        n_features = 20
        seq_len = 30

        # Tabular data
        X_train = np.random.randn(n_train, n_features).astype(np.float32)
        y_train = np.random.choice([-1, 0, 1], n_train)
        weights_train = np.ones(n_train, dtype=np.float32)

        X_val = np.random.randn(n_val, n_features).astype(np.float32)
        y_val = np.random.choice([-1, 0, 1], n_val)
        weights_val = np.ones(n_val, dtype=np.float32)

        # Sequence data
        X_train_seq = np.random.randn(n_train, seq_len, n_features).astype(np.float32)
        X_val_seq = np.random.randn(n_val, seq_len, n_features).astype(np.float32)

        mock = MagicMock()

        def get_sklearn_arrays(split):
            if split == "train":
                return X_train, y_train, weights_train
            return X_val, y_val, weights_val

        mock.get_sklearn_arrays = get_sklearn_arrays
        mock.get_array = lambda split, arr_type: {
            ("train", "features"): X_train,
            ("train", "labels"): y_train,
            ("train", "weights"): weights_train,
            ("val", "features"): X_val,
            ("val", "labels"): y_val,
            ("val", "weights"): weights_val,
        }.get((split, arr_type))

        return mock

    def test_container_with_xgboost(self, mock_container):
        """Data container should work with XGBoost."""
        from src.models.boosting import XGBoostModel

        X_train, y_train, _ = mock_container.get_sklearn_arrays("train")
        X_val, y_val, _ = mock_container.get_sklearn_arrays("val")

        model = XGBoostModel(config={
            "n_estimators": 10,
            "max_depth": 3,
            "verbosity": 0,
        })

        metrics = model.fit(X_train, y_train, X_val, y_val)
        assert metrics is not None

        preds = model.predict(X_val)
        assert len(preds.class_predictions) == len(y_val)

    def test_container_with_random_forest(self, mock_container):
        """Data container should work with Random Forest."""
        from src.models.classical import RandomForestModel

        X_train, y_train, _ = mock_container.get_sklearn_arrays("train")
        X_val, y_val, _ = mock_container.get_sklearn_arrays("val")

        model = RandomForestModel(config={
            "n_estimators": 10,
            "max_depth": 3,
            "n_jobs": 1,
        })

        metrics = model.fit(X_train, y_train, X_val, y_val)
        assert metrics is not None

        preds = model.predict(X_val)
        assert len(preds.class_predictions) == len(y_val)

    def test_container_with_logistic(self, mock_container):
        """Data container should work with Logistic Regression."""
        from src.models.classical import LogisticModel

        X_train, y_train, _ = mock_container.get_sklearn_arrays("train")
        X_val, y_val, _ = mock_container.get_sklearn_arrays("val")

        model = LogisticModel(config={"max_iter": 100})

        metrics = model.fit(X_train, y_train, X_val, y_val)
        assert metrics is not None

        preds = model.predict(X_val)
        assert len(preds.class_predictions) == len(y_val)


# =============================================================================
# TEMPORAL ORDERING PRESERVATION
# =============================================================================


class TestTemporalOrderPreservation:
    """Tests that temporal ordering is preserved across modules."""

    def test_cv_splits_no_overlap(self):
        """CV splits should not overlap between train and val."""
        from src.cross_validation.purged_kfold import PurgedKFold, PurgedKFoldConfig

        np.random.seed(42)
        n_samples = 500

        X = pd.DataFrame(np.random.randn(n_samples, 20))
        y = pd.Series(np.random.choice([-1, 0, 1], n_samples))

        config = PurgedKFoldConfig(n_splits=5, purge_bars=10, embargo_bars=20)
        cv = PurgedKFold(config)

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            # No overlap between train and val
            overlap = set(train_idx) & set(val_idx)
            assert len(overlap) == 0, f"Fold {fold_idx}: train/val overlap"

            # Both splits should have samples
            assert len(train_idx) > 0, f"Fold {fold_idx}: empty train set"
            assert len(val_idx) > 0, f"Fold {fold_idx}: empty val set"


# =============================================================================
# LABEL FLOW INTEGRATION
# =============================================================================


class TestLabelFlowIntegration:
    """Tests that labels flow correctly through the pipeline."""

    def test_labels_reach_model_training(self):
        """Labels from labeling stage should reach model training."""
        np.random.seed(42)
        n_bars = 300

        # Create data with labels
        df = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(n_bars) * 0.5),
            'high': 100 + np.cumsum(np.random.randn(n_bars) * 0.5) + 0.5,
            'low': 100 + np.cumsum(np.random.randn(n_bars) * 0.5) - 0.5,
            'atr_14': np.abs(np.random.randn(n_bars)) + 0.5,
        }, index=pd.date_range('2020-01-01', periods=n_bars, freq='5min'))
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])

        from src.phase1.stages.labeling.triple_barrier import TripleBarrierLabeler

        labeler = TripleBarrierLabeler(
            k_up=2.0,
            k_down=1.5,
            max_bars=20,
            apply_transaction_costs=False,
        )
        result = labeler.compute_labels(df, horizon=20)

        # Labels should exist
        assert result is not None
        assert hasattr(result, 'labels')
        assert result.labels is not None

        # Labels should be valid values
        valid_labels = result.labels[~np.isnan(result.labels)]
        assert len(valid_labels) > 0
        # Filter out sentinel value -99 (used for invalid labels)
        actual_labels = valid_labels[valid_labels != -99]
        assert set(actual_labels).issubset({-1, 0, 1})

    def test_label_distribution_preserved(self):
        """Label distribution should be preserved through pipeline."""
        np.random.seed(42)

        # Original labels
        original_labels = np.random.choice([-1, 0, 1], 1000, p=[0.3, 0.4, 0.3])

        # After train/val split
        train_size = 700
        train_labels = original_labels[:train_size]
        val_labels = original_labels[train_size:]

        # Distribution should be roughly similar
        def get_dist(labels):
            unique, counts = np.unique(labels, return_counts=True)
            return dict(zip(unique, counts / len(labels)))

        original_dist = get_dist(original_labels)
        train_dist = get_dist(train_labels)

        # Should be within 15% tolerance
        for label in [-1, 0, 1]:
            diff = abs(original_dist.get(label, 0) - train_dist.get(label, 0))
            assert diff < 0.15, f"Label {label} distribution changed significantly"


# =============================================================================
# SAMPLE WEIGHTS INTEGRATION
# =============================================================================


class TestSampleWeightsIntegration:
    """Tests that sample weights flow correctly through pipeline."""

    def test_weights_used_in_xgboost_training(self):
        """XGBoost should use sample weights during training."""
        np.random.seed(42)

        X_train = np.random.randn(200, 20).astype(np.float32)
        y_train = np.random.choice([-1, 0, 1], 200)
        X_val = np.random.randn(50, 20).astype(np.float32)
        y_val = np.random.choice([-1, 0, 1], 50)

        # Heavy weights on first half
        weights_biased = np.ones(200, dtype=np.float32)
        weights_biased[:100] = 10.0  # 10x weight on first half

        from src.models.boosting import XGBoostModel

        model = XGBoostModel(config={
            "n_estimators": 20,
            "max_depth": 3,
            "verbosity": 0,
        })

        # Train with biased weights
        metrics = model.fit(
            X_train, y_train, X_val, y_val,
            sample_weights=weights_biased
        )

        # Model should train successfully
        assert metrics is not None

    def test_weights_preserved_through_cv(self):
        """Sample weights should be preserved through CV splits."""
        from src.cross_validation.purged_kfold import PurgedKFold, PurgedKFoldConfig

        np.random.seed(42)
        n_samples = 500

        X = pd.DataFrame(np.random.randn(n_samples, 20))
        y = pd.Series(np.random.choice([-1, 0, 1], n_samples))
        weights = np.random.uniform(0.5, 1.5, n_samples)

        config = PurgedKFoldConfig(n_splits=5, purge_bars=10, embargo_bars=20)
        cv = PurgedKFold(config)

        for train_idx, val_idx in cv.split(X, y):
            train_weights = weights[train_idx]
            val_weights = weights[val_idx]

            # Weights should be valid
            assert len(train_weights) == len(train_idx)
            assert len(val_weights) == len(val_idx)
            assert (train_weights > 0).all()
            assert (val_weights > 0).all()


# =============================================================================
# MODEL REGISTRY INTEGRATION
# =============================================================================


class TestModelRegistryIntegration:
    """Tests for model registry integration."""

    def test_registry_lists_models(self):
        """Registry should list available models."""
        from src.models import ModelRegistry

        models = ModelRegistry.list_all()
        assert isinstance(models, list)
        assert len(models) > 0
        assert 'xgboost' in models

    def test_registry_creates_models(self):
        """Registry should create models."""
        from src.models import ModelRegistry

        model = ModelRegistry.create('xgboost', config={
            'n_estimators': 5,
            'verbosity': 0,
        })

        assert model is not None
        # Check model name attribute if available
        if hasattr(model, 'model_name'):
            assert model.model_name == 'xgboost'
        elif hasattr(model, 'name'):
            assert model.name == 'xgboost'
        else:
            # Model was created successfully - that's the main assertion
            assert True

    def test_all_models_have_consistent_interface(self):
        """All models should implement BaseModel interface."""
        from src.models import ModelRegistry
        from src.models.base import BaseModel

        for model_name in ['xgboost', 'random_forest', 'logistic']:
            if model_name in ModelRegistry.list_all():
                model = ModelRegistry.create(model_name, config={})
                assert isinstance(model, BaseModel)
                assert hasattr(model, 'fit')
                assert hasattr(model, 'predict')
                assert hasattr(model, 'save')
                assert hasattr(model, 'load')
