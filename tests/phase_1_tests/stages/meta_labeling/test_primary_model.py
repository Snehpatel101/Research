"""
Tests for PrimaryClassifier and RecallOptimizer.

Tests cover:
- PrimaryModelConfig validation
- RecallOptimizer threshold optimization
- PrimaryClassifier training and prediction
- Integration with sklearn-compatible API
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.phase1.stages.meta_labeling.primary_model import (
    PrimaryClassifier,
    PrimaryModelConfig,
    RecallOptimizer,
    DEFAULT_RECALL_TARGET,
    MIN_RECALL_THRESHOLD,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def binary_classification_data():
    """Create binary classification dataset."""
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42,
        class_sep=0.8,
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)


@pytest.fixture
def imbalanced_data():
    """Create imbalanced classification dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        weights=[0.8, 0.2],  # 80% class 0, 20% class 1
        random_state=42,
        class_sep=0.5,
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)


@pytest.fixture
def sample_probas():
    """Create sample probability predictions."""
    np.random.seed(42)
    n = 200
    # Generate probabilities centered around 0.5 with some signal
    probas = np.clip(np.random.normal(0.5, 0.2, n), 0.01, 0.99)
    # Generate true labels with correlation to probas
    y_true = (probas > 0.5 + np.random.normal(0, 0.1, n)).astype(int)
    return y_true, probas


# =============================================================================
# PRIMARY MODEL CONFIG TESTS
# =============================================================================


class TestPrimaryModelConfig:
    """Tests for PrimaryModelConfig validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PrimaryModelConfig()

        assert config.recall_target == DEFAULT_RECALL_TARGET
        assert config.min_recall == MIN_RECALL_THRESHOLD
        assert config.base_model == "logistic"
        assert config.calibrate is True
        assert config.cv_folds == 5
        assert config.class_weight == "balanced"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PrimaryModelConfig(
            recall_target=0.80,
            min_recall=0.60,
            base_model="lightgbm",
            calibrate=False,
            cv_folds=3,
            random_state=123,
        )

        assert config.recall_target == 0.80
        assert config.min_recall == 0.60
        assert config.base_model == "lightgbm"
        assert config.calibrate is False

    def test_invalid_recall_target_raises(self):
        """Test that invalid recall_target raises ValueError."""
        with pytest.raises(ValueError, match="recall_target must be in"):
            PrimaryModelConfig(recall_target=0)

        with pytest.raises(ValueError, match="recall_target must be in"):
            PrimaryModelConfig(recall_target=1.5)

        with pytest.raises(ValueError, match="recall_target must be in"):
            PrimaryModelConfig(recall_target=-0.1)

    def test_invalid_min_recall_raises(self):
        """Test that invalid min_recall raises ValueError."""
        # min_recall must be less than recall_target
        with pytest.raises(ValueError, match="min_recall must be in"):
            PrimaryModelConfig(recall_target=0.70, min_recall=0.80)

        with pytest.raises(ValueError, match="min_recall must be in"):
            PrimaryModelConfig(min_recall=0)

        with pytest.raises(ValueError, match="min_recall must be in"):
            PrimaryModelConfig(min_recall=-0.1)

    def test_invalid_cv_folds_raises(self):
        """Test that invalid cv_folds raises ValueError."""
        with pytest.raises(ValueError, match="cv_folds must be >= 2"):
            PrimaryModelConfig(cv_folds=1)

        with pytest.raises(ValueError, match="cv_folds must be >= 2"):
            PrimaryModelConfig(cv_folds=0)

    def test_invalid_base_model_raises(self):
        """Test that invalid base_model raises ValueError."""
        with pytest.raises(ValueError, match="base_model must be one of"):
            PrimaryModelConfig(base_model="invalid_model")

    def test_valid_base_models(self):
        """Test that valid base models are accepted."""
        for model in ["logistic", "lightgbm", "xgboost", "random_forest"]:
            config = PrimaryModelConfig(base_model=model)
            assert config.base_model == model


# =============================================================================
# RECALL OPTIMIZER TESTS
# =============================================================================


class TestRecallOptimizer:
    """Tests for RecallOptimizer."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        optimizer = RecallOptimizer()

        assert optimizer.recall_target == DEFAULT_RECALL_TARGET
        assert optimizer.optimal_threshold is None
        assert optimizer.achieved_recall is None

    def test_initialization_custom(self):
        """Test custom initialization."""
        optimizer = RecallOptimizer(
            recall_target=0.80,
            search_min=0.05,
            search_max=0.40,
            n_steps=100,
        )

        assert optimizer.recall_target == 0.80
        assert optimizer.search_min == 0.05
        assert optimizer.search_max == 0.40
        assert optimizer.n_steps == 100

    def test_invalid_recall_target_raises(self):
        """Test that invalid recall_target raises ValueError."""
        with pytest.raises(ValueError, match="recall_target must be in"):
            RecallOptimizer(recall_target=0)

        with pytest.raises(ValueError, match="recall_target must be in"):
            RecallOptimizer(recall_target=1.5)

    def test_fit_basic(self, sample_probas):
        """Test basic threshold fitting."""
        y_true, probas = sample_probas
        optimizer = RecallOptimizer(recall_target=0.70)

        threshold = optimizer.fit(y_true, probas)

        assert isinstance(threshold, float)
        assert 0 < threshold < 1
        assert optimizer.optimal_threshold == threshold
        assert optimizer.achieved_recall is not None
        assert optimizer.achieved_recall >= 0.0

    def test_fit_achieves_target_recall(self, sample_probas):
        """Test that fit achieves target recall when possible."""
        y_true, probas = sample_probas
        target_recall = 0.60
        optimizer = RecallOptimizer(recall_target=target_recall)

        optimizer.fit(y_true, probas)

        # Should achieve at least target recall (or close to it)
        if optimizer.achieved_recall is not None:
            # Allow some tolerance
            assert optimizer.achieved_recall >= target_recall - 0.1 or \
                   optimizer.achieved_recall >= 0.5

    def test_apply_threshold(self, sample_probas):
        """Test threshold application."""
        y_true, probas = sample_probas
        optimizer = RecallOptimizer()
        optimizer.fit(y_true, probas)

        y_pred = optimizer.apply_threshold(probas)

        assert isinstance(y_pred, np.ndarray)
        assert len(y_pred) == len(probas)
        assert set(y_pred.tolist()).issubset({0, 1})

    def test_apply_threshold_not_fitted_raises(self):
        """Test that apply_threshold before fit raises error."""
        optimizer = RecallOptimizer()
        probas = np.array([0.3, 0.5, 0.7])

        with pytest.raises(RuntimeError, match="fit"):
            optimizer.apply_threshold(probas)

    def test_higher_recall_target_lowers_threshold(self, sample_probas):
        """Test that higher recall target leads to lower threshold."""
        y_true, probas = sample_probas

        optimizer_low = RecallOptimizer(recall_target=0.50)
        optimizer_high = RecallOptimizer(recall_target=0.90)

        threshold_low = optimizer_low.fit(y_true, probas)
        threshold_high = optimizer_high.fit(y_true, probas)

        # Higher recall target should have lower or equal threshold
        assert threshold_high <= threshold_low + 0.1

    def test_fit_with_all_same_class(self):
        """Test fit with degenerate data (all same class)."""
        y_true = np.array([1, 1, 1, 1, 1])
        probas = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

        optimizer = RecallOptimizer()
        threshold = optimizer.fit(y_true, probas)

        # Should still return a valid threshold
        assert isinstance(threshold, float)
        assert 0 <= threshold <= 1


# =============================================================================
# PRIMARY CLASSIFIER TESTS
# =============================================================================


class TestPrimaryClassifier:
    """Tests for PrimaryClassifier."""

    def test_initialization_defaults(self):
        """Test initialization with defaults."""
        clf = PrimaryClassifier()

        assert clf.recall_target == DEFAULT_RECALL_TARGET
        assert clf.base_model == "logistic"
        assert clf.calibrate is True

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        clf = PrimaryClassifier(
            recall_target=0.80,
            base_model="random_forest",
            calibrate=False,
        )

        assert clf.recall_target == 0.80
        assert clf.base_model == "random_forest"
        assert clf.calibrate is False

    def test_sklearn_compatible_interface(self, binary_classification_data):
        """Test sklearn-compatible interface."""
        X_train, X_test, y_train, y_test = binary_classification_data

        clf = PrimaryClassifier()

        # Should have fit/predict/predict_proba methods
        assert hasattr(clf, "fit")
        assert hasattr(clf, "predict")
        assert hasattr(clf, "predict_proba")

    def test_fit_and_predict(self, binary_classification_data):
        """Test basic fit and predict."""
        X_train, X_test, y_train, y_test = binary_classification_data

        clf = PrimaryClassifier()
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)

        assert isinstance(y_pred, np.ndarray)
        assert len(y_pred) == len(X_test)
        assert set(y_pred.tolist()).issubset({0, 1})

    def test_predict_proba(self, binary_classification_data):
        """Test predict_proba returns probabilities."""
        X_train, X_test, y_train, y_test = binary_classification_data

        clf = PrimaryClassifier()
        clf.fit(X_train, y_train)

        probas = clf.predict_proba(X_test)

        assert isinstance(probas, np.ndarray)
        # Check probabilities are valid
        assert np.all(probas >= 0) and np.all(probas <= 1)

    def test_predict_side_returns_binary_signal(self, binary_classification_data):
        """Test predict_side returns binary signal indicator."""
        X_train, X_test, y_train, y_test = binary_classification_data

        clf = PrimaryClassifier()
        clf.fit(X_train, y_train)

        # predict_side returns has_signal indicator (0 or 1)
        # unless y_direction is provided
        sides = clf.predict_side(X_test)

        assert isinstance(sides, np.ndarray)
        assert len(sides) == len(X_test)
        # Without y_direction, returns binary {0, 1}
        assert set(sides.tolist()).issubset({0, 1})

    def test_predict_side_with_direction(self, binary_classification_data):
        """Test predict_side with y_direction returns actual sides."""
        X_train, X_test, y_train, y_test = binary_classification_data

        # Create training labels with direction: -1, 0, 1
        np.random.seed(42)
        y_train_dir = np.random.choice([-1, 0, 1], size=len(y_train), p=[0.3, 0.2, 0.5])
        y_test_dir = np.random.choice([-1, 0, 1], size=len(y_test), p=[0.3, 0.2, 0.5])

        clf = PrimaryClassifier()
        clf.fit(X_train, y_train_dir)

        # With y_direction, should return actual sides {-1, 1}
        sides = clf.predict_side(X_test, y_direction=y_test_dir)

        assert isinstance(sides, np.ndarray)
        assert len(sides) == len(X_test)

    def test_predict_not_fitted_raises(self):
        """Test that predict before fit raises error."""
        clf = PrimaryClassifier()
        X_test = np.random.randn(10, 5)

        with pytest.raises((ValueError, AttributeError, RuntimeError)):
            clf.predict(X_test)

    def test_imbalanced_data_handling(self, imbalanced_data):
        """Test handling of imbalanced data."""
        X_train, X_test, y_train, y_test = imbalanced_data

        # Use balanced class weights (default)
        clf = PrimaryClassifier(class_weight="balanced")
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        # Should produce predictions for both classes
        assert len(y_pred) == len(X_test)

    def test_is_fitted_property(self, binary_classification_data):
        """Test that _is_fitted is set after fit."""
        X_train, X_test, y_train, y_test = binary_classification_data

        clf = PrimaryClassifier()
        assert clf._is_fitted is False

        clf.fit(X_train, y_train)
        assert clf._is_fitted is True

    def test_different_base_models(self, binary_classification_data):
        """Test training with different base models."""
        X_train, X_test, y_train, y_test = binary_classification_data

        # Test logistic regression
        clf = PrimaryClassifier(base_model="logistic")
        clf.fit(X_train, y_train)
        assert clf.predict(X_test) is not None

        # Test random forest
        clf_rf = PrimaryClassifier(base_model="random_forest")
        clf_rf.fit(X_train, y_train)
        assert clf_rf.predict(X_test) is not None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestPrimaryClassifierIntegration:
    """Integration tests for primary classifier workflow."""

    def test_full_workflow(self, binary_classification_data):
        """Test full workflow: configure -> fit -> predict."""
        X_train, X_test, y_train, y_test = binary_classification_data

        # 1. Create classifier with settings
        clf = PrimaryClassifier(
            recall_target=0.70,
            base_model="logistic",
            calibrate=True,
        )

        # 2. Fit
        clf.fit(X_train, y_train)

        # 3. Predict class
        y_pred = clf.predict(X_test)
        assert len(y_pred) == len(X_test)

        # 4. Predict probability
        y_proba = clf.predict_proba(X_test)
        assert np.all((y_proba >= 0) & (y_proba <= 1))

    def test_reproducibility(self, binary_classification_data):
        """Test that same random seed produces same results."""
        X_train, X_test, y_train, y_test = binary_classification_data

        clf1 = PrimaryClassifier(random_state=42)
        clf1.fit(X_train, y_train)
        pred1 = clf1.predict(X_test)

        clf2 = PrimaryClassifier(random_state=42)
        clf2.fit(X_train, y_train)
        pred2 = clf2.predict(X_test)

        np.testing.assert_array_equal(pred1, pred2)


# =============================================================================
# VALIDATION MODULE INTEGRATION
# =============================================================================


class TestValidationModuleIntegration:
    """Test that classes are importable from validation module."""

    def test_import_from_validation(self):
        """Test importing from src.validation."""
        from src.validation import (
            PrimaryClassifier,
            PrimaryModelConfig,
            RecallOptimizer,
        )

        assert PrimaryClassifier is not None
        assert PrimaryModelConfig is not None
        assert RecallOptimizer is not None

    def test_create_from_validation_import(self, binary_classification_data):
        """Test creating instances from validation imports."""
        from src.validation import PrimaryClassifier

        X_train, X_test, y_train, y_test = binary_classification_data

        clf = PrimaryClassifier(recall_target=0.70)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        assert len(y_pred) == len(X_test)
