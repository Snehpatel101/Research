"""
Tests for Conformal Prediction Module.

Tests the conformal predictor for prediction sets with coverage guarantees.
Uses deterministic synthetic data for reproducibility.
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_probabilities():
    """Generate synthetic probability predictions."""
    np.random.seed(42)
    n_samples = 500
    n_classes = 3

    # Generate probabilities (normalized to sum to 1)
    probs = np.random.rand(n_samples, n_classes)
    probs = probs / probs.sum(axis=1, keepdims=True)

    return probs


@pytest.fixture
def sample_labels():
    """Generate synthetic true labels."""
    np.random.seed(42)
    n_samples = 500
    n_classes = 3

    return np.random.randint(0, n_classes, n_samples)


@pytest.fixture
def calibrated_probabilities():
    """Generate well-calibrated probabilities."""
    np.random.seed(42)
    n_samples = 500
    n_classes = 3

    # Generate labels first
    labels = np.random.randint(0, n_classes, n_samples)

    # Generate probabilities that are well-calibrated
    probs = np.random.rand(n_samples, n_classes) * 0.2  # Base noise
    for i in range(n_samples):
        probs[i, labels[i]] += 0.6  # True class gets higher probability

    probs = probs / probs.sum(axis=1, keepdims=True)

    return probs, labels


# =============================================================================
# CONFIG TESTS
# =============================================================================

class TestConformalConfig:
    """Test conformal prediction configuration."""

    def test_default_config(self):
        """Test default config is valid."""
        from src.models.calibration import ConformalConfig

        config = ConformalConfig()
        assert config.confidence_level == 0.90
        assert config.method == "lac"

    def test_invalid_confidence_level(self):
        """Test confidence level validation."""
        from src.models.calibration import ConformalConfig

        with pytest.raises(ValueError, match="confidence_level"):
            ConformalConfig(confidence_level=0.3)  # Too low

        with pytest.raises(ValueError, match="confidence_level"):
            ConformalConfig(confidence_level=1.0)  # Too high

    def test_invalid_method(self):
        """Test method validation."""
        from src.models.calibration import ConformalConfig

        with pytest.raises(ValueError, match="Unknown method"):
            ConformalConfig(method="invalid")


# =============================================================================
# CONFORMAL PREDICTOR TESTS
# =============================================================================

class TestConformalPredictor:
    """Test conformal predictor functionality."""

    def test_predictor_creation(self):
        """Test predictor can be created."""
        from src.models.calibration import ConformalPredictor, ConformalConfig

        config = ConformalConfig(confidence_level=0.90)
        predictor = ConformalPredictor(config)

        assert not predictor.is_fitted
        assert predictor.config.confidence_level == 0.90

    def test_fit(self, sample_probabilities, sample_labels):
        """Test fitting the predictor."""
        from src.models.calibration import ConformalPredictor

        predictor = ConformalPredictor()
        metrics = predictor.fit(sample_labels, sample_probabilities)

        assert predictor.is_fitted
        assert predictor.threshold > 0
        assert metrics.empirical_coverage > 0

    def test_predict_sets(self, sample_probabilities, sample_labels):
        """Test generating prediction sets."""
        from src.models.calibration import ConformalPredictor

        predictor = ConformalPredictor()
        predictor.fit(sample_labels[:300], sample_probabilities[:300])

        pred_sets, set_sizes = predictor.predict_sets(sample_probabilities[300:])

        assert pred_sets.shape == (200, 3)  # (n_samples, n_classes)
        assert len(set_sizes) == 200
        assert pred_sets.dtype == np.int32
        assert np.all(set_sizes >= 1)  # No empty sets by default

    def test_coverage_guarantee(self, calibrated_probabilities):
        """Test that coverage is approximately correct."""
        from src.models.calibration import ConformalPredictor, ConformalConfig

        probs, labels = calibrated_probabilities

        # Split into calibration and test
        cal_probs, test_probs = probs[:300], probs[300:]
        cal_labels, test_labels = labels[:300], labels[300:]

        config = ConformalConfig(confidence_level=0.90)
        predictor = ConformalPredictor(config)
        predictor.fit(cal_labels, cal_probs)

        pred_sets, _ = predictor.predict_sets(test_probs)

        # Check coverage on test set
        covered = [pred_sets[i, test_labels[i]] for i in range(len(test_labels))]
        empirical_coverage = np.mean(covered)

        # Coverage should be close to 0.90 (allow some variance)
        assert empirical_coverage >= 0.80, f"Coverage too low: {empirical_coverage}"
        assert empirical_coverage <= 1.0

    def test_predict_not_fitted_raises(self, sample_probabilities):
        """Test predict raises if not fitted."""
        from src.models.calibration import ConformalPredictor

        predictor = ConformalPredictor()

        with pytest.raises(RuntimeError, match="not fitted"):
            predictor.predict_sets(sample_probabilities)

    def test_lac_method(self, sample_probabilities, sample_labels):
        """Test LAC conformal method."""
        from src.models.calibration import ConformalPredictor, ConformalConfig

        config = ConformalConfig(method="lac")
        predictor = ConformalPredictor(config)
        predictor.fit(sample_labels, sample_probabilities)

        pred_sets, set_sizes = predictor.predict_sets(sample_probabilities)

        assert np.all(set_sizes >= 1)

    def test_aps_method(self, sample_probabilities, sample_labels):
        """Test APS conformal method."""
        from src.models.calibration import ConformalPredictor, ConformalConfig

        config = ConformalConfig(method="aps")
        predictor = ConformalPredictor(config)
        predictor.fit(sample_labels, sample_probabilities)

        pred_sets, set_sizes = predictor.predict_sets(sample_probabilities)

        assert np.all(set_sizes >= 1)

    def test_naive_method(self, sample_probabilities, sample_labels):
        """Test naive conformal method."""
        from src.models.calibration import ConformalPredictor, ConformalConfig

        config = ConformalConfig(method="naive")
        predictor = ConformalPredictor(config)
        predictor.fit(sample_labels, sample_probabilities)

        pred_sets, set_sizes = predictor.predict_sets(sample_probabilities)

        assert np.all(set_sizes >= 1)

    def test_allow_empty_sets(self, sample_probabilities, sample_labels):
        """Test allowing empty prediction sets."""
        from src.models.calibration import ConformalPredictor, ConformalConfig

        config = ConformalConfig(allow_empty_sets=True)
        predictor = ConformalPredictor(config)
        predictor.fit(sample_labels, sample_probabilities)

        # With allow_empty_sets, empty sets are possible
        # (though unlikely with typical data)
        pred_sets, set_sizes = predictor.predict_sets(sample_probabilities)

        assert pred_sets.shape[0] == len(sample_probabilities)


# =============================================================================
# PREDICTION WITH REJECTION TESTS
# =============================================================================

class TestPredictionWithRejection:
    """Test prediction with rejection for ambiguous cases."""

    def test_predict_with_rejection(self, sample_probabilities, sample_labels):
        """Test prediction with rejection."""
        from src.models.calibration import ConformalPredictor

        predictor = ConformalPredictor()
        predictor.fit(sample_labels[:300], sample_probabilities[:300])

        predictions, rejected = predictor.predict_with_rejection(
            sample_probabilities[300:],
            reject_threshold=2,
        )

        assert len(predictions) == 200
        assert len(rejected) == 200
        assert predictions.dtype == np.int32
        # Rejected samples have prediction -1
        assert np.all(predictions[rejected] == -1)
        # Non-rejected should have valid predictions
        assert np.all(predictions[~rejected] >= 0)


# =============================================================================
# METRICS TESTS
# =============================================================================

class TestConformalMetrics:
    """Test conformal prediction metrics."""

    def test_evaluate(self, sample_probabilities, sample_labels):
        """Test evaluation on test set."""
        from src.models.calibration import ConformalPredictor

        predictor = ConformalPredictor()
        predictor.fit(sample_labels[:300], sample_probabilities[:300])

        metrics = predictor.evaluate(sample_labels[300:], sample_probabilities[300:])

        assert 0 <= metrics.empirical_coverage <= 1
        assert metrics.average_set_size >= 1
        assert 0 <= metrics.singleton_rate <= 1
        assert 0 <= metrics.empty_set_rate <= 1
        assert metrics.n_samples == 200

    def test_metrics_to_dict(self, sample_probabilities, sample_labels):
        """Test metrics can be serialized."""
        from src.models.calibration import ConformalPredictor

        predictor = ConformalPredictor()
        metrics = predictor.fit(sample_labels, sample_probabilities)

        metrics_dict = metrics.to_dict()

        assert "empirical_coverage" in metrics_dict
        assert "average_set_size" in metrics_dict
        assert "threshold" in metrics_dict


# =============================================================================
# COVERAGE VALIDATION TESTS
# =============================================================================

class TestValidateCoverage:
    """Test coverage validation function."""

    def test_validate_coverage_passing(self):
        """Test coverage validation passes."""
        from src.models.calibration import validate_coverage

        # Perfect coverage
        y_true = np.array([0, 1, 2, 0, 1])
        pred_sets = np.array([
            [1, 0, 0],  # Covers 0
            [0, 1, 0],  # Covers 1
            [0, 0, 1],  # Covers 2
            [1, 0, 0],  # Covers 0
            [0, 1, 0],  # Covers 1
        ])

        result = validate_coverage(y_true, pred_sets, expected_coverage=1.0)

        assert result["empirical_coverage"] == 1.0
        assert result["passed"] is True

    def test_validate_coverage_failing(self):
        """Test coverage validation fails."""
        from src.models.calibration import validate_coverage

        # Poor coverage
        y_true = np.array([0, 1, 2, 0, 1])
        pred_sets = np.array([
            [0, 1, 0],  # Misses 0
            [1, 0, 0],  # Misses 1
            [0, 0, 1],  # Covers 2
            [1, 0, 0],  # Covers 0
            [0, 1, 0],  # Covers 1
        ])

        result = validate_coverage(y_true, pred_sets, expected_coverage=0.90, tolerance=0.05)

        assert result["empirical_coverage"] == 0.6
        assert result["passed"] is False


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================

class TestConformalSerialization:
    """Test conformal predictor save/load."""

    def test_save_and_load(self, sample_probabilities, sample_labels):
        """Test saving and loading predictor."""
        from src.models.calibration import ConformalPredictor, ConformalConfig

        config = ConformalConfig(confidence_level=0.85)
        predictor = ConformalPredictor(config)
        predictor.fit(sample_labels, sample_probabilities)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "conformal.pkl"

            # Save
            predictor.save(path)
            assert path.exists()

            # Load
            loaded = ConformalPredictor.load(path)

            assert loaded.is_fitted
            assert loaded.config.confidence_level == 0.85
            assert loaded.threshold == predictor.threshold
            assert loaded.n_classes == predictor.n_classes

    def test_save_not_fitted_raises(self):
        """Test save raises if not fitted."""
        from src.models.calibration import ConformalPredictor

        predictor = ConformalPredictor()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "conformal.pkl"

            with pytest.raises(RuntimeError, match="unfitted"):
                predictor.save(path)


# =============================================================================
# EDGE CASES
# =============================================================================

class TestConformalEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_probabilities_shape(self, sample_labels):
        """Test invalid probability shape raises error."""
        from src.models.calibration import ConformalPredictor

        predictor = ConformalPredictor()

        with pytest.raises(ValueError, match="must be 2D"):
            predictor.fit(sample_labels, np.random.rand(len(sample_labels)))

    def test_mismatched_lengths(self, sample_probabilities):
        """Test mismatched lengths raises error."""
        from src.models.calibration import ConformalPredictor

        predictor = ConformalPredictor()
        short_labels = np.array([0, 1, 2])

        with pytest.raises(ValueError, match="Length mismatch"):
            predictor.fit(short_labels, sample_probabilities)

    def test_wrong_n_classes_at_predict(self, sample_probabilities, sample_labels):
        """Test wrong number of classes at predict time."""
        from src.models.calibration import ConformalPredictor

        predictor = ConformalPredictor()
        predictor.fit(sample_labels, sample_probabilities)

        wrong_probs = np.random.rand(10, 5)  # 5 classes instead of 3
        wrong_probs = wrong_probs / wrong_probs.sum(axis=1, keepdims=True)

        with pytest.raises(ValueError, match="Expected 3 classes"):
            predictor.predict_sets(wrong_probs)

    def test_label_normalization(self, sample_probabilities):
        """Test labels are normalized correctly."""
        from src.models.calibration import ConformalPredictor

        predictor = ConformalPredictor()

        # Labels in -1, 0, 1 format
        labels = np.random.choice([-1, 0, 1], size=len(sample_probabilities))

        # Should work with -1, 0, 1 labels
        predictor.fit(labels, sample_probabilities)
        assert predictor.is_fitted
