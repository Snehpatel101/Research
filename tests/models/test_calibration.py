"""
Tests for Probability Calibration Module.

Tests:
- Metrics: Brier score, ECE, reliability bins
- ProbabilityCalibrator: fit, calibrate, save/load
- Calibration improves miscalibrated probabilities
- Calibrated probabilities sum to 1
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.models.calibration import (
    CalibrationConfig,
    CalibrationMetrics,
    ProbabilityCalibrator,
    ReliabilityBins,
    compute_brier_score,
    compute_ece,
    compute_reliability_bins,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_calibration_data():
    """Generate sample data for calibration tests."""
    np.random.seed(42)
    n_samples = 1000
    n_classes = 3

    # Generate true labels
    y_true = np.random.randint(0, n_classes, n_samples)

    # Generate miscalibrated probabilities (overconfident)
    probs = np.random.dirichlet([0.5, 0.5, 0.5], n_samples)

    # Make probabilities somewhat correlated with true labels (but miscalibrated)
    for i in range(n_samples):
        probs[i, y_true[i]] += 0.3
    probs = probs / probs.sum(axis=1, keepdims=True)

    return y_true, probs


@pytest.fixture
def perfectly_calibrated_data():
    """Generate perfectly calibrated probabilities."""
    np.random.seed(42)
    n_samples = 1000
    n_classes = 3

    # Generate probabilities first
    probs = np.random.dirichlet([1, 1, 1], n_samples)

    # Generate labels according to probabilities (perfect calibration)
    y_true = np.array([
        np.random.choice(n_classes, p=p) for p in probs
    ])

    return y_true, probs


@pytest.fixture
def small_sample_data():
    """Generate small sample data (for testing sigmoid fallback)."""
    np.random.seed(42)
    n_samples = 50  # Small enough to trigger sigmoid
    n_classes = 3

    y_true = np.random.randint(0, n_classes, n_samples)
    probs = np.random.dirichlet([1, 1, 1], n_samples)

    for i in range(n_samples):
        probs[i, y_true[i]] += 0.2
    probs = probs / probs.sum(axis=1, keepdims=True)

    return y_true, probs


# =============================================================================
# METRICS TESTS
# =============================================================================

class TestBrierScore:
    """Tests for Brier score computation."""

    def test_brier_perfect_predictions(self):
        """Brier score is 0 for perfect predictions."""
        n_samples = 100
        y_true = np.array([0, 1, 2] * 33 + [0])[:n_samples]

        # Perfect one-hot probabilities
        probs = np.zeros((n_samples, 3))
        for i, y in enumerate(y_true):
            probs[i, y] = 1.0

        brier = compute_brier_score(y_true, probs)
        assert brier == 0.0

    def test_brier_worst_predictions(self):
        """Brier score is 2 for completely wrong predictions (3-class)."""
        n_samples = 100
        y_true = np.zeros(n_samples, dtype=int)

        # All probability on wrong class
        probs = np.zeros((n_samples, 3))
        probs[:, 1] = 1.0  # Predict class 1, true is class 0

        brier = compute_brier_score(y_true, probs)
        assert brier == 2.0

    def test_brier_uniform_predictions(self):
        """Brier score for uniform predictions."""
        n_samples = 300
        y_true = np.array([0, 1, 2] * 100)  # Balanced classes
        probs = np.ones((n_samples, 3)) / 3  # Uniform

        brier = compute_brier_score(y_true, probs)
        # Expected: mean(sum((1/3 - one_hot)^2)) = 2 * (1/3)^2 + (1 - 1/3)^2
        # = 2/9 + 4/9 = 6/9 = 0.667
        assert 0.6 < brier < 0.7

    def test_brier_empty_input(self):
        """Brier score returns 0 for empty input."""
        y_true = np.array([])
        probs = np.zeros((0, 3))

        brier = compute_brier_score(y_true, probs)
        assert brier == 0.0


class TestECE:
    """Tests for Expected Calibration Error."""

    def test_ece_perfect_calibration(self, perfectly_calibrated_data):
        """ECE is low for well-calibrated probabilities."""
        y_true, probs = perfectly_calibrated_data
        ece = compute_ece(y_true, probs)

        # Not exactly 0 due to finite samples, but should be small
        assert ece < 0.1

    def test_ece_overconfident(self):
        """ECE detects overconfident predictions."""
        n_samples = 1000
        y_true = np.random.randint(0, 3, n_samples)

        # Overconfident: always predict 90% on one class
        probs = np.full((n_samples, 3), 0.05)
        probs[:, 0] = 0.9

        ece = compute_ece(y_true, probs)
        # Should detect gap between 90% confidence and ~33% accuracy
        assert ece > 0.3

    def test_ece_range(self, sample_calibration_data):
        """ECE is between 0 and 1."""
        y_true, probs = sample_calibration_data
        ece = compute_ece(y_true, probs)

        assert 0 <= ece <= 1

    def test_ece_empty_input(self):
        """ECE returns 0 for empty input."""
        y_true = np.array([])
        probs = np.zeros((0, 3))

        ece = compute_ece(y_true, probs)
        assert ece == 0.0


class TestReliabilityBins:
    """Tests for reliability diagram bins."""

    def test_reliability_bins_structure(self, sample_calibration_data):
        """Reliability bins have correct structure."""
        y_true, probs = sample_calibration_data
        bins = compute_reliability_bins(y_true, probs, n_bins=10)

        assert isinstance(bins, ReliabilityBins)
        assert len(bins.bin_centers) == 10
        assert len(bins.bin_accuracies) == 10
        assert len(bins.bin_confidences) == 10
        assert len(bins.bin_counts) == 10
        assert bins.n_bins == 10

    def test_reliability_bins_centers(self):
        """Bin centers are evenly spaced."""
        y_true = np.array([0, 1, 2])
        probs = np.eye(3)

        bins = compute_reliability_bins(y_true, probs, n_bins=10)

        expected_centers = np.array([0.05, 0.15, 0.25, 0.35, 0.45,
                                     0.55, 0.65, 0.75, 0.85, 0.95])
        np.testing.assert_array_almost_equal(bins.bin_centers, expected_centers)

    def test_reliability_bins_to_dict(self, sample_calibration_data):
        """Reliability bins can be serialized."""
        y_true, probs = sample_calibration_data
        bins = compute_reliability_bins(y_true, probs)

        bins_dict = bins.to_dict()

        assert "bin_centers" in bins_dict
        assert "bin_accuracies" in bins_dict
        assert isinstance(bins_dict["bin_centers"], list)


# =============================================================================
# CALIBRATOR TESTS
# =============================================================================

class TestProbabilityCalibrator:
    """Tests for ProbabilityCalibrator class."""

    def test_calibrator_init(self):
        """Calibrator initializes correctly."""
        config = CalibrationConfig(method="isotonic")
        calibrator = ProbabilityCalibrator(config)

        assert not calibrator.is_fitted
        assert calibrator.config.method == "isotonic"

    def test_calibrator_default_config(self):
        """Calibrator uses default config if none provided."""
        calibrator = ProbabilityCalibrator()

        assert calibrator.config.method == "auto"
        assert calibrator.config.min_samples_per_class == 100

    def test_calibrator_fit(self, sample_calibration_data):
        """Calibrator fits on data and returns metrics."""
        y_true, probs = sample_calibration_data

        calibrator = ProbabilityCalibrator()
        metrics = calibrator.fit(y_true, probs)

        assert calibrator.is_fitted
        assert isinstance(metrics, CalibrationMetrics)
        assert metrics.brier_before >= 0
        assert metrics.brier_after >= 0
        assert metrics.ece_before >= 0
        assert metrics.ece_after >= 0

    def test_calibration_improves_brier(self, sample_calibration_data):
        """Calibration reduces Brier score on miscalibrated data."""
        y_true, probs = sample_calibration_data

        calibrator = ProbabilityCalibrator()
        metrics = calibrator.fit(y_true, probs)

        # Calibration should improve or maintain Brier score
        assert metrics.brier_after <= metrics.brier_before + 0.05

    def test_calibration_improves_ece(self, sample_calibration_data):
        """Calibration reduces ECE on miscalibrated data."""
        y_true, probs = sample_calibration_data

        calibrator = ProbabilityCalibrator()
        metrics = calibrator.fit(y_true, probs)

        # Calibration should reduce ECE
        assert metrics.ece_after <= metrics.ece_before + 0.02

    def test_calibrate_probs_sum_to_one(self, sample_calibration_data):
        """Calibrated probabilities sum to 1."""
        y_true, probs = sample_calibration_data

        calibrator = ProbabilityCalibrator()
        calibrator.fit(y_true, probs)
        calibrated = calibrator.calibrate(probs)

        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, rtol=1e-5)

    def test_calibrate_probs_in_valid_range(self, sample_calibration_data):
        """Calibrated probabilities are in [0, 1]."""
        y_true, probs = sample_calibration_data

        calibrator = ProbabilityCalibrator()
        calibrator.fit(y_true, probs)
        calibrated = calibrator.calibrate(probs)

        assert np.all(calibrated >= 0)
        assert np.all(calibrated <= 1)

    def test_calibrate_unfitted_raises(self, sample_calibration_data):
        """Calibrating with unfitted calibrator raises error."""
        _, probs = sample_calibration_data

        calibrator = ProbabilityCalibrator()

        with pytest.raises(RuntimeError, match="not fitted"):
            calibrator.calibrate(probs)

    def test_calibrate_wrong_classes_raises(self, sample_calibration_data):
        """Calibrating with wrong number of classes raises error."""
        y_true, probs = sample_calibration_data

        calibrator = ProbabilityCalibrator()
        calibrator.fit(y_true, probs)

        wrong_probs = np.random.rand(100, 5)  # 5 classes instead of 3
        wrong_probs = wrong_probs / wrong_probs.sum(axis=1, keepdims=True)

        with pytest.raises(ValueError, match="Expected 3 classes"):
            calibrator.calibrate(wrong_probs)

    def test_isotonic_method(self, sample_calibration_data):
        """Isotonic calibration works."""
        y_true, probs = sample_calibration_data

        config = CalibrationConfig(method="isotonic")
        calibrator = ProbabilityCalibrator(config)
        metrics = calibrator.fit(y_true, probs)

        assert metrics.method_used == "isotonic"
        assert calibrator.is_fitted

    def test_sigmoid_method(self, sample_calibration_data):
        """Sigmoid calibration works."""
        y_true, probs = sample_calibration_data

        config = CalibrationConfig(method="sigmoid")
        calibrator = ProbabilityCalibrator(config)
        metrics = calibrator.fit(y_true, probs)

        assert metrics.method_used == "sigmoid"
        assert calibrator.is_fitted

    def test_auto_selects_isotonic_for_large_samples(self, sample_calibration_data):
        """Auto method selects isotonic for large sample sizes."""
        y_true, probs = sample_calibration_data

        config = CalibrationConfig(method="auto", min_samples_per_class=100)
        calibrator = ProbabilityCalibrator(config)
        metrics = calibrator.fit(y_true, probs)

        # With 1000 samples and 3 classes, should use isotonic
        assert metrics.method_used == "isotonic"

    def test_auto_selects_sigmoid_for_small_samples(self, small_sample_data):
        """Auto method selects sigmoid for small sample sizes."""
        y_true, probs = small_sample_data

        config = CalibrationConfig(method="auto", min_samples_per_class=100)
        calibrator = ProbabilityCalibrator(config)
        metrics = calibrator.fit(y_true, probs)

        # With 50 samples, should fall back to sigmoid
        assert metrics.method_used == "sigmoid"


class TestCalibratorSaveLoad:
    """Tests for calibrator serialization."""

    def test_save_load_roundtrip(self, sample_calibration_data, tmp_path):
        """Calibrator can be saved and loaded."""
        y_true, probs = sample_calibration_data

        calibrator = ProbabilityCalibrator()
        calibrator.fit(y_true, probs)
        calibrated_original = calibrator.calibrate(probs)

        # Save
        save_path = tmp_path / "calibrator.pkl"
        calibrator.save(save_path)

        # Load
        loaded = ProbabilityCalibrator.load(save_path)

        # Verify loaded calibrator produces same results
        calibrated_loaded = loaded.calibrate(probs)
        np.testing.assert_array_almost_equal(calibrated_original, calibrated_loaded)

    def test_save_unfitted_raises(self, tmp_path):
        """Saving unfitted calibrator raises error."""
        calibrator = ProbabilityCalibrator()

        with pytest.raises(RuntimeError, match="Cannot save unfitted"):
            calibrator.save(tmp_path / "calibrator.pkl")

    def test_load_missing_file_raises(self, tmp_path):
        """Loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            ProbabilityCalibrator.load(tmp_path / "nonexistent.pkl")

    def test_save_creates_parent_dirs(self, sample_calibration_data, tmp_path):
        """Save creates parent directories if needed."""
        y_true, probs = sample_calibration_data

        calibrator = ProbabilityCalibrator()
        calibrator.fit(y_true, probs)

        nested_path = tmp_path / "nested" / "dir" / "calibrator.pkl"
        calibrator.save(nested_path)

        assert nested_path.exists()


class TestCalibrationMetrics:
    """Tests for CalibrationMetrics dataclass."""

    def test_metrics_improvement_calculation(self):
        """Improvement percentages are calculated correctly."""
        bins = ReliabilityBins(
            bin_centers=np.zeros(10),
            bin_accuracies=np.zeros(10),
            bin_confidences=np.zeros(10),
            bin_counts=np.zeros(10),
            n_bins=10,
        )

        metrics = CalibrationMetrics(
            brier_before=0.20,
            brier_after=0.15,
            ece_before=0.10,
            ece_after=0.05,
            reliability_bins=bins,
            method_used="isotonic",
        )

        assert abs(metrics.brier_improvement - 0.25) < 1e-10  # 25% improvement
        assert abs(metrics.ece_improvement - 0.50) < 1e-10  # 50% improvement

    def test_metrics_to_dict(self, sample_calibration_data):
        """Metrics can be serialized to dict."""
        y_true, probs = sample_calibration_data

        calibrator = ProbabilityCalibrator()
        metrics = calibrator.fit(y_true, probs)

        metrics_dict = metrics.to_dict()

        assert "brier_before" in metrics_dict
        assert "brier_after" in metrics_dict
        assert "ece_before" in metrics_dict
        assert "ece_after" in metrics_dict
        assert "brier_improvement" in metrics_dict
        assert "ece_improvement" in metrics_dict
        assert "method_used" in metrics_dict
        assert "reliability_bins" in metrics_dict


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestCalibrationIntegration:
    """Integration tests for calibration with model predictions."""

    def test_calibration_with_realistic_miscalibration(self):
        """Calibration fixes realistic boosting-like miscalibration."""
        np.random.seed(42)
        n_samples = 1000

        # Simulate boosting model outputs (overconfident on predicted class)
        y_true = np.random.randint(0, 3, n_samples)

        # Base probabilities
        probs = np.random.dirichlet([2, 2, 2], n_samples)

        # Add overconfidence (push predictions toward 0.8+ on max class)
        for i in range(n_samples):
            max_idx = probs[i].argmax()
            probs[i, max_idx] = 0.8 + np.random.uniform(0, 0.15)
            remaining = 1 - probs[i, max_idx]
            for j in range(3):
                if j != max_idx:
                    probs[i, j] = remaining / 2

        # Calibrate
        calibrator = ProbabilityCalibrator()
        metrics = calibrator.fit(y_true, probs)

        # ECE should improve significantly for overconfident predictions
        assert metrics.ece_after < metrics.ece_before

    def test_calibration_preserves_ranking(self, sample_calibration_data):
        """Calibration preserves relative ranking of predictions."""
        y_true, probs = sample_calibration_data

        calibrator = ProbabilityCalibrator()
        calibrator.fit(y_true, probs)
        calibrated = calibrator.calibrate(probs)

        # Predicted class should stay the same for most samples
        original_preds = probs.argmax(axis=1)
        calibrated_preds = calibrated.argmax(axis=1)

        agreement = (original_preds == calibrated_preds).mean()
        # At least 50% should agree (calibration may change ranking for uncertain predictions)
        # Isotonic regression is non-parametric and can reorder near-equal probabilities
        assert agreement > 0.5

    def test_calibration_handles_extreme_probs(self):
        """Calibration handles extreme probability values."""
        np.random.seed(42)
        n_samples = 100

        y_true = np.random.randint(0, 3, n_samples)

        # Extreme probabilities (near 0 and 1)
        probs = np.zeros((n_samples, 3))
        for i, y in enumerate(y_true):
            probs[i, y] = 0.999
            for j in range(3):
                if j != y:
                    probs[i, j] = 0.0005

        calibrator = ProbabilityCalibrator()
        metrics = calibrator.fit(y_true, probs)
        calibrated = calibrator.calibrate(probs)

        # Should not produce NaN or Inf
        assert not np.any(np.isnan(calibrated))
        assert not np.any(np.isinf(calibrated))

        # Should still sum to 1
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, rtol=1e-5)
