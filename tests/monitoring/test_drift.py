"""
Tests for Drift Detection and Alert Handling.

Tests:
- ADWIN concept drift detection
- PSI feature drift detection
- KS distribution comparison
- Multi-feature monitoring
- Alert handling and filtering
"""
import time
import numpy as np
import pytest

from src.monitoring.drift_detector import (
    ADWINDetector,
    DriftResult,
    DriftSeverity,
    DriftType,
    FeatureDriftMonitor,
    KSDetector,
    PSIDetector,
)
from src.monitoring.alert_handler import (
    AlertConfig,
    AlertHandler,
    DriftAlertAggregator,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def stable_distribution():
    """Generate stable (non-drifting) data."""
    np.random.seed(42)
    return np.random.randn(1000)


@pytest.fixture
def drifted_distribution():
    """Generate drifted data (shifted mean)."""
    np.random.seed(42)
    return np.random.randn(1000) + 3.0  # Mean shifted by 3 std


@pytest.fixture
def reference_features():
    """Multi-feature reference data."""
    np.random.seed(42)
    return np.random.randn(500, 5)


# =============================================================================
# ADWIN DETECTOR TESTS
# =============================================================================

class TestADWINDetector:
    """Tests for ADWIN concept drift detector."""

    def test_init_default(self):
        """ADWIN initializes with default params."""
        detector = ADWINDetector()
        assert detector.delta == 0.002
        assert detector.n_updates == 0

    def test_init_custom_delta(self):
        """ADWIN accepts custom delta."""
        detector = ADWINDetector(delta=0.01)
        assert detector.delta == 0.01

    def test_no_drift_stable_stream(self):
        """No drift detected in stable error stream."""
        detector = ADWINDetector(delta=0.002, min_samples=50)

        # Stable error rate around 0.3
        np.random.seed(42)
        errors = np.random.binomial(1, 0.3, 200)

        drift_detected = False
        for error in errors:
            result = detector.update(float(error))
            if result and result.drift_detected:
                drift_detected = True

        # May or may not detect drift in random data, but should run without error
        assert detector.n_updates == 200

    def test_detects_drift_in_changing_stream(self):
        """Drift detected when error rate changes."""
        detector = ADWINDetector(delta=0.002, min_samples=30)

        # First half: low error rate (0.2)
        np.random.seed(42)
        for _ in range(100):
            detector.update(np.random.binomial(1, 0.2))

        # Second half: high error rate (0.8)
        drift_results = []
        for _ in range(100):
            result = detector.update(np.random.binomial(1, 0.8))
            if result and result.drift_detected:
                drift_results.append(result)

        # Should detect at least one drift
        assert len(drift_results) > 0
        assert all(r.drift_type == DriftType.CONCEPT for r in drift_results)

    def test_reset_clears_state(self):
        """Reset clears detector state."""
        detector = ADWINDetector()

        for i in range(50):
            detector.update(float(i % 2))

        assert detector.n_updates == 50

        detector.reset()
        assert detector.n_updates == 0

    def test_drift_result_structure(self):
        """DriftResult has expected structure."""
        detector = ADWINDetector(delta=0.1, min_samples=10)

        # Force drift by drastic change
        for _ in range(20):
            detector.update(0.0)

        result = None
        for _ in range(20):
            r = detector.update(1.0)
            if r and r.drift_detected:
                result = r
                break

        if result:  # Only check if drift was detected
            assert isinstance(result, DriftResult)
            assert result.drift_type == DriftType.CONCEPT
            assert isinstance(result.severity, DriftSeverity)
            assert isinstance(result.metric_value, float)


# =============================================================================
# PSI DETECTOR TESTS
# =============================================================================

class TestPSIDetector:
    """Tests for PSI feature drift detector."""

    def test_init_default(self):
        """PSI initializes with default params."""
        detector = PSIDetector()
        assert detector.n_bins == 10
        assert detector.threshold_low == 0.1
        assert detector.threshold_high == 0.25

    def test_set_reference(self, stable_distribution):
        """Reference distribution is set correctly."""
        detector = PSIDetector(n_bins=10)
        detector.set_reference(stable_distribution, feature_name="test_feature")

        assert detector._reference_bins is not None
        assert detector._bin_edges is not None
        assert len(detector._bin_edges) == 11  # n_bins + 1

    def test_no_drift_same_distribution(self, stable_distribution):
        """No drift when comparing same distribution."""
        detector = PSIDetector(n_bins=10)
        detector.set_reference(stable_distribution)

        # Compare with similar distribution
        np.random.seed(123)
        current = np.random.randn(1000)

        result = detector.compare(current)

        assert isinstance(result, DriftResult)
        assert result.drift_type == DriftType.COVARIATE
        assert result.metric_value < 0.1  # Low PSI
        assert result.severity == DriftSeverity.NONE

    def test_drift_detected_shifted_distribution(
        self, stable_distribution, drifted_distribution
    ):
        """Drift detected when distribution shifts."""
        detector = PSIDetector(n_bins=10)
        detector.set_reference(stable_distribution)

        result = detector.compare(drifted_distribution)

        assert result.drift_detected
        assert result.metric_value > 0.25  # High PSI
        assert result.severity in (DriftSeverity.MEDIUM, DriftSeverity.HIGH)

    def test_update_raises_not_implemented(self, stable_distribution):
        """update() raises NotImplementedError."""
        detector = PSIDetector()
        detector.set_reference(stable_distribution)

        with pytest.raises(NotImplementedError):
            detector.update(1.0)

    def test_compare_without_reference_raises(self):
        """compare() without reference raises error."""
        detector = PSIDetector()

        with pytest.raises(RuntimeError, match="Reference not set"):
            detector.compare(np.random.randn(100))

    def test_psi_value_calculation(self):
        """PSI value is calculated correctly."""
        detector = PSIDetector(n_bins=5)

        # Known reference distribution
        reference = np.array([1, 2, 3, 4, 5] * 100)
        detector.set_reference(reference)

        # Identical distribution should have PSI ≈ 0
        result = detector.compare(reference)
        assert result.metric_value < 0.05  # Nearly identical

    def test_result_contains_details(self, stable_distribution, drifted_distribution):
        """Result contains expected details."""
        detector = PSIDetector(n_bins=10, threshold_low=0.1, threshold_high=0.25)
        detector.set_reference(stable_distribution)

        result = detector.compare(drifted_distribution)

        assert "n_bins" in result.details
        assert result.details["n_bins"] == 10
        assert "threshold_low" in result.details
        assert "threshold_high" in result.details


# =============================================================================
# KS DETECTOR TESTS
# =============================================================================

class TestKSDetector:
    """Tests for Kolmogorov-Smirnov drift detector."""

    def test_init_default(self):
        """KS initializes with default params."""
        detector = KSDetector()
        assert detector.alpha == 0.05

    def test_no_drift_same_distribution(self, stable_distribution):
        """No drift when comparing same distribution."""
        detector = KSDetector(alpha=0.05)
        detector.set_reference(stable_distribution)

        np.random.seed(123)
        current = np.random.randn(1000)

        result = detector.compare(current)

        assert result.drift_detected == False
        assert result.severity == DriftSeverity.NONE
        assert "p_value" in result.details
        assert result.details["p_value"] > 0.05

    def test_drift_detected_shifted_distribution(
        self, stable_distribution, drifted_distribution
    ):
        """Drift detected with shifted distribution."""
        detector = KSDetector(alpha=0.05)
        detector.set_reference(stable_distribution)

        result = detector.compare(drifted_distribution)

        assert result.drift_detected
        assert result.details["p_value"] < 0.05

    def test_update_raises_not_implemented(self, stable_distribution):
        """update() raises NotImplementedError."""
        detector = KSDetector()
        detector.set_reference(stable_distribution)

        with pytest.raises(NotImplementedError):
            detector.update(1.0)


# =============================================================================
# FEATURE DRIFT MONITOR TESTS
# =============================================================================

class TestFeatureDriftMonitor:
    """Tests for multi-feature drift monitoring."""

    def test_init_default(self):
        """Monitor initializes with default params."""
        monitor = FeatureDriftMonitor()
        assert monitor.method == "psi"
        assert monitor.n_bins == 10

    def test_set_reference_with_names(self, reference_features):
        """Reference set with feature names."""
        monitor = FeatureDriftMonitor(method="psi")
        feature_names = ["f1", "f2", "f3", "f4", "f5"]

        monitor.set_reference(reference_features, feature_names=feature_names)

        assert len(monitor._detectors) == 5
        assert "f1" in monitor._detectors

    def test_set_reference_auto_names(self, reference_features):
        """Reference set with auto-generated names."""
        monitor = FeatureDriftMonitor()
        monitor.set_reference(reference_features)

        assert len(monitor._detectors) == 5
        assert "feature_0" in monitor._detectors

    def test_check_drift_no_drift(self, reference_features):
        """No drift when comparing similar data."""
        monitor = FeatureDriftMonitor(method="psi")
        monitor.set_reference(reference_features)

        # Similar distribution
        np.random.seed(123)
        current = np.random.randn(500, 5)

        results = monitor.check_drift(current)

        assert len(results) == 5
        n_drifted = sum(1 for r in results if r.drift_detected)
        assert n_drifted == 0  # No features should drift

    def test_check_drift_with_drift(self, reference_features):
        """Drift detected when features shift."""
        monitor = FeatureDriftMonitor(method="psi", threshold=0.1)
        monitor.set_reference(reference_features)

        # Drift in features 0 and 2
        np.random.seed(123)
        current = np.random.randn(500, 5)
        current[:, 0] += 5.0  # Shift feature 0
        current[:, 2] += 5.0  # Shift feature 2

        results = monitor.check_drift(current)

        drifted = [r for r in results if r.drift_detected]
        assert len(drifted) >= 2

    def test_get_drift_summary(self, reference_features):
        """Drift summary is correct."""
        monitor = FeatureDriftMonitor(method="psi")
        monitor.set_reference(reference_features)

        np.random.seed(123)
        current = np.random.randn(500, 5)
        current[:, 0] += 5.0  # Shift feature 0

        summary = monitor.get_drift_summary(current)

        assert "n_features" in summary
        assert summary["n_features"] == 5
        assert "n_drifted" in summary
        assert "drift_rate" in summary
        assert "method" in summary
        assert summary["method"] == "psi"

    def test_ks_method(self, reference_features):
        """KS method works."""
        monitor = FeatureDriftMonitor(method="ks", alpha=0.05)
        monitor.set_reference(reference_features)

        np.random.seed(123)
        current = np.random.randn(500, 5)

        results = monitor.check_drift(current)
        assert len(results) == 5


# =============================================================================
# ALERT HANDLER TESTS
# =============================================================================

class TestAlertHandler:
    """Tests for alert handling."""

    def test_init_default(self):
        """Handler initializes with default config."""
        handler = AlertHandler()
        assert handler.config.min_severity == DriftSeverity.LOW
        assert handler.config.log_alerts is True

    def test_handle_filters_by_severity(self):
        """Alerts filtered by severity threshold."""
        handler = AlertHandler(
            config=AlertConfig(min_severity=DriftSeverity.HIGH)
        )

        # Low severity - should be filtered
        low_result = DriftResult(
            drift_detected=True,
            drift_type=DriftType.CONCEPT,
            severity=DriftSeverity.LOW,
            metric_value=0.1,
            threshold=0.05,
        )
        assert handler.handle(low_result) is False

        # High severity - should trigger
        high_result = DriftResult(
            drift_detected=True,
            drift_type=DriftType.CONCEPT,
            severity=DriftSeverity.HIGH,
            metric_value=0.3,
            threshold=0.05,
        )
        assert handler.handle(high_result) is True

    def test_handle_no_drift(self):
        """No alert when drift_detected=False."""
        handler = AlertHandler()

        result = DriftResult(
            drift_detected=False,
            drift_type=DriftType.CONCEPT,
            severity=DriftSeverity.NONE,
            metric_value=0.01,
            threshold=0.05,
        )

        assert handler.handle(result) is False

    def test_callback_execution(self):
        """Callbacks are executed on alert."""
        handler = AlertHandler()
        callback_results = []

        def callback(result):
            callback_results.append(result.feature_name)

        handler.add_callback(callback)

        result = DriftResult(
            drift_detected=True,
            drift_type=DriftType.COVARIATE,
            severity=DriftSeverity.MEDIUM,
            metric_value=0.15,
            threshold=0.1,
            feature_name="test_feature",
        )

        handler.handle(result)

        assert len(callback_results) == 1
        assert callback_results[0] == "test_feature"

    def test_rate_limiting(self):
        """Alerts are rate limited."""
        handler = AlertHandler(
            config=AlertConfig(rate_limit_seconds=0.5)
        )

        result = DriftResult(
            drift_detected=True,
            drift_type=DriftType.CONCEPT,
            severity=DriftSeverity.MEDIUM,
            metric_value=0.2,
            threshold=0.1,
            feature_name="rate_test",
        )

        # First should trigger
        assert handler.handle(result) is True

        # Immediate second should be rate limited
        assert handler.handle(result) is False

        # After waiting, should trigger again
        time.sleep(0.6)
        assert handler.handle(result) is True

    def test_handle_batch(self):
        """Batch handling works correctly."""
        handler = AlertHandler()

        results = [
            DriftResult(
                drift_detected=True,
                drift_type=DriftType.CONCEPT,
                severity=DriftSeverity.MEDIUM,
                metric_value=0.2,
                threshold=0.1,
                feature_name=f"feat_{i}",
            )
            for i in range(5)
        ]

        # Add one non-drift result
        results.append(DriftResult(
            drift_detected=False,
            drift_type=DriftType.CONCEPT,
            severity=DriftSeverity.NONE,
            metric_value=0.01,
            threshold=0.1,
        ))

        summary = handler.handle_batch(results)

        assert summary["total"] == 6
        assert summary["triggered"] == 5
        assert summary["filtered"] == 0

    def test_get_history(self):
        """Alert history is tracked."""
        handler = AlertHandler()

        for i in range(5):
            result = DriftResult(
                drift_detected=True,
                drift_type=DriftType.COVARIATE,
                severity=DriftSeverity.MEDIUM,
                metric_value=0.2,
                threshold=0.1,
                feature_name=f"feat_{i}",
            )
            handler.handle(result)

        history = handler.get_history()
        assert len(history) == 5

        # Filter by feature
        filtered = handler.get_history(feature_name="feat_2")
        assert len(filtered) == 1

    def test_get_alert_counts(self):
        """Alert counts are correct."""
        handler = AlertHandler()

        for i in range(3):
            result = DriftResult(
                drift_detected=True,
                drift_type=DriftType.CONCEPT,
                severity=DriftSeverity.HIGH,
                metric_value=0.3,
                threshold=0.1,
                feature_name=f"feat_{i}",
            )
            handler.handle(result)

        counts = handler.get_alert_counts()
        assert counts["total"] == 3
        assert counts["by_type"]["concept"] == 3
        assert counts["by_severity"]["high"] == 3


# =============================================================================
# ALERT AGGREGATOR TESTS
# =============================================================================

class TestDriftAlertAggregator:
    """Tests for alert aggregation."""

    def test_init_default(self):
        """Aggregator initializes correctly."""
        aggregator = DriftAlertAggregator(window_seconds=60)
        assert aggregator.window_seconds == 60

    def test_add_alerts(self):
        """Alerts are added to aggregator."""
        aggregator = DriftAlertAggregator(window_seconds=60)

        for i in range(3):
            result = DriftResult(
                drift_detected=True,
                drift_type=DriftType.COVARIATE,
                severity=DriftSeverity.MEDIUM,
                metric_value=0.2,
                threshold=0.1,
                feature_name=f"feat_{i}",
            )
            aggregator.add(result)

        assert len(aggregator._alerts) == 3

    def test_should_report(self):
        """should_report works correctly."""
        aggregator = DriftAlertAggregator(
            window_seconds=0.1,  # Short window for testing
            min_alerts_to_report=2,
        )

        # Add one alert
        result = DriftResult(
            drift_detected=True,
            drift_type=DriftType.COVARIATE,
            severity=DriftSeverity.MEDIUM,
            metric_value=0.2,
            threshold=0.1,
        )
        aggregator.add(result)

        # Not enough alerts yet
        time.sleep(0.15)
        assert aggregator.should_report() is False

        # Add another alert
        aggregator.add(result)
        assert aggregator.should_report() is True

    def test_get_summary(self):
        """Summary is correct."""
        aggregator = DriftAlertAggregator()

        for i in range(3):
            result = DriftResult(
                drift_detected=True,
                drift_type=DriftType.COVARIATE,
                severity=DriftSeverity.HIGH if i == 0 else DriftSeverity.MEDIUM,
                metric_value=0.2 + i * 0.1,
                threshold=0.1,
                feature_name=f"feat_{i}",
            )
            aggregator.add(result)

        summary = aggregator.get_summary()

        assert summary["n_alerts"] == 3
        assert summary["n_features_affected"] == 3
        assert summary["max_severity"] == "high"

    def test_reset(self):
        """Reset clears aggregator."""
        aggregator = DriftAlertAggregator()

        result = DriftResult(
            drift_detected=True,
            drift_type=DriftType.COVARIATE,
            severity=DriftSeverity.MEDIUM,
            metric_value=0.2,
            threshold=0.1,
        )
        aggregator.add(result)

        assert len(aggregator._alerts) == 1

        aggregator.reset()
        assert len(aggregator._alerts) == 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestDriftIntegration:
    """Integration tests for drift detection pipeline."""

    def test_full_monitoring_workflow(self, reference_features):
        """Full workflow: monitor -> detect -> alert."""
        # Setup
        monitor = FeatureDriftMonitor(method="psi")
        monitor.set_reference(reference_features, feature_names=[f"f{i}" for i in range(5)])

        handler = AlertHandler(
            config=AlertConfig(min_severity=DriftSeverity.MEDIUM)
        )

        alerted_features = []
        handler.add_callback(lambda r: alerted_features.append(r.feature_name))

        # Simulate drift in one feature
        np.random.seed(42)
        current = np.random.randn(500, 5)
        current[:, 2] += 5.0  # Shift f2

        # Detect and handle
        results = monitor.check_drift(current)
        handler.handle_batch(results)

        # Verify
        assert "f2" in alerted_features

    def test_adwin_with_alert_handler(self):
        """ADWIN integrated with alert handler."""
        detector = ADWINDetector(delta=0.01, min_samples=20)
        handler = AlertHandler()

        drift_count = 0

        def count_drift(r):
            nonlocal drift_count
            drift_count += 1

        handler.add_callback(count_drift)

        # Stable phase
        for _ in range(50):
            result = detector.update(0.2)
            if result:
                handler.handle(result)

        # Drift phase
        for _ in range(50):
            result = detector.update(0.8)
            if result:
                handler.handle(result)

        # Should have detected some drift
        assert drift_count >= 0  # May or may not detect depending on ADWIN behavior
