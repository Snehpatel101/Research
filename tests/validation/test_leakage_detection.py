"""
Tests for feature-label leakage detection.

Tests cover:
- Feature-label correlation checks
- Temporal leakage detection
- Information leakage via mutual information
- Comprehensive leakage check
"""

import numpy as np
import pandas as pd
import pytest

from src.validation.leakage_detection import (
    LeakageCheckResult,
    LeakageReport,
    check_feature_label_correlation,
    check_information_leakage,
    check_temporal_leakage,
    comprehensive_leakage_check,
)


class TestCheckFeatureLabelCorrelation:
    """Tests for feature-label correlation check."""

    def test_no_leakage_random_features(self):
        """Random features should not show leakage."""
        np.random.seed(42)
        n = 500
        features = pd.DataFrame(
            {
                "feature_1": np.random.randn(n),
                "feature_2": np.random.randn(n),
                "feature_3": np.random.randn(n),
            }
        )
        labels = np.random.randint(-1, 2, n)

        report = check_feature_label_correlation(
            features, labels, correlation_threshold=0.5
        )

        assert isinstance(report, LeakageReport)
        assert report.n_suspicious == 0  # Random features should not be suspicious
        assert len(report.suspicious_features) == 0

    def test_detects_perfect_correlation(self):
        """Should detect feature that perfectly predicts label."""
        np.random.seed(42)
        n = 200
        labels = np.random.randint(-1, 2, n)
        features = pd.DataFrame(
            {
                "random": np.random.randn(n),
                "leaky": labels.astype(float) + np.random.randn(n) * 0.01,  # Near-perfect
            }
        )

        report = check_feature_label_correlation(
            features, labels, correlation_threshold=0.5
        )

        assert report.n_suspicious > 0
        suspicious_names = report.get_suspicious_feature_names()
        assert "leaky" in suspicious_names

    def test_all_results_computed(self):
        """All feature results should be in all_results."""
        np.random.seed(42)
        n = 200
        features = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
        labels = np.random.randint(-1, 2, n)

        report = check_feature_label_correlation(features, labels)

        assert len(report.all_results) == 2
        assert all(isinstance(r, LeakageCheckResult) for r in report.all_results)

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        np.random.seed(42)
        n = 300
        labels = np.random.randint(0, 2, n)
        # Create feature with moderate correlation
        features = pd.DataFrame({"moderate": labels * 0.5 + np.random.randn(n) * 0.3})

        # Low threshold should flag more
        report_low = check_feature_label_correlation(
            features, labels, correlation_threshold=0.1
        )

        # High threshold should flag less
        report_high = check_feature_label_correlation(
            features, labels, correlation_threshold=0.9
        )

        assert report_low.correlation_threshold == 0.1
        assert report_high.correlation_threshold == 0.9

    def test_handles_nan_values(self):
        """Should handle NaN values in features."""
        np.random.seed(42)
        n = 100
        features = pd.DataFrame(
            {
                "with_nan": np.where(
                    np.random.rand(n) > 0.9, np.nan, np.random.randn(n)
                ),
                "clean": np.random.randn(n),
            }
        )
        labels = np.random.randint(-1, 2, n)

        # Should not raise
        report = check_feature_label_correlation(features, labels)
        assert isinstance(report, LeakageReport)

    def test_summary_on_leakage(self):
        """Should provide summary when leakage detected."""
        np.random.seed(42)
        n = 200
        labels = np.random.randint(-1, 2, n)
        features = pd.DataFrame({"leaky": labels.astype(float)})

        report = check_feature_label_correlation(
            features, labels, correlation_threshold=0.5
        )

        assert report.n_suspicious > 0
        assert "WARNING" in report.summary

    def test_leakage_ratio(self):
        """leakage_ratio property should work."""
        np.random.seed(42)
        n = 200
        labels = np.random.randint(-1, 2, n)
        features = pd.DataFrame(
            {
                "leaky": labels.astype(float),
                "random": np.random.randn(n),
            }
        )

        report = check_feature_label_correlation(
            features, labels, correlation_threshold=0.5
        )

        assert 0 <= report.leakage_ratio <= 1
        assert report.leakage_ratio == report.n_suspicious / report.n_features


class TestCheckTemporalLeakage:
    """Tests for temporal leakage detection."""

    def test_returns_leakage_report(self):
        """Should return LeakageReport."""
        np.random.seed(42)
        n = 500
        features = pd.DataFrame(
            {
                "f1": np.random.randn(n),
                "f2": np.random.randn(n),
            }
        )
        labels = np.random.randint(0, 2, n)

        report = check_temporal_leakage(features, labels, max_lag=5)

        assert isinstance(report, LeakageReport)
        assert report.n_features == 2

    def test_detects_forward_looking_feature(self):
        """Should detect feature using future information."""
        np.random.seed(42)
        n = 500

        # Labels based on future movement
        prices = np.random.randn(n).cumsum() + 100
        labels = (np.roll(prices, -1) > prices).astype(int)  # Next price > current

        # Leaky feature: uses future price
        features = pd.DataFrame(
            {
                "future_price": np.roll(prices, -1),  # Forward looking!
                "past_price": np.roll(prices, 1),
            }
        )

        report = check_temporal_leakage(features, labels, max_lag=3)

        # Future-looking feature should be flagged
        assert report.n_features == 2

    def test_lag_in_details(self):
        """Should include lag info in results."""
        np.random.seed(42)
        n = 300
        features = pd.DataFrame({"f1": np.random.randn(n)})
        labels = np.random.randint(0, 2, n)

        report = check_temporal_leakage(features, labels, max_lag=5)

        # Check that details contain lag info
        assert len(report.all_results) > 0
        assert "max_lag" in report.all_results[0].details

    def test_handles_constant_feature(self):
        """Should handle constant features."""
        np.random.seed(42)
        n = 200
        features = pd.DataFrame(
            {
                "constant": np.ones(n),
                "varying": np.random.randn(n),
            }
        )
        labels = np.random.randint(0, 2, n)

        report = check_temporal_leakage(features, labels)

        assert report.n_features == 2


class TestCheckInformationLeakage:
    """Tests for information leakage via mutual information."""

    def test_returns_leakage_report(self):
        """Should return LeakageReport."""
        np.random.seed(42)
        n = 500
        features = pd.DataFrame(
            {"random_1": np.random.randn(n), "random_2": np.random.randn(n)}
        )
        labels = np.random.randint(0, 2, n)

        report = check_information_leakage(features, labels, mi_threshold=0.5)

        assert isinstance(report, LeakageReport)
        assert report.n_features == 2

    def test_detects_high_mi_feature(self):
        """Should detect feature with high mutual information."""
        np.random.seed(42)
        n = 500
        labels = np.random.randint(0, 2, n)

        # Create feature that's highly informative about label
        features = pd.DataFrame(
            {
                "informative": labels + np.random.randn(n) * 0.1,
                "random": np.random.randn(n),
            }
        )

        report = check_information_leakage(features, labels, mi_threshold=0.3)

        assert report.n_suspicious > 0

    def test_mi_in_details(self):
        """Mutual information values should be in details."""
        np.random.seed(42)
        n = 200
        features = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
        labels = np.random.randint(0, 2, n)

        report = check_information_leakage(features, labels)

        # Check details contain MI info
        assert len(report.all_results) == 2
        for result in report.all_results:
            assert "mutual_information_raw" in result.details
            assert "mutual_information_normalized" in result.details

    def test_binary_labels(self):
        """Should work with binary labels."""
        np.random.seed(42)
        n = 200
        features = pd.DataFrame({"f1": np.random.randn(n)})
        labels = np.random.randint(0, 2, n)

        report = check_information_leakage(features, labels)

        assert isinstance(report, LeakageReport)

    def test_multiclass_labels(self):
        """Should work with multiclass labels."""
        np.random.seed(42)
        n = 200
        features = pd.DataFrame({"f1": np.random.randn(n)})
        labels = np.random.randint(-1, 2, n)  # -1, 0, 1

        report = check_information_leakage(features, labels)

        assert isinstance(report, LeakageReport)


class TestComprehensiveLeakageCheck:
    """Tests for comprehensive_leakage_check function."""

    def test_runs_all_checks(self):
        """Should run correlation, temporal, and MI checks."""
        np.random.seed(42)
        n = 300
        features = pd.DataFrame(
            {"f1": np.random.randn(n), "f2": np.random.randn(n), "f3": np.random.randn(n)}
        )
        labels = np.random.randint(-1, 2, n)

        results = comprehensive_leakage_check(features, labels)

        assert isinstance(results, dict)
        assert "correlation" in results
        assert "temporal" in results
        assert "mutual_information" in results

        # Each should be a LeakageReport
        for key, report in results.items():
            assert isinstance(report, LeakageReport)

    def test_custom_thresholds(self):
        """Custom thresholds should be passed to individual checks."""
        np.random.seed(42)
        n = 200
        features = pd.DataFrame({"f1": np.random.randn(n)})
        labels = np.random.randint(-1, 2, n)

        results = comprehensive_leakage_check(
            features,
            labels,
            correlation_threshold=0.3,
            mi_threshold=0.2,
            max_lag=10,
        )

        # Verify thresholds were used
        assert results["correlation"].correlation_threshold == 0.3
        # temporal uses temporal_threshold
        # mutual_information uses mi_threshold

    def test_all_reports_have_results(self):
        """All reports should have results."""
        np.random.seed(42)
        n = 200
        features = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
        labels = np.random.randint(-1, 2, n)

        results = comprehensive_leakage_check(features, labels)

        for key, report in results.items():
            assert report.n_features == 2


class TestLeakageCheckResult:
    """Tests for LeakageCheckResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = LeakageCheckResult(
            feature_name="test_feature",
            correlation=0.8,
            p_value=0.001,
            is_suspicious=True,
            leakage_score=0.8,
            details={"key": "value"},
        )

        result_dict = result.to_dict()

        assert result_dict["feature_name"] == "test_feature"
        assert result_dict["correlation"] == 0.8
        assert result_dict["p_value"] == 0.001
        assert result_dict["is_suspicious"] is True
        assert result_dict["leakage_score"] == 0.8
        assert result_dict["details"]["key"] == "value"


class TestLeakageReport:
    """Tests for LeakageReport dataclass."""

    def test_leakage_ratio_property(self):
        """leakage_ratio should be computed correctly."""
        results = [
            LeakageCheckResult("f1", 0.9, 0.001, True, 0.9, {}),
            LeakageCheckResult("f2", 0.1, 0.5, False, 0.0, {}),
        ]
        report = LeakageReport(
            n_features=2,
            n_suspicious=1,
            suspicious_features=[results[0]],
            all_results=results,
            correlation_threshold=0.5,
            summary="Test",
        )

        assert report.leakage_ratio == 0.5  # 1/2

    def test_get_suspicious_feature_names(self):
        """Should return suspicious feature names."""
        suspicious = [
            LeakageCheckResult("f1", 0.9, 0.001, True, 0.9, {}),
            LeakageCheckResult("f2", 0.8, 0.001, True, 0.8, {}),
        ]
        report = LeakageReport(
            n_features=3,
            n_suspicious=2,
            suspicious_features=suspicious,
            all_results=suspicious,
            correlation_threshold=0.5,
            summary="Test",
        )

        names = report.get_suspicious_feature_names()

        assert names == ["f1", "f2"]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        suspicious = [LeakageCheckResult("f1", 0.9, 0.001, True, 0.9, {})]
        report = LeakageReport(
            n_features=2,
            n_suspicious=1,
            suspicious_features=suspicious,
            all_results=suspicious,
            correlation_threshold=0.5,
            summary="Test summary",
        )

        report_dict = report.to_dict()

        assert report_dict["n_features"] == 2
        assert report_dict["n_suspicious"] == 1
        assert report_dict["leakage_ratio"] == 0.5
        assert report_dict["correlation_threshold"] == 0.5
        assert len(report_dict["suspicious_features"]) == 1
        assert report_dict["summary"] == "Test summary"

    def test_zero_features(self):
        """Should handle zero features case."""
        report = LeakageReport(
            n_features=0,
            n_suspicious=0,
            suspicious_features=[],
            all_results=[],
            correlation_threshold=0.5,
            summary="No features",
        )

        assert report.leakage_ratio == 0.0
