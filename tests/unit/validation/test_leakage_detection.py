"""
Tests for feature-label leakage detection.

Verifies that correlation-based leakage detection correctly identifies
both clean data and data with obvious leakage, while handling edge
cases like NaN values and constant features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validation.leakage_detection import (
    LeakageDetectedError,
    LeakageReport,
    check_feature_label_correlation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def clean_data(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Random features and random labels -- no leakage expected."""
    n = 500
    features = pd.DataFrame(
        {
            "feat_a": rng.standard_normal(n),
            "feat_b": rng.standard_normal(n),
            "feat_c": rng.standard_normal(n),
        }
    )
    labels = rng.choice([-1, 0, 1], size=n).astype(float)
    return features, labels


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCleanData:
    """Random features should show no leakage."""

    def test_no_suspicious_features(self, clean_data: tuple[pd.DataFrame, np.ndarray]) -> None:
        features, labels = clean_data
        report = check_feature_label_correlation(features, labels, raise_on_leakage=False)
        assert isinstance(report, LeakageReport)
        assert report.n_suspicious == 0
        assert report.leakage_ratio == pytest.approx(0.0)


class TestPerfectLeakage:
    """A feature that is an exact copy of the labels should be detected."""

    def test_perfect_copy_detected(self, rng: np.random.Generator) -> None:
        n = 500
        labels = rng.choice([-1, 0, 1], size=n).astype(float)
        features = pd.DataFrame(
            {
                "random_feat": rng.standard_normal(n),
                "leaked_feat": labels.copy(),
            }
        )
        report = check_feature_label_correlation(features, labels, raise_on_leakage=False)
        assert report.n_suspicious >= 1
        suspicious_names = report.get_suspicious_feature_names()
        assert "leaked_feat" in suspicious_names

    def test_raises_when_blocking(self, rng: np.random.Generator) -> None:
        n = 500
        labels = rng.choice([-1, 0, 1], size=n).astype(float)
        features = pd.DataFrame({"leaked": labels.copy()})
        with pytest.raises(LeakageDetectedError):
            check_feature_label_correlation(features, labels, raise_on_leakage=True)


class TestNearPerfectCorrelation:
    """Feature that is labels + small noise should still be caught."""

    def test_noisy_copy_detected(self, rng: np.random.Generator) -> None:
        n = 1000
        labels = rng.choice([-1, 0, 1], size=n).astype(float)
        noise = rng.standard_normal(n) * 0.01
        features = pd.DataFrame({"near_leaked": labels + noise})
        report = check_feature_label_correlation(
            features,
            labels,
            correlation_threshold=0.5,
            raise_on_leakage=False,
        )
        assert report.n_suspicious >= 1
        assert report.suspicious_features[0].feature_name == "near_leaked"


class TestNaNHandling:
    """NaN values in features should not cause crashes."""

    def test_nan_features_no_crash(self, rng: np.random.Generator) -> None:
        n = 500
        labels = rng.choice([-1, 0, 1], size=n).astype(float)
        feat = rng.standard_normal(n)
        # Inject NaN in 20% of values
        nan_idx = rng.choice(n, size=n // 5, replace=False)
        feat[nan_idx] = np.nan
        features = pd.DataFrame({"nan_feat": feat})
        # Should not raise
        report = check_feature_label_correlation(features, labels, raise_on_leakage=False)
        assert isinstance(report, LeakageReport)

    def test_all_nan_feature(self) -> None:
        """A fully NaN feature should be handled gracefully."""
        n = 500
        labels = np.zeros(n)
        features = pd.DataFrame({"all_nan": np.full(n, np.nan)})
        report = check_feature_label_correlation(features, labels, raise_on_leakage=False)
        assert report.n_suspicious == 0


class TestConstantFeature:
    """Constant features should not be flagged as suspicious."""

    def test_constant_not_suspicious(self, rng: np.random.Generator) -> None:
        n = 500
        labels = rng.choice([-1, 0, 1], size=n).astype(float)
        features = pd.DataFrame({"const_feat": np.ones(n)})
        report = check_feature_label_correlation(features, labels, raise_on_leakage=False)
        assert report.n_suspicious == 0
        # The constant feature should have 0.0 correlation
        result = report.all_results[0]
        assert result.correlation == pytest.approx(0.0)
        assert result.details.get("note") == "constant feature"


class TestReportAPI:
    """Verify LeakageReport exposes expected attributes."""

    def test_report_to_dict(self, clean_data: tuple[pd.DataFrame, np.ndarray]) -> None:
        features, labels = clean_data
        report = check_feature_label_correlation(features, labels, raise_on_leakage=False)
        d = report.to_dict()
        assert "n_features" in d
        assert "n_suspicious" in d
        assert "leakage_ratio" in d
        assert "summary" in d
