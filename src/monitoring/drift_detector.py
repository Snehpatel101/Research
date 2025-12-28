"""
Online Drift Detection for ML Models.

Implements multiple drift detection methods:
- ADWIN (Adaptive Windowing) via river library for concept drift
- PSI (Population Stability Index) for feature distribution drift
- KS (Kolmogorov-Smirnov) test for distribution comparison

Use Cases:
- Monitor prediction accuracy degradation (concept drift)
- Detect input feature distribution shifts (covariate drift)
- Trigger model retraining alerts
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# DRIFT TYPES AND RESULTS
# =============================================================================

class DriftType(Enum):
    """Types of drift that can be detected."""
    CONCEPT = "concept"  # Model performance degradation
    COVARIATE = "covariate"  # Input feature distribution shift
    PREDICTION = "prediction"  # Prediction distribution shift


class DriftSeverity(Enum):
    """Severity levels for drift alerts."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftResult:
    """Result from drift detection."""
    drift_detected: bool
    drift_type: DriftType
    severity: DriftSeverity
    metric_value: float
    threshold: float
    feature_name: Optional[str] = None
    timestamp: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "drift_detected": self.drift_detected,
            "drift_type": self.drift_type.value,
            "severity": self.severity.value,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "feature_name": self.feature_name,
            "timestamp": self.timestamp,
            "details": self.details,
        }


# =============================================================================
# BASE DETECTOR
# =============================================================================

class BaseDriftDetector(ABC):
    """Abstract base class for drift detectors."""

    def __init__(self, name: str = "detector") -> None:
        self.name = name
        self._n_updates = 0

    @abstractmethod
    def update(self, value: float) -> Optional[DriftResult]:
        """Update detector with new value and check for drift."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset detector state."""
        pass

    @property
    def n_updates(self) -> int:
        """Number of updates received."""
        return self._n_updates


# =============================================================================
# ADWIN DETECTOR (Concept Drift)
# =============================================================================

class ADWINDetector(BaseDriftDetector):
    """
    ADWIN (ADaptive WINdowing) detector for concept drift.

    Uses the river library's ADWIN implementation for efficient
    online drift detection with automatic window sizing.

    Best for:
    - Monitoring error rates over time
    - Detecting accuracy degradation
    - Online learning scenarios

    Example:
        >>> detector = ADWINDetector(delta=0.002)
        >>> for error in error_stream:
        ...     result = detector.update(error)
        ...     if result and result.drift_detected:
        ...         print("Concept drift detected!")
    """

    def __init__(
        self,
        delta: float = 0.002,
        name: str = "adwin",
        min_samples: int = 30,
    ) -> None:
        """
        Initialize ADWIN detector.

        Args:
            delta: Confidence parameter (lower = more sensitive).
                   Typical values: 0.002 (sensitive) to 0.1 (less sensitive)
            name: Detector name for identification
            min_samples: Minimum samples before drift detection activates
        """
        super().__init__(name)
        self.delta = delta
        self.min_samples = min_samples
        self._adwin = None
        self._last_mean = None
        self._init_adwin()

    def _init_adwin(self) -> None:
        """Initialize or reset ADWIN instance."""
        try:
            from river.drift import ADWIN
            self._adwin = ADWIN(delta=self.delta)
        except ImportError:
            logger.warning(
                "river library not installed. ADWIN will use fallback. "
                "Install with: pip install river"
            )
            self._adwin = None
            self._fallback_window: List[float] = []
            self._fallback_max_size = 1000

    def update(self, value: float) -> Optional[DriftResult]:
        """
        Update ADWIN with new value and check for drift.

        Args:
            value: New observation (e.g., prediction error, loss)

        Returns:
            DriftResult if drift detected, None otherwise
        """
        self._n_updates += 1

        if self._adwin is not None:
            # Use river ADWIN
            drift_detected = self._adwin.update(value)
            current_mean = self._adwin.estimation

            if drift_detected and self._n_updates >= self.min_samples:
                severity = self._compute_severity(
                    self._last_mean or current_mean, current_mean
                )
                result = DriftResult(
                    drift_detected=True,
                    drift_type=DriftType.CONCEPT,
                    severity=severity,
                    metric_value=current_mean,
                    threshold=self.delta,
                    details={
                        "window_size": self._adwin.width,
                        "n_updates": self._n_updates,
                        "previous_mean": self._last_mean,
                    },
                )
                self._last_mean = current_mean
                return result

            self._last_mean = current_mean
            return None
        else:
            # Fallback implementation
            return self._fallback_update(value)

    def _fallback_update(self, value: float) -> Optional[DriftResult]:
        """Simple fallback when river not available."""
        self._fallback_window.append(value)
        if len(self._fallback_window) > self._fallback_max_size:
            self._fallback_window.pop(0)

        if len(self._fallback_window) < self.min_samples * 2:
            return None

        # Simple mean comparison between halves
        mid = len(self._fallback_window) // 2
        first_half = np.mean(self._fallback_window[:mid])
        second_half = np.mean(self._fallback_window[mid:])

        # Use t-test for drift detection
        _, p_value = stats.ttest_ind(
            self._fallback_window[:mid],
            self._fallback_window[mid:]
        )

        if p_value < self.delta:
            severity = self._compute_severity(first_half, second_half)
            return DriftResult(
                drift_detected=True,
                drift_type=DriftType.CONCEPT,
                severity=severity,
                metric_value=second_half,
                threshold=self.delta,
                details={
                    "p_value": p_value,
                    "first_half_mean": first_half,
                    "second_half_mean": second_half,
                    "fallback_mode": True,
                },
            )

        return None

    def _compute_severity(
        self, old_mean: float, new_mean: float
    ) -> DriftSeverity:
        """Compute severity based on mean change."""
        if old_mean == 0:
            change = abs(new_mean)
        else:
            change = abs(new_mean - old_mean) / abs(old_mean)

        if change < 0.05:
            return DriftSeverity.LOW
        elif change < 0.15:
            return DriftSeverity.MEDIUM
        elif change < 0.30:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL

    def reset(self) -> None:
        """Reset detector state."""
        self._n_updates = 0
        self._last_mean = None
        self._init_adwin()

    @property
    def current_mean(self) -> Optional[float]:
        """Current estimated mean."""
        if self._adwin is not None:
            return self._adwin.estimation
        elif self._fallback_window:
            return np.mean(self._fallback_window)
        return None

    @property
    def window_size(self) -> int:
        """Current window size."""
        if self._adwin is not None:
            return self._adwin.width
        return len(getattr(self, "_fallback_window", []))


# =============================================================================
# PSI DETECTOR (Feature Drift)
# =============================================================================

class PSIDetector(BaseDriftDetector):
    """
    Population Stability Index (PSI) detector for feature drift.

    PSI measures how much a distribution has shifted from a reference.
    Commonly used in credit scoring and financial modeling.

    Interpretation:
    - PSI < 0.1: No significant shift
    - PSI 0.1-0.25: Moderate shift, investigate
    - PSI > 0.25: Significant shift, action required

    Example:
        >>> detector = PSIDetector(n_bins=10)
        >>> detector.set_reference(reference_data)
        >>> result = detector.compare(current_data)
        >>> if result.drift_detected:
        ...     print(f"Feature drift: PSI={result.metric_value:.3f}")
    """

    def __init__(
        self,
        n_bins: int = 10,
        threshold_low: float = 0.1,
        threshold_high: float = 0.25,
        name: str = "psi",
    ) -> None:
        """
        Initialize PSI detector.

        Args:
            n_bins: Number of bins for histogram comparison
            threshold_low: PSI threshold for low severity (default: 0.1)
            threshold_high: PSI threshold for high severity (default: 0.25)
            name: Detector name
        """
        super().__init__(name)
        self.n_bins = n_bins
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self._reference_bins: Optional[np.ndarray] = None
        self._bin_edges: Optional[np.ndarray] = None
        self._feature_name: Optional[str] = None

    def set_reference(
        self,
        reference_data: np.ndarray,
        feature_name: Optional[str] = None,
    ) -> None:
        """
        Set reference distribution for comparison.

        Args:
            reference_data: Reference (training) data distribution
            feature_name: Optional name for the feature
        """
        reference_data = np.asarray(reference_data).ravel()

        # Create bins from reference data
        self._reference_bins, self._bin_edges = np.histogram(
            reference_data, bins=self.n_bins
        )
        # Convert to proportions and add small epsilon to avoid division by zero
        self._reference_bins = (
            self._reference_bins / len(reference_data)
        ) + 1e-10
        self._feature_name = feature_name

        logger.debug(f"PSI reference set with {len(reference_data)} samples")

    def update(self, value: float) -> Optional[DriftResult]:
        """
        Not used for PSI - use compare() instead.

        PSI is a batch comparison method, not online.
        """
        raise NotImplementedError(
            "PSI uses batch comparison. Use compare() instead of update()."
        )

    def compare(
        self,
        current_data: np.ndarray,
        feature_name: Optional[str] = None,
    ) -> DriftResult:
        """
        Compare current distribution to reference.

        Args:
            current_data: Current data distribution
            feature_name: Optional feature name (overrides constructor)

        Returns:
            DriftResult with PSI value and drift detection
        """
        if self._reference_bins is None or self._bin_edges is None:
            raise RuntimeError(
                "Reference not set. Call set_reference() first."
            )

        current_data = np.asarray(current_data).ravel()
        feature_name = feature_name or self._feature_name

        # Compute current distribution using reference bin edges
        current_bins, _ = np.histogram(current_data, bins=self._bin_edges)
        current_bins = (current_bins / len(current_data)) + 1e-10

        # Calculate PSI
        psi = np.sum(
            (current_bins - self._reference_bins) *
            np.log(current_bins / self._reference_bins)
        )

        # Determine severity
        if psi < self.threshold_low:
            severity = DriftSeverity.NONE
            drift_detected = False
        elif psi < self.threshold_high:
            severity = DriftSeverity.MEDIUM
            drift_detected = True
        else:
            severity = DriftSeverity.HIGH
            drift_detected = True

        self._n_updates += 1

        return DriftResult(
            drift_detected=drift_detected,
            drift_type=DriftType.COVARIATE,
            severity=severity,
            metric_value=psi,
            threshold=self.threshold_low,
            feature_name=feature_name,
            details={
                "n_bins": self.n_bins,
                "reference_size": int(self._reference_bins.sum() * len(current_data)),
                "current_size": len(current_data),
                "threshold_low": self.threshold_low,
                "threshold_high": self.threshold_high,
            },
        )

    def reset(self) -> None:
        """Reset detector state."""
        self._n_updates = 0
        self._reference_bins = None
        self._bin_edges = None
        self._feature_name = None


# =============================================================================
# KS DETECTOR (Distribution Comparison)
# =============================================================================

class KSDetector(BaseDriftDetector):
    """
    Kolmogorov-Smirnov test for distribution drift.

    Non-parametric test that compares two distributions.
    More sensitive to shape differences than PSI.

    Example:
        >>> detector = KSDetector(alpha=0.05)
        >>> detector.set_reference(reference_data)
        >>> result = detector.compare(current_data)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        name: str = "ks",
    ) -> None:
        """
        Initialize KS detector.

        Args:
            alpha: Significance level for the test
            name: Detector name
        """
        super().__init__(name)
        self.alpha = alpha
        self._reference_data: Optional[np.ndarray] = None
        self._feature_name: Optional[str] = None

    def set_reference(
        self,
        reference_data: np.ndarray,
        feature_name: Optional[str] = None,
    ) -> None:
        """Set reference distribution."""
        self._reference_data = np.asarray(reference_data).ravel()
        self._feature_name = feature_name

    def update(self, value: float) -> Optional[DriftResult]:
        """Not used for KS - use compare() instead."""
        raise NotImplementedError(
            "KS uses batch comparison. Use compare() instead of update()."
        )

    def compare(
        self,
        current_data: np.ndarray,
        feature_name: Optional[str] = None,
    ) -> DriftResult:
        """Compare current distribution to reference using KS test."""
        if self._reference_data is None:
            raise RuntimeError("Reference not set. Call set_reference() first.")

        current_data = np.asarray(current_data).ravel()
        feature_name = feature_name or self._feature_name

        # Perform KS test
        statistic, p_value = stats.ks_2samp(
            self._reference_data, current_data
        )

        drift_detected = p_value < self.alpha

        # Determine severity based on p-value
        if p_value >= self.alpha:
            severity = DriftSeverity.NONE
        elif p_value >= self.alpha / 5:
            severity = DriftSeverity.LOW
        elif p_value >= self.alpha / 25:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.HIGH

        self._n_updates += 1

        return DriftResult(
            drift_detected=drift_detected,
            drift_type=DriftType.COVARIATE,
            severity=severity,
            metric_value=statistic,
            threshold=self.alpha,
            feature_name=feature_name,
            details={
                "p_value": p_value,
                "ks_statistic": statistic,
                "alpha": self.alpha,
                "reference_size": len(self._reference_data),
                "current_size": len(current_data),
            },
        )

    def reset(self) -> None:
        """Reset detector state."""
        self._n_updates = 0
        self._reference_data = None
        self._feature_name = None


# =============================================================================
# MULTI-FEATURE DRIFT MONITOR
# =============================================================================

class FeatureDriftMonitor:
    """
    Monitor multiple features for drift simultaneously.

    Combines PSI and KS tests for comprehensive drift detection
    across all input features.

    Example:
        >>> monitor = FeatureDriftMonitor(method="psi")
        >>> monitor.set_reference(X_train, feature_names=feature_cols)
        >>> results = monitor.check_drift(X_current)
        >>> drifted_features = [r.feature_name for r in results if r.drift_detected]
    """

    def __init__(
        self,
        method: str = "psi",
        n_bins: int = 10,
        threshold: float = 0.1,
        alpha: float = 0.05,
    ) -> None:
        """
        Initialize feature drift monitor.

        Args:
            method: Detection method ("psi" or "ks")
            n_bins: Number of bins for PSI
            threshold: PSI threshold or KS alpha
            alpha: Significance level for KS test
        """
        self.method = method
        self.n_bins = n_bins
        self.threshold = threshold
        self.alpha = alpha
        self._detectors: Dict[str, Union[PSIDetector, KSDetector]] = {}
        self._feature_names: List[str] = []

    def set_reference(
        self,
        X_reference: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Set reference distribution for all features.

        Args:
            X_reference: Reference data, shape (n_samples, n_features)
            feature_names: Optional feature names
        """
        X_reference = np.asarray(X_reference)
        if X_reference.ndim == 1:
            X_reference = X_reference.reshape(-1, 1)

        n_features = X_reference.shape[1]
        self._feature_names = feature_names or [
            f"feature_{i}" for i in range(n_features)
        ]

        if len(self._feature_names) != n_features:
            raise ValueError(
                f"feature_names length ({len(self._feature_names)}) "
                f"!= n_features ({n_features})"
            )

        self._detectors = {}
        for i, name in enumerate(self._feature_names):
            if self.method == "psi":
                detector = PSIDetector(
                    n_bins=self.n_bins,
                    threshold_low=self.threshold,
                    name=f"psi_{name}",
                )
            else:
                detector = KSDetector(alpha=self.alpha, name=f"ks_{name}")

            detector.set_reference(X_reference[:, i], feature_name=name)
            self._detectors[name] = detector

        logger.info(
            f"FeatureDriftMonitor initialized with {n_features} features "
            f"using {self.method.upper()} method"
        )

    def check_drift(
        self,
        X_current: np.ndarray,
    ) -> List[DriftResult]:
        """
        Check all features for drift.

        Args:
            X_current: Current data, shape (n_samples, n_features)

        Returns:
            List of DriftResult for each feature
        """
        if not self._detectors:
            raise RuntimeError("Reference not set. Call set_reference() first.")

        X_current = np.asarray(X_current)
        if X_current.ndim == 1:
            X_current = X_current.reshape(-1, 1)

        results = []
        for i, name in enumerate(self._feature_names):
            detector = self._detectors[name]
            result = detector.compare(X_current[:, i])
            results.append(result)

        n_drifted = sum(1 for r in results if r.drift_detected)
        if n_drifted > 0:
            logger.warning(
                f"Drift detected in {n_drifted}/{len(results)} features"
            )

        return results

    def get_drift_summary(
        self,
        X_current: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Get summary of drift detection across all features.

        Returns:
            Dict with drift summary statistics
        """
        results = self.check_drift(X_current)

        drifted = [r for r in results if r.drift_detected]
        severity_counts = {}
        for r in results:
            sev = r.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        return {
            "n_features": len(results),
            "n_drifted": len(drifted),
            "drift_rate": len(drifted) / len(results) if results else 0,
            "drifted_features": [r.feature_name for r in drifted],
            "severity_counts": severity_counts,
            "max_metric": max(r.metric_value for r in results) if results else 0,
            "method": self.method,
        }

    def reset(self) -> None:
        """Reset all detectors."""
        self._detectors = {}
        self._feature_names = []


__all__ = [
    "DriftType",
    "DriftSeverity",
    "DriftResult",
    "BaseDriftDetector",
    "ADWINDetector",
    "PSIDetector",
    "KSDetector",
    "FeatureDriftMonitor",
]
