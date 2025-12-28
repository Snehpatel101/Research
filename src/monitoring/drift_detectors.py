"""
Statistical drift detection implementations.

Implements ADWIN (concept drift), PSI (feature drift), and KS (distribution comparison).
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from scipy import stats

from .drift_types import (
    BaseDriftDetector,
    DriftResult,
    DriftSeverity,
    DriftType,
)

logger = logging.getLogger(__name__)


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
        # Use epsilon to prevent division by zero or near-zero
        epsilon = 1e-8
        if abs(old_mean) < epsilon:
            change = abs(new_mean)
        else:
            change = abs(new_mean - old_mean) / max(abs(old_mean), epsilon)

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


__all__ = [
    "ADWINDetector",
    "PSIDetector",
    "KSDetector",
]
