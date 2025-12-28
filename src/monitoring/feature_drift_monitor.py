"""
Multi-feature drift monitoring.

Monitors multiple features simultaneously using PSI or KS tests.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .drift_detectors import KSDetector, PSIDetector
from .drift_types import DriftResult

logger = logging.getLogger(__name__)


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
    "FeatureDriftMonitor",
]
