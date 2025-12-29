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

This module maintains backward compatibility by re-exporting all components
from the refactored submodules.
"""
from __future__ import annotations

# Import all detector implementations
from .drift_detectors import (
    ADWINDetector,
    KSDetector,
    PSIDetector,
)

# Import all types and base classes
from .drift_types import (
    BaseDriftDetector,
    DriftResult,
    DriftSeverity,
    DriftType,
)

# Import multi-feature monitor
from .feature_drift_monitor import FeatureDriftMonitor

# Maintain original __all__ for backward compatibility
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
