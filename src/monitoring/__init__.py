"""
Monitoring module for ML pipeline.

Provides online drift detection and alerting:
- ADWIN for concept drift (via river library)
- PSI for feature distribution drift
- KS test for distribution comparison
- Alert handlers with rate limiting and callbacks
"""

from src.monitoring.drift_detector import (
    ADWINDetector,
    BaseDriftDetector,
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
    AlertRecord,
    DriftAlertAggregator,
)

__all__ = [
    # Drift types
    "DriftType",
    "DriftSeverity",
    "DriftResult",
    # Detectors
    "BaseDriftDetector",
    "ADWINDetector",
    "PSIDetector",
    "KSDetector",
    "FeatureDriftMonitor",
    # Alert handling
    "AlertConfig",
    "AlertHandler",
    "AlertRecord",
    "DriftAlertAggregator",
]
