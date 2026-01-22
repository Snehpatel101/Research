"""
Monitoring module for ML pipeline.

Import paths:
    # New (preferred):
    from src.validation.monitoring import DriftResult, AlertHandler

    # Legacy (still works, deprecation warning):
    from src.monitoring import DriftResult, AlertHandler

Provides online drift detection and alerting:
- ADWIN for concept drift (via river library)
- PSI for feature distribution drift
- KS test for distribution comparison
- Alert handlers with rate limiting and callbacks
"""

from .alert_handler import (
    AlertConfig,
    AlertHandler,
    AlertRecord,
    DriftAlertAggregator,
)
from .drift_detector import (
    ADWINDetector,
    BaseDriftDetector,
    DriftResult,
    DriftSeverity,
    DriftType,
    FeatureDriftMonitor,
    KSDetector,
    PSIDetector,
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
