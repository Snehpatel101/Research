"""
DEPRECATED: Import from 'src.validation.monitoring' instead.

This module will be removed in v2.0.0.

Migration:
    # Old (deprecated)
    from src.monitoring import DriftResult, AlertHandler

    # New (preferred)
    from src.validation.monitoring import DriftResult, AlertHandler
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

# Names that will be lazily loaded from src.validation.monitoring
_MONITORING_EXPORTS = [
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


def __getattr__(name: str):
    """Lazy import to avoid circular imports."""
    if name in _MONITORING_EXPORTS:
        # Show deprecation warning
        warnings.warn(
            f"Importing '{name}' from 'src.monitoring' is deprecated. "
            f"Use 'from src.validation.monitoring import {name}' instead. "
            "This will be removed in v2.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Import from new location
        from src.validation import monitoring as monitoring_module
        return getattr(monitoring_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Return list of available names."""
    return _MONITORING_EXPORTS


__all__ = _MONITORING_EXPORTS
