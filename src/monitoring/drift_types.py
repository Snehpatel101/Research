"""
Drift detection core types and base classes.

Defines common enums, result types, and abstract base detector.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

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
    feature_name: str | None = None
    timestamp: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
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
    def update(self, value: float) -> DriftResult | None:
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


__all__ = [
    "DriftType",
    "DriftSeverity",
    "DriftResult",
    "BaseDriftDetector",
]
