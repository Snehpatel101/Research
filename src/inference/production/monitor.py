"""
Production model monitoring.

Tracks:
- Feature drift (PSI, KS tests)
- Model performance degradation
- Model freshness (time since training)
- Prediction distribution shifts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.inference.bundle import ModelBundle

logger = logging.getLogger(__name__)


@dataclass
class ModelHealthMetrics:
    """
    Model health metrics for production monitoring.

    Attributes:
        accuracy: Current model accuracy
        drift_features_count: Number of features with detected drift
        model_age_days: Days since model was trained
        avg_latency_ms: Average inference latency in milliseconds
        error_rate: Inference error rate (0-1)
        last_checked: Timestamp of last health check
    """

    accuracy: float
    drift_features_count: int
    model_age_days: int
    avg_latency_ms: float
    error_rate: float
    last_checked: datetime


class ProductionMonitor:
    """
    Unified production monitoring for deployed models.

    This monitor provides comprehensive health checks for models in production:
    - Drift detection via PSI (Population Stability Index)
    - Performance degradation tracking
    - Model freshness validation
    - Alert generation for critical issues

    Example:
        from src.inference.bundle import ModelBundle
        from src.inference.production import ProductionMonitor

        bundle = ModelBundle.load('models/xgboost_h20/')
        monitor = ProductionMonitor(
            bundle,
            baseline_accuracy=0.65,
            max_model_age_days=30
        )

        # Run health checks
        health = monitor.run_all_checks(X_current, y_true, y_pred)
        print(f"Drift features: {health.drift_features_count}")
        print(f"Current accuracy: {health.accuracy:.3f}")
    """

    def __init__(
        self,
        model_bundle: ModelBundle,
        baseline_accuracy: float = 0.60,
        max_model_age_days: int = 30,
        alert_handler: Any | None = None,
    ) -> None:
        """
        Initialize production monitor.

        Args:
            model_bundle: ModelBundle to monitor
            baseline_accuracy: Expected baseline accuracy (for degradation detection)
            max_model_age_days: Maximum acceptable model age in days
            alert_handler: Optional alert handler (e.g., SlackAlertConnector)
        """
        self.bundle = model_bundle
        self.baseline_accuracy = baseline_accuracy
        self.max_model_age_days = max_model_age_days
        self.alert_handler = alert_handler

        # Initialize drift monitor
        self.drift_monitor = self._create_drift_monitor()

        # Performance tracking
        self.prediction_history: list[dict[str, Any]] = []

        # Latency and error tracking
        self._latency_samples: list[float] = []  # Recent latency samples in ms
        self._error_count: int = 0
        self._total_predictions: int = 0
        self._max_latency_samples: int = 1000  # Keep last 1000 samples

    def record_prediction(
        self,
        latency_ms: float,
        success: bool = True,
    ) -> None:
        """
        Record a prediction for latency and error tracking.

        Call this after each inference to track metrics.

        Args:
            latency_ms: Inference latency in milliseconds
            success: Whether the prediction succeeded (False for errors)
        """
        self._total_predictions += 1

        if success:
            self._latency_samples.append(latency_ms)
            # Keep only recent samples
            if len(self._latency_samples) > self._max_latency_samples:
                self._latency_samples = self._latency_samples[-self._max_latency_samples :]
        else:
            self._error_count += 1

    def get_avg_latency_ms(self) -> float:
        """Get average inference latency in milliseconds."""
        if not self._latency_samples:
            return 0.0
        return sum(self._latency_samples) / len(self._latency_samples)

    def get_error_rate(self) -> float:
        """Get error rate (0-1) from prediction history."""
        if self._total_predictions == 0:
            return 0.0
        return self._error_count / self._total_predictions

    def _create_drift_monitor(self) -> Any:
        """Create and configure drift monitor."""
        try:
            from src.validation.monitoring import FeatureDriftMonitor

            return FeatureDriftMonitor(method="psi", threshold=0.2)
        except ImportError:
            logger.warning("FeatureDriftMonitor not available, drift detection disabled")
            return None

    def check_model_freshness(self) -> tuple[bool, str]:
        """
        Check if model is too old.

        Models should be retrained regularly to maintain performance.
        This check alerts if the model exceeds the configured age threshold.

        Returns:
            Tuple of (is_fresh, message)
        """
        # Check if bundle has metadata with creation timestamp
        if not hasattr(self.bundle, "metadata"):
            return True, "No metadata available for freshness check"

        metadata = self.bundle.metadata
        if not hasattr(metadata, "created_at"):
            return True, "No created_at timestamp in metadata"

        try:
            created_at = datetime.fromisoformat(metadata.created_at)
        except (ValueError, AttributeError):
            return True, "Invalid created_at timestamp"

        age_days = (datetime.now() - created_at).days

        if age_days > self.max_model_age_days:
            msg = f"Model age {age_days} days exceeds max {self.max_model_age_days}"
            if self.alert_handler:
                self.alert_handler.send_alert("model_freshness", "HIGH", msg)
            return False, msg

        return True, f"Model age: {age_days} days (OK)"

    def check_performance_degradation(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> tuple[bool, dict[str, float]]:
        """
        Check if accuracy has degraded significantly.

        Alerts if current accuracy drops more than 20% below baseline.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Tuple of (is_healthy, metrics_dict)
        """
        try:
            from sklearn.metrics import accuracy_score

            current_acc = accuracy_score(y_true, y_pred)
        except ImportError:
            logger.warning("sklearn not available for accuracy calculation")
            return True, {"current": 0.0, "baseline": self.baseline_accuracy}

        # Alert if accuracy drops more than 20%
        if current_acc < self.baseline_accuracy * 0.80:
            msg = f"Accuracy degraded: {self.baseline_accuracy:.3f} → {current_acc:.3f}"
            if self.alert_handler:
                self.alert_handler.send_alert("accuracy_degradation", "CRITICAL", msg)
            return False, {"current": current_acc, "baseline": self.baseline_accuracy}

        return True, {"current": current_acc, "baseline": self.baseline_accuracy}

    def check_drift(self, X_current: np.ndarray) -> tuple[int, list[Any]]:
        """
        Check for feature drift.

        Args:
            X_current: Current feature data

        Returns:
            Tuple of (drift_count, drift_results)
        """
        if self.drift_monitor is None:
            return 0, []

        try:
            drift_results = self.drift_monitor.check_drift(X_current)
            drift_count = sum(1 for r in drift_results if r.drift_detected)

            # Alert on high/critical drift
            if self.alert_handler:
                for result in drift_results:
                    if result.drift_detected and hasattr(result, "severity"):
                        severity_val = (
                            result.severity.value
                            if hasattr(result.severity, "value")
                            else str(result.severity)
                        )
                        if severity_val in ("HIGH", "CRITICAL"):
                            self.alert_handler.send_drift_alert(
                                feature_name=result.feature_name,
                                severity=severity_val,
                                metric_value=result.metric_value,
                                threshold=result.threshold,
                            )

            return drift_count, drift_results
        except Exception as e:
            logger.warning(f"Drift detection failed: {e}")
            return 0, []

    def run_all_checks(
        self,
        X_current: np.ndarray,
        y_true: np.ndarray | None = None,
        y_pred: np.ndarray | None = None,
    ) -> ModelHealthMetrics:
        """
        Run all monitoring checks and return health metrics.

        Args:
            X_current: Current feature data
            y_true: True labels (optional, for performance tracking)
            y_pred: Predicted labels (optional, for performance tracking)

        Returns:
            ModelHealthMetrics with current health status
        """
        # Check freshness
        freshness_ok, freshness_msg = self.check_model_freshness()
        if not freshness_ok:
            logger.warning(freshness_msg)

        # Check drift
        drift_count, drift_results = self.check_drift(X_current)

        # Check performance degradation (if labels provided)
        perf_ok = True
        perf_metrics: dict[str, float] = {}
        if y_true is not None and y_pred is not None:
            perf_ok, perf_metrics = self.check_performance_degradation(y_true, y_pred)
            if not perf_ok:
                logger.error(f"Performance degradation detected: {perf_metrics}")

        # Calculate model age
        model_age_days = 0
        if hasattr(self.bundle, "metadata") and hasattr(self.bundle.metadata, "created_at"):
            try:
                created_at = datetime.fromisoformat(self.bundle.metadata.created_at)
                model_age_days = (datetime.now() - created_at).days
            except (ValueError, AttributeError):
                pass

        return ModelHealthMetrics(
            accuracy=perf_metrics.get("current", 0.0),
            drift_features_count=drift_count,
            model_age_days=model_age_days,
            avg_latency_ms=self.get_avg_latency_ms(),
            error_rate=self.get_error_rate(),
            last_checked=datetime.now(),
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ModelHealthMetrics",
    "ProductionMonitor",
]
