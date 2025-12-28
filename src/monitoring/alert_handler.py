"""
Alert Handlers for Drift Detection.

Provides flexible alert handling for drift events:
- Logging alerts
- Callback-based alerts
- Threshold-based filtering
- Alert aggregation and rate limiting

Example:
    >>> handler = AlertHandler()
    >>> handler.add_callback(lambda r: print(f"DRIFT: {r.feature_name}"))
    >>> handler.handle(drift_result)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.monitoring.drift_detector import DriftResult, DriftSeverity, DriftType

logger = logging.getLogger(__name__)


@dataclass
class AlertConfig:
    """Configuration for alert handling."""
    min_severity: DriftSeverity = DriftSeverity.LOW
    rate_limit_seconds: float = 60.0  # Min time between alerts per feature
    aggregate_window_seconds: float = 300.0  # Window for aggregation
    log_alerts: bool = True
    log_level: int = logging.WARNING


@dataclass
class AlertRecord:
    """Record of a triggered alert."""
    result: DriftResult
    timestamp: float
    acknowledged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertHandler:
    """
    Handles drift detection alerts with filtering and callbacks.

    Features:
    - Severity-based filtering
    - Rate limiting per feature
    - Custom callback support
    - Alert history tracking

    Example:
        >>> handler = AlertHandler(config=AlertConfig(min_severity=DriftSeverity.MEDIUM))
        >>> handler.add_callback(send_slack_alert)
        >>> handler.add_callback(trigger_retraining)
        >>>
        >>> for result in drift_results:
        ...     handler.handle(result)
    """

    def __init__(self, config: Optional[AlertConfig] = None) -> None:
        """
        Initialize AlertHandler.

        Args:
            config: Alert configuration
        """
        self.config = config or AlertConfig()
        self._callbacks: List[Callable[[DriftResult], None]] = []
        self._history: List[AlertRecord] = []
        self._last_alert_time: Dict[str, float] = {}  # feature -> timestamp
        self._max_history = 1000

    def add_callback(
        self,
        callback: Callable[[DriftResult], None],
    ) -> None:
        """
        Add a callback function to be called on alerts.

        Args:
            callback: Function that takes DriftResult as argument
        """
        self._callbacks.append(callback)

    def remove_callback(
        self,
        callback: Callable[[DriftResult], None],
    ) -> bool:
        """
        Remove a callback function.

        Returns:
            True if callback was found and removed
        """
        try:
            self._callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def handle(self, result: DriftResult) -> bool:
        """
        Handle a drift detection result.

        Args:
            result: DriftResult from a detector

        Returns:
            True if alert was triggered, False if filtered
        """
        # Check if drift was detected
        if not result.drift_detected:
            return False

        # Check severity threshold
        if not self._meets_severity_threshold(result.severity):
            return False

        # Check rate limiting
        if not self._check_rate_limit(result):
            return False

        # Record alert
        record = AlertRecord(
            result=result,
            timestamp=time.time(),
        )
        self._history.append(record)
        self._trim_history()

        # Update rate limit tracking
        feature_key = result.feature_name or "global"
        self._last_alert_time[feature_key] = time.time()

        # Log if configured
        if self.config.log_alerts:
            self._log_alert(result)

        # Execute callbacks
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        return True

    def handle_batch(
        self,
        results: List[DriftResult],
    ) -> Dict[str, int]:
        """
        Handle multiple drift results.

        Args:
            results: List of DriftResult objects

        Returns:
            Dict with counts of triggered/filtered alerts
        """
        triggered = 0
        filtered = 0

        for result in results:
            if self.handle(result):
                triggered += 1
            elif result.drift_detected:
                filtered += 1

        return {
            "triggered": triggered,
            "filtered": filtered,
            "total": len(results),
        }

    def _meets_severity_threshold(self, severity: DriftSeverity) -> bool:
        """Check if severity meets minimum threshold."""
        severity_order = [
            DriftSeverity.NONE,
            DriftSeverity.LOW,
            DriftSeverity.MEDIUM,
            DriftSeverity.HIGH,
            DriftSeverity.CRITICAL,
        ]
        return severity_order.index(severity) >= severity_order.index(
            self.config.min_severity
        )

    def _check_rate_limit(self, result: DriftResult) -> bool:
        """Check if alert is rate limited."""
        feature_key = result.feature_name or "global"
        last_time = self._last_alert_time.get(feature_key, 0)
        time_since = time.time() - last_time

        return time_since >= self.config.rate_limit_seconds

    def _log_alert(self, result: DriftResult) -> None:
        """Log an alert."""
        feature_str = f" for '{result.feature_name}'" if result.feature_name else ""
        msg = (
            f"DRIFT ALERT{feature_str}: "
            f"type={result.drift_type.value}, "
            f"severity={result.severity.value}, "
            f"metric={result.metric_value:.4f}"
        )
        logger.log(self.config.log_level, msg)

    def _trim_history(self) -> None:
        """Trim history to max size."""
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def get_history(
        self,
        n_recent: Optional[int] = None,
        drift_type: Optional[DriftType] = None,
        feature_name: Optional[str] = None,
    ) -> List[AlertRecord]:
        """
        Get alert history with optional filtering.

        Args:
            n_recent: Number of recent alerts to return
            drift_type: Filter by drift type
            feature_name: Filter by feature name

        Returns:
            Filtered list of AlertRecord
        """
        records = self._history

        if drift_type is not None:
            records = [r for r in records if r.result.drift_type == drift_type]

        if feature_name is not None:
            records = [r for r in records if r.result.feature_name == feature_name]

        if n_recent is not None:
            records = records[-n_recent:]

        return records

    def get_alert_counts(
        self,
        window_seconds: Optional[float] = None,
    ) -> Dict[str, int]:
        """
        Get alert counts by type and severity.

        Args:
            window_seconds: Time window (None = all history)

        Returns:
            Dict with alert counts
        """
        window = window_seconds or float("inf")
        cutoff = time.time() - window

        recent = [r for r in self._history if r.timestamp >= cutoff]

        counts = {
            "total": len(recent),
            "by_type": {},
            "by_severity": {},
            "by_feature": {},
        }

        for record in recent:
            # By type
            dtype = record.result.drift_type.value
            counts["by_type"][dtype] = counts["by_type"].get(dtype, 0) + 1

            # By severity
            sev = record.result.severity.value
            counts["by_severity"][sev] = counts["by_severity"].get(sev, 0) + 1

            # By feature
            feat = record.result.feature_name or "unknown"
            counts["by_feature"][feat] = counts["by_feature"].get(feat, 0) + 1

        return counts

    def clear_history(self) -> int:
        """
        Clear alert history.

        Returns:
            Number of records cleared
        """
        n = len(self._history)
        self._history = []
        self._last_alert_time = {}
        return n

    def acknowledge(
        self,
        feature_name: Optional[str] = None,
    ) -> int:
        """
        Acknowledge alerts.

        Args:
            feature_name: Feature to acknowledge (None = all)

        Returns:
            Number of alerts acknowledged
        """
        count = 0
        for record in self._history:
            if not record.acknowledged:
                if feature_name is None or record.result.feature_name == feature_name:
                    record.acknowledged = True
                    count += 1
        return count


class DriftAlertAggregator:
    """
    Aggregates drift alerts over time windows.

    Useful for reducing alert noise by summarizing
    multiple drift events into periodic reports.

    Example:
        >>> aggregator = DriftAlertAggregator(window_seconds=300)
        >>> aggregator.add(result)
        >>> if aggregator.should_report():
        ...     summary = aggregator.get_summary()
        ...     send_report(summary)
        ...     aggregator.reset()
    """

    def __init__(
        self,
        window_seconds: float = 300.0,
        min_alerts_to_report: int = 1,
    ) -> None:
        """
        Initialize aggregator.

        Args:
            window_seconds: Aggregation window in seconds
            min_alerts_to_report: Minimum alerts before reporting
        """
        self.window_seconds = window_seconds
        self.min_alerts_to_report = min_alerts_to_report
        self._alerts: List[DriftResult] = []
        self._window_start = time.time()

    def add(self, result: DriftResult) -> None:
        """Add a drift result to the aggregation window."""
        if result.drift_detected:
            self._alerts.append(result)

    def should_report(self) -> bool:
        """Check if aggregator should generate a report."""
        window_elapsed = time.time() - self._window_start >= self.window_seconds
        has_enough_alerts = len(self._alerts) >= self.min_alerts_to_report

        return window_elapsed and has_enough_alerts

    def get_summary(self) -> Dict[str, Any]:
        """Get aggregated summary of alerts."""
        if not self._alerts:
            return {"n_alerts": 0, "features": [], "max_severity": None}

        # Find max severity
        severity_order = [
            DriftSeverity.NONE,
            DriftSeverity.LOW,
            DriftSeverity.MEDIUM,
            DriftSeverity.HIGH,
            DriftSeverity.CRITICAL,
        ]
        max_sev = max(self._alerts, key=lambda r: severity_order.index(r.severity))

        # Group by feature
        features = {}
        for alert in self._alerts:
            feat = alert.feature_name or "unknown"
            if feat not in features:
                features[feat] = {"count": 0, "max_metric": 0}
            features[feat]["count"] += 1
            features[feat]["max_metric"] = max(
                features[feat]["max_metric"], alert.metric_value
            )

        return {
            "n_alerts": len(self._alerts),
            "window_seconds": self.window_seconds,
            "features": features,
            "n_features_affected": len(features),
            "max_severity": max_sev.severity.value,
            "drift_types": list(set(a.drift_type.value for a in self._alerts)),
        }

    def reset(self) -> None:
        """Reset aggregator for new window."""
        self._alerts = []
        self._window_start = time.time()


__all__ = [
    "AlertConfig",
    "AlertRecord",
    "AlertHandler",
    "DriftAlertAggregator",
]
