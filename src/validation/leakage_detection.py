"""
Feature-label leakage detection for ML pipelines.

Detects potential data leakage by analyzing correlations between
features at time t and labels at time t. High correlations may
indicate forward-looking features.

Detection methods:
- Point-in-time correlation: Features should not correlate strongly with current labels
- Temporal correlation: Features should correlate more with past than future labels
- Information gain: Features shouldn't provide excessive information about labels

References:
- Kaufman, S. et al. (2012). "Leakage in data mining: Formulation, detection, and avoidance"
- López de Prado, M. (2018). "Advances in Financial Machine Learning"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy import stats  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class LeakageDetectedError(Exception):
    """Raised when data leakage is detected and blocking mode is enabled.

    This exception is raised by leakage detection functions when:
    1. Leakage is detected (suspicious features found)
    2. The `raise_on_leakage=True` parameter is set

    The exception contains the full LeakageReport for inspection.

    Example:
        >>> try:
        ...     report = check_feature_label_correlation(
        ...         features, labels, raise_on_leakage=True
        ...     )
        ... except LeakageDetectedError as e:
        ...     print(f"Leakage found: {e.report.n_suspicious} suspicious features")
        ...     for feat in e.report.suspicious_features:
        ...         print(f"  - {feat.feature_name}: r={feat.correlation:.3f}")
    """

    def __init__(self, report: LeakageReport) -> None:
        self.report = report
        super().__init__(
            f"Data leakage detected: {report.n_suspicious} suspicious features "
            f"(threshold={report.correlation_threshold}). {report.summary}"
        )


@dataclass
class LeakageCheckResult:
    """Result of a leakage check for a single feature."""

    feature_name: str
    correlation: float  # Correlation with current label
    p_value: float  # Statistical significance
    is_suspicious: bool  # Above threshold
    leakage_score: float  # Combined leakage score (0-1)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "correlation": self.correlation,
            "p_value": self.p_value,
            "is_suspicious": self.is_suspicious,
            "leakage_score": self.leakage_score,
            "details": self.details,
        }


@dataclass
class LeakageReport:
    """Complete leakage analysis report."""

    n_features: int
    n_suspicious: int
    suspicious_features: list[LeakageCheckResult]
    all_results: list[LeakageCheckResult]
    correlation_threshold: float
    summary: str

    @property
    def leakage_ratio(self) -> float:
        """Ratio of suspicious features."""
        return self.n_suspicious / self.n_features if self.n_features > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_features": self.n_features,
            "n_suspicious": self.n_suspicious,
            "leakage_ratio": self.leakage_ratio,
            "correlation_threshold": self.correlation_threshold,
            "suspicious_features": [r.to_dict() for r in self.suspicious_features],
            "summary": self.summary,
        }

    def get_suspicious_feature_names(self) -> list[str]:
        """Get names of suspicious features."""
        return [r.feature_name for r in self.suspicious_features]


def check_feature_label_correlation(
    features: pd.DataFrame | np.ndarray,
    labels: np.ndarray,
    feature_names: list[str] | None = None,
    correlation_threshold: float = 0.5,
    p_value_threshold: float = 0.01,
    method: str = "spearman",
    raise_on_leakage: bool = False,
) -> LeakageReport:
    """
    Check for feature-label leakage via correlation analysis.

    High correlation between features at time t and labels at time t
    may indicate forward-looking features (leakage).

    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Label array (n_samples,)
        feature_names: Optional list of feature names
        correlation_threshold: Correlation above this is suspicious
        p_value_threshold: Only consider statistically significant correlations
        method: Correlation method ("spearman" or "pearson")
        raise_on_leakage: If True, raise LeakageDetectedError when leakage is found.
            Default is False for backward compatibility.

    Returns:
        LeakageReport with analysis results

    Raises:
        LeakageDetectedError: If raise_on_leakage=True and suspicious features found.

    Example:
        >>> report = check_feature_label_correlation(X_train, y_train)
        >>> if report.n_suspicious > 0:
        ...     print(f"WARNING: {report.n_suspicious} suspicious features")
        ...     for feat in report.suspicious_features:
        ...         print(f"  - {feat.feature_name}: r={feat.correlation:.3f}")

        >>> # Blocking mode for training pipelines
        >>> report = check_feature_label_correlation(
        ...     X_train, y_train, raise_on_leakage=True
        ... )  # Raises LeakageDetectedError if leakage found
    """
    # Convert to numpy if DataFrame
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = features.columns.tolist()
        features = features.values

    features = np.asarray(features)
    labels = np.asarray(labels).flatten()

    if len(features) != len(labels):
        raise ValueError("Features and labels must have same number of samples")

    n_samples, n_features = features.shape

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    if len(feature_names) != n_features:
        raise ValueError("feature_names must match number of features")

    # Compute correlations
    results = []

    for i, name in enumerate(feature_names):
        feature_col = features[:, i]

        # Skip if constant
        if np.std(feature_col) == 0:
            results.append(
                LeakageCheckResult(
                    feature_name=name,
                    correlation=0.0,
                    p_value=1.0,
                    is_suspicious=False,
                    leakage_score=0.0,
                    details={"note": "constant feature"},
                )
            )
            continue

        # Handle NaN values
        valid_mask = ~(np.isnan(feature_col) | np.isnan(labels))
        if valid_mask.sum() < 10:
            results.append(
                LeakageCheckResult(
                    feature_name=name,
                    correlation=0.0,
                    p_value=1.0,
                    is_suspicious=False,
                    leakage_score=0.0,
                    details={"note": "insufficient valid samples"},
                )
            )
            continue

        # Compute correlation
        if method == "spearman":
            corr, p_val = stats.spearmanr(feature_col[valid_mask], labels[valid_mask])
        elif method == "pearson":
            corr, p_val = stats.pearsonr(feature_col[valid_mask], labels[valid_mask])
        else:
            raise ValueError(f"Unknown method: {method}")

        # Handle NaN correlation
        if np.isnan(corr):
            corr = 0.0
            p_val = 1.0

        abs_corr = abs(corr)
        is_suspicious = abs_corr >= correlation_threshold and p_val < p_value_threshold

        # Leakage score: combines correlation magnitude and significance
        if p_val < p_value_threshold:
            leakage_score = abs_corr
        else:
            leakage_score = 0.0

        results.append(
            LeakageCheckResult(
                feature_name=name,
                correlation=float(corr),
                p_value=float(p_val),
                is_suspicious=is_suspicious,
                leakage_score=float(leakage_score),
                details={
                    "abs_correlation": float(abs_corr),
                    "method": method,
                    "n_valid_samples": int(valid_mask.sum()),
                },
            )
        )

    # Sort by leakage score
    results.sort(key=lambda r: r.leakage_score, reverse=True)
    suspicious = [r for r in results if r.is_suspicious]

    # Generate summary
    if len(suspicious) == 0:
        summary = "No suspicious features detected."
    else:
        top_features = ", ".join([r.feature_name for r in suspicious[:5]])
        summary = (
            f"LEAKAGE WARNING: {len(suspicious)} features have high correlation "
            f"with labels (threshold={correlation_threshold}). "
            f"Top suspicious: {top_features}"
        )

    report = LeakageReport(
        n_features=n_features,
        n_suspicious=len(suspicious),
        suspicious_features=suspicious,
        all_results=results,
        correlation_threshold=correlation_threshold,
        summary=summary,
    )

    if raise_on_leakage and report.n_suspicious > 0:
        raise LeakageDetectedError(report)

    return report


def check_temporal_leakage(
    features: pd.DataFrame | np.ndarray,
    labels: np.ndarray,
    feature_names: list[str] | None = None,
    max_lag: int = 5,
    correlation_threshold: float = 0.3,
    raise_on_leakage: bool = False,
) -> LeakageReport:
    """
    Check for temporal leakage by comparing forward vs backward correlations.

    If a feature correlates more strongly with future labels than past labels,
    it may contain forward-looking information.

    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Label array (n_samples,)
        feature_names: Optional list of feature names
        max_lag: Maximum lag to check
        correlation_threshold: Threshold for suspicious forward correlation
        raise_on_leakage: If True, raise LeakageDetectedError when leakage is found.
            Default is False for backward compatibility.

    Returns:
        LeakageReport with temporal leakage analysis

    Raises:
        LeakageDetectedError: If raise_on_leakage=True and temporal leakage found.

    Example:
        >>> report = check_temporal_leakage(X_train, y_train, max_lag=10)
        >>> print(report.summary)

        >>> # Blocking mode
        >>> report = check_temporal_leakage(
        ...     X_train, y_train, raise_on_leakage=True
        ... )  # Raises if leakage found
    """
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = features.columns.tolist()
        features = features.values

    features = np.asarray(features)
    labels = np.asarray(labels).flatten()

    n_samples, n_features = features.shape

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    results = []

    for i, name in enumerate(feature_names):
        feature_col = features[:, i]

        if np.std(feature_col) == 0:
            results.append(
                LeakageCheckResult(
                    feature_name=name,
                    correlation=0.0,
                    p_value=1.0,
                    is_suspicious=False,
                    leakage_score=0.0,
                    details={"note": "constant feature"},
                )
            )
            continue

        # Compute correlations at different lags
        forward_corrs = []  # Feature at t vs label at t+k (should be low)
        backward_corrs = []  # Feature at t vs label at t-k (expected)

        for lag in range(1, max_lag + 1):
            # Forward: does feature predict future labels?
            if lag < n_samples:
                f_corr, _ = stats.spearmanr(feature_col[:-lag], labels[lag:], nan_policy="omit")
                if not np.isnan(f_corr):
                    forward_corrs.append(abs(f_corr))

            # Backward: does feature relate to past labels?
            if lag < n_samples:
                b_corr, _ = stats.spearmanr(feature_col[lag:], labels[:-lag], nan_policy="omit")
                if not np.isnan(b_corr):
                    backward_corrs.append(abs(b_corr))

        # Concurrent correlation
        concurrent_corr, p_val = stats.spearmanr(feature_col, labels, nan_policy="omit")
        if np.isnan(concurrent_corr):
            concurrent_corr = 0.0
            p_val = 1.0

        # Leakage indicator: strong forward correlation or forward > backward
        avg_forward = np.mean(forward_corrs) if forward_corrs else 0.0
        avg_backward = np.mean(backward_corrs) if backward_corrs else 0.0

        # Suspicious if forward correlation is strong or forward > backward
        is_suspicious = avg_forward > correlation_threshold or (
            avg_forward > avg_backward * 1.5 and avg_forward > 0.1
        )

        leakage_score = max(
            avg_forward,
            abs(concurrent_corr) if abs(concurrent_corr) > correlation_threshold else 0,
        )

        results.append(
            LeakageCheckResult(
                feature_name=name,
                correlation=float(concurrent_corr),
                p_value=float(p_val),
                is_suspicious=is_suspicious,
                leakage_score=float(leakage_score),
                details={
                    "avg_forward_correlation": float(avg_forward),
                    "avg_backward_correlation": float(avg_backward),
                    "forward_backward_ratio": (
                        float(avg_forward / avg_backward)
                        if avg_backward > 0
                        else float("inf") if avg_forward > 0 else 0.0
                    ),
                    "max_lag": max_lag,
                },
            )
        )

    results.sort(key=lambda r: r.leakage_score, reverse=True)
    suspicious = [r for r in results if r.is_suspicious]

    if len(suspicious) == 0:
        summary = "No temporal leakage detected."
    else:
        top_features = ", ".join([r.feature_name for r in suspicious[:5]])
        summary = (
            f"TEMPORAL LEAKAGE WARNING: {len(suspicious)} features show "
            f"suspicious forward correlations. Top: {top_features}"
        )

    report = LeakageReport(
        n_features=n_features,
        n_suspicious=len(suspicious),
        suspicious_features=suspicious,
        all_results=results,
        correlation_threshold=correlation_threshold,
        summary=summary,
    )

    if raise_on_leakage and report.n_suspicious > 0:
        raise LeakageDetectedError(report)

    return report


def check_information_leakage(
    features: pd.DataFrame | np.ndarray,
    labels: np.ndarray,
    feature_names: list[str] | None = None,
    mi_threshold: float = 0.5,
    raise_on_leakage: bool = False,
) -> LeakageReport:
    """
    Check for information leakage using mutual information.

    High mutual information between features and labels may indicate
    that features contain information that wouldn't be available
    at prediction time.

    Args:
        features: Feature matrix
        labels: Label array
        feature_names: Optional feature names
        mi_threshold: Mutual information threshold (normalized)
        raise_on_leakage: If True, raise LeakageDetectedError when leakage is found.
            Default is False for backward compatibility.

    Returns:
        LeakageReport with mutual information analysis

    Raises:
        LeakageDetectedError: If raise_on_leakage=True and information leakage found.
    """
    from sklearn.feature_selection import mutual_info_classif  # type: ignore[import-untyped]

    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = features.columns.tolist()
        features = features.values

    features = np.asarray(features)
    labels = np.asarray(labels).flatten()

    n_samples, n_features = features.shape

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # Handle NaN values
    valid_mask = ~np.isnan(features).any(axis=1) & ~np.isnan(labels)
    if valid_mask.sum() < 100:
        logger.warning("Insufficient valid samples for MI analysis")
        return LeakageReport(
            n_features=n_features,
            n_suspicious=0,
            suspicious_features=[],
            all_results=[],
            correlation_threshold=mi_threshold,
            summary="Insufficient valid samples for MI analysis",
        )

    features_valid = features[valid_mask]
    labels_valid = labels[valid_mask]

    # Compute mutual information
    try:
        mi_scores = mutual_info_classif(
            features_valid, labels_valid, random_state=42, n_neighbors=5
        )
    except Exception as e:
        logger.warning(f"MI computation failed: {e}")
        return LeakageReport(
            n_features=n_features,
            n_suspicious=0,
            suspicious_features=[],
            all_results=[],
            correlation_threshold=mi_threshold,
            summary=f"MI computation failed: {e}",
        )

    # Normalize MI scores (0-1)
    max_mi = np.max(mi_scores) if np.max(mi_scores) > 0 else 1.0
    mi_normalized = mi_scores / max_mi

    results = []
    for i, name in enumerate(feature_names):
        mi_score = mi_normalized[i]
        is_suspicious = mi_score >= mi_threshold

        results.append(
            LeakageCheckResult(
                feature_name=name,
                correlation=float(mi_scores[i]),  # Raw MI
                p_value=0.0,  # MI doesn't have p-value
                is_suspicious=is_suspicious,
                leakage_score=float(mi_score),
                details={
                    "mutual_information_raw": float(mi_scores[i]),
                    "mutual_information_normalized": float(mi_score),
                },
            )
        )

    results.sort(key=lambda r: r.leakage_score, reverse=True)
    suspicious = [r for r in results if r.is_suspicious]

    if len(suspicious) == 0:
        summary = "No information leakage detected via MI analysis."
    else:
        top_features = ", ".join([r.feature_name for r in suspicious[:5]])
        summary = (
            f"INFORMATION LEAKAGE WARNING: {len(suspicious)} features have "
            f"high mutual information with labels. Top: {top_features}"
        )

    report = LeakageReport(
        n_features=n_features,
        n_suspicious=len(suspicious),
        suspicious_features=suspicious,
        all_results=results,
        correlation_threshold=mi_threshold,
        summary=summary,
    )

    if raise_on_leakage and report.n_suspicious > 0:
        raise LeakageDetectedError(report)

    return report


def comprehensive_leakage_check(
    features: pd.DataFrame | np.ndarray,
    labels: np.ndarray,
    feature_names: list[str] | None = None,
    correlation_threshold: float = 0.5,
    temporal_threshold: float = 0.3,
    mi_threshold: float = 0.5,
    max_lag: int = 5,
    raise_on_leakage: bool = False,
) -> dict[str, LeakageReport]:
    """
    Run comprehensive leakage detection with multiple methods.

    Args:
        features: Feature matrix
        labels: Label array
        feature_names: Optional feature names
        correlation_threshold: Threshold for correlation check
        temporal_threshold: Threshold for temporal check
        mi_threshold: Threshold for MI check
        max_lag: Max lag for temporal analysis
        raise_on_leakage: If True, raise LeakageDetectedError when ANY check
            finds leakage. Default is False for backward compatibility.

    Returns:
        Dict with results from all leakage checks

    Raises:
        LeakageDetectedError: If raise_on_leakage=True and any leakage found.
            The exception's report will be the first check that found leakage.

    Example:
        >>> results = comprehensive_leakage_check(X_train, y_train)
        >>> for check_name, report in results.items():
        ...     if report.n_suspicious > 0:
        ...         print(f"{check_name}: {report.summary}")

        >>> # Blocking mode for training pipelines
        >>> results = comprehensive_leakage_check(
        ...     X_train, y_train, raise_on_leakage=True
        ... )  # Raises if any check finds leakage
    """
    results = {}

    # Correlation check
    logger.info("Running correlation leakage check...")
    results["correlation"] = check_feature_label_correlation(
        features,
        labels,
        feature_names=feature_names,
        correlation_threshold=correlation_threshold,
        raise_on_leakage=raise_on_leakage,
    )

    # Temporal check
    logger.info("Running temporal leakage check...")
    results["temporal"] = check_temporal_leakage(
        features,
        labels,
        feature_names=feature_names,
        max_lag=max_lag,
        correlation_threshold=temporal_threshold,
        raise_on_leakage=raise_on_leakage,
    )

    # MI check
    logger.info("Running mutual information leakage check...")
    results["mutual_information"] = check_information_leakage(
        features,
        labels,
        feature_names=feature_names,
        mi_threshold=mi_threshold,
        raise_on_leakage=raise_on_leakage,
    )

    return results


__all__ = [
    "LeakageCheckResult",
    "LeakageDetectedError",
    "LeakageReport",
    "check_feature_label_correlation",
    "check_temporal_leakage",
    "check_information_leakage",
    "comprehensive_leakage_check",
]
