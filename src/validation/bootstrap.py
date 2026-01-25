"""
Bootstrap confidence intervals for ML metrics.

Provides uncertainty quantification for performance metrics using
bootstrap resampling. Essential for understanding the reliability
of reported metrics.

Features:
- Bootstrap CI for any metric function
- Specialized functions for common metrics (Sharpe, drawdown, accuracy)
- Percentile and BCa (bias-corrected accelerated) methods
- Parallel computation support

References:
- Efron, B. & Tibshirani, R. (1993). "An Introduction to the Bootstrap"
- DiCiccio, T.J. & Efron, B. (1996). "Bootstrap confidence intervals"
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    """Result of bootstrap analysis."""

    estimate: float  # Point estimate
    ci_lower: float  # Lower bound of CI
    ci_upper: float  # Upper bound of CI
    ci_level: float  # Confidence level (e.g., 0.95)
    std_error: float  # Bootstrap standard error
    n_bootstrap: int  # Number of bootstrap samples
    method: str  # CI method used
    bootstrap_distribution: np.ndarray | None = None  # Full distribution if stored

    @property
    def ci_width(self) -> float:
        """Width of confidence interval."""
        return self.ci_upper - self.ci_lower

    @property
    def relative_ci_width(self) -> float:
        """CI width relative to estimate (coefficient of variation)."""
        if self.estimate == 0:
            return float("inf")
        return self.ci_width / abs(self.estimate)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "estimate": self.estimate,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "ci_level": self.ci_level,
            "std_error": self.std_error,
            "n_bootstrap": self.n_bootstrap,
            "method": self.method,
            "ci_width": self.ci_width,
            "relative_ci_width": self.relative_ci_width,
        }

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"{self.estimate:.4f} [{self.ci_lower:.4f}, {self.ci_upper:.4f}] "
            f"({self.ci_level*100:.0f}% CI)"
        )


def bootstrap_metric(
    data: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    method: str = "percentile",
    random_state: int | None = None,
    return_distribution: bool = False,
) -> BootstrapResult:
    """
    Compute bootstrap confidence interval for any metric.

    Args:
        data: Input data array
        metric_fn: Function that computes metric from data
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level (default 0.95 for 95% CI)
        method: CI method - "percentile" or "bca" (bias-corrected accelerated)
        random_state: Random seed for reproducibility
        return_distribution: Whether to store full bootstrap distribution

    Returns:
        BootstrapResult with estimate and confidence interval

    Example:
        >>> returns = np.random.randn(252) * 0.01  # Daily returns
        >>> def sharpe(r): return np.mean(r) / np.std(r) * np.sqrt(252)
        >>> result = bootstrap_metric(returns, sharpe)
        >>> print(f"Sharpe: {result}")
    """
    data = np.asarray(data)
    n = len(data)

    if n < 2:
        raise ValueError("Need at least 2 data points for bootstrap")

    rng = np.random.default_rng(random_state)

    # Point estimate
    point_estimate = metric_fn(data)

    # Bootstrap resampling
    bootstrap_estimates = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = rng.integers(0, n, size=n)
        bootstrap_sample = data[indices]
        try:
            bootstrap_estimates[i] = metric_fn(bootstrap_sample)
        except Exception:
            # Handle edge cases (e.g., zero std in Sharpe)
            bootstrap_estimates[i] = np.nan

    # Remove NaN values
    valid_estimates = bootstrap_estimates[~np.isnan(bootstrap_estimates)]
    if len(valid_estimates) < n_bootstrap * 0.5:
        logger.warning(
            f"Many bootstrap samples failed ({n_bootstrap - len(valid_estimates)}/{n_bootstrap})"
        )

    if len(valid_estimates) == 0:
        raise ValueError("All bootstrap samples failed")

    # Bootstrap standard error
    std_error = np.std(valid_estimates, ddof=1)

    # Confidence interval
    alpha = 1 - ci_level
    if method == "percentile":
        ci_lower = np.percentile(valid_estimates, alpha / 2 * 100)
        ci_upper = np.percentile(valid_estimates, (1 - alpha / 2) * 100)
    elif method == "bca":
        ci_lower, ci_upper = _bca_interval(
            data, metric_fn, valid_estimates, point_estimate, alpha, rng
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'percentile' or 'bca'")

    return BootstrapResult(
        estimate=float(point_estimate),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        ci_level=ci_level,
        std_error=float(std_error),
        n_bootstrap=n_bootstrap,
        method=method,
        bootstrap_distribution=valid_estimates if return_distribution else None,
    )


def _bca_interval(
    data: np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    bootstrap_estimates: np.ndarray,
    point_estimate: float,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """
    Compute BCa (bias-corrected and accelerated) confidence interval.

    BCa adjusts for bias and skewness in the bootstrap distribution,
    providing more accurate coverage than simple percentile method.
    """
    n = len(data)

    # Bias correction factor
    z0 = stats.norm.ppf(np.mean(bootstrap_estimates < point_estimate))
    if np.isinf(z0):
        # Fall back to percentile if z0 is infinite
        z0 = 0.0

    # Acceleration factor (jackknife estimate)
    jackknife_estimates = np.zeros(n)
    for i in range(n):
        jackknife_sample = np.delete(data, i)
        try:
            jackknife_estimates[i] = metric_fn(jackknife_sample)
        except Exception:
            jackknife_estimates[i] = np.nan

    valid_jack = jackknife_estimates[~np.isnan(jackknife_estimates)]
    if len(valid_jack) < n * 0.5:
        # Fall back to percentile
        return (
            np.percentile(bootstrap_estimates, alpha / 2 * 100),
            np.percentile(bootstrap_estimates, (1 - alpha / 2) * 100),
        )

    jack_mean = np.mean(valid_jack)
    num = np.sum((jack_mean - valid_jack) ** 3)
    denom = 6 * (np.sum((jack_mean - valid_jack) ** 2) ** 1.5)

    a = 0.0 if denom == 0 else num / denom

    # BCa percentiles
    z_alpha_lower = stats.norm.ppf(alpha / 2)
    z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

    def bca_percentile(z_alpha: float) -> float:
        num = z0 + z_alpha
        denom = 1 - a * num
        if denom == 0:
            return z_alpha
        return float(stats.norm.cdf(z0 + num / denom))

    p_lower = bca_percentile(z_alpha_lower) * 100
    p_upper = bca_percentile(z_alpha_upper) * 100

    # Ensure valid percentiles
    p_lower = max(0, min(100, p_lower))
    p_upper = max(0, min(100, p_upper))

    return (
        np.percentile(bootstrap_estimates, p_lower),
        np.percentile(bootstrap_estimates, p_upper),
    )


def bootstrap_sharpe_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    method: str = "percentile",
    random_state: int | None = None,
) -> BootstrapResult:
    """
    Bootstrap confidence interval for Sharpe ratio.

    Args:
        returns: Array of returns (daily, monthly, etc.)
        periods_per_year: Annualization factor (252 for daily, 12 for monthly)
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level
        method: CI method
        random_state: Random seed

    Returns:
        BootstrapResult for Sharpe ratio

    Example:
        >>> daily_returns = portfolio_returns  # Array of daily returns
        >>> result = bootstrap_sharpe_ratio(daily_returns)
        >>> print(f"Sharpe: {result.estimate:.2f} [{result.ci_lower:.2f}, {result.ci_upper:.2f}]")
    """

    def sharpe_fn(r: np.ndarray) -> float:
        if np.std(r, ddof=1) == 0:
            return 0.0
        return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(periods_per_year))

    return bootstrap_metric(
        returns,
        sharpe_fn,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        method=method,
        random_state=random_state,
    )


def bootstrap_max_drawdown(
    returns: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    method: str = "percentile",
    random_state: int | None = None,
) -> BootstrapResult:
    """
    Bootstrap confidence interval for maximum drawdown.

    Args:
        returns: Array of returns
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level
        method: CI method
        random_state: Random seed

    Returns:
        BootstrapResult for max drawdown (as positive percentage)
    """

    def max_drawdown_fn(r: np.ndarray) -> float:
        # Compute equity curve
        equity = np.cumprod(1 + r)
        # Running maximum
        running_max = np.maximum.accumulate(equity)
        # Drawdown
        drawdown = (running_max - equity) / running_max
        return float(np.max(drawdown) * 100)  # As percentage

    return bootstrap_metric(
        returns,
        max_drawdown_fn,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        method=method,
        random_state=random_state,
    )


def bootstrap_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    method: str = "percentile",
    random_state: int | None = None,
) -> BootstrapResult:
    """
    Bootstrap confidence interval for classification accuracy.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level
        method: CI method
        random_state: Random seed

    Returns:
        BootstrapResult for accuracy
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    # Stack for paired resampling
    paired_data = np.column_stack([y_true, y_pred])

    def accuracy_fn(data: np.ndarray) -> float:
        return float(np.mean(data[:, 0] == data[:, 1]))

    return bootstrap_metric(
        paired_data,
        accuracy_fn,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        method=method,
        random_state=random_state,
    )


def bootstrap_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    method: str = "percentile",
    random_state: int | None = None,
) -> BootstrapResult:
    """
    Bootstrap confidence interval for F1 score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ("macro", "micro", "weighted")
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level
        method: CI method
        random_state: Random seed

    Returns:
        BootstrapResult for F1 score
    """
    from sklearn.metrics import f1_score  # type: ignore[import-untyped]

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    paired_data = np.column_stack([y_true, y_pred])

    def f1_fn(data: np.ndarray) -> float:
        return float(f1_score(data[:, 0], data[:, 1], average=average, zero_division=0))

    return bootstrap_metric(
        paired_data,
        f1_fn,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        method=method,
        random_state=random_state,
    )


def bootstrap_win_rate(
    trade_returns: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    method: str = "percentile",
    random_state: int | None = None,
) -> BootstrapResult:
    """
    Bootstrap confidence interval for trading win rate.

    Args:
        trade_returns: Array of individual trade returns
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level
        method: CI method
        random_state: Random seed

    Returns:
        BootstrapResult for win rate (as percentage)
    """

    def win_rate_fn(r: np.ndarray) -> float:
        return float(np.mean(r > 0) * 100)

    return bootstrap_metric(
        trade_returns,
        win_rate_fn,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        method=method,
        random_state=random_state,
    )


def bootstrap_multiple_metrics(
    data: np.ndarray,
    metric_fns: dict[str, Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    method: str = "percentile",
    random_state: int | None = None,
) -> dict[str, BootstrapResult]:
    """
    Compute bootstrap CIs for multiple metrics efficiently.

    Shares bootstrap samples across metrics for consistency.

    Args:
        data: Input data array
        metric_fns: Dict mapping metric names to functions
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level
        method: CI method
        random_state: Random seed

    Returns:
        Dict mapping metric names to BootstrapResult

    Example:
        >>> metrics = {
        ...     "sharpe": lambda r: np.mean(r)/np.std(r)*np.sqrt(252),
        ...     "mean": np.mean,
        ...     "std": np.std,
        ... }
        >>> results = bootstrap_multiple_metrics(returns, metrics)
    """
    data = np.asarray(data)
    n = len(data)
    rng = np.random.default_rng(random_state)

    # Point estimates
    point_estimates = {name: fn(data) for name, fn in metric_fns.items()}

    # Generate shared bootstrap indices
    bootstrap_indices = rng.integers(0, n, size=(n_bootstrap, n))

    # Compute all metrics for each bootstrap sample
    bootstrap_results: dict[str, list[float]] = {name: [] for name in metric_fns}

    for i in range(n_bootstrap):
        bootstrap_sample = data[bootstrap_indices[i]]
        for name, fn in metric_fns.items():
            try:
                bootstrap_results[name].append(fn(bootstrap_sample))
            except Exception:
                bootstrap_results[name].append(np.nan)

    # Compute CIs for each metric
    results = {}
    alpha = 1 - ci_level

    for name, estimates in bootstrap_results.items():
        valid_estimates = np.array([e for e in estimates if not np.isnan(e)])

        if len(valid_estimates) == 0:
            continue

        std_error = np.std(valid_estimates, ddof=1)

        if method == "percentile":
            ci_lower = np.percentile(valid_estimates, alpha / 2 * 100)
            ci_upper = np.percentile(valid_estimates, (1 - alpha / 2) * 100)
        else:
            ci_lower, ci_upper = _bca_interval(
                data, metric_fns[name], valid_estimates, point_estimates[name], alpha, rng
            )

        results[name] = BootstrapResult(
            estimate=float(point_estimates[name]),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            ci_level=ci_level,
            std_error=float(std_error),
            n_bootstrap=n_bootstrap,
            method=method,
        )

    return results


__all__ = [
    "BootstrapResult",
    "bootstrap_metric",
    "bootstrap_sharpe_ratio",
    "bootstrap_max_drawdown",
    "bootstrap_accuracy",
    "bootstrap_f1_score",
    "bootstrap_win_rate",
    "bootstrap_multiple_metrics",
]
