"""
Statistical tests for model comparison.

Provides rigorous statistical tests to determine if one model
significantly outperforms another, avoiding false conclusions
from random variation.

Tests implemented:
- Diebold-Mariano: Compare forecast accuracy
- Paired t-test: Compare paired observations
- Wilcoxon signed-rank: Non-parametric paired comparison

References:
- Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive Accuracy"
- Harvey, D., Leybourne, S., & Newbold, P. (1997). "Testing the equality
  of prediction mean squared errors"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class LossFunction(Enum):
    """Loss functions for Diebold-Mariano test."""

    MSE = "mse"  # Mean Squared Error
    MAE = "mae"  # Mean Absolute Error
    MAPE = "mape"  # Mean Absolute Percentage Error


@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""

    test_name: str
    statistic: float
    p_value: float
    significant: bool  # At alpha level
    alpha: float
    alternative: str  # "two-sided", "greater", "less"
    conclusion: str  # Human-readable conclusion
    effect_size: float | None = None  # Cohen's d or similar
    confidence_interval: tuple[float, float] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "significant": self.significant,
            "alpha": self.alpha,
            "alternative": self.alternative,
            "conclusion": self.conclusion,
            "effect_size": self.effect_size,
            "confidence_interval": self.confidence_interval,
        }


def diebold_mariano_test(
    y_true: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    h: int = 1,
    loss: LossFunction | str = LossFunction.MSE,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    alpha: float = 0.05,
) -> StatisticalTestResult:
    """
    Diebold-Mariano test for comparing forecast accuracy.

    Tests whether the forecasts from model 1 and model 2 have
    significantly different accuracy.

    Args:
        y_true: True values
        pred1: Predictions from model 1
        pred2: Predictions from model 2
        h: Forecast horizon (for HAC variance estimation)
        loss: Loss function to use (MSE, MAE, or MAPE)
        alternative: Type of alternative hypothesis
            - "two-sided": pred1 != pred2
            - "greater": pred1 > pred2 (model 2 is better)
            - "less": pred1 < pred2 (model 1 is better)
        alpha: Significance level

    Returns:
        StatisticalTestResult with test outcome

    Example:
        >>> result = diebold_mariano_test(y_true, pred_xgboost, pred_lstm)
        >>> if result.significant:
        ...     print(f"Models differ significantly: {result.conclusion}")
    """
    y_true = np.asarray(y_true).flatten()
    pred1 = np.asarray(pred1).flatten()
    pred2 = np.asarray(pred2).flatten()

    if len(y_true) != len(pred1) or len(y_true) != len(pred2):
        raise ValueError("All arrays must have the same length")

    n = len(y_true)

    # Convert string to enum if needed
    if isinstance(loss, str):
        loss = LossFunction(loss.lower())

    # Compute loss differentials
    if loss == LossFunction.MSE:
        e1 = (y_true - pred1) ** 2
        e2 = (y_true - pred2) ** 2
    elif loss == LossFunction.MAE:
        e1 = np.abs(y_true - pred1)
        e2 = np.abs(y_true - pred2)
    elif loss == LossFunction.MAPE:
        # Avoid division by zero
        mask = y_true != 0
        e1 = np.zeros(n)
        e2 = np.zeros(n)
        e1[mask] = np.abs((y_true[mask] - pred1[mask]) / y_true[mask])
        e2[mask] = np.abs((y_true[mask] - pred2[mask]) / y_true[mask])
    else:
        raise ValueError(f"Unknown loss function: {loss}")

    # Loss differential
    d = e1 - e2
    d_mean = np.mean(d)

    # Harvey, Leybourne, Newbold (1997) modification for small samples
    # Uses a modified test statistic with t-distribution

    # Estimate variance with Newey-West HAC estimator
    # Truncation lag = h - 1 for h-step ahead forecasts
    gamma_0 = np.var(d, ddof=1)

    if h > 1:
        # Add autocovariance terms
        gamma_sum = 0.0
        for k in range(1, h):
            gamma_k = np.cov(d[:-k], d[k:])[0, 1] if len(d) > k else 0
            gamma_sum += 2 * gamma_k
        var_d = (gamma_0 + gamma_sum) / n
    else:
        var_d = gamma_0 / n

    # Handle edge case of zero variance
    if var_d <= 0:
        logger.warning("Zero variance in loss differentials, models may be identical")
        return StatisticalTestResult(
            test_name="Diebold-Mariano",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            alternative=alternative,
            conclusion="Cannot distinguish models (zero variance in loss differentials)",
            effect_size=0.0,
        )

    # DM statistic
    dm_stat = d_mean / np.sqrt(var_d)

    # Harvey et al. (1997) small-sample correction
    # Modified statistic follows t-distribution with n-1 df
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat_corrected = dm_stat * correction

    # Compute p-value using t-distribution
    df = n - 1
    if alternative == "two-sided":
        p_value = 2 * (1 - stats.t.cdf(abs(dm_stat_corrected), df))
    elif alternative == "greater":
        p_value = 1 - stats.t.cdf(dm_stat_corrected, df)
    elif alternative == "less":
        p_value = stats.t.cdf(dm_stat_corrected, df)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # Effect size (standardized mean difference)
    effect_size = d_mean / np.std(d, ddof=1) if np.std(d, ddof=1) > 0 else 0.0

    # Conclusion
    significant = p_value < alpha
    if significant:
        if d_mean > 0:
            conclusion = f"Model 2 significantly better than Model 1 (p={p_value:.4f})"
        else:
            conclusion = f"Model 1 significantly better than Model 2 (p={p_value:.4f})"
    else:
        conclusion = f"No significant difference between models (p={p_value:.4f})"

    return StatisticalTestResult(
        test_name="Diebold-Mariano",
        statistic=float(dm_stat_corrected),
        p_value=float(p_value),
        significant=significant,
        alpha=alpha,
        alternative=alternative,
        conclusion=conclusion,
        effect_size=float(effect_size),
    )


def paired_ttest(
    metrics1: np.ndarray,
    metrics2: np.ndarray,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    alpha: float = 0.05,
) -> StatisticalTestResult:
    """
    Paired t-test for comparing model performance across folds/runs.

    Tests whether the mean difference in metrics is significantly
    different from zero.

    Args:
        metrics1: Performance metrics from model 1 (e.g., per-fold F1 scores)
        metrics2: Performance metrics from model 2
        alternative: Type of alternative hypothesis
            - "two-sided": metrics1 != metrics2
            - "greater": metrics1 > metrics2
            - "less": metrics1 < metrics2
        alpha: Significance level

    Returns:
        StatisticalTestResult with test outcome

    Example:
        >>> # Compare 5-fold CV results
        >>> f1_model1 = [0.72, 0.75, 0.71, 0.73, 0.74]
        >>> f1_model2 = [0.68, 0.70, 0.69, 0.71, 0.70]
        >>> result = paired_ttest(f1_model1, f1_model2)
    """
    metrics1 = np.asarray(metrics1).flatten()
    metrics2 = np.asarray(metrics2).flatten()

    if len(metrics1) != len(metrics2):
        raise ValueError("Both arrays must have the same length")

    if len(metrics1) < 2:
        raise ValueError("Need at least 2 paired observations")

    # Perform paired t-test
    statistic, p_value_two_sided = stats.ttest_rel(metrics1, metrics2)

    # Adjust p-value for one-sided alternatives
    if alternative == "two-sided":
        p_value = p_value_two_sided
    elif alternative == "greater":
        p_value = p_value_two_sided / 2 if statistic > 0 else 1 - p_value_two_sided / 2
    elif alternative == "less":
        p_value = p_value_two_sided / 2 if statistic < 0 else 1 - p_value_two_sided / 2
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # Cohen's d effect size
    diff = metrics1 - metrics2
    effect_size = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0

    # Confidence interval for mean difference
    n = len(diff)
    se = np.std(diff, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
    ci = (np.mean(diff) - t_crit * se, np.mean(diff) + t_crit * se)

    # Conclusion
    significant = p_value < alpha
    mean_diff = np.mean(diff)
    if significant:
        if mean_diff > 0:
            conclusion = f"Model 1 significantly better (mean diff={mean_diff:.4f}, p={p_value:.4f})"
        else:
            conclusion = f"Model 2 significantly better (mean diff={mean_diff:.4f}, p={p_value:.4f})"
    else:
        conclusion = f"No significant difference (mean diff={mean_diff:.4f}, p={p_value:.4f})"

    return StatisticalTestResult(
        test_name="Paired t-test",
        statistic=float(statistic),
        p_value=float(p_value),
        significant=significant,
        alpha=alpha,
        alternative=alternative,
        conclusion=conclusion,
        effect_size=float(effect_size),
        confidence_interval=ci,
    )


def wilcoxon_test(
    metrics1: np.ndarray,
    metrics2: np.ndarray,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    alpha: float = 0.05,
) -> StatisticalTestResult:
    """
    Wilcoxon signed-rank test for non-parametric paired comparison.

    Use when normality assumption of paired t-test is violated or
    when sample size is small.

    Args:
        metrics1: Performance metrics from model 1
        metrics2: Performance metrics from model 2
        alternative: Type of alternative hypothesis
        alpha: Significance level

    Returns:
        StatisticalTestResult with test outcome

    Example:
        >>> # Compare when normality is questionable
        >>> sharpe_model1 = [0.8, 1.2, 0.5, 1.5, 0.9]
        >>> sharpe_model2 = [0.6, 0.9, 0.4, 1.1, 0.7]
        >>> result = wilcoxon_test(sharpe_model1, sharpe_model2)
    """
    metrics1 = np.asarray(metrics1).flatten()
    metrics2 = np.asarray(metrics2).flatten()

    if len(metrics1) != len(metrics2):
        raise ValueError("Both arrays must have the same length")

    diff = metrics1 - metrics2

    # Remove zero differences (ties at zero)
    nonzero_diff = diff[diff != 0]

    if len(nonzero_diff) < 2:
        return StatisticalTestResult(
            test_name="Wilcoxon signed-rank",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            alternative=alternative,
            conclusion="Insufficient non-zero differences for test",
            effect_size=0.0,
        )

    # Perform Wilcoxon signed-rank test
    try:
        statistic, p_value = stats.wilcoxon(
            metrics1, metrics2, alternative=alternative, zero_method="wilcox"
        )
    except ValueError as e:
        # Handle edge cases
        logger.warning(f"Wilcoxon test failed: {e}")
        return StatisticalTestResult(
            test_name="Wilcoxon signed-rank",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            alternative=alternative,
            conclusion=f"Test failed: {e}",
            effect_size=0.0,
        )

    # Effect size: r = Z / sqrt(N)
    # Approximate Z from p-value
    n = len(nonzero_diff)
    z_approx = stats.norm.ppf(1 - p_value / 2) if p_value < 1 else 0
    effect_size = z_approx / np.sqrt(n) if n > 0 else 0.0

    # Conclusion
    significant = p_value < alpha
    median_diff = np.median(diff)
    if significant:
        if median_diff > 0:
            conclusion = f"Model 1 significantly better (median diff={median_diff:.4f}, p={p_value:.4f})"
        else:
            conclusion = f"Model 2 significantly better (median diff={median_diff:.4f}, p={p_value:.4f})"
    else:
        conclusion = f"No significant difference (median diff={median_diff:.4f}, p={p_value:.4f})"

    return StatisticalTestResult(
        test_name="Wilcoxon signed-rank",
        statistic=float(statistic),
        p_value=float(p_value),
        significant=significant,
        alpha=alpha,
        alternative=alternative,
        conclusion=conclusion,
        effect_size=float(effect_size),
    )


def compare_models(
    y_true: np.ndarray | None = None,
    pred1: np.ndarray | None = None,
    pred2: np.ndarray | None = None,
    metrics1: np.ndarray | None = None,
    metrics2: np.ndarray | None = None,
    alpha: float = 0.05,
) -> dict[str, StatisticalTestResult]:
    """
    Run multiple statistical tests to compare two models.

    Provide either (y_true, pred1, pred2) for prediction comparison
    or (metrics1, metrics2) for metric comparison.

    Args:
        y_true: True values (for DM test)
        pred1: Model 1 predictions (for DM test)
        pred2: Model 2 predictions (for DM test)
        metrics1: Model 1 metrics across folds/runs
        metrics2: Model 2 metrics across folds/runs
        alpha: Significance level

    Returns:
        Dict with results from all applicable tests

    Example:
        >>> # Compare using predictions
        >>> results = compare_models(y_true=y_test, pred1=pred_a, pred2=pred_b)
        >>>
        >>> # Compare using CV metrics
        >>> results = compare_models(metrics1=cv_f1_a, metrics2=cv_f1_b)
    """
    results = {}

    # Diebold-Mariano test (requires predictions)
    if y_true is not None and pred1 is not None and pred2 is not None:
        results["diebold_mariano"] = diebold_mariano_test(
            y_true, pred1, pred2, alpha=alpha
        )

    # Paired tests (require metrics)
    if metrics1 is not None and metrics2 is not None:
        results["paired_ttest"] = paired_ttest(metrics1, metrics2, alpha=alpha)
        results["wilcoxon"] = wilcoxon_test(metrics1, metrics2, alpha=alpha)

    if not results:
        raise ValueError(
            "Provide either (y_true, pred1, pred2) or (metrics1, metrics2)"
        )

    return results


__all__ = [
    "LossFunction",
    "StatisticalTestResult",
    "diebold_mariano_test",
    "paired_ttest",
    "wilcoxon_test",
    "compare_models",
]
