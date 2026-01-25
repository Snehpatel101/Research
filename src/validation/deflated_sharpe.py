"""
Deflated Sharpe Ratio (DSR) for Multiple Testing Bias Correction.

When evaluating N strategies or hyperparameter configurations and selecting the best,
the observed Sharpe ratio is inflated due to selection bias. DSR corrects for this
by deflating the Sharpe to account for the number of trials evaluated.

Academic Reference:
    Bailey, D.H., and Lopez de Prado, M. (2014)
    "The Deflated Sharpe Ratio: Correcting for Selection Bias,
    Backtest Overfitting, and Non-Normality"
    Journal of Portfolio Management, 40(5), 94-107
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551

Key Concepts:
    - Selection Bias: Choosing the best of N trials inflates expected Sharpe
    - Gamma Correction: Adjusts for non-normality (skewness/kurtosis) of returns
    - DSR Threshold: DSR > 0.0 suggests true skill; DSR > 0.5 reasonable for deployment

Formula:
    DSR = SR - gamma * sqrt(Var(SR) * (n-1) / n)

    where gamma accounts for higher moments:
        gamma = (1 - skew*SR + (kurt-3)/4*SR^2)^(-1)

Example:
    >>> from src.validation.deflated_sharpe import compute_deflated_sharpe
    >>> sharpes = np.array([0.8, 1.2, 0.9, 1.5, 0.7])  # 5 trials
    >>> result = compute_deflated_sharpe(
    ...     sharpe_ratio=1.5,  # Best trial
    ...     trial_sharpes=sharpes,
    ...     n_trials=5
    ... )
    >>> print(f"DSR: {result.deflated_sharpe:.3f}, Deploy: {result.should_deploy}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class DSRConfig:
    """
    Configuration for DSR computation.

    Attributes:
        significance_threshold: DSR threshold for statistical significance (default 0.0)
        deployment_threshold: DSR threshold for deployment recommendation (default 0.5)
        gamma_clip_min: Minimum value for gamma to prevent numerical instability
        gamma_clip_max: Maximum value for gamma to prevent numerical instability
        min_trials_for_kurtosis: Minimum trials needed to compute kurtosis (default 4)
    """

    significance_threshold: float = 0.0
    deployment_threshold: float = 0.5
    gamma_clip_min: float = 0.1
    gamma_clip_max: float = 10.0
    min_trials_for_kurtosis: int = 4

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.significance_threshold >= self.deployment_threshold:
            raise ValueError(
                f"significance_threshold ({self.significance_threshold}) must be < "
                f"deployment_threshold ({self.deployment_threshold})"
            )
        if self.gamma_clip_min <= 0:
            raise ValueError(f"gamma_clip_min must be > 0, got {self.gamma_clip_min}")
        if self.gamma_clip_min >= self.gamma_clip_max:
            raise ValueError(
                f"gamma_clip_min ({self.gamma_clip_min}) must be < "
                f"gamma_clip_max ({self.gamma_clip_max})"
            )
        if self.min_trials_for_kurtosis < 4:
            raise ValueError(
                f"min_trials_for_kurtosis must be >= 4 for valid kurtosis, "
                f"got {self.min_trials_for_kurtosis}"
            )


# =============================================================================
# DSR RESULT
# =============================================================================


@dataclass
class DSRResult:
    """
    Result from Deflated Sharpe Ratio computation.

    Attributes:
        sharpe_ratio: Original (inflated) Sharpe ratio from best trial
        deflated_sharpe: Corrected Sharpe ratio accounting for selection bias
        n_trials: Number of trials/strategies evaluated
        variance_sharpe: Variance of Sharpe estimates across trials
        skewness_sharpe: Skewness of Sharpe estimates (0 = normal)
        kurtosis_sharpe: Excess kurtosis of Sharpe estimates (0 = normal)
        gamma_correction: Gamma factor accounting for non-normality
        is_significant: True if DSR > significance_threshold (default 0.0)
        should_deploy: True if DSR > deployment_threshold (default 0.5)
        config: Configuration used for computation
    """

    sharpe_ratio: float
    deflated_sharpe: float
    n_trials: int
    variance_sharpe: float
    skewness_sharpe: float
    kurtosis_sharpe: float
    gamma_correction: float
    is_significant: bool
    should_deploy: bool
    config: DSRConfig

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sharpe_ratio": self.sharpe_ratio,
            "deflated_sharpe": self.deflated_sharpe,
            "n_trials": self.n_trials,
            "variance_sharpe": self.variance_sharpe,
            "skewness_sharpe": self.skewness_sharpe,
            "kurtosis_sharpe": self.kurtosis_sharpe,
            "gamma_correction": self.gamma_correction,
            "is_significant": self.is_significant,
            "should_deploy": self.should_deploy,
            "significance_threshold": self.config.significance_threshold,
            "deployment_threshold": self.config.deployment_threshold,
        }

    def get_risk_level(self) -> str:
        """Get human-readable risk assessment."""
        if self.should_deploy:
            if self.deflated_sharpe > 1.0:
                return "STRONG: High confidence in true skill"
            return "OK: Reasonable confidence for deployment"
        elif self.is_significant:
            return "CAUTION: Statistically significant but below deployment threshold"
        else:
            return "WARNING: Observed Sharpe likely due to selection bias"

    def get_deflation_pct(self) -> float:
        """Get percentage reduction from raw to deflated Sharpe."""
        if abs(self.sharpe_ratio) < 1e-10:
            return 0.0
        return float((self.sharpe_ratio - self.deflated_sharpe) / self.sharpe_ratio * 100)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _validate_inputs(
    sharpe_ratio: float,
    trial_sharpes: np.ndarray,
    n_trials: int,
) -> None:
    """
    Validate inputs for DSR computation.

    Args:
        sharpe_ratio: Best observed Sharpe ratio
        trial_sharpes: Array of all trial Sharpe values
        n_trials: Number of trials evaluated

    Raises:
        ValueError: If inputs are invalid
    """
    if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
        raise ValueError(f"sharpe_ratio must be finite, got {sharpe_ratio}")

    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    if not isinstance(trial_sharpes, np.ndarray):
        raise TypeError(f"trial_sharpes must be numpy array, got {type(trial_sharpes)}")

    if len(trial_sharpes) == 0:
        raise ValueError("trial_sharpes cannot be empty")

    if len(trial_sharpes) != n_trials:
        raise ValueError(
            f"trial_sharpes length ({len(trial_sharpes)}) must equal n_trials ({n_trials})"
        )

    # Check for all NaN
    valid_sharpes = trial_sharpes[~np.isnan(trial_sharpes)]
    if len(valid_sharpes) == 0:
        raise ValueError("trial_sharpes contains all NaN values")


def _compute_trial_variance(trial_sharpes: np.ndarray) -> float:
    """
    Compute variance of Sharpe estimates across trials.

    Args:
        trial_sharpes: Array of Sharpe values from all trials

    Returns:
        Variance of Sharpe estimates (uses ddof=1 for unbiased estimate)
    """
    valid_sharpes = trial_sharpes[~np.isnan(trial_sharpes)]

    if len(valid_sharpes) < 2:
        return 0.0

    return float(np.var(valid_sharpes, ddof=1))


def _compute_trial_skewness(trial_sharpes: np.ndarray) -> float:
    """
    Compute skewness of Sharpe estimates across trials.

    Args:
        trial_sharpes: Array of Sharpe values from all trials

    Returns:
        Sample skewness (Fisher's definition)
        Returns 0.0 if insufficient data
    """
    valid_sharpes = trial_sharpes[~np.isnan(trial_sharpes)]

    if len(valid_sharpes) < 3:
        logger.debug("Insufficient data for skewness (n < 3), returning 0.0")
        return 0.0

    # Use scipy for Fisher's skewness (bias-corrected)
    skew = stats.skew(valid_sharpes, bias=False)
    return float(skew) if not np.isnan(skew) else 0.0


def _compute_trial_kurtosis(trial_sharpes: np.ndarray, min_trials: int = 4) -> float:
    """
    Compute excess kurtosis of Sharpe estimates across trials.

    Args:
        trial_sharpes: Array of Sharpe values from all trials
        min_trials: Minimum number of valid trials for kurtosis computation

    Returns:
        Excess kurtosis (Fisher's definition, normal = 0)
        Returns 0.0 if insufficient data
    """
    valid_sharpes = trial_sharpes[~np.isnan(trial_sharpes)]

    if len(valid_sharpes) < min_trials:
        logger.debug(
            f"Insufficient data for kurtosis (n={len(valid_sharpes)} < {min_trials}), "
            "returning 0.0"
        )
        return 0.0

    # Use scipy for Fisher's kurtosis (excess, bias-corrected)
    kurt = stats.kurtosis(valid_sharpes, fisher=True, bias=False)
    return float(kurt) if not np.isnan(kurt) else 0.0


def _compute_gamma_correction(
    sharpe: float,
    trial_sharpes: np.ndarray,
    config: DSRConfig,
) -> float:
    """
    Compute gamma skewness/kurtosis correction factor.

    The gamma factor adjusts for non-normality in the distribution of Sharpe
    ratios. Higher skewness or kurtosis increases gamma, leading to more
    aggressive deflation.

    Formula from Bailey & Lopez de Prado (2014):
        gamma = (1 - skew*SR + (kurt-3)/4*SR^2)^(-1)

    Note: We use excess kurtosis (kurt - 3 already subtracted in scipy's
    fisher=True mode), so our formula becomes:
        gamma = (1 - skew*SR + kurt/4*SR^2)^(-1)

    Args:
        sharpe: Best observed Sharpe ratio
        trial_sharpes: Array of all trial Sharpe values
        config: DSRConfig with gamma clipping bounds

    Returns:
        Gamma correction factor, clipped to [gamma_clip_min, gamma_clip_max]
    """
    skew = _compute_trial_skewness(trial_sharpes)
    kurt = _compute_trial_kurtosis(trial_sharpes, config.min_trials_for_kurtosis)

    # Formula: gamma = 1 / (1 - skew*SR + (excess_kurt)/4*SR^2)
    # Note: scipy's kurtosis with fisher=True already returns excess kurtosis
    denominator = 1.0 - skew * sharpe + (kurt / 4.0) * (sharpe**2)

    # Prevent division by zero or very small values
    gamma = config.gamma_clip_max if abs(denominator) < 1e-10 else 1.0 / denominator

    # Clip for numerical stability
    gamma = np.clip(gamma, config.gamma_clip_min, config.gamma_clip_max)

    return float(gamma)


def _compute_expected_max_sharpe(n_trials: int, variance: float) -> float:
    """
    Compute expected maximum Sharpe from N trials under null hypothesis.

    Under the null hypothesis (no true skill), the expected maximum Sharpe
    from N independent trials with variance sigma^2 follows approximately:
        E[max(SR_1, ..., SR_N)] ~ sigma * sqrt(2 * ln(N))

    This represents the selection bias - even with no skill, picking the
    best of N trials gives an inflated Sharpe.

    Args:
        n_trials: Number of trials evaluated
        variance: Variance of Sharpe estimates

    Returns:
        Expected maximum Sharpe due to selection bias
    """
    if n_trials <= 1:
        return 0.0

    if variance <= 0:
        return 0.0

    sigma = np.sqrt(variance)
    expected_max = sigma * np.sqrt(2.0 * np.log(n_trials))
    return float(expected_max)


# =============================================================================
# MAIN DSR COMPUTATION
# =============================================================================


def compute_deflated_sharpe(
    sharpe_ratio: float,
    trial_sharpes: np.ndarray,
    n_trials: int,
    deployment_threshold: float = 0.5,
    config: DSRConfig | None = None,
) -> DSRResult:
    """
    Compute Deflated Sharpe Ratio correcting for multiple testing bias.

    When evaluating N strategies/hyperparameters and selecting the best,
    the observed Sharpe is inflated due to selection bias. DSR deflates
    it to account for the number of trials.

    Academic Reference:
        Bailey, D.H., and Lopez de Prado, M. (2014)
        "The Deflated Sharpe Ratio: Correcting for Selection Bias,
        Backtest Overfitting, and Non-Normality"
        Journal of Portfolio Management, 40(5), 94-107

    The deflation formula accounts for:
        1. Number of trials (more trials = more selection bias)
        2. Variance of Sharpe estimates across trials
        3. Non-normality via gamma correction (skewness, kurtosis)

    Args:
        sharpe_ratio: Best observed Sharpe from trials
        trial_sharpes: Array of all trial Sharpe values
        n_trials: Number of trials evaluated
        deployment_threshold: Min DSR for deployment (default: 0.5)
        config: DSRConfig for advanced settings (optional)

    Returns:
        DSRResult with deflated Sharpe and deployment recommendation

    Raises:
        ValueError: If inputs are invalid (n_trials < 1, NaN sharpe, etc.)

    Example:
        >>> sharpes = np.array([0.8, 1.2, 0.9, 1.5, 0.7, 1.1, 0.6, 0.95, 1.3, 1.0])
        >>> result = compute_deflated_sharpe(
        ...     sharpe_ratio=1.5,
        ...     trial_sharpes=sharpes,
        ...     n_trials=10
        ... )
        >>> print(f"Raw SR: {result.sharpe_ratio:.2f}")
        >>> print(f"Deflated SR: {result.deflated_sharpe:.2f}")
        >>> print(f"Deflation: {result.get_deflation_pct():.1f}%")
        >>> if result.should_deploy:
        ...     print("Recommendation: Deploy")
    """
    # Handle configuration
    if config is None:
        config = DSRConfig(deployment_threshold=deployment_threshold)
    else:
        # Override deployment threshold if explicitly provided and different
        if deployment_threshold != 0.5 and deployment_threshold != config.deployment_threshold:
            config = DSRConfig(
                significance_threshold=config.significance_threshold,
                deployment_threshold=deployment_threshold,
                gamma_clip_min=config.gamma_clip_min,
                gamma_clip_max=config.gamma_clip_max,
                min_trials_for_kurtosis=config.min_trials_for_kurtosis,
            )

    # Validate inputs
    _validate_inputs(sharpe_ratio, trial_sharpes, n_trials)

    # Compute statistics
    variance = _compute_trial_variance(trial_sharpes)
    skewness = _compute_trial_skewness(trial_sharpes)
    kurtosis = _compute_trial_kurtosis(trial_sharpes, config.min_trials_for_kurtosis)
    gamma = _compute_gamma_correction(sharpe_ratio, trial_sharpes, config)

    # Compute expected maximum Sharpe from selection bias
    expected_max_sr = _compute_expected_max_sharpe(n_trials, variance)

    # Deflate the Sharpe ratio
    # DSR = SR - gamma * E[max(SR)]
    # This subtracts the expected inflation from selection bias,
    # adjusted by gamma for non-normality
    deflated = sharpe_ratio - gamma * expected_max_sr

    # Determine significance and deployment flags
    is_significant = deflated > config.significance_threshold
    should_deploy = deflated > config.deployment_threshold

    logger.info(
        f"DSR computed: SR={sharpe_ratio:.3f} -> DSR={deflated:.3f} "
        f"(n={n_trials}, gamma={gamma:.3f}, deflation={expected_max_sr:.3f})"
    )

    return DSRResult(
        sharpe_ratio=sharpe_ratio,
        deflated_sharpe=deflated,
        n_trials=n_trials,
        variance_sharpe=variance,
        skewness_sharpe=skewness,
        kurtosis_sharpe=kurtosis,
        gamma_correction=gamma,
        is_significant=is_significant,
        should_deploy=should_deploy,
        config=config,
    )


# =============================================================================
# OPTUNA INTEGRATION
# =============================================================================


def compute_dsr_from_optuna_study(
    study: optuna.Study,
    deployment_threshold: float = 0.5,
    config: DSRConfig | None = None,
) -> DSRResult:
    """
    Compute DSR from Optuna study.

    Extracts trial history from Optuna study and computes DSR to assess
    whether the best hyperparameters are genuinely good or just selected
    due to optimization bias.

    Args:
        study: Completed Optuna study
        deployment_threshold: Min DSR for deployment (default: 0.5)
        config: DSRConfig for advanced settings (optional)

    Returns:
        DSRResult with deflated Sharpe based on optimization history

    Raises:
        ValueError: If study is None or has no completed trials or all trials failed

    Example:
        >>> import optuna
        >>> study = optuna.create_study(direction='maximize')
        >>> # ... run optimization targeting Sharpe ratio ...
        >>> dsr_result = compute_dsr_from_optuna_study(study)
        >>> if dsr_result.should_deploy:
        ...     print("Low overfitting risk - deploy model")
        >>> else:
        ...     print(f"WARNING: DSR={dsr_result.deflated_sharpe:.2f}, likely overfit")
    """
    # Validate input
    if study is None:
        raise ValueError("Optuna study cannot be None")

    # Get completed trials only
    completed_trials = [t for t in study.trials if t.state.is_finished() and t.value is not None]

    if not completed_trials:
        raise ValueError(
            "Optuna study has no completed trials with values. "
            "Ensure trials completed successfully and returned a value."
        )

    # Extract trial values (assumed to be Sharpe ratios or similar metric)
    trial_values = np.array([t.value for t in completed_trials])

    # Handle case where all values are NaN
    valid_values = trial_values[~np.isnan(trial_values)]
    if len(valid_values) == 0:
        raise ValueError("All trial values are NaN, cannot compute DSR")

    # Get best value (depends on study direction)
    if study.direction.name == "MINIMIZE":
        # For minimization, lower is better, but DSR expects higher = better
        # So we negate the values for DSR computation
        logger.warning(
            "Study direction is MINIMIZE. Negating values for DSR computation. "
            "Ensure the metric being optimized is appropriate (e.g., negative Sharpe)."
        )
        trial_values = -trial_values
        best_value = -study.best_value
    else:
        best_value = study.best_value

    n_trials = len(trial_values)

    logger.info(
        f"Computing DSR from Optuna study: {n_trials} completed trials, "
        f"best value: {best_value:.4f}"
    )

    return compute_deflated_sharpe(
        sharpe_ratio=best_value,
        trial_sharpes=trial_values,
        n_trials=n_trials,
        deployment_threshold=deployment_threshold,
        config=config,
    )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def dsr_gate(
    dsr_result: DSRResult,
    strict: bool = False,
) -> tuple[bool, str]:
    """
    Gate function for deployment decisions based on DSR.

    Args:
        dsr_result: DSRResult from compute_deflated_sharpe
        strict: If True, require higher confidence for deployment

    Returns:
        Tuple of (should_proceed, reason)

    Example:
        >>> result = compute_deflated_sharpe(...)
        >>> proceed, reason = dsr_gate(result)
        >>> if not proceed:
        ...     raise ValueError(f"Deployment blocked: {reason}")
    """
    if strict:
        # In strict mode, require DSR > 1.0 for strong confidence
        threshold = max(1.0, dsr_result.config.deployment_threshold)
    else:
        threshold = dsr_result.config.deployment_threshold

    if dsr_result.deflated_sharpe < threshold:
        return False, (
            f"DSR ({dsr_result.deflated_sharpe:.3f}) below threshold ({threshold:.2f}). "
            f"Raw Sharpe {dsr_result.sharpe_ratio:.3f} deflated by "
            f"{dsr_result.get_deflation_pct():.1f}% due to {dsr_result.n_trials} trials. "
            f"Risk: {dsr_result.get_risk_level()}"
        )

    return True, (
        f"DSR ({dsr_result.deflated_sharpe:.3f}) exceeds threshold ({threshold:.2f}). "
        f"Risk: {dsr_result.get_risk_level()}"
    )


def analyze_selection_bias(
    trial_sharpes: np.ndarray,
    strategy_names: list[str] | None = None,
    config: DSRConfig | None = None,
) -> dict[str, Any]:
    """
    Comprehensive selection bias analysis for multiple strategies.

    Computes DSR for the best strategy and provides detailed breakdown
    of the selection bias correction.

    Args:
        trial_sharpes: Array of Sharpe ratios from all trials/strategies
        strategy_names: Optional names for each strategy
        config: DSRConfig for advanced settings

    Returns:
        Dict with analysis results including DSR, statistics, and recommendations

    Example:
        >>> sharpes = np.array([0.8, 1.2, 1.5, 0.9, 1.1])
        >>> names = ["model_a", "model_b", "model_c", "model_d", "model_e"]
        >>> analysis = analyze_selection_bias(sharpes, strategy_names=names)
        >>> print(f"Best model: {analysis['best_strategy_name']}")
        >>> print(f"DSR: {analysis['deflated_sharpe']:.3f}")
    """
    if config is None:
        config = DSRConfig()

    n_trials = len(trial_sharpes)

    if strategy_names is None:
        strategy_names = [f"strategy_{i}" for i in range(n_trials)]

    if len(strategy_names) != n_trials:
        raise ValueError(
            f"strategy_names length ({len(strategy_names)}) must match "
            f"trial_sharpes length ({n_trials})"
        )

    # Find best strategy
    valid_mask = ~np.isnan(trial_sharpes)
    if not valid_mask.any():
        raise ValueError("All trial Sharpe values are NaN")

    best_idx = int(np.nanargmax(trial_sharpes))
    best_sharpe = float(trial_sharpes[best_idx])
    best_name = strategy_names[best_idx]

    # Compute DSR
    dsr_result = compute_deflated_sharpe(
        sharpe_ratio=best_sharpe,
        trial_sharpes=trial_sharpes,
        n_trials=n_trials,
        config=config,
    )

    # Per-strategy analysis
    strategy_analysis = []
    for i, (name, sharpe) in enumerate(zip(strategy_names, trial_sharpes, strict=False)):
        strategy_analysis.append(
            {
                "name": name,
                "sharpe_ratio": float(sharpe) if not np.isnan(sharpe) else None,
                "rank": int(np.sum(trial_sharpes >= sharpe)) if not np.isnan(sharpe) else None,
                "is_best": i == best_idx,
            }
        )

    return {
        "n_trials": n_trials,
        "best_strategy_idx": best_idx,
        "best_strategy_name": best_name,
        "raw_sharpe": dsr_result.sharpe_ratio,
        "deflated_sharpe": dsr_result.deflated_sharpe,
        "deflation_pct": dsr_result.get_deflation_pct(),
        "variance_sharpe": dsr_result.variance_sharpe,
        "skewness_sharpe": dsr_result.skewness_sharpe,
        "kurtosis_sharpe": dsr_result.kurtosis_sharpe,
        "gamma_correction": dsr_result.gamma_correction,
        "is_significant": dsr_result.is_significant,
        "should_deploy": dsr_result.should_deploy,
        "risk_level": dsr_result.get_risk_level(),
        "strategy_analysis": strategy_analysis,
    }


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    "DSRConfig",
    "DSRResult",
    "compute_deflated_sharpe",
    "compute_dsr_from_optuna_study",
    "dsr_gate",
    "analyze_selection_bias",
]
