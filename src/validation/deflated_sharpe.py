"""
Probabilistic and Deflated Sharpe Ratio (PSR / DSR).

When N strategy configurations are tried and the best one is kept, its Sharpe
ratio is inflated by selection bias. The Deflated Sharpe Ratio answers: *what
is the probability that the true Sharpe ratio of the selected strategy is
positive, after accounting for the number of trials, the sample length and the
non-normality of returns?*

Academic Reference:
    Bailey, D.H., and Lopez de Prado, M. (2014)
    "The Deflated Sharpe Ratio: Correcting for Selection Bias,
    Backtest Overfitting, and Non-Normality"
    Journal of Portfolio Management, 40(5), 94-107
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551

Formulas (all Sharpe ratios are NON-annualized, per-period):

    PSR(SR*) = Phi( (SR_hat - SR*) * sqrt(T - 1)
                    / sqrt(1 - g3 * SR_hat + ((g4 - 1) / 4) * SR_hat^2) )

        SR_hat  per-period Sharpe of the selected strategy's returns
        T       number of returns
        g3      skewness of the returns
        g4      kurtosis of the returns (NON-excess: normal = 3)

    SR0 = sqrt(V[{SR_n}]) * ((1 - gamma) * Phi^-1(1 - 1/N)
                             + gamma * Phi^-1(1 - 1/(N * e)))

        V[{SR_n}]  variance of the N trial Sharpe ratios (per-period)
        N          number of independent trials
        gamma      Euler-Mascheroni constant (~0.5772)

    DSR = PSR(SR0)            -- a probability in [0, 1]

Gate: deploy only when DSR >= 0.95 (configurable).

Example:
    >>> from src.validation.deflated_sharpe import compute_deflated_sharpe_from_returns
    >>> result = compute_deflated_sharpe_from_returns(
    ...     returns=best_strategy_returns,        # per-period returns, length T
    ...     trial_sharpes=all_trial_sharpes,      # per-period Sharpe of every trial
    ... )
    >>> print(f"DSR: {result.dsr:.3f}, Deploy: {result.should_deploy}")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)

EULER_MASCHERONI = 0.5772156649015329
"""Euler-Mascheroni constant used in the expected-maximum approximation."""

NORMAL_KURTOSIS = 3.0
"""Kurtosis (non-excess) of the normal distribution."""


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class DSRComputeConfig:
    """
    Gate thresholds for the Deflated Sharpe Ratio.

    Both thresholds are probabilities (DSR and PSR live in [0, 1]).

    Attributes:
        deployment_threshold: Minimum DSR required to recommend deployment
            (default 0.95, i.e. 95% confidence that the selected strategy's true
            Sharpe is positive after deflating for selection bias).
        strict_threshold: Minimum DSR used by ``dsr_gate(strict=True)``
            (default 0.99).
    """

    deployment_threshold: float = 0.95
    strict_threshold: float = 0.99

    def __post_init__(self) -> None:
        """Validate configuration."""
        for name in ("deployment_threshold", "strict_threshold"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {value}")
        if self.strict_threshold < self.deployment_threshold:
            raise ValueError(
                f"strict_threshold ({self.strict_threshold}) must be >= "
                f"deployment_threshold ({self.deployment_threshold})"
            )


# =============================================================================
# DSR RESULT
# =============================================================================


@dataclass
class DSRResult:
    """
    Result of a Deflated Sharpe Ratio computation.

    Attributes:
        sharpe_ratio: Observed per-period (non-annualized) Sharpe of the
            selected strategy (SR_hat).
        dsr: Deflated Sharpe Ratio = PSR(SR0), a probability in [0, 1].
        psr: PSR(0) -- probability that the true Sharpe is > 0 ignoring
            selection bias (useful as a reference; always >= dsr when SR0 >= 0).
        expected_max_sharpe: SR0, the expected maximum per-period Sharpe of
            N independent zero-skill trials.
        n_trials: Number of independent trials N.
        n_observations: Number of returns T behind ``sharpe_ratio``.
        skewness: Skewness g3 of the selected strategy's returns.
        kurtosis: Kurtosis g4 (non-excess) of the selected strategy's returns.
        variance_trial_sharpes: V[{SR_n}], variance of the per-period trial Sharpes.
        is_significant: True if PSR(0) >= deployment_threshold.
        should_deploy: True if DSR >= deployment_threshold.
        config: Thresholds used.
    """

    sharpe_ratio: float
    dsr: float
    psr: float
    expected_max_sharpe: float
    n_trials: int
    n_observations: int
    skewness: float
    kurtosis: float
    variance_trial_sharpes: float
    is_significant: bool
    should_deploy: bool
    config: DSRComputeConfig

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sharpe_ratio": self.sharpe_ratio,
            "dsr": self.dsr,
            "psr": self.psr,
            "expected_max_sharpe": self.expected_max_sharpe,
            "n_trials": self.n_trials,
            "n_observations": self.n_observations,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "variance_trial_sharpes": self.variance_trial_sharpes,
            "is_significant": self.is_significant,
            "should_deploy": self.should_deploy,
            "deployment_threshold": self.config.deployment_threshold,
        }

    def get_risk_level(self) -> str:
        """Get human-readable risk assessment."""
        if self.dsr >= self.config.strict_threshold:
            return "STRONG: Sharpe survives deflation with high confidence"
        if self.should_deploy:
            return "OK: Sharpe survives deflation for selection bias"
        if self.is_significant:
            return "CAUTION: Significant on its own, but not after deflating for trials"
        return "WARNING: Observed Sharpe is consistent with selection bias / noise"


# =============================================================================
# CORE FORMULAS
# =============================================================================


def _finite(values: np.ndarray) -> np.ndarray:
    """Return the finite entries of a 1D array."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def sharpe_ratio_per_period(returns: np.ndarray) -> float:
    """
    Per-period (non-annualized) Sharpe ratio: mean / sample std (ddof=1).

    Returns 0.0 for fewer than 2 finite returns or zero dispersion.
    """
    r = _finite(returns)
    if len(r) < 2:
        return 0.0
    std = float(np.std(r, ddof=1))
    if std < 1e-15:
        return 0.0
    return float(np.mean(r) / std)


def return_moments(returns: np.ndarray) -> tuple[float, float]:
    """
    Skewness g3 and NON-excess kurtosis g4 of a return series.

    Uses bias-corrected sample estimators when at least 4 finite returns are
    available; otherwise returns the normal values (0, 3).
    """
    r = _finite(returns)
    if len(r) < 4 or float(np.std(r)) < 1e-15:
        return 0.0, NORMAL_KURTOSIS
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))
    if not np.isfinite(skew) or not np.isfinite(kurt):
        return 0.0, NORMAL_KURTOSIS
    return skew, kurt


def probabilistic_sharpe_ratio(
    sharpe_ratio: float,
    benchmark_sharpe: float,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = NORMAL_KURTOSIS,
) -> float:
    """
    Probabilistic Sharpe Ratio PSR(SR*).

    Args:
        sharpe_ratio: Observed per-period Sharpe SR_hat (NOT annualized).
        benchmark_sharpe: Benchmark per-period Sharpe SR*.
        n_observations: Number of returns T (>= 2).
        skewness: Skewness g3 of the returns.
        kurtosis: Kurtosis g4 of the returns, NON-excess (normal = 3).

    Returns:
        Probability that the true Sharpe exceeds ``benchmark_sharpe``.

    Raises:
        ValueError: On non-finite inputs, T < 2, or a non-positive variance term
            (impossible for a real distribution since g4 >= 1 + g3^2).
    """
    if not (np.isfinite(sharpe_ratio) and np.isfinite(benchmark_sharpe)):
        raise ValueError(
            f"Sharpe ratios must be finite, got SR={sharpe_ratio}, SR*={benchmark_sharpe}"
        )
    if n_observations < 2:
        raise ValueError(f"n_observations must be >= 2, got {n_observations}")

    variance_term = 1.0 - skewness * sharpe_ratio + ((kurtosis - 1.0) / 4.0) * sharpe_ratio**2
    if variance_term <= 0:
        raise ValueError(
            f"Non-positive PSR variance term ({variance_term:.4g}) for "
            f"skewness={skewness}, kurtosis={kurtosis}. Kurtosis must be the "
            "non-excess kurtosis (normal = 3) and satisfy kurtosis >= 1 + skewness^2."
        )

    z = (sharpe_ratio - benchmark_sharpe) * math.sqrt(n_observations - 1) / math.sqrt(variance_term)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, variance_trial_sharpes: float) -> float:
    """
    Expected maximum Sharpe SR0 of N independent zero-skill trials.

        SR0 = sqrt(V) * ((1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N e)))

    Args:
        n_trials: Number of independent trials N (>= 1). N = 1 gives SR0 = 0.
        variance_trial_sharpes: Variance of the per-period trial Sharpes.

    Returns:
        SR0 in the same (per-period) units as the trial Sharpes.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if n_trials == 1 or variance_trial_sharpes <= 0:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    factor = (1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2
    return float(math.sqrt(variance_trial_sharpes) * factor)


# =============================================================================
# MAIN DSR COMPUTATION
# =============================================================================


def compute_deflated_sharpe(
    sharpe_ratio: float,
    trial_sharpes: np.ndarray,
    *,
    n_observations: int,
    n_trials: int | None = None,
    skewness: float = 0.0,
    kurtosis: float = NORMAL_KURTOSIS,
    config: DSRComputeConfig | None = None,
) -> DSRResult:
    """
    Deflated Sharpe Ratio of a selected strategy (Bailey & Lopez de Prado, 2014).

    Args:
        sharpe_ratio: Per-period (non-annualized) Sharpe SR_hat of the selected
            strategy.
        trial_sharpes: Per-period Sharpe ratios of ALL trials (including the
            selected one). Their variance feeds SR0. Non-finite values
            (failed trials) are ignored.
        n_observations: Number of returns T behind ``sharpe_ratio``.
        n_trials: Number of independent trials N. Defaults to the number of
            finite ``trial_sharpes``; pass a smaller effective N when trials
            are strongly correlated.
        skewness: Skewness g3 of the selected strategy's returns.
        kurtosis: NON-excess kurtosis g4 of the selected strategy's returns.
        config: Gate thresholds (default: deploy at DSR >= 0.95).

    Returns:
        DSRResult with DSR = PSR(SR0) and gate flags.

    Raises:
        ValueError: If inputs are invalid (non-finite Sharpe, T < 2, no finite
            trial Sharpes, N < 1).
    """
    if config is None:
        config = DSRComputeConfig()

    valid = _finite(np.asarray(trial_sharpes))
    if len(valid) == 0:
        raise ValueError("trial_sharpes contains no finite values")

    n_eff = len(valid) if n_trials is None else int(n_trials)
    if n_eff < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_eff}")

    variance = float(np.var(valid, ddof=1)) if len(valid) >= 2 else 0.0
    sr0 = expected_max_sharpe(n_eff, variance)

    dsr = probabilistic_sharpe_ratio(sharpe_ratio, sr0, n_observations, skewness, kurtosis)
    psr = probabilistic_sharpe_ratio(sharpe_ratio, 0.0, n_observations, skewness, kurtosis)

    logger.info(
        f"DSR computed: SR={sharpe_ratio:.4f} (per-period), SR0={sr0:.4f}, "
        f"T={n_observations}, N={n_eff}, skew={skewness:.3f}, kurt={kurtosis:.3f} "
        f"-> DSR={dsr:.4f}, PSR(0)={psr:.4f}"
    )

    return DSRResult(
        sharpe_ratio=float(sharpe_ratio),
        dsr=dsr,
        psr=psr,
        expected_max_sharpe=sr0,
        n_trials=n_eff,
        n_observations=int(n_observations),
        skewness=float(skewness),
        kurtosis=float(kurtosis),
        variance_trial_sharpes=variance,
        is_significant=psr >= config.deployment_threshold,
        should_deploy=dsr >= config.deployment_threshold,
        config=config,
    )


def compute_deflated_sharpe_from_returns(
    returns: np.ndarray,
    trial_sharpes: np.ndarray,
    n_trials: int | None = None,
    config: DSRComputeConfig | None = None,
) -> DSRResult:
    """
    DSR of a selected strategy given its per-period returns.

    Derives SR_hat, T, skewness and (non-excess) kurtosis from ``returns``.

    Args:
        returns: Per-period returns of the selected strategy (NaNs dropped).
        trial_sharpes: Per-period Sharpe ratios of all trials.
        n_trials: Number of independent trials (default: finite trial count).
        config: Gate thresholds.
    """
    r = _finite(np.asarray(returns))
    skew, kurt = return_moments(r)
    return compute_deflated_sharpe(
        sharpe_ratio=sharpe_ratio_per_period(r),
        trial_sharpes=trial_sharpes,
        n_observations=len(r),
        n_trials=n_trials,
        skewness=skew,
        kurtosis=kurt,
        config=config,
    )


# =============================================================================
# OPTUNA INTEGRATION
# =============================================================================


def is_sharpe_like_metric(metric_name: str) -> bool:
    """
    Check whether a metric name corresponds to a Sharpe-like ratio.

    DSR is derived for Sharpe ratios. Applying it to bounded metrics
    (F1 in [0,1], accuracy in [0,1]) is statistically invalid.

    Args:
        metric_name: The optimization metric name (e.g., "sharpe_ratio", "f1_weighted").

    Returns:
        True if the metric is Sharpe-like and DSR deflation is meaningful.
    """
    _SHARPE_LIKE_METRICS = {
        "sharpe_ratio",
        "sortino_ratio",
        "sharpe",
        "sortino",
        "calmar_ratio",
        "calmar",
    }
    return metric_name.lower().strip() in _SHARPE_LIKE_METRICS


def compute_dsr_from_optuna_study(
    study: optuna.Study,
    n_observations: int,
    *,
    periods_per_year: float = 1.0,
    skewness: float = 0.0,
    kurtosis: float = NORMAL_KURTOSIS,
    n_trials: int | None = None,
    config: DSRComputeConfig | None = None,
    metric_name: str | None = None,
) -> DSRResult:
    """
    DSR of the best trial of an Optuna study whose objective is a Sharpe ratio.

    Trial values are converted to per-period Sharpe by dividing by
    ``sqrt(periods_per_year)`` (pass 1.0 if the objective is already
    per-period). Failed / pruned / non-finite trials are ignored.

    Optuna does not keep the best trial's return series, so its moments must
    be supplied (``skewness``/``kurtosis``); the defaults assume normal returns.

    Args:
        study: Completed Optuna study.
        n_observations: Number of returns T behind each trial's Sharpe
            (e.g. total out-of-sample bars across CV folds).
        periods_per_year: Annualization used by the objective (bars per year).
        skewness: Skewness of the best trial's returns.
        kurtosis: NON-excess kurtosis of the best trial's returns.
        n_trials: Effective number of independent trials (default: all finite
            completed trials).
        config: Gate thresholds.
        metric_name: If given and not Sharpe-like, raises ValueError.

    Raises:
        ValueError: For non-Sharpe metrics, a missing study, or no finite trials.
    """
    if metric_name is not None and not is_sharpe_like_metric(metric_name):
        raise ValueError(
            f"DSR deflation skipped: metric '{metric_name}' is not a Sharpe-like ratio. "
            "DSR is only defined for Sharpe ratios, not bounded metrics like F1 or accuracy."
        )
    if study is None:
        raise ValueError("Optuna study cannot be None")
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be > 0, got {periods_per_year}")

    values = np.array(
        [t.value for t in study.trials if t.state.is_finished() and t.value is not None],
        dtype=np.float64,
    )
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("Optuna study has no completed trials with finite values")

    if study.direction.name == "MINIMIZE":
        logger.warning(
            "Study direction is MINIMIZE. Negating values for DSR computation. "
            "Ensure the metric being optimized is a negative Sharpe."
        )
        values = -values

    per_period = values / math.sqrt(periods_per_year)
    best = float(np.max(per_period))

    logger.info(
        f"Computing DSR from Optuna study: {len(per_period)} finite trials, "
        f"best per-period Sharpe {best:.4f}, T={n_observations}"
    )

    return compute_deflated_sharpe(
        sharpe_ratio=best,
        trial_sharpes=per_period,
        n_observations=n_observations,
        n_trials=n_trials,
        skewness=skewness,
        kurtosis=kurtosis,
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
    Deployment gate based on DSR.

    Args:
        dsr_result: DSRResult from compute_deflated_sharpe.
        strict: Use ``config.strict_threshold`` instead of ``deployment_threshold``.

    Returns:
        Tuple of (should_proceed, reason).
    """
    cfg = dsr_result.config
    threshold = cfg.strict_threshold if strict else cfg.deployment_threshold

    if dsr_result.dsr < threshold:
        return False, (
            f"DSR ({dsr_result.dsr:.3f}) below threshold ({threshold:.2f}). "
            f"Per-period Sharpe {dsr_result.sharpe_ratio:.4f} vs expected max under "
            f"the null SR0={dsr_result.expected_max_sharpe:.4f} "
            f"(N={dsr_result.n_trials}, T={dsr_result.n_observations}). "
            f"Risk: {dsr_result.get_risk_level()}"
        )

    return True, (
        f"DSR ({dsr_result.dsr:.3f}) meets threshold ({threshold:.2f}). "
        f"Risk: {dsr_result.get_risk_level()}"
    )


def analyze_selection_bias(
    returns_matrix: np.ndarray,
    strategy_names: list[str] | None = None,
    config: DSRComputeConfig | None = None,
) -> dict[str, Any]:
    """
    Selection-bias analysis for N strategies with per-period returns.

    Args:
        returns_matrix: Array of shape (T, N) -- per-period returns of each
            strategy configuration (NaNs are ignored per column).
        strategy_names: Optional names for the N strategies.
        config: Gate thresholds.

    Returns:
        Dict with the best strategy, its DSR and per-strategy Sharpe ratios.
    """
    matrix = np.asarray(returns_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"returns_matrix must be 2D (T, N), got shape {matrix.shape}")
    n_strategies = matrix.shape[1]

    if strategy_names is None:
        strategy_names = [f"strategy_{i}" for i in range(n_strategies)]
    if len(strategy_names) != n_strategies:
        raise ValueError(
            f"strategy_names length ({len(strategy_names)}) must match "
            f"number of strategies ({n_strategies})"
        )

    sharpes = np.array([sharpe_ratio_per_period(matrix[:, j]) for j in range(n_strategies)])
    best_idx = int(np.argmax(sharpes))

    dsr_result = compute_deflated_sharpe_from_returns(
        returns=matrix[:, best_idx],
        trial_sharpes=sharpes,
        config=config,
    )

    order = np.argsort(-sharpes)
    ranks = np.empty(n_strategies, dtype=int)
    ranks[order] = np.arange(1, n_strategies + 1)

    return {
        "n_trials": n_strategies,
        "best_strategy_idx": best_idx,
        "best_strategy_name": strategy_names[best_idx],
        "raw_sharpe": dsr_result.sharpe_ratio,
        "dsr": dsr_result.dsr,
        "psr": dsr_result.psr,
        "expected_max_sharpe": dsr_result.expected_max_sharpe,
        "n_observations": dsr_result.n_observations,
        "skewness": dsr_result.skewness,
        "kurtosis": dsr_result.kurtosis,
        "variance_trial_sharpes": dsr_result.variance_trial_sharpes,
        "is_significant": dsr_result.is_significant,
        "should_deploy": dsr_result.should_deploy,
        "risk_level": dsr_result.get_risk_level(),
        "strategy_analysis": [
            {
                "name": name,
                "sharpe_ratio": float(sharpes[i]),
                "rank": int(ranks[i]),
                "is_best": i == best_idx,
            }
            for i, name in enumerate(strategy_names)
        ],
    }


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    "EULER_MASCHERONI",
    "DSRComputeConfig",
    "DSRResult",
    "analyze_selection_bias",
    "compute_deflated_sharpe",
    "compute_deflated_sharpe_from_returns",
    "compute_dsr_from_optuna_study",
    "dsr_gate",
    "expected_max_sharpe",
    "is_sharpe_like_metric",
    "probabilistic_sharpe_ratio",
    "return_moments",
    "sharpe_ratio_per_period",
]
