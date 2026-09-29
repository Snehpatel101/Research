"""
Probability of Backtest Overfitting (PBO) via CSCV.

PBO is the probability that the configuration selected as best in-sample ranks
at or below the median out-of-sample. It is estimated with Combinatorially
Symmetric Cross-Validation (CSCV) over a T x N matrix of per-period returns.

Reference:
    Bailey, D.H., Borwein, J., Lopez de Prado, M., Zhu, Q.J. (2017)
    "The Probability of Backtest Overfitting", Journal of Computational Finance.
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253

Algorithm (CSCV):
    1. M is a T x N matrix: column n holds the per-period returns of strategy
       configuration n over the same T periods.
    2. Split the rows of M into S contiguous blocks of equal size (S even).
    3. For every combination C of S/2 blocks:
         IS  = rows in C, OOS = remaining rows
         n*  = argmax_n Sharpe_IS(n)
         w   = rank of n* among the OOS Sharpes / (N + 1)     (rank 1 = worst)
         lam = ln(w / (1 - w))
    4. PBO = fraction of combinations with lam <= 0.

Sharpe ratios are per-period (NOT annualized) -- ranks are scale invariant.

Example:
    >>> from src.validation.cv.pbo import compute_pbo
    >>> result = compute_pbo(returns_matrix)   # shape (T, N)
    >>> print(f"PBO: {result.pbo:.3f}, Overfit: {result.is_overfit}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class PBOConfig:
    """
    Configuration for CSCV-based PBO.

    Attributes:
        n_partitions: Number of row blocks S (must be even; paper uses 16).
            All C(S, S/2) combinations are evaluated.
        warn_threshold: PBO above which ``is_overfit`` is set (default 0.5).
        block_threshold: PBO above which ``should_block`` is set (default 0.8).
    """

    n_partitions: int = 16
    warn_threshold: float = 0.5
    block_threshold: float = 0.8

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.n_partitions < 2 or self.n_partitions % 2 != 0:
            raise ValueError(f"n_partitions must be an even integer >= 2, got {self.n_partitions}")
        if not 0 < self.warn_threshold < 1:
            raise ValueError(f"warn_threshold must be in (0, 1), got {self.warn_threshold}")
        if not 0 < self.block_threshold <= 1:
            raise ValueError(f"block_threshold must be in (0, 1], got {self.block_threshold}")
        if self.warn_threshold >= self.block_threshold:
            raise ValueError("warn_threshold must be < block_threshold")


# =============================================================================
# PBO RESULT
# =============================================================================


@dataclass
class PBOResult:
    """
    Result of a CSCV PBO computation.

    Attributes:
        pbo: Probability of Backtest Overfitting = fraction of logits <= 0.
        logit_distribution: lambda_c for every CSCV combination c.
        is_sharpe_best: IS Sharpe of the IS-selected strategy, per combination.
        oos_sharpe_best: OOS Sharpe of the IS-selected strategy, per combination.
        performance_degradation: Slope of OOS on IS Sharpe of the selected
            strategy across combinations (paper Sec. 5; < 1 means decay,
            <= 0 means IS performance carries no OOS information).
        prob_oos_loss: Fraction of combinations where the selected strategy
            has a negative OOS Sharpe.
        rank_correlation: Mean Spearman correlation between IS and OOS Sharpe
            ranks across combinations.
        is_overfit: pbo > warn_threshold.
        should_block: pbo > block_threshold.
        n_combinations: Number of CSCV combinations evaluated, C(S, S/2).
        n_strategies: N.
        n_observations: Rows of M used (T truncated to a multiple of S).
        best_is_strategy_idx: Strategy with the best full-sample Sharpe.
        best_is_oos_rank: Mean relative OOS rank w of the IS-selected strategy.
        config: PBOConfig used.
    """

    pbo: float
    logit_distribution: np.ndarray
    is_sharpe_best: np.ndarray
    oos_sharpe_best: np.ndarray
    performance_degradation: float
    prob_oos_loss: float
    rank_correlation: float
    is_overfit: bool
    should_block: bool
    n_combinations: int
    n_strategies: int
    n_observations: int
    best_is_strategy_idx: int
    best_is_oos_rank: float
    config: PBOConfig

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pbo": self.pbo,
            "logit_distribution": self.logit_distribution.tolist(),
            "performance_degradation": self.performance_degradation,
            "prob_oos_loss": self.prob_oos_loss,
            "rank_correlation": self.rank_correlation,
            "is_overfit": self.is_overfit,
            "should_block": self.should_block,
            "n_combinations": self.n_combinations,
            "n_strategies": self.n_strategies,
            "n_observations": self.n_observations,
            "n_partitions": self.config.n_partitions,
            "best_is_strategy_idx": self.best_is_strategy_idx,
            "best_is_oos_rank": self.best_is_oos_rank,
            "warn_threshold": self.config.warn_threshold,
            "block_threshold": self.config.block_threshold,
        }

    def get_risk_level(self) -> str:
        """Get human-readable risk level."""
        if self.should_block:
            return "CRITICAL: High overfitting risk, block deployment"
        if self.is_overfit:
            return "WARNING: Moderate overfitting risk, review carefully"
        if self.pbo > 0.3:
            return "CAUTION: Some overfitting signal detected"
        return "OK: Low overfitting risk"


# =============================================================================
# CSCV
# =============================================================================


def _sharpe_from_moments(sums: np.ndarray, sumsq: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Per-period Sharpe (ddof=1) from sums, sums of squares and counts."""
    mean = sums / counts
    var = (sumsq - counts * mean**2) / np.maximum(counts - 1, 1)
    std = np.sqrt(np.maximum(var, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(std > 1e-15, mean / std, 0.0)
    return sharpe


def _rowwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation of matching rows of a and b (0 where undefined)."""
    a_c = a - a.mean(axis=1, keepdims=True)
    b_c = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((a_c**2).sum(axis=1) * (b_c**2).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, (a_c * b_c).sum(axis=1) / denom, 0.0)
    return corr


def compute_pbo(
    returns_matrix: np.ndarray,
    config: PBOConfig | None = None,
) -> PBOResult:
    """
    Probability of Backtest Overfitting via CSCV.

    Args:
        returns_matrix: Array of shape (T, N) -- per-period returns of N
            strategy configurations over the same T periods. Must be finite
            (use 0.0 for periods with no position).
        config: PBOConfig (default S = 16).

    Returns:
        PBOResult with PBO and the logit distribution.

    Raises:
        ValueError: If N < 2, the matrix has non-finite values, or there are
            fewer than 2 rows per block.
    """
    if config is None:
        config = PBOConfig()

    matrix = np.asarray(returns_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"returns_matrix must be 2D (T, N), got shape {matrix.shape}")
    n_rows, n_strategies = matrix.shape
    if n_strategies < 2:
        raise ValueError(f"Need at least 2 strategies (columns), got {n_strategies}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("returns_matrix contains non-finite values; clean or zero-fill first")

    n_blocks = config.n_partitions
    block_size = n_rows // n_blocks
    if block_size < 2:
        raise ValueError(
            f"Need at least 2 rows per block: T={n_rows} rows, S={n_blocks} partitions"
        )
    n_used = block_size * n_blocks
    if n_used < n_rows:
        logger.info(
            f"CSCV: dropping the oldest {n_rows - n_used} rows so T is a multiple of S={n_blocks}"
        )
    used = matrix[n_rows - n_used :]

    # Per-block sufficient statistics: (S, N)
    blocks = used.reshape(n_blocks, block_size, n_strategies)
    block_sums = blocks.sum(axis=1)
    block_sumsq = (blocks**2).sum(axis=1)
    total_sums = block_sums.sum(axis=0)
    total_sumsq = block_sumsq.sum(axis=0)

    # IS membership of every combination: (C, S)
    half = n_blocks // 2
    combos = list(combinations(range(n_blocks), half))
    membership = np.zeros((len(combos), n_blocks), dtype=np.float64)
    for c, combo in enumerate(combos):
        membership[c, list(combo)] = 1.0

    is_count = float(half * block_size)
    oos_count = float(n_used - half * block_size)
    is_sums = membership @ block_sums
    is_sumsq = membership @ block_sumsq
    is_sharpe = _sharpe_from_moments(is_sums, is_sumsq, np.full_like(is_sums, is_count))
    oos_sharpe = _sharpe_from_moments(
        total_sums - is_sums, total_sumsq - is_sumsq, np.full_like(is_sums, oos_count)
    )

    rows = np.arange(len(combos))
    best_is = np.argmax(is_sharpe, axis=1)
    oos_ranks = stats.rankdata(oos_sharpe, axis=1)  # 1 = worst, N = best (ties averaged)
    omega = oos_ranks[rows, best_is] / (n_strategies + 1)
    logits = np.log(omega / (1.0 - omega))
    pbo = float(np.mean(logits <= 0))

    is_best = is_sharpe[rows, best_is]
    oos_best = oos_sharpe[rows, best_is]
    has_spread = np.ptp(is_best) > 1e-15
    degradation = float(np.polyfit(is_best, oos_best, 1)[0]) if has_spread else 0.0
    prob_oos_loss = float(np.mean(oos_best < 0))
    rank_corr = float(np.mean(_rowwise_pearson(stats.rankdata(is_sharpe, axis=1), oos_ranks)))

    full_sharpe = _sharpe_from_moments(
        total_sums, total_sumsq, np.full_like(total_sums, float(n_used))
    )

    logger.info(
        f"CSCV PBO: {pbo:.3f} over {len(combos)} combinations "
        f"(T={n_used}, N={n_strategies}, S={n_blocks})"
    )

    return PBOResult(
        pbo=pbo,
        logit_distribution=logits,
        is_sharpe_best=is_best,
        oos_sharpe_best=oos_best,
        performance_degradation=degradation,
        prob_oos_loss=prob_oos_loss,
        rank_correlation=rank_corr,
        is_overfit=pbo > config.warn_threshold,
        should_block=pbo > config.block_threshold,
        n_combinations=len(combos),
        n_strategies=n_strategies,
        n_observations=n_used,
        best_is_strategy_idx=int(np.argmax(full_sharpe)),
        best_is_oos_rank=float(np.mean(omega)),
        config=config,
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def directional_strategy_returns(
    predictions: np.ndarray,
    forward_returns: np.ndarray,
    cost_per_turnover: float | np.ndarray = 0.0,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """
    Per-bar returns of trading the sign of a {-1, 0, +1} prediction.

    position_t = sign(prediction_t)
    r_t = position_t * forward_return_t - cost_t * |position_t - position_{t-1}|

    Args:
        predictions: Directional class predictions (-1 short, 0 flat, +1 long).
            NaN predictions are treated as flat.
        forward_returns: Return from bar t to bar t+1, aligned with
            ``predictions`` (NaN -> 0, e.g. last bar of a series).
        cost_per_turnover: Cost in return units per unit of position change
            (one side of a round trip), scalar or per-bar array.
        groups: Optional per-bar series id (e.g. symbol). Positions are not
            carried across group boundaries; each group starts flat.

    Returns:
        Array of per-bar strategy returns (finite).
    """
    preds = np.asarray(predictions, dtype=np.float64)
    fwd = np.nan_to_num(np.asarray(forward_returns, dtype=np.float64), nan=0.0)
    if preds.shape != fwd.shape:
        raise ValueError(f"predictions {preds.shape} and forward_returns {fwd.shape} differ")

    position = np.sign(np.nan_to_num(preds, nan=0.0))
    previous = np.empty_like(position)
    previous[0] = 0.0
    previous[1:] = position[:-1]
    if groups is not None:
        g = np.asarray(groups)
        starts = np.empty(len(g), dtype=bool)
        starts[0] = True
        starts[1:] = g[1:] != g[:-1]
        previous[starts] = 0.0

    turnover = np.abs(position - previous)
    cost = np.nan_to_num(np.asarray(cost_per_turnover, dtype=np.float64), nan=0.0)
    return position * fwd - cost * turnover


def pbo_gate(
    pbo_result: PBOResult,
    strict: bool = False,
) -> tuple[bool, str]:
    """
    Gate function for deployment decisions based on PBO.

    Args:
        pbo_result: PBOResult from compute_pbo
        strict: If True, compare against block_threshold; otherwise warn_threshold.

    Returns:
        Tuple of (should_proceed, reason)
    """
    threshold = pbo_result.config.block_threshold if strict else pbo_result.config.warn_threshold

    if pbo_result.pbo > threshold:
        return False, (
            f"PBO ({pbo_result.pbo:.3f}) exceeds threshold ({threshold:.2f}). "
            f"Risk level: {pbo_result.get_risk_level()}"
        )

    return True, (
        f"PBO ({pbo_result.pbo:.3f}) within threshold ({threshold:.2f}). "
        f"Risk level: {pbo_result.get_risk_level()}"
    )


def analyze_overfitting_risk(
    returns_matrix: np.ndarray,
    strategy_names: list[str] | None = None,
    config: PBOConfig | None = None,
) -> dict[str, Any]:
    """
    PBO plus per-strategy full-sample statistics.

    Args:
        returns_matrix: (T, N) per-period returns matrix
        strategy_names: Optional names for the N strategies
        config: PBOConfig

    Returns:
        Dict with analysis results
    """
    matrix = np.asarray(returns_matrix, dtype=np.float64)
    n_rows, n_strategies = matrix.shape

    if strategy_names is None:
        strategy_names = [f"strategy_{i}" for i in range(n_strategies)]
    if len(strategy_names) != n_strategies:
        raise ValueError(
            f"strategy_names length ({len(strategy_names)}) must match N ({n_strategies})"
        )

    pbo_result = compute_pbo(matrix, config)

    sums = matrix.sum(axis=0)
    sumsq = (matrix**2).sum(axis=0)
    sharpes = _sharpe_from_moments(sums, sumsq, np.full_like(sums, float(n_rows)))

    strategy_analysis = [
        {
            "name": name,
            "mean_return": float(matrix[:, i].mean()),
            "std_return": float(matrix[:, i].std(ddof=1)),
            "sharpe_per_period": float(sharpes[i]),
            "is_best_full_sample": i == pbo_result.best_is_strategy_idx,
        }
        for i, name in enumerate(strategy_names)
    ]

    return {
        "pbo": pbo_result.pbo,
        "is_overfit": pbo_result.is_overfit,
        "should_block": pbo_result.should_block,
        "risk_level": pbo_result.get_risk_level(),
        "performance_degradation": pbo_result.performance_degradation,
        "prob_oos_loss": pbo_result.prob_oos_loss,
        "rank_correlation": pbo_result.rank_correlation,
        "n_strategies": n_strategies,
        "n_observations": n_rows,
        "strategy_analysis": strategy_analysis,
        "best_full_sample_strategy": strategy_names[pbo_result.best_is_strategy_idx],
    }


__all__ = [
    "PBOConfig",
    "PBOResult",
    "compute_pbo",
    "directional_strategy_returns",
    "pbo_gate",
    "analyze_overfitting_risk",
]
