"""
Probability of Backtest Overfitting (PBO) Computation.

PBO quantifies the probability that a backtest-optimal strategy will
underperform out-of-sample. High PBO indicates the strategy selection
process is likely overfit to historical data.

Reference: Bailey et al. (2014) "The Probability of Backtest Overfitting"
           https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253

Key concepts:
- CSCV: Combinatorially Symmetric Cross-Validation for unbiased estimation
- Logit transformation: Maps rank ratios to unbounded scale
- PBO threshold: PBO > 0.5 suggests overfitting; PBO > 0.8 is severe

Example:
    >>> from src.cross_validation.pbo import compute_pbo, PBOResult
    >>> # performance_matrix: (n_strategies, n_paths) matrix of returns/metrics
    >>> result = compute_pbo(performance_matrix, n_partitions=16)
    >>> print(f"PBO: {result.pbo:.3f}, Overfit: {result.is_overfit}")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PBOConfig:
    """
    Configuration for PBO computation.

    Attributes:
        n_partitions: Number of CSCV partitions (S in paper, typically 8-16)
        warn_threshold: PBO threshold for warning (default 0.5)
        block_threshold: PBO threshold for blocking deployment (default 0.8)
        min_paths: Minimum paths required for reliable PBO estimate
        use_sharpe: If True, use Sharpe ratio; if False, use raw returns
    """
    n_partitions: int = 16
    warn_threshold: float = 0.5
    block_threshold: float = 0.8
    min_paths: int = 6
    use_sharpe: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.n_partitions < 2:
            raise ValueError(f"n_partitions must be >= 2, got {self.n_partitions}")
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
    Result from PBO computation.

    Attributes:
        pbo: Probability of Backtest Overfitting (0-1)
        logit_distribution: Distribution of logit values across CSCV combinations
        performance_degradation: Mean OOS / mean IS ratio
        rank_correlation: Spearman correlation between IS and OOS ranks
        is_overfit: True if PBO exceeds warn_threshold
        should_block: True if PBO exceeds block_threshold
        n_paths_evaluated: Number of CSCV paths evaluated
        best_is_strategy_idx: Index of best in-sample strategy
        best_is_oos_rank: OOS rank of best IS strategy (lower = better)
    """
    pbo: float
    logit_distribution: np.ndarray
    performance_degradation: float
    rank_correlation: float
    is_overfit: bool
    should_block: bool
    n_paths_evaluated: int
    best_is_strategy_idx: int
    best_is_oos_rank: float
    config: PBOConfig

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pbo": self.pbo,
            "performance_degradation": self.performance_degradation,
            "rank_correlation": self.rank_correlation,
            "is_overfit": self.is_overfit,
            "should_block": self.should_block,
            "n_paths_evaluated": self.n_paths_evaluated,
            "best_is_strategy_idx": self.best_is_strategy_idx,
            "best_is_oos_rank": self.best_is_oos_rank,
            "warn_threshold": self.config.warn_threshold,
            "block_threshold": self.config.block_threshold,
        }

    def get_risk_level(self) -> str:
        """Get human-readable risk level."""
        if self.should_block:
            return "CRITICAL: High overfitting risk, block deployment"
        elif self.is_overfit:
            return "WARNING: Moderate overfitting risk, review carefully"
        elif self.pbo > 0.3:
            return "CAUTION: Some overfitting signal detected"
        else:
            return "OK: Low overfitting risk"


# =============================================================================
# PBO COMPUTATION
# =============================================================================

def _compute_sharpe(returns: np.ndarray, annualization: float = 252.0) -> float:
    """Compute Sharpe ratio from returns array."""
    if len(returns) < 2:
        return 0.0
    mean_ret = np.nanmean(returns)
    std_ret = np.nanstd(returns, ddof=1)
    if std_ret < 1e-10:
        return 0.0
    return float(mean_ret / std_ret * np.sqrt(annualization))


def _compute_logit(w_bar: float) -> float:
    """
    Compute logit transformation of rank ratio.

    Args:
        w_bar: Relative rank of best IS strategy in OOS (0-1)
               w_bar < 0.5 means best IS strategy ranks below median OOS

    Returns:
        Logit value: negative if overfit, positive if robust
    """
    # Clip to avoid log(0) or log(inf)
    w_bar = np.clip(w_bar, 1e-10, 1 - 1e-10)
    return float(np.log(w_bar / (1 - w_bar)))


def compute_pbo(
    performance_matrix: np.ndarray,
    config: PBOConfig | None = None,
) -> PBOResult:
    """
    Compute Probability of Backtest Overfitting.

    Uses Combinatorially Symmetric Cross-Validation (CSCV) to estimate
    the probability that the best in-sample strategy underperforms
    out-of-sample.

    Args:
        performance_matrix: Matrix of shape (n_strategies, n_paths) where each
                           entry is the performance metric (returns/Sharpe) for
                           strategy i in path j
        config: PBOConfig (uses defaults if None)

    Returns:
        PBOResult with PBO estimate and related metrics

    Raises:
        ValueError: If matrix is too small for reliable estimation

    Algorithm:
        1. Partition paths into S groups
        2. For each half-split of groups (CSCV):
           a. Compute IS performance (mean over train paths)
           b. Compute OOS performance (mean over test paths)
           c. Find best IS strategy by rank
           d. Compute relative OOS rank of best IS strategy
           e. Apply logit transformation
        3. PBO = fraction of logit values < 0
    """
    if config is None:
        config = PBOConfig()

    n_strategies, n_paths = performance_matrix.shape

    if n_strategies < 2:
        raise ValueError(f"Need at least 2 strategies, got {n_strategies}")

    if n_paths < config.min_paths:
        raise ValueError(
            f"Need at least {config.min_paths} paths for reliable PBO, got {n_paths}"
        )

    # Partition paths into groups
    n_partitions = min(config.n_partitions, n_paths)
    paths_per_partition = n_paths // n_partitions

    # Create partition indices
    partition_indices = []
    for i in range(n_partitions):
        start = i * paths_per_partition
        if i == n_partitions - 1:
            end = n_paths
        else:
            end = (i + 1) * paths_per_partition
        partition_indices.append(list(range(start, end)))

    # Generate CSCV combinations (half-splits)
    n_test_partitions = n_partitions // 2
    all_combos = list(combinations(range(n_partitions), n_test_partitions))

    # Limit combinations if too many
    max_combos = 100
    if len(all_combos) > max_combos:
        step = len(all_combos) / max_combos
        selected = [int(i * step) for i in range(max_combos)]
        all_combos = [all_combos[i] for i in selected]

    logit_values = []
    is_oos_pairs = []  # For correlation computation

    for test_partition_ids in all_combos:
        train_partition_ids = tuple(
            p for p in range(n_partitions) if p not in test_partition_ids
        )

        # Get path indices for train (IS) and test (OOS)
        is_paths = []
        for p in train_partition_ids:
            is_paths.extend(partition_indices[p])

        oos_paths = []
        for p in test_partition_ids:
            oos_paths.extend(partition_indices[p])

        if not is_paths or not oos_paths:
            continue

        # Compute IS and OOS performance for each strategy
        is_perf = np.nanmean(performance_matrix[:, is_paths], axis=1)
        oos_perf = np.nanmean(performance_matrix[:, oos_paths], axis=1)

        # Handle NaN values
        valid_mask = ~(np.isnan(is_perf) | np.isnan(oos_perf))
        if valid_mask.sum() < 2:
            continue

        is_perf_valid = is_perf[valid_mask]
        oos_perf_valid = oos_perf[valid_mask]
        n_valid = len(is_perf_valid)

        # Rank strategies by IS performance (higher rank = better)
        is_ranks = stats.rankdata(is_perf_valid)
        oos_ranks = stats.rankdata(oos_perf_valid)

        # Find best IS strategy (highest IS rank)
        best_is_idx = np.argmax(is_ranks)
        best_is_rank = is_ranks[best_is_idx]

        # Get OOS rank of best IS strategy
        best_is_oos_rank = oos_ranks[best_is_idx]

        # Compute relative OOS rank (w_bar): fraction of strategies it beats OOS
        # w_bar = (oos_rank - 1) / (n - 1) for ranks in [1, n]
        # If best IS strategy has low OOS rank, w_bar < 0.5 → overfit signal
        if n_valid > 1:
            w_bar = (best_is_oos_rank - 1) / (n_valid - 1)
        else:
            w_bar = 0.5

        logit_val = _compute_logit(w_bar)
        logit_values.append(logit_val)

        # Store for correlation
        is_oos_pairs.append((is_perf_valid, oos_perf_valid))

    if not logit_values:
        raise ValueError("No valid CSCV combinations produced logit values")

    logit_array = np.array(logit_values)

    # PBO = probability that logit < 0 (i.e., best IS strategy underperforms OOS)
    pbo = float(np.mean(logit_array < 0))

    # Compute performance degradation
    all_is = np.concatenate([p[0] for p in is_oos_pairs])
    all_oos = np.concatenate([p[1] for p in is_oos_pairs])
    mean_is = np.nanmean(all_is)
    mean_oos = np.nanmean(all_oos)
    if abs(mean_is) > 1e-10:
        perf_degradation = float(mean_oos / mean_is)
    else:
        perf_degradation = 1.0

    # Compute rank correlation between IS and OOS
    # Use the last CSCV split for this
    if is_oos_pairs:
        last_is, last_oos = is_oos_pairs[-1]
        if len(last_is) >= 3:
            try:
                rank_corr, _ = stats.spearmanr(last_is, last_oos)
                rank_corr = float(rank_corr) if not np.isnan(rank_corr) else 0.0
            except Exception:
                rank_corr = 0.0
        else:
            rank_corr = 0.0
    else:
        rank_corr = 0.0

    # Find overall best IS strategy
    overall_is = np.nanmean(performance_matrix, axis=1)
    best_is_strategy_idx = int(np.argmax(overall_is))

    # Compute its average OOS rank across all combos
    oos_ranks_best = []
    for is_perf, oos_perf in is_oos_pairs:
        oos_rank = stats.rankdata(oos_perf)[np.argmax(is_perf)]
        oos_ranks_best.append(oos_rank / len(oos_perf))
    avg_oos_rank = float(np.mean(oos_ranks_best)) if oos_ranks_best else 0.5

    return PBOResult(
        pbo=pbo,
        logit_distribution=logit_array,
        performance_degradation=perf_degradation,
        rank_correlation=rank_corr,
        is_overfit=pbo > config.warn_threshold,
        should_block=pbo > config.block_threshold,
        n_paths_evaluated=len(all_combos),
        best_is_strategy_idx=best_is_strategy_idx,
        best_is_oos_rank=avg_oos_rank,
        config=config,
    )


def compute_pbo_from_returns(
    returns_matrix: np.ndarray,
    config: PBOConfig | None = None,
    use_sharpe: bool = True,
) -> PBOResult:
    """
    Compute PBO from a matrix of strategy returns.

    Convenience wrapper that converts returns to Sharpe ratios before
    computing PBO.

    Args:
        returns_matrix: Matrix of shape (n_strategies, n_periods) with returns
        config: PBOConfig
        use_sharpe: If True, convert to Sharpe; if False, use cumulative returns

    Returns:
        PBOResult
    """
    n_strategies, n_periods = returns_matrix.shape

    if use_sharpe:
        # Compute rolling Sharpe ratios as performance metric
        window = max(20, n_periods // 10)
        n_paths = n_periods // window

        perf_matrix = np.zeros((n_strategies, n_paths))
        for path in range(n_paths):
            start = path * window
            end = start + window
            for strat in range(n_strategies):
                perf_matrix[strat, path] = _compute_sharpe(
                    returns_matrix[strat, start:end],
                    annualization=252.0
                )
    else:
        # Use cumulative returns as performance metric
        window = max(20, n_periods // 10)
        n_paths = n_periods // window

        perf_matrix = np.zeros((n_strategies, n_paths))
        for path in range(n_paths):
            start = path * window
            end = start + window
            for strat in range(n_strategies):
                perf_matrix[strat, path] = float(
                    np.nansum(returns_matrix[strat, start:end])
                )

    return compute_pbo(perf_matrix, config)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def pbo_gate(
    pbo_result: PBOResult,
    strict: bool = False,
) -> tuple[bool, str]:
    """
    Gate function for deployment decisions based on PBO.

    Args:
        pbo_result: PBOResult from compute_pbo
        strict: If True, use block_threshold; if False, use warn_threshold

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
    performance_matrix: np.ndarray,
    strategy_names: list[str] | None = None,
    config: PBOConfig | None = None,
) -> dict[str, Any]:
    """
    Comprehensive overfitting risk analysis.

    Args:
        performance_matrix: (n_strategies, n_paths) performance matrix
        strategy_names: Optional names for each strategy
        config: PBOConfig

    Returns:
        Dict with analysis results
    """
    n_strategies, n_paths = performance_matrix.shape

    if strategy_names is None:
        strategy_names = [f"strategy_{i}" for i in range(n_strategies)]

    # Compute PBO
    pbo_result = compute_pbo(performance_matrix, config)

    # Per-strategy analysis
    strategy_analysis = []
    for i, name in enumerate(strategy_names):
        strat_perf = performance_matrix[i, :]
        strategy_analysis.append({
            "name": name,
            "mean_performance": float(np.nanmean(strat_perf)),
            "std_performance": float(np.nanstd(strat_perf)),
            "min_performance": float(np.nanmin(strat_perf)),
            "max_performance": float(np.nanmax(strat_perf)),
            "is_best_is": i == pbo_result.best_is_strategy_idx,
        })

    return {
        "pbo": pbo_result.pbo,
        "is_overfit": pbo_result.is_overfit,
        "should_block": pbo_result.should_block,
        "risk_level": pbo_result.get_risk_level(),
        "performance_degradation": pbo_result.performance_degradation,
        "rank_correlation": pbo_result.rank_correlation,
        "n_strategies": n_strategies,
        "n_paths": n_paths,
        "strategy_analysis": strategy_analysis,
        "best_is_strategy": strategy_names[pbo_result.best_is_strategy_idx],
    }


__all__ = [
    "PBOConfig",
    "PBOResult",
    "compute_pbo",
    "compute_pbo_from_returns",
    "pbo_gate",
    "analyze_overfitting_risk",
]
