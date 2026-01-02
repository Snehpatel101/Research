"""
Optuna TPE-based optimization for barrier parameters.

Replaces DEAP genetic algorithm with Optuna's Tree-structured Parzen Estimator (TPE),
which is 27% more sample-efficient for hyperparameter optimization.

Public API:
    - run_optuna_optimization: Run TPE optimization for a single horizon
    - OptunaConvergenceCallback: Track optimization progress for plotting
"""

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler

from src.phase1.stages.labeling import triple_barrier_numba

from .fitness import calculate_fitness

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Search space bounds (same as GA)
K_MIN, K_MAX = 0.8, 2.5
MAX_BARS_MIN, MAX_BARS_MAX = 2.0, 3.0


@dataclass
class ConvergenceRecord:
    """Track convergence across trials for compatibility with plotting."""

    trials: list[dict[str, Any]] = field(default_factory=list)

    def add_trial(
        self,
        trial_number: int,
        value: float,
        best_value: float,
        params: dict[str, float],
    ) -> None:
        """Add a trial record."""
        self.trials.append(
            {
                "trial": trial_number,
                "value": value,
                "best_value": best_value,
                "params": params,
            }
        )

    def to_convergence_list(self) -> list[dict[str, Any]]:
        """
        Convert to the format expected by plotting.plot_convergence.

        Returns list of dicts with: gen, avg, max, min, std
        """
        if not self.trials:
            return []

        # Group trials into generations (batches of 5 for smoothing)
        batch_size = max(1, len(self.trials) // 30)  # ~30 generations
        convergence = []

        for gen_idx in range(0, len(self.trials), batch_size):
            batch = self.trials[gen_idx : gen_idx + batch_size]
            values = [t["value"] for t in batch if t["value"] > -500]  # Filter failures

            if values:
                convergence.append(
                    {
                        "gen": gen_idx // batch_size,
                        "avg": float(np.mean(values)),
                        "max": float(np.max(values)),
                        "min": float(np.min(values)),
                        "std": float(np.std(values)) if len(values) > 1 else 0.0,
                    }
                )

        return convergence


class OptunaConvergenceCallback:
    """Callback to track convergence during Optuna optimization."""

    def __init__(self) -> None:
        self.record = ConvergenceRecord()
        self.best_value: float = float("-inf")

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Called after each trial completes."""
        if trial.value is not None:
            # Track best (remember we're maximizing, so use study direction)
            if study.direction == optuna.study.StudyDirection.MAXIMIZE:
                if trial.value > self.best_value:
                    self.best_value = trial.value
            else:
                # Minimizing (we negate fitness)
                if trial.value < self.best_value:
                    self.best_value = trial.value

            self.record.add_trial(
                trial_number=trial.number,
                value=(
                    -trial.value
                    if study.direction == optuna.study.StudyDirection.MINIMIZE
                    else trial.value
                ),
                best_value=(
                    -self.best_value
                    if study.direction == optuna.study.StudyDirection.MINIMIZE
                    else self.best_value
                ),
                params=trial.params.copy(),
            )


def get_seeded_trials(symbol: str) -> list[dict[str, float]]:
    """
    Get symbol-specific seeded starting points for Optuna.

    These are the same starting points used by the GA.

    Parameters:
    -----------
    symbol : 'MES' or 'MGC'

    Returns:
    --------
    List of parameter dicts to enqueue as initial trials
    """
    if symbol == "MGC":
        # MGC: Symmetric barriers for mean-reverting gold
        return [
            {"k_up": 1.2, "k_down": 1.2, "max_bars_mult": 2.5},
            {"k_up": 1.5, "k_down": 1.5, "max_bars_mult": 2.5},
            {"k_up": 1.0, "k_down": 1.0, "max_bars_mult": 2.0},
            {"k_up": 1.3, "k_down": 1.3, "max_bars_mult": 2.8},
            {"k_up": 1.8, "k_down": 1.8, "max_bars_mult": 2.2},
        ]
    else:
        # MES: Asymmetric barriers for equity drift (k_up > k_down)
        return [
            {"k_up": 1.5, "k_down": 1.0, "max_bars_mult": 2.5},
            {"k_up": 1.8, "k_down": 1.2, "max_bars_mult": 2.5},
            {"k_up": 1.3, "k_down": 0.9, "max_bars_mult": 2.0},
            {"k_up": 2.0, "k_down": 1.4, "max_bars_mult": 2.8},
            {"k_up": 1.6, "k_down": 1.1, "max_bars_mult": 2.2},
        ]


def create_objective(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_prices: np.ndarray,
    atr: np.ndarray,
    horizon: int,
    symbol: str,
    regime: str = "low_vol",
    include_slippage: bool = True,
) -> Callable[[optuna.Trial], float]:
    """
    Create an Optuna objective function for barrier optimization.

    Parameters:
    -----------
    close, high, low, open_prices, atr : Price/indicator arrays
    horizon : Horizon value
    symbol : Symbol name for symbol-specific constraints
    regime : Volatility regime (default: 'low_vol')
    include_slippage : Include slippage in cost calculation (default: True)

    Returns:
    --------
    objective : Callable that takes an Optuna trial and returns fitness
    """

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function (minimizes negative fitness = maximizes fitness)."""
        # Sample parameters
        k_up = trial.suggest_float("k_up", K_MIN, K_MAX)
        k_down = trial.suggest_float("k_down", K_MIN, K_MAX)
        max_bars_mult = trial.suggest_float("max_bars_mult", MAX_BARS_MIN, MAX_BARS_MAX)

        # Apply symbol-specific asymmetry bonus (same logic as GA)
        avg_k = (k_up + k_down) / 2.0

        if symbol == "MGC":
            # MGC: Strict symmetry - penalize asymmetry > 10%
            if abs(k_up - k_down) > avg_k * 0.10:
                asymmetry_bonus = -abs(k_up - k_down) * 5.0
            else:
                asymmetry_bonus = 0.5  # Small reward for good symmetry
        else:
            # MES: REWARD k_up > k_down to counteract equity drift
            if k_up > k_down:
                asymmetry_ratio = k_up / k_down if k_down > 0 else 1.0
                if 1.2 <= asymmetry_ratio <= 1.8:
                    asymmetry_bonus = (k_up - k_down) * 2.0
                elif asymmetry_ratio > 1.8:
                    asymmetry_bonus = (k_up - k_down) * 0.5
                else:
                    asymmetry_bonus = (k_up - k_down) * 1.0
            else:
                # WRONG direction: k_down > k_up amplifies long bias
                asymmetry_bonus = -(k_down - k_up) * 3.0

        # Decode max_bars
        max_bars = int(horizon * max_bars_mult)
        max_bars = max(horizon * 2, min(max_bars, horizon * 3))

        # Run triple-barrier labeling
        try:
            labels, bars_to_hit, mae, mfe, _ = triple_barrier_numba(
                close, high, low, open_prices, atr, k_up, k_down, max_bars
            )
        except Exception as e:
            logger.warning(f"Triple barrier failed: {e}")
            return float("inf")  # Return inf for failed trials (we minimize)

        # Calculate fitness
        atr_mean = np.mean(atr[atr > 0]) if np.any(atr > 0) else 1.0

        fitness = calculate_fitness(
            labels,
            bars_to_hit,
            mae,
            mfe,
            horizon,
            atr_mean=atr_mean,
            symbol=symbol,
            k_up=k_up,
            k_down=k_down,
            regime=regime,
            include_slippage=include_slippage,
        )
        fitness += asymmetry_bonus

        # Report intermediate value for pruning
        trial.report(fitness, step=0)

        # Return negative fitness (Optuna minimizes by default)
        return -fitness

    return objective


def run_optuna_optimization(
    df: pd.DataFrame,
    horizon: int,
    symbol: str = "MES",
    n_trials: int = 100,
    subset_fraction: float = 0.3,
    atr_column: str = "atr_14",
    seed: int = 42,
    show_progress: bool = True,
    n_startup_trials: int = 10,
    regime: str = "low_vol",
    include_slippage: bool = True,
) -> tuple[dict[str, Any], ConvergenceRecord]:
    """
    Run Optuna TPE optimization to find optimal labeling parameters.

    This replaces the DEAP genetic algorithm with Optuna's Tree-structured
    Parzen Estimator (TPE), which is more sample-efficient.

    Parameters:
    -----------
    df : DataFrame with OHLCV and ATR
    horizon : Horizon to optimize for
    symbol : 'MES' or 'MGC' for symbol-specific optimization
    n_trials : Number of optimization trials (default: 100)
    subset_fraction : Fraction of data to use for speed (default: 0.3)
    atr_column : ATR column name (default: 'atr_14')
    seed : Random seed for reproducibility (default: 42)
    show_progress : Show progress bar (default: True)
    n_startup_trials : Random trials before TPE kicks in (default: 10)
    regime : Volatility regime (default: 'low_vol')
    include_slippage : Include slippage in cost calculation (default: True)

    Returns:
    --------
    results : dict with best parameters and statistics
    convergence_record : ConvergenceRecord for plotting
    """
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)

    logger.info(f"\nOptimizing parameters for {symbol} horizon {horizon} (Optuna TPE)")
    logger.info(f"  Trials: {n_trials}, Startup: {n_startup_trials}")
    logger.info(
        f"  Symbol-specific mode: {'SYMMETRIC (gold)' if symbol == 'MGC' else 'ASYMMETRIC (equity)'}"
    )
    logger.info(f"  Random seed: {seed}")

    # Get contiguous subset for temporal integrity
    from .operators import get_contiguous_subset

    df_subset = get_contiguous_subset(df, subset_fraction, seed=seed)
    logger.info(
        f"  Using {len(df_subset):,} contiguous samples ({subset_fraction*100:.0f}% of data)"
    )

    # Extract arrays
    close = df_subset["close"].values
    high = df_subset["high"].values
    low = df_subset["low"].values
    open_prices = df_subset["open"].values
    atr = df_subset[atr_column].values

    # Create objective function
    objective = create_objective(
        close, high, low, open_prices, atr, horizon, symbol, regime, include_slippage
    )

    # Create Optuna study with TPE sampler
    sampler = TPESampler(
        seed=seed,
        n_startup_trials=n_startup_trials,
        multivariate=True,  # Consider parameter correlations
    )

    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="minimize",  # Minimize negative fitness = maximize fitness
        sampler=sampler,
        study_name=f"{symbol}_h{horizon}",
    )

    # Enqueue seeded trials (same starting points as GA)
    for seed_params in get_seeded_trials(symbol):
        study.enqueue_trial(seed_params)

    # Create convergence callback
    callback = OptunaConvergenceCallback()

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[callback],
        show_progress_bar=show_progress,
        n_jobs=1,  # Single-threaded for reproducibility
    )

    # Get best parameters
    best_trial = study.best_trial
    best_k_up = best_trial.params["k_up"]
    best_k_down = best_trial.params["k_down"]
    best_max_bars_mult = best_trial.params["max_bars_mult"]
    best_fitness = -best_trial.value  # Negate back to positive fitness

    logger.info("\n  OPTIMIZATION COMPLETE (Optuna TPE)")
    logger.info(f"  Best fitness: {best_fitness:.4f}")
    logger.info(
        f"  Best params: k_up={best_k_up:.3f}, k_down={best_k_down:.3f}, "
        f"max_bars={int(horizon * best_max_bars_mult)}"
    )
    logger.info(f"  Total trials: {len(study.trials)}")

    # Validate on full data
    logger.info("  Validating on full dataset...")
    full_close = df["close"].values
    full_high = df["high"].values
    full_low = df["low"].values
    full_open = df["open"].values
    full_atr = df[atr_column].values

    max_bars = int(horizon * best_max_bars_mult)

    val_labels, val_bars, val_mae, val_mfe, _ = triple_barrier_numba(
        full_close, full_high, full_low, full_open, full_atr, best_k_up, best_k_down, max_bars
    )

    n_total = len(val_labels)
    n_long = (val_labels == 1).sum()
    n_short = (val_labels == -1).sum()
    n_neutral = (val_labels == 0).sum()

    logger.info("  Full data label distribution:")
    logger.info(f"    Long:    {n_long:,} ({100*n_long/n_total:.1f}%)")
    logger.info(f"    Short:   {n_short:,} ({100*n_short/n_total:.1f}%)")
    logger.info(f"    Neutral: {n_neutral:,} ({100*n_neutral/n_total:.1f}%)")

    # Recalculate fitness on FULL data (not subset) for accurate validation
    # The subset fitness may be misleading if the subset has different characteristics
    full_atr_mean = np.mean(full_atr[full_atr > 0]) if np.any(full_atr > 0) else 1.0
    validation_fitness = calculate_fitness(
        val_labels,
        val_bars,
        val_mae,
        val_mfe,
        horizon,
        atr_mean=full_atr_mean,
        symbol=symbol,
        k_up=best_k_up,
        k_down=best_k_down,
        regime=regime,
        include_slippage=include_slippage,
    )

    # Calculate asymmetry_bonus (must match logic in fitness.py evaluate_individual)
    avg_k = (best_k_up + best_k_down) / 2.0
    if symbol == "MGC":
        # MGC: Strict symmetry - penalize asymmetry > 10%
        if abs(best_k_up - best_k_down) > avg_k * 0.10:
            asymmetry_bonus = max(-5.0, -abs(best_k_up - best_k_down) * 5.0)
        else:
            asymmetry_bonus = 0.5
    else:
        # MES: REWARD k_up > k_down to counteract equity drift
        if best_k_up > best_k_down:
            asymmetry_ratio = best_k_up / best_k_down if best_k_down > 0 else 1.0
            if 1.2 <= asymmetry_ratio <= 1.8:
                asymmetry_bonus = min(3.0, (best_k_up - best_k_down) * 2.0)
            elif asymmetry_ratio > 1.8:
                asymmetry_bonus = min(1.5, (best_k_up - best_k_down) * 0.5)
            else:
                asymmetry_bonus = min(2.0, (best_k_up - best_k_down) * 1.0)
        else:
            asymmetry_bonus = max(-5.0, -(best_k_down - best_k_up) * 3.0)

    # Add asymmetry bonus to validation fitness for consistency with optimization objective
    validation_fitness += asymmetry_bonus

    # Log both fitnesses for debugging
    logger.info(f"  Subset fitness: {best_fitness:.4f}")
    logger.info(
        f"  Full-data fitness: {validation_fitness:.4f} (with asymmetry_bonus={asymmetry_bonus:.2f})"
    )

    # Use validation fitness (full data) for stored results
    # This is more representative of actual model performance
    final_fitness = validation_fitness

    # Prepare results (same format as GA for compatibility)
    results = {
        "horizon": horizon,
        "best_k_up": float(best_k_up),
        "best_k_down": float(best_k_down),
        "best_max_bars": int(horizon * best_max_bars_mult),
        "best_fitness": float(final_fitness),
        "n_trials": n_trials,
        "optimizer": "optuna_tpe",
        "validation": {
            "n_total": int(n_total),
            "n_long": int(n_long),
            "n_short": int(n_short),
            "n_neutral": int(n_neutral),
            "pct_long": float(100 * n_long / n_total),
            "pct_short": float(100 * n_short / n_total),
            "pct_neutral": float(100 * n_neutral / n_total),
            "signal_rate": float((n_long + n_short) / n_total),
        },
        "convergence": callback.record.to_convergence_list(),
    }

    return results, callback.record


def run_optuna_optimization_safe(
    df: pd.DataFrame,
    horizon: int,
    symbol: str = "MES",
    train_ratio: float = 0.70,
    n_trials: int = 100,
    subset_fraction: float = 0.3,
    atr_column: str = "atr_14",
    seed: int = 42,
    show_progress: bool = True,
    n_startup_trials: int = 10,
    regime: str = "low_vol",
    include_slippage: bool = True,
) -> tuple[dict[str, Any], ConvergenceRecord]:
    """
    Run Optuna TPE optimization using ONLY training data portion to prevent test data leakage.

    This is the SAFE version of run_optuna_optimization that ensures no data from the
    future test set influences the optimized barrier parameters. This prevents the
    critical data leakage issue where parameters are tuned on data that will later
    be used for testing.

    IMPORTANT: This function enforces temporal integrity by only using the first
    `train_ratio` portion of the data for optimization. The remaining data is
    reserved for validation/test and is NEVER seen during parameter optimization.

    Parameters:
    -----------
    df : DataFrame with OHLCV and ATR (time-ordered)
    horizon : Horizon to optimize for
    symbol : 'MES' or 'MGC' for symbol-specific optimization
    train_ratio : Fraction of data to use for optimization (default: 0.70)
                  This MUST match the train_ratio used in the splits stage.
    n_trials : Number of optimization trials (default: 100)
    subset_fraction : Fraction of TRAINING data to use for speed (default: 0.3)
    atr_column : ATR column name (default: 'atr_14')
    seed : Random seed for reproducibility (default: 42)
    show_progress : Show progress bar (default: True)
    n_startup_trials : Random trials before TPE kicks in (default: 10)
    regime : Volatility regime (default: 'low_vol')
    include_slippage : Include slippage in cost calculation (default: True)

    Returns:
    --------
    results : dict with best parameters and statistics
    convergence_record : ConvergenceRecord for plotting

    Example:
    --------
    >>> # Use only first 70% of data (same as train split)
    >>> results, record = run_optuna_optimization_safe(
    ...     df=full_df,
    ...     horizon=20,
    ...     train_ratio=0.70,  # Must match splits stage
    ...     symbol='MES'
    ... )
    """
    n_total = len(df)
    train_end = int(n_total * train_ratio)

    # Enforce minimum training size
    if train_end < 500:
        raise ValueError(
            f"Training portion too small: {train_end} samples. "
            f"Need at least 500 samples for meaningful optimization."
        )

    # Extract ONLY training portion (first train_ratio% of data)
    df_train = df.iloc[:train_end].copy()

    logger.info("")
    logger.info("=" * 60)
    logger.info("SAFE MODE: Test Data Leakage Prevention ENABLED")
    logger.info("=" * 60)
    logger.info(f"  Total dataset size: {n_total:,} samples")
    logger.info(f"  Training portion: {train_end:,} samples ({train_ratio*100:.0f}%)")
    logger.info(
        f"  Reserved for test: {n_total - train_end:,} samples ({(1-train_ratio)*100:.0f}%)"
    )
    logger.info("")
    logger.info("  Optimization will ONLY use training portion.")
    logger.info(
        f"  Test data (last {(1 - train_ratio) * 100:.0f}%) is NEVER seen during parameter tuning."
    )
    logger.info("=" * 60)

    # Run optimization on training data only
    results, convergence = run_optuna_optimization(
        df=df_train,  # ONLY training data
        horizon=horizon,
        symbol=symbol,
        n_trials=n_trials,
        subset_fraction=subset_fraction,
        atr_column=atr_column,
        seed=seed,
        show_progress=show_progress,
        n_startup_trials=n_startup_trials,
        regime=regime,
        include_slippage=include_slippage,
    )

    # Mark results as safe-mode
    results["safe_mode"] = True
    results["train_ratio_used"] = train_ratio
    results["train_samples"] = train_end
    results["total_samples"] = n_total

    # Update validation section to clarify it's train-only
    results["validation"]["note"] = "Statistics computed on training portion only (safe mode)"

    logger.info("")
    logger.info("SAFE MODE optimization complete.")
    logger.info(f"  Parameters were tuned on {train_end:,} training samples only.")
    logger.info(f"  Test data ({n_total - train_end:,} samples) was NOT used.")

    return results, convergence
