"""
Label Optimization - Triple-Barrier Parameter Tuning with Optuna.

This module provides optimization of triple-barrier labeling parameters
to achieve target class distributions or maximize model performance.

Key Components:
    TripleBarrierConfig: Configuration for triple-barrier labeling
    LabelOptimizationResult: Result container for label optimization
    LabelOptimizer: Optuna-based optimizer for label parameters

Optimization Objectives:
    1. Class Balance: Minimize deviation from target distribution (e.g., 33/33/33)
    2. Model Performance: Maximize F1 score with candidate model
    3. Combined: Balance both objectives

Usage:
    from src.optimization.labels import LabelOptimizer, TripleBarrierConfig

    optimizer = LabelOptimizer(n_trials=100, objective="class_balance")
    result = optimizer.optimize(ohlcv_df, target_distribution={-1: 0.33, 0: 0.33, 1: 0.33})

    # Apply optimized config
    labeler = TripleBarrierLabeler(**result.best_config.to_dict())
    labels = labeler.compute_labels(ohlcv_df, horizon=20)

Reference: Lopez de Prado (2018) "Advances in Financial Machine Learning"
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler

from src.core import (
    DEFAULT_HORIZON,
    DEFAULT_LABEL_OPTIMIZATION_TRIALS,
    DEFAULT_OPTUNA_RANDOM_STATE,
)
from src.optimization.scoring import purged_train_val_split

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================


@dataclass
class TripleBarrierConfig:
    """
    Configuration for triple-barrier labeling.

    Attributes:
        k_up: Upper barrier multiplier (profit target = k_up * ATR)
        k_down: Lower barrier multiplier (stop loss = k_down * ATR)
        max_bars: Maximum holding period before timeout
        atr_period: Period for ATR calculation (default: 14)
        apply_transaction_costs: Whether to adjust for costs
        horizon: Prediction horizon in bars
    """

    k_up: float = 2.0
    k_down: float = 2.0
    max_bars: int = 20
    atr_period: int = 14
    apply_transaction_costs: bool = True
    horizon: int = DEFAULT_HORIZON

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "k_up": self.k_up,
            "k_down": self.k_down,
            "max_bars": self.max_bars,
            "atr_period": self.atr_period,
            "apply_transaction_costs": self.apply_transaction_costs,
            "horizon": self.horizon,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TripleBarrierConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def __repr__(self) -> str:
        return (
            f"TripleBarrierConfig(k_up={self.k_up:.3f}, k_down={self.k_down:.3f}, "
            f"max_bars={self.max_bars}, horizon={self.horizon})"
        )


# =============================================================================
# RESULT DATACLASS
# =============================================================================


@dataclass
class LabelOptimizationResult:
    """
    Result container for label optimization.

    Attributes:
        best_config: Optimal triple-barrier configuration
        best_score: Best optimization score achieved
        target_distribution: Target class distribution
        achieved_distribution: Actual distribution with best config
        distribution_error: L2 distance from target distribution
        n_trials: Number of optimization trials
        study: Optuna Study object
        optimization_time: Time taken in seconds
        trial_history: List of (config, score) for each trial
    """

    best_config: TripleBarrierConfig
    best_score: float
    target_distribution: dict[int, float]
    achieved_distribution: dict[int, float]
    distribution_error: float
    n_trials: int
    study: optuna.Study
    optimization_time: float = 0.0
    trial_history: list[tuple[TripleBarrierConfig, float]] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"LabelOptimizationResult(\n"
            f"  best_config={self.best_config},\n"
            f"  achieved_distribution={self.achieved_distribution},\n"
            f"  distribution_error={self.distribution_error:.4f},\n"
            f"  n_trials={self.n_trials}\n"
            f")"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "best_config": self.best_config.to_dict(),
            "best_score": self.best_score,
            "target_distribution": self.target_distribution,
            "achieved_distribution": self.achieved_distribution,
            "distribution_error": self.distribution_error,
            "n_trials": self.n_trials,
            "optimization_time": self.optimization_time,
        }


# =============================================================================
# LABEL OPTIMIZER
# =============================================================================


class LabelOptimizer:
    """
    Optuna-based optimizer for triple-barrier labeling parameters.

    Optimizes k_up, k_down, and max_bars to achieve target class distributions
    or maximize model performance.

    Args:
        n_trials: Number of optimization trials (default: 100)
        objective: Optimization objective ("class_balance", "model_f1", "combined")
        random_state: Random seed for reproducibility
        horizon: Prediction horizon in bars
        verbose: Verbosity level (0=silent, 1=progress, 2=debug)

    Search Space:
        k_up: [0.5, 5.0] - Upper barrier multiplier
        k_down: [0.5, 5.0] - Lower barrier multiplier
        max_bars: [5, 100] - Maximum holding period

    Example:
        >>> optimizer = LabelOptimizer(n_trials=100)
        >>> result = optimizer.optimize(
        ...     ohlcv_df,
        ...     target_distribution={-1: 0.33, 0: 0.34, 1: 0.33}
        ... )
        >>> print(f"Best config: {result.best_config}")
    """

    # Search space bounds
    K_UP_RANGE = (0.5, 5.0)
    K_DOWN_RANGE = (0.5, 5.0)
    MAX_BARS_RANGE = (5, 100)

    def __init__(
        self,
        n_trials: int = DEFAULT_LABEL_OPTIMIZATION_TRIALS,
        objective: str = "class_balance",
        random_state: int = DEFAULT_OPTUNA_RANDOM_STATE,
        horizon: int = DEFAULT_HORIZON,
        verbose: int = 1,
    ):
        self.n_trials = n_trials
        self.objective = objective
        self.random_state = random_state
        self.horizon = horizon
        self.verbose = verbose

        # Validate objective
        valid_objectives = ["class_balance", "model_f1", "combined"]
        if objective not in valid_objectives:
            raise ValueError(f"Invalid objective: {objective}. Must be one of {valid_objectives}")

        # Configure Optuna logging
        if verbose == 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        elif verbose == 1:
            optuna.logging.set_verbosity(optuna.logging.INFO)

    def _compute_labels(
        self,
        ohlcv_df: pd.DataFrame,
        config: TripleBarrierConfig,
    ) -> np.ndarray:
        """
        Compute labels using triple-barrier method.

        Args:
            ohlcv_df: OHLCV DataFrame with ATR column
            config: Triple-barrier configuration

        Returns:
            Array of labels (-1, 0, 1, or -99 for invalid)
        """
        try:
            from src.data.pipeline.stages.labeling import TripleBarrierLabeler

            labeler = TripleBarrierLabeler(
                k_up=config.k_up,
                k_down=config.k_down,
                max_bars=config.max_bars,
                apply_transaction_costs=config.apply_transaction_costs,
            )

            result = labeler.compute_labels(
                ohlcv_df,
                horizon=config.horizon,
            )
            return result.labels

        except ImportError:
            # Fallback to basic implementation if labeling module not available
            return self._compute_labels_basic(ohlcv_df, config)

    def _compute_labels_basic(
        self,
        ohlcv_df: pd.DataFrame,
        config: TripleBarrierConfig,
    ) -> np.ndarray:
        """
        Basic triple-barrier labeling (fallback implementation).

        Args:
            ohlcv_df: OHLCV DataFrame
            config: Triple-barrier configuration

        Returns:
            Array of labels
        """
        close = ohlcv_df["close"].values
        high = ohlcv_df["high"].values
        low = ohlcv_df["low"].values

        # Compute ATR
        atr_col = f"atr_{config.atr_period}"
        if atr_col in ohlcv_df.columns:
            atr = ohlcv_df[atr_col].values
        else:
            # Simple ATR calculation
            tr = np.maximum(
                high - low,
                np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))),
            )
            tr[0] = high[0] - low[0]
            atr = pd.Series(tr).rolling(config.atr_period).mean().values

        n = len(close)
        labels = np.zeros(n, dtype=np.int8)

        for i in range(n - config.max_bars):
            entry_price = close[i]
            entry_atr = atr[i]

            if np.isnan(entry_atr) or entry_atr <= 0:
                labels[i] = -99
                continue

            upper_barrier = entry_price + config.k_up * entry_atr
            lower_barrier = entry_price - config.k_down * entry_atr

            hit = False
            for j in range(1, config.max_bars + 1):
                idx = i + j
                if idx >= n:
                    break

                if high[idx] >= upper_barrier:
                    labels[i] = 1
                    hit = True
                    break
                elif low[idx] <= lower_barrier:
                    labels[i] = -1
                    hit = True
                    break

            if not hit:
                labels[i] = 0

        # Mark last max_bars as invalid
        labels[-config.max_bars :] = -99

        return labels

    def _compute_distribution(self, labels: np.ndarray) -> dict[int, float]:
        """
        Compute class distribution from labels.

        Args:
            labels: Array of labels

        Returns:
            Dict mapping class -> proportion
        """
        valid_mask = labels != -99
        valid_labels = labels[valid_mask]

        if len(valid_labels) == 0:
            return {-1: 0.0, 0: 0.0, 1: 0.0}

        counts = {-1: 0, 0: 0, 1: 0}
        for label in valid_labels:
            if label in counts:
                counts[label] += 1

        total = len(valid_labels)
        return {k: v / total for k, v in counts.items()}

    def _distribution_error(
        self,
        achieved: dict[int, float],
        target: dict[int, float],
    ) -> float:
        """
        Compute L2 distance between distributions.

        Args:
            achieved: Achieved class distribution
            target: Target class distribution

        Returns:
            L2 distance (lower is better)
        """
        error = 0.0
        for k in [-1, 0, 1]:
            error += (achieved.get(k, 0) - target.get(k, 0.333)) ** 2
        return float(np.sqrt(error))

    def optimize(
        self,
        ohlcv_df: pd.DataFrame,
        target_distribution: dict[int, float] | None = None,
        model_factory: Callable | None = None,
        feature_df: pd.DataFrame | None = None,
    ) -> LabelOptimizationResult:
        """
        Optimize triple-barrier parameters.

        Args:
            ohlcv_df: OHLCV DataFrame with required columns
            target_distribution: Target class distribution (default: balanced)
            model_factory: Optional model factory for model_f1 objective
            feature_df: Optional feature DataFrame for model_f1 objective

        Returns:
            LabelOptimizationResult with optimal configuration
        """
        start_time = time.time()

        # Default to balanced distribution
        if target_distribution is None:
            target_distribution = {-1: 0.333, 0: 0.334, 1: 0.333}

        # Normalize target distribution
        total = sum(target_distribution.values())
        target_distribution = {k: v / total for k, v in target_distribution.items()}

        if self.verbose >= 1:
            print(f"\n{'='*60}")
            print("Optimizing triple-barrier labels")
            print(f"  n_trials={self.n_trials}, objective={self.objective}")
            print(f"  target_distribution={target_distribution}")
            print(f"{'='*60}")

        # Trial history
        trial_history: list[tuple[TripleBarrierConfig, float]] = []

        def objective(trial: optuna.Trial) -> float:
            # Suggest parameters
            k_up = trial.suggest_float("k_up", *self.K_UP_RANGE)
            k_down = trial.suggest_float("k_down", *self.K_DOWN_RANGE)
            max_bars = trial.suggest_int("max_bars", *self.MAX_BARS_RANGE)

            config = TripleBarrierConfig(
                k_up=k_up,
                k_down=k_down,
                max_bars=max_bars,
                horizon=self.horizon,
            )

            # Compute labels
            labels = self._compute_labels(ohlcv_df, config)

            # Compute achieved distribution
            achieved = self._compute_distribution(labels)

            # Store in trial for later retrieval
            trial.set_user_attr("achieved_distribution", achieved)
            trial.set_user_attr("config", config.to_dict())

            # Compute objective
            if self.objective == "class_balance":
                # Minimize distribution error (convert to maximization)
                error = self._distribution_error(achieved, target_distribution)
                score = -error  # Negate for maximization

            elif self.objective == "model_f1":
                if model_factory is None or feature_df is None:
                    raise ValueError("model_f1 objective requires model_factory and feature_df")

                # Train model and compute F1
                from sklearn.metrics import f1_score

                valid_mask = labels != -99
                X = feature_df.values[valid_mask]
                y = labels[valid_mask]

                # Purged train/val split to prevent leakage
                train_end, val_start = purged_train_val_split(len(X))
                X_train, X_val = X[:train_end], X[val_start:]
                y_train, y_val = y[:train_end], y[val_start:]

                try:
                    model = model_factory({})
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                    score = f1_score(y_val, preds, average="weighted")
                except Exception as e:
                    logger.warning(f"Model training failed: {e}")
                    score = 0.0

            elif self.objective == "combined":
                # Combine class balance and model F1
                error = self._distribution_error(achieved, target_distribution)
                balance_score = 1.0 - error / np.sqrt(3)  # Normalize to [0, 1]

                if model_factory is not None and feature_df is not None:
                    from sklearn.metrics import f1_score

                    valid_mask = labels != -99
                    X = feature_df.values[valid_mask]
                    y = labels[valid_mask]

                    train_end, val_start = purged_train_val_split(len(X))
                    X_train, X_val = X[:train_end], X[val_start:]
                    y_train, y_val = y[:train_end], y[val_start:]

                    try:
                        model = model_factory({})
                        model.fit(X_train, y_train)
                        preds = model.predict(X_val)
                        f1 = f1_score(y_val, preds, average="weighted")
                    except Exception as e:
                        logger.warning(f"Model training/scoring failed in label optimization: {e}. Using default F1 0.5.")
                        f1 = 0.5

                    score = 0.5 * balance_score + 0.5 * f1
                else:
                    score = balance_score

            else:
                raise ValueError(f"Unknown objective: {self.objective}")

            trial_history.append((config, score))
            return score

        # Create study
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=(self.verbose >= 1),
        )

        optimization_time = time.time() - start_time

        # Get best trial
        best_trial = study.best_trial
        best_config_dict = best_trial.user_attrs.get("config", {})
        best_config = TripleBarrierConfig.from_dict(best_config_dict)
        achieved_distribution = best_trial.user_attrs.get("achieved_distribution", {})

        # Compute final distribution error
        distribution_error = self._distribution_error(achieved_distribution, target_distribution)

        if self.verbose >= 1:
            print("\nOptimization complete:")
            print(f"  Best config: {best_config}")
            print(f"  Achieved distribution: {achieved_distribution}")
            print(f"  Distribution error: {distribution_error:.4f}")
            print(f"  Time: {optimization_time:.1f}s")

        return LabelOptimizationResult(
            best_config=best_config,
            best_score=study.best_value,
            target_distribution=target_distribution,
            achieved_distribution=achieved_distribution,
            distribution_error=distribution_error,
            n_trials=len(study.trials),
            study=study,
            optimization_time=optimization_time,
            trial_history=trial_history,
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "TripleBarrierConfig",
    "LabelOptimizationResult",
    "LabelOptimizer",
]
