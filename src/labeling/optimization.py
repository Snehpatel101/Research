"""
Optuna-based Label Optimization for PHASE_1B.

Optimizes triple-barrier labeling parameters using Optuna's TPESampler
to achieve balanced class distributions while maintaining predictability.

Optimization Objectives:
    1. Class Balance: Even distribution across long/neutral/short
    2. Predictability: Labels should be learnable from features
    3. Minimum Samples: Each class should have sufficient samples

Search Spaces:
    - upper_mult: float(0.5, 4.0) - Upper barrier ATR multiplier
    - lower_mult: float(0.5, 4.0) - Lower barrier ATR multiplier
    - horizon: int(5, 60) - Maximum holding period
    - atr_period: int(7, 28) - ATR calculation window
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.trial import Trial

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
    TPESampler = None
    Trial = None

from src.core.constants import (
    DEFAULT_LABEL_OPTIMIZATION_TRIALS,
    DEFAULT_OPTUNA_RANDOM_STATE,
    N_CLASSES,
)
from src.core.validation import ValidationError, validate_ohlcv

from src.labeling.triple_barrier import TripleBarrierConfig, TripleBarrierLabeler


logger = logging.getLogger(__name__)


# =============================================================================
# OPTIMIZATION RESULT
# =============================================================================


@dataclass
class LabelOptimizationResult:
    """
    Result of label parameter optimization.

    Contains the best configuration found, optimization metrics,
    and the full Optuna study for further analysis.

    Attributes:
        best_config: Optimal TripleBarrierConfig.
        best_score: Best objective score achieved.
        class_distribution: Class distribution with best config.
        n_trials: Number of trials completed.
        study: Optuna Study object for analysis.
        optimization_history: List of trial results.

    Example:
        >>> result = optimizer.optimize(df)
        >>> print(f"Best score: {result.best_score:.4f}")
        >>> print(f"Best config: {result.best_config}")
        >>> labeler = TripleBarrierLabeler(result.best_config)
    """

    best_config: TripleBarrierConfig
    best_score: float
    class_distribution: Dict[str, float]
    n_trials: int
    study: Any  # optuna.Study (Any for optional import)
    optimization_history: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "best_config": self.best_config.to_dict(),
            "best_score": self.best_score,
            "class_distribution": self.class_distribution,
            "n_trials": self.n_trials,
            "optimization_history": self.optimization_history,
        }

    def get_param_importance(self) -> Dict[str, float]:
        """
        Get parameter importance from Optuna study.

        Returns:
            dict: Parameter name -> importance score mapping.
        """
        if not OPTUNA_AVAILABLE or self.study is None:
            return {}

        try:
            importance = optuna.importance.get_param_importances(self.study)
            return dict(importance)
        except Exception as e:
            logger.warning(f"Could not compute parameter importance: {e}")
            return {}

    def plot_optimization_history(self) -> None:
        """Plot optimization history using Optuna visualization."""
        if not OPTUNA_AVAILABLE or self.study is None:
            logger.warning("Optuna not available for plotting")
            return

        try:
            from optuna.visualization import plot_optimization_history

            fig = plot_optimization_history(self.study)
            fig.show()
        except Exception as e:
            logger.warning(f"Could not plot optimization history: {e}")

    def plot_param_importances(self) -> None:
        """Plot parameter importances using Optuna visualization."""
        if not OPTUNA_AVAILABLE or self.study is None:
            logger.warning("Optuna not available for plotting")
            return

        try:
            from optuna.visualization import plot_param_importances

            fig = plot_param_importances(self.study)
            fig.show()
        except Exception as e:
            logger.warning(f"Could not plot parameter importances: {e}")


# =============================================================================
# LABEL OPTIMIZER
# =============================================================================


class LabelOptimizer:
    """
    Optuna-based optimization for triple-barrier labeling parameters.

    Uses Tree-structured Parzen Estimator (TPE) sampling to efficiently
    search the parameter space for configurations that produce:
        1. Balanced class distributions
        2. Predictable labels (if features provided)
        3. Sufficient samples per class

    Search Spaces:
        - upper_mult: float(0.5, 4.0)
        - lower_mult: float(0.5, 4.0)
        - horizon: int(5, 60)
        - atr_period: int(7, 28)

    Attributes:
        n_trials: Number of optimization trials.
        target_distribution: Target class distribution (default: balanced).
        min_samples_per_class: Minimum samples required per class.
        use_predictability_score: Include predictability in objective.
        random_state: Random seed for reproducibility.

    Example:
        >>> optimizer = LabelOptimizer(n_trials=100, min_samples_per_class=500)
        >>> result = optimizer.optimize(ohlcv_df, feature_df)
        >>> print(f"Best config: {result.best_config}")
        >>> print(f"Distribution: {result.class_distribution}")
    """

    # Default search space bounds
    SEARCH_SPACE = {
        "upper_mult": (0.5, 4.0),
        "lower_mult": (0.5, 4.0),
        "horizon": (5, 60),
        "atr_period": (7, 28),
    }

    def __init__(
        self,
        n_trials: int = DEFAULT_LABEL_OPTIMIZATION_TRIALS,
        target_distribution: Optional[Dict[str, float]] = None,
        min_samples_per_class: int = 500,
        use_predictability_score: bool = True,
        random_state: int = DEFAULT_OPTUNA_RANDOM_STATE,
        balance_weight: float = 0.6,
        predictability_weight: float = 0.3,
        sample_penalty_weight: float = 0.1,
        verbose: int = 1,
    ) -> None:
        """
        Initialize the label optimizer.

        Args:
            n_trials: Number of Optuna trials to run.
            target_distribution: Target class distribution.
                Default: {'long': 0.33, 'neutral': 0.34, 'short': 0.33}
            min_samples_per_class: Minimum samples per class.
                Configurations with fewer samples are penalized.
            use_predictability_score: If True and features provided,
                include predictability in the objective.
            random_state: Random seed for reproducibility.
            balance_weight: Weight for balance score in objective (0-1).
            predictability_weight: Weight for predictability score (0-1).
            sample_penalty_weight: Weight for sample count penalty (0-1).
            verbose: Verbosity level (0=silent, 1=progress, 2=debug).

        Raises:
            ImportError: If Optuna is not installed.
            ValidationError: If weights don't sum to 1.0.
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for label optimization. "
                "Install with: pip install optuna>=3.4.0"
            )

        self.n_trials = n_trials
        self.target_distribution = target_distribution or {
            "long": 0.33,
            "neutral": 0.34,
            "short": 0.33,
        }
        self.min_samples_per_class = min_samples_per_class
        self.use_predictability_score = use_predictability_score
        self.random_state = random_state
        self.balance_weight = balance_weight
        self.predictability_weight = predictability_weight
        self.sample_penalty_weight = sample_penalty_weight
        self.verbose = verbose

        # Validate weights
        total_weight = balance_weight + predictability_weight + sample_penalty_weight
        if abs(total_weight - 1.0) > 0.01:
            raise ValidationError(
                "Objective weights must sum to 1.0",
                field="weights",
                expected=1.0,
                actual=total_weight,
            )

        # Internal state
        self._df: Optional[pd.DataFrame] = None
        self._feature_df: Optional[pd.DataFrame] = None
        self._optimization_history: List[Dict[str, Any]] = []

    def optimize(
        self,
        df: pd.DataFrame,
        feature_df: Optional[pd.DataFrame] = None,
        search_space: Optional[Dict[str, Tuple[float, float]]] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> LabelOptimizationResult:
        """
        Optimize triple-barrier labeling parameters.

        Runs Optuna optimization to find the best combination of:
            - upper_mult: Upper barrier multiplier
            - lower_mult: Lower barrier multiplier
            - horizon: Maximum holding period
            - atr_period: ATR calculation period

        Args:
            df: OHLCV DataFrame with DatetimeIndex.
            feature_df: Optional feature DataFrame for predictability scoring.
                Should be aligned with df (same index).
            search_space: Optional custom search space bounds.
                Default: {'upper_mult': (0.5, 4.0), ...}
            callbacks: Optional list of Optuna callback functions.

        Returns:
            LabelOptimizationResult: Optimization results with best config.

        Raises:
            ValidationError: If DataFrame is invalid.
            RuntimeError: If optimization fails.
        """
        # Validate input
        validate_ohlcv(df, context="LabelOptimizer")

        if feature_df is not None:
            if len(feature_df) != len(df):
                raise ValidationError(
                    "feature_df must have same length as df",
                    field="feature_df",
                    expected=len(df),
                    actual=len(feature_df),
                )

        # Store for objective function
        self._df = df
        self._feature_df = feature_df
        self._optimization_history = []

        # Use custom or default search space
        space = search_space or self.SEARCH_SPACE

        # Create Optuna study
        sampler = TPESampler(seed=self.random_state)

        # Set log level based on verbosity
        if self.verbose == 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        elif self.verbose == 1:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.DEBUG)

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name="triple_barrier_optimization",
        )

        # Define objective with search space
        def objective(trial: Trial) -> float:
            return self._objective(trial, space)

        # Run optimization
        logger.info(f"Starting label optimization with {self.n_trials} trials...")

        try:
            study.optimize(
                objective,
                n_trials=self.n_trials,
                callbacks=callbacks,
                show_progress_bar=(self.verbose >= 1),
            )
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise RuntimeError(f"Label optimization failed: {e}") from e

        # Extract best config
        best_params = study.best_params
        best_config = TripleBarrierConfig(
            upper_mult=best_params["upper_mult"],
            lower_mult=best_params["lower_mult"],
            horizon=best_params["horizon"],
            atr_period=best_params["atr_period"],
        )

        # Get class distribution with best config
        labeler = TripleBarrierLabeler(best_config)
        best_labels = labeler.create_labels(df)
        best_distribution = labeler.get_class_distribution(best_labels)

        logger.info(
            f"Optimization complete. Best score: {study.best_value:.4f}, "
            f"Distribution: long={best_distribution['long']:.2%}, "
            f"neutral={best_distribution['neutral']:.2%}, "
            f"short={best_distribution['short']:.2%}"
        )

        return LabelOptimizationResult(
            best_config=best_config,
            best_score=study.best_value,
            class_distribution=best_distribution,
            n_trials=len(study.trials),
            study=study,
            optimization_history=self._optimization_history,
        )

    def _objective(
        self, trial: Trial, search_space: Dict[str, Tuple[float, float]]
    ) -> float:
        """
        Optuna objective function.

        Combines multiple objectives:
            1. Balance score: How balanced is the class distribution?
            2. Predictability score: How learnable are the labels?
            3. Sample penalty: Are there enough samples per class?

        Args:
            trial: Optuna trial object.
            search_space: Parameter search bounds.

        Returns:
            float: Combined objective score (higher is better).
        """
        # Sample parameters
        upper_mult = trial.suggest_float(
            "upper_mult",
            search_space["upper_mult"][0],
            search_space["upper_mult"][1],
        )
        lower_mult = trial.suggest_float(
            "lower_mult",
            search_space["lower_mult"][0],
            search_space["lower_mult"][1],
        )
        horizon = trial.suggest_int(
            "horizon",
            search_space["horizon"][0],
            search_space["horizon"][1],
        )
        atr_period = trial.suggest_int(
            "atr_period",
            search_space["atr_period"][0],
            search_space["atr_period"][1],
        )

        # Create config and labels
        try:
            config = TripleBarrierConfig(
                upper_mult=upper_mult,
                lower_mult=lower_mult,
                horizon=horizon,
                atr_period=atr_period,
            )
            labeler = TripleBarrierLabeler(config)
            labels = labeler.create_labels(self._df)
        except Exception as e:
            logger.debug(f"Trial failed with config error: {e}")
            return 0.0

        # Get class distribution
        dist = labeler.get_class_distribution(labels)

        # Handle empty distributions
        if dist["n_samples"] == 0:
            return 0.0

        # Compute scores
        balance_score = self._compute_balance_score(dist)
        sample_penalty = self._compute_sample_penalty(dist)

        # Predictability score (optional)
        if self.use_predictability_score and self._feature_df is not None:
            predictability_score = self._compute_predictability(
                self._feature_df, labels, trial
            )
        else:
            predictability_score = 0.5  # Neutral if not using

        # Combined objective (weighted sum)
        objective_value = (
            self.balance_weight * balance_score
            + self.predictability_weight * predictability_score
            + self.sample_penalty_weight * (1.0 - sample_penalty)
        )

        # Store trial metadata
        trial_result = {
            "trial_number": trial.number,
            "params": {
                "upper_mult": upper_mult,
                "lower_mult": lower_mult,
                "horizon": horizon,
                "atr_period": atr_period,
            },
            "scores": {
                "objective": objective_value,
                "balance": balance_score,
                "predictability": predictability_score,
                "sample_penalty": sample_penalty,
            },
            "class_distribution": dist,
        }
        self._optimization_history.append(trial_result)

        # Set user attributes for Optuna dashboard
        trial.set_user_attr("balance_score", balance_score)
        trial.set_user_attr("predictability_score", predictability_score)
        trial.set_user_attr("sample_penalty", sample_penalty)
        trial.set_user_attr("n_long", dist.get("n_long", 0))
        trial.set_user_attr("n_neutral", dist.get("n_neutral", 0))
        trial.set_user_attr("n_short", dist.get("n_short", 0))

        return objective_value

    def _compute_balance_score(self, dist: Dict[str, float]) -> float:
        """
        Compute class balance score.

        Uses normalized entropy to measure how balanced the distribution is.
        Score of 1.0 = perfectly balanced, 0.0 = all one class.

        Additionally penalizes deviation from target distribution.

        Args:
            dist: Class distribution dictionary.

        Returns:
            float: Balance score (0.0 to 1.0).
        """
        probs = [dist["long"], dist["neutral"], dist["short"]]

        # Filter out zero probabilities
        probs = [p for p in probs if p > 0]

        if len(probs) <= 1:
            return 0.0

        # Normalized entropy
        entropy = -sum(p * np.log(p) for p in probs)
        max_entropy = np.log(N_CLASSES)
        entropy_score = entropy / max_entropy

        # Distance from target distribution
        target_probs = [
            self.target_distribution["long"],
            self.target_distribution["neutral"],
            self.target_distribution["short"],
        ]
        actual_probs = [dist["long"], dist["neutral"], dist["short"]]

        # L1 distance (normalized by 2 for max possible distance)
        l1_distance = sum(abs(a - t) for a, t in zip(actual_probs, target_probs))
        target_score = 1.0 - (l1_distance / 2.0)

        # Combine entropy and target scores
        balance_score = 0.5 * entropy_score + 0.5 * target_score

        return float(balance_score)

    def _compute_sample_penalty(self, dist: Dict[str, float]) -> float:
        """
        Compute penalty for insufficient samples per class.

        Args:
            dist: Class distribution with sample counts.

        Returns:
            float: Penalty score (0.0 = enough samples, 1.0 = severe shortage).
        """
        n_long = dist.get("n_long", 0)
        n_neutral = dist.get("n_neutral", 0)
        n_short = dist.get("n_short", 0)

        min_count = min(n_long, n_neutral, n_short)

        if min_count >= self.min_samples_per_class:
            return 0.0

        # Linear penalty based on shortage
        shortage_ratio = min_count / self.min_samples_per_class
        penalty = 1.0 - shortage_ratio

        return float(penalty)

    def _compute_predictability(
        self,
        feature_df: pd.DataFrame,
        labels: pd.Series,
        trial: Trial,
    ) -> float:
        """
        Compute predictability score using a simple model.

        Uses a small random forest to estimate how well features
        can predict the labels. This helps identify configurations
        where labels are noise vs. signal.

        Args:
            feature_df: Feature DataFrame.
            labels: Generated labels.
            trial: Optuna trial (for pruning if needed).

        Returns:
            float: Predictability score (0.0 to 1.0, higher = more predictable).
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
        except ImportError:
            logger.warning("scikit-learn not available for predictability scoring")
            return 0.5

        # Align features and labels
        valid_mask = ~labels.isna()
        X = feature_df.loc[valid_mask].values
        y = labels.loc[valid_mask].values

        if len(y) < 100:
            return 0.0

        # Handle NaN in features
        if np.isnan(X).any():
            # Simple imputation with column means
            col_means = np.nanmean(X, axis=0)
            nan_mask = np.isnan(X)
            X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        # Check for remaining NaN/Inf
        if np.isnan(X).any() or np.isinf(X).any():
            return 0.0

        try:
            # Quick random forest for predictability estimation
            clf = RandomForestClassifier(
                n_estimators=10,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=1,
            )

            # Use 3-fold CV for speed
            n_splits = min(3, len(np.unique(y)))
            if n_splits < 2:
                return 0.0

            scores = cross_val_score(
                clf, X, y, cv=n_splits, scoring="accuracy", n_jobs=1
            )

            # Normalize score: 0.33 (random) -> 0.0, 1.0 -> 1.0
            avg_score = np.mean(scores)
            baseline = 1.0 / N_CLASSES
            normalized_score = (avg_score - baseline) / (1.0 - baseline)
            normalized_score = max(0.0, min(1.0, normalized_score))

            return float(normalized_score)

        except Exception as e:
            logger.debug(f"Predictability scoring failed: {e}")
            return 0.5

    def get_search_space_info(self) -> Dict[str, Any]:
        """
        Get information about the search space.

        Returns:
            dict: Search space bounds and descriptions.
        """
        return {
            "upper_mult": {
                "bounds": self.SEARCH_SPACE["upper_mult"],
                "description": "ATR multiplier for upper (profit) barrier",
            },
            "lower_mult": {
                "bounds": self.SEARCH_SPACE["lower_mult"],
                "description": "ATR multiplier for lower (stop) barrier",
            },
            "horizon": {
                "bounds": self.SEARCH_SPACE["horizon"],
                "description": "Maximum holding period in bars",
            },
            "atr_period": {
                "bounds": self.SEARCH_SPACE["atr_period"],
                "description": "ATR calculation window",
            },
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def optimize_labels(
    df: pd.DataFrame,
    feature_df: Optional[pd.DataFrame] = None,
    n_trials: int = 100,
    min_samples_per_class: int = 500,
    random_state: int = 42,
    verbose: int = 1,
) -> LabelOptimizationResult:
    """
    Convenience function to optimize triple-barrier labels.

    Args:
        df: OHLCV DataFrame.
        feature_df: Optional feature DataFrame for predictability.
        n_trials: Number of optimization trials.
        min_samples_per_class: Minimum samples per class.
        random_state: Random seed.
        verbose: Verbosity level.

    Returns:
        LabelOptimizationResult: Optimization results.

    Example:
        >>> result = optimize_labels(ohlcv_df, feature_df, n_trials=50)
        >>> labels = TripleBarrierLabeler(result.best_config).create_labels(ohlcv_df)
    """
    optimizer = LabelOptimizer(
        n_trials=n_trials,
        min_samples_per_class=min_samples_per_class,
        use_predictability_score=(feature_df is not None),
        random_state=random_state,
        verbose=verbose,
    )

    return optimizer.optimize(df, feature_df)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "LabelOptimizationResult",
    "LabelOptimizer",
    "optimize_labels",
]
