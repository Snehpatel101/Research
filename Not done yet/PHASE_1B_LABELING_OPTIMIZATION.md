# PHASE 1B: LABELING & OPTUNA OPTIMIZATION

**Status:** PLANNING
**Created:** 2026-01-17
**Purpose:** Comprehensive labeling and Optuna-based optimization for features, labels, and hyperparameters

---

## Overview

Phase 1B handles all label generation and Optuna-based optimization. This phase sits between feature engineering (Phase 1) and adapter integration (Phase 2). All optimization uses Optuna for systematic hyperparameter search.

**Key Components:**
1. Triple-barrier labeling with Optuna optimization
2. Feature selection optimization with Optuna
3. Feature pruning with Optuna
4. Model hyperparameter optimization with Optuna

---

## Task 1B.1: Triple-Barrier Labeling System

### File: `src/labeling/triple_barrier.py`

```python
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np
import pandas as pd

from src.core.constants import DEFAULT_HORIZONS


@dataclass
class TripleBarrierConfig:
    """Configuration for triple-barrier labeling."""

    # Barrier parameters
    upper_mult: float = 2.0          # ATR multiplier for upper barrier
    lower_mult: float = 2.0          # ATR multiplier for lower barrier
    horizon: int = 20                 # Max holding period (bars)

    # ATR calculation
    atr_period: int = 14             # ATR lookback

    # Volatility adjustment
    use_adaptive_barriers: bool = False
    vol_lookback: int = 60           # For adaptive barriers

    # Class balance targets
    target_long_pct: float = 0.33
    target_short_pct: float = 0.33
    target_neutral_pct: float = 0.34


class TripleBarrierLabeler:
    """
    Triple-barrier labeling method (Lopez de Prado).

    Labels:
        -1 = Short (lower barrier hit first)
         0 = Neutral (timeout - neither barrier hit)
        +1 = Long (upper barrier hit first)
    """

    def __init__(self, config: TripleBarrierConfig):
        self.config = config

    def compute_atr(self, df: pd.DataFrame) -> pd.Series:
        """Compute Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return tr.rolling(self.config.atr_period).mean()

    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Apply triple-barrier method to generate labels.

        Args:
            df: OHLCV DataFrame with DatetimeIndex

        Returns:
            Series of labels (-1, 0, +1)
        """
        close = df["close"].values
        atr = self.compute_atr(df).values
        n = len(close)

        labels = np.zeros(n, dtype=np.int8)

        for i in range(n - self.config.horizon):
            entry_price = close[i]
            entry_atr = atr[i]

            if np.isnan(entry_atr):
                labels[i] = 0
                continue

            upper_barrier = entry_price + self.config.upper_mult * entry_atr
            lower_barrier = entry_price - self.config.lower_mult * entry_atr

            # Look forward up to horizon
            for j in range(1, self.config.horizon + 1):
                future_idx = i + j
                if future_idx >= n:
                    break

                future_price = close[future_idx]

                # Check barriers
                if future_price >= upper_barrier:
                    labels[i] = 1  # Long
                    break
                elif future_price <= lower_barrier:
                    labels[i] = -1  # Short
                    break
            # If loop completes without break, label stays 0 (neutral/timeout)

        # Mark last horizon bars as NaN (can't label)
        labels[-self.config.horizon:] = 0

        return pd.Series(labels, index=df.index, name="label")

    def get_class_distribution(self, labels: pd.Series) -> dict:
        """Get class distribution statistics."""
        counts = labels.value_counts()
        total = len(labels)
        return {
            "short_count": counts.get(-1, 0),
            "neutral_count": counts.get(0, 0),
            "long_count": counts.get(1, 0),
            "short_pct": counts.get(-1, 0) / total,
            "neutral_pct": counts.get(0, 0) / total,
            "long_pct": counts.get(1, 0) / total,
        }
```

---

## Task 1B.2: Optuna Label Optimization

### File: `src/labeling/optimization.py`

```python
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

from .triple_barrier import TripleBarrierConfig, TripleBarrierLabeler


@dataclass
class LabelOptimizationResult:
    """Result from Optuna label optimization."""

    best_config: TripleBarrierConfig
    best_score: float
    class_distribution: dict
    n_trials: int
    study: optuna.Study
    optimization_history: list


class LabelOptimizer:
    """
    Optuna-based optimization for triple-barrier parameters.

    Optimizes:
    - upper_mult: Upper barrier ATR multiplier
    - lower_mult: Lower barrier ATR multiplier
    - horizon: Maximum holding period
    - atr_period: ATR calculation window

    Objectives:
    - Class balance (minimize deviation from target distribution)
    - Label predictability (via quick model validation)
    - Sufficient samples per class
    """

    def __init__(
        self,
        n_trials: int = 100,
        target_distribution: dict = None,
        min_samples_per_class: int = 500,
        use_predictability_score: bool = True,
        random_state: int = 42,
    ):
        self.n_trials = n_trials
        self.target_distribution = target_distribution or {
            "long": 0.33,
            "neutral": 0.34,
            "short": 0.33,
        }
        self.min_samples_per_class = min_samples_per_class
        self.use_predictability_score = use_predictability_score
        self.random_state = random_state

    def optimize(
        self,
        df: pd.DataFrame,
        feature_df: Optional[pd.DataFrame] = None,
    ) -> LabelOptimizationResult:
        """
        Optimize triple-barrier parameters using Optuna.

        Args:
            df: OHLCV DataFrame
            feature_df: Optional features for predictability scoring

        Returns:
            LabelOptimizationResult with best config
        """
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name="triple_barrier_optimization",
        )

        def objective(trial: optuna.Trial) -> float:
            # Sample parameters
            config = TripleBarrierConfig(
                upper_mult=trial.suggest_float("upper_mult", 0.5, 4.0),
                lower_mult=trial.suggest_float("lower_mult", 0.5, 4.0),
                horizon=trial.suggest_int("horizon", 5, 60),
                atr_period=trial.suggest_int("atr_period", 7, 28),
            )

            # Generate labels
            labeler = TripleBarrierLabeler(config)
            labels = labeler.create_labels(df)

            # Score 1: Class balance (0-1, higher = better)
            dist = labeler.get_class_distribution(labels)
            balance_score = self._compute_balance_score(dist)

            # Score 2: Minimum samples check
            min_count = min(dist["short_count"], dist["neutral_count"], dist["long_count"])
            if min_count < self.min_samples_per_class:
                return 0.0  # Reject configurations with too few samples

            # Score 3: Predictability (optional)
            predictability_score = 1.0
            if self.use_predictability_score and feature_df is not None:
                predictability_score = self._compute_predictability(
                    feature_df, labels, trial
                )

            # Combined score
            total_score = 0.6 * balance_score + 0.4 * predictability_score

            # Store metrics for analysis
            trial.set_user_attr("balance_score", balance_score)
            trial.set_user_attr("predictability_score", predictability_score)
            trial.set_user_attr("class_distribution", dist)

            return total_score

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
            n_jobs=1,  # Single-threaded for reproducibility
        )

        # Extract best config
        best_params = study.best_trial.params
        best_config = TripleBarrierConfig(
            upper_mult=best_params["upper_mult"],
            lower_mult=best_params["lower_mult"],
            horizon=best_params["horizon"],
            atr_period=best_params["atr_period"],
        )

        # Generate final labels with best config
        final_labeler = TripleBarrierLabeler(best_config)
        final_labels = final_labeler.create_labels(df)
        final_dist = final_labeler.get_class_distribution(final_labels)

        return LabelOptimizationResult(
            best_config=best_config,
            best_score=study.best_value,
            class_distribution=final_dist,
            n_trials=self.n_trials,
            study=study,
            optimization_history=[
                {"trial": t.number, "value": t.value, "params": t.params}
                for t in study.trials
            ],
        )

    def _compute_balance_score(self, dist: dict) -> float:
        """Compute how balanced the class distribution is."""
        target_long = self.target_distribution["long"]
        target_neutral = self.target_distribution["neutral"]
        target_short = self.target_distribution["short"]

        deviation = (
            abs(dist["long_pct"] - target_long) +
            abs(dist["neutral_pct"] - target_neutral) +
            abs(dist["short_pct"] - target_short)
        )

        # Convert deviation to score (0 deviation = 1.0 score)
        return max(0, 1.0 - deviation)

    def _compute_predictability(
        self,
        feature_df: pd.DataFrame,
        labels: pd.Series,
        trial: optuna.Trial,
    ) -> float:
        """Quick predictability check using simple model."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        # Align features and labels
        common_idx = feature_df.index.intersection(labels.index)
        X = feature_df.loc[common_idx].values
        y = labels.loc[common_idx].values

        # Remove NaN labels
        valid_mask = ~np.isnan(y) & (y != 0) | (y == 0)
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < 1000:
            return 0.5  # Not enough samples

        # Quick RF check with limited trees
        clf = RandomForestClassifier(
            n_estimators=20,
            max_depth=5,
            random_state=self.random_state,
            n_jobs=-1,
        )

        try:
            scores = cross_val_score(clf, X[:5000], y[:5000], cv=3, scoring="f1_weighted")
            return float(np.mean(scores))
        except Exception:
            return 0.5
```

---

## Task 1B.3: Feature Selection Optimization with Optuna

### File: `src/features/selection.py`

```python
from dataclasses import dataclass
from typing import List, Optional, Callable
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler


@dataclass
class FeatureSelectionResult:
    """Result from Optuna feature selection."""

    selected_features: List[str]
    all_features: List[str]
    n_selected: int
    n_total: int
    selection_ratio: float
    best_score: float
    feature_importance: dict
    study: optuna.Study


class FeatureSelector:
    """
    Optuna-based feature selection.

    Strategies:
    1. Binary selection: Include/exclude each feature
    2. Family selection: Select entire feature families
    3. Importance-based: Use model importance for guidance
    """

    def __init__(
        self,
        n_trials: int = 100,
        min_features: int = 10,
        max_features: int = 150,
        selection_strategy: str = "binary",  # binary, family, importance
        cv_folds: int = 3,
        random_state: int = 42,
    ):
        self.n_trials = n_trials
        self.min_features = min_features
        self.max_features = max_features
        self.selection_strategy = selection_strategy
        self.cv_folds = cv_folds
        self.random_state = random_state

    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model_fn: Callable,
        scoring: str = "f1_weighted",
    ) -> FeatureSelectionResult:
        """
        Select optimal feature subset using Optuna.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels
            feature_names: List of feature column names
            model_fn: Function that returns a fresh model instance
            scoring: Sklearn scoring metric

        Returns:
            FeatureSelectionResult with selected features
        """
        from sklearn.model_selection import cross_val_score

        n_features = len(feature_names)
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name="feature_selection",
        )

        def objective(trial: optuna.Trial) -> float:
            if self.selection_strategy == "binary":
                # Binary selection: each feature on/off
                selected_idx = []
                for i, fname in enumerate(feature_names):
                    if trial.suggest_categorical(f"f_{i}", [True, False]):
                        selected_idx.append(i)
            else:
                # Top-K selection
                k = trial.suggest_int("n_features", self.min_features, self.max_features)
                # Use importance scores if available
                selected_idx = list(range(min(k, n_features)))

            # Enforce minimum features
            if len(selected_idx) < self.min_features:
                return 0.0

            # Subset features
            X_subset = X[:, selected_idx]

            # Cross-validate
            model = model_fn()
            try:
                scores = cross_val_score(
                    model, X_subset, y,
                    cv=self.cv_folds,
                    scoring=scoring,
                    n_jobs=-1,
                )
                score = float(np.mean(scores))
            except Exception:
                score = 0.0

            # Store selected count
            trial.set_user_attr("n_selected", len(selected_idx))

            return score

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        # Extract selected features from best trial
        best_params = study.best_trial.params
        selected_features = [
            feature_names[i]
            for i in range(n_features)
            if best_params.get(f"f_{i}", False)
        ]

        # Compute feature importance based on selection frequency
        feature_importance = self._compute_selection_frequency(study, feature_names)

        return FeatureSelectionResult(
            selected_features=selected_features,
            all_features=feature_names,
            n_selected=len(selected_features),
            n_total=n_features,
            selection_ratio=len(selected_features) / n_features,
            best_score=study.best_value,
            feature_importance=feature_importance,
            study=study,
        )

    def _compute_selection_frequency(
        self,
        study: optuna.Study,
        feature_names: List[str],
    ) -> dict:
        """Compute how often each feature was selected in top trials."""
        # Get top 20% of trials
        trials = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)
        top_trials = trials[:max(1, len(trials) // 5)]

        freq = {fname: 0 for fname in feature_names}
        for trial in top_trials:
            for i, fname in enumerate(feature_names):
                if trial.params.get(f"f_{i}", False):
                    freq[fname] += 1

        # Normalize
        n_top = len(top_trials)
        return {fname: count / n_top for fname, count in freq.items()}
```

---

## Task 1B.4: Feature Pruning with Optuna

### File: `src/features/pruning.py`

```python
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner


@dataclass
class FeaturePruningResult:
    """Result from feature pruning."""

    original_features: List[str]
    pruned_features: List[str]
    removed_features: List[str]
    n_original: int
    n_pruned: int
    n_removed: int
    pruning_ratio: float
    score_before: float
    score_after: float
    improvement: float


class FeaturePruner:
    """
    Optuna-based feature pruning.

    Starts with full feature set and iteratively removes
    low-importance features while monitoring performance.

    Methods:
    1. Importance-based: Remove lowest importance features
    2. Correlation-based: Remove highly correlated features
    3. Null-importance: Remove features with null importance
    """

    def __init__(
        self,
        n_trials: int = 50,
        pruning_strategy: str = "importance",  # importance, correlation, null
        min_features: int = 20,
        importance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        cv_folds: int = 3,
        random_state: int = 42,
    ):
        self.n_trials = n_trials
        self.pruning_strategy = pruning_strategy
        self.min_features = min_features
        self.importance_threshold = importance_threshold
        self.correlation_threshold = correlation_threshold
        self.cv_folds = cv_folds
        self.random_state = random_state

    def prune_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model_fn,
        scoring: str = "f1_weighted",
    ) -> FeaturePruningResult:
        """
        Prune features using Optuna with early stopping.

        Args:
            X: Feature matrix
            y: Labels
            feature_names: Feature names
            model_fn: Model factory function
            scoring: Scoring metric

        Returns:
            FeaturePruningResult
        """
        from sklearn.model_selection import cross_val_score

        # Get baseline score with all features
        model = model_fn()
        baseline_scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=scoring)
        score_before = float(np.mean(baseline_scores))

        # Get feature importance from baseline model
        model.fit(X, y)
        importance = self._get_feature_importance(model, feature_names)

        # Sort features by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        # Optuna optimization with pruning
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        study = optuna.create_study(
            direction="maximize",
            pruner=pruner,
            study_name="feature_pruning",
        )

        def objective(trial: optuna.Trial) -> float:
            # How many features to keep (from top)
            n_keep = trial.suggest_int("n_features", self.min_features, len(feature_names))

            # Select top-N features by importance
            keep_features = [f for f, _ in sorted_features[:n_keep]]
            keep_idx = [feature_names.index(f) for f in keep_features]

            X_pruned = X[:, keep_idx]

            # Cross-validate
            model = model_fn()
            scores = cross_val_score(model, X_pruned, y, cv=self.cv_folds, scoring=scoring)

            # Report intermediate value for pruning
            for fold_idx, score in enumerate(scores):
                trial.report(score, fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(scores))

        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        # Get optimal feature count
        best_n = study.best_trial.params["n_features"]
        pruned_features = [f for f, _ in sorted_features[:best_n]]
        removed_features = [f for f in feature_names if f not in pruned_features]

        # Get final score
        keep_idx = [feature_names.index(f) for f in pruned_features]
        model = model_fn()
        final_scores = cross_val_score(model, X[:, keep_idx], y, cv=self.cv_folds, scoring=scoring)
        score_after = float(np.mean(final_scores))

        return FeaturePruningResult(
            original_features=feature_names,
            pruned_features=pruned_features,
            removed_features=removed_features,
            n_original=len(feature_names),
            n_pruned=len(pruned_features),
            n_removed=len(removed_features),
            pruning_ratio=len(removed_features) / len(feature_names),
            score_before=score_before,
            score_after=score_after,
            improvement=score_after - score_before,
        )

    def _get_feature_importance(self, model, feature_names: List[str]) -> dict:
        """Extract feature importance from fitted model."""
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            # Default to uniform importance
            imp = np.ones(len(feature_names)) / len(feature_names)

        return dict(zip(feature_names, imp))

    def prune_by_correlation(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> List[str]:
        """Remove highly correlated features."""
        corr_matrix = np.corrcoef(X.T)

        # Find pairs above threshold
        to_remove = set()
        for i in range(len(feature_names)):
            if feature_names[i] in to_remove:
                continue
            for j in range(i + 1, len(feature_names)):
                if feature_names[j] in to_remove:
                    continue
                if abs(corr_matrix[i, j]) > self.correlation_threshold:
                    # Remove the second one
                    to_remove.add(feature_names[j])

        return [f for f in feature_names if f not in to_remove]

    def null_importance_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model_fn,
        n_permutations: int = 20,
    ) -> List[str]:
        """Remove features with null importance (permutation test)."""
        from sklearn.inspection import permutation_importance

        model = model_fn()
        model.fit(X, y)

        # Permutation importance
        result = permutation_importance(
            model, X, y,
            n_repeats=n_permutations,
            random_state=self.random_state,
            n_jobs=-1,
        )

        # Keep features with positive importance (mean > 0)
        keep_mask = result.importances_mean > 0
        return [f for f, keep in zip(feature_names, keep_mask) if keep]
```

---

## Task 1B.5: Model Hyperparameter Optimization with Optuna

### File: `src/optimization/hyperparameters.py`

```python
from dataclasses import dataclass
from typing import Dict, Any, Callable, Optional
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner


@dataclass
class HyperparameterResult:
    """Result from hyperparameter optimization."""

    model_name: str
    best_params: Dict[str, Any]
    best_score: float
    default_score: float
    improvement: float
    n_trials: int
    study: optuna.Study


# Default search spaces per model family
HYPERPARAMETER_SPACES = {
    "xgboost": {
        "n_estimators": ("int", 100, 1000),
        "max_depth": ("int", 3, 10),
        "learning_rate": ("log_float", 0.01, 0.3),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
        "min_child_weight": ("int", 1, 10),
        "gamma": ("float", 0, 5),
        "reg_alpha": ("log_float", 1e-8, 10.0),
        "reg_lambda": ("log_float", 1e-8, 10.0),
    },
    "lightgbm": {
        "n_estimators": ("int", 100, 1000),
        "max_depth": ("int", 3, 12),
        "learning_rate": ("log_float", 0.01, 0.3),
        "num_leaves": ("int", 20, 150),
        "feature_fraction": ("float", 0.6, 1.0),
        "bagging_fraction": ("float", 0.6, 1.0),
        "bagging_freq": ("int", 1, 7),
        "min_child_samples": ("int", 5, 100),
        "reg_alpha": ("log_float", 1e-8, 10.0),
        "reg_lambda": ("log_float", 1e-8, 10.0),
    },
    "catboost": {
        "iterations": ("int", 100, 1000),
        "depth": ("int", 4, 10),
        "learning_rate": ("log_float", 0.01, 0.3),
        "l2_leaf_reg": ("log_float", 1, 10),
        "border_count": ("int", 32, 255),
        "bagging_temperature": ("float", 0, 1),
    },
    "random_forest": {
        "n_estimators": ("int", 50, 500),
        "max_depth": ("int", 5, 30),
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 10),
        "max_features": ("categorical", ["sqrt", "log2", None]),
    },
    "lstm": {
        "hidden_size": ("categorical", [32, 64, 128, 256]),
        "num_layers": ("int", 1, 4),
        "dropout": ("float", 0.0, 0.5),
        "learning_rate": ("log_float", 1e-4, 1e-2),
        "batch_size": ("categorical", [32, 64, 128, 256]),
    },
    "transformer": {
        "d_model": ("categorical", [32, 64, 128]),
        "n_heads": ("categorical", [2, 4, 8]),
        "n_layers": ("int", 1, 4),
        "d_ff": ("categorical", [64, 128, 256, 512]),
        "dropout": ("float", 0.0, 0.3),
        "learning_rate": ("log_float", 1e-5, 1e-3),
    },
}


class HyperparameterOptimizer:
    """
    Optuna-based hyperparameter optimization.

    Supports:
    - All boosting models (XGBoost, LightGBM, CatBoost)
    - Classical models (RF, Logistic, SVM)
    - Neural models (LSTM, GRU, Transformer)
    """

    def __init__(
        self,
        n_trials: int = 100,
        cv_folds: int = 3,
        scoring: str = "f1_weighted",
        use_pruning: bool = True,
        random_state: int = 42,
    ):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.use_pruning = use_pruning
        self.random_state = random_state

    def optimize(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        model_factory: Callable,
        search_space: Optional[Dict] = None,
    ) -> HyperparameterResult:
        """
        Optimize hyperparameters for a model.

        Args:
            model_name: Name of the model
            X: Feature matrix
            y: Labels
            model_factory: Function(params) -> model
            search_space: Optional custom search space

        Returns:
            HyperparameterResult
        """
        from sklearn.model_selection import cross_val_score

        # Get search space
        space = search_space or HYPERPARAMETER_SPACES.get(model_name, {})

        # Baseline score with defaults
        default_model = model_factory({})
        default_scores = cross_val_score(default_model, X, y, cv=self.cv_folds, scoring=self.scoring)
        default_score = float(np.mean(default_scores))

        # Setup study
        sampler = TPESampler(seed=self.random_state)
        pruner = HyperbandPruner() if self.use_pruning else None
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=f"{model_name}_hyperparam_opt",
        )

        def objective(trial: optuna.Trial) -> float:
            params = {}

            for param_name, param_spec in space.items():
                param_type = param_spec[0]

                if param_type == "int":
                    params[param_name] = trial.suggest_int(param_name, param_spec[1], param_spec[2])
                elif param_type == "float":
                    params[param_name] = trial.suggest_float(param_name, param_spec[1], param_spec[2])
                elif param_type == "log_float":
                    params[param_name] = trial.suggest_float(param_name, param_spec[1], param_spec[2], log=True)
                elif param_type == "categorical":
                    params[param_name] = trial.suggest_categorical(param_name, param_spec[1])

            model = model_factory(params)

            try:
                scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=self.scoring)

                # Report for pruning
                for fold_idx, score in enumerate(scores):
                    trial.report(score, fold_idx)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                return float(np.mean(scores))
            except Exception as e:
                return 0.0

        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        return HyperparameterResult(
            model_name=model_name,
            best_params=study.best_trial.params,
            best_score=study.best_value,
            default_score=default_score,
            improvement=study.best_value - default_score,
            n_trials=self.n_trials,
            study=study,
        )
```

---

## Task 1B.6: Unified Optimization Pipeline

### File: `src/optimization/pipeline.py`

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

from src.labeling.optimization import LabelOptimizer, LabelOptimizationResult
from src.features.selection import FeatureSelector, FeatureSelectionResult
from src.features.pruning import FeaturePruner, FeaturePruningResult
from .hyperparameters import HyperparameterOptimizer, HyperparameterResult


@dataclass
class FullOptimizationResult:
    """Complete optimization results."""

    label_result: LabelOptimizationResult
    selection_result: FeatureSelectionResult
    pruning_result: FeaturePruningResult
    hyperparam_results: Dict[str, HyperparameterResult]

    final_features: List[str]
    final_label_config: Any
    final_hyperparams: Dict[str, Dict[str, Any]]

    total_trials: int
    optimization_time_seconds: float


class OptimizationPipeline:
    """
    Unified Optuna optimization pipeline.

    Runs all optimizations in sequence:
    1. Label optimization (triple-barrier parameters)
    2. Feature selection (binary/importance-based)
    3. Feature pruning (remove low-value features)
    4. Hyperparameter optimization (per model)
    """

    def __init__(
        self,
        label_trials: int = 100,
        feature_trials: int = 100,
        pruning_trials: int = 50,
        hyperparam_trials: int = 100,
        random_state: int = 42,
    ):
        self.label_trials = label_trials
        self.feature_trials = feature_trials
        self.pruning_trials = pruning_trials
        self.hyperparam_trials = hyperparam_trials
        self.random_state = random_state

    def run_full_optimization(
        self,
        ohlcv_df: pd.DataFrame,
        feature_df: pd.DataFrame,
        models: List[str],
        model_factories: Dict[str, Any],
    ) -> FullOptimizationResult:
        """
        Run complete optimization pipeline.

        Args:
            ohlcv_df: Raw OHLCV data
            feature_df: Computed features
            models: List of model names to optimize
            model_factories: Dict of model_name -> factory function

        Returns:
            FullOptimizationResult with all optimization outputs
        """
        import time
        start_time = time.time()
        total_trials = 0

        # 1. Optimize labels
        print("=== Stage 1: Label Optimization ===")
        label_optimizer = LabelOptimizer(
            n_trials=self.label_trials,
            random_state=self.random_state,
        )
        label_result = label_optimizer.optimize(ohlcv_df, feature_df)
        total_trials += self.label_trials

        # Generate optimized labels
        from src.labeling.triple_barrier import TripleBarrierLabeler
        labeler = TripleBarrierLabeler(label_result.best_config)
        labels = labeler.create_labels(ohlcv_df)

        # Align features and labels
        common_idx = feature_df.index.intersection(labels.index)
        X = feature_df.loc[common_idx].values
        y = labels.loc[common_idx].values
        feature_names = list(feature_df.columns)

        # 2. Feature selection
        print("=== Stage 2: Feature Selection ===")
        selector = FeatureSelector(
            n_trials=self.feature_trials,
            random_state=self.random_state,
        )
        selection_result = selector.select_features(
            X, y, feature_names,
            model_fn=lambda: model_factories[models[0]]({}),
        )
        total_trials += self.feature_trials

        # Update feature set
        selected_idx = [feature_names.index(f) for f in selection_result.selected_features]
        X_selected = X[:, selected_idx]

        # 3. Feature pruning
        print("=== Stage 3: Feature Pruning ===")
        pruner = FeaturePruner(
            n_trials=self.pruning_trials,
            random_state=self.random_state,
        )
        pruning_result = pruner.prune_features(
            X_selected, y, selection_result.selected_features,
            model_fn=lambda: model_factories[models[0]]({}),
        )
        total_trials += self.pruning_trials

        # Final feature set
        final_features = pruning_result.pruned_features
        final_idx = [selection_result.selected_features.index(f) for f in final_features]
        X_final = X_selected[:, final_idx]

        # 4. Hyperparameter optimization per model
        print("=== Stage 4: Hyperparameter Optimization ===")
        hyperparam_optimizer = HyperparameterOptimizer(
            n_trials=self.hyperparam_trials,
            random_state=self.random_state,
        )

        hyperparam_results = {}
        final_hyperparams = {}
        for model_name in models:
            print(f"  Optimizing {model_name}...")
            result = hyperparam_optimizer.optimize(
                model_name, X_final, y,
                model_factory=model_factories[model_name],
            )
            hyperparam_results[model_name] = result
            final_hyperparams[model_name] = result.best_params
            total_trials += self.hyperparam_trials

        elapsed = time.time() - start_time

        return FullOptimizationResult(
            label_result=label_result,
            selection_result=selection_result,
            pruning_result=pruning_result,
            hyperparam_results=hyperparam_results,
            final_features=final_features,
            final_label_config=label_result.best_config,
            final_hyperparams=final_hyperparams,
            total_trials=total_trials,
            optimization_time_seconds=elapsed,
        )
```

---

## Implementation Checklist

### Task 1B.1: Triple-Barrier Labeling
- [ ] Create `src/labeling/triple_barrier.py`
- [ ] `TripleBarrierConfig` dataclass
- [ ] `TripleBarrierLabeler` class
- [ ] ATR computation
- [ ] Label generation with barriers

### Task 1B.2: Label Optimization
- [ ] Create `src/labeling/optimization.py`
- [ ] `LabelOptimizationResult` dataclass
- [ ] `LabelOptimizer` with Optuna
- [ ] Balance scoring
- [ ] Predictability scoring

### Task 1B.3: Feature Selection
- [ ] Create `src/features/selection.py`
- [ ] `FeatureSelectionResult` dataclass
- [ ] `FeatureSelector` with binary selection
- [ ] Family-based selection
- [ ] Selection frequency importance

### Task 1B.4: Feature Pruning
- [ ] Create `src/features/pruning.py`
- [ ] `FeaturePruningResult` dataclass
- [ ] `FeaturePruner` with importance-based pruning
- [ ] Correlation-based pruning
- [ ] Null importance test

### Task 1B.5: Hyperparameter Optimization
- [ ] Create `src/optimization/hyperparameters.py`
- [ ] `HyperparameterResult` dataclass
- [ ] `HYPERPARAMETER_SPACES` for all models
- [ ] `HyperparameterOptimizer` with Optuna
- [ ] Hyperband pruning

### Task 1B.6: Unified Pipeline
- [ ] Create `src/optimization/pipeline.py`
- [ ] `FullOptimizationResult` dataclass
- [ ] `OptimizationPipeline` class
- [ ] Sequential optimization stages

---

## Integration with PipelineConfig

```python
@dataclass
class PipelineConfig:
    # ... existing fields ...

    # ═══════════════════════════════════════════════════════════
    # OPTUNA OPTIMIZATION CONFIGURATION
    # ═══════════════════════════════════════════════════════════

    # Label optimization
    optimize_labels: bool = True
    label_optimization_trials: int = 100
    target_class_distribution: dict = None  # {"long": 0.33, "neutral": 0.34, "short": 0.33}

    # Feature optimization
    optimize_features: bool = True
    feature_selection_trials: int = 100
    feature_pruning_trials: int = 50
    min_features: int = 20

    # Hyperparameter optimization
    optimize_hyperparams: bool = True
    hyperparam_trials: int = 100

    # Optuna settings
    optuna_random_state: int = 42
    optuna_n_jobs: int = 1  # Parallel trials
```

---

## Document Metadata

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Created | 2026-01-17 |
| Purpose | Labeling and Optuna optimization |
| Related Docs | PHASE_1_UNIFIED_FEATURES.md, PHASE_3_TRAINING_ORCHESTRATION.md |
