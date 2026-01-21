# SNwH Implementation: Phase 5 - Feature Strategy Integration

## Overview

Phase 5 wires `MODEL_FEATURE_STRATEGIES` (from `src/features/strategies.py`) into the trainer configuration flow. This ensures each model gets its baseline features based on its declared strategy.

**Current State**: `MODEL_FEATURE_STRATEGIES` is defined but not integrated into the trainer.

**Target State**: Trainer automatically selects features based on model's declared strategy.

---

## 5.1 Feature Strategy Manager

### New File: `src/features/strategy_manager.py`

```python
"""
Feature Strategy Manager - Integrates MODEL_FEATURE_STRATEGIES into training.

This module bridges the gap between:
1. MODEL_FEATURE_STRATEGIES (static definitions)
2. TrainerConfig (runtime configuration)
3. Adapters (data transformation)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .strategies import MODEL_FEATURE_STRATEGIES, ModelFeatureStrategy, get_strategy_for_model

logger = logging.getLogger(__name__)


@dataclass
class ResolvedFeatureSet:
    """
    Resolved feature set for a model.

    Contains the final list of features after:
    1. Baseline selection from strategy
    2. Availability filtering (features present in data)
    3. Optional optimization/pruning
    """
    model_name: str
    feature_columns: list[str]
    n_features: int = 0

    # Resolution metadata
    baseline_requested: int = 0  # Features in strategy baseline
    baseline_available: int = 0  # Baseline features found in data
    optimized: bool = False
    optimization_method: str = ""

    # Feature families
    included_families: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.n_features = len(self.feature_columns)

    def __repr__(self) -> str:
        status = "optimized" if self.optimized else "baseline"
        return (
            f"ResolvedFeatureSet({self.model_name}: "
            f"{self.n_features} features [{status}])"
        )


class FeatureStrategyManager:
    """
    Manages feature selection based on model strategies.

    The manager:
    1. Retrieves the strategy for a model
    2. Filters baseline features to those available in the data
    3. Validates feature count against strategy bounds
    4. Optionally integrates with feature optimization

    Usage:
        manager = FeatureStrategyManager(df)
        features = manager.get_features_for_model("xgboost")
    """

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        available_features: list[str] | None = None,
    ):
        """
        Initialize manager with available features.

        Args:
            df: DataFrame to extract available features from
            available_features: Explicit list of available features
        """
        if available_features:
            self.available_features = set(available_features)
        elif df is not None:
            self.available_features = set(self._detect_features(df))
        else:
            self.available_features = set()

        self._resolved_cache: dict[str, ResolvedFeatureSet] = {}

    def _detect_features(self, df: pd.DataFrame) -> list[str]:
        """Detect feature columns in DataFrame."""
        from src.phase1.utils.constants import METADATA_COLUMNS
        from src.phase1.utils.feature_sets import _is_label_column

        return [
            col for col in df.columns
            if col not in METADATA_COLUMNS and not _is_label_column(col)
        ]

    def get_strategy(self, model_name: str) -> ModelFeatureStrategy:
        """
        Get the feature strategy for a model.

        Args:
            model_name: Name of the model

        Returns:
            ModelFeatureStrategy for the model
        """
        return get_strategy_for_model(model_name)

    def get_features_for_model(
        self,
        model_name: str,
        custom_baseline: list[str] | None = None,
        strict: bool = True,
    ) -> ResolvedFeatureSet:
        """
        Get resolved features for a model.

        Args:
            model_name: Name of the model
            custom_baseline: Override baseline features
            strict: If True, raise error if too few features available

        Returns:
            ResolvedFeatureSet with filtered features

        Raises:
            ValueError: If strict=True and insufficient features
        """
        # Check cache
        cache_key = f"{model_name}_{custom_baseline is not None}"
        if cache_key in self._resolved_cache:
            return self._resolved_cache[cache_key]

        strategy = self.get_strategy(model_name)

        # Get baseline features
        if custom_baseline:
            baseline = custom_baseline
        else:
            baseline = strategy.baseline_features

        # Filter to available features
        available_baseline = [f for f in baseline if f in self.available_features]

        # Log filtering results
        n_requested = len(baseline)
        n_available = len(available_baseline)

        if n_available < n_requested:
            missing = set(baseline) - set(available_baseline)
            logger.warning(
                f"{model_name}: {n_available}/{n_requested} baseline features available. "
                f"Missing: {sorted(missing)[:5]}..."
            )

        # Validate against strategy bounds
        if strict and n_available < strategy.min_features:
            raise ValueError(
                f"{model_name}: Only {n_available} features available, "
                f"but strategy requires >= {strategy.min_features}. "
                f"Missing baseline features: {sorted(set(baseline) - set(available_baseline))[:10]}"
            )

        if n_available > strategy.max_features:
            logger.warning(
                f"{model_name}: {n_available} features exceeds max {strategy.max_features}. "
                f"Consider feature selection/optimization."
            )

        result = ResolvedFeatureSet(
            model_name=model_name,
            feature_columns=available_baseline,
            baseline_requested=n_requested,
            baseline_available=n_available,
            optimized=False,
            included_families=strategy.preferred_families,
        )

        self._resolved_cache[cache_key] = result
        return result

    def get_features_for_ensemble(
        self,
        base_models: list[str],
    ) -> dict[str, ResolvedFeatureSet]:
        """
        Get resolved features for all models in an ensemble.

        Args:
            base_models: List of base model names

        Returns:
            Dict mapping model_name -> ResolvedFeatureSet
        """
        return {
            model: self.get_features_for_model(model)
            for model in base_models
        }

    def validate_feature_diversity(
        self,
        base_models: list[str],
        min_unique_ratio: float = 0.3,
    ) -> tuple[bool, list[str]]:
        """
        Validate that ensemble base models have diverse features.

        Heterogeneous ensembles benefit from feature diversity.
        This check warns if models share too many features.

        Args:
            base_models: List of base model names
            min_unique_ratio: Minimum ratio of unique features per model

        Returns:
            (is_diverse, list_of_warnings)
        """
        warnings = []

        # Get feature sets
        feature_sets = {}
        for model in base_models:
            resolved = self.get_features_for_model(model, strict=False)
            feature_sets[model] = set(resolved.feature_columns)

        # Check pairwise overlap
        models = list(feature_sets.keys())
        for i, m1 in enumerate(models):
            for m2 in models[i + 1:]:
                f1 = feature_sets[m1]
                f2 = feature_sets[m2]

                if not f1 or not f2:
                    continue

                overlap = f1 & f2
                overlap_ratio = len(overlap) / min(len(f1), len(f2))

                if overlap_ratio > (1 - min_unique_ratio):
                    warnings.append(
                        f"{m1} and {m2} share {overlap_ratio:.1%} features. "
                        f"Consider more diverse strategies."
                    )

        return len(warnings) == 0, warnings


def get_features_for_model(
    model_name: str,
    df: pd.DataFrame,
    strict: bool = True,
) -> list[str]:
    """
    Convenience function to get features for a model.

    Args:
        model_name: Name of the model
        df: DataFrame with available features
        strict: Raise error if insufficient features

    Returns:
        List of feature column names
    """
    manager = FeatureStrategyManager(df=df)
    resolved = manager.get_features_for_model(model_name, strict=strict)
    return resolved.feature_columns


__all__ = [
    "ResolvedFeatureSet",
    "FeatureStrategyManager",
    "get_features_for_model",
]
```

---

## 5.2 Integration with TrainerConfig

### File: `src/models/config/trainer_config.py`

**Add feature strategy integration**

```python
# ADD this method to TrainerConfig class

def get_feature_strategy(self) -> "ModelFeatureStrategy":
    """
    Get the feature strategy for this model.

    Returns:
        ModelFeatureStrategy for the configured model
    """
    from src.features.strategies import get_strategy_for_model
    return get_strategy_for_model(self.model_name)

def get_baseline_features(self) -> list[str]:
    """
    Get baseline feature list from strategy.

    Returns:
        List of baseline feature names
    """
    strategy = self.get_feature_strategy()
    return strategy.baseline_features.copy()

def resolve_features(
    self,
    available_features: list[str],
    strict: bool = True,
) -> list[str]:
    """
    Resolve features based on strategy and availability.

    Args:
        available_features: Features available in the data
        strict: Raise error if insufficient features

    Returns:
        List of resolved feature names
    """
    from src.features.strategy_manager import FeatureStrategyManager

    manager = FeatureStrategyManager(available_features=available_features)
    resolved = manager.get_features_for_model(self.model_name, strict=strict)

    # Store resolution result for logging
    self._resolved_features = resolved

    return resolved.feature_columns
```

---

## 5.3 Integration with Trainer

### File: `src/models/training/features.py`

**Modify TrainerFeaturesMixin to use strategies**

```python
# REPLACE the existing feature resolution logic with strategy-aware version

class TrainerFeaturesMixin:
    """Mixin for feature handling in Trainer."""

    def _get_feature_columns(
        self,
        df: pd.DataFrame,
        use_strategy: bool = True,
    ) -> list[str]:
        """
        Get feature columns for training.

        Args:
            df: DataFrame to extract features from
            use_strategy: Whether to use model's feature strategy

        Returns:
            List of feature column names
        """
        if use_strategy:
            return self._get_strategy_features(df)
        else:
            return self._get_all_features(df)

    def _get_strategy_features(self, df: pd.DataFrame) -> list[str]:
        """
        Get features based on model's declared strategy.

        This is the SNwH-aware feature selection that ensures
        each model gets features tailored to its inductive biases.

        Args:
            df: DataFrame with available features

        Returns:
            List of feature column names
        """
        from src.features.strategy_manager import FeatureStrategyManager

        manager = FeatureStrategyManager(df=df)

        try:
            resolved = manager.get_features_for_model(
                self.config.model_name,
                strict=True,
            )

            logger.info(
                f"Using strategy features for {self.config.model_name}: "
                f"{resolved.n_features} features "
                f"({resolved.baseline_available}/{resolved.baseline_requested} baseline available)"
            )

            return resolved.feature_columns

        except ValueError as e:
            logger.warning(
                f"Strategy feature resolution failed: {e}. "
                f"Falling back to all available features."
            )
            return self._get_all_features(df)

    def _get_all_features(self, df: pd.DataFrame) -> list[str]:
        """
        Get all available feature columns (legacy behavior).

        Args:
            df: DataFrame to extract features from

        Returns:
            List of all feature column names
        """
        from src.phase1.utils.constants import METADATA_COLUMNS
        from src.phase1.utils.feature_sets import _is_label_column

        return [
            col for col in df.columns
            if col not in METADATA_COLUMNS and not _is_label_column(col)
        ]

    def _validate_features_for_model(
        self,
        feature_columns: list[str],
    ) -> tuple[bool, list[str]]:
        """
        Validate feature set against model requirements.

        Args:
            feature_columns: List of feature column names

        Returns:
            (is_valid, list_of_warnings)
        """
        from src.features.strategies import get_strategy_for_model
        from src.contracts import get_model_contract

        warnings = []
        strategy = get_strategy_for_model(self.config.model_name)
        contract = get_model_contract(self.config.model_name)

        n_features = len(feature_columns)

        # Check bounds
        if n_features < strategy.min_features:
            warnings.append(
                f"Feature count ({n_features}) below minimum ({strategy.min_features})"
            )

        if n_features > strategy.max_features:
            warnings.append(
                f"Feature count ({n_features}) above maximum ({strategy.max_features})"
            )

        # Check contract bounds
        if n_features < contract.min_features:
            warnings.append(
                f"Feature count ({n_features}) below contract minimum ({contract.min_features})"
            )

        if n_features > contract.max_features:
            warnings.append(
                f"Feature count ({n_features}) above contract maximum ({contract.max_features})"
            )

        return len(warnings) == 0, warnings
```

---

## 5.4 Feature Optimization Integration

### New File: `src/features/optimization.py`

```python
"""
Feature Optimization - Optuna-based feature pruning from baseline to optimal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .strategies import get_strategy_for_model
from .strategy_manager import FeatureStrategyManager, ResolvedFeatureSet

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of feature optimization."""
    model_name: str
    original_features: list[str]
    optimized_features: list[str]
    n_original: int
    n_optimized: int
    improvement: float  # Metric improvement (higher is better)
    best_trial_score: float
    n_trials: int


class FeatureOptimizer:
    """
    Optuna-based feature optimizer.

    Prunes from baseline feature set to optimal subset using
    cross-validated metric optimization.

    Usage:
        optimizer = FeatureOptimizer(model_name="xgboost", n_trials=30)
        result = optimizer.optimize(X_train, y_train, X_val, y_val)
        optimized_features = result.optimized_features
    """

    def __init__(
        self,
        model_name: str,
        n_trials: int = 30,
        metric: str = "f1_weighted",
        min_features: int | None = None,
        random_seed: int = 42,
    ):
        """
        Initialize optimizer.

        Args:
            model_name: Name of the model to optimize for
            n_trials: Number of Optuna trials
            metric: Metric to optimize (f1_weighted, accuracy, etc.)
            min_features: Minimum features to keep (None = use strategy)
            random_seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.n_trials = n_trials
        self.metric = metric
        self.random_seed = random_seed

        strategy = get_strategy_for_model(model_name)
        self.min_features = min_features or strategy.min_features
        self.max_features = strategy.max_features

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str],
        sample_weights: np.ndarray | None = None,
    ) -> OptimizationResult:
        """
        Optimize feature set using Optuna.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: Names of features
            sample_weights: Optional sample weights

        Returns:
            OptimizationResult with optimized feature set
        """
        import optuna
        from src.models.registry import ModelRegistry

        n_features = len(feature_names)

        if n_features <= self.min_features:
            logger.info(
                f"Feature count ({n_features}) <= min ({self.min_features}). "
                f"Skipping optimization."
            )
            return OptimizationResult(
                model_name=self.model_name,
                original_features=feature_names,
                optimized_features=feature_names,
                n_original=n_features,
                n_optimized=n_features,
                improvement=0.0,
                best_trial_score=0.0,
                n_trials=0,
            )

        # Define objective
        def objective(trial: optuna.Trial) -> float:
            # Select subset of features
            n_select = trial.suggest_int(
                "n_features",
                self.min_features,
                min(n_features, self.max_features),
            )

            # Select which features to include
            selected_indices = []
            for i, name in enumerate(feature_names):
                if trial.suggest_categorical(f"include_{i}", [True, False]):
                    selected_indices.append(i)

            # Ensure we have at least min_features
            if len(selected_indices) < self.min_features:
                # Add random features to meet minimum
                remaining = [i for i in range(n_features) if i not in selected_indices]
                np.random.shuffle(remaining)
                selected_indices.extend(remaining[: self.min_features - len(selected_indices)])

            # Limit to max_features
            if len(selected_indices) > self.max_features:
                selected_indices = selected_indices[: self.max_features]

            if not selected_indices:
                return 0.0

            # Train and evaluate with selected features
            X_train_subset = X_train[:, selected_indices]
            X_val_subset = X_val[:, selected_indices]

            try:
                model = ModelRegistry.create(self.model_name)
                model.fit(
                    X_train_subset,
                    y_train,
                    X_val_subset,
                    y_val,
                    sample_weights=sample_weights,
                )
                predictions = model.predict(X_val_subset)

                # Compute metric
                if self.metric == "f1_weighted":
                    from sklearn.metrics import f1_score
                    score = f1_score(y_val, predictions.class_predictions, average="weighted")
                elif self.metric == "accuracy":
                    from sklearn.metrics import accuracy_score
                    score = accuracy_score(y_val, predictions.class_predictions)
                else:
                    raise ValueError(f"Unknown metric: {self.metric}")

                # Store selected features for this trial
                trial.set_user_attr("selected_indices", selected_indices)

                return score

            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0

        # Run optimization
        sampler = optuna.samplers.TPESampler(seed=self.random_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
            catch=(Exception,),
        )

        # Get best features
        best_trial = study.best_trial
        best_indices = best_trial.user_attrs.get("selected_indices", list(range(n_features)))
        optimized_features = [feature_names[i] for i in best_indices]

        # Compute baseline score for comparison
        baseline_model = ModelRegistry.create(self.model_name)
        baseline_model.fit(X_train, y_train, X_val, y_val, sample_weights=sample_weights)
        baseline_pred = baseline_model.predict(X_val)

        if self.metric == "f1_weighted":
            from sklearn.metrics import f1_score
            baseline_score = f1_score(y_val, baseline_pred.class_predictions, average="weighted")
        else:
            from sklearn.metrics import accuracy_score
            baseline_score = accuracy_score(y_val, baseline_pred.class_predictions)

        improvement = best_trial.value - baseline_score

        logger.info(
            f"Feature optimization complete: {n_features} -> {len(optimized_features)} features. "
            f"Score: {baseline_score:.4f} -> {best_trial.value:.4f} "
            f"(improvement: {improvement:+.4f})"
        )

        return OptimizationResult(
            model_name=self.model_name,
            original_features=feature_names,
            optimized_features=optimized_features,
            n_original=n_features,
            n_optimized=len(optimized_features),
            improvement=improvement,
            best_trial_score=best_trial.value,
            n_trials=self.n_trials,
        )


__all__ = [
    "OptimizationResult",
    "FeatureOptimizer",
]
```

---

## 5.5 Update Package Exports

### File: `src/features/__init__.py`

```python
"""
Features package - Feature strategies, management, and optimization.
"""

from .strategies import (
    ModelFeatureStrategy,
    MODEL_FEATURE_STRATEGIES,
    get_strategy_for_model,
    get_baseline_features,
)
from .strategy_manager import (
    ResolvedFeatureSet,
    FeatureStrategyManager,
    get_features_for_model,
)
from .optimization import (
    OptimizationResult,
    FeatureOptimizer,
)

__all__ = [
    # Strategies
    "ModelFeatureStrategy",
    "MODEL_FEATURE_STRATEGIES",
    "get_strategy_for_model",
    "get_baseline_features",
    # Strategy manager
    "ResolvedFeatureSet",
    "FeatureStrategyManager",
    "get_features_for_model",
    # Optimization
    "OptimizationResult",
    "FeatureOptimizer",
]
```

---

## Summary: Phase 5 Changes

| File | Type | Purpose |
|------|------|---------|
| `src/features/strategy_manager.py` | NEW | FeatureStrategyManager, ResolvedFeatureSet |
| `src/features/optimization.py` | NEW | FeatureOptimizer (Optuna-based) |
| `src/features/__init__.py` | MODIFY | Export new classes |
| `src/models/config/trainer_config.py` | MODIFY | Add feature strategy methods |
| `src/models/training/features.py` | MODIFY | Use strategy-based feature selection |

## Dependencies

- Phase 0-4 must be complete
- `src/features/strategies.py` (existing) provides MODEL_FEATURE_STRATEGIES
- Optuna required for optimization

## Data Flow After Integration

```
TrainerConfig.model_name
        |
        v
get_strategy_for_model() --> ModelFeatureStrategy
        |
        v
FeatureStrategyManager.get_features_for_model()
        |
        v
ResolvedFeatureSet (baseline features filtered to available)
        |
        v
[Optional] FeatureOptimizer.optimize()
        |
        v
Final feature list --> Adapter.transform()
```

## Usage Example

```python
from src.features import (
    FeatureStrategyManager,
    FeatureOptimizer,
    get_features_for_model,
)

# Simple usage
features = get_features_for_model("xgboost", df)

# With manager for multiple models
manager = FeatureStrategyManager(df=train_df)
xgb_features = manager.get_features_for_model("xgboost")
lstm_features = manager.get_features_for_model("lstm")

# Validate diversity for ensemble
is_diverse, warnings = manager.validate_feature_diversity(
    ["xgboost", "lstm", "patchtst"]
)

# With optimization
optimizer = FeatureOptimizer("xgboost", n_trials=30)
result = optimizer.optimize(X_train, y_train, X_val, y_val, feature_names)
optimized_features = result.optimized_features
```

## Backward Compatibility

- Existing code using `container.feature_columns` continues to work
- Strategy-based selection is opt-in via `use_strategy=True` parameter
- Falls back to all features if strategy resolution fails

## Complete SNwH Integration Flow

After Phase 5, the complete flow is:

```
1. Model declares contract (Phase 0)
2. TrainerConfig created from contract (Phase 1)
3. Adapter selected based on contract (Phase 2)
4. TimeframeCoordinator loads correct timeframe (Phase 3)
5. OOF aligned across models (Phase 4)
6. Features selected from strategy (Phase 5)
7. Training proceeds with model-specific data
```
