# ARCHIVED (Legacy)

This document is preserved for historical context. It contains spec-heavy content and may not match the current codebase.

For the current Phase 3 documentation, see `docs/phases/PHASE_3.md`.

---

# Phase 3: Cross-Validation and Out-of-Sample Predictions

## Current Status: IMPLEMENTED

**IMPLEMENTATION STATUS:**
- [x] PurgedKFold cross-validation - `src/cross_validation/purged_kfold.py`
- [x] Walk-forward feature selection - `src/cross_validation/feature_selector.py`
- [x] Optuna hyperparameter tuning - `src/cross_validation/cv_runner.py`
- [x] OOF prediction generation - `src/cross_validation/oof_generator.py`
- [x] Stacking dataset creation - `src/cross_validation/oof_generator.py`
- [x] Scripts (`scripts/run_cv.py`) - Complete with CLI

**DEPENDENCIES:**
- [x] Phase 1 (Data Pipeline) - **COMPLETE**
- [x] Phase 2 (Model Factory) - **COMPLETE** - Models registered in `src/models/`

**TESTS:**
- 125 tests in `tests/cross_validation/` - All passing
  - `test_purged_kfold.py` - Fold generation, purge/embargo zones, label leakage prevention
  - `test_feature_selector.py` - MDI/MDA importance, walk-forward selection, stable feature identification
  - `test_oof_generator.py` - OOF prediction generation, stacking dataset creation, prediction correlation
  - `test_cv_runner.py` - CVRunner initialization, fold metrics, result aggregation

**READY TO RUN:**
```bash
# Run CV for boosting models
python scripts/run_cv.py --models xgboost,lightgbm --horizons 5,10,15,20

# Run with hyperparameter tuning
python scripts/run_cv.py --models xgboost --horizons 20 --tune --n-trials 100

# Run all models
python scripts/run_cv.py --models all --horizons all
```

Phase 3 applies purged k-fold cross-validation to generate truly out-of-sample predictions. These predictions become training data for the Phase 4 ensemble. This phase also includes walk-forward feature selection and hyperparameter tuning to ensure robust, generalizable models.

---

## Overview

```
Trained Models (Phase 2)  -->  Purged K-Fold CV  -->  OOS Predictions  -->  Stacking Dataset
                                     |                      |
                              [Fold 1..5]           [Per-model probabilities]
                                     |
                         Walk-Forward Feature Selection
                                     |
                         Hyperparameter Tuning (Optuna)
```

**Key Objectives:**
1. Generate truly out-of-sample predictions for ensemble training
2. Measure model stability across different time periods
3. Perform walk-forward feature selection for robust feature sets
4. Tune hyperparameters with time-series aware validation
5. Analyze prediction correlations for ensemble diversity

---

## Why Purged Cross-Validation?

### The Problem with Standard K-Fold

Standard k-fold cross-validation breaks temporal order and causes information leakage:

```
Standard K-Fold (WRONG for time series):
|--Train--|--Test--|--Train--|--Train--|--Train--|
     ^         ^         ^
     |         |         |
  Uses future data to predict past = LEAKAGE
```

### Purged K-Fold Solution

Purged CV maintains temporal order and removes contaminated samples:

```
Purged K-Fold (CORRECT):
|----Train----|PURGE|--Test--|EMBARGO|----Train----|
                 ^              ^
                 |              |
     Removes samples whose      Buffer to break
     labels depend on test      serial correlation
```

### Mathematical Justification

From Lopez de Prado's "Advances in Financial Machine Learning":

1. **Label Leakage:** If sample at time t has label that depends on prices at t+k, and we train on t+k, we've leaked information
2. **Serial Correlation:** Financial returns exhibit autocorrelation; embargo breaks this dependency
3. **Purge Calculation:** Remove samples where `label_end_time > test_start_time - purge_bars`

---

## Purged K-Fold Implementation

### Core Algorithm

```python
from dataclasses import dataclass
from typing import List, Tuple, Iterator
import numpy as np
import pandas as pd


@dataclass
class PurgedKFoldConfig:
    """Configuration for purged k-fold cross-validation."""
    n_splits: int = 5
    purge_bars: int = 60          # 3x max horizon (20 bars)
    embargo_bars: int = 1440       # 5 trading days at 5-min
    min_train_size: float = 0.3    # Minimum training set fraction


class PurgedKFold:
    """
    Time-series cross-validation with purging and embargo.

    Implements the purged k-fold CV from Lopez de Prado (2018),
    which prevents information leakage in overlapping labels.
    """

    def __init__(self, config: PurgedKFoldConfig):
        self.config = config

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        groups: pd.Series = None,
        label_end_times: pd.Series = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for each fold.

        Args:
            X: Features DataFrame with DatetimeIndex
            y: Labels (optional)
            groups: Symbol groups for symbol isolation (optional)
            label_end_times: When each label's outcome is known (optional)

        Yields:
            Tuple of (train_indices, test_indices) for each fold
        """
        indices = np.arange(len(X))
        timestamps = X.index

        # Calculate fold boundaries
        fold_size = len(X) // self.config.n_splits
        min_train = int(len(X) * self.config.min_train_size)

        for fold_idx in range(self.config.n_splits):
            # Test fold boundaries
            test_start = fold_idx * fold_size
            test_end = (fold_idx + 1) * fold_size if fold_idx < self.config.n_splits - 1 else len(X)

            test_indices = indices[test_start:test_end]

            # Training indices: everything except test + purge + embargo
            train_mask = np.ones(len(X), dtype=bool)

            # Remove test period
            train_mask[test_start:test_end] = False

            # Apply purge before test
            purge_start = max(0, test_start - self.config.purge_bars)
            train_mask[purge_start:test_start] = False

            # Apply embargo after test
            embargo_end = min(len(X), test_end + self.config.embargo_bars)
            train_mask[test_end:embargo_end] = False

            # Additional purge for overlapping labels
            if label_end_times is not None:
                test_start_time = timestamps[test_start]
                for i in range(test_start):
                    if label_end_times.iloc[i] >= test_start_time:
                        train_mask[i] = False

            train_indices = indices[train_mask]

            # Ensure minimum training size
            if len(train_indices) < min_train:
                raise ValueError(
                    f"Fold {fold_idx}: Training set too small "
                    f"({len(train_indices)} < {min_train})"
                )

            yield train_indices, test_indices

    def get_fold_info(self, X: pd.DataFrame) -> List[dict]:
        """Get information about each fold for logging."""
        info = []
        for fold_idx, (train_idx, test_idx) in enumerate(self.split(X)):
            info.append({
                "fold": fold_idx,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "train_start": X.index[train_idx[0]],
                "train_end": X.index[train_idx[-1]],
                "test_start": X.index[test_idx[0]],
                "test_end": X.index[test_idx[-1]],
            })
        return info
```

### Fold Visualization

```
5-Fold Purged CV Example (3 years of data):

Fold 1: |===TRAIN===|PURGE|==TEST==|EMBARGO|===TRAIN===|===TRAIN===|===TRAIN===|
        |  Year 1   |     | H1 Y1  |       |   H2 Y1   |   Year 2   |   Year 3  |

Fold 2: |===TRAIN===|PURGE|==TEST==|EMBARGO|===TRAIN===|===TRAIN===|
        |   H1 Y1   |     | H2 Y1  |       |   Year 2   |   Year 3  |

Fold 3: |===TRAIN===|===TRAIN===|PURGE|==TEST==|EMBARGO|===TRAIN===|
        |   Year 1   |   H1 Y2   |     | H2 Y2  |       |   Year 3  |

Fold 4: |===TRAIN===|===TRAIN===|===TRAIN===|PURGE|==TEST==|EMBARGO|
        |   Year 1   |   Year 2   |   H1 Y3   |     | H2 Y3  |       |

Fold 5: |===TRAIN===|===TRAIN===|===TRAIN===|===TRAIN===|PURGE|==TEST==|
        |   Year 1   |   Year 2   |   H1 Y3   |   Q3 Y3   |     | Q4 Y3  |
```

---

## Walk-Forward Feature Selection

### Why Walk-Forward?

Standard feature selection uses full dataset, causing lookahead bias. Walk-forward selection trains only on past data:

```
Time -->
|----Select Features----|----Validate----|
|                       ^
|                       |
        Only use data before this point
```

### Implementation

```python
from typing import List, Dict, Set
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class WalkForwardFeatureSelector:
    """
    Feature selection with walk-forward methodology.

    Prevents lookahead bias by selecting features using only
    historical data at each point in time.
    """

    def __init__(
        self,
        n_features_to_select: int = 50,
        selection_method: str = "mda",  # mda, mdi, or hybrid
        n_estimators: int = 100,
        min_feature_frequency: float = 0.6
    ):
        self.n_features = n_features_to_select
        self.method = selection_method
        self.n_estimators = n_estimators
        self.min_frequency = min_feature_frequency

    def select_features_walkforward(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_splits: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, any]:
        """
        Perform walk-forward feature selection across CV folds.

        Args:
            X: Feature DataFrame
            y: Labels
            cv_splits: List of (train_idx, test_idx) tuples

        Returns:
            Dictionary with selected features and selection stats
        """
        feature_selections = []
        feature_importance_history = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]

            # Train feature selector
            importance = self._compute_importance(X_train, y_train)

            # Select top features
            top_features = importance.nlargest(self.n_features).index.tolist()
            feature_selections.append(set(top_features))

            feature_importance_history.append({
                "fold": fold_idx,
                "importance": importance.to_dict()
            })

        # Find stable features (appear in min_frequency of folds)
        all_features = set().union(*feature_selections)
        feature_counts = {f: sum(f in s for s in feature_selections) for f in all_features}

        stable_features = [
            f for f, count in feature_counts.items()
            if count >= len(cv_splits) * self.min_frequency
        ]

        return {
            "stable_features": stable_features,
            "feature_counts": feature_counts,
            "per_fold_selections": feature_selections,
            "importance_history": feature_importance_history
        }

    def _compute_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.Series:
        """Compute feature importance using specified method."""

        if self.method == "mdi":
            return self._mdi_importance(X, y)
        elif self.method == "mda":
            return self._mda_importance(X, y)
        else:  # hybrid
            mdi = self._mdi_importance(X, y)
            mda = self._mda_importance(X, y)
            return (mdi.rank() + mda.rank()) / 2

    def _mdi_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Mean Decrease in Impurity (built-in RF importance)."""
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=5,
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X, y)
        return pd.Series(rf.feature_importances_, index=X.columns)

    def _mda_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        Mean Decrease in Accuracy (permutation importance).

        More reliable than MDI for correlated features.
        Reference: Lopez de Prado (2018)
        """
        from sklearn.inspection import permutation_importance

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=5,
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X, y)

        # Use OOB samples for permutation importance
        result = permutation_importance(
            rf, X, y,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )

        return pd.Series(result.importances_mean, index=X.columns)
```

### Clustered Feature Importance

For highly correlated features, use clustered importance:

```python
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def clustered_mda_importance(
    X: pd.DataFrame,
    y: pd.Series,
    max_clusters: int = 20
) -> pd.Series:
    """
    MDA importance with feature clustering.

    Groups correlated features and computes importance per cluster,
    then distributes importance within cluster.

    Reference: Lopez de Prado (2018), Chapter 8
    """
    # Compute correlation matrix
    corr = X.corr()

    # Hierarchical clustering
    dist = 1 - corr.abs()
    linkage_matrix = linkage(squareform(dist), method='ward')
    clusters = fcluster(linkage_matrix, t=max_clusters, criterion='maxclust')

    # Map features to clusters
    feature_clusters = pd.Series(clusters, index=X.columns)

    # Compute importance per cluster
    cluster_importance = {}
    for cluster_id in np.unique(clusters):
        cluster_features = feature_clusters[feature_clusters == cluster_id].index
        X_cluster = X[cluster_features].mean(axis=1).to_frame('cluster_mean')

        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X_cluster, y)

        cluster_importance[cluster_id] = rf.feature_importances_[0]

    # Distribute importance within cluster
    feature_importance = {}
    for feature in X.columns:
        cluster_id = feature_clusters[feature]
        n_features_in_cluster = (feature_clusters == cluster_id).sum()
        feature_importance[feature] = cluster_importance[cluster_id] / n_features_in_cluster

    return pd.Series(feature_importance)
```

---

## Hyperparameter Tuning with Optuna

### Time-Series Aware Tuning

```python
import optuna
from optuna.samplers import TPESampler
from typing import Callable, Dict, Any


class TimeSeriesOptunaTuner:
    """
    Hyperparameter tuning with purged cross-validation.

    Uses Optuna's TPE sampler with time-series aware objective.
    """

    def __init__(
        self,
        model_class: type,
        cv: PurgedKFold,
        n_trials: int = 100,
        direction: str = "maximize",
        metric: str = "sharpe"
    ):
        self.model_class = model_class
        self.cv = cv
        self.n_trials = n_trials
        self.direction = direction
        self.metric = metric

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series = None,
        param_space: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run hyperparameter tuning.

        Args:
            X: Features
            y: Labels
            sample_weights: Quality weights (optional)
            param_space: Search space definition (optional, uses defaults)

        Returns:
            Best parameters and study results
        """
        study = optuna.create_study(
            direction=self.direction,
            sampler=TPESampler(seed=42)
        )

        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            params = self._sample_params(trial, param_space)

            # Cross-validation
            scores = []
            for train_idx, val_idx in self.cv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                w_train = None
                if sample_weights is not None:
                    w_train = sample_weights.iloc[train_idx]

                # Train and evaluate
                model = self.model_class(**params)
                model.fit(X_train, y_train, sample_weight=w_train)
                score = self._evaluate(model, X_val, y_val)
                scores.append(score)

            # Return mean score (with penalty for high variance)
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            # Penalize high variance (stability matters)
            penalty = 0.1 * std_score
            return mean_score - penalty

        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "study": study
        }

    def _sample_params(
        self,
        trial: optuna.Trial,
        param_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sample parameters from search space."""
        if param_space is None:
            param_space = self._default_param_space()

        params = {}
        for name, spec in param_space.items():
            if spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif spec["type"] == "float":
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"],
                    log=spec.get("log", False)
                )
            elif spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])

        return params

    def _default_param_space(self) -> Dict[str, Any]:
        """Default search space for XGBoost."""
        return {
            "n_estimators": {"type": "int", "low": 100, "high": 500},
            "max_depth": {"type": "int", "low": 3, "high": 10},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
            "reg_alpha": {"type": "float", "low": 0.0, "high": 10.0},
            "reg_lambda": {"type": "float", "low": 0.0, "high": 10.0},
        }
```

### Model-Specific Search Spaces

```python
PARAM_SPACES = {
    # --- BOOSTING MODELS ---
    "xgboost": {
        "n_estimators": {"type": "int", "low": 100, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
        "min_child_weight": {"type": "int", "low": 1, "high": 20},
        "gamma": {"type": "float", "low": 0.0, "high": 5.0},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 10.0},
        "reg_lambda": {"type": "float", "low": 0.0, "high": 10.0},
    },

    "lightgbm": {
        "n_estimators": {"type": "int", "low": 100, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "num_leaves": {"type": "int", "low": 20, "high": 100},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
        "min_child_samples": {"type": "int", "low": 5, "high": 50},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 10.0},
        "reg_lambda": {"type": "float", "low": 0.0, "high": 10.0},
    },

    "catboost": {
        "iterations": {"type": "int", "low": 100, "high": 500},
        "depth": {"type": "int", "low": 4, "high": 10},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 10.0},
        "bagging_temperature": {"type": "float", "low": 0.0, "high": 1.0},
        "random_strength": {"type": "float", "low": 0.0, "high": 1.0},
        "border_count": {"type": "int", "low": 32, "high": 255},
    },

    # --- RECURRENT NEURAL NETWORKS ---
    "lstm": {
        "hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
        "num_layers": {"type": "int", "low": 1, "high": 3},
        "dropout": {"type": "float", "low": 0.1, "high": 0.5},
        "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "batch_size": {"type": "categorical", "choices": [64, 128, 256, 512]},
        "sequence_length": {"type": "categorical", "choices": [30, 60, 90, 120]},
    },

    "gru": {
        "hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
        "num_layers": {"type": "int", "low": 1, "high": 3},
        "dropout": {"type": "float", "low": 0.1, "high": 0.5},
        "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "batch_size": {"type": "categorical", "choices": [64, 128, 256, 512]},
        "sequence_length": {"type": "categorical", "choices": [30, 60, 90, 120]},
        "bidirectional": {"type": "categorical", "choices": [True, False]},
    },

    # --- TEMPORAL CONVOLUTIONAL NETWORKS ---
    "tcn": {
        "num_channels": {"type": "categorical", "choices": [[32, 64], [64, 128], [64, 128, 256]]},
        "kernel_size": {"type": "int", "low": 2, "high": 7},
        "dropout": {"type": "float", "low": 0.1, "high": 0.4},
        "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "batch_size": {"type": "categorical", "choices": [64, 128, 256]},
        "sequence_length": {"type": "categorical", "choices": [60, 120, 180]},
        "dilation_base": {"type": "int", "low": 2, "high": 3},
    },

    # --- TRANSFORMER MODELS ---
    "patchtst": {
        "patch_len": {"type": "categorical", "choices": [8, 16, 24]},
        "stride": {"type": "categorical", "choices": [4, 8, 12]},
        "d_model": {"type": "categorical", "choices": [64, 128, 256]},
        "n_heads": {"type": "categorical", "choices": [4, 8]},
        "n_layers": {"type": "int", "low": 2, "high": 4},
        "dropout": {"type": "float", "low": 0.1, "high": 0.3},
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
        "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
        "context_length": {"type": "categorical", "choices": [96, 192, 336]},
    },

    "itransformer": {
        "d_model": {"type": "categorical", "choices": [64, 128, 256]},
        "n_heads": {"type": "categorical", "choices": [4, 8]},
        "e_layers": {"type": "int", "low": 2, "high": 4},
        "d_ff": {"type": "categorical", "choices": [128, 256, 512]},
        "dropout": {"type": "float", "low": 0.1, "high": 0.3},
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
        "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
        "seq_len": {"type": "categorical", "choices": [96, 192]},
    },

    "tft": {  # Temporal Fusion Transformer
        "hidden_size": {"type": "categorical", "choices": [32, 64, 128]},
        "attention_head_size": {"type": "categorical", "choices": [1, 2, 4]},
        "dropout": {"type": "float", "low": 0.1, "high": 0.4},
        "hidden_continuous_size": {"type": "categorical", "choices": [8, 16, 32]},
        "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
        "max_encoder_length": {"type": "categorical", "choices": [60, 120, 180]},
    },

    # --- CLASSICAL MODELS ---
    "random_forest": {
        "n_estimators": {"type": "int", "low": 100, "high": 500},
        "max_depth": {"type": "int", "low": 5, "high": 20},
        "min_samples_split": {"type": "int", "low": 5, "high": 50},
        "min_samples_leaf": {"type": "int", "low": 2, "high": 20},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2", 0.5]},
    },
}
```

### Walk-Forward CV for Different Model Families

Different model families require different CV strategies due to training costs and stationarity assumptions:

```python
CV_STRATEGIES = {
    "boosting": {
        # Fast training allows more folds
        "n_splits": 5,
        "tuning_trials": 100,
        "description": "Full purged k-fold, fast retraining per fold"
    },
    "neural_rnn": {
        # Moderate training time
        "n_splits": 3,
        "tuning_trials": 50,
        "description": "Fewer folds, early stopping within each fold"
    },
    "transformer": {
        # Expensive training
        "n_splits": 3,
        "tuning_trials": 30,
        "description": "Minimal folds, transfer learning between folds"
    },
}


class ModelAwareCV:
    """Cross-validation strategy adapted to model training costs."""

    def __init__(self, model_family: str, base_cv: PurgedKFold):
        self.strategy = CV_STRATEGIES[model_family]
        self.base_cv = base_cv

    def get_cv_splits(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Return appropriate number of splits for model family."""
        n_splits = self.strategy["n_splits"]

        # Adjust base CV if needed
        if n_splits != self.base_cv.config.n_splits:
            adjusted_config = PurgedKFoldConfig(
                n_splits=n_splits,
                purge_bars=self.base_cv.config.purge_bars,
                embargo_bars=self.base_cv.config.embargo_bars
            )
            cv = PurgedKFold(adjusted_config)
        else:
            cv = self.base_cv

        return cv.split(X)

    def get_tuning_trials(self) -> int:
        """Return appropriate number of Optuna trials."""
        return self.strategy["tuning_trials"]
```

---

## Out-of-Sample Prediction Generation

### CV Runner

```python
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import pandas as pd


@dataclass
class CVResult:
    """Results from cross-validation run."""
    model_name: str
    horizon: int
    fold_metrics: List[Dict]
    oos_predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    tuned_params: Dict[str, Any]


class CrossValidationRunner:
    """
    Orchestrates cross-validation for generating OOS predictions.
    """

    def __init__(
        self,
        cv: PurgedKFold,
        models: List[str],
        horizons: List[int],
        tune_hyperparams: bool = True,
        select_features: bool = True
    ):
        self.cv = cv
        self.models = models
        self.horizons = horizons
        self.tune_hyperparams = tune_hyperparams
        self.select_features = select_features

    def run(
        self,
        container: "TimeSeriesDataContainer"
    ) -> Dict[str, CVResult]:
        """
        Run cross-validation for all models and horizons.

        Returns:
            Dictionary mapping (model, horizon) to CVResult
        """
        results = {}

        for model_name in self.models:
            for horizon in self.horizons:
                print(f"Running CV for {model_name} on H{horizon}...")

                result = self._run_single_cv(
                    container,
                    model_name,
                    horizon
                )
                results[(model_name, horizon)] = result

        return results

    def _run_single_cv(
        self,
        container: "TimeSeriesDataContainer",
        model_name: str,
        horizon: int
    ) -> CVResult:
        """Run CV for single model/horizon combination."""

        # Get data for this horizon
        X, y, weights = container.get_sklearn_arrays("train", horizon=horizon)

        # Feature selection (if enabled)
        if self.select_features:
            selector = WalkForwardFeatureSelector(n_features_to_select=50)
            selection_result = selector.select_features_walkforward(
                X, y, list(self.cv.split(X))
            )
            selected_features = selection_result["stable_features"]
            X = X[selected_features]
        else:
            selected_features = X.columns.tolist()

        # Hyperparameter tuning (if enabled)
        if self.tune_hyperparams:
            tuner = TimeSeriesOptunaTuner(
                model_class=ModelRegistry.get_class(model_name),
                cv=self.cv,
                n_trials=50
            )
            tuning_result = tuner.tune(X, y, weights)
            best_params = tuning_result["best_params"]
        else:
            best_params = ModelRegistry.get_default_config(model_name)

        # Generate OOS predictions
        oos_predictions = []
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(self.cv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            w_train = weights.iloc[train_idx] if weights is not None else None

            # Train model
            model = ModelRegistry.create(model_name, config=best_params)
            model.fit(X_train, y_train, sample_weights=w_train)

            # Generate predictions
            predictions = model.predict(X_test)

            # Store OOS predictions
            fold_preds = pd.DataFrame({
                "datetime": X_test.index,
                "fold": fold_idx,
                "y_true": y_test.values,
                "y_pred": predictions.class_predictions,
                "prob_short": predictions.class_probabilities[:, 0],
                "prob_neutral": predictions.class_probabilities[:, 1],
                "prob_long": predictions.class_probabilities[:, 2],
                "confidence": predictions.confidence
            })
            oos_predictions.append(fold_preds)

            # Compute fold metrics
            metrics = self._compute_fold_metrics(y_test, predictions)
            metrics["fold"] = fold_idx
            fold_metrics.append(metrics)

        # Aggregate results
        oos_df = pd.concat(oos_predictions, ignore_index=True)
        oos_df = oos_df.sort_values("datetime")

        return CVResult(
            model_name=model_name,
            horizon=horizon,
            fold_metrics=fold_metrics,
            oos_predictions=oos_df,
            feature_importance=pd.DataFrame(),  # Populate if needed
            tuned_params=best_params
        )
```

---

## How CV Generates OOF Predictions for Stacking

### The OOF Prediction Process

Out-of-Fold (OOF) predictions are the foundation of stacking ensembles. Each sample in the training set gets a prediction from a model that **never saw that sample during training**:

```
Dataset: [Sample 1, Sample 2, Sample 3, Sample 4, Sample 5, ...]

Fold 1: Train on [2,3,4,5] -> Predict [1]     -> OOF_pred[1]
Fold 2: Train on [1,3,4,5] -> Predict [2]     -> OOF_pred[2]
Fold 3: Train on [1,2,4,5] -> Predict [3]     -> OOF_pred[3]
Fold 4: Train on [1,2,3,5] -> Predict [4]     -> OOF_pred[4]
Fold 5: Train on [1,2,3,4] -> Predict [5]     -> OOF_pred[5]

Result: Every sample has a truly out-of-sample prediction
```

### Why OOF Predictions Matter for Stacking

**Problem with in-sample predictions:**
If we train a meta-learner on predictions from models trained on the same data:
- Base models will be overconfident on training data
- Meta-learner learns to trust overfitted predictions
- Catastrophic overfitting in production

**Solution with OOF predictions:**
- Each prediction comes from a model that never saw that sample
- Meta-learner sees realistic prediction quality
- Better generalization to new data

### OOF Generation Pipeline

```python
class OOFGenerator:
    """
    Generate out-of-fold predictions for stacking.

    Each sample gets a prediction from a model trained
    without seeing that sample.
    """

    def __init__(
        self,
        cv: PurgedKFold,
        model_registry: "ModelRegistry"
    ):
        self.cv = cv
        self.registry = model_registry

    def generate_oof_predictions(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_configs: Dict[str, Dict],
        sample_weights: pd.Series = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate OOF predictions for all models.

        Args:
            X: Feature DataFrame
            y: Labels
            model_configs: {model_name: hyperparameters}
            sample_weights: Optional quality weights

        Returns:
            {model_name: DataFrame with OOF predictions}
        """
        oof_results = {}

        for model_name, config in model_configs.items():
            print(f"Generating OOF for {model_name}...")

            # Initialize OOF storage
            n_samples = len(X)
            n_classes = 3  # short, neutral, long
            oof_probs = np.full((n_samples, n_classes), np.nan)
            oof_preds = np.full(n_samples, np.nan)

            # Generate predictions fold by fold
            for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X)):
                print(f"  Fold {fold_idx + 1}...")

                # Extract fold data
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train = y.iloc[train_idx]
                w_train = sample_weights.iloc[train_idx] if sample_weights is not None else None

                # Train model
                model = self.registry.create(model_name, config)
                model.fit(X_train, y_train, sample_weights=w_train)

                # Generate predictions for validation fold
                probs = model.predict_proba(X_val)
                preds = probs.argmax(axis=1) - 1  # -1, 0, 1

                # Store OOF predictions
                oof_probs[val_idx] = probs
                oof_preds[val_idx] = preds

            # Validate no NaN remaining
            if np.isnan(oof_probs).any():
                raise ValueError(f"Some samples missing OOF predictions for {model_name}")

            # Create result DataFrame
            oof_df = pd.DataFrame({
                "datetime": X.index,
                f"{model_name}_prob_short": oof_probs[:, 0],
                f"{model_name}_prob_neutral": oof_probs[:, 1],
                f"{model_name}_prob_long": oof_probs[:, 2],
                f"{model_name}_pred": oof_preds,
                f"{model_name}_confidence": oof_probs.max(axis=1),
            })
            oof_df.set_index("datetime", inplace=True)

            oof_results[model_name] = oof_df

        return oof_results

    def validate_oof_coverage(
        self,
        oof_predictions: Dict[str, pd.DataFrame],
        original_index: pd.Index
    ) -> Dict:
        """Validate that OOF predictions cover all samples."""
        validation = {"passed": True, "issues": []}

        for model_name, oof_df in oof_predictions.items():
            missing = set(original_index) - set(oof_df.index)
            if missing:
                validation["passed"] = False
                validation["issues"].append({
                    "model": model_name,
                    "missing_samples": len(missing)
                })

        return validation
```

### Feature Selection Integration with CV

Feature selection must also be performed in a walk-forward manner to avoid lookahead:

```python
class CVIntegratedFeatureSelector:
    """
    Integrate feature selection with CV to prevent lookahead.

    Strategy:
    1. For each CV fold, select features using ONLY training data
    2. Track which features are stable across folds
    3. Final feature set = features selected in >= min_frequency folds
    """

    def __init__(
        self,
        cv: PurgedKFold,
        n_features: int = 50,
        min_frequency: float = 0.6,
        method: str = "mda"  # mda, mdi, or boruta
    ):
        self.cv = cv
        self.n_features = n_features
        self.min_frequency = min_frequency
        self.method = method

    def select_and_generate_oof(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        model_config: Dict
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Select features and generate OOF in single pass.

        Returns:
            Tuple of (stable_features, oof_predictions)
        """
        feature_selections = []
        oof_probs = np.full((len(X), 3), np.nan)

        for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X)):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]

            # Select features on training data only
            fold_features = self._select_features_single_fold(X_train, y_train)
            feature_selections.append(set(fold_features))

            # Train model on selected features
            X_train_selected = X_train[fold_features]
            X_val_selected = X.iloc[val_idx][fold_features]

            model = ModelRegistry.create(model_name, model_config)
            model.fit(X_train_selected, y_train)

            # Generate OOF predictions
            probs = model.predict_proba(X_val_selected)
            oof_probs[val_idx] = probs

        # Identify stable features
        all_features = set().union(*feature_selections)
        feature_counts = {
            f: sum(f in s for s in feature_selections)
            for f in all_features
        }
        stable_features = [
            f for f, count in feature_counts.items()
            if count >= len(feature_selections) * self.min_frequency
        ]

        # Build OOF DataFrame
        oof_df = pd.DataFrame(
            oof_probs,
            index=X.index,
            columns=["prob_short", "prob_neutral", "prob_long"]
        )

        return stable_features, oof_df

    def _select_features_single_fold(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[str]:
        """Select top N features for a single fold."""
        if self.method == "mda":
            importance = self._mda_importance(X, y)
        elif self.method == "mdi":
            importance = self._mdi_importance(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return importance.nlargest(self.n_features).index.tolist()
```

---

## Stacking Dataset Generation

### Building the Ensemble Training Data

```python
def build_stacking_dataset(
    cv_results: Dict[Tuple[str, int], CVResult],
    horizons: List[int]
) -> Dict[int, pd.DataFrame]:
    """
    Build stacking dataset from OOS predictions.

    For each horizon, creates a DataFrame with:
    - datetime, symbol (metadata)
    - model1_prob_short, model1_prob_neutral, model1_prob_long
    - model2_prob_short, model2_prob_neutral, model2_prob_long
    - ...
    - y_true (label)

    Args:
        cv_results: Results from CrossValidationRunner
        horizons: List of horizons to process

    Returns:
        Dictionary mapping horizon to stacking DataFrame
    """
    stacking_datasets = {}

    for horizon in horizons:
        # Collect predictions for this horizon
        model_predictions = {}

        for (model_name, h), result in cv_results.items():
            if h != horizon:
                continue

            preds = result.oos_predictions.set_index("datetime")
            model_predictions[model_name] = preds[[
                "prob_short", "prob_neutral", "prob_long", "y_true"
            ]].rename(columns={
                "prob_short": f"{model_name}_prob_short",
                "prob_neutral": f"{model_name}_prob_neutral",
                "prob_long": f"{model_name}_prob_long",
            })

        # Merge all model predictions
        stacking_df = None
        for model_name, preds in model_predictions.items():
            if stacking_df is None:
                stacking_df = preds
            else:
                # Use y_true from first model only
                preds = preds.drop(columns=["y_true"], errors="ignore")
                stacking_df = stacking_df.join(preds, how="inner")

        # Add extended features for meta-learner
        stacking_df = add_stacking_features(stacking_df, model_predictions.keys())

        stacking_datasets[horizon] = stacking_df

    return stacking_datasets


def add_stacking_features(
    df: pd.DataFrame,
    model_names: List[str]
) -> pd.DataFrame:
    """Add derived features for meta-learner."""

    # Model confidence (max probability)
    for model in model_names:
        prob_cols = [f"{model}_prob_short", f"{model}_prob_neutral", f"{model}_prob_long"]
        df[f"{model}_confidence"] = df[prob_cols].max(axis=1)

    # Model predictions (argmax)
    for model in model_names:
        prob_cols = [f"{model}_prob_short", f"{model}_prob_neutral", f"{model}_prob_long"]
        df[f"{model}_pred"] = df[prob_cols].values.argmax(axis=1) - 1  # -1, 0, 1

    # Agreement features
    pred_cols = [f"{model}_pred" for model in model_names]
    df["models_agree"] = (df[pred_cols].nunique(axis=1) == 1).astype(int)
    df["agreement_count"] = df[pred_cols].apply(lambda x: x.value_counts().max(), axis=1)

    # Average confidence
    conf_cols = [f"{model}_confidence" for model in model_names]
    df["avg_confidence"] = df[conf_cols].mean(axis=1)

    # Prediction entropy (uncertainty)
    for model in model_names:
        prob_cols = [f"{model}_prob_short", f"{model}_prob_neutral", f"{model}_prob_long"]
        probs = df[prob_cols].values
        df[f"{model}_entropy"] = -np.sum(probs * np.log(probs + 1e-10), axis=1)

    return df
```

---

## Stability Analysis

### Fold-to-Fold Consistency

```python
def analyze_cv_stability(
    cv_results: Dict[Tuple[str, int], CVResult]
) -> pd.DataFrame:
    """
    Analyze stability of models across CV folds.

    Returns DataFrame with stability metrics per model/horizon.
    """
    stability_data = []

    for (model_name, horizon), result in cv_results.items():
        metrics_df = pd.DataFrame(result.fold_metrics)

        for metric in ["accuracy", "f1", "sharpe", "win_rate"]:
            if metric not in metrics_df.columns:
                continue

            values = metrics_df[metric].values
            mean_val = np.mean(values)
            std_val = np.std(values)
            cv = std_val / mean_val if mean_val > 0 else np.inf

            stability_data.append({
                "model": model_name,
                "horizon": horizon,
                "metric": metric,
                "mean": mean_val,
                "std": std_val,
                "cv": cv,
                "min": np.min(values),
                "max": np.max(values),
                "stability_grade": grade_stability(cv)
            })

    return pd.DataFrame(stability_data)


def grade_stability(cv: float) -> str:
    """Grade stability based on coefficient of variation."""
    if cv < 0.15:
        return "Excellent"
    elif cv < 0.25:
        return "Good"
    elif cv < 0.40:
        return "Acceptable"
    elif cv < 0.60:
        return "Poor"
    else:
        return "Unstable"
```

### Prediction Correlation Analysis

```python
def analyze_prediction_correlation(
    stacking_df: pd.DataFrame,
    model_names: List[str]
) -> pd.DataFrame:
    """
    Analyze correlation between model predictions.

    Low correlation = good diversity for ensemble.
    """
    # Get predicted class for each model
    pred_cols = [f"{model}_pred" for model in model_names]
    pred_df = stacking_df[pred_cols]

    # Compute correlation matrix
    corr_matrix = pred_df.corr()

    # Summarize
    summary = []
    for i, model_i in enumerate(model_names):
        for j, model_j in enumerate(model_names):
            if i < j:
                corr = corr_matrix.loc[f"{model_i}_pred", f"{model_j}_pred"]
                summary.append({
                    "model_1": model_i,
                    "model_2": model_j,
                    "correlation": corr,
                    "diversity_grade": grade_diversity(corr)
                })

    return pd.DataFrame(summary)


def grade_diversity(corr: float) -> str:
    """Grade ensemble diversity based on prediction correlation."""
    if corr < 0.3:
        return "Excellent (highly diverse)"
    elif corr < 0.5:
        return "Good"
    elif corr < 0.7:
        return "Moderate"
    elif corr < 0.85:
        return "Low"
    else:
        return "Poor (models too similar)"
```

---

## Output Structure

### Directory Layout

```
data/stacking/
|
+-- h5/
|   +-- stacking_dataset.parquet     # OOS predictions for ensemble
|   +-- cv_metrics.json              # Per-fold metrics
|   +-- stability_analysis.json      # Stability grades
|
+-- h10/
|   +-- (same structure)
|
+-- h15/
|   +-- (same structure)
|
+-- h20/
|   +-- (same structure)
|
+-- tuned_params/
|   +-- xgboost_h5.json
|   +-- xgboost_h10.json
|   +-- lightgbm_h5.json
|   +-- lstm_h5.json
|   +-- ...
|
+-- feature_selection/
|   +-- stable_features_h5.json
|   +-- selection_history_h5.json
|   +-- ...

reports/phase3/
|
+-- cv_summary.html                  # Interactive CV report
+-- fold_metrics.csv                 # All fold metrics
+-- stability_analysis.csv           # Stability analysis
+-- prediction_correlation.png       # Model agreement heatmap
+-- feature_importance.png           # Walk-forward feature importance
```

### Stacking Dataset Schema

```json
{
  "columns": {
    "datetime": "Timestamp of prediction",
    "symbol": "MES or MGC",

    "xgboost_prob_short": "XGBoost P(short)",
    "xgboost_prob_neutral": "XGBoost P(neutral)",
    "xgboost_prob_long": "XGBoost P(long)",
    "xgboost_confidence": "XGBoost max probability",
    "xgboost_pred": "XGBoost predicted class (-1, 0, 1)",
    "xgboost_entropy": "XGBoost prediction uncertainty",

    "lstm_prob_short": "LSTM P(short)",
    "...": "...",

    "models_agree": "1 if all models predict same class",
    "agreement_count": "Number of models predicting mode class",
    "avg_confidence": "Average confidence across models",

    "y_true": "True label (-1, 0, 1)"
  },
  "samples": 50000,
  "models": ["xgboost", "lightgbm", "lstm"],
  "horizon": 20
}
```

---

## Computational Requirements

### Estimated Time

| Component | Time per Model | Total (3 models, 4 horizons) |
|-----------|---------------|------------------------------|
| Feature Selection | 15-30 min | 3-6 hours |
| Hyperparameter Tuning | 2-4 hours | 24-48 hours |
| CV Training (5 folds) | 1-3 hours | 12-36 hours |
| **Total** | **3-7 hours** | **39-90 hours** |

### Parallelization Strategy

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


def run_parallel_cv(
    models: List[str],
    horizons: List[int],
    container: "TimeSeriesDataContainer",
    n_workers: int = None
) -> Dict:
    """Run CV in parallel across models and horizons."""

    if n_workers is None:
        n_workers = min(mp.cpu_count() - 1, len(models) * len(horizons))

    tasks = [(model, horizon) for model in models for horizon in horizons]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(run_single_cv, model, horizon, container): (model, horizon)
            for model, horizon in tasks
        }

        results = {}
        for future in futures:
            model, horizon = futures[future]
            results[(model, horizon)] = future.result()

    return results
```

---

## Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Folds completed | 5 per model per horizon | Count |
| OOS coverage | 100% of validation set has OOS prediction | Coverage check |
| Per-fold Sharpe CV | < 0.5 | Stability analysis |
| Prediction correlation | 0.3 - 0.7 between models | Correlation matrix |
| Feature selection stability | > 60% features stable across folds | Selection frequency |
| Tuning convergence | Optuna study converged | Study analysis |

---

## Usage Examples

**NOTE: These scripts do not currently exist. Phase 3 requires Phase 2 to be completed first.**

```bash
# PLANNED (not yet implemented):
# python scripts/run_cv.py --models xgboost,lightgbm,lstm --horizons 5,10,15,20

# CURRENT STATUS:
# Phase 3 is PLANNED and cannot be run until Phase 2 (Model Factory) is complete.
#
# Implementation order:
# 1. Complete Phase 2: Train base models (XGBoost, LightGBM, LSTM, etc.)
# 2. Implement PurgedKFold cross-validation
# 3. Generate out-of-fold predictions for stacking
# 4. Proceed to Phase 4 for ensemble training

# Current implementation:
# Only Phase 1 data pipeline is implemented:
./pipeline run --symbols MES,MGC
```

---

## Next Step

Phase 3 stacking dataset feeds into Phase 4 (Ensemble Meta-Learner) where model predictions are combined using logistic regression, XGBoost, or other meta-learners to produce final ensemble predictions.
