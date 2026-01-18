# Hyperparameter Optimization Guide

**Purpose:** Comprehensive guide for hyperparameter optimization using Optuna across the 16-stage ML pipeline
**Audience:** ML engineers, quant analysts
**Last Updated:** 2026-01-18

---

## Table of Contents

1. [Optimization Philosophy](#optimization-philosophy)
2. [Stage 7: Triple Barrier Optimization with Optuna](#stage-7-triple-barrier-optimization-with-optuna)
3. [Stage 8: Feature Selection Optimization with Optuna](#stage-8-feature-selection-optimization-with-optuna)
4. [Stage 9: Feature Pruning Optimization with Optuna](#stage-9-feature-pruning-optimization-with-optuna)
5. [Stage 13: Per-Model Hyperparameter Optimization](#stage-13-per-model-hyperparameter-optimization)
6. [Search Space Design](#search-space-design)
7. [Objective Functions](#objective-functions)
8. [Pruning Strategies](#pruning-strategies)
9. [Multi-Objective Optimization](#multi-objective-optimization)
10. [Integration with Cross-Validation](#integration-with-cross-validation)
11. [Computational Budget](#computational-budget)
12. [Adding New Parameters](#adding-new-parameters)

---

## Optimization Philosophy

### Optuna-Unified Optimization Pipeline

The ML Factory uses Optuna for all optimization stages, providing a consistent and efficient approach across the entire pipeline. Four distinct Optuna optimization stages are integrated into the 16-stage pipeline:

| Pipeline Stage | Optuna Use | Trials | Parameters |
|----------------|------------|--------|------------|
| **Stage 7** | Label Optimization | 100 | upper_mult, lower_mult, horizon, atr_period |
| **Stage 8** | Feature Selection | 100 | Binary include/exclude per feature |
| **Stage 9** | Feature Pruning | 50 | Importance-based removal thresholds |
| **Stage 13** | Hyperparameter Optimization | 100 per model | Model-specific hyperparameters |

### When to Use Different Search Strategies

| Method | Use Case | Pros | Cons | Typical Use |
|--------|----------|------|------|-------------|
| **Optuna (TPE)** | All pipeline stages | Fast, adaptive, pruning, unified | May get stuck in local optima | All 4 optimization stages |
| **Grid Search** | Small discrete spaces | Exhaustive, deterministic | Exponential growth | Testing 2-3 discrete options |
| **Random Search** | Baseline | Fast, simple | Not adaptive | Quick baseline before Optuna |

### Key Principles

1. **Four-stage Optuna optimization:**
   - **Stage 7:** Optimize triple-barrier label parameters (100 trials)
   - **Stage 8:** Optimize feature selection via binary include/exclude (100 trials)
   - **Stage 9:** Optimize feature pruning via importance thresholds (50 trials)
   - **Stage 13:** Optimize model hyperparameters (100 trials per model)

2. **Nested cross-validation:**
   - **Outer CV:** Evaluate generalization (5 folds)
   - **Inner CV:** Hyperparameter tuning (3 folds)
   - Prevents overfitting on validation set

3. **Symbol-specific optimization:**
   - MES (micro E-mini S&P) has different volatility than MGC (micro Gold)
   - Optimize label parameters separately per symbol

4. **Transaction cost penalties:**
   - Objective function includes trading costs (slippage + commission)
   - Prevents over-trading

5. **Time budget allocation:**
   - GA (label params): 1-2 hours
   - Optuna (boosting): 100 trials × 2 min = 3-4 hours
   - Optuna (neural): 50 trials × 10 min = 8-10 hours

---

## Stage 7: Triple Barrier Optimization with Optuna

### Overview

**Pipeline Stage:** 7 of 16
**Goal:** Optimize triple-barrier labeling parameters to maximize risk-adjusted returns.
**Trials:** 100 Optuna trials

**Parameters to optimize:**
- `upper_mult`: Upper barrier multiplier (profit target as ATR multiple)
- `lower_mult`: Lower barrier multiplier (stop loss as ATR multiple)
- `horizon`: Maximum holding period in bars
- `atr_period`: ATR lookback period for volatility normalization

**Objective:** Maximize Sharpe ratio with transaction cost penalty and class balance.

### Search Space Definition

```python
# src/phase1/stages/label_optimize/optuna_optimizer.py
import optuna

def label_param_search_space(trial: optuna.Trial, symbol: str) -> dict:
    """
    Optuna search space for triple-barrier label parameters.

    Args:
        trial: Optuna trial object
        symbol: Trading symbol (MES, MGC, etc.)

    Returns:
        Dict of sampled parameters
    """
    # Symbol-specific bounds
    bounds = SYMBOL_BOUNDS[symbol]

    return {
        'upper_mult': trial.suggest_float(
            'upper_mult',
            bounds['upper_mult'][0],
            bounds['upper_mult'][1],
            log=False
        ),
        'lower_mult': trial.suggest_float(
            'lower_mult',
            bounds['lower_mult'][0],
            bounds['lower_mult'][1],
            log=False
        ),
        'horizon': trial.suggest_int(
            'horizon',
            bounds['horizon'][0],
            bounds['horizon'][1]
        ),
        'atr_period': trial.suggest_int(
            'atr_period',
            bounds['atr_period'][0],
            bounds['atr_period'][1]
        ),
    }
```

### Parameter Bounds by Symbol

```python
# src/phase1/stages/label_optimize/bounds.py

SYMBOL_BOUNDS = {
    'MES': {
        'upper_mult': (1.0, 3.0),    # 1x to 3x ATR for profit target
        'lower_mult': (0.5, 2.0),    # 0.5x to 2x ATR for stop loss
        'horizon': (5, 30),          # 5 to 30 bars max holding period
        'atr_period': (10, 30),      # 10 to 30 bars ATR lookback
    },
    'MGC': {
        'upper_mult': (1.5, 4.0),    # Higher for more volatile gold
        'lower_mult': (0.75, 2.5),
        'horizon': (5, 30),
        'atr_period': (10, 30),
    }
}
```

**Why these parameter ranges?**
- `upper_mult` > `lower_mult`: Encourages positive risk/reward ratio (asymmetric barriers)
- `atr_period`: Balances responsiveness vs stability of volatility estimate
- `horizon`: Prevents holding positions too long in uncertain markets
- MES vs MGC: Gold has higher volatility, requiring wider barrier multipliers

### Objective Function

```python
# src/phase1/stages/label_optimize/objective.py
import optuna
import numpy as np
import pandas as pd
from src.phase1.stages.labeling.triple_barrier import apply_triple_barrier_labels
from src.metrics.sharpe import compute_sharpe_ratio

def label_optimization_objective(
    trial: optuna.Trial,
    df: pd.DataFrame,
    config: dict
) -> float:
    """
    Objective function for Stage 7 label parameter optimization.

    Args:
        trial: Optuna trial
        df: OHLCV DataFrame with ATR computed
        config: Config with symbol, bounds, etc.

    Returns:
        Negative Sharpe ratio (Optuna minimizes, we want to maximize Sharpe)
    """
    # Sample label parameters
    params = label_param_search_space(trial, config['symbol'])

    upper_mult = params['upper_mult']
    lower_mult = params['lower_mult']
    horizon = params['horizon']
    atr_period = params['atr_period']

    # Compute ATR for barrier calculation
    atr = compute_atr(df, period=atr_period)

    # Apply triple-barrier labels with ATR-based barriers
    labels_df = apply_triple_barrier_labels(
        df,
        upper_barrier=atr * upper_mult,
        lower_barrier=atr * lower_mult,
        max_horizon=horizon,
    )

    # Simulate trading strategy
    positions = labels_df['label']  # -1 (short), 0 (neutral), +1 (long)
    returns = df['returns'].values

    # Strategy returns
    strategy_returns = positions.values * returns

    # Transaction costs
    position_changes = np.abs(np.diff(positions.values, prepend=0))
    cost_per_trade = config.get('cost_per_trade', 0.0002)  # 2 bps
    transaction_costs = position_changes * cost_per_trade

    # Net returns
    net_returns = strategy_returns - transaction_costs

    # Sharpe ratio (annualized)
    sharpe = compute_sharpe_ratio(net_returns, annualization_factor=252)

    # Penalty for class imbalance (want roughly balanced labels)
    label_counts = labels_df['label'].value_counts(normalize=True)
    imbalance_penalty = 0.0
    for label in [-1, 0, 1]:
        if label in label_counts:
            # Penalize if any class is less than 15% or more than 50%
            ratio = label_counts[label]
            if ratio < 0.15:
                imbalance_penalty += (0.15 - ratio) * 2.0
            elif ratio > 0.50:
                imbalance_penalty += (ratio - 0.50) * 2.0

    # Penalty for excessive trading
    turnover = position_changes.sum() / len(positions)
    turnover_penalty = max(0, turnover - 0.5) * 0.5

    # Final objective (negative because Optuna minimizes)
    return -(sharpe - turnover_penalty - imbalance_penalty)
```

### Running Label Optimization (Stage 7)

```bash
# Optimize label parameters for MES using Optuna
python -m src.phase1.stages.label_optimize.run \
    --symbol MES \
    --n-trials 100 \
    --timeout 3600  # 1 hour max
```

**Output:**

```yaml
# Saved to: experiments/label_optimize/MES_20260118_120000/best_params.yaml
symbol: MES
stage: 7_label_optimization
best_params:
  upper_mult: 2.15
  lower_mult: 1.35
  horizon: 20
  atr_period: 14

metrics:
  sharpe_ratio: 1.92
  win_rate: 0.541
  label_balance:
    long: 0.32
    neutral: 0.38
    short: 0.30
  turnover: 0.35
  num_trials: 100
  optimization_time: 2834.2  # seconds
```

### Integration with Pipeline

After Stage 7 optimization, the best parameters flow to Stage 9 (Final Labels):

```python
# src/phase1/stages/final_labels/run.py
from src.phase1.stages.label_optimize.results import load_best_label_params

def run_final_labels_stage(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    # Load Optuna-optimized label parameters
    best_params = load_best_label_params(symbol=config.symbol)

    # Compute ATR with optimized period
    atr = compute_atr(df, period=best_params['atr_period'])

    # Apply triple-barrier labels with optimized params
    labels_df = apply_triple_barrier_labels(
        df,
        upper_barrier=atr * best_params['upper_mult'],
        lower_barrier=atr * best_params['lower_mult'],
        max_horizon=best_params['horizon'],
    )

    return labels_df
```

### Validation: Prevent Overfitting on Train Set

**Problem:** Optimization on training data may overfit.

**Solution:** Use walk-forward validation during optimization:

```python
def label_objective_walk_forward(trial: optuna.Trial, df: pd.DataFrame, config: dict) -> float:
    """
    Objective function with walk-forward validation.

    Splits data into 3 periods:
    - Train (60%): Generate labels
    - Val (20%): Evaluate Sharpe (optimization target)
    - Test (20%): Holdout for final validation
    """
    # Sample parameters
    params = label_param_search_space(trial, config['symbol'])

    # Compute ATR and apply labels
    atr = compute_atr(df, period=params['atr_period'])
    labels_df = apply_triple_barrier_labels(
        df,
        upper_barrier=atr * params['upper_mult'],
        lower_barrier=atr * params['lower_mult'],
        max_horizon=params['horizon'],
    )

    # Walk-forward split
    n = len(labels_df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    # Compute strategy returns on validation set ONLY
    val_labels = labels_df.iloc[train_end:val_end]
    val_returns = val_labels['label'].values * df.iloc[train_end:val_end]['returns'].values

    # Evaluate on validation set
    sharpe_val = compute_sharpe_ratio(val_returns, annualization_factor=252)

    return -sharpe_val
```

---

## Stage 8: Feature Selection Optimization with Optuna

### Overview

**Pipeline Stage:** 8 of 16
**Goal:** Select optimal subset of features using binary include/exclude decisions.
**Trials:** 100 Optuna trials

**Approach:** Each trial samples a binary mask for each feature group, training a lightweight model to evaluate the feature subset's predictive power.

### Search Space Definition

```python
# src/phase1/stages/feature_select/optuna_optimizer.py
import optuna
from typing import Dict, List

# Feature groups for optimization (reduce search space vs individual features)
FEATURE_GROUPS = {
    'momentum': ['rsi_14', 'macd', 'macd_signal', 'macd_hist', 'cci_20', 'stoch_k', 'stoch_d'],
    'volatility': ['atr_14', 'bollinger_width', 'keltner_width', 'historical_vol'],
    'volume': ['volume_ratio', 'obv', 'vwap_distance', 'mfi_14'],
    'trend': ['adx_14', 'aroon_up', 'aroon_down', 'supertrend'],
    'moving_avg': ['sma_20', 'sma_50', 'ema_12', 'ema_26', 'price_to_sma20'],
    'microstructure': ['order_flow_imbalance', 'amihud_illiquidity', 'roll_spread'],
    'wavelets': ['wavelet_trend', 'wavelet_detail_1', 'wavelet_detail_2', 'wavelet_detail_3'],
    'temporal': ['hour_sin', 'hour_cos', 'day_of_week_sin', 'session_progress'],
    'regime': ['volatility_regime', 'trend_regime', 'composite_regime'],
}

def feature_selection_search_space(trial: optuna.Trial) -> Dict[str, bool]:
    """
    Optuna search space for feature selection (binary include/exclude).

    Args:
        trial: Optuna trial object

    Returns:
        Dict mapping feature group names to include (True) or exclude (False)
    """
    selected_groups = {}

    for group_name in FEATURE_GROUPS.keys():
        # Binary decision: include or exclude this feature group
        selected_groups[group_name] = trial.suggest_categorical(
            f'include_{group_name}',
            [True, False]
        )

    # Ensure at least one group is selected
    if not any(selected_groups.values()):
        # Force momentum to be included as fallback
        selected_groups['momentum'] = True

    return selected_groups
```

### Objective Function

```python
# src/phase1/stages/feature_select/objective.py
import optuna
import numpy as np
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier

def feature_selection_objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    config: dict
) -> float:
    """
    Objective function for Stage 8 feature selection optimization.

    Args:
        trial: Optuna trial
        X: Full feature matrix
        y: Labels
        feature_names: List of feature names
        config: Configuration dict

    Returns:
        Negative cross-validation score (Optuna minimizes)
    """
    # Get selected feature groups
    selected_groups = feature_selection_search_space(trial)

    # Build list of selected features
    selected_features = []
    for group_name, include in selected_groups.items():
        if include:
            selected_features.extend(FEATURE_GROUPS[group_name])

    # Get indices of selected features
    selected_indices = [
        i for i, name in enumerate(feature_names)
        if any(feat in name for feat in selected_features)
    ]

    if len(selected_indices) == 0:
        return float('inf')  # No features selected

    # Subset features
    X_subset = X[:, selected_indices]

    # Train lightweight model for evaluation
    model = LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        verbose=-1,
        random_state=42
    )

    # Cross-validation score (use Sharpe-proxy or accuracy)
    scores = cross_val_score(
        model, X_subset, y,
        cv=3,
        scoring='balanced_accuracy',
        n_jobs=-1
    )

    mean_score = np.mean(scores)

    # Penalty for using too many features (regularization)
    n_features = len(selected_indices)
    feature_penalty = config.get('feature_penalty', 0.001) * n_features

    # Return negative score (Optuna minimizes)
    return -(mean_score - feature_penalty)
```

### Running Feature Selection (Stage 8)

```bash
# Optimize feature selection for MES
python -m src.phase1.stages.feature_select.run \
    --symbol MES \
    --n-trials 100 \
    --timeout 1800  # 30 min max
```

**Output:**

```yaml
# Saved to: experiments/feature_select/MES_20260118_130000/best_features.yaml
symbol: MES
stage: 8_feature_selection
selected_groups:
  momentum: true
  volatility: true
  volume: false
  trend: true
  moving_avg: true
  microstructure: false
  wavelets: true
  temporal: true
  regime: true

metrics:
  n_features_selected: 87
  n_features_total: 162
  cross_val_score: 0.632
  num_trials: 100
  optimization_time: 1245.3  # seconds
```

---

## Stage 9: Feature Pruning Optimization with Optuna

### Overview

**Pipeline Stage:** 9 of 16
**Goal:** Prune low-importance features based on model-derived importance scores.
**Trials:** 50 Optuna trials

**Approach:** After Stage 8 selects feature groups, Stage 9 fine-tunes by removing individual low-importance features based on learned thresholds.

### Search Space Definition

```python
# src/phase1/stages/feature_prune/optuna_optimizer.py
import optuna

def feature_pruning_search_space(trial: optuna.Trial) -> dict:
    """
    Optuna search space for feature pruning thresholds.

    Args:
        trial: Optuna trial object

    Returns:
        Dict of pruning parameters
    """
    return {
        # Minimum importance threshold (features below this are pruned)
        'importance_threshold': trial.suggest_float(
            'importance_threshold',
            0.0001,
            0.01,
            log=True
        ),
        # Maximum correlation allowed (higher correlation = redundant)
        'max_correlation': trial.suggest_float(
            'max_correlation',
            0.85,
            0.99
        ),
        # Minimum variance threshold (low variance = uninformative)
        'min_variance_percentile': trial.suggest_float(
            'min_variance_percentile',
            0.01,
            0.10
        ),
        # Method for computing importance
        'importance_method': trial.suggest_categorical(
            'importance_method',
            ['gain', 'split', 'permutation']
        ),
    }
```

### Objective Function

```python
# src/phase1/stages/feature_prune/objective.py
import optuna
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance

def feature_pruning_objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict
) -> float:
    """
    Objective function for Stage 9 feature pruning optimization.

    Args:
        trial: Optuna trial
        X: Training feature matrix (post Stage 8 selection)
        y: Training labels
        feature_names: List of feature names
        X_val: Validation feature matrix
        y_val: Validation labels
        config: Configuration dict

    Returns:
        Negative validation score (Optuna minimizes)
    """
    # Get pruning thresholds
    params = feature_pruning_search_space(trial)

    # Train model to get feature importances
    model = LGBMClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        verbose=-1,
        random_state=42
    )
    model.fit(X, y)

    # Get feature importances
    if params['importance_method'] == 'gain':
        importances = model.booster_.feature_importance(importance_type='gain')
    elif params['importance_method'] == 'split':
        importances = model.booster_.feature_importance(importance_type='split')
    else:  # permutation
        perm_result = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42)
        importances = perm_result.importances_mean

    # Normalize importances
    importances = importances / importances.sum()

    # Prune features below threshold
    keep_mask = importances >= params['importance_threshold']

    # Also prune low-variance features
    variances = np.var(X, axis=0)
    variance_threshold = np.percentile(variances, params['min_variance_percentile'] * 100)
    keep_mask &= (variances >= variance_threshold)

    # Apply correlation pruning
    if keep_mask.sum() > 1:
        corr_matrix = np.corrcoef(X[:, keep_mask].T)
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if abs(corr_matrix[i, j]) > params['max_correlation']:
                    # Remove feature with lower importance
                    indices = np.where(keep_mask)[0]
                    if importances[indices[i]] < importances[indices[j]]:
                        keep_mask[indices[i]] = False
                    else:
                        keep_mask[indices[j]] = False

    if keep_mask.sum() == 0:
        return float('inf')

    # Subset and evaluate
    X_pruned = X[:, keep_mask]
    X_val_pruned = X_val[:, keep_mask]

    model_pruned = LGBMClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        verbose=-1,
        random_state=42
    )
    model_pruned.fit(X_pruned, y)

    # Evaluate on validation set
    val_score = model_pruned.score(X_val_pruned, y_val)

    # Slight penalty for feature count (prefer parsimony)
    n_features = keep_mask.sum()
    parsimony_bonus = config.get('parsimony_bonus', 0.0005) * (X.shape[1] - n_features)

    return -(val_score + parsimony_bonus)
```

### Running Feature Pruning (Stage 9)

```bash
# Prune low-importance features for MES
python -m src.phase1.stages.feature_prune.run \
    --symbol MES \
    --n-trials 50 \
    --timeout 900  # 15 min max
```

**Output:**

```yaml
# Saved to: experiments/feature_prune/MES_20260118_140000/pruned_features.yaml
symbol: MES
stage: 9_feature_pruning
pruning_params:
  importance_threshold: 0.0023
  max_correlation: 0.92
  min_variance_percentile: 0.03
  importance_method: gain

metrics:
  n_features_before: 87
  n_features_after: 52
  features_removed: 35
  val_accuracy_before: 0.618
  val_accuracy_after: 0.627
  num_trials: 50
  optimization_time: 687.1  # seconds

removed_features:
  - wavelet_detail_3
  - aroon_down
  - stoch_d
  # ... (32 more)
```

---

## Stage 13: Per-Model Hyperparameter Optimization

### Overview

**Pipeline Stage:** 13 of 16
**Goal:** Optimize model hyperparameters using Optuna's Tree-structured Parzen Estimator (TPE).
**Trials:** 100 trials per model (23 models = 2,300 total trials if running all models)

**Integration:** `src/cross_validation/cv_runner.py`

### Trial Budget by Model Family

| Model Family | Models | Trials per Model | Time per Trial | Total Time |
|--------------|--------|------------------|----------------|------------|
| **Boosting** | XGBoost, LightGBM, CatBoost | 100 | ~2 min | ~10 hours |
| **Classical** | Random Forest, Logistic, SVM | 100 | ~1 min | ~5 hours |
| **Neural (RNN)** | LSTM, GRU | 100 | ~10 min | ~33 hours |
| **Neural (CNN)** | TCN, InceptionTime, ResNet1D | 100 | ~8 min | ~40 hours |
| **Transformers** | Transformer, PatchTST, iTransformer, TFT | 100 | ~15 min | ~100 hours |
| **Foundation** | N-BEATS | 100 | ~5 min | ~8 hours |

### Search Spaces per Model Family

Located in `src/cross_validation/param_spaces.py`:

---

### Boosting Models (XGBoost, LightGBM, CatBoost)

```python
# src/cross_validation/param_spaces.py

def xgboost_param_space(trial: optuna.Trial) -> dict:
    """
    Hyperparameter search space for XGBoost.

    Returns:
        Dict of hyperparameters
    """
    return {
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),

        # Fixed params
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'early_stopping_rounds': 50,
        'num_boost_round': 1000,
    }

def lightgbm_param_space(trial: optuna.Trial) -> dict:
    """LightGBM search space."""
    return {
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),

        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'early_stopping_rounds': 50,
    }

def catboost_param_space(trial: optuna.Trial) -> dict:
    """CatBoost search space."""
    return {
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),

        'iterations': 1000,
        'early_stopping_rounds': 50,
        'loss_function': 'MultiClass',
        'verbose': False,
    }
```

**Key hyperparameters explained:**

| Parameter | Description | Impact |
|-----------|-------------|--------|
| `learning_rate` | Step size for gradient descent | Lower = slower, more accurate; higher = faster, may overfit |
| `max_depth` | Maximum tree depth | Higher = more complex trees, more overfitting risk |
| `num_leaves` | Max leaves per tree (LightGBM) | Higher = more complex, more overfitting |
| `subsample` | Fraction of samples per tree | <1.0 adds randomness, reduces overfitting |
| `colsample_bytree` | Fraction of features per tree | <1.0 reduces correlation between trees |
| `reg_alpha` | L1 regularization | Higher = sparser models |
| `reg_lambda` | L2 regularization | Higher = smoother models |

---

### Neural Sequence Models (LSTM, GRU, TCN)

```python
def lstm_param_space(trial: optuna.Trial) -> dict:
    """LSTM search space."""
    return {
        'hidden_size': trial.suggest_int('hidden_size', 32, 256, log=True),
        'num_layers': trial.suggest_int('num_layers', 1, 4),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512]),
        'seq_len': trial.suggest_categorical('seq_len', [30, 60, 90, 120]),
        'gradient_clip_norm': trial.suggest_float('gradient_clip_norm', 0.5, 5.0),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),

        # Fixed
        'max_epochs': 100,
        'patience': 10,
    }

def gru_param_space(trial: optuna.Trial) -> dict:
    """GRU search space (similar to LSTM)."""
    return lstm_param_space(trial)  # Same search space

def tcn_param_space(trial: optuna.Trial) -> dict:
    """TCN (Temporal Convolutional Network) search space."""
    return {
        'num_channels': trial.suggest_categorical('num_channels', [[32, 64], [64, 128], [128, 256]]),
        'kernel_size': trial.suggest_categorical('kernel_size', [3, 5, 7]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'seq_len': trial.suggest_categorical('seq_len', [30, 60, 90]),

        'max_epochs': 100,
        'patience': 10,
    }
```

**Key hyperparameters explained:**

| Parameter | Description | Impact |
|-----------|-------------|--------|
| `hidden_size` | LSTM/GRU hidden state dimension | Larger = more capacity, more overfitting risk |
| `num_layers` | Number of stacked RNN layers | More layers = deeper model, slower training |
| `dropout` | Dropout probability | Higher = more regularization |
| `seq_len` | Lookback window length | Longer = more history, more computation |
| `batch_size` | Samples per gradient update | Larger = faster, less noisy gradients |
| `gradient_clip_norm` | Max gradient norm | Prevents exploding gradients |

---

### Transformers (PatchTST, iTransformer, TFT)

```python
def patchtst_param_space(trial: optuna.Trial) -> dict:
    """PatchTST search space."""
    return {
        'd_model': trial.suggest_categorical('d_model', [64, 128, 256, 512]),
        'nhead': trial.suggest_categorical('nhead', [4, 8, 16]),
        'num_layers': trial.suggest_int('num_layers', 2, 8),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        'patch_len': trial.suggest_categorical('patch_len', [8, 16, 32, 64]),
        'stride': trial.suggest_categorical('stride', [4, 8, 16]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),

        'max_epochs': 100,
        'patience': 15,
    }

def itransformer_param_space(trial: optuna.Trial) -> dict:
    """iTransformer search space."""
    return {
        'd_model': trial.suggest_categorical('d_model', [64, 128, 256]),
        'nhead': trial.suggest_categorical('nhead', [4, 8, 16]),
        'num_layers': trial.suggest_int('num_layers', 2, 6),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),

        'max_epochs': 100,
        'patience': 15,
    }

def tft_param_space(trial: optuna.Trial) -> dict:
    """Temporal Fusion Transformer search space."""
    return {
        'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128, 256]),
        'num_attention_heads': trial.suggest_categorical('num_attention_heads', [1, 2, 4]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),

        'max_epochs': 100,
        'patience': 15,
    }
```

---

### CNN Models (InceptionTime, ResNet)

```python
def inceptiontime_param_space(trial: optuna.Trial) -> dict:
    """InceptionTime search space."""
    return {
        'num_filters': trial.suggest_categorical('num_filters', [32, 64, 128, 256]),
        'bottleneck_size': trial.suggest_categorical('bottleneck_size', [16, 32, 64]),
        'num_blocks': trial.suggest_int('num_blocks', 2, 8),
        'dropout': trial.suggest_float('dropout', 0.0, 0.4),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),

        'max_epochs': 100,
        'patience': 10,
    }

def resnet_param_space(trial: optuna.Trial) -> dict:
    """ResNet search space."""
    return {
        'num_filters': trial.suggest_categorical('num_filters', [32, 64, 128]),
        'num_blocks': trial.suggest_int('num_blocks', 2, 8),
        'dropout': trial.suggest_float('dropout', 0.0, 0.4),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),

        'max_epochs': 100,
        'patience': 10,
    }
```

---

### Classical Models (Random Forest, Logistic, SVM)

```python
def random_forest_param_space(trial: optuna.Trial) -> dict:
    """Random Forest search space."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),

        'random_state': 42,
        'n_jobs': -1,
    }

def logistic_param_space(trial: optuna.Trial) -> dict:
    """Logistic Regression search space."""
    return {
        'C': trial.suggest_float('C', 1e-4, 100.0, log=True),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
        'solver': trial.suggest_categorical('solver', ['saga']),  # saga supports all penalties
        'max_iter': trial.suggest_int('max_iter', 100, 2000),

        'random_state': 42,
        'n_jobs': -1,
    }

def svm_param_space(trial: optuna.Trial) -> dict:
    """SVM search space."""
    return {
        'C': trial.suggest_float('C', 1e-3, 100.0, log=True),
        'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]),
        'degree': trial.suggest_int('degree', 2, 5),  # For poly kernel

        'random_state': 42,
    }
```

---

## Objective Functions

### Single-Objective: Sharpe Ratio

```python
# src/cross_validation/cv_runner.py

def optuna_objective_sharpe(trial: optuna.Trial, model_name: str, data) -> float:
    """
    Objective function: Maximize Sharpe ratio on validation set.

    Args:
        trial: Optuna trial
        model_name: Model to optimize
        data: (X_train, y_train, X_val, y_val, weights_train, weights_val)

    Returns:
        Negative Sharpe (Optuna minimizes, we want to maximize)
    """
    X_train, y_train, X_val, y_val, weights_train, weights_val = data

    # Get param space
    param_space_fn = get_param_space_fn(model_name)
    params = param_space_fn(trial)

    # Train model
    model_class = ModelRegistry.get(model_name)
    model = model_class(config=params)

    try:
        model.fit(X_train, y_train, X_val, y_val, weights_train)
    except Exception as e:
        # If training fails, return worst score
        return float('inf')

    # Predict on validation set
    output = model.predict(X_val)
    y_pred = output.predictions

    # Map labels to positions
    position_map = {0: -1, 1: 0, 2: 1}
    positions = np.array([position_map[p] for p in y_pred])

    # Compute returns (assuming we have returns in data)
    returns = compute_returns_from_labels(y_val, positions)

    # Sharpe ratio
    sharpe = compute_sharpe_ratio(returns, annualization_factor=252)

    # Return negative (Optuna minimizes)
    return -sharpe
```

### Multi-Objective: Sharpe vs Max Drawdown

```python
def optuna_objective_multi(trial: optuna.Trial, model_name: str, data) -> tuple[float, float]:
    """
    Multi-objective: Maximize Sharpe, Minimize Max Drawdown.

    Returns:
        (-sharpe, max_drawdown)  # Both to minimize
    """
    X_train, y_train, X_val, y_val, weights_train, weights_val = data

    # ... train model, get predictions ...

    # Compute metrics
    sharpe = compute_sharpe_ratio(returns)
    max_dd = compute_max_drawdown(returns)

    # Return both objectives (Optuna will find Pareto front)
    return (-sharpe, max_dd)
```

**Run multi-objective optimization:**

```python
import optuna

study = optuna.create_study(
    directions=['minimize', 'minimize'],  # Minimize both objectives
    sampler=optuna.samplers.NSGAIISampler()  # Multi-objective sampler
)

study.optimize(lambda trial: optuna_objective_multi(trial, 'xgboost', data), n_trials=100)

# Get Pareto front
pareto_trials = study.best_trials

# Visualize
optuna.visualization.plot_pareto_front(study)
```

---

## Pruning Strategies

**Idea:** Stop unpromising trials early to save computation.

### Median Pruner

```python
import optuna

# Create study with median pruner
study = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=10,  # Don't prune first 10 trials
        n_warmup_steps=20,    # Wait 20 steps before pruning
        interval_steps=5      # Check every 5 steps
    )
)
```

**How it works:**
- After `n_warmup_steps`, compute median of all trials at that step
- If current trial is worse than median, prune it
- Saves ~30-50% of computation

### Integration with Model Training

```python
def fit_with_pruning(model, X_train, y_train, X_val, y_val, trial):
    """
    Train model with pruning callback.

    Args:
        trial: Optuna trial for pruning
    """
    for epoch in range(max_epochs):
        # Train one epoch
        train_loss = model.train_epoch(X_train, y_train)
        val_loss = model.validate_epoch(X_val, y_val)

        # Report intermediate value
        trial.report(val_loss, epoch)

        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

    return model
```

---

## Multi-Objective Optimization

### Pareto Front: Sharpe vs Drawdown

**Goal:** Find trade-off between high returns (Sharpe) and low risk (drawdown).

```python
import optuna

# Create multi-objective study
study = optuna.create_study(
    directions=['minimize', 'minimize'],  # (-Sharpe, max_dd)
    sampler=optuna.samplers.NSGAIISampler(population_size=50)
)

# Optimize
study.optimize(
    lambda trial: multi_objective_fn(trial, model_name='xgboost', data=data),
    n_trials=200
)

# Get Pareto-optimal trials
pareto_trials = study.best_trials

# Example: Select trial with best Sharpe subject to max_dd < 0.15
best_trial = min(
    [t for t in pareto_trials if t.values[1] < 0.15],  # Drawdown < 15%
    key=lambda t: t.values[0]  # Minimize -Sharpe (i.e., max Sharpe)
)

print(f"Best params: {best_trial.params}")
print(f"Sharpe: {-best_trial.values[0]:.2f}, Max DD: {best_trial.values[1]:.2%}")
```

**Visualization:**

```python
import optuna.visualization as vis

# Pareto front plot
fig = vis.plot_pareto_front(study, target_names=['Sharpe (neg)', 'Max Drawdown'])
fig.show()

# Hyperparameter importances (for first objective)
fig = vis.plot_param_importances(study, target=lambda t: t.values[0])
fig.show()
```

---

## Integration with Cross-Validation

### Nested CV for Hyperparameter Tuning

**Outer CV:** Evaluate generalization (5 folds)
**Inner CV:** Hyperparameter tuning (3 folds per outer fold)

```python
# src/cross_validation/cv_runner.py
from src.cross_validation.purged_kfold import PurgedKFold
import optuna

def nested_cv_with_optuna(
    X, y, weights,
    model_name: str,
    n_outer_splits: int = 5,
    n_inner_splits: int = 3,
    n_trials_per_fold: int = 50
):
    """
    Nested cross-validation with Optuna hyperparameter tuning.

    Returns:
        oof_predictions, best_params_per_fold
    """
    outer_kfold = PurgedKFold(n_splits=n_outer_splits, purge_bars=60, embargo_bars=480)
    inner_kfold = PurgedKFold(n_splits=n_inner_splits, purge_bars=60, embargo_bars=480)

    oof_predictions = np.zeros((len(X), 3))  # (n_samples, 3 classes)
    best_params_per_fold = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_kfold.split(X)):
        print(f"Outer fold {fold_idx + 1}/{n_outer_splits}")

        X_train_outer = X[train_idx]
        y_train_outer = y[train_idx]
        weights_train_outer = weights[train_idx]

        X_test_outer = X[test_idx]
        y_test_outer = y[test_idx]

        # Inner CV for hyperparameter tuning
        def inner_objective(trial):
            param_space_fn = get_param_space_fn(model_name)
            params = param_space_fn(trial)

            inner_scores = []

            for inner_train_idx, inner_val_idx in inner_kfold.split(X_train_outer):
                X_train_inner = X_train_outer[inner_train_idx]
                y_train_inner = y_train_outer[inner_train_idx]
                weights_train_inner = weights_train_outer[inner_train_idx]

                X_val_inner = X_train_outer[inner_val_idx]
                y_val_inner = y_train_outer[inner_val_idx]

                # Train model
                model_class = ModelRegistry.get(model_name)
                model = model_class(config=params)
                model.fit(X_train_inner, y_train_inner, X_val_inner, y_val_inner, weights_train_inner)

                # Evaluate
                output = model.predict(X_val_inner)
                score = compute_sharpe_from_predictions(output.predictions, y_val_inner)
                inner_scores.append(score)

            # Return mean score across inner folds
            return -np.mean(inner_scores)  # Negative because Optuna minimizes

        # Run Optuna on inner CV
        study = optuna.create_study(direction='minimize')
        study.optimize(inner_objective, n_trials=n_trials_per_fold, show_progress_bar=False)

        # Best params for this outer fold
        best_params = study.best_params
        best_params_per_fold.append(best_params)

        # Train final model on full outer train set with best params
        model_class = ModelRegistry.get(model_name)
        model = model_class(config=best_params)
        model.fit(X_train_outer, y_train_outer, X_test_outer[:500], y_test_outer[:500], weights_train_outer)

        # Predict on outer test set
        output = model.predict(X_test_outer)
        oof_predictions[test_idx] = output.probabilities

    return oof_predictions, best_params_per_fold
```

### Running Nested CV

```bash
python scripts/run_cv.py \
    --models xgboost \
    --horizons 20 \
    --n-splits 5 \
    --tune \
    --n-trials 50
```

**Output:**

```
Outer fold 1/5
  Inner CV: 50 trials, best Sharpe: 1.82
  Best params: {'learning_rate': 0.05, 'max_depth': 8, ...}
  Outer test Sharpe: 1.76

Outer fold 2/5
  Inner CV: 50 trials, best Sharpe: 1.91
  Best params: {'learning_rate': 0.04, 'max_depth': 7, ...}
  Outer test Sharpe: 1.85

...

Final OOF Sharpe: 1.79 ± 0.08
```

---

## Computational Budget

### Budget Allocation by Optimization Stage

| Stage | Description | Trials | Estimated Time |
|-------|-------------|--------|----------------|
| **Stage 7** | Label Optimization | 100 | ~1 hour |
| **Stage 8** | Feature Selection | 100 | ~30 min |
| **Stage 9** | Feature Pruning | 50 | ~15 min |
| **Stage 13** | Hyperparameter Optimization | 100 per model | See below |

### Stage 13: Per-Model Trial Budget

| Model Family | Trials per Model | Time per Trial | Total Time |
|--------------|------------------|----------------|------------|
| **Boosting** (XGBoost, LightGBM, CatBoost) | 100 | 2 min | ~10 hours |
| **Neural (LSTM, GRU)** | 100 | 10 min | ~33 hours |
| **Transformers** (PatchTST, iTransformer, TFT) | 100 | 15-20 min | ~80 hours |
| **CNN** (InceptionTime, ResNet1D, TCN) | 100 | 8 min | ~40 hours |
| **Classical** (RF, Logistic, SVM) | 100 | 1 min | ~5 hours |
| **Foundation** (N-BEATS) | 100 | 5 min | ~8 hours |

**Total Optuna trials across all stages:**
- Label: 100
- Feature Selection: 100
- Feature Pruning: 50
- Hyperparameters: 100 x N_models
- **Grand Total:** 250 + (100 x N_models)

### Parallel Optimization

Run multiple Optuna studies in parallel:

```python
import optuna
from joblib import Parallel, delayed

def optimize_model(model_name, data, n_trials):
    """Run Optuna for one model."""
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: optuna_objective(trial, model_name, data),
        n_trials=n_trials
    )
    return model_name, study.best_params, study.best_value

# Parallel optimization for multiple models
models = ['xgboost', 'lightgbm', 'catboost', 'lstm']
results = Parallel(n_jobs=4)(
    delayed(optimize_model)(model, data, n_trials=50)
    for model in models
)

# Results: [(model_name, best_params, best_value), ...]
```

---

## Adding New Parameters

### Step 1: Define Parameter in Search Space

```python
# src/cross_validation/param_spaces.py

def my_model_param_space(trial: optuna.Trial) -> dict:
    return {
        # Existing params
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),

        # NEW PARAMETER
        'new_param': trial.suggest_int('new_param', 1, 10),

        # ...
    }
```

### Step 2: Use Parameter in Model

```python
# src/models/my_family/my_model.py

def fit(self, X_train, y_train, X_val, y_val, sample_weights=None, config=None):
    params = self._get_params(config)

    # Use new parameter
    new_param_value = params.get('new_param', default_value)

    # Initialize model with new parameter
    self.model = SomeModel(new_param=new_param_value, ...)
```

### Step 3: Run Optuna

```bash
python scripts/run_cv.py --models my_model --tune --n-trials 100
```

Optuna will automatically search over the new parameter.

---

## Summary

### Four-Stage Optuna Optimization Workflow

| Stage | What | Trials | Key Parameters |
|-------|------|--------|----------------|
| **Stage 7** | Label Optimization | 100 | upper_mult, lower_mult, horizon, atr_period |
| **Stage 8** | Feature Selection | 100 | Binary include/exclude per feature group |
| **Stage 9** | Feature Pruning | 50 | importance_threshold, max_correlation, importance_method |
| **Stage 13** | Model Hyperparameters | 100/model | Model-specific (learning_rate, depth, etc.) |

### Optimization Workflow

1. **Stage 7 (Label Opt):** Optimize triple-barrier parameters (upper_mult, lower_mult, horizon, atr_period) with 100 trials
2. **Stage 8 (Feature Select):** Binary include/exclude optimization for feature groups with 100 trials
3. **Stage 9 (Feature Prune):** Importance-based feature removal with 50 trials
4. **Stage 13 (Hyperparameters):** Per-model optimization with 100 trials each
5. **Nested CV:** Use inner CV for tuning, outer CV for evaluation
6. **Pruning:** Use MedianPruner to save ~30-50% computation
7. **Multi-objective:** Optimize Sharpe vs drawdown for risk-adjusted models

### Key Takeaways

- **Optuna everywhere:** All 4 optimization stages use Optuna for consistency
- **ATR-based barriers:** Label parameters use ATR multipliers for volatility-adaptive thresholds
- **Hierarchical feature selection:** Stage 8 selects groups, Stage 9 prunes individuals
- **100 trials per model:** Standard budget ensures thorough hyperparameter search
- **Nested CV prevents overfitting:** Inner CV tunes, outer CV evaluates
- **Symbol-specific optimization:** MES and MGC have different parameter bounds
- **Transaction cost penalties:** Prevents over-trading in label optimization

### File Paths Reference

- Label optimization: `src/phase1/stages/label_optimize/`
- Feature selection: `src/phase1/stages/feature_select/`
- Feature pruning: `src/phase1/stages/feature_prune/`
- Optuna integration: `src/cross_validation/cv_runner.py`
- Search spaces: `src/cross_validation/param_spaces.py`
- Nested CV: `src/cross_validation/nested_cv.py`

### Pipeline Stage Reference (16 Stages)

```
Stages 1-6:   Data preparation (Ingest, Clean, Sessions, MTF, Features, Regime)
Stage 7:      OPTUNA Label Optimization (100 trials)
Stage 8:      OPTUNA Feature Selection (100 trials)
Stage 9:      OPTUNA Feature Pruning (50 trials)
Stages 10-12: Splits, Scaling, Adaptation
Stage 13:     OPTUNA Hyperparameter Optimization (100 trials per model)
Stages 14-16: Training, Stacking, Bundling
```

---

**Next steps:** After hyperparameter optimization, evaluate models using walk-forward validation and CPCV/PBO to ensure robustness.
