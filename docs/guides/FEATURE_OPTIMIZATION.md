# Feature Optimization with Optuna

**Purpose:** Comprehensive guide for feature optimization using Optuna in the 16-stage ML pipeline
**Audience:** ML engineers, data scientists, quant researchers
**Last Updated:** 2026-01-18

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Integration](#pipeline-integration)
3. [Stage 8: Feature Selection Optimization](#stage-8-feature-selection-optimization)
4. [Stage 9: Feature Pruning Optimization](#stage-9-feature-pruning-optimization)
5. [Objective Functions](#objective-functions)
6. [Pruning Callbacks](#pruning-callbacks)
7. [Samplers and Search Strategies](#samplers-and-search-strategies)
8. [Per-Model Feature Optimization](#per-model-feature-optimization)
9. [Configuration Reference](#configuration-reference)
10. [Running Feature Optimization](#running-feature-optimization)
11. [Visualization and Analysis](#visualization-and-analysis)
12. [Best Practices](#best-practices)
13. [Troubleshooting](#troubleshooting)

---

## Overview

Feature optimization is a critical step in the ML Factory pipeline that automatically discovers the optimal feature subset for each model. The pipeline uses Optuna for two dedicated feature optimization stages:

| Stage | Name | Trials | Purpose |
|-------|------|--------|---------|
| **8** | Feature Selection | 100 | Binary include/exclude decisions for feature groups |
| **9** | Feature Pruning | 50 | Importance-based removal of individual features |

### Why Two Stages?

1. **Stage 8 (Selection):** Broad strokes - select which feature groups to include
2. **Stage 9 (Pruning):** Fine-tuning - remove low-importance features within selected groups

This hierarchical approach reduces the search space while maintaining optimization quality.

### Benefits of Optuna-Based Feature Optimization

- **Automatic:** No manual feature selection required
- **Adaptive:** Uses TPE sampler that learns from previous trials
- **Efficient:** Pruning callbacks stop unpromising trials early
- **Reproducible:** Seed-based sampling ensures consistent results
- **Scalable:** Supports parallel optimization across multiple workers

---

## Pipeline Integration

Feature optimization occurs after feature engineering and labeling, but before training:

```
Stage 5: Features (162 indicators)
         |
Stage 6: Regime Detection
         |
Stage 7: OPTUNA Label Optimization (100 trials)
         |
         v
+---------------------------+
| Stage 8: Feature Selection |  <-- 100 Optuna trials
| Binary include/exclude     |
+---------------------------+
         |
         v
+---------------------------+
| Stage 9: Feature Pruning   |  <-- 50 Optuna trials
| Importance-based removal   |
+---------------------------+
         |
         v
Stage 10: Splits (70/15/15)
         |
Stage 11: Scaling (train-only)
         |
Stage 12: Adaptation (2D/3D/4D)
```

### Data Flow

```python
# Input to Stage 8
X: np.ndarray  # Shape: (n_samples, 162) - all features
y: np.ndarray  # Shape: (n_samples,) - labels from Stage 7

# Output from Stage 8
selected_features: List[str]  # ~60-100 features

# Input to Stage 9
X_selected: np.ndarray  # Shape: (n_samples, ~80) - selected features
importance_scores: Dict[str, float]  # From quick model training

# Output from Stage 9
final_features: List[str]  # ~30-60 features (pruned)
```

---

## Stage 8: Feature Selection Optimization

### Overview

**Goal:** Identify the optimal subset of features using binary include/exclude decisions for feature groups.

**Trials:** 100 (configurable)

**Search Space:** 10 feature groups x 2 options (include/exclude) = 1,024 possible combinations

### Feature Groups

Features are organized into logical groups to reduce search space:

```python
FEATURE_GROUPS = {
    'momentum': [
        'rsi_14', 'rsi_7', 'macd', 'macd_signal', 'macd_hist',
        'cci_20', 'stoch_k', 'stoch_d', 'williams_r', 'roc_10',
        'roc_20', 'mfi_14'
    ],
    'volatility': [
        'atr_14', 'atr_7', 'bb_width', 'bb_position',
        'realized_vol_10', 'realized_vol_20', 'parkinson_vol',
        'gk_vol', 'rs_vol'
    ],
    'volume': [
        'volume_ratio', 'obv', 'vwap_distance', 'dollar_volume',
        'twap', 'volume_ma_ratio'
    ],
    'trend': [
        'adx_14', 'aroon_up', 'aroon_down', 'supertrend',
        'di_plus', 'di_minus', 'trend_strength'
    ],
    'moving_avg': [
        'sma_10', 'sma_20', 'sma_50', 'ema_10', 'ema_20',
        'price_to_sma20', 'price_to_sma50'
    ],
    'microstructure': [
        'spread', 'spread_pct', 'imbalance', 'trade_intensity',
        'kyle_lambda', 'amihud', 'roll_spread'
    ],
    'wavelets': [
        'wavelet_trend', 'wavelet_detail_1', 'wavelet_detail_2',
        'wavelet_detail_3', 'wavelet_energy', 'wavelet_entropy'
    ],
    'temporal': [
        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
        'session_progress', 'time_to_close'
    ],
    'regime': [
        'volatility_regime', 'trend_regime', 'composite_regime'
    ],
    'entropy': [
        'shannon_entropy', 'approx_entropy', 'sample_entropy',
        'hurst_exponent', 'lempel_ziv'
    ],
}
```

### Objective Function

```python
import optuna
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score

def feature_selection_objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    config: dict
) -> float:
    """
    Objective function for Stage 8 feature selection.

    Args:
        trial: Optuna trial object
        X: Full feature matrix (n_samples, 162)
        y: Labels (n_samples,)
        feature_names: List of feature names
        config: Configuration dict

    Returns:
        Score to maximize (F1 macro)
    """
    # Binary decision for each feature group
    selected_groups = {}
    for group_name in FEATURE_GROUPS.keys():
        selected_groups[group_name] = trial.suggest_categorical(
            f'include_{group_name}',
            [True, False]
        )

    # Build list of selected feature names
    selected_features = []
    for group_name, include in selected_groups.items():
        if include:
            selected_features.extend(FEATURE_GROUPS[group_name])

    # Get indices of selected features
    selected_indices = [
        i for i, name in enumerate(feature_names)
        if any(feat in name for feat in selected_features)
    ]

    # Constraint: minimum features
    min_features = config.get('min_features', 10)
    if len(selected_indices) < min_features:
        # Penalize: return low score
        return 0.0

    # Constraint: maximum features
    max_features = config.get('max_features', 100)
    if len(selected_indices) > max_features:
        return 0.0

    # Subset features
    X_subset = X[:, selected_indices]

    # Train lightweight model for evaluation
    model = LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        verbose=-1,
        random_state=42,
        n_jobs=-1
    )

    # Cross-validation score
    cv_splits = config.get('cv_splits', 3)
    scores = cross_val_score(
        model, X_subset, y,
        cv=cv_splits,
        scoring='f1_macro',
        n_jobs=-1
    )

    mean_score = np.mean(scores)

    # Regularization: penalize large feature counts
    regularization = config.get('regularization_lambda', 0.001)
    feature_penalty = regularization * len(selected_indices)

    # Final score (higher is better)
    return mean_score - feature_penalty
```

### Search Space Configuration

```yaml
# config/optimization/feature_selection.yaml
search_space:
  feature_selection_mode: binary

  feature_groups:
    price_features:
      enabled: true
    returns_features:
      enabled: true
    volume_features:
      enabled: true
    momentum_features:
      enabled: true
    volatility_features:
      enabled: true
    trend_features:
      enabled: true
    microstructure_features:
      enabled: true

constraints:
  min_features: 10
  max_features: 100
  required_features:
    - returns_1
    - volatility_20
    - volume_ratio
```

### Running Feature Selection

```bash
# CLI
python -m src.optimization.feature_selection \
    --symbol MES \
    --n-trials 100 \
    --timeout 14400 \
    --config config/optimization/feature_selection.yaml

# Python API
from src.optimization import FeatureSelectionOptimizer

optimizer = FeatureSelectionOptimizer(
    config_path='config/optimization/feature_selection.yaml'
)

best_features = optimizer.optimize(
    X=X_train,
    y=y_train,
    feature_names=feature_names,
    n_trials=100
)
```

### Output

```yaml
# experiments/optuna/feature_selection/MES_20260118/results.yaml
symbol: MES
stage: 8_feature_selection
n_trials: 100
optimization_time_seconds: 1245.3

best_params:
  include_momentum: true
  include_volatility: true
  include_volume: false
  include_trend: true
  include_moving_avg: true
  include_microstructure: false
  include_wavelets: true
  include_temporal: true
  include_regime: true
  include_entropy: false

metrics:
  best_score: 0.632
  n_features_selected: 87
  n_features_total: 162

selected_features:
  - rsi_14
  - macd
  - atr_14
  # ... 84 more
```

---

## Stage 9: Feature Pruning Optimization

### Overview

**Goal:** Fine-tune feature selection by removing low-importance individual features.

**Trials:** 50 (configurable)

**Search Space:** Continuous thresholds + categorical method selection

### Search Space

```python
def feature_pruning_search_space(trial: optuna.Trial) -> dict:
    """
    Search space for importance-based feature pruning.
    """
    return {
        # Importance threshold (log scale for better exploration)
        'importance_threshold': trial.suggest_float(
            'importance_threshold',
            0.001,  # Keep almost all
            0.1,    # Remove most
            log=True
        ),

        # Maximum correlation allowed between features
        'max_correlation': trial.suggest_float(
            'max_correlation',
            0.85,
            0.99
        ),

        # Minimum variance percentile to retain
        'min_variance_percentile': trial.suggest_float(
            'min_variance_percentile',
            0.01,
            0.10
        ),

        # Importance calculation method
        'importance_method': trial.suggest_categorical(
            'importance_method',
            ['gain', 'split', 'permutation', 'shap']
        ),

        # Aggregation across CV folds
        'aggregation': trial.suggest_categorical(
            'aggregation',
            ['mean', 'median', 'min']
        ),
    }
```

### Objective Function

```python
import optuna
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance

def feature_pruning_objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    config: dict
) -> float:
    """
    Objective function for Stage 9 feature pruning.

    Args:
        trial: Optuna trial object
        X: Training features (post Stage 8 selection)
        y: Training labels
        X_val: Validation features
        y_val: Validation labels
        feature_names: List of feature names
        config: Configuration dict

    Returns:
        Score to maximize
    """
    params = feature_pruning_search_space(trial)

    # Step 1: Train model to get feature importances
    model = LGBMClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        verbose=-1,
        random_state=42
    )
    model.fit(X, y)

    # Step 2: Calculate feature importances
    if params['importance_method'] == 'gain':
        importances = model.booster_.feature_importance(importance_type='gain')
    elif params['importance_method'] == 'split':
        importances = model.booster_.feature_importance(importance_type='split')
    elif params['importance_method'] == 'permutation':
        result = permutation_importance(
            model, X_val, y_val,
            n_repeats=5,
            random_state=42,
            n_jobs=-1
        )
        importances = result.importances_mean
    else:  # shap
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val[:1000])  # Subsample for speed
        importances = np.abs(shap_values).mean(axis=0).mean(axis=0)

    # Normalize importances to [0, 1]
    importances = importances / (importances.max() + 1e-8)

    # Step 3: Create pruning mask
    keep_mask = importances >= params['importance_threshold']

    # Step 4: Variance filtering
    variances = np.var(X, axis=0)
    var_threshold = np.percentile(
        variances,
        params['min_variance_percentile'] * 100
    )
    keep_mask &= (variances >= var_threshold)

    # Step 5: Correlation filtering
    if keep_mask.sum() > 1:
        X_keep = X[:, keep_mask]
        corr_matrix = np.corrcoef(X_keep.T)
        keep_indices = np.where(keep_mask)[0]

        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if abs(corr_matrix[i, j]) > params['max_correlation']:
                    # Remove feature with lower importance
                    idx_i, idx_j = keep_indices[i], keep_indices[j]
                    if importances[idx_i] < importances[idx_j]:
                        keep_mask[idx_i] = False
                    else:
                        keep_mask[idx_j] = False

    # Constraint: minimum features
    min_retained = config.get('min_features_retained', 20)
    if keep_mask.sum() < min_retained:
        return 0.0

    # Step 6: Evaluate pruned feature set
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

    # Validation score
    from sklearn.metrics import f1_score
    y_pred = model_pruned.predict(X_val_pruned)
    val_score = f1_score(y_val, y_pred, average='macro')

    # Parsimony bonus: reward fewer features at same performance
    efficiency_bonus = config.get('efficiency_bonus_weight', 0.1)
    removed_pct = 1 - (keep_mask.sum() / len(keep_mask))
    parsimony_bonus = efficiency_bonus * removed_pct

    return val_score + parsimony_bonus
```

### Running Feature Pruning

```bash
# CLI
python -m src.optimization.feature_pruning \
    --symbol MES \
    --n-trials 50 \
    --timeout 3600 \
    --config config/optimization/feature_pruning.yaml

# Python API
from src.optimization import FeaturePruningOptimizer

optimizer = FeaturePruningOptimizer(
    config_path='config/optimization/feature_pruning.yaml'
)

pruned_features = optimizer.optimize(
    X=X_selected,
    y=y_train,
    X_val=X_val,
    y_val=y_val,
    feature_names=selected_feature_names,
    n_trials=50
)
```

### Output

```yaml
# experiments/optuna/feature_pruning/MES_20260118/results.yaml
symbol: MES
stage: 9_feature_pruning
n_trials: 50
optimization_time_seconds: 687.1

best_params:
  importance_threshold: 0.0023
  max_correlation: 0.92
  min_variance_percentile: 0.03
  importance_method: gain
  aggregation: mean

metrics:
  features_before: 87
  features_after: 52
  features_removed: 35
  accuracy_before: 0.618
  accuracy_after: 0.627

removed_features:
  - wavelet_detail_3
  - aroon_down
  - stoch_d
  # ... 32 more

feature_importance_ranking:
  - feature: rsi_14
    importance: 0.082
  - feature: macd
    importance: 0.071
  # ... 50 more
```

---

## Objective Functions

### Available Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| `f1_macro` | Macro-averaged F1 score | Balanced classification (default) |
| `f1_weighted` | Weighted F1 score | Imbalanced classification |
| `accuracy` | Overall accuracy | Quick evaluation |
| `balanced_accuracy` | Balanced accuracy | Imbalanced classes |
| `roc_auc_ovr` | ROC AUC (one-vs-rest) | Probability calibration |
| `sharpe_proxy` | Returns-based Sharpe | Trading performance |

### Custom Objective Functions

```python
def custom_feature_objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    config: dict
) -> float:
    """
    Custom objective combining multiple metrics.
    """
    # Get selected features
    selected_indices = get_selected_features(trial, feature_names)
    X_subset = X[:, selected_indices]

    # Train model
    model = LGBMClassifier(n_estimators=100, verbose=-1)

    # Multi-metric cross-validation
    from sklearn.model_selection import cross_validate
    cv_results = cross_validate(
        model, X_subset, y,
        cv=3,
        scoring=['f1_macro', 'accuracy', 'roc_auc_ovr'],
        n_jobs=-1
    )

    # Weighted combination
    f1 = cv_results['test_f1_macro'].mean()
    acc = cv_results['test_accuracy'].mean()
    auc = cv_results['test_roc_auc_ovr'].mean()

    # Weights from config
    w_f1 = config.get('weight_f1', 0.5)
    w_acc = config.get('weight_accuracy', 0.3)
    w_auc = config.get('weight_auc', 0.2)

    combined_score = w_f1 * f1 + w_acc * acc + w_auc * auc

    # Feature count penalty
    n_features = len(selected_indices)
    penalty = config.get('feature_penalty', 0.001) * n_features

    return combined_score - penalty
```

---

## Pruning Callbacks

Pruning callbacks stop unpromising trials early, saving computation time.

### Median Pruner

Prunes trials worse than the median of previous trials at the same step.

```python
import optuna

# Create study with median pruner
study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=10,    # Don't prune first 10 trials
        n_warmup_steps=5,       # Wait 5 CV folds before pruning
        interval_steps=1        # Check every fold
    )
)
```

### Hyperband Pruner

Aggressive early stopping using successive halving.

```python
study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=1,         # Minimum folds before pruning
        max_resource=100,       # Maximum folds/epochs
        reduction_factor=3      # Reduction ratio per rung
    )
)
```

### Integration with Cross-Validation

```python
def objective_with_pruning(trial: optuna.Trial, X, y, config) -> float:
    """
    Objective function with per-fold pruning.
    """
    # Get features
    selected_indices = get_selected_features(trial)
    X_subset = X[:, selected_indices]

    model = LGBMClassifier(n_estimators=100, verbose=-1)

    # Manual CV loop for pruning support
    from sklearn.model_selection import StratifiedKFold
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_subset, y)):
        X_train, X_val = X_subset[train_idx], X_subset[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        from sklearn.metrics import f1_score
        fold_score = f1_score(y_val, y_pred, average='macro')
        scores.append(fold_score)

        # Report intermediate value for pruning
        trial.report(np.mean(scores), fold_idx)

        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)
```

### Pruner Configuration

```yaml
# config/optimization/feature_selection.yaml
pruner:
  type: hyperband          # Options: median, hyperband, percentile, none
  min_resource: 1          # Min CV folds before pruning
  max_resource: 100        # Max resources (folds, epochs)
  reduction_factor: 3      # Successive halving factor

  # For median pruner
  # n_startup_trials: 10
  # n_warmup_steps: 5
  # interval_steps: 1
```

### Pruning Statistics

After optimization, check pruning effectiveness:

```python
# After study.optimize()
n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

print(f"Completed: {n_completed}")
print(f"Pruned: {n_pruned}")
print(f"Pruning rate: {n_pruned / len(study.trials):.1%}")
# Typical: 30-50% pruning rate with MedianPruner
```

---

## Samplers and Search Strategies

### TPE Sampler (Default)

Tree-structured Parzen Estimator - learns from previous trials.

```python
sampler = optuna.samplers.TPESampler(
    seed=42,
    n_startup_trials=10,    # Random trials before TPE
    multivariate=True,      # Model parameter dependencies
    group=True,             # Group related parameters
    constant_liar=True      # For parallel optimization
)
```

### CMA-ES Sampler

Covariance Matrix Adaptation - good for continuous parameters.

```python
sampler = optuna.samplers.CmaEsSampler(
    seed=42,
    n_startup_trials=5,
    restart_strategy='ipop'  # Increasing population restarts
)
```

### Random Sampler

For baseline comparison or when search space is small.

```python
sampler = optuna.samplers.RandomSampler(seed=42)
```

### Sampler Configuration

```yaml
# config/optimization/feature_selection.yaml
sampler:
  type: tpe                # Options: tpe, cmaes, random, grid
  seed: 42
  n_startup_trials: 20
  multivariate: true
  group: true
  constant_liar: true      # For parallel optimization
```

---

## Per-Model Feature Optimization

Different model families benefit from different feature subsets.

### Model-Specific Strategies

| Model Family | Recommended Groups | Excluded | Feature Count |
|--------------|-------------------|----------|---------------|
| **Boosting** | All | None | 100-150 |
| **LSTM/GRU** | momentum, volatility, wavelets | microstructure, entropy | 30-60 |
| **Transformer** | momentum, volume, temporal | wavelets | 25-40 |
| **CNN** | wavelets, volatility | microstructure | 40-60 |
| **Classical** | momentum, trend, moving_avg | entropy, wavelets | 50-80 |

### Per-Model Optimization

```python
def optimize_features_for_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    model_family: str,
    n_trials: int = 100
) -> list[str]:
    """
    Optimize features specifically for a model family.
    """
    # Model-specific regularization
    if model_family in ['lstm', 'gru', 'transformer']:
        # Neural models prefer fewer features
        feature_penalty = 0.002
        max_features = 60
    elif model_family in ['xgboost', 'lightgbm', 'catboost']:
        # Boosting handles many features
        feature_penalty = 0.0005
        max_features = 150
    else:
        feature_penalty = 0.001
        max_features = 100

    config = {
        'feature_penalty': feature_penalty,
        'max_features': max_features,
        'model_family': model_family,
    }

    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Optimize
    study.optimize(
        lambda trial: feature_selection_objective(
            trial, X, y, feature_names, config
        ),
        n_trials=n_trials
    )

    return extract_selected_features(study.best_params, feature_names)
```

### Running Per-Model Optimization

```bash
# Optimize for XGBoost
python -m src.optimization.feature_selection \
    --symbol MES \
    --model-family boosting \
    --n-trials 100

# Optimize for LSTM
python -m src.optimization.feature_selection \
    --symbol MES \
    --model-family neural \
    --n-trials 100
```

---

## Configuration Reference

### Feature Selection Config

```yaml
# config/optimization/feature_selection.yaml

optimization:
  name: feature_selection
  framework: optuna
  n_trials: 100
  timeout: 14400  # 4 hours
  direction: maximize

sampler:
  type: tpe
  seed: 42
  n_startup_trials: 20
  multivariate: true
  group: true

pruner:
  type: hyperband
  min_resource: 1
  max_resource: 100
  reduction_factor: 3

search_space:
  feature_selection_mode: binary
  feature_groups:
    price_features:
      enabled: true
    volume_features:
      enabled: true
    momentum_features:
      enabled: true
    volatility_features:
      enabled: true
    trend_features:
      enabled: true
    microstructure_features:
      enabled: true

objective:
  metric: f1_macro
  direction: maximize
  regularization:
    enabled: true
    type: l1
    lambda: 0.001

evaluation:
  model: xgboost
  cv_splits: 3
  cv_type: purged_kfold
  purge_bars: 60
  embargo_bars: 60

constraints:
  min_features: 10
  max_features: 100
  required_features:
    - returns_1
    - volatility_20

output:
  save_study: true
  storage: sqlite:///experiments/optuna/feature_selection.db
  save_features: true
  features_path: config/features/selected_features.yaml

random_seed: 42
```

### Feature Pruning Config

```yaml
# config/optimization/feature_pruning.yaml

optimization:
  name: feature_pruning
  framework: optuna
  n_trials: 50
  timeout: 3600  # 1 hour
  direction: maximize

sampler:
  type: tpe
  seed: 42
  n_startup_trials: 10
  multivariate: true

pruner:
  type: median
  n_startup_trials: 5
  n_warmup_steps: 5
  interval_steps: 1

search_space:
  importance_threshold:
    type: float
    low: 0.001
    high: 0.1
    log: true

  importance_method:
    type: categorical
    choices: [permutation, shap, gain, split]

  max_correlation:
    type: float
    low: 0.85
    high: 0.99

  min_variance_percentile:
    type: float
    low: 0.01
    high: 0.10

objective:
  metric: f1_macro
  direction: maximize
  efficiency_bonus:
    enabled: true
    weight: 0.1

importance_calculation:
  model: xgboost
  n_repeats: 5  # For permutation importance

evaluation:
  cv_splits: 3
  cv_type: purged_kfold

constraints:
  min_features_retained: 20
  max_features_removed_pct: 0.7
  required_features:
    - returns_1
    - volatility_20

output:
  save_study: true
  storage: sqlite:///experiments/optuna/feature_pruning.db
  save_importance: true
  importance_path: experiments/feature_importance/

random_seed: 42
```

---

## Running Feature Optimization

### Full Pipeline

```bash
# Run Stage 8 + Stage 9 sequentially
python -m src.pipeline.run \
    --symbol MES \
    --stages 8,9 \
    --config config/pipeline.yaml
```

### Individual Stages

```bash
# Stage 8: Feature Selection
python -m src.optimization.feature_selection \
    --symbol MES \
    --n-trials 100 \
    --timeout 14400 \
    --config config/optimization/feature_selection.yaml

# Stage 9: Feature Pruning
python -m src.optimization.feature_pruning \
    --symbol MES \
    --n-trials 50 \
    --timeout 3600 \
    --config config/optimization/feature_pruning.yaml
```

### Python API

```python
from src.optimization import FeatureOptimizationPipeline

# Run both stages
pipeline = FeatureOptimizationPipeline(
    symbol='MES',
    feature_selection_trials=100,
    feature_pruning_trials=50
)

# Execute optimization
results = pipeline.run(
    X=X_train,
    y=y_train,
    X_val=X_val,
    y_val=y_val,
    feature_names=feature_names
)

# Access results
print(f"Selected features: {len(results.selected_features)}")
print(f"Pruned features: {len(results.pruned_features)}")
print(f"Best selection score: {results.selection_score:.4f}")
print(f"Best pruning score: {results.pruning_score:.4f}")
```

### Parallel Optimization

```python
import optuna

# Create study with SQLite storage for distributed optimization
study = optuna.create_study(
    study_name='feature_selection_MES',
    storage='sqlite:///experiments/optuna/feature_selection.db',
    load_if_exists=True,
    direction='maximize',
    sampler=optuna.samplers.TPESampler(
        seed=42,
        constant_liar=True  # For parallel workers
    )
)

# Run in parallel (can be on multiple machines)
study.optimize(
    objective_fn,
    n_trials=100,
    n_jobs=4,  # Parallel workers
    show_progress_bar=True
)
```

---

## Visualization and Analysis

### Optuna Dashboard

```bash
# Start Optuna Dashboard
optuna-dashboard sqlite:///experiments/optuna/feature_selection.db

# Access at http://localhost:8080
```

### Built-in Visualizations

```python
import optuna.visualization as vis

# Load study
study = optuna.load_study(
    study_name='feature_selection_MES',
    storage='sqlite:///experiments/optuna/feature_selection.db'
)

# Optimization history
fig = vis.plot_optimization_history(study)
fig.write_html('reports/optimization_history.html')

# Parameter importances
fig = vis.plot_param_importances(study)
fig.write_html('reports/param_importances.html')

# Parallel coordinate plot
fig = vis.plot_parallel_coordinate(study)
fig.write_html('reports/parallel_coordinate.html')

# Slice plot (parameter vs objective)
fig = vis.plot_slice(study)
fig.write_html('reports/slice_plot.html')

# Contour plot (2 params vs objective)
fig = vis.plot_contour(study, params=['include_momentum', 'include_volatility'])
fig.write_html('reports/contour_plot.html')
```

### Feature Importance Analysis

```python
import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance(importance_dict: dict, top_n: int = 30):
    """Plot top feature importances."""
    df = pd.DataFrame([
        {'feature': k, 'importance': v}
        for k, v in importance_dict.items()
    ]).sort_values('importance', ascending=True).tail(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(df['feature'], df['importance'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.savefig('reports/feature_importance.png', dpi=150)
    plt.close()
```

---

## Best Practices

### 1. Set Appropriate Trial Budgets

| Optimization | Recommended Trials | Time Estimate |
|--------------|-------------------|---------------|
| Feature Selection | 100 | ~30 min |
| Feature Pruning | 50 | ~15 min |
| Per-Model Selection | 50-100 | ~20-30 min each |

### 2. Use Required Features

Always require essential features that should never be excluded:

```yaml
constraints:
  required_features:
    - returns_1      # Basic price change
    - volatility_20  # Risk measure
    - volume_ratio   # Liquidity signal
```

### 3. Enable Pruning

Pruning saves 30-50% computation time:

```python
study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=5
    )
)
```

### 4. Use Purged Cross-Validation

Prevent leakage in time-series data:

```python
from src.cross_validation import PurgedKFold

cv = PurgedKFold(
    n_splits=3,
    purge_bars=60,    # Remove 60 bars after train
    embargo_bars=60   # Skip 60 bars before val
)
```

### 5. Save Study for Resumption

```python
study = optuna.create_study(
    storage='sqlite:///experiments/optuna/study.db',
    load_if_exists=True  # Resume if exists
)
```

### 6. Set Reproducibility Seeds

```yaml
random_seed: 42

sampler:
  seed: 42
```

---

## Troubleshooting

### Issue: All Trials Return Same Score

**Cause:** Search space too constrained or features too correlated.

**Solution:**
```yaml
# Relax constraints
constraints:
  min_features: 5   # Lower minimum
  max_features: 150 # Higher maximum
```

### Issue: Pruning Too Aggressive

**Cause:** Warmup too short, trials pruned before convergence.

**Solution:**
```yaml
pruner:
  n_startup_trials: 20  # More trials before pruning
  n_warmup_steps: 10    # More folds before checking
```

### Issue: Optimization Takes Too Long

**Cause:** Too many trials, slow evaluation model.

**Solution:**
```python
# Use faster evaluation model
model = LGBMClassifier(
    n_estimators=50,  # Fewer trees
    max_depth=4,      # Shallower
    verbose=-1
)

# Reduce CV folds
cv_splits = 2  # Instead of 3

# Set timeout
study.optimize(objective, n_trials=100, timeout=3600)  # 1 hour max
```

### Issue: Memory Error with SHAP

**Cause:** SHAP calculation on full dataset.

**Solution:**
```python
# Subsample for SHAP
shap_values = explainer.shap_values(X_val[:500])  # Only 500 samples
```

### Issue: Feature Selection Selects No Features

**Cause:** Penalty too high or constraints too strict.

**Solution:**
```yaml
objective:
  regularization:
    lambda: 0.0001  # Lower penalty

constraints:
  min_features: 5  # Lower minimum
```

---

## Summary

### Feature Optimization Workflow

1. **Stage 8: Feature Selection (100 trials)**
   - Binary include/exclude for feature groups
   - Reduces 162 features to ~80
   - Uses TPE sampler with Hyperband pruner

2. **Stage 9: Feature Pruning (50 trials)**
   - Importance-based individual feature removal
   - Reduces ~80 features to ~50
   - Optimizes threshold, correlation, variance

### Key Takeaways

- **Two-stage optimization** provides both breadth (group selection) and depth (individual pruning)
- **Optuna pruning callbacks** save 30-50% computation time
- **Per-model optimization** produces better features for each model family
- **Required features** ensure essential signals are never excluded
- **Purged CV** prevents information leakage in time-series evaluation

### File Reference

- Feature Selection Config: `config/optimization/feature_selection.yaml`
- Feature Pruning Config: `config/optimization/feature_pruning.yaml`
- Implementation: `src/optimization/feature_selection.py`, `src/optimization/feature_pruning.py`
- Results: `experiments/optuna/feature_selection/`, `experiments/optuna/feature_pruning/`

---

**Related Documentation:**
- [HYPERPARAMETER_TUNING.md](./HYPERPARAMETER_TUNING.md) - Stage 13 hyperparameter optimization
- [FEATURE_ENGINEERING.md](./FEATURE_ENGINEERING.md) - Stage 5 feature computation
- [PHASE_3_FEATURES.md](../implementation/PHASE_3_FEATURES.md) - Feature implementation details
- [config/optimization/README.md](../../config/optimization/README.md) - Configuration reference
