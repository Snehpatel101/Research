# Optimization Configuration

This directory contains configuration files for Optuna-based optimization across the ML pipeline.

## Overview

The ML Factory uses Optuna for hyperparameter optimization across four key areas:

| Optimization Type | Config File | Trials | Purpose |
|-------------------|-------------|--------|---------|
| Triple Barrier Labeling | `label_optimization.yaml` | 100 | Optimize labeling parameters |
| Feature Selection | `feature_selection.yaml` | 100 | Binary feature include/exclude |
| Feature Pruning | `feature_pruning.yaml` | 50 | Importance-based pruning |
| Hyperparameter Tuning | `hyperparameter.yaml` | 100/model | Per-model HP optimization |

## Folders

- `ga_results/` - Genetic algorithm results (legacy, symbol-specific)

## Configuration Files

### 1. Triple Barrier Label Optimization (`label_optimization.yaml`)

Optimizes the triple barrier labeling parameters to maximize label quality and predictability.

#### Triple Barrier Method Overview

The triple barrier method labels each sample based on which of three barriers is hit first:

```
                    Upper Barrier (Profit Target)
                    price_upper = close * (1 + ATR * upper_mult)
     ────────────────────────────────────────────────────────
              /\          /\
             /  \        /  \
     ───────/────\──────/────\───────  Entry Price (close)
           /      \    /      \
          /        \  /        \
     ──────────────────────────────────────────────────────
                    Lower Barrier (Stop Loss)
                    price_lower = close * (1 - ATR * lower_mult)

     |<─────────── Time Horizon (max bars) ──────────────>|
```

**Label Assignment:**
- **Long (+1)**: Upper barrier (profit target) hit first
- **Short (-1)**: Lower barrier (stop loss) hit first
- **Neutral (0)**: Time barrier reached (neither price barrier hit)

#### Optimized Parameters

| Parameter | Description | Search Space | Recommended |
|-----------|-------------|--------------|-------------|
| `upper_mult` | Upper barrier ATR multiplier (profit target) | [0.5, 4.0] | 2.0 |
| `lower_mult` | Lower barrier ATR multiplier (stop loss) | [0.5, 4.0] | 1.5 |
| `horizon` | Maximum holding period (bars) | [5, 60] | 20 |
| `atr_period` | ATR calculation period | [7, 28] | 14 |

#### Optimization Objective

The objective function is a weighted combination:

```
score = 0.4 * class_balance + 0.3 * barrier_hit_rate + 0.3 * model_f1
```

| Component | Weight | Description |
|-----------|--------|-------------|
| `class_balance` | 40% | Minimize deviation from balanced 33% per class |
| `barrier_hit_rate` | 30% | Maximize profit/loss hits vs timeout labels |
| `model_f1` | 30% | Downstream F1 score with XGBoost validation |

#### Transaction Cost Integration

Labels are evaluated considering realistic transaction costs:
- Commission: $1.00 round-trip
- Slippage: 2-3 basis points
- Minimum profitable move threshold

#### Related Configuration

- `config/labeling.yaml` - Static labeling parameters (symbol-specific barriers, horizons, quality weights)
- `docs/implementation/PHASE_4_LABELING.md` - Full implementation details

**Example Configuration:**
```yaml
# label_optimization.yaml
optimization:
  name: triple_barrier_optimization
  framework: optuna
  n_trials: 100
  timeout: 7200  # 2 hours

sampler:
  type: tpe
  seed: 42
  n_startup_trials: 10

pruner:
  type: median
  n_startup_trials: 5
  n_warmup_steps: 10

search_space:
  upper_mult:
    type: float
    low: 0.5
    high: 4.0
    log: false
  lower_mult:
    type: float
    low: 0.5
    high: 4.0
    log: false
  horizon:
    type: int
    low: 5
    high: 60
  atr_period:
    type: int
    low: 7
    high: 28

objective:
  metric: label_quality_score
  direction: maximize
  components:
    - name: class_balance
      weight: 0.4
    - name: barrier_hit_rate
      weight: 0.3
    - name: model_f1
      weight: 0.3

transaction_costs:
  enabled: true
  commission: 1.00
  slippage_bps: 2.0

constraints:
  min_samples_per_class: 1000
  max_class_imbalance: 0.4  # Max deviation from 33%
```

### 2. Feature Selection Optimization (`feature_selection.yaml`)

Comprehensive feature selection optimization using multiple Optuna-driven strategies.

**Reference Guide:** `docs/guides/FEATURE_SELECTION_OPTIMIZATION.md` - Comprehensive guide with examples

**Selection Strategies (in priority order):**

| Strategy | Description | Search Space |
|----------|-------------|--------------|
| **Binary Group Selection** | Select entire feature groups (momentum, volatility, etc.) | 10 binary decisions |
| **Binary Individual Selection** | Fine-grained selection within groups | ~100 binary decisions |
| **Importance-Based Selection** | Select features above importance threshold | threshold, method |
| **Recursive Feature Elimination (RFE)** | Iteratively remove least important features | n_features, step |
| **Correlation-Based Selection** | Remove highly correlated features | max_correlation |

**Importance Methods:**

| Method | Description | Speed | Model Dependency |
|--------|-------------|-------|------------------|
| `gain` | Total gain from tree splits | Fast | Tree-based only |
| `split` | Number of times feature used in splits | Fast | Tree-based only |
| `permutation` | Performance drop when feature shuffled | Medium | Any model |
| `shap` | SHAP value magnitudes | Slow | Any model |
| `mutual_info` | Mutual information with target | Medium | Model-agnostic |
| `lasso` | LASSO coefficient magnitude | Fast | Linear models |

**Recursive Feature Elimination (RFE):**

RFE iteratively removes the least important features until the desired number remains:

```python
# RFE with Optuna optimization
from sklearn.feature_selection import RFECV

def rfe_objective(trial: optuna.Trial, X, y) -> float:
    n_features = trial.suggest_int('n_features', 20, 80)
    step = trial.suggest_categorical('step', [1, 5, 10])

    estimator = LGBMClassifier(n_estimators=100, verbose=-1)

    rfe = RFECV(
        estimator=estimator,
        step=step,
        min_features_to_select=n_features,
        cv=3,
        scoring='f1_macro'
    )

    rfe.fit(X, y)
    return rfe.cv_results_['mean_test_score'].max()
```

**Example Configuration:**
```yaml
# feature_selection.yaml
optimization:
  name: feature_selection
  framework: optuna
  n_trials: 100
  timeout: 14400  # 4 hours

sampler:
  type: tpe
  seed: 42
  n_startup_trials: 20

pruner:
  type: hyperband
  min_resource: 1
  max_resource: 100
  reduction_factor: 3

# Multiple selection strategies
selection_strategies:
  binary_group_selection:
    enabled: true
    priority: 1
  importance_based_selection:
    enabled: true
    priority: 2
  recursive_feature_elimination:
    enabled: true
    priority: 3
  correlation_based_selection:
    enabled: true
    priority: 4

# Importance-based search space
importance_based:
  importance_method:
    type: categorical
    choices: [gain, split, permutation, shap, mutual_info, lasso]
  importance_threshold:
    type: float
    low: 0.001
    high: 0.1
    log: true

# RFE search space
recursive_feature_elimination:
  n_features_to_select:
    type: int
    low: 20
    high: 80
  step:
    type: categorical
    choices: [1, 5, 10, 0.1, 0.2]

# Correlation-based search space
correlation_based:
  max_correlation:
    type: float
    low: 0.80
    high: 0.99
  correlation_method:
    type: categorical
    choices: [pearson, spearman, kendall]

# Feature groups (binary selection)
search_space:
  feature_selection_mode: binary
  feature_groups:
    price_features: true
    volume_features: true
    momentum_features: true
    volatility_features: true
    microstructure_features: true

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

constraints:
  min_features: 10
  max_features: 100
  required_features:
    - returns_1
    - volatility_20
```

### 3. Feature Pruning Optimization (`feature_pruning.yaml`)

Importance-based feature pruning to reduce dimensionality.

**Reference Guide:** `docs/guides/FEATURE_SELECTION_OPTIMIZATION.md#stage-9-feature-pruning-with-optuna`

**Pruning Strategy:**
- Compute feature importance scores
- Optimize importance threshold
- Remove features below threshold
- Validate with model performance

**Example Configuration:**
```yaml
# feature_pruning.yaml
optimization:
  name: feature_pruning
  framework: optuna
  n_trials: 50
  timeout: 3600  # 1 hour

sampler:
  type: tpe
  seed: 42

search_space:
  importance_threshold:
    type: float
    low: 0.001
    high: 0.1
    log: true

  importance_method:
    type: categorical
    choices:
      - permutation
      - shap
      - gain
      - split

objective:
  metric: f1_macro
  direction: maximize

importance_calculation:
  model: xgboost
  n_repeats: 5  # For permutation importance

evaluation:
  cv_splits: 3

constraints:
  min_features_retained: 20
  max_features_removed_pct: 0.7
```

### 4. Hyperparameter Optimization (`hyperparameter.yaml`)

Per-model hyperparameter tuning with Optuna.

**Model-Specific Search Spaces:**

#### Boosting Models (XGBoost, LightGBM, CatBoost)
```yaml
boosting_search_space:
  n_estimators:
    type: int
    low: 100
    high: 2000
    step: 100
  max_depth:
    type: int
    low: 3
    high: 12
  learning_rate:
    type: float
    low: 0.001
    high: 0.3
    log: true
  subsample:
    type: float
    low: 0.5
    high: 1.0
  colsample_bytree:
    type: float
    low: 0.5
    high: 1.0
  reg_alpha:
    type: float
    low: 1e-8
    high: 10.0
    log: true
  reg_lambda:
    type: float
    low: 1e-8
    high: 10.0
    log: true
```

#### Neural Models (LSTM, GRU, TCN, Transformer)
```yaml
neural_search_space:
  hidden_size:
    type: categorical
    choices: [64, 128, 256, 512]
  num_layers:
    type: int
    low: 1
    high: 4
  dropout:
    type: float
    low: 0.0
    high: 0.5
  learning_rate:
    type: float
    low: 1e-5
    high: 1e-2
    log: true
  batch_size:
    type: categorical
    choices: [32, 64, 128, 256]
  weight_decay:
    type: float
    low: 1e-6
    high: 1e-2
    log: true
```

#### Classical Models (Random Forest, SVM, Logistic)
```yaml
classical_search_space:
  # Random Forest
  rf_n_estimators:
    type: int
    low: 100
    high: 1000
  rf_max_depth:
    type: int
    low: 5
    high: 30
  rf_min_samples_split:
    type: int
    low: 2
    high: 20

  # SVM
  svm_C:
    type: float
    low: 1e-3
    high: 1e3
    log: true
  svm_gamma:
    type: categorical
    choices: [scale, auto, 0.001, 0.01, 0.1, 1.0]
```

**Example Configuration:**
```yaml
# hyperparameter.yaml
optimization:
  name: hyperparameter_optimization
  framework: optuna
  n_trials: 100
  timeout: 7200  # 2 hours per model

sampler:
  type: tpe
  seed: 42
  n_startup_trials: 10
  multivariate: true

pruner:
  type: hyperband
  min_resource: 10
  max_resource: 100
  reduction_factor: 3

early_stopping:
  enabled: true
  patience: 20
  min_delta: 0.001

objective:
  metric: f1_macro
  direction: maximize

evaluation:
  cv_type: purged_kfold
  cv_splits: 5
  purge_bars: 60
  embargo_bars: 60

# Per-model configurations
models:
  xgboost:
    n_trials: 100
    search_space: boosting_search_space
    early_stopping_rounds: 50

  lightgbm:
    n_trials: 100
    search_space: boosting_search_space
    early_stopping_rounds: 50

  catboost:
    n_trials: 100
    search_space: boosting_search_space
    early_stopping_rounds: 50

  lstm:
    n_trials: 100
    search_space: neural_search_space
    max_epochs: 100

  transformer:
    n_trials: 100
    search_space: neural_search_space
    max_epochs: 100

  random_forest:
    n_trials: 50
    search_space: classical_search_space
```

## Usage

### Running Label Optimization
```python
from src.labeling.optimization import optimize_triple_barrier

best_params = optimize_triple_barrier(
    data=df,
    config_path="config/optimization/label_optimization.yaml",
    symbol="MES"
)
print(f"Best params: {best_params}")
```

### Running Feature Selection
```python
from src.features.optimization import optimize_feature_selection

selected_features = optimize_feature_selection(
    X=features,
    y=labels,
    config_path="config/optimization/feature_selection.yaml"
)
print(f"Selected {len(selected_features)} features")
```

### Running Hyperparameter Optimization
```bash
python scripts/tune_model.py \
    --model xgboost \
    --horizon 20 \
    --config config/optimization/hyperparameter.yaml \
    --n-trials 100
```

### CLI Commands
```bash
# Optimize labels
python scripts/optimize_labels.py --config config/optimization/label_optimization.yaml

# Optimize feature selection
python scripts/optimize_features.py --mode selection --config config/optimization/feature_selection.yaml

# Optimize feature pruning
python scripts/optimize_features.py --mode pruning --config config/optimization/feature_pruning.yaml

# Optimize hyperparameters
python scripts/tune_model.py --model all --config config/optimization/hyperparameter.yaml
```

## Optuna Dashboard

Visualize optimization progress with Optuna Dashboard:

```bash
# Start dashboard
optuna-dashboard sqlite:///experiments/optuna/study.db

# Or use in-memory storage
python -c "
import optuna
from optuna_dashboard import run_server

storage = optuna.storages.InMemoryStorage()
run_server(storage, host='127.0.0.1', port=8080)
"
```

## Best Practices

### 1. Trial Budget Allocation
- Label optimization: 100 trials (quick evaluation)
- Feature selection: 100 trials (binary search space)
- Feature pruning: 50 trials (simple threshold)
- Hyperparameter tuning: 100 trials per model

### 2. Reproducibility
- Always set `seed` in sampler configuration
- Use deterministic model training
- Store optimization history

### 3. Pruning Strategy
- Use MedianPruner for label optimization
- Use HyperbandPruner for hyperparameter tuning
- Set appropriate warmup steps

### 4. Resource Management
- Set timeout to prevent runaway trials
- Use early stopping for neural models
- Monitor memory usage with large search spaces

## Related Docs

- [Feature Selection Optimization Guide](../../docs/guides/FEATURE_SELECTION_OPTIMIZATION.md) - **NEW** Comprehensive guide
- [Hyperparameter Tuning Guide](../../docs/guides/HYPERPARAMETER_TUNING.md)
- [Feature Engineering Guide](../../docs/guides/FEATURE_ENGINEERING.md)
- [Unified Pipeline Architecture](../../docs/implementation/UNIFIED_PIPELINE_ARCHITECTURE.md)
- [Pipeline Configuration](../pipeline/README.md)

---

*Last Updated: 2026-01-18*
