# Hyperparameter Optimization Guide

**Purpose:** Comprehensive guide for hyperparameter optimization using Genetic Algorithms (GA) and Optuna
**Audience:** ML engineers, quant analysts
**Last Updated:** 2025-12-30

---

## Table of Contents

1. [Optimization Philosophy](#optimization-philosophy)
2. [Phase 1: GA for Label Parameters](#phase-1-ga-for-label-parameters)
3. [Phase 2: Optuna for Model Hyperparameters](#phase-2-optuna-for-model-hyperparameters)
4. [Search Space Design](#search-space-design)
5. [Objective Functions](#objective-functions)
6. [Pruning Strategies](#pruning-strategies)
7. [Multi-Objective Optimization](#multi-objective-optimization)
8. [Integration with Cross-Validation](#integration-with-cross-validation)
9. [Computational Budget](#computational-budget)
10. [Adding New Parameters](#adding-new-parameters)

---

## Optimization Philosophy

### When to Use GA vs Optuna vs Grid Search

| Method | Use Case | Pros | Cons | Typical Use |
|--------|----------|------|------|-------------|
| **Genetic Algorithm (GA)** | Label parameters | Handles complex interactions, finds asymmetric barriers | Slow (1-2 hours) | Triple-barrier profit/stop optimization |
| **Optuna (TPE)** | Model hyperparameters | Fast, adaptive, pruning | May get stuck in local optima | XGBoost, LSTM hyperparameters |
| **Grid Search** | Small discrete spaces | Exhaustive, deterministic | Exponential growth | Testing 2-3 discrete options |
| **Random Search** | Baseline | Fast, simple | Not adaptive | Quick baseline before Optuna |

### Key Principles

1. **Two-stage optimization:**
   - **Stage 1 (Phase 1):** Optimize label parameters (profit_target, stop_loss) using GA
   - **Stage 2 (Phase 2):** Optimize model hyperparameters using Optuna

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

## Phase 1: GA for Label Parameters

### Overview

**Goal:** Optimize triple-barrier labeling parameters to maximize risk-adjusted returns.

**Parameters to optimize:**
- `profit_target`: Profit-taking threshold (% move)
- `stop_loss`: Stop-loss threshold (% move)
- `min_return`: Minimum return to label as long/short (noise filter)

**Objective:** Maximize Sharpe ratio with transaction cost penalty.

### Current Implementation

Located in `src/phase1/stages/ga_optimize/`:

```
src/phase1/stages/ga_optimize/
├── optimizer.py         # GA optimizer using Optuna
├── objective.py         # Objective function (Sharpe - cost penalty)
├── bounds.py            # Parameter bounds per symbol
└── run.py               # CLI entry point
```

### Parameter Bounds by Symbol

```python
# src/phase1/stages/ga_optimize/bounds.py

SYMBOL_BOUNDS = {
    'MES': {
        'profit_target': (0.001, 0.01),  # 0.1% to 1.0% (micro S&P is less volatile)
        'stop_loss': (0.0005, 0.008),    # Asymmetric: 0.05% to 0.8%
        'min_return': (0.0001, 0.002),   # Noise filter: 0.01% to 0.2%
        'horizon_multiplier': (1.0, 3.0) # Horizon scaling
    },
    'MGC': {
        'profit_target': (0.002, 0.015), # 0.2% to 1.5% (gold more volatile)
        'stop_loss': (0.001, 0.01),
        'min_return': (0.0002, 0.003),
        'horizon_multiplier': (1.0, 3.0)
    }
}
```

**Why asymmetric bounds?**
- `stop_loss` max < `profit_target` max: Encourage positive risk/reward ratio
- MES vs MGC: Gold has higher volatility → wider bounds

### Objective Function

```python
# src/phase1/stages/ga_optimize/objective.py
import optuna
import numpy as np
from src.phase1.stages.labeling.triple_barrier import apply_triple_barrier_labels
from src.metrics.sharpe import compute_sharpe_ratio

def objective_function(trial: optuna.Trial, df: pd.DataFrame, config: dict) -> float:
    """
    Objective function for GA optimization.

    Args:
        trial: Optuna trial
        df: OHLCV DataFrame with returns
        config: Config with symbol, horizons, etc.

    Returns:
        Negative Sharpe ratio (Optuna minimizes, we want to maximize Sharpe)
    """
    # Sample parameters
    profit_target = trial.suggest_float('profit_target', *config['bounds']['profit_target'])
    stop_loss = trial.suggest_float('stop_loss', *config['bounds']['stop_loss'])
    min_return = trial.suggest_float('min_return', *config['bounds']['min_return'])

    # Apply triple-barrier labels
    labels_df = apply_triple_barrier_labels(
        df,
        profit_target=profit_target,
        stop_loss=stop_loss,
        min_return=min_return,
        horizons=config['horizons']
    )

    # Simulate trading strategy
    positions = labels_df['label']  # 0 (short), 1 (neutral), 2 (long)
    returns = df['returns'].values

    # Map labels to positions: 0→-1, 1→0, 2→+1
    position_map = {0: -1, 1: 0, 2: 1}
    positions_numeric = positions.map(position_map).values

    # Strategy returns
    strategy_returns = positions_numeric * returns

    # Transaction costs
    # Cost when position changes: |pos[t] - pos[t-1]| * cost_per_trade
    position_changes = np.abs(np.diff(positions_numeric, prepend=0))
    cost_per_trade = config.get('cost_per_trade', 0.0002)  # 2 bps
    transaction_costs = position_changes * cost_per_trade

    # Net returns
    net_returns = strategy_returns - transaction_costs

    # Sharpe ratio (annualized)
    sharpe = compute_sharpe_ratio(net_returns, annualization_factor=252)

    # Penalty for excessive trading
    turnover = position_changes.sum() / len(positions)
    turnover_penalty = max(0, turnover - 0.5) * 0.5  # Penalize if turnover > 50%

    # Final objective (negative because Optuna minimizes)
    return -(sharpe - turnover_penalty)
```

### Running GA Optimization

```bash
# Optimize label parameters for MES
python -m src.phase1.stages.ga_optimize.run \
    --symbol MES \
    --horizons 5,10,15,20 \
    --n-trials 100 \
    --timeout 7200  # 2 hours
```

**Output:**

```yaml
# Saved to: experiments/ga_optimize/MES_20250101_120000/best_params.yaml
symbol: MES
best_params:
  profit_target: 0.0065
  stop_loss: 0.0043
  min_return: 0.0008
  horizon_multiplier: 2.1

metrics:
  sharpe_ratio: 1.87
  win_rate: 0.523
  avg_trade_return: 0.00043
  turnover: 0.38
  num_trials: 100
  optimization_time: 6834.2  # seconds
```

### Integration with Pipeline

After GA optimization, the best parameters are saved and used in the labeling stage:

```python
# src/phase1/stages/final_labels/run.py
from src.phase1.stages.ga_optimize.results import load_best_params

def run_final_labels_stage(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    # Load GA-optimized parameters
    best_params = load_best_params(symbol=config.symbol)

    # Apply triple-barrier labels with optimized params
    labels_df = apply_triple_barrier_labels(
        df,
        profit_target=best_params['profit_target'],
        stop_loss=best_params['stop_loss'],
        min_return=best_params['min_return'],
        horizons=config.label_horizons
    )

    return labels_df
```

### Validation: Prevent Overfitting on Train Set

**Problem:** GA optimizes on training data → may overfit.

**Solution:** Use walk-forward validation during GA:

```python
def objective_function_walk_forward(trial, df, config):
    """
    Objective function with walk-forward validation.

    Splits data into 3 periods:
    - Train (60%): Fit labels
    - Val (20%): Evaluate Sharpe
    - Test (20%): Holdout for final validation
    """
    # Sample parameters
    profit_target = trial.suggest_float('profit_target', ...)
    stop_loss = trial.suggest_float('stop_loss', ...)

    # Apply labels
    labels_df = apply_triple_barrier_labels(df, profit_target, stop_loss, ...)

    # Walk-forward split
    n = len(labels_df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    val_returns = labels_df.iloc[train_end:val_end]['net_returns']

    # Evaluate on validation set ONLY
    sharpe_val = compute_sharpe_ratio(val_returns)

    return -sharpe_val
```

---

## Phase 2: Optuna for Model Hyperparameters

### Overview

**Goal:** Optimize model hyperparameters using Optuna's Tree-structured Parzen Estimator (TPE).

**Integration:** `src/cross_validation/cv_runner.py`

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

### Budget Allocation by Model Family

| Model Family | Trials | Time per Trial | Total Time | Recommended Budget |
|--------------|--------|----------------|------------|-------------------|
| **Boosting** | 100 | 2 min | 3-4 hours | 100 trials |
| **Neural (LSTM/GRU)** | 50 | 10 min | 8-10 hours | 50 trials |
| **Transformers** | 30 | 20-30 min | 10-15 hours | 30 trials |
| **CNN** | 50 | 10 min | 8-10 hours | 50 trials |
| **Classical (RF)** | 50 | 1 min | 1 hour | 50 trials |
| **Classical (SVM)** | 30 | 5 min | 2-3 hours | 30 trials |

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

### Optimization Workflow

1. **Phase 1 (GA):** Optimize label parameters (profit_target, stop_loss) → 1-2 hours
2. **Phase 2 (Optuna):** Optimize model hyperparameters → 3-15 hours depending on model
3. **Nested CV:** Use inner CV for tuning, outer CV for evaluation
4. **Pruning:** Use MedianPruner to save ~30-50% computation
5. **Multi-objective:** Optimize Sharpe vs drawdown for risk-adjusted models

### Key Takeaways

- **GA for label params:** Handles complex interactions, asymmetric barriers
- **Optuna for model params:** Fast, adaptive, supports pruning
- **Nested CV prevents overfitting:** Inner CV tunes, outer CV evaluates
- **Symbol-specific optimization:** MES ≠ MGC, optimize separately
- **Transaction cost penalties:** Prevents over-trading
- **Computational budget:** 100 trials for boosting, 50 for neural, 30 for transformers

### File Paths Reference

- GA optimization: `src/phase1/stages/ga_optimize/`
- Optuna integration: `src/cross_validation/cv_runner.py`
- Search spaces: `src/cross_validation/param_spaces.py`
- Nested CV: `src/cross_validation/nested_cv.py`

---

**Next steps:** After hyperparameter optimization, evaluate models using walk-forward validation and CPCV/PBO to ensure robustness.
