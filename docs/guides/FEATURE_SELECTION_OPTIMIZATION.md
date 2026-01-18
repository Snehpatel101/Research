# Feature Selection Optimization with Optuna

**Purpose:** Comprehensive guide for automated feature selection and pruning using Optuna
**Audience:** ML engineers, data scientists, quant analysts
**Last Updated:** 2026-01-18

---

## Table of Contents

1. [Overview](#overview)
2. [Feature Selection Pipeline (Stages 8-9)](#feature-selection-pipeline-stages-8-9)
3. [Stage 8: Feature Selection with Optuna](#stage-8-feature-selection-with-optuna)
4. [Stage 9: Feature Pruning with Optuna](#stage-9-feature-pruning-with-optuna)
5. [Selection Strategies](#selection-strategies)
6. [Per-Model Feature Selection](#per-model-feature-selection)
7. [Running Feature Selection](#running-feature-selection)
8. [Interpreting Results](#interpreting-results)
9. [Best Practices](#best-practices)
10. [Integration with Pipeline](#integration-with-pipeline)

---

## Overview

### Why Automated Feature Selection?

**Problem:** With 162 engineered features across 12 feature families, not all features are useful for every model. Irrelevant features increase:
- Training time
- Overfitting risk
- Model complexity
- Memory usage

**Solution:** Use Optuna to automatically select the optimal feature subset for each model family, maximizing predictive performance while minimizing feature count.

### Two-Stage Approach

The ML Factory uses a **hierarchical two-stage feature optimization** approach:

| Stage | Name | Trials | Goal | Output |
|-------|------|--------|------|--------|
| **Stage 8** | Feature Selection | 100 | Select feature groups/individuals via binary decisions | Feature mask (typically 60-100 features retained) |
| **Stage 9** | Feature Pruning | 50 | Remove low-importance features via thresholds | Pruned mask (typically 30-60 features retained) |

**Why two stages?**
- **Stage 8** performs coarse filtering at the group level (fast, reduces search space)
- **Stage 9** performs fine-grained pruning at the individual level (slower, but more precise)
- Hierarchical approach is more efficient than single-stage optimization

### Feature Selection vs Feature Engineering

**Feature Engineering** (Stage 5):
- Generates 162 features from raw OHLCV
- Includes momentum, volatility, volume, trend, microstructure, wavelets, etc.
- Creates features that *might* be useful

**Feature Selection** (Stages 8-9):
- Selects subset of engineered features
- Removes redundant, noisy, or irrelevant features
- Optimizes for model performance, not feature count

**Both are necessary** for optimal model performance.

---

## Feature Selection Pipeline (Stages 8-9)

### Data Flow

```
Stage 5: Features (162 features)
         │
         ▼
┌─────────────────────────────────────────┐
│  Stage 8: Feature Selection (Optuna)    │
│  100 trials                              │
│  Binary include/exclude decisions        │
│                                          │
│  Search Space:                           │
│  - Binary group selection (10 groups)   │
│  - Binary individual selection (162)     │
│  - RFE (recursive elimination)           │
│  - Importance-based selection            │
│  - Correlation-based filtering           │
└─────────────┬───────────────────────────┘
              ▼
         Selected Features
         (typically 60-100)
              │
              ▼
┌─────────────────────────────────────────┐
│  Stage 9: Feature Pruning (Optuna)      │
│  50 trials                               │
│  Importance-based removal                │
│                                          │
│  Search Space:                           │
│  - Importance threshold (0.001-0.1)     │
│  - Max correlation (0.80-0.99)          │
│  - Min variance percentile (0.01-0.15)  │
│  - Importance method (gain/shap/perm)   │
└─────────────┬───────────────────────────┘
              ▼
         Pruned Features
         (typically 30-60)
              │
              ▼
    Stage 10: Splits
```

### Integration with 16-Stage Pipeline

```
Stage 1:  Ingestion
Stage 2:  Cleaning
Stage 3:  Sessions
Stage 4:  MTF Upscaling
Stage 5:  Features (162 indicators)
Stage 6:  Regime Detection
Stage 7:  OPTUNA Label Optimization (100 trials)
Stage 8:  OPTUNA Feature Selection (100 trials) ← THIS GUIDE
Stage 9:  OPTUNA Feature Pruning (50 trials)    ← THIS GUIDE
Stage 10: Splits
Stage 11: Scaling
Stage 12: Adaptation
Stage 13: OPTUNA Hyperparameter Optimization
Stage 14: Training
Stage 15: Stacking
Stage 16: Bundling
```

---

## Stage 8: Feature Selection with Optuna

### Overview

**Goal:** Select optimal feature subset using binary include/exclude decisions per feature group or individual feature.

**Trial Budget:** 100 trials
**Typical Runtime:** 15-30 minutes
**Search Strategy:** TPE (Tree-structured Parzen Estimator)

### Feature Groups

Features are organized into 10 logical groups to reduce search space:

```python
FEATURE_GROUPS = {
    'momentum': [
        'rsi_14', 'rsi_7', 'rsi_21',
        'macd', 'macd_signal', 'macd_hist',
        'cci_20', 'stoch_k', 'stoch_d', 'williams_r',
        'roc_10', 'roc_20', 'mfi_14',
        'adx_14', 'di_plus', 'di_minus'
    ],  # ~16 features

    'volatility': [
        'atr_14', 'atr_7', 'atr_21',
        'bollinger_upper', 'bollinger_lower', 'bollinger_width', 'bollinger_position',
        'keltner_upper', 'keltner_lower', 'keltner_width',
        'historical_vol', 'parkinson_vol', 'gk_vol'
    ],  # ~13 features

    'volume': [
        'volume', 'volume_ratio', 'volume_ma_ratio',
        'obv', 'vwap', 'vwap_distance',
        'mfi_14', 'dollar_volume', 'twap'
    ],  # ~9 features

    'trend': [
        'sma_10', 'sma_20', 'sma_50',
        'ema_10', 'ema_20', 'ema_26',
        'adx_14', 'aroon_up', 'aroon_down',
        'supertrend', 'di_plus', 'di_minus'
    ],  # ~12 features

    'moving_avg': [
        'price_to_sma10', 'price_to_sma20', 'price_to_sma50',
        'sma_crossover_10_20', 'sma_crossover_20_50',
        'ema_crossover_12_26'
    ],  # ~6 features

    'microstructure': [
        'order_flow_imbalance', 'amihud_illiquidity',
        'roll_spread', 'corwin_schultz', 'kyle_lambda',
        'effective_spread', 'realized_spread'
    ],  # ~7 features

    'wavelets': [
        'wavelet_trend_1h', 'wavelet_detail_30m', 'wavelet_detail_15m',
        'wavelet_detail_5m', 'wavelet_energy',
        # ... (16 total wavelet features)
    ],  # ~16 features

    'temporal': [
        'hour_sin', 'hour_cos',
        'day_of_week_sin', 'day_of_week_cos',
        'session_progress', 'time_to_close'
    ],  # ~6 features

    'regime': [
        'volatility_regime', 'trend_regime',
        'composite_regime', 'regime_strength'
    ],  # ~4 features

    'entropy': [
        'shannon_entropy', 'approx_entropy',
        'sample_entropy', 'hurst_exponent'
    ],  # ~4 features
}
# Total: ~100 base features + ~60 MTF features (if enabled) = 162 features
```

### Selection Strategies

Stage 8 supports **5 selection strategies**, enabled via configuration:

#### 1. Binary Group Selection (Default)

**Concept:** Include or exclude entire feature groups.

**Search Space:** 10 binary decisions (one per group)

**Advantages:**
- Fast (only 2^10 = 1,024 possible combinations)
- Interpretable (group-level decisions)
- Prevents removing all features from a critical group

**Example Trial:**
```python
def binary_group_objective(trial: optuna.Trial, X, y, feature_names):
    selected_groups = {}
    for group_name in FEATURE_GROUPS.keys():
        selected_groups[group_name] = trial.suggest_categorical(
            f'include_{group_name}',
            [True, False]
        )

    # Build feature mask
    selected_features = []
    for group_name, include in selected_groups.items():
        if include:
            selected_features.extend(FEATURE_GROUPS[group_name])

    # Get indices
    selected_indices = [
        i for i, name in enumerate(feature_names)
        if any(feat in name for feat in selected_features)
    ]

    if len(selected_indices) < 10:
        return 0.0  # Too few features

    # Evaluate
    X_subset = X[:, selected_indices]
    model = LGBMClassifier(n_estimators=100, max_depth=6, verbose=-1)
    scores = cross_val_score(model, X_subset, y, cv=3, scoring='f1_macro')

    return np.mean(scores)
```

**Typical Result:**
```yaml
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
  entropy: false

features_retained: 87
features_removed: 75
cv_score: 0.632
```

#### 2. Binary Individual Selection

**Concept:** Include or exclude each feature individually.

**Search Space:** 162 binary decisions (one per feature)

**Advantages:**
- Fine-grained control
- Can remove specific noisy features within a group

**Disadvantages:**
- Huge search space (2^162 combinations)
- Slower convergence
- May remove all features from a group

**Example Trial:**
```python
def binary_individual_objective(trial: optuna.Trial, X, y, feature_names):
    feature_mask = []
    for i, feature_name in enumerate(feature_names):
        include = trial.suggest_categorical(f'feat_{i}', [True, False])
        feature_mask.append(include)

    # Ensure minimum features
    if sum(feature_mask) < 20:
        return 0.0

    X_selected = X[:, feature_mask]
    model = LGBMClassifier(n_estimators=100, verbose=-1)
    scores = cross_val_score(model, X_selected, y, cv=3, scoring='f1_macro')

    return np.mean(scores)
```

**Recommendation:** Use binary group selection first, then optionally run individual selection for fine-tuning.

#### 3. Recursive Feature Elimination (RFE)

**Concept:** Iteratively remove the least important features until reaching the desired count.

**Search Space:**
- `n_features_to_select`: 20-80
- `step`: 1, 5, 10, or percentage (0.1, 0.2)
- `estimator`: xgboost, lightgbm, random_forest, logistic

**Advantages:**
- Model-aware (uses feature importance)
- Proven method (sklearn standard)
- Finds optimal subset size automatically (with RFECV)

**Example:**
```python
from sklearn.feature_selection import RFECV

def rfe_objective(trial: optuna.Trial, X, y):
    n_features = trial.suggest_int('n_features_to_select', 20, 80)
    step = trial.suggest_categorical('step', [1, 5, 10])
    estimator_name = trial.suggest_categorical(
        'estimator', ['xgboost', 'lightgbm', 'random_forest']
    )

    # Create estimator
    if estimator_name == 'lightgbm':
        estimator = LGBMClassifier(n_estimators=100, verbose=-1)
    elif estimator_name == 'xgboost':
        from xgboost import XGBClassifier
        estimator = XGBClassifier(n_estimators=100, verbosity=0)
    else:
        from sklearn.ensemble import RandomForestClassifier
        estimator = RandomForestClassifier(n_estimators=100, n_jobs=-1)

    # RFE with cross-validation
    rfe = RFECV(
        estimator=estimator,
        step=step,
        min_features_to_select=n_features,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1
    )

    rfe.fit(X, y)

    # Return best CV score
    best_score = rfe.cv_results_['mean_test_score'].max()
    return best_score
```

**Typical Result:**
```yaml
best_params:
  n_features_to_select: 52
  step: 5
  estimator: lightgbm

optimal_n_features: 52
cv_score: 0.641
```

#### 4. Importance-Based Selection

**Concept:** Select features above an importance threshold.

**Search Space:**
- `importance_threshold`: 0.001-0.1 (log scale)
- `importance_method`: gain, split, permutation, shap, mutual_info, lasso
- `top_k_features`: 20-100 (optional)
- `aggregation`: mean, median, min (for CV folds)

**Advantages:**
- Fast (single model training + importance calculation)
- Multiple importance methods available
- Threshold is interpretable

**Example:**
```python
def importance_objective(trial: optuna.Trial, X, y, X_val, y_val):
    threshold = trial.suggest_float('importance_threshold', 0.001, 0.1, log=True)
    method = trial.suggest_categorical(
        'importance_method',
        ['gain', 'split', 'permutation', 'shap']
    )

    # Train model
    model = LGBMClassifier(n_estimators=200, max_depth=8, verbose=-1)
    model.fit(X, y)

    # Compute importance
    if method == 'gain':
        importances = model.booster_.feature_importance(importance_type='gain')
    elif method == 'split':
        importances = model.booster_.feature_importance(importance_type='split')
    elif method == 'permutation':
        from sklearn.inspection import permutation_importance
        result = permutation_importance(
            model, X_val, y_val, n_repeats=5, random_state=42
        )
        importances = result.importances_mean
    else:  # shap
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val[:5000])
        if isinstance(shap_values, list):
            importances = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            importances = np.abs(shap_values).mean(axis=0)

    # Normalize
    importances = importances / (importances.sum() + 1e-8)

    # Create mask
    mask = importances >= threshold

    if mask.sum() < 10:
        return 0.0

    # Evaluate
    X_selected = X[:, mask]
    X_val_selected = X_val[:, mask]

    model_eval = LGBMClassifier(n_estimators=100, verbose=-1)
    model_eval.fit(X_selected, y)
    score = model_eval.score(X_val_selected, y_val)

    return score
```

**Typical Result:**
```yaml
best_params:
  importance_threshold: 0.0032
  importance_method: gain

features_retained: 64
cv_score: 0.638
```

#### 5. Correlation-Based Selection

**Concept:** Remove highly correlated features to reduce redundancy.

**Search Space:**
- `max_correlation`: 0.80-0.99
- `correlation_method`: pearson, spearman, kendall
- `keep_criterion`: higher_importance, higher_variance, first_in_list

**Advantages:**
- Removes redundant features
- Improves model interpretability
- Reduces multicollinearity (important for linear models)

**Example:**
```python
def correlation_objective(trial: optuna.Trial, X, y, feature_names, base_importances):
    max_corr = trial.suggest_float('max_correlation', 0.80, 0.99)
    corr_method = trial.suggest_categorical(
        'correlation_method', ['pearson', 'spearman']
    )
    keep_criterion = trial.suggest_categorical(
        'keep_criterion',
        ['higher_importance', 'higher_variance', 'first_in_list']
    )

    # Compute correlation matrix
    import pandas as pd
    df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df.corr(method=corr_method).abs()

    # Find correlated pairs
    mask = np.ones(len(feature_names), dtype=bool)
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if upper.iloc[i, j] > max_corr and mask[i] and mask[j]:
                # Decide which to remove
                if keep_criterion == 'higher_importance' and base_importances is not None:
                    remove_idx = i if base_importances[i] < base_importances[j] else j
                elif keep_criterion == 'higher_variance':
                    var_i, var_j = np.var(X[:, i]), np.var(X[:, j])
                    remove_idx = i if var_i < var_j else j
                else:  # first_in_list
                    remove_idx = j

                mask[remove_idx] = False

    if mask.sum() < 10:
        return 0.0

    # Evaluate
    X_selected = X[:, mask]
    model = LGBMClassifier(n_estimators=100, verbose=-1)
    scores = cross_val_score(model, X_selected, y, cv=3, scoring='f1_macro')

    return np.mean(scores)
```

**Typical Result:**
```yaml
best_params:
  max_correlation: 0.92
  correlation_method: pearson
  keep_criterion: higher_importance

correlated_pairs_removed: 23
features_retained: 73
cv_score: 0.635
```

### Combined Selection Pipeline

**Recommended approach:** Run strategies in sequence for best results.

```python
class CombinedFeatureSelector:
    """Combined feature selection using multiple strategies."""

    def run_full_pipeline(self, X, y, X_val, y_val, feature_names):
        current_mask = np.ones(len(feature_names), dtype=bool)

        # Step 1: Binary group selection (coarse filtering)
        if self.config.get('binary_group_selection', {}).get('enabled', True):
            group_mask = self._run_group_selection(X, y, feature_names)
            current_mask &= group_mask
            print(f"After group selection: {current_mask.sum()} features")

        # Step 2: Importance-based selection
        if self.config.get('importance_based', {}).get('enabled', True):
            X_subset = X[:, current_mask]
            importance_study = optuna.create_study(direction='maximize')
            importance_study.optimize(
                lambda trial: self._importance_objective(
                    trial, X_subset, y, X_val[:, current_mask], y_val
                ),
                n_trials=50
            )
            # Apply importance filtering to current_mask
            print(f"After importance selection: {current_mask.sum()} features")

        # Step 3: RFE (optional, if needed)
        if self.config.get('rfe', {}).get('enabled', False):
            X_subset = X[:, current_mask]
            subset_features = [f for f, m in zip(feature_names, current_mask) if m]
            rfe_selected, _ = self._run_rfe(X_subset, y, subset_features)
            # Update mask based on RFE results
            print(f"After RFE: {len(rfe_selected)} features")

        # Step 4: Correlation filtering
        if self.config.get('correlation', {}).get('enabled', True):
            X_subset = X[:, current_mask]
            corr_study = optuna.create_study(direction='maximize')
            corr_study.optimize(
                lambda trial: self._correlation_objective(
                    trial, X_subset, y, feature_names[current_mask]
                ),
                n_trials=30
            )
            print(f"After correlation filtering: {current_mask.sum()} features")

        selected_features = [f for f, m in zip(feature_names, current_mask) if m]

        return FeatureSelectionResult(
            selected_features=selected_features,
            selection_mask=current_mask,
            best_params={},
            metrics={
                'features_before': len(feature_names),
                'features_after': len(selected_features),
                'reduction_ratio': 1 - len(selected_features) / len(feature_names)
            }
        )
```

### Objective Function Configuration

**Metrics:** Stage 8 optimizes for validation performance (not feature count).

**Regularization:** Optional L1 penalty on feature count.

```python
def feature_selection_objective_with_regularization(
    trial: optuna.Trial,
    X, y,
    feature_names,
    lambda_l1=0.001
):
    # ... select features ...

    # Evaluate performance
    model = LGBMClassifier(n_estimators=100, verbose=-1)
    scores = cross_val_score(model, X_selected, y, cv=3, scoring='f1_macro')
    mean_score = np.mean(scores)

    # L1 regularization (penalize feature count)
    n_features = X_selected.shape[1]
    penalty = lambda_l1 * n_features

    return mean_score - penalty
```

**Without regularization:** Optuna may select too many features (overfitting risk).
**With regularization (λ=0.001):** Optuna balances performance vs parsimony.

---

## Stage 9: Feature Pruning with Optuna

### Overview

**Goal:** Fine-tune feature selection by removing low-importance individual features.

**Trial Budget:** 50 trials
**Typical Runtime:** 8-15 minutes
**Input:** Features selected from Stage 8 (typically 60-100 features)
**Output:** Pruned features (typically 30-60 features)

### Search Space

```python
def feature_pruning_search_space(trial: optuna.Trial) -> dict:
    return {
        # Importance threshold (features below this are removed)
        'importance_threshold': trial.suggest_float(
            'importance_threshold', 0.0001, 0.01, log=True
        ),

        # Maximum correlation between features
        'max_correlation': trial.suggest_float(
            'max_correlation', 0.85, 0.99
        ),

        # Minimum variance percentile (remove low-variance features)
        'min_variance_percentile': trial.suggest_float(
            'min_variance_percentile', 0.01, 0.10
        ),

        # Importance calculation method
        'importance_method': trial.suggest_categorical(
            'importance_method',
            ['gain', 'split', 'permutation']
        ),
    }
```

### Pruning Strategies

#### 1. Threshold-Based Pruning (Default)

**Concept:** Remove features with importance below threshold.

```python
def threshold_pruning_objective(trial, X, y, X_val, y_val):
    params = feature_pruning_search_space(trial)

    # Train model
    model = LGBMClassifier(n_estimators=200, max_depth=8, verbose=-1)
    model.fit(X, y)

    # Get importances
    if params['importance_method'] == 'gain':
        importances = model.booster_.feature_importance(importance_type='gain')
    elif params['importance_method'] == 'split':
        importances = model.booster_.feature_importance(importance_type='split')
    else:  # permutation
        from sklearn.inspection import permutation_importance
        result = permutation_importance(model, X_val, y_val, n_repeats=5)
        importances = result.importances_mean

    # Normalize
    importances = importances / (importances.sum() + 1e-8)

    # Prune by threshold
    keep_mask = importances >= params['importance_threshold']

    # Variance filtering
    variances = np.var(X, axis=0)
    var_threshold = np.percentile(variances, params['min_variance_percentile'] * 100)
    keep_mask &= (variances >= var_threshold)

    # Correlation filtering
    if keep_mask.sum() > 1:
        corr = np.corrcoef(X[:, keep_mask].T)
        for i in range(len(corr)):
            for j in range(i + 1, len(corr)):
                if abs(corr[i, j]) > params['max_correlation']:
                    indices = np.where(keep_mask)[0]
                    drop_idx = indices[i] if importances[indices[i]] < importances[indices[j]] else indices[j]
                    keep_mask[drop_idx] = False

    if keep_mask.sum() < 20:
        return 0.0

    # Evaluate
    X_pruned = X[:, keep_mask]
    X_val_pruned = X_val[:, keep_mask]

    model_pruned = LGBMClassifier(n_estimators=200, verbose=-1)
    model_pruned.fit(X_pruned, y)
    val_score = model_pruned.score(X_val_pruned, y_val)

    # Parsimony bonus
    removed_features = X.shape[1] - keep_mask.sum()
    parsimony_bonus = 0.0005 * removed_features

    return val_score + parsimony_bonus
```

#### 2. Top-K Selection

**Concept:** Keep only the top K most important features.

```python
def top_k_pruning_objective(trial, X, y):
    k = trial.suggest_int('top_k_features', 20, 80)
    method = trial.suggest_categorical('importance_method', ['gain', 'shap'])

    # Train and get importance
    model = LGBMClassifier(n_estimators=200, verbose=-1)
    model.fit(X, y)

    if method == 'gain':
        importances = model.booster_.feature_importance(importance_type='gain')
    else:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X[:5000])
        importances = np.abs(shap_values).mean(axis=0) if not isinstance(shap_values, list) else \
                      np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)

    # Select top K
    top_indices = np.argsort(importances)[-k:]
    mask = np.zeros(len(importances), dtype=bool)
    mask[top_indices] = True

    X_selected = X[:, mask]
    scores = cross_val_score(model, X_selected, y, cv=3, scoring='f1_macro')

    return np.mean(scores)
```

#### 3. Cumulative Importance

**Concept:** Keep features until cumulative importance reaches threshold (e.g., 95%).

```python
def cumulative_importance_objective(trial, X, y):
    target_cumulative = trial.suggest_float('target_cumulative', 0.90, 0.99)

    # Train and get importance
    model = LGBMClassifier(n_estimators=200, verbose=-1)
    model.fit(X, y)
    importances = model.booster_.feature_importance(importance_type='gain')

    # Normalize
    importances = importances / importances.sum()

    # Sort by importance
    sorted_indices = np.argsort(importances)[::-1]
    cumsum = np.cumsum(importances[sorted_indices])

    # Find cutoff
    n_features = np.searchsorted(cumsum, target_cumulative) + 1

    # Create mask
    mask = np.zeros(len(importances), dtype=bool)
    mask[sorted_indices[:n_features]] = True

    X_selected = X[:, mask]
    scores = cross_val_score(model, X_selected, y, cv=3, scoring='f1_macro')

    return np.mean(scores)
```

### Importance Methods Comparison

| Method | Speed | Reliability | Model Dependency | Use Case |
|--------|-------|-------------|------------------|----------|
| **gain** | Fast | High | Tree models only | Default for boosting |
| **split** | Fast | Medium | Tree models only | Quick baseline |
| **permutation** | Slow | Very High | Any model | Most reliable |
| **shap** | Slow | Very High | Any model | Interpretability |
| **weight** | Fast | Low | Linear models | Linear models only |

**Recommendation:**
- Use **gain** for speed (default)
- Use **permutation** for reliability (if time permits)
- Use **shap** for interpretability (research/debugging)

### Constraints

```yaml
constraints:
  # Minimum features to retain (safety floor)
  min_features_retained: 20

  # Maximum percentage of features to remove
  max_features_removed_pct: 0.7  # Never remove more than 70%

  # Required features (never pruned)
  required_features:
    - returns_1
    - volatility_20
    - volume_ratio

  # Feature groups that should be pruned together
  grouped_features:
    bollinger:
      - bollinger_upper
      - bollinger_lower
      - bollinger_width
    macd:
      - macd
      - macd_signal
      - macd_hist

  # Minimum performance threshold
  min_performance:
    metric: f1_macro
    threshold: 0.55  # Reject pruning if F1 < 0.55
```

---

## Per-Model Feature Selection

### Model-Specific Strategies

Different model families benefit from different feature selection approaches:

| Model Family | Recommended Strategy | Max Features | Key Groups |
|--------------|---------------------|--------------|------------|
| **Boosting** (XGBoost, LightGBM, CatBoost) | All strategies, prefer gain importance | 60-100 | All groups |
| **Neural** (LSTM, GRU) | RFE + importance, fewer features | 30-50 | momentum, volatility, wavelets |
| **Transformers** (PatchTST, iTransformer) | Minimal selection, raw features | 20-40 | Skip engineered, use raw OHLCV |
| **CNN** (InceptionTime, ResNet1D) | Wavelet emphasis, RFE | 40-60 | wavelets, volatility |
| **Classical** (RF, SVM, Logistic) | All strategies, strong regularization | 30-50 | momentum, trend, moving_avg |

### Example: Boosting Models

```python
# config/models/xgboost_feature_selection.yaml
feature_selection:
  stage_8:
    enabled: true
    n_trials: 100
    strategies:
      binary_group_selection:
        enabled: true
        required_groups:
          - momentum
          - volatility
          - trend
      importance_based:
        enabled: true
        methods: [gain, split]
        threshold_range: [0.001, 0.05]
      correlation_based:
        enabled: true
        max_correlation: 0.95

  stage_9:
    enabled: true
    n_trials: 50
    importance_method: gain
    threshold_range: [0.002, 0.02]
    max_correlation: 0.92

  constraints:
    min_features: 30
    max_features: 100
    required_features:
      - returns_1
      - rsi_14
      - atr_14
      - volume_ratio
```

### Example: Neural Models (LSTM)

```python
# config/models/lstm_feature_selection.yaml
feature_selection:
  stage_8:
    enabled: true
    n_trials: 100
    strategies:
      binary_group_selection:
        enabled: true
        required_groups:
          - momentum
          - volatility
          - wavelets
        excluded_groups:
          - microstructure  # Too noisy for LSTM
          - entropy  # Too noisy
      rfe:
        enabled: true
        n_features_range: [25, 45]
        estimator: lightgbm

  stage_9:
    enabled: true
    n_trials: 50
    importance_method: permutation  # More reliable for neural
    threshold_range: [0.005, 0.03]
    max_correlation: 0.85  # Stricter for neural

  constraints:
    min_features: 20
    max_features: 50
    required_features:
      - returns_1
      - volatility_20
      - wavelet_trend
```

### Example: Transformers (PatchTST)

```python
# config/models/patchtst_feature_selection.yaml
feature_selection:
  stage_8:
    enabled: false  # Skip feature selection
    # PatchTST uses raw OHLCV, no engineered features

  stage_9:
    enabled: false

  feature_mode: raw_ohlcv
  # Use raw OHLCV from multi-resolution streams
  # No feature engineering needed
```

---

## Running Feature Selection

### Command-Line Interface

```bash
# Run Stage 8 only (feature selection)
python -m src.phase1.stages.feature_select.run \
    --symbol MES \
    --model-family boosting \
    --n-trials 100 \
    --config config/optimization/feature_selection.yaml

# Run Stage 9 only (feature pruning)
python -m src.phase1.stages.feature_prune.run \
    --symbol MES \
    --n-trials 50 \
    --importance-method gain \
    --config config/optimization/feature_pruning.yaml

# Run both stages (recommended)
python -m src.phase1.stages.feature_optimize.run \
    --symbol MES \
    --selection-trials 100 \
    --pruning-trials 50
```

### Python API

```python
from src.phase1.stages.feature_select import FeatureSelectionOptimizer
from src.phase1.stages.feature_prune import FeaturePruningOptimizer

# Stage 8: Feature Selection
selector = FeatureSelectionOptimizer(config={
    'n_trials': 100,
    'strategy': 'binary_group',
    'model_family': 'boosting'
})

selected_features, best_params = selector.optimize(
    X=X_train,
    y=y_train,
    feature_names=feature_names
)

print(f"Selected {len(selected_features)} features")
print(f"Best params: {best_params}")

# Stage 9: Feature Pruning
pruner = FeaturePruningOptimizer(config={
    'n_trials': 50,
    'importance_method': 'gain'
})

pruned_features, pruning_params = pruner.optimize(
    X=X_train[:, selected_mask],
    y=y_train,
    X_val=X_val[:, selected_mask],
    y_val=y_val,
    feature_names=selected_features
)

print(f"Pruned to {len(pruned_features)} features")
```

### Integration with Unified Pipeline

```python
from src.pipeline.unified import MLPipeline
from src.pipeline.config import MLConfig

config = MLConfig(
    symbol="MES",
    horizons=[20],
    models=["xgboost", "lstm"],

    # Enable feature optimization
    optimize_features=True,
    feature_selection_trials=100,
    feature_pruning_trials=50,

    # Per-model optimization
    per_model_feature_selection=True
)

pipeline = MLPipeline(config)

# Runs all 16 stages including Stages 8-9
pipeline.run()

# Or run feature optimization only
pipeline.run_optimization()  # Runs Stages 7-9
```

---

## Interpreting Results

### Stage 8 Output

```yaml
# experiments/feature_select/MES_20260118_130000/best_features.yaml
symbol: MES
stage: 8_feature_selection
timestamp: 2026-01-18T13:25:42

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
  entropy: false

metrics:
  n_features_total: 162
  n_features_selected: 87
  n_features_removed: 75
  reduction_ratio: 0.463
  cross_val_score: 0.632
  num_trials: 100
  optimization_time: 1245.3

best_params:
  include_momentum: true
  include_volatility: true
  include_volume: false
  # ... (10 binary decisions)

selected_features:
  - rsi_14
  - rsi_21
  - macd
  - macd_signal
  - atr_14
  - bollinger_width
  - adx_14
  # ... (87 total)
```

**Key Metrics:**
- `reduction_ratio`: 0.463 = removed 46.3% of features
- `cross_val_score`: 0.632 = validation F1 score with selected features
- Selected 7 out of 10 feature groups

**Interpretation:**
- Volume features excluded (not predictive for this symbol)
- Microstructure features excluded (too noisy)
- Entropy features excluded (not useful)
- Core momentum, volatility, trend retained

### Stage 9 Output

```yaml
# experiments/feature_prune/MES_20260118_140000/pruned_features.yaml
symbol: MES
stage: 9_feature_pruning
timestamp: 2026-01-18T14:12:18

pruning_params:
  importance_threshold: 0.0023
  max_correlation: 0.92
  min_variance_percentile: 0.03
  importance_method: gain

metrics:
  n_features_before: 87
  n_features_after: 52
  features_removed: 35
  reduction_ratio: 0.402
  accuracy_before: 0.618
  accuracy_after: 0.627
  num_trials: 50
  optimization_time: 687.1

removed_features:
  - wavelet_detail_3  # Low importance
  - aroon_down  # Correlated with aroon_up
  - stoch_d  # Correlated with stoch_k
  - cci_20  # Low importance
  - williams_r  # Correlated with rsi_14
  # ... (35 total)

retained_features:
  - rsi_14
  - rsi_21
  - macd
  - macd_signal
  - atr_14
  - bollinger_width
  # ... (52 total)

importance_ranking:
  - feature: rsi_14
    importance: 0.087
    rank: 1
  - feature: atr_14
    importance: 0.065
    rank: 2
  - feature: macd
    importance: 0.054
    rank: 3
  # ... (52 total)
```

**Key Metrics:**
- `features_removed`: 35 (40.2% reduction from Stage 8)
- `accuracy_after`: 0.627 (improved from 0.618)
- Removed low-importance and correlated features

**Interpretation:**
- Feature pruning **improved** performance (0.618 → 0.627)
- Removed redundant indicators (e.g., williams_r correlated with rsi_14)
- Top 3 features: rsi_14, atr_14, macd

### Visualization

#### 1. Feature Importance Plot

```python
import matplotlib.pyplot as plt
import pandas as pd

# Load pruning results
results = pd.read_csv('experiments/feature_prune/MES_20260118_140000/importance_ranking.csv')

# Plot top 20 features
top_20 = results.head(20)
plt.figure(figsize=(10, 8))
plt.barh(top_20['feature'], top_20['importance'])
plt.xlabel('Importance')
plt.title('Top 20 Most Important Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
```

#### 2. Feature Correlation Heatmap

```python
import seaborn as sns

# Load retained features
X_pruned = load_pruned_features()  # Shape: (N, 52)
corr = pd.DataFrame(X_pruned, columns=retained_features).corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap='coolwarm', center=0, square=True)
plt.title('Feature Correlation Heatmap (After Pruning)')
plt.tight_layout()
plt.savefig('feature_correlation.png')
```

#### 3. Optimization History

```python
import optuna

# Load Optuna study
study = optuna.load_study(
    study_name='feature_selection_MES',
    storage='sqlite:///experiments/optuna/feature_selection.db'
)

# Plot optimization history
fig = optuna.visualization.plot_optimization_history(study)
fig.write_image('optimization_history.png')

# Plot parameter importances
fig = optuna.visualization.plot_param_importances(study)
fig.write_image('param_importances.png')
```

---

## Best Practices

### 1. Trial Budget Allocation

**Stage 8:**
- Minimum: 50 trials (for binary group selection)
- Recommended: 100 trials (for combined strategies)
- Maximum: 200 trials (if using RFE or individual selection)

**Stage 9:**
- Minimum: 30 trials (for threshold-based pruning)
- Recommended: 50 trials (default)
- Maximum: 100 trials (if using multiple importance methods)

### 2. Validation Strategy

**Use PurgedKFold** to prevent data leakage:

```python
from src.cross_validation.purged_kfold import PurgedKFold

def feature_selection_objective_with_purged_cv(trial, X, y):
    # ... select features ...

    kfold = PurgedKFold(n_splits=3, purge_bars=60, embargo_bars=60)
    scores = []

    for train_idx, val_idx in kfold.split(X_selected):
        model = LGBMClassifier(n_estimators=100, verbose=-1)
        model.fit(X_selected[train_idx], y[train_idx])
        pred = model.predict(X_selected[val_idx])
        scores.append(f1_score(y[val_idx], pred, average='macro'))

    return np.mean(scores)
```

### 3. Feature Group Design

**Good feature groups:**
- Logically cohesive (all momentum indicators together)
- Similar feature count (10-20 features per group)
- Non-overlapping (no feature in multiple groups)

**Bad feature groups:**
- Mixed concepts (momentum + volume in same group)
- Unbalanced sizes (1 feature vs 50 features)
- Overlapping (rsi_14 in both 'momentum' and 'trend')

### 4. Importance Method Selection

**Use gain for:**
- Boosting models (XGBoost, LightGBM, CatBoost)
- Fast optimization (Stage 9 with 50 trials)
- Initial exploration

**Use permutation for:**
- Neural models (LSTM, GRU, TCN)
- Final validation
- Research/debugging

**Use SHAP for:**
- Interpretability (explain model predictions)
- Debugging feature interactions
- Model auditing

### 5. Regularization Tuning

```python
# No regularization (may select too many features)
lambda_l1 = 0.0

# Light regularization (recommended)
lambda_l1 = 0.001  # Penalize 1 feature = 0.001 score

# Strong regularization (aggressive pruning)
lambda_l1 = 0.005

# Adaptive regularization (based on model family)
if model_family == 'boosting':
    lambda_l1 = 0.0005  # Boosting handles many features well
elif model_family == 'neural':
    lambda_l1 = 0.002  # Neural models prefer fewer features
else:
    lambda_l1 = 0.001
```

### 6. Checkpointing and Resume

```python
# Enable checkpointing
config = {
    'n_trials': 100,
    'checkpoint': {
        'enabled': True,
        'checkpoint_dir': 'experiments/runs/{run_id}/checkpoints/stage_8/',
        'save_every_n_trials': 10
    },
    'resume': {
        'enabled': True,
        'resume_from_study': True
    }
}

# Run optimization
optimizer = FeatureSelectionOptimizer(config)
result = optimizer.optimize(X, y, feature_names)

# If interrupted, resume from checkpoint
optimizer.resume(run_id='20260118_120000')
```

### 7. Parallel Optimization

```python
# Parallelize trials across cores
config = {
    'n_trials': 100,
    'n_jobs': -1,  # Use all available cores
    'storage': 'sqlite:///experiments/optuna/feature_selection.db'
}

# Or distribute across multiple machines
config = {
    'n_trials': 100,
    'storage': 'postgresql://user:pass@host/db',
    'distributed': True
}
```

---

## Integration with Pipeline

### Full Pipeline Flow

```
Stage 1-6: Data Preparation
  ↓
Stage 7: OPTUNA Label Optimization (100 trials)
  ↓
┌─────────────────────────────────────┐
│ Stage 8: Feature Selection (100)    │
│ - Binary group selection             │
│ - Importance-based selection         │
│ - RFE (optional)                     │
│ - Correlation filtering              │
│                                      │
│ Input: 162 features                  │
│ Output: ~60-100 features             │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ Stage 9: Feature Pruning (50)       │
│ - Threshold-based pruning            │
│ - Correlation filtering              │
│ - Variance filtering                 │
│                                      │
│ Input: ~60-100 features              │
│ Output: ~30-60 features              │
└──────────────┬──────────────────────┘
  ↓
Stage 10: Splits
Stage 11: Scaling
Stage 12: Adaptation
  ↓
Stage 13: OPTUNA Hyperparameter Optimization (100 trials per model)
  ↓
Stage 14-16: Training, Stacking, Bundling
```

### Pipeline State Management

```python
# Pipeline tracks which stages are complete
from src.pipeline.state import PipelineState

state = PipelineState(run_id='20260118_120000')

# Check if feature selection is complete
if state.is_stage_complete('feature_selection'):
    # Load selected features
    selected_features = state.get_checkpoint('feature_selection', 'selected_features.json')
else:
    # Run feature selection
    result = pipeline.run_optimization()
    state.mark_stage_complete('feature_selection', result)
```

### Model-Specific Feature Sets

```python
# Each model can have its own feature selection
config = MLConfig(
    models=[
        ModelConfig(
            name="xgboost",
            feature_selection_trials=100,
            feature_pruning_trials=50,
            feature_groups=['momentum', 'volatility', 'trend', 'wavelets']
        ),
        ModelConfig(
            name="lstm",
            feature_selection_trials=100,
            feature_pruning_trials=50,
            feature_groups=['momentum', 'volatility', 'wavelets']  # No microstructure
        ),
        ModelConfig(
            name="patchtst",
            feature_selection_trials=0,  # Skip feature selection
            feature_mode='raw_ohlcv'  # Use raw OHLCV
        )
    ],
    per_model_feature_selection=True
)
```

---

## Summary

### Key Takeaways

1. **Two-stage optimization:** Stage 8 (selection) → Stage 9 (pruning)
2. **Multiple strategies:** Binary group, RFE, importance, correlation
3. **Per-model optimization:** Different models need different features
4. **Optuna integration:** Automated hyperparameter search for selection thresholds
5. **Validation:** Use PurgedKFold to prevent data leakage
6. **Regularization:** Balance performance vs feature count

### Trial Budget Summary

| Stage | Trials | Time | Output |
|-------|--------|------|--------|
| Stage 8 | 100 | 15-30 min | 60-100 features |
| Stage 9 | 50 | 8-15 min | 30-60 features |
| **Total** | **150** | **25-45 min** | **Final feature set** |

### Feature Reduction Example

```
Stage 5:  162 features (engineered)
           ↓
Stage 8:  87 features (group selection + importance)
           ↓ (46% reduction)
Stage 9:  52 features (individual pruning)
           ↓ (68% total reduction)
Training: 52 features (optimal subset)
```

### Configuration Files

- Feature selection: `config/optimization/feature_selection.yaml`
- Feature pruning: `config/optimization/feature_pruning.yaml`
- Per-model configs: `config/models/{model_name}_feature_selection.yaml`

### Related Documentation

- Feature Engineering: `docs/guides/FEATURE_ENGINEERING.md`
- Hyperparameter Tuning: `docs/guides/HYPERPARAMETER_TUNING.md`
- Unified Pipeline: `docs/implementation/UNIFIED_PIPELINE_ARCHITECTURE.md`
- Optimization README: `config/optimization/README.md`

---

**Last Updated:** 2026-01-18
**Author:** ML Factory Team
**Version:** 1.0.0
