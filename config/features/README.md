# Feature Configuration

Feature-selection and feature-engineering configuration lives here.

## Files

- `model_features.yaml` - Per-model feature selection rules / defaults
- `mtf_strategies.yaml` - Multi-timeframe (MTF) feature strategies
- `selection_methods.yaml` - Feature selection method configuration

## Feature Selection with Optuna

The ML Factory supports Optuna-based feature selection for optimal feature subset discovery.

### Binary Feature Selection

Each feature is treated as a binary variable (include/exclude), allowing Optuna to efficiently explore the feature space.

**Configuration:** See `config/optimization/feature_selection.yaml`

```yaml
# Example: Feature selection config reference
optimization:
  framework: optuna
  n_trials: 100

search_space:
  feature_selection_mode: binary
  # Each feature becomes a suggest_categorical(feature_name, [True, False])

constraints:
  min_features: 10
  max_features: 100
```

**Usage:**
```python
from src.features.optimization import optimize_feature_selection

selected = optimize_feature_selection(
    X=features,
    y=labels,
    config_path="config/optimization/feature_selection.yaml"
)
```

### Feature Group Selection

Select entire feature groups rather than individual features:

```yaml
feature_groups:
  price_features:
    include: true
    features: [close, high, low, open, vwap]

  volume_features:
    include: true
    features: [volume, tick_volume, dollar_volume]

  momentum_features:
    include: true
    features: [rsi_14, macd, macd_signal, momentum_10]

  volatility_features:
    include: true
    features: [atr_14, bb_width, realized_vol_20]

  microstructure_features:
    include: true
    features: [spread, imbalance, trade_intensity]
```

## Feature Pruning with Optuna

Importance-based feature pruning removes low-importance features to reduce dimensionality.

**Configuration:** See `config/optimization/feature_pruning.yaml`

```yaml
# Example: Feature pruning config reference
optimization:
  framework: optuna
  n_trials: 50

search_space:
  importance_threshold:
    type: float
    low: 0.001
    high: 0.1
    log: true

  importance_method:
    type: categorical
    choices: [permutation, shap, gain, split]

constraints:
  min_features_retained: 20
```

**Importance Methods:**

| Method | Description | Speed | Accuracy |
|--------|-------------|-------|----------|
| `gain` | Total gain across all splits | Fast | Medium |
| `split` | Number of times feature is used | Fast | Low |
| `permutation` | Performance drop when shuffled | Medium | High |
| `shap` | SHAP value magnitudes | Slow | High |

**Usage:**
```python
from src.features.optimization import optimize_feature_pruning

retained_features = optimize_feature_pruning(
    X=features,
    y=labels,
    config_path="config/optimization/feature_pruning.yaml"
)
```

## Per-Model Feature Strategies

Different model families benefit from different feature sets.

### Boosting Models
- Handle high-dimensional features well
- Prefer raw features over normalized
- Can utilize all feature types

```yaml
# model_features.yaml
boosting:
  feature_set: boosting_optimal
  normalization: false
  feature_count: 50-100
  include_interactions: true
```

### Neural Models
- Require normalized features
- Benefit from dimensionality reduction
- Prefer stationary features

```yaml
neural:
  feature_set: neural_optimal
  normalization: true
  feature_count: 30-50
  sequence_features: true
```

### Classical Models
- Sensitive to feature scaling
- Benefit from feature selection
- Prefer lower dimensionality

```yaml
classical:
  feature_set: classical_optimal
  normalization: true
  feature_count: 20-40
  pca_reduction: optional
```

## Per-Model Optimization

Run feature selection separately for each model family:

```bash
# Optimize for boosting models
python scripts/optimize_features.py \
    --mode selection \
    --model-family boosting \
    --n-trials 100

# Optimize for neural models
python scripts/optimize_features.py \
    --mode selection \
    --model-family neural \
    --n-trials 100
```

## Feature Selection Workflow

```
1. Start with full feature set (~150 features)
           |
           v
2. Apply mandatory filters (NaN threshold, variance)
           |
           v
3. Run Optuna feature selection (100 trials)
           |
           v
4. Run Optuna feature pruning (50 trials)
           |
           v
5. Per-model feature adaptation
           |
           v
6. Final optimized feature set (~40-60 features)
```

## Configuration Files

### model_features.yaml
```yaml
# Per-model feature configuration
models:
  xgboost:
    feature_set: boosting_optimal
    max_features: 80
    required_features:
      - returns_1
      - volatility_20
      - volume_ratio

  lstm:
    feature_set: neural_optimal
    max_features: 50
    sequence_length: 60
    normalization: z_score

  random_forest:
    feature_set: classical_optimal
    max_features: 40
    use_pca: false
```

### mtf_strategies.yaml
```yaml
# Multi-timeframe feature strategies
mtf:
  enabled: true
  base_timeframe: 5min

  higher_timeframes:
    - timeframe: 15min
      features: [sma_20, rsi_14, atr_14]
    - timeframe: 1h
      features: [sma_50, macd, bb_width]
    - timeframe: 4h
      features: [trend_direction, volatility_regime]

  aggregation:
    method: concat  # concat, mean, weighted
```

### selection_methods.yaml
```yaml
# Feature selection method configuration
methods:
  variance_threshold:
    enabled: true
    threshold: 0.01

  correlation_filter:
    enabled: true
    threshold: 0.95

  mutual_information:
    enabled: true
    n_neighbors: 5

  optuna_selection:
    enabled: true
    config_path: config/optimization/feature_selection.yaml

  optuna_pruning:
    enabled: true
    config_path: config/optimization/feature_pruning.yaml
```

## Related Docs

- [Feature Engineering Guide](../../docs/guides/FEATURE_ENGINEERING.md)
- [Feature Selection by Architecture](../../docs/FEATURE_SELECTION_BY_ARCHITECTURE.md)
- [Optimization Configuration](../optimization/README.md)
- [Feature Selection Optimization](../optimization/feature_selection.yaml)
- [Feature Pruning Optimization](../optimization/feature_pruning.yaml)

---

*Last Updated: 2026-01-18*
