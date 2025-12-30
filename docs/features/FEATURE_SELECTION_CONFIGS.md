# Feature Selection Configurations

**Walk-Forward Feature Selection for Time Series ML**

This document specifies exact configurations for feature selection methods used in the ML Model Factory, including MDA, MDI, SHAP, and hybrid approaches.

---

## Table of Contents

- [Overview](#overview)
- [Selection Methods](#selection-methods)
- [Configuration Parameters](#configuration-parameters)
- [Walk-Forward Methodology](#walk-forward-methodology)
- [Feature Importance Scores](#feature-importance-scores)
- [Stability Analysis](#stability-analysis)
- [Usage Examples](#usage-examples)

---

## Overview

### Purpose

Feature selection reduces dimensionality, prevents overfitting, and improves model generalization by identifying the most predictive features from the 150+ engineered features.

### Anti-Lookahead Guarantee

**All feature selection is performed using walk-forward methodology:**
- Features are selected using ONLY training data at each CV fold
- No future data leaks into feature importance calculation
- Features must appear consistently across multiple folds to be considered stable

### Methods Implemented

| Method | Type | Speed | Reliability | Multicollinearity Handling |
|--------|------|-------|-------------|---------------------------|
| MDI | Impurity-based | Fast | Low (biased) | Poor |
| MDA | Permutation-based | Slow | High | Good |
| Hybrid | MDI + MDA ranking | Medium | Medium-High | Good |
| Clustered MDA | Correlation-aware | Slowest | Highest | Excellent |

**Implementation:** `src/cross_validation/feature_selector.py`

---

## Selection Methods

### 1. MDI (Mean Decrease in Impurity)

**Method:** Built-in Random Forest feature importance based on Gini impurity reduction.

**Algorithm:**
```python
1. Train Random Forest on training data
2. For each feature f:
   importance[f] = Σ (impurity_reduction at splits using f) / n_trees
3. Normalize importance to sum to 1
4. Select top N features by importance
```

**Pros:**
- Very fast (computed during training)
- No extra computational cost

**Cons:**
- Biased towards high-cardinality features
- Inflates importance of correlated features
- Unreliable for feature selection in presence of multicollinearity

**Configuration:**

```python
{
    "method": "mdi",
    "n_estimators": 100,  # RF trees for importance
    "max_depth": 5,       # Prevent overfitting
    "n_features_to_select": 50,
    "min_feature_frequency": 0.6  # Stability threshold
}
```

**When to Use:**
- Quick baseline feature selection
- Features are known to be uncorrelated
- Speed is critical

### 2. MDA (Mean Decrease in Accuracy)

**Method:** Permutation importance - measures performance drop when feature is randomly shuffled.

**Algorithm:**
```python
1. Train Random Forest with OOB scoring enabled
2. For each feature f:
   a. Record baseline OOB accuracy
   b. Shuffle feature f values
   c. Compute OOB accuracy on permuted data
   d. importance[f] = baseline_accuracy - permuted_accuracy
   e. Repeat 10 times and average
3. Select top N features by importance
```

**Reference:** Breiman (2001), "Random Forests"; Lopez de Prado (2018), Chapter 8

**Pros:**
- More reliable than MDI
- Handles correlated features better
- Directly measures predictive power

**Cons:**
- Computationally expensive (10 permutations per feature)
- Still affected by multicollinearity (correlated features share importance)

**Configuration:**

```python
{
    "method": "mda",
    "n_estimators": 100,
    "max_depth": 5,
    "n_repeats": 10,       # Permutation repeats
    "n_features_to_select": 50,
    "min_feature_frequency": 0.6
}
```

**When to Use:**
- Default recommendation for feature selection
- Better reliability needed than MDI
- Sufficient computational budget

### 3. Hybrid (MDI + MDA Ranking)

**Method:** Combine MDI and MDA by averaging their rank-based importance.

**Algorithm:**
```python
1. Compute MDI importance → rank_mdi
2. Compute MDA importance → rank_mda
3. For each feature f:
   hybrid_importance[f] = (rank_mdi[f] + rank_mda[f]) / 2
4. Select top N features by hybrid_importance
```

**Rationale:** Ranks are more robust than raw scores when combining methods with different scales.

**Pros:**
- Balances speed (MDI) and reliability (MDA)
- More stable than either method alone
- Reduces method-specific bias

**Cons:**
- Slower than MDI alone
- Still somewhat affected by multicollinearity

**Configuration:**

```python
{
    "method": "hybrid",
    "n_estimators": 100,
    "max_depth": 5,
    "n_repeats": 10,
    "n_features_to_select": 50,
    "min_feature_frequency": 0.6
}
```

**When to Use:**
- Balanced approach between speed and reliability
- When unsure which method is better for the dataset

### 4. Clustered MDA (Correlation-Aware)

**Method:** Group correlated features into clusters, compute MDA importance per cluster, distribute within cluster.

**Algorithm:**
```python
1. Compute feature correlation matrix
2. Hierarchical clustering on distance = 1 - |correlation|
3. Create max_clusters feature clusters
4. For each cluster c:
   a. Compute cluster representative = mean(cluster_features)
   b. Train RF on representative feature
   c. Compute MDA importance → cluster_importance[c]
5. For each feature f in cluster c:
   importance[f] = cluster_importance[c] / n_features_in_cluster
6. Select top N features by importance
```

**Reference:** Lopez de Prado (2018), Chapter 8 - "Clustered Feature Importance"

**Pros:**
- Handles multicollinearity optimally
- Prevents redundant features from dominating selection
- Most reliable method for correlated features

**Cons:**
- Slowest method
- Requires tuning max_clusters parameter

**Configuration:**

```python
{
    "method": "mda",
    "use_clustered_importance": True,
    "max_clusters": 20,  # Feature cluster limit
    "n_estimators": 50,  # Fewer trees (already slow)
    "max_depth": 5,
    "n_features_to_select": 50,
    "min_feature_frequency": 0.6
}
```

**When to Use:**
- Features are known to be highly correlated (e.g., multiple volatility estimators, MTF features)
- Maximum reliability needed
- Computational budget allows

---

## Configuration Parameters

### Core Parameters

**`n_features_to_select: int`**
- Number of top features to select per CV fold
- **Default:** 50
- **Range:** [10, 100]
- **Guideline:**
  - Too few: Underfitting, missing important features
  - Too many: Overfitting, redundant features
  - Rule of thumb: `sqrt(total_features)` to `0.5 * total_features`

**`selection_method: str`**
- Feature importance method
- **Options:** `"mdi"`, `"mda"`, `"hybrid"`
- **Default:** `"mda"`
- **Recommendation:** Use `"mda"` unless speed is critical

**`min_feature_frequency: float`**
- Minimum fraction of CV folds feature must appear in to be considered stable
- **Default:** 0.6
- **Range:** [0.4, 1.0]
- **Interpretation:**
  - 0.6: Feature selected in ≥60% of folds
  - 0.8: Feature selected in ≥80% of folds (very stable)
  - 1.0: Feature selected in 100% of folds (extremely stable, may be too restrictive)

### Random Forest Parameters

**`n_estimators: int`**
- Number of trees in Random Forest for importance calculation
- **Default:** 100
- **Range:** [50, 200]
- **Trade-off:** More trees → more stable importance, slower computation

**`max_depth: int`**
- Maximum tree depth for Random Forest
- **Default:** 5
- **Range:** [3, 10]
- **Rationale:** Shallow trees prevent overfitting during feature selection

### MDA-Specific Parameters

**`n_repeats: int`**
- Number of permutations per feature for MDA
- **Default:** 10
- **Range:** [5, 20]
- **Trade-off:** More repeats → more stable importance, slower computation

### Clustered MDA Parameters

**`use_clustered_importance: bool`**
- Whether to use clustered MDA (correlation-aware)
- **Default:** False
- **When to enable:** Features are highly correlated

**`max_clusters: int`**
- Maximum number of feature clusters
- **Default:** 20
- **Range:** [10, 50]
- **Guideline:** Higher for more features, lower for fewer features

---

## Walk-Forward Methodology

### Anti-Lookahead Algorithm

**Feature selection is performed walk-forward to prevent lookahead bias:**

```python
for fold in cv_splits:
    train_idx, test_idx = fold

    # Select features using ONLY training data
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    # Compute importance on training data only
    importance = compute_importance(X_train, y_train)

    # Select top N features
    selected_features = importance.nlargest(n_features_to_select)

    # Track selected features
    fold_selections.append(selected_features)
```

**Stable Features Extraction:**

```python
# Count how many folds each feature was selected in
feature_counts = count_selections(fold_selections)

# Filter for stability
min_count = n_folds * min_feature_frequency
stable_features = [
    f for f, count in feature_counts.items()
    if count >= min_count
]

# Sort by selection frequency (most stable first)
stable_features.sort(key=lambda f: feature_counts[f], reverse=True)
```

### Per-Fold vs Final Selection

**Per-Fold Selection:**
- Used during cross-validation
- Each fold may use different feature subset
- Prevents lookahead bias during CV

**Final Selection (Stable Features):**
- Used for final model training
- Only includes features selected in ≥ `min_feature_frequency` of folds
- Ensures robust feature set

**Example (5-fold CV, min_frequency=0.6):**

| Feature | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Count | Stable? |
|---------|--------|--------|--------|--------|--------|-------|---------|
| `rsi_14` | ✓ | ✓ | ✓ | ✓ | ✓ | 5/5 | ✓ (100%) |
| `atr_14` | ✓ | ✓ | ✓ | ✓ | - | 4/5 | ✓ (80%) |
| `sma_50` | ✓ | ✓ | ✓ | - | - | 3/5 | ✓ (60%, exactly at threshold) |
| `volume_ratio` | ✓ | ✓ | - | - | - | 2/5 | ✗ (40%, below threshold) |
| `wavelet_close_d1` | ✓ | - | - | - | - | 1/5 | ✗ (20%, unstable) |

**Min count = 5 * 0.6 = 3 folds**

**Stable features:** `rsi_14`, `atr_14`, `sma_50`

---

## Feature Importance Scores

### MDI Importance

**Properties:**
- Non-negative
- Normalized to sum to 1.0
- Unitless (relative importance only)

**Interpretation:**
- Higher = more informative for splitting criteria
- Represents total impurity reduction across all trees

**Limitations:**
- Inflated for high-cardinality features
- Inflated for correlated features
- Does not directly measure predictive power

### MDA Importance

**Properties:**
- Can be negative (feature hurts performance when shuffled)
- Not normalized
- Units: accuracy drop (e.g., -0.05 = 5% accuracy decrease)

**Interpretation:**
- Higher = larger performance drop when permuted
- Directly measures predictive contribution
- Negative importance → redundant or harmful feature

**Typical Range:**
- Top features: 0.01 - 0.10 (1% - 10% accuracy drop)
- Marginal features: 0.001 - 0.01 (0.1% - 1%)
- Irrelevant features: ≈ 0 or negative

### Hybrid Importance

**Properties:**
- Rank-based (average of MDI and MDA ranks)
- Non-negative
- Units: average rank (lower = more important)

**Interpretation:**
- Combines position in MDI and MDA rankings
- Reduces bias from either method alone

---

## Stability Analysis

### Stability Score

**Definition:**
```python
stability_score = n_folds_selected / n_total_folds
```

**Interpretation:**
- 1.0: Feature selected in all folds (extremely stable)
- 0.8: Feature selected in 80% of folds (very stable)
- 0.6: Feature selected in 60% of folds (moderately stable)
- 0.4: Feature selected in 40% of folds (unstable)
- 0.2: Feature selected in 20% of folds (very unstable)

### Stability Thresholds

**Recommended `min_feature_frequency` values:**

| Use Case | Threshold | Rationale |
|----------|-----------|-----------|
| High-frequency trading (overfitting risk) | 0.8 - 1.0 | Require very stable features |
| Standard classification | 0.6 - 0.7 | Balance stability and coverage |
| Exploratory analysis | 0.4 - 0.5 | Allow more features |

### Stability Diagnostics

**Low stability symptoms:**
- Many features with stability < 0.5
- Different features selected across folds
- High variance in fold-wise performance

**Causes:**
- Insufficient training data
- High noise in labels
- Many redundant/correlated features
- Overfitting during feature selection

**Solutions:**
- Increase training data
- Reduce `n_features_to_select`
- Increase `min_feature_frequency`
- Use clustered MDA for correlated features
- Review labeling quality

---

## Usage Examples

### Example 1: Default MDA Selection

**Config:**
```yaml
# config/features/selection_methods.yaml
feature_selection:
  method: mda
  n_features_to_select: 50
  min_feature_frequency: 0.6
  n_estimators: 100
  max_depth: 5
  n_repeats: 10
  random_state: 42
```

**Python:**
```python
from src.cross_validation import WalkForwardFeatureSelector

selector = WalkForwardFeatureSelector(
    n_features_to_select=50,
    selection_method="mda",
    min_feature_frequency=0.6,
    random_state=42
)

# Run walk-forward selection
result = selector.select_features_walkforward(
    X=features_df,
    y=labels,
    cv_splits=cv_splits,
    sample_weights=sample_weights
)

# Get stable features
stable_features = result.stable_features
print(f"Selected {len(stable_features)} stable features")

# Check stability scores
stability_scores = result.get_stability_scores()
for feature in stable_features[:10]:
    print(f"{feature}: {stability_scores[feature]:.1%}")
```

**Output:**
```
Selected 45 stable features
rsi_14: 100.0%
atr_14: 100.0%
hvol_20: 100.0%
sma_50: 80.0%
macd_hist: 80.0%
bb_width: 80.0%
return_5: 80.0%
ema_21: 60.0%
volume_ratio: 60.0%
micro_amihud_20: 60.0%
```

### Example 2: Fast MDI Selection

**Config:**
```python
selector = WalkForwardFeatureSelector(
    n_features_to_select=30,
    selection_method="mdi",  # Fast method
    min_feature_frequency=0.6,
    n_estimators=50,  # Fewer trees
    random_state=42
)
```

**Use Case:** Quick baseline feature selection for rapid prototyping.

### Example 3: Clustered MDA for Correlated Features

**Config:**
```python
selector = WalkForwardFeatureSelector(
    n_features_to_select=50,
    selection_method="mda",
    use_clustered_importance=True,  # Enable clustering
    max_clusters=20,
    min_feature_frequency=0.7,  # Higher threshold
    n_estimators=50,
    random_state=42
)
```

**Use Case:** Dataset with MTF features (high correlation across timeframes), multiple volatility estimators.

### Example 4: Hybrid Selection

**Config:**
```python
selector = WalkForwardFeatureSelector(
    n_features_to_select=50,
    selection_method="hybrid",  # Combine MDI + MDA
    min_feature_frequency=0.6,
    n_estimators=100,
    random_state=42
)
```

**Use Case:** Balanced approach when unsure of best method.

### Example 5: Stability Analysis

**Python:**
```python
# Run selection
result = selector.select_features_walkforward(X, y, cv_splits)

# Analyze stability distribution
stability_scores = result.get_stability_scores()
stability_values = list(stability_scores.values())

import numpy as np
print(f"Stability distribution:")
print(f"  Min:    {np.min(stability_values):.1%}")
print(f"  25%:    {np.percentile(stability_values, 25):.1%}")
print(f"  Median: {np.median(stability_values):.1%}")
print(f"  75%:    {np.percentile(stability_values, 75):.1%}")
print(f"  Max:    {np.max(stability_values):.1%}")

# Identify unstable features
unstable = [f for f, s in stability_scores.items() if s < 0.4]
print(f"\nUnstable features (<40% selection): {len(unstable)}")
```

**Output:**
```
Stability distribution:
  Min:    20.0%
  25%:    40.0%
  Median: 60.0%
  75%:    80.0%
  Max:    100.0%

Unstable features (<40% selection): 12
```

---

## Integration with Cross-Validation

### CV-Integrated Feature Selection

**Pattern: Feature selection within CV loop**

```python
from src.cross_validation import CVIntegratedFeatureSelector

cv_selector = CVIntegratedFeatureSelector(
    n_features=50,
    min_frequency=0.6,
    method="mda"
)

oof_predictions = []
fold_features = []

for train_idx, test_idx in cv_splits:
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]

    # Select features on training data only
    selected = cv_selector.select_single_fold(
        X_train, y_train, sample_weights[train_idx]
    )
    fold_features.append(selected)

    # Train model on selected features
    model.fit(X_train[selected], y_train)

    # Predict on test fold (OOF)
    oof_predictions.append(model.predict(X_test[selected]))

# Extract stable features across all folds
from collections import Counter
feature_counts = Counter()
for features in fold_features:
    feature_counts.update(features)

min_count = len(cv_splits) * 0.6
stable_features = [
    f for f, count in feature_counts.items()
    if count >= min_count
]
```

**Benefit:** Feature selection and CV happen in a single pass, ensuring no lookahead.

---

## Comparison of Methods

### Performance Characteristics

| Method | Computation Time (5 folds, 150 features) | Memory Usage | Reliability |
|--------|------------------------------------------|--------------|-------------|
| MDI | ~30 sec | Low | Low |
| MDA | ~5 min | Medium | High |
| Hybrid | ~5.5 min | Medium | Medium-High |
| Clustered MDA | ~10 min | High | Very High |

**Benchmark:** Intel i7, 16GB RAM, 150 features, 100k samples, 5-fold CV

### Recommended Workflow

**Phase 1: Baseline (MDI)**
- Quick feature selection for prototyping
- Identify obviously important features
- Establish performance baseline

**Phase 2: Refined (MDA)**
- Run walk-forward MDA selection
- Validate stability across folds
- Compare performance vs MDI baseline

**Phase 3: Final (Clustered MDA or Hybrid)**
- If features are highly correlated → Clustered MDA
- If features are diverse → Hybrid
- Use for final model training

---

## Configuration Files

### YAML Config Template

```yaml
# config/features/selection_methods.yaml

feature_selection:
  # Core parameters
  method: mda  # Options: mdi, mda, hybrid
  n_features_to_select: 50
  min_feature_frequency: 0.6

  # Random Forest parameters
  n_estimators: 100
  max_depth: 5
  random_state: 42

  # MDA-specific
  n_repeats: 10

  # Clustered MDA (optional)
  use_clustered_importance: false
  max_clusters: 20

# Presets for different use cases
presets:
  fast:
    method: mdi
    n_features_to_select: 30
    n_estimators: 50

  default:
    method: mda
    n_features_to_select: 50
    n_estimators: 100

  robust:
    method: mda
    use_clustered_importance: true
    n_features_to_select: 50
    max_clusters: 20
    min_feature_frequency: 0.7
```

### Loading Config

```python
import yaml
from pathlib import Path

config_path = Path("config/features/selection_methods.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

# Use default preset
params = config['presets']['default']
selector = WalkForwardFeatureSelector(**params)

# Or use custom config
params = config['feature_selection']
selector = WalkForwardFeatureSelector(**params)
```

---

## See Also

- [Feature Catalog](./FEATURE_CATALOG.md) - Complete feature documentation
- [Multi-Timeframe Feature Configs](./MTF_FEATURE_CONFIGS.md) - MTF-specific selection
- [Model Feature Requirements](./MODEL_FEATURE_REQUIREMENTS.md) - Per-model feature needs
- [Cross-Validation Guide](/docs/guides/CROSS_VALIDATION_GUIDE.md) - CV best practices
