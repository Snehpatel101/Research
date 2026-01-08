# Feature Selection Guide for ML Factory Pipeline

## Executive Summary

This guide synthesizes research best practices and pipeline-specific analysis to provide comprehensive feature selection recommendations for the OHLCV ML Factory. The factory implements 23 models across 4 families with a sophisticated walk-forward feature selection system, but several optimization opportunities exist.

**Key Finding:** Your pipeline has ~168 features from Phase 1 with an additional ~70 MTF features (~238 total). For typical training datasets of 50-100K samples, this creates borderline sample-to-feature ratios that can lead to overfitting, particularly for tabular models.

---

## Part 1: Understanding Your Current Architecture

### 1.1 Feature Selection Infrastructure (3,980+ lines)

Your pipeline implements a **three-tier feature selection architecture**:

```
Tier 1: Feature Set Pre-Selection (Phase 1 - config)
  └── 9 predefined feature sets tailored to model families
        ├── boosting_optimal: 50-100 features
        ├── neural_optimal: 20-30 features
        ├── transformer_raw: 5-10 features (minimal)
        └── ... 6 more specialized sets

Tier 2: Walk-Forward Selection (Phase 6 - training)
  └── MDA/MDI/Hybrid importance with PurgedKFold CV
        ├── Per-fold importance calculation
        ├── Stability tracking (min_feature_frequency)
        └── Clustered MDA for multicollinearity

Tier 3: Persistence (Inference)
  └── PersistedFeatureSelection saved alongside model
        ├── selected_features + indices
        ├── stability_scores + importance_scores
        └── Reproducible inference path
```

### 1.2 Current Model-Family Defaults

| Family | Selection Enabled | n_features | Method | min_frequency |
|--------|------------------|------------|--------|---------------|
| **Boosting** (XGBoost, LightGBM, CatBoost) | Yes | 50 | MDA | 0.6 |
| **Classical** (RF, Logistic, SVM) | Yes | 40 | MDA | 0.7 |
| **Neural** (LSTM, GRU, TCN, etc.) | **No** | 0 (all) | - | - |
| **Ensemble** | No | 0 | - | - |

### 1.3 Feature Counts by Pipeline Stage

```
Stage                    Feature Count
─────────────────────────────────────
Base OHLCV derivs        ~12 features (returns, ranges, body, wicks)
Momentum indicators      ~25 features (RSI, MACD, Stochastic, etc.)
Trend indicators         ~20 features (ADX, Supertrend, MAs)
Volatility indicators    ~25 features (ATR, BB, KC, HVol variants)
Volume indicators        ~15 features (OBV, VWAP, ratios)
Wavelets/Advanced        ~20 features (DWT decomposition)
Microstructure           ~10 features (Roll, Amihud, Kyle lambda)
Temporal/Session         ~15 features (hour, day, session flags)
Regime features          ~10 features (trend_regime, vol_regime)
─────────────────────────────────────
SUBTOTAL (Base)          ~152-168 features

MTF Features (5 TFs)     ~70 features (same indicators x timeframes)
─────────────────────────────────────
TOTAL (Full MTF)         ~220-238 features
```

---

## Part 2: The Feature Selection Problem

### 2.1 Sample-to-Feature Ratio Analysis

Based on Lopez de Prado and current ML research, these are the recommended ratios:

| Model Family | Minimum Ratio | Recommended | Your Pipeline Risk |
|--------------|---------------|-------------|-------------------|
| Tabular (Boosting) | 20:1 | 30:1 | **Medium** - ~168 features needs 5K+ samples |
| Classical (RF, SVM) | 30:1 | 50:1 | **High** - 40 features after selection OK |
| Neural (LSTM/GRU) | 50:1 | 100:1 | **High** - currently getting ALL features |
| Transformer | 100:1 | 500:1+ | **OK** - designed for high-dimensional |

**For a typical 1-year 5-min dataset (~75K samples):**
- 168 features → 446:1 ratio (acceptable)
- 238 features (MTF) → 315:1 ratio (borderline)

**For a 6-month dataset (~37K samples):**
- 168 features → 223:1 ratio (risky for classical)
- 238 features → 158:1 ratio (concerning)

### 2.2 The Core Question: "Am I Shoving Too Many Features?"

**Answer: It depends on your model family and dataset size.**

**For Tabular Models (XGBoost, LightGBM, CatBoost):**
- With 75K+ samples: ~150 features is acceptable
- With <50K samples: Reduce to 50-80 features
- **Recommendation:** Use `boosting_optimal` set + walk-forward selection to 50 features

**For Classical Models (Random Forest, Logistic, SVM):**
- These are more sensitive to the curse of dimensionality
- **Recommendation:** Aggressive selection to 30-40 features

**For Neural Models (LSTM, GRU, TCN):**
- Currently receiving ALL features (feature selection disabled)
- This is problematic for small datasets
- **Recommendation:** Pre-select 40-60 features for LSTM/GRU/TCN

**For Transformers (PatchTST, iTransformer, TFT):**
- Designed to learn from high-dimensional raw data
- Internal attention mechanisms act as learned feature selection
- **Recommendation:** Keep feature selection disabled; use minimal preprocessing

---

## Part 3: How to Find Optimal Features

### 3.1 Step-by-Step Protocol (Leakage-Safe)

```
STEP 1: Start with a Smaller Candidate Universe
─────────────────────────────────────────────
Instead of mtf_plus (238 features), start with:
- boosting_optimal (50-100) for XGBoost/LightGBM/CatBoost
- neural_optimal (20-30) for LSTM/GRU/TCN
- transformer_raw (5-10) for PatchTST/iTransformer

Why: Optimizing within a huge noisy space introduces variance.


STEP 2: Treat "#features" as a Hyperparameter
─────────────────────────────────────────────
Run walk-forward selection with different n_features targets:

python scripts/run_cv.py \
  --models xgboost \
  --horizons 20 \
  --feature-selection-sweep 20,40,60,80,100

Track: OOF balanced accuracy, Sharpe, profit factor per sweep value


STEP 3: Require Stability, Not Just Performance
─────────────────────────────────────────────
A feature selected in 5/5 folds (stability=1.0) is more reliable
than a feature selected in 2/5 folds (stability=0.4).

Configure:
  min_feature_frequency: 0.6  # At least 60% of folds

Features appearing inconsistently are likely fitting noise.


STEP 4: Prune Redundancy
─────────────────────────────────────────────
Enable clustered importance to avoid keeping 10 variants of RSI:

python scripts/train_model.py \
  --model xgboost \
  --use-clustered-importance \
  --max-clusters 20

This groups correlated features and distributes importance.


STEP 5: Lock Per (Symbol, Horizon, Model) and Judge Once
─────────────────────────────────────────────
Feature selection should be:
- Re-run each training window (walk-forward)
- Specific to model and prediction horizon
- NEVER tuned on test performance

Save with model: experiments/runs/{run_id}/feature_selection.json
```

### 3.2 Concrete Starting Configurations

Based on your pipeline analysis, here are recommended starting points:

#### For CatBoost/XGBoost/LightGBM (Boosting Family)

```python
# TrainerConfig settings
use_feature_selection = True
feature_selection_n_features = 50      # Start here, sweep 30-80
feature_selection_method = "mda"       # MDA handles multicollinearity better
feature_selection_cv_splits = 5

# Advanced: enable clustered importance for highly correlated features
use_clustered_importance = True
max_clusters = 15
```

**Why 50 features?**
- Sweet spot for boosting models based on research
- Your defaults already use this; validated in AFML
- Gives 1500:1 ratio with 75K samples (very safe)

#### For Random Forest/Logistic/SVM (Classical Family)

```python
# TrainerConfig settings
use_feature_selection = True
feature_selection_n_features = 40      # More aggressive
feature_selection_method = "mda"
feature_selection_cv_splits = 5
min_feature_frequency = 0.7            # Higher stability requirement
```

**Why 40 features with higher stability?**
- Classical models more sensitive to noise features
- Logistic regression benefits from interpretable feature sets
- SVM suffers in high dimensions without aggressive selection

#### For LSTM/GRU/TCN (Basic Sequence Models)

**CURRENT:** Feature selection disabled (all features passed through)

**RECOMMENDED:** Enable pre-selection for these models:

```python
# Proposed change to ModelFamilyDefaults in config.py
NEURAL_BASIC = {
    "enabled": True,         # Change from False
    "n_features": 50,        # Pre-select 50 features
    "method": "mda",
    "min_feature_frequency": 0.5,
    "n_estimators": 50,
}
```

**Why enable for LSTM/GRU/TCN?**
- These models don't have sophisticated feature selection mechanisms
- Dropout helps but doesn't replace feature selection
- Too many features leads to longer training, worse generalization
- Research shows tree-based pre-selection improves NN performance

#### For Transformer/PatchTST/iTransformer/TFT (Advanced Sequence)

**CURRENT:** Feature selection disabled - **KEEP THIS**

```python
# Keep current settings
NEURAL_ADVANCED = {
    "enabled": False,        # Correct - keep disabled
    "n_features": 0,
    "method": "mda",
    "min_feature_frequency": 0.5,
}
```

**Why keep disabled?**
- Attention mechanisms learn feature importance internally
- PatchTST/iTransformer designed for high-dimensional time series
- External selection can remove features the model learns to use
- Use `transformer_raw` or `patchtst_optimal` feature sets for minimal preprocessing

---

## Part 4: Advanced Feature Selection Strategies

### 4.1 Regime-Aware Feature Selection

Your pipeline has regime detection (`src/phase1/stages/regime/`) but doesn't use it for feature selection.

**Implementation Strategy:**

```python
# Approach 1: Union of Regime-Specific Features
def regime_aware_selection(X, y, regime_labels):
    regime_features = {}

    for regime in ['trending', 'mean_reverting', 'volatile']:
        mask = regime_labels == regime
        X_regime = X[mask]
        y_regime = y[mask]

        # Run selection on each regime
        selector = WalkForwardFeatureSelector(n_features=40)
        selected = selector.select_features(X_regime, y_regime)
        regime_features[regime] = set(selected)

    # Core: features important in ALL regimes
    core = regime_features['trending'] & regime_features['mean_reverting'] & regime_features['volatile']

    # Conditional: regime-specific additions
    conditional = {
        'trending': regime_features['trending'] - core,
        'mean_reverting': regime_features['mean_reverting'] - core,
        'volatile': regime_features['volatile'] - core,
    }

    return core, conditional
```

**Benefit:** Different features work in different market conditions. RSI works well in mean-reverting markets but poorly in trends.

### 4.2 MTF Feature Selection Strategy

Your MTF features (~70 additional) have high inter-timeframe correlation.

**Problem:** `rsi_14` at 5-min is highly correlated with `rsi_14` at 15-min.

**Solution: MTF-Specific Correlation Pruning**

```python
# In feature engineering, add MTF correlation pruning
def prune_mtf_redundancy(features_df, base_tf='5min', correlation_threshold=0.85):
    """
    For each indicator, keep only the most informative timeframe.
    """
    mtf_groups = {}

    # Group by base indicator name
    for col in features_df.columns:
        base_name = extract_base_indicator(col)  # e.g., 'rsi_14' from 'rsi_14_15m'
        if base_name not in mtf_groups:
            mtf_groups[base_name] = []
        mtf_groups[base_name].append(col)

    features_to_keep = []

    for base_name, variants in mtf_groups.items():
        if len(variants) == 1:
            features_to_keep.append(variants[0])
            continue

        # Keep base timeframe + one MTF with lowest correlation
        correlations = features_df[variants].corr()
        # ... selection logic

    return features_to_keep
```

### 4.3 SHAP Integration for Interpretability

Your codebase has SHAP imports but they're not integrated with feature selection.

**Recommended Integration:**

```python
import shap

def shap_feature_importance(model, X_train, X_val):
    """
    Compute SHAP values for interpretable feature importance.
    Works with XGBoost, LightGBM, CatBoost, Random Forest.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    # Global importance: mean absolute SHAP value
    importance = np.abs(shap_values).mean(axis=0)

    return pd.Series(importance, index=X_train.columns).sort_values(ascending=False)
```

**Benefits over MDA/MDI:**
- More accurate for correlated features
- Provides per-sample explanations
- Industry standard for model interpretability
- Can detect feature interactions

### 4.4 CPCV for Feature Selection Validation

Your pipeline has CPCV in `scripts/run_cpcv_pbo.py` but it's for model validation, not feature selection.

**Adaptation for Feature Selection:**

```python
def cpcv_feature_selection(X, y, n_features_candidates=[30, 50, 70, 100]):
    """
    Use CPCV to validate feature selection hyperparameter.
    """
    from src.cross_validation.cpcv import CPCV

    results = {}

    for n_features in n_features_candidates:
        # Run feature selection + model training under CPCV
        cpcv = CPCV(n_splits=10, n_test_splits=2)

        metrics = []
        for train_idx, test_idx in cpcv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Feature selection on train only
            selector = WalkForwardFeatureSelector(n_features=n_features)
            selected = selector.select_features(X_train, y_train)

            # Train and evaluate
            model = XGBClassifier()
            model.fit(X_train[selected], y_train)
            preds = model.predict_proba(X_test[selected])

            metrics.append(calculate_metrics(y_test, preds))

        results[n_features] = {
            'mean_sharpe': np.mean([m['sharpe'] for m in metrics]),
            'pbo': calculate_pbo(metrics)  # Probability of Backtest Overfitting
        }

    return results
```

---

## Part 5: Specific Recommendations for Your Pipeline

### 5.1 Immediate Optimizations (High Impact, Low Effort)

#### 1. Enable Feature Selection for Basic Neural Models

**File:** `src/models/feature_selection/config.py`

```python
# Change NEURAL defaults to separate basic vs advanced
NEURAL_BASIC = {  # For LSTM, GRU, TCN
    "enabled": True,          # Changed from False
    "n_features": 50,         # Pre-select top 50
    "method": "mda",
    "min_feature_frequency": 0.5,
    "n_estimators": 50,
}

NEURAL_ADVANCED = {  # For Transformer, PatchTST, iTransformer, TFT
    "enabled": False,         # Keep disabled
    "n_features": 0,
    "method": "mda",
    "min_feature_frequency": 0.5,
    "n_estimators": 50,
}
```

**Why:** LSTM/GRU/TCN don't have built-in feature selection mechanisms. Feeding 168+ features causes:
- Longer training times
- Higher memory usage
- Worse generalization on small datasets

#### 2. Enable Clustered Importance by Default

**File:** `src/models/feature_selection/config.py`

```python
BOOSTING = {
    "enabled": True,
    "n_features": 50,
    "method": "mda",
    "min_feature_frequency": 0.6,
    "n_estimators": 100,
    "use_clustered_importance": True,   # Changed from False
    "max_clusters": 15,
}
```

**Why:** Your feature set has many correlated indicators:
- RSI_14 correlated with Stochastic_K
- ATR_14 correlated with Bollinger_Width
- Multiple MA variants highly correlated

Clustered importance prevents selecting 5 variants of the same signal.

#### 3. Add Correlation Pre-Filtering

Before walk-forward selection, prune obviously redundant features:

```python
# In manager.py or as preprocessing step
def correlation_prefilter(X_df, threshold=0.95):
    """Remove features with >95% correlation to another feature."""
    corr_matrix = X_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return X_df.drop(columns=to_drop)
```

### 5.2 Medium-Term Improvements (High Impact, Medium Effort)

#### 1. Per-Model Feature Sets in Heterogeneous Ensembles

**Current:** All base models in stacking use same features.

**Recommended:** Each base model gets its optimized feature set:

```python
# In trainer.py for heterogeneous stacking
def train_heterogeneous_ensemble(base_models, X, y):
    base_predictions = []

    for model_name in base_models:
        family = get_model_family(model_name)

        if family == 'boosting':
            feature_set = 'boosting_optimal'  # ~50 features
            fs_enabled = True
        elif family == 'neural' and model_name in ['lstm', 'gru', 'tcn']:
            feature_set = 'neural_optimal'    # ~30 features
            fs_enabled = True
        elif family == 'neural':  # transformer, patchtst, etc.
            feature_set = 'transformer_raw'   # ~10 features
            fs_enabled = False

        X_model = apply_feature_set(X, feature_set)

        if fs_enabled:
            selector = FeatureSelectionManager.from_model_family(family)
            X_model = selector.select_and_apply(X_model, y)

        model = ModelRegistry.get(model_name)()
        model.fit(X_model, y)
        base_predictions.append(model.predict_proba(X_model))

    return stack_predictions(base_predictions)
```

**Benefit:** Maximizes diversity in heterogeneous ensembles. Different models see different features → less correlated errors.

#### 2. Integrate SHAP for Feature Selection

Add SHAP-based importance as an option alongside MDA/MDI:

```python
# In feature_selector.py
def _shap_importance(self, X, y):
    """Compute SHAP-based feature importance."""
    model = LGBMClassifier(n_estimators=100, random_state=self.random_state)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):  # Multi-class
        importance = np.abs(np.array(shap_values)).mean(axis=0).mean(axis=0)
    else:
        importance = np.abs(shap_values).mean(axis=0)

    return importance
```

Then add to config:
```python
selection_method: str = "shap"  # New option alongside mda, mdi, hybrid
```

#### 3. Add Feature Stability Report to Training Output

After each training run, generate a stability report:

```python
def generate_stability_report(feature_selection_result):
    """
    Generate report showing which features are stable vs unstable.
    """
    stability = feature_selection_result.stability_scores

    very_stable = [f for f, s in stability.items() if s >= 0.8]
    stable = [f for f, s in stability.items() if 0.6 <= s < 0.8]
    unstable = [f for f, s in stability.items() if s < 0.6]

    report = f"""
    Feature Stability Report
    ========================
    Very Stable (80%+ folds): {len(very_stable)} features
      {', '.join(very_stable[:10])}{'...' if len(very_stable) > 10 else ''}

    Stable (60-80% folds): {len(stable)} features
      {', '.join(stable[:10])}{'...' if len(stable) > 10 else ''}

    Unstable (<60% folds): {len(unstable)} features
      {', '.join(unstable[:10])}{'...' if len(unstable) > 10 else ''}

    Recommendation: Consider removing unstable features from candidate pool.
    """
    return report
```

### 5.3 Long-Term Enhancements (High Impact, High Effort)

#### 1. Regime-Aware Feature Selection Pipeline

Add regime conditioning to feature selection:

```python
# New module: src/models/feature_selection/regime_aware.py
class RegimeAwareFeatureSelector:
    """
    Selects features conditionally on market regime.
    """
    def __init__(self, core_features=30, conditional_features=10):
        self.core_features = core_features
        self.conditional_features = conditional_features

    def select_features(self, X, y, regime_labels):
        # 1. Get regime-specific importance
        regime_importance = {}
        for regime in np.unique(regime_labels):
            mask = regime_labels == regime
            importance = self._compute_importance(X[mask], y[mask])
            regime_importance[regime] = importance

        # 2. Find core features (important in ALL regimes)
        core = self._find_intersection(regime_importance, n=self.core_features)

        # 3. Find conditional features (regime-specific)
        conditional = {}
        for regime, importance in regime_importance.items():
            remaining = importance.drop(core)
            conditional[regime] = remaining.nlargest(self.conditional_features).index.tolist()

        return RegimeAwareSelection(core=core, conditional=conditional)
```

#### 2. Temporal Feature Decay Tracking

Some features lose predictive power over time (alpha decay):

```python
# New module: src/models/feature_selection/temporal_decay.py
class TemporalDecayTracker:
    """
    Tracks feature importance over time to identify decaying features.
    """
    def track_decay(self, X, y, window_size=5000, n_windows=10):
        importance_history = []

        for i in range(n_windows):
            start = i * window_size
            end = (i + 1) * window_size
            X_window = X[start:end]
            y_window = y[start:end]

            importance = self._compute_importance(X_window, y_window)
            importance_history.append(importance)

        # Compute decay metrics
        decay_scores = {}
        for feature in X.columns:
            ranks = [imp.rank()[feature] for imp in importance_history]
            # Positive slope = improving, negative = decaying
            decay_scores[feature] = np.polyfit(range(len(ranks)), ranks, 1)[0]

        return decay_scores
```

#### 3. Automated Feature Engineering Evaluation

Integrate with tsfresh for automated feature discovery:

```python
from tsfresh import extract_features
from tsfresh.feature_selection import select_features

def discover_new_features(X_ohlcv, y, max_features=20):
    """
    Use tsfresh to discover potentially useful features not in current set.
    """
    # Extract all possible features
    extracted = extract_features(X_ohlcv, column_id='symbol', column_sort='timestamp')

    # Select only significant features
    selected = select_features(extracted, y, fdr_level=0.01)

    # Return top features not already in pipeline
    current_features = get_current_feature_names()
    new_features = [f for f in selected.columns if f not in current_features]

    return new_features[:max_features]
```

---

## Part 6: Quick Reference Tables

### 6.1 Recommended Configurations by Model

| Model | Feature Set | n_features | Method | Clustered | Notes |
|-------|-------------|------------|--------|-----------|-------|
| XGBoost | boosting_optimal | 50 | MDA | Yes | Default recommended |
| LightGBM | boosting_optimal | 50 | MDA | Yes | Same as XGBoost |
| CatBoost | boosting_optimal | 40-50 | MDA | Yes | Handles categoricals natively |
| Random Forest | boosting_optimal | 40 | MDA | Yes | More aggressive reduction |
| Logistic | core_min | 30 | MDA | No | Fewer features, more interpretable |
| SVM | neural_optimal | 30 | MDA | No | Dimensionality sensitive |
| LSTM | neural_optimal | 50 | MDA | No | Enable selection (currently disabled) |
| GRU | neural_optimal | 50 | MDA | No | Enable selection (currently disabled) |
| TCN | tcn_optimal | 50 | MDA | No | Enable selection (currently disabled) |
| Transformer | transformer_raw | ALL | N/A | N/A | Keep selection disabled |
| PatchTST | patchtst_optimal | ALL | N/A | N/A | Minimal preprocessing |
| iTransformer | transformer_raw | ALL | N/A | N/A | Raw OHLCV preferred |
| TFT | transformer_raw | ALL | N/A | N/A | Has built-in variable selection |

### 6.2 Sample Size Guidelines

| Samples (After Split) | Max Features (Tabular) | Max Features (Neural) | Recommendation |
|----------------------|----------------------|---------------------|----------------|
| 10,000 | 100 | 50 | Use core_min, aggressive selection |
| 25,000 | 150 | 60 | Use boosting_optimal, select to 50 |
| 50,000 | 200 | 80 | Standard pipeline settings OK |
| 100,000+ | 300+ | 100+ | Can relax constraints |

### 6.3 When to Use Each Importance Method

| Method | Best For | Avoid When |
|--------|----------|------------|
| **MDI** | Quick prototyping, initial exploration | Highly correlated features |
| **MDA** | Production, correlated features | Speed is critical |
| **Hybrid** | Robust selection, safety | Interpretability needed |
| **Clustered MDA** | Many correlated indicators (RSI variants, MAs) | Low correlation features |
| **SHAP** | Interpretability, explaining to stakeholders | Time-sensitive pipelines |

### 6.4 Anti-Patterns to Avoid

| Don't Do This | Do This Instead |
|---------------|-----------------|
| Select features using test data | Use PurgedKFold on train only |
| Keep all 238 MTF features | Apply correlation pruning first |
| Same features for all models | Model-family-specific feature sets |
| Feature selection disabled for LSTM | Enable with 50 features |
| MDI without clustering | Use MDA or clustered MDI |
| Single-period selection | Walk-forward with stability tracking |
| Tune n_features on test Sharpe | Use CPCV for selection validation |

---

## Part 7: Implementation Checklist

### Immediate Actions (This Week)

- [ ] Enable feature selection for LSTM/GRU/TCN in `ModelFamilyDefaults`
- [ ] Set `use_clustered_importance=True` for boosting family
- [ ] Add correlation pre-filtering (threshold=0.95) before selection
- [ ] Verify your training dataset has >50K samples after split

### Short-Term (Next 2 Weeks)

- [ ] Implement per-model feature sets for heterogeneous ensembles
- [ ] Add SHAP as feature selection method option
- [ ] Create feature stability report as training artifact
- [ ] Sweep n_features=[30, 50, 70] for your key models

### Medium-Term (Next Month)

- [ ] Implement regime-aware feature selection
- [ ] Add MTF-specific correlation pruning
- [ ] Integrate CPCV for feature selection hyperparameter validation
- [ ] Build automated feature stability monitoring

---

## Summary

Your ML Factory pipeline has sophisticated feature selection infrastructure. The key optimizations are:

1. **Enable selection for basic neural models** (LSTM, GRU, TCN) - they don't have built-in selection mechanisms

2. **Enable clustered importance** - your feature set has many correlated indicators

3. **Use model-specific feature sets** - transformers want raw data, boosting wants engineered features

4. **Validate with CPCV** - ensures feature selection isn't overfitting to specific periods

5. **Track stability** - features appearing in <60% of folds are likely noise

The goal is not minimizing features, but **finding the right features for each model family** while maintaining strict leakage prevention through PurgedKFold and proper purge/embargo settings.
