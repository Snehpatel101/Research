# ML Pipeline Optimization Research
*Compiled 2026-02-08 — Johnny*

---

## 1. Walk-Forward Validation: The Gold Standard

### Why It Matters
Traditional backtesting = one-shot optimization + validation. **90% of academic strategies fail in production** because of:
- Overfitting to in-sample data
- Lookahead bias (using future info)
- Static parameters in dynamic markets

### Walk-Forward Protocol (MANDATORY)
```
For each rolling window:
1. Train on window[t-N : t] (in-sample)
2. Validate on window[t : t+M] (out-of-sample)
3. Record OOS performance
4. Slide window forward by M
5. Repeat until data exhausted
```

**Your ML Factory should use**: 5-year training, 1-year OOS, roll annually
- Train 2020-2024 → Test 2025
- Train 2021-2025 → Test 2026
- etc.

### Key Insight from Research
> "Aggregate statistics mask important regime-dependent heterogeneity" — arxiv:2512.12924

Your regime system (CHOP/STRESS) is right concept but wrong execution. Walk-forward needs **regime-aware windows** — train on similar volatility periods, not just chronological.

---

## 2. PurgedKFold: Preventing Temporal Leakage

### The Problem
Standard KFold leaks info because:
- Fold 3 test sample at t=100 might train on t=99 and t=101
- Auto-correlation in returns = free lookahead

### The Fix: Purge + Embargo
```python
from sklearn.model_selection import PurgedKFold

cv = PurgedKFold(
    n_splits=5,
    purge_gap=10,    # Remove 10 bars BEFORE test
    embargo_gap=10   # Remove 10 bars AFTER test
)
```

**Your audit found**: 6 files now using PurgedKFold ✓
**Still needed**: Ensure ALL optimization stages use it, especially stacking/ensembles

### Embargo Formula
```
embargo_bars = max(
    lookahead_horizon,  # If predicting 5 bars out, embargo >= 5
    autocorrelation_lag,  # Where ACF drops below 0.05
    feature_window  # If using 20-bar features, embargo >= 20
)
```

---

## 3. Optuna Best Practices for Trading

### Sampler Selection
| Task | Sampler | Pruner |
|------|---------|--------|
| Tree models (XGB, LGBM) | TPESampler | HyperbandPruner |
| Neural nets | TPESampler | MedianPruner |
| Hyperband for expensive trials | TPESampler | HyperbandPruner |

### Trading-Specific Objective Functions

**DON'T optimize for**:
- Raw returns (invites leverage/risk)
- Accuracy alone (ignores profitability)
- Sharpe without penalty (inflates in low-vol)

**DO optimize for**:
```python
def objective(trial):
    # Your model params
    params = {
        'learning_rate': trial.suggest_float('lr', 1e-4, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        # etc
    }
    
    # Run walk-forward backtest
    results = walk_forward_test(params)
    
    # Multi-objective: Sharpe + Sortino + Max Drawdown penalty
    score = (
        0.4 * results['sharpe'] +
        0.3 * results['sortino'] +
        0.3 * (1 - abs(results['max_drawdown']))  # Penalize drawdown
    )
    
    # Prune if losing money in OOS
    if results['cumulative_return'] < 0:
        raise optuna.TrialPruned()
    
    return score
```

### Overfitting Detection
```python
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42)
)

# After optimization, check:
in_sample_score = best_trial.value
out_of_sample_score = evaluate_oos(best_trial.params)

overfit_ratio = in_sample_score / out_of_sample_score
if overfit_ratio > 1.5:
    print("⚠️ LIKELY OVERFIT — IS/OOS divergence > 50%")
```

---

## 4. Feature Selection for Profitable Models

### Feature Categories (Priority Order)

**1. Volume/Liquidity Features** (highest signal)
- Dollar volume imbalance
- Bid-ask spread dynamics
- Trade count acceleration
- Volume-weighted price deviation

**2. Volatility Features**
- Realized vol (5/15/30 min)
- VoV (volatility of volatility) — YOU FIXED THIS ✓
- ATR ratios across timeframes
- Parkinson/Garman-Klass estimators

**3. Momentum Features** (careful of decay)
- RSI divergences (not raw RSI)
- MACD histogram acceleration
- Price vs VWAP deviation
- Trend strength (ADX filtered)

**4. Microstructure Features** (if you have tick data)
- Order flow imbalance (OFI)
- Kyle's lambda
- VPIN (Volume-synchronized PIN)

### Feature Engineering Pipeline
```python
def create_features(df):
    # 1. Raw technical indicators
    raw = compute_technicals(df)
    
    # 2. Normalize by volatility
    normalized = raw / df['atr_20']
    
    # 3. Cross-sectional ranks (if multi-asset)
    ranked = normalized.rank(pct=True)
    
    # 4. Lag all features to prevent lookahead
    lagged = ranked.shift(1)
    
    # 5. Remove highly correlated (>0.9)
    final = drop_correlated(lagged, threshold=0.9)
    
    return final
```

### Feature Importance Validation
```python
# SHAP for tree models
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Check feature stability across folds
fold_importances = []
for train, test in cv.split(X):
    model.fit(X[train], y[train])
    fold_importances.append(model.feature_importances_)

# Features should be important in ALL folds, not just some
stable_features = np.std(fold_importances, axis=0) < 0.1
```

---

## 5. Meta-Labeling: The Secret Sauce

### What It Is
Instead of predicting direction, predict **whether to take the trade**.

```
Primary Model → Side prediction (long/short)
Meta Model → Bet sizing (0-1 confidence)
```

### Implementation
```python
# Step 1: Primary model predicts direction
primary_signal = primary_model.predict(X)  # 1=long, -1=short

# Step 2: Meta-labels using Triple Barrier
# Label = 1 if primary signal was PROFITABLE
# Label = 0 if primary signal would have LOST

# Step 3: Train meta-model on different features
meta_features = [
    'volatility_regime',
    'market_breadth', 
    'primary_confidence',
    'time_of_day',
    'days_since_last_signal'
]
meta_model.fit(meta_features, meta_labels)

# Step 4: Execute only when meta says so
if primary_signal != 0 and meta_model.predict_proba() > 0.6:
    execute_trade(primary_signal)
```

### Triple Barrier Method
```
Upper barrier = entry + profit_target * volatility
Lower barrier = entry - stop_loss * volatility  
Vertical barrier = max_holding_period

Label = +1 if upper hit first (profit)
Label = -1 if lower hit first (stop)
Label = 0 if vertical hit (timeout)
```

**Your meta-labeling leakage issue**: Labels were computed on FULL data before train/test split. Fix = compute labels ONLY on training data, then apply same barrier to test.

---

## 6. Pipeline Cohesion Checklist

### Data Flow
```
Raw Data
    ↓
├── Feature Engineering (lag everything!)
├── Label Generation (triple barrier)
    ↓
Walk-Forward Split
    ↓
For each fold:
    ├── Train purged/embargo
    ├── Optuna hyperparam search (inner CV)
    ├── Fit final model
    ├── Meta-label generation
    ├── Meta-model training
    └── OOS evaluation
    ↓
Aggregate OOS metrics
    ↓
Final model ensemble
```

### Cohesion Rules

1. **Single source of truth for config**
   - All embargo_bars from ONE config
   - All horizons from ONE config
   - You fixed this with get_config_horizons() ✓

2. **Temporal consistency**
   - Every stage must respect time ordering
   - No information from future anywhere
   - Validate with causality checks

3. **Reproducibility**
   - Set random seeds everywhere
   - Log all hyperparameters
   - Version control data snapshots

4. **Validation hierarchy**
   ```
   Inner CV (for hyperparams) 
       ⊂ 
   Walk-forward fold (for model selection)
       ⊂
   Holdout test (final sanity check, NEVER touch during dev)
   ```

---

## 7. Recommended Optuna Config for Your Models

### For XGBoost/LightGBM
```python
def xgb_params(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('alpha', 1e-8, 10, log=True),
        'reg_lambda': trial.suggest_float('lambda', 1e-8, 10, log=True),
        'min_child_weight': trial.suggest_int('min_child', 1, 10),
    }
```

### For Neural Nets
```python
def nn_params(trial):
    return {
        'hidden_layers': trial.suggest_int('layers', 1, 4),
        'hidden_units': trial.suggest_int('units', 32, 256),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'learning_rate': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch', [32, 64, 128, 256]),
        'weight_decay': trial.suggest_float('wd', 1e-6, 1e-2, log=True),
    }
```

### Trial Budget
- Quick exploration: 50 trials
- Serious optimization: 200-500 trials
- Final production: 1000+ trials with early stopping

---

## 8. Key Takeaways for Profitable Models

1. **Walk-forward is non-negotiable** — single backtest is useless
2. **Purge/embargo everywhere** — you're fixing this ✓
3. **Optimize for risk-adjusted returns, not raw returns**
4. **Meta-labeling > raw direction prediction**
5. **Features must be lagged and volatility-normalized**
6. **Check IS/OOS ratio — if > 1.5x, you overfit**
7. **Regime-aware training windows** — don't mix CHOP with TREND
8. **Kill correlated features (>0.9)** — they add noise not signal

---

## Sources

- arxiv:2512.12924 — Walk-Forward Validation Framework
- QuantInsti Blog — WFO Implementation
- Hudson & Thames — Meta-Labeling Research
- Optuna Documentation — Sampler/Pruner Best Practices
- Marcos Lopez de Prado — Advances in Financial Machine Learning


---

## 9. ML FACTORY SPECIFIC FIXES (Your 9 CRITICALs)

### Issue 1: Optimization Stages Use 80/20 Split Without Purge
**Location**: `src/optimization/pipeline.py:398-405`

**Problem**: Triple-barrier labels look ahead `max_bars` (up to 180 bars). Samples near split boundary leak.

**Fix**:
```python
# BEFORE (broken)
split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]

# AFTER (correct)
from sklearn.model_selection import PurgedKFold
embargo = max(config.max_bars, 180)  # At least as long as label horizon
cv = PurgedKFold(n_splits=5, purge_gap=embargo, embargo_gap=embargo)
for train_idx, val_idx in cv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
```

### Issue 2: Hardcoded embargo_bars=10 in model_trainer.py
**Location**: `src/models/training/model_trainer.py:420`

**Problem**: 10 bars = 10 min at 1-min resolution. Labels can look ahead 180 bars.

**Fix**:
```python
# BEFORE
embargo_bars = 10

# AFTER - Use config with minimum safety floor
embargo_bars = max(
    self.config.get('embargo_bars', 180),
    self.config.get('label_max_bars', 180),
    180  # Safety floor
)
```

### Issue 3: Meta-Labeling Primary Model Sees Future
**Location**: `src/data/pipeline/stages/meta_labeling/run.py:383-418`

**Problem**: Primary classifier trained on ALL data including test set, then generates features for training.

**Fix**:
```python
# Split FIRST, then train primary model on train only
train_idx = get_train_indices(df, config)
primary_model.fit(X[train_idx], y[train_idx])

# Generate meta-labels ONLY for train set
meta_labels = primary_model.predict_proba(X[train_idx])

# For test: DON'T generate meta-labels, or use walk-forward
```

### Issue 4: Same Validation Set for 4 Optimization Stages = 6.25M Implicit Trials
**Location**: `src/optimization/pipeline.py`

**Problem**: 50 trials × 4 stages = 50^4 effective combinations on ONE val set.

**Fix**:
```python
# Different temporal windows per stage (already in your arch docs!)
stage_splits = {
    'label_opt': {'train': 'Q1', 'val': 'Q2'},
    'feature_select': {'train': 'Q1+Q2', 'val': 'Q3'},
    'feature_prune': {'train': 'Q1+Q2+Q3', 'val': 'Q4'},
    'hyperparam_opt': {'train': 'Q1+Q2+Q3+Q4', 'val': 'Q5'},  # Holdout
}
```

### Issue 5: 5D Objective Uses Wrong Proxy Model
**Location**: `src/optimization/five_dimension_objective.py:798-815`

**Problem**: Optimizing for RF proxy when target is LSTM/PatchTST/etc.

**Fix Options**:
1. Use actual target model (slower but accurate)
2. Use model-family proxies (XGB proxy for boosters, small LSTM for RNNs)
3. Discount RF-optimized params when applying to neural nets

```python
# At minimum, match complexity class
if target_family == 'neural':
    proxy = MLPClassifier(hidden_layer_sizes=(64,), max_iter=100)
elif target_family == 'boosting':
    proxy = LGBMClassifier(n_estimators=50, max_depth=5)
else:
    proxy = RandomForestClassifier(n_estimators=50, max_depth=5)
```

### Issue 6: MDA Importance on Training Data
**Location**: `src/optimization/feature_selection/walk_forward.py:223-236`

**Problem**: `permutation_importance(model, X_train, y_train)` — overfit features look important.

**Fix**:
```python
# BEFORE
rf.fit(X, y)
importance = permutation_importance(rf, X, y)

# AFTER - Use OOB or holdout
rf = RandomForestClassifier(n_estimators=100, oob_score=True, max_samples=0.7)
rf.fit(X_train, y_train)

# Option A: OOB importance
oob_indices = ~rf.estimators_samples_  # Bootstrap excluded samples
importance = permutation_importance(rf, X[oob_indices], y[oob_indices])

# Option B: Separate holdout
importance = permutation_importance(rf, X_holdout, y_holdout)
```

### Issue 7: Multi-Stream TF Alignment via np.repeat
**Location**: `src/data/adapters/multi_stream.py`

**Problem**: Higher TF data repeated by ratio, ignoring gaps/holidays.

**Fix**:
```python
# Use actual timestamp alignment
def align_higher_tf(anchor_df, higher_tf_df):
    """Forward-fill higher TF to anchor timestamps."""
    # Reindex higher TF to anchor index
    aligned = higher_tf_df.reindex(anchor_df.index, method='ffill')
    
    # Add feature indicating freshness
    aligned['bars_since_update'] = (
        anchor_df.index - aligned.index.to_series().shift(1)
    ).dt.total_seconds() // 60
    
    return aligned
```

### Issue 8: Stacking OOF Leakage (Safe Mode Exists but Can Be Disabled)
**Location**: `src/models/ensemble/stacking.py:355-397`

**Fix**: Remove the unsafe path entirely.
```python
# DELETE this option
# use_default_configs_for_oof=False  # NEVER ALLOW THIS

# ALWAYS use default (untuned) configs for OOF generation
# Then tune the meta-learner separately
```

### Issue 9: DataFrame Checksum Sampling
**Location**: `src/data/store/cache.py:609-647`

**Problem**: Intermediate rows can differ, same hash returned.

**Fix**:
```python
# BEFORE
sample_rows = df.iloc[[0, -1] + random.sample(range(len(df)), 3)]
checksum = hash(sample_rows.values.tobytes())

# AFTER - Full content hash (use xxhash for speed)
import xxhash
def compute_checksum(df):
    return xxhash.xxh128(df.values.tobytes()).hexdigest()

# Or if too slow, use statistical fingerprint
def compute_checksum_fast(df):
    fingerprint = (
        df.shape,
        tuple(df.dtypes),
        df.values.sum(),  # Sum catches most changes
        df.values.std(),
        hash(df.iloc[::100].values.tobytes())  # Strided sample
    )
    return hash(fingerprint)
```

---

## 10. Optuna Config Specific to Your Models

Based on your model registry:

### For Boosting Family (xgboost, lightgbm, catboost)
```python
def boosting_objective(trial, model_name='xgboost'):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('lr', 0.005, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('alpha', 1e-8, 100, log=True),
        'reg_lambda': trial.suggest_float('lambda', 1e-8, 100, log=True),
    }
    if model_name == 'catboost':
        params['l2_leaf_reg'] = params.pop('reg_lambda')
    return params
```

### For Neural Family (lstm, gru, tcn, nbeats, resnet1d, inceptiontime)
```python
def neural_objective(trial, model_name='lstm'):
    params = {
        'hidden_size': trial.suggest_int('hidden', 32, 256),
        'num_layers': trial.suggest_int('layers', 1, 4),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'learning_rate': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch', [32, 64, 128]),
        'weight_decay': trial.suggest_float('wd', 1e-7, 1e-3, log=True),
    }
    if model_name in ['tcn', 'inceptiontime', 'resnet1d']:
        params['kernel_size'] = trial.suggest_int('kernel', 2, 7)
        params['num_filters'] = trial.suggest_int('filters', 32, 128)
    return params
```

### For Transformer Family (patchtst, itransformer)
```python
def transformer_objective(trial):
    return {
        'd_model': trial.suggest_categorical('d_model', [64, 128, 256]),
        'n_heads': trial.suggest_categorical('n_heads', [4, 8]),
        'n_layers': trial.suggest_int('n_layers', 1, 4),
        'patch_len': trial.suggest_int('patch_len', 8, 32),
        'stride': trial.suggest_int('stride', 4, 16),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
    }
```

### For Meta-Learners (ridge_meta, xgboost_meta, mlp_meta)
```python
def meta_learner_objective(trial):
    # Meta-learner sees OOF predictions (n_base_models × n_classes)
    # Keep it simple to avoid double-overfitting
    return {
        'alpha': trial.suggest_float('alpha', 0.01, 100, log=True),  # Ridge
        'hidden': trial.suggest_int('hidden', 16, 64),  # MLP
        'max_depth': trial.suggest_int('depth', 2, 5),  # XGB meta
    }
```

---

## 11. Your Regime Features (ML Factory)

Your `regime.py` computes volatility-based regimes comparing short vs long-term vol.

**Enhancement suggestions**:
```python
# 1. Add trend regime for regime-aware training
def compute_composite_regime(df, vol_window=20, trend_window=50):
    vol_regime = compute_volatility_regime(df)  # Your existing
    trend_regime = compute_trend_regime(df)     # New
    
    # Composite: 4 states
    # 0 = low_vol + no_trend (ranging)
    # 1 = low_vol + trend (clean trend)
    # 2 = high_vol + no_trend (choppy)
    # 3 = high_vol + trend (crisis/breakout)
    return vol_regime * 2 + trend_regime

# 2. Use regime as training filter
def get_regime_filtered_data(df, target_regime):
    """Train only on similar regime data."""
    regime = compute_composite_regime(df)
    return df[regime == target_regime]
```

