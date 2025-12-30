# Feature Engineering Documentation - Summary

**Complete reference for ML Model Factory feature engineering and selection**

This directory contains comprehensive documentation for the feature engineering pipeline, including exact formulas, parameters, selection strategies, and model-specific recommendations.

---

## Documentation Structure

### Core Documentation

| Document | Description | Features Documented |
|----------|-------------|---------------------|
| **[FEATURE_CATALOG.md](./FEATURE_CATALOG.md)** | Complete feature reference with exact formulas and parameters | 175+ base features |
| **[FEATURE_SELECTION_CONFIGS.md](./FEATURE_SELECTION_CONFIGS.md)** | MDA/MDI/SHAP selection methods and configurations | 4 selection methods |
| **[MTF_FEATURE_CONFIGS.md](./MTF_FEATURE_CONFIGS.md)** | Multi-timeframe feature strategies and optimization | 6 MTF strategies |

### Configuration Files

| Config File | Purpose | Presets |
|-------------|---------|---------|
| **`config/features/selection_methods.yaml`** | Feature selection method configurations | 7 presets (fast, default, robust, etc.) |
| **`config/features/mtf_strategies.yaml`** | MTF timeframe strategies | 7 strategies (default, minimal, aggressive, etc.) |
| **`config/features/model_features.yaml`** | Model-specific feature requirements | 13 models (configs for all) |

---

## Quick Reference

### Feature Catalog Statistics

**Total Base Features: ~175**

| Category | Count | Example Features |
|----------|-------|-----------------|
| Price Features | 20 | `return_5`, `return_20`, `hl_ratio`, `clv` |
| Moving Averages | 25 | `sma_20`, `sma_50`, `ema_21`, `price_to_sma_50` |
| Momentum | 23 | `rsi_14`, `macd_hist`, `stoch_k`, `roc_10` |
| Volatility | 34 | `atr_14`, `bb_width`, `hvol_20`, `yz_vol` |
| Volume | 13 | `obv`, `vwap`, `volume_ratio`, `dollar_volume` |
| Trend | 6 | `adx_14`, `plus_di_14`, `supertrend` |
| Temporal | 9 | `hour_sin`, `hour_cos`, `session_asia` |
| Regime | 2 | `volatility_regime`, `trend_regime` |
| Microstructure | 19 | `micro_amihud`, `micro_roll_spread`, `micro_kyle_lambda` |
| Wavelets | 24 | `wavelet_close_approx`, `wavelet_close_d1`, energy features |

**MTF Features (Variable):**
- Per timeframe: ~20-25 features
- Default (15min + 60min): ~40-50 features
- Aggressive (5 timeframes): ~100-125 features

---

## Feature Selection Quick Start

### Default MDA Selection

**Python:**
```python
from src.cross_validation import WalkForwardFeatureSelector

selector = WalkForwardFeatureSelector(
    n_features_to_select=50,
    selection_method="mda",
    min_feature_frequency=0.6,
    random_state=42
)

result = selector.select_features_walkforward(X, y, cv_splits)
stable_features = result.stable_features
print(f"Selected {len(stable_features)} stable features")
```

**YAML Config:**
```yaml
# config/features/selection_methods.yaml
feature_selection:
  method: mda
  n_features_to_select: 50
  min_feature_frequency: 0.6
  n_estimators: 100
  max_depth: 5
```

### Selection Method Comparison

| Method | Speed | Reliability | Correlated Features | Best For |
|--------|-------|-------------|---------------------|----------|
| **MDI** | Fast (30s) | Low | Poor | Quick baseline |
| **MDA** | Medium (5min) | High | Good | Default choice |
| **Hybrid** | Medium (5.5min) | Medium-High | Good | Balanced |
| **Clustered MDA** | Slow (10min) | Very High | Excellent | MTF datasets |

**Benchmark:** 5 folds, 150 features, 100k samples

### Recommended Presets

| Use Case | Preset | Method | Features | Stability |
|----------|--------|--------|----------|-----------|
| Fast prototyping | `fast` | MDI | 30 | 0.5 |
| Standard training | `default` | MDA | 50 | 0.6 |
| MTF datasets | `mtf` | Clustered MDA | 60 | 0.6 |
| Production models | `robust` | Clustered MDA | 50 | 0.7 |
| Limited data | `stable` | MDA | 40 | 0.8 |

---

## MTF Feature Quick Start

### Default MTF Strategy

**Python:**
```python
from src.phase1.stages.features import FeatureEngineer

engineer = FeatureEngineer(
    input_dir="data/clean",
    output_dir="data/features",
    timeframe="5min",
    enable_mtf=True,
    mtf_timeframes=["15min", "60min"],
    mtf_include_ohlcv=True,
    mtf_include_indicators=True
)

df_features, report = engineer.engineer_features(df, symbol="MES")
print(f"MTF features: {report['mtf_feature_count']}")
```

**YAML Config:**
```yaml
# config/features/mtf_strategies.yaml
strategies:
  default:
    base_timeframe: "5min"
    mtf_timeframes: ["15min", "60min"]
    mtf_include_ohlcv: true
    mtf_include_indicators: true
```

### MTF Strategy Comparison

| Strategy | Timeframes | MTF Features | Total Features | Speed | Data Needed |
|----------|------------|--------------|----------------|-------|-------------|
| **disabled** | None | 0 | ~175 | Fastest | 5k bars (~1 week) |
| **minimal** | [15min] | ~25 | ~200 | Fast | 10k bars (~2 weeks) |
| **default** | [15min, 60min] | ~45 | ~220 | Medium | 50k bars (~4 months) |
| **aggressive** | [15min, 30min, 60min, 4h, daily] | ~120 | ~295 | Slow | 100k bars (~1 year) |

**Note:** All MTF strategies require **Clustered MDA** feature selection to handle correlation.

---

## Model-Specific Recommendations

### Boosting Models (XGBoost, LightGBM, CatBoost)

**Features:** 50-70
**MTF:** Recommended (default strategy)
**Selection:** MDA or Hybrid
**Config:**
```yaml
# Complete pipeline config
base_features: 175
mtf_strategy: default  # +40-50 features
total_before_selection: ~220
feature_selection:
  method: mda
  n_features: 60
final_features: 60
```

### Neural Networks (LSTM, GRU, TCN, Transformer)

**Features:** 30-50
**Sequence Length:** 60
**MTF:** Optional (minimal strategy)
**Selection:** MDA (optional)
**Config:**
```yaml
base_features: 175
mtf_strategy: minimal  # +25 features
total_before_selection: ~200
feature_selection:
  method: mda
  n_features: 40
final_features: 40
sequence_length: 60
```

### Classical ML (Random Forest, Logistic, SVM)

**Features:** 20-40
**MTF:** Not recommended
**Selection:** MDA (strict stability=0.8)
**Config:**
```yaml
base_features: 175
mtf_strategy: disabled
total_before_selection: 175
feature_selection:
  method: mda
  n_features: 30
  min_feature_frequency: 0.8
final_features: 30
```

### Ensemble Models (Voting, Stacking, Blending)

**Features:** 50-80
**MTF:** Recommended (default or aggressive)
**Selection:** Clustered MDA (required)
**Config:**
```yaml
base_features: 175
mtf_strategy: default  # +45 features
total_before_selection: ~220
feature_selection:
  method: clustered_mda
  n_features: 70
  max_clusters: 30
final_features: 70
```

---

## Feature Engineering Pipeline Flow

```
Raw OHLCV Data (5min resampled)
    ↓
[ Base Feature Engineering ] (~175 features)
    - Returns & Price Ratios (20)
    - Moving Averages (25)
    - Momentum (23)
    - Volatility (34)
    - Volume (13)
    - Trend (6)
    - Temporal (9)
    - Regime (2)
    - Microstructure (19)
    - Wavelets (24, if enabled)
    ↓
[ Multi-Timeframe Features ] (~40-50 features, if enabled)
    - Resample to [15min, 60min]
    - MTF OHLCV (10)
    - MTF Indicators (30-40)
    ↓
[ NaN Cleaning ]
    - Drop columns with >90% NaN
    - Drop rows with any NaN
    ↓
Total Features: ~175 (base) + ~45 (MTF) = ~220
    ↓
[ Walk-Forward Feature Selection ]
    - Method: MDA or Clustered MDA
    - Select top 50-70 features
    - Require ≥60% fold stability
    ↓
Final Selected Features: 50-70
    ↓
[ Model Training ]
```

---

## Key Parameters Reference

### Feature Engineering

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `timeframe` | `5min` | `1min`, `5min`, `15min`, `60min` | Base data timeframe |
| `enable_mtf` | `True` | `True/False` | Enable MTF features |
| `mtf_timeframes` | `[15min, 60min]` | Any > base TF | Higher timeframes |
| `enable_wavelets` | `True` | `True/False` | Enable wavelet features |
| `wavelet_type` | `db4` | `db4`, `db8`, `sym5`, `haar` | Wavelet family |
| `wavelet_level` | `3` | `2-5` | Decomposition levels |
| `nan_threshold` | `0.9` | `0.0-1.0` | Column NaN drop threshold |

### Feature Selection

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `method` | `mda` | `mdi`, `mda`, `hybrid` | Importance method |
| `n_features_to_select` | `50` | `20-100` | Features per fold |
| `min_feature_frequency` | `0.6` | `0.4-1.0` | Stability threshold |
| `use_clustered_importance` | `False` | `True/False` | Handle correlation |
| `max_clusters` | `20` | `10-50` | Feature clusters |
| `n_estimators` | `100` | `50-200` | RF trees |
| `n_repeats` | `10` | `5-20` | MDA permutations |

### Period Scaling

**All indicator periods automatically scale with timeframe to maintain lookback duration.**

**Example: RSI-14**
| Timeframe | Scaled Period | Lookback |
|-----------|---------------|----------|
| 1min | 70 | 70 min |
| 5min (base) | 14 | 70 min |
| 15min | 5 | 75 min |
| 60min | 2 | 120 min (min=2 enforced) |

**Formula:**
```python
scaled_period = round((period × source_tf_minutes) / target_tf_minutes)
scaled_period = max(2, scaled_period)  # Enforce minimum
```

---

## Anti-Lookahead Guarantees

**All features are shifted by 1 bar to prevent lookahead bias:**

```python
# WRONG (lookahead):
df['sma_20'] = df['close'].rolling(20).mean()

# CORRECT (no lookahead):
df['sma_20'] = df['close'].rolling(20).mean().shift(1)
```

**Result:** Feature at bar `t` uses data only up to bar `t-1`.

**Exception:** Temporal features (hour, minute, session) have no lag since they describe the current bar's timestamp.

**Feature Selection:** Walk-forward methodology ensures features are selected using only training data (no future information).

---

## Common Workflows

### Workflow 1: Standard Training Pipeline

```python
# Step 1: Generate features
engineer = FeatureEngineer(
    timeframe="5min",
    enable_mtf=True,
    mtf_timeframes=["15min", "60min"]
)
df_features, _ = engineer.engineer_features(df, "MES")
# Output: ~220 features

# Step 2: Feature selection
selector = WalkForwardFeatureSelector(
    n_features_to_select=60,
    selection_method="mda",
    use_clustered_importance=True,
    max_clusters=30
)
result = selector.select_features_walkforward(X, y, cv_splits)
# Output: ~60 stable features

# Step 3: Train model
X_selected = X[result.stable_features]
model.fit(X_selected, y)
```

### Workflow 2: Fast Prototyping (No MTF, MDI Selection)

```python
# Step 1: Features (no MTF)
engineer = FeatureEngineer(timeframe="5min", enable_mtf=False)
df_features, _ = engineer.engineer_features(df, "MES")
# Output: ~175 features

# Step 2: Fast selection (MDI)
selector = WalkForwardFeatureSelector(
    n_features_to_select=30,
    selection_method="mdi",  # Fast
    min_feature_frequency=0.5
)
result = selector.select_features_walkforward(X, y, cv_splits)
# Output: ~30 features

# Step 3: Train
model.fit(X[result.stable_features], y)
```

### Workflow 3: Production Ensemble (Aggressive MTF, Clustered MDA)

```python
# Step 1: Maximum features
engineer = FeatureEngineer(
    timeframe="5min",
    enable_mtf=True,
    mtf_timeframes=["15min", "30min", "60min", "4h", "daily"]
)
df_features, _ = engineer.engineer_features(df, "MES")
# Output: ~295 features

# Step 2: Robust selection
selector = WalkForwardFeatureSelector(
    n_features_to_select=80,
    selection_method="mda",
    use_clustered_importance=True,
    max_clusters=50,
    min_feature_frequency=0.7
)
result = selector.select_features_walkforward(X, y, cv_splits)
# Output: ~70-80 stable features

# Step 3: Train ensemble
ensemble = VotingEnsemble(base_models=[xgb, lgb, cat])
ensemble.fit(X[result.stable_features], y)
```

---

## Performance Benchmarks

### Feature Generation Time (100k bars, 5min base)

| Configuration | Time | Memory |
|---------------|------|--------|
| Base only (no MTF) | 5-8 sec | 140 MB |
| MTF minimal ([15min]) | 8-10 sec | 160 MB |
| MTF default ([15min, 60min]) | 15-20 sec | 180 MB |
| MTF aggressive (5 TFs) | 30-40 sec | 250 MB |

### Feature Selection Time (5 folds, CV)

| Method | Features | Time | Memory |
|--------|----------|------|--------|
| MDI | 150 | 30 sec | 500 MB |
| MDA | 150 | 5 min | 1 GB |
| Hybrid | 150 | 5.5 min | 1 GB |
| Clustered MDA | 220 (MTF) | 12 min | 1.5 GB |

**Hardware:** Intel i7, 16GB RAM, 100k samples

---

## Troubleshooting

### Issue: Too Few Features Selected

**Symptoms:** `len(stable_features) << n_features_to_select`

**Causes:**
- `min_feature_frequency` too high
- Noisy labels (features unstable across folds)
- Insufficient training data

**Solutions:**
- Lower `min_feature_frequency` (e.g., 0.5 instead of 0.6)
- Increase training data
- Review label quality

### Issue: MTF Features Dominating Selection

**Symptoms:** Most selected features are MTF (`_15m`, `_1h` suffixes)

**Causes:**
- Not using Clustered MDA (correlation inflates MTF importance)

**Solutions:**
- Enable Clustered MDA: `use_clustered_importance=True`
- Increase `max_clusters` (e.g., 30-50)

### Issue: Slow Feature Generation

**Symptoms:** MTF generation takes >1 minute

**Causes:**
- Too many MTF timeframes
- Wavelets enabled on large dataset

**Solutions:**
- Use selective MTF timeframes ([15min, 60min] instead of 5 TFs)
- Disable wavelets if not needed: `enable_wavelets=False`
- Disable MTF indicators: `mtf_include_indicators=False`

### Issue: High Memory Usage

**Symptoms:** OOM errors during feature engineering

**Causes:**
- Too many features
- Large dataset
- `dtype=float64` (default)

**Solutions:**
- Use `dtype=float32` (50% memory reduction)
- Process symbols sequentially (not all in memory)
- Disable MTF for memory-constrained environments

---

## See Also

- [Feature Engineering Implementation Guide](/docs/guides/FEATURE_ENGINEERING_GUIDE.md)
- [Cross-Validation Guide](/docs/guides/CROSS_VALIDATION_GUIDE.md)
- [Model Training Guide](/docs/guides/MODEL_TRAINING_GUIDE.md)
- [Performance Tuning](/docs/guides/PERFORMANCE_TUNING.md)

---

## References

**Academic Papers:**
- Breiman, L. (2001). "Random Forests"
- Lopez de Prado, M. (2018). "Advances in Financial Machine Learning" (Chapter 8: Feature Importance)
- Amihud, Y. (2002). "Illiquidity and Stock Returns"
- Corwin & Schultz (2012). "A Simple Way to Estimate Bid-Ask Spreads"
- Yang & Zhang (2000). "Drift-Independent Volatility Estimation"

**Implementation:**
- Feature Engineering: `src/phase1/stages/features/`
- Feature Selection: `src/cross_validation/feature_selector.py`
- MTF Generation: `src/phase1/stages/mtf/`
- Model Registry: `src/models/registry.py`

---

**Last Updated:** 2025-12-30
**ML Model Factory Version:** Phase 1 Complete, Phase 2 Complete, Phase 3 Complete, Phase 4 In Progress
