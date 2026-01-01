# Multi-Timeframe (MTF) Strategy Guide

**Audience:** ML engineers choosing data preparation strategies for OHLCV time series models
**Last Updated:** 2026-01-01
**Status:** Strategy 2 partially implemented (5/9 timeframes), Strategies 1 & 3 not implemented

---

## Table of Contents

1. [Overview](#overview)
2. [Strategy Comparison](#strategy-comparison)
3. [Strategy 1: Single-Timeframe Training](#strategy-1-single-timeframe-training)
4. [Strategy 2: MTF Indicators](#strategy-2-mtf-indicators)
5. [Strategy 3: MTF Ingestion](#strategy-3-mtf-ingestion)
6. [Model-Strategy Compatibility Matrix](#model-strategy-compatibility-matrix)
7. [Decision Tree: When to Use Which Strategy](#decision-tree-when-to-use-which-strategy)
8. [Configuration Examples](#configuration-examples)
9. [Performance Implications](#performance-implications)
10. [Current Limitations](#current-limitations)
11. [Migration Path](#migration-path)

---

## Overview

### What are MTF Strategies?

**Multi-Timeframe (MTF) strategies** determine how models consume temporal data at different resolutions. Instead of training on a single fixed timeframe, MTF strategies enable models to learn from patterns across multiple time scales simultaneously.

**Core Concept:**
```
Raw 1-minute OHLCV bars
         ↓
Resample to multiple timeframes (5m, 15m, 30m, 1h, 4h, daily)
         ↓
Choose strategy based on model architecture:
├── Strategy 1: Use one timeframe only (baseline)
├── Strategy 2: Compute indicators across timeframes (tabular models)
└── Strategy 3: Feed raw multi-resolution bars (sequence models)
```

### Why MTF Matters

Different timeframes capture different market dynamics:
- **Short timeframes** (1min, 5min): Microstructure, immediate momentum
- **Medium timeframes** (15min, 30min, 1h): Intraday trends, session patterns
- **Long timeframes** (4h, daily): Swing trends, multi-day patterns

**Problem:** Choosing the "right" timeframe is hard. MTF strategies let models learn from ALL timeframes.

---

## Strategy Comparison

| Aspect | Strategy 1: Single-TF | Strategy 2: MTF Indicators | Strategy 3: MTF Ingestion |
|--------|----------------------|---------------------------|--------------------------|
| **Data Input** | One timeframe only | Indicators from 9 timeframes | Raw OHLCV from 9 timeframes |
| **Feature Count** | ~40-50 per TF | ~180 (base + MTF indicators) | Variable (multi-resolution tensors) |
| **Input Shape** | 2D or 3D (depends on model) | 2D: `(n, 180)` for tabular | 4D: `(n, n_tfs, seq_len, 4)` |
| **Model Families** | All models | Tabular: Boosting + Classical | Sequence: Neural + CNN + Advanced |
| **Strengths** | Simple, fast, interpretable | Dense cross-TF features | Multi-resolution temporal learning |
| **Weaknesses** | Misses cross-scale patterns | Pre-computed features lose raw structure | Complex, slower, harder to debug |
| **Use Case** | Baselines, ablation studies | Tabular models (XGBoost, RF) | Advanced sequence models (TFT, PatchTST) |
| **Implementation Status** | ❌ Not implemented | ⚠️ Partial (5/9 TFs) | ❌ Not implemented |
| **Effort to Implement** | 1 week | 2 weeks (9-TF completion) | 3-4 weeks |

---

## Strategy 1: Single-Timeframe Training

### Overview

Train models on a **single fixed timeframe** without any multi-timeframe features.

**Data Flow:**
```
1min raw OHLCV
      ↓
Resample to training_timeframe (e.g., 15min)
      ↓
Compute ~40-50 base features on 15min bars
      ↓
Train model (NO MTF features)
```

### When to Use

✅ **Use Strategy 1 when:**
- Establishing baselines for comparison
- Running ablation studies (does MTF help?)
- Training simple interpretable models
- Testing new model architectures before adding MTF complexity
- Working with high-frequency data where resampling loses information

❌ **Avoid Strategy 1 when:**
- You need cross-scale pattern recognition
- Market exhibits multi-timeframe regime shifts
- Competing models use MTF and you want fair comparison

### Features Generated

**Base features (~40-50):**
- Returns: `return_1`, `log_return_1`, `return_5`
- Momentum: `rsi_14`, `macd`, `stoch_k`, `williams_r`
- Volatility: `atr_14`, `bb_position`, `hvol_20`
- Volume: `volume_ratio`, `obv`, `vwap`
- Temporal: `hour_sin`, `hour_cos`, `day_of_week`
- Wavelets: `wavelet_cA3`, `wavelet_cD1` (optional)
- Microstructure: `micro_spread`, `micro_liquidity` (optional)

**No MTF features** - all features computed on the single training timeframe.

### Configuration

**Status:** ❌ Not implemented

**Future config:**
```python
# config/mtf_strategy_1.yaml
training_timeframe: '15min'  # Train on 15-min bars
mtf_strategy: 'single_tf'    # No MTF features
enable_wavelets: true
enable_microstructure: false
```

**Future CLI:**
```bash
python scripts/train_model.py \
    --model xgboost \
    --training-timeframe 15min \
    --mtf-strategy single_tf \
    --horizon 20
```

### Performance Implications

**Training Speed:** ⚡ Fastest (fewest features)
**Inference Speed:** ⚡ Fastest
**Memory Usage:** ⚡ Lowest
**Expected Performance:** Baseline (worse than MTF strategies on most datasets)

**Typical Results (expected):**
- Sharpe ratio: 0.8-1.2 (lower than MTF)
- Win rate: 52-55%
- Overfitting risk: Low (fewer features)

---

## Strategy 2: MTF Indicators

### Overview

Compute **indicators on multiple timeframes**, then flatten into a single feature vector for tabular models.

**Data Flow:**
```
1min raw OHLCV
      ↓
Resample to 9 timeframes (5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)
      ↓
Compute ~150 base indicators on training_timeframe (e.g., 15min)
      ↓
Compute ~30 MTF indicators on 8 higher timeframes
      ↓
Flatten to single vector: ~180 features
      ↓
Train tabular model (XGBoost, LightGBM, RF, etc.)
```

### When to Use

✅ **Use Strategy 2 when:**
- Training **tabular models** (XGBoost, LightGBM, CatBoost, Random Forest)
- You want cross-timeframe feature engineering
- Interpretability matters (can analyze feature importance)
- Fast training/inference is critical

❌ **Avoid Strategy 2 when:**
- Training **sequence models** (LSTM, Transformer) - use Strategy 3 instead
- Feature dimensionality is already too high
- You suspect indicator redundancy across timeframes

### Features Generated

**Base features (~150) on training_timeframe:**
- All features from Strategy 1
- Extended momentum indicators (15+ indicators)
- Wavelet decompositions (24 features)
- Microstructure proxies (10 features)
- Regime indicators (volatility, trend, composite)

**MTF features (~30) from higher timeframes:**

| Timeframe | Features | Example Columns |
|-----------|----------|-----------------|
| 15min | 6 features | `rsi_14_15m`, `sma_20_15m`, `atr_14_15m` |
| 30min | 6 features | `rsi_14_30m`, `sma_20_30m`, `close_sma20_ratio_30m` |
| 1h | 6 features | `rsi_14_1h`, `ema_50_1h`, `bb_position_1h` |
| 4h | 6 features | `rsi_14_4h`, `macd_hist_4h`, `atr_14_4h` |
| daily | 6 features | `rsi_14_1d`, `sma_200_1d`, `close_sma200_ratio_1d` |

**Total:** ~180 features (150 base + 30 MTF)

### Anti-Lookahead Protection

**CRITICAL:** MTF features use `.shift(1)` before forward-filling to prevent lookahead bias.

```python
# Example: Computing 1h RSI at 15min resolution
df_1h = df_15min.resample('1h').agg({'close': 'last', ...})
df_1h['rsi_14_1h'] = compute_rsi(df_1h['close'], period=14)

# CRITICAL: Shift by 1 before merging back to 15min
df_1h['rsi_14_1h_shifted'] = df_1h['rsi_14_1h'].shift(1)

# Forward-fill to 15min resolution
df_15min = df_15min.merge(df_1h[['rsi_14_1h_shifted']], ...)
df_15min['rsi_14_1h'] = df_15min['rsi_14_1h_shifted'].ffill()
```

**Why shift?** At time `t`, the 1h bar ending at `t` uses OHLCV up to `t`, which is not known until the bar closes. Shifting ensures we only use the **previous** completed 1h bar.

### Configuration

**Status:** ⚠️ Partially implemented (5 of 9 timeframes: 15min, 30min, 1h, 4h, daily)

**Current config (partial):**
```python
# src/phase1/config/features.py
MTF_TIMEFRAMES = ['15min', '30min', '1h', '4h', 'daily']  # Only 5 TFs
MTF_MODE = 'both'  # Generates both OHLCV bars + indicators
```

**Future config (9-TF complete):**
```python
# config/mtf_strategy_2.yaml
training_timeframe: '15min'
mtf_strategy: 'mtf_indicators'
mtf_source_timeframes: ['5min', '10min', '15min', '20min', '25min', '30min', '45min', '1h']
enable_feature_selection: true
n_features_to_select: 50  # Optional: reduce via MDA/MDI
```

**Future CLI:**
```bash
python scripts/train_model.py \
    --model xgboost \
    --training-timeframe 15min \
    --mtf-strategy mtf_indicators \
    --mtf-source-timeframes 5min 10min 15min 20min 25min 30min 45min 1h \
    --horizon 20
```

### Performance Implications

**Training Speed:** 🟡 Medium (more features than Strategy 1)
**Inference Speed:** 🟡 Medium
**Memory Usage:** 🟡 Medium (~180 features)
**Expected Performance:** Better than Strategy 1 for tabular models

**Typical Results (expected):**
- Sharpe ratio: 1.2-1.6 (10-30% improvement over Strategy 1)
- Win rate: 54-58%
- Overfitting risk: Medium (feature correlation across timeframes)

**Mitigation:** Use walk-forward feature selection (MDA/MDI) to reduce to 50-80 most important features.

---

## Strategy 3: MTF Ingestion

### Overview

Feed **raw OHLCV bars** from multiple timeframes as **separate tensors** to sequence models, enabling multi-resolution temporal learning.

**Data Flow:**
```
1min raw OHLCV
      ↓
Resample to 9 timeframes
      ↓
For each sample at time t, extract synchronized windows:
├── 5min:  last 60 bars → (60, 4) OHLC tensor
├── 15min: last 20 bars → (20, 4) OHLC tensor
├── 30min: last 10 bars → (10, 4) OHLC tensor
└── 1h:    last 5 bars  → (5, 4) OHLC tensor
      ↓
Feed to sequence model (LSTM, Transformer, TFT, PatchTST)
      ↓
Model learns cross-scale patterns via attention/convolution
```

### When to Use

✅ **Use Strategy 3 when:**
- Training **sequence models** (LSTM, GRU, TCN, Transformer)
- Training **advanced models** (PatchTST, iTransformer, TFT, N-BEATS)
- Market exhibits multi-scale regime shifts
- You need state-of-the-art forecasting performance
- Model architecture supports multi-resolution inputs

❌ **Avoid Strategy 3 when:**
- Training **tabular models** (they can't consume multi-resolution tensors)
- Interpretability is critical (raw bars are less interpretable than indicators)
- You need fast inference (multi-resolution is slower)

### Multi-Resolution Tensor Structure

**Example:** Training on 15min with MTF from [5min, 15min, 30min, 1h]

For each sample at time `t`:
```python
{
    '5min':  np.ndarray(shape=(60, 4)),  # Last 60 5min bars (5 hours)
    '15min': np.ndarray(shape=(20, 4)),  # Last 20 15min bars (5 hours)
    '30min': np.ndarray(shape=(10, 4)),  # Last 10 30min bars (5 hours)
    '1h':    np.ndarray(shape=(5, 4)),   # Last 5 1h bars (5 hours)
}
# All windows cover the same lookback period (5 hours)
# Shape: (seq_len, 4) where 4 = [open, high, low, close]
```

**Concatenation Strategy (for LSTM, TCN):**
```python
# Concatenate along sequence dimension
X_concat = np.concatenate([X_5min, X_15min, X_30min, X_1h], axis=1)
# Shape: (n_samples, 95, 4) where 95 = 60+20+10+5
```

**Stacking Strategy (for Transformers, TFT):**
```python
# Stack into 4D tensor
X_stacked = np.stack([X_5min_padded, X_15min_padded, X_30min_padded, X_1h_padded], axis=1)
# Shape: (n_samples, 4, max_seq_len, 4)
# Models use attention to weight different resolutions
```

### Model-Specific Ingestion

| Model | Input Strategy | Shape | Notes |
|-------|---------------|-------|-------|
| LSTM, GRU | Concatenate | `(n, 95, 4)` | Sequential processing across all bars |
| TCN | Concatenate | `(n, 95, 4)` | Dilated convolutions across concatenated sequence |
| Transformer | Stack + Flatten | `(n, 380)` | Flattened multi-resolution as "long" sequence |
| TFT (Temporal Fusion) | Stack | `(n, 4, seq, 4)` | Attention over timeframes + temporal fusion |
| PatchTST | Patched Stack | `(n, 4, patches, patch_len)` | Patch-based multi-resolution |
| iTransformer | Stack | `(n, 4, seq, 4)` | Inverted attention (across variables first) |

### Configuration

**Status:** ❌ Not implemented

**Future config:**
```python
# config/mtf_strategy_3.yaml
training_timeframe: '15min'
mtf_strategy: 'mtf_ingestion'
mtf_input_timeframes: ['5min', '15min', '30min', '1h']
lookback_minutes: 300  # 5 hours of history
ingestion_mode: 'concatenate'  # or 'stack' for Transformers
```

**Future CLI:**
```bash
python scripts/train_model.py \
    --model lstm \
    --training-timeframe 15min \
    --mtf-strategy mtf_ingestion \
    --mtf-input-timeframes 5min 15min 30min 1h \
    --ingestion-mode concatenate \
    --horizon 20
```

### Performance Implications

**Training Speed:** 🔴 Slow (larger tensors, more complex models)
**Inference Speed:** 🔴 Slower
**Memory Usage:** 🔴 High (multi-resolution tensors)
**Expected Performance:** Best for sequence models

**Typical Results (expected):**
- Sharpe ratio: 1.4-1.8+ (20-40% improvement over Strategy 2 for sequence models)
- Win rate: 56-60%
- Overfitting risk: High (complex multi-scale patterns, need strong regularization)

**Best Practices:**
- Use dropout (0.2-0.4) in sequence models
- Use gradient clipping (max_norm=1.0)
- Monitor validation loss carefully (early stopping)
- Start with 2-3 timeframes, add more if validation improves

---

## Model-Strategy Compatibility Matrix

| Model | Strategy 1 | Strategy 2 | Strategy 3 | Recommended |
|-------|-----------|-----------|-----------|-------------|
| **Boosting Models** | | | | |
| XGBoost | ✅ Baseline | ✅ **Best** | ❌ Can't use | **Strategy 2** |
| LightGBM | ✅ Baseline | ✅ **Best** | ❌ Can't use | **Strategy 2** |
| CatBoost | ✅ Baseline | ✅ **Best** | ❌ Can't use | **Strategy 2** |
| **Classical Models** | | | | |
| Random Forest | ✅ Baseline | ✅ **Best** | ❌ Can't use | **Strategy 2** |
| Logistic Regression | ✅ Baseline | ✅ Good | ❌ Can't use | **Strategy 2** |
| SVM | ✅ Baseline | ⚠️ May overfit | ❌ Can't use | **Strategy 1** |
| **Neural Models** | | | | |
| LSTM | ✅ Baseline | ⚠️ Suboptimal | ✅ **Best** | **Strategy 3** |
| GRU | ✅ Baseline | ⚠️ Suboptimal | ✅ **Best** | **Strategy 3** |
| TCN | ✅ Baseline | ⚠️ Suboptimal | ✅ **Best** | **Strategy 3** |
| Transformer | ✅ Baseline | ⚠️ Suboptimal | ✅ **Best** | **Strategy 3** |
| **CNN Models (Planned)** | | | | |
| InceptionTime | ✅ Baseline | ❌ Can't use | ✅ **Best** | **Strategy 3** |
| 1D ResNet | ✅ Baseline | ❌ Can't use | ✅ **Best** | **Strategy 3** |
| **Advanced Models (Planned)** | | | | |
| PatchTST | ✅ Baseline | ❌ Can't use | ✅ **Best** | **Strategy 3** |
| iTransformer | ✅ Baseline | ❌ Can't use | ✅ **Best** | **Strategy 3** |
| TFT | ✅ Baseline | ⚠️ Suboptimal | ✅ **Best** | **Strategy 3** |
| N-BEATS | ✅ Baseline | ❌ Can't use | ✅ **Best** | **Strategy 3** |

**Legend:**
- ✅ **Best** - Optimal for this model architecture
- ✅ Good - Works well, expected to improve performance
- ⚠️ Suboptimal - Works but misses architectural strengths
- ❌ Can't use - Incompatible with model input requirements

---

## Decision Tree: When to Use Which Strategy

```
START: Choose MTF Strategy
         |
         ├── What's your model family?
         │   ├── Tabular (XGBoost, LightGBM, RF, etc.)
         │   │   └── Use Strategy 2: MTF Indicators
         │   │       └── Why? Tabular models excel at dense cross-TF features
         │   │
         │   ├── Sequence (LSTM, Transformer, TCN, etc.)
         │   │   └── Is Strategy 3 implemented yet?
         │   │       ├── Yes → Use Strategy 3: MTF Ingestion
         │   │       │   └── Why? Multi-resolution learning is SOTA
         │   │       │
         │   │       └── No → Use Strategy 2 (temporary fallback)
         │   │           └── Warning: Suboptimal, upgrade when Strategy 3 ready
         │   │
         │   └── Unknown/New model
         │       └── Start with Strategy 1: Single-TF (baseline)
         │           └── Run ablation: Strategy 1 vs 2 vs 3
         │
         ├── What's your goal?
         │   ├── Baseline comparison → Strategy 1
         │   ├── Production model → Strategy 2 (tabular) or 3 (sequence)
         │   └── Research/experimentation → Try all 3, compare
         │
         └── What are your constraints?
             ├── Need fast inference → Strategy 1 or 2
             ├── Limited memory → Strategy 1
             ├── Need interpretability → Strategy 2 (feature importance)
             └── No constraints → Strategy 2 (tabular) or 3 (sequence)
```

**Quick Reference:**

| Your Situation | Recommended Strategy |
|----------------|---------------------|
| Training XGBoost/LightGBM | **Strategy 2** |
| Training LSTM/Transformer | **Strategy 3** (when available) |
| Need baseline for comparison | **Strategy 1** |
| Optimizing for speed | **Strategy 1** |
| Optimizing for accuracy | **Strategy 2 or 3** |
| Working with high-frequency data | **Strategy 1** (less resampling) |
| Limited training data | **Strategy 1** (fewer features = less overfitting) |

---

## Configuration Examples

### Example 1: XGBoost with Strategy 2 (MTF Indicators)

**Config file: `config/xgboost_mtf.yaml`**
```yaml
# Data pipeline
training_timeframe: '15min'
mtf_strategy: 'mtf_indicators'
mtf_source_timeframes: ['5min', '10min', '15min', '20min', '25min', '30min', '45min', '1h']

# Feature engineering
enable_wavelets: true
enable_microstructure: true
enable_feature_selection: true
n_features_to_select: 60

# Model
model: 'xgboost'
horizon: 20

# Training
n_estimators: 500
max_depth: 6
learning_rate: 0.05
subsample: 0.8
colsample_bytree: 0.8
```

**CLI:**
```bash
python scripts/train_model.py --config config/xgboost_mtf.yaml
```

### Example 2: LSTM with Strategy 3 (MTF Ingestion) - Future

**Config file: `config/lstm_mtf_ingestion.yaml`**
```yaml
# Data pipeline
training_timeframe: '15min'
mtf_strategy: 'mtf_ingestion'
mtf_input_timeframes: ['5min', '15min', '30min', '1h']
lookback_minutes: 300  # 5 hours
ingestion_mode: 'concatenate'

# Model
model: 'lstm'
horizon: 20
seq_len: 60  # Unused with multi-resolution (calculated from lookback)

# Architecture
hidden_size: 128
num_layers: 2
dropout: 0.3
bidirectional: false

# Training
learning_rate: 0.001
batch_size: 64
max_epochs: 100
early_stopping_patience: 10
gradient_clip_norm: 1.0
```

**CLI:**
```bash
python scripts/train_model.py --config config/lstm_mtf_ingestion.yaml
```

### Example 3: Random Forest with Strategy 1 (Single-TF Baseline)

**Config file: `config/rf_baseline.yaml`**
```yaml
# Data pipeline
training_timeframe: '30min'
mtf_strategy: 'single_tf'  # No MTF features
enable_wavelets: false
enable_microstructure: false

# Model
model: 'random_forest'
horizon: 20

# Hyperparameters
n_estimators: 300
max_depth: 10
min_samples_split: 20
min_samples_leaf: 10
max_features: 'sqrt'
```

**CLI:**
```bash
python scripts/train_model.py --config config/rf_baseline.yaml
```

---

## Performance Implications

### Computational Complexity

| Strategy | Feature Computation | Training Complexity | Inference Latency | Memory Usage |
|----------|-------------------|-------------------|------------------|--------------|
| Strategy 1 | O(n × f) | O(n × f × log(n)) | ~1ms | Low |
| Strategy 2 | O(n × f × k) | O(n × (f×k) × log(n)) | ~2-3ms | Medium |
| Strategy 3 | O(n × k) | O(n × (k×s) × h²) | ~5-10ms | High |

**Legend:**
- `n` = number of samples
- `f` = features per timeframe (~40)
- `k` = number of timeframes (9)
- `s` = sequence length per timeframe
- `h` = hidden size (for sequence models)

### Expected Performance Gains

**Tabular Models (XGBoost, LightGBM, CatBoost):**

| Metric | Strategy 1 (Baseline) | Strategy 2 (MTF Indicators) | Improvement |
|--------|----------------------|---------------------------|-------------|
| Sharpe Ratio | 1.0-1.2 | 1.2-1.6 | +10-30% |
| Win Rate | 52-55% | 54-58% | +2-3% |
| Max Drawdown | -15% to -20% | -12% to -18% | -2-3% |
| Training Time | 1x (baseline) | 1.5-2x | N/A |

**Sequence Models (LSTM, Transformer):**

| Metric | Strategy 1 (Baseline) | Strategy 2 (MTF Indicators) | Strategy 3 (MTF Ingestion) |
|--------|----------------------|---------------------------|---------------------------|
| Sharpe Ratio | 0.9-1.1 | 1.1-1.4 | 1.4-1.8+ |
| Win Rate | 51-54% | 53-56% | 56-60% |
| Max Drawdown | -18% to -22% | -15% to -20% | -12% to -16% |
| Training Time | 1x (baseline) | 1.5x | 3-5x |

**Key Insights:**
1. Strategy 2 improves **tabular models** significantly
2. Strategy 2 provides **marginal gains** for sequence models (suboptimal)
3. Strategy 3 unlocks **full potential** of sequence models (20-40% improvement)
4. Trade-off: Better performance → Longer training time

### Overfitting Risks

| Strategy | Overfitting Risk | Mitigation |
|----------|-----------------|------------|
| Strategy 1 | **Low** (fewest features) | None needed |
| Strategy 2 | **Medium** (correlated MTF features) | Feature selection (MDA/MDI), reduce to 50-80 features |
| Strategy 3 | **High** (complex multi-scale patterns) | Dropout (0.3-0.4), early stopping, gradient clipping |

---

## Current Limitations

### What's Implemented (as of 2026-01-01)

✅ **Partial Strategy 2 (MTF Indicators):**
- 5 of 9 timeframes: 15min, 30min, 1h, 4h, daily
- Anti-lookahead protection (shift + forward-fill)
- ~30 MTF indicator features
- All models receive same ~180 indicator-derived features

### What's Missing

❌ **Strategy 1 (Single-TF):**
- No config option to disable MTF features
- Cannot train single-timeframe baselines
- Estimated implementation: **1 week**

❌ **Strategy 2 (Complete 9-TF Ladder):**
- Missing timeframes: 1min, 5min, 10min, 20min, 25min, 45min
- Missing: Configurable `mtf_source_timeframes`
- Missing: Model-specific feature selection
- Estimated implementation: **2 weeks**

❌ **Strategy 3 (MTF Ingestion):**
- No multi-resolution tensor builder
- No `TimeSeriesDataContainer.get_multi_resolution_bars()`
- No concatenate/stack utilities
- Sequence models receive indicators instead of raw bars
- Estimated implementation: **3-4 weeks**

### Current Behavior (Temporary Universal Pipeline)

**ALL models currently receive:**
- ~180 indicator-derived features
- MTF from 5 timeframes (15min, 30min, 1h, 4h, daily)
- Served in model-appropriate shapes:
  - Tabular models: 2D `(n_samples, 180)`
  - Sequence models: 3D `(n_samples, 60, 180)` - **SUBOPTIMAL**

**Problem:** Sequence models receive pre-computed indicators when they should receive raw multi-resolution OHLCV bars.

**Impact:**
- Tabular models: ✅ Receive optimal data (indicators)
- Sequence models: ⚠️ Handicapped (should use Strategy 3)

---

## Migration Path

### Phase 1: Implement Strategy 1 (1 week)

**Goal:** Enable single-timeframe training for baselines

**Tasks:**
1. Add `mtf_strategy='single_tf'` config option
2. Skip MTF feature generation when `single_tf` selected
3. Update CLI to accept `--mtf-strategy` flag
4. Add validation tests

**Output:** All models can train on single timeframe (~40-50 features)

### Phase 2: Complete Strategy 2 (2 weeks)

**Goal:** Full 9-timeframe MTF indicator ladder

**Tasks:**
1. Add 20min, 25min to `MTF_TIMEFRAMES` constants
2. Implement `mtf_source_timeframes` config parameter
3. Add model-specific feature selection hooks
4. Update to 9-TF ladder: 5min, 10min, 15min, 20min, 25min, 30min, 45min, 1h
5. Add feature importance tracking per model
6. Add walk-forward feature selection integration

**Output:** Tabular models get optimal MTF indicator features from 9 timeframes

### Phase 3: Implement Strategy 3 (3-4 weeks)

**Goal:** Multi-resolution tensor ingestion for sequence models

**Tasks:**
1. Create `MultiResolutionDatasetBuilder` class
2. Add `get_multi_resolution_bars()` to `TimeSeriesDataContainer`
3. Implement concatenate/stack tensor utilities
4. Update sequence model trainers to accept multi-resolution inputs
5. Add configuration for `mtf_input_timeframes`
6. Add integration tests for all sequence models
7. Benchmark Strategy 3 vs Strategy 2 for LSTM, Transformer

**Output:** Sequence models receive raw OHLCV bars from 9 timeframes, unlocking multi-resolution learning

### Timeline

```
Week 1: Strategy 1 (Single-TF)
    ↓
Weeks 2-3: Strategy 2 (9-TF Indicators)
    ↓
Weeks 4-7: Strategy 3 (MTF Ingestion)
    ↓
Week 8: Integration, documentation, benchmarking
```

**Total Effort:** 6-8 weeks (1 engineer) | 4-5 weeks (2 engineers)

### Success Criteria

**Phase 1 Complete:**
- [ ] All models can train with `mtf_strategy='single_tf'`
- [ ] Feature count = ~40-50 (no MTF features)
- [ ] Tests pass for all model families

**Phase 2 Complete:**
- [ ] All 9 timeframes in MTF ladder
- [ ] Tabular models get ~180 MTF indicator features (9 TFs)
- [ ] Feature selection reduces to top 50-80 features
- [ ] Tests pass for Strategy 2 with all timeframes

**Phase 3 Complete:**
- [ ] Sequence models receive multi-resolution raw OHLCV bars
- [ ] LSTM/GRU/TCN support concatenated multi-resolution
- [ ] Transformer/TFT/PatchTST support stacked multi-resolution
- [ ] Benchmarks show 20-40% improvement over Strategy 2 for sequence models

---

## References

- **Implementation Roadmap:** `docs/roadmaps/MTF_IMPLEMENTATION_ROADMAP.md`
- **Current Limitations:** `docs/CURRENT_LIMITATIONS.md`
- **Architecture Analysis:** `docs/CURRENT_VS_INTENDED_ARCHITECTURE.md`
- **Refactoring Plan:** `docs/analysis/IMPLEMENTATION_TASKS.md`
- **Phase 1 Details:** `docs/phases/PHASE_1.md`

---

**Last Updated:** 2026-01-01
**Status:** Strategy 2 partially implemented (5/9 timeframes), Strategies 1 & 3 not implemented
**Next Steps:** See `docs/analysis/IMPLEMENTATION_TASKS.md` for week-by-week implementation plan
