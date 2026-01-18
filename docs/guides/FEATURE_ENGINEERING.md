# Feature Engineering Guide

**Purpose:** Comprehensive guide for engineering features for each model family in the ML factory
**Audience:** Data engineers, quant analysts, ML engineers
**Last Updated:** 2026-01-18

---

## Table of Contents

1. [Feature Set Selection](#feature-set-selection) **(Start here!)**
2. [Feature Engineering Philosophy](#feature-engineering-philosophy)
3. [Feature Sets by Model Family](#feature-sets-by-model-family)
4. [MTF Feature Construction](#mtf-feature-construction)
5. [Stage 8: Feature Selection Optimization with Optuna](#stage-8-feature-selection-optimization-with-optuna)
6. [Stage 9: Feature Pruning Optimization with Optuna](#stage-9-feature-pruning-optimization-with-optuna)
7. [Feature Selection Strategies (Manual)](#feature-selection-strategies-manual)
8. [Feature Validation](#feature-validation)
9. [Feature Scaling](#feature-scaling)
10. [Adding New Feature Groups](#adding-new-feature-groups)
11. [Performance Considerations](#performance-considerations)
12. [Code Examples](#code-examples)

---

## Feature Set Selection

> **Authoritative Source:** `src/phase1/config/feature_sets.py`

The pipeline supports multiple pre-defined feature sets optimized for different model families. Select the appropriate feature set based on your model architecture.

### Quick Reference Table

| Feature Set | Use Case | Model Families | Feature Count | MTF |
|-------------|----------|----------------|---------------|-----|
| `core_min` | Minimal baseline | All | ~40 | No |
| `core_full` | All base features | All | ~80 | No |
| `mtf_plus` | Full with MTF | Tabular, Sequential | ~150 | Yes |
| `boosting_optimal` | Gradient boosting | XGBoost, LightGBM, CatBoost | 50-100 | No |
| `neural_optimal` | RNN/CNN sequence models | LSTM, GRU, TCN, InceptionTime, ResNet1D, TFT | 40-60 | No |
| `transformer_raw` | Attention models | Transformer, PatchTST, iTransformer | 10-15 | No |
| `ensemble_base` | Ensemble diversity | Stacking, Blending, Voting | 60-80 | Yes |

### Model-Specific Feature Sets

| Feature Set | Target Model | Key Characteristics |
|-------------|--------------|---------------------|
| `tcn_optimal` | TCN | Longer sequences (120), local pattern features |
| `patchtst_optimal` | PatchTST | Minimal features, long sequences (256) |
| `nbeats_optimal` | N-BEATS | Near-univariate (2-3 features) |
| `inceptiontime_optimal` | InceptionTime | Multi-scale wavelets (55 features) |
| `resnet1d_optimal` | ResNet1D | Wavelets + residual-friendly (48 features) |
| `itransformer_optimal` | iTransformer | Grouped features for cross-attention (25-40) |
| `tft_optimal` | TFT | Rich features for variable selection (60-80) |

### Usage

```python
from src.phase1.config.feature_sets import (
    FEATURE_SET_DEFINITIONS,
    resolve_feature_set_name,
    get_feature_set_columns,
)

# Resolve alias to canonical name
canonical = resolve_feature_set_name("lstm")  # Returns "neural_optimal"

# Get columns matching a feature set
columns = get_feature_set_columns(df.columns.tolist(), "boosting_optimal")
```

### Selection Guidelines

1. **Boosting models (XGBoost, LightGBM, CatBoost):** Use `boosting_optimal` - handles many features well
2. **RNN models (LSTM, GRU):** Use `neural_optimal` - bounded oscillators, normalized ratios
3. **Transformer models (PatchTST, iTransformer):** Use `transformer_raw` or model-specific sets
4. **CNN models (InceptionTime, ResNet1D):** Use model-specific sets with wavelets
5. **N-BEATS:** Use `nbeats_optimal` - minimal features for decomposition
6. **Ensembles:** Use `ensemble_base` for diversity or match base model feature sets

---

## Feature Engineering Philosophy

### Why Different Models Need Different Features

**Key Insight:** More features ≠ better performance. Different model families have different feature needs based on their architecture and learning mechanisms.

| Model Family | Feature Count | Feature Type | Rationale |
|--------------|---------------|--------------|-----------|
| **Boosting** | 150-200 | Dense MTF indicators | Gradient boosting benefits from rich feature sets, handles multicollinearity well |
| **Sequence Models** | 25-30 | Minimal + wavelets | RNNs learn temporal patterns from raw data, too many features cause gradient instability |
| **Transformers** | 5-20 | Multi-resolution raw | Attention mechanisms learn cross-scale patterns, feature engineering reduces inductive bias |
| **Classical** | 150-200 | Dense MTF indicators | Traditional ML needs engineered features, cannot learn complex interactions |
| **Foundation Models** | 4 | Raw OHLC only | Pre-trained on millions of series, feature engineering breaks transfer learning |

### Core Principles

1. **Model-Specific Feature Engineering:** Tabular models need dense features; sequence models need minimal features
2. **MTF Strategy Alignment:** Strategy 2 (MTF indicators) adds cross-timeframe features; Strategy 1/3 do not
3. **Leakage Prevention:** All features are causal (no lookahead), forward-fill for alignment
4. **Quality Over Quantity:** 30 high-quality features > 200 noisy features (for neural models)
5. **Computational Cost:** Feature generation is 20-40% of pipeline runtime - optimize carefully

---

## Feature Sets by Model Family

### Tabular Models (XGBoost, LightGBM, CatBoost, RF, Logistic, SVM)

**Recommended MTF Strategy:** Strategy 2 (MTF indicators) - optional
**Training timeframe:** Configurable (5m/10m/15m/1h) - choose per experiment
**Source timeframes (if MTF enabled):** Varies based on primary TF
**Feature count:** 40-200 features (depending on MTF strategy)

#### Feature Groups

**1. Base Timeframe Features (configurable: 5m/10m/15m/1h) - 40 features**

```python
# Price features (4)
- returns_15m: log return over 15min bar
- returns_std_15m: rolling volatility (20-bar window)
- high_low_range_15m: (high - low) / close
- close_position_15m: (close - low) / (high - low)

# Momentum indicators (12)
- rsi_14_15m: Relative Strength Index (14 periods)
- macd_15m: MACD line (12, 26, 9)
- macd_signal_15m: Signal line
- macd_hist_15m: Histogram
- cci_20_15m: Commodity Channel Index
- stoch_k_15m: Stochastic %K
- stoch_d_15m: Stochastic %D
- williams_r_15m: Williams %R
- roc_10_15m: Rate of Change
- mfi_14_15m: Money Flow Index
- adx_14_15m: Average Directional Index
- aroon_up_15m, aroon_down_15m: Aroon indicators

# Volatility features (8)
- atr_14_15m: Average True Range
- bollinger_upper_15m: Upper Bollinger Band
- bollinger_lower_15m: Lower Bollinger Band
- bollinger_width_15m: Band width
- keltner_upper_15m: Keltner Channel upper
- keltner_lower_15m: Keltner Channel lower
- historical_vol_15m: Historical volatility (20 periods)
- parkinson_vol_15m: Parkinson volatility estimator

# Volume features (6)
- volume_15m: Raw volume
- volume_sma_15m: Volume SMA (20 periods)
- volume_ratio_15m: volume / volume_sma
- obv_15m: On-Balance Volume
- vwap_15m: Volume-Weighted Average Price
- vwap_distance_15m: (close - vwap) / vwap

# Trend features (6)
- sma_20_15m: Simple Moving Average (20)
- sma_50_15m: SMA (50)
- ema_12_15m: Exponential MA (12)
- ema_26_15m: EMA (26)
- price_to_sma20_15m: close / sma_20
- price_to_sma50_15m: close / sma_50

# Microstructure features (4) - 15min only, not from 1min
- realized_variance_15m: Sum of squared 1min returns in 15min bar
- bid_ask_spread_proxy_15m: (high - low) / close
- order_flow_imbalance_15m: (close - open) / (high - low)
- price_impact_15m: returns / volume
```

**2. Multi-Timeframe (MTF) Features - 110 features**

MTF features duplicate select indicators from other timeframes and align them to the 15min grid using **forward-fill** (no lookahead).

```python
# From 1min (microstructure signals) - 25 features
mtf_1m_features = [
    'returns_1m', 'returns_std_1m',  # Ultra-short momentum
    'rsi_14_1m', 'macd_1m', 'cci_20_1m',  # Fast indicators
    'atr_14_1m', 'bollinger_width_1m',  # Intrabar volatility
    'volume_ratio_1m', 'obv_1m',  # Tick-level volume
    # ... 15 more microstructure features
]

# From 5min (short-term momentum) - 25 features
mtf_5m_features = [
    'returns_5m', 'returns_std_5m',
    'rsi_14_5m', 'macd_5m', 'macd_hist_5m',
    'atr_14_5m', 'bollinger_width_5m',
    'volume_ratio_5m', 'vwap_5m',
    'sma_20_5m', 'ema_12_5m',
    # ... 14 more short-term features
]

# From 30min (mid-term trend) - 30 features
mtf_30m_features = [
    'returns_30m', 'returns_std_30m',
    'rsi_14_30m', 'macd_30m', 'macd_hist_30m',
    'atr_14_30m', 'bollinger_width_30m',
    'volume_ratio_30m', 'vwap_30m',
    'sma_20_30m', 'sma_50_30m', 'ema_12_30m',
    'adx_14_30m', 'aroon_up_30m', 'aroon_down_30m',
    # ... 15 more mid-term features
]

# From 1h (regime context) - 30 features
mtf_1h_features = [
    'returns_1h', 'returns_std_1h',
    'rsi_14_1h', 'macd_1h', 'macd_hist_1h',
    'atr_14_1h', 'bollinger_width_1h',
    'volume_ratio_1h', 'vwap_1h',
    'sma_20_1h', 'sma_50_1h', 'ema_12_1h', 'ema_26_1h',
    'adx_14_1h', 'aroon_up_1h', 'aroon_down_1h',
    # ... 14 more regime features
]
```

**Alignment Example:**

```python
# Compute features on 1h timeframe
df_1h = resample_ohlcv(df_1m, '1h')
df_1h['rsi_14_1h'] = compute_rsi(df_1h, 14)

# Align to 15min grid (forward-fill, no lookahead)
df_15m = df_15m.merge(
    df_1h[['rsi_14_1h']],
    left_index=True,
    right_index=True,
    how='left'
)
df_15m['rsi_14_1h'] = df_15m['rsi_14_1h'].shift(1).ffill()  # Shift prevents lookahead
```

**3. Wavelet Features (NOT for tabular models in Strategy 2)**

Tabular models with MTF indicators skip wavelets (redundant with MTF). Wavelets are for Strategy 1 (single-TF) or sequence models.

**Total Feature Count for Tabular + MTF:**
- Base: 40
- MTF (1m + 5m + 30m + 1h): 110
- **Total: 150 features**

---

### Sequence Models (LSTM, GRU, TCN)

**Recommended MTF Strategy:** Strategy 1 (single-timeframe) or Strategy 3 (MTF ingestion)
**Training timeframe:** Configurable (5m/10m/15m/1h) - choose per experiment
**Feature count:** 25-30 features (Strategy 1) or 4-20 (Strategy 3: raw OHLCV)
**Input shape:** `(n_samples, 60, 25)` - 60 timesteps, 25 features

#### Rationale for Minimal Features

1. **RNNs learn temporal patterns:** No need for momentum indicators; LSTM learns them from raw prices
2. **Gradient stability:** Too many features → vanishing gradients → poor training
3. **Overfitting risk:** 150 features × 60 timesteps = 9000 inputs per sample → massive overfit on small datasets

#### Feature Groups

**1. Raw OHLCV (5 features)**

```python
# No transformation, just raw prices + volume
features = ['open', 'high', 'low', 'close', 'volume']
```

**Important:** These are **scaled** by Phase 1 robust scaler (train-only fit).

**2. Returns and Volatility (3 features)**

```python
# Basic engineered features
- returns: log(close / close.shift(1))
- returns_std: returns.rolling(20).std()
- atr_14: Average True Range (volatility proxy)
```

**3. Wavelet Decomposition (16 features)**

Wavelet features capture multi-scale patterns without needing MTF data. They decompose the price series into low-frequency (trend) and high-frequency (noise) components.

```python
import pywt

# Discrete Wavelet Transform on close prices
coeffs = pywt.wavedec(close_prices, wavelet='db4', level=4)

# Level 4: cA4 (approximation, ~1h scale at 15min bars)
# Level 3: cD4 (detail, ~30min scale)
# Level 2: cD3 (detail, ~15min scale)
# Level 1: cD2 (detail, ~5min scale)
# Level 0: cD1 (detail, ~1min scale, noise)

wavelet_features = {
    'wavelet_trend_1h': cA4,  # Low-frequency trend
    'wavelet_detail_30m': cD4,  # Medium-frequency cycles
    'wavelet_detail_15m': cD3,  # Primary frequency
    'wavelet_detail_5m': cD2,  # High-frequency
    # ... (16 total wavelet features)
}
```

**Why wavelets for sequence models?**
- Captures multi-scale patterns **without** needing MTF data
- Stationary features (better for RNN training)
- Reduces noise (high-frequency components can be dropped)

**4. Volume Features (2 features)**

```python
- volume_ratio: volume / volume.rolling(20).mean()
- obv: On-Balance Volume (cumulative)
```

**Total Feature Count: 5 + 3 + 16 + 2 = 26 features**

**Input Example:**

```python
# Shape: (n_samples, seq_len, n_features) = (14940, 60, 26)
X_train, y_train, weights_train = container.get_sequence_data(
    split='train',
    seq_len=60
)

# Each sample is 60 bars × 26 features = last 15 hours of 15min data
```

---

### CNN Models (InceptionTime, ResNet)

**Recommended MTF Strategy:** Strategy 3 (multi-resolution ingestion) OR Strategy 1 (single-TF)
**Training timeframe:** Configurable (5m/10m/15m/1h) - choose per experiment
**Feature count:** 5-10 features (raw OHLCV + minimal engineering)
**Input shape:** `(n_samples, 60, 5)` for single-TF or `(n_samples, total_seq_len, 5)` for multi-res

#### Feature Groups

**Option 1: Single-Resolution (Strategy 1)**

```python
# Just raw OHLCV
features = ['open', 'high', 'low', 'close', 'volume']

# CNNs learn local patterns via convolutions
# No need for indicators - kernels detect them
```

**Option 2: Multi-Resolution (Strategy 3)**

```python
# Concatenate 3 timeframes: 1min, 5min, 15min
X_1m = last_15_bars_at_1min  # (n, 15, 5)
X_5m = last_3_bars_at_5min   # (n, 3, 5)
X_15m = last_1_bar_at_15min  # (n, 1, 5)

# Concatenate along sequence dimension
X_multi = np.concatenate([X_1m, X_5m, X_15m], axis=1)  # (n, 19, 5)

# CNN applies 1D convolutions across all timeframes
# Learns scale-invariant patterns
```

**Why minimal features for CNNs?**
- CNNs learn features via convolutions (no manual engineering needed)
- Multiple kernel sizes capture different pattern scales
- Inception modules combine kernels (3, 5, 8, 11, 23) for multi-scale learning

---

### Transformers (PatchTST, iTransformer, TFT)

**Recommended MTF Strategy:** Strategy 3 (multi-resolution ingestion) - optional
**Training timeframe:** Configurable (5m/10m/15m/1h) - choose per experiment
**Feature count:** 5 features (single-TF) or 5 × n_timeframes (MTF ingestion)
**Input shape:** `(n_samples, total_seq_len, 5)` or `(n_samples, n_timeframes, seq_len, 5)`

#### Feature Groups

**Raw OHLCV from Multiple Timeframes**

```python
# 4 timeframes: 1min, 5min, 15min, 1h
timeframes = ['1min', '5min', '15min', '1h']

# For each timeframe, include raw OHLCV
features_per_tf = ['open', 'high', 'low', 'close', 'volume']

# Total: 5 features × 4 timeframes = 20 features
```

**Multi-Resolution Tensor Construction:**

```python
# For each 15min timestamp, get synchronized windows
X_1m = get_lookback_window(df_1m, lookback_minutes=60)   # (n, 60, 5)
X_5m = get_lookback_window(df_5m, lookback_minutes=60)   # (n, 12, 5)
X_15m = get_lookback_window(df_15m, lookback_minutes=60) # (n, 4, 5)
X_1h = get_lookback_window(df_1h, lookback_minutes=60)   # (n, 1, 5)

# Option A: Concatenate (for standard Transformers)
X_concat = np.concatenate([X_1m, X_5m, X_15m, X_1h], axis=1)  # (n, 77, 5)

# Option B: Stack (for PatchTST, cross-scale attention)
X_stacked = np.stack([X_1m_padded, X_5m_padded, X_15m_padded, X_1h_padded], axis=1)  # (n, 4, 60, 5)
```

**Why raw features for Transformers?**
- Self-attention learns feature interactions automatically
- Positional encodings capture temporal structure
- Cross-scale attention (in PatchTST) learns multi-resolution patterns
- Feature engineering reduces model's inductive bias

**Additional Features for TFT (Temporal Fusion Transformer):**

TFT distinguishes between "known" and "unknown" features:

```python
# Known features (available at prediction time)
known_features = [
    'hour_of_day',  # 0-23
    'day_of_week',  # 0-6
    'is_market_open',  # Binary
]

# Unknown features (not available at prediction time)
unknown_features = [
    'open', 'high', 'low', 'close', 'volume'
]

# TFT uses separate encoders for known vs unknown
```

---

### Foundation Models (Chronos, TimesFM, Moirai, TimeGPT)

**Recommended MTF Strategy:** Strategy 1 (single-timeframe) - NO MTF
**Training timeframe:** 1min (highest resolution for zero-shot)
**Feature count:** 4 (OHLC only, drop volume)
**Input shape:** `(n_samples, 512)` - context length is 512 bars

#### Feature Groups

**CRITICAL: Minimal feature engineering to preserve pre-training**

```python
# ONLY OHLC (no volume, no indicators)
features = ['open', 'high', 'low', 'close']

# Context length: 512 bars
# For 1min data: 512 minutes ≈ 8.5 hours of history
```

**Normalization (CRITICAL):**

Foundation models expect **zero-mean, unit-variance** inputs.

```python
# Per-sample normalization (NOT global scaler)
def normalize_for_foundation(X):
    """
    X: (n_samples, 512, 4) - OHLC sequences

    Returns: Normalized X with mean=0, std=1 per sample
    """
    mean = X.mean(axis=1, keepdims=True)  # (n_samples, 1, 4)
    std = X.std(axis=1, keepdims=True)    # (n_samples, 1, 4)

    X_norm = (X - mean) / (std + 1e-8)
    return X_norm, mean, std

# Denormalize predictions
y_pred = y_pred_norm * std + mean
```

**Why no feature engineering?**

1. **Pre-training mismatch:** Foundation models were trained on millions of raw time series
2. **Distribution shift:** Adding RSI, MACD, etc. creates input distribution different from pre-training
3. **Zero-shot performance:** Best results come from raw prices that match pre-training data
4. **Feature engineering breaks transfer learning**

**Exception: Chronos Fine-Tuning**

If fine-tuning Chronos on your data (not zero-shot inference), you can add features:

```python
# Fine-tuning mode: add minimal features
features = [
    'open', 'high', 'low', 'close',
    'returns',  # log(close / close.shift(1))
    'volume_ratio'  # volume / volume.rolling(20).mean()
]
# Still avoid heavy engineering (no MTF, no 50+ indicators)
```

---

### Ensemble Models

**Feature Strategy:** Depends on base models

#### Voting Ensemble

Uses predictions from base models, no feature engineering needed:

```python
# Input: same as base models
# Features: determined by base model family

# Example: Voting([XGBoost, LightGBM, CatBoost])
# Uses 150 MTF features (tabular models)
```

#### Stacking Ensemble

Uses **out-of-fold (OOF) predictions** as meta-features:

```python
# Base models: XGBoost, LightGBM, CatBoost, LSTM
# Each base model generates OOF predictions (3 classes, so 3 probabilities)

oof_features = [
    'xgboost_prob_long',
    'xgboost_prob_neutral',
    'xgboost_prob_short',
    'lightgbm_prob_long',
    'lightgbm_prob_neutral',
    'lightgbm_prob_short',
    'catboost_prob_long',
    'catboost_prob_neutral',
    'catboost_prob_short',
    'lstm_prob_long',
    'lstm_prob_neutral',
    'lstm_prob_short',
]
# Total: 12 features (4 models × 3 classes)

# Optional: Add regime features
meta_features = [
    *oof_features,
    'regime_volatility',  # Current vol regime (low, medium, high)
    'regime_trend',  # Current trend regime (up, sideways, down)
]
# Total: 14 features
```

**Meta-learner trains on OOF features:**

```python
# Meta-learner (e.g., Logistic Regression)
meta_model = LogisticRegression()
meta_model.fit(oof_features, y_train)

# Inference: get base model predictions, feed to meta-model
base_preds = [model.predict(X_test).probabilities for model in base_models]
meta_input = np.hstack(base_preds)
final_pred = meta_model.predict(meta_input)
```

---

## MTF Feature Construction

### MTF Strategies Overview

MTF enrichment is **optional**. Choose the strategy that fits your model and experiment:

| Strategy | Description | Best For | Status |
|----------|-------------|----------|--------|
| **Strategy 1: Single-TF** | Train on primary timeframe only, no MTF | Sequence models, baselines | Implemented |
| **Strategy 2: MTF Indicators** | Add indicator features from multiple TFs | Tabular models | Implemented |
| **Strategy 3: MTF Ingestion** | Raw OHLCV bars from multiple TFs as tensors | CNN, Transformer models | Planned |

### Strategy 1: Single-TF (No MTF)

**Goal:** Train on a single timeframe without MTF enrichment.

**Use when:**
- Building baseline models for comparison
- Using sequence models that learn patterns from raw data
- Avoiding complexity in initial experiments

**Configuration:**
```python
# config/pipeline.yaml
mtf:
  enabled: false
  primary_timeframe: "15min"  # or 5m/10m/1h
```

### Strategy 2: MTF Indicators (Optional Enrichment)

**Goal:** Add features from multiple timeframes to a single training timeframe.

**Example:** Train on 15min, add features from 1min, 5min, 30min, 1h.

#### Step 1: Resample to All Timeframes

```python
# src/phase1/stages/mtf/generator.py
from src.phase1/stages/clean/resample import resample_ohlcv

def generate_mtf_dataframes(df_1min: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Resample 1min data to all MTF timeframes.

    Returns:
        {
            '1min': df_1min,
            '5min': df_5min,
            '15min': df_15min,
            '30min': df_30min,
            '1h': df_1h,
        }
    """
    mtf_dfs = {'1min': df_1min}

    for tf in ['5min', '15min', '30min', '1h']:
        mtf_dfs[tf] = resample_ohlcv(df_1min, freq=tf)

    return mtf_dfs
```

#### Step 2: Compute Features on Each Timeframe

```python
def compute_features_all_timeframes(mtf_dfs: dict) -> dict:
    """
    Compute features on each timeframe.

    Returns:
        {
            '1min': df_1min_with_features,
            '5min': df_5min_with_features,
            ...
        }
    """
    from src.phase1.stages.features.engineer import compute_base_features

    mtf_features = {}
    for tf, df in mtf_dfs.items():
        mtf_features[tf] = compute_base_features(df, timeframe=tf)

    return mtf_features
```

#### Step 3: Align Features to Training Timeframe

**CRITICAL:** Use **forward-fill** with **shift(1)** to prevent lookahead.

```python
def align_mtf_features(
    df_train_tf: pd.DataFrame,  # 15min
    mtf_features: dict,
    source_timeframes: list[str]
) -> pd.DataFrame:
    """
    Align MTF features to training timeframe.

    Args:
        df_train_tf: Primary training timeframe (e.g., 15min)
        mtf_features: Dict of feature DataFrames per timeframe
        source_timeframes: Which timeframes to pull features from

    Returns:
        df_train_tf with additional MTF features
    """
    df_aligned = df_train_tf.copy()

    for source_tf in source_timeframes:
        if source_tf == '15min':
            continue  # Skip base timeframe (already have those features)

        # Select features to align
        feature_cols = [c for c in mtf_features[source_tf].columns if c.endswith(f'_{source_tf}')]

        # Merge (left join on timestamp)
        df_aligned = df_aligned.merge(
            mtf_features[source_tf][feature_cols],
            left_index=True,
            right_index=True,
            how='left'
        )

        # Forward-fill with shift to prevent lookahead
        for col in feature_cols:
            # Shift(1) ensures we only use PAST values from higher timeframe
            df_aligned[col] = df_aligned[col].shift(1).ffill()

    return df_aligned
```

**Why shift(1)?**

Without shift, a 1h bar that completes at 10:00 would be available at 10:00. But in reality, we only know it completed AFTER 10:00. Shift(1) ensures we use the 09:00-10:00 bar (previous 1h bar) for the 10:00 15min prediction.

**Example:**

```
Timestamp (15min) | 1h RSI (no shift) | 1h RSI (shift + ffill)
09:00             | 55.0              | NaN (no data yet)
09:15             | 55.0 (lookahead!) | 55.0 (from 08:00-09:00)
09:30             | 55.0 (lookahead!) | 55.0 (from 08:00-09:00)
09:45             | 55.0 (lookahead!) | 55.0 (from 08:00-09:00)
10:00             | 57.2 (lookahead!) | 55.0 (from 08:00-09:00)
10:15             | 57.2              | 57.2 (from 09:00-10:00) ✓
```

---

## Stage 8: Feature Selection Optimization with Optuna

**COMPREHENSIVE GUIDE:** See `docs/guides/FEATURE_SELECTION_OPTIMIZATION.md` for detailed coverage with examples

### Overview

**Pipeline Stage:** 8 of 16
**Goal:** Automatically select the optimal subset of features using Optuna-driven binary include/exclude decisions.
**Trials:** 100 Optuna trials

This stage integrates with the 16-stage ML Factory pipeline, running after feature engineering (Stage 5) and regime detection (Stage 6), but before splits and scaling.

### How Features Feed into Optimization

```
Stage 5 (Features)     Stage 6 (Regime)     Stage 7 (Label Opt)
       │                     │                     │
       ▼                     ▼                     ▼
   162 indicators  +   regime features   +   optimized labels
       │                     │                     │
       └─────────────────────┴─────────────────────┘
                             │
                             ▼
                   ┌─────────────────────┐
                   │  Stage 8: Feature   │
                   │  Selection (Optuna) │
                   │  100 trials         │
                   │  Binary decisions   │
                   └─────────┬───────────┘
                             │
                             ▼
                   Selected feature groups
                             │
                             ▼
                   ┌─────────────────────┐
                   │  Stage 9: Feature   │
                   │  Pruning (Optuna)   │
                   │  50 trials          │
                   │  Importance-based   │
                   └─────────────────────┘
```

### Feature Groups for Selection

Features are organized into logical groups to reduce the search space:

```python
FEATURE_GROUPS = {
    'momentum': ['rsi_14', 'macd', 'macd_signal', 'macd_hist', 'cci_20', 'stoch_k', 'stoch_d', 'williams_r', 'roc_10', 'mfi_14'],
    'volatility': ['atr_14', 'bollinger_width', 'keltner_width', 'historical_vol', 'parkinson_vol', 'gk_vol'],
    'volume': ['volume_ratio', 'obv', 'vwap_distance', 'mfi_14', 'dollar_volume', 'twap'],
    'trend': ['adx_14', 'aroon_up', 'aroon_down', 'supertrend', 'di_plus', 'di_minus'],
    'moving_avg': ['sma_20', 'sma_50', 'ema_12', 'ema_26', 'price_to_sma20', 'price_to_sma50'],
    'microstructure': ['order_flow_imbalance', 'amihud_illiquidity', 'roll_spread', 'corwin_schultz', 'kyle_lambda'],
    'wavelets': ['wavelet_trend', 'wavelet_detail_1', 'wavelet_detail_2', 'wavelet_detail_3', 'wavelet_energy'],
    'temporal': ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'session_progress'],
    'regime': ['volatility_regime', 'trend_regime', 'composite_regime'],
    'entropy': ['shannon_entropy', 'approx_entropy', 'sample_entropy', 'hurst_exponent'],
}
```

### Per-Model Feature Selection Strategies

Different model families benefit from different feature subsets:

| Model Family | Recommended Groups | Excluded Groups | Rationale |
|--------------|-------------------|-----------------|-----------|
| **Boosting** | All groups | None (handles redundancy) | Tree models handle correlated features well |
| **LSTM/GRU** | momentum, volatility, wavelets | microstructure, entropy | RNNs learn temporal patterns; avoid noise |
| **Transformers** | momentum, volume, temporal | wavelets (model learns patterns) | Attention learns cross-feature relationships |
| **CNN** | wavelets, volatility | microstructure | CNNs excel at multi-scale patterns |
| **Classical** | momentum, trend, moving_avg | entropy, wavelets | Traditional signals work best |

### Example: Feature Selection Objective

```python
# src/phase1/stages/feature_select/objective.py
import optuna
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score

def feature_selection_objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    model_family: str,
    config: dict
) -> float:
    """
    Objective function for Stage 8 feature selection.

    Args:
        trial: Optuna trial
        X: Full feature matrix (162 features)
        y: Labels
        feature_names: List of feature names
        model_family: Target model family (affects regularization)
        config: Configuration dict

    Returns:
        Negative score (Optuna minimizes)
    """
    # Binary decisions for each feature group
    selected_groups = {}
    for group_name in FEATURE_GROUPS.keys():
        selected_groups[group_name] = trial.suggest_categorical(
            f'include_{group_name}',
            [True, False]
        )

    # Get selected feature indices
    selected_features = []
    for group_name, include in selected_groups.items():
        if include:
            selected_features.extend(FEATURE_GROUPS[group_name])

    selected_indices = [
        i for i, name in enumerate(feature_names)
        if any(feat in name for feat in selected_features)
    ]

    if len(selected_indices) == 0:
        return float('inf')

    X_subset = X[:, selected_indices]

    # Evaluate with cross-validation
    model = LGBMClassifier(n_estimators=100, max_depth=6, verbose=-1)
    scores = cross_val_score(model, X_subset, y, cv=3, scoring='balanced_accuracy')

    # Regularization based on model family
    n_features = len(selected_indices)
    if model_family in ['lstm', 'gru', 'transformer']:
        # Neural models prefer fewer features
        feature_penalty = 0.002 * n_features
    else:
        # Tabular models handle more features
        feature_penalty = 0.0005 * n_features

    return -(np.mean(scores) - feature_penalty)
```

### Running Feature Selection

```bash
python -m src.phase1.stages.feature_select.run \
    --symbol MES \
    --model-family boosting \
    --n-trials 100
```

---

## Stage 9: Feature Pruning Optimization with Optuna

**COMPREHENSIVE GUIDE:** See `docs/guides/FEATURE_SELECTION_OPTIMIZATION.md#stage-9-feature-pruning-with-optuna` for detailed coverage

### Overview

**Pipeline Stage:** 9 of 16
**Goal:** Fine-tune feature selection by pruning low-importance individual features.
**Trials:** 50 Optuna trials

After Stage 8 selects feature groups, Stage 9 removes individual features within those groups based on learned importance thresholds.

### Importance-Based Pruning Methods

Stage 9 optimizes three pruning parameters:

1. **Importance Threshold:** Minimum feature importance to retain
2. **Correlation Threshold:** Maximum correlation allowed between features
3. **Variance Threshold:** Minimum variance percentile to retain

```python
# src/phase1/stages/feature_prune/optuna_optimizer.py
import optuna

def feature_pruning_search_space(trial: optuna.Trial) -> dict:
    """
    Search space for importance-based feature pruning.
    """
    return {
        'importance_threshold': trial.suggest_float(
            'importance_threshold', 0.0001, 0.01, log=True
        ),
        'max_correlation': trial.suggest_float(
            'max_correlation', 0.85, 0.99
        ),
        'min_variance_percentile': trial.suggest_float(
            'min_variance_percentile', 0.01, 0.10
        ),
        'importance_method': trial.suggest_categorical(
            'importance_method', ['gain', 'split', 'permutation']
        ),
    }
```

### Pruning Strategy by Model Family

| Model Family | Importance Method | Correlation Threshold | Variance Threshold |
|--------------|-------------------|----------------------|-------------------|
| **Boosting** | gain | 0.95 | 0.01 |
| **LSTM/GRU** | permutation | 0.85 | 0.05 |
| **Transformers** | gain | 0.90 | 0.03 |
| **CNN** | gain | 0.88 | 0.02 |
| **Classical** | split | 0.90 | 0.03 |

### Example: Feature Pruning Objective

```python
# src/phase1/stages/feature_prune/objective.py
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
    """
    params = feature_pruning_search_space(trial)

    # Train model to get importances
    model = LGBMClassifier(n_estimators=200, max_depth=8, verbose=-1)
    model.fit(X, y)

    # Get feature importances
    if params['importance_method'] == 'gain':
        importances = model.booster_.feature_importance(importance_type='gain')
    elif params['importance_method'] == 'split':
        importances = model.booster_.feature_importance(importance_type='split')
    else:
        result = permutation_importance(model, X_val, y_val, n_repeats=5)
        importances = result.importances_mean

    # Normalize
    importances = importances / (importances.sum() + 1e-8)

    # Create pruning mask
    keep_mask = importances >= params['importance_threshold']

    # Variance filtering
    variances = np.var(X, axis=0)
    var_threshold = np.percentile(variances, params['min_variance_percentile'] * 100)
    keep_mask &= (variances >= var_threshold)

    # Correlation filtering
    if keep_mask.sum() > 1:
        corr = np.corrcoef(X[:, keep_mask].T)
        keep_indices = np.where(keep_mask)[0]
        for i in range(len(corr)):
            for j in range(i + 1, len(corr)):
                if abs(corr[i, j]) > params['max_correlation']:
                    # Drop feature with lower importance
                    drop_idx = keep_indices[i] if importances[keep_indices[i]] < importances[keep_indices[j]] else keep_indices[j]
                    keep_mask[drop_idx] = False

    if keep_mask.sum() == 0:
        return float('inf')

    # Evaluate pruned feature set
    X_pruned, X_val_pruned = X[:, keep_mask], X_val[:, keep_mask]
    model_pruned = LGBMClassifier(n_estimators=200, max_depth=8, verbose=-1)
    model_pruned.fit(X_pruned, y)
    val_score = model_pruned.score(X_val_pruned, y_val)

    # Parsimony bonus
    removed_features = X.shape[1] - keep_mask.sum()
    parsimony_bonus = 0.0005 * removed_features

    return -(val_score + parsimony_bonus)
```

### Running Feature Pruning

```bash
python -m src.phase1.stages.feature_prune.run \
    --symbol MES \
    --n-trials 50 \
    --importance-method gain
```

### Output Example

```yaml
# experiments/feature_prune/MES_20260118/pruned_features.yaml
symbol: MES
stage: 9_feature_pruning
pruning_params:
  importance_threshold: 0.0023
  max_correlation: 0.92
  min_variance_percentile: 0.03
  importance_method: gain

results:
  features_before: 87
  features_after: 52
  features_removed: 35
  accuracy_before: 0.618
  accuracy_after: 0.627

removed_features:
  - wavelet_detail_3
  - aroon_down
  - stoch_d
  - corwin_schultz
  # ... 31 more
```

---

## Advanced Feature Selection with Optuna

**COMPREHENSIVE GUIDE:** See `docs/guides/FEATURE_SELECTION_OPTIMIZATION.md` for standalone guide with all strategies

This section provides comprehensive coverage of all Optuna-driven feature selection strategies, including implementation details, best practices, and integration with the unified pipeline.

### Overview of Selection Strategies

The ML Factory supports five primary feature selection strategies, all optimized by Optuna:

| Strategy | Stage | Trials | Search Space | Best For |
|----------|-------|--------|--------------|----------|
| **Binary Group Selection** | 8 | 100 | 10 binary decisions | Quick dimensionality reduction |
| **Binary Individual Selection** | 8 | 100 | ~100 binary decisions | Fine-grained optimization |
| **Importance-Based Selection** | 8/9 | 50-100 | threshold, method | Model-aware selection |
| **Recursive Feature Elimination** | 8 | 100 | n_features, step | Optimal subset discovery |
| **Correlation-Based Selection** | 9 | 50 | max_correlation | Redundancy removal |

### Strategy 1: Recursive Feature Elimination (RFE) with Optuna

**Concept:** RFE iteratively removes the least important features until reaching the optimal subset size.

```python
# src/phase1/stages/feature_select/rfe_optimizer.py
import optuna
import numpy as np
from sklearn.feature_selection import RFECV, RFE
from lightgbm import LGBMClassifier
from typing import List, Tuple

class RFEOptunaOptimizer:
    """
    Optuna-optimized Recursive Feature Elimination.

    Optimizes:
    - n_features_to_select: Number of features to keep
    - step: Features removed per iteration
    - estimator: Base model for importance calculation
    """

    def __init__(self, config: dict):
        self.config = config
        self.n_trials = config.get('n_trials', 100)

    def create_objective(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ):
        """Create Optuna objective for RFE optimization."""

        def objective(trial: optuna.Trial) -> float:
            # Sample RFE hyperparameters
            n_features = trial.suggest_int('n_features_to_select', 20, 80)
            step = trial.suggest_categorical('step', [1, 5, 10])
            estimator_name = trial.suggest_categorical(
                'estimator', ['xgboost', 'lightgbm', 'random_forest']
            )

            # Create base estimator
            estimator = self._get_estimator(estimator_name)

            # Run RFE with cross-validation
            rfe = RFECV(
                estimator=estimator,
                step=step,
                min_features_to_select=n_features,
                cv=3,
                scoring='f1_macro',
                n_jobs=-1
            )

            rfe.fit(X, y)

            # Get best score and optimal number of features
            best_score = rfe.cv_results_['mean_test_score'].max()
            optimal_n_features = rfe.n_features_

            # Slight penalty for more features (prefer parsimony)
            parsimony_penalty = 0.0005 * optimal_n_features

            return best_score - parsimony_penalty

        return objective

    def _get_estimator(self, name: str):
        """Get estimator by name."""
        if name == 'lightgbm':
            return LGBMClassifier(n_estimators=100, max_depth=6, verbose=-1)
        elif name == 'xgboost':
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=100, max_depth=6, verbosity=0)
        elif name == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, max_depth=6, n_jobs=-1)

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[List[str], dict]:
        """
        Run RFE optimization with Optuna.

        Returns:
            Tuple of (selected_feature_names, best_params)
        """
        study = optuna.create_study(direction='maximize')
        objective = self.create_objective(X, y, feature_names)

        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        # Get best parameters
        best_params = study.best_params

        # Run final RFE with best params
        estimator = self._get_estimator(best_params['estimator'])
        rfe = RFE(
            estimator=estimator,
            n_features_to_select=best_params['n_features_to_select'],
            step=best_params['step']
        )
        rfe.fit(X, y)

        # Get selected feature names
        selected_features = [
            feature_names[i]
            for i in range(len(feature_names))
            if rfe.support_[i]
        ]

        return selected_features, best_params
```

### Strategy 2: Importance-Based Selection with Multiple Methods

**Concept:** Select features above an importance threshold, where both the threshold and importance method are optimized by Optuna.

```python
# src/phase1/stages/feature_select/importance_optimizer.py
import optuna
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
from typing import List, Callable

class ImportanceOptunaOptimizer:
    """
    Optuna-optimized importance-based feature selection.

    Supports multiple importance methods:
    - gain: Tree-based gain importance
    - split: Tree-based split importance
    - permutation: Model-agnostic permutation importance
    - shap: SHAP value magnitudes
    - mutual_info: Mutual information with target
    - lasso: LASSO coefficient magnitude
    """

    IMPORTANCE_METHODS = {
        'gain': '_compute_gain_importance',
        'split': '_compute_split_importance',
        'permutation': '_compute_permutation_importance',
        'shap': '_compute_shap_importance',
        'mutual_info': '_compute_mutual_info_importance',
        'lasso': '_compute_lasso_importance',
    }

    def __init__(self, config: dict):
        self.config = config

    def create_objective(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str]
    ):
        """Create Optuna objective for importance-based selection."""

        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            importance_method = trial.suggest_categorical(
                'importance_method',
                ['gain', 'split', 'permutation', 'shap', 'mutual_info']
            )
            threshold = trial.suggest_float(
                'importance_threshold', 0.001, 0.1, log=True
            )
            top_k = trial.suggest_int('top_k_features', 20, 100)
            use_threshold = trial.suggest_categorical('use_threshold', [True, False])

            # Compute feature importance
            importance_fn = getattr(self, self.IMPORTANCE_METHODS[importance_method])
            importances = importance_fn(X, y, X_val, y_val)

            # Normalize importances
            importances = importances / (importances.sum() + 1e-8)

            # Create selection mask
            if use_threshold:
                mask = importances >= threshold
            else:
                # Top-K selection
                top_indices = np.argsort(importances)[-top_k:]
                mask = np.zeros(len(importances), dtype=bool)
                mask[top_indices] = True

            # Ensure minimum features
            if mask.sum() < 10:
                top_indices = np.argsort(importances)[-10:]
                mask = np.zeros(len(importances), dtype=bool)
                mask[top_indices] = True

            # Evaluate with selected features
            from lightgbm import LGBMClassifier
            from sklearn.model_selection import cross_val_score

            X_selected = X[:, mask]
            model = LGBMClassifier(n_estimators=100, verbose=-1)
            scores = cross_val_score(model, X_selected, y, cv=3, scoring='f1_macro')

            return np.mean(scores)

        return objective

    def _compute_gain_importance(
        self, X: np.ndarray, y: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray
    ) -> np.ndarray:
        """Compute tree-based gain importance."""
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(n_estimators=200, max_depth=8, verbose=-1)
        model.fit(X, y)
        return model.booster_.feature_importance(importance_type='gain')

    def _compute_split_importance(
        self, X: np.ndarray, y: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray
    ) -> np.ndarray:
        """Compute tree-based split importance."""
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(n_estimators=200, max_depth=8, verbose=-1)
        model.fit(X, y)
        return model.booster_.feature_importance(importance_type='split')

    def _compute_permutation_importance(
        self, X: np.ndarray, y: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray
    ) -> np.ndarray:
        """Compute permutation importance (model-agnostic)."""
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(n_estimators=200, max_depth=8, verbose=-1)
        model.fit(X, y)

        result = permutation_importance(
            model, X_val, y_val,
            n_repeats=5,
            random_state=42,
            n_jobs=-1
        )
        return result.importances_mean

    def _compute_shap_importance(
        self, X: np.ndarray, y: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray
    ) -> np.ndarray:
        """Compute SHAP-based importance."""
        import shap
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(n_estimators=200, max_depth=8, verbose=-1)
        model.fit(X, y)

        # Use TreeSHAP for speed
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val[:5000])  # Sample for speed

        # Mean absolute SHAP value per feature
        if isinstance(shap_values, list):
            # Multi-class: average across classes
            shap_importance = np.mean([
                np.abs(sv).mean(axis=0) for sv in shap_values
            ], axis=0)
        else:
            shap_importance = np.abs(shap_values).mean(axis=0)

        return shap_importance

    def _compute_mutual_info_importance(
        self, X: np.ndarray, y: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray
    ) -> np.ndarray:
        """Compute mutual information with target."""
        return mutual_info_classif(X, y, n_neighbors=5, random_state=42)

    def _compute_lasso_importance(
        self, X: np.ndarray, y: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray
    ) -> np.ndarray:
        """Compute LASSO-based importance via coefficient magnitude."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        # Standardize features for LASSO
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit LASSO logistic regression
        model = LogisticRegression(
            penalty='l1',
            solver='saga',
            C=0.1,
            max_iter=1000,
            random_state=42
        )
        model.fit(X_scaled, y)

        # Mean absolute coefficient across classes
        return np.abs(model.coef_).mean(axis=0)
```

### Strategy 3: Correlation-Based Selection with Optuna

**Concept:** Remove highly correlated features to reduce redundancy while preserving predictive power.

```python
# src/phase1/stages/feature_select/correlation_optimizer.py
import optuna
import numpy as np
from typing import List, Tuple

class CorrelationOptunaOptimizer:
    """
    Optuna-optimized correlation-based feature selection.

    Optimizes:
    - max_correlation: Maximum allowed correlation between features
    - correlation_method: pearson, spearman, or kendall
    - keep_criterion: Which feature to keep when removing correlated pairs
    """

    def __init__(self, config: dict):
        self.config = config

    def create_objective(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        base_importances: np.ndarray = None
    ):
        """Create Optuna objective for correlation-based selection."""

        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
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

            # Find correlated pairs and remove one
            mask = np.ones(len(feature_names), dtype=bool)
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if upper.iloc[i, j] > max_corr and mask[i] and mask[j]:
                        # Decide which to remove
                        if keep_criterion == 'higher_importance':
                            if base_importances is not None:
                                remove_idx = i if base_importances[i] < base_importances[j] else j
                            else:
                                remove_idx = j  # Default to removing second
                        elif keep_criterion == 'higher_variance':
                            var_i = np.var(X[:, i])
                            var_j = np.var(X[:, j])
                            remove_idx = i if var_i < var_j else j
                        else:  # first_in_list
                            remove_idx = j

                        mask[remove_idx] = False

            # Evaluate with selected features
            if mask.sum() < 10:
                return 0.0  # Too few features

            X_selected = X[:, mask]

            from lightgbm import LGBMClassifier
            from sklearn.model_selection import cross_val_score

            model = LGBMClassifier(n_estimators=100, verbose=-1)
            scores = cross_val_score(model, X_selected, y, cv=3, scoring='f1_macro')

            # Bonus for removing more correlated features
            removal_bonus = 0.001 * (len(feature_names) - mask.sum())

            return np.mean(scores) + removal_bonus

        return objective
```

### Combined Feature Selection Pipeline

**Concept:** Run all selection strategies in sequence, with each stage refining the feature set.

```python
# src/phase1/stages/feature_select/combined_optimizer.py
import optuna
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class FeatureSelectionResult:
    """Result from combined feature selection."""
    selected_features: List[str]
    selection_mask: np.ndarray
    best_params: Dict[str, Any]
    optimization_history: List[Dict]
    metrics: Dict[str, float]

class CombinedFeatureSelector:
    """
    Combined feature selection pipeline using multiple Optuna-optimized strategies.

    Pipeline order:
    1. Binary group selection (coarse filtering)
    2. Importance-based selection (model-aware filtering)
    3. RFE (optimal subset discovery)
    4. Correlation-based selection (redundancy removal)
    """

    def __init__(self, config: dict):
        self.config = config
        self.rfe_optimizer = RFEOptunaOptimizer(config.get('rfe', {}))
        self.importance_optimizer = ImportanceOptunaOptimizer(config.get('importance', {}))
        self.correlation_optimizer = CorrelationOptunaOptimizer(config.get('correlation', {}))

    def run_full_pipeline(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        model_family: str = 'boosting'
    ) -> FeatureSelectionResult:
        """
        Run full feature selection pipeline.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: List of feature names
            model_family: Target model family for tailored selection

        Returns:
            FeatureSelectionResult with selected features and metadata
        """
        results = {}
        current_mask = np.ones(len(feature_names), dtype=bool)
        current_features = feature_names.copy()

        # Step 1: Binary Group Selection (if enabled)
        if self.config.get('binary_group_selection', {}).get('enabled', True):
            group_mask = self._run_group_selection(X, y, feature_names, model_family)
            current_mask &= group_mask
            results['group_selection'] = {
                'features_before': len(feature_names),
                'features_after': current_mask.sum()
            }

        # Step 2: Importance-Based Selection
        if self.config.get('importance_based', {}).get('enabled', True):
            X_subset = X[:, current_mask]
            X_val_subset = X_val[:, current_mask]
            subset_features = [f for f, m in zip(feature_names, current_mask) if m]

            importance_study = optuna.create_study(direction='maximize')
            importance_obj = self.importance_optimizer.create_objective(
                X_subset, y, X_val_subset, y_val, subset_features
            )
            importance_study.optimize(importance_obj, n_trials=50)

            results['importance_selection'] = importance_study.best_params

        # Step 3: RFE (if enabled)
        if self.config.get('rfe', {}).get('enabled', True):
            X_subset = X[:, current_mask]
            subset_features = [f for f, m in zip(feature_names, current_mask) if m]

            rfe_selected, rfe_params = self.rfe_optimizer.optimize(
                X_subset, y, subset_features
            )
            results['rfe'] = {
                'params': rfe_params,
                'features_selected': len(rfe_selected)
            }

            # Update mask based on RFE results
            rfe_mask = np.array([f in rfe_selected for f in subset_features])
            temp_mask = np.zeros(len(feature_names), dtype=bool)
            temp_mask[current_mask] = rfe_mask
            current_mask = temp_mask

        # Step 4: Correlation-Based Selection
        if self.config.get('correlation', {}).get('enabled', True):
            X_subset = X[:, current_mask]
            subset_features = [f for f, m in zip(feature_names, current_mask) if m]

            corr_study = optuna.create_study(direction='maximize')
            corr_obj = self.correlation_optimizer.create_objective(
                X_subset, y, subset_features
            )
            corr_study.optimize(corr_obj, n_trials=30)

            results['correlation_selection'] = corr_study.best_params

        # Final selected features
        selected_features = [f for f, m in zip(feature_names, current_mask) if m]

        return FeatureSelectionResult(
            selected_features=selected_features,
            selection_mask=current_mask,
            best_params=results,
            optimization_history=[],
            metrics={
                'features_before': len(feature_names),
                'features_after': len(selected_features),
                'reduction_ratio': 1 - len(selected_features) / len(feature_names)
            }
        )

    def _run_group_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model_family: str
    ) -> np.ndarray:
        """Run binary group selection."""
        # Model-family-specific group defaults
        GROUP_DEFAULTS = {
            'boosting': {
                'momentum': True, 'volatility': True, 'volume': True,
                'trend': True, 'microstructure': True, 'wavelets': False,
                'temporal': True, 'regime': True
            },
            'neural': {
                'momentum': True, 'volatility': True, 'volume': False,
                'trend': False, 'microstructure': False, 'wavelets': True,
                'temporal': True, 'regime': True
            },
            'transformer': {
                'momentum': False, 'volatility': True, 'volume': False,
                'trend': False, 'microstructure': False, 'wavelets': False,
                'temporal': True, 'regime': True
            }
        }

        defaults = GROUP_DEFAULTS.get(model_family, GROUP_DEFAULTS['boosting'])

        # Create mask based on feature groups
        mask = np.ones(len(feature_names), dtype=bool)

        for group_name, include in defaults.items():
            if not include:
                for i, feat in enumerate(feature_names):
                    if group_name in feat.lower():
                        mask[i] = False

        return mask
```

### Per-Model Feature Selection Strategies

Different model families benefit from different selection strategies:

| Model Family | Recommended Strategy | Key Parameters |
|--------------|---------------------|----------------|
| **Boosting** (XGBoost, LightGBM, CatBoost) | All strategies, prefer gain importance | max_features: 60-100, correlation: 0.95 |
| **Neural** (LSTM, GRU) | RFE + importance, fewer features | max_features: 30-50, correlation: 0.85 |
| **Transformers** (PatchTST, iTransformer) | Minimal selection, raw features preferred | max_features: 20-40, skip engineered |
| **CNN** (InceptionTime, ResNet1D) | Wavelet emphasis, RFE | max_features: 40-60, wavelets: always |
| **Classical** (RF, SVM, Logistic) | All strategies, strong regularization | max_features: 30-50, LASSO importance |

---

## Feature Selection Strategies (Manual)

In addition to Optuna-based optimization (Stages 8-9), manual feature selection methods are available for exploration and debugging.

### Mean Decrease Accuracy (MDA)

**Idea:** Permute a feature and measure performance drop. Large drop = important feature.

```python
# src/cross_validation/feature_selector.py
from sklearn.inspection import permutation_importance

def select_features_mda(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 10,
    threshold: float = 0.01
) -> list[str]:
    """
    Select features using Mean Decrease Accuracy.

    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation labels
        feature_names: List of feature names
        n_repeats: Permutation repeats
        threshold: Minimum importance to keep feature

    Returns:
        List of selected feature names
    """
    result = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1
    )

    # Select features with importance > threshold
    selected_features = [
        feature_names[i]
        for i in range(len(feature_names))
        if result.importances_mean[i] > threshold
    ]

    return selected_features
```

**When to use:**
- **Tabular models:** Reduce 200 features → 100 most important
- **After initial training:** Use MDA to identify redundant features
- **Model-specific:** Different models may select different features

**Limitations:**
- Computationally expensive (requires many re-evaluations)
- Correlated features may have low individual importance but high joint importance

---

### Mean Decrease Impurity (MDI)

**Idea:** For tree-based models, sum the total reduction in impurity (Gini, entropy) from splits on each feature.

```python
def select_features_mdi(
    model,  # XGBoost, LightGBM, or RandomForest
    feature_names: list[str],
    threshold: float = 0.001
) -> list[str]:
    """
    Select features using Mean Decrease Impurity (tree-based models only).

    Args:
        model: Trained tree-based model
        feature_names: List of feature names
        threshold: Minimum importance to keep feature

    Returns:
        List of selected feature names
    """
    # Get feature importances from model
    importances = model.feature_importances_

    # Select features with importance > threshold
    selected_features = [
        feature_names[i]
        for i in range(len(feature_names))
        if importances[i] > threshold
    ]

    return selected_features
```

**When to use:**
- **Boosting/tree models only:** LightGBM, XGBoost, CatBoost, RandomForest
- **Fast:** No re-evaluation needed
- **Interpretability:** Understand which features drive splits

**Limitations:**
- **Biased toward high-cardinality features**
- **Not applicable to neural models**

---

### Correlation Filtering

**Idea:** Remove highly correlated features to reduce redundancy.

```python
def select_features_correlation(
    X: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.95
) -> list[str]:
    """
    Remove features with correlation > threshold.

    Args:
        X: Feature matrix
        feature_names: List of feature names
        threshold: Correlation threshold (0.95 typical)

    Returns:
        List of selected feature names (uncorrelated)
    """
    import pandas as pd

    df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df.corr().abs()

    # Find pairs with correlation > threshold
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Drop features with correlation > threshold
    to_drop = [
        column for column in upper_tri.columns
        if any(upper_tri[column] > threshold)
    ]

    selected_features = [f for f in feature_names if f not in to_drop]

    return selected_features
```

**When to use:**
- **Before training:** Reduce 200 features → 150 uncorrelated features
- **All model types:** Helps prevent multicollinearity (though boosting handles it well)
- **Interpretability:** Easier to interpret uncorrelated features

**Threshold guidance:**
- 0.95: Very conservative (remove near-duplicates only)
- 0.90: Moderate (recommended)
- 0.80: Aggressive (may remove useful feature combinations)

---

### Walk-Forward Feature Selection

**Idea:** Select features using cross-validation, re-selecting for each fold.

```python
def walk_forward_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    model_class,
    n_splits: int = 5
) -> list[str]:
    """
    Select features using walk-forward CV.

    Returns:
        Features selected in majority of folds
    """
    from src.cross_validation.purged_kfold import PurgedKFold

    kfold = PurgedKFold(n_splits=n_splits, purge_bars=60, embargo_bars=480)

    feature_votes = {f: 0 for f in feature_names}

    for train_idx, val_idx in kfold.split(X_train):
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_val = y_train[val_idx]

        # Train model
        model = model_class()
        model.fit(X_fold_train, y_fold_train, X_fold_val, y_fold_val)

        # Select features using MDA
        selected = select_features_mda(model, X_fold_val, y_fold_val, feature_names)

        # Vote
        for f in selected:
            feature_votes[f] += 1

    # Keep features selected in majority of folds
    final_selected = [f for f, votes in feature_votes.items() if votes >= n_splits // 2]

    return final_selected
```

**When to use:**
- **Robust feature selection:** Avoid selecting features that only work on one fold
- **Prevents overfitting:** Features must generalize across folds
- **Expensive:** Requires training n_splits models

---

## Feature Validation

### NaN Checks

```python
def validate_no_nans(df: pd.DataFrame, stage: str):
    """
    Ensure no NaNs in features.

    Args:
        df: Feature DataFrame
        stage: Stage name for error message
    """
    nan_cols = df.columns[df.isna().any()].tolist()

    if nan_cols:
        nan_counts = df[nan_cols].isna().sum().to_dict()
        raise ValueError(
            f"[{stage}] NaN values detected in features: {nan_counts}\n"
            "Possible causes:\n"
            "  - Insufficient history for indicator calculation\n"
            "  - Missing data in raw OHLCV\n"
            "  - MTF alignment issue (check forward-fill)\n"
            "Fix: Drop NaN rows or impute"
        )
```

### Infinite Value Checks

```python
def validate_no_infs(df: pd.DataFrame, stage: str):
    """
    Ensure no infinite values in features.

    Args:
        df: Feature DataFrame
        stage: Stage name for error message
    """
    inf_cols = df.columns[np.isinf(df).any()].tolist()

    if inf_cols:
        raise ValueError(
            f"[{stage}] Infinite values detected in features: {inf_cols}\n"
            "Possible causes:\n"
            "  - Division by zero (check volume, ATR, etc.)\n"
            "  - Log of negative number\n"
            "  - Overflow in calculations\n"
            "Fix: Clip values or fix calculation"
        )
```

### Feature Correlation Validation

```python
def validate_feature_correlation(
    X: np.ndarray,
    feature_names: list[str],
    max_correlation: float = 0.99
):
    """
    Warn if features are extremely correlated (near-duplicates).

    Args:
        X: Feature matrix
        feature_names: Feature names
        max_correlation: Warn if abs(corr) > this threshold
    """
    import pandas as pd

    df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df.corr().abs()

    # Find pairs with correlation > threshold
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if corr_matrix.iloc[i, j] > max_correlation:
                high_corr_pairs.append((
                    feature_names[i],
                    feature_names[j],
                    corr_matrix.iloc[i, j]
                ))

    if high_corr_pairs:
        warnings.warn(
            f"Found {len(high_corr_pairs)} feature pairs with correlation > {max_correlation}:\n" +
            "\n".join([f"  {f1} <-> {f2}: {corr:.4f}" for f1, f2, corr in high_corr_pairs[:10]])
        )
```

### Feature Quality Score

```python
def compute_feature_quality(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str]
) -> pd.DataFrame:
    """
    Compute quality metrics for each feature.

    Returns:
        DataFrame with columns: feature, variance, nan_pct, inf_pct, unique_pct
    """
    quality = []

    for i, feature in enumerate(feature_names):
        feat_values = X_train[:, i]

        quality.append({
            'feature': feature,
            'variance': np.var(feat_values),
            'nan_pct': np.isnan(feat_values).mean(),
            'inf_pct': np.isinf(feat_values).mean(),
            'unique_pct': len(np.unique(feat_values)) / len(feat_values),
            'mean': np.mean(feat_values),
            'std': np.std(feat_values),
        })

    return pd.DataFrame(quality)
```

---

## Feature Scaling

Phase 1 applies robust scaling (median + IQR) to all features. **Scaler is fit on train set only** to prevent leakage.

```python
# src/phase1/stages/scaling/scaler.py
from sklearn.preprocessing import RobustScaler

def fit_scaler_train_only(
    X_train: np.ndarray,
    feature_names: list[str]
) -> RobustScaler:
    """
    Fit RobustScaler on training set only.

    Returns:
        Fitted scaler
    """
    scaler = RobustScaler()
    scaler.fit(X_train)

    return scaler

def apply_scaler(
    X: np.ndarray,
    scaler: RobustScaler
) -> np.ndarray:
    """
    Apply pre-fitted scaler to data.

    Args:
        X: Unscaled features
        scaler: Fitted scaler (from train set)

    Returns:
        Scaled features
    """
    return scaler.transform(X)
```

### Why RobustScaler?

- **Robust to outliers:** Uses median and IQR instead of mean and std
- **Financial data has outliers:** Flash crashes, extreme moves, gaps
- **Better than StandardScaler:** StandardScaler is sensitive to outliers

### Scaling by Model Family

| Model Family | Scaling Required? | Scaler Type |
|--------------|-------------------|-------------|
| **Boosting** | No (optional) | RobustScaler (slight improvement) |
| **Neural** | **Yes** (critical) | RobustScaler or MinMaxScaler |
| **Classical** | **Yes** (critical for SVM, Logistic) | RobustScaler or StandardScaler |
| **Foundation** | **Yes** (per-sample normalization) | Z-score (mean=0, std=1) |

---

## Adding New Feature Groups

### Step 1: Define Feature Function

```python
# src/phase1/stages/features/indicators.py

def compute_my_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute custom features.

    Args:
        df: OHLCV DataFrame with columns [open, high, low, close, volume]

    Returns:
        DataFrame with new feature columns
    """
    df = df.copy()

    # Example: Heikin-Ashi candles
    df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['ha_open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
    df['ha_high'] = df[['high', 'ha_open', 'ha_close']].max(axis=1)
    df['ha_low'] = df[['low', 'ha_open', 'ha_close']].min(axis=1)

    # Example: Custom momentum indicator
    df['custom_momentum'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)

    return df
```

### Step 2: Integrate into Feature Pipeline

```python
# src/phase1/stages/features/run.py

from src.phase1.stages.features.indicators import compute_my_custom_features

def run_features_stage(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    # ... existing feature groups ...

    # Add custom features
    if config.enable_custom_features:
        logger.info("Computing custom features...")
        df = compute_my_custom_features(df)

    return df
```

### Step 3: Add Configuration Flag

```python
# src/phase1/pipeline_config.py

@dataclass
class PipelineConfig:
    # ... existing fields ...

    enable_custom_features: bool = False  # Enable custom feature group
```

### Step 4: Add Tests

```python
# tests/phase_1_tests/stages/test_custom_features.py

def test_custom_features():
    df = create_sample_ohlcv(n_bars=1000)

    df_features = compute_my_custom_features(df)

    # Validate features exist
    assert 'ha_close' in df_features.columns
    assert 'custom_momentum' in df_features.columns

    # Validate no NaNs (after dropna)
    df_features = df_features.dropna()
    assert df_features['ha_close'].isna().sum() == 0
```

---

## Performance Considerations

### Feature Computation Cost

**Profiling results** (15min timeframe, 2 years of data):

| Feature Group | Computation Time | % of Pipeline |
|---------------|------------------|---------------|
| Raw OHLCV | 0.1s | 0.5% |
| Basic indicators (RSI, MACD, etc.) | 2.5s | 12% |
| Wavelet decomposition | 5.0s | 24% |
| MTF resampling | 1.5s | 7% |
| MTF feature alignment | 3.0s | 14% |
| Volume features | 0.8s | 4% |
| Microstructure features | 1.2s | 6% |
| **Total feature engineering** | **14.1s** | **67%** |

**Feature engineering is the bottleneck in Phase 1.**

### Optimization Strategies

1. **Vectorization:** Use NumPy/Pandas vectorized operations, avoid Python loops
2. **Caching:** Cache MTF dataframes (don't recompute for each horizon)
3. **Parallel processing:** Compute features for different timeframes in parallel
4. **Selective features:** Only compute features needed for your model family

### Example: Parallel MTF Feature Computation

```python
from multiprocessing import Pool

def compute_features_parallel(mtf_dfs: dict) -> dict:
    """
    Compute features on multiple timeframes in parallel.

    Args:
        mtf_dfs: Dict of DataFrames per timeframe

    Returns:
        Dict of feature DataFrames per timeframe
    """
    with Pool(processes=4) as pool:
        # Parallel map: compute_base_features on each timeframe
        timeframes = list(mtf_dfs.keys())
        dfs = list(mtf_dfs.values())

        results = pool.starmap(compute_base_features, [(df, tf) for df, tf in zip(dfs, timeframes)])

    return {tf: result for tf, result in zip(timeframes, results)}
```

**Speedup:** 3-4x on 4-core CPU for MTF feature computation.

---

## Code Examples

### Complete Example: Tabular Model Feature Engineering

```python
# Full pipeline for tabular models (XGBoost, LightGBM)
from src.phase1.stages.clean.resample import resample_ohlcv
from src.phase1.stages.features.engineer import compute_base_features
from src.phase1.stages.mtf.generator import MTFFeatureGenerator

# Step 1: Load 1min data
df_1min = pd.read_parquet('data/raw/MES_1m.parquet')

# Step 2: Resample to training timeframe (15min)
df_15min = resample_ohlcv(df_1min, freq='15min')

# Step 3: Compute base features (15min)
df_15min = compute_base_features(df_15min, timeframe='15min')
# Now has: open, high, low, close, volume + 40 indicators

# Step 4: Generate MTF features (Strategy 2)
mtf_generator = MTFFeatureGenerator(
    base_timeframe='15min',
    mtf_timeframes=['1min', '5min', '30min', '1h'],
    mode='indicators'
)

df_15min = mtf_generator.add_mtf_features(df_1min, df_15min)
# Now has: 40 base + 110 MTF = 150 features

# Step 5: Drop NaNs
df_15min = df_15min.dropna()

# Step 6: Extract features and labels
feature_cols = [c for c in df_15min.columns if c not in ['label', 'sample_weight']]
X = df_15min[feature_cols].values
y = df_15min['label'].values

# Step 7: Scale (train-only fit)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 8: Train model
from src.models.boosting.xgboost_model import XGBoostModel
model = XGBoostModel()
model.fit(X_train_scaled, y_train, X_test_scaled[:500], y_test[:500])
```

### Complete Example: Sequence Model Feature Engineering

```python
# Full pipeline for sequence models (LSTM, GRU)
from src.phase1.stages.clean.resample import resample_ohlcv
from src.phase1.stages.features.engineer import compute_sequence_features

# Step 1: Load 1min data
df_1min = pd.read_parquet('data/raw/MES_1m.parquet')

# Step 2: Resample to training timeframe (15min)
df_15min = resample_ohlcv(df_1min, freq='15min')

# Step 3: Compute minimal features
df_15min = compute_sequence_features(df_15min)
# Features: open, high, low, close, volume, returns, returns_std, atr_14, wavelets (16)

# Step 4: Drop NaNs
df_15min = df_15min.dropna()

# Step 5: Build sequences
from src.phase1.stages.datasets.sequence_builder import build_sequences

X, y, weights = build_sequences(
    df_15min,
    seq_len=60,
    feature_cols=['open', 'high', 'low', 'close', 'volume', 'returns', 'returns_std', 'atr_14', ...],
    label_col='label',
    weight_col='sample_weight'
)
# X: (n_samples, 60, 26)
# y: (n_samples,)
# weights: (n_samples,)

# Step 6: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 7: Scale (already done in Phase 1, but can re-scale if needed)
from sklearn.preprocessing import RobustScaler
n_samples, seq_len, n_features = X_train.shape

# Reshape to 2D for scaling
X_train_2d = X_train.reshape(-1, n_features)
scaler = RobustScaler()
X_train_scaled_2d = scaler.fit_transform(X_train_2d)
X_train_scaled = X_train_scaled_2d.reshape(n_samples, seq_len, n_features)

# Step 8: Train model
from src.models.neural.lstm_model import LSTMModel
model = LSTMModel(config={'hidden_size': 128, 'num_layers': 2})
model.fit(X_train_scaled, y_train, X_test_scaled[:500], y_test[:500])
```

---

## Summary

### Feature Engineering by Model Family

| Model Family | Strategy | Features | Feature Count | Computation Cost |
|--------------|----------|----------|---------------|------------------|
| **Tabular** | MTF indicators | Dense cross-TF | 150-200 | High (14s) |
| **Sequence** | Single-TF + wavelets | Minimal + wavelets | 25-30 | Medium (7s) |
| **CNN** | Single/Multi-res raw | Raw OHLCV | 5-10 | Low (2s) |
| **Transformer** | Multi-res raw | Raw OHLCV × TFs | 20 (5×4) | Low (2s) |
| **Foundation** | Zero-shot | Raw OHLC only | 4 | Very low (0.5s) |

### Optuna Feature Optimization Summary

| Stage | Optimization | Trials | What It Does |
|-------|--------------|--------|--------------|
| **Stage 8** | Feature Selection | 100 | Binary include/exclude per feature group |
| **Stage 9** | Feature Pruning | 50 | Importance-based individual feature removal |

### Key Takeaways

1. **Model-specific feature engineering is critical:** Don't use 150 features for LSTM or 5 features for XGBoost
2. **MTF Strategy determines features:** Strategy 1 (no MTF), Strategy 2 (MTF indicators), Strategy 3 (multi-res tensors)
3. **Leakage prevention:** Always shift(1) before forward-fill for MTF alignment
4. **Optuna-driven selection:** Stage 8 selects feature groups; Stage 9 prunes individuals
5. **Per-model selection strategies:** Neural models need fewer features than tabular models
6. **Validation is essential:** Check for NaNs, infinities, and extreme correlations
7. **Scaling matters:** Neural models require scaling; foundation models require per-sample normalization

### Pipeline Stage Reference (16 Stages)

```
Stage 1:  Ingestion - Load raw OHLCV
Stage 2:  Cleaning - Resample, gap handling
Stage 3:  Sessions - Trading hours filtering
Stage 4:  MTF Upscaling - 9 timeframes from 1-min
Stage 5:  Features - 162 indicators (this guide)
Stage 6:  Regime - Market regime detection
Stage 7:  OPTUNA Label Optimization - 100 trials
Stage 8:  OPTUNA Feature Selection - 100 trials (this guide)
Stage 9:  OPTUNA Feature Pruning - 50 trials (this guide)
Stage 10: Splits - Train/val/test (70/15/15)
Stage 11: Scaling - Train-only robust scaling
Stage 12: Adaptation - 2D/3D/4D tensor per model type
Stage 13: OPTUNA Hyperparameter Optimization - 100 trials per model
Stage 14: Training - PurgedKFold CV, OOF generation
Stage 15: Stacking - OOF alignment, meta-learner
Stage 16: Bundling - Model + Scaler + Graph -> Artifact
```

### File Paths Reference

- Feature engineering: `src/phase1/stages/features/engineer.py`
- MTF generation: `src/phase1/stages/mtf/generator.py`
- Feature selection (Optuna): `src/phase1/stages/feature_select/`
- Feature pruning (Optuna): `src/phase1/stages/feature_prune/`
- Feature selection (manual): `src/cross_validation/feature_selector.py`
- Scaling: `src/phase1/stages/scaling/scaler.py`
- Wavelets: `src/phase1/stages/features/wavelets.py`
