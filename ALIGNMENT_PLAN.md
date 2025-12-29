# ML Factory Charter & Implementation Plan

**Generated:** 2025-12-29 (Production deployment system for live trading)
**Purpose:** **PRODUCTION SYSTEM** - Train and deploy live trading models, not research
**Architecture:** Dynamic, registry-based ML factory for OHLCV trading
**Model Count:** **19 models** (optimized for production deployment)
**Estimated Effort:** 12-14 weeks (1 engineer) | 7-9 weeks (2 engineers)

---

## ⚠️ CRITICAL: This Is a Production System

**This is NOT a research project.** This factory exists to:
- ✅ Train production-grade models for live futures trading
- ✅ Deploy ensembles to real markets with real capital
- ✅ Mix and match models to build simple → complex ensembles
- ✅ Generate reliable, reproducible, deployable artifacts

**Every component must be production-safe:**
- Zero leakage (purge/embargo enforced)
- Deterministic (same data + seed = same results)
- Robust (handles failures gracefully)
- Auditable (version tracking, checksums, test-set integrity)

**If a model/feature/ensemble isn't production-ready, it doesn't belong here.**

---

## Table of Contents

1. [Factory Charter](#factory-charter)
2. [Core Design Principles](#core-design-principles)
3. [Final Model Suite (19 Models)](#final-model-suite-19-models)
4. [Sample Ensemble Configurations](#sample-ensemble-configurations)
5. [Robustness Requirements](#robustness-requirements)
6. [Dynamic Registry Architecture](#dynamic-registry-architecture)
7. [Contracts & Artifacts](#contracts--artifacts)
8. [Critical Data Pipeline Bugs](#critical-data-pipeline-bugs)
9. [Implementation Roadmap](#implementation-roadmap)
10. [Testing Strategy](#testing-strategy)

---

## Factory Charter

### Objective

Build a **production-grade, model-agnostic ML factory** for **single-contract OHLCV time-series research and deployment**. The factory ingests **one canonical dataset** (one contract, one base timeframe) and **deterministically** produces:

* **Multi-timeframe training corpora** (aligned, leakage-safe)
* **Model-specific representations** (feature matrices, window tensors, patches)
* **Standardized artifacts** (train/eval/inference bundles)

**Goal:** Any model (single or ensemble) can be trained and compared under **identical experimental controls**.

---

### Core Philosophy: One Data Source → Many Model Backends

**Not this:** "Force all models to consume the same 2D feature matrix"
**Instead:** "Adapt the data contract per model family while preserving leakage discipline"

**Examples:**
- **Tabular models** (LightGBM, XGBoost) → Feature matrix engineered for stationarity, monotonic relationships, sparse interactions
- **Sequence models** (PatchTST, TCN, LSTM) → Windowed tensors with correct normalization and temporal alignment
- **Foundation models** (Chronos, TimesFM) → Normalized OHLCV windows for zero-shot inference

**Non-negotiables:** Reproducibility, leakage control, fair evaluation, deterministic dataset generation.

---

### Inference Is First-Class (Not an Afterthought)

Training and serving must share identical pipelines:

* **Consistent feature pipelines** between train and serve (no training-only transforms)
* **Deterministic resampling** (multi-timeframe alignment must match exactly at runtime)
* **Fixed schema contracts** (explicit column sets, dtypes, shapes, time alignment rules)
* **Packaging** that emits predictable outputs:
  - `p_up` (class probabilities)
  - `E[r]` (expected return)
  - `q05/q50/q95` (quantiles for risk bands)
  - `uncertainty` (calibrated confidence intervals)
  - `regime_score` (volatility/trend/structure regime)

**Goal:** Research artifacts become **deployable inference components** consumable by a trading engine with clear semantics and risk-aware thresholds.

---

### Meta-Learning / Stacking Layer (Leakage-Safe)

Ensembling must be principled, not a naive average.

The factory supports a **stacking / meta-learning stage** where base models act as feature generators:

* Generate **out-of-fold (OOF) predictions** for all base learners
* Train meta-models only on **OOF meta-features** (prevents leakage)
* Enforce fold-consistent preprocessing and **time-series-aware splits** (PurgedKFold)
* Ensemble learns **when to trust which model** under different market conditions

**Supported ensemble methods:**
- **Stacking** (OOF-trained meta-learner: Ridge, Elastic Net, LightGBM meta, Logistic)
- **Voting** (weighted averaging, soft/hard)
- **Gating** (regime-conditional weighting: HMM, Softmax, Markov Switching)

---

### Factory Must Be Dynamic (Config-Driven Selection)

**Nothing is "the one model trio."** The system allows **configurable selections:**

* Train **any single model** from any family
* Train **any set of base models** across families (heterogeneous ensembles)
* Optionally attach **ensemble stage** (stacking/voting/gating)
* Optionally attach **calibration** (Temperature Scaling, Isotonic, CQR)
* Optionally attach **RL policy layer** (SAC, TD3, PPO for adaptive position sizing)
* Preserve **identical evaluation controls** across all runs

**Selection via YAML config, not code changes.**

---

## Core Design Principles

### 1. Single-Contract Architecture

**This is a single-contract ML factory.** Each contract is trained in complete isolation.

- **One contract at a time:** Pipeline processes exactly one futures contract per run (MES, MGC, etc.)
- **Complete isolation:** No cross-symbol correlation or feature engineering
- **Symbol configurability:** Easy to switch between contracts via config

```yaml
symbol: "MES"  # or "MGC", "ES", "GC"
```

---

## Data Flow Architecture: One Source → Many Representations

### The Critical Question: How Does One Dataset Feed Different Models?

**Short answer:** One canonical 1-minute OHLCV dataset → Multi-timeframe resampling (9 timeframes) → MTF features OR multi-resolution ingestion → Model-specific feature sets → Model-specific input views

```
Raw 1min OHLCV (MES)
       ↓
[ Clean & Validate → 1min base timeframe ]
       ↓
[ Multi-Timeframe (MTF) Resampling - 9 Timeframes ]
       ├─ 1m  (base)
       ├─ 5m  (5 bars → 1 bar)
       ├─ 10m (10 bars → 1 bar)
       ├─ 15m (15 bars → 1 bar)
       ├─ 20m (20 bars → 1 bar)
       ├─ 25m (25 bars → 1 bar)
       ├─ 30m (30 bars → 1 bar)
       ├─ 45m (45 bars → 1 bar)
       └─ 1h  (60 bars → 1 bar)
       ↓
[ Two MTF Approaches - Model-Specific ]
       ├─ Approach 1: MTF Feature Construction
       │   ├─ Compute indicators on all 9 timeframes
       │   ├─ Align all features to 1m grid (forward-fill)
       │   └─ Result: 200+ aligned features
       │
       └─ Approach 2: Multi-Resolution Ingestion
           ├─ Stack multiple timeframe tensors (1m + 5m + 15m)
           ├─ Models process multi-scale inputs directly
           └─ Result: Multi-resolution 3D/4D tensors
       ↓
[ Model-Specific Feature Selection ]
       ├─ Tabular models → MTF features (100+ from 9 timeframes)
       ├─ Sequence models → OHLCV + wavelets (30 features across 3 timeframes)
       ├─ Transformers → Multi-resolution OHLCV stacks (1m+5m+15m)
       └─ Foundation models → Normalized 1m OHLCV only
       ↓
[ Model-Specific Input Views ]
       ├─ Tabular: (n_samples, n_features) 2D matrix
       ├─ Sequence: (n_samples, seq_len, n_features) 3D tensor
       └─ Multi-res: (n_samples, n_timeframes, seq_len, n_features) 4D tensor
       ↓
[ Training ] → Different models, same experimental controls
```

---

### Multi-Timeframe (MTF) Feature Engineering

**CRITICAL DISTINCTION:**
- **Ingest timeframe:** **1 minute** (raw data ingestion, always 1m)
- **Training timeframe:** **Configurable** (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, or 1h)

**MTF Timeframe Ladder (9 timeframes):**
```
1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h
```

**Rationale for this ladder:**
- **1m:** Microstructure, tick-level noise, order flow
- **5m-15m:** Short-term momentum, scalping signals
- **20m-30m:** Mid-term trend confirmation
- **45m-1h:** Session trends, regime transitions
- **Gaps (no 2m, 3m, etc.):** Reduces noise, focuses on meaningful resampling intervals

**Pipeline flow:**
```
1min raw data (ingest) → Resample to 9 timeframes → Choose training strategy:
  ├─ Strategy 1: Single-timeframe (train on 15m, no MTF)
  ├─ Strategy 2: MTF indicators (train on 15m, add features from 1m/5m/30m/1h)
  └─ Strategy 3: MTF ingestion (feed 1m+5m+15m tensors together)
```

---

### Three MTF Strategies (Mix-and-Match)

#### Strategy 1: Single-Timeframe Training (No MTF)

**Use case:** Simplest approach, good for baseline models

**Config:**
```yaml
training_timeframe: "15m"  # Train on 15m data only
mtf_indicators: false       # Don't include features from other timeframes
mtf_ingestion: false        # Don't feed multiple timeframes together
```

**Result:**
- Train on 15m OHLCV only
- Features computed from 15m data only
- No cross-timeframe information

**Example:** XGBoost trained on 15m bars with 15m RSI, 15m ATR, 15m VWAP

---

#### Strategy 2: MTF Indicators (Train on One TF, Include Features from Other TFs)

**Use case:** Tabular models that benefit from cross-timeframe features

**Config:**
```yaml
training_timeframe: "15m"   # Train on 15m data (primary)
mtf_indicators: true        # Include features from other timeframes
mtf_source_timeframes:      # Which timeframes to pull features from
  - "1m"   # Microstructure signals
  - "5m"   # Short-term momentum
  - "30m"  # Mid-term trend
  - "1h"   # Regime context
```

**Result:**
- Train on 15m bars (n_samples = 15m bar count)
- Each 15m bar has features computed from:
  - 15m indicators (RSI_15m, ATR_15m, VWAP_15m)
  - 1m indicators (microstructure, order flow) - aligned to 15m grid
  - 5m indicators (short momentum) - aligned to 15m grid
  - 30m indicators (trend confirmation) - aligned to 15m grid
  - 1h indicators (regime context) - aligned to 15m grid

**Feature alignment (forward-fill, no lookahead):**
```python
# src/phase1/stages/mtf/align_features.py

# Train on 15m bars
df_15m = resample_ohlcv(df_1m, freq="15min")  # Primary training timeframe

# Compute features from other timeframes
df_1m_features = compute_features(df_1m)   # Microstructure
df_5m_features = compute_features(df_5m)   # Short momentum
df_30m_features = compute_features(df_30m) # Mid-term trend
df_1h_features = compute_features(df_1h)   # Regime

# Align all features to 15m grid (forward-fill, no lookahead)
df_15m = align_mtf_features(
    primary_tf=df_15m,
    mtf_features={
        "1m": df_1m_features,
        "5m": df_5m_features,
        "30m": df_30m_features,
        "1h": df_1h_features,
    }
)

# Result: Each 15m bar has ~150 features from 5 timeframes
```

**Example:** XGBoost trained on 15m bars with:
- 15m RSI, 15m ATR, 15m VWAP (primary TF features)
- 1m order_flow_imbalance, 1m bid_ask_spread (microstructure)
- 5m RSI, 5m MACD (short momentum)
- 30m RSI, 30m ATR (trend confirmation)
- 1h regime_state (regime context)

---

#### Strategy 3: MTF Ingestion (Feed Multiple Timeframes Together)

**Use case:** Sequence/transformer models that learn cross-scale patterns

**Config:**
```yaml
training_timeframe: "15m"   # Primary timeframe for labels/splits
mtf_ingestion: true         # Feed multiple timeframes together
mtf_input_timeframes:       # Which timeframes to stack/concatenate
  - "1m"   # Microstructure
  - "5m"   # Short momentum
  - "15m"  # Primary (trend)
  - "1h"   # Regime context
```

**Result:**
- Labels/splits defined on 15m bars
- Model receives multi-resolution tensors (1m + 5m + 15m + 1h OHLCV)
- No feature engineering (raw OHLCV at multiple scales)

**Input construction:**
```python
# src/phase1/stages/datasets/multi_resolution.py

# For each 15m timestamp, get synchronized lookback windows
X_1m = get_lookback_window(df_1m, lookback=15)   # Last 15 minutes at 1m resolution (15 bars)
X_5m = get_lookback_window(df_5m, lookback=3)    # Last 15 minutes at 5m resolution (3 bars)
X_15m = get_lookback_window(df_15m, lookback=1)  # Current 15m bar (1 bar)
X_1h = get_lookback_window(df_1h, lookback=1)    # Current 1h context (1 bar)

# Stack into 4D tensor
X_multi = np.stack([X_1m, X_5m, X_15m, X_1h], axis=1)  # (n_15m_bars, 4, varying_seq_len, 5)

# Or concatenate along sequence dimension
X_concat = np.concatenate([X_1m, X_5m, X_15m, X_1h], axis=1)  # (n_15m_bars, 20, 5)
```

**Example:** PatchTST trained on 15m bars, receiving:
- 1m OHLCV (15 bars, microstructure)
- 5m OHLCV (3 bars, short momentum)
- 15m OHLCV (1 bar, current trend)
- 1h OHLCV (1 bar, regime context)

---

### Choosing the Primary Training Timeframe

**Key decision:** Which timeframe to use as the primary training dataset?

**Factors to consider:**

| Training TF | Pros | Cons | Best For |
|-------------|------|------|----------|
| **1m** | Max data (high sample count), microstructure signals | Noisy, high computational cost | HFT strategies, order flow models |
| **5m** | Good balance of signal/noise, still high sample count | Some noise, moderate compute | Scalping, short-term momentum |
| **15m** | Clean signals, lower noise, manageable compute | Moderate sample count | Intraday swing, trend following |
| **30m** | Very clean signals, low noise | Lower sample count, miss short-term moves | Session trends, position trading |
| **1h** | Cleanest signals, regime context | Low sample count, miss intraday | Multi-session trends, regime models |

**Sample count comparison (1 year of data):**
- 1m: ~100,000 bars
- 5m: ~20,000 bars
- 15m: ~6,500 bars
- 30m: ~3,250 bars
- 1h: ~1,625 bars

**Configuration:**
```yaml
# config.yaml
data:
  ingest_timeframe: "1m"      # Always 1m (raw data)
  training_timeframe: "15m"   # Primary timeframe for model training

  mtf_strategy: "mtf_indicators"  # or "single_tf" or "mtf_ingestion"

  # For Strategy 2 (MTF indicators)
  mtf_indicators:
    enabled: true
    source_timeframes: ["1m", "5m", "30m", "1h"]  # Optional: which TFs to pull features from

  # For Strategy 3 (MTF ingestion)
  mtf_ingestion:
    enabled: false
    input_timeframes: ["1m", "5m", "15m", "1h"]
```

---

### Implementation Details

#### Step 1: Resample 1min to all 9 timeframes (always done)

```python
# src/phase1/stages/mtf/resample.py
df_1m = load_raw_ohlcv("MES_1m.parquet")  # Ingest timeframe (always 1m)

# Resample to 8 higher timeframes
df_5m = resample_ohlcv(df_1m, freq="5min")
df_10m = resample_ohlcv(df_1m, freq="10min")
df_15m = resample_ohlcv(df_1m, freq="15min")
df_20m = resample_ohlcv(df_1m, freq="20min")
df_25m = resample_ohlcv(df_1m, freq="25min")
df_30m = resample_ohlcv(df_1m, freq="30min")
df_45m = resample_ohlcv(df_1m, freq="45min")
df_1h = resample_ohlcv(df_1m, freq="1h")

# Now we have 9 timeframes available for any training strategy
```

#### Step 2: Choose training strategy (config-driven)

**Strategy 1: Single-timeframe (no MTF)**
```python
# Train on 15m bars, no cross-timeframe features
df_train = df_15m
features = compute_features(df_train)  # Only 15m features
```

**Strategy 2: MTF indicators**
```python
# Train on 15m bars, with features from 1m/5m/30m/1h
df_train = df_15m  # Primary training timeframe

# Compute features from all specified timeframes
features_1m = compute_features(df_1m, prefix="1m_")
features_5m = compute_features(df_5m, prefix="5m_")
features_15m = compute_features(df_15m, prefix="15m_")
features_30m = compute_features(df_30m, prefix="30m_")
features_1h = compute_features(df_1h, prefix="1h_")

# Align all to 15m grid (forward-fill)
df_train = align_mtf_features(
    primary_tf=df_train,
    mtf_features={
        "1m": features_1m,
        "5m": features_5m,
        "15m": features_15m,
        "30m": features_30m,
        "1h": features_1h,
    }
)
# Result: Each 15m bar has ~150 features from 5 timeframes
```

**Strategy 3: MTF ingestion**
```python
# Train on 15m bars, feed 1m+5m+15m+1h tensors
labels_df = df_15m  # Labels/splits defined on 15m

# Create multi-resolution tensors
X_1m = create_sequences(df_1m, lookback=15)   # (n_15m, 15, 5)
X_5m = create_sequences(df_5m, lookback=3)    # (n_15m, 3, 5)
X_15m = create_sequences(df_15m, lookback=1)  # (n_15m, 1, 5)
X_1h = create_sequences(df_1h, lookback=1)    # (n_15m, 1, 5)

# Stack or concatenate
X_multi = np.stack([X_1m, X_5m, X_15m, X_1h], axis=1)  # (n_15m, 4, varying, 5)
# OR
X_concat = np.concatenate([X_1m, X_5m, X_15m, X_1h], axis=1)  # (n_15m, 20, 5)
```

---

### Feature Count by Strategy

**Strategy 1 (Single-TF):**
- Training on 15m → ~40 features (15m indicators only)
- Training on 1m → ~50 features (1m indicators + microstructure)

**Strategy 2 (MTF Indicators):**
- Training on 15m with 4 MTF sources → ~150 features
- Training on 1m with 8 MTF sources → ~200 features
- Training on 1h with 3 MTF sources → ~100 features

**Strategy 3 (MTF Ingestion):**
- 4-timeframe stack (1m+5m+15m+1h) → 5 features × 4 TFs = 20 feature channels
- 3-timeframe stack (1m+5m+15m) → 5 features × 3 TFs = 15 feature channels

---

### Recommended Strategy by Model Family

| Model Family | Recommended Strategy | Training TF | MTF Sources | Rationale |
|--------------|---------------------|-------------|-------------|-----------|
| **XGBoost/LightGBM** | Strategy 2 (MTF indicators) | 15m | 1m, 5m, 30m, 1h | Tabular models benefit from dense cross-TF features |
| **Logistic Regression** | Strategy 1 (Single-TF) | 15m | None | Simple baseline, avoid overfitting |
| **LSTM/TCN** | Strategy 1 or 3 | 15m | None or 1m+5m+15m | Single-TF with wavelets OR multi-res ingestion |
| **InceptionTime/ResNet** | Strategy 3 (MTF ingestion) | 15m | 1m+5m+15m | CNN benefits from multi-scale raw inputs |
| **PatchTST/iTransformer** | Strategy 3 (MTF ingestion) | 15m | 1m+5m+15m+1h | Transformers learn cross-scale attention |
| **TFT** | Strategy 3 (MTF ingestion) | 15m | 1m+5m+15m+1h | Explicit multi-TF support (known/unknown features) |
| **Chronos/TimesFM** | Strategy 1 (Single-TF) | 1m | None | Foundation models: no MTF, matches pre-training |

**General guidelines:**
- **Tabular models:** Prefer Strategy 2 (MTF indicators) with 3-5 source timeframes
- **Sequence models:** Prefer Strategy 1 (single-TF) or Strategy 3 (MTF ingestion) with 3-4 timeframes
- **Transformers:** Prefer Strategy 3 (MTF ingestion) for cross-scale attention
- **Foundation models:** Use Strategy 1 (single-TF) only, no MTF

---

### Model-Specific Feature Sets

**Different models have different feature requirements.** The pipeline can produce features from any combination of the 9 timeframes, depending on the chosen MTF strategy and training timeframe.

---

#### Tabular Models (XGBoost, LightGBM, Logistic)

**Recommended:** Strategy 2 (MTF indicators)

**Example config:**
```yaml
training_timeframe: "15m"
mtf_strategy: "mtf_indicators"
mtf_source_timeframes: ["1m", "5m", "30m", "1h"]
```

**Features used:** ~150 features (training on 15m with 4 MTF sources)
- ✅ **15m indicators** (primary): RSI, MACD, ATR, VWAP, etc. - 40 features
- ✅ **1m indicators** (microstructure): order flow, bid-ask spread, tick direction - 15 features
- ✅ **5m indicators** (short momentum): RSI, MACD, Stochastic - 20 features
- ✅ **30m indicators** (trend confirmation): RSI, ATR, Bollinger Bands - 20 features
- ✅ **1h indicators** (regime context): regime state, volatility regime - 15 features
- ✅ **Interaction features**: RSI_15m × vol_regime_1h, MACD_5m × trend_30m - 20 features
- ✅ **Wavelet features**: Multi-scale decomposition on 15m close - 20 features

**Total:** ~150 features aligned to 15m grid

**Why this approach?**
- Tabular models excel at **cross-timeframe feature interactions**
- Training on 15m reduces noise vs 1m (cleaner signals)
- MTF indicators from 1m/5m/30m/1h provide multi-scale context
- XGBoost/LightGBM handle high dimensionality well

**Input shape:** `(n_15m_bars, 150)` 2D matrix

---

#### Sequence Models (LSTM, TCN, InceptionTime, 1D ResNet)

**Recommended:** Strategy 1 (single-TF) for LSTM/TCN, Strategy 3 (MTF ingestion) for CNN models

**Option A: Single-timeframe with wavelets (LSTM, TCN)**
```yaml
training_timeframe: "15m"
mtf_strategy: "single_tf"
```

**Features used:** ~25 features
- ✅ Raw OHLCV at **15m** (5 features)
- ✅ Returns (log returns, realized volatility) - 2 features
- ✅ Wavelet decompositions (4 levels: D1, D2, D3, A3) - 16 features
- ✅ Basic volume features (volume MA, volume std) - 2 features

**Input shape:** `(n_15m_bars, seq_len=60, 25)` where seq_len=60 bars = 15 hours of 15m data

**Option B: Multi-resolution (InceptionTime, 1D ResNet)**
```yaml
training_timeframe: "15m"
mtf_strategy: "mtf_ingestion"
mtf_input_timeframes: ["1m", "5m", "15m"]
```

**Concatenated shape:** `(n_15m_bars, 19, 5)` where:
- 15 bars from 1m (last 15 minutes)
- 3 bars from 5m (last 15 minutes)
- 1 bar from 15m (current bar)

**Stacked shape:** `(n_15m_bars, 3, 15, 5)` 4D tensor

**Why this approach?**
- RNNs (LSTM/TCN) work well with single-TF + wavelets
- CNNs (InceptionTime/ResNet) benefit from multi-scale raw inputs
- Fewer features than tabular prevents gradient issues

---

#### Transformers (PatchTST, iTransformer, TFT)

**Recommended:** Strategy 3 (MTF ingestion)

**Config:**
```yaml
training_timeframe: "15m"
mtf_strategy: "mtf_ingestion"
mtf_input_timeframes: ["1m", "5m", "15m", "1h"]
```

**Multi-resolution OHLCV:**
- ✅ **1m OHLCV:** 15 bars (last 15 minutes)
- ✅ **5m OHLCV:** 3 bars (last 15 minutes)
- ✅ **15m OHLCV:** 1 bar (current bar)
- ✅ **1h OHLCV:** 1 bar (current hour context)

**Concatenated input:** `(n_15m_bars, 20, 5)` - 20 total bars across 4 timeframes

**Stacked input (requires multi-scale architecture):** `(n_15m_bars, 4, 15, 5)` 4D tensor

**Why multi-resolution?**
- Transformers **learn cross-scale attention patterns** automatically
- Patching (PatchTST) extracts features from each timeframe independently
- Inverted attention (iTransformer) treats timeframes as feature channels
- TFT explicitly supports known/unknown/static feature splits across timeframes

**Model-specific notes:**
- **PatchTST:** Use stacked 4D input with separate patch embeddings per timeframe
- **iTransformer:** Use concatenated input (treats each timeframe as separate feature)
- **TFT:** Use stacked input with explicit known (1m, 5m) vs unknown (15m, 1h) splits

---

#### Foundation Models (Chronos, TimesFM)

**Recommended:** Strategy 1 (single-TF), 1m only

**Config:**
```yaml
training_timeframe: "1m"
mtf_strategy: "single_tf"
```

**Features used:** Normalized OHLCV at **1m** (4 features)
- ✅ Open, High, Low, Close (z-score normalized per 512-bar lookback window)

**Why 1m only, no MTF?**
- Foundation models are **pre-trained on pure OHLCV at single resolutions**
- Adding engineered features **breaks pre-trained representations**
- Adding MTF data **mismatches training distribution**
- Zero-shot inference requires **exact match to pre-training format**

**Input shape:** `(n_1m_bars, context_length=512, 4)`

---

### Feature Selection in Code

**Where feature selection happens:** `src/phase1/stages/datasets/`

```python
# src/phase1/stages/datasets/model_views.py

TABULAR_FEATURES = [
    # Momentum (9 timeframes × 4 indicators = 36 features)
    "rsi_1m", "rsi_5m", "rsi_10m", "rsi_15m", "rsi_20m", "rsi_25m", "rsi_30m", "rsi_45m", "rsi_1h",
    "macd_1m", "macd_5m", "macd_10m", "macd_15m", "macd_20m", "macd_25m", "macd_30m", "macd_45m", "macd_1h",
    "stoch_1m", "stoch_5m", ...,  # (36 momentum features)

    # Volatility (9 timeframes × 4 metrics = 36 features)
    "atr_1m", "atr_5m", "atr_10m", "atr_15m", "atr_20m", "atr_25m", "atr_30m", "atr_45m", "atr_1h",
    "parkinson_1m", "parkinson_5m", ...,  # (36 volatility features)

    # Volume (9 timeframes × 4 metrics = 36 features)
    "vwap_1m", "vwap_5m", "vwap_10m", "vwap_15m", "vwap_20m", "vwap_25m", "vwap_30m", "vwap_45m", "vwap_1h",
    "obv_1m", "obv_5m", ...,  # (36 volume features)

    # Microstructure (1m only, 12 features)
    "bid_ask_spread", "order_flow_imbalance", "tick_direction", "trade_intensity",
    "price_impact", "effective_spread", ...,

    # Regime (3 timeframes × 5 metrics = 15 features)
    "volatility_regime_1m", "volatility_regime_15m", "volatility_regime_1h",
    "trend_regime_1m", "trend_regime_15m", "trend_regime_1h",
    "composite_regime_1m", "composite_regime_15m", "composite_regime_1h",

    # Interactions (30 features)
    "rsi_1m_x_vol_regime", "macd_5m_x_trend_regime", ...,

    # Wavelets (20 features)
    "wavelet_D1", "wavelet_D2", "wavelet_D3", "wavelet_A3", ...,

    # Statistics (5 timeframes × 4 stats = 20 features)
    "mean_1m", "mean_5m", "mean_15m", "mean_30m", "mean_1h",
    "std_1m", "std_5m", ...,
]
# Total: ~200 features

SEQUENCE_FEATURES_SINGLE_TF = [
    # Option A: Single timeframe (1m) with wavelets
    "open", "high", "low", "close", "volume",  # (5 features)
    "log_returns", "realized_vol",  # (2 features)
    "wavelet_D1", "wavelet_D2", "wavelet_D3", "wavelet_A4",  # (16 features, 4 per OHLC)
    "volume_ma_20", "volume_std_20",  # (2 features)
]
# Total: 25 features, shape: (n, 60, 25)

SEQUENCE_FEATURES_MULTI_RES = {
    # Option B: Multi-resolution (1m + 5m + 15m)
    "1m": ["open", "high", "low", "close", "volume"],  # (n, 60, 5)
    "5m": ["open", "high", "low", "close", "volume"],  # (n, 12, 5)
    "15m": ["open", "high", "low", "close", "volume"],  # (n, 4, 5)
    # Concatenated shape: (n, 76, 5) OR Stacked shape: (n, 3, 60, 5) 4D
}

TRANSFORMER_FEATURES_SINGLE_TF = [
    # Option A: Single timeframe (1m) minimal features
    "open", "high", "low", "close", "volume",  # (5 features)
    "volatility_regime_embedding", "trend_regime_embedding",  # (2 features)
]
# Total: 7 features, shape: (n, 512, 7) → patches

TRANSFORMER_FEATURES_MULTI_RES = {
    # Option B: Multi-resolution (1m + 5m + 15m + 1h)
    "1m": ["open", "high", "low", "close", "volume"],  # (n, 60, 5)
    "5m": ["open", "high", "low", "close", "volume"],  # (n, 12, 5)
    "15m": ["open", "high", "low", "close", "volume"],  # (n, 4, 5)
    "1h": ["open", "high", "low", "close", "volume"],  # (n, 1, 5)
    # Concatenated shape: (n, 77, 5) OR Stacked shape: (n, 4, 60, 5) 4D
}

FOUNDATION_FEATURES = [
    "open", "high", "low", "close",  # Normalized OHLCV only, 1m timeframe
]
# Total: 4 features, shape: (n, 512, 4)

def get_feature_view(df: pd.DataFrame, model_family: str, multi_res: bool = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Get model-specific feature view.

    Args:
        df: Input dataframe with all 200+ features
        model_family: "boosting", "classical", "rnn", "cnn", "transformer", "foundation"
        multi_res: If True, return multi-resolution dict for sequence/transformer models

    Returns:
        2D DataFrame for tabular models
        Dict of DataFrames for multi-resolution sequence/transformer models
    """
    if model_family in ["boosting", "classical"]:
        return df[TABULAR_FEATURES]  # (n, 200)

    elif model_family in ["rnn", "cnn"]:
        if multi_res:
            return {
                "1m": df[SEQUENCE_FEATURES_MULTI_RES["1m"]],
                "5m": df[SEQUENCE_FEATURES_MULTI_RES["5m"]],
                "15m": df[SEQUENCE_FEATURES_MULTI_RES["15m"]],
            }
        else:
            return df[SEQUENCE_FEATURES_SINGLE_TF]  # (n, 25)

    elif model_family == "transformer":
        if multi_res:
            return {
                "1m": df[TRANSFORMER_FEATURES_MULTI_RES["1m"]],
                "5m": df[TRANSFORMER_FEATURES_MULTI_RES["5m"]],
                "15m": df[TRANSFORMER_FEATURES_MULTI_RES["15m"]],
                "1h": df[TRANSFORMER_FEATURES_MULTI_RES["1h"]],
            }
        else:
            return df[TRANSFORMER_FEATURES_SINGLE_TF]  # (n, 7)

    elif model_family == "foundation":
        return df[FOUNDATION_FEATURES]  # (n, 4)

    else:
        raise ValueError(f"Unknown model family: {model_family}")
```

---

### Ensemble Compatibility: Mixing Tabular + Sequence

**Question:** Can we ensemble XGBoost (tabular, 80 features) + LSTM (sequence, 20 features) + PatchTST (transformer, 5 features)?

**Answer:** **YES**, via OOF-based stacking.

**How it works:**

1. **Phase 1: Generate base model predictions (OOF)**
   ```python
   # Each model trains on its own feature view
   xgboost_oof = train_oof(XGBoost, feature_view=TABULAR_FEATURES)  # (n_samples, 1)
   lstm_oof = train_oof(LSTM, feature_view=SEQUENCE_FEATURES)        # (n_samples, 1)
   patchtst_oof = train_oof(PatchTST, feature_view=TRANSFORMER_FEATURES)  # (n_samples, 1)
   ```

2. **Phase 2: Stack OOF predictions**
   ```python
   # Meta-learner input: OOF predictions (all same shape now!)
   X_meta = np.column_stack([xgboost_oof, lstm_oof, patchtst_oof])  # (n_samples, 3)

   # Add regime features to help meta-learner learn when to trust each model
   X_meta = np.column_stack([X_meta, df["volatility_regime"], df["trend_regime"]])  # (n_samples, 5)

   # Train meta-learner (Ridge, Elastic Net, or LightGBM)
   meta_model.fit(X_meta, y_train)
   ```

**Key insight:** OOF stacking **collapses heterogeneous inputs** (80 features vs 20 features vs 5 features) into **homogeneous predictions** (3 probabilities), which the meta-learner can combine.

---

### Summary: One Dataset, Many Views

| Model Family | Training TF | MTF Strategy | Feature Count | Input Shape | Rationale |
|--------------|-------------|--------------|---------------|-------------|-----------|
| **XGBoost/LightGBM** | 15m | Strategy 2 (MTF indicators) | ~150 | `(n_15m, 150)` | Dense cross-TF features |
| **Logistic Regression** | 15m | Strategy 1 (Single-TF) | ~40 | `(n_15m, 40)` | Simple baseline |
| **LSTM/TCN** | 15m | Strategy 1 (Single-TF) | ~25 | `(n_15m, 60, 25)` | Single-TF + wavelets |
| **InceptionTime/ResNet** | 15m | Strategy 3 (MTF ingestion) | 5×3 TFs | `(n_15m, 19, 5)` | Multi-scale CNN |
| **PatchTST/iTransformer** | 15m | Strategy 3 (MTF ingestion) | 5×4 TFs | `(n_15m, 20, 5)` | Cross-scale attention |
| **TFT** | 15m | Strategy 3 (MTF ingestion) | 5×4 TFs | `(n_15m, 4, 15, 5)` | Explicit multi-TF |
| **Chronos/TimesFM** | 1m | Strategy 1 (Single-TF) | 4 | `(n_1m, 512, 4)` | Zero-shot, no MTF |

**All models share (experimental controls):**
- ✅ Same **1min ingest data** (raw OHLCV ingestion)
- ✅ Same **9-timeframe resampling** available (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)
- ✅ Same **deterministic resampling** (fixed seeds)
- ✅ Same **leakage protections** (purge/embargo, train-only scaling, causal models)
- ✅ Same **transaction costs** in triple-barrier labels

**Different per model (configuration choices):**
- ❌ **Training timeframe** (15m recommended for most, 1m for foundation models)
- ❌ **MTF strategy** (single-TF vs MTF indicators vs MTF ingestion)
- ❌ **Feature subset** (tabular uses 150 cross-TF, transformers use 5 raw OHLCV × 4 TFs)
- ❌ **Input shape** (2D matrix vs 3D tensor vs 4D multi-res tensor)
- ❌ **Feature engineering** (hand-crafted vs learned)

**Key architectural insights:**
1. **Training timeframe is configurable** - 1m (ingest) ≠ 15m (training)
2. **Three MTF strategies** - mix-and-match based on model architecture
3. **Tabular models:** Dense feature engineering (Strategy 2, MTF indicators)
4. **Sequence models:** Single-TF + wavelets OR multi-res ingestion
5. **Transformers:** Multi-res ingestion for cross-scale attention
6. **Foundation models:** Single-TF only (1m, no MTF)

**Result:** Fair comparison under identical experimental controls (same ingest data, same resampling, same leakage protections), but each model gets the **training timeframe, MTF strategy, and feature representation** it needs to perform best.

---

### 2. Leakage Paranoia

All models trained on **same leakage-safe data:**

- **Purge:** 60 bars (prevents label leakage from overlapping barrier windows)
- **Embargo:** 1440 bars (~5 days at 5-min bars, prevents serial correlation)
- **Train-only scaling:** Fit scalers on train set, transform all splits
- **Causal models only:** No future-attending transformers (basic Transformer removed)

### 3. Fair Evaluation Under Identical Controls

- Identical transaction costs, slippage, risk constraints
- Standardized metrics: Sharpe, win rate, max drawdown, regime performance
- Cross-validation: PurgedKFold, walk-forward, CPCV, PBO

### 4. Reproducibility

- Dataset fingerprint + config hash for every run
- Deterministic resampling (fixed seeds)
- Versioned artifacts (preproc + schema + weights + metrics)

---

## Final Model Suite (19 Models)

**Optimized from 29 → 19** by removing 10 redundant/weak models.

### Models Removed (10 Total)

| Model | Reason | Replacement |
|-------|--------|-------------|
| CatBoost | 3rd boosting model is overkill | XGBoost (stable) + LightGBM (fast) |
| GRU | Too similar to LSTM (<2% difference) | LSTM (better long-term memory) |
| Transformer (basic) | Non-causal, leaks future data | PatchTST (causal, SOTA) |
| WaveNet | Redundant with TCN (both dilated CNN) | TCN (same architecture class) |
| Random Forest | Inferior to boosting in every way | XGBoost/LightGBM |
| ExtraTrees | Extra randomization adds minimal value | (removed with Random Forest) |
| SVM | O(n²-n³) too slow for OHLCV | XGBoost (10-100x faster) |
| Blending | Wastes 20% data on holdout | Stacking (OOF) + Voting (speed) |
| Informer | Replaced by better transformers | PatchTST/iTransformer |
| TSMixer | Covered by N-BEATS/N-HiTS/DLinear | (MLP baselines sufficient) |

---

### Final 19 Models by Family

#### 1. Boosting (2 models)

**1. XGBoost**
- **Use case:** Stable benchmark, SHAP importance, mature tooling
- **Input:** `feature_matrix` (2D)
- **Outputs:** `p_up`, `E[r]`
- **Effort:** ✅ Already implemented

**2. LightGBM**
- **Use case:** Fastest training, lowest memory, leaf-wise growth
- **Input:** `feature_matrix` (2D)
- **Outputs:** `p_up`, `E[r]`
- **Effort:** ✅ Already implemented

---

#### 2. Neural (RNN) - 1 model

**3. LSTM**
- **Use case:** Long-term dependencies, classic RNN baseline
- **Input:** `window_tensor` (3D)
- **Outputs:** `p_up`, `E[r]`
- **Causal:** Yes (bidirectional=False)
- **Effort:** ✅ Already implemented

---

#### 3. Neural (CNN) - 3 models

**4. TCN (Temporal Convolutional Network)**
- **Use case:** Causal dilations, parallelizable, modern RNN alternative
- **Input:** `window_tensor` (3D)
- **Outputs:** `p_up`, `E[r]`
- **Causal:** Yes (dilated causal convolutions)
- **Effort:** ✅ Already implemented

**5. InceptionTime**
- **Use case:** Multi-scale kernels (3x1, 5x1, 7x1), ensemble-in-architecture
- **Input:** `window_tensor` (3D)
- **Outputs:** `p_up`, `E[r]`
- **Effort:** 3 days

**6. 1D ResNet**
- **Use case:** Residual learning for deep networks, stable baseline
- **Input:** `window_tensor` (3D)
- **Outputs:** `p_up`, `E[r]`
- **Effort:** 2 days

---

#### 4. Transformers (Advanced) - 3 models

**7. PatchTST**
- **Use case:** SOTA long-term forecasting, 21% MSE reduction vs vanilla
- **Architecture:** Patch-based attention (16-token patches, 512-bar context)
- **Input:** `window_tensor` (3D) → patches
- **Outputs:** `p_up`, `E[r]`, `q05`, `q50`, `q95`
- **Causal:** Yes (production-safe)
- **Effort:** 4 days

**8. iTransformer**
- **Use case:** Multivariate correlations (features as tokens, not time steps)
- **Architecture:** Inverted attention over features
- **Input:** `window_tensor` (3D)
- **Outputs:** `p_up`, `E[r]`
- **Effort:** 3 days

**9. TFT (Temporal Fusion Transformer)**
- **Use case:** Interpretable attention + variable selection + multi-horizon
- **Architecture:** Gating + multi-head attention + static/known/unknown features
- **Input:** `window_tensor` (3D) + covariates
- **Outputs:** `p_up`, `E[r]`, `q05`, `q50`, `q95`
- **Effort:** 5 days

---

#### 5. Probabilistic Sequence (2 models)

**10. DeepAR**
- **Use case:** Distribution forecasting, calibrated uncertainty
- **Architecture:** Auto-regressive RNN with probabilistic outputs
- **Input:** `window_tensor` (3D)
- **Outputs:** `q05`, `q50`, `q95`
- **Effort:** 4 days

**11. Quantile RNN**
- **Use case:** Direct q05/q50/q95 for risk bands, position sizing
- **Architecture:** LSTM/GRU with quantile loss
- **Input:** `window_tensor` (3D)
- **Outputs:** `q05`, `q50`, `q95`
- **Effort:** 2 days

---

#### 6. MLP / Linear Baselines (3 models)

**12. N-BEATS**
- **Use case:** Interpretable decomposition (trend + seasonal), M4 winner
- **Architecture:** Stacked blocks with basis expansion
- **Input:** `window_tensor` (3D)
- **Outputs:** `E[r]`, `trend`, `seasonal`
- **Effort:** 1 day

**13. N-HiTS**
- **Use case:** Hierarchical N-BEATS, 2x faster, multi-scale
- **Architecture:** Multi-rate inputs (short/medium/long-term)
- **Input:** `window_tensor` (3D)
- **Outputs:** `E[r]`
- **Effort:** 1 day

**14. DLinear**
- **Use case:** Ultra-fast sanity gate, trend/seasonality baseline
- **Architecture:** Decomposition + two linear layers
- **Input:** `window_tensor` (3D)
- **Outputs:** `E[r]`
- **Effort:** 4 hours

---

#### 7. Foundation Models (2 models)

**15. Chronos-Bolt**
- **Use case:** Zero-shot baseline (51%+ directional accuracy, no training)
- **Architecture:** Pre-trained transformer (Amazon, 200M params)
- **Input:** `window_tensor` (3D)
- **Outputs:** `p_up`
- **Zero-shot:** Yes (no training required)
- **Effort:** 3 days (API wrapper)

**16. TimesFM 2.5**
- **Use case:** Probabilistic zero-shot forecasts (quantiles)
- **Architecture:** Decoder-only transformer (Google, 200M params)
- **Input:** `window_tensor` (3D)
- **Outputs:** `q05`, `q50`, `q95`
- **Zero-shot:** Yes
- **Effort:** 3 days (API wrapper)

---

#### 8. Classical (1 model)

**17. Logistic Regression**
- **Use case:** Fast baseline, meta-learner for stacking ensembles
- **Input:** `feature_matrix` (2D)
- **Outputs:** `p_up`
- **Effort:** ✅ Already implemented

---

#### 9. Ensemble (2 models)

**18. Voting Ensemble**
- **Use case:** Simple weighted averaging, low latency (~6ms for 3 models)
- **Input:** Base model predictions
- **Outputs:** `p_up`, `E[r]`
- **Effort:** ✅ Already implemented

**19. Stacking Ensemble**
- **Use case:** OOF-based meta-learning, best ensemble performance
- **Architecture:** PurgedKFold OOF + Ridge/Elastic Net/LightGBM meta-learner
- **Input:** OOF predictions + optional regime features
- **Outputs:** `p_up`, `E[r]`
- **Effort:** ✅ Already implemented (extend with regime features)

---

### Summary Table

| Family | Models | Count | Primary Use Case |
|--------|--------|-------|------------------|
| **Boosting** | XGBoost, LightGBM | 2 | Fast tabular baselines |
| **Neural (RNN)** | LSTM | 1 | Long-term dependencies |
| **Neural (CNN)** | TCN, InceptionTime, 1D ResNet | 3 | Causal + multi-scale + residual |
| **Transformers** | PatchTST, iTransformer, TFT | 3 | SOTA long-term, multivariate, interpretable |
| **Probabilistic** | DeepAR, Quantile RNN | 2 | Uncertainty quantification, risk bands |
| **MLP/Linear** | N-BEATS, N-HiTS, DLinear | 3 | Interpretable + hierarchical + ultra-fast |
| **Foundation** | Chronos, TimesFM | 2 | Zero-shot baselines |
| **Classical** | Logistic Regression | 1 | Meta-learner |
| **Ensemble** | Voting, Stacking | 2 | Model combination |

**Total: 19 models** (13 existing + 6 new after cutting 10 redundant)

---

## Sample Ensemble Configurations

### Config 1: Fast Baseline (Single Model)

**Use case:** Quick iteration, debugging, baseline comparison

```yaml
base_models:
  - name: lightgbm
    view: feature_matrix
    outputs: [p_up]
    config:
      n_estimators: 500
      learning_rate: 0.05
      max_depth: 7

ensemble:
  enabled: false

inference:
  calibrate: [temperature_scaling]
  uncertainty: null

policy:
  enabled: false
```

**Expected runtime:** 2-5 minutes (train + eval)
**Expected Sharpe:** 0.3-0.6 (baseline)

---

### Config 2: Tabular Duo + Ridge Stacking

**Use case:** Fast ensemble, interpretable meta-learner

```yaml
base_models:
  - name: xgboost
    view: feature_matrix
    outputs: [p_up]
    config:
      n_estimators: 500
      max_depth: 6

  - name: lightgbm
    view: feature_matrix
    outputs: [p_up]
    config:
      n_estimators: 500
      learning_rate: 0.05

ensemble:
  enabled: true
  method: stacking
  meta_learner: ridge
  oof:
    n_folds: 5
    purge_bars: 60
    embargo_bars: 1440

inference:
  calibrate: [temperature_scaling]
  uncertainty: null

policy:
  enabled: false
```

**Expected runtime:** 10-15 minutes
**Expected Sharpe:** 0.5-0.8 (ensemble boost)
**Meta-learner:** Ridge regression on 2 OOF predictions

---

### Config 3: Heterogeneous Trio (Tabular + CNN + Transformer)

**Use case:** Model diversity across families

```yaml
base_models:
  - name: lightgbm
    view: feature_matrix
    outputs: [p_up]

  - name: tcn
    view: window_tensor
    outputs: [E_r]
    config:
      num_channels: [64, 128, 256]
      kernel_size: 3
      dropout: 0.2
      seq_len: 60

  - name: patchtst
    view: window_tensor
    outputs: [p_up, E_r]
    config:
      patch_length: 16
      stride: 8
      context_length: 512
      d_model: 256
      n_heads: 8
      causal: true

ensemble:
  enabled: true
  method: stacking
  meta_learner: ridge
  oof:
    n_folds: 5
    purge_bars: 60
    embargo_bars: 1440

inference:
  calibrate: [temperature_scaling]
  uncertainty: null

policy:
  enabled: false
```

**Expected runtime:** 30-45 minutes
**Expected Sharpe:** 0.7-1.1 (heterogeneous boost)
**Meta-learner:** Ridge on 3 diverse predictions (boosting + CNN + transformer)

---

### Config 4: Probabilistic Trio + CQR Uncertainty

**Use case:** Risk-aware trading with quantile forecasts

```yaml
base_models:
  - name: patchtst
    view: window_tensor
    outputs: [q05, q50, q95]
    config:
      patch_length: 16
      d_model: 256
      causal: true

  - name: deepar
    view: window_tensor
    outputs: [q05, q50, q95]
    config:
      hidden_size: 128
      num_layers: 3

  - name: timesfm
    view: window_tensor
    outputs: [q05, q50, q95]
    zero_shot: true
    config:
      model_id: "google/timesfm-2.5-200m"

ensemble:
  enabled: true
  method: voting
  voting_type: soft  # Average quantiles across models

inference:
  calibrate: null  # Quantiles already calibrated
  uncertainty:
    method: cqr  # Conformalized Quantile Regression
    coverage: 0.90
    holdout_fraction: 0.2

policy:
  enabled: false
```

**Expected runtime:** 40-60 minutes (foundation model is slower)
**Expected Sharpe:** 0.6-0.9 (probabilistic trading)
**Output:** Conformalized 90% prediction intervals for position sizing

---

### Config 5: Full Stack (3 Base + Meta-Learner + Gating + RL)

**Use case:** Production deployment with adaptive position sizing

```yaml
base_models:
  - name: lightgbm
    view: feature_matrix
    outputs: [p_up]

  - name: patchtst
    view: window_tensor
    outputs: [E_r, q05, q50, q95]
    config:
      patch_length: 16
      d_model: 256
      causal: true

  - name: chronos
    view: window_tensor
    outputs: [p_up]
    zero_shot: true

ensemble:
  enabled: true
  method: stacking
  meta_learner: lightgbm_meta  # Nonlinear meta-learner
  oof:
    n_folds: 5
    purge_bars: 60
    embargo_bars: 1440
  gating:
    enabled: true
    method: hmm_regime  # HMM learns regime-conditional weights
    regime_features: [volatility_regime, trend_regime, structure_regime]

inference:
  calibrate: [isotonic, temperature_scaling]  # Double calibration
  uncertainty:
    method: jackknife_plus
    coverage: 0.95

policy:
  enabled: true
  algorithm: sac  # Soft Actor-Critic for continuous position sizing
  inputs: [p_up, E_r, q05, q50, q95, uncertainty, regime_score]
  outputs: [position_size, entry_threshold, exit_threshold]
  config:
    learning_rate: 0.0003
    gamma: 0.99
    tau: 0.005
```

**Expected runtime:** 1-2 hours
**Expected Sharpe:** 0.9-1.4 (full adaptive system)
**Components:**
- 3 diverse base models (boosting + transformer + foundation)
- LightGBM nonlinear meta-learner
- HMM regime-aware gating
- Double calibration (isotonic + temperature scaling)
- Jackknife+ 95% conformal intervals
- SAC RL policy for adaptive position sizing

---

### Config 6: Research Comparison (All Model Families)

**Use case:** Benchmark all 19 models, no ensemble

```yaml
base_models:
  # Boosting
  - {name: xgboost, view: feature_matrix, outputs: [p_up]}
  - {name: lightgbm, view: feature_matrix, outputs: [p_up]}

  # RNN
  - {name: lstm, view: window_tensor, outputs: [p_up], config: {seq_len: 60}}

  # CNN
  - {name: tcn, view: window_tensor, outputs: [E_r], config: {seq_len: 60}}
  - {name: inceptiontime, view: window_tensor, outputs: [p_up], config: {seq_len: 60}}
  - {name: resnet_1d, view: window_tensor, outputs: [p_up], config: {seq_len: 60}}

  # Transformers
  - {name: patchtst, view: window_tensor, outputs: [p_up, q05, q50, q95]}
  - {name: itransformer, view: window_tensor, outputs: [p_up]}
  - {name: tft, view: window_tensor, outputs: [p_up, q05, q50, q95]}

  # Probabilistic
  - {name: deepar, view: window_tensor, outputs: [q05, q50, q95]}
  - {name: quantile_rnn, view: window_tensor, outputs: [q05, q50, q95]}

  # MLP
  - {name: nbeats, view: window_tensor, outputs: [E_r]}
  - {name: nhits, view: window_tensor, outputs: [E_r]}
  - {name: dlinear, view: window_tensor, outputs: [E_r]}

  # Foundation
  - {name: chronos, view: window_tensor, outputs: [p_up], zero_shot: true}
  - {name: timesfm, view: window_tensor, outputs: [q05, q50, q95], zero_shot: true}

  # Classical
  - {name: logistic, view: feature_matrix, outputs: [p_up]}

ensemble:
  enabled: false  # Train all models individually

inference:
  calibrate: [temperature_scaling]  # Apply to all

policy:
  enabled: false

# Output: Comparison table of all 19 models
# Metrics: Sharpe, win rate, max DD, training time, inference latency
```

**Expected runtime:** 3-6 hours (all 19 models)
**Expected output:** Benchmark table ranking models by Sharpe/speed/interpretability

---

## Robustness Requirements

### 1. Data Pipeline Robustness

#### A) Leakage Prevention (CRITICAL)

**Requirements:**
- [ ] HMM regime detection uses only past data (`expanding=False` default)
- [ ] GA optimization confined to train set (70% of data, before splits)
- [ ] Transaction costs included in triple-barrier labels
- [ ] All feature engineering uses only past data (no lookahead)
- [ ] Scaling fit only on train set, transform all splits
- [ ] Purge (60 bars) and embargo (1440 bars) enforced in all CV folds

**Test Coverage:**
- Unit tests for each stage verify no future data access
- Integration test: Generate features at time T, verify no data from T+1 onwards used
- Regression tests: Lock down leakage fixes with assertions

---

#### B) Deterministic Resampling

**Requirements:**
- [ ] OHLCV resampling (1min → 5min) is deterministic (fixed seeds)
- [ ] Multi-timeframe alignment is reproducible
- [ ] Same resampling code used in training AND inference
- [ ] Gap handling (missing bars) is consistent

**Implementation:**
```python
# src/phase1/stages/clean/resample.py
def resample_ohlcv(df, freq="5min", seed=42):
    """Deterministic OHLCV resampling."""
    np.random.seed(seed)  # For any gap filling
    # ... resampling logic
    return df_resampled

# MUST use same function in:
# - Training: scripts/run_pipeline.py
# - Inference: src/inference/pipeline.py
```

**Test Coverage:**
- Run resampling 100 times with same seed → assert identical output
- Verify training and inference use same function (import check)

---

#### C) Schema Validation

**Requirements:**
- [ ] Every pipeline stage validates input schema (column names, dtypes, shapes)
- [ ] Feature schema is versioned and stored with artifacts
- [ ] Inference validates incoming data matches training schema
- [ ] Missing columns raise clear errors (not silent failures)

**Implementation:**
```python
# src/phase1/validation/schema.py
@dataclass
class FeatureSchema:
    columns: List[str]
    dtypes: Dict[str, str]
    shape: Tuple[int, int]
    hash: str  # SHA256 of sorted column names

def validate_schema(df: pd.DataFrame, expected: FeatureSchema):
    if set(df.columns) != set(expected.columns):
        raise SchemaError(f"Missing columns: {set(expected.columns) - set(df.columns)}")
    # ... dtype, shape validation
```

**Test Coverage:**
- Test schema mismatch detection (missing column, wrong dtype)
- Test schema hash collisions

---

### 2. Model Training Robustness

#### A) Reproducibility

**Requirements:**
- [ ] All models seed RNGs (NumPy, PyTorch, TensorFlow, XGBoost)
- [ ] GPU operations are deterministic (set `CUDA_DETERMINISTIC=True`)
- [ ] Training produces identical results across runs (same data + seed)

**Implementation:**
```python
# src/models/trainer.py
def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
```

**Test Coverage:**
- Train model 10 times with same seed → assert identical metrics
- Test GPU determinism (if available)

---

#### B) Hyperparameter Validation

**Requirements:**
- [ ] All hyperparameters validated at config load (not runtime)
- [ ] Invalid configs raise errors before training starts
- [ ] Sensible defaults provided for all optional params

**Implementation:**
```python
# src/models/config/validation.py
def validate_model_config(config: Dict[str, Any], model_name: str):
    schema = MODEL_CONFIG_SCHEMAS[model_name]
    for param, constraints in schema.items():
        if param not in config:
            config[param] = constraints["default"]
        if not constraints["validator"](config[param]):
            raise ConfigError(f"{param}={config[param]} violates {constraints}")
    return config
```

**Test Coverage:**
- Test invalid configs raise errors (negative learning rate, etc.)
- Test defaults are applied correctly

---

#### C) Early Stopping & Checkpointing

**Requirements:**
- [ ] All models support early stopping (patience=10 epochs default)
- [ ] Best model checkpoint saved (not last epoch)
- [ ] Training can resume from checkpoint (if interrupted)

**Implementation:**
```python
# src/models/trainer.py
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # Don't stop
        else:
            self.counter += 1
            return self.counter >= self.patience  # Stop if patience exceeded
```

**Test Coverage:**
- Test early stopping triggers correctly
- Test checkpoint restoration produces identical predictions

---

### 3. Ensemble Robustness

#### A) OOF Prediction Integrity

**Requirements:**
- [ ] OOF predictions cover 100% of training samples (no gaps)
- [ ] Each sample predicted exactly once by out-of-fold model
- [ ] OOF predictions never leak fold information

**Validation:**
```python
# src/cross_validation/oof_validation.py
def validate_oof_coverage(oof_preds: np.ndarray, y_true: np.ndarray):
    assert len(oof_preds) == len(y_true), "OOF length mismatch"
    assert not np.any(np.isnan(oof_preds)), "OOF has NaN predictions"
    # For sequence models, first seq_len samples may be NaN (no lookback)
    # ... handle edge cases
```

**Test Coverage:**
- Test OOF coverage for all model types (tabular + sequence)
- Test fold assignments don't leak

---

#### B) Meta-Learner Regularization

**Requirements:**
- [ ] Meta-learners use regularization to prevent overfitting to base predictions
- [ ] Ridge/Elastic Net: alpha >= 0.1 (default: 1.0)
- [ ] LightGBM meta: max_depth <= 3, n_estimators <= 100

**Implementation:**
```python
# src/models/ensemble/stacking.py
DEFAULT_META_CONFIGS = {
    "ridge": {"alpha": 1.0},
    "elastic_net": {"alpha": 1.0, "l1_ratio": 0.5},
    "lightgbm_meta": {"max_depth": 3, "n_estimators": 50, "learning_rate": 0.05},
}
```

**Test Coverage:**
- Test meta-learner doesn't overfit to OOF predictions
- Compare meta-learner train/val performance (gap should be small)

---

#### C) Heterogeneous Ensemble Compatibility

**Requirements:**
- [ ] Validate all base models have compatible output shapes
- [ ] Tabular + sequence models can be mixed (stacking handles different input views)
- [ ] Error raised if outputs don't align (e.g., mixing `p_up` and `q05/q50/q95`)

**Implementation:**
```python
# src/models/ensemble/validator.py
def validate_ensemble_compatibility(base_models: List[BaseModel]):
    output_types = [model.output_contract for model in base_models]
    if len(set(output_types)) > 1:
        raise EnsembleCompatibilityError(
            f"Base models have incompatible outputs: {output_types}"
        )
```

**Test Coverage:**
- Test tabular + sequence mixing works
- Test incompatible output error handling

---

### 4. Inference Robustness

#### A) Bundle Integrity

**Requirements:**
- [ ] ModelBundle includes all artifacts (preproc, schema, weights, metrics)
- [ ] Bundle checksum validates integrity on load
- [ ] Missing artifacts raise errors (not silent failures)

**Implementation:**
```python
# src/inference/bundle.py
@dataclass
class ModelBundle:
    model: BaseModel
    scaler: StandardScaler
    feature_schema: FeatureSchema
    calibrator: Optional[Calibrator]
    metadata: Dict[str, Any]
    checksum: str  # SHA256 of all artifacts

    def save(self, path: Path):
        # Save all artifacts + compute checksum
        checksum = compute_bundle_checksum(path)
        metadata["checksum"] = checksum

    @classmethod
    def load(cls, path: Path):
        # Load + verify checksum
        expected_checksum = metadata["checksum"]
        actual_checksum = compute_bundle_checksum(path)
        if expected_checksum != actual_checksum:
            raise BundleCorruptionError("Checksum mismatch")
```

**Test Coverage:**
- Test bundle save/load round-trip
- Test checksum validation catches corruption

---

#### B) Calibration Validation

**Requirements:**
- [ ] Calibrated probabilities are valid (0 ≤ p ≤ 1)
- [ ] Reliability diagrams show calibration quality
- [ ] Calibration fit only on validation set (not train or test)

**Validation:**
```python
# src/models/calibration/validation.py
def validate_calibration(y_true: np.ndarray, y_prob: np.ndarray):
    assert np.all((y_prob >= 0) & (y_prob <= 1)), "Probabilities outside [0,1]"
    # Compute reliability diagram (bin probabilities, compare to empirical)
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(y_prob, bins)
    reliability = []
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if mask.sum() > 0:
            reliability.append((bins[i], y_true[mask].mean()))
    return reliability
```

**Test Coverage:**
- Test calibration improves reliability diagrams
- Test calibration fit only on validation set

---

#### C) Conformal Prediction Coverage

**Requirements:**
- [ ] Conformal intervals achieve target coverage (e.g., 90%)
- [ ] Coverage validated on holdout set (not calibration set)
- [ ] Adaptive intervals (wider when uncertain, tighter when confident)

**Validation:**
```python
# src/models/calibration/conformal.py
def validate_coverage(y_true, intervals, target_coverage=0.90):
    lower, upper = intervals
    coverage = ((y_true >= lower) & (y_true <= upper)).mean()
    assert abs(coverage - target_coverage) < 0.05, \
        f"Coverage {coverage:.2f} too far from target {target_coverage}"
```

**Test Coverage:**
- Test CQR achieves 90% coverage
- Test intervals adapt to uncertainty

---

### 5. Production Deployment Robustness

#### A) Drift Detection

**Requirements:**
- [ ] Monitor feature distributions (KL divergence, Wasserstein distance)
- [ ] Alert when drift exceeds threshold (e.g., KL > 0.5)
- [ ] Log drift metrics for post-mortem analysis

**Implementation:**
```python
# src/monitoring/drift_detector.py
class DriftDetector:
    def __init__(self, reference_data: np.ndarray, threshold=0.5):
        self.reference_mean = reference_data.mean(axis=0)
        self.reference_std = reference_data.std(axis=0)
        self.threshold = threshold

    def detect(self, new_data: np.ndarray):
        kl_div = compute_kl_divergence(self.reference_data, new_data)
        if kl_div > self.threshold:
            raise DriftAlert(f"KL divergence {kl_div:.2f} exceeds threshold")
```

**Test Coverage:**
- Test drift detection on synthetic drifted data
- Test alert mechanism

---

#### B) Model Versioning

**Requirements:**
- [ ] Every model has a semantic version (v1.2.3)
- [ ] Breaking changes increment major version
- [ ] Inference API is versioned (v1, v2 routes)

**Implementation:**
```python
# src/inference/versioning.py
@dataclass
class ModelVersion:
    major: int  # Breaking changes
    minor: int  # New features (backward compatible)
    patch: int  # Bug fixes

    def __str__(self):
        return f"v{self.major}.{self.minor}.{self.patch}"
```

**Test Coverage:**
- Test version comparison (v1.2.3 < v2.0.0)
- Test API routes (/v1/predict, /v2/predict)

---

#### C) Graceful Degradation

**Requirements:**
- [ ] If primary model fails, fall back to simpler model (e.g., LightGBM)
- [ ] Log failures for debugging
- [ ] Never return NaN predictions (use fallback)

**Implementation:**
```python
# src/inference/pipeline.py
class InferencePipeline:
    def __init__(self, primary_model, fallback_model):
        self.primary = primary_model
        self.fallback = fallback_model

    def predict(self, X):
        try:
            return self.primary.predict(X)
        except Exception as e:
            logger.error(f"Primary model failed: {e}. Using fallback.")
            return self.fallback.predict(X)
```

**Test Coverage:**
- Test fallback triggers on primary failure
- Test fallback predictions are valid

---

## Dynamic Registry Architecture

### Philosophy: Configuration > Code

```yaml
# Example: config.yaml
base_models:
  - {name: lightgbm, view: feature_matrix, outputs: [p_up]}
  - {name: patchtst, view: window_tensor, outputs: [E_r, q05, q50, q95]}

ensemble:
  method: stacking
  meta_learner: ridge

inference:
  calibrate: [temperature_scaling]
  uncertainty: {method: cqr, coverage: 0.90}
```

### Four Registry Types

#### 1. Model Registry

**Purpose:** All trainable models (tabular, sequence, foundation)

**Entry Schema:**
```python
@dataclass
class ModelSpec:
    name: str                    # "lightgbm", "patchtst"
    family: str                  # "boosting", "transformer"
    input_view: str              # "feature_matrix" or "window_tensor"
    outputs: List[str]           # ["p_up", "E_r", "q05", "q50", "q95"]
    is_causal: bool              # Production safety
    zero_shot: bool              # Foundation models
```

**Registration:**
```python
@register_model(
    name="patchtst",
    family="transformer",
    input_view="window_tensor",
    outputs=["p_up", "E_r", "q05", "q50", "q95"],
    is_causal=True,
    zero_shot=False
)
class PatchTSTModel(BaseModel):
    ...
```

---

#### 2. Ensemble Registry

**Purpose:** Stacking, voting, gating strategies

**Entry Schema:**
```python
@dataclass
class EnsembleSpec:
    name: str                    # "ridge_stacking"
    method: str                  # "stacking", "voting", "gating"
    meta_learner: Optional[str]  # "ridge", "elastic_net", "lightgbm_meta"
    requires_oof: bool           # True for stacking
    supports_heterogeneous: bool # Can mix tabular + sequence?
```

---

#### 3. Inference Registry

**Purpose:** Calibration, conformal prediction, gating

**Entry Schema:**
```python
@dataclass
class InferenceSpec:
    name: str                    # "temperature_scaling", "cqr"
    category: str                # "calibration", "conformal", "gating"
    requires_holdout: bool       # True for conformal methods
```

---

#### 4. Policy Registry (Optional RL)

**Purpose:** Adaptive decision-making

**Entry Schema:**
```python
@dataclass
class PolicySpec:
    name: str                    # "sac", "ppo"
    algorithm: str               # "sac", "td3", "ppo", "dqn"
    action_space: str            # "continuous", "discrete"
```

---

## Contracts & Artifacts

### Input View Contract

| View | Shape | Models |
|------|-------|--------|
| `feature_matrix` | `(n_samples, n_features)` | XGBoost, LightGBM, Logistic |
| `window_tensor` | `(n_samples, seq_len, n_features)` | LSTM, TCN, InceptionTime, 1D ResNet, PatchTST, iTransformer, TFT, DeepAR, Quantile RNN, N-BEATS, N-HiTS, DLinear, Chronos, TimesFM |

### Output Contract

| Output | Type | Description |
|--------|------|-------------|
| `p_up` | `float [0,1]` | Probability of up move |
| `E[r]` | `float` | Expected return |
| `q05` | `float` | 5th percentile |
| `q50` | `float` | Median |
| `q95` | `float` | 95th percentile |
| `regime_score` | `float [0,1]` | Regime confidence |

### Artifact Contract

```python
{
    "preproc": "path/to/scaler.pkl",
    "schema": "path/to/feature_schema.json",
    "weights": "path/to/model.pth",
    "metrics": "path/to/metrics.json",
    "inference_signature": {
        "inputs": ["feature_matrix"],
        "outputs": ["p_up", "E_r"],
        "version": "v1.2.3"
    }
}
```

For ensembles, add:
```python
{
    "oof_preds": "path/to/oof_predictions.parquet",
    "meta_model": "path/to/meta_learner.pkl",
    "calibrator": "path/to/calibrator.pkl",
    "gating_model": "path/to/gate.pkl"  # Optional
}
```

---

## Critical Data Pipeline Bugs

### 🔴 Bug 1: HMM Lookahead Bias

**File:** `src/phase1/stages/regime/hmm.py:329-354`

**Problem:** HMM trains on entire dataset including future data when `expanding=True`.

**Fix:**
```python
expanding = kwargs.get("expanding", False)  # Change default to False
```

**Estimated Effort:** 2 days

---

### 🔴 Bug 2: GA Test Data Leakage

**File:** `src/phase1/stages/ga_optimize/optuna_optimizer.py`

**Problem:** Optuna optimization uses full dataset before train/val/test splits.

**Fix:**
```python
train_end_idx = int(0.7 * len(df))
train_df = df.iloc[:train_end_idx]
study.optimize(objective, train_df)  # Only train portion
```

**Estimated Effort:** 2 days

---

### 🔴 Bug 3: No Transaction Costs in Labels

**File:** `src/phase1/stages/labeling/triple_barrier.py`

**Problem:** Triple-barrier labels assume zero transaction costs.

**Fix:**
```python
cost_in_atr = (cost_ticks * tick_value) / atr
upper_barrier = entry_price + (k_up - cost_in_atr) * atr
lower_barrier = entry_price - (k_down + cost_in_atr) * atr
```

**Estimated Effort:** 2 days

---

## Implementation Roadmap

### Phase 1: Fix Data Pipeline Bugs (Week 1-2) ⚠️ CRITICAL

**Goal:** Zero-leakage pipeline

**Tasks:**
1. Fix HMM lookahead bias (2 days)
2. Fix GA test data leakage (2 days)
3. Fix transaction costs in labels (2 days)
4. Regression test suite (2 days)

**Success Criteria:**
- ✅ All 3 bugs fixed
- ✅ All 2,060 tests pass

**Priority:** **CRITICAL** (blocks all production use)

---

### Phase 2: Build Dynamic Registry System (Week 3-4)

**Goal:** Config-driven model selection

**Tasks:**
1. Implement ModelRegistry + ModelSpec (3 days)
2. Implement EnsembleRegistry + EnsembleSpec (2 days)
3. Implement InferenceRegistry + InferenceSpec (2 days)
4. Implement PolicyRegistry + PolicySpec (2 days)
5. Update all 13 existing models to use @register_model (1 day)
6. Add YAML config loading (1 day)

**New Files:**
```
src/registry/
├── model_registry.py
├── ensemble_registry.py
├── inference_registry.py
├── policy_registry.py
└── contracts.py
```

**Success Criteria:**
- ✅ All 13 models registered
- ✅ Config-driven training: `python scripts/train_model.py --config config.yaml`

**Estimated Effort:** 2 weeks

---

### Phase 3: Add 6 New Models (Week 5-7)

**Goal:** 19 total models

#### Week 5: Foundation + MLP (Quick Wins)
1. Chronos-Bolt (3 days)
2. TimesFM (3 days)
3. DLinear (4 hours)

#### Week 6: Advanced Transformers
4. PatchTST (4 days)
5. iTransformer (3 days)

#### Week 7: CNN + Probabilistic
6. InceptionTime (3 days)
7. 1D ResNet (2 days)
8. TFT (5 days - complex)
9. DeepAR (4 days)
10. Quantile RNN (2 days)
11. N-BEATS (1 day)
12. N-HiTS (1 day)

**Success Criteria:**
- ✅ 6 new models registered (total: 19)
- ✅ All trainable via config
- ✅ Benchmarks documented

**Estimated Effort:** 3 weeks

---

### Phase 4: Add Inference Layer (Week 8-9)

**Goal:** Calibration + conformal + gating

**Components:**
1. Temperature Scaling (1 day)
2. Isotonic Regression (1 day)
3. CQR (2 days)
4. Split Conformal (2 days)
5. HMM Regime Gating (3 days)
6. Elastic Net meta-learner (1 day)
7. LightGBM meta-learner (1 day)

**Success Criteria:**
- ✅ 7 inference components registered
- ✅ Config-driven: `inference: {calibrate: [temperature_scaling], uncertainty: cqr}`

**Estimated Effort:** 2 weeks

---

### Phase 5: Add Optional RL Policy Layer (Week 10-11)

**Goal:** Adaptive decision-making

**Algorithms:**
1. SAC (4 days)
2. TD3 (3 days)
3. PPO (4 days)

**Success Criteria:**
- ✅ 3 RL policies registered
- ✅ Optional via config (default: `enabled: false`)

**Estimated Effort:** 2 weeks

---

### Phase 6: Production Infrastructure (Week 12-14)

**Goal:** Automated train→bundle→deploy

**Tasks:**
1. Auto-bundle generation (3 days)
2. Test set evaluation script (2 days)
3. Drift detection daemon (4 days)
4. CI/CD pipeline (3 days)

**Success Criteria:**
- ✅ One-command: `python scripts/train_model.py --config config.yaml --create-bundle`
- ✅ Drift monitoring running

**Estimated Effort:** 2-3 weeks

---

## Testing Strategy

### Phase 1: Bug Fix Tests
- [ ] HMM: `expanding=False` uses only past data
- [ ] GA: Optimization confined to train set
- [ ] Transaction costs: Barrier adjustment math validated

### Phase 2: Registry Tests
- [ ] ModelRegistry: Create models from config
- [ ] EnsembleRegistry: Validate compatibility checks
- [ ] InferenceRegistry: Calibration correctness

### Phase 3: New Model Tests
- [ ] Each new model: `test_fit()`, `test_predict()`, `test_save_load()`
- [ ] PatchTST: Causal masking verified
- [ ] Foundation models: Zero-shot prediction (no training)

### Phase 4: Inference Tests
- [ ] Calibration: Reliability diagrams match
- [ ] Conformal: Coverage matches target (90%)
- [ ] Gating: Regime weights sum to 1.0

### Phase 5: RL Tests
- [ ] SAC/TD3/PPO: Policy converges on simple envs
- [ ] Integration: RL consumes inference outputs correctly

---

## Success Metrics

| Phase | Metric | Target |
|-------|--------|--------|
| Phase 1 | Zero leakage bugs | All 3 bugs fixed |
| Phase 2 | Registry coverage | All 13 models registered |
| Phase 3 | Model count | 19 models |
| Phase 4 | Inference components | 7 calibration/conformal/gating methods |
| Phase 5 | RL policies | 3 algorithms (optional) |
| Phase 6 | Deployment time | <5 min train→bundle→deploy |

**Final System Capabilities:**

- ✅ 19 production-safe models (no redundancy)
- ✅ Config-driven selection (no code changes)
- ✅ 7+ inference components (calibration, conformal, gating)
- ✅ Optional RL policy layer (3 algorithms)
- ✅ One-command deploy
- ✅ Automated drift detection
- ✅ Zero leakage (all 3 bugs fixed)
- ✅ Robust artifacts (checksums, versioning, schema validation)
