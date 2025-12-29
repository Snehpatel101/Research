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

**Short answer:** One canonical OHLCV dataset → Multi-timeframe features → Model-specific feature sets → Model-specific input views

```
Raw 1min OHLCV (MES)
       ↓
[ Clean & Resample → 5min base timeframe ]
       ↓
[ Multi-Timeframe Feature Engineering ]
       ├─ 5min features (150+ indicators)
       ├─ 15min features (resampled + indicators)
       ├─ 1h features (resampled + indicators)
       ├─ 4h features (resampled + indicators)
       └─ Daily features (resampled + indicators)
       ↓
[ Feature Alignment ] (all timeframes aligned to 5min grid)
       ↓
[ Model-Specific Feature Selection ]
       ├─ Tabular models → momentum + microstructure + regime (80 features)
       ├─ Sequence models → OHLCV + volume + wavelets (20 features)
       ├─ Transformers → raw OHLCV + volume (5 features, learn their own features)
       └─ Foundation models → normalized OHLCV only (4 features)
       ↓
[ Model-Specific Input Views ]
       ├─ Tabular: (n_samples, n_features) 2D matrix
       └─ Sequence: (n_samples, seq_len, n_features) 3D tensor
       ↓
[ Training ] → Different models, same experimental controls
```

---

### Multi-Timeframe (MTF) Feature Engineering

**Base timeframe:** 5 minutes (after resampling from 1min raw data)

**Upsampling mechanics:**

1. **Resample OHLCV to higher timeframes:**
   ```python
   # src/phase1/stages/mtf/
   df_15m = resample_ohlcv(df_5m, freq="15min")  # 3 bars → 1 bar
   df_1h = resample_ohlcv(df_5m, freq="1h")      # 12 bars → 1 bar
   df_4h = resample_ohlcv(df_5m, freq="4h")      # 48 bars → 1 bar
   df_daily = resample_ohlcv(df_5m, freq="1D")   # 288 bars → 1 bar
   ```

2. **Compute indicators at each timeframe:**
   ```python
   # Example: RSI computed on 15min, 1h, daily
   df_5m["rsi_15m"] = compute_rsi(df_15m, period=14)    # Slow-moving
   df_5m["rsi_1h"] = compute_rsi(df_1h, period=14)      # Trend signal
   df_5m["rsi_daily"] = compute_rsi(df_daily, period=14)  # Regime signal
   ```

3. **Align all timeframes to base 5min grid:**
   ```python
   # Forward-fill higher timeframe features to 5min (no lookahead)
   df_5m = df_5m.merge(df_15m[["rsi_15m"]], left_index=True, right_index=True, how="left")
   df_5m["rsi_15m"].ffill(inplace=True)  # Use most recent 15min value
   ```

**Result:** Every 5min bar has features from 5 timeframes (5m, 15m, 1h, 4h, daily), all aligned and leakage-safe.

**Why MTF?**
- **5min features:** Microstructure, noise
- **15min-1h features:** Intraday trends
- **4h-daily features:** Regime context (volatility, trend strength)

---

### Model-Specific Feature Sets

**Different models have different feature requirements.** The pipeline produces 150+ features, but each model uses a subset:

#### Tabular Models (XGBoost, LightGBM, Logistic)

**Features used:** ~80 features
- ✅ Momentum indicators (RSI, MACD, Stochastic) across 5 timeframes
- ✅ Microstructure (bid-ask spread, order flow imbalance, tick direction)
- ✅ Volatility (ATR, Parkinson, Garman-Klass) across 5 timeframes
- ✅ Volume features (VWAP, OBV, volume ratios)
- ✅ Regime features (volatility regime, trend regime, composite regime)
- ✅ Interaction features (RSI * volatility_regime, MACD * trend_regime)

**Why these features?**
- Tabular models excel at capturing **monotonic relationships** and **sparse interactions**
- Engineered features provide **domain knowledge shortcuts** (e.g., RSI overbought/oversold)
- Multi-timeframe features give **regime context**

---

#### Sequence Models (LSTM, TCN, InceptionTime, 1D ResNet)

**Features used:** ~20 features
- ✅ Raw OHLCV (5 features: open, high, low, close, volume)
- ✅ Returns (log returns, realized volatility)
- ✅ Wavelet decompositions (4 levels: D1, D2, D3, A3)
- ✅ Basic volume features (volume MA, volume std)

**Why fewer features?**
- Sequence models **learn temporal patterns automatically** (don't need pre-computed indicators)
- Wavelets help with **multi-scale patterns** (short-term vs long-term)
- Feeding 150 features to RNNs causes **gradient vanishing** and **overfitting**

**Input shape:** `(n_samples, seq_len, 20)` where `seq_len=60` (5 hours of 5min bars)

---

#### Transformers (PatchTST, iTransformer, TFT)

**Features used:** ~5-10 features (raw OHLCV + volume + regime embeddings)
- ✅ Raw OHLCV (4 features)
- ✅ Volume
- ✅ Regime embeddings (volatility state, trend state)

**Why so few features?**
- Transformers **learn their own feature representations** via attention
- Patching (PatchTST) and inverted attention (iTransformer) need **raw inputs**
- Over-engineering features **limits transformer's ability to discover patterns**

**Input shape:**
- PatchTST: `(n_samples, context_length=512, 5)` → patches of 16 bars
- iTransformer: `(n_samples, seq_len=60, 5)`
- TFT: `(n_samples, seq_len=60, 10)` with static/known/unknown feature splits

---

#### Foundation Models (Chronos, TimesFM)

**Features used:** Normalized OHLCV only (4 features)
- ✅ Open, High, Low, Close (z-score normalized per lookback window)

**Why only OHLCV?**
- Foundation models are **pre-trained on pure OHLCV patterns** (no custom features)
- Adding engineered features **breaks their pre-trained representations**
- Zero-shot inference requires **matching their training distribution**

**Input shape:** `(n_samples, context_length=512, 4)`

---

### Feature Selection in Code

**Where feature selection happens:** `src/phase1/stages/datasets/`

```python
# src/phase1/stages/datasets/model_views.py

TABULAR_FEATURES = [
    # Momentum (5 timeframes × 5 indicators)
    "rsi_5m", "rsi_15m", "rsi_1h", "rsi_4h", "rsi_daily",
    "macd_5m", "macd_15m", ...,

    # Microstructure
    "bid_ask_spread", "order_flow_imbalance", "tick_direction",

    # Volatility
    "atr_5m", "atr_15m", "atr_1h", "atr_4h", "atr_daily",

    # Volume
    "vwap", "obv", "volume_ratio_5m", "volume_ratio_15m",

    # Regime
    "volatility_regime", "trend_regime", "composite_regime",
]

SEQUENCE_FEATURES = [
    "open", "high", "low", "close", "volume",
    "log_returns", "realized_vol",
    "wavelet_D1", "wavelet_D2", "wavelet_D3", "wavelet_A3",
    "volume_ma_20", "volume_std_20",
]

TRANSFORMER_FEATURES = [
    "open", "high", "low", "close", "volume",
    "volatility_regime_embedding", "trend_regime_embedding",
]

FOUNDATION_FEATURES = [
    "open", "high", "low", "close",  # Normalized only
]

def get_feature_view(df: pd.DataFrame, model_family: str) -> pd.DataFrame:
    if model_family in ["boosting", "classical"]:
        return df[TABULAR_FEATURES]
    elif model_family in ["rnn", "cnn"]:
        return df[SEQUENCE_FEATURES]
    elif model_family == "transformer":
        return df[TRANSFORMER_FEATURES]
    elif model_family == "foundation":
        return df[FOUNDATION_FEATURES]
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

| Model Family | Feature Count | Input Shape | Why This Design? |
|--------------|---------------|-------------|------------------|
| **Tabular** | ~80 features | `(n, 80)` | Needs engineered features + domain knowledge |
| **Sequence** | ~20 features | `(n, 60, 20)` | Learns temporal patterns, wavelets for multi-scale |
| **Transformer** | ~5 features | `(n, 512, 5)` | Learns own features via attention |
| **Foundation** | 4 features | `(n, 512, 4)` | Pre-trained on raw OHLCV only |

**All models:**
- ✅ Same base 5min dataset (leakage-safe)
- ✅ Same multi-timeframe features available (5m/15m/1h/4h/daily)
- ✅ Same train/val/test splits (70/15/15)
- ✅ Same purge (60) and embargo (1440)
- ✅ Same transaction costs in labels

**Different per model:**
- ❌ Feature subset (tabular uses 80, transformers use 5)
- ❌ Input shape (2D matrix vs 3D tensor)
- ❌ Feature engineering philosophy (hand-crafted vs learned)

**Result:** Fair comparison under identical experimental controls, but each model gets the **feature representation it needs** to perform best.

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
