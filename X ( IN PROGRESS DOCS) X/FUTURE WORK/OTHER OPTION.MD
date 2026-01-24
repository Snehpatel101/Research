# ML Factory: Build-a-Bear for Trading Ensembles

**Generated:** 2025-01-23  
**Status:** Architecture Redesign  
**Philosophy:** Simple choices, powerful defaults, impossible to misconfigure

---

## Vision

**The Promise:** Users configure *what* they want with simple, high-level choices. The factory handles *how* — compatibility, shape alignment, leakage prevention, calibration, and clean stage handoffs.

**The Experience:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    BUILD-A-BEAR INTERFACE                       │
├─────────────────────────────────────────────────────────────────┤
│  1. Pick a Preset     →  "Balanced" / "Speed" / "Max Alpha"    │
│  2. Choose Base Models →  [CatBoost] [TCN] [PatchTST]          │
│  3. Pick Ensemble Style →  Stack / Blend / MoE / Adaptive      │
│  4. Tune Knobs (optional) → Risk, Horizons, Calibration        │
│  5. Run                 →  ml-factory run --config my_bear.yaml │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FACTORY HANDLES AUTOMATICALLY                │
├─────────────────────────────────────────────────────────────────┤
│  • Feature shape alignment (2D↔3D↔4D)                          │
│  • OOF-safe training (no leakage, purged folds)                │
│  • Model compatibility checks                                   │
│  • Calibration pipeline                                         │
│  • Regime detection (if needed)                                 │
│  • Optuna optimization (5 dimensions)                           │
│  • Bundle packaging for deployment                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                  │
├─────────────────────────────────────────────────────────────────┤
│  experiments/{run_id}/                                          │
│  ├── deployment/inference_bundle.tar.gz  ← Production-ready    │
│  ├── metrics/backtest_report.json        ← Realistic P&L       │
│  └── plots/                              ← Visual diagnostics   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

1. [The Lego Blocks](#1-the-lego-blocks) — Every swappable component
2. [Presets & Knobs](#2-presets--knobs) — Simple UX layer
3. [Ensemble Styles](#3-ensemble-styles) — Static to Dynamic spectrum
4. [Architecture Layers](#4-architecture-layers) — How pieces connect
5. [Current State & Gap Analysis](#5-current-state--gap-analysis)
6. [Recommended File Structure](#6-recommended-file-structure)
7. [Implementation Roadmap](#7-implementation-roadmap)

---

## 1. The Lego Blocks

Every component users can swap. The factory validates compatibility automatically.

### 1.1 Base Learners (The "Stuffing")

| Block | Input Shape | Speed | Best For | Status |
|-------|-------------|-------|----------|--------|
| **Boosting Family** |
| `xgboost` | 2D (N, F) | ⚡⚡⚡ | Tabular, fast iteration | ✅ |
| `lightgbm` | 2D (N, F) | ⚡⚡⚡ | Tabular, large datasets | ✅ |
| `catboost` | 2D (N, F) | ⚡⚡ | Tabular, categorical | ✅ |
| **RNN Family** |
| `lstm` | 3D (N, T, F) | ⚡ | Sequential patterns | ✅ |
| `gru` | 3D (N, T, F) | ⚡⚡ | Sequential, faster | ✅ |
| **CNN Family** |
| `tcn` | 3D (N, T, F) | ⚡⚡ | Local patterns | ✅ |
| `inception_time` | 3D (N, T, F) | ⚡ | Multi-scale patterns | ❌ |
| `resnet_1d` | 3D (N, T, F) | ⚡ | Deep residual | ❌ |
| **Transformer Family** |
| `patchtst` | 4D (N, TF, T, C) | 🐢 | Multi-resolution | ❌ |
| `itransformer` | 4D (N, TF, T, C) | 🐢 | Channel-independent | ❌ |
| `tft` | 4D (N, TF, T, C) | 🐢 | Interpretable attention | ❌ |
| **Decomposition Family** |
| `nbeats` | 4D (N, TF, T, C) | ⚡ | Trend/seasonality | ❌ |

**Compatibility Rules (factory enforces):**
```yaml
shape_requirements:
  2D_models: [xgboost, lightgbm, catboost]
  3D_models: [lstm, gru, tcn, inception_time, resnet_1d]
  4D_models: [patchtst, itransformer, tft, nbeats]

# Factory auto-routes to correct canonical store:
# 2D → data/canonical/engineered/
# 3D → data/canonical/engineered/ + windowing
# 4D → data/canonical/raw_mtf/ + Multi-Res adapter
```

### 1.2 Feature Sets (The "Accessories")

| Block | Features | Used By | Status |
|-------|----------|---------|--------|
| `momentum` | RSI, MACD, ROC, Stoch, Williams %R | All 2D/3D | ✅ |
| `volatility` | ATR, BB, Keltner, Parkinson, GK | All 2D/3D | ✅ |
| `volume` | OBV, VWAP, AD, MFI, Volume MA | All 2D/3D | ✅ |
| `microstructure` | Spread, depth imbalance, trade flow | Boosting | ✅ |
| `wavelets` | DWT coefficients, multi-scale energy | Neural | ⚠️ Partial |
| `entropy` | Sample entropy, permutation entropy | Transformer | ❌ |
| `mtf_indicators` | Same indicators across 9 timeframes | All 2D/3D | ✅ |
| `raw_ohlcv` | Pure OHLCV (no engineering) | 4D models | ❌ Store missing |

**Feature Set Contracts:**
```yaml
base_feature_sets:
  boosting:
    families: [momentum, volatility, volume, microstructure, mtf_indicators]
    base_count: 180
    optuna_can_select: true  # Reduce from 180

  neural_sequence:
    families: [momentum, volatility, volume, wavelets]
    base_count: 120
    optuna_can_select: true

  transformer_raw:
    families: [raw_ohlcv]
    base_count: 4  # O, H, L, C per timeframe
    optuna_can_select: false  # Must use all
    requires_4d_adapter: true

  transformer_engineered:
    families: [momentum, volatility, wavelets, entropy]
    base_count: 100
    optuna_can_select: true
```

### 1.3 Ensemble Methods (The "Outfit")

| Block | Type | Adapts? | Best For | Status |
|-------|------|---------|----------|--------|
| **Static** |
| `voting_hard` | Voting | No | Classification, interpretable | ✅ |
| `voting_soft` | Voting | No | Calibrated probabilities | ✅ |
| `blending` | Weighted Avg | No | Simple combination | ✅ |
| `stacking` | Meta-learner | No | Heterogeneous models | ✅ |
| **Dynamic** |
| `moe` | Gating Network | Per-sample | Regime-aware routing | ❌ |
| `adaptive` | Decaying Weights | Over time | Non-stationary data | ❌ |
| `regime_routed` | Hard Switching | Per-regime | Explicit regime logic | ❌ |
| `uncertainty_weighted` | Confidence | Per-sample | Risk-aware | ❌ |
| `cascaded` | Early Exit | Per-sample | Fast inference | ❌ |

### 1.4 Meta-Learners (For Stacking/MoE)

| Block | Type | Regularized | Status |
|-------|------|-------------|--------|
| `ridge` | Linear | Yes | ✅ |
| `lasso` | Linear | Yes (sparse) | ✅ |
| `elastic_net` | Linear | Yes | ✅ |
| `logistic` | Linear | Yes | ✅ |
| `mlp_meta` | Neural | Dropout | ✅ |
| `xgb_meta` | Boosting | Built-in | ✅ |
| `gating_mlp` | Neural (MoE) | Dropout | ❌ |
| `gating_attention` | Attention (MoE) | Built-in | ❌ |

### 1.5 Calibration Methods

| Block | What It Does | Status |
|-------|--------------|--------|
| `platt` | Logistic scaling | ✅ |
| `isotonic` | Non-parametric | ✅ |
| `temperature` | Single scalar | ⚠️ Partial |
| `beta` | Beta distribution | ❌ |
| `none` | Skip calibration | ✅ |

### 1.6 Regime Detectors (For Dynamic Ensembles)

| Block | Method | Status |
|-------|--------|--------|
| `hmm_volatility` | HMM on volatility | ❌ |
| `hmm_returns` | HMM on returns | ❌ |
| `rule_based` | Threshold rules | ❌ |
| `clustering` | K-means on features | ❌ |
| `changepoint` | PELT/BOCPD | ❌ |

### 1.7 Optimization Dimensions

| Dimension | What's Optimized | Status |
|-----------|------------------|--------|
| `barriers` | Profit/loss thresholds, holding period | ✅ |
| `feature_selection` | Which features from base set | ❌ |
| `feature_params` | RSI period, ATR window, etc. | ❌ |
| `feature_timeframes` | Which TF per feature | ❌ |
| `hyperparameters` | Model-specific params | ✅ |

---

## 2. Presets & Knobs

### 2.1 Presets (One-Click Configurations)

```yaml
# User picks ONE preset, factory configures everything else
presets:
  
  quick_test:
    description: "Fast iteration, single model, no Optuna"
    models: [catboost]
    ensemble: none
    optuna: false
    horizons: [20]
    # Total time: ~5 min
  
  balanced:
    description: "Good default, heterogeneous stack, moderate Optuna"
    models: [catboost, lightgbm, tcn]
    ensemble: stacking
    meta_learner: ridge
    optuna:
      n_trials: 50
      dimensions: [barriers, hyperparameters]
    horizons: [5, 20, 60]
    calibration: isotonic
    # Total time: ~1-2 hours
  
  max_alpha:
    description: "Full power, all models, exhaustive optimization"
    models: [catboost, lightgbm, xgboost, lstm, gru, tcn, patchtst]
    ensemble: moe
    gating: attention
    optuna:
      n_trials: 200
      dimensions: [barriers, features, feature_params, hyperparameters]
    horizons: [5, 10, 20, 60, 120]
    calibration: beta
    # Total time: ~8-12 hours
  
  speed:
    description: "Production latency focus, fast models only"
    models: [lightgbm, catboost]
    ensemble: blending
    optuna:
      n_trials: 30
      dimensions: [hyperparameters]
    horizons: [5, 20]
    calibration: temperature
    # Total time: ~30 min
  
  adaptive:
    description: "Non-stationary focus, regime-aware"
    models: [catboost, tcn, lstm]
    ensemble: adaptive
    decay_halflife: 100
    regime_detector: hmm_volatility
    optuna:
      n_trials: 100
      dimensions: [barriers, hyperparameters]
    horizons: [20, 60]
    # Total time: ~3-4 hours
```

### 2.2 Knobs (Fine-Tuning)

After picking a preset, users can twist these knobs:

```yaml
# experiment_config.yaml
preset: balanced  # Start here

# KNOB 1: Risk Profile
risk:
  profile: moderate  # conservative | moderate | aggressive
  # Affects: position sizing, barrier thresholds, drawdown limits

# KNOB 2: Horizons
horizons: [5, 20]  # Override preset's horizons

# KNOB 3: Model Swaps (keep ensemble style, change models)
models:
  remove: [tcn]
  add: [lstm]

# KNOB 4: Calibration
calibration: platt  # Override preset

# KNOB 5: Optuna Budget
optuna:
  n_trials: 75  # More than preset's 50

# KNOB 6: MTF Toggle
disable_mtf_features: false  # Ablation experiments

# KNOB 7: Contract/Symbol
contract: MES
symbols: [MES, MNQ]  # Multi-symbol training
```

### 2.3 Advanced Mode (Full Control)

Power users can bypass presets entirely:

```yaml
# advanced_config.yaml
preset: none  # I know what I'm doing

contract: MES
primary_timeframe: 5min
horizons: [5, 10, 20, 60, 120]

data:
  train_start: "2020-01-01"
  train_end: "2023-12-31"
  test_start: "2024-01-01"
  test_end: "2024-06-30"
  purge_days: 5
  embargo_days: 2

feature_sets:
  catboost:
    families: [momentum, volatility, microstructure]
    exclude: [wavelets, entropy]
  tcn:
    families: [momentum, volatility, wavelets]
  patchtst:
    families: [raw_ohlcv]

models:
  - name: catboost
    mtf_mode: indicators
    optuna:
      depth: [4, 10]
      learning_rate: [0.01, 0.1]
      iterations: [500, 2000]
      
  - name: tcn
    mtf_mode: indicators
    sequence_length: 60
    optuna:
      num_filters: [32, 128]
      kernel_size: [2, 7]
      dropout: [0.1, 0.4]
      
  - name: patchtst
    mtf_mode: multi_stream
    patch_length: 16
    optuna:
      d_model: [64, 256]
      n_heads: [4, 8]
      n_layers: [2, 6]

ensemble:
  method: moe
  gating:
    type: attention
    features: [realized_vol_20, trend_strength, volume_regime]
    hidden_dims: [64, 32]
    dropout: 0.2
  
  fallback: stacking  # If MoE gating fails validation
  meta_learner: ridge
  
  calibration:
    method: beta
    cv_folds: 5

optimization:
  optuna:
    n_trials: 200
    sampler: tpe
    pruner: hyperband
    dimensions:
      barriers: true
      feature_selection: true
      feature_params: true
      feature_timeframes: false  # Skip this dimension
      hyperparameters: true

validation:
  method: purged_kfold
  n_splits: 5
  purge_pct: 0.02
  embargo_pct: 0.01

inference:
  position_sizing: kelly_fractional
  kelly_fraction: 0.25
  max_position: 3
  
backtest:
  costs:
    commission_per_side: 2.50
    slippage_ticks: 1
  
output:
  save_optuna_studies: true
  save_oof_predictions: true
  generate_plots: true
```

---

## 3. Ensemble Styles

### 3.1 Static Ensembles (What You Have)

**Voting:**
```
Base₁ ──→ Prediction₁ ──┐
Base₂ ──→ Prediction₂ ──┼──→ Majority/Average ──→ Final
Base₃ ──→ Prediction₃ ──┘
```
- Hard voting: argmax(class counts)
- Soft voting: mean(probabilities)
- Weighted: sum(weight_i × pred_i)

**Blending:**
```
Train │ Base₁, Base₂, Base₃ trained on Train
      ↓
Blend │ Holdout predictions ──→ Fit fixed weights
      ↓
Test  │ weighted_avg(predictions)
```

**Stacking (OOF-Safe):**
```
Fold 1 │ Train on 2,3,4,5 ──→ Predict Fold 1 (OOF)
Fold 2 │ Train on 1,3,4,5 ──→ Predict Fold 2 (OOF)
  ...
       ↓
Meta   │ Train meta-learner on ALL OOF predictions
       ↓
Final  │ Retrain bases on full train, combine via meta-learner
```

### 3.2 Dynamic Ensembles (What You Need)

**Mixture of Experts (MoE):**
```
                    ┌──────────────────┐
                    │  Gating Network  │
                    │  g(x) → weights  │
                    └────────┬─────────┘
                             │ [0.7, 0.2, 0.1]  ← Changes per sample!
        ┌────────────────────┼────────────────────┐
        ↓                    ↓                    ↓
   ┌─────────┐          ┌─────────┐          ┌─────────┐
   │ Expert₁ │          │ Expert₂ │          │ Expert₃ │
   │CatBoost │          │   TCN   │          │ PatchTST│
   └────┬────┘          └────┬────┘          └────┬────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             ↓
                    Σ(weight_i × expert_i(x))
```

**Why MoE for trading:**
- Markets have regimes (trending, ranging, volatile, quiet)
- Different models excel in different regimes
- Gating network *learns* which model to trust *given current conditions*
- No manual regime rules needed

**OOF-Safe MoE Training:**
```
Phase 1: Generate OOF predictions from all experts (standard stacking)
Phase 2: Train gating network on OOF predictions + features
         └── Gating sees: [OOF_pred₁, OOF_pred₂, OOF_pred₃, regime_features]
         └── Gating outputs: weights for each expert
Phase 3: Final retrain experts on full train
Phase 4: Deploy gating + experts together
```

**Adaptive Weighting:**
```
t=0   │ weights = [0.33, 0.33, 0.33]  # Equal start
t=100 │ Model₁ crushing it ──→ weights = [0.50, 0.30, 0.20]
t=200 │ Model₁ decaying    ──→ weights = [0.35, 0.40, 0.25]
t=300 │ Model₃ hot streak  ──→ weights = [0.25, 0.30, 0.45]
```

```python
# Exponential decay on rolling performance
def update_weights(weights, recent_performance, alpha=0.1):
    # recent_performance: accuracy/sharpe over last N bars
    raw_weights = alpha * recent_performance + (1 - alpha) * weights
    return raw_weights / raw_weights.sum()  # Normalize
```

**Regime-Routed:**
```
┌─────────────────────┐
│   Regime Detector   │
│  (HMM / Rules / ?)  │
└──────────┬──────────┘
           │ regime = "high_vol_trending"
           ↓
    ┌──────────────────────────────┐
    │     Routing Table            │
    ├──────────────────────────────┤
    │ low_vol_ranging  → CatBoost  │
    │ low_vol_trending → TCN       │
    │ high_vol_ranging → LSTM      │
    │ high_vol_trending→ [TCN,LSTM]│  ← Can route to mini-ensemble
    │ crisis           → ABSTAIN   │
    └──────────────────────────────┘
```

**Uncertainty-Weighted:**
```
Model₁ ──→ pred=0.7, uncertainty=0.1 ──→ weight = 1/0.1 = 10
Model₂ ──→ pred=0.6, uncertainty=0.3 ──→ weight = 1/0.3 = 3.3
Model₃ ──→ pred=0.8, uncertainty=0.5 ──→ weight = 1/0.5 = 2.0

Final = Σ(weight_i × pred_i) / Σ(weight_i)
```

Uncertainty sources:
- MC Dropout (run inference N times with dropout)
- Ensemble variance (disagreement among sub-models)
- Conformal prediction intervals

**Cascaded (Early Exit):**
```
Sample x arrives
      ↓
┌─────────────┐
│  CatBoost   │──→ confidence > 0.85? ──→ YES ──→ Return prediction
│   (fast)    │                                   (exit early)
└──────┬──────┘
       │ NO (uncertain)
       ↓
┌─────────────┐
│    TCN      │──→ confidence > 0.80? ──→ YES ──→ Return prediction
│  (medium)   │
└──────┬──────┘
       │ NO (still uncertain)
       ↓
┌─────────────┐
│ Full Stack  │──→ Return prediction (always)
│  (slowest)  │
└─────────────┘
```

Benefits: Most samples exit early → faster inference, lower compute.

### 3.3 Ensemble Compatibility Matrix

| Ensemble | 2D Models | 3D Models | 4D Models | Requires |
|----------|-----------|-----------|-----------|----------|
| voting | ✅ | ✅ | ✅ | Same output shape |
| blending | ✅ | ✅ | ✅ | Holdout set |
| stacking | ✅ | ✅ | ✅ | OOF generation |
| moe | ✅ | ✅ | ✅ | Gating network |
| adaptive | ✅ | ✅ | ✅ | Rolling performance tracker |
| regime_routed | ✅ | ✅ | ✅ | Regime detector |
| uncertainty_weighted | ⚠️ | ✅ | ✅ | Uncertainty estimates |
| cascaded | ✅ | ✅ | ✅ | Confidence thresholds |

---

## 4. Architecture Layers

### 4.1 The Three Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                     LAYER 1: USER INTERFACE                     │
│                                                                 │
│   • Presets (one-click)                                        │
│   • Knobs (simple overrides)                                   │
│   • Advanced config (full control)                             │
│   • CLI: ml-factory run --config my_bear.yaml                  │
│                                                                 │
│   User's mental model: "I picked some models and an ensemble"  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   LAYER 2: ORCHESTRATION                        │
│                                                                 │
│   ConfigValidator     → Checks compatibility, fills defaults   │
│   PipelineOrchestrator → Sequences phases, handles failures    │
│   CompatibilityEngine → Routes models to correct data stores   │
│   ArtifactManager     → Tracks all outputs, ensures repro      │
│                                                                 │
│   Factory's job: "Make it work, don't leak, stay reproducible" │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     LAYER 3: COMPONENTS                         │
│                                                                 │
│   DataStore           → Canonical parquets (engineered + MTF)  │
│   FeatureEngine       → Compute features on demand             │
│   ModelRegistry       → All base learner implementations       │
│   EnsembleRegistry    → All ensemble method implementations    │
│   OptunaEngine        → 5-dimension optimization               │
│   CalibrationEngine   → Post-hoc probability calibration       │
│   RegimeDetector      → HMM, rules, clustering                 │
│   InferenceEngine     → Production prediction pipeline         │
│   BacktestEngine      → Realistic P&L simulation               │
│                                                                 │
│   Components: "I do one thing well, I'm swappable"             │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Flow (Canonical Stores)

```
Raw OHLCV (1min)
      │
      ├──────────────────────────────────────────────────────────┐
      │                                                          │
      ↓                                                          ↓
┌─────────────────────────────┐          ┌─────────────────────────────┐
│   CANONICAL STORE 1         │          │   CANONICAL STORE 2         │
│   Engineered Features       │          │   Raw MTF OHLCV             │
├─────────────────────────────┤          ├─────────────────────────────┤
│ Location:                   │          │ Location:                   │
│   data/canonical/engineered │          │   data/canonical/raw_mtf/   │
│                             │          │                             │
│ Files:                      │          │ Files:                      │
│   {symbol}_train.parquet    │          │   {symbol}_1min_train.pq    │
│   {symbol}_val.parquet      │          │   {symbol}_5min_train.pq    │
│   {symbol}_test.parquet     │          │   {symbol}_10min_train.pq   │
│                             │          │   ... (9 TFs × 3 splits)    │
│ Shape: (N, ~180)            │          │                             │
│                             │          │ Shape per file: (N_tf, 4)   │
│ Contains:                   │          │                             │
│   • Momentum indicators     │          │ Contains:                   │
│   • Volatility indicators   │          │   • Raw O, H, L, C only     │
│   • Volume indicators       │          │   • Aligned to base TF      │
│   • MTF indicator columns   │          │   • shift(1) anti-lookahead │
│   • Microstructure          │          │                             │
└──────────────┬──────────────┘          └──────────────┬──────────────┘
               │                                        │
               │                                        │
      ┌────────┴────────┐                    ┌──────────┴──────────┐
      │   Model Router  │                    │  Multi-Res Adapter  │
      └────────┬────────┘                    └──────────┬──────────┘
               │                                        │
    ┌──────────┼──────────┐                            │
    ↓          ↓          ↓                            ↓
┌───────┐ ┌───────┐ ┌───────┐                    ┌───────────┐
│2D Model│ │3D Model│ │3D Model│                   │ 4D Model  │
│CatBoost│ │ LSTM  │ │  TCN  │                    │ PatchTST  │
│(N, F)  │ │(N,T,F)│ │(N,T,F)│                    │(N,9,T,4)  │
└───────┘ └───────┘ └───────┘                    └───────────┘
```

### 4.3 OOF-Safe Training Flow

```
PHASE 1: DATA PREPARATION
═════════════════════════
Raw OHLCV ──→ MTF Alignment ──→ Feature Engineering ──→ Labels ──→ Splits
                                                                      │
                                                         ┌────────────┴────────────┐
                                                         │    Purged K-Fold        │
                                                         │    (5 folds, purge=2%)  │
                                                         └────────────┬────────────┘
                                                                      │
PHASE 2: OPTUNA OPTIMIZATION (per model)                              │
════════════════════════════════════════                              │
                                                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  For each trial:                                                            │
│    1. Sample barrier params (Dim 1)                                         │
│    2. Sample feature subset (Dim 2)                                         │
│    3. Sample feature params (Dim 3)                                         │
│    4. Sample feature TFs (Dim 4)                                            │
│    5. Sample hyperparameters (Dim 5)                                        │
│    6. Train on folds 2-5, validate on fold 1 (with purge)                   │
│    7. Repeat for all folds                                                  │
│    8. Return mean CV score                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ↓ Best trial per model
                              ┌───────────────┐
                              │  FeatureSpec  │  ← Saved artifact
                              │   Artifact    │
                              └───────┬───────┘
                                      │
PHASE 3: OOF GENERATION                │
═══════════════════════                ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  For each model (using best FeatureSpec):                                   │
│    Fold 1: Train on 2,3,4,5 → Predict fold 1 → Store OOF₁                   │
│    Fold 2: Train on 1,3,4,5 → Predict fold 2 → Store OOF₂                   │
│    Fold 3: Train on 1,2,4,5 → Predict fold 3 → Store OOF₃                   │
│    Fold 4: Train on 1,2,3,5 → Predict fold 4 → Store OOF₄                   │
│    Fold 5: Train on 1,2,3,4 → Predict fold 5 → Store OOF₅                   │
│                                                                             │
│  Result: Full OOF predictions for entire train set (no leakage)             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ OOF predictions from ALL models
                                      ↓
PHASE 4: META-LEARNER / GATING TRAINING
═══════════════════════════════════════
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stack OOF predictions: X_meta = [OOF_catboost, OOF_tcn, OOF_lstm, ...]     │
│                                                                             │
│  For stacking:                                                              │
│    Train meta-learner on (X_meta, y_train)                                  │
│                                                                             │
│  For MoE:                                                                   │
│    X_gating = [X_meta, regime_features]                                     │
│    Train gating network to predict optimal weights                          │
│                                                                             │
│  For adaptive:                                                              │
│    Initialize weight tracker with OOF performance                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
PHASE 5: FINAL RETRAIN                 │
══════════════════════                 │
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  Retrain ALL base models on FULL training set (using best FeatureSpecs)    │
│  This gives us the final production models                                  │
│                                                                             │
│  Meta-learner/gating already trained on OOF (no retrain needed)             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
PHASE 6: CALIBRATION                   │
════════════════════                   │
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  Using validation set (held out from all training):                         │
│    1. Get ensemble predictions                                              │
│    2. Fit calibrator (Platt/isotonic/beta)                                  │
│    3. Store calibration parameters                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
PHASE 7: TEST & BACKTEST               │
════════════════════════               │
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  Final evaluation on test set (never seen before):                          │
│    1. Feature engineering (same pipeline)                                   │
│    2. Base model predictions                                                │
│    3. Ensemble combination (meta/gating/adaptive)                           │
│    4. Calibration                                                           │
│    5. Position sizing                                                       │
│    6. Backtest with realistic costs                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
PHASE 8: BUNDLE & DEPLOY               │
════════════════════════               │
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  experiments/{run_id}/                                                      │
│  ├── config.yaml                 # Frozen config                            │
│  ├── feature_specs/              # All FeatureSpec artifacts                │
│  ├── models/                     # Serialized base models                   │
│  ├── ensemble/                   # Meta-learner/gating weights              │
│  ├── calibration/                # Calibrator state                         │
│  ├── metrics/                    # All performance metrics                  │
│  └── deployment/                                                            │
│      └── inference_bundle.tar.gz # Everything needed to predict             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Current State & Gap Analysis

### 5.1 What's Working

| Component | Status | Notes |
|-----------|--------|-------|
| **Data Pipeline** |
| Raw OHLCV ingestion | ✅ | |
| MTF alignment | ✅ | 9 timeframes |
| Feature engineering | ✅ | ~180 features |
| Label generation (triple barrier) | ✅ | |
| Train/val/test splits | ✅ | Time-based |
| Scaling | ✅ | |
| **Models** |
| XGBoost | ✅ | 2D |
| LightGBM | ✅ | 2D |
| CatBoost | ✅ | 2D |
| LSTM | ✅ | 3D |
| GRU | ✅ | 3D |
| TCN | ✅ | 3D |
| **Ensemble** |
| Voting (hard/soft) | ✅ | Static |
| Blending | ✅ | Static |
| Stacking (OOF) | ✅ | Static |
| **Meta-Learners** |
| Ridge/Lasso/ElasticNet | ✅ | |
| Logistic | ✅ | |
| MLP | ✅ | |
| **Optimization** |
| Barrier optimization | ✅ | Optuna Dim 1 |
| Hyperparameter tuning | ✅ | Optuna Dim 5 |
| **Calibration** |
| Platt scaling | ✅ | |
| Isotonic regression | ✅ | |
| **Inference** |
| Model bundles | ✅ | Serialization |
| Preprocessing graph | ✅ | Train/serve parity |
| Backtesting | ✅ | Realistic costs |
| Position sizing | ✅ | Kelly/Vol-target |

### 5.2 Critical Gaps

```
PRIORITY 1: BLOCKING 4D MODELS (HIGH IMPACT)
════════════════════════════════════════════

GAP 1.1: Raw MTF Canonical Store
├── What's missing: data/canonical/raw_mtf/{symbol}_{tf}_{split}.parquet
├── Why it matters: 4D models (PatchTST, iTransformer, etc.) need raw OHLCV
├── Impact: 0/6 advanced models can be trained
└── Effort: Medium (modify Phase 2 pipeline)

GAP 1.2: TimeSeriesDataContainer 4D Support
├── What's missing: Container only handles 2D/3D, not (N, 9, T, 4)
├── Why it matters: Can't feed 4D data to models
└── Effort: Medium

GAP 1.3: Multi-Resolution 4D Adapter
├── What's missing: Transform raw MTF → 4D tensor with proper alignment
├── Why it matters: Bridge between canonical store and 4D models
├── Depends on: Gap 1.1, Gap 1.2
└── Effort: Medium-High

GAP 1.4: Advanced Model Implementations
├── Missing models: InceptionTime, ResNet1D, PatchTST, iTransformer, TFT, N-BEATS
├── 3D models (can start now): InceptionTime, ResNet1D
├── 4D models (blocked): PatchTST, iTransformer, TFT, N-BEATS
└── Effort: High (6 model implementations)


PRIORITY 2: OPTUNA FEATURE OPTIMIZATION (HIGH IMPACT)
═════════════════════════════════════════════════════

GAP 2.1: Feature Selection Optimization (Dim 2)
├── What's missing: Optuna selecting which features from base set
├── Why it matters: Current uses all features → overfitting risk
└── Effort: Medium

GAP 2.2: Feature Parameter Optimization (Dim 3)
├── What's missing: RSI period, ATR window, etc. as Optuna params
├── Why it matters: Hardcoded feature params may not be optimal
└── Effort: Medium

GAP 2.3: Feature Timeframe Optimization (Dim 4)
├── What's missing: Which TF each feature computed on
├── Why it matters: Some features better on 5min, others on 60min
└── Effort: Medium

GAP 2.4: Per-Model Base Feature Sets
├── What's missing: Each model type has defined feature families
├── Why it matters: Boosting needs different features than transformers
└── Effort: Low-Medium

GAP 2.5: FeatureSpec Artifact Flow
├── What's missing: Save all 5 dims to JSON, embed in bundle
├── Why it matters: Reproducibility requires exact config
├── Depends on: Gap 2.1-2.4
└── Effort: Medium


PRIORITY 3: DYNAMIC ENSEMBLES (DIFFERENTIATING FEATURE)
═══════════════════════════════════════════════════════

GAP 3.1: Mixture of Experts (MoE)
├── What's missing: Gating network implementation
├── Components needed:
│   ├── GatingMLP / GatingAttention classes
│   ├── MoE training loop (OOF-safe)
│   └── MoE inference wrapper
└── Effort: High

GAP 3.2: Adaptive Weighting
├── What's missing: Rolling performance tracker, weight decay
├── Components needed:
│   ├── PerformanceTracker class
│   ├── WeightUpdater with decay
│   └── AdaptiveEnsemble wrapper
└── Effort: Medium

GAP 3.3: Regime Detection
├── What's missing: HMM, rule-based, clustering detectors
├── Components needed:
│   ├── HMMRegimeDetector
│   ├── RuleBasedDetector
│   ├── ClusteringDetector
│   └── RegimeRouter
└── Effort: Medium-High

GAP 3.4: Uncertainty Estimation
├── What's missing: MC Dropout, ensemble variance, conformal
├── Why it matters: Needed for uncertainty-weighted ensembles
└── Effort: Medium

GAP 3.5: Cascaded/Early Exit
├── What's missing: Confidence routing logic
├── Effort: Low-Medium


PRIORITY 4: UX LAYER (BUILD-A-BEAR EXPERIENCE)
══════════════════════════════════════════════

GAP 4.1: Preset System
├── What's missing: Pre-configured "recipes"
├── Effort: Low

GAP 4.2: Config Validator
├── What's missing: Compatibility checks, default filling
├── Effort: Medium

GAP 4.3: MLFactory Unified Entry Point
├── What's missing: Single class that orchestrates everything
├── Effort: Medium

GAP 4.4: CLI Polish
├── What's missing: ml-factory run --config, progress bars, clean output
├── Effort: Low

GAP 4.5: MTF Ablation Flag
├── What's missing: Simple toggle to disable MTF features
├── Effort: Low
```

### 5.3 Gap Summary

| Priority | Category | Gaps | Blocking? | Effort |
|----------|----------|------|-----------|--------|
| 1 | 4D Infrastructure | 4 | Yes - blocks 6 models | High |
| 2 | Optuna Features | 5 | No - but limits quality | Medium |
| 3 | Dynamic Ensembles | 5 | No - but key differentiator | High |
| 4 | UX Layer | 5 | No - but needed for "Build-a-Bear" | Medium |

---

## 6. Recommended File Structure

Your current structure is probably getting tangled. Here's a clean organization for long-term maintainability:

```
ml_factory/
│
├── README.md
├── pyproject.toml                    # Modern Python packaging
├── setup.py                          # Backwards compatibility
│
├── configs/                          # ALL CONFIGURATION LIVES HERE
│   ├── presets/                      # One-click configurations
│   │   ├── quick_test.yaml
│   │   ├── balanced.yaml
│   │   ├── max_alpha.yaml
│   │   ├── speed.yaml
│   │   └── adaptive.yaml
│   │
│   ├── feature_sets/                 # Base feature definitions
│   │   ├── boosting.yaml             # Features for XGB/LGB/CB
│   │   ├── neural_sequence.yaml      # Features for LSTM/GRU/TCN
│   │   ├── transformer_raw.yaml      # Raw OHLCV only
│   │   └── transformer_engineered.yaml
│   │
│   ├── model_defaults/               # Default hyperparams per model
│   │   ├── catboost.yaml
│   │   ├── lightgbm.yaml
│   │   ├── xgboost.yaml
│   │   ├── lstm.yaml
│   │   ├── tcn.yaml
│   │   └── patchtst.yaml
│   │
│   └── schemas/                      # JSON schemas for validation
│       ├── experiment_config.schema.json
│       ├── feature_spec.schema.json
│       └── model_bundle.schema.json
│
├── ml_factory/                       # MAIN PACKAGE
│   │
│   ├── __init__.py
│   ├── factory.py                    # MLFactory - main entry point
│   ├── cli.py                        # Command-line interface
│   │
│   ├── config/                       # Configuration handling
│   │   ├── __init__.py
│   │   ├── loader.py                 # Load and merge configs
│   │   ├── validator.py              # Validate compatibility
│   │   ├── resolver.py               # Resolve presets + overrides
│   │   └── defaults.py               # Default values
│   │
│   ├── data/                         # Data pipeline
│   │   ├── __init__.py
│   │   ├── loaders/
│   │   │   ├── __init__.py
│   │   │   ├── ohlcv_loader.py       # Raw data ingestion
│   │   │   └── parquet_loader.py     # Canonical store loading
│   │   │
│   │   ├── stores/                   # Canonical data stores
│   │   │   ├── __init__.py
│   │   │   ├── engineered_store.py   # data/canonical/engineered/
│   │   │   └── raw_mtf_store.py      # data/canonical/raw_mtf/  ❌ MISSING
│   │   │
│   │   ├── mtf/                      # Multi-timeframe handling
│   │   │   ├── __init__.py
│   │   │   ├── aligner.py            # Align TFs to base
│   │   │   ├── resampler.py          # Resample to different TFs
│   │   │   └── adapter_4d.py         # Multi-Res 4D adapter  ❌ MISSING
│   │   │
│   │   ├── splits/
│   │   │   ├── __init__.py
│   │   │   ├── time_series_split.py  # Train/val/test
│   │   │   └── purged_kfold.py       # PurgedKFold CV
│   │   │
│   │   └── containers/
│   │       ├── __init__.py
│   │       └── ts_container.py       # TimeSeriesDataContainer
│   │                                 # (needs 4D support)  ⚠️ PARTIAL
│   │
│   ├── features/                     # Feature engineering
│   │   ├── __init__.py
│   │   ├── registry.py               # Feature function registry
│   │   │
│   │   ├── families/                 # Feature implementations by family
│   │   │   ├── __init__.py
│   │   │   ├── momentum.py           # RSI, MACD, ROC, etc.
│   │   │   ├── volatility.py         # ATR, BB, Keltner, etc.
│   │   │   ├── volume.py             # OBV, VWAP, MFI, etc.
│   │   │   ├── microstructure.py     # Spread, depth, flow
│   │   │   ├── wavelets.py           # DWT features  ⚠️ PARTIAL
│   │   │   └── entropy.py            # Entropy features  ❌ MISSING
│   │   │
│   │   ├── pipeline.py               # Feature computation pipeline
│   │   ├── spec.py                   # FeatureSpec dataclass
│   │   └── selector.py               # Feature selection utilities
│   │
│   ├── labels/                       # Label generation
│   │   ├── __init__.py
│   │   ├── triple_barrier.py         # Triple barrier labeling
│   │   ├── fixed_horizon.py          # Simple fixed-horizon
│   │   └── meta_labeling.py          # Meta-labeling (side + size)
│   │
│   ├── models/                       # Model implementations
│   │   ├── __init__.py
│   │   ├── registry.py               # Model registry
│   │   ├── base.py                   # BaseModel ABC
│   │   │
│   │   ├── boosting/                 # 2D models
│   │   │   ├── __init__.py
│   │   │   ├── xgboost_model.py
│   │   │   ├── lightgbm_model.py
│   │   │   └── catboost_model.py
│   │   │
│   │   ├── rnn/                      # 3D sequence models
│   │   │   ├── __init__.py
│   │   │   ├── lstm_model.py
│   │   │   └── gru_model.py
│   │   │
│   │   ├── cnn/                      # 3D convolutional models
│   │   │   ├── __init__.py
│   │   │   ├── tcn_model.py
│   │   │   ├── inception_time.py     # ❌ MISSING
│   │   │   └── resnet_1d.py          # ❌ MISSING
│   │   │
│   │   ├── transformer/              # 4D transformer models
│   │   │   ├── __init__.py
│   │   │   ├── patchtst.py           # ❌ MISSING
│   │   │   ├── itransformer.py       # ❌ MISSING
│   │   │   └── tft.py                # ❌ MISSING
│   │   │
│   │   └── decomposition/            # Decomposition models
│   │       ├── __init__.py
│   │       └── nbeats.py             # ❌ MISSING
│   │
│   ├── ensemble/                     # Ensemble methods
│   │   ├── __init__.py
│   │   ├── registry.py               # Ensemble registry
│   │   ├── base.py                   # BaseEnsemble ABC
│   │   │
│   │   ├── static/                   # Static ensemble methods
│   │   │   ├── __init__.py
│   │   │   ├── voting.py             # Hard/soft voting
│   │   │   ├── blending.py           # Fixed weight blending
│   │   │   └── stacking.py           # OOF stacking
│   │   │
│   │   ├── dynamic/                  # Dynamic ensemble methods
│   │   │   ├── __init__.py
│   │   │   ├── moe.py                # Mixture of Experts  ❌ MISSING
│   │   │   ├── adaptive.py           # Adaptive weighting  ❌ MISSING
│   │   │   ├── regime_routed.py      # Regime routing  ❌ MISSING
│   │   │   ├── uncertainty.py        # Uncertainty-weighted  ❌ MISSING
│   │   │   └── cascaded.py           # Early exit  ❌ MISSING
│   │   │
│   │   ├── meta_learners/            # Meta-learner implementations
│   │   │   ├── __init__.py
│   │   │   ├── linear.py             # Ridge, Lasso, ElasticNet
│   │   │   ├── mlp.py                # Neural meta-learner
│   │   │   └── boosting.py           # XGB as meta-learner
│   │   │
│   │   └── gating/                   # MoE gating networks
│   │       ├── __init__.py
│   │       ├── mlp_gating.py         # ❌ MISSING
│   │       └── attention_gating.py   # ❌ MISSING
│   │
│   ├── regime/                       # Regime detection
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── hmm.py                    # HMM detector  ❌ MISSING
│   │   ├── rules.py                  # Rule-based  ❌ MISSING
│   │   └── clustering.py             # Clustering-based  ❌ MISSING
│   │
│   ├── calibration/                  # Probability calibration
│   │   ├── __init__.py
│   │   ├── platt.py                  # Platt scaling
│   │   ├── isotonic.py               # Isotonic regression
│   │   ├── temperature.py            # Temperature scaling  ⚠️ PARTIAL
│   │   └── beta.py                   # Beta calibration  ❌ MISSING
│   │
│   ├── optimization/                 # Optuna optimization
│   │   ├── __init__.py
│   │   ├── engine.py                 # Main Optuna engine
│   │   │
│   │   ├── dimensions/               # Individual optimization dimensions
│   │   │   ├── __init__.py
│   │   │   ├── barriers.py           # Dim 1: Barrier params ✅
│   │   │   ├── feature_selection.py  # Dim 2: Which features  ❌ MISSING
│   │   │   ├── feature_params.py     # Dim 3: Feature params  ❌ MISSING
│   │   │   ├── feature_timeframes.py # Dim 4: Feature TFs  ❌ MISSING
│   │   │   └── hyperparameters.py    # Dim 5: Model hyperparams ✅
│   │   │
│   │   ├── samplers.py               # Custom Optuna samplers
│   │   └── pruners.py                # Custom pruners
│   │
│   ├── training/                     # Training orchestration
│   │   ├── __init__.py
│   │   ├── trainer.py                # Main trainer class
│   │   ├── oof_generator.py          # OOF prediction generation
│   │   └── callbacks.py              # Training callbacks
│   │
│   ├── inference/                    # Inference pipeline
│   │   ├── __init__.py
│   │   ├── engine.py                 # Inference engine
│   │   ├── preprocessor.py           # Feature preprocessing
│   │   └── position_sizer.py         # Kelly, vol-target, etc.
│   │
│   ├── backtest/                     # Backtesting
│   │   ├── __init__.py
│   │   ├── engine.py                 # Backtest engine
│   │   ├── costs.py                  # Commission, slippage
│   │   └── metrics.py                # Sharpe, drawdown, etc.
│   │
│   ├── artifacts/                    # Artifact management
│   │   ├── __init__.py
│   │   ├── feature_spec.py           # FeatureSpec save/load
│   │   ├── model_bundle.py           # ModelBundle packaging
│   │   └── experiment.py             # Experiment tracking
│   │
│   ├── uncertainty/                  # Uncertainty estimation
│   │   ├── __init__.py
│   │   ├── mc_dropout.py             # MC Dropout  ❌ MISSING
│   │   ├── ensemble_variance.py      # Ensemble disagreement  ❌ MISSING
│   │   └── conformal.py              # Conformal prediction  ❌ MISSING
│   │
│   └── utils/                        # Utilities
│       ├── __init__.py
│       ├── logging.py                # Structured logging
│       ├── timing.py                 # Performance timing
│       ├── validation.py             # Data validation
│       └── visualization.py          # Plotting utilities
│
├── tests/                            # Test suite
│   ├── conftest.py
│   ├── fixtures/                     # Test data fixtures
│   │
│   ├── unit/
│   │   ├── test_config_validator.py
│   │   ├── test_feature_families.py
│   │   ├── test_models/
│   │   ├── test_ensembles/
│   │   └── ...
│   │
│   ├── integration/
│   │   ├── test_full_pipeline.py
│   │   ├── test_oof_flow.py
│   │   └── test_moe_training.py
│   │
│   └── e2e/
│       └── test_build_a_bear.py      # Full config → bundle
│
├── notebooks/                        # Jupyter notebooks
│   ├── colab_quickstart.ipynb        # One-click Colab demo
│   ├── advanced_config.ipynb         # Power user walkthrough
│   └── regime_analysis.ipynb         # Regime detection exploration
│
├── experiments/                      # Experiment outputs (gitignored)
│   └── {run_id}/
│       ├── config.yaml
│       ├── feature_specs/
│       ├── models/
│       ├── ensemble/
│       ├── metrics/
│       ├── plots/
│       └── deployment/
│
├── data/                             # Data directory (gitignored)
│   ├── raw/                          # Raw OHLCV
│   └── canonical/
│       ├── engineered/               # Engineered feature parquets
│       └── raw_mtf/                  # Raw MTF OHLCV parquets
│
└── scripts/                          # Utility scripts
    ├── download_sample_data.py
    ├── validate_installation.py
    └── benchmark_models.py
```

### 6.1 Key Organizational Principles

**1. Configs are First-Class Citizens**
```
configs/
├── presets/          # User-facing "recipes"
├── feature_sets/     # Declarative feature definitions
├── model_defaults/   # Sensible defaults per model
└── schemas/          # Validation schemas
```
Configs live OUTSIDE the Python package. Easy to edit, version, share.

**2. Registry Pattern for All Swappables**
```python
# ml_factory/models/registry.py
MODEL_REGISTRY = {
    "catboost": CatBoostModel,
    "tcn": TCNModel,
    "patchtst": PatchTSTModel,
}

# ml_factory/ensemble/registry.py
ENSEMBLE_REGISTRY = {
    "stacking": StackingEnsemble,
    "moe": MixtureOfExperts,
    "adaptive": AdaptiveEnsemble,
}
```
Add new models/ensembles by registering them. No core code changes.

**3. Dimension-Organized Optimization**
```
optimization/dimensions/
├── barriers.py           # Dim 1
├── feature_selection.py  # Dim 2
├── feature_params.py     # Dim 3
├── feature_timeframes.py # Dim 4
└── hyperparameters.py    # Dim 5
```
Each dimension is a self-contained module. Easy to enable/disable.

**4. Static vs Dynamic Ensemble Split**
```
ensemble/
├── static/   # Voting, blending, stacking
└── dynamic/  # MoE, adaptive, regime-routed
```
Clear separation of complexity. Static is done, dynamic is the frontier.

**5. Feature Families**
```
features/families/
├── momentum.py
├── volatility.py
├── volume.py
├── microstructure.py
├── wavelets.py
└── entropy.py
```
Each family is independent. Easy to add new feature types.

---

## 7. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
**Goal:** Clean up structure, implement presets

```
□ Reorganize codebase to match recommended structure
□ Implement config loader + validator
□ Create 5 preset configs (quick_test, balanced, max_alpha, speed, adaptive)
□ Build MLFactory entry point class
□ Polish CLI (ml-factory run --config)
□ Add MTF ablation flag
```

### Phase 2: 4D Infrastructure (Weeks 3-4)
**Goal:** Enable 4D models

```
□ Implement raw MTF canonical store
□ Add 4D support to TimeSeriesDataContainer
□ Build Multi-Resolution 4D adapter
□ Implement InceptionTime, ResNet1D (3D, can start now)
```

### Phase 3: Optuna Expansion (Weeks 5-6)
**Goal:** Full 5-dimension optimization

```
□ Define per-model base feature sets
□ Implement Dim 2: Feature selection
□ Implement Dim 3: Feature parameters
□ Implement Dim 4: Feature timeframes
□ Build FeatureSpec artifact flow (save + embed)
```

### Phase 4: 4D Models (Weeks 7-8)
**Goal:** Advanced transformer models

```
□ Implement PatchTST
□ Implement iTransformer
□ Implement TFT
□ Implement N-BEATS
```

### Phase 5: Dynamic Ensembles (Weeks 9-11)
**Goal:** MoE and adaptive methods

```
□ Implement MoE gating (MLP + Attention)
□ Implement MoE training loop (OOF-safe)
□ Implement adaptive weighting
□ Implement regime detectors (HMM, rules)
□ Implement regime-routed ensemble
□ Implement uncertainty estimation (MC Dropout, variance)
□ Implement uncertainty-weighted ensemble
□ Implement cascaded/early exit
```

### Phase 6: Polish & Documentation (Week 12)
**Goal:** Production-ready

```
□ End-to-end Colab notebook
□ Comprehensive documentation
□ Example configs for common use cases
□ Performance benchmarks
□ Test coverage > 80%
```

---

## Appendix A: Config Examples

### A.1 Minimal Config (Uses Preset)
```yaml
# my_bear.yaml
preset: balanced
contract: MES
```

### A.2 Preset + Knobs
```yaml
# my_bear.yaml
preset: balanced
contract: MES
horizons: [5, 20]

models:
  add: [lstm]
  remove: [tcn]

optuna:
  n_trials: 100
```

### A.3 MoE Config
```yaml
# moe_bear.yaml
preset: none
contract: MES
horizons: [20]

models:
  - catboost
  - tcn
  - lstm

ensemble:
  method: moe
  gating:
    type: mlp
    features: [realized_vol_20, trend_strength_20, volume_zscore]
    hidden_dims: [64, 32]
    dropout: 0.2

optuna:
  n_trials: 100
  dimensions: [barriers, feature_selection, hyperparameters]
```

### A.4 Regime-Routed Config
```yaml
# regime_bear.yaml
preset: none
contract: MES

regime:
  detector: hmm_volatility
  n_states: 3

models:
  - catboost
  - tcn
  - lstm

ensemble:
  method: regime_routed
  routing:
    state_0:  # Low vol
      models: [catboost]
      meta: none
    state_1:  # Medium vol
      models: [catboost, tcn]
      meta: blending
    state_2:  # High vol
      models: [lstm, tcn]
      meta: stacking
```

### A.5 Adaptive Config
```yaml
# adaptive_bear.yaml
preset: adaptive
contract: MES

ensemble:
  method: adaptive
  decay_halflife: 100
  min_weight: 0.1
  performance_metric: sharpe
  lookback_bars: 500
```

---

## Appendix B: Compatibility Matrix

### B.1 Model → Feature Set Compatibility

| Model | momentum | volatility | volume | micro | wavelets | entropy | raw_ohlcv |
|-------|----------|------------|--------|-------|----------|---------|-----------|
| xgboost | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ |
| lightgbm | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ |
| catboost | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ |
| lstm | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| gru | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| tcn | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| inception | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| patchtst | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| itransformer | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| tft | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| nbeats | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

### B.2 Ensemble → Model Shape Compatibility

| Ensemble | 2D | 3D | 4D | Mixed | Notes |
|----------|----|----|----|----|-------|
| voting | ✅ | ✅ | ✅ | ✅ | All output same shape |
| blending | ✅ | ✅ | ✅ | ✅ | All output same shape |
| stacking | ✅ | ✅ | ✅ | ✅ | OOF unifies shapes |
| moe | ✅ | ✅ | ✅ | ✅ | Gating handles routing |
| adaptive | ✅ | ✅ | ✅ | ✅ | Weights per model |
| regime_routed | ✅ | ✅ | ✅ | ✅ | Routes to subsets |
| uncertainty | ⚠️ | ✅ | ✅ | ⚠️ | Needs uncertainty estimates |
| cascaded | ✅ | ✅ | ✅ | ✅ | Sequential evaluation |

---

*Document serves as single source of truth for ML Factory architecture.*
*Last updated: 2025-01-23*
