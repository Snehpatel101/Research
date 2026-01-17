# ML Factory: Unified Pipeline Refactoring Proposal

**Date:** 2026-01-16
**Analysis:** Sequential specialized agents (ML Engineer, Backend Architect, MLOps Engineer)
**Scope:** Transform two disconnected pipelines into ONE cohesive ML factory

---

## Executive Summary

Your codebase has **all the pieces** for a world-class ML factory:
- Per-model feature strategies (23 models)
- Model family adapters (2D/3D/4D)
- Sophisticated heterogeneous stacking (1280 lines)
- 4 meta-learners (Ridge, MLP, XGBoost, Calibrated)
- Inference bundles with preprocessing graphs

**But they're disconnected.** The codebase contains two pipelines:
1. `phase1/` - Data processing (21 stages)
2. `models/` + `training/` - Model training (orphaned adapters, disconnected inference)

**This proposal reorganizes everything into ONE smooth pipeline** where data flows seamlessly from raw OHLCV to deployed inference bundle.

---

## Table of Contents

1. [The Core Problem: Two Pipelines](#1-the-core-problem-two-pipelines)
2. [The Solution: One Unified Pipeline](#2-the-solution-one-unified-pipeline)
3. [Layer 1: Data & Timeframes](#3-layer-1-data--timeframes)
4. [Layer 2: Features & Per-Model Selection](#4-layer-2-features--per-model-selection)
5. [Layer 3: Model Family Adapters](#5-layer-3-model-family-adapters)
6. [Layer 4: Training & Ensembles](#6-layer-4-training--ensembles)
7. [Layer 5: Meta-Learners](#7-layer-5-meta-learners)
8. [Layer 6: Inference](#8-layer-6-inference)
9. [The Single Configuration Surface](#9-the-single-configuration-surface)
10. [Implementation Roadmap](#10-implementation-roadmap)

---

## 1. The Core Problem: Two Pipelines

### Current Architecture (Disconnected)

```
┌─────────────────────────────────────────────────────────────────────┐
│ PIPELINE 1: phase1/                                                  │
│                                                                      │
│ Raw OHLCV → Resample → Features → Labels → Splits → Container       │
│                                                                      │
│ Outputs: TimeSeriesDataContainer                                     │
│ Problem: Uses prefix-based feature sets (FEATURE_SET_DEFINITIONS)    │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
              [DISCONNECT: Two feature systems]
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PIPELINE 2: models/ + training/                                      │
│                                                                      │
│ TimeSeriesDataContainer → Trainer → Model → Save                    │
│                                                                      │
│ Problem: Uses explicit name lists (MODEL_FEATURE_STRATEGIES)         │
│ Problem: Adapters exist but are ORPHANED (not used by Trainer)       │
│ Problem: Inference bundles not created by Trainer                    │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
              [DISCONNECT: Training doesn't create bundles]
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ ORPHANED: inference/                                                 │
│                                                                      │
│ ModelBundle → InferencePipeline → Predictions                       │
│                                                                      │
│ Problem: Must manually create bundles from training artifacts        │
│ Problem: No meta-learner stacking support                            │
└─────────────────────────────────────────────────────────────────────┘
```

### Specific Disconnections Found

| System A | System B | Disconnect |
|----------|----------|------------|
| `FEATURE_SET_DEFINITIONS` (prefix-based) | `MODEL_FEATURE_STRATEGIES` (explicit names) | Two parallel feature systems |
| `phase1/` feature engineering | `features/strategies.py` | Different naming conventions |
| `adapters/` (TabularAdapter, SequenceAdapter) | `Trainer` | Trainer bypasses adapters entirely |
| `Trainer.run()` | `ModelBundle` | Training doesn't create inference bundles |
| `StackingEnsemble` (meta-learners) | `InferencePipeline` | No meta-learner inference support |

---

## 2. The Solution: One Unified Pipeline

### New Architecture (Cohesive)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ONE UNIFIED PIPELINE                              │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ LAYER 1: Data & Timeframes                                    │   │
│  │                                                               │   │
│  │ Raw 1-min OHLCV → Resample → 9 Timeframe Artifacts           │   │
│  │ (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              ↓                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ LAYER 2: Features & Per-Model Selection                       │   │
│  │                                                               │   │
│  │ 180+ Features Generated → FEATURE_REGISTRY (unified)         │   │
│  │                              ↓                                │   │
│  │ MODEL_FEATURE_STRATEGIES → Baseline per model (23 configs)   │   │
│  │                              ↓                                │   │
│  │ FeatureOptimizer → Optuna pruning (baseline → optimal)       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              ↓                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ LAYER 3: Model Family Adapters                                │   │
│  │                                                               │   │
│  │ AdapterRegistry.get_for_model() → Correct data format        │   │
│  │   ├─ TabularAdapter   → (n_samples, n_features)      2D     │   │
│  │   ├─ SequenceAdapter  → (n_samples, seq_len, n_feat) 3D     │   │
│  │   └─ MultiStreamAdapter → (n, n_tf, seq_len, n_feat) 4D     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              ↓                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ LAYER 4: Training & Ensembles                                 │   │
│  │                                                               │   │
│  │ TrainingOrchestrator → Per-model training                    │   │
│  │                              ↓                                │   │
│  │ OOF Generation → PurgedKFold → OOF predictions               │   │
│  │                              ↓                                │   │
│  │ StackingDatasetBuilder → Aligned OOF for meta-learner        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              ↓                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ LAYER 5: Meta-Learners                                        │   │
│  │                                                               │   │
│  │ 4 Meta-Learners: Ridge, MLP, XGBoost, Calibrated             │   │
│  │   ├─ Input: Stacked OOF probabilities (always 2D)            │   │
│  │   └─ Output: Final ensemble prediction                        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              ↓                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ LAYER 6: Inference                                            │   │
│  │                                                               │   │
│  │ Trainer → ModelBundle (automatic)                            │   │
│  │   ├─ model, scaler, calibrator, feature_columns              │   │
│  │   └─ preprocessing_graph (raw OHLCV → features)              │   │
│  │                              ↓                                │   │
│  │ InferencePipeline → Real-time + Batch predictions            │   │
│  │   └─ MetaLearnerInference (NEW) → Heterogeneous ensembles    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Layer 1: Data & Timeframes

### Current State
- 9 intraday timeframes supported (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)
- All resampled from canonical 1-min OHLCV
- **Working correctly** - no changes needed

### Data Flow

```
data/raw/MES_1min.parquet (canonical source)
         ↓
    [Resampler]
         ↓
data/timeframes/MES/
    ├── 1min.parquet
    ├── 5min.parquet
    ├── 10min.parquet
    ├── 15min.parquet
    ├── 20min.parquet
    ├── 25min.parquet
    ├── 30min.parquet
    ├── 45min.parquet
    └── 60min.parquet
```

### Per-Model Timeframe Selection

Each model independently chooses its primary timeframe:

| Model | Primary TF | Derived From |
|-------|-----------|--------------|
| XGBoost | 15min | 1-min canonical |
| LSTM | 5min | 1-min canonical |
| PatchTST | 1min | 1-min canonical (direct) |
| TCN | 5min | 1-min canonical |

---

## 4. Layer 2: Features & Per-Model Selection

### The Problem: Two Parallel Feature Systems

**System 1: `phase1/config/feature_sets/` (prefix-based)**
```python
# Uses include_prefixes to match features
FeatureSetDefinition(
    name="boosting_optimal",
    include_prefixes=["return_", "rsi_", "macd_", ...],
    exclude_prefixes=["sma_", "ema_", ...],
)
```

**System 2: `src/features/strategies.py` (explicit names)**
```python
# Uses explicit feature name lists
MODEL_FEATURE_STRATEGIES["xgboost"] = ModelFeatureStrategy(
    baseline_features=["returns", "rsi_14", "macd_line", ...],  # ~100 features
)
```

**These don't communicate.** Phase1 generates `return_1`, `return_5` but strategies expects `returns`.

### The Solution: Unified Feature Registry

```
┌────────────────────────────────────────────────────────────────────┐
│ UNIFIED FEATURE SYSTEM                                              │
│                                                                     │
│ 1. Phase1 generates ~180 features with consistent naming           │
│    └─ output: features_{tf}.parquet with columns like:             │
│       rsi_14, macd_line, atr_14, bb_upper, volume_sma_20, etc.    │
│                                                                     │
│ 2. FEATURE_REGISTRY (complete, 180+ entries)                       │
│    └─ Maps each feature to: family, recommended_models, cost       │
│                                                                     │
│ 3. MODEL_FEATURE_STRATEGIES (references families, not names)       │
│    └─ Each model: baseline_families + mtf_mode + optimization      │
│                                                                     │
│ 4. FeatureStrategyManager resolves families → actual columns       │
│    └─ get_features_for_model("xgboost", df) → ~100 columns        │
│                                                                     │
│ 5. FeatureOptimizer (optional) prunes baseline → optimal           │
│    └─ Optuna-based pruning: ~100 → ~60 features                    │
└────────────────────────────────────────────────────────────────────┘
```

### Per-Model Feature Strategies (All 23 Models)

| Model | Family | Baseline Features | MTF Mode | Min/Max |
|-------|--------|-------------------|----------|---------|
| **Boosting (3)** |
| xgboost | boosting | momentum + volatility + volume + microstructure + MTF | indicators | 40/120 |
| lightgbm | boosting | momentum + volatility + volume + microstructure + MTF | indicators | 40/120 |
| catboost | boosting | momentum + volatility + volume + microstructure + MTF | indicators | 40/120 |
| **Classical (3)** |
| random_forest | classical | momentum + volatility + volume | none | 30/80 |
| logistic | classical | momentum + volatility (low corr) | none | 20/50 |
| svm | classical | momentum + volatility (normalized) | none | 20/50 |
| **Neural RNN (2)** |
| lstm | neural | momentum + volatility + wavelets + MTF | indicators | 50/150 |
| gru | neural | momentum + volatility + wavelets + MTF | indicators | 50/150 |
| **Neural CNN (3)** |
| tcn | cnn | momentum + volatility + volume | none | 30/100 |
| inceptiontime | cnn | momentum + volatility | none | 30/80 |
| resnet1d | cnn | momentum + volatility + volume | none | 30/100 |
| **Transformers (5)** |
| transformer | transformer | momentum + volatility + wavelets | indicators | 40/100 |
| patchtst | transformer | raw_ohlcv only | multi_stream | 4/10 |
| itransformer | transformer | raw_ohlcv only | multi_stream | 4/10 |
| tft | transformer | momentum + volatility + regime | indicators | 40/120 |
| nbeats | transformer | close + volume only | none | 2/10 |
| **Meta-Learners (4)** |
| ridge_meta | meta_learner | OOF probabilities | none | N/A |
| mlp_meta | meta_learner | OOF probabilities | none | N/A |
| xgboost_meta | meta_learner | OOF probabilities | none | N/A |
| calibrated_meta | meta_learner | OOF probabilities | none | N/A |
| **Ensembles (3)** |
| voting | ensemble | (uses base model features) | varies | N/A |
| stacking | ensemble | (uses base model features) | varies | N/A |
| blending | ensemble | (uses base model features) | varies | N/A |

### Feature Flow Diagram

```
Phase1 generates 180+ features
         ↓
┌────────────────────────────────────────────────────────────────┐
│ Per-Model Feature Selection                                     │
│                                                                 │
│ User: "Train XGBoost"                                          │
│         ↓                                                       │
│ FeatureStrategyManager.get_features_for_model("xgboost", df)   │
│         ↓                                                       │
│ Lookup MODEL_FEATURE_STRATEGIES["xgboost"]                     │
│   baseline_families = ["momentum", "volatility", "volume",     │
│                        "microstructure", "mtf"]                │
│         ↓                                                       │
│ Resolve families → actual columns via FEATURE_REGISTRY         │
│   momentum → [rsi_14, macd_line, macd_signal, stoch_k, ...]   │
│   volatility → [atr_14, bb_upper, bb_lower, bb_width, ...]    │
│   etc.                                                          │
│         ↓                                                       │
│ Filter to columns present in df                                │
│         ↓                                                       │
│ Returns: ~100 feature columns for XGBoost                      │
└────────────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────────┐
│ Optional: Feature Optimization                                  │
│                                                                 │
│ FeatureOptimizer("xgboost", n_trials=30).optimize(...)         │
│         ↓                                                       │
│ Optuna suggests feature subsets                                │
│ Trains model, evaluates F1, selects best                       │
│         ↓                                                       │
│ Returns: ~60 optimized features (from ~100 baseline)           │
└────────────────────────────────────────────────────────────────┘
         ↓
Final feature set for XGBoost training
```

---

## 5. Layer 3: Model Family Adapters

### Current State: Orphaned But Well-Designed

The adapter system exists in `src/adapters/` with excellent implementations:
- `TabularAdapter` - 2D arrays for boosting/classical
- `SequenceAdapter` - 3D sequences for RNN/CNN
- `MultiStreamAdapter` - 4D multi-TF for transformers

**But Trainer doesn't use them.** The Trainer bypasses adapters entirely.

### The Solution: Adapter-Aware Training

```
┌────────────────────────────────────────────────────────────────┐
│ ADAPTER-AWARE TRAINING FLOW                                     │
│                                                                 │
│ 1. Model selected: "lstm"                                      │
│         ↓                                                       │
│ 2. Get model contract:                                         │
│    ModelContract.for_model("lstm")                             │
│    → input_rank: SEQUENCE_3D                                   │
│    → sequence_length: 60                                       │
│         ↓                                                       │
│ 3. Get appropriate adapter:                                    │
│    AdapterRegistry.get_for_model("lstm")                       │
│    → SequenceAdapter(sequence_length=60)                       │
│         ↓                                                       │
│ 4. Get model's feature columns:                                │
│    FeatureStrategyManager.get_features_for_model("lstm", df)   │
│    → ~80 feature columns                                       │
│         ↓                                                       │
│ 5. Transform data:                                             │
│    adapter = SequenceAdapter(                                  │
│        sequence_length=60,                                     │
│        feature_columns=feature_cols,                           │
│    )                                                           │
│    result = adapter.transform(df)                              │
│    → AdapterResult(                                            │
│        X: (n_samples, 60, 80),  # 3D                          │
│        y: (n_samples,),                                        │
│        data_rank: SEQUENCE_3D,                                 │
│        original_indices: [...],                                │
│      )                                                         │
│         ↓                                                       │
│ 6. Train model:                                                │
│    model.fit(result.X, result.y, ...)                         │
└────────────────────────────────────────────────────────────────┘
```

### Adapter Mapping

| Model | Family | Adapter | Output Shape |
|-------|--------|---------|--------------|
| xgboost | boosting | TabularAdapter | (n, ~100) |
| lightgbm | boosting | TabularAdapter | (n, ~100) |
| catboost | boosting | TabularAdapter | (n, ~100) |
| random_forest | classical | TabularAdapter | (n, ~80) |
| logistic | classical | TabularAdapter | (n, ~50) |
| svm | classical | TabularAdapter | (n, ~50) |
| lstm | neural | SequenceAdapter | (n, 60, ~80) |
| gru | neural | SequenceAdapter | (n, 60, ~80) |
| tcn | cnn | SequenceAdapter | (n, 60, ~60) |
| inceptiontime | cnn | SequenceAdapter | (n, 60, ~60) |
| resnet1d | cnn | SequenceAdapter | (n, 60, ~60) |
| transformer | transformer | SequenceAdapter | (n, 60, ~80) |
| patchtst | transformer | **MultiStreamAdapter** | (n, 3, 60, 5) |
| itransformer | transformer | **MultiStreamAdapter** | (n, 3, 60, 5) |
| tft | transformer | SequenceAdapter | (n, 60, ~100) |
| nbeats | transformer | SequenceAdapter | (n, 60, 2) |

### Heterogeneous Ensemble Adaptation

For stacking ensembles with mixed model types:

```
Heterogeneous Stacking: XGBoost + LSTM + PatchTST
         ↓
┌────────────────────────────────────────────────────────────────┐
│ PARALLEL ADAPTATION                                             │
│                                                                 │
│ XGBoost path:                                                  │
│   TabularAdapter → (n_samples, ~100)                           │
│   Train XGBoost on 2D                                          │
│   Generate OOF: (n_samples, 3) probabilities                   │
│                                                                 │
│ LSTM path:                                                     │
│   SequenceAdapter → (n_samples - 59, 60, ~80)                  │
│   Train LSTM on 3D                                             │
│   Generate OOF: (n_samples - 59, 3) probabilities              │
│   NOTE: Loses 59 samples at start due to lookback              │
│                                                                 │
│ PatchTST path:                                                 │
│   MultiStreamAdapter → (n_samples - 59, 3, 60, 5)              │
│   Train PatchTST on 4D                                         │
│   Generate OOF: (n_samples - 59, 3) probabilities              │
└────────────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────────┐
│ OOF ALIGNMENT (OOFAlignmentValidator)                          │
│                                                                 │
│ XGBoost OOF: indices [0, n)                                    │
│ LSTM OOF: indices [59, n)                                      │
│ PatchTST OOF: indices [59, n)                                  │
│                                                                 │
│ Common range: [59, n) → n_common samples                       │
│                                                                 │
│ Aligned OOF stack: (n_common, 3 models × 3 classes) = (_, 9)   │
└────────────────────────────────────────────────────────────────┘
         ↓
Meta-learner trains on aligned 2D OOF
```

---

## 6. Layer 4: Training & Ensembles

### Training Orchestrator Flow

```
ExperimentConfig
    ↓
┌────────────────────────────────────────────────────────────────┐
│ TrainingOrchestrator.run()                                      │
│                                                                 │
│ 1. Validate config                                             │
│ 2. Load data for each (symbol, horizon, timeframe)             │
│ 3. For each model in config.models:                            │
│    a. Get feature strategy                                     │
│    b. Get adapter                                              │
│    c. Transform data                                           │
│    d. (Optional) Optimize features via Optuna                  │
│    e. Train model                                              │
│    f. Generate OOF predictions                                 │
│    g. Store results                                            │
│ 4. If config.build_ensemble:                                   │
│    a. Build stacking dataset from OOF                          │
│    b. Train meta-learner                                       │
│ 5. Create ModelBundles                                         │
│ 6. Save all artifacts                                          │
└────────────────────────────────────────────────────────────────┘
```

### OOF Generation for Stacking

```
┌────────────────────────────────────────────────────────────────┐
│ OOF GENERATION (per model)                                      │
│                                                                 │
│ PurgedKFold (n_folds=5, purge=60, embargo=1440)               │
│                                                                 │
│ For each fold:                                                 │
│   1. Split data with purge gap and embargo period              │
│   2. Scale train data only (leakage prevention)                │
│   3. Train model on fold_train                                 │
│   4. Predict on fold_val                                       │
│   5. Store predictions at fold_val indices                     │
│                                                                 │
│ Output: OOFPrediction                                          │
│   predictions: DataFrame with columns:                         │
│     - {model}_prob_short, {model}_prob_neutral, {model}_prob_long │
│     - {model}_pred, {model}_confidence                         │
│   fold_info: per-fold metrics                                  │
│   coverage: fraction with predictions                          │
└────────────────────────────────────────────────────────────────┘
```

### Three Ensemble Methods

| Method | Homogeneous Only | Description |
|--------|------------------|-------------|
| **Voting** | Yes | Simple soft/hard vote (no meta-learner) |
| **Blending** | Yes | Holdout-based (simpler than stacking) |
| **Stacking** | **No** | OOF-based with meta-learner (**supports heterogeneous**) |

---

## 7. Layer 5: Meta-Learners

### The Four Meta-Learners

All meta-learners receive OOF predictions as input (always 2D, regardless of base model type):

```
┌────────────────────────────────────────────────────────────────┐
│ META-LEARNER INPUT                                              │
│                                                                 │
│ Stacked OOF: (n_common, n_models × 3)                          │
│                                                                 │
│ Example: 3 base models → (n_common, 9) input                   │
│   [xgb_prob_short, xgb_prob_neutral, xgb_prob_long,            │
│    lstm_prob_short, lstm_prob_neutral, lstm_prob_long,         │
│    ptst_prob_short, ptst_prob_neutral, ptst_prob_long]         │
│                                                                 │
│ + Derived features (optional):                                 │
│   models_agree, agreement_count, avg_confidence,               │
│   min_confidence, max_confidence, {model}_entropy, avg_entropy │
└────────────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────────┐
│ META-LEARNER SELECTION                                          │
│                                                                 │
│ ridge_meta:                                                    │
│   Method: L2-regularized Ridge classifier                      │
│   Strength: Fast, interpretable, robust to multicollinearity   │
│   Use case: Well-calibrated base models                        │
│                                                                 │
│ mlp_meta:                                                      │
│   Method: Shallow neural network (32, 16)                      │
│   Strength: Non-linear interactions                            │
│   Use case: Complementary error patterns                       │
│                                                                 │
│ xgboost_meta:                                                  │
│   Method: Shallow XGBoost (max_depth=3)                        │
│   Strength: Complex non-linear, feature importance             │
│   Use case: Diverse error patterns                             │
│                                                                 │
│ calibrated_meta:                                               │
│   Method: Logistic + Isotonic calibration                      │
│   Strength: Well-calibrated probabilities                      │
│   Use case: Threshold-based trading decisions                  │
└────────────────────────────────────────────────────────────────┘
         ↓
Final ensemble prediction
```

### Meta-Learner Flow

```
Base Models (trained on OOF folds)
    ├── XGBoost → OOF probs (n, 3)
    ├── LSTM → OOF probs (n-59, 3)
    └── PatchTST → OOF probs (n-59, 3)
         ↓
OOF Alignment → Common range [59, n)
         ↓
StackingDatasetBuilder → (n_common, 9)
         ↓
Meta-Learner (e.g., ridge_meta)
    ├── Train on OOF
    └── Validate on val set predictions
         ↓
Full retrain base models on all training data
         ↓
Inference: base predictions → meta-learner → final prediction
```

---

## 8. Layer 6: Inference

### Current Gap: Training Doesn't Create Bundles

**Current flow (broken):**
```
Trainer.run()
    → saves model to checkpoints/best_model/
    → saves scaler separately
    → saves feature_selection.json
    → DOES NOT create ModelBundle

User must manually:
    → Locate all artifacts
    → Create ModelBundle.from_training()
    → bundle.save()
```

### The Solution: Automatic Bundle Creation

```
┌────────────────────────────────────────────────────────────────┐
│ AUTOMATIC BUNDLE CREATION                                       │
│                                                                 │
│ Trainer.run()                                                  │
│     ↓                                                          │
│ train model                                                    │
│     ↓                                                          │
│ fit calibrator                                                 │
│     ↓                                                          │
│ TrainerArtifactsMixin._save_all_artifacts()                    │
│     ↓                                                          │
│ ModelBundle.from_training(                                     │
│     model=self.model,                                          │
│     scaler=self.scaler,                                        │
│     feature_columns=self._feature_set_columns,                 │
│     horizon=self.config.horizon,                               │
│     calibrator=self.calibrator,                                │
│     preprocessing_graph=self._create_preprocessing_graph(),    │
│     symbol=self.config.symbol,                                 │
│     training_metrics=metrics_dict,                             │
│ )                                                              │
│     ↓                                                          │
│ bundle.save(self.output_path / "bundle")                       │
└────────────────────────────────────────────────────────────────┘
```

### ModelBundle Contents

```
bundle_dir/
    manifest.json               # File listing + MD5 checksums
    metadata.json               # Model info (name, family, horizon, features)
    features.json               # Ordered feature column names
    scaler.pkl                  # Fitted scaler
    calibrator.pkl              # Fitted probability calibrator [optional]
    preprocessing_graph.json    # Raw OHLCV → features config [optional]
    model/                      # Model artifacts
```

### Inference Pipeline (Including Meta-Learners)

```
┌────────────────────────────────────────────────────────────────┐
│ INFERENCE MODES                                                 │
│                                                                 │
│ Mode 1: Single Model                                           │
│   bundle = ModelBundle.load("./bundles/xgb_h20")               │
│   pipeline = InferencePipeline([bundle])                       │
│   result = pipeline.predict(X_new)                             │
│                                                                 │
│ Mode 2: Homogeneous Voting (existing)                          │
│   bundles = [xgb_bundle, lgbm_bundle, catboost_bundle]         │
│   pipeline = InferencePipeline(bundles)                        │
│   result = pipeline.predict_ensemble(X_new, method="soft_vote")│
│                                                                 │
│ Mode 3: Heterogeneous Stacking (NEW)                           │
│   pipeline = MetaLearnerInferencePipeline(                     │
│       base_bundles=[xgb_bundle, lstm_bundle, ptst_bundle],     │
│       meta_learner_bundle=ridge_meta_bundle,                   │
│   )                                                            │
│   result = pipeline.predict_stacking(X_new, X_seq=X_new_3d)    │
│                                                                 │
│ Mode 4: Raw OHLCV End-to-End                                   │
│   bundle = ModelBundle.load("./bundles/xgb_h20_with_graph")    │
│   raw_df = pd.read_parquet("live_data/recent_bars.parquet")    │
│   result = bundle.predict_from_raw(raw_df)                     │
└────────────────────────────────────────────────────────────────┘
```

### New: MetaLearnerInferencePipeline

```python
class MetaLearnerInferencePipeline:
    """Inference for heterogeneous stacking ensembles."""

    def __init__(
        self,
        base_bundles: list[ModelBundle],
        meta_learner_bundle: ModelBundle,
    ):
        self.base_bundles = base_bundles
        self.meta_learner_bundle = meta_learner_bundle

        # Separate tabular vs sequence bundles
        self.tabular_bundles = [b for b in base_bundles if not b.requires_sequences]
        self.sequence_bundles = [b for b in base_bundles if b.requires_sequences]

    def predict_stacking(
        self,
        X: np.ndarray,                    # 2D for tabular
        X_seq: np.ndarray | None = None,  # 3D for sequence
    ) -> PredictionOutput:
        # 1. Get predictions from all base models
        base_probs = []
        for bundle in self.tabular_bundles:
            probs = bundle.predict(X).class_probabilities
            base_probs.append(probs)
        for bundle in self.sequence_bundles:
            probs = bundle.predict(X_seq).class_probabilities
            base_probs.append(probs)

        # 2. Stack base predictions
        stacked = np.hstack(base_probs)  # (n_samples, n_models * 3)

        # 3. Meta-learner predicts final output
        return self.meta_learner_bundle.predict(stacked)
```

---

## 9. The Single Configuration Surface

### User Experience: 10 Lines of Config

```python
from src.training import TrainingOrchestrator, ExperimentConfig, ModelConfig

config = ExperimentConfig(
    symbol="MES",
    horizons=[20],
    models=[
        ModelConfig(name="xgboost", timeframe="15min", optimize_features=True),
        ModelConfig(name="lstm", timeframe="5min", sequence_length=60),
        ModelConfig(name="patchtst", timeframe="1min"),
    ],
    build_ensemble=True,
    meta_learner="ridge_meta",
)

orchestrator = TrainingOrchestrator(config)
results = orchestrator.run()
orchestrator.display_results()
```

**System automatically:**
1. Derives feature requirements from model strategies
2. Gets appropriate adapters (2D, 3D, 4D)
3. Transforms data per model
4. Trains base models
5. Generates OOF with PurgedKFold
6. Aligns OOF for heterogeneous models
7. Trains meta-learner on stacked OOF
8. Creates ModelBundles for deployment
9. Saves inference bundles

### Configuration Dataclasses

```python
@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str                           # "xgboost", "lstm", "patchtst"
    timeframe: str = "5min"            # Primary training timeframe
    sequence_length: int | None = None  # For sequence models
    optimize_features: bool = False     # Run Optuna pruning?
    feature_opt_trials: int = 30
    hyperparams: dict = field(default_factory=dict)

@dataclass
class ExperimentConfig:
    """Single source of truth for ML experiments."""

    # ========== REQUIRED ==========
    models: list[ModelConfig]           # Which models to train
    symbol: str                         # "MES", "MGC"

    # ========== DATA ==========
    horizons: list[int] = field(default_factory=lambda: [20])

    # ========== ENSEMBLE ==========
    build_ensemble: bool = False
    meta_learner: str = "ridge_meta"    # ridge_meta, mlp_meta, xgboost_meta, calibrated_meta

    # ========== TRAINING MODE ==========
    training_mode: str = "standard"     # standard, walk_forward, regime_aware

    # ========== ARTIFACTS ==========
    run_id: str | None = None           # Auto-generated if None
    save_bundles: bool = True           # Create inference bundles
```

---

## 10. Implementation Roadmap

### Phase 1: Unify Feature System (Week 1-2)

**Goal:** Merge two parallel feature systems into one.

**Tasks:**
- [ ] Complete FEATURE_REGISTRY with all 180+ features
- [ ] Change MODEL_FEATURE_STRATEGIES to reference families (not explicit names)
- [ ] Update FeatureStrategyManager to resolve families → columns
- [ ] Fix optimize_features_for_model() bug (user_attrs extraction)
- [ ] Deprecate FEATURE_SET_DEFINITIONS (phase1 prefix system)
- [ ] Tests for unified feature resolution

**Deliverables:**
- `src/features/registry.py` - Complete 180+ feature registry
- `src/features/strategies.py` - Family-based strategies
- Migration guide for existing experiments

---

### Phase 2: Integrate Adapters into Trainer (Week 3-4)

**Goal:** Trainer uses adapters instead of bypassing them.

**Tasks:**
- [ ] Refactor Trainer to accept AdapterResult
- [ ] Route data through AdapterRegistry based on model
- [ ] Pass feature_columns to adapter from feature strategy
- [ ] Handle heterogeneous ensembles in StackingEnsemble (dual data loading)
- [ ] Fix ModelContract.input_rank for patchtst/itransformer (should be 4D)
- [ ] Tests for adapter integration

**Deliverables:**
- `src/models/training/trainer.py` - Adapter-aware training
- Updated ModelContracts with correct ranks

---

### Phase 3: Automatic Bundle Creation (Week 5-6)

**Goal:** Trainer automatically creates inference bundles.

**Tasks:**
- [ ] Add bundle creation to TrainerArtifactsMixin
- [ ] Create preprocessing_graph from pipeline config
- [ ] Add bundle validation and checksums
- [ ] Create bundle CLI script
- [ ] Tests for bundle creation and loading

**Deliverables:**
- `src/models/training/artifacts.py` - Auto bundle creation
- `scripts/create_bundle.py` - CLI for manual bundling

---

### Phase 4: Meta-Learner Inference (Week 7-8)

**Goal:** InferencePipeline supports trained meta-learners.

**Tasks:**
- [ ] Implement MetaLearnerInferencePipeline class
- [ ] Handle heterogeneous base bundles (2D + 3D)
- [ ] Add stacking bundle format (base bundles + meta bundle)
- [ ] Tests for meta-learner inference

**Deliverables:**
- `src/inference/meta_learner_pipeline.py` - Heterogeneous inference
- `src/inference/stacking_bundle.py` - Stacking bundle format

---

### Phase 5: Cleanup & Documentation (Week 9-10)

**Goal:** Remove legacy code, update docs.

**Tasks:**
- [ ] Remove FEATURE_SET_DEFINITIONS (phase1 prefix system)
- [ ] Remove duplicate feature resolution code
- [ ] Update CLAUDE.md with unified architecture
- [ ] Update notebooks for new API
- [ ] Migration script for existing experiments

**Deliverables:**
- Clean codebase with single feature system
- Updated documentation
- Migration guide

---

## Summary: Before vs After

| Aspect | Before (Disconnected) | After (Unified) |
|--------|----------------------|-----------------|
| Feature systems | 2 parallel (prefix + explicit) | 1 unified (family-based) |
| Adapters | Orphaned (Trainer bypasses) | Integrated (Trainer uses) |
| Bundle creation | Manual | Automatic |
| Meta-learner inference | Not supported | Fully supported |
| Config complexity | 85 classes, 200+ fields | 2 dataclasses, ~15 fields |
| User experience | 50+ lines config | 10 lines config |

**Result:** ONE smooth, cohesive pipeline where a beginner can write:

```python
config = ExperimentConfig(
    models=[
        ModelConfig(name="lstm"),
        ModelConfig(name="tcn"),
        ModelConfig(name="gru"),
    ],
    symbol="MES",
    build_ensemble=True,
    meta_learner="ridge_meta",
)
orchestrator = TrainingOrchestrator(config)
results = orchestrator.run()  # Just works!
```

And the system handles everything: feature selection, data adaptation, training, OOF generation, meta-learner training, and inference bundle creation.

---

**Document Version:** 2.0
**Last Updated:** 2026-01-16
**Analysis By:** Sequential Specialized Agents (ML Engineer, Backend Architect, MLOps Engineer)
