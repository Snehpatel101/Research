# Target Architecture: ONE Unified Pipeline

## Core Principle

**Single pipeline that ingests canonical OHLCV and deterministically derives model-specific representations.**

**NOT separate pipelines** - ONE workflow with adapters.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CANONICAL DATASET (SINGLE SOURCE)                │
│                                                                     │
│  Raw OHLCV → MTF → Features → Labels → Splits → Scaling           │
│  Output: data/splits/scaled/{symbol}_{split}.parquet               │
│  (~180 features, 70/15/15 splits, purge/embargo applied)           │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     MODEL-FAMILY ADAPTERS                           │
│                     (Deterministic Transformations)                 │
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐           │
│  │   Tabular    │   │   Sequence   │   │  Multi-Res   │           │
│  │   Adapter    │   │   Adapter    │   │   Adapter    │           │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘           │
│         ↓                  ↓                  ↓                     │
│    2D Arrays          3D Windows         4D Tensors                │
│    (N, 180)           (N, T, 180)        (N, 9, T, 4)              │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING (PHASE 6)                       │
│                                                                     │
│  Tabular (2D)         Sequence (3D)       Multi-Res (4D)           │
│  ├─ XGBoost           ├─ LSTM             ├─ PatchTST             │
│  ├─ LightGBM          ├─ GRU              ├─ iTransformer         │
│  ├─ CatBoost          ├─ TCN              ├─ TFT                  │
│  ├─ Random Forest     ├─ Transformer      ├─ N-BEATS              │
│  ├─ Logistic          ├─ InceptionTime    └─ ...                  │
│  └─ SVM               └─ 1D ResNet                                 │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│         HETEROGENEOUS BASE MODELS (PHASE 6)                        │
│                                                                     │
│  Select 1 model per family (3-4 total):                            │
│  ├→ Tabular: CatBoost (engineered features)                        │
│  ├→ CNN/TCN: TCN (local patterns)                                  │
│  ├→ Transformer: PatchTST (long context)                           │
│  └→ Optional 4th: Ridge (linear baseline)                          │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│         META-LEARNER STACKING (PHASE 7)                            │
│                                                                     │
│  OOF Predictions → Logistic/Ridge/MLP → Final Predictions         │
│  (Direct stacking from heterogeneous bases, no same-family rule)  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### 1. Canonical Dataset (Single Source of Truth)

**Input:** Raw OHLCV data (`data/raw/{symbol}_1m.parquet`)

**Processing Pipeline:**
1. **Phase 1:** Ingest + validate + clean
2. **Phase 2:** MTF upscaling (9 timeframes: 1m → 1h)
3. **Phase 3:** Feature engineering (~180 features)
4. **Phase 4:** Triple-barrier labeling + splits + scaling

**Output:** `data/splits/scaled/{symbol}_{split}.parquet`
- Train/val/test splits (70/15/15)
- ~180 indicator-derived features
- Triple-barrier labels (3 classes: long/short/neutral)
- Quality-weighted samples
- Robust scaling (train-only fit)

**Key Properties:**
- **Deterministic:** Same input → same output
- **Leakage-free:** Purge (60) + embargo (1440) applied
- **Single storage:** Stored once, consumed by all models
- **Reproducible:** All models train on identical data

---

### 2. Model-Family Adapters (Deterministic Transformations)

**Purpose:** Transform canonical dataset to model-specific formats.

**Three Adapter Types:**

| Adapter | Output Shape | Model Families | Status |
|---------|--------------|----------------|--------|
| **Tabular** | 2D `(N, F)` | Boosting, Classical | ✅ Complete |
| **Sequence** | 3D `(N, T, F)` | Neural, CNN | ✅ Complete |
| **Multi-Resolution** | 4D `(N, TF, T, 4)` | Advanced (PatchTST, etc.) | ❌ Planned |

**Adapter Responsibilities:**
- Read canonical dataset from `data/splits/scaled/`
- **Select model-specific features** (different models get different feature sets)
- Transform to model-appropriate shape (2D, 3D, 4D)
- **Deterministic** (same input → same output)
- Return `TimeSeriesDataContainer` (in-memory)

**Feature Selection by Model:**
- **Tabular models:** ~200 engineered features (indicators, wavelets, MTF indicators from multiple TFs)
- **Sequence models:** ~150 base features (indicators, wavelets, single primary TF)
- **Advanced transformers:** Raw OHLCV bars from multiple timeframes (multi-stream ingestion)

**See:** `.serena/knowledge/per_model_feature_selection.md` for detailed per-model feature strategies

**Example: Tabular Adapter**
```python
# Read canonical dataset
df = pd.read_parquet("data/splits/scaled/MES_train.parquet")

# Extract 2D arrays
X = df[feature_cols].values  # (N, 180)
y = df["label"].values        # (N,)

# Return container
return TimeSeriesDataContainer(X_train=X, y_train=y, ...)
```

**Example: Sequence Adapter**
```python
# Read canonical dataset
df = pd.read_parquet("data/splits/scaled/MES_train.parquet")

# Create 3D windows (lookback = 30 bars)
X = create_windows(df[feature_cols].values, seq_len=30)  # (N, 30, 180)
y = df["label"].values[30:]  # (N,) - skip first 30 samples

# Return container
return TimeSeriesDataContainer(X_train=X, y_train=y, ...)
```

**Example: Multi-Resolution Adapter (Planned)**
```python
# Read raw MTF OHLCV (9 timeframes)
mtf_data = load_mtf_ohlcv(timeframes=[1, 5, 10, 15, 20, 25, 30, 45, 60])

# Build 4D tensor (N, 9, T, 4)
X = build_multiresolution_tensor(mtf_data, lookback_window=60)

# Return container
return TimeSeriesDataContainer(X_train=X, y_train=y, ...)
```

---

### 3. Model Training (Plugin-Based Registry)

**Plugin System:**
```python
from src.models import register, BaseModel

@register(name="my_model", family="boosting")
class MyModel(BaseModel):
    def fit(self, X_train, y_train, X_val, y_val, ...):
        # Training logic
        pass

    def predict(self, X):
        # Prediction logic
        pass
```

**Automatic Discovery:**
- Models register themselves via `@register` decorator
- `ModelRegistry.list_all()` returns all available models
- CLI automatically supports new models

**Model Families:**

| Family | Models | Input Shape | Status |
|--------|--------|-------------|--------|
| **Boosting** | XGBoost, LightGBM, CatBoost | 2D `(N, 180)` | ✅ Complete (3) |
| **Classical** | Random Forest, Logistic, SVM | 2D `(N, 180)` | ✅ Complete (3) |
| **Neural** | LSTM, GRU, TCN, Transformer | 3D `(N, T, 180)` | ✅ Complete (4) |
| **CNN** | InceptionTime, 1D ResNet | 3D `(N, T, 180)` | ❌ Planned (2) |
| **Advanced** | PatchTST, iTransformer, TFT | 4D `(N, 9, T, 4)` | ❌ Planned (3) |
| **MLP** | N-BEATS | 3D/4D | ❌ Planned (1) |
| **Ensemble** | Voting, Stacking, Blending | Mixed | ✅ Complete (3) |

**Total:** 19 models (13 implemented, 6 planned)

---

### 4. Heterogeneous Ensemble Support

**Architecture:** Select 1 model per family (3-4 total) for maximum diversity.

**Recommended Configurations:**
```python
# 3-base standard (recommended)
["catboost", "tcn", "patchtst"]

# 4-base maximum diversity
["lightgbm", "tcn", "tft", "ridge"]

# 2-base minimal (prototyping)
["xgboost", "lstm"]
```

**Training Protocol:**
1. Generate OOF predictions for each base model (PurgedKFold)
2. Stack OOF predictions as meta-features
3. Train meta-learner (Logistic/Ridge/MLP) on stacked OOF
4. Full retrain base models on complete training set
5. Test evaluation: meta-learner combines base predictions

**Meta-Learner Options:**

| Meta-Learner | Method | Use Case |
|--------------|--------|----------|
| **Logistic** | L2-regularized stacking | Default, calibrated probabilities |
| **Ridge** | L2-regularized regression | Continuous predictions |
| **Small MLP** | 1-2 hidden layers | Learned non-linear blending |
| **Calibrated Blender** | Voting + temperature scaling | Calibrated confidence scores |

---

### 5. Why Heterogeneous > Homogeneous

**Diversity of Inductive Biases:**
- Tabular models excel at feature interactions and engineered indicators
- CNN/TCN models capture local temporal patterns and multi-scale features
- Transformers capture long-range dependencies and global context

**Reduced Error Correlation:**
- Errors from diverse model families are less correlated
- Meta-learner can learn when to trust each base model
- Overall ensemble is more robust than any single family

**Comparison:**

| Ensemble Type | Base Selection | Error Correlation | Diversity |
|---------------|----------------|-------------------|-----------|
| **Homogeneous** | Same family (XGB+LGB+Cat) | High | Low |
| **Heterogeneous** | Different families (Cat+TCN+PatchTST) | Low | High |

---

## Key Architectural Principles

### 1. Single Source of Truth
- Canonical dataset stored **once** in `data/splits/scaled/`
- All models consume the **same data**
- Adapters transform deterministically (no feature engineering)

**Why:**
- Reproducibility (same data for all models)
- Storage efficiency (store data once, not per model)
- Consistency (no divergence between model inputs)

---

### 2. Deterministic Adapters
- Adapters are **pure functions** (same input → same output)
- No randomness, no side effects, no state
- Only shape transformations (2D → 3D, 2D → 4D, etc.)

**Why:**
- Reproducibility (same adapter run → same output)
- Debuggability (easy to trace data flow)
- Testability (deterministic = easy to test)

---

### 3. Plugin-Based Models
- Models register via `@register` decorator
- Automatic discovery (no manual registration)
- Uniform interface (`BaseModel` contract)

**Why:**
- Extensibility (add new models trivially)
- Consistency (all models follow same contract)
- Testability (uniform interface = uniform tests)

---

### 4. Leakage Prevention
- **MTF shift(1):** Prevent lookahead in multi-timeframe features
- **Purge (60 bars):** Remove overlapping labels between splits
- **Embargo (1440 bars):** Prevent serial correlation leakage
- **Train-only scaling:** Fit scaler on train only, transform all splits
- **OOF predictions:** Stacking meta-learner uses out-of-fold preds

**Why:**
- Realistic performance estimates (no leakage = accurate metrics)
- Production readiness (leakage-free = generalizes to live data)

---

### 5. Single-Contract Isolation
- Pipeline processes **one contract at a time** (MES, MGC, etc.)
- No cross-symbol correlation or feature engineering
- Complete isolation between contracts

**Why:**
- Simpler data management (no alignment across symbols)
- Prevents information leakage between contracts
- Easier to reason about feature engineering and labeling

---

## Configuration System

**Pipeline Config:** `config/pipeline.yaml`
```yaml
phase2:
  mtf:
    base_timeframe: "5min"
    timeframes: [5, 10, 15, 20, 25, 30, 45, 60]  # 9 TFs (minutes)

phase4:
  barriers:
    MES: {profit_threshold: 0.015, loss_threshold: 0.010}
  splits: {train_pct: 0.70, val_pct: 0.15, test_pct: 0.15}
  purge_bars: 60
  embargo_bars: 1440
```

**Model Configs:** `config/models/{model_name}.yaml`
```yaml
# XGBoost example
model_params:
  objective: "multi:softprob"
  max_depth: 6
  learning_rate: 0.1

training:
  num_boost_round: 1000
  early_stopping_rounds: 50
```

---

## Quick Commands

### Run Data Pipeline
```bash
./pipeline run --symbols MES
# Output: data/splits/scaled/{MES_train,MES_val,MES_test}.parquet
```

### Train Single Model
```bash
# Tabular model (2D input)
python scripts/train_model.py --model xgboost --horizon 20

# Sequence model (3D input)
python scripts/train_model.py --model lstm --horizon 20 --seq-len 30
```

### Train Ensemble
```bash
# Tabular ensemble
python scripts/train_model.py \
  --model voting \
  --base-models xgboost,lightgbm,catboost \
  --horizon 20

# Sequence ensemble
python scripts/train_model.py \
  --model stacking \
  --base-models lstm,gru,tcn \
  --horizon 20 \
  --seq-len 30
```

---

## Extension Points

### Adding a New Model
1. Create `src/models/{family}/{model_name}.py`
2. Implement `BaseModel` interface
3. Register with `@register(name="...", family="...")`
4. Create `config/models/{model_name}.yaml`
5. Done! Model automatically available in CLI.

### Adding a New Feature
1. Add calculation in `src/phase1/stages/features/indicators/`
2. Register in `src/phase1/stages/features/feature_engineer.py`
3. Done! Feature automatically available to all models.

### Adding a New Timeframe
1. Update `config/pipeline.yaml` (add timeframe to list)
2. Update `src/phase1/stages/mtf/mtf_scaler.py` (if needed)
3. Done! Timeframe automatically upscaled and aligned.

---

**Last Updated:** 2026-01-01
