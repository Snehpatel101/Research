# ML Factory Overview - TOPSTEPX Single-Contract Trading

## North Star

**Production-grade, model-agnostic ML factory for systematic futures trading research and deployment.**

- **Input**: Single canonical 1-min OHLCV dataset for one futures contract (MES, MGC)
- **Output**: Reproducible training datasets, trained model artifacts, evaluation reports, inference-ready predictors

## What "Factory" Means

1. **ONE canonical pipeline**: Single data flow from raw 1-min OHLCV to trained models
2. **Deterministic adapters**: Model-family specific feature selection + shape transformations (2D/3D/4D)
3. **Per-model feature selection**: Each model gets DIFFERENT features tailored to its inductive biases
4. **Single source of truth**: 1-min canonical OHLCV → ALL 9 timeframes derived → models choose features
5. **Fair comparison**: Same timestamps, labels, splits, leakage prevention, metrics/backtest assumptions
6. **Research → inference parity**: Exact same preprocessing graph at training and runtime

## Architecture (7 Phases, ONE Pipeline)

### Phases 1-4: Data Pipeline (`src/phase1/`)
```
Phase 1: Raw OHLCV → Ingest → Clean → Validate (1-min canonical source)
Phase 2: MTF Upscaling → Derive ALL 9 timeframes (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)
         ⚠️ Currently only 5 TFs implemented (15m, 30m, 1h, 4h, daily)
Phase 3: Feature Engineering → ~180 features (base indicators + wavelets + microstructure + MTF)
Phase 4: Labeling + Splits → Triple-barrier + 70/15/15 splits + Robust scaling
```

**Output**: Canonical dataset in `data/splits/scaled/` (single source of truth)
**Leakage protection**: Purge (60), embargo (1440), shift(1) on MTF, train-only scaling

### Phase 5: Model-Family Adapters (`src/phase1/stages/datasets/`)

**Purpose**: Transform canonical dataset to model-specific formats with feature selection

| Adapter | Output Shape | Feature Selection | Models |
|---------|--------------|-------------------|--------|
| **Tabular** | 2D `(N, ~200)` | Base + MTF indicators + wavelets | Boosting, Classical |
| **Sequence** | 3D `(N, T, ~150)` | Base + wavelets (NO MTF) | Neural, CNN, MLP |
| **Multi-Res** | 4D `(N, TF, T, 4)` | Raw OHLCV streams (NO indicators) | Advanced Transformers |

**Key**: Adapters do BOTH feature selection AND shape transformation (not shape-only)

### Phase 6: Model Training (`src/models/`)
```
19 total models:
├─ 13 implemented: Boosting (3), Neural (4), Classical (3), Ensemble (3)
├─ 6 planned: CNN (2), Advanced Transformers (3), MLP (1)
└─ 5 families: Boosting, Classical, Neural, Advanced, Inference/Meta
```

**Model Interface**: `BaseModel` with `fit()`, `predict()`, `save()`, `load()`
**Registry**: Plugin-based `@register()` decorator system

### Phase 7: Meta-Learner Stacking (`src/cross_validation/`)

**Architecture**: Heterogeneous ensemble (3-4 bases → 1 meta-learner)

```
Base Model Selection (1 per family, 3-4 total)
  ├→ Tabular: CatBoost (~200 features, 15min + MTF indicators)
  ├→ CNN/TCN: TCN (~150 features, 5min, single-TF)
  ├→ Transformer: PatchTST (raw OHLCV multi-stream)
  └→ Optional 4th: Ridge (1h, single-TF)
       ↓
  OOF Generation (PurgedKFold)
       ↓
  Meta-Learner (Logistic/Ridge/MLP) → Final Predictions
```

**Key Features**:
- Direct stacking on OOF predictions (no "ensembles of ensembles")
- Heterogeneous bases (different families, NOT same-family constraint)
- Time-series aware CV with purge/embargo

## Per-Model Feature Selection (Core Architecture)

**SAME Underlying Data:**
- ✅ 1-min canonical OHLCV source
- ✅ Same timestamps (aligned after resampling)
- ✅ Same target labels
- ✅ Same train/val/test splits

**DIFFERENT Features:**
- **Tabular models (CatBoost, XGBoost, LightGBM):** ~200 features (base + MTF indicators + wavelets + microstructure)
- **Sequence models (TCN, LSTM, GRU, Transformer):** ~150 features (base + wavelets, NO MTF)
- **Advanced transformers (PatchTST, iTransformer, TFT):** Raw multi-stream OHLCV (12-17 raw features)

**Why Different:**
1. **Inductive bias alignment**: Tabular models need engineered features; transformers learn from raw data
2. **Ensemble diversity**: Different features → reduced error correlation → better meta-learner performance
3. **Efficiency**: Sequence models have temporal memory (don't need MTF indicators)

## Single-Contract Isolation (NON-NEGOTIABLE)

- **One contract per pipeline run** (MES or MGC, never both)
- **Complete data isolation** - no cross-symbol correlation or features
- **Symbol configurability** via config: `./pipeline run --symbols MES`

## Model Families (5 Total)

| Family | Models | Input Shape | Primary TF | Features | Status |
|--------|--------|-------------|-----------|----------|--------|
| **Boosting** | XGBoost, LightGBM, CatBoost | 2D `(N, 200)` | 15min | Base + MTF indicators | ✅ Complete |
| **Classical** | Random Forest, Logistic, SVM | 2D `(N, 150)` | 1h | Base only | ✅ Complete |
| **Neural** | LSTM, GRU, TCN, Transformer | 3D `(N, T, 150)` | 5min | Base only | ✅ Complete |
| **Advanced** | PatchTST, iTransformer, TFT | 4D `(N, 3, T, 4)` | 1min | Raw OHLCV | ❌ Planned |
| **Inference/Meta** | Logistic Meta, Ridge Meta, MLP Meta, Calibrated Blender | Mixed | N/A | OOF predictions | ⚠️ Partial |

**Note**: Each model independently chooses primary TF from the 9 derived timeframes (configurable per-model)

## Pipeline Implementation Status

| Phase | Component | Status | Gap |
|-------|-----------|--------|-----|
| 1 | OHLCV Ingestion | ✅ Complete | None |
| 2 | MTF Upscaling | ⚠️ Partial | 4 of 9 timeframes missing (5min, 10min, 20min, 25min, 45min) |
| 3 | Feature Engineering | ✅ Complete | MTF features only from 5 timeframes (intended: 9) |
| 4 | Labeling + Splits | ✅ Complete | None |
| 5 | Adapters | ⚠️ Partial | Multi-res adapter (4D) not implemented, per-model feature selection not fully implemented |
| 6 | Model Training | ⚠️ Partial | 6 advanced models not implemented (CNN, transformers, MLP) |
| 7 | Meta-Learner Stacking | ⚠️ Partial | Heterogeneous stacking implemented, calibrated blender not implemented |

**Priority Tasks**:
1. Complete 9-timeframe MTF ladder (1-2 days)
2. Implement per-model feature selection in adapters (3-5 days)
3. Implement multi-resolution adapter (3 days)
4. Add advanced models (14-18 days)
5. Complete meta-learner strategies (5-7 days)

## Key Parameters

```python
SYMBOL = 'MES'  # One symbol per run
LABEL_HORIZONS = [5, 10, 15, 20]  # Prediction horizons in bars
TRAIN/VAL/TEST = 70/15/15
PURGE_BARS = 60  # 3× max_horizon (prevents label leakage)
EMBARGO_BARS = 1440  # ~5 days at 5-min (prevents serial correlation)
MTF_TIMEFRAMES = [1, 5, 10, 15, 20, 25, 30, 45, 60]  # All 9 timeframes (minutes)
```

## File Size Limits

- **Target**: 650 lines per file
- **Maximum**: 800 lines per file
- Beyond 800 → refactor and split responsibilities

## Documentation

**Canonical references:**
- Architecture: `docs/ARCHITECTURE.md`
- Per-model features: `.serena/knowledge/per_model_feature_selection.md`
- Heterogeneous ensembles: `.serena/knowledge/heterogeneous_ensemble_architecture.md`
- MTF strategies: `docs/implementation/MTF_IMPLEMENTATION_ROADMAP.md`
- Models catalog: `docs/reference/MODELS.md`

**Last Updated:** 2026-01-01
