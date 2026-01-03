# ML Pipeline Analysis Report

**Date:** 2026-01-03
**Analyzed by:** 3 Specialized ML Agents (MLOps Engineer, Data Scientist, ML Engineer)

---

## Executive Summary

The pipeline is a **well-architected heterogeneous ML model factory** that exceeds charter requirements in model count (23 vs 13 stated) with proper infrastructure for heterogeneous ensembles, comprehensive leakage prevention, and clean plugin architecture.

**Overall Verdict: GOOD FOUNDATION, MINOR GAPS**

---

## 1. Pipeline Architecture Analysis (MLOps Engineer)

### 1.1 Pipeline Structure (`src/phase1/`)

**Status: Excellent**

The pipeline is organized into 14+ discrete stages:
```
ingest → clean → sessions → features → regime → mtf → labeling →
ga_optimize → final_labels → splits → scaling → datasets → validation → reporting
```

**Key Files:**
- `src/phase1/stages/datasets/container.py` (687 lines): Unified `TimeSeriesDataContainer`
  - Lines 168-687: Provides `get_sklearn_arrays()` (2D), `get_pytorch_sequences()` (3D), `get_multi_resolution_4d()` (4D)
  - Lines 411-436: Proper support for label end times for PurgedKFold
  - Lines 481-521: Symbol-isolated sequence generation

### 1.2 Model Registry (`src/models/registry.py`)

**Status: Excellent (435 lines)**

Clean plugin architecture:
- Lines 62-139: `@register` decorator with validation
- Lines 141-175: `create()` factory method
- Lines 293-330: `get_model_info()` returns `requires_sequences`, `requires_scaling`, etc.
- **23 models registered** (vs 13 stated in charter)

### 1.3 Data Adapters

**Status: Complete**

| Adapter | Output Shape | Location |
|---------|--------------|----------|
| Tabular (2D) | `(n_samples, n_features)` | `container.get_sklearn_arrays()` |
| Sequence (3D) | `(n_samples, seq_len, n_features)` | `container.get_pytorch_sequences()` |
| Multi-Resolution (4D) | `(n_samples, n_tf, seq_len, features)` | `adapters/multi_resolution.py` |

### 1.4 Cross-Validation (`src/cross_validation/`)

**Status: Excellent**

- `purged_kfold.py` (303 lines): PurgedKFold with timeframe-aware embargo
- `oof_stacking.py` (302 lines): StackingDatasetBuilder with derived meta-features
- Lines 131-152 in purged_kfold.py: Model-family CV strategies (5 splits for boosting, 3 for neural)

### 1.5 Heterogeneous Stacking (`src/models/ensemble/stacking.py`)

**Status: Complete (755 lines)**

- Lines 69-75: Tracks `_is_heterogeneous`, `_tabular_models`, `_sequence_models`
- Lines 132-143: `fit()` accepts both `X_train` (2D) and `X_train_seq` (3D)
- Lines 372-499: `_generate_oof_predictions()` routes correct data to each model type
- Lines 253-277: `use_default_configs_for_oof=True` prevents meta-learner leakage

---

## 2. Data Flow Analysis (Data Scientist)

### 2.1 Data Ingestion (`src/phase1/stages/ingest/`)

**Status: Complete**

- `__init__.py` (309 lines): Main `DataIngestor` class
- `validators.py` (289 lines): OHLCV relationship validation with auto-fix
- Standardizes to canonical format: `datetime`, `open`, `high`, `low`, `close`, `volume`
- **Leakage Risk: LOW**

### 2.2 MTF Upscaling (`src/phase1/stages/mtf/`)

**Status: Complete (9 TFs defined)**

**9 Timeframes in `FULL_MTF_TIMEFRAMES`:**
```python
["1min", "5min", "10min", "15min", "20min", "25min", "30min", "45min", "1h"]
```

**Anti-Lookahead Measures:**
- `shift(1)` applied to MTF data before alignment (`generator.py:313`)
- Forward-fill uses only **completed** higher TF bars
- `validate_no_lookahead()` checks first valid index

**Gap:** CLAUDE.md says "5/9 TFs implemented" but code shows all 9 defined. Clarification needed.

### 2.3 Feature Engineering (`src/phase1/stages/features/`)

**Status: Complete (~180 features)**

| Category | Indicators |
|----------|------------|
| Momentum | RSI, MACD, Stochastic, Williams %R, ROC, CCI, MFI |
| Volatility | ATR, Bollinger, Keltner, Parkinson, Garman-Klass, Yang-Zhang |
| Microstructure | Amihud, Roll spread, Kyle lambda, Corwin-Schultz |
| Wavelets | DWT coefficients, energy, trend decomposition |
| Volume | OBV, VWAP, dollar volume |
| Trend | ADX, Supertrend |
| Temporal | Hour/minute/day sin/cos encoding |
| Regime | Volatility regime, trend regime classification |

### 2.4 Triple-Barrier Labeling (`src/phase1/stages/labeling/`)

**Status: Complete**

- `triple_barrier.py` (652 lines): Numba-optimized implementation
- Asymmetric barriers (`k_up` != `k_down`) to correct long bias
- Transaction cost adjustment via `cost_in_atr`
- Invalid label sentinel (`-99`) for last `max_bars` samples
- **Leakage Risk: LOW**

### 2.5 Leakage Prevention Summary

| Area | Protection | Location |
|------|------------|----------|
| MTF Features | `shift(1)` on higher TF | `generator.py:313` |
| Train/Test Split | Chronological + purge + embargo | `splits/core.py:155-314` |
| Purge | 60 bars at boundaries | Configurable |
| Embargo | 1440 bars buffer | Configurable |
| Scaling | Train-only fitting | `scaling/core.py` |
| Labels | Last `max_bars` invalid | `triple_barrier.py:166-173` |
| Sequences | Never cross symbol boundaries | `sequences.py:146-143` |

### 2.6 Per-Model Feature Selection

**Status: Configurable but NOT enforced**

Infrastructure exists:
- `FeatureSetDefinition` in `src/phase1/config/feature_sets.py`
- `resolve_feature_set()` in `src/phase1/utils/feature_sets.py`

**Gap:** Models receive same full feature set by default. Requires explicit configuration per training run.

---

## 3. Model Implementation Analysis (ML Engineer)

### 3.1 Base Model Interface (`src/models/base.py`)

**Status: Excellent (453 lines)**

| Method | Purpose | Lines |
|--------|---------|-------|
| `fit()` | Train with X_train, y_train, X_val, y_val, sample_weights | 278-368 |
| `predict()` | Return PredictionOutput | 278-368 |
| `save(path)` | Persist to disk | 278-368 |
| `load(path)` | Load from disk | 278-368 |

Supporting dataclasses:
- `PredictionOutput` (lines 27-87): `class_predictions`, `class_probabilities`, `confidence`
- `TrainingMetrics` (lines 94-159): `train_loss`, `val_loss`, `train_f1`, `val_f1`

### 3.2 Model Inventory (23 Total)

#### Boosting Family (3 models)
| Model | File | Lines | Status |
|-------|------|-------|--------|
| XGBoost | `boosting/xgboost_model.py` | 369 | Complete |
| LightGBM | `boosting/lightgbm_model.py` | 431 | Complete |
| CatBoost | `boosting/catboost_model.py` | 361 | Complete |

#### Neural Family (10 models)
| Model | File | Lines | Status |
|-------|------|-------|--------|
| LSTM | `neural/lstm_model.py` | 166 | Complete |
| GRU | `neural/gru_model.py` | 197 | Complete |
| TCN | `neural/tcn_model.py` | 253 | Complete |
| Transformer | `neural/transformer_model.py` | 550 | Complete |
| PatchTST | `neural/patchtst_model.py` | 483 | **Bonus** |
| iTransformer | `neural/itransformer_model.py` | 612 | **Bonus** |
| TFT | `neural/tft_model.py` | 795 | **Bonus** |
| N-BEATS | `neural/nbeats.py` | 753 | **Bonus** |
| InceptionTime | `neural/cnn.py` | Part of 1049 | **Bonus** |
| ResNet1D | `neural/cnn.py` | Part of 1049 | **Bonus** |

#### Classical Family (3 models)
| Model | File | Lines | Status |
|-------|------|-------|--------|
| Random Forest | `classical/random_forest.py` | 266 | Complete |
| Logistic | `classical/logistic.py` | 294 | Complete |
| SVM | `classical/svm.py` | 320 | Complete |

#### Ensemble Family (3 models)
| Model | File | Lines | Status |
|-------|------|-------|--------|
| Voting | `ensemble/voting.py` | 532 | Complete |
| Stacking | `ensemble/stacking.py` | 755 | Complete |
| Blending | `ensemble/blending.py` | 469 | Complete |

#### Meta-Learners (4 models)
| Model | File | Lines | Status |
|-------|------|-------|--------|
| Ridge | `ensemble/meta_learners.py` | 48-311 | Complete |
| MLP | `ensemble/meta_learners.py` | 319-610 | Complete |
| Calibrated | `ensemble/meta_learners.py` | 618-919 | Complete |
| XGBoost Meta | `ensemble/meta_learners.py` | 927-1260 | Complete |

### 3.3 Heterogeneous Ensemble Support

**Status: Excellent**

`StackingEnsemble` properly supports mixed tabular + sequence base models:

```python
# stacking.py lines 451-477
# Tabular models get 2D, sequence models get 3D
if self._is_heterogeneous and model_name in self._sequence_models:
    model_X_train = X_seq_fold_train_cache  # 3D
else:
    model_X_train = X_fold_train  # 2D
```

**Validator** (`validator.py`, 347 lines):
- `HETEROGENEOUS_ENSEMBLE_TYPES = {"stacking"}` (line 25)
- `classify_base_models()` separates tabular vs sequence (lines 305-335)

### 3.4 File Size Violations

| File | Lines | Issue |
|------|-------|-------|
| `neural/cnn.py` | 1049 | **Over 800 limit** - Contains InceptionTime + ResNet1D |
| `ensemble/meta_learners.py` | 1267 | **Over 800 limit** - Contains 4 meta-learners |
| `neural/tft_model.py` | 795 | At limit - Acceptable (TFT is complex) |
| `ensemble/stacking.py` | 755 | At limit - Acceptable (heterogeneous support) |

---

## 4. Architecture Diagram (As Implemented)

```
Raw 1-min OHLCV (Canonical Source)
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  MTF Upscaling                                               │
│  9 TFs: 1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h            │
│  Anti-lookahead: shift(1) before alignment                   │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Feature Engineering (~180 indicators)                       │
│  Momentum | Volatility | Microstructure | Wavelets | Regime  │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Triple-Barrier Labeling                                     │
│  Optuna-optimized | Asymmetric barriers | Cost-adjusted      │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Data Adapters                                               │
│  ├─ get_sklearn_arrays()      → 2D (n, ~180)                │
│  ├─ get_pytorch_sequences()   → 3D (n, seq_len, ~180)       │
│  └─ get_multi_resolution_4d() → 4D (n, n_tf, seq_len, f)    │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Model Registry (23 models via @register)                    │
│  ├─ Boosting (3):  XGBoost, LightGBM, CatBoost              │
│  ├─ Neural (10):   LSTM, GRU, TCN, Transformer, PatchTST,   │
│  │                 iTransformer, TFT, N-BEATS, Inception,   │
│  │                 ResNet1D                                  │
│  ├─ Classical (3): RF, Logistic, SVM                        │
│  ├─ Ensemble (3):  Voting, Stacking, Blending               │
│  └─ Meta (4):      Ridge, MLP, Calibrated, XGBoost          │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Heterogeneous Stacking                                      │
│  Tabular bases → 2D data                                     │
│  Sequence bases → 3D data                                    │
│  OOF predictions → Meta-learner (always 2D)                  │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. Gaps vs Charter

| Gap | Severity | Details |
|-----|----------|---------|
| **MTF Documentation** | Medium | Code has all 9 TFs; CLAUDE.md says "5/9 implemented" |
| **Per-Model Feature Selection** | Medium | Configurable but not enforced by default |
| **`train_ensemble.py` Script** | Medium | No CLI for heterogeneous ensemble training |
| **File Size Violations** | Low | `cnn.py` (1049), `meta_learners.py` (1267) exceed 800 lines |
| **Documentation Mismatch** | Low | CLAUDE.md says 13 models; reality is 23 |

---

## 6. Documentation vs Reality

| CLAUDE.md Says | Reality |
|----------------|---------|
| "13 base models" | **23 models registered** |
| "5/9 MTF timeframes" | All 9 defined in `FULL_MTF_TIMEFRAMES` |
| "Per-model feature selection" | Configurable, not automatic |
| "4 meta-learners" | 4 implemented (correct) |

---

## 7. Charter Compliance Summary

| Charter Goal | Status | Notes |
|--------------|--------|-------|
| Plugin-based model registry | ✅ Complete | 23 models via `@register` |
| 13+ base models | ✅ Exceeded | 23 implemented |
| Heterogeneous ensembles | ✅ Complete | Stacking routes 2D/3D correctly |
| Per-model feature selection | ⚠️ Partial | Infrastructure exists, not enforced |
| 9-timeframe ladder | ⚠️ Clarify | All 9 defined, doc says 5/9 |
| Proper purge/embargo | ✅ Complete | 60/1440 bars configurable |
| Train-only scaling | ✅ Complete | RobustScaler on train only |
| Leakage prevention | ✅ Complete | Comprehensive protections |

---

## 8. Recommended Actions

### High Priority

1. **Clarify MTF Status**
   - Update CLAUDE.md: all 9 TFs are defined in code
   - Document which TFs are actively tested vs just defined
   - Location: `src/phase1/stages/mtf/constants.py`

2. **Create `scripts/train_ensemble.py`**
   - CLI for heterogeneous ensemble training
   - Usage: `--base-models catboost,tcn,patchtst --meta-learner logistic`
   - Coordinates 2D + 3D data loading automatically

### Medium Priority

3. **Enforce Per-Model Feature Selection**
   - Create model-family-specific configs that auto-apply
   - Tabular: ~200 features (base + MTF)
   - Sequence: ~150 features (single-TF)
   - Transformers: Raw OHLCV streams

4. **Update CLAUDE.md Model Count**
   - Change "13 base models" to "23 models"
   - Document the 6 bonus advanced neural models

### Low Priority

5. **Split Large Files**
   - `neural/cnn.py` → `inceptiontime.py` + `resnet1d.py`
   - `ensemble/meta_learners.py` → 4 separate files

6. **Model-Family-Aware Scaling**
   - Boosting: No scaling
   - Neural: Robust scaling
   - Transformers: Standard scaling

---

## 9. Conclusion

This is a **solid heterogeneous ML factory** that exceeds charter requirements. The core architecture properly supports:

- Plugin-based model registration (23 models)
- Heterogeneous ensembles (mixed tabular + sequence bases)
- Comprehensive leakage prevention
- Multiple data formats (2D, 3D, 4D)

The gaps are primarily:
- Documentation clarity (model count, MTF status)
- Convenience tooling (missing CLI script)
- Per-model feature enforcement (infrastructure exists, not automatic)

**None of the gaps are fundamental architectural blockers.** The factory can train any model type on OHLCV data with proper time-series CV.
