# Pipeline Implementation Status

## Current State (What Works)

### Phase 1: Canonical OHLCV Ingestion ✅
- Schema validation (OHLCV columns, data types)
- Duplicate removal (keep last)
- Gap detection (preserved, not filled)
- Session filtering (regular vs extended hours)
- Output: `data/processed/{symbol}_1m_clean.parquet`

### Phase 2: MTF Upscaling ⚠️ Partial
- **Implemented:** 5 of 9 timeframes (15min, 30min, 1h, 4h, daily)
- Resample to higher timeframes (OHLCV aggregation)
- Align to 5-minute base index (forward-fill)
- Apply shift(1) to prevent lookahead
- Output: `data/processed/{symbol}_{timeframe}.parquet`

### Phase 3: Feature Engineering ✅
- Base indicators (~150): RSI, MACD, ATR, Bollinger, ADX
- Wavelets (~30): Db4/Haar decomposition (3 levels)
- Microstructure (~20): Spread proxies, order flow
- MTF indicators (~30): Indicators from 5 timeframes
- Total: ~180 features
- Output: `data/features/{symbol}_features.parquet`

### Phase 4: Triple-Barrier Labeling ✅
- Optuna barrier optimization (100 trials, ~2 minutes)
- Triple-barrier labeling (profit/loss/time)
- Quality weighting (0.5x-1.5x)
- Time-series splits (70/15/15) with purge (60) + embargo (1440)
- Robust scaling (train-only fit)
- Output: `data/splits/scaled/{symbol}_{split}.parquet`

### Phase 5: Model-Family Adapters ✅ (Partial)
- **Implemented:**
  - Tabular adapter (2D): `(N, 180)` for boosting + classical
  - Sequence adapter (3D): `(N, seq_len, 180)` for neural
- **Output:** `TimeSeriesDataContainer` (in-memory)

### Phase 6: Model Training ✅
- **Implemented:** 17 models across 5 families
  - Boosting (3): XGBoost, LightGBM, CatBoost
  - Neural (4): LSTM, GRU, TCN, Transformer
  - Classical (3): Random Forest, Logistic, SVM
  - Ensemble (3): Voting, Stacking, Blending
  - Meta-learners (4): Ridge Meta, MLP Meta, Calibrated Meta, XGBoost Meta
- Training with early stopping, sample weighting
- Output: `experiments/runs/{run_id}/models/`

### Phase 7: Heterogeneous Ensemble Training ⚠️ PLANNED
- **Status:** Meta-learners implemented, training script not yet created
- **Missing:** scripts/train_ensemble.py for automated heterogeneous stacking
- **Workaround:** Manual training of base models + meta-learner

### Cross-Validation (Phase 3) ✅
- PurgedKFold with purge/embargo
- OOF prediction generation (tabular + sequence)
- Stacking dataset builder
- Optuna hyperparameter tuning
- Output: CV results, OOF predictions

---

## Gaps (What's Missing)

### 1. Configurable Primary Training Timeframe
**Missing:** Primary training timeframe is currently hardcoded to 5min

**Current:** Fixed 5-min base timeframe

**Intended:** Configurable per experiment (5m/10m/15m/1h)

**Impact:**
- Cannot experiment with different primary timeframes
- Limited flexibility for different trading strategies

**Effort:** 1 day

**Files to modify:**
- `config/pipeline.yaml` (add primary_timeframe config)
- `src/phase1/stages/clean/resample.py` (parameterize base TF)

---

### 2. MTF as Optional Enrichment
**Missing:** MTF is always-on, cannot run single-TF training

**Current:** MTF indicators always computed and added

**Intended:** Three MTF strategies
- Strategy 1: Single-TF (no MTF)
- Strategy 2: MTF Indicators (optional enrichment for tabular)
- Strategy 3: MTF Ingestion (raw OHLCV for sequence models)

**Impact:**
- Cannot build single-TF baselines
- Cannot compare MTF vs. single-TF performance

**Effort:** 1-2 days

**Files to modify:**
- `config/pipeline.yaml` (add mtf.enabled flag)
- `src/phase1/stages/mtf/` (make MTF stages conditional)

---

### 3. Phase 2 - MTF Upscaling (Incomplete)
**Missing:** 4 timeframes (5min, 10min, 20min, 25min, 45min)

**Current:** Only 15min, 30min, 1h, 4h, daily implemented

**Intended:** 9-timeframe ladder
- 1min (base)
- 5min, 10min, 15min, 20min, 25min, 30min, 45min, 1h

**Impact:**
- Advanced models (PatchTST, iTransformer, TFT) require 9 timeframes
- Incomplete MTF feature set
- Strategy 3 (multi-resolution raw OHLCV) blocked

**Effort:** 1-2 days

**Files to modify:**
- `src/phase1/stages/mtf/mtf_scaler.py`
- `config/pipeline.yaml` (add missing timeframes)
- `src/phase1/stages/features/mtf_features.py` (add MTF features for new TFs)

---

### 4. Phase 5 - Multi-Resolution Adapter (Not Started)
**Missing:** Multi-resolution adapter for 4D tensors

**Current:** Only 2D (tabular) and 3D (sequence) adapters exist

**Intended:** 4D multi-resolution adapter
- Shape: `(N, 9, T, 4)` where:
  - N: samples
  - 9: timeframes
  - T: lookback window (varies by timeframe)
  - 4: OHLC features

**Impact:**
- Advanced models (PatchTST, iTransformer, TFT, N-BEATS) cannot be trained
- Strategy 3 (multi-resolution raw OHLCV) blocked

**Effort:** 3 days

**Files to create:**
- `src/phase1/stages/datasets/multi_resolution_adapter.py`
- Update `src/phase1/stages/datasets/dataset_builder.py` to support 4D
- Add multi-resolution loader to `TimeSeriesDataContainer`

---

### 5. Phase 6 - Advanced Models (Not Started)
**Missing:** 6 models across 3 families

**CNN (2 models):**
- InceptionTime (multi-scale pattern detection)
- 1D ResNet (deep residual learning)
- Input: 3D sequences `(N, seq_len, 180)` OR 4D multi-res `(N, 9, T, 4)`

**Advanced Transformers (3 models):**
- PatchTST (patch-based transformer)
- iTransformer (inverted transformer)
- TFT (Temporal Fusion Transformer)
- Input: 4D multi-res `(N, 9, T, 4)` (requires multi-res adapter)

**MLP (1 model):**
- N-BEATS (neural basis expansion)
- Input: 3D sequences `(N, seq_len, 180)` OR 4D multi-res `(N, 9, T, 4)`

**Impact:**
- Missing SOTA time-series models
- Cannot leverage multi-resolution temporal learning

**Effort:** 14-18 days (see `docs/archive/roadmaps/ADVANCED_MODELS_ROADMAP.md`)

**Files to create:**
- `src/models/cnn/inception_time.py`
- `src/models/cnn/resnet_1d.py`
- `src/models/advanced/patch_tst.py`
- `src/models/advanced/itransformer.py`
- `src/models/advanced/tft.py`
- `src/models/mlp/nbeats.py`
- `config/models/{model_name}.yaml` (6 files)

---

### 6. Heterogeneous Ensemble Support (Partially Implemented)
**Status:** Phase 7 meta-learner stacking implemented, but heterogeneous base model support needs enhancement

**Current:** Ensemble methods (Voting/Stacking/Blending) exist but assume same-family bases

**Intended:** Full heterogeneous ensemble support
- Select 1 model per family (3-4 total)
- Generate OOF predictions from heterogeneous bases
- Stack via meta-learner (Logistic/Ridge/MLP)

**Impact:**
- Cannot fully leverage diversity of different model families
- Missing heterogeneous ensemble training script

**Effort:** 2-3 days

**Files to create/modify:**
- `src/ensemble/heterogeneous_stacker.py`
- `scripts/train_ensemble.py` (heterogeneous training)
- `config/ensemble.yaml` (meta-learner configs)

---

## What We're Building

### ONE Unified Pipeline Architecture

**Core Principle:** Single pipeline that ingests canonical OHLCV and deterministically derives model-specific representations.

**NOT separate pipelines** - ONE workflow with adapters.

**Data Flow:**
```
Raw OHLCV → Configurable TF → Features → Labels → Adapters → Training → Artifacts
```

**Key Components:**
1. **Configurable primary timeframe:** ⚠️ Planned - currently hardcoded to 5min
2. **Optional MTF enrichment:** Single-TF, MTF indicators, or MTF ingestion
3. **Model-family adapters:** Transform canonical data to model-specific formats (2D, 3D, 4D)
4. **Plugin-based models:** 23 total models (17 implemented + 6 planned)
   - 10 base models (boosting, neural, classical)
   - 3 ensemble models (voting, stacking, blending)
   - 4 meta-learners (ridge, mlp, calibrated, xgboost)
   - 6 planned advanced (CNN, transformers, MLP)
5. **Heterogeneous ensembles:** ⚠️ Planned - meta-learners exist, training script not yet created

**Result:** Reproducible, deterministic, storage-efficient ML factory.

---

## Priority Tasks

### 1. Complete 9-Timeframe MTF Ladder (1-2 days)
**Goal:** Add missing timeframes to Phase 2

**Tasks:**
- Add 5min, 10min, 20min, 25min, 45min to `config/pipeline.yaml`
- Update `src/phase1/stages/mtf/mtf_scaler.py` to generate all 9 TFs
- Add MTF features for new timeframes in `src/phase1/stages/features/mtf_features.py`
- Update tests in `tests/phase1/stages/mtf/`

**Deliverable:** 9 MTF views (1min → 1h)

---

### 2. Implement Multi-Resolution Adapter (3 days)
**Goal:** Enable 4D tensor input for advanced models

**Tasks:**
- Create `src/phase1/stages/datasets/multi_resolution_adapter.py`
- Build 4D tensor loader: `(N, 9, T, 4)` from raw MTF OHLCV
- Update `TimeSeriesDataContainer` to support 4D data
- Add adapter tests in `tests/phase1/stages/datasets/`
- Update `scripts/train_model.py` to support 4D models

**Deliverable:** Multi-resolution adapter ready for advanced models

---

### 3. Add Advanced Models (14-18 days)
**Goal:** Implement 6 advanced models (CNN, transformers, MLP)

**Phased approach:**
- **Week 1:** InceptionTime + 1D ResNet (CNN family, 3D input)
- **Week 2:** PatchTST + iTransformer (4D input, requires multi-res adapter)
- **Week 3:** TFT + N-BEATS (4D input)

**Tasks per model:**
- Create model class in `src/models/{family}/`
- Register model with `@register` decorator
- Create config file in `config/models/`
- Add tests in `tests/models/{family}/`
- Update documentation

**Deliverable:** 19 total models (13 existing + 6 new)

---

### 4. Build Meta-Learner Training (5-7 days)
**Goal:** Adaptive ensemble selection

**Tasks:**
- Implement regime-aware weighting (cluster market states, weight by regime)
- Implement confidence-based selection (weight by prediction confidence)
- Implement adaptive tracking (weight by recent performance)
- Create meta-learner training script
- Add tests for meta-learner logic

**Deliverable:** Meta-learner layer for adaptive ensemble selection

---

## Implementation Sequence

**Critical Path:**
1. Complete 9-TF MTF ladder (blocks Strategy 3)
2. Implement multi-res adapter (blocks advanced models)
3. Add CNN models (validate 3D/4D pipeline)
4. Add advanced transformer models (PatchTST, iTransformer, TFT)
5. Add N-BEATS (MLP family)
6. Build meta-learners (adaptive ensemble selection)

**Total Timeline:** 4-5 weeks

**Milestones:**
- Week 1: 9-TF MTF + multi-res adapter
- Week 2-3: CNN + advanced transformers
- Week 4: N-BEATS + meta-learners
- Week 5: Testing, documentation, validation

---

## Success Criteria

### Phase 2 Complete
- [ ] 9 MTF views generated (1min → 1h)
- [ ] MTF features computed for all 9 timeframes
- [ ] Tests pass for all MTF stages
- [ ] Documentation updated

### Phase 5 Complete
- [ ] Multi-resolution adapter implemented
- [ ] 4D tensor loader working
- [ ] TimeSeriesDataContainer supports 4D
- [ ] Tests pass for multi-res adapter

### Phase 6 Complete
- [ ] 19 models implemented (13 + 6)
- [ ] All models register correctly
- [ ] Training works for all families (tabular, sequence, multi-res)
- [ ] Tests pass for all models
- [ ] Documentation updated

### Phase 8 Complete
- [ ] Meta-learners implemented (regime-aware, confidence-based, adaptive)
- [ ] Training script supports meta-learners
- [ ] Tests pass for meta-learner logic
- [ ] Documentation updated

---

**Last Updated:** 2026-01-01
