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
- **Implemented:** 23 models across 6 families (22 if CatBoost unavailable)
  - Boosting (3): XGBoost, LightGBM, CatBoost
  - Neural (10): LSTM, GRU, TCN, Transformer, PatchTST, iTransformer, TFT, N-BEATS, InceptionTime, ResNet1D
  - Classical (3): Random Forest, Logistic, SVM
  - Ensemble (3): Voting, Stacking, Blending
  - Meta-learners (4): Ridge Meta, MLP Meta, Calibrated Meta, XGBoost Meta
- Training with early stopping, sample weighting
- Output: `experiments/runs/{run_id}/models/`

### Phase 7: Heterogeneous Ensemble Training ✅
- **Status:** Complete - heterogeneous stacking implemented in trainer.py
- **Features:** Dual data loading for mixed tabular/sequence bases
- **Usage:** `scripts/train_model.py --model stacking --base-models xgboost,lstm,patchtst --meta-learner ridge_meta`

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

### 2. Phase 2 - MTF Upscaling ✅ COMPLETE
**Status:** All 9 intraday timeframes implemented (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)

**Implemented:**
- Full 9-TF ladder from 1-min canonical source
- Configurable via `--process-all-timeframes` or `--output-timeframes`
- Per-model primary TF selection (CatBoost→15min, TCN→5min, PatchTST→1min)
- All stages iterate over `effective_output_timeframes`

**Resolved:** No longer blocking advanced model training or heterogeneous ensembles

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
4. **Plugin-based models:** 23 total models (22 if CatBoost unavailable)
   - 6 tabular models (boosting + classical)
   - 10 neural models (LSTM, GRU, TCN, Transformer, PatchTST, iTransformer, TFT, N-BEATS, InceptionTime, ResNet1D)
   - 3 ensemble models (voting, stacking, blending)
   - 4 meta-learners (ridge, mlp, calibrated, xgboost)
5. **Heterogeneous ensembles:** ✅ Complete - dual data loading in trainer.py supports mixed tabular/sequence bases

**Result:** Reproducible, deterministic, storage-efficient ML factory.

---

## Priority Tasks (P0 Architecture Improvements)

### 1. Lineage Unification (1-2 days)
**Goal:** Link training runs to pipeline artifacts via metadata + checksums

**Tasks:**
- Add `pipeline_run_id` to TrainerConfig
- Compute checksums for scaled datasets at pipeline completion
- Store checksums in pipeline artifacts
- Validate checksums at training time
- Add lineage validation to Trainer

**Deliverable:** Reproducible artifact lineage from pipeline → training

---

### 2. Timestamp Alignment Checks (1-2 days)
**Goal:** Ensure datetime alignment (not just index) for heterogeneous stacking

**Tasks:**
- Add datetime alignment validation in stacking dataset builder
- Validate timestamp consistency across base model predictions
- Add tests for timestamp alignment edge cases
- Update stacking logic to enforce datetime matching

**Deliverable:** Robust heterogeneous ensemble alignment

---

### 3. Standardize Evaluation Reports (4 hours)
**Goal:** Unified JSON + markdown artifacts per model run

**Tasks:**
- Create evaluation report schema (JSON format)
- Generate markdown summary from JSON
- Save reports to `experiments/runs/{run_id}/reports/`
- Add report generation to Trainer
- Update tests for report validation

**Deliverable:** Consistent evaluation artifacts across all models

---

## Implementation Sequence

**Current Status:** Phases 1-7 complete (23 models implemented)

**Next Steps (P0 Architecture Improvements):**
1. Lineage unification (1-2 days)
2. Timestamp alignment checks (1-2 days)
3. Standardize evaluation reports (4 hours)

**Total Timeline:** 2-3 days

**Optional Future Work:**
- Phase 8: Advanced meta-learners (regime-aware, adaptive)
- Phase 9: Real-time inference pipeline

---

## Success Criteria

### Phase 1-7: ✅ Complete
- [x] Canonical OHLCV ingestion pipeline
- [x] 9 intraday MTF timeframes (1m-1h)
- [x] ~180 engineered features
- [x] Triple-barrier labeling with Optuna optimization
- [x] 2D/3D adapters for tabular/sequence models
- [x] 23 models across 6 families (22 if CatBoost unavailable)
- [x] Heterogeneous ensemble stacking support
- [x] 3442 tests passing

### P0 Architecture Improvements: ⏳ Pending
- [ ] Lineage unification (pipeline→training artifacts)
- [ ] Timestamp alignment checks for stacking
- [ ] Standardized evaluation reports (JSON + markdown)

---

**Last Updated:** 2026-01-15
