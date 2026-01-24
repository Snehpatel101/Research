# Pipeline Implementation Status

**Last Audit:** 2026-01-23 (Phase 1 Deep Analysis)

## Current State (What Works)

### SNwH Phase 0: Canonical Contracts ✅
- **Status:** Complete (81 tests passing)
- **Implemented:** 2026-01-16
- **Files Created:**
  - `src/contracts/data_contract.py` - DataContract, DataRank, FeatureMode, MTFMode
  - `src/contracts/model_contract.py` - ModelContract, MODEL_CONTRACTS (23 models)
  - `src/contracts/artifact_manifest.py` - ArtifactManifest for reproducibility
- **Key Features:**
  - All 23 models have contracts with input_rank, primary_timeframe, mtf_mode
  - DataContract validates arrays and DataFrames
  - Schema hash for reproducibility
  - Artifact manifest for lineage tracking
- **See:** `.serena/knowledge/phase0_contracts_implementation.md` for details

### Phase 1: Canonical OHLCV Ingestion ✅
- Schema validation (OHLCV columns, data types)
- Duplicate removal (keep last)
- Gap detection (preserved, not filled)
- Session filtering (regular vs extended hours)
- Output: `data/processed/{symbol}_1m_clean.parquet`

### Phase 2: MTF Upscaling ✅ (Complete)
- **Status:** All 9 intraday timeframes now implemented (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)
- Resample to higher timeframes (OHLCV aggregation)
- Align to primary timeframe index (forward-fill)
- Apply shift(1) to prevent lookahead
- Output: `data/processed/{symbol}_{timeframe}.parquet`

### Phase 3: Feature Engineering ✅
- **Feature Count:** 160+ features in 9 families (per Phase 1 Agent #5 analysis)
- Feature Families:
  1. Raw OHLCV (4): Base for transformers
  2. Momentum (~40): RSI, MACD, Stochastic, MAs
  3. Volatility (~30): ATR, Bollinger, Historical Vol
  4. Volume (~20): VWAP, OBV, Volume Ratios
  5. Microstructure (~30): VPIN, Kyle's Lambda
  6. Wavelets (~30): Daubechies db4 decomposition
  7. MTF (~20): Multi-timeframe indicators
  8. Regime (~10): Volatility/Trend detection
  9. Temporal: Hour/Day-of-week encoding
- Output: `data/features/{symbol}_features.parquet`

### Phase 4: Triple-Barrier Labeling ✅
- **Optuna Optimization:** TPE sampler, 27% more sample-efficient than traditional GA
- **Search Space:** k_up [0.8-2.5], k_down [0.8-2.5], max_bars_mult [2.0-3.0]
- **Fitness Components:** Neutral Score (PRIMARY), Long/Short Balance (+2), Speed Score (+1.5), Profit Factor (+2)
- **Safe Mode:** Uses only first 70% of data to prevent test leakage
- **Symbol-Specific Seeds:** MES asymmetric (k_up > k_down), MGC symmetric
- Quality weighting (Tier 1: 1.5x, Tier 2: 1.0x, Tier 3: 0.5x)
- Time-series splits (70/15/15) with purge (60) + embargo (1440)
- Robust scaling (train-only fit)
- Output: `data/splits/scaled/{symbol}_{split}.parquet`

### Phase 5: Model-Family Adapters ✅ (Partial)
- **Implemented:**
  - Tabular adapter (2D): `(N, 180)` for boosting + classical
  - Sequence adapter (3D): `(N, seq_len, 180)` for neural
- **Data Adapters (4,771 lines):**
  - TabularAdapter: 2D for XGBoost/LightGBM
  - SequenceAdapter: 3D for LSTM/GRU
  - MultiStreamAdapter: 4D for PatchTST/iTransformer
  - AdapterFactory: Unified entry point
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

### Data Store/Versioning System ✅
- **Total Lines:** 7,522
- **FeatureStore (897 lines):** Unified storage with Parquet caching
- **FeatureCache (647 lines):** Content-addressable storage (SHA256)
- **LineageTracker (620 lines):** Full audit trail with TransformationType enum
- **VersionManager (445 lines):** Semantic versioning (major/minor/patch bumps)

---

## Known Issues (From Phase 1 Analysis - 2026-01-23)

### Critical Issues

#### PIPE-008: Single-Symbol Enforcement
**Status:** Open
**Impact:** Limits batch processing capability
**Description:** Pipeline currently enforces single-symbol processing, preventing batch operations across multiple symbols.

#### CFG-001: Dual Configuration Hierarchy
**Status:** Open
**Impact:** Developer confusion, maintenance burden
**Description:** UnifiedConfig vs PipelineConfig creates confusion. 71+ duplicated `_get_global_or_default()` patterns exist.
**Recommendation:** Migrate all to `get_config_value()`, add cross-config validation (CompositeValidator)

#### CFG-010: Constants Scattered
**Status:** Open
**Impact:** Maintenance difficulty
**Description:** Constants scattered across multiple locations instead of centralized.

### Medium Issues

#### SCALE-001: Fragile Feature Column Detection
**Status:** Open
**Impact:** May break with new features
**Description:** Feature column detection uses fragile string matching patterns.

#### VAL-001: Hardcoded Validation Thresholds
**Status:** Open
**Impact:** Not tunable per experiment
**Description:** Validation thresholds hardcoded (0.85 correlation, 0.01 variance) instead of configurable.

#### LBL-001: ATR Dependency Not Declared
**Status:** Fixed (2024-12)
**Description:** ATR dependency was not declared in feature toggles.

### Technical Debt

#### Missing core.py Pattern
**Status:** Open
**Impact:** Inconsistent stage architecture
**Description:** Only 2/12 pipeline stages fully implement the run.py/core.py pattern. 10 stages need extraction.

#### Asymmetry Bonus Duplication
**Status:** Open
**Impact:** Code duplication
**Description:** Asymmetry bonus logic duplicated between fitness.py and optuna_optimizer.py. Extract to shared module.

#### Lineage Query Performance
**Status:** Open
**Impact:** Slow queries on large lineage graphs
**Description:** Lineage queries are O(n) with no indexing. Add lineage indexing for large graphs.

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
