# Repository Organization & Discrepancy Analysis

**Generated:** 2025-12-30
**Purpose:** Comprehensive analysis of codebase vs documentation discrepancies
**Scope:** Identify gaps, organize structure, create unified roadmap

---

## Executive Summary

**Critical Findings:**

1. **MODEL COUNT MISMATCH:** Documentation claims 19 models, but only **13 are implemented**
2. **MTF TIMEFRAMES INCOMPLETE:** Missing 20min and 25min from 9-timeframe ladder
3. **DOCUMENTATION CONFLICTS:** CLAUDE.md vs ALIGNMENT_PLAN.md vs actual code
4. **6 ADVANCED MODELS MISSING:** PatchTST, iTransformer, TFT, InceptionTime, N-BEATS, Chronos, etc.
5. **REMOVED MODELS STILL EXIST:** CatBoost, GRU, Random Forest, SVM, Blending (marked for removal but still in code)

---

## Section 1: Model Inventory Discrepancy

### What Documentation Says (ALIGNMENT_PLAN.md)

**Claims:** 19 models (13 existing + 6 new)

**Removal List (10 models marked for deletion):**
- ❌ CatBoost → "3rd boosting model is overkill"
- ❌ GRU → "Too similar to LSTM"
- ❌ Transformer (basic) → "Non-causal, leaks future data"  - ❌ Random Forest → "Inferior to boosting"
- ❌ SVM → "O(n²-n³) too slow"
- ❌ Blending → "Wastes 20% data on holdout"

**Final 19 Models (per ALIGNMENT_PLAN):**
1. XGBoost
2. LightGBM
3. LSTM
4. TCN
5. **InceptionTime** (NOT implemented)
6. **1D ResNet** (NOT implemented)
7. **PatchTST** (NOT implemented)
8. **iTransformer** (NOT implemented)
9. **TFT** (NOT implemented)
10. **DeepAR** (NOT implemented)
11. **Quantile RNN** (NOT implemented)
12. **N-BEATS** (NOT implemented)
13. **N-HiTS** (NOT implemented)
14. **DLinear** (NOT implemented)
15. **Chronos** (NOT implemented)
16. **TimesFM** (NOT implemented)
17. Logistic Regression
18. Voting Ensemble
19. Stacking Ensemble

### What Actually Exists (src/models/)

**Actual Count:** 13 models

**Boosting (3):**
1. ✅ XGBoost (`boosting/xgboost_model.py`)
2. ✅ LightGBM (`boosting/lightgbm_model.py`)
3. ✅ **CatBoost** (`boosting/catboost_model.py`) ← **SHOULD BE REMOVED per docs**

**Neural Sequence (4):**
4. ✅ LSTM (`neural/lstm_model.py`)
5. ✅ **GRU** (`neural/gru_model.py`) ← **SHOULD BE REMOVED per docs**
6. ✅ TCN (`neural/tcn_model.py`)
7. ✅ **Transformer** (`neural/transformer_model.py`) ← **SHOULD BE REMOVED per docs**

**Classical (3):**
8. ✅ Logistic Regression (`classical/logistic.py`)
9. ✅ **Random Forest** (`classical/random_forest.py`) ← **SHOULD BE REMOVED per docs**
10. ✅ **SVM** (`classical/svm.py`) ← **SHOULD BE REMOVED per docs**

**Ensemble (3):**
11. ✅ Voting (`ensemble/voting.py`)
12. ✅ Stacking (`ensemble/stacking.py`)
13. ✅ **Blending** (`ensemble/blending.py`) ← **SHOULD BE REMOVED per docs**

### Missing Advanced Models (6+ models)

**NOT FOUND in codebase:**
- ❌ InceptionTime (CNN)
- ❌ 1D ResNet (CNN)
- ❌ PatchTST (Transformer)
- ❌ iTransformer (Transformer)
- ❌ TFT (Transformer)
- ❌ DeepAR (Probabilistic)
- ❌ Quantile RNN (Probabilistic)
- ❌ N-BEATS (MLP)
- ❌ N-HiTS (MLP)
- ❌ DLinear (Linear)
- ❌ Chronos (Foundation)
- ❌ TimesFM (Foundation)

**Gap:** 6-12 models depending on how many are actually planned

---

## Section 2: MTF (Multi-Timeframe) Discrepancy

### What Documentation Says

**MTF_IMPLEMENTATION_ROADMAP.md:**
- 9-timeframe ladder: `1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h`
- Task 1.1: "Add 20m and 25m to timeframe ladder"

**ALIGNMENT_PLAN.md:**
- Training timeframe selection guide shows all 9 timeframes
- Three MTF strategies documented

### What Actually Exists

**src/phase1/stages/mtf/constants.py:**

```python
MTF_TIMEFRAMES = {
    '1min': 1,
    '5min': 5,
    '10min': 10,
    '15min': 15,
    '30min': 30,
    '45min': 45,
    '60min': 60,
    '1h': 60,
    '4h': 240,      # ← EXTRA (not in 9-TF ladder)
    'daily': 1440,  # ← EXTRA (not in 9-TF ladder)
}
```

**Missing:**
- ❌ **20min** timeframe
- ❌ **25min** timeframe

**Extra (not in 9-TF ladder):**
- 4h (240min)
- daily (1440min)

**Default MTF:**
```python
DEFAULT_MTF_TIMEFRAMES = ['15min', '30min', '1h', '4h', 'daily']
```

**Status:** Partially implemented, needs 20m and 25m, should remove/deprecate 4h and daily

---

## Section 3: MTF Strategy Implementation Status

### Three Strategies (per ALIGNMENT_PLAN)

**Strategy 1: Single-Timeframe**
- **Status:** ❓ Unknown if implemented
- **Files:** Should be in `src/phase1/stages/features/run.py`
- **Conditional:** Skip MTF feature generation

**Strategy 2: MTF Indicators**
- **Status:** ✅ Partially implemented
- **Files:** `src/phase1/stages/mtf/generator.py`
- **Issue:** No clear `mtf_strategy` config parameter in `PipelineConfig`

**Strategy 3: MTF Ingestion (Multi-Resolution Tensors)**
- **Status:** ❌ NOT implemented
- **Files:** Missing `src/phase1/stages/datasets/multi_resolution.py`
- **Missing:** `get_multi_resolution_tensors()` method in `TimeSeriesDataContainer`

---

## Section 4: Documentation Conflicts

### Root Documentation Files

| File | Claims | Accuracy | Issues |
|------|--------|----------|--------|
| **CLAUDE.md** | 13 models, factory pattern | ⚠️ Partially accurate | Says CatBoost/GRU/RF/SVM should be removed but they exist |
| **CLAUDE2.md** | OOF stacking, inference-first | ✅ Accurate | Good charter document |
| **ALIGNMENT_PLAN.md** | 19 models, 3 MTF strategies | ❌ Inaccurate | Claims 6 new models don't exist, removal list not applied |
| **PIPELINE_FLOW.md** | 14 stages, 13 models | ✅ Accurate | Good visual flow |
| **MTF_IMPLEMENTATION_ROADMAP.md** | 6-phase MTF implementation | ⚠️ Future plan | Not implemented yet |
| **MODEL_INTEGRATION_GUIDE.md** | How to add models | ✅ Accurate | Good guide (just created) |
| **FEATURE_ENGINEERING_GUIDE.md** | Feature sets per model | ✅ Accurate | Good guide (just created) |
| **HYPERPARAMETER_OPTIMIZATION_GUIDE.md** | GA + Optuna | ✅ Accurate | Good guide (just created) |
| **MODEL_INFRASTRUCTURE_REQUIREMENTS.md** | Hardware reqs | ✅ Accurate | Good guide (just created) |

**Recommendation:**
- **IGNORE:** CLAUDE.md (user confirmed inaccurate)
- **PRIMARY:** CLAUDE2.md, PIPELINE_FLOW.md, new guides
- **UPDATE:** ALIGNMENT_PLAN.md to match reality

---

## Section 5: Phase 1 Pipeline Implementation Status

### 14 Stages (per PIPELINE_FLOW.md)

| Stage | Directory | Status | Notes |
|-------|-----------|--------|-------|
| 1. Ingest | `ingest/` | ✅ Implemented | OHLCV validation |
| 2. Clean | `clean/` | ✅ Implemented | Resample 1m→5m (but should be configurable) |
| 3. Sessions | `sessions/` | ✅ Implemented | Session filtering |
| 4. Features | `features/` | ✅ Implemented | 150+ indicators |
| 5. Regime | `regime/` | ✅ Implemented | HMM, volatility, trend |
| 6. MTF | `mtf/` | ⚠️ Partial | Exists but missing 20m/25m, no Strategy 3 |
| 7. Labeling | `labeling/` | ✅ Implemented | Triple-barrier |
| 8. GA Optimize | `ga_optimize/` | ✅ Implemented | Optuna label params |
| 9. Final Labels | `final_labels/` | ✅ Implemented | Apply optimized params |
| 10. Splits | `splits/` | ✅ Implemented | Train/val/test + purge/embargo |
| 11. Scaling | `scaling/` | ✅ Implemented | RobustScaler (train-only) |
| 12. Datasets | `datasets/` | ⚠️ Partial | TabularDataContainer exists, missing MultiResolution |
| 13. Validation | `validation/` | ✅ Implemented | Feature correlation, drift |
| 14. Reporting | `reporting/` | ✅ Implemented | Pipeline reports |

**Issues:**
- `clean/` stage hardcodes 5min resampling, should use `training_timeframe` config
- `datasets/` missing multi-resolution builder for Strategy 3
- No clear `mtf_strategy` config parameter

---

## Section 6: Configuration System Gaps

### Current Config (src/phase1/pipeline_config.py)

**Missing Parameters:**

```python
@dataclass
class PipelineConfig:
    # ... existing ...

    # MISSING: MTF strategy selection
    # mtf_strategy: str = 'single_tf'  # Options: 'single_tf', 'mtf_indicators', 'mtf_ingestion'

    # MISSING: Training timeframe (separate from ingest)
    # ingest_timeframe: str = '1min'  # Always 1min
    # training_timeframe: str = '5min'  # Configurable

    # MISSING: MTF source timeframes (Strategy 2)
    # mtf_source_timeframes: list[str] | None = None

    # MISSING: MTF input timeframes (Strategy 3)
    # mtf_input_timeframes: list[str] | None = None
```

**Current Hardcoded Assumptions:**
- Base timeframe is hardcoded to 5min in `clean/pipeline.py`
- No way to choose between 3 MTF strategies
- No way to specify which source timeframes for MTF features

---

## Section 7: Cross-Validation Implementation Status

### Phase 3 Components

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| Purged K-Fold | `purged_kfold.py` | ✅ Implemented | Time-series aware CV |
| Feature Selection | `feature_selector.py` | ✅ Implemented | MDA/MDI |
| OOF Core (Tabular) | `oof_core.py` | ✅ Implemented | Tabular OOF generation |
| OOF Sequence | `oof_sequence.py` | ✅ Implemented | Sequence model OOF |
| OOF Stacking | `oof_stacking.py` | ✅ Implemented | Stacking dataset builder |
| OOF Validation | `oof_validation.py` | ✅ Implemented | Coverage validation |
| OOF I/O | `oof_io.py` | ✅ Implemented | Save/load OOF |
| CV Runner | `cv_runner.py` | ✅ Implemented | Main CV orchestration |
| Param Spaces | `param_spaces.py` | ⚠️ Partial | Spaces for 13 models, missing 6 new |
| Walk Forward | `walk_forward.py` | ✅ Implemented | Rolling window CV |
| CPCV | `cpcv.py` | ✅ Implemented | Combinatorial purged CV |
| PBO | `pbo.py` | ✅ Implemented | Probability of backtest overfitting |

**Status:** Phase 3 is well-implemented for existing 13 models

---

## Section 8: Inference/Deployment Status

### Inference Components

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| Inference Pipeline | `inference/pipeline.py` | ✅ Implemented | Feature pipeline + predict |
| Bundle | `inference/bundle.py` | ✅ Implemented | Package model artifacts |
| Batch Inference | `inference/batch.py` | ✅ Implemented | Batch predictions |
| Server | `inference/server.py` | ✅ Implemented | FastAPI server |

**Status:** Inference infrastructure exists

---

## Section 9: Directory Structure

### Current Structure (Good)

```
Research/
├── src/
│   ├── phase1/              ✅ Well organized
│   │   └── stages/          ✅ 14 stages implemented
│   ├── models/              ✅ Registry architecture
│   │   ├── base.py          ✅ BaseModel interface
│   │   ├── registry.py      ✅ Plugin system
│   │   ├── boosting/        ✅ 3 models
│   │   ├── neural/          ✅ 4 models
│   │   ├── classical/       ✅ 3 models
│   │   └── ensemble/        ✅ 3 models
│   ├── cross_validation/    ✅ 12 files, well organized
│   ├── inference/           ✅ 4 files
│   └── utils/               ✅ Utilities
├── scripts/                 ✅ CLI tools
├── config/                  ✅ YAML configs
├── docs/                    ✅ Comprehensive docs
│   ├── phases/              ✅ PHASE_1-5.md
│   ├── reference/           ✅ ARCHITECTURE.md, FEATURES.md
│   ├── research/            ✅ Best practices research
│   └── getting-started/     ✅ Quickstart guides
├── tests/                   ⚠️ Needs verification
└── data/                    ✅ Standard layout
```

**Issues:**
- `tests/` coverage unknown
- Root has too many markdown files (13+ files)
- No clear separation of "planning docs" vs "implementation guides"

---

## Section 10: Recommended Reorganization

### Root Documentation Cleanup

**Current root .md files (13):**
1. CLAUDE.md ← IGNORE (inaccurate)
2. CLAUDE2.md ← KEEP (good charter)
3. README.md ← UPDATE
4. ALIGNMENT_PLAN.md ← UPDATE (fix discrepancies)
5. PIPELINE_FLOW.md ← KEEP (good visual)
6. ML_PIPELINE_AUDIT_REPORT.md ← ARCHIVE
7. MTF_IMPLEMENTATION_ROADMAP.md ← KEEP (future plan)
8. MODEL_INTEGRATION_GUIDE.md ← KEEP (new, good)
9. FEATURE_ENGINEERING_GUIDE.md ← KEEP (new, good)
10. HYPERPARAMETER_OPTIMIZATION_GUIDE.md ← KEEP (new, good)
11. MODEL_INFRASTRUCTURE_REQUIREMENTS.md ← KEEP (new, good)
12. REPO_ORGANIZATION_ANALYSIS.md ← THIS FILE

**Proposed Structure:**

```
Research/
├── README.md                      # Main entry point
├── QUICKSTART.md                  # Get started in 5 minutes
│
├── docs/
│   ├── PROJECT_CHARTER.md         # Vision, goals, architecture (merge CLAUDE2 + ALIGNMENT_PLAN)
│   ├── PIPELINE_FLOW.md           # Visual pipeline flow (keep)
│   │
│   ├── guides/                    # Implementation guides (MOVE from root)
│   │   ├── MODEL_INTEGRATION.md
│   │   ├── FEATURE_ENGINEERING.md
│   │   ├── HYPERPARAMETER_OPTIMIZATION.md
│   │   └── MODEL_INFRASTRUCTURE.md
│   │
│   ├── roadmaps/                  # Future implementation plans
│   │   ├── MTF_IMPLEMENTATION.md
│   │   └── ADVANCED_MODELS.md     # Plan for 6 missing models
│   │
│   ├── phases/                    # Phase documentation (keep)
│   ├── reference/                 # Reference docs (keep)
│   └── archive/                   # Old/deprecated docs
│       ├── CLAUDE.md              # MOVE here (inaccurate)
│       └── ML_PIPELINE_AUDIT_REPORT.md
│
└── .github/
    └── CONTRIBUTING.md            # How to add models, features, etc.
```

---

## Section 11: Priority Action Items

### Critical (Week 1)

1. **Fix ALIGNMENT_PLAN.md** to match actual implementation (13 models, not 19)
2. **Add missing MTF timeframes** (20min, 25min) to constants
3. **Add `mtf_strategy` config parameter** to PipelineConfig
4. **Create PROJECT_CHARTER.md** (merge CLAUDE2 + corrected ALIGNMENT_PLAN)
5. **Update README.md** with accurate model count and status

### High Priority (Week 2-3)

6. **Implement Strategy 1** (single-timeframe, skip MTF)
7. **Implement Strategy 3** (multi-resolution tensors)
8. **Add `training_timeframe` config** (separate from hardcoded 5min)
9. **Create test coverage report**
10. **Reorganize root documentation** per proposed structure

### Medium Priority (Week 4-6)

11. **Decide on model removal:** Actually remove CatBoost/GRU/RF/SVM/Blending OR keep them and update docs
12. **Advanced model roadmap:** Plan for InceptionTime, PatchTST, etc. (if wanted)
13. **CI/CD setup:** pytest, linting, pre-commit hooks
14. **Model-specific MTF configurations** per model family

### Low Priority (Future)

15. **Foundation models:** Chronos, TimesFM wrappers
16. **Advanced transformers:** PatchTST, iTransformer, TFT
17. **Probabilistic models:** DeepAR, Quantile RNN
18. **MLP baselines:** N-BEATS, N-HiTS, DLinear

---

## Section 12: Testing Status

### Test Directory Structure

```
tests/
├── phase_1_tests/         ✅ Exists
│   └── README.md
├── phase_2_tests/         ❓ Unknown status
├── phase_3_tests/         ❓ Unknown status
└── integration/           ❓ Unknown status
```

**Required Investigation:**
- Count existing tests
- Check test coverage (codecov or pytest-cov)
- Identify gaps in test coverage

**Recommended Test Coverage:**
- Phase 1: All 14 stages
- Phase 2: All 13 models (fit/predict/save/load)
- Phase 3: CV, OOF generation, feature selection
- Integration: End-to-end pipeline runs

---

## Section 13: Development Workflow Gaps

### Missing Components

1. **No CI/CD pipeline** (GitHub Actions, GitLab CI)
2. **No pre-commit hooks** (black, ruff, mypy)
3. **No automated testing** on PR/commit
4. **No code coverage reporting**
5. **No linting enforcement**
6. **No type checking** (mypy)

### Recommended Dev Workflow

```
Developer makes change
    ↓
Pre-commit hook runs (black, ruff, mypy)
    ↓
Push to branch
    ↓
CI runs (pytest, coverage)
    ↓
PR review
    ↓
Merge to main
    ↓
Docs auto-deploy (if applicable)
```

---

## Section 14: Key Discrepancies Summary

| Category | Documentation Says | Reality | Impact |
|----------|-------------------|---------|--------|
| **Model Count** | 19 models | 13 models | HIGH - Confusing, misleading |
| **Models Removed** | CatBoost, GRU, RF, SVM, Blending removed | Still exist | HIGH - Docs don't match code |
| **Advanced Models** | InceptionTime, PatchTST, etc. exist | Not implemented | HIGH - False expectations |
| **MTF Timeframes** | 9-TF ladder (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h) | Missing 20m, 25m; has extra 4h, daily | MEDIUM - Needs update |
| **MTF Strategies** | 3 strategies (single-TF, indicators, ingestion) | Only Strategy 2 partial | HIGH - Strategy 1 and 3 missing |
| **Training Timeframe** | Configurable per run | Hardcoded to 5min | MEDIUM - Limits flexibility |
| **Config Parameters** | mtf_strategy, training_timeframe | Don't exist | HIGH - Can't configure MTF |

---

## Section 15: Recommendations

### Immediate Actions (This Week)

1. **Create accurate PROJECT_CHARTER.md**
   - Merge CLAUDE2.md + corrected ALIGNMENT_PLAN.md
   - State actual model count (13)
   - Clarify what's implemented vs planned

2. **Fix ALIGNMENT_PLAN.md**
   - Change "19 models" to "13 models implemented, 6 planned"
   - Add clear "Status" column to model tables
   - Remove misleading "removal list" or clarify it's future work

3. **Add MTF timeframes**
   - Implement Task 1.1 from MTF_IMPLEMENTATION_ROADMAP.md
   - Add 20min and 25min to constants.py

4. **Update README.md**
   - Accurate model count
   - Clear status of what works today
   - Link to PROJECT_CHARTER for vision

### Short-Term Actions (Next 2-4 Weeks)

5. **Implement MTF configuration**
   - Add `mtf_strategy`, `training_timeframe` to PipelineConfig
   - Implement Strategy 1 (single-TF)
   - Implement Strategy 3 (multi-resolution)

6. **Reorganize root documentation**
   - Move guides to `docs/guides/`
   - Move roadmaps to `docs/roadmaps/`
   - Archive inaccurate docs

7. **Test coverage audit**
   - Run pytest with coverage
   - Identify gaps
   - Create test plan

8. **Dev workflow setup**
   - Add pre-commit config
   - Add GitHub Actions CI
   - Add linting (ruff, black, mypy)

### Long-Term Actions (Next 1-3 Months)

9. **Model roadmap decision**
   - Decide: Keep 13 or expand to 19?
   - If expand: Prioritize which 6 to add
   - If keep 13: Update all docs to reflect

10. **Advanced model implementation**
    - If expanding: Start with highest-value models
    - InceptionTime (3 days)
    - PatchTST (4 days)
    - N-BEATS (1 day)

11. **Production deployment guide**
    - Inference server setup
    - Model monitoring
    - A/B testing framework

---

## Section 16: Questions for Decision

### Critical Decisions Needed

1. **Model Count Strategy:**
   - **Option A:** Keep 13 models, update all docs to reflect reality
   - **Option B:** Implement 6 new models to reach 19
   - **Option C:** Remove "redundant" models (CatBoost, GRU, RF, SVM, Blending) → down to 8
   - **Recommendation:** Option A (keep 13, update docs) - most pragmatic

2. **MTF Timeframe Ladder:**
   - **Option A:** 9-TF ladder (add 20m, 25m, remove 4h and daily)
   - **Option B:** Keep current TFs, update docs to match
   - **Recommendation:** Option A (9-TF ladder for consistency)

3. **Documentation Structure:**
   - **Option A:** Keep everything in root
   - **Option B:** Reorganize into docs/ subdirectories
   - **Recommendation:** Option B (cleaner, more discoverable)

4. **Development Priority:**
   - **Option A:** Focus on MTF implementation (Strategies 1 & 3)
   - **Option B:** Focus on adding new advanced models
   - **Option C:** Focus on testing and CI/CD
   - **Recommendation:** Option A then C (MTF first, then quality infrastructure)

---

## Section 17: Effort Estimates

### MTF Implementation (Strategies 1 & 3)

- Task 1.1-1.5 (Phase 1 infra): 1 week
- Task 2.1-2.4 (Strategy 1): 4 days
- Task 4.1-4.5 (Strategy 3): 2 weeks
- **Total:** 3-4 weeks (1 engineer)

### Documentation Reorganization

- Create PROJECT_CHARTER.md: 2 hours
- Fix ALIGNMENT_PLAN.md: 1 hour
- Update README.md: 1 hour
- Reorganize root files: 2 hours
- **Total:** 6 hours (1 engineer)

### Test Coverage & CI/CD

- Audit existing tests: 4 hours
- Add missing unit tests: 1-2 weeks
- Set up pre-commit: 2 hours
- Set up GitHub Actions: 4 hours
- **Total:** 1.5-2.5 weeks (1 engineer)

### Advanced Models (if pursuing)

- InceptionTime: 3 days
- PatchTST: 4 days
- iTransformer: 3 days
- N-BEATS: 1 day
- DLinear: 4 hours
- Chronos wrapper: 3 days
- **Total:** 2-3 weeks per model (including tests)

---

## Section 18: Next Steps

### Immediate (Today/Tomorrow)

1. ✅ **THIS DOCUMENT** - Analyze discrepancies
2. ⏭️ Create PROJECT_CHARTER.md (merge CLAUDE2 + corrected ALIGNMENT_PLAN)
3. ⏭️ Fix ALIGNMENT_PLAN.md to match reality
4. ⏭️ Update README.md with accurate status

### Week 1

5. ⏭️ Add 20min/25min timeframes
6. ⏭️ Add mtf_strategy config parameter
7. ⏭️ Implement Strategy 1 (single-TF)
8. ⏭️ Reorganize root documentation

### Week 2-4

9. ⏭️ Implement Strategy 3 (multi-resolution)
10. ⏭️ Test coverage audit
11. ⏭️ Set up CI/CD
12. ⏭️ Create CONTRIBUTING.md

---

## Conclusion

**The repository is in good shape** - 13 models working, Phase 1 pipeline complete, cross-validation robust.

**The main issues are documentation discrepancies,** not code problems. Once docs are updated to match reality and MTF is fully implemented, this will be a production-grade system.

**Priority:** Fix docs first (low effort, high clarity), then finish MTF (medium effort, high value), then decide on advanced models (high effort, medium value).
