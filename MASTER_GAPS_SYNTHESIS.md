# Master Gaps Synthesis: ML Factory Status Report

**Date:** 2026-01-03
**Status:** NOT PRODUCTION-READY
**Test Pass Rate:** 99.3% (16 failing / 2309 total)

---

## Executive Summary

The ML Factory has a **solid architectural foundation** but is **not production-ready** due to:

1. **16 failing tests** (integration mocks, MTF defaults, stacking shape bug)
2. **Missing `train_ensemble.py`** CLI script
3. **Per-model timeframe routing not implemented** (all models use same TF)
4. **Feature selection not enforced** (all models get same ~180 features)
5. **Documentation mismatches** (13 models documented, 23 exist)

---

## Part 1: Test Failures (16 Total)

### 1.1 Integration Test Failures (13 Tests) - CRITICAL

**Root Cause:** `tests/integration/conftest.py:48-56`
**Issue:** Mock `create_mock_container()` missing `return_df=True` parameter

**Failing Tests:**
| Test File | Count | Tests |
|-----------|-------|-------|
| `test_full_pipeline.py` | 6 | xgboost, rf, logistic, save/load, voting, stacking |
| `test_model_comparison.py` | 7 | boosting_vs_classical, 3_class, importance, ensemble, metrics |

**Fix:** Update mock to match real signature:
```python
# Current (broken)
def get_sklearn_arrays(split: str):

# Required
def get_sklearn_arrays(split: str, return_df: bool = False):
```

### 1.2 MTF Test Failures (3 Tests)

**Root Cause:** `src/phase1/stages/mtf/constants.py:50-59`
**Issue:** Default changed from 5-TF to 8-TF, includes invalid `5min` (same as base)

**Failing Tests:**
- `test_pipeline_config.py::test_default_mtf_timeframes`
- `test_mtf_features.py::test_default_initialization`
- `test_mtf_features.py::test_insufficient_data`

**Error:** `ValueError: MTF timeframe 5min must be > base 5min`

**Fix:** Remove `5min` from defaults, update test expectations

### 1.3 Heterogeneous Stacking Shape Bug (1 Test)

**Root Cause:** `src/models/ensemble/stacking.py:584`
**Issue:** Meta-learner validates 2D input against `family="sequence"`

**Failing Test:** `test_heterogeneous_stacking.py::test_heterogeneous_stacking_training`

**Error:**
```
ValueError: X must be 3D (n_samples, seq_len, n_features) for sequential models, got shape (60, 10)
```

**Fix:** Meta-learner should always use `family="tabular"` since it operates on 2D stacked OOF predictions

---

## Part 2: Missing Infrastructure

### 2.1 `scripts/train_ensemble.py` - MISSING

**Status:** File does not exist
**Impact:** Cannot train heterogeneous ensembles from CLI

**Documented (broken):**
```bash
python scripts/train_ensemble.py --base-models catboost,tcn,patchtst \
  --meta-learner logistic --horizon 20
```

**What exists:**
- ✅ `scripts/train_model.py`
- ✅ `scripts/benchmark_ensemble.py`
- ❌ `scripts/train_ensemble.py` **MISSING**

### 2.2 Heterogeneous OOF Pipeline - NOT AUTOMATED

**Components exist but no orchestration:**
- ✅ `src/cross_validation/oof_generator.py`
- ✅ `src/cross_validation/oof_core.py`
- ✅ `src/cross_validation/oof_sequence.py`
- ✅ `src/cross_validation/oof_stacking.py`
- ❌ No script to orchestrate multi-model OOF generation
- ❌ No validation that base models use same CV splits

### 2.3 Per-Model Timeframe Routing - NOT IMPLEMENTED

**Current state:** ALL models train on SAME timeframe (5min from pipeline)

**Charter says:**
- CatBoost → 15min primary TF
- TCN → 5min primary TF
- PatchTST → 1min primary TF

**What's missing:**
- ❌ Per-model primary timeframe configuration
- ❌ Per-model MTF strategy selection
- ❌ Multi-stream data prep for transformers

---

## Part 3: Feature Selection Gap

### 3.1 Research Finding: Models Should NOT Get Same Features

| Model Family | Recommended | Current |
|--------------|-------------|---------|
| **CatBoost/Boosting** | 80-120 engineered + MTF | ~180 (all) |
| **LSTM/GRU** | 40-80 single-TF | ~180 (all) |
| **TCN** | 60-100 single-TF | ~180 (all) |
| **PatchTST** | 4-12 raw OHLCV | ~180 (all) |

### 3.2 Why Different Features Matter

1. **Diversity:** Different features → different errors → better ensemble
2. **Inductive biases:** Transformers learn representations; boosting needs engineered features
3. **Research:** PatchTST with raw OHLCV achieves 21% better MSE than with indicators

### 3.3 Current Implementation

**Feature selection infrastructure EXISTS:**
- ✅ `src/models/feature_selection/manager.py`
- ✅ `src/models/feature_selection/config.py`
- ✅ Walk-forward MDA/MDI selection

**But NOT ENFORCED:**
- ❌ All models receive same feature set by default
- ❌ No per-model feature configs auto-applied
- ❌ PatchTST gets indicators instead of raw OHLCV

---

## Part 4: Documentation Mismatches

| Document Says | Reality |
|---------------|---------|
| "13 base models" | **23 models registered** |
| "5/9 MTF timeframes" | 8 TFs defined (but 5min invalid) |
| "Per-model feature selection" | Configurable, not enforced |
| "`train_ensemble.py` CLI" | **File missing** |
| "Phase 7 complete" | Stacking has shape bug |

---

## Part 5: Architecture Status

### What's Working Well

| Component | Status | Notes |
|-----------|--------|-------|
| Model Registry | ✅ Excellent | 23 models via `@register` |
| Base Interface | ✅ Excellent | Clean `fit/predict/save/load` |
| Leakage Prevention | ✅ Complete | shift(1), purge/embargo, train-only scaling |
| Data Adapters | ✅ Complete | 2D, 3D, 4D adapters |
| PurgedKFold CV | ✅ Complete | Label-aware purging |
| Stacking Logic | ⚠️ Bug | Shape validation error |

### What's Incomplete

| Component | Status | Gap |
|-----------|--------|-----|
| MTF Ladder | ⚠️ Partial | 8 TFs defined, 1 invalid |
| Per-Model Features | ⚠️ Partial | Infrastructure exists, not enforced |
| Per-Model Timeframes | ❌ Missing | All models use same TF |
| `train_ensemble.py` | ❌ Missing | No CLI for heterogeneous |
| Heterogeneous OOF | ❌ Missing | No automated orchestration |

---

## Part 6: Fix Priority

### Immediate (Unblock Tests) - 30 minutes

| Priority | Issue | Fix Location | Time |
|----------|-------|--------------|------|
| 1 | Integration mock | `tests/integration/conftest.py:48` | 5 min |
| 2 | MTF defaults | `src/phase1/stages/mtf/constants.py:50` | 10 min |
| 3 | Stacking shape | `src/models/ensemble/stacking.py:584` | 15 min |

### High Priority (Infrastructure) - 4 hours

| Priority | Issue | Deliverable | Time |
|----------|-------|-------------|------|
| 4 | Missing CLI | Create `scripts/train_ensemble.py` | 2-3 hrs |
| 5 | OOF automation | Add orchestration to OOF pipeline | 1-2 hrs |

### Medium Priority (Architecture) - 2 days

| Priority | Issue | Deliverable | Time |
|----------|-------|-------------|------|
| 6 | Per-model TF routing | Config-driven timeframe selection | 1 day |
| 7 | Per-model features | Auto-apply feature configs by family | 1 day |

### Low Priority (Documentation) - 2 hours

| Priority | Issue | Deliverable | Time |
|----------|-------|-------------|------|
| 8 | Model count | Update CLAUDE.md (13 → 23) | 30 min |
| 9 | MTF status | Clarify 7-TF ladder status | 30 min |
| 10 | Goals vs reality | Audit all claims against code | 1 hr |

---

## Part 7: Recommended Action Plan

### Option A: Quick Fix (30 min)
Fix the 16 failing tests only:
1. Update integration conftest mock signature
2. Fix MTF defaults (remove 5min)
3. Fix stacking meta-learner family

**Result:** Tests pass, but infrastructure gaps remain

### Option B: Stabilization (1 day)
Fix tests + add missing CLI:
1. All Option A fixes
2. Create `scripts/train_ensemble.py`
3. Automate heterogeneous OOF pipeline

**Result:** Heterogeneous ensembles trainable from CLI

### Option C: Full Remediation (1 week)
Complete the architecture:
1. All Option B deliverables
2. Per-model timeframe routing
3. Per-model feature enforcement
4. Documentation alignment
5. End-to-end integration tests

**Result:** Production-ready heterogeneous ML factory

---

## Appendix A: File Size Violations

| File | Lines | Limit | Action |
|------|-------|-------|--------|
| `neural/cnn.py` | 1049 | 800 | Split into `inceptiontime.py` + `resnet1d.py` |
| `ensemble/meta_learners.py` | 1267 | 800 | Split into 4 files |

---

## Appendix B: Feature Selection Summary

### Research-Backed Recommendations

| Model | Features | MTF | Input Type |
|-------|----------|-----|------------|
| CatBoost | 80-120 | Yes (indicators) | Engineered |
| LightGBM | 80-120 | Yes (indicators) | Engineered |
| XGBoost | 80-120 | Yes (indicators) | Engineered |
| LSTM | 40-80 | No | Light engineering |
| GRU | 40-80 | No | Light engineering |
| TCN | 60-100 | No | Light engineering |
| PatchTST | 4-12 | Multi-stream | Raw OHLCV |
| iTransformer | 5-10 | No | Raw OHLCV |
| TFT | 50-100 | Yes (indicators) | Engineered |

### Key Insight

> "All models getting 180 features" violates research best practices. Transformers should get raw OHLCV (4-12 features), not engineered indicators.

---

## Appendix C: Heterogeneous Ensemble Architecture

### Current Flow (Broken)
```
All models → Same 5min TF → Same 180 features → Stacking (shape bug)
```

### Target Flow (Per Charter)
```
1-min Canonical OHLCV
       │
       ├── CatBoost: 15min + MTF → 120 features
       ├── TCN: 5min single-TF → 80 features
       └── PatchTST: 1min multi-stream → 12 raw features
       │
       ▼
   Meta-Learner (2D OOF predictions)
```

---

## Summary

**Bottom Line:** The ML Factory has excellent foundations but needs:

1. **16 test fixes** (30 min)
2. **Missing `train_ensemble.py`** (2-3 hrs)
3. **Per-model routing** (1-2 days)
4. **Documentation sync** (2 hrs)

Total estimated remediation: **3-5 days** for production-ready status.
