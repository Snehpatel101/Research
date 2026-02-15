# Architecture Integrity Audit

**Auditor:** Architecture Integrity Agent
**Date:** 2026-02-15
**Scope:** Core types, contracts, canonical locations, enforcement

---

## Overview

The ML Factory's architectural foundation is **well-structured** with clear canonical locations for types, contracts, and constants. The contract system covers all 23 models across 6 families. However, there are **duplicate enum definitions** that violate the single-source-of-truth principle, and the CLAUDE.md documentation is outdated regarding model count (says 12, actual is 23).

**Overall Grade: B+** — Strong foundation with some cleanup needed on duplicate enums.

---

## 1. Types Audit (`src/core/types.py`)

### Enums Defined (7 total)

| Enum | Type | Values | Complete? |
|------|------|--------|-----------|
| `DataRank` | `int, Enum` | TABULAR_2D(2), SEQUENCE_3D(3), MULTI_TF_4D(4) | YES — covers all tensor ranks used |
| `ModelFamily` | `str, Enum` | BOOSTING, CLASSICAL, NEURAL, ENSEMBLE, META_LEARNER, TRANSFORMER | YES — 6 families match constants.py |
| `FeatureFamily` | `str, Enum` | RAW, MOMENTUM, MOVING_AVERAGE, VOLATILITY, VOLUME, TREND, PRICE, MICROSTRUCTURE, ENTROPY, WAVELETS, TEMPORAL, REGIME, MTF | YES — 13 values covering all feature categories |
| `TrainingMode` | `str, Enum` | STANDARD, WALK_FORWARD, REGIME_AWARE, META_LABELING | YES |
| `CVMethod` | `str, Enum` | PURGED_KFOLD, CPCV, WALK_FORWARD, PBO, STANDARD | YES |
| `AdapterType` | `str, Enum` | TABULAR, SEQUENCE, MULTI_STREAM | YES — maps 1:1 to DataRank |
| `LabelingMethod` | `str, Enum` | TRIPLE_BARRIER, DIRECTIONAL, THRESHOLD, REGRESSION | YES |

### Type Aliases (8 total)

| Alias | Definition | Purpose |
|-------|-----------|---------|
| `Features` | `np.ndarray \| pd.DataFrame` | Feature input type |
| `Labels` | `np.ndarray` | Label array type |
| `ModelType` | `TypeVar("ModelType", bound="ModelContract")` | Generic model handling |
| `Array1D-4D` | `np.ndarray` (4 aliases) | Shape-documented array types |
| `DatetimeIndex` | `pd.DatetimeIndex` | Index type |
| `Index` | `np.ndarray \| pd.Index` | General index type |

### Singleton Check: PASS
- `class DataRank` — 1 definition (src/core/types.py:32) ✅
- `class ModelFamily` — 1 definition (src/core/types.py:69) ✅

### ModelFamily Coverage vs MODEL_FAMILIES in constants.py

| ModelFamily Enum | constants.py families | Match? |
|-----------------|----------------------|--------|
| BOOSTING | boosting (3 models) | ✅ |
| CLASSICAL | classical (3 models) | ✅ |
| NEURAL | neural (7 models) | ✅ |
| TRANSFORMER | transformer (3 models) | ✅ |
| ENSEMBLE | ensemble (3 models) | ✅ |
| META_LEARNER | meta_learner (4 models) | ✅ |

All 6 families match. Total: 23 models across all families.

---

## 2. Contracts Audit (`src/core/contracts/`)

### Files

| File | Purpose | Status |
|------|---------|--------|
| `__init__.py` | Re-exports all contract components | ✅ Clean |
| `model_contract.py` | ModelContract dataclass + MODEL_CONTRACTS registry | ✅ Complete |
| `data_contract.py` | DataContract + DataContractSchema + FeatureMode/MTFMode enums | ✅ Complete |
| `feature_spec.py` | FeatureSpec (5-dimension optimization) | ✅ Complete |

### ModelContract Dataclass Fields

| Field | Type | Purpose |
|-------|------|---------|
| `model_name` | str | Identity |
| `model_family` | str | Family group |
| `input_rank` | DataRank | 2D/3D/4D |
| `feature_mode` | FeatureMode | engineered/raw/hybrid/oof_probs |
| `mtf_mode` | MTFMode | none/indicators/multi_stream |
| `primary_timeframe` | str | Base timeframe |
| `mtf_timeframes` | tuple[str,...] | Extra timeframes for 4D |
| `sequence_length` | int | For 3D/4D models |
| `patch_length` | int\|None | PatchTST specific |
| `requires_scaling` | bool | Scaling needed |
| `scaler_type` | str | robust/standard/minmax/none |
| `min_features` / `max_features` | int | Feature bounds |

### MODEL_CONTRACTS Registry — All 23 Models

| Category | Models | Rank | Scaling | Count |
|----------|--------|------|---------|-------|
| **Boosting** | xgboost, lightgbm, catboost | 2D | No | 3 |
| **Classical** | random_forest, logistic, svm | 2D | Mixed | 3 |
| **Neural/RNN** | lstm, gru | 3D | Yes | 2 |
| **Neural/CNN** | tcn, inceptiontime, resnet1d | 3D | Yes | 3 |
| **Neural/Other** | tft, nbeats | 3D | Yes | 2 |
| **Transformer** | transformer, patchtst, itransformer | 3D/4D | Yes | 3 |
| **Ensemble** | voting, stacking, blending | 2D | No | 3 |
| **Meta-Learner** | ridge_meta, mlp_meta, calibrated_meta, xgboost_meta | 2D | Mixed | 4 |
| | | | **Total** | **23** |

### DataRank Coverage in Contracts

| DataRank | Models Using It |
|----------|----------------|
| TABULAR_2D | xgboost, lightgbm, catboost, random_forest, logistic, svm, voting, stacking, blending, ridge_meta, mlp_meta, calibrated_meta, xgboost_meta (13) |
| SEQUENCE_3D | lstm, gru, tcn, transformer, tft, nbeats, inceptiontime, resnet1d (8) |
| MULTI_TF_4D | patchtst, itransformer (2) |

All three DataRank values are used. ✅

---

## 3. Contract Enforcement Check

### `get_model_contract()` Usage (13 call sites)

| File | Line(s) | Context |
|------|---------|---------|
| `src/config/unified.py` | 914, 918 | Config validation |
| `src/models/training_utils.py` | 89, 93 | Feature mode checks |
| `src/models/config/trainer_config.py` | 272, 274 | Trainer setup |
| `src/models/config/per_model_config.py` | 91, 93, 347, 349 | Per-model config + meta-learner |
| `src/models/training/evaluation.py` | 224, 226 | Evaluation setup |
| `src/models/training/trainer.py` | 394, 397, 449, 457, 690, 692 | Training pipeline (3 locations) |
| `src/models/training/unified_orchestrator.py` | 309, 311, 387, 393, 443, 445, 694, 703 | Orchestrator (4 locations) |
| `src/models/training/features.py` | 239, 260 | Feature preparation |
| `src/data/adapters/registry.py` | 109, 111 | Adapter routing |

**Assessment:** Contract enforcement is thorough. All critical paths (training, orchestration, adapter routing, config validation) call `get_model_contract()`. ✅

### Key validation methods on ModelContract:
- `validate_data_contract()` — returns (bool, issues)
- `validate_data_contract_strict()` — raises `ModelContractViolation`
- `adapter_id` property — routes to correct adapter
- `requires_sequences` / `requires_multi_timeframe` — shape checks

---

## 4. Canonical Location Compliance

### PASS — Clean Checks

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| `class DataRank` definitions | 1 | 1 | ✅ |
| `class ModelFamily` definitions | 1 | 1 | ✅ |
| `from src.coordination` imports | 0 | 0 | ✅ |
| `from src.feature_selection` imports | 0 | 0 | ✅ |

### ISSUE — Duplicate Enum Definitions

The following enums are defined in **both** `src/core/types.py` AND `src/config/` modules:

| Enum | types.py | Also in | Identical? | Who imports duplicate? |
|------|----------|---------|------------|----------------------|
| `CVMethod` | core/types.py:163 | config/cv.py:28 | YES (same 5 values) | config/__init__.py |
| `LabelingMethod` | core/types.py:213 | config/data.py:52 | YES (same 4 values) | No external imports found |
| `MTFMode` | — | config/data.py:61 | N/A — different scope | data/pipeline/stages/mtf/constants.py |

**Note on MTFMode:** There are TWO different MTFMode concepts:
1. `src/config/data.py:MTFMode` — Pipeline-level (5 values: NONE, BARS, INDICATORS, BOTH, MULTI_STREAM)
2. `src/core/contracts/data_contract.py:ModelMTFMode` (aliased as `MTFMode`) — Model-level (3 values: NONE, INDICATORS, MULTI_STREAM)

These are intentionally different (pipeline vs. model), well-documented in `data_contract.py:42-58`. ✅

**Additional enums in contracts not in types.py:**
- `FeatureMode` (data_contract.py:33) — ENGINEERED, RAW, HYBRID, OOF_PROBS
- `ModelMTFMode` (data_contract.py:42) — NONE, INDICATORS, MULTI_STREAM

These are contract-specific and live in the contracts module. Reasonable location.

---

## 5. Constants Integrity (`src/core/constants.py`)

### Model Registry Consistency

| Check | Status |
|-------|--------|
| `MODEL_FAMILIES` has 6 families | ✅ |
| `ALL_MODELS` assert == 23 | ✅ |
| `MODEL_TO_FAMILY` derived from `MODEL_FAMILIES` | ✅ |
| `MODEL_DATA_RANKS` derived from `MODEL_CONTRACTS` (lazy) | ✅ |
| `MODEL_ADAPTER_MAP` derived from `MODEL_CONTRACTS` (lazy) | ✅ |

Single source of truth chain: `MODEL_CONTRACTS` → derived maps in `constants.py` → used by `types.py` methods. Clean. ✅

### ArtifactManifest

Defined in `src/core/common/manifest.py`, re-exported via `src/core/contracts/__init__.py`. Clean. ✅

---

## 6. Issues Found

### Issue 1: Duplicate CVMethod Enum (LOW)
- **Location:** `src/core/types.py:163` AND `src/config/cv.py:28`
- **Impact:** Low — identical values, but violates single-source-of-truth principle
- **Who uses which:** `src/core/config.py` imports from types.py; `src/config/__init__.py` imports from cv.py
- **Recommendation:** Consolidate to one location (types.py is canonical per CLAUDE.md), have cv.py import from there

### Issue 2: Duplicate LabelingMethod Enum (LOW)
- **Location:** `src/core/types.py:213` AND `src/config/data.py:52`
- **Impact:** Low — identical values, no external imports of the config.data version found
- **Recommendation:** Remove from config/data.py, import from types.py

### Issue 3: CLAUDE.md Model Count Outdated (DOC)
- **CLAUDE.md says:** "All 12 models are production-ready" with a 4-category table
- **Actual:** 23 models across 6 families (boosting, classical, neural, transformer, ensemble, meta_learner)
- **CLAUDE.md lists:** XGBoost, LightGBM, CatBoost, LSTM, GRU, TCN, InceptionTime, 1D ResNet, PatchTST, iTransformer, TFT, N-BEATS
- **Missing from CLAUDE.md:** random_forest, logistic, svm, transformer (vanilla), voting, stacking, blending, ridge_meta, mlp_meta, calibrated_meta, xgboost_meta
- **Recommendation:** Update CLAUDE.md model table to reflect all 23 models

### Issue 4: ModelContract.model_family Uses str, Not ModelFamily Enum (MINOR)
- **Location:** `src/core/contracts/model_contract.py:57` — `model_family: str`
- **Impact:** Minor — model_family field is a plain string rather than using the `ModelFamily` enum from types.py
- **Recommendation:** Consider using `ModelFamily` enum for type safety, though this would be a broader refactor

### Issue 5: TFT Family Classification (OBSERVATION)
- **In MODEL_CONTRACTS:** TFT is classified as `model_family="neural"` (model_contract.py:409)
- **In MODEL_FAMILIES constant:** TFT is in the "neural" list (constants.py:89)
- **Observation:** TFT (Temporal Fusion Transformer) could arguably be "transformer" family. Current classification is internally consistent but worth noting.

---

## 7. Summary

| Category | Status | Details |
|----------|--------|---------|
| Core Types (types.py) | ✅ PASS | All 7 enums complete and well-documented |
| DataRank coverage | ✅ PASS | All 3 ranks used by contracts |
| ModelFamily coverage | ✅ PASS | All 6 families match constants.py |
| Model Contracts | ✅ PASS | All 23 models registered with complete specs |
| Contract Enforcement | ✅ PASS | 13 call sites across all critical paths |
| Dead imports | ✅ PASS | No src.coordination or src.feature_selection imports |
| Singleton classes | ✅ PASS | DataRank and ModelFamily each defined once |
| Duplicate enums | ⚠️ WARN | CVMethod and LabelingMethod duplicated in config/ |
| Documentation | ⚠️ WARN | CLAUDE.md lists 12 models, actual is 23 |
| Type safety | ℹ️ INFO | ModelContract.model_family is str, not ModelFamily enum |

**No critical issues found.** The architecture is sound with proper separation of concerns. The duplicate enums are a cleanup opportunity but don't cause runtime issues since the values are identical.
