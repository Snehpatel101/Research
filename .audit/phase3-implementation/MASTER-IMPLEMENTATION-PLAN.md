# Master Implementation Plan: Universal Inference Pipeline

**Date:** 2026-02-15
**Status:** FINAL DELIVERABLE
**Scope:** ~1,680 lines across 26 files (8 new, 18 modified)
**Sources:** Phase 1 audit, Phase 2 roadmap, Phase 3A-3D implementation plans, validation test plan, architecture constraints check

---

## Executive Summary

**What we're building:** A universal inference pipeline that enables all 12 core ML Factory models to go from raw OHLCV data to predictions in a single call. Currently, only 3 of 12 models (the boosting family) support `predict_from_raw()`. The remaining 9 (LSTM, GRU, TCN, InceptionTime, ResNet1D, PatchTST, iTransformer, TFT, N-BEATS) require callers to manually shape tensors to 3D or 4D before prediction.

**Why it matters:** This is the single most impactful gap identified across all 6 Phase 1 audit reports. Without adapter integration in the inference path, bundles produced by training are only partially usable. Colab users download bundles they can't easily run. The system trains 12 model families but can only serve 3 end-to-end.

**Total scope:** ~1,680 lines of change across 4 phases (3A Foundation, 3B Core Inference, 3C Integration, 3D Cleanup), producing 8 new files and modifying 18 existing files.

---

## Phase 3A: Foundation (~237 lines, 5 files)

**Goal:** Establish TrainerProtocol, extend BundleMetadata, make BundleBuilder protocol-aware, fix calibrator transfer, add FeatureSpec auto-generation.

**Execution order:**
```
3A-1 (TrainerProtocol) -> 3A-2 (Trainer properties) -> 3A-5 (Calibrator transfer)
                                                     -> 3A-3 (BundleMetadata) -> 3A-4 (BundleBuilder) -> 3A-6 (FeatureSpec)
```

### Task 3A-1: Create TrainerProtocol
- **File:** `src/core/protocols.py` (NEW, ~45 lines)
- **What:** `@runtime_checkable` Protocol class defining what BundleBuilder expects from trainers: `model`, `scaler`, `feature_columns`, `calibrator`, `training_config`, `model_key`
- **Register:** Add import + `__all__` entry in `src/core/__init__.py`

### Task 3A-2: Add Trainer Properties
- **File:** `src/models/training/trainer.py` (EDIT, ~25 lines)
- **Changes:**
  - Add `self.scaler: Any | None = None` in `__init__` (after line ~107)
  - Add `@property feature_columns` exposing `self._feature_set_columns`
  - Add `@property training_config` returning `self.config.to_dict()`
  - Add `@property model_key` returning `f"{self.config.model_name}_h{self.config.horizon}"`
  - In `run_prepared()` (~line 951): capture `prepared.scaler` and `prepared.feature_names`

### Task 3A-3: Extend BundleMetadata
- **File:** `src/inference/bundle.py` (EDIT, ~30 lines)
- **Changes:**
  - Bump `BUNDLE_VERSION` from `"1.2.0"` to `"1.3.0"`
  - Add 6 new fields: `scaling_source`, `arch_version`, `label_mapping`, `feature_names`, `scaler_type`, `training_run_id`
  - Update `to_dict()` and `from_dict()` (using `.get()` with safe defaults for backward compat)

### Task 3A-4: Protocol-Aware BundleBuilder
- **File:** `src/inference/builder.py` (EDIT, ~80 lines)
- **Changes:**
  - Import `TrainerProtocol` from `src.core.protocols`
  - `_extract_model()`: Check `isinstance(trainer, TrainerProtocol)` first, legacy duck-typing fallback with warning
  - `_extract_scaler()`: Same protocol-first pattern
  - `_extract_feature_columns()`: Same protocol-first pattern
  - `_extract_calibrator()`: Protocol-first + check `model_result.calibrator` (from orchestrator)
  - `_create_preprocessing_graph()`: Replace hardcoded `"1min"`, `"5min"`, `"robust"` with `getattr(self.config, ...)` values

### Task 3A-5: Calibrator Transfer Fix
- **File:** `src/models/training/unified_orchestrator.py` (EDIT, ~15 lines)
- **Changes:**
  - Add `calibrator: Any | None = None` field to `ModelTrainingResult` dataclass
  - Capture `calibrator=getattr(service_result, "calibrator", None)` at both parallel path (line ~799) and sequential path (line ~912)
  - Update `to_dict()` to include `has_calibrator` flag

### Task 3A-6: FeatureSpec Auto-Generation
- **File:** `src/inference/builder.py` (EDIT, ~40 lines)
- **Changes:**
  - New method `_auto_generate_feature_spec(model_result, trainer) -> FeatureSpec | None`
  - Called in `build_from_training_result()` when `feature_spec is None`
  - Best-effort: if FeatureSpec constructor needs more args, try/except handles gracefully

### Phase 3A Validation
```bash
python -c "from src.core.protocols import TrainerProtocol; print('3A-1 OK')"
python -c "from src.core import TrainerProtocol; print('3A-1 re-export OK')"
python -c "from src.inference.bundle import BundleMetadata, BUNDLE_VERSION; assert BUNDLE_VERSION == '1.3.0'; print('3A-3 OK')"
python -c "from src.inference.builder import BundleBuilder; from src.core.protocols import TrainerProtocol; print('3A-4 OK')"
python -c "from src.models.training.unified_orchestrator import ModelTrainingResult; r = ModelTrainingResult(model_name='x', horizon=20, calibrator='test'); assert r.calibrator == 'test'; print('3A-5 OK')"
ruff check src/core/protocols.py src/inference/bundle.py src/inference/builder.py src/models/training/trainer.py src/models/training/unified_orchestrator.py
```

---

## Phase 3B: Core Inference (~870 lines, 6 files)

**Goal:** Wire adapter routing into inference, build UniversalInferencePipeline, fix EnsembleBundle, add MTF generation, bridge type alignment.

**Depends on:** Phase 3A complete.

**Execution order:**
```
3B-1e (skip_scaling fix) -> 3B-1a-d (adapter routing) -> 3B-4 (MTF gen, part of 3B-1c)
                                                       -> 3B-3 (EnsembleBundle fixes, independent)
                                                       -> 3B-5 (type bridge, independent)
                                                       -> 3B-2 (UIP, depends on all above)
```

### Task 3B-1: Adapter Routing in ModelBundle
- **File:** `src/inference/bundle.py` (EDIT, ~180 lines added)
- **Key methods added:**
  - `_apply_adapter(features_2d, raw_df)` - Routes based on `metadata.requires_4d` / `metadata.requires_sequences`
  - `_build_3d_input(features_2d)` - Sliding window via `numpy.lib.stride_tricks.sliding_window_view`, output `(n_seq, seq_len, n_feat)`
  - `_build_4d_input(features_2d, raw_df)` - MTF resample + per-TF windowing, output `(n, tf, seq, feat)`
  - `_generate_mtf_dataframes(raw_1min_df, timeframes)` - Resample 1min OHLCV to multiple timeframes
- **Critical fix:** `preprocess()` now passes `skip_scaling=True` to `preprocessing_graph.transform()` to prevent double-scaling (bundle's own scaler handles it)
- **Updated:** `predict_from_raw()` chains: preprocess -> adapter -> predict

**Adapter routing decision table:**

| BundleMetadata Flags | Adapter Path | Output Shape | Models |
|---------------------|--------------|--------------|--------|
| `requires_4d=True` | `_build_4d_input()` | `(n, tf, seq, feat)` | PatchTST, iTransformer |
| `requires_sequences=True` | `_build_3d_input()` | `(n, seq, feat)` | LSTM, GRU, TCN, InceptionTime, ResNet1D, TFT, N-BEATS |
| Both `False` | Pass through 2D | `(n, feat)` | XGBoost, LightGBM, CatBoost |

### Task 3B-2: UniversalInferencePipeline
- **Files:** `src/inference/universal_pipeline.py` (NEW, ~520 lines), `src/inference/errors.py` (NEW, ~40 lines)
- **Key class:** `UniversalInferencePipeline` - THE single entry point for all inference
- **3 input modes:**
  - Mode 1: Raw OHLCV -> features -> adapt -> predict (`predict_from_raw`)
  - Mode 2: Pre-computed features (2D DataFrame) -> adapt -> predict (`predict`)
  - Mode 3: Pre-shaped tensors (ndarray) -> predict directly (`predict`)
- **Class methods:** `from_bundle()`, `from_bundles()`, `from_experiment()`, `from_training_result()`
- **Core methods:** `predict()`, `predict_from_raw()`, `predict_all()`, `predict_ensemble()`, `predict_batch()`, `predict_with_uncertainty()`
- **`ScalingSource` enum:** `BUNDLE` (default), `PREPROCESSING`, `NONE` - prevents double-scaling
- **Design invariant:** UIP calls `bundle.model.predict()` directly (NOT `bundle.predict()`) to control scaling timing
- **`InferenceShapeMismatchError`:** Custom error with model name, expected rank, actual shape, and contract details

### Task 3B-3: EnsembleBundle Fixes
- **File:** `src/inference/ensemble_bundle.py` (EDIT, ~70 lines)
- **Fix `save()`:** Store relative paths (relative to ensemble bundle parent) instead of absolute
- **Fix `load()`:** Resolve relative paths with absolute fallback for backward compat
- **New method:** `predict_from_raw(raw_df)` - loads base bundles, calls each `predict_from_raw()` (which handles adapter routing), combines via meta-learner

### Task 3B-4: MTF Inference Data Generation
- Included in 3B-1 (`_generate_mtf_dataframes` method on ModelBundle)
- Resamples raw 1min OHLCV to requested timeframes using standard OHLCV aggregation

### Task 3B-5: Type Alignment Bridge
- **File:** `src/models/training/services/ensemble_service.py` (EDIT, ~50 lines)
- **New function:** `to_ensemble_result(service_result, config)` - bridges EnsembleServiceResult to EnsembleResult format for EnsembleBundle
- **File:** `src/inference/__init__.py` (EDIT, ~10 lines) - export UIP + errors

### Double-Scaling Prevention (3 levels)

| Level | Mechanism | Where |
|-------|-----------|-------|
| PreprocessingGraph | `skip_scaling=True` in `predict_from_raw()` | `bundle.py` (3B-1e) |
| UniversalInferencePipeline | `ScalingSource` enum controls single scaling point | `universal_pipeline.py` |
| ModelBundle.predict() | Applies `self.scaler` once (unchanged) | `bundle.py` (existing) |

### Phase 3B Validation
```bash
python -c "from src.inference.bundle import ModelBundle; assert all(hasattr(ModelBundle, m) for m in ['_apply_adapter','_build_3d_input','_build_4d_input','_generate_mtf_dataframes']); print('3B-1 OK')"
python -c "from src.inference.universal_pipeline import UniversalInferencePipeline, ScalingSource; print('3B-2 OK')"
python -c "from src.inference.errors import InferenceShapeMismatchError; assert issubclass(InferenceShapeMismatchError, ValueError); print('3B-2 errors OK')"
python -c "from src.inference.ensemble_bundle import EnsembleBundle; assert hasattr(EnsembleBundle, 'predict_from_raw'); print('3B-3 OK')"
python -c "from src.models.training.services.ensemble_service import to_ensemble_result; print('3B-5 OK')"
python -c "import inspect; from src.inference.bundle import ModelBundle; assert 'skip_scaling=True' in inspect.getsource(ModelBundle.preprocess); print('skip_scaling OK')"
ruff check src/inference/universal_pipeline.py src/inference/errors.py src/inference/bundle.py src/inference/ensemble_bundle.py
```

---

## Phase 3C: Integration (~1,150 lines, 9 files)

**Goal:** Wire everything into consumers: notebook, server, batch, special mode bundles.

**Depends on:** Phase 3B complete (soft dependency for some tasks).

**Execution order:**
```
3C-3 (new bundle files) -> 3C-3e (BundleBuilder methods) -> 3C-5 (__init__.py exports)
3C-2 (server/batch migration) -- parallel, soft dep on 3B
3C-4 (Colab polish) -> 3C-1 (Colab inference demo) -- parallel
```

### Task 3C-1: Colab Inference Demo
- **File:** `notebooks/ml_factory_colab.ipynb` (new cells)
- **Cell 8:** Load best bundle, run predictions on sample data, display SHORT/HOLD/LONG distribution + mean confidence
- **Cell 9:** Inference-only export - zip only `bundles/` directory, show size comparison vs full output

### Task 3C-2: server.py + batch.py Migration
- **Files:** `src/inference/server.py`, `src/inference/batch.py` (EDIT, ~15 lines each)
- Conditional import of `UniversalInferencePipeline` with fallback to `InferencePipeline`
- `from_bundle()` and `from_bundles()` prefer UIP when available
- `__init__` type hints broadened to `InferencePipeline | Any`

### Task 3C-3: Special Mode Bundles (4 new files)

**`src/inference/regime_detector.py`** (NEW, ~160 lines)
- `RegimeDetector`: Serializable market regime detection (volatility percentile, ADX trend)
- `RegimeDetectorConfig`: Configurable thresholds, window sizes, regime names
- Save/load to JSON for exact training-time replay

**`src/inference/walk_forward_bundle.py`** (NEW, ~150 lines)
- `WalkForwardBundle`: Thin wrapper around latest-window ModelBundle
- Delegates prediction to `self.latest_bundle.predict_from_raw()`
- Preserves window metadata (n_windows, window_type, boundaries, aggregated_metrics)

**`src/inference/regime_bundle.py`** (NEW, ~170 lines)
- `RegimeBundle`: Per-regime model routing
- `predict_from_raw()`: detect regime from recent OHLCV -> route to correct per-regime ModelBundle
- Fallback to first regime if detection fails

**`src/inference/meta_labeling_bundle.py`** (NEW, ~200 lines)
- `MetaLabelingBundle`: Primary model + meta-model for position sizing
- `predict_meta()`: direction prediction + P(primary correct) + threshold filtering
- `MetaLabelingPrediction` dataclass: directions, meta_probabilities, positions, trade_mask

All special bundles implement the InferenceBundle protocol: `predict()`, `predict_from_raw()`, `save()`, `load()`, `validate()`.

### Task 3C-3e: BundleBuilder Additions
- **File:** `src/inference/builder.py` (EDIT, +120 lines)
- 3 new methods: `build_walk_forward_bundle()`, `build_regime_bundle()`, `build_meta_labeling_bundle()`

### Task 3C-4: Colab Polish
- **File:** `notebooks/ml_factory_colab.ipynb` (6 cell modifications + 1 new cell)
- Cell 0: Add inference section to Quick Start
- Cell 1: Add torch version check (>=2.2.0)
- Cell 2: Add bundling config toggles (`CREATE_BUNDLE`, `BUNDLE_FORMAT`, `INCLUDE_OOF`, `INCLUDE_FEATURE_IMPORTANCE`)
- Cell 3: Add memory/VRAM warnings for large experiments
- Cell 5: Add ephemeral filesystem warning
- Cell 6: Add bundle summary section
- Cell 7b (NEW): Google Drive mount + save bundles for persistence

### Task 3C-5: Update `__init__.py` Exports
- **File:** `src/inference/__init__.py` (EDIT, +20 lines)
- Export: `WalkForwardBundle`, `RegimeBundle`, `MetaLabelingBundle`, `RegimeDetector`, `UniversalInferencePipeline`
- Add deprecation `__getattr__` for old class names

### Phase 3C Validation
```bash
python -c "from src.inference.regime_detector import RegimeDetector; print('OK')"
python -c "from src.inference.walk_forward_bundle import WalkForwardBundle; print('OK')"
python -c "from src.inference.regime_bundle import RegimeBundle; print('OK')"
python -c "from src.inference.meta_labeling_bundle import MetaLabelingBundle; print('OK')"
python -c "from src.inference import WalkForwardBundle, RegimeBundle, MetaLabelingBundle; print('OK')"
python -c "from src.inference.server import ModelServer; print('OK')"
python -c "from src.inference.batch import BatchPredictor; print('OK')"
ruff check src/inference/regime_detector.py src/inference/walk_forward_bundle.py src/inference/regime_bundle.py src/inference/meta_labeling_bundle.py
```

---

## Phase 3D: Cleanup (~80 lines added, ~70 removed, 17 files)

**Goal:** Dead code removal, enum consolidation, pickle security, neural versioning, deprecation warnings.

**Dependencies:** NONE. Runs fully parallel with 3A-3C.

### Task 3D-1: Remove `_apply_regime()` No-Op
- **File:** `src/inference/preprocessing_graph.py` (DELETE 8 lines)
- Remove call site (lines 496-498) and method definition (lines 702-706)

### Task 3D-2: Consolidate CVMethod Enum
- **File:** `src/config/cv.py` (EDIT, ~4 lines)
- Remove duplicate `class CVMethod` definition, replace with `from src.core.types import CVMethod`
- **NOTE:** Verify this isn't already done (architecture check flagged it may be a no-op)

### Task 3D-3: Consolidate LabelingMethod Enum
- **File:** `src/config/data.py` (EDIT, ~4 lines)
- Remove duplicate `class LabelingMethod` definition, replace with `from src.core.types import LabelingMethod`
- **NOTE:** Same verification needed as 3D-2

### Task 3D-4: `safe_pickle_load()` Utility
- **File:** `src/core/utils/safe_pickle.py` (NEW, ~40 lines)
- Centralizes pickle loading with path validation and optional type checking
- **16 call sites** across 13 files to migrate (see detailed list below)
- **NOTE:** Verify call sites still exist before building (W-3 from architecture check)

**Call sites:**
1. `src/factory.py:474`
2. `src/models/boosting/xgboost_model.py:295`
3. `src/models/boosting/catboost_model.py:281`
4. `src/models/boosting/lightgbm_model.py:352`
5. `src/models/calibration/conformal.py:482`
6. `src/models/calibration/calibrator.py:307`
7. `src/models/ensemble/xgboost_meta.py:279`
8. `src/data/pipeline/stages/scaling/scaler.py:499`
9. `src/core/utils/checkpoint_manager.py:208`
10. `src/core/utils/checkpoint_manager.py:224`
11. `src/core/utils/cache.py:190`
12. `src/inference/bundle.py:644`
13. `src/inference/bundle.py:653`
14. `src/inference/ensemble_bundle.py:561`
15. `src/inference/ensemble_bundle.py:580`
16. `src/inference/preprocessing_graph.py:431`

### Task 3D-5: Neural Architecture Versioning
- **File:** `src/models/neural/base_rnn.py` (EDIT, ~15 lines)
- Add `ARCH_VERSION = "1.0"` constant
- Save in checkpoint dict during `save()`
- Validate on `load()` with warning (not error) for version mismatch

### Task 3D-6: Feature Names in BundleBuilder
- **File:** `src/inference/builder.py` (EDIT, ~3 lines)
- After extracting model, call `model.set_feature_names(feature_columns)` if supported

### Task 3D-7: Deprecation Warnings
- **Files:** `src/inference/pipeline.py`, `src/inference/orchestrator.py` (EDIT, ~10 lines each)
- Add `warnings.warn("... deprecated. Use UniversalInferencePipeline instead.", DeprecationWarning)` in `__init__()`

### Phase 3D Validation
```bash
grep -c "_apply_regime" src/inference/preprocessing_graph.py  # Should be 0
grep -r "class CVMethod" src/ --include="*.py" | wc -l  # Should be 1
grep -r "class LabelingMethod" src/ --include="*.py" | wc -l  # Should be 1
python -c "from src.core.utils.safe_pickle import safe_pickle_load; print('OK')"
grep -r "pickle\.load(" src/ --include="*.py" | grep -v safe_pickle | grep -v "#" | wc -l  # Should be 0
python -c "from src.models.neural.base_rnn import ARCH_VERSION; print(f'ARCH_VERSION={ARCH_VERSION}')"
```

---

## Architecture Compliance

### Constraint Check Results (from architecture-constraints-check.md)

| Area | Status | Notes |
|------|--------|-------|
| Canonical locations | **PASS** (with 2 warnings) | W-1: Move `ScalingSource` to `src/core/types.py`. W-2: Move `InferenceBundle` protocol to `src/core/protocols.py` |
| No duplicate definitions | **PASS** | All new classes are unique |
| Import patterns | **PASS** | All imports follow canonical patterns |
| Clean code principles | **PASS** | Dead code removed, no magic numbers |
| No data leakage | **PASS** | Double-scaling prevented, past-only windows, feature columns locked |
| Backward compatibility | **PASS** | Old bundles load, old trainers work via fallback, deprecated classes still importable |
| Project structure | **PASS** | All 8 new files fit existing directory structure |
| Linting compliance | **PASS** (with note) | All new files must include `from __future__ import annotations` |

### Warnings to Address During Implementation

| ID | Issue | Fix |
|----|-------|-----|
| W-1 | `ScalingSource` enum in `universal_pipeline.py` instead of `src/core/types.py` | Define in `src/core/types.py`, import in UIP |
| W-2 | `InferenceBundle` protocol in `src/inference/` instead of `src/core/` | Define in `src/core/protocols.py` alongside `TrainerProtocol` |
| W-3 | `safe_pickle_load` call sites may not exist | Verify with `grep -r "pickle.load" src/` before building |
| N-2 | Tasks 3D-2/3D-3 may already be complete | Verify before executing |
| N-3 | All new files need `from __future__ import annotations` | Add to implementation checklist |

---

## Dependency-Safe Execution Order

### Critical Path
```
Step 1:  3A-1 TrainerProtocol (NEW file)                    [CHECKPOINT: import test]
Step 2:  3A-2 Trainer properties                             [CHECKPOINT: protocol satisfaction test]
Step 3:  3A-3 BundleMetadata extensions                      [CHECKPOINT: round-trip test]
Step 4:  3A-4 Protocol-aware BundleBuilder                   [CHECKPOINT: legacy fallback test]
Step 5:  3A-5 Calibrator transfer fix                        [CHECKPOINT: calibrator field test]
Step 6:  3A-6 FeatureSpec auto-generation
         === PHASE 3A COMPLETE === [CHECKPOINT: full 3A validation + ruff/black]

Step 7:  3B-1e Fix skip_scaling in preprocess()              [CHECKPOINT: inspect source]
Step 8:  3B-1a-d Adapter routing methods on ModelBundle      [CHECKPOINT: method existence test]
Step 9:  3B-3 EnsembleBundle fixes (parallel with 8)
Step 10: 3B-5 Type alignment bridge (parallel with 8)
Step 11: 3B-2 UniversalInferencePipeline (NEW files)         [CHECKPOINT: import + method test]
         === PHASE 3B COMPLETE === [CHECKPOINT: full 3B validation + ruff/black]

Step 12: 3C-3 Special mode bundle files (4 NEW)              [CHECKPOINT: import tests]
Step 13: 3C-3e BundleBuilder additions                       [CHECKPOINT: method existence]
Step 14: 3C-5 Update __init__.py exports                     [CHECKPOINT: re-export test]
Step 15: 3C-2 server.py + batch.py migration (parallel)
Step 16: 3C-4 Colab polish (parallel)
Step 17: 3C-1 Colab inference demo cells
         === PHASE 3C COMPLETE === [CHECKPOINT: full 3C validation + ruff/black]
```

### Parallel Track (Phase 3D - any time)
```
Step D1: 3D-1 Remove _apply_regime() no-op
Step D2: 3D-2 + 3D-3 Enum consolidation (verify first!)
Step D3: 3D-7 Deprecation warnings
Step D4: 3D-6 Feature names in BundleBuilder
Step D5: 3D-5 Neural arch versioning
Step D6: 3D-4 safe_pickle_load + 16 call site migrations    [CHECKPOINT: grep for raw pickle.load]
         === PHASE 3D COMPLETE === [CHECKPOINT: full 3D validation + ruff/black]
```

### What Can Be Parallelized vs Sequential

| Parallel Groups | Sequential Dependencies |
|----------------|----------------------|
| 3A-3 and 3A-5 (independent) | 3A-1 -> 3A-2 -> 3A-4 -> 3A-6 |
| 3B-3, 3B-5 (independent of 3B-1) | 3B-1 -> 3B-2 |
| 3C-2, 3C-4, 3C-1 (mostly independent) | 3C-3 -> 3C-3e -> 3C-5 |
| ALL of 3D (parallel with 3A-3C) | 3D-4 call sites are sequential |

---

## File Change Summary

### 8 New Files

| File | Phase | Lines | Description |
|------|-------|-------|-------------|
| `src/core/protocols.py` | 3A | ~45 | TrainerProtocol + InferenceBundle protocols |
| `src/inference/universal_pipeline.py` | 3B | ~520 | UniversalInferencePipeline, ScalingSource, InferenceResult, EnsembleInferenceResult |
| `src/inference/errors.py` | 3B | ~40 | InferenceShapeMismatchError |
| `src/inference/regime_detector.py` | 3C | ~160 | RegimeDetector, RegimeDetectorConfig |
| `src/inference/walk_forward_bundle.py` | 3C | ~150 | WalkForwardBundle, WindowConfig |
| `src/inference/regime_bundle.py` | 3C | ~170 | RegimeBundle |
| `src/inference/meta_labeling_bundle.py` | 3C | ~200 | MetaLabelingBundle, MetaLabelingPrediction |
| `src/core/utils/safe_pickle.py` | 3D | ~40 | safe_pickle_load() utility |

### 18 Modified Files

| File | Phase | Change Summary |
|------|-------|---------------|
| `src/core/__init__.py` | 3A | Add TrainerProtocol import + __all__ |
| `src/models/training/trainer.py` | 3A | Add scaler attr, 3 properties, scaler capture in run_prepared() |
| `src/inference/bundle.py` | 3A+3B | 6 new metadata fields, version bump, adapter routing methods, skip_scaling fix, MTF generation |
| `src/inference/builder.py` | 3A+3C | Protocol-aware extraction, FeatureSpec auto-gen, 3 special bundle builders, feature names |
| `src/models/training/unified_orchestrator.py` | 3A | Calibrator field on ModelTrainingResult |
| `src/inference/ensemble_bundle.py` | 3B | Relative paths in save/load, predict_from_raw() |
| `src/models/training/services/ensemble_service.py` | 3B | to_ensemble_result() bridge function |
| `src/inference/__init__.py` | 3B+3C | Export UIP, special bundles, deprecation __getattr__ |
| `src/inference/server.py` | 3C | Conditional UIP import, prefer UIP in from_bundle() |
| `src/inference/batch.py` | 3C | Conditional UIP import, prefer UIP in from_bundle() |
| `notebooks/ml_factory_colab.ipynb` | 3C | 6 cell edits + 3 new cells |
| `src/inference/preprocessing_graph.py` | 3D | Remove _apply_regime() no-op |
| `src/config/cv.py` | 3D | Import CVMethod from core.types instead of defining |
| `src/config/data.py` | 3D | Import LabelingMethod from core.types instead of defining |
| `src/models/neural/base_rnn.py` | 3D | ARCH_VERSION constant, save/load versioning |
| `src/inference/pipeline.py` | 3D | DeprecationWarning in __init__() |
| `src/inference/orchestrator.py` | 3D | DeprecationWarning in __init__() |
| 13 files with pickle.load | 3D | Migrate to safe_pickle_load() |

---

## Risk Register

### Top 5 Risks with Mitigations

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| 1 | **Double scaling** - pipeline scaler + bundle scaler apply twice | MEDIUM | ScalingSource enum + skip_scaling=True in predict_from_raw() + validate in tests |
| 2 | **Backward incompatibility** - old bundles fail to load with new metadata | MEDIUM | All new BundleMetadata fields use `.get()` with safe defaults; version bump to 1.3.0 |
| 3 | **FeatureSpec constructor mismatch** - auto-generation may fail if constructor needs more args | LOW | try/except wrapping; best-effort auto-generation, not required |
| 4 | **Pickle call sites may not exist** - 3D-4 utility may be unnecessary | LOW | Verify with grep before building; skip if already clean |
| 5 | **4D model inference needs raw 1min data** - callers may not have it | LOW | Clear error message; store `mtf_timeframes` in metadata; document requirement |

### Rollback Strategy

If issues arise during implementation:

1. **Per-phase rollback:** Each phase is a clean git commit boundary. `git revert` the phase commit to undo all changes.
2. **Legacy fallback always works:** All protocol-aware code falls back to duck-typing with warnings. Old trainers, old bundles, old prediction paths remain functional.
3. **New classes are additive:** `UniversalInferencePipeline`, special mode bundles, and `safe_pickle_load` are new code that doesn't modify existing behavior. Deleting the new files restores prior state.
4. **Deprecation warnings only:** `InferencePipeline` and `InferenceOrchestrator` are deprecated with warnings, NOT deleted. They continue to work.
5. **BundleMetadata is forward-compatible:** Old metadata (v1.2.0) loads fine with defaults. New metadata (v1.3.0) round-trips correctly. Removing new fields from the class restores v1.2.0 behavior.

---

## Validation Checklist

### Final Smoke Test Sequence (run after all phases)

```bash
#!/bin/bash
set -e

echo "=== New Imports ==="
python -c "from src.core.protocols import TrainerProtocol; print('  TrainerProtocol OK')"
python -c "from src.inference.universal_pipeline import UniversalInferencePipeline, ScalingSource; print('  UIP OK')"
python -c "from src.inference.errors import InferenceShapeMismatchError; print('  Errors OK')"
python -c "from src.inference.walk_forward_bundle import WalkForwardBundle; print('  WalkForwardBundle OK')"
python -c "from src.inference.regime_bundle import RegimeBundle; print('  RegimeBundle OK')"
python -c "from src.inference.regime_detector import RegimeDetector; print('  RegimeDetector OK')"
python -c "from src.inference.meta_labeling_bundle import MetaLabelingBundle; print('  MetaLabelingBundle OK')"
python -c "from src.core.utils.safe_pickle import safe_pickle_load; print('  safe_pickle_load OK')"

echo ""
echo "=== Existing Imports (Must Not Break) ==="
python -c "from src.inference import ModelBundle; print('  ModelBundle OK')"
python -c "from src.inference import InferencePipeline; print('  InferencePipeline OK')"
python -c "from src.inference import InferenceOrchestrator; print('  InferenceOrchestrator OK')"
python -c "from src.inference import EnsembleBundle; print('  EnsembleBundle OK')"
python -c "from src.inference import BundleBuilder; print('  BundleBuilder OK')"
python -c "from src.config.cv import CVMethod; print('  CVMethod OK')"
python -c "from src.config.data import LabelingMethod; print('  LabelingMethod OK')"
python -c "from src.core.types import DataRank, ModelFamily; print('  DataRank/ModelFamily OK')"
python -c "from src.core.contracts import get_model_contract; print('  get_model_contract OK')"

echo ""
echo "=== Single Definition Checks ==="
echo -n "  TrainerProtocol definitions: "; grep -r "class TrainerProtocol" src/ --include="*.py" | wc -l
echo -n "  CVMethod definitions: "; grep -r "class CVMethod" src/ --include="*.py" | wc -l
echo -n "  LabelingMethod definitions: "; grep -r "class LabelingMethod" src/ --include="*.py" | wc -l
echo -n "  Raw pickle.load calls: "; grep -r "pickle\.load(" src/ --include="*.py" | grep -v "safe_pickle" | grep -v "#" | wc -l

echo ""
echo "=== Ruff + Black ==="
ruff check src/
black --check src/

echo ""
echo "=== ALL VALIDATIONS PASSED ==="
```

### Critical Path Tests (Go/No-Go)

1. BundleMetadata backward compat - old bundles must load
2. TrainerProtocol + legacy fallback - old trainers must still work
3. Adapter routing shape correctness - 3D windowing must be `(n, seq, feat)`
4. Existing imports unbroken - all items in validation script
5. Single definition counts - TrainerProtocol=1, CVMethod=1, LabelingMethod=1
6. Zero raw pickle.load after 3D-4
7. Ruff + Black clean

---

## Colab Deliverables Summary

| Priority | Cell | Type | Description |
|----------|------|------|-------------|
| 1 | Cell 2 | MODIFY | Add 4 bundling config toggles |
| 2 | Cell 5 | MODIFY | BundlingSection + ephemeral FS warning |
| 3 | Cell 6 | MODIFY | Bundle summary (model count, families, size) |
| 4 | Cell 7b | NEW | Google Drive mount + save bundles |
| 5 | Cell 9 | NEW | Inference-only export (smaller zip) |
| 6 | Cell 8 | NEW | Inference demo (load bundle, predict, display results) |
| 7 | Cell 0 | MODIFY | Add inference section to Quick Start |
| 8 | Cell 1 | MODIFY | Torch version check (>=2.2.0) |
| 9 | Cell 3 | MODIFY | Memory/VRAM warnings for large experiments |

---

## Reference

For full code diffs, exact line numbers, and complete new file contents, see the detailed per-phase plans:
- **Phase 3A details:** `phase3a-foundation-impl.md`
- **Phase 3B details:** `phase3b-core-inference-impl.md`
- **Phase 3C details:** `phase3c-integration-impl.md`
- **Phase 3D details:** `phase3d-cleanup-impl.md`
- **Test plan:** `validation-test-plan.md`
- **Architecture check:** `architecture-constraints-check.md`

---

*Generated: 2026-02-15*
*This document is the FINAL deliverable of the 3-phase audit operation (Phase 1: Audit, Phase 2: Planning, Phase 3: Implementation Planning).*
*It is sufficient for an engineer to implement the entire universal inference pipeline without additional context.*
