# 01 - Document Extraction: Foundational Context for Deployable Artifact Implementation

**Date:** 2026-02-15
**Agent:** 1/10 (Document Extraction)
**Purpose:** Structured handoff of all planning decisions, constraints, and findings for subsequent implementation agents.

---

## 1. Deployable Artifact Objective

**Source:** `.audit/phase3-implementation/PHASE0-DEPLOYABLE-ARTIFACT-PLAN.md`

### Target Behavior

After `notebooks/ml_factory_colab.ipynb` finishes training, the system must output **one deployable artifact per horizon** that accepts raw OHLCV bars and returns predictions in a single call:

```python
prediction = artifact.predict_from_raw(raw_bars_df)
```

No manual feature shaping, no manual adapter calls, no manual model-family branching.

### Two Artifact Types (Same API)

1. **ModelBundle** -- single best model per horizon (when training produces individual models)
2. **EnsembleBundle** -- ensemble artifact per horizon (when ensemble mode is active)

Both must expose the identical `predict_from_raw(raw_df) -> PredictionResult` contract.

### Per-Horizon Output Spec

```
experiments/runs/{run_id}/
  deploy/
    manifest.json              # Run-level manifest
    h{horizon}/
      artifact/                # Selected deployable artifact (directory form)
      artifact.tar.gz          # Optional packaged form
      validation.json          # Smoke validation report
```

### `deploy/manifest.json` Required Fields

- `run_id`
- `created_at`
- `horizons`
- `selected_artifacts` (single vs ensemble)
- `runtime_profile` (`native` / `onnx` / `torchscript`)
- `compatibility` (min runtime versions)

### Runtime Contract (All Artifacts Must Satisfy)

| Method | Purpose |
|--------|---------|
| `predict_from_raw(raw_df) -> PredictionResult` | Raw OHLCV to prediction, single call |
| `predict(features_or_tensor) -> PredictionResult` | Pre-shaped input prediction |
| `validate() -> dict` | Smoke test report |
| `save(path)` / `load(path)` | Serialization |

Protocol location: `src/core/protocols.py` (class `InferenceBundle`).

### Selection Policy

- If ensemble exists and passes validation: choose ensemble artifact.
- Otherwise: best base model by configured metric.
- Decision persisted in `deploy/manifest.json` with metric snapshot.

### Priority Order (from PHASE0 plan)

1. **P0-A:** Bundle reliability and selection (artifacts load, no unresolved refs)
2. **P0-B:** Universal raw-bars inference path (all 12 model families)
3. **P0-C:** Notebook deploy UX (print paths, download, inference demo)
4. **P0-D:** Optional runtime profile (ONNX/TorchScript, never blocks native)

### Definition of Done

A run is complete only if ALL are true:
1. Notebook outputs `deploy/` with one selected artifact per horizon.
2. Artifact loads in a clean process with one command.
3. Artifact accepts raw OHLCV bars and returns predictions in one call.
4. Same API shape for single-model and ensemble artifacts.
5. Simple smoke validation report generated alongside artifact.

---

## 2. Architecture Decisions Already Made

**Source:** `.audit/phase3-implementation/HIGH-LEVEL-DEPLOYABLE-ARTIFACT-ARCHITECTURE.md` and `MASTER-IMPLEMENTATION-PLAN.md`

### Locked-In Patterns

#### 2.1 Runtime Flow (Fixed)

```
raw OHLCV bars
   -> preprocessing graph replay (train/serve parity)
   -> adapter routing (2D / 3D / 4D)
   -> model inference (single or ensemble)
   -> calibration (if present)
   -> PredictionResult
```

#### 2.2 Adapter Routing Decision Table (Fixed)

| BundleMetadata Flags | Adapter Path | Output Shape | Models |
|---------------------|--------------|--------------|--------|
| `requires_4d=True` | `_build_4d_input()` | `(n, tf, seq, feat)` | PatchTST, iTransformer |
| `requires_sequences=True` | `_build_3d_input()` | `(n, seq, feat)` | LSTM, GRU, TCN, InceptionTime, ResNet1D, TFT, N-BEATS |
| Both `False` | Pass through 2D | `(n, feat)` | XGBoost, LightGBM, CatBoost |

#### 2.3 Double-Scaling Prevention (Fixed, 3 Levels)

| Level | Mechanism | Where |
|-------|-----------|-------|
| PreprocessingGraph | `skip_scaling=True` in `predict_from_raw()` | `src/inference/bundle.py` |
| UniversalInferencePipeline | `ScalingSource` enum controls single scaling point | `src/inference/universal_pipeline.py` |
| ModelBundle.predict() | Applies `self.scaler` once (unchanged) | `src/inference/bundle.py` |

#### 2.4 Phase Structure (Fixed)

- **Phase 3A** (Foundation): TrainerProtocol, BundleMetadata extensions, protocol-aware BundleBuilder, calibrator fix, FeatureSpec auto-gen
- **Phase 3B** (Core Inference): Adapter routing in ModelBundle, UniversalInferencePipeline, EnsembleBundle fixes, MTF generation, type alignment
- **Phase 3C** (Integration): Colab cells, server/batch migration, special mode bundles, __init__.py exports
- **Phase 3D** (Cleanup): Dead code, enum consolidation, safe_pickle_load, neural versioning, deprecation warnings

#### 2.5 New Files to Create (Fixed List -- 8 Files)

| File | Phase | Key Classes |
|------|-------|-------------|
| `src/core/protocols.py` | 3A | `TrainerProtocol`, `InferenceBundle` |
| `src/inference/universal_pipeline.py` | 3B | `UniversalInferencePipeline`, `ScalingSource` |
| `src/inference/errors.py` | 3B | `InferenceShapeMismatchError` |
| `src/inference/regime_detector.py` | 3C | `RegimeDetector`, `RegimeDetectorConfig` |
| `src/inference/walk_forward_bundle.py` | 3C | `WalkForwardBundle` |
| `src/inference/regime_bundle.py` | 3C | `RegimeBundle` |
| `src/inference/meta_labeling_bundle.py` | 3C | `MetaLabelingBundle`, `MetaLabelingPrediction` |
| `src/core/utils/safe_pickle.py` | 3D | `safe_pickle_load()` |

#### 2.6 Existing Files to Modify (Fixed List -- 18 Files)

Key modifications:
- `src/inference/bundle.py` -- 6 new BundleMetadata fields, version bump to `1.3.0`, adapter routing methods, `skip_scaling` fix
- `src/inference/builder.py` -- Protocol-aware extraction, FeatureSpec auto-gen, 3 special bundle builders
- `src/models/training/trainer.py` -- `self.scaler` attr, 3 properties, scaler capture
- `src/models/training/unified_orchestrator.py` -- `calibrator` field on `ModelTrainingResult`
- `src/inference/ensemble_bundle.py` -- Relative paths, `predict_from_raw()`

#### 2.7 ONNX Strategy (Fixed)

- ONNX is an **optional runtime profile**, not a mandatory replacement.
- Profiles: `native` (default), `onnx` (optional), `torchscript` (optional for torch models).
- If ONNX export fails, artifact falls back to `native` and records reason in validation output.
- ONNX never blocks native artifact generation.

### Open for Design (Not Yet Locked)

1. **Deploy selector implementation details** -- The selection policy is defined (ensemble if valid, else best model by metric) but the actual selector class/function has not been specified.
2. **Deploy manifest writer** -- The manifest schema is defined but no implementation is prescribed for writing it.
3. **Artifact packaging implementation** -- tar.gz packaging exists (`ModelBundle.package_bundle()`) but the deploy-specific packaging flow is not specified.
4. **Validation report format** -- Required checks are listed (loadability, schema, smoke predict, timing, runtime profile) but the exact JSON schema is open.
5. **Notebook cell content** -- Cell descriptions exist but exact cell code is not prescribed.

---

## 3. Hard Constraints

**Source:** `/home/jake/Desktop/Research/CLAUDE.md`

### 3.1 Canonical Locations (Enforced)

| Thing | Required Location | Consequence |
|-------|------------------|-------------|
| All enums/types | `src/core/types.py` | `ScalingSource` enum MUST be defined here (Warning W-1 in MASTER-IMPLEMENTATION-PLAN) |
| Protocol contracts | `src/core/protocols.py` | `TrainerProtocol` and `InferenceBundle` go here (Warning W-2) |
| Model contracts | `src/core/contracts/` | Do NOT move or duplicate |
| Adapters | `src/data/adapters/` | Inference uses these via registry |
| Feature selection | `src/optimization/feature_selection/` | |
| Validation | `src/validation/` | |

### 3.2 No Duplicate Definitions Rule

- Every class/type/enum must have exactly ONE definition in `src/`.
- Import from canonical location, never redefine.
- Verification: `grep -r "class ClassName" src/ --include="*.py" | wc -l` must return `1`.

### 3.3 Backward Compatibility Requirements

From PHASE0 plan section 3.3:
- Existing bundle loading behavior must remain backward-compatible.
- Existing `predict(X)` must continue to work for pre-shaped inputs.
- Legacy pipeline/orchestrator can be deprecated later, not removed immediately.

From MASTER-IMPLEMENTATION-PLAN rollback strategy:
- All new BundleMetadata fields use `.get()` with safe defaults.
- BundleBuilder falls back to duck-typing if TrainerProtocol not satisfied.
- `InferencePipeline` and `InferenceOrchestrator` get deprecation warnings, NOT deletion.
- Old neural checkpoints load with missing `arch_version` defaulting to `"0.0"`.

### 3.4 Data Leakage Prevention

| Rule | Implementation |
|------|---------------|
| No forward lookups in inference | Past-only windows; all MTF operations use `shift(1)` |
| Exactly one scaling source | `ScalingSource` enum + `skip_scaling=True` |
| Feature ordering pinned to training | Feature columns locked at bundle creation, used at inference |
| Purge/embargo in all CV splits | Already implemented in validation module |

### 3.5 Single Scaling Source Rule

Exactly one scaler applies per prediction path:
- In `predict_from_raw()`: PreprocessingGraph runs with `skip_scaling=True`, then bundle's own scaler applies.
- In UIP: `ScalingSource` enum (`BUNDLE` / `PREPROCESSING` / `NONE`) controls which scaler runs.
- Bundle scaler is canonical (saved as pickle alongside model).

### 3.6 Code Quality Requirements

- `ruff check src/` must pass (0 errors).
- `black src/` must be applied.
- All new files must include `from __future__ import annotations`.
- No magic numbers without context.
- Delete dead code, don't comment it out.

---

## 4. Current Phase Status

**Source:** `/home/jake/Desktop/Research/DIRECTION.md`

### What's Done (Phases 0-50)

The codebase has been through 50+ phases of cleanup and improvement. Key completed work:

| Phase | What Was Done |
|-------|--------------|
| 0-6 | Removed ~5,336 lines of duplicate code, contract enforcement, 4D data infra, adapter error handling, feature manifest, performance optimizations, deprecation cleanup |
| 24-50 | See COMPLETION.md for full details |
| 43 | Pipeline robustness + TCN timeframe fix (auto-resampling in `src/data/adapters/preparation.py`) |
| 47 | Critical fixes: data leakage (bfill->ffill), thread-safe random seeds, notebook model names |
| 48 | Medium fixes: embargo defaults, 3-class probabilities, 5D feature mismatch, registry fallback |
| 49 | Ruff clean sweep: 51 lint issues fixed, all files black-formatted |
| 50 | Speed optimizations, config cleanup, MGC readiness |

### Infrastructure That Exists

| Component | Location | Status |
|-----------|----------|--------|
| MODEL_CONTRACTS (all 12 core models) | `src/core/contracts/` | Complete, includes `requires_sequences`, `requires_4d`, `sequence_length`, `n_timeframes` |
| AdapterRegistry | `src/data/adapters/registry.py` | Complete, `get_for_model()` returns correct adapter |
| PreprocessingGraph | `src/inference/preprocessing_graph.py` | Working for 2D output; does NOT handle 3D/4D |
| BundleMetadata shape flags | `src/inference/bundle.py` | `requires_sequences`, `requires_4d`, `sequence_length`, `n_timeframes` already stored |
| FeatureManifest | `src/data/adapters/` | `BaseAdapter.from_manifest()` exists |
| ModelBundle.predict(X) | `src/inference/bundle.py` | Works for pre-shaped 2D/3D/4D inputs |
| ModelBundle.predict_from_raw(df) | `src/inference/bundle.py` | Works for tabular (4 tabular: 3 boosting + MLP) ONLY; broken for 10/12 models (8 needing 3D + 2 needing 4D) |
| EnsembleBundle | `src/inference/ensemble_bundle.py` | Works but paths stored as raw strings (relative by default); no `predict_from_raw()` |
| BundleBuilder | `src/inference/builder.py` | Works but uses duck-typing, hardcoded assumptions |
| TCN timeframe auto-resampling | `src/data/adapters/preparation.py` | `_detect_timeframe()` and `_resample_for_model()` added in Phase 43 |

### What Does NOT Exist Yet

- `src/core/protocols.py` (TrainerProtocol, InferenceBundle)
- `src/inference/universal_pipeline.py` (UniversalInferencePipeline)
- `src/inference/errors.py` (InferenceShapeMismatchError)
- Special mode bundles (WalkForwardBundle, RegimeBundle, MetaLabelingBundle)
- `src/core/utils/safe_pickle.py`
- Deploy directory structure (`deploy/manifest.json`, per-horizon artifacts)
- Adapter routing inside ModelBundle (`_apply_adapter`, `_build_3d_input`, `_build_4d_input`)

---

## 5. Prior Audit Findings

**Sources:** `.audit/phase1-audit/CONSOLIDATED-FINDINGS.md` and `.audit/phase2-planning/UNIFIED-ROADMAP.md`

### 5.1 Key Gaps Identified (Phase 1 Audit -- 6 Reports)

**THE #1 GAP (identified independently by all 6 audit reports):**

> Only 4 of 12 core models (3 boosting + MLP) have complete raw-OHLCV-to-prediction inference. The other 10 require manual tensor preparation (8 needing 3D + 2 needing 4D). PreprocessingGraph outputs 2D only; neural/transformer models need 3D/4D tensors via SequenceAdapter/MultiStreamAdapter.

| Gap | Severity | Location |
|-----|----------|----------|
| No adapter integration in inference | CRITICAL | `predict_from_raw()` outputs 2D; 10/12 models need 3D/4D (8 needing 3D + 2 needing 4D) |
| Fragile trainer extraction (duck-typing) | HIGH | `src/inference/builder.py` uses `model`, `_model`, `estimator`, `_estimator`, `get_model()` fallback chains |
| End-to-end preprocessing exists for tabular only | HIGH | Exists for tabular via PreprocessingGraph.transform() + predict_from_raw(); missing adapter routing for 3D/4D models |
| Double scaling risk | MEDIUM | Pipeline Stage 7.5 scales parquet; AdapterScaler may scale again |
| FeatureSpec not auto-populated | MEDIUM | Must be passed explicitly to BundleBuilder |
| Calibrator transfer broken | MEDIUM | Calibrator on service result, not transferred to ModelTrainingResult |
| Preprocessing graph hardcodes assumptions | MEDIUM | Hardcodes `source_timeframe="1min"`, `target_timeframe="5min"`, `scaler_type="robust"` |
| No inference demo in notebook | HIGH | Notebook stops at "download zip" |

### 5.2 Serialization Gaps Per Model Family

| Family | Gap |
|--------|-----|
| Boosting (3) | Feature names may be None; shows f0,f1,... instead of names |
| Neural (9) | No architecture version tag; code changes cause cryptic shape mismatches |
| Ensemble | Paths stored as raw strings (relative by default); type mismatch between EnsembleService result and BundleBuilder |
| All | Pickle for scaler/calibrator has no validation; 16 confirmed raw `pickle.load()` call sites |

### 5.3 Critical Path Items

From CONSOLIDATED-FINDINGS.md section 6:

**Must build first (sequential dependencies):**
1. TrainerProtocol (no deps) -- unblocks reliable bundle extraction
2. Adapter integration in ModelBundle (depends on adapter registry) -- unblocks universal raw-to-prediction
3. Double-scaling resolution (no deps) -- unblocks correct feature values
4. Calibrator transfer fix (depends on TrainerProtocol) -- unblocks probability calibration

**Can be parallelized:**
- Colab inference demo, MTF inference data gen, inference-only export, FeatureSpec auto-gen, ensemble path cleanup, dead code removal, enum consolidation, neural arch versioning

### 5.4 Risk Items Relevant to Deployable Artifacts

| Risk | Severity | Mitigation (from plans) |
|------|----------|------------------------|
| Double scaling in inference | MEDIUM | ScalingSource enum + skip_scaling=True + validation |
| Old bundles fail with new metadata | MEDIUM | All new fields use `.get()` with safe defaults; version bump to 1.3.0 |
| 4D models need raw 1min data at inference | LOW | Store `mtf_timeframes` in metadata; clear error if insufficient |
| FeatureSpec auto-gen may fail | LOW | try/except wrapping; best-effort, not required |
| Pickle call sites may not exist as documented | LOW | Verify with grep before building |

### 5.5 Contradictions Found in Phase 1 Audit

1. **Model count:** CLAUDE.md says "12 models production-ready" but MODEL_CONTRACTS has 23 entries (includes classical, ensemble, meta-learner variants). Resolution: 12 are core prediction models; 23 includes all variants.
2. **TFT classification:** MODEL_CONTRACTS says `model_family="neural"` with `input_rank=SEQUENCE_3D`; CLAUDE.md lists TFT under "Transformer." Impact: minor -- TFT uses 3D (not 4D), so "neural" classification is functionally correct for adapter routing.
3. **PredictionResult duplication:** Two classes found in `src.core.interfaces` and `src.models.base`. Needs verification: are they the same class re-exported, or genuinely different?

---

## 6. Conflicts or Ambiguities Found

### 6.1 ScalingSource Location Conflict

- **MASTER-IMPLEMENTATION-PLAN (Task 3B-2)** places `ScalingSource` enum inside `src/inference/universal_pipeline.py`.
- **CLAUDE.md** requires all enums/types in `src/core/types.py`.
- **Architecture constraints check (Warning W-1)** explicitly flags this: "Define in `src/core/types.py`, import in UIP."
- **Resolution for downstream agents:** Define `ScalingSource` in `src/core/types.py`. Import it in `src/inference/universal_pipeline.py`.

### 6.2 InferenceBundle Protocol Location Conflict

- **MASTER-IMPLEMENTATION-PLAN** and **UNIFIED-ROADMAP** describe the `InferenceBundle` protocol as living in `src/inference/`.
- **Architecture constraints check (Warning W-2)** says: "Define in `src/core/protocols.py` alongside TrainerProtocol."
- **CLAUDE.md** requires protocol contracts in `src/core/protocols.py`.
- **Resolution for downstream agents:** Define `InferenceBundle` in `src/core/protocols.py`.

### 6.3 Adapter Integration: ModelBundle vs Separate Layer

- **UNIFIED-ROADMAP Conflict Resolution** (section 1): Both approaches are used. ModelBundle gets `_build_3d_input()`/`_build_4d_input()` for backward compat. UIP uses them OR its own `_adapt_input()` via contracts. UIP is recommended path; ModelBundle methods are fallback.
- **Ambiguity for downstream agents:** When implementing UIP's `_adapt_input()`, should it call `bundle._build_3d_input()` or implement its own windowing? The plans say "both" but the exact delegation is unspecified.

### 6.4 Tasks That May Already Be Done

- **3D-2 (CVMethod consolidation)** and **3D-3 (LabelingMethod consolidation)** -- MASTER-IMPLEMENTATION-PLAN notes "Verify this isn't already done" (architecture check flagged as potential no-op).
- **3D-4 (safe_pickle_load call sites)** -- MASTER-IMPLEMENTATION-PLAN Warning W-3: "Verify call sites still exist before building."
- **Downstream agents MUST verify** with grep before implementing these tasks.

### 6.5 `src/core/protocols.py` -- New File or Existing?

- MASTER-IMPLEMENTATION-PLAN says `src/core/protocols.py` is a NEW file (~45 lines).
- **Downstream agents must check** whether this file already exists before creating it. The plan was written before Phases 43-50 completed; it may have been created during a prior phase.

### 6.6 BundleMetadata `BUNDLE_VERSION` -- Current Value Unknown

- MASTER-IMPLEMENTATION-PLAN says bump from `"1.2.0"` to `"1.3.0"`.
- This was written before Phases 43-50. The current value may differ.
- **Downstream agents must read** `src/inference/bundle.py` to verify the current version before bumping.

### 6.7 Deploy Directory vs Existing Bundle Directory

- The deploy plan specifies `deploy/h{horizon}/artifact/` as the output location.
- The existing system outputs bundles to `experiments/runs/{run_id}/bundles/`.
- **Open question:** Does the deploy selector copy/move from `bundles/` to `deploy/`, or does it create a new artifact? The plans imply copy/selection but the mechanism is unspecified.

### 6.8 Ensemble predict_from_raw Delegation

- HIGH-LEVEL-ARCHITECTURE says EnsembleBundle's `predict_from_raw()` should "load base artifacts, run base `predict_from_raw(raw_df)`, stack/align base outputs, run meta-learner."
- MASTER-IMPLEMENTATION-PLAN (3B-3) says: "Call each `base_bundle.predict_from_raw(raw_df, calibrate=False)`."
- **Note:** The `calibrate=False` parameter is important -- calibration happens at the ensemble level, not per base model. Downstream agents must implement this correctly.

---

## Summary for Downstream Agents

### What you MUST do before writing any code:
1. Verify `src/core/protocols.py` exists or not.
2. Verify current `BUNDLE_VERSION` in `src/inference/bundle.py`.
3. Verify whether CVMethod/LabelingMethod are already consolidated (grep for `class CVMethod` and `class LabelingMethod`).
4. Verify pickle.load call sites still exist (grep for `pickle.load`).
5. Define `ScalingSource` in `src/core/types.py`, NOT in `universal_pipeline.py`.
6. Define `InferenceBundle` protocol in `src/core/protocols.py`, NOT in `src/inference/`.

### Critical execution order:
```
3A-1 (TrainerProtocol) -> 3A-2 (Trainer props) -> 3A-3 (BundleMetadata) -> 3A-4 (BundleBuilder)
    -> 3A-5 (Calibrator) -> 3A-6 (FeatureSpec)
    -> 3B-1 (Adapter routing) -> 3B-2 (UIP) -> 3B-3 (EnsembleBundle)
    -> 3C (Integration) + 3D (Cleanup, parallel)
```

### The single most important thing to get right:
The `predict_from_raw(raw_df) -> PredictionResult` path must work for ALL 12 core model families (3 boosting + 2 RNN + 3 CNN + 3 transformer + 1 MLP) with zero caller-side tensor shaping. This is the entire point of the deployable artifact.
