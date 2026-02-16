# 05 - Gap Analysis: Synthesized Findings from All Inspection Reports

**Date:** 2026-02-15
**Agent:** 5/10 (Gap Analyst)
**Purpose:** Evidence-based synthesis of reports 01-04 into a precise gap analysis with dependency mapping, conflict resolution, and severity-ranked gap table.

---

## Executive Summary

The deployable-artifact objective requires `predict_from_raw(raw_df) -> PredictionResult` to work for all 12 core model families via a single call. Today, this works for only 4/12 models (3 boosting + MLP). The critical path is: (1) fix the calibrator transfer (works via Trainer attribute but lost in orchestrator's ModelTrainingResult conversion path), (2) fix the confirmed double-scaling bug in `predict_from_raw()`, (3) add adapter routing (2D->3D windowing for 8 models needing 3D, raw multi-TF preprocessing for 2 models needing 4D), (4) make `EnsembleBundle` satisfy the same `predict_from_raw()` contract, and (5) create the deploy directory structure with per-horizon selection and manifest. No new protocols or types files exist yet (`protocols.py`, `ScalingSource` enum). The BundleBuilder's ensemble output is incompatible with `EnsembleBundle.load()`, and the notebook has zero post-training inference or deploy cells.

---

## Current State (Evidence-Based)

### Factory (`src/factory.py`)

- **Entry:** `run()` at L233 returns `ExperimentResult` (L66-97)
- **Bundling:** `_create_bundle()` at L673 calls only `builder.build_from_training_result(training_result)` at L693
- **Output:** `ExperimentResult.bundle_path` = `output_dir / "bundles"` (a flat directory, L694-695)
- **Does NOT:** create `deploy/` directory, write `manifest.json`, select per-horizon artifacts, call `build_ensemble_bundle()`, pass `feature_specs`
- **Dead import:** L54 references non-existent `TrainingResult` (actual type is `TrainingRunResult`)

### Orchestrator (`src/models/training/unified_orchestrator.py`)

- **Entry:** `train()` at L580 returns `TrainingRunResult` (L115-166)
- **Per-model results:** Keyed `"{model}_h{horizon}"` in `self._model_results` (L866)
- **Calibrator flow:** Fitted at L981-982, attached to service result at L993 (`result.calibrator = calibrator`), then LOST at L912-920 when orchestrator creates new `ModelTrainingResult` without a `calibrator` field
- **Ensemble:** Built once across ALL horizons (L633-643), horizon hardcoded to `config.horizons[0]` (L1058) -- NOT per-horizon
- **Two `ModelTrainingResult` classes:** Orchestrator's at L78-111 vs service's at `services/model_training.py` L39-48; neither has a `calibrator` field

### BundleBuilder (`src/inference/builder.py`)

- **Model extraction:** Duck-typing chain at L557-580 (`model`, `_model`, `estimator`, `_estimator`, `get_model()`)
- **Calibrator extraction:** Duck-typing at L630-644, searches trainer for `calibrator`/`_calibrator`/`prob_calibrator` -- always returns `None` because calibrator was set on service result, not trainer
- **Ensemble building:** `build_ensemble_bundle()` at L370-437 saves custom files (`ensemble_metadata.json`, `stacking_dataset.parquet`, `aligned_oof_info.json`) that are NOT loadable by `EnsembleBundle.load()` (expects `metadata.json`, `manifest.json`, `meta_learner/`, `stacking_features.json`, `base_bundles.json`, `alignment_config.json`)
- **PreprocessingGraph creation:** `_create_preprocessing_graph()` at L512-555 uses hardcoded values (`source_timeframe="1min"`, `target_timeframe="5min"`, `scaler_type="robust"`) rather than actual training config

### ModelBundle (`src/inference/bundle.py`)

- **`predict(X)`** at L701-761: Works correctly for pre-shaped 2D, 3D, and 4D inputs (scaling handles all ranks via reshape)
- **`predict_from_raw(raw_df)`** at L1056-1077: Calls `preprocess()` -> returns 2D DataFrame -> passes to `predict()`. Fails for 10/12 models (8 needing 3D + 2 needing 4D) because `predict()` shape validation rejects 2D input for 3D/4D models (L769-775 for 4D DataFrame, L803-804 for 3D ndim check)
- **Double-scaling bug:** `preprocess()` at L1038-1042 calls `transform(skip_scaling=False)` (hardcoded), then `predict()` applies bundle scaler again at L720-752
- **BUNDLE_VERSION:** `"1.2.0"` at L54 (plans say bump to `"1.3.0"`)
- **BundleMetadata** at L70-92: Has `requires_sequences`, `requires_4d`, `sequence_length`, `n_timeframes` but MISSING `primary_timeframe`, `mtf_timeframes`, `feature_mode`

### EnsembleBundle (`src/inference/ensemble_bundle.py`)

- **No `predict_from_raw()`:** Has `predict()` (requires pre-stacked base predictions, L596-628), `predict_from_base_features()` (passes same X to ALL base models, L664-698) -- neither accepts raw OHLCV
- **Base paths stored as raw strings:** `base_bundle_paths` saved as raw strings (relative by default) at L447; absolute paths break on relocation
- **`predict_from_base_features()`:** Passes identical `X` to all base models (L694) -- fails for heterogeneous ensembles with mixed 2D/3D models
- **Meta-learner loading:** Fragile fallback at L552-571; silently sets `meta_learner = None` on import failure

### PreprocessingGraph (`src/inference/preprocessing_graph.py`)

- **`transform()`** at L451-517: Always returns 2D DataFrame (flat features, one row per timestep)
- **Does NOT:** create 3D sliding windows, produce multi-TF raw OHLCV, handle 4D tensor construction
- **Hardcoded assumptions:** `source_timeframe="1min"`, `target_timeframe="5min"` (via BundleBuilder L527-528)

### Notebook (`notebooks/ml_factory_colab.ipynb`)

- **8 cells total** (1 markdown + 7 code): Setup, config, validation, data load, training, results display, save/download
- **Post-training:** Cell 6 prints metrics + paths, Cell 7 zips and downloads
- **Missing:** No deploy directory creation, no artifact selection, no manifest, no inference demo, no bundle validation, no bundle listing

### Core Types & Protocols

- **`src/core/types.py`** (275 lines): 7 enums defined (`DataRank`, `ModelFamily`, `FeatureFamily`, `TrainingMode`, `CVMethod`, `AdapterType`, `LabelingMethod`)
- **`ScalingSource` enum:** Does NOT exist anywhere -- safe to create in `types.py`
- **`src/core/protocols.py`:** Does NOT exist -- must be created
- **`FeatureMode` and `ModelMTFMode`:** Live in `src/core/contracts/data_contract.py` L33-58, NOT in `types.py` (known deviation from CLAUDE.md rule)
- **Model contracts:** 23 total in `MODEL_CONTRACTS` (12 core + 11 non-core), all with `input_rank`, `adapter_id`, `sequence_length`, `feature_mode`, `mtf_timeframes` fields

---

## Target State

### Factory
- After `factory.run()`, produce `deploy/` directory under `output_dir` with per-horizon artifacts and `manifest.json`
- Per-horizon structure: `deploy/h{horizon}/artifact/` containing selected ModelBundle or EnsembleBundle
- Selection policy: ensemble if valid, else best model by configured metric
- `deploy/manifest.json` with `run_id`, `created_at`, `horizons`, `selected_artifacts`, `runtime_profile`, `compatibility`

### Orchestrator
- Transfer calibrators from service result through to `ModelTrainingResult` so BundleBuilder can extract them
- Produce complete `TrainingRunResult` with calibrators intact on trainers or as explicit fields

### BundleBuilder
- Protocol-aware extraction via `TrainerProtocol` (replaces duck-typing fallback chains)
- `build_ensemble_bundle()` must use `EnsembleBundle.from_ensemble_result()` + `.save()` to produce loadable ensemble bundles
- Auto-bundling creates both individual ModelBundles AND EnsembleBundles
- FeatureSpec auto-generation from training context

### ModelBundle
- `predict_from_raw(raw_df)` works for ALL 12 core model types:
  - 2D (boosting): PreprocessingGraph -> pass through -> predict
  - 3D (neural): PreprocessingGraph(skip_scaling=True) -> sliding window -> predict
  - 4D (transformer): Bypass PreprocessingGraph, resample raw OHLCV to multiple TFs -> 4D tensor -> predict
- Single scaling source: `skip_scaling=True` in preprocess(), bundle scaler applies once in predict()
- BundleMetadata v1.3.0 with `primary_timeframe`, `mtf_timeframes`, `feature_mode` fields

### EnsembleBundle
- `predict_from_raw(raw_df)` that loads base bundles, calls `base_bundle.predict_from_raw(raw_df, calibrate=False)` on each, stacks, runs meta-learner
- Relative base bundle paths for portability
- Correct handling of heterogeneous base models (different input shapes)

### Notebook
- Cell 8: Bundle inventory (list all bundles with metadata)
- Cell 9: Deploy artifact selection (per-horizon best model or ensemble, create deploy/ structure, write manifest)
- Cell 10: Validation + inference demo (load artifact, validate(), predict_from_raw() on sample data)
- Cell 11: Export + download (package_bundle(), zip deploy/, trigger download)

### Core Types & Protocols
- `ScalingSource` enum in `src/core/types.py` with values `BUNDLE`, `PREPROCESSING`, `NONE`
- `src/core/protocols.py` with `TrainerProtocol` (defines model, scaler, feature_columns, calibrator attrs) and `InferenceBundle` protocol (defines `predict_from_raw()`, `predict()`, `validate()`, `save()`/`load()`)
- `src/inference/universal_pipeline.py` with `UniversalInferencePipeline` using `ScalingSource`
- `src/inference/errors.py` with `InferenceShapeMismatchError`

---

## Gap Table

| # | Component | Current State | Target State | Gap | Severity | Evidence |
|---|-----------|--------------|-------------|-----|----------|----------|
| G1 | ModelBundle.predict_from_raw | Works for 4/12 models (3 boosting + MLP); fails with ValueError for 3D models at L803-804 and 4D models at L769-775 | Works for all 12 core models via adapter routing | Missing adapter routing between preprocess() and predict() for 10 broken models (8 needing 3D + 2 needing 4D) | **CRITICAL** | bundle.py L1056-1077, L769-775, L803-804 (Report 03 Sec 5) |
| G2 | ModelBundle.preprocess | Calls transform(skip_scaling=False) at L1041 | Must call transform(skip_scaling=True) | Double-scaling bug: preprocessing graph scales, then bundle scaler scales again | **CRITICAL** | bundle.py L1038-1042, L720-752 (Report 03 Sec 7) |
| G3 | Calibrator transfer | Calibrator works via Trainer attribute but lost in orchestrator's ModelTrainingResult conversion path (L912-920); BundleBuilder extraction (builder.py L640-644) always returns None | Calibrator reaches ModelBundle for probability calibration | Calibrator never makes it into bundles | **CRITICAL** | orchestrator.py L912-920, L993; builder.py L640-644 (Report 02 Sec 3.4, 4.3) |
| G4 | EnsembleBundle.predict_from_raw | Method does not exist; only predict() (pre-stacked) and predict_from_base_features() (same X to all) | predict_from_raw(raw_df) orchestrates base bundles and meta-learner | No raw-to-prediction path for ensemble artifacts | **CRITICAL** | ensemble_bundle.py (Report 03 Sec 8, Gaps 1-2) |
| G5 | Deploy directory structure | Factory outputs to flat `bundles/` directory (factory.py L694) | `deploy/h{horizon}/artifact/` + `deploy/manifest.json` per run | No deploy directory, no per-horizon selection, no manifest | **CRITICAL** | factory.py L673-704 (Report 02 Sec 2.6) |
| G6 | BundleBuilder.build_ensemble_bundle | Saves custom file layout (ensemble_metadata.json, stacking_dataset.parquet) at L370-437; NOT loadable by EnsembleBundle.load() | Uses EnsembleBundle.from_ensemble_result() + .save() | Ensemble directory format incompatible with EnsembleBundle.load() | **HIGH** | builder.py L370-437, L398-433 vs ensemble_bundle.py load() L498-594 (Report 03 Sec 9) |
| G7 | Factory ensemble bundling | Factory only calls build_from_training_result() (L693), never build_ensemble_bundle() or build_all() | Factory calls build_all() to create both model and ensemble bundles | Ensemble bundles never created during standard flow | **HIGH** | factory.py L693 (Report 02 Sec 4.5) |
| G8 | Ensemble per-horizon | Ensemble built once across ALL horizons; horizon hardcoded to config.horizons[0] (orchestrator.py L1058) | One ensemble per horizon for per-horizon deploy artifacts | Cannot produce per-horizon ensemble artifacts | **HIGH** | orchestrator.py L633-643, L1058 (Report 02 Sec 3.6) |
| G9 | EnsembleBundle base paths | Paths stored as raw strings (relative by default) at L447; absolute paths break on relocation | Relative paths for portability | Ensemble bundles not portable when absolute | **HIGH** | ensemble_bundle.py L447, L543 (Report 03 Sec 8, Gap 3) |
| G10 | protocols.py | File does not exist | TrainerProtocol + InferenceBundle protocol defined | No protocol-based extraction or inference contract | **HIGH** | Report 04 Sec 5 (confirmed non-existent) |
| G11 | ScalingSource enum | Does not exist anywhere in codebase | Defined in src/core/types.py with BUNDLE/PREPROCESSING/NONE values | No scaling source control enum | **HIGH** | Report 04 Sec 2 (confirmed non-existent); Report 01 Sec 6.1 |
| G12 | UniversalInferencePipeline | File does not exist | src/inference/universal_pipeline.py with ScalingSource-controlled scaling | No unified inference pipeline | **HIGH** | Report 01 Sec 4 (files that don't exist) |
| G13 | Notebook inference demo | No inference or deploy cells; stops at results display + zip download (Cells 6-7) | 4 new cells: bundle inventory, deploy selection, inference demo, export | No post-training inference or deploy UX | **HIGH** | Report 04 Sec 1 (8 cells total, no deploy cells) |
| G14 | BundleMetadata fields | Missing primary_timeframe, mtf_timeframes, feature_mode | All three fields present for complete inference routing | 4D models cannot determine TF resampling targets from metadata alone | **HIGH** | bundle.py L70-92; Report 04 Sec 6.2 |
| G15 | 4D model preprocessing | PreprocessingGraph returns 2D engineered features; 4D models need raw OHLCV at multiple TFs | Dedicated _preprocess_4d() path that bypasses PreprocessingGraph | Fundamentally different preprocessing path needed for patchtst, itransformer | **HIGH** | preprocessing_graph.py L451-517; Report 03 Sec 6; Report 04 Sec 7 |
| G16 | 3D model inference path | preprocess() returns 2D DataFrame; no windowing step before predict() | _adapt_for_model() creates sliding windows using sequence_length from metadata | Missing 2D->3D sliding window step for 7 neural models | **HIGH** | bundle.py L1056-1077 (Report 03 Sec 12) |
| G17 | Feature specs not passed | Factory never passes feature_specs to BundleBuilder (factory.py L693) | feature_specs passed for train/serve parity | Bundles lack feature specification | **MEDIUM** | factory.py L693; builder.py L243 (Report 02 Sec 2.4) |
| G18 | PreprocessingGraph hardcoded config | source_timeframe="1min", target_timeframe="5min", scaler_type="robust" (builder.py L527-547) | Derive from actual pipeline config used during training | Preprocessing graph may not match training preprocessing | **MEDIUM** | builder.py L512-555 (Report 03 Sec 6) |
| G19 | EnsembleBundle.predict_from_base_features | Passes identical X to ALL base models (L694) | Route per-model based on model type (or use predict_from_raw per base) | Fails for heterogeneous ensembles with mixed 2D/3D base models | **MEDIUM** | ensemble_bundle.py L664-698 (Report 03 Sec 8, Gap 2) |
| G20 | BUNDLE_VERSION | "1.2.0" (bundle.py L54) | "1.3.0" after new metadata fields added | Version not bumped for planned metadata extensions | **MEDIUM** | bundle.py L54 (Report 02 Sec 5.6, Report 03 Sec 2) |
| G21 | Two ModelTrainingResult classes | Orchestrator L78-111 vs service L39-48; manual field copy at L912-920 loses dynamic attrs | Single result class or explicit calibrator field on orchestrator version | Confusing dual-class pattern causes attribute loss | **MEDIUM** | orchestrator.py L78-111; services/model_training.py L39-48 (Report 02 Sec 5.1-5.2) |
| G22 | InferenceShapeMismatchError | Does not exist | src/inference/errors.py with descriptive error for shape mismatches | Generic ValueError used instead of domain-specific error | **MEDIUM** | Report 01 Sec 2.5 (planned file) |
| G23 | Validation/smoke test reports | No validation report generated during bundling | validation.json per horizon in deploy/ directory | No artifact validation output | **MEDIUM** | Report 02 Sec 2.6 |
| G24 | EnsembleBundle meta-learner loading | Fragile fallback at L552-571; silently None on import failure | Robust loading with clear error on failure | Meta-learner can silently disappear | **MEDIUM** | ensemble_bundle.py L552-571 (Report 03 Sec 8, Gap 4) |
| G25 | nbeats feature_mode=RAW | PreprocessingGraph generates engineered features; nbeats contract specifies feature_mode=RAW | Different feature set (raw OHLCV or minimal features) + sliding window | nbeats gets wrong feature set from standard preprocessing | **MEDIUM** | Report 04 Sec 3 (nbeats contract); Report 03 Sec 13 |
| G26 | FeatureMode/ModelMTFMode location | In data_contract.py L33-58, not in types.py | Per CLAUDE.md, all enums should be in types.py | Known deviation from canonical location rule | **LOW** | Report 04 Sec 2 (data_contract.py L33-58) |
| G27 | Dead TYPE_CHECKING import | factory.py L54 references non-existent TrainingResult | Remove or fix to TrainingRunResult | Dead import | **LOW** | factory.py L54 (Report 02 Sec 2.6) |
| G28 | tar extractall() deprecation | extract_bundle() at bundle.py L585 uses extractall() without filter param | Use filter parameter per Python 3.12+ recommendation | Deprecation warning in Python 3.12+ | **LOW** | bundle.py L579-585 (Report 03 Sec 10) |
| G29 | Pickle load validation | 16 confirmed raw pickle.load() call sites across codebase (per audit) | safe_pickle_load() with validation | No pickle deserialization validation | **LOW** | Report 01 Sec 5.2 (Warning W-3: verify sites still exist) |

---

## Dependency Map

```
FOUNDATION LAYER (no dependencies, can be parallel):
  G10 (protocols.py)          -- unblocks G6 (protocol-aware BundleBuilder)
  G11 (ScalingSource enum)    -- unblocks G12 (UniversalInferencePipeline)
  G2  (double-scaling fix)    -- one-line fix, unblocks correct feature values
  G3  (calibrator transfer)   -- unblocks calibrated predictions in all bundles
  G14 (BundleMetadata fields) -- unblocks G15, G16 (adapter routing needs metadata)
  G20 (BUNDLE_VERSION bump)   -- pairs with G14

ADAPTER ROUTING LAYER (depends on Foundation):
  G16 (3D windowing)          -- depends on G2, G14
                              -- unblocks G4 (EnsembleBundle.predict_from_raw)
                              -- unblocks G1 for 8/12 models (needing 3D)
  G15 (4D preprocessing)      -- depends on G14
                              -- unblocks G1 for 2/12 models (patchtst, itransformer)
  G25 (nbeats RAW features)   -- depends on G16
                              -- special case within 3D path

ENSEMBLE LAYER (depends on Adapter Routing):
  G6  (fix build_ensemble_bundle)  -- depends on G10 (protocol-aware builder)
  G8  (per-horizon ensemble)       -- depends on G6
  G4  (EnsembleBundle.predict_from_raw) -- depends on G16 (base models must work first)
  G9  (relative paths)             -- depends on G6
  G19 (heterogeneous predict)      -- depends on G4

DEPLOY LAYER (depends on Ensemble):
  G5  (deploy directory)           -- depends on G6, G8
  G7  (factory ensemble bundling)  -- depends on G6
  G23 (validation reports)         -- depends on G5

NOTEBOOK LAYER (depends on Deploy):
  G13 (notebook cells)             -- depends on G5, G4, G1

INFRASTRUCTURE (parallel, lower priority):
  G12 (UniversalInferencePipeline) -- depends on G11
  G22 (InferenceShapeMismatchError)-- no deps
  G17 (feature specs)              -- no deps
  G18 (preprocessing config)       -- no deps
  G21 (dual ModelTrainingResult)   -- pairs with G3
  G24 (meta-learner loading)       -- no deps
  G26-G29 (cleanup)                -- no deps
```

### Critical Path

```
G2 (double-scaling fix) ──────────────────────────────────────┐
G3 (calibrator transfer) ─────────────────────────────────────┤
G14 (BundleMetadata fields) + G20 (version bump) ────────────┤
G10 (protocols.py) ───────────────────────────────────────────┤
                                                              v
                                              G16 (3D windowing) + G15 (4D preprocessing)
                                                              |
                                                              v
                                              G6 (fix build_ensemble_bundle)
                                              G4 (EnsembleBundle.predict_from_raw)
                                                              |
                                                              v
                                              G8 (per-horizon ensemble)
                                              G5 (deploy directory + manifest)
                                              G7 (factory ensemble bundling)
                                                              |
                                                              v
                                              G13 (notebook deploy cells)
                                                              |
                                                              v
                                              DONE: predict_from_raw() for all 12 models
                                              + deploy/ directory with manifest
                                              + notebook inference demo
```

**Minimum viable path (fewest changes to unblock 12/12 models):**
1. G2 -- one-line fix (skip_scaling=True)
2. G3 -- attach calibrator to trainer object in orchestrator
3. G14 + G20 -- add 3 fields to BundleMetadata, bump version
4. G16 -- add _window_2d_to_3d() + _adapt_for_model() to ModelBundle (~20 lines)
5. G15 -- add _preprocess_4d() to ModelBundle (complex, ~80-120 lines)

After these 5, `predict_from_raw()` works for 12/12 models on ModelBundle.

---

## Conflicts Between Docs and Code

### Conflict 1: ScalingSource Location

| Source | Says |
|--------|------|
| MASTER-IMPLEMENTATION-PLAN (Task 3B-2) | Define `ScalingSource` inside `src/inference/universal_pipeline.py` |
| CLAUDE.md | All enums/types in `src/core/types.py` |
| Warning W-1 in plan | "Define in `src/core/types.py`, import in UIP" |

**Resolution:** Define in `src/core/types.py`, import in `universal_pipeline.py`. Warning W-1 is correct; CLAUDE.md rule takes precedence.

### Conflict 2: InferenceBundle Protocol Location

| Source | Says |
|--------|------|
| MASTER-IMPLEMENTATION-PLAN | `InferenceBundle` in `src/inference/` |
| UNIFIED-ROADMAP | `InferenceBundle` in `src/inference/` |
| Warning W-2 in plan | "Define in `src/core/protocols.py` alongside TrainerProtocol" |
| CLAUDE.md | Protocol contracts in `src/core/protocols.py` |

**Resolution:** Define in `src/core/protocols.py`. Warning W-2 is correct; CLAUDE.md rule takes precedence.

### Conflict 3: Adapter Integration -- ModelBundle vs Separate Layer

| Source | Says |
|--------|------|
| UNIFIED-ROADMAP | Both approaches: ModelBundle gets `_build_3d_input()`/`_build_4d_input()` for backward compat; UIP uses own `_adapt_input()` |
| MASTER-IMPLEMENTATION-PLAN (3B-1) | Add methods to ModelBundle |
| Current code | Neither exists |

**Resolution:** Add methods directly to ModelBundle (`_adapt_for_model()`) since it is the primary predict_from_raw() host. UIP can delegate to these methods when it is built. This avoids two parallel implementations.

### Conflict 4: Ensemble Per-Horizon vs Cross-Horizon

| Source | Says |
|--------|------|
| Deploy plan | One deployable artifact per horizon (including ensemble) |
| Current code | Ensemble built once across ALL horizons (orchestrator.py L633-643, L1058) |

**Resolution:** For Phase 3A/3B, the ensemble artifact can be assigned to its training horizon (horizons[0]). Per-horizon ensembles require orchestrator changes (building separate ensembles per horizon) which is a larger refactor. Document this as a known limitation: ensemble artifact covers horizon[0] only until the orchestrator is updated.

### Conflict 5: skip_scaling in predict_from_raw()

| Source | Says |
|--------|------|
| Plan (architecture decision 2.3) | `skip_scaling=True` in `predict_from_raw()` |
| Current code | `skip_scaling=False` hardcoded at bundle.py L1041 |

**Resolution:** Fix code to match plan. Change L1041 to `skip_scaling=True`. This is the double-scaling bug (G2).

### Conflict 6: BundleBuilder Ensemble Output vs EnsembleBundle Format

| Source | Says |
|--------|------|
| BundleBuilder.build_ensemble_bundle() | Saves `ensemble_metadata.json`, `stacking_dataset.parquet`, `aligned_oof_info.json` |
| EnsembleBundle.load() | Expects `metadata.json`, `manifest.json`, `meta_learner/`, `stacking_features.json`, `base_bundles.json`, `alignment_config.json` |

**Resolution:** Rewrite `build_ensemble_bundle()` to use `EnsembleBundle.from_ensemble_result()` + `.save()`. The current custom format is dead code in practice since nothing can load it.

### Conflict 7: Model Count (12 vs 23)

| Source | Says |
|--------|------|
| CLAUDE.md | "12 models production-ready" |
| MODEL_CONTRACTS | 23 entries |

**Resolution:** Already resolved in Phase 1 audit. 12 are core prediction models (used in notebook). 11 additional are classical, ensemble, and meta-learner variants. The deployable artifact objective targets the 12 core models.

### Conflict 8: TFT Classification

| Source | Says |
|--------|------|
| CLAUDE.md | Lists TFT under "Transformer" category |
| MODEL_CONTRACTS | `model_family="neural"`, `input_rank=SEQUENCE_3D` |

**Resolution:** Functionally correct for adapter routing. TFT uses 3D input (sequence adapter), not 4D. The CLAUDE.md categorization is conceptual; the contract's `input_rank=SEQUENCE_3D` is what matters for inference routing.

### Conflict 9: PredictionResult Source

| Source | Says |
|--------|------|
| inference/__init__.py L113 | Re-exports `PredictionResult` from `orchestrator` module |
| src/core/interfaces | Also defines/exports `PredictionResult` |
| src/models/base | Also references `PredictionResult` |

**Resolution:** Needs verification before implementation. Check if all three are the same class re-exported, or genuinely different. If different, consolidate to canonical location per CLAUDE.md rules.

---

*This document synthesizes findings from reports 01-04. Every claim maps to specific file:line evidence from those reports. No code has been modified.*
