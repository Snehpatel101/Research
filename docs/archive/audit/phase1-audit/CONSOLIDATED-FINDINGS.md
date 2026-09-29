# Phase 1 Consolidated Findings

**Date:** 2026-02-15
**Sources:** 6 Phase 1 audit reports (inference-pipeline, model-packaging, data-flow, architecture-integrity, colab-readiness, training-output-flow)
**Purpose:** Primary input for Phase 2 planning — universal inference pipeline

---

## 1. Current State Map

### Training Path (Working)

```
Raw OHLCV (parquet)
    ↓
MLFactory.run() [src/factory.py]
    ├── Phase 1: Data Pipeline (12 stages via runner.py)
    │     Raw → Clean → Features (~150+) → Labels → Splits → Scale → Validate
    ├── Phase 2: Training
    │     ExperimentConfig → PipelineConfig → UnifiedTrainingOrchestrator.train()
    │     ├── DataPreparer → UnifiedDataPreparation → Adapter routing (2D/3D/4D)
    │     ├── Parallel boosting + sequential neural training
    │     ├── OOF generation → EnsembleService → Meta-learner
    │     └── ArtifactManager saves models, OOF, config, metrics
    ├── Phase 3: Evaluation (backtesting)
    └── Phase 4: Bundling (BundleBuilder → ModelBundle/EnsembleBundle)
```

### Inference Path (Partially Working)

```
ModelBundle.load(bundle_dir)
    ├── predict(X)              — Works for pre-shaped inputs (2D/3D/4D)
    ├── predict_from_raw(df)    — Works for TABULAR models only (2D)
    │     └── PreprocessingGraph.transform() → 2D features → predict
    └── BROKEN for neural/transformer: predict_from_raw() outputs 2D,
        but 3D/4D models need adapter reshaping that isn't integrated
```

### Notebook Path (Training Only)

```
Colab notebook [notebooks/ml_factory_colab.ipynb]
    ├── Cells 1-4: Setup, config, validation, data load
    ├── Cell 5: MLFactory.run() (triggers all 4 phases including bundling)
    ├── Cell 6: Display results (prints bundle_path but doesn't use it)
    └── Cell 7: Zip & download entire output_dir (no inference demo)
```

### Key Connections and Disconnections

| Connection | Status | Details |
|-----------|--------|---------|
| Training → ModelBundle | **Working** | BundleBuilder.build_from_training_result() extracts model, scaler, features |
| Training → EnsembleBundle | **Working** | BundleBuilder.build_ensemble_bundle() saves meta-learner + alignment |
| Bundle → Predict (tabular) | **Working** | ModelBundle.predict(X_2d) for boosting models |
| Bundle → Predict (neural) | **Partial** | Requires caller to pre-shape input to 3D/4D |
| Bundle → Raw OHLCV → Predict | **Broken for 8/12 core models** | PreprocessingGraph outputs 2D only; no adapter integration. 4 tabular models (XGBoost, LightGBM, CatBoost, MLP) work end-to-end; 8 others need 3D/4D adapter routing |
| Notebook → Bundles | **Created but unused** | Bundles exist in zip but no inference demo cell |
| Notebook → Deployment | **Missing** | No ONNX/TorchScript export; bundles require full src/ package |

---

## 2. Gap Analysis

### 2.1 Training → Inference Automation Gaps

| Gap | Severity | Report Source | Details |
|-----|----------|--------------|---------|
| **No adapter integration in inference** | CRITICAL | inference-pipeline, data-flow | `PreprocessingGraph.transform()` produces 2D DataFrame; neural/transformer models need 3D/4D tensors via SequenceAdapter/MultiStreamAdapter. The `predict_from_raw()` path is incomplete for 8 of 12 core models (6 needing 3D + 2 needing 4D). 4 tabular models (XGBoost, LightGBM, CatBoost, MLP) work end-to-end via PreprocessingGraph.transform() + predict_from_raw(). |
| **Fragile trainer extraction** | HIGH | training-output-flow | BundleBuilder uses duck-typing fallback chains (`model`, `_model`, `estimator`, `_estimator`, `get_model()`) instead of a defined protocol. Silent failure if naming changes. |
| **No end-to-end preprocessing replay for 3D/4D models** | HIGH | data-flow | PreprocessingGraph.transform() + predict_from_raw() provide end-to-end raw OHLCV → predictions for tabular/boosting models. The gap is specifically for 3D/4D models where adapter routing is missing in the inference path. |
| **Double scaling risk** | MEDIUM | data-flow | Pipeline Stage 7.5 scales features in parquet. UnifiedDataPreparation applies AdapterScaler again by default. If both apply, features are scaled twice. |
| **FeatureSpec not auto-populated** | MEDIUM | training-output-flow | BundleBuilder accepts FeatureSpec but it must be passed explicitly. No auto-generation from training config. |
| **Calibrator transfer partially broken** | MEDIUM | training-output-flow | Calibrator IS stored on Trainer.calibrator and BundleBuilder CAN find it via duck-typing. However, the orchestrator also sets calibrator on the service result object (L993) which is LOST during ModelTrainingResult conversion (L912-920) since that dataclass has no calibrator field. Net effect: calibrator transfer works for direct Trainer access but fails in the orchestrator conversion path. |
| **Preprocessing graph hardcodes assumptions** | MEDIUM | training-output-flow | `BundleBuilder._create_preprocessing_graph()` hardcodes source_timeframe="1min", target_timeframe="5min", scaler_type="robust" instead of pulling from PipelineConfig. |
| **No auto-bundling in orchestrator** | MEDIUM | training-output-flow | UnifiedTrainingOrchestrator doesn't call BundleBuilder. Only MLFactory Phase 4 does. Direct `train_pipeline()` users get no bundles. |

### 2.2 Colab Integration Gaps

| Gap | Severity | Details |
|-----|----------|---------|
| **No inference demo cell** | HIGH | Notebook stops at "download zip." No cell loads a bundle and runs predictions. |
| **No inference-only download** | MEDIUM | Downloaded zip contains cache, checkpoints, raw data — not just bundles. |
| **BundlingSection not exposed** | MEDIUM | Cell 2 configures everything except bundling options. |
| **No Drive mount helper** | LOW | Notebook detects Drive if mounted but doesn't help mount it. |
| **No ONNX/TorchScript export** | LOW | Bundles require full src/ package to load. No standalone deployment. |

### 2.3 Universal (All Model Types) Inference Gaps

| Model Family | predict(X_preshaped) | predict_from_raw(df) | Gap |
|-------------|---------------------|---------------------|-----|
| Tabular (4: XGB, LGB, CB, MLP) | Works (2D) | Works | None — predict_from_raw() works end-to-end |
| RNN (2) | Works (needs 3D) | **Broken** | No SequenceAdapter in inference path |
| CNN (3) | Works (needs 3D) | **Broken** | No SequenceAdapter in inference path |
| Transformer (3) | Works (needs 3D/4D) | **Broken** | No MultiStreamAdapter; needs multi-TF data |
| Ensemble | Works | **Partial** | Base model predictions need adapter routing |

**Key finding:** 4 of 12 core models (XGBoost, LightGBM, CatBoost, MLP/N-BEATS) have complete raw-OHLCV-to-prediction inference via PreprocessingGraph.transform() + predict_from_raw(). The other 8 (6 needing 3D + 2 needing 4D) require manual tensor preparation.

---

## 3. Risk Register

### 3.1 Data Leakage Risks in Inference

| Risk | Severity | Location | Details |
|------|----------|----------|---------|
| Double scaling | MEDIUM | data-flow GAP 1 | If pipeline-scaled data is re-scaled by AdapterScaler during inference, feature distributions will be wrong. Need to enforce exactly one scaling step. |
| Feature column auto-detection drift | LOW-MEDIUM | data-flow GAP 2 | Heuristic exclusion lists differ between pipeline (`EXCLUDED_COLUMNS`) and adapters (inline list). Could select different feature sets at inference vs training. |
| Scaler type mismatch on load | LOW | model-packaging 5.1 | Bundle stores scaler as raw pickle, no validation that loaded scaler type matches model contract's `scaler_type`. |
| Label mapping not stored | LOW | model-packaging 5.1 | The -1,0,1 → 0,1,2 mapping is hardcoded. If it changes, old models break silently. |

### 3.2 Serialization Gaps Per Model Family

| Family | Serialization | Gap | Impact |
|--------|--------------|-----|--------|
| **Boosting** (3) | Native format (JSON/text/cbm) + pickle metadata | Feature names require manual `set_feature_names()` call; may be None | Feature importance shows f0,f1,... instead of names |
| **Neural** (9) | PyTorch checkpoint (.pt) | No architecture version tag; code changes → cryptic shape mismatch errors | Model loading fails silently between code versions |
| **Ensemble** | Meta-learner + base bundle paths (relative by default) | Ensemble orchestrator uses relative paths by default (./experiments/exp_001). EnsembleBundle saves paths as raw str(p) at L447 — whether absolute depends on input. Not hardcoded absolute. Type mismatch between EnsembleService result and what BundleBuilder expects. | Robustness issues |
| **All** | Pickle for scaler/calibrator | Security risk with untrusted files; not validated against contract | Incorrect scaling if wrong scaler loaded |

### 3.3 Colab-Specific Constraints

| Constraint | Risk | Mitigation Status |
|-----------|------|-------------------|
| Ephemeral filesystem | Bundle lost on disconnect | Drive save exists but optional and not prompted |
| 15GB VRAM (T4) | Walk-forward with all 12 models may OOM | No memory estimation/warning |
| pandas==2.2.2 pin | Limits available features/dependencies | Documented, handled in requirements-colab.txt |
| torch version mismatch | Colab may have older torch than required >=2.2.0 | Not validated at runtime |
| No persistent state | Multi-session training impossible without Drive | No guidance in notebook |

---

## 4. Key Questions for Planning

### Architecture Decisions

1. **Where should adapter integration live in the inference path?**
   - Option A: Inside `ModelBundle.predict_from_raw()` — bundle knows its model's data rank and invokes the right adapter
   - Option B: Inside `PreprocessingGraph.transform()` — graph outputs the correct tensor shape
   - Option C: New `UnifiedInferencePipeline` that chains PreprocessingGraph → Adapter → Model
   - Trade-off: A is simplest but couples bundle to adapters; C is cleanest but adds a new abstraction layer

2. **How to handle multi-timeframe data at inference time for 4D models?**
   - PatchTST/iTransformer need data from multiple timeframes (1min, 5min, 15min, 60min)
   - Bundle metadata stores `n_timeframes` and `requires_4d` but not the specific timeframe list or how to obtain the data
   - Must the caller provide all timeframe DataFrames, or should the inference system generate them from 1min data?

3. **Should we fix the double-scaling issue by removing one scaling stage or by adding a flag?**
   - Pipeline Stage 7.5 scales parquet files. AdapterScaler scales numpy arrays.
   - At inference, which scaler should be canonical? The pipeline scaler (saved as `feature_scaler.pkl`) or the adapter scaler?
   - Recommendation: Pipeline scaler should be canonical since it's already saved and validated. Adapter scaling should be `apply_scaling=False` when consuming pre-scaled data.

4. **Should special training modes (walk-forward, regime, meta-labeling) get specialized bundles?**
   - Currently only STANDARD mode bundles cleanly
   - Walk-forward produces multiple model versions per window
   - Regime-aware produces per-regime models
   - Meta-labeling produces primary + meta model pairs
   - All three are production-relevant but none have inference bundle support

### Trade-offs Identified

| Trade-off | Option A | Option B | Recommendation |
|-----------|----------|----------|----------------|
| Adapter in bundle vs separate layer | Simpler, self-contained bundle | Cleaner separation of concerns | Option A for MVP, Option C long-term |
| Store MTF data generation vs require caller | Self-contained but complex | Simple but requires caller knowledge | Store generation config in bundle |
| Fix extraction duck-typing vs define protocol | Quick fix (hardcode names) | Clean but larger refactor | Define protocol (TrainerProtocol) |
| Bundle requires src/ vs standalone export | Works now | Enables deployment to any environment | ONNX export as Phase 3+ goal |

---

## 5. Recommended Architecture Patterns

### 5.1 Patterns to Reuse (Already Exist)

| Pattern | Location | Why Reuse |
|---------|----------|-----------|
| **Contract-driven routing** | `MODEL_CONTRACTS` → `get_model_contract()` → `adapter_id` | 13 call sites already enforce this. Inference should use the same contracts to select adapters. |
| **Adapter registry** | `AdapterRegistry.get_for_model()` → TabularAdapter/SequenceAdapter/MultiStreamAdapter | Clean decorator-based registry. Inference path just needs to call it. |
| **BundleMetadata shape flags** | `requires_sequences`, `requires_4d`, `sequence_length`, `n_timeframes` | Already stored in bundle metadata. Inference path can use these to select the right adapter. |
| **PreprocessingGraph config capture** | `PreprocessingGraph.from_pipeline_config()` | Already captures feature engineering config. Needs to be extended, not replaced. |
| **FeatureManifest** | `BaseAdapter.from_manifest()` | Explicit feature column specification instead of auto-detection. Should be required, not optional. |
| **ModelBundle.package_bundle() / extract_bundle()** | tar.gz packaging with path traversal protection | Deployment packaging already works. |

### 5.2 What Needs to Be Built New

| Component | Purpose | Complexity |
|-----------|---------|------------|
| **Adapter integration in inference** | Chain PreprocessingGraph → Adapter routing → Model prediction | MEDIUM — the pieces exist, need orchestration |
| **TrainerProtocol** | Explicit interface (model, scaler, feature_columns, calibrator) replacing duck-typing | LOW — define Protocol class, update Trainer base class |
| **MTF data generation at inference time** | Generate multi-timeframe DataFrames from 1min raw data for 4D models | MEDIUM — MTFFeatureGenerator exists but isn't wired into inference |
| **Bundle-level scaling flag** | Record whether the bundle's scaler was the pipeline scaler or adapter scaler, prevent double-scaling | LOW — add field to BundleMetadata |
| **Inference demo cell for Colab** | Load bundle → predict on new data → display results | LOW — straightforward notebook cell |
| **Inference-only export** | Extract just bundles/ directory from output for deployment | LOW — filter zip contents |

### 5.3 Recommended Universal Inference Pipeline Pattern

```
                          ┌─────────────────────────────────┐
                          │   UniversalInferencePipeline     │
                          │   (new orchestration layer)      │
                          └─────────┬───────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
    ┌─────────▼──────────┐  ┌──────▼───────┐  ┌─────────▼──────────┐
    │ Raw OHLCV Input     │  │ Pre-computed │  │ Pre-shaped Input   │
    │ (end-to-end)        │  │ Features     │  │ (2D/3D/4D)         │
    └─────────┬──────────┘  └──────┬───────┘  └─────────┬──────────┘
              │                     │                     │
    ┌─────────▼──────────┐         │                     │
    │ PreprocessingGraph  │         │                     │
    │ .transform(raw_df)  │         │                     │
    │ → 2D features       │         │                     │
    └─────────┬──────────┘         │                     │
              │                     │                     │
              ▼                     ▼                     │
    ┌─────────────────────────────────────┐               │
    │ Adapter Routing (from BundleMetadata)│              │
    │ requires_sequences? → SequenceAdapter│              │
    │ requires_4d? → MultiStreamAdapter    │              │
    │ else → pass through (already 2D)     │              │
    └─────────────────┬───────────────────┘               │
                      │                                   │
                      ▼                                   ▼
            ┌─────────────────────────────────────────────┐
            │ ModelBundle.predict(shaped_input)            │
            │ → scale → model.predict → calibrate         │
            │ → PredictionResult                          │
            └─────────────────────────────────────────────┘
```

**Key design decisions in this pattern:**
1. PreprocessingGraph stays 2D-output (feature engineering is always tabular)
2. Adapter routing happens AFTER feature engineering, BEFORE prediction
3. Bundle metadata (`requires_sequences`, `requires_4d`, `sequence_length`) drives adapter selection
4. Pre-shaped inputs bypass both preprocessing and adapter steps
5. The scaler in the bundle is the ONLY scaler applied (no double-scaling)

---

## 6. Critical Path Items

### Must Be Built First (Dependencies)

```
1. TrainerProtocol (no dependencies)
   └── Standardize what trainers expose: model, scaler, feature_columns, calibrator
   └── Unblocks: reliable bundle extraction

2. Adapter integration in ModelBundle (depends on: existing adapter registry)
   └── Add adapter routing to predict_from_raw() using BundleMetadata shape flags
   └── Unblocks: universal raw→prediction for all 12 models

3. Double-scaling resolution (no dependencies)
   └── Add scaling_source flag to BundleMetadata
   └── Ensure exactly one scaling step in inference path
   └── Unblocks: correct feature values at inference time

4. Calibrator transfer fix (depends on: TrainerProtocol)
   └── Ensure calibrator flows training → result → bundle
   └── Unblocks: probability calibration in production
```

### Can Be Parallelized

```
A. Colab inference demo cell (depends on: adapter integration #2)
   └── Load bundle → predict on new data → display

B. MTF inference data generation (depends on: adapter integration #2)
   └── Generate multi-TF DataFrames from 1min data for 4D models

C. Inference-only export packaging (no dependencies)
   └── Add "download bundles only" to notebook

D. FeatureSpec auto-generation (no dependencies)
   └── Generate from PipelineConfig + training results

E. Ensemble path cleanup (no dependencies)
   └── Relative base bundle paths, type alignment between services

F. Dead code removal (no dependencies)
   └── validate_distribution(), unused time calls, no-op _apply_regime()

G. Duplicate enum consolidation (no dependencies)
   └── CVMethod and LabelingMethod → import from types.py only

H. Architecture version tag for neural models (no dependencies)
   └── Save arch_version in checkpoint, validate on load
```

### Suggested Phase 2 Execution Order

```
Phase 2a (Foundation — sequential):
  1. TrainerProtocol
  2. Adapter integration in inference path
  3. Double-scaling resolution

Phase 2b (Parallel work — after 2a):
  4. Calibrator transfer fix
  5. Colab inference demo cell
  6. MTF inference data generation
  7. FeatureSpec auto-generation

Phase 2c (Cleanup — parallel, any time):
  8. Inference-only export
  9. Ensemble path fixes
  10. Dead code removal
  11. Duplicate enum consolidation
  12. Neural architecture versioning
```

---

## Cross-Report Contradictions and Observations

### Contradiction: Model Count
- **CLAUDE.md** says "All 12 models are production-ready"
- **Architecture integrity audit** found **23 models** across 6 families in MODEL_CONTRACTS
- **Resolution:** The 12 in CLAUDE.md are the core prediction models. The 23 includes classical (3), ensemble (3), and meta-learner (4) variants. Both are correct depending on scope, but CLAUDE.md should be updated.

### Contradiction: TFT Classification
- **MODEL_CONTRACTS** classifies TFT as `model_family="neural"` with `input_rank=SEQUENCE_3D`
- **CLAUDE.md** lists TFT under "Transformer" category
- **Impact:** Minor — TFT uses 3D input (not 4D like PatchTST/iTransformer), so "neural" classification is functionally correct for adapter routing

### Overlapping Concern: PredictionResult Duplication
- **Inference-pipeline report** found two PredictionResult classes: `src.core.interfaces` and `src.models.base`
- **Needs investigation:** Are these the same class re-exported, or genuinely different? If different, inference path may return wrong type.

### Overlapping Concern: InferencePipeline vs InferenceOrchestrator
- Both provide load-models-and-predict functionality
- Pipeline is lower-level (used by server, batch). Orchestrator is higher-level (supports raw input).
- **Risk:** Maintaining two similar abstractions increases surface area for bugs
- **Recommendation:** Decide on one entry point for Phase 2 universal inference

### Consistent Finding Across All Reports
All 6 reports independently identified the **adapter integration gap** as the single most impactful missing piece. The system has all the components (PreprocessingGraph, Adapters, ModelBundle) but they aren't wired together in the inference path. This is the #1 priority for Phase 2.

---

## VERIFICATION NOTES

The following corrections were applied based on automated verification on 2026-02-15:

1. **Adapter gap model count**: Corrected from "3 boosting models work / 9 broken" to "4 tabular models (XGBoost, LightGBM, CatBoost, MLP) work end-to-end; 8 of 12 core models broken (6 needing 3D + 2 needing 4D)". N-BEATS/MLP uses TabularAdapter per MODEL_ADAPTER_MAP.
2. **Calibrator transfer**: Nuanced from "broken" to "partially broken" — Calibrator IS on Trainer.calibrator (BundleBuilder can find it), but LOST during orchestrator's ModelTrainingResult conversion (L912-920).
3. **Ensemble paths**: Corrected from "absolute (not portable)" to "relative by default". EnsembleBundle saves paths as raw str(p) at L447; whether absolute depends on input.
4. **End-to-end preprocessing**: Corrected from "no end-to-end replay exists" to "PreprocessingGraph.transform() + predict_from_raw() provide end-to-end for tabular models; gap is specifically for 3D/4D models."
5. **MTFMode naming**: Clarified that data_contract.py defines `ModelMTFMode` (not `MTFMode`); separate `MTFMode` exists in config/data.py.
6. **Pickle count**: No "17 call sites" claim found in phase1-audit files; no correction needed here.
