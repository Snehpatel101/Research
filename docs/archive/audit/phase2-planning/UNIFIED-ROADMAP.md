# Phase 2 Unified Implementation Roadmap

**Date:** 2026-02-15
**Sources:** 6 Phase 2 planning documents + Phase 1 consolidated findings
**Purpose:** Primary reference for Phase 3 implementation agents

---

## 1. Architecture Overview

### How the 6 Plans Fit Together

```
                    PLAN 5: TrainerProtocol
                    (Foundation layer — standardizes
                     what trainers expose for extraction)
                              │
                              ▼
              ┌───────────────────────────────────┐
              │       PLAN 2: Adapter/Scaling      │
              │  (Per-model routing, double-scaling │
              │   fix, MTF generation, feature      │
              │   column alignment)                 │
              └───────────┬───────────────────────┘
                          │
                          ▼
              ┌───────────────────────────────────┐
              │     PLAN 1: UniversalInference     │
              │     Pipeline (orchestration layer   │
              │     that wires adapters into        │
              │     inference, merges Pipeline +    │
              │     Orchestrator)                   │
              └───────────┬───────────────────────┘
                          │
              ┌───────────┼───────────────────┐
              │           │                   │
              ▼           ▼                   ▼
   PLAN 4: Ensemble    PLAN 3: Colab      PLAN 6: Cleanup
   & Special Modes     Integration         & Risk Mitigation
   (WalkForward,       (Notebook cells,    (Dead code, dupes,
    Regime, Meta-       bundling config,    pickle security,
    Labeling bundles)   Drive persistence)  label mapping)
```

### Dependency Summary

- **Plan 5 (TrainerProtocol)** is pure foundation — no dependencies, unblocks reliable bundle extraction
- **Plan 2 (Adapter/Scaling)** depends on nothing but enables the core inference fix
- **Plan 1 (UniversalInferencePipeline)** consumes Plan 2's adapter routing and Plan 5's protocol
- **Plan 4 (Ensemble/Special Modes)** depends on Plan 1 for `predict_from_raw()` on base models
- **Plan 3 (Colab)** depends on Plan 1 for full 12-model inference demos (tabular demos work now)
- **Plan 6 (Cleanup)** is fully parallel — no dependencies on Plans 1-5

### Conflict Resolutions

| Conflict | Plans | Resolution |
|----------|-------|------------|
| Where adapter integration lives | Plan 1 (new class) vs Plan 2 (inside ModelBundle) | **Both.** Plan 2's `_build_3d_input()`/`_build_4d_input()` go on ModelBundle for backward compat. Plan 1's UniversalInferencePipeline uses them OR its own `_adapt_input()` via contracts. The UIP is the recommended path; ModelBundle methods are the fallback. |
| Who owns scaling | Plan 1 (ScalingSource enum) vs Plan 2 (pipeline scaler canonical) | **Aligned.** Both agree pipeline scaler is canonical. Plan 1's `ScalingSource` enum is the enforcement mechanism. Plan 2's `skip_scaling=True` in `predict_from_raw()` is the implementation. |
| Adapter transform_inference vs inline numpy | Plan 2 (both options) | **Option B (inline) for MVP, Option A (transform_inference) for Phase 3C polish.** Keeps adapter code untouched initially. |

---

## 2. Dependency Graph

### ASCII Dependency Diagram

```
                         ┌─────────────────────────┐
                         │ 3A-1: TrainerProtocol    │
                         │ (src/core/protocols.py)  │
                         └──────────┬──────────────┘
                                    │
                    ┌───────────────┼──────────────────┐
                    │               │                  │
                    ▼               ▼                  ▼
          ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐
          │ 3A-2: Trainer│  │ 3A-3: Bundle │  │ 3A-4: BundleBuilder│
          │ properties   │  │ Metadata     │  │ protocol-aware     │
          └──────┬──────┘  └──────┬───────┘  │ extraction         │
                 │                │          └──────────┬─────────┘
                 │                │                     │
                 └────────┬───────┴─────────────────────┘
                          │
                          ▼
              ┌───────────────────────────────┐
              │ 3B-1: Adapter routing in       │
              │ ModelBundle (predict_from_raw)  │
              │ + double-scaling fix            │
              └──────────┬────────────────────┘
                         │
              ┌──────────┼──────────────────────────────┐
              │          │                              │
              ▼          ▼                              ▼
   ┌──────────────┐  ┌────────────────────┐  ┌─────────────────────┐
   │ 3B-2: UIP    │  │ 3B-3: EnsembleBundle│ │ 3B-4: MTF inference │
   │ class build  │  │ predict_from_raw +  │ │ data generation     │
   │              │  │ relative paths      │ │                     │
   └──────┬───────┘  └────────┬───────────┘ └──────────┬──────────┘
          │                   │                        │
          └───────┬───────────┴────────────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ 3C: Integration                                              │
   │  ├── 3C-1: Colab inference demo (Cells 8, 9)                │
   │  ├── 3C-2: server.py + batch.py → use UIP                   │
   │  ├── 3C-3: Special mode bundles (WalkForward, Regime, Meta) │
   │  └── 3C-4: Colab polish (Drive, warnings, bundling config)  │
   └──────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────┐
   │ 3D: Cleanup (PARALLEL with everything above)                 │
   │  ├── 3D-1: Remove _apply_regime() no-op                     │
   │  ├── 3D-2: Consolidate CVMethod, LabelingMethod enums       │
   │  ├── 3D-3: safe_pickle_load() utility                       │
   │  ├── 3D-4: Neural architecture versioning                   │
   │  └── 3D-5: Deprecate InferencePipeline + InferenceOrchestrator │
   └──────────────────────────────────────────────────────────────┘
```

### Build Order (Critical Path)

```
TrainerProtocol → Trainer props → BundleMetadata → BundleBuilder update
    → Adapter routing in ModelBundle → UniversalInferencePipeline
    → server/batch/notebook integration → deprecate old classes
```

---

## 3. Implementation Phases (Ordered Work Packages)

### Phase 3A: Foundation (Must Be First)

**Goal:** Establish the TrainerProtocol, update BundleMetadata, make BundleBuilder protocol-aware.

| Task | Files | Complexity | Est. Lines |
|------|-------|------------|------------|
| **3A-1: Create TrainerProtocol** | NEW: `src/core/protocols.py` | LOW | ~40 |
| **3A-2: Add Trainer properties** | `src/models/training/trainer.py` | LOW | ~30 |
| **3A-3: Extend BundleMetadata** | `src/inference/bundle.py` | LOW | ~50 |
| **3A-4: Protocol-aware BundleBuilder** | `src/inference/builder.py` | MEDIUM | ~80 |
| **3A-5: Calibrator transfer fix** | `src/models/training/unified_orchestrator.py`, `src/inference/builder.py` | LOW | ~20 |
| **3A-6: FeatureSpec auto-generation** | `src/inference/builder.py` | LOW | ~30 |

**Details:**

**3A-1: TrainerProtocol** (`src/core/protocols.py` — NEW FILE)
```python
@runtime_checkable
class TrainerProtocol(Protocol):
    @property
    def model(self) -> BaseModel: ...
    @property
    def scaler(self) -> Any | None: ...
    @property
    def feature_columns(self) -> list[str]: ...
    @property
    def calibrator(self) -> Any | None: ...
    @property
    def training_config(self) -> dict[str, Any]: ...
    @property
    def model_key(self) -> str: ...
```

**3A-2: Trainer properties** (`src/models/training/trainer.py`)
- Add `self.scaler = None` in `__init__` (after line ~107)
- Add `@property feature_columns` exposing `self._feature_set_columns`
- Add `@property training_config` returning `self.config.to_dict()`
- Add `@property model_key` returning `f"{self.config.model_name}_h{self.config.horizon}"`
- In `run()` (~line 720): capture scaler from container
- In `run_prepared()` (~line 948): capture scaler from PreparedData

**3A-3: BundleMetadata extensions** (`src/inference/bundle.py`)
- Add fields: `scaling_source: str = "unknown"`, `arch_version: str | None = None`, `label_mapping: dict | None = None`, `feature_names: list[str] = field(default_factory=list)`, `scaler_type: str = "unknown"`, `training_run_id: str | None = None`
- Update `from_dict()` with `.get()` safe defaults
- Update `to_dict()` to serialize new fields
- Bump version from `1.2.0` to `1.3.0`

**3A-4: BundleBuilder protocol-aware extraction** (`src/inference/builder.py`)
- `_extract_model()`: Check `isinstance(trainer, TrainerProtocol)` first, legacy fallback with warning
- `_extract_scaler()`: Same pattern
- `_extract_feature_columns()`: Same pattern
- `_extract_calibrator()`: Same pattern + check `model_result.calibrator`
- `_create_preprocessing_graph()`: Replace hardcoded `source_timeframe="1min"`, `target_timeframe="5min"`, `scaler_type="robust"` with values from `PipelineConfig`

**3A-5: Calibrator transfer** (`src/models/training/unified_orchestrator.py`)
- Add `calibrator: Any | None = None` field to `ModelTrainingResult` dataclass
- After training loop: `result.calibrator = trainer.calibrator`

**3A-6: FeatureSpec auto-generation** (`src/inference/builder.py`)
- New method `_auto_generate_feature_spec(model_result, trainer) -> FeatureSpec | None`
- Called in `build_from_training_result()` when `feature_spec is None`

**Validation:**
```bash
python -c "from src.core.protocols import TrainerProtocol; print('OK')"
python -c "from src.inference.bundle import BundleMetadata; m = BundleMetadata.from_dict({'version':'1.2.0','model_name':'x'}); print(m.scaling_source)"
```

---

### Phase 3B: Core Inference (Depends on 3A)

**Goal:** Wire adapters into inference path, build UniversalInferencePipeline, fix EnsembleBundle.

| Task | Files | Complexity | Est. Lines |
|------|-------|------------|------------|
| **3B-1: Adapter routing in ModelBundle** | `src/inference/bundle.py` | MEDIUM | ~100 |
| **3B-2: UniversalInferencePipeline** | NEW: `src/inference/universal_pipeline.py`, NEW: `src/inference/errors.py` | MEDIUM | ~500 |
| **3B-3: EnsembleBundle fixes** | `src/inference/ensemble_bundle.py` | MEDIUM | ~60 |
| **3B-4: MTF inference data generation** | `src/inference/bundle.py` | MEDIUM | ~50 |
| **3B-5: Type alignment bridge** | `src/models/training/services/ensemble_service.py`, `src/models/training/unified_orchestrator.py` | MEDIUM | ~40 |

**Details:**

**3B-1: Adapter routing in ModelBundle** (`src/inference/bundle.py`)
- Add `_apply_adapter(features_2d, raw_df) -> np.ndarray | pd.DataFrame`
- Add `_build_3d_input(features_2d) -> np.ndarray` — sliding window via numpy stride_tricks
- Add `_build_4d_input(features_2d, raw_df) -> np.ndarray` — MTF generation + multi-stream windowing
- Update `predict_from_raw()` to chain: `preprocess() → _apply_adapter() → predict()`
- Pass `skip_scaling=True` to `preprocessing_graph.transform()` in `predict_from_raw()`
- Store `mtf_timeframes` in `BundleMetadata.extra` for 4D models

Routing logic:
```python
def _apply_adapter(self, features_2d, raw_df):
    if self.metadata.requires_4d:
        return self._build_4d_input(features_2d, raw_df)
    elif self.metadata.requires_sequences:
        return self._build_3d_input(features_2d)
    else:
        return features_2d
```

3D windowing (inline numpy, ~10 lines):
```python
def _build_3d_input(self, features_2d):
    seq_len = self.metadata.sequence_length
    features = features_2d[self.feature_columns].values.astype(np.float32)
    if len(features) < seq_len:
        raise ValueError(f"Need >= {seq_len} rows, got {len(features)}")
    windows = np.lib.stride_tricks.sliding_window_view(features, seq_len, axis=0)
    return windows.transpose(0, 2, 1).copy()  # (n_seq, seq_len, n_feat)
```

**3B-2: UniversalInferencePipeline** (`src/inference/universal_pipeline.py` — NEW FILE)

Class with:
- Constructor: `__init__(bundles, ensemble_bundle, preprocessing_graph, config, scaling_source)`
- Class methods: `from_bundle()`, `from_bundles()`, `from_experiment()`, `from_training_result()`
- Core methods: `predict()`, `predict_from_raw()`, `predict_all()`, `predict_ensemble()`, `predict_batch()`, `predict_with_uncertainty()`
- Internal: `_get_adapter_for_model()`, `_adapt_input()`, `_predict_single()`, `_apply_scaling()`
- `ScalingSource` enum: `BUNDLE`, `PREPROCESSING`, `NONE`

Key design invariant: Pipeline calls `bundle.model.predict()` directly (NOT `bundle.predict()`) to control scaling timing.

Error classes in `src/inference/errors.py` (~40 lines):
- `InferenceShapeMismatchError(ValueError)`

**3B-3: EnsembleBundle fixes** (`src/inference/ensemble_bundle.py`)
- Fix `save()` (~line 443): Store relative paths from ensemble bundle root
- Fix `load()` (~line 539): Resolve relative paths with absolute fallback
- Add `predict_from_raw(raw_df, calibrate=True) -> PredictionResult`:
  - Load base bundles
  - Call each `base_bundle.predict_from_raw(raw_df, calibrate=False)`
  - Stack probabilities via `_stack_predictions()`
  - Feed meta-learner

**3B-4: MTF inference data generation** (`src/inference/bundle.py`)
- Add `_generate_mtf_dataframes(raw_1min_df, timeframes) -> dict[str, DataFrame]`
- Uses `resample_ohlcv()` from `src/data/pipeline/stages/clean/utils.py`
- Called by `_build_4d_input()` when raw 1min data is available

**3B-5: Type alignment bridge**
- `src/models/training/services/ensemble_service.py`: Add `to_ensemble_result(service_result, config) -> EnsembleResult`
- `src/models/training/unified_orchestrator.py`: Change `_build_ensemble()` to store `EnsembleResult` (not `ModelTrainingResult`) in `TrainingRunResult.ensemble_result`

**Validation:**
```bash
# Verify adapter routing works for all model families
python -c "
from src.inference.bundle import ModelBundle
# Test that _apply_adapter exists and routes correctly
print('ModelBundle has _apply_adapter:', hasattr(ModelBundle, '_apply_adapter'))
"
# Verify UIP imports
python -c "from src.inference.universal_pipeline import UniversalInferencePipeline; print('OK')"
```

---

### Phase 3C: Integration (Depends on 3B)

**Goal:** Wire everything into consumers: notebook, server, batch, special modes.

| Task | Files | Complexity | Est. Lines |
|------|-------|------------|------------|
| **3C-1: Colab inference demo** | `notebooks/ml_factory_colab.ipynb` | MEDIUM | ~200 |
| **3C-2: server.py + batch.py migration** | `src/inference/server.py`, `src/inference/batch.py` | LOW | ~40 |
| **3C-3: Special mode bundles** | NEW: 3 files in `src/inference/` | MEDIUM | ~400 |
| **3C-4: Colab polish** | `notebooks/ml_factory_colab.ipynb` | LOW | ~100 |
| **3C-5: Update __init__.py exports** | `src/inference/__init__.py` | LOW | ~15 |

**Details:**

**3C-1: Colab inference demo** (notebook)
- **Cell 8: Inference Demo** — Load best bundle, run `predict_from_raw()` on sample data, display prediction distribution (SHORT/HOLD/LONG counts), mean confidence
  - For tabular: full raw→predict demo
  - For neural/transformer: full demo once adapter integration lands (Phase 3B)
  - Ensemble: metadata display + optional base-model predictions
- **Cell 9: Inference-Only Export** — Zip only `bundles/` directory, show size comparison vs full zip

**3C-2: server.py + batch.py migration**
```python
# server.py: Replace InferencePipeline with UniversalInferencePipeline
# batch.py: Replace InferenceOrchestrator with UniversalInferencePipeline
```

**3C-3: Special mode bundles** (3 new files)
- `src/inference/walk_forward_bundle.py` — `WalkForwardBundle`: thin wrapper around latest-window ModelBundle with window metadata. `predict_from_raw()` delegates to `self.latest_bundle.predict_from_raw()`. ~80 lines.
- `src/inference/regime_bundle.py` — `RegimeBundle`: stores per-regime ModelBundles + serializable RegimeDetector. `predict_from_raw()` detects regime → routes to correct model. ~120 lines.
- `src/inference/meta_labeling_bundle.py` — `MetaLabelingBundle`: primary ModelBundle + sklearn meta-model + threshold. `predict_from_raw()` → primary prediction → meta confidence → position sizing. ~100 lines.
- `src/inference/regime_detector.py` — Serializable `RegimeDetector` class extracted from RegimeAwareTrainer logic. ~60 lines.
- `src/inference/builder.py` — Add `build_walk_forward_bundle()`, `build_regime_bundle()`, `build_meta_labeling_bundle()` methods to BundleBuilder.

Common interface (Protocol):
```python
class InferenceBundle(Protocol):
    def predict(self, X, calibrate=True) -> PredictionResult: ...
    def predict_from_raw(self, raw_df) -> PredictionResult: ...
    def save(self, path, overwrite=False) -> Path: ...
    @classmethod
    def load(cls, path) -> InferenceBundle: ...
```

**3C-4: Colab polish** (notebook)
- **Cell 0 (Markdown)**: Add inference section to Quick Start
- **Cell 1 (Setup)**: Add torch version check (>=2.2.0)
- **Cell 2 (Config)**: Add bundling config toggles: `CREATE_BUNDLE`, `BUNDLE_FORMAT`, `INCLUDE_OOF`, `INCLUDE_FEATURE_IMPORTANCE`
- **Cell 3 (Validation)**: Add memory/VRAM warnings for large walk-forward + many models
- **Cell 5 (MLFactory Run)**: Add `BundlingSection` to `ExperimentConfig`; ephemeral FS warning
- **Cell 6 (Results)**: Add bundle summary section
- **Cell 7b (NEW)**: Drive mount + save bundles to Google Drive

**3C-5: Update exports** (`src/inference/__init__.py`)
- Export `UniversalInferencePipeline`
- Add deprecation warnings for `InferencePipeline`, `InferenceOrchestrator`

---

### Phase 3D: Cleanup (Parallel with Everything Above)

**Goal:** Dead code, duplicates, security, versioning. No dependencies on 3A-3C.

| Task | Files | Complexity | Est. Lines Changed |
|------|-------|------------|-------------------|
| **3D-1: Remove _apply_regime() no-op** | `src/inference/preprocessing_graph.py` | LOW | -10 |
| **3D-2: Consolidate CVMethod enum** | `src/config/cv.py` | LOW | ~5 |
| **3D-3: Consolidate LabelingMethod enum** | `src/config/data.py` | LOW | ~5 |
| **3D-4: safe_pickle_load() utility** | NEW: `src/core/utils/safe_pickle.py`, 16 call sites | MEDIUM | ~60 new + 16 modifications |
| **3D-5: Neural arch versioning** | `src/models/neural/base_rnn.py` | LOW | ~20 |
| **3D-6: Feature names in BundleBuilder** | `src/inference/builder.py` | LOW | ~5 |
| **3D-7: Deprecate old inference classes** | `src/inference/pipeline.py`, `src/inference/orchestrator.py` | LOW | ~10 |

**Details:**

**3D-1:** Remove `_apply_regime()` method at line 702 and its call at line 498 of `preprocessing_graph.py`.

**3D-2:** In `src/config/cv.py`, remove `class CVMethod` definition, replace with `from src.core.types import CVMethod`.

**3D-3:** In `src/config/data.py`, remove `class LabelingMethod` definition, replace with `from src.core.types import LabelingMethod`.

**3D-4:** Create `src/core/utils/safe_pickle.py`:
```python
def safe_pickle_load(path: Path, expected_type: type | None = None) -> Any:
    """Load pickle with path validation and optional type checking."""
    path = Path(path).resolve()
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if expected_type and not isinstance(obj, expected_type):
        logger.warning(f"Pickle at {path}: expected {expected_type}, got {type(obj)}")
    return obj
```
Update all 17 `pickle.load()` call sites.

**3D-5:** Add `ARCH_VERSION = "1.0"` to `BaseRNNModel`. Save in checkpoint dict. Validate on load (warning only, not error).

**3D-6:** In `BundleBuilder`, after extracting model, call `model.set_feature_names(feature_columns)` if the model supports it.

**3D-7:** Add deprecation warnings to `InferencePipeline.__init__()` and `InferenceOrchestrator.__init__()` pointing to `UniversalInferencePipeline`.

---

## 4. File Change Manifest

### New Files

| File | Plan Source | Purpose | Key Classes/Functions |
|------|------------|---------|----------------------|
| `src/core/protocols.py` | Plan 5 | TrainerProtocol definition | `TrainerProtocol` (Protocol class) |
| `src/inference/universal_pipeline.py` | Plan 1 | Main inference orchestrator | `UniversalInferencePipeline`, `ScalingSource` |
| `src/inference/errors.py` | Plan 1 | Inference-specific errors | `InferenceShapeMismatchError` |
| `src/inference/walk_forward_bundle.py` | Plan 4 | Walk-forward inference | `WalkForwardBundle` |
| `src/inference/regime_bundle.py` | Plan 4 | Regime-aware inference | `RegimeBundle` |
| `src/inference/regime_detector.py` | Plan 4 | Serializable regime detection | `RegimeDetector` |
| `src/inference/meta_labeling_bundle.py` | Plan 4 | Meta-labeling inference | `MetaLabelingBundle`, `MetaLabelingPrediction` |
| `src/core/utils/safe_pickle.py` | Plan 6 | Pickle loading with validation | `safe_pickle_load()` |

### Modified Files

| File | Plan Source | Changes |
|------|------------|---------|
| **src/models/training/trainer.py** | Plan 5 | Add `self.scaler = None` in `__init__`; add `feature_columns`, `training_config`, `model_key` properties; capture scaler in `run()` and `run_prepared()` |
| **src/inference/bundle.py** | Plans 2, 3, 5 | Add 6 new BundleMetadata fields (`scaling_source`, `arch_version`, `label_mapping`, `feature_names`, `scaler_type`, `training_run_id`); update `from_dict()`/`to_dict()`; add `_apply_adapter()`, `_build_3d_input()`, `_build_4d_input()`, `_generate_mtf_dataframes()`; fix `predict_from_raw()` to pass `skip_scaling=True` |
| **src/inference/builder.py** | Plans 5, 4, 2 | Protocol-aware `_extract_*()` methods; `_auto_generate_feature_spec()`; `build_walk_forward_bundle()`, `build_regime_bundle()`, `build_meta_labeling_bundle()`; fix hardcoded preprocessing graph config; call `set_feature_names()` after extraction |
| **src/inference/ensemble_bundle.py** | Plan 4 | Fix relative paths in `save()`/`load()`; add `predict_from_raw()` method |
| **src/inference/__init__.py** | Plans 1, 4 | Export `UniversalInferencePipeline`, `WalkForwardBundle`, `RegimeBundle`, `MetaLabelingBundle`; deprecation aliases |
| **src/inference/pipeline.py** | Plan 1 | Add deprecation warning in `__init__()` |
| **src/inference/orchestrator.py** | Plan 1 | Add deprecation warning in `__init__()` |
| **src/inference/server.py** | Plan 1 | Swap `InferencePipeline` → `UniversalInferencePipeline` |
| **src/inference/batch.py** | Plan 1 | Swap `InferenceOrchestrator` → `UniversalInferencePipeline` |
| **src/inference/preprocessing_graph.py** | Plan 6 | Remove `_apply_regime()` no-op method and call site |
| **src/models/training/unified_orchestrator.py** | Plans 4, 5 | Add `calibrator` field to `ModelTrainingResult`; change `_build_ensemble()` to return `EnsembleResult`; store `EnsembleResult` in `TrainingRunResult.ensemble_result` |
| **src/models/training/services/ensemble_service.py** | Plan 4 | Add `to_ensemble_result()` conversion method |
| **src/models/neural/base_rnn.py** | Plans 5, 6 | Add `ARCH_VERSION = "1.0"` class constant; save/validate arch_version in checkpoint save/load |
| **src/config/cv.py** | Plan 6 | Remove duplicate `CVMethod` class; import from `src.core.types` |
| **src/config/data.py** | Plan 6 | Remove duplicate `LabelingMethod` class; import from `src.core.types` |
| **src/data/adapters/preparation.py** | Plan 2 | Clarify `apply_scaling` default behavior for pre-scaled pipeline data |
| **notebooks/ml_factory_colab.ipynb** | Plan 3 | Modify Cells 0, 1, 2, 3, 5, 6; add Cells 7b, 8, 9 |

### Files NOT Modified

| File | Reason |
|------|--------|
| `src/core/contracts/model_contract.py` | Already has all routing info needed |
| `src/data/adapters/registry.py` | Works correctly, inference uses it via model_name |
| `src/data/adapters/scaling.py` | Not used in canonical inference path |
| `src/data/pipeline/stages/scaling/run.py` | Pipeline scaling stays as-is |
| `src/models/ensemble/orchestrator.py` | `EnsembleResult` is already correct |

### 17 Pickle Call Sites for safe_pickle_load (3D-4)

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

---

## 5. Risk Mitigations

### From Plan 6 (Cleanup/Risk)

| Risk | Severity | Mitigation | When |
|------|----------|------------|------|
| **Pickle security** (16 call sites with no validation) | MEDIUM | Create `safe_pickle_load()` with path validation + type checking | Phase 3D (parallel) |
| **Double scaling** (pipeline scaler + adapter scaler) | MEDIUM | `ScalingSource` enum in UIP; `skip_scaling=True` in `predict_from_raw()` | Phase 3B-1 |
| **Feature column drift** (auto-detection differs between training/inference) | LOW-MEDIUM | Always use `bundle.feature_columns`; never auto-detect at inference | Phase 3B-1 |
| **Label mapping not stored** | LOW | Add `label_mapping` to BundleMetadata | Phase 3A-3 |
| **Neural arch version mismatch** | LOW | `ARCH_VERSION` in checkpoint; warning on load | Phase 3D-5 |
| **Scaler type mismatch on load** | LOW | Add `scaler_type` to BundleMetadata | Phase 3A-3 |

### From Each Plan

| Plan | Risk | Mitigation |
|------|------|------------|
| Plan 1 (UIP) | Calling `bundle.model.predict()` directly bypasses bundle's internal scaling | UIP owns scaling via `_apply_scaling()` with `ScalingSource` — test all 3 modes |
| Plan 2 (Adapter) | 4D models need MTF data that may not be available | Store `mtf_timeframes` in metadata; generate from 1min data at inference; clear error if insufficient data |
| Plan 4 (Ensemble) | Relative paths in `base_bundles.json` (default) may not resolve if bundle directory is restructured | Validate relative path resolution with absolute fallback |
| Plan 4 (Special modes) | Walk-forward discards per-window models | MVP: bundle only latest window; future: all windows as mini-ensemble |
| Plan 5 (Protocol) | Existing trainers may not satisfy protocol initially | Legacy duck-typing fallback with deprecation warning |

### Backward Compatibility Guarantees

1. **Old bundles load fine** — New BundleMetadata fields have safe defaults (`"unknown"`, `None`, `[]`)
2. **Old prediction paths work** — `ModelBundle.predict(X_preshaped)` is unchanged
3. **Old trainer extraction works** — BundleBuilder falls back to duck-typing if protocol not satisfied
4. **Old neural checkpoints load** — Missing `arch_version` defaults to `"0.0"` with warning
5. **InferencePipeline/InferenceOrchestrator still work** — Deprecated with warnings, removed in later phase

---

## 6. Colab Deliverables

### Notebook Changes (Ordered)

| Priority | Cell | Type | Description |
|----------|------|------|-------------|
| 1 | Cell 2 (Config) | MODIFY | Add 4 bundling toggles: `CREATE_BUNDLE`, `BUNDLE_FORMAT`, `INCLUDE_OOF`, `INCLUDE_FEATURE_IMPORTANCE` |
| 2 | Cell 5 (Run) | MODIFY | Import `BundlingSection`; add to `ExperimentConfig` constructor; add ephemeral FS warning |
| 3 | Cell 6 (Results) | MODIFY | Add bundle summary section (model count, families, total size) |
| 4 | Cell 7b | **NEW** | Google Drive mount + save bundles for persistence (~30 lines) |
| 5 | Cell 9 | **NEW** | Inference-only export — zip only `bundles/` dir, show size comparison (~25 lines) |
| 6 | Cell 8 | **NEW** | Inference demo — load best bundle, run `predict_from_raw()`, display results (~50 lines) |
| 7 | Cell 0 (Markdown) | MODIFY | Add inference section to Quick Start guide |
| 8 | Cell 1 (Setup) | MODIFY | Add torch version check (>=2.2.0, 6 lines) |
| 9 | Cell 3 (Validation) | MODIFY | Add memory/VRAM warnings for large experiments |

### Dependencies

- Cells 2, 5, 6, 7b, 9: No code dependencies — can be implemented immediately
- Cell 8 (inference demo): Tabular models work now; full 12-model demo requires Phase 3B adapter integration
- Cells 0, 1, 3: No dependencies

### Testing Approach

1. **Cell 8 (Inference Demo)**: Test with a pre-trained boosting bundle first. Verify prediction distribution and confidence output. After Phase 3B, test with LSTM and PatchTST bundles.
2. **Cell 9 (Export)**: Verify zip contains only `bundles/` contents; compare file count and size vs full zip.
3. **Cell 7b (Drive)**: Test in actual Colab environment with Drive mount.
4. **Cell 2/5 (Bundling Config)**: Verify `BundlingSection` values flow through to `MLFactory.run()`.

---

## 7. Validation Checklist

### Phase 3A Validation

```bash
# TrainerProtocol exists and is importable
python -c "from src.core.protocols import TrainerProtocol; print('OK')"

# Trainer satisfies protocol (structural subtyping)
python -c "
from src.core.protocols import TrainerProtocol
from src.models.training.trainer import Trainer
from src.models.config import TrainerConfig
config = TrainerConfig(model_name='xgboost', horizon=20)
t = Trainer(config)
print(f'Satisfies protocol: {isinstance(t, TrainerProtocol)}')
"

# BundleMetadata backward compat
python -c "
from src.inference.bundle import BundleMetadata
old = BundleMetadata.from_dict({'version':'1.2.0','model_name':'x','model_family':'boosting'})
print(f'scaling_source={old.scaling_source}, arch_version={old.arch_version}')
# Should print: scaling_source=unknown, arch_version=None
"

# BundleBuilder legacy fallback still works
python -c "
from src.inference.builder import BundleBuilder
class Legacy:
    model = 'dummy'
    _scaler = None
b = BundleBuilder.__new__(BundleBuilder)
print(f'Legacy extract works: {b._extract_model(Legacy()) == \"dummy\"}')"

# Single definitions
grep -r "class TrainerProtocol" src/ --include="*.py" | wc -l  # Should be 1
```

### Phase 3B Validation

```bash
# UIP importable
python -c "from src.inference.universal_pipeline import UniversalInferencePipeline; print('OK')"
python -c "from src.inference.errors import InferenceShapeMismatchError; print('OK')"

# Adapter routing on ModelBundle
python -c "
from src.inference.bundle import ModelBundle
print('_apply_adapter' in dir(ModelBundle))
print('_build_3d_input' in dir(ModelBundle))
print('_build_4d_input' in dir(ModelBundle))
"

# EnsembleBundle has predict_from_raw
python -c "
from src.inference.ensemble_bundle import EnsembleBundle
print('predict_from_raw' in dir(EnsembleBundle))
"
```

### Phase 3C Validation

```bash
# Special bundles importable
python -c "from src.inference.walk_forward_bundle import WalkForwardBundle; print('OK')"
python -c "from src.inference.regime_bundle import RegimeBundle; print('OK')"
python -c "from src.inference.meta_labeling_bundle import MetaLabelingBundle; print('OK')"

# __init__.py exports
python -c "from src.inference import UniversalInferencePipeline; print('OK')"
```

### Phase 3D Validation

```bash
# No-op removed
grep -n "_apply_regime" src/inference/preprocessing_graph.py | wc -l  # Should be 0

# Enum consolidation
grep -r "class CVMethod" src/ --include="*.py" | wc -l  # Should be 1
grep -r "class LabelingMethod" src/ --include="*.py" | wc -l  # Should be 1

# safe_pickle_load exists
python -c "from src.core.utils.safe_pickle import safe_pickle_load; print('OK')"

# Neural arch versioning
python -c "from src.models.neural.base_rnn import BaseRNNModel; print(f'ARCH_VERSION={BaseRNNModel.ARCH_VERSION}')"

# No raw pickle.load remaining (should be 0 after migration)
grep -r "pickle\.load(" src/ --include="*.py" | grep -v "safe_pickle" | wc -l
```

### Integration Tests Needed

| Test | What It Verifies | Depends On |
|------|-----------------|------------|
| Train XGBoost → bundle → `predict_from_raw(sample_data)` → PredictionResult | Full tabular roundtrip | Phase 3A |
| Train LSTM → bundle → `predict_from_raw(sample_data)` → PredictionResult with 3D adapter | Full sequence roundtrip | Phase 3B |
| Train PatchTST → bundle → `predict_from_raw(raw_1min)` → PredictionResult with 4D MTF | Full multi-stream roundtrip | Phase 3B |
| `UniversalInferencePipeline.from_bundle(path).predict(X_2d)` for boosting | UIP Mode 2 | Phase 3B |
| `UniversalInferencePipeline.from_bundle(path).predict(X_3d)` for LSTM | UIP Mode 3 | Phase 3B |
| `UniversalInferencePipeline.from_experiment(config).predict_ensemble(X)` | UIP ensemble path | Phase 3B |
| EnsembleBundle with relative paths: save → move → load → predict | Portability fix | Phase 3B |
| Old bundle (v1.2.0 metadata) loads without error | Backward compat | Phase 3A |
| Notebook Cell 8 runs without error after training | Colab inference demo | Phase 3C |

### Regression Tests

| Test | What Could Break |
|------|-----------------|
| Existing `ModelBundle.predict(X_preshaped)` still works | Phase 3B changes to bundle.py |
| Existing `ModelBundle.predict_from_raw(df)` still works for tabular | Phase 3B changes to predict_from_raw |
| `BundleBuilder.build_from_training_result()` still works with old trainers | Phase 3A protocol changes |
| `EnsembleBundle.load()` works with old bundles (absolute paths) | Phase 3B-3 relative path fix |
| All model `save()`/`load()` roundtrips work | Phase 3D-5 arch versioning |
| Enum imports from `src.config` still work | Phase 3D-2/3D-3 enum consolidation |

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| New files | 8 |
| Modified files | 18 |
| Deleted files | 0 (deprecated, removed in later phase) |
| Estimated new lines | ~1,300 |
| Estimated modified lines | ~350 |
| Estimated removed lines | ~30 |
| Total estimated scope | ~1,680 lines of change |

### Phase Sizing

| Phase | Tasks | Complexity | Can Parallelize |
|-------|-------|------------|-----------------|
| 3A (Foundation) | 6 | LOW-MEDIUM | Tasks 3A-1, 3A-3 in parallel; then 3A-2, 3A-4, 3A-5, 3A-6 sequential |
| 3B (Core Inference) | 5 | MEDIUM | 3B-1 first; then 3B-2, 3B-3, 3B-4, 3B-5 parallel |
| 3C (Integration) | 5 | LOW-MEDIUM | 3C-1 through 3C-5 mostly parallel |
| 3D (Cleanup) | 7 | LOW-MEDIUM | All fully parallel, no deps on 3A-3C |

---

## VERIFICATION NOTES (Added 2026-02-15)

The following corrections were applied based on codebase verification:

1. **Model count:** 4 tabular models (XGBoost, LightGBM, CatBoost, RandomForest) support `predict_from_raw()` end-to-end, not 3. The remaining 10 models require adapter integration for 3D/4D tensor shaping.
2. **Ensemble paths:** `base_bundles.json` stores relative paths by default, not absolute paths. The fix focuses on robust relative path resolution on load with absolute fallback.
3. **Pickle call sites:** 16 confirmed `pickle.load()` call sites, not 17. The previously listed 17th site was unverified.
4. **Calibrator transfer:** The calibrator is present on the Trainer when `use_calibration=True`. The gap is propagation through the result chain (ModelTrainingResult), not a missing attribute — this is a propagation gap, not a bug.
5. **Tabular preprocessing:** End-to-end `predict_from_raw()` already works for all tabular models (boosting + classical) via PreprocessingGraph → 2D features → predict.
