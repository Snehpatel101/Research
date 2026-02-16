# 02 - Factory & Orchestrator Inspection: Training-Side Code Paths

**Date:** 2026-02-15
**Agent:** 2/10 (Factory & Orchestrator Inspection)
**Purpose:** Deep inspection of training-side code paths that produce model artifacts, documenting entry points, artifact production, bundling integration, and gaps relevant to the deployable-artifact objective.

---

## 1. Files Inspected

| File | Path | Lines | Exists |
|------|------|-------|--------|
| MLFactory | `/home/jake/Desktop/Research/src/factory.py` | 781 | Yes |
| UnifiedTrainingOrchestrator | `/home/jake/Desktop/Research/src/models/training/unified_orchestrator.py` | 2076 | Yes |
| Top-level orchestrator | `/home/jake/Desktop/Research/src/orchestrator.py` | ~50 | **EXISTS** — contains MLPipeline class (thin wrapper around UnifiedTrainingOrchestrator with bundling support) |
| BundleBuilder | `/home/jake/Desktop/Research/src/inference/builder.py` | 784 | Yes |
| ModelTrainingService | `/home/jake/Desktop/Research/src/models/training/services/model_training.py` | ~180 | Yes |
| EnsembleService | `/home/jake/Desktop/Research/src/models/training/services/ensemble_service.py` | 412 | Yes |
| ArtifactManager | `/home/jake/Desktop/Research/src/models/training/services/artifact_persistence.py` | ~219 | Yes |
| ModelBundle | `/home/jake/Desktop/Research/src/inference/bundle.py` | ~1097 | Yes |
| EnsembleBundle | `/home/jake/Desktop/Research/src/inference/ensemble_bundle.py` | ~910 | Yes |

---

## 2. MLFactory (`src/factory.py`)

### 2.1 Entry Points

| Method | Signature | Return Type | Line |
|--------|-----------|-------------|------|
| `run()` | `(resume: bool = False) -> ExperimentResult` | `ExperimentResult` | L233 |
| `resume_from_checkpoint()` | `() -> ExperimentResult` | `ExperimentResult` | L333 |

### 2.2 Training Flow (Step by Step)

The `run()` method at L233 executes 4 phases:

1. **Phase 1 - Data Pipeline** (L267-273): `_run_data_pipeline()` at L487
   - Loads raw OHLCV from parquet (L500)
   - Normalizes columns, ensures datetime index (L504-512)
   - Runs FeatureEngineer (L531-538)
   - Runs TripleBarrierLabeler for each horizon (L549-558)
   - Creates default `label` column from first horizon (L562)
   - Returns DataFrame with features + labels

2. **Phase 2 - Training** (L276-279): `_run_training(df)` at L578
   - Creates `PipelineConfig` via `config.to_pipeline_config()` (L594)
   - Instantiates `UnifiedTrainingOrchestrator(pipeline_config)` (L595)
   - Calls `orchestrator.train(df)` (L596)
   - Returns `TrainingRunResult` (typed as `Any` in factory)

3. **Phase 3 - Evaluation** (L284-291): `_run_evaluation(df, training_result)` at L603
   - Optional backtest via `Backtester` (L619)
   - Extracts predictions from OOF (aligned ensemble OOF first, then best model OOF) (L706-751)

4. **Phase 4 - Bundling** (L294-296): `_create_bundle(training_result)` at L673
   - Gated by `config.bundling.create_bundle` (L683)
   - Creates `BundleBuilder(pipeline_config)` (L691)
   - Calls `builder.build_from_training_result(training_result)` (L693)
   - Returns `bundle_path = output_dir / "bundles"` (L694)

### 2.3 Artifact Production

The `ExperimentResult` dataclass (L66-97) contains:

| Field | Type | Source |
|-------|------|--------|
| `run_id` | `str` | `config.run_id` |
| `config` | `ExperimentConfig` | Passed through |
| `success` | `bool` | True if no exception |
| `duration_seconds` | `float` | Wall clock |
| `n_models` | `int` | From `training_result.n_models` |
| `best_model` | `str | None` | From `training_result.best_model` |
| `metrics` | `dict[str, dict[str, float]]` | From `training_result.get_metrics_summary()` |
| `ensemble_metrics` | `dict[str, float]` | From `training_result.ensemble_result.metrics` (L763-765) |
| `backtest_metrics` | `dict[str, float]` | From backtester |
| `bundle_path` | `Path | None` | From `_create_bundle()` |
| `output_dir` | `Path | None` | Factory output dir |

### 2.4 Bundling Integration

**Key point:** Bundling in MLFactory is **optional** and **separate** from training (L683-704).

- BundleBuilder is instantiated with `PipelineConfig` (L691)
- Calls `build_from_training_result(training_result)` (L693)
- Does NOT pass ensemble_result separately -- only the `TrainingRunResult`
- Does NOT pass feature_specs
- The `ExperimentResult.bundle_path` is just the directory `output_dir / "bundles"`, not per-model paths

### 2.5 Per-Horizon Handling

MLFactory itself does NOT iterate over horizons for training -- it delegates entirely to `UnifiedTrainingOrchestrator.train(df)`. The factory only iterates horizons during:
- Label generation (L549-558): Creates `label_h{horizon}` columns

### 2.6 Key Gaps for Deployable Artifact

| Gap | Location | Severity |
|-----|----------|----------|
| **No deploy/ directory creation** | `_create_bundle()` L673 | HIGH -- outputs to `bundles/`, not `deploy/h{horizon}/artifact/` |
| **No per-horizon artifact selection** | `_create_bundle()` L693 | HIGH -- no selection policy (ensemble vs best model) |
| **No deploy manifest** | Entire file | HIGH -- no `manifest.json` generation |
| **No validation report** | Entire file | MEDIUM -- no smoke test report |
| **Bundle path is just a directory** | L694-695 | MEDIUM -- `ExperimentResult.bundle_path` is `output_dir / "bundles"`, not meaningful |
| **TYPE_CHECKING import references non-existent `TrainingResult`** | L54 | LOW -- dead import, actual type is `TrainingRunResult` |

---

## 3. UnifiedTrainingOrchestrator (`src/models/training/unified_orchestrator.py`)

### 3.1 Entry Points

| Method | Signature | Return Type | Line |
|--------|-----------|-------------|------|
| `train()` | `(df, additional_dfs=None, generate_financial_report=True) -> TrainingRunResult` | `TrainingRunResult` | L580 |
| `get_trained_model()` | `(model_key: str) -> Any | None` | Trainer or None | L1860 |
| `get_oof_predictions()` | `(model_key: str) -> OOFPrediction | None` | OOFPrediction or None | L1872 |
| `get_meta_labeling_models()` | `(horizon: int) -> tuple | None` | (primary, meta, threshold) or None | L1884 |
| `predict_meta_labeling()` | `(X, horizon) -> tuple | None` | (directions, probabilities, positions) | L1920 |

Convenience functions:
| Function | Line |
|----------|------|
| `train_pipeline(config, df)` | L1986 |
| `train_meta_labeling(config, df)` | L2019 |

### 3.2 Training Flow (Standard Mode -- the primary path)

`train()` at L580:

1. **Pre-training validation** (L612): `_pre_training_validation(df)` -- contract/leakage/lookahead checks
2. **Mode routing** (L615-626): Routes to `_train_standard`, `_train_walk_forward`, `_train_regime_aware`, or `_train_meta_labeling`
3. **Ensemble building** (L633-643): If `config.build_ensemble` and >1 model results
4. **Save results** (L646): `_save_results()` delegates to `ArtifactManager`
5. **Financial reports** (L658-659): Optional report generation
6. **Return TrainingRunResult** (L661-670)

**Standard training** (`_train_standard` at L672):

For each horizon (L696):
1. Separate models into boosting (CPU-parallelizable) vs neural (sequential) (L700-707)
2. **Boosting parallel** (L710-720): If 2+ boosting models, call `_train_boosting_parallel()`
3. **Sequential** (L726-732): For each remaining model, call `_train_model_sequential()`
4. **Clear cache** (L735): `_clear_prepared_cache()` after each horizon

**Per-model training** (`_train_model_sequential` at L829):

1. `_prepare_with_cache(df, model_name)` (L850) -- data preparation via adapters with caching
2. `_train_single_model(model_name, prepared, horizon)` (L859) -- delegates to `ModelTrainingService`
3. Store result in `self._model_results[key]` where key = `"{model_name}_h{horizon}"` (L866)
4. Generate OOF if enabled (L869-872)

**Single model training** (`_train_single_model` at L884):

1. Creates `ModelTrainingRequest` (L891-900)
2. Calls `self._model_service.train_model(request)` (L902)
3. **Calibration** (L905-906): If `config.auto_calibrate`, calls `_calibrate_model(result, prepared, model_name)` -- **CRITICAL: see calibrator gap below**
4. Stores trainer in `self._trained_models` (L909)
5. Converts service `ModelTrainingResult` to orchestrator's `ModelTrainingResult` (L912-920) -- **CRITICAL: calibrator is NOT transferred** (calibrator works via Trainer attribute but lost in orchestrator's ModelTrainingResult conversion path)

### 3.3 Artifact Production

The `train()` method returns `TrainingRunResult` (L661-670) containing:

| Field | Type | Source | Line |
|-------|------|--------|------|
| `run_id` | `str` | Generated timestamp | L246-249 |
| `config` | `PipelineConfig` | Passed through | L663 |
| `model_results` | `dict[str, ModelTrainingResult]` | `self._model_results` | L664 |
| `ensemble_result` | `ModelTrainingResult | None` | From `_build_ensemble()` | L665 |
| `stacking_dataset` | `StackingDataset | None` | From `_build_ensemble()` | L666 |
| `aligned_oof` | `AlignedOOFResult | None` | From `_build_ensemble()` | L667 |
| `total_time_seconds` | `float` | Wall clock | L668 |
| `output_dir` | `Path` | Run output directory | L669 |

Internal state also saved by `_save_results()` via `ArtifactManager`:
- `config.json` -- PipelineConfig
- `metrics_summary.json` -- all model metrics
- `oof/` directory -- OOF parquet files per model
- `models/` directory -- pickled trainer files per model

### 3.4 Calibrator Flow (CRITICAL GAP)

The calibration happens in `_calibrate_model()` at L922-999:

1. Calibrator is fitted on validation probabilities (L981-982)
2. **Attached to the service result** via `result.calibrator = calibrator` (L993) -- this sets an attribute on `ModelTrainingResult` from `services/model_training.py`
3. **BUT**: In `_train_single_model()` at L912-920, the orchestrator creates a NEW `ModelTrainingResult` (orchestrator's own dataclass) by copying fields one-by-one: `model_name`, `horizon`, `metrics`, `trainer`, `training_time_seconds`, `n_features`, `data_rank`
4. **The calibrator is NOT copied** -- the orchestrator's `ModelTrainingResult` (L78-111) has NO `calibrator` field (calibrator works via Trainer attribute but lost in orchestrator's ModelTrainingResult conversion path)
5. The calibrator is only accessible if the **trainer** object has it as an attribute -- but `_calibrate_model()` sets it on the service result, NOT on the trainer

**Result:** Calibrator is lost during the conversion from service result to orchestrator result. This is confirmed by Agent 1's finding: "Calibrator transfer broken -- Calibrator on service result, not transferred to ModelTrainingResult."

**Parallel path also affected:** In `_train_boosting_parallel()` at L788-823, the same pattern occurs:
- Calibration at L793
- New `ModelTrainingResult` at L799-807 -- no calibrator field

### 3.5 Ensemble Construction

`_build_ensemble()` at L1018:

1. Creates `EnsembleRequest` with `self._oof_predictions`, config, df (L1038-1042)
2. Calls `self._ensemble_service.build_ensemble(request)` (L1043)
3. Returns `EnsembleServiceResult` containing:
   - `aligned_oof: AlignedOOFResult | None`
   - `stacking_dataset: StackingDataset | None`
   - `ensemble_metrics: dict`
   - `meta_learner: Any | None`
   - `training_time_seconds: float`
   - `diversity_metrics: DiversityMetrics | None`
4. Converts to `ModelTrainingResult` at L1056-1062 -- the ensemble result is a `ModelTrainingResult` with `trainer=result.meta_learner`
5. **After ensemble building**, OOF predictions are cleared to free memory (L641-643)

**EnsembleService** (`services/ensemble_service.py`):
- Aligns OOF predictions via `OOFAligner` (L134)
- Builds stacking dataset (L168-183)
- Trains meta-learner directly (not through Trainer) via `meta_learner.fit(X_train, y_train, X_val, y_val)` (L304-309)
- Returns `EnsembleServiceResult` (L192-199)

### 3.6 Per-Horizon Handling

Horizons are iterated in `_train_standard()` at L696:
```python
for horizon in self.config.horizons:
```

Each model within a horizon produces a result keyed as `"{model_name}_h{horizon}"`. PreparedData cache is cleared after each horizon completes (L735).

**Ensemble is NOT per-horizon.** The ensemble is built once across ALL model results (L633-643), using the first horizon for the ensemble result (L1058). This means:
- If training produces `xgboost_h5`, `xgboost_h10`, `lightgbm_h5`, `lightgbm_h10`
- The ensemble is built from ALL of those OOF predictions together
- The ensemble result horizon is hardcoded to `config.horizons[0]` (L1058)

**This is a gap for per-horizon deploy artifacts:** The plan requires one deployable artifact per horizon, but the ensemble is a single entity spanning all horizons.

### 3.7 What's Returned to Caller

`TrainingRunResult` (L115-166):

```python
@dataclass
class TrainingRunResult:
    run_id: str
    config: PipelineConfig
    model_results: dict[str, ModelTrainingResult]  # key: "{model}_h{horizon}"
    ensemble_result: ModelTrainingResult | None
    stacking_dataset: StackingDataset | None
    aligned_oof: AlignedOOFResult | None
    total_time_seconds: float
    output_dir: Path
```

Properties:
- `n_models` (L140-142): `len(self.model_results)`
- `best_model` (L145-151): Key with highest `val_f1`
- `get_metrics_summary()` (L153-155): Dict of metrics per model key

---

## 4. BundleBuilder (`src/inference/builder.py`)

### 4.1 Entry Points

| Method | Signature | Return Type | Line |
|--------|-----------|-------------|------|
| `build_from_training_result()` | `(training_result, include_preprocessing_graph=True, include_calibrator=True, feature_specs=None) -> BundleBuildResult` | `BundleBuildResult` | L238 |
| `build_ensemble_bundle()` | `(ensemble_result, base_bundles=None) -> Path` | `Path` | L370 |
| `build_all()` | `(training_result=None, ensemble_result=None, ...) -> BundleBuildResult` | `BundleBuildResult` | L439 |
| `validate_bundles()` | `() -> dict` | Validation dict | L665 |

### 4.2 Bundle Building Flow

`build_from_training_result()` at L238:

1. Creates `PreprocessingGraph` if `include_preprocessing_graph` (L280-281)
2. For each `(key, model_result)` in `training_result.model_results` (L284):
   a. Gets trainer from `model_result.trainer` (L292)
   b. Extracts model via duck-typing: tries `model`, `_model`, `estimator`, `_estimator`, `get_model()` (L571-580)
   c. Extracts scaler via duck-typing: tries `scaler`, `_scaler`, `feature_scaler`, `_feature_scaler` (L592-596)
   d. Extracts feature columns via duck-typing (L614-628)
   e. **Extracts calibrator** via duck-typing: tries `calibrator`, `_calibrator`, `prob_calibrator` on **trainer** (L640-644)
   f. Creates `ModelBundle.from_training(...)` (L322-337)
   g. Saves bundle to `bundles_dir / "{model_name}_h{horizon}"` (L340-341)

### 4.3 Calibrator Extraction Chain

The BundleBuilder tries to extract calibrator from the **trainer** object (L640-644):
```python
def _extract_calibrator(self, trainer: Any) -> Any | None:
    for attr in ["calibrator", "_calibrator", "prob_calibrator"]:
        calibrator = getattr(trainer, attr, None)
        if calibrator is not None:
            return calibrator
    return None
```

**But the calibrator was set on the service result, not the trainer.** The chain is:
1. `_calibrate_model()` sets `result.calibrator = calibrator` (on service `ModelTrainingResult`)
2. Orchestrator creates NEW `ModelTrainingResult` without calibrator field
3. Orchestrator stores `result.trainer` (the Trainer object)
4. BundleBuilder looks for `trainer.calibrator` -- **which was never set**
5. **Result: calibrator is always None in bundles** (calibrator works via Trainer attribute but lost in orchestrator's ModelTrainingResult conversion path)

### 4.4 Ensemble Bundle Building

`build_ensemble_bundle()` at L370:
- Saves metadata JSON (L398-413)
- Saves stacking dataset as parquet (L416-422)
- Saves aligned OOF info (L425-433)
- Does NOT save the meta-learner model as a pickle (only metadata)
- Does NOT create a loadable `EnsembleBundle` -- just raw files in `bundles/ensemble/`
- **Returns a Path, not an EnsembleBundle object**

**Gap:** The BundleBuilder's `build_ensemble_bundle()` does NOT call `EnsembleBundle.from_ensemble_result()` (which exists in `ensemble_bundle.py`). It creates a custom directory structure that is NOT the same format as `EnsembleBundle.save()` produces.

### 4.5 How MLFactory Calls BundleBuilder

In `factory.py` L673-704:
```python
def _create_bundle(self, training_result: Any) -> Path | None:
    if not self.config.bundling.create_bundle:
        return None
    builder = BundleBuilder(pipeline_config)
    bundle_result = builder.build_from_training_result(training_result)
    bundle_path = self.output_dir / "bundles"
    ...
    return bundle_path
```

**Critical observations:**
1. Only calls `build_from_training_result()`, NOT `build_all()` or `build_ensemble_bundle()`
2. Does NOT pass `ensemble_result` separately -- the ensemble model is inside `training_result.model_results` only if it was added as a `ModelTrainingResult` (which it is, at orchestrator L1056-1062)
3. Does NOT pass `feature_specs`
4. Returns just the directory path, not per-bundle paths

---

## 5. Key Dataclasses

### 5.1 ModelTrainingResult (Orchestrator) -- L78-111

```python
@dataclass
class ModelTrainingResult:
    model_name: str
    horizon: int
    metrics: dict[str, float] = field(default_factory=dict)
    oof_prediction: OOFPrediction | None = None
    trainer: Any | None = None
    training_time_seconds: float = 0.0
    n_features: int = 0
    data_rank: int = 2
```

**Note:** NO `calibrator` field. NO `scaler` field. These must be extracted from `trainer` by duck-typing.

### 5.2 ModelTrainingResult (Service) -- `services/model_training.py` L39-48

```python
@dataclass
class ModelTrainingResult:
    model_name: str
    horizon: int
    trainer: Any
    metrics: dict[str, float] = field(default_factory=dict)
    training_time_seconds: float = 0.0
    n_features: int = 0
    data_rank: int = 2
```

**Note:** Same name, different module. Also NO `calibrator` field. The calibrator is dynamically attached via `result.calibrator = calibrator` in `_calibrate_model()`.

### 5.3 TrainingRunResult -- L115-166

```python
@dataclass
class TrainingRunResult:
    run_id: str
    config: PipelineConfig
    model_results: dict[str, ModelTrainingResult] = field(default_factory=dict)
    ensemble_result: ModelTrainingResult | None = None
    stacking_dataset: StackingDataset | None = None
    aligned_oof: AlignedOOFResult | None = None
    total_time_seconds: float = 0.0
    output_dir: Path = field(default_factory=lambda: Path("."))
```

Properties: `n_models`, `best_model`, `get_metrics_summary()`, `to_dict()`

### 5.4 EnsembleServiceResult -- `services/ensemble_service.py` L40-48

```python
@dataclass
class EnsembleServiceResult:
    aligned_oof: AlignedOOFResult | None
    stacking_dataset: StackingDataset | None
    ensemble_metrics: dict[str, Any]
    meta_learner: Any | None = None
    training_time_seconds: float = 0.0
    diversity_metrics: DiversityMetrics | None = None
```

### 5.5 BundleBuildResult -- `inference/builder.py` L55-71

```python
@dataclass
class BundleBuildResult:
    bundle_paths: list[Path] = field(default_factory=list)
    ensemble_bundle_path: Path | None = None
    n_bundles: int = 0
    total_size_mb: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 5.6 BundleMetadata -- `inference/bundle.py` L71-92

```python
@dataclass
class BundleMetadata:
    version: str                              # BUNDLE_VERSION = "1.2.0"
    created_at: str
    model_name: str
    model_family: str
    horizon: int
    n_features: int
    feature_hash: str
    requires_sequences: bool = False
    requires_4d: bool = False
    sequence_length: int = 0
    n_timeframes: int = 0
    has_calibrator: bool = False
    has_preprocessing_graph: bool = False
    preprocessing_graph_hash: str = ""
    has_feature_spec: bool = False
    feature_spec_hash: str = ""
    symbol: str = ""
    training_metrics: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
```

---

## 6. Relationship Diagram

```
MLFactory.run()
    |
    +-- _run_data_pipeline() -> DataFrame (features + labels)
    |
    +-- _run_training(df) -> TrainingRunResult
    |       |
    |       +-- UnifiedTrainingOrchestrator(PipelineConfig)
    |       |       |
    |       |       +-- _train_standard(df)
    |       |       |       |
    |       |       |       +-- for horizon in horizons:
    |       |       |       |       +-- _train_boosting_parallel() or _train_model_sequential()
    |       |       |       |       |       +-- _prepare_with_cache(df, model_name) -> PreparedData
    |       |       |       |       |       +-- _train_single_model(model, prepared, horizon)
    |       |       |       |       |       |       +-- ModelTrainingService.train_model(request)
    |       |       |       |       |       |       +-- _calibrate_model(result, prepared)  <-- calibrator SET on service result
    |       |       |       |       |       |       +-- return ModelTrainingResult(...)       <-- calibrator LOST here
    |       |       |       |       |       +-- _generate_oof(model, prepared, horizon) -> OOFPrediction
    |       |       |       |       +-- _clear_prepared_cache()
    |       |       |
    |       |       +-- _build_ensemble(df) -> (aligned_oof, stacking_dataset, ensemble_result)
    |       |       |       +-- EnsembleService.build_ensemble(request) -> EnsembleServiceResult
    |       |       |       +-- _analyze_ensemble_diversity()
    |       |       |
    |       |       +-- _save_results() -> ArtifactManager.save_all()
    |       |       +-- _generate_financial_reports(df)
    |       |       +-- return TrainingRunResult(...)
    |       |
    |       +-- TrainingRunResult { model_results, ensemble_result, aligned_oof, ... }
    |
    +-- _run_evaluation(df, training_result) -> backtest_metrics
    |
    +-- _create_bundle(training_result) -> Path | None
    |       |
    |       +-- BundleBuilder(PipelineConfig)
    |       +-- builder.build_from_training_result(training_result)
    |               |
    |               +-- for each model_result:
    |                       +-- _extract_model(trainer)      <-- duck-typing
    |                       +-- _extract_scaler(trainer)      <-- duck-typing
    |                       +-- _extract_calibrator(trainer)  <-- duck-typing, ALWAYS FAILS
    |                       +-- ModelBundle.from_training(...)
    |                       +-- bundle.save(bundles_dir / "{model}_h{horizon}")
    |
    +-- return ExperimentResult(bundle_path=output_dir/"bundles", ...)
```

---

## 7. Conflicts with Deployable-Artifact Objective

### 7.1 CRITICAL: Calibrator Transfer is Broken

**Location:** `unified_orchestrator.py` L912-920 and L799-807
**Impact:** Calibrators are NEVER included in bundles

The calibrator is fitted in `_calibrate_model()` (L993: `result.calibrator = calibrator`) but this sets it on the **service result**, not on the **trainer** object. When the orchestrator creates its own `ModelTrainingResult` at L912-920, it copies `trainer` but not `calibrator`. The BundleBuilder then looks for `trainer.calibrator` (builder.py L640-644) which is never set.

**Fix needed:** Either:
- (a) Set calibrator on the trainer: `result.trainer.calibrator = calibrator` in `_calibrate_model()`
- (b) Add `calibrator` field to orchestrator's `ModelTrainingResult` and pass it through

### 7.2 CRITICAL: No Auto-Bundling in Deploy Format

**Location:** `factory.py` L673-704
**Impact:** No deploy/ directory, no per-horizon artifacts, no manifest

The factory creates bundles in `output_dir/bundles/` but:
- No `deploy/` directory structure
- No `manifest.json` with run metadata
- No per-horizon selection (ensemble vs best model)
- No validation report
- Ensemble bundle NOT created (only `build_from_training_result()` called, not `build_all()`)

### 7.3 HIGH: Ensemble Bundle Not Properly Created

**Location:** `factory.py` L693 only calls `build_from_training_result()`, builder.py `build_ensemble_bundle()` L370-437
**Impact:** Even if called, ensemble would not be a loadable EnsembleBundle

Two separate issues:
1. Factory does NOT call `build_ensemble_bundle()` at all
2. BundleBuilder's `build_ensemble_bundle()` saves raw files, NOT an `EnsembleBundle`-compatible directory

### 7.4 HIGH: Ensemble is NOT Per-Horizon

**Location:** `unified_orchestrator.py` L633-643, L1058
**Impact:** Cannot produce one deployable artifact per horizon for ensemble mode

The ensemble is built from ALL model results across ALL horizons. The resulting `ensemble_result` has `horizon=config.horizons[0]` (hardcoded first horizon). For per-horizon artifacts, the ensemble would need to be built separately per horizon.

### 7.5 HIGH: `predict_from_raw()` Missing on EnsembleBundle

**Location:** `src/inference/ensemble_bundle.py`
**Impact:** Ensemble artifacts cannot satisfy the `predict_from_raw(raw_df) -> PredictionResult` contract

`EnsembleBundle` has `predict()`, `predict_proba()`, `predict_classes()`, and `predict_from_base_features()` -- but NO `predict_from_raw()`. Only `ModelBundle` has `predict_from_raw()`, and even that only works for tabular/boosting models (2D).

### 7.6 MEDIUM: Feature Specs Never Passed

**Location:** `factory.py` L693 -- `build_from_training_result(training_result)` without `feature_specs`
**Impact:** Bundles lack feature specification for train/serve parity

BundleBuilder supports `feature_specs` parameter (builder.py L243), but MLFactory never passes it.

### 7.7 MEDIUM: Two ModelTrainingResult Classes

**Location:** `unified_orchestrator.py` L78 vs `services/model_training.py` L39
**Impact:** Confusing conversion, lost attributes

Both share the same name but are different dataclasses. The service version has `trainer: Any` (required), while the orchestrator version has `trainer: Any | None` and adds `oof_prediction`. The conversion at L912-920 copies fields manually, losing any dynamically-attached attributes.

### 7.8 MEDIUM: Bundle Version Still 1.2.0

**Location:** `inference/bundle.py` L54
**Impact:** Plan specifies bumping to 1.3.0 for new metadata fields

`BUNDLE_VERSION = "1.2.0"` -- Agent 1 noted this needs verification before bumping.

### 7.9 LOW: Dead TYPE_CHECKING Import in Factory

**Location:** `factory.py` L54
**Impact:** References non-existent `TrainingResult` class

```python
if TYPE_CHECKING:
    from src.models.training.result import TrainingResult
```

This module/class does not exist. The actual type is `TrainingRunResult` from `unified_orchestrator.py`.

---

## 8. Summary for Downstream Agents

### What the training pipeline produces today:
1. `TrainingRunResult` with `model_results` dict (keyed `"{model}_h{horizon}"`)
2. Each `ModelTrainingResult` contains a `trainer` object (with model, scaler inside)
3. Optional `ensemble_result` (single `ModelTrainingResult` with meta-learner as trainer)
4. Optional `aligned_oof` and `stacking_dataset`
5. Saved artifacts: `config.json`, `metrics_summary.json`, `oof/`, `models/`

### What the training pipeline does NOT produce:
1. No `deploy/` directory structure
2. No per-horizon artifact selection
3. No deploy manifest
4. No calibrator in bundles (calibrator works via Trainer attribute but lost in orchestrator's ModelTrainingResult conversion path)
5. No EnsembleBundle objects (BundleBuilder makes raw files)
6. No `predict_from_raw()` for ensembles
7. No `predict_from_raw()` for neural/transformer models (10/12 models: 8 needing 3D + 2 needing 4D)
8. No validation/smoke test reports

### Critical path for implementation:
```
Fix calibrator transfer (7.1)
  -> Fix ensemble per-horizon (7.4)
  -> Create deploy selector + manifest writer (7.2)
  -> Add predict_from_raw to EnsembleBundle (7.5)
  -> Wire auto-bundling into MLFactory (7.2)
  -> Pass feature_specs (7.6)
```
