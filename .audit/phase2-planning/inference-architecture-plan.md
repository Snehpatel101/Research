# UniversalInferencePipeline Architecture Plan

**Date:** 2026-02-15
**Author:** Architecture Planner (Phase 2)
**Input:** Phase 1 Consolidated Findings (sections 1, 2.1, 5.3, 6)

---

## 1. Architecture Decision: Where Adapter Integration Lives

### Decision: Option C — New `UniversalInferencePipeline` orchestration layer

**Why not Option A (inside ModelBundle.predict_from_raw)?**
- `ModelBundle` is a serialization/storage container. Adding adapter orchestration couples packaging to data transformation.
- `predict_from_raw()` currently returns 2D features from `PreprocessingGraph.transform()` and passes them directly to `predict()`. Inserting adapter routing here means the bundle must import and manage `AdapterRegistry`, `AdapterFactory`, `SequenceAdapter`, `MultiStreamAdapter`, and their dependencies — violating single responsibility.
- Different models need different adapter configuration (sequence_length, timeframes, stride). The bundle metadata stores these as flags but not as full adapter configs.

**Why not Option B (inside PreprocessingGraph.transform)?**
- `PreprocessingGraph` is feature engineering: raw OHLCV → 2D feature DataFrame. This is inherently tabular — indicators, wavelets, and regime features produce columns, not tensors.
- Forcing 3D/4D output from the graph would break its clean responsibility boundary and make it model-aware (it currently has no knowledge of which model will consume its output).

**Why Option C wins:**
- Clean separation: `PreprocessingGraph` does feature engineering (always 2D), adapters do shape transformation (2D→3D/4D), `ModelBundle` does prediction.
- Reuses the exact same contract-driven adapter routing used in training (`AdapterRegistry.get_for_model()` → `adapter.transform()`), ensuring train/serve parity.
- Single orchestration point that can handle all 3 input modes without any existing class needing modification.
- The pieces already exist; this layer just wires them together.

---

## 2. Overlap Resolution: InferencePipeline vs InferenceOrchestrator

### Current State

| Aspect | `InferencePipeline` | `InferenceOrchestrator` |
|--------|--------------------|-----------------------|
| **Location** | `src/inference/pipeline.py` | `src/inference/orchestrator.py` |
| **Input** | `list[ModelBundle]` | `PipelineConfig` + lazy bundle loading |
| **predict()** | Delegates to `bundle.predict(X)` | Delegates to `bundle.predict(X)` |
| **predict_from_raw()** | Not supported | Uses `PreprocessingGraph.transform()` → 2D only |
| **Ensemble** | Soft/hard/weighted vote across bundles | Meta-learner via EnsembleBundle or averaging |
| **Batch** | Not supported | `predict_batch()` with chunking |
| **Unique** | `InferenceResult`/`EnsembleResult` wrappers, timing | `predict_with_uncertainty()`, config-driven loading |
| **Used by** | `server.py`, `batch.py` (lower-level) | Recommended entry point per `__init__.py` |

### Decision: Merge into `UniversalInferencePipeline`, deprecate both

Neither class handles the adapter gap. Rather than patching one and leaving the other as dead weight, we build `UniversalInferencePipeline` as the replacement that subsumes both:

- **From InferenceOrchestrator:** Config-driven loading (`from_experiment`, `from_bundle`, `from_bundles`, `from_training_result`), ensemble support, batch inference, `predict_from_raw()`.
- **From InferencePipeline:** `InferenceResult`/`EnsembleResult` structured outputs, ensemble voting methods, timing metadata.
- **New:** Adapter integration for all 12 core models.

**Migration path:**
1. Build `UniversalInferencePipeline` in new file `src/inference/universal_pipeline.py`.
2. Keep `InferencePipeline` and `InferenceOrchestrator` temporarily with deprecation warnings pointing to `UniversalInferencePipeline`.
3. Update `server.py`, `batch.py`, and notebook to use `UniversalInferencePipeline`.
4. Remove deprecated classes in Phase 2c cleanup.

---

## 3. Class Design: UniversalInferencePipeline

### Constructor and Core State

```python
class UniversalInferencePipeline:
    """
    THE single entry point for all inference in ML Factory.

    Supports 3 input modes:
      Mode 1: Raw OHLCV → features → adapt → predict
      Mode 2: Pre-computed features (2D DataFrame) → adapt → predict
      Mode 3: Pre-shaped tensors (2D/3D/4D ndarray) → predict directly

    Handles all 12 core model families (boosting, RNN, CNN, transformer, MLP)
    via contract-driven adapter routing.
    """

    def __init__(
        self,
        bundles: dict[str, ModelBundle],
        ensemble_bundle: EnsembleBundle | None = None,
        preprocessing_graph: PreprocessingGraph | None = None,
        config: PipelineConfig | None = None,
        scaling_source: ScalingSource = ScalingSource.BUNDLE,
    ) -> None:
        self._bundles = bundles
        self._ensemble_bundle = ensemble_bundle
        self._preprocessing_graph = preprocessing_graph
        self._config = config
        self._scaling_source = scaling_source
        # Cache: model_name -> instantiated adapter
        self._adapter_cache: dict[str, BaseAdapter] = {}
```

### Enum: ScalingSource

```python
class ScalingSource(str, Enum):
    """Controls which scaler is applied during inference."""
    BUNDLE = "bundle"           # Scaler in ModelBundle (default)
    PREPROCESSING = "preprocessing"  # Scaler in PreprocessingGraph
    NONE = "none"               # No scaling (caller pre-scaled)
```

This resolves the double-scaling risk (Findings 2.1, GAP 1). The pipeline enforces exactly one scaling step by checking `scaling_source` before prediction.

### Class Methods (Constructors)

```python
@classmethod
def from_bundle(cls, path: str | Path, config: PipelineConfig | None = None) -> Self:
    """Load single bundle."""

@classmethod
def from_bundles(cls, paths: list[str | Path], config: PipelineConfig | None = None) -> Self:
    """Load multiple bundles."""

@classmethod
def from_experiment(cls, config: PipelineConfig, load_ensemble: bool = True) -> Self:
    """Load all bundles from experiment output directory."""

@classmethod
def from_training_result(cls, training_result: Any, config: PipelineConfig | None = None) -> Self:
    """Load from PHASE_3 TrainingRunResult."""
```

### Core Prediction Methods

```python
def predict(
    self,
    X: pd.DataFrame | np.ndarray,
    model_name: str | None = None,
    calibrate: bool = True,
) -> InferenceResult:
    """
    Predict from pre-computed features (2D DataFrame) or pre-shaped tensors.

    Routing logic:
    1. If X is ndarray with ndim >= 2: Mode 3 (pre-shaped) → skip adapter, predict directly
    2. If X is DataFrame: Mode 2 → route through adapter based on model contract → predict

    For Mode 2 (DataFrame input):
      a. Look up model contract via get_model_contract(model_name)
      b. If contract.input_rank == TABULAR_2D: extract columns, predict (no adapter needed)
      c. If contract.input_rank == SEQUENCE_3D: get SequenceAdapter, transform → 3D, predict
      d. If contract.input_rank == MULTI_TF_4D: get MultiStreamAdapter, transform → 4D, predict
    """

def predict_from_raw(
    self,
    raw_df: pd.DataFrame,
    model_name: str | None = None,
    calibrate: bool = True,
    additional_dfs: dict[str, pd.DataFrame] | None = None,
) -> InferenceResult:
    """
    End-to-end: Raw OHLCV → features → adapt → predict.

    Steps:
    1. PreprocessingGraph.transform(raw_df) → 2D features DataFrame
    2. Route through predict() which handles adapter transformation

    For 4D models (PatchTST, iTransformer):
      - additional_dfs provides pre-computed MTF DataFrames
      - OR raw_df is 1min data and MTFFeatureGenerator creates higher TFs
    """

def predict_all(
    self,
    X: pd.DataFrame | np.ndarray,
    calibrate: bool = True,
) -> dict[str, InferenceResult]:
    """Get predictions from all loaded models."""

def predict_ensemble(
    self,
    X: pd.DataFrame | np.ndarray,
    method: str = "soft_vote",
    weights: list[float] | None = None,
    calibrate: bool = True,
) -> EnsembleResult:
    """Ensemble prediction combining all models."""

def predict_batch(
    self,
    data: pd.DataFrame | Path,
    batch_size: int = 10000,
    model_name: str | None = None,
    output_path: Path | None = None,
    calibrate: bool = True,
) -> pd.DataFrame:
    """Batch inference for large datasets."""

def predict_with_uncertainty(
    self,
    X: pd.DataFrame | np.ndarray,
    calibrate: bool = True,
) -> dict[str, Any]:
    """Predictions with uncertainty estimates from model disagreement."""
```

### Internal Adapter Routing

```python
def _get_adapter_for_model(self, model_name: str) -> BaseAdapter | None:
    """
    Get or create cached adapter for model.

    Uses contract-driven routing:
      contract = get_model_contract(model_name)
      if contract.input_rank == TABULAR_2D: return None (no adapter needed)
      adapter = AdapterRegistry.get_for_model(model_name, ...)
      return adapter
    """
    if model_name in self._adapter_cache:
        return self._adapter_cache[model_name]

    from src.core.contracts import get_model_contract
    contract = get_model_contract(model_name)

    if contract.input_rank == DataRank.TABULAR_2D:
        self._adapter_cache[model_name] = None
        return None

    from src.data.adapters import AdapterRegistry
    adapter = AdapterRegistry.get_for_model(
        model_name,
        feature_columns=bundle.feature_columns,
        sequence_length=bundle.metadata.sequence_length,
    )
    self._adapter_cache[model_name] = adapter
    return adapter

def _adapt_input(
    self,
    X: pd.DataFrame,
    model_name: str,
    bundle: ModelBundle,
    additional_dfs: dict[str, pd.DataFrame] | None = None,
) -> np.ndarray:
    """
    Transform 2D DataFrame to model-expected shape.

    Returns:
      2D ndarray for boosting/classical
      3D ndarray for RNN/CNN/MLP
      4D ndarray for transformers
    """
    adapter = self._get_adapter_for_model(model_name)

    if adapter is None:
        # Tabular: just extract columns as array
        return X[bundle.feature_columns].values.astype(np.float32)

    # Use adapter transform (same path as training)
    from src.core.contracts import get_model_contract
    contract = get_model_contract(model_name)
    result = adapter.transform(X, model_contract=contract, additional_dfs=additional_dfs)
    return result.X

def _predict_single(
    self,
    bundle: ModelBundle,
    X: pd.DataFrame | np.ndarray,
    calibrate: bool,
    additional_dfs: dict[str, pd.DataFrame] | None = None,
) -> InferenceResult:
    """
    Internal single-model prediction with adapter routing.

    Flow:
    1. If X is DataFrame → _adapt_input() to get correctly shaped array
    2. Apply scaling (respecting scaling_source to avoid double-scaling)
    3. model.predict(X_shaped) → PredictionResult
    4. Apply calibration if requested
    5. Wrap in InferenceResult with timing
    """
```

### Scaling Resolution

```python
def _apply_scaling(
    self,
    X: np.ndarray,
    bundle: ModelBundle,
) -> np.ndarray:
    """
    Apply exactly one scaling step based on scaling_source.

    ScalingSource.BUNDLE: Use bundle.scaler (default — training scaler)
    ScalingSource.PREPROCESSING: Already scaled by PreprocessingGraph, skip
    ScalingSource.NONE: Caller pre-scaled, skip

    For 3D/4D: reshape → scale → reshape back (same as current bundle.predict)
    """
    if self._scaling_source != ScalingSource.BUNDLE:
        return X
    if bundle.scaler is None:
        return X

    # Handle dimensionality (same logic as current ModelBundle.predict)
    if X.ndim == 2:
        return bundle.scaler.transform(X)
    elif X.ndim == 3:
        orig_shape = X.shape
        X_flat = X.reshape(-1, orig_shape[-1])
        X_scaled = bundle.scaler.transform(X_flat)
        return X_scaled.reshape(orig_shape)
    elif X.ndim == 4:
        orig_shape = X.shape
        X_flat = X.reshape(-1, orig_shape[-1])
        X_scaled = bundle.scaler.transform(X_flat)
        return X_scaled.reshape(orig_shape)
    return X
```

---

## 4. Data Flow Diagrams

### Mode 1: Raw OHLCV → features → adapt → predict

```
                    ┌──────────────────────────┐
                    │  raw_df (OHLCV DataFrame) │
                    │  + optional additional_dfs│
                    │    for MTF timeframes     │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  PreprocessingGraph       │
                    │  .transform(raw_df)       │
                    │                           │
                    │  1. Validate OHLCV cols   │
                    │  2. Resample 1min→5min    │
                    │  3. Compute indicators    │
                    │  4. Compute MTF features  │
                    │  5. Regime detection      │
                    │  6. Drop NaN              │
                    │  7. Scale (if scaling_    │
                    │     source=PREPROCESSING) │
                    │  8. Select feature cols   │
                    │                           │
                    │  Output: 2D DataFrame     │
                    │  (~150 feature columns)   │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Contract Lookup          │
                    │  get_model_contract(name) │
                    │  → input_rank, adapter_id │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
     input_rank=2D      input_rank=3D       input_rank=4D
              │                  │                   │
     ┌────────▼───────┐  ┌──────▼────────┐  ┌──────▼───────────┐
     │ Extract cols   │  │ Sequence      │  │ MultiStream      │
     │ → np.float32   │  │ Adapter       │  │ Adapter          │
     │                │  │ .transform()  │  │ .transform()     │
     │ (n, features)  │  │               │  │                  │
     │                │  │ (n, seq, feat)│  │ (n, tf, seq, f)  │
     └────────┬───────┘  └──────┬────────┘  └──────┬───────────┘
              │                  │                   │
              └──────────────────┼───────────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Scaling (if scaling_     │
                    │  source=BUNDLE)           │
                    │  bundle.scaler.transform()│
                    │  (reshape for 3D/4D)      │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  model.predict(X_shaped)  │
                    │  → PredictionResult       │
                    │                           │
                    │  class_predictions: [-1,0,1]
                    │  class_probabilities: (n,3)
                    │  confidence: (n,)         │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Calibration (optional)   │
                    │  calibrator.calibrate()   │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  InferenceResult          │
                    │  (predictions, timing,    │
                    │   model_name, metadata)   │
                    └──────────────────────────┘
```

### Mode 2: Pre-computed features → adapt → predict

```
                    ┌──────────────────────────┐
                    │  features_df (2D)         │
                    │  Already feature-         │
                    │  engineered DataFrame     │
                    │  (~150 columns)           │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Contract Lookup          │
                    │  get_model_contract(name) │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
     input_rank=2D      input_rank=3D       input_rank=4D
              │                  │                   │
     ┌────────▼───────┐  ┌──────▼────────┐  ┌──────▼───────────┐
     │ Extract cols   │  │ Sequence      │  │ MultiStream      │
     │ df[feat_cols]  │  │ Adapter       │  │ Adapter          │
     │ .values        │  │ .transform()  │  │ .transform()     │
     └────────┬───────┘  └──────┬────────┘  └──────┬───────────┘
              │                  │                   │
              └──────────────────┼───────────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Scale → Predict →        │
                    │  Calibrate → Return       │
                    │  (same as Mode 1 tail)    │
                    └──────────────────────────┘
```

### Mode 3: Pre-shaped tensors → predict directly

```
                    ┌──────────────────────────┐
                    │  X_shaped (np.ndarray)    │
                    │  Already correct shape:   │
                    │  2D (n, feat)             │
                    │  3D (n, seq, feat)        │
                    │  4D (n, tf, seq, feat)    │
                    └────────────┬─────────────┘
                                 │
                                 │  (bypass adapter — shape already correct)
                                 │
                    ┌────────────▼─────────────┐
                    │  Validate shape against   │
                    │  BundleMetadata:          │
                    │  requires_sequences,      │
                    │  requires_4d,             │
                    │  sequence_length,         │
                    │  n_timeframes,            │
                    │  n_features               │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Scale → Predict →        │
                    │  Calibrate → Return       │
                    │  (same as Modes 1&2 tail) │
                    └──────────────────────────┘
```

**Routing decision rule:**
- `isinstance(X, np.ndarray)` → Mode 3 (pre-shaped, skip adapter)
- `isinstance(X, pd.DataFrame)` → Mode 2 (or Mode 1 if called via `predict_from_raw`)

---

## 5. Integration Points

### 5.1 ModelBundle Integration

`UniversalInferencePipeline` does NOT modify `ModelBundle`. Instead, it wraps bundle usage:

```python
# Pipeline uses bundle.model.predict() directly (bypassing bundle.predict()
# to control scaling separately)
bundle = self._bundles[model_name]
X_adapted = self._adapt_input(features_df, model_name, bundle)
X_scaled = self._apply_scaling(X_adapted, bundle)
output = bundle.model.predict(X_scaled)
# Apply calibration if available
if calibrate and bundle.calibrator is not None:
    output = bundle._apply_calibration(output)
```

This avoids double-scaling: `bundle.predict()` applies `bundle.scaler` internally, but we need to control when scaling happens (before or after adapter transformation). By calling `bundle.model.predict()` directly, the pipeline owns the scaling decision.

### 5.2 EnsembleBundle Integration

For ensemble inference, the pipeline:
1. Calls `predict()` on each base model bundle (with adapter routing)
2. Stacks base predictions into meta-learner input
3. Calls `ensemble_bundle.predict(base_predictions)`

```python
def _predict_ensemble_from_features(self, X, calibrate):
    base_preds = {}
    for name, bundle in self._bundles.items():
        result = self._predict_single(bundle, X, calibrate=False)
        base_preds[name] = result.predictions.class_probabilities

    if self._ensemble_bundle is not None:
        return self._ensemble_bundle.predict(base_preds, calibrate=calibrate)
    else:
        # Fallback: soft vote
        return self._soft_vote(base_preds)
```

### 5.3 server.py Integration

```python
# Before (server.py uses InferencePipeline):
pipeline = InferencePipeline.from_bundle(bundle_path)
result = pipeline.predict(X)

# After:
pipeline = UniversalInferencePipeline.from_bundle(bundle_path)
result = pipeline.predict(X)  # Now works for ALL model types
result = pipeline.predict_from_raw(raw_df)  # Also works for all types
```

### 5.4 batch.py Integration

```python
# Before (batch.py uses InferencePipeline or InferenceOrchestrator):
orchestrator = InferenceOrchestrator.from_experiment(config)
df = orchestrator.predict_batch(data, batch_size=10000)

# After:
pipeline = UniversalInferencePipeline.from_experiment(config)
df = pipeline.predict_batch(data, batch_size=10000)
```

### 5.5 Colab Notebook Integration

```python
# New inference demo cell:
from src.inference import UniversalInferencePipeline

pipeline = UniversalInferencePipeline.from_experiment(config)

# Works for ANY model type (boosting, neural, transformer)
result = pipeline.predict_from_raw(new_ohlcv_data)
print(f"Predictions: {result.predictions.class_predictions}")
print(f"Confidence: {result.predictions.confidence.mean():.3f}")
```

---

## 6. Error Handling

### 6.1 Contract Mismatch Errors

```python
class InferenceShapeMismatchError(ValueError):
    """Raised when input shape doesn't match model contract."""

    def __init__(self, model_name, expected_rank, actual_shape, contract):
        self.model_name = model_name
        self.expected_rank = expected_rank
        self.actual_shape = actual_shape
        self.contract = contract
        super().__init__(
            f"Model '{model_name}' expects {expected_rank.name} input "
            f"(rank {expected_rank.value}), but received shape {actual_shape}. "
            f"Contract: adapter_id='{contract.adapter_id}', "
            f"sequence_length={contract.sequence_length}, "
            f"min_features={contract.min_features}."
        )
```

### 6.2 Specific Error Scenarios

| Scenario | Detection Point | Error / Recovery |
|----------|----------------|-----------------|
| DataFrame passed to 4D model | `_adapt_input()` | MultiStreamAdapter transforms it; if `additional_dfs` missing for MTF, raise `ValueError` with instruction to provide MTF data |
| ndarray wrong rank (e.g., 2D for LSTM) | `_predict_single()` shape check | Raise `InferenceShapeMismatchError` with expected vs actual |
| Feature columns missing from DataFrame | `_adapt_input()` column check | Raise `ValueError` listing missing columns (first 10) |
| Scaler feature count mismatch | `_apply_scaling()` | Raise `ValueError` with scaler vs input feature counts |
| PreprocessingGraph not available for raw input | `predict_from_raw()` | Raise `RuntimeError` suggesting `predict()` with pre-computed features |
| No bundles loaded | Any predict method | Raise `RuntimeError` listing available constructors |
| Adapter not registered for model | `_get_adapter_for_model()` | Raise `KeyError` with available adapter IDs |
| Sequence too short for adapter | `SequenceAdapter.transform()` | Raise `ValueError` with min required rows |

### 6.3 Warning Scenarios (non-fatal)

| Scenario | Action |
|----------|--------|
| Feature hash mismatch (training vs inference) | `logger.warning()` — features may have drifted |
| Calibrator expected but missing | `logger.warning()` — return uncalibrated |
| Different horizons across bundles in ensemble | `logger.warning()` — predictions may not be comparable |

---

## 7. Backward Compatibility

### 7.1 Existing `predict()` Continues to Work

`ModelBundle.predict(X)` is unchanged. Users who already pass pre-shaped tensors (Mode 3) continue working exactly as before. `UniversalInferencePipeline` is an opt-in improvement.

### 7.2 Existing `predict_from_raw()` Continues to Work

`ModelBundle.predict_from_raw(raw_df)` still works for tabular models (boosting). It produces 2D features and predicts — same as today. For neural/transformer models, it will still fail as it does today, but the new `UniversalInferencePipeline.predict_from_raw()` provides the working path.

### 7.3 Migration Timeline

```
Phase 2a (Build):
  - Add UniversalInferencePipeline in new file
  - No existing code modified
  - Both old and new paths work

Phase 2b (Integrate):
  - Update server.py, batch.py, notebook to use UniversalInferencePipeline
  - Add deprecation warnings to InferencePipeline and InferenceOrchestrator
  - Old code still works with warnings

Phase 2c (Cleanup):
  - Remove InferencePipeline (pipeline.py)
  - Remove InferenceOrchestrator (orchestrator.py)
  - Update __init__.py exports
```

### 7.4 Import Compatibility

```python
# Temporary re-exports during transition
# src/inference/__init__.py additions:

# Deprecated aliases (Phase 2b)
from src.inference.universal_pipeline import UniversalInferencePipeline

# Keep old imports working with deprecation warning
# InferencePipeline = _deprecated_alias(UniversalInferencePipeline, "InferencePipeline")
# InferenceOrchestrator = _deprecated_alias(UniversalInferencePipeline, "InferenceOrchestrator")
```

---

## 8. File Layout

### New Files

```
src/inference/
├── universal_pipeline.py    # NEW: UniversalInferencePipeline class
├── errors.py                # NEW: InferenceShapeMismatchError, etc.
├── pipeline.py              # DEPRECATED in Phase 2b, REMOVED in Phase 2c
├── orchestrator.py          # DEPRECATED in Phase 2b, REMOVED in Phase 2c
├── bundle.py                # UNCHANGED
├── ensemble_bundle.py       # UNCHANGED
├── preprocessing_graph.py   # UNCHANGED
├── builder.py               # UNCHANGED
├── batch.py                 # UPDATED to use UniversalInferencePipeline
├── server.py                # UPDATED to use UniversalInferencePipeline
└── __init__.py              # UPDATED: add UniversalInferencePipeline export
```

### Estimated Scope

| File | Lines (est.) | Complexity |
|------|-------------|------------|
| `universal_pipeline.py` | ~450 | Medium — mostly orchestration wiring |
| `errors.py` | ~40 | Low — custom exception classes |
| `__init__.py` changes | ~10 | Low — add exports |
| `batch.py` changes | ~20 | Low — swap pipeline class |
| `server.py` changes | ~20 | Low — swap pipeline class |

---

## 9. Key Design Invariants

1. **PreprocessingGraph always outputs 2D** — Feature engineering is tabular. Shape transformation happens after.
2. **Contract drives adapter selection** — `get_model_contract(model_name).adapter_id` determines which adapter class is used. Same routing as training.
3. **Exactly one scaling step** — `ScalingSource` enum prevents double-scaling. Default is `BUNDLE` (bundle's scaler applies after adapter transformation).
4. **Pre-shaped input bypasses adapter** — If caller passes ndarray, trust they shaped it correctly. Only validate dimensions match metadata.
5. **Adapter instances are cached** — One adapter per model, reused across predictions.
6. **Pipeline does NOT call `bundle.predict()`** — It calls `bundle.model.predict()` directly to control scaling timing. This is critical for correct 3D/4D scaling.

---

## 10. Open Questions for Implementation

| # | Question | Recommendation |
|---|----------|---------------|
| 1 | Should `predict()` auto-detect Mode 2 vs Mode 3 based on input type? | Yes — DataFrame = Mode 2, ndarray = Mode 3. Simple, unambiguous. |
| 2 | For 4D models in Mode 1, should we auto-generate MTF data from 1min? | Yes, if PreprocessingGraph has MTF config and `additional_dfs` not provided. Use MTFFeatureGenerator. |
| 3 | Should adapter transform handle scaling internally? | No — keep adapter as pure shape transformation. Pipeline owns scaling. |
| 4 | Should we support mixed-rank ensemble predictions? | Yes — pipeline adapts each model independently, then stacks probabilities for meta-learner. |
| 5 | PredictionResult duplication (core.interfaces vs models.base)? | Use `src.core.interfaces.PredictionResult` as canonical. Verify they're the same or consolidate. |

---

## 11. Dependency Graph

```
UniversalInferencePipeline
    ├── ModelBundle (src/inference/bundle.py)
    │     ├── BaseModel (src/models/base.py)
    │     ├── Scaler (sklearn)
    │     └── Calibrator (pickle)
    ├── EnsembleBundle (src/inference/ensemble_bundle.py)
    ├── PreprocessingGraph (src/inference/preprocessing_graph.py)
    ├── ModelContract (src/core/contracts/model_contract.py)
    │     └── get_model_contract()
    ├── AdapterRegistry (src/data/adapters/registry.py)
    │     ├── TabularAdapter
    │     ├── SequenceAdapter
    │     └── MultiStreamAdapter
    ├── DataRank (src/core/types.py)
    ├── PredictionResult (src/core/interfaces.py)
    └── PipelineConfig (src/core.py) [optional]
```

No new external dependencies. All imports are from existing `src/` modules.
