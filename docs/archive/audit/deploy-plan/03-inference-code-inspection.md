# 03 - Inference Code Inspection: Bundle, Builder, PreprocessingGraph, Ensemble

**Date:** 2026-02-15
**Agent:** 3/10 (Inference-Side Code Paths)
**Purpose:** Surgical inspection of every inference-side file, documenting exact API surfaces, data flows, and the precise gaps that block `predict_from_raw()` for 10/12 model types (8 needing 3D + 2 needing 4D).

---

## 1. Files Inspected

| File | Path | Lines | Status |
|------|------|-------|--------|
| ModelBundle | `/home/jake/Desktop/Research/src/inference/bundle.py` | 1107 | Exists |
| EnsembleBundle | `/home/jake/Desktop/Research/src/inference/ensemble_bundle.py` | 922 | Exists |
| BundleBuilder | `/home/jake/Desktop/Research/src/inference/builder.py` | 784 | Exists |
| PreprocessingGraph | `/home/jake/Desktop/Research/src/inference/preprocessing_graph.py` | 885 | Exists |
| `__init__.py` | `/home/jake/Desktop/Research/src/inference/__init__.py` | 195 | Exists |

---

## 2. BundleMetadata Schema (bundle.py L70-139)

### Fields (L74-92)

| Field | Type | Default | Source |
|-------|------|---------|--------|
| `version` | `str` | *required* | `BUNDLE_VERSION = "1.2.0"` (L54) |
| `created_at` | `str` | *required* | `datetime.now().isoformat()` |
| `model_name` | `str` | *required* | `model._get_model_type()` (L285) |
| `model_family` | `str` | `"unknown"` | `model.model_family` (L286) |
| `horizon` | `int` | *required* | Passed in |
| `n_features` | `int` | *required* | `len(feature_columns)` or 5/0 for 4D (L295-311) |
| `feature_hash` | `str` | *required* | MD5 of comma-joined feature columns (L314) |
| `requires_sequences` | `bool` | `False` | `model.requires_sequences` (L287) |
| `requires_4d` | `bool` | `False` | `model.requires_4d` (L288) |
| `sequence_length` | `int` | `0` | From `model._config.get("sequence_length", 60)` (L291) |
| `n_timeframes` | `int` | `0` | `1 + len(contract.mtf_timeframes)` for 4D (L302) |
| `has_calibrator` | `bool` | `False` | `calibrator is not None` |
| `has_preprocessing_graph` | `bool` | `False` | `preprocessing_graph is not None` |
| `preprocessing_graph_hash` | `str` | `""` | From graph config hash |
| `has_feature_spec` | `bool` | `False` | `feature_spec is not None` |
| `feature_spec_hash` | `str` | `""` | From spec schema hash |
| `symbol` | `str` | `""` | Passed in |
| `training_metrics` | `dict[str, Any]` | `{}` | Passed in |
| `extra` | `dict[str, Any]` | `{}` | Passed in |

### Backward Compatibility

`from_dict()` (L118-139) uses `.get()` with safe defaults for all optional fields. Fields `version`, `created_at`, `model_name`, `horizon`, `n_features`, `feature_hash` are required (no `.get()` fallback). `model_family` uses `.get()` with `"unknown"` default (L123).

### Current Version

**`BUNDLE_VERSION = "1.2.0"` at L54.** Plans specify bump to `"1.3.0"` but this has NOT been done yet.

---

## 3. ModelBundle Public API Surface (bundle.py L171-1097)

### Constructor (L219-247)

```python
ModelBundle(
    model: BaseModel,
    scaler: RobustScaler | StandardScaler | None,
    feature_columns: list[str],
    metadata: BundleMetadata,
    calibrator: Any | None = None,
    preprocessing_graph: Any | None = None,
    feature_spec: Any | None = None,
)
```

### Class Methods

| Method | Signature | Line |
|--------|-----------|------|
| `from_training()` | `(model, scaler, feature_columns, horizon, calibrator=None, preprocessing_graph=None, feature_spec=None, symbol="", training_metrics=None, extra_metadata=None) -> ModelBundle` | L249-360 |
| `load()` | `(path: str|Path) -> ModelBundle` | L599-699 |
| `extract_bundle()` | `(tarball_path, extract_dir=None) -> Path` | L531-597 |

### Instance Methods

| Method | Signature | Returns | Line |
|--------|-----------|---------|------|
| `predict()` | `(X: DataFrame|ndarray, calibrate=True) -> PredictionResult` | PredictionResult | L701-761 |
| `predict_from_raw()` | `(raw_df: DataFrame, calibrate=True, skip_cleaning=False) -> PredictionResult` | PredictionResult | L1056-1077 |
| `preprocess()` | `(raw_df: DataFrame, skip_cleaning=False) -> DataFrame` | DataFrame (2D) | L1008-1054 |
| `save()` | `(path, overwrite=False) -> Path` | Path | L362-457 |
| `package_bundle()` | `(bundle_dir, output_path=None, compression="gz") -> Path` | Path | L459-529 |
| `validate()` | `() -> dict[str, Any]` | dict | L921-965 |
| `validate_distribution()` | `(X_current, method="ks", threshold=0.05) -> tuple[bool, list[str]]` | (bool, list) | L832-919 |
| `set_preprocessing_graph()` | `(graph) -> None` | None | L985-1006 |
| `set_feature_spec()` | `(feature_spec) -> None` | None | L967-983 |

---

## 4. ModelBundle.predict() -- Exact Flow (L701-761)

### Step-by-step:

1. **`_prepare_input(X)`** (L717, calls L763-817):
   - If `X` is DataFrame:
     - **4D models: RAISES ValueError** (L769-775) -- "4D models require ndarray input"
     - Validates feature columns exist, reorders to `self.feature_columns` (L778-782)
     - Converts to values array (L782)
   - Converts to `np.float32` (L784)
   - **Shape validation** (L787-815):
     - 4D: checks ndim==4, n_timeframes, sequence_length, n_features
     - 3D: checks ndim==3, n_features at axis 2
     - 2D: checks ndim==2, n_features at axis 1

2. **Scaling** (L720-752):
   - If `self.scaler is not None`:
     - **4D path** (L721-732): Reshape `(n, tf, seq, feat)` -> `(-1, feat)`, transform, reshape back
     - **3D path** (L733-744): Reshape `(n, seq, feat)` -> `(-1, feat)`, transform, reshape back
     - **2D path** (L745-752): Direct `scaler.transform(X_array)`
   - Validates scaler feature count matches input feature count

3. **Model prediction** (L755): `output = self.model.predict(X_array)`

4. **Calibration** (L758-759): If `calibrate=True` and `self.calibrator is not None`, calls `_apply_calibration(output)` (L819-830)

### Key observations:
- `predict()` ALREADY handles 2D, 3D, and 4D inputs correctly **if they arrive pre-shaped**
- The scaler handles reshape/transform/reshape for all ranks
- The gap is NOT in `predict()` -- it is in getting the data to the right shape before `predict()` is called

---

## 5. ModelBundle.predict_from_raw() -- Exact Flow (L1056-1077)

```python
def predict_from_raw(self, raw_df, calibrate=True, skip_cleaning=False):
    features = self.preprocess(raw_df, skip_cleaning=skip_cleaning)  # L1076
    return self.predict(features, calibrate=calibrate)                # L1077
```

### preprocess() flow (L1008-1054):

1. **Checks preprocessing_graph exists** (L1031-1035) -- raises RuntimeError if None
2. **Calls `self.preprocessing_graph.transform(raw_df, skip_cleaning, skip_scaling=False)`** (L1038-1042)
   - **CRITICAL: `skip_scaling=False` is hardcoded** -- the preprocessing graph WILL scale the data
   - This contradicts the plan which specifies `skip_scaling=True` here to avoid double-scaling
   - The bundle's own scaler will then scale again in `predict()` (L720-752)
   - **THIS IS A DOUBLE-SCALING BUG** (for models that have both a preprocessing graph scaler AND a bundle scaler)
3. **Filters to `self.feature_columns`** (L1045-1054) -- keeps only columns present in both
4. **Returns a 2D DataFrame** (always)

### WHY predict_from_raw() FAILS for neural/transformer models:

The complete chain is:
```
raw_df (OHLCV)
  -> self.preprocess(raw_df)
    -> self.preprocessing_graph.transform(raw_df)
    -> returns 2D DataFrame
  -> self.predict(features_2d_dataframe)
    -> _prepare_input(X)
      -> if DataFrame and requires_4d: RAISES ValueError (L769-775)
      -> if DataFrame: converts to 2D array via X[columns].values (L782)
    -> shape validation:
      -> if requires_sequences and ndim != 3: RAISES ValueError (L803-804)
      -> if requires_4d and ndim != 4: RAISES ValueError (L788-789)
```

**There are TWO failure points:**

| Failure | Location | Trigger | Models Affected |
|---------|----------|---------|----------------|
| **4D DataFrame rejection** | `_prepare_input()` L769-775 | `requires_4d=True` and input is DataFrame | PatchTST, iTransformer |
| **3D shape validation** | `_prepare_input()` L803-804 | `requires_sequences=True` and ndim != 3 | LSTM, GRU, TCN, InceptionTime, ResNet1D, TFT, N-BEATS, Transformer |

**The missing link is clear:** Between `preprocess()` returning a 2D DataFrame and `predict()` expecting 3D/4D ndarray, there is NO adapter routing step. The code needs to:

1. Take the 2D feature DataFrame from `preprocess()`
2. Look at `self.metadata.requires_sequences` and `self.metadata.requires_4d`
3. Route through the appropriate adapter:
   - `requires_4d=True` -> MultiStreamAdapter -> 4D ndarray `(n, tf, seq, feat)`
   - `requires_sequences=True` -> SequenceAdapter -> 3D ndarray `(n, seq, feat)`
   - Neither -> pass through as 2D

---

## 6. PreprocessingGraph -- Exact Analysis (preprocessing_graph.py)

### Public API

| Method | Signature | Returns | Line |
|--------|-----------|---------|------|
| `__init__()` | `(config: PreprocessingGraphConfig)` | None | L282-291 |
| `from_pipeline_config()` | `(pipeline_config, feature_columns=None, scaling_params=None, symbol="", horizon=20) -> PreprocessingGraph` | PreprocessingGraph | L293-392 |
| `from_training_run()` | `(run_path: Path) -> PreprocessingGraph` | PreprocessingGraph | L394-424 |
| `transform()` | `(raw_df, skip_cleaning=False, skip_scaling=False) -> DataFrame` | DataFrame (2D) | L451-517 |
| `set_scaler()` | `(scaler) -> None` | None | L441-449 |
| `save()` | `(path: Path) -> None` | None | L748-769 |
| `load()` | `(path: Path) -> PreprocessingGraph` | PreprocessingGraph | L771-809 |
| `validate()` | `() -> dict[str, Any]` | dict | L821-858 |

### transform() flow (L451-517)

1. **Validate input** (L481): Checks for OHLCV columns + datetime
2. **Cleaning/Resampling** (L486-487): Resamples from `source_timeframe` to `target_timeframe` (e.g., 1min -> 5min)
3. **Feature engineering** (L490): Calls `_apply_features()` (L572-680) -- applies ALL indicators (SMA, EMA, RSI, MACD, ATR, wavelets, etc.)
4. **MTF features** (L493-494): Calls `_apply_mtf()` (L682-700) -- generates multi-timeframe features
5. **Regime detection** (L497-498): Currently a no-op (L702-706) -- regime features already in step 3
6. **NaN drop** (L501)
7. **Scaling** (L503-505): If `skip_scaling=False` AND scaler exists, applies scaling
8. **Column selection** (L507-516): If `feature_columns` configured, select only those

### CRITICAL: transform() ALWAYS returns a 2D DataFrame

The output is a flat DataFrame with one row per timestep, containing all engineered features as columns. There is NO windowing, NO sequence creation, NO multi-resolution stacking. This is the core gap.

### What 3D models need that transform() does NOT provide:

| Requirement | What SequenceAdapter does | What transform() does |
|-------------|--------------------------|----------------------|
| Sliding window | Creates `(n_samples, seq_len, n_features)` windows | Returns flat `(n_rows, n_features)` DataFrame |
| Label alignment | Aligns labels to last timestep of each window | No label handling |
| Sequence length | Uses `contract.sequence_length` (e.g., 60) | Not applicable |

### What 4D models need that transform() does NOT provide:

| Requirement | What MultiStreamAdapter does | What transform() does |
|-------------|------------------------------|----------------------|
| Multi-resolution stacking | Creates `(n, n_timeframes, seq_len, n_features)` | Returns single-timeframe 2D DataFrame |
| Per-timeframe resampling | Resamples to 1min, 5min, 15min independently | Resamples to single target_timeframe |
| Raw OHLCV per stream | Uses raw OHLCV (4-5 features) per timeframe | Generates 100+ engineered features |

### Hardcoded assumptions in _create_preprocessing_graph() (builder.py L512-555):

| Assumption | Value | Location |
|------------|-------|----------|
| `source_timeframe` | `"1min"` | builder.py L527 |
| `target_timeframe` | `"5min"` | builder.py L528 |
| `base_timeframe` | `"5min"` | builder.py L531 |
| `scaler_type` | `"robust"` | builder.py L547 |
| `enable_mtf` | `True` | builder.py L535 |
| `enable_wavelets` | `True` | builder.py L540 |
| `regime.enabled` | `True` | builder.py L543 |

These are reasonable defaults for boosting models but wrong for 4D models (PatchTST/iTransformer) which need raw OHLCV at multiple timeframes, not engineered features.

---

## 7. Double-Scaling Bug (CONFIRMED)

### Current flow in predict_from_raw():

```
raw_df
  -> preprocess() calls transform(skip_scaling=False)     # L1041: scaling APPLIED
    -> PreprocessingGraph._apply_scaling() runs            # L503-505
  -> predict(features)
    -> scaler.transform(X_array)                           # L720-752: scaling APPLIED AGAIN
```

**Both the preprocessing graph scaler AND the bundle scaler apply.** This is the double-scaling bug documented in the plans.

### Where the fix goes:

In `preprocess()` at L1038-1042, change:
```python
# CURRENT (L1041):
skip_scaling=False,
# SHOULD BE:
skip_scaling=True,
```

This ensures the PreprocessingGraph generates features WITHOUT scaling, and the bundle's own scaler in `predict()` applies scaling exactly once.

---

## 8. EnsembleBundle -- Full Analysis (ensemble_bundle.py)

### EnsembleBundleMetadata (L81-130)

| Field | Type | Default |
|-------|------|---------|
| `version` | `str` | Required |
| `created_at` | `str` | Required |
| `meta_learner_name` | `str` | Required |
| `base_model_names` | `list[str]` | Required |
| `horizon` | `int` | Required |
| `n_base_models` | `int` | Required |
| `n_stacking_features` | `int` | Required |
| `symbol` | `str` | `""` |
| `coverage` | `float` | `1.0` |
| `alignment_offset` | `int` | `0` |
| `metrics` | `dict[str, float]` | `{}` |
| `extra` | `dict[str, Any]` | `{}` |

### What EnsembleBundle stores:
- `meta_learner` -- the trained meta-learner model (L246)
- `base_bundle_paths` -- list of Path objects to base ModelBundle directories (L248)
- `stacking_feature_names` -- ordered list of stacking feature names (L249)
- `scaler` -- optional scaler for stacking features (L250)
- `alignment_config` -- AlignmentConfig for OOF alignment (L251)
- `_base_bundles` -- lazy-loaded dict of model_name -> ModelBundle (L254)

### Public API

| Method | Signature | Returns | Line |
|--------|-----------|---------|------|
| `from_ensemble_result()` | `(ensemble_result, base_bundle_paths=None, config=None, scaler=None) -> EnsembleBundle` | EnsembleBundle | L256-349 |
| `from_orchestrator()` | `(orchestrator, base_bundle_paths=None, scaler=None) -> EnsembleBundle` | EnsembleBundle | L351-392 |
| `predict()` | `(base_predictions: dict[str, ndarray], calibrate=True) -> Any` | PredictionResult | L596-628 |
| `predict_proba()` | `(base_predictions) -> ndarray` | ndarray | L630-645 |
| `predict_classes()` | `(base_predictions) -> ndarray` | ndarray | L647-662 |
| `predict_from_base_features()` | `(X: DataFrame|ndarray, calibrate=True) -> Any` | PredictionResult | L664-698 |
| `save()` | `(path, overwrite=False) -> Path` | Path | L394-496 |
| `load()` | `(path) -> EnsembleBundle` | EnsembleBundle | L498-594 |
| `validate()` | `() -> dict[str, Any]` | dict | L827-860 |
| `summary()` | `() -> str` | str | L862-901 |

### predict() flow (L596-628):
1. Check meta_learner loaded (L615-616)
2. `_stack_predictions(base_predictions)` (L619) -- stacks base model probabilities into 2D meta-features
3. Apply scaler if present (L622-623)
4. `self.meta_learner.predict(stacking_X)` (L626)

### predict_from_base_features() flow (L664-698):
1. `_ensure_base_bundles_loaded()` (L685) -- lazy loads base ModelBundles from `self.base_bundle_paths`
2. For each base bundle: `bundle.predict(X, calibrate=False)` (L694) -- **passes same X to ALL base models**
3. Collects `output.class_probabilities` per model (L695)
4. Calls `self.predict(base_predictions)` (L698)

### CRITICAL GAPS in EnsembleBundle:

**GAP 1: No `predict_from_raw()` method exists** (confirmed)

The EnsembleBundle has NO method that accepts raw OHLCV and produces predictions. It has:
- `predict()` -- requires pre-stacked base predictions
- `predict_from_base_features()` -- requires pre-shaped features

Neither accepts raw OHLCV.

**GAP 2: `predict_from_base_features()` passes SAME input to ALL base models** (L694)

This is wrong for heterogeneous ensembles. If the ensemble has both XGBoost (2D) and LSTM (3D) base models, passing the same `X` to both will fail because:
- XGBoost expects `(n, feat)` 2D
- LSTM expects `(n, seq, feat)` 3D

The fix requires calling `base_bundle.predict_from_raw(raw_df)` instead of `base_bundle.predict(X)`, but that requires `predict_from_raw()` to work for all model types first.

**GAP 3: Base bundle paths are absolute** (L447)

```python
"paths": [str(p) for p in self.base_bundle_paths],  # L447
```

On `load()` (L543):
```python
base_bundle_paths = [Path(p) for p in json.load(f).get("paths", [])]  # L543
```

These are paths stored as raw strings (relative by default). If absolute paths are used and the bundle is moved to a different machine or directory, the paths break. The `validate()` method (L851) correctly checks for missing bundles but cannot fix the paths.

**GAP 4: Meta-learner loading has fragile fallback** (L552-571)

Load path tries:
1. `meta_dir / "model.pkl"` via pickle (L556-561)
2. `get_meta_learner(name)` + `meta_learner.load(meta_dir)` (L564-569)
3. Falls through to `meta_learner = None` (implicit)

If `src.models.ensemble.get_meta_learner` import fails, the meta-learner is silently None.

---

## 9. BundleBuilder -- Full Analysis (builder.py)

### Public API

| Method | Signature | Returns | Line |
|--------|-----------|---------|------|
| `__init__()` | `(config: PipelineConfig)` | None | L182-195 |
| `from_config()` | `(config) -> BundleBuilder` | BundleBuilder | L197-208 |
| `from_training_run()` | `(run_path) -> BundleBuilder` | BundleBuilder | L210-236 |
| `build_from_training_result()` | `(training_result, include_preprocessing_graph=True, include_calibrator=True, feature_specs=None) -> BundleBuildResult` | BundleBuildResult | L238-368 |
| `build_ensemble_bundle()` | `(ensemble_result, base_bundles=None) -> Path` | Path | L370-437 |
| `build_all()` | `(training_result=None, ensemble_result=None, include_preprocessing_graph=True, feature_specs=None) -> BundleBuildResult` | BundleBuildResult | L439-510 |
| `validate_bundles()` | `() -> dict[str, Any]` | dict | L665-711 |

### Duck-Typing Extraction Chains

**Model extraction** (`_extract_model` L557-580):
```
trainer.model -> trainer._model -> trainer.estimator -> trainer._estimator -> trainer.get_model()
```

**Scaler extraction** (`_extract_scaler` L582-596):
```
trainer.scaler -> trainer._scaler -> trainer.feature_scaler -> trainer._feature_scaler
```

**Feature columns extraction** (`_extract_feature_columns` L598-628):
```
trainer.feature_columns -> trainer._feature_columns -> trainer.feature_names -> trainer._feature_names
  -> scaler.feature_names_in_
  -> fallback: [f"f0", f"f1", ...]
```

**Calibrator extraction** (`_extract_calibrator` L630-644):
```
trainer.calibrator -> trainer._calibrator -> trainer.prob_calibrator
```
**ALWAYS returns None** because calibrator is set on service result, not trainer (see Agent 2 findings).

### build_from_training_result() flow (L238-368):

1. Create `PreprocessingGraph` if `include_preprocessing_graph` (L279-281) via `_create_preprocessing_graph()` (L512-555)
2. For each `(key, model_result)` in `training_result.model_results` (L284):
   a. Get trainer from `model_result.trainer` (L291)
   b. Extract model via duck-typing (L298)
   c. Extract scaler via duck-typing (L305)
   d. Extract feature columns via duck-typing (L308)
   e. Extract calibrator via duck-typing (L313) -- **always None** (calibrator works via Trainer attribute but lost in orchestrator's ModelTrainingResult conversion path)
   f. Get feature_spec from `feature_specs` dict if provided (L317-318)
   g. Create `ModelBundle.from_training(...)` (L322-337)
   h. Save to `bundles_dir / "{model_name}_h{horizon}"` (L340-341)
3. Return `BundleBuildResult`

### build_ensemble_bundle() flow (L370-437):

**DOES NOT CREATE an EnsembleBundle object.** Instead saves raw files:
- `ensemble_metadata.json` (L398-413) -- custom format, NOT `EnsembleBundleMetadata`
- `stacking_dataset.parquet` (L416-422) -- the raw stacking dataset
- `aligned_oof_info.json` (L425-433) -- alignment info

**DOES NOT:**
- Save the meta-learner model
- Create a loadable `EnsembleBundle`
- Use `EnsembleBundle.from_ensemble_result()`
- Use `EnsembleBundle.save()`

**Result:** The ensemble directory created by `build_ensemble_bundle()` CANNOT be loaded by `EnsembleBundle.load()` because:
- File names differ (`ensemble_metadata.json` vs `metadata.json`)
- No `manifest.json`
- No `meta_learner/` directory
- No `stacking_features.json`
- No `base_bundles.json`
- No `alignment_config.json`

### Where PreprocessingGraph is called:

Only in `_create_preprocessing_graph()` (L512-555), which is called from `build_from_training_result()` at L281. The graph is created from hardcoded config values (see Section 6 above), NOT from the actual pipeline config that was used during training. This means the preprocessing graph may not exactly match training preprocessing.

---

## 10. Package/Extract -- tar.gz Bundling (bundle.py)

### package_bundle() (L459-529)
- Validates bundle directory exists and has `manifest.json` (L498-507)
- Creates tarball with `tar.add(bundle_dir, arcname=bundle_dir.name)` (L524)
- Compression options: `gz` (default), `bz2`, `xz`, `""` (none)
- Returns Path to created tarball

### extract_bundle() (L531-597)
- **Path traversal protection** (L579-584):
  ```python
  for member in tar.getmembers():
      if member.name.startswith("/") or ".." in member.name:
          raise ValueError("Unsafe path in tarball...")
  ```
- Uses `tar.extractall()` (L585) -- still has deprecation warnings in Python 3.12+
- Finds bundle name from first member's top-level directory (L590-591)

### Security note:
- Path traversal check exists but is basic (only checks `/` prefix and `..`)
- Does not use `filter` parameter on `extractall()` (Python 3.12 recommendation)
- Pickle loading throughout (scaler L644, calibrator L653, meta-learner L561) has no validation

---

## 11. Inference __init__.py Exports (L1-195)

### All exports organized by module:

| Module | Exports |
|--------|---------|
| `batch` | BatchInference, BatchInferenceResult, BatchPredictor, BatchProgress, BatchResult, ModelPrediction, run_batch_inference |
| `builder` | BundleBuilder, BundleBuildResult, build_bundles, build_from_run |
| `bundle` | BUNDLE_FEATURE_SPEC_FILE, BUNDLE_PREPROCESSING_GRAPH_FILE, BUNDLE_VERSION, BundleManifest, BundleMetadata, ModelBundle |
| `ensemble_bundle` | ENSEMBLE_BUNDLE_VERSION, AlignmentConfig, EnsembleBundle, EnsembleBundleManifest, EnsembleBundleMetadata |
| `orchestrator` | InferenceOrchestrator, PredictionResult, load_inference, predict_batch_from_bundle, predict_from_bundle |
| `pipeline` | EnsembleResult, InferencePipeline, InferenceResult |
| `preprocessing_graph` | All config classes + PreprocessingGraph + PREPROCESSING_GRAPH_VERSION + PREPROCESSING_GRAPH_FILE |
| `server` | ModelServer, ServerConfig, start_server |

### Note:
- The `orchestrator` and `pipeline` modules are imported (L111-122) but NOT inspected by this agent. They may have their own `predict_from_raw()` implementations.
- `PredictionResult` is re-exported from `orchestrator` (L113) -- this may shadow or conflict with `PredictionResult` from `src.core.interfaces` (used by `BaseModel`).

---

## 12. Gap Map: What Must Change for Universal predict_from_raw()

### The Missing Link (Visual)

```
Current flow (WORKS for tabular, 4/12 models — 3 boosting + MLP):

  raw_df -> PreprocessingGraph.transform() -> 2D DataFrame -> predict() -> PredictionResult
                                                    |
                                              [2D validation passes]


Current flow (FAILS for neural, 8/12 models needing 3D):

  raw_df -> PreprocessingGraph.transform() -> 2D DataFrame -> predict()
                                                    |
                                              [3D validation FAILS at L803]


Current flow (FAILS for transformer 4D, 2/12 models):

  raw_df -> PreprocessingGraph.transform() -> 2D DataFrame -> predict()
                                                    |
                                              [4D rejection at L769 OR 4D validation FAILS at L788]


Required flow (ALL 12 models):

  raw_df -> PreprocessingGraph.transform(skip_scaling=True) -> 2D DataFrame
         -> _adapt_for_model()                                             [NEW]
            if requires_4d:    -> MultiStreamAdapter -> 4D ndarray
            if requires_seq:   -> SequenceAdapter    -> 3D ndarray
            else:              -> pass through       -> 2D ndarray
         -> predict(adapted_input) -> PredictionResult
```

### Exact Changes Required Per File

#### bundle.py -- ModelBundle

| Change | Location | What | Why |
|--------|----------|------|-----|
| **Fix double-scaling** | L1041 | Change `skip_scaling=False` to `skip_scaling=True` | Prevents double scaling (graph + bundle scaler) |
| **Add adapter routing** | New method between L1054 and L1056 | `_adapt_for_model(features_2d: DataFrame) -> ndarray` | Routes 2D features through correct adapter based on `self.metadata.requires_sequences` / `self.metadata.requires_4d` |
| **Update predict_from_raw()** | L1076-1077 | Insert adapter routing between preprocess() and predict() | Connects the 2D output to model-appropriate shape |

The new `predict_from_raw()` should be approximately:
```python
def predict_from_raw(self, raw_df, calibrate=True, skip_cleaning=False):
    features = self.preprocess(raw_df, skip_cleaning=skip_cleaning)  # 2D DataFrame
    adapted = self._adapt_for_model(features)                         # 2D/3D/4D ndarray
    return self.predict(adapted, calibrate=calibrate)
```

Where `_adapt_for_model()` must:
1. Use `self.metadata.requires_4d` and `self.metadata.requires_sequences` to determine path
2. For 3D: Create sliding windows of `self.metadata.sequence_length` from the 2D features
3. For 4D: This is more complex -- needs raw OHLCV at multiple timeframes, which `preprocess()` does NOT provide. The 4D path may need a completely different preprocessing chain.

#### 4D Models -- Special Case

For PatchTST and iTransformer, the entire `preprocess()` path is wrong because:
- These models want **raw OHLCV** (4-5 features) at **multiple timeframes** (1min, 5min, 15min)
- `PreprocessingGraph.transform()` outputs **100+ engineered features** at a **single timeframe**
- The adapter routing alone is insufficient -- a different preprocessing path is needed

Options (for implementation agents):
1. **Option A**: Add a `_preprocess_4d(raw_df)` method that bypasses PreprocessingGraph entirely, resamples raw data to each timeframe, and builds 4D tensors directly
2. **Option B**: Modify PreprocessingGraph to support a `mode="raw_multi_stream"` that outputs per-timeframe OHLCV instead of engineered features
3. **Option C**: Store the raw multi-timeframe data alongside the engineered features, let the adapter select the right path

#### ensemble_bundle.py -- EnsembleBundle

| Change | Location | What | Why |
|--------|----------|------|-----|
| **Add predict_from_raw()** | After L698 | New method that loads base bundles and calls `bundle.predict_from_raw(raw_df)` on each | Required by InferenceBundle protocol |
| **Fix path storage** | L447 | Ensure relative paths (relative to ensemble bundle dir); paths stored as raw strings, relative by default | Portability |
| **Fix predict_from_base_features()** | L694 | Cannot pass same X to heterogeneous models | Different models need different input shapes |

#### builder.py -- BundleBuilder

| Change | Location | What | Why |
|--------|----------|------|-----|
| **Fix build_ensemble_bundle()** | L370-437 | Use `EnsembleBundle.from_ensemble_result()` + `.save()` instead of custom file layout | Creates loadable ensemble bundles |
| **Fix calibrator extraction** | L630-644 | Also check `model_result.calibrator` (once calibrator transfer is fixed in orchestrator) | Calibrators actually get into bundles |
| **Store model contract info** | L322-337 | Pass contract's `sequence_length`, `mtf_timeframes`, `feature_mode` to BundleMetadata | Adapter routing needs this info at inference time |

---

## 13. Model-by-Model Routing Table

Based on `MODEL_CONTRACTS` in `/home/jake/Desktop/Research/src/core/contracts/model_contract.py`:

| Model | DataRank | adapter_id | sequence_length | mtf_timeframes | feature_mode | predict_from_raw status |
|-------|----------|-----------|-----------------|----------------|-------------|------------------------|
| xgboost | TABULAR_2D | tabular | N/A | N/A | ENGINEERED | WORKS |
| lightgbm | TABULAR_2D | tabular | N/A | N/A | ENGINEERED | WORKS |
| catboost | TABULAR_2D | tabular | N/A | N/A | ENGINEERED | WORKS |
| lstm | SEQUENCE_3D | sequence | 60 | N/A | ENGINEERED | BROKEN -- needs 3D windowing |
| gru | SEQUENCE_3D | sequence | 60 | N/A | ENGINEERED | BROKEN -- needs 3D windowing |
| tcn | SEQUENCE_3D | sequence | 120 | N/A | ENGINEERED | BROKEN -- needs 3D windowing |
| transformer | SEQUENCE_3D | sequence | 128 | N/A | HYBRID | BROKEN -- needs 3D windowing |
| tft | SEQUENCE_3D | sequence | 60 | N/A | HYBRID | BROKEN -- needs 3D windowing |
| nbeats | SEQUENCE_3D | sequence | 60 | N/A | RAW | BROKEN -- needs 3D windowing + different features |
| inceptiontime | SEQUENCE_3D | sequence | 60 | N/A | ENGINEERED | BROKEN -- needs 3D windowing |
| resnet1d | SEQUENCE_3D | sequence | 60 | N/A | ENGINEERED | BROKEN -- needs 3D windowing |
| patchtst | MULTI_TF_4D | multi_stream | 60 | (5min, 15min) | RAW | BROKEN -- needs entirely different preprocessing |
| itransformer | MULTI_TF_4D | multi_stream | 60 | (5min, 15min) | RAW | BROKEN -- needs entirely different preprocessing |

### Difficulty tiers for implementation:

| Tier | Models | Effort | What's needed |
|------|--------|--------|---------------|
| **Already works** | xgboost, lightgbm, catboost | None | N/A |
| **Medium** | lstm, gru, tcn, transformer, tft, inceptiontime, resnet1d | Add `SequenceAdapter` windowing in `_adapt_for_model()` | Take 2D DataFrame, create sliding windows of `sequence_length`, return 3D ndarray |
| **Hard** | nbeats | Add windowing + handle `feature_mode=RAW` (fewer features) | May need different feature set than engineered features |
| **Hardest** | patchtst, itransformer | Different preprocessing path entirely | Need raw OHLCV at 1min/5min/15min, NOT engineered features; need `MultiStreamAdapter` or equivalent |

---

## 14. Summary for Downstream Agents

### The 4 things to fix in priority order:

1. **Double-scaling bug** (bundle.py L1041): Change `skip_scaling=False` to `skip_scaling=True`. Surgical one-line fix.

2. **3D adapter routing** (bundle.py, new code): Add `_adapt_for_model()` method that creates sliding windows from 2D features for `requires_sequences=True` models. Use `self.metadata.sequence_length` for window size.

3. **4D preprocessing path** (bundle.py, new code): Add alternative preprocessing for `requires_4d=True` models that bypasses engineered features and builds multi-resolution raw OHLCV tensors. This is the hardest part.

4. **EnsembleBundle.predict_from_raw()** (ensemble_bundle.py, new method): Add method that calls `base_bundle.predict_from_raw(raw_df)` for each base model, collects probabilities, stacks, and runs meta-learner. Depends on items 2 and 3 being done first.

### Dependencies:
```
Fix double-scaling (1) -- no deps, do first
  -> 3D adapter routing (2) -- depends on correct scaling
    -> EnsembleBundle.predict_from_raw (4) -- depends on base models working
4D preprocessing (3) -- independent of 2, can be parallel
```

### Files that implementation agents will modify:
- `/home/jake/Desktop/Research/src/inference/bundle.py` -- L1041 fix + new methods
- `/home/jake/Desktop/Research/src/inference/ensemble_bundle.py` -- new predict_from_raw() + path fix
- `/home/jake/Desktop/Research/src/inference/builder.py` -- fix build_ensemble_bundle() to use EnsembleBundle properly

### Files that implementation agents will READ (not modify):
- `/home/jake/Desktop/Research/src/core/contracts/model_contract.py` -- MODEL_CONTRACTS registry, DataRank enum usage
- `/home/jake/Desktop/Research/src/core/types.py` -- DataRank enum (L32-43)
- `/home/jake/Desktop/Research/src/data/adapters/sequence.py` -- SequenceAdapter for reference implementation
- `/home/jake/Desktop/Research/src/data/adapters/multi_stream.py` -- MultiStreamAdapter for reference implementation
- `/home/jake/Desktop/Research/src/data/adapters/registry.py` -- AdapterRegistry.get_for_model()
