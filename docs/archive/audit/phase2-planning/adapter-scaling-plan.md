# Adapter Integration & Scaling Resolution Plan

**Date:** 2026-02-15
**Author:** Adapter/Scaling Planning Agent
**Scope:** Per-model adapter routing in inference, double-scaling resolution, MTF data generation, feature column alignment

---

## 1. Per-Model-Family Routing Table

The routing table below is derived from `MODEL_CONTRACTS` in `src/core/contracts/model_contract.py`. At inference time, `BundleMetadata` fields (`requires_sequences`, `requires_4d`, `sequence_length`, `n_timeframes`, `model_name`, `model_family`) drive adapter selection.

| Model Name | Family | Adapter ID | Input Rank | Seq Len | MTF TFs | Feature Mode | Scaler Type | BundleMetadata Flags |
|------------|--------|-----------|-----------|---------|---------|-------------|-------------|---------------------|
| **xgboost** | boosting | `tabular` | 2D | - | - | engineered | none | `requires_sequences=False, requires_4d=False` |
| **lightgbm** | boosting | `tabular` | 2D | - | - | engineered | none | same |
| **catboost** | boosting | `tabular` | 2D | - | - | engineered | none | same |
| **random_forest** | classical | `tabular` | 2D | - | - | engineered | none | same |
| **logistic** | classical | `tabular` | 2D | - | - | engineered | standard | same |
| **svm** | classical | `tabular` | 2D | - | - | engineered | standard | same |
| **lstm** | neural | `sequence` | 3D | 60 | - | engineered | robust | `requires_sequences=True, sequence_length=60` |
| **gru** | neural | `sequence` | 3D | 60 | - | engineered | robust | same |
| **tcn** | neural | `sequence` | 3D | 120 | - | engineered | robust | `requires_sequences=True, sequence_length=120` |
| **transformer** | transformer | `sequence` | 3D | 128 | - | hybrid | standard | `requires_sequences=True, sequence_length=128` |
| **tft** | neural | `sequence` | 3D | 60 | - | hybrid | robust | `requires_sequences=True, sequence_length=60` |
| **nbeats** | neural | `sequence` | 3D | 60 | - | raw | robust | `requires_sequences=True, sequence_length=60` |
| **inceptiontime** | neural | `sequence` | 3D | 60 | - | engineered | robust | same |
| **resnet1d** | neural | `sequence` | 3D | 60 | - | engineered | robust | same |
| **patchtst** | transformer | `multi_stream` | 4D | 60 | 5min,15min | raw | standard | `requires_4d=True, n_timeframes=3, sequence_length=60` |
| **itransformer** | transformer | `multi_stream` | 4D | 60 | 5min,15min | raw | robust | `requires_4d=True, n_timeframes=3, sequence_length=60` |
| **voting** | ensemble | `tabular` | 2D | - | - | engineered | none | tabular flags |
| **stacking** | ensemble | `tabular` | 2D | - | - | oof_probs | none | tabular flags |
| **blending** | ensemble | `tabular` | 2D | - | - | oof_probs | none | tabular flags |
| **ridge_meta** | meta_learner | `tabular` | 2D | - | - | oof_probs | none | tabular flags |
| **mlp_meta** | meta_learner | `tabular` | 2D | - | - | oof_probs | standard | tabular flags |
| **calibrated_meta** | meta_learner | `tabular` | 2D | - | - | oof_probs | none | tabular flags |
| **xgboost_meta** | meta_learner | `tabular` | 2D | - | - | oof_probs | none | tabular flags |

### Inference-Time Routing Logic

```python
# In ModelBundle or UniversalInferencePipeline:
def _get_adapter_for_bundle(metadata: BundleMetadata) -> str:
    if metadata.requires_4d:
        return "multi_stream"
    elif metadata.requires_sequences:
        return "sequence"
    else:
        return "tabular"
```

This already aligns with `ModelContract.adapter_id` property logic. The key insight: **BundleMetadata already stores enough information to route correctly**. No new flags needed for adapter selection.

---

## 2. Double-Scaling Resolution

### Current State (The Problem)

Two scaling points exist:

1. **Pipeline Stage 7.5** (`FeatureScaler` in `src/data/pipeline/stages/scaling/run.py`):
   - Fits `RobustScaler` on training split only
   - Saves `feature_scaler.pkl` + `scaling_metadata.json`
   - Transforms val/test splits
   - Output: scaled parquet files in `{splits_dir}/scaled/`

2. **Adapter Scaling** (`AdapterScaler` in `src/data/adapters/scaling.py`):
   - Applied by `UnifiedDataPreparation.prepare()` when `apply_scaling=True` (default)
   - Fits on adapter-transformed training data
   - Handles 2D/3D/4D reshaping for sklearn

**Risk:** If training consumes Stage 7.5 pre-scaled parquet files AND `apply_scaling=True` in `UnifiedDataPreparation`, features are scaled twice.

### Design Decision: Pipeline Scaler is Canonical

**The Pipeline Stage 7.5 scaler (`feature_scaler.pkl`) is the canonical scaler for inference.** Rationale:

1. It's already saved automatically during pipeline execution
2. It's already loaded into `ModelBundle.scaler` by `BundleBuilder`
3. It operates on the 2D feature space before adapter transformation
4. It's validated for leakage (train-only fit) by Stage 7.7

### Resolution Design

#### A. Add `scaling_source` to BundleMetadata

```python
@dataclass
class BundleMetadata:
    # ... existing fields ...
    scaling_source: str = "pipeline"  # "pipeline" | "adapter" | "none"
    scaling_applied_during_training: bool = True
```

- `"pipeline"`: Bundle scaler is the Stage 7.5 `feature_scaler.pkl` (default)
- `"adapter"`: Bundle scaler is the `AdapterScaler` from `UnifiedDataPreparation`
- `"none"`: No scaling was applied (boosting models with `requires_scaling=False`)

#### B. Enforce Single Scaling in Training Path

In `UnifiedDataPreparation.prepare()`, detect if input data is already scaled:

```python
def prepare(self, df, model_name, ..., apply_scaling=True):
    # Check if pipeline scaling was already applied
    # Indicator: scaling_metadata.json exists alongside the data
    if apply_scaling and self._is_already_scaled(df):
        logger.info("Input data already pipeline-scaled, skipping adapter scaling")
        apply_scaling = False
    # ... rest of prepare()
```

Better approach: **make `apply_scaling=False` the default** when consuming pipeline output, since the pipeline always scales. The training orchestrator should explicitly pass `apply_scaling=False`.

#### C. Enforce Single Scaling in Inference Path

In `ModelBundle.predict()`, the scaler is already applied once (lines 720-752 of `bundle.py`). The key rule:

> **Inference scaling rule:** `ModelBundle.predict()` applies `self.scaler` exactly once. The preprocessing graph (`PreprocessingGraph.transform()`) should set `skip_scaling=False` only if no scaler is in the bundle. Since all bundles from pipeline training have the pipeline scaler, `PreprocessingGraph.transform()` should default to `skip_scaling=True` when called from `predict_from_raw()`.

Fix in `ModelBundle.predict_from_raw()`:

```python
def predict_from_raw(self, raw_df, calibrate=True, skip_cleaning=False):
    # Preprocess WITHOUT scaling (bundle.predict() will scale)
    features = self.preprocessing_graph.transform(
        raw_df,
        skip_cleaning=skip_cleaning,
        skip_scaling=True,  # CRITICAL: bundle.predict() handles scaling
    )
    return self.predict(features, calibrate=calibrate)
```

#### D. Concrete Changes

| File | Change |
|------|--------|
| `src/inference/bundle.py:BundleMetadata` | Add `scaling_source: str = "pipeline"` field |
| `src/inference/bundle.py:BundleMetadata.to_dict/from_dict` | Serialize/deserialize `scaling_source` |
| `src/inference/bundle.py:ModelBundle.predict_from_raw` | Pass `skip_scaling=True` to preprocessing graph |
| `src/inference/bundle.py:ModelBundle.from_training` | Set `scaling_source` based on whether scaler is Pipeline or Adapter type |
| `src/data/adapters/preparation.py:UnifiedDataPreparation.prepare` | Default `apply_scaling=False` when input is pre-scaled pipeline output |

---

## 3. Adapter Wiring in Inference Path

### Current State

`ModelBundle.predict_from_raw()` calls `PreprocessingGraph.transform()` which outputs a 2D DataFrame. Then calls `self.predict(features)` which expects:
- 2D ndarray for tabular models (works)
- 3D ndarray for sequence models (BROKEN - gets 2D)
- 4D ndarray for multi-stream models (BROKEN - gets 2D)

### Design: Adapter Integration Inside `ModelBundle.predict_from_raw()`

Add adapter reshaping between preprocessing and prediction. The bundle already knows its data rank via `metadata.requires_sequences` and `metadata.requires_4d`.

```python
def predict_from_raw(self, raw_df, calibrate=True, skip_cleaning=False):
    # Step 1: Feature engineering (2D output)
    features_2d = self.preprocess(raw_df, skip_cleaning=skip_cleaning)

    # Step 2: Adapter reshaping (2D → 3D or 4D if needed)
    model_input = self._apply_adapter(features_2d, raw_df)

    # Step 3: Predict (scaling + model + calibration)
    return self.predict(model_input, calibrate=calibrate)
```

#### `_apply_adapter` Implementation

```python
def _apply_adapter(self, features_2d: pd.DataFrame, raw_df: pd.DataFrame) -> np.ndarray | pd.DataFrame:
    """Reshape 2D features to model-required tensor shape."""

    if self.metadata.requires_4d:
        # 4D: Need multi-timeframe data
        return self._build_4d_input(features_2d, raw_df)

    elif self.metadata.requires_sequences:
        # 3D: Build sliding windows
        return self._build_3d_input(features_2d)

    else:
        # 2D: Pass through
        return features_2d
```

#### `_build_3d_input` Implementation

```python
def _build_3d_input(self, features_2d: pd.DataFrame) -> np.ndarray:
    """Convert 2D DataFrame to 3D sequences using SequenceAdapter."""
    from src.data.adapters.sequence import SequenceAdapter

    seq_len = self.metadata.sequence_length
    adapter = SequenceAdapter(
        feature_columns=self.feature_columns,
        label_column=None,       # No labels at inference
        weight_column=None,      # No weights at inference
        sequence_length=seq_len,
        stride=1,
        symbol_column=None,      # Single-symbol inference
    )

    # Use adapter's windowing logic but skip label extraction
    features_array = features_2d[self.feature_columns].values.astype(np.float32)
    n_rows = len(features_array)

    if n_rows < seq_len:
        raise ValueError(
            f"Need at least {seq_len} rows for sequence model, got {n_rows}"
        )

    # Sliding window view (same as SequenceAdapter._build_sequences)
    windows = np.lib.stride_tricks.sliding_window_view(
        features_array, seq_len, axis=0
    )
    X = windows.transpose(0, 2, 1).copy()  # (n_sequences, seq_len, n_features)
    return X
```

**Note:** This creates sequences without labels (inference mode). The SequenceAdapter currently requires labels. We need either:
- Option A: Add an inference-mode `transform_inference()` method to SequenceAdapter
- Option B: Implement windowing directly in ModelBundle (shown above, simpler)

**Recommendation:** Option B for now (windowing is 5 lines of numpy), Option A later for consistency.

#### `_build_4d_input` Implementation

```python
def _build_4d_input(self, features_2d: pd.DataFrame, raw_df: pd.DataFrame) -> np.ndarray:
    """Build 4D tensor for multi-stream models from raw 1min data."""
    from src.data.adapters.multi_stream import MultiStreamAdapter

    # Get MTF config from bundle metadata
    mtf_timeframes = self.metadata.extra.get("mtf_timeframes", ["1min", "5min", "15min"])
    seq_len = self.metadata.sequence_length
    n_features = self.metadata.n_features  # Should be 5 for raw OHLCV

    # Generate multi-TF DataFrames from raw 1min data
    tf_dfs = self._generate_mtf_dataframes(raw_df, mtf_timeframes)

    # Use MultiStreamAdapter to build 4D tensor
    adapter = MultiStreamAdapter(
        feature_columns=["open", "high", "low", "close", "volume"],
        label_column=None,       # No labels at inference
        weight_column=None,
        sequence_length=seq_len,
        timeframes=mtf_timeframes,
    )

    # Need inference-mode transform (see Section 4)
    return adapter.transform_inference(tf_dfs)
```

### Concrete Changes

| File | Change |
|------|--------|
| `src/inference/bundle.py:ModelBundle` | Add `_apply_adapter()`, `_build_3d_input()`, `_build_4d_input()` methods |
| `src/inference/bundle.py:ModelBundle.predict_from_raw` | Insert adapter step between preprocess and predict |
| `src/inference/bundle.py:ModelBundle.predict` | Handle 3D/4D numpy arrays as input (currently only handles DataFrame or pre-shaped ndarray) |

---

## 4. MTF Data Generation at Inference

### Problem

4D models (PatchTST, iTransformer) need `(n_samples, n_timeframes, seq_len, n_features)` tensors. During training, multi-TF data comes from the raw MTF store or `additional_dfs`. At inference, the caller provides 1min raw OHLCV — we must generate the higher-TF DataFrames internally.

### Design: MTF Generation from 1min Data

The multi-timeframe OHLCV can be generated from 1min bars by resampling:

```python
def _generate_mtf_dataframes(
    self,
    raw_1min_df: pd.DataFrame,
    timeframes: list[str],
) -> dict[str, pd.DataFrame]:
    """Generate multi-TF OHLCV DataFrames from 1min data."""
    from src.data.pipeline.stages.clean.utils import resample_ohlcv

    tf_dfs = {}
    for tf in timeframes:
        if tf == "1min":
            tf_dfs[tf] = raw_1min_df
        else:
            tf_dfs[tf] = resample_ohlcv(raw_1min_df, tf)

    return tf_dfs
```

### What Must Be Stored in Bundle

The bundle needs to know which timeframes were used during training. Currently:
- `BundleMetadata.n_timeframes` stores the count
- `BundleMetadata.requires_4d` is True

**Missing:** The specific timeframe list.

**Fix:** Store `mtf_timeframes` in `BundleMetadata.extra`:

```python
# In ModelBundle.from_training():
if requires_4d:
    extra_metadata = extra_metadata or {}
    contract = get_model_contract(model_name)
    extra_metadata["mtf_timeframes"] = [contract.primary_timeframe] + list(contract.mtf_timeframes)
```

### Inference-Mode Adapter Transform

Both `SequenceAdapter.transform()` and `MultiStreamAdapter.transform()` currently require labels. For inference, we need label-free variants.

**Option A (Recommended):** Add `transform_inference()` to both adapters:

```python
class SequenceAdapter(BaseAdapter):
    def transform_inference(self, df: pd.DataFrame) -> np.ndarray:
        """Transform DataFrame to 3D sequences without label extraction."""
        feature_cols = self._get_feature_columns(df)
        features = df[feature_cols].values.astype(np.float32)
        windows = np.lib.stride_tricks.sliding_window_view(features, self.sequence_length, axis=0)
        return windows.transpose(0, 2, 1).copy()

class MultiStreamAdapter(BaseAdapter):
    def transform_inference(
        self, tf_dfs: dict[str, pd.DataFrame]
    ) -> np.ndarray:
        """Build 4D tensor from timeframe DataFrames without label extraction."""
        # Same as _build_multi_stream but skip label/weight extraction
        ...
```

**Option B (Simpler for MVP):** Inline the numpy windowing in `ModelBundle._build_3d_input()` (shown in Section 3). This avoids modifying adapter code.

### Concrete Changes

| File | Change |
|------|--------|
| `src/inference/bundle.py:ModelBundle.from_training` | Store `mtf_timeframes` in `extra` metadata for 4D models |
| `src/inference/bundle.py:ModelBundle` | Add `_generate_mtf_dataframes()` method |
| `src/data/adapters/sequence.py:SequenceAdapter` | Add `transform_inference()` (no labels/weights) |
| `src/data/adapters/multi_stream.py:MultiStreamAdapter` | Add `transform_inference()` (no labels/weights) |

---

## 5. Feature Column Alignment

### Problem

Feature columns can drift between training and inference due to:
1. Auto-detection heuristics differ (pipeline `EXCLUDED_COLUMNS` vs adapter inline exclusions)
2. DataFrame column sets differ (new metadata columns added)
3. Feature ordering not enforced

### Design: Strict Feature Column Enforcement

The bundle already stores `feature_columns` (ordered list) in `features.json`. The rule is:

> **At inference, ONLY use `bundle.feature_columns`.** Never auto-detect.

#### Current Enforcement (Already Working)

`ModelBundle._prepare_input()` (bundle.py:763-782) already:
1. Validates missing columns: `missing = set(self.feature_columns) - set(X.columns)`
2. Reorders columns: `X = X[self.feature_columns].values`

#### Gaps to Fix

1. **`PreprocessingGraph.transform()` may output extra columns.** The preprocessing graph generates all features, not just the model's selected subset. The `ModelBundle.preprocess()` method (bundle.py:1044-1054) does filter to `self.feature_columns`, but with a soft warning for missing columns.

   **Fix:** Make the warning an error (or at least log at WARNING level with the specific missing columns).

2. **Adapter auto-detection at inference.** If someone creates a `SequenceAdapter` at inference time without passing `feature_columns`, the adapter will auto-detect, potentially selecting different columns.

   **Fix:** When `ModelBundle._build_3d_input()` creates the adapter, always pass `feature_columns=self.feature_columns`.

3. **FeatureManifest not required.** The `FeatureManifest` exists for explicit column specification but isn't mandatory.

   **Fix:** `ModelBundle.from_training()` should always populate `feature_columns` from the training data. This is already the case (it's a required parameter).

### Concrete Changes

| File | Change |
|------|--------|
| `src/inference/bundle.py:ModelBundle.preprocess` | Raise error (not just warning) if critical features are missing (>10% missing = error) |
| `src/inference/bundle.py:ModelBundle._build_3d_input` | Pass `feature_columns=self.feature_columns` to adapter |
| `src/inference/bundle.py:ModelBundle._build_4d_input` | Pass `feature_columns` from contract (OHLCV for raw mode) |

---

## 6. Concrete Code Changes Summary

### Files to Modify

| # | File | Changes | Priority |
|---|------|---------|----------|
| 1 | `src/inference/bundle.py` | **BundleMetadata**: Add `scaling_source` field. **ModelBundle**: Add `_apply_adapter()`, `_build_3d_input()`, `_build_4d_input()`, `_generate_mtf_dataframes()`. Update `predict_from_raw()` to chain preprocessing → adapter → predict. Update `from_training()` to store `mtf_timeframes` and `scaling_source`. Fix preprocessing `skip_scaling` in `predict_from_raw()`. | **P0** |
| 2 | `src/data/adapters/sequence.py` | Add `transform_inference(df) -> np.ndarray` method for label-free windowing | **P1** |
| 3 | `src/data/adapters/multi_stream.py` | Add `transform_inference(tf_dfs) -> np.ndarray` method for label-free 4D tensor building | **P1** |
| 4 | `src/data/adapters/preparation.py` | Change `apply_scaling` default to `False` when consuming pre-scaled pipeline data. Add docstring clarifying scaling responsibility. | **P1** |
| 5 | `src/data/adapters/base.py` | Add `transform_inference()` abstract method stub to `BaseAdapter` (optional, returns NotImplemented) | **P2** |
| 6 | `src/inference/preprocessing_graph.py` | Ensure `transform()` respects `skip_scaling` parameter cleanly. Verify scaler is NOT applied when `skip_scaling=True`. | **P1** |

### Files to NOT Modify

- `src/core/contracts/model_contract.py` — No changes needed, already has all required info
- `src/data/adapters/registry.py` — Already works, inference uses it via model_name
- `src/data/adapters/scaling.py` — No changes needed, just won't be used in canonical inference path
- `src/data/pipeline/stages/scaling/run.py` — Pipeline scaling stays as-is

### New Fields Added

```python
# BundleMetadata additions:
scaling_source: str = "pipeline"  # "pipeline" | "adapter" | "none"

# BundleMetadata.extra additions (for 4D models):
extra["mtf_timeframes"] = ["1min", "5min", "15min"]  # Specific TFs used in training
```

### Validation Checklist

After implementation, verify:

1. **Tabular models (4):** `ModelBundle.predict_from_raw(raw_df)` works end-to-end for xgboost, lightgbm, catboost, random_forest
2. **Sequence models (8):** `ModelBundle.predict_from_raw(raw_df)` produces 3D tensors → correct predictions for lstm, gru, tcn, transformer, tft, nbeats, inceptiontime, resnet1d
3. **Multi-stream models (2):** `ModelBundle.predict_from_raw(raw_1min_df)` generates MTF data → 4D tensors → correct predictions for patchtst, itransformer
4. **No double scaling:** With `scaling_source="pipeline"`, data is scaled exactly once (in `ModelBundle.predict()`)
5. **Feature alignment:** `bundle.feature_columns` matches training features exactly
6. **Round-trip test:** Train model → bundle → predict_from_raw → compare with training predictions on same data

---

## 7. Architecture Diagram

```
                    predict_from_raw(raw_df)
                             │
                             ▼
                ┌─────────────────────────┐
                │ PreprocessingGraph       │
                │ .transform(raw_df,      │
                │   skip_scaling=True)    │
                │ → 2D features DataFrame │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ _apply_adapter()         │
                │                          │
                │ if requires_4d:          │
                │   _generate_mtf_dfs()    │
                │   _build_4d_input()      │
                │   → 4D ndarray           │
                │                          │
                │ elif requires_sequences: │
                │   _build_3d_input()      │
                │   → 3D ndarray           │
                │                          │
                │ else:                    │
                │   pass through 2D        │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ predict(shaped_input)    │
                │                          │
                │ 1. _prepare_input()      │
                │    (validate shape)      │
                │ 2. scaler.transform()    │
                │    (single scaling)      │
                │ 3. model.predict()       │
                │ 4. calibrator (optional) │
                │ → PredictionResult       │
                └─────────────────────────┘
```

**Key invariant:** Scaling happens ONCE, inside `predict()`, using the bundle's pipeline scaler. The preprocessing graph and adapters operate on unscaled features.

---

## 8. Dependencies and Ordering

```
Task 1: Add scaling_source to BundleMetadata (no deps)
Task 2: Add mtf_timeframes to BundleMetadata.extra (no deps)
Task 3: Fix predict_from_raw skip_scaling (depends on: Task 1)
Task 4: Add _build_3d_input to ModelBundle (no deps)
Task 5: Add _build_4d_input + _generate_mtf_dataframes (depends on: Task 2)
Task 6: Add _apply_adapter orchestration to predict_from_raw (depends on: Tasks 3, 4, 5)
Task 7: Add transform_inference to SequenceAdapter (optional, enhances Task 4)
Task 8: Add transform_inference to MultiStreamAdapter (optional, enhances Task 5)
Task 9: Fix apply_scaling default in UnifiedDataPreparation (independent)
Task 10: Add validation/testing (depends on: Task 6)
```

**Critical path:** Tasks 1 → 3 → 4 → 5 → 6 → 10
**Parallel work:** Tasks 7, 8, 9 can happen independently

---

*End of plan.*
