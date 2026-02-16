# 04 - Notebook, Types, Contracts & Adapter Inspection

**Date:** 2026-02-15
**Agent:** 4/10 (Notebook, Types, Contracts, Adapters)
**Purpose:** Comprehensive inspection of the notebook post-training flow, core type inventory, model contracts (all 23), adapter system internals, and protocols.py status. Produces the definitive MODEL CONTRACT TABLE for inference routing.

---

## 1. Notebook Post-Training Flow

**File:** `/home/jake/Desktop/Research/notebooks/ml_factory_colab.ipynb`
**Total cells:** 8 (1 markdown header + 7 code cells)

### Cell-by-Cell Inventory

| Cell | ID | Type | Purpose | What It Does |
|------|-----|------|---------|--------------|
| 0 | cell-0 | markdown | Header/intro | Model table, quick-start instructions |
| 1 | cell-1 | code | Setup | Clone repo, pip install, GPU check, import `MLFactory` |
| 2 | cell-2 | code | Configuration | All user-facing toggles (models, horizons, epochs, walk-forward, ensemble, optuna) |
| 3 | cell-3 | code | Validation | Validates model names, data path, GPU availability, training mode params |
| 4 | cell-4 | code | Data load | Read parquet/CSV, normalize columns, validate OHLCV, datetime index, preview |
| 5 | cell-5 | code | Training | Build `ExperimentConfig`, instantiate `MLFactory`, call `factory.run()` |
| 6 | cell-6 | code | Results | Print metrics table, ensemble metrics, backtest metrics, display up to 6 PNG plots, print bundle/output paths |
| 7 | cell-7 | code | Save/Download | Zip results, auto-download in Colab, optionally copy to Google Drive |

### What Happens After Training

**Cell 5** calls `factory.run()` which returns `ExperimentResult`. If successful, the result has:
- `result.metrics` -- per-model metric dict
- `result.ensemble_metrics` -- ensemble metric dict
- `result.backtest_metrics` -- backtest metric dict
- `result.bundle_path` -- `Path` to `output_dir/bundles/` (or None)
- `result.output_dir` -- `Path` to run output directory

**Cell 6** displays metrics in tables, renders up to 6 PNG plots, and prints `bundle_path` and `output_dir` paths.

**Cell 7** zips the output directory and triggers download (Colab) or prints the local path.

### What's Missing (Notebook Gaps)

| Gap | Severity | What's Needed |
|-----|----------|---------------|
| **No deploy directory creation** | CRITICAL | No cell creates `deploy/h{horizon}/artifact/` structure |
| **No artifact selection** | CRITICAL | No cell selects best model vs ensemble per horizon |
| **No deploy manifest** | CRITICAL | No cell writes `deploy/manifest.json` |
| **No inference demo** | HIGH | No cell loads a bundle and calls `predict_from_raw()` on sample data |
| **No bundle validation** | HIGH | No cell runs `bundle.validate()` or smoke test |
| **No bundle listing** | MEDIUM | Cell 6 prints `bundle_path` directory but doesn't list individual bundles |
| **No artifact packaging** | MEDIUM | No cell calls `bundle.package_bundle()` for tar.gz export |
| **No feature spec display** | LOW | No cell shows which features were selected or their lineage |

### Cells That Need to Be Added

For the deployable-artifact flow, the notebook needs approximately 4 new cells after Cell 6:

| New Cell | Purpose | Key Operations |
|----------|---------|----------------|
| **Cell 8: Bundle Inventory** | List all bundles created, show metadata | Iterate `bundles/` dir, load each `manifest.json`, display table of model_name, horizon, requires_sequences, requires_4d, n_features |
| **Cell 9: Deploy Artifact Selection** | Select best artifact per horizon | For each horizon: pick ensemble if valid, else best model by metric; create `deploy/h{horizon}/artifact/`; write `deploy/manifest.json` |
| **Cell 10: Validation & Inference Demo** | Smoke test + inference demo | Load selected artifact; call `validate()`; load sample raw OHLCV; call `predict_from_raw(sample_df)`; display prediction + timing |
| **Cell 11: Export & Download** | Package and download deploy artifacts | Call `package_bundle()` for each horizon artifact; zip `deploy/` directory; trigger download |

---

## 2. Core Types Inventory

**File:** `/home/jake/Desktop/Research/src/core/types.py` (275 lines)

### Every Enum Currently Defined

| Enum | Base | Values | Line | Purpose |
|------|------|--------|------|---------|
| `DataRank` | `int, Enum` | `TABULAR_2D=2`, `SEQUENCE_3D=3`, `MULTI_TF_4D=4` | L32-43 | Tensor dimensionality for model input |
| `ModelFamily` | `str, Enum` | `BOOSTING`, `CLASSICAL`, `NEURAL`, `ENSEMBLE`, `META_LEARNER`, `TRANSFORMER` | L69-84 | Model grouping for defaults/routing |
| `FeatureFamily` | `str, Enum` | `RAW`, `MOMENTUM`, `MOVING_AVERAGE`, `VOLATILITY`, `VOLUME`, `TREND`, `PRICE`, `MICROSTRUCTURE`, `ENTROPY`, `WAVELETS`, `TEMPORAL`, `REGIME`, `MTF` | L102-134 | Feature grouping (13 families) |
| `TrainingMode` | `str, Enum` | `STANDARD`, `WALK_FORWARD`, `REGIME_AWARE`, `META_LABELING` | L142-155 | Training procedure mode |
| `CVMethod` | `str, Enum` | `PURGED_KFOLD`, `CPCV`, `WALK_FORWARD`, `PBO`, `STANDARD` | L163-178 | Cross-validation method |
| `AdapterType` | `str, Enum` | `TABULAR`, `SEQUENCE`, `MULTI_STREAM` | L186-205 | Adapter routing type |
| `LabelingMethod` | `str, Enum` | `TRIPLE_BARRIER`, `DIRECTIONAL`, `THRESHOLD`, `REGRESSION` | L213-226 | Target generation method |

**Total: 7 enums currently in types.py**

### Type Aliases Defined

| Alias | Definition | Line |
|-------|-----------|------|
| `Features` | `np.ndarray \| pd.DataFrame` | L234 |
| `Labels` | `np.ndarray` | L235 |
| `ModelType` | `TypeVar("ModelType", bound="ModelContract")` | L238 |
| `Array1D` | `np.ndarray` (shape: n) | L241 |
| `Array2D` | `np.ndarray` (shape: n, features) | L242 |
| `Array3D` | `np.ndarray` (shape: n, seq_len, features) | L243 |
| `Array4D` | `np.ndarray` (shape: n, timeframes, seq_len, features) | L244 |
| `DatetimeIndex` | `pd.DatetimeIndex` | L247 |
| `Index` | `np.ndarray \| pd.Index` | L248 |

### Enums Defined OUTSIDE types.py (in data_contract.py)

| Enum | Location | Purpose |
|------|----------|---------|
| `FeatureMode` | `src/core/contracts/data_contract.py` L33-39 | `ENGINEERED`, `RAW`, `HYBRID`, `OOF_PROBS` |
| `ModelMTFMode` (aliased as `MTFMode`) | `src/core/contracts/data_contract.py` L42-58 | `NONE`, `INDICATORS`, `MULTI_STREAM` |

**Note:** `FeatureMode` and `MTFMode` are NOT in `types.py`. They live in `data_contract.py` and are imported by `model_contract.py`. Per CLAUDE.md rules, all enums should be in `types.py`, but moving these would be a refactor. They are re-exported from `src.core.contracts.__init__`.

### NEW Enums Needed (from plans)

| Enum | Planned Location | Values | Conflict Risk |
|------|-----------------|--------|---------------|
| `ScalingSource` | `src/core/types.py` (per W-1 warning) | `BUNDLE`, `PREPROCESSING`, `NONE` | **No conflict** -- does not exist anywhere yet |

**ScalingSource** can be safely added to `types.py` without any conflicts. No existing enum or class with this name exists in the codebase.

---

## 3. Model Contracts -- Complete Registry

**File:** `/home/jake/Desktop/Research/src/core/contracts/model_contract.py`

### ModelContract Dataclass Fields

| Field | Type | Default | Inference-Relevant |
|-------|------|---------|-------------------|
| `model_name` | `str` | required | Yes -- routing key |
| `model_family` | `str` | required | Yes -- family grouping |
| `input_rank` | `DataRank` | `TABULAR_2D` | Yes -- determines adapter path |
| `feature_mode` | `FeatureMode` | `ENGINEERED` | Yes -- what features to generate |
| `mtf_mode` | `MTFMode` | `NONE` | Yes -- multi-timeframe strategy |
| `primary_timeframe` | `str` | `"5min"` | Yes -- base timeframe |
| `mtf_timeframes` | `tuple[str, ...]` | `()` | Yes -- additional timeframes for 4D |
| `sequence_length` | `int` | `60` | Yes -- window size for 3D/4D |
| `patch_length` | `int \| None` | `None` | Info -- PatchTST only |
| `requires_scaling` | `bool` | `True` | Yes -- whether scaler applies |
| `scaler_type` | `str` | `"robust"` | Yes -- which scaler |
| `min_features` | `int` | `4` | Validation |
| `max_features` | `int` | `200` | Validation |
| `description` | `str` | `""` | Info |

### Computed Properties

| Property | Returns | Logic |
|----------|---------|-------|
| `requires_sequences` | `bool` | `input_rank.value >= 3` |
| `requires_multi_timeframe` | `bool` | `input_rank == MULTI_TF_4D` |
| `adapter_id` | `str` | `"tabular"` / `"sequence"` / `"multi_stream"` based on `input_rank` |

### COMPLETE MODEL CONTRACT TABLE (All 23 Models)

This table is the definitive reference for inference routing. The 12 "core" models (used in notebook) are marked with asterisks.

#### Core Prediction Models (12) -- Used in Notebook

| model_name | model_family | input_rank | adapter_id | requires_sequences | requires_4d | sequence_length | primary_tf | mtf_timeframes | feature_mode | requires_scaling | scaler_type | patch_length |
|-----------|-------------|-----------|-----------|-------------------|------------|----------------|-----------|---------------|-------------|-----------------|------------|-------------|
| **xgboost*** | boosting | TABULAR_2D | tabular | False | False | 60 | 15min | () | ENGINEERED | False | none | None |
| **lightgbm*** | boosting | TABULAR_2D | tabular | False | False | 60 | 15min | () | ENGINEERED | False | none | None |
| **catboost*** | boosting | TABULAR_2D | tabular | False | False | 60 | 15min | () | ENGINEERED | False | none | None |
| **lstm*** | neural | SEQUENCE_3D | sequence | True | False | 60 | 5min | () | ENGINEERED | True | robust | None |
| **gru*** | neural | SEQUENCE_3D | sequence | True | False | 60 | 5min | () | ENGINEERED | True | robust | None |
| **tcn*** | neural | SEQUENCE_3D | sequence | True | False | 120 | 5min | () | ENGINEERED | True | robust | None |
| **inceptiontime*** | neural | SEQUENCE_3D | sequence | True | False | 60 | 5min | () | ENGINEERED | True | robust | None |
| **resnet1d*** | neural | SEQUENCE_3D | sequence | True | False | 60 | 5min | () | ENGINEERED | True | robust | None |
| **tft*** | neural | SEQUENCE_3D | sequence | True | False | 60 | 5min | () | HYBRID | True | robust | None |
| **nbeats*** | neural | SEQUENCE_3D | sequence | True | False | 60 | 5min | () | RAW | True | robust | None |
| **patchtst*** | transformer | MULTI_TF_4D | multi_stream | True | True | 60 | 1min | (5min, 15min) | RAW | True | standard | 16 |
| **itransformer*** | transformer | MULTI_TF_4D | multi_stream | True | True | 60 | 1min | (5min, 15min) | RAW | True | robust | None |

#### Non-Core Models (11) -- Classical, Ensemble, Meta-Learner

| model_name | model_family | input_rank | adapter_id | requires_sequences | requires_4d | sequence_length | primary_tf | mtf_timeframes | feature_mode | requires_scaling | scaler_type |
|-----------|-------------|-----------|-----------|-------------------|------------|----------------|-----------|---------------|-------------|-----------------|------------|
| random_forest | classical | TABULAR_2D | tabular | False | False | 60 | 15min | () | ENGINEERED | False | none |
| logistic | classical | TABULAR_2D | tabular | False | False | 60 | 15min | () | ENGINEERED | True | standard |
| svm | classical | TABULAR_2D | tabular | False | False | 60 | 15min | () | ENGINEERED | True | standard |
| transformer | transformer | SEQUENCE_3D | sequence | True | False | 128 | 5min | () | HYBRID | True | standard |
| voting | ensemble | TABULAR_2D | tabular | False | False | 60 | 5min | () | ENGINEERED | False | none |
| stacking | ensemble | TABULAR_2D | tabular | False | False | 60 | 5min | () | OOF_PROBS | False | none |
| blending | ensemble | TABULAR_2D | tabular | False | False | 60 | 5min | () | OOF_PROBS | False | none |
| ridge_meta | meta_learner | TABULAR_2D | tabular | False | False | 60 | 5min | () | OOF_PROBS | False | none |
| mlp_meta | meta_learner | TABULAR_2D | tabular | False | False | 60 | 5min | () | OOF_PROBS | True | standard |
| calibrated_meta | meta_learner | TABULAR_2D | tabular | False | False | 60 | 5min | () | OOF_PROBS | False | none |
| xgboost_meta | meta_learner | TABULAR_2D | tabular | False | False | 60 | 5min | () | OOF_PROBS | False | none |

### Inference Routing Summary (Core 12 Only)

| Adapter Path | Models | Input Shape | Feature Source | Preprocessing Strategy |
|-------------|--------|-------------|---------------|----------------------|
| **tabular** (2D) | xgboost, lightgbm, catboost | `(n, feat)` | Engineered features (~150-200) | PreprocessingGraph.transform() -> 2D DataFrame -> pass through |
| **sequence** (3D) | lstm, gru, tcn, inceptiontime, resnet1d, tft | `(n, seq, feat)` | Engineered features (~50-150) | PreprocessingGraph.transform() -> 2D DataFrame -> SequenceAdapter sliding window |
| **sequence** (3D, RAW) | nbeats | `(n, seq, feat)` | Raw OHLCV (~2-20 features) | Different feature set needed; sliding window still applies |
| **sequence** (3D, HYBRID) | transformer | `(n, seq, feat)` | Mix raw + indicators (~20-100) | PreprocessingGraph.transform() subset -> sliding window |
| **multi_stream** (4D) | patchtst, itransformer | `(n, tf, seq, feat)` | Raw OHLCV at multiple TFs (4-10 features) | Bypass PreprocessingGraph entirely; load raw OHLCV per TF; MultiStreamAdapter |

**Note on `transformer` (vanilla):** This is listed in MODEL_CONTRACTS but NOT in the notebook's 12 model toggles. The notebook uses `patchtst`, `itransformer`, and `tft` as its transformer models. The vanilla `transformer` contract exists for completeness but may not be actively tested.

---

## 4. Adapter System

### Registered Adapters

Three adapters are registered via `@AdapterRegistry.register()`:

| adapter_id | Class | File | Output Rank |
|-----------|-------|------|-------------|
| `"tabular"` | `TabularAdapter` | `src/data/adapters/tabular.py` | `TABULAR_2D` |
| `"sequence"` | `SequenceAdapter` | `src/data/adapters/sequence.py` | `SEQUENCE_3D` |
| `"multi_stream"` | `MultiStreamAdapter` | `src/data/adapters/multi_stream.py` | `MULTI_TF_4D` |

### AdapterRegistry API

| Method | Purpose |
|--------|---------|
| `AdapterRegistry.get(adapter_id)` | Get adapter class by ID |
| `AdapterRegistry.create(adapter_id, **kwargs)` | Create adapter instance |
| `AdapterRegistry.get_for_model(model_name, **kwargs)` | Look up contract -> get adapter_id -> create instance |
| `AdapterRegistry.get_for_contract(contract, **kwargs)` | Create adapter from contract.adapter_id |
| `get_adapter(model_name=None, adapter_id=None, **kwargs)` | Convenience function |

### How Each Adapter's `transform()` Works

#### TabularAdapter.transform(df, model_contract=None)

1. Validate input (label column exists)
2. Auto-detect feature columns (or use explicit list)
3. Validate feature count against contract bounds
4. Extract `X = df[feature_cols].values.astype(float32)` -- shape `(n_samples, n_features)`
5. Extract `y = df[label_column].values.astype(int64)` -- shape `(n_samples,)`
6. Extract optional weights
7. Create `DataContract`, validate against model contract
8. Return `AdapterResult(X, y, weights, data_rank=TABULAR_2D)`

**For inference replication:** Simply select feature columns from 2D DataFrame and convert to float32 array. Trivial.

#### SequenceAdapter.transform(df, model_contract=None)

1. Validate input, get feature columns
2. Override `sequence_length` from contract if provided
3. If symbol column present: build sequences per-symbol (temporal isolation)
4. `_build_sequences()`:
   - Extract `features = df[feature_cols].values.astype(float32)` -- 2D array
   - Use `np.lib.stride_tricks.sliding_window_view()` for vectorized windowing
   - Windows at stride intervals, transpose to `(n_sequences, seq_len, n_features)`
   - Labels extracted at LAST timestep of each window (no lookahead)
   - `label_positions = np.arange(n_sequences) * stride + (seq_len - 1)`
5. Return `AdapterResult(X, y, weights, data_rank=SEQUENCE_3D)`

**For inference replication:** Take 2D feature array, create sliding windows of `sequence_length`. The key operation is:
```python
windows = np.lib.stride_tricks.sliding_window_view(features_2d, seq_len, axis=0)
X_3d = windows[::stride].transpose(0, 2, 1).copy()
```
At inference time, for a SINGLE prediction, you need at least `sequence_length` rows of 2D features, then the last window is the input.

#### MultiStreamAdapter.transform(df, model_contract=None, additional_dfs=None)

1. Validate input
2. Resolve timeframes from contract: `[primary_timeframe] + list(mtf_timeframes)`
3. Resolve sequence length from contract
4. Resolve feature columns (defaults to `["open", "high", "low", "close", "volume"]`)
5. Collect DataFrames for each timeframe:
   - Anchor TF from `df` parameter
   - Others from `additional_dfs`, raw MTF store, or data_dir
6. `_build_multi_stream()`:
   - Anchor TF determines sample count
   - For anchor: direct sliding window extraction
   - For higher TFs: timestamp-based alignment via `pd.merge_asof()` (backward lookup)
   - Deduplication of consecutive identical higher-TF bars
   - Padding with earliest bar if fewer unique bars than seq_len
7. Return `AdapterResult(X, y, weights, data_rank=MULTI_TF_4D)` -- shape `(n_seq, n_tf, seq_len, n_feat)`

**For inference replication:** This is the hardest adapter to replicate at inference time because:
- Needs raw OHLCV at MULTIPLE timeframes (not engineered features)
- Needs timestamp-based alignment between timeframes
- Needs `merge_asof` or equivalent for correct gap handling
- Cannot use PreprocessingGraph (which outputs single-timeframe engineered features)
- At minimum, need raw 1min OHLCV data, then resample to 5min and 15min

### What's Needed to Replicate Adapters at Inference Time

| Adapter | Replication Difficulty | What's Needed in Bundle |
|---------|----------------------|------------------------|
| TabularAdapter | Trivial | `feature_columns` list (already in BundleMetadata) |
| SequenceAdapter | Easy | `sequence_length` (already in BundleMetadata), `feature_columns` |
| MultiStreamAdapter | Hard | `timeframes` list, `feature_columns` (OHLCV), raw OHLCV per TF, timestamp alignment logic |

**Critical insight:** For SequenceAdapter replication at inference, the bundle does NOT need a full SequenceAdapter instance. It only needs to:
1. Take the 2D feature DataFrame from PreprocessingGraph
2. Apply `sliding_window_view(features, seq_len, axis=0)`
3. Take the last window (for single-point prediction) or all windows (for batch)

This can be implemented as a simple method inside `ModelBundle._adapt_for_model()` without importing SequenceAdapter.

For MultiStreamAdapter replication, a different approach is needed because PreprocessingGraph cannot produce the required multi-timeframe raw OHLCV. Options:
- Store `mtf_timeframes` in BundleMetadata (already available from contract)
- Add a `_preprocess_4d()` method that takes raw OHLCV and resamples per timeframe
- Or require callers of 4D model bundles to provide `additional_dfs`

---

## 5. protocols.py Status

**File:** `/home/jake/Desktop/Research/src/core/protocols.py`

**STATUS: DOES NOT EXIST**

Confirmed via filesystem check. The file has not been created in any prior phase (0-50).

Per the plans, this file needs to contain:
- `TrainerProtocol` -- Protocol class defining what a Trainer must expose for bundle extraction
- `InferenceBundle` -- Protocol class defining the universal `predict_from_raw()` contract

Both `ModelBundle` and `EnsembleBundle` should satisfy `InferenceBundle`.

**Estimated size:** ~45 lines (from MASTER-IMPLEMENTATION-PLAN)

**Must be created in Phase 3A** before BundleBuilder can use protocol-based extraction instead of duck-typing.

---

## 6. Key Findings for Downstream Agents

### 6.1 BundleMetadata Already Has Routing Fields

The existing `BundleMetadata` dataclass (bundle.py L70-92) already stores:
- `requires_sequences: bool`
- `requires_4d: bool`
- `sequence_length: int`
- `n_timeframes: int`
- `model_family: str`
- `model_name: str`

These are populated from the model object during `ModelBundle.from_training()` (bundle.py L285-311). This means **inference routing can use BundleMetadata directly** without needing to look up MODEL_CONTRACTS at inference time.

### 6.2 Missing BundleMetadata Fields for Full Inference

The following are NOT in BundleMetadata but ARE needed for complete inference routing:

| Missing Field | Source | Why Needed |
|--------------|--------|------------|
| `primary_timeframe` | `contract.primary_timeframe` | 4D models need to know base TF for resampling |
| `mtf_timeframes` | `contract.mtf_timeframes` | 4D models need to know which TFs to resample to |
| `feature_mode` | `contract.feature_mode` | Determines whether to use engineered features or raw OHLCV |
| `scaler_type` | `contract.scaler_type` | Which scaler to apply |
| `adapter_id` | `contract.adapter_id` | Direct adapter routing (can be derived from requires_sequences/requires_4d) |

**Recommendation:** Add `primary_timeframe`, `mtf_timeframes`, and `feature_mode` to BundleMetadata as part of the version 1.3.0 bump.

### 6.3 SequenceAdapter Windowing Is Self-Contained

The core windowing logic in SequenceAdapter._build_sequences() uses only:
- `np.lib.stride_tricks.sliding_window_view()`
- `sequence_length` parameter
- `stride` parameter (default 1)

This can be trivially replicated in ModelBundle without importing SequenceAdapter. A ~10-line `_window_2d_to_3d()` method would suffice.

### 6.4 MultiStreamAdapter Has Complex Dependencies

MultiStreamAdapter requires:
- Raw OHLCV DataFrames per timeframe (not engineered features)
- Timestamp-based alignment via `pd.merge_asof()`
- The raw MTF store (`src/data/store`) or `additional_dfs`
- Timeframe normalization via `src/core/common/timeframes`

Replicating this inside ModelBundle is non-trivial. The simplest approach for Phase 3B:
1. Add a `_preprocess_4d(raw_df)` method to ModelBundle
2. This method resamples raw OHLCV to each required timeframe
3. Builds the 4D tensor using the same windowing + alignment logic
4. Does NOT use PreprocessingGraph at all for 4D models

### 6.5 Notebook Configuration Matches Contract Expectations

The notebook's Cell 2 selects exactly the 12 core models. The `HORIZONS = [20]` default means only `label_h20` is generated. Walk-forward with 5 windows is the default training mode. This is consistent with the model contracts which all default to `label_h20` parsing.

### 6.6 FeatureMode and MTFMode Are NOT in types.py

These two enums live in `src/core/contracts/data_contract.py`. The CLAUDE.md rule says "all enums/types in src/core/types.py." If moved, all imports from `data_contract.py` would need updating. This is a Phase 3D cleanup task, not blocking for Phase 3A/3B.

---

## 7. Adapter Routing Decision Table for Inference

This table combines BundleMetadata flags with the contract data to give the complete routing picture:

```
                              BundleMetadata Flags
                    ┌────────────────────────────────────────┐
                    │ requires_4d=True                       │
                    │   -> _preprocess_4d(raw_df)            │
                    │   -> builds 4D from raw OHLCV at       │
                    │      [1min, 5min, 15min]               │
                    │   -> shape: (n, 3, 60, 5)              │
                    │   -> Models: patchtst, itransformer    │
                    ├────────────────────────────────────────┤
                    │ requires_sequences=True, requires_4d=F │
                    │   -> PreprocessingGraph.transform()    │
                    │      with skip_scaling=True            │
                    │   -> _window_2d_to_3d(features_2d,     │
                    │      sequence_length)                  │
                    │   -> shape: (n, seq_len, feat)         │
                    │   -> Models: lstm, gru, tcn,           │
                    │      inceptiontime, resnet1d, tft,     │
                    │      nbeats, transformer               │
                    ├────────────────────────────────────────┤
                    │ requires_sequences=False, requires_4d=F│
                    │   -> PreprocessingGraph.transform()    │
                    │      with skip_scaling=True            │
                    │   -> pass through 2D DataFrame         │
                    │   -> shape: (n, feat)                  │
                    │   -> Models: xgboost, lightgbm,        │
                    │      catboost                          │
                    └────────────────────────────────────────┘
```

---

## 8. Summary for Downstream Agents

### For Agent 5 (Trainer/Scaler Inspection):
- `TrainerProtocol` does not exist yet -- must be created
- Current extraction is pure duck-typing (see Agent 2/3 findings)
- The `scaler_type` field on contracts tells you what scaler each model expects

### For Agent 6 (Architecture Planner):
- Use the MODEL CONTRACT TABLE in Section 3 as the definitive routing reference
- BundleMetadata already has `requires_sequences`, `requires_4d`, `sequence_length`, `n_timeframes`
- Missing from BundleMetadata: `primary_timeframe`, `mtf_timeframes`, `feature_mode`
- `ScalingSource` enum can be safely added to `types.py`
- `protocols.py` must be created fresh (does not exist)
- The notebook needs 4 new cells (Section 1)
- SequenceAdapter windowing is trivially replicable (~10 lines)
- MultiStreamAdapter replication is complex (needs raw OHLCV at multiple TFs)

### For Agent 7+ (Implementation):
- The 3 adapter classes are registered and working for training
- `AdapterRegistry.get_for_model(model_name)` is the canonical way to get an adapter
- At inference time, do NOT instantiate full adapters -- use lightweight windowing methods in ModelBundle
- 4D models (patchtst, itransformer) need a fundamentally different preprocessing path than PreprocessingGraph provides

### Critical Data Points:
- **BUNDLE_VERSION** is currently `"1.2.0"` (confirmed by Agent 3)
- **protocols.py** does NOT exist (confirmed)
- **ScalingSource** does NOT exist anywhere (safe to create)
- **FeatureMode/MTFMode** are in `data_contract.py`, not `types.py` (known deviation)
- **Notebook has 7 cells** (0=markdown, 1-7=code), needs ~4 more for deploy flow
