# Data Flow Audit: Raw OHLCV to Model Input

## Overview

ML Factory transforms raw OHLCV data into model-ready numpy arrays through a two-phase pipeline:

1. **Pipeline Phase** (12 stages): Raw data -> cleaned, feature-engineered, labeled, scaled parquet files
2. **Adapter Phase**: Scaled DataFrames -> model-specific numpy arrays (2D/3D/4D)

The system supports heterogeneous ensembles where different model families consume the same underlying data in different tensor shapes. All data preparation enforces a "no bypass" policy through `UnifiedDataPreparation`.

---

## Pipeline Stages (12 Stages)

The pipeline is defined in `src/data/pipeline/stage_registry.py` and executed by `src/data/pipeline/runner.py`.

| Stage | Name | Number | Description |
|-------|------|--------|-------------|
| 1 | `data_generation` | 1 | Generate or validate raw data files |
| 2 | `data_cleaning` | 2 | Clean and resample OHLCV data (bar building, gap handling) |
| 3 | `feature_engineering` | 3 | Generate technical features (indicators, wavelets, MTF, regime) |
| 4 | `initial_labeling` | 4 | Apply initial triple-barrier labeling |
| 5 | `ga_optimize` | 5 | GA/Optuna optimization of barrier parameters |
| 6 | `final_labels` | 6 | Apply optimized labels with quality scores |
| 7 | `create_splits` | 7 | Create chronological train/val/test splits |
| 7.5 | `feature_scaling` | 7.5 | Train-only feature scaling (RobustScaler default) |
| 7.6 | `build_datasets` | 7.6 | Build dataset splits and feature set manifests |
| 7.7 | `validate_scaled` | 7.7 | Post-scale drift validation |
| 8 | `validate` | 8 | Comprehensive data validation |
| 9 | `generate_report` | 9 | Generate completion report |

### Key Pipeline Properties

- **Dependencies are enforced**: Each stage must complete before its dependents run
- **State is persisted**: `pipeline_state.json` tracks completed stages for resume capability
- **Artifact manifest**: All outputs tracked via `ArtifactManifest`
- **Schema validation**: Between-stage transition validation (Phase 43) catches column mismatches
- **Lineage tracking**: `PipelineLineage` records pipeline run parameters and dataset checksums
- **Multi-TF support**: Stages 7.5, 7.6 process each timeframe separately when in multi-TF mode

### Stage 7.5: Feature Scaling (Critical for Reproducibility)

Location: `src/data/pipeline/stages/scaling/run.py`

1. Loads combined labeled data and split indices
2. Identifies feature columns (excludes OHLCV, labels, metadata via `EXCLUDED_COLUMNS`/`EXCLUDED_PREFIXES`)
3. Fits `FeatureScaler` on **training data only** (leakage prevention)
4. Transforms val/test using training statistics
5. Validates scaling integrity and checks for data leakage
6. Saves scaled parquet files and `feature_scaler.pkl`
7. Saves `scaling_metadata.json` with feature columns and scaler config

**Scaler is saved at**: `{splits_dir}/scaled/feature_scaler.pkl`
**Scaling metadata at**: `{splits_dir}/scaled/scaling_metadata.json`

### Stage 7.6: Dataset Build

Location: `src/data/pipeline/stages/datasets/run.py`

- Organizes scaled data into feature set / horizon combinations
- Creates `feature_set_manifest.json` and `dataset_manifest.json`
- Validates feature schema consistency across splits (PIPE-004)
- In multi-TF mode, validates cross-timeframe column consistency

---

## Adapter System

### Architecture

```
AdapterRegistry (class registry, decorator-based)
    ├── TabularAdapter   @register("tabular")   -> 2D (n_samples, n_features)
    ├── SequenceAdapter  @register("sequence")   -> 3D (n_samples, seq_len, n_features)
    └── MultiStreamAdapter @register("multi_stream") -> 4D (n_samples, n_tf, seq_len, n_features)
```

### Adapter Routing

Model-to-adapter mapping is defined in `src/core/constants.py` via `MODEL_ADAPTER_MAP`:

| Adapter | Models |
|---------|--------|
| **tabular** | xgboost, lightgbm, catboost, nbeats |
| **sequence** | lstm, gru, tcn, inceptiontime, resnet1d |
| **multi_stream** | patchtst, itransformer, tft |

Two entry points for creating adapters:
1. **`AdapterFactory`** (`src/data/adapters/factory.py`): Config-driven creation, integrates with `PipelineConfig` for sequence_length and mtf_timeframes
2. **`AdapterRegistry.get_for_model()`** / **`get_adapter()`**: Direct registry lookup via model contract

### TabularAdapter (`src/data/adapters/tabular.py`)

- **Input**: DataFrame with features + labels + optional weights
- **Output**: `AdapterResult` with `X: (n_samples, n_features) float32`, `y: (n_samples,) int64`
- **Feature detection**: Explicit `feature_columns` or auto-detection (excludes `label_*`, `sample_weight_*`, `regime_*`, OHLCV, metadata via prefix/exact matching)
- **Validation**: Checks against `ModelContract` min/max feature bounds
- **Creates**: `DataContract` with symbol, timeframe, horizon, split metadata

### SequenceAdapter (`src/data/adapters/sequence.py`)

- **Input**: Same DataFrame, plus `sequence_length` and `stride` params
- **Output**: `AdapterResult` with `X: (n_sequences, seq_len, n_features) float32`
- **Windowing**: Sliding window via `np.lib.stride_tricks.sliding_window_view` (vectorized, no Python loop)
- **Label alignment**: Label at LAST timestep of each window (no future leakage)
- **Per-symbol isolation**: If `symbol` column present, builds sequences per-symbol then concatenates (prevents cross-symbol leakage)
- **Index tracking**: `original_indices` maps each sequence back to source DataFrame row
- **Contract override**: Can get `sequence_length` from `ModelContract`

### MultiStreamAdapter (`src/data/adapters/multi_stream.py`)

- **Input**: Anchor DataFrame (smallest TF) + additional DataFrames for higher TFs
- **Output**: `AdapterResult` with `X: (n_sequences, n_timeframes, seq_len, n_features) float32`
- **Default features**: `["open", "high", "low", "close", "volume"]` (raw OHLCV)
- **Default timeframes**: `["1min", "5min", "15min", "60min"]`
- **Timestamp alignment**: Uses `pd.merge_asof` for proper alignment across timeframes (handles overnight/weekend gaps correctly)
- **Fallback**: Ratio-based alignment when no DatetimeIndex available (with warning)
- **Data loading**: Three methods:
  1. `additional_dfs` parameter (explicit)
  2. Raw MTF store (`symbol` + `split` + `base_path`)
  3. Legacy `data_dir` file patterns
- **Sequence extraction**: For higher TFs, deduplicates consecutive same-bar references, pads/truncates to `seq_len`

---

## Feature Flow

### Feature Computation (`src/data/features/compute/`)

15+ feature modules organized by category:
- `price.py`, `momentum.py`, `volatility.py`, `volume.py`, `trend.py`
- `moving_average.py`, `microstructure.py`, `order_flow.py`, `entropy.py`
- `liquidity.py`, `mean_reversion.py`, `regime.py`, `temporal.py`
- `wavelets.py`, `mtf.py`, `raw.py`

### Feature Selection (`src/data/features/selection.py`)

- Optuna-based feature selection with 3 strategies: binary, family, importance
- Produces `FeatureSelectionResult` with `selected_features`, `feature_importance`
- Cross-validated scoring with configurable metrics
- Runs on 2D tabular data only

### Feature Manifest (`src/data/pipeline/feature_manifest.py`)

- `FeatureManifest`: Records feature columns, label columns, and per-feature metadata
- `FeatureMetadata`: Per-feature computation parameters (category, params, source_columns, checksum)
- Supports `validate_reproducibility()` for comparing manifests
- `from_dataframe()`: Auto-creates manifest from DataFrame column naming conventions
- Adapters can load feature columns from manifest via `BaseAdapter.from_manifest()`

---

## Preprocessing Chain (Full Flow)

```
Raw OHLCV (1min parquet)
    │
    ▼ Stage 1: data_generation
Validated raw data
    │
    ▼ Stage 2: data_cleaning
Cleaned OHLCV (resampled, gaps filled, outliers handled)
    │
    ▼ Stage 3: feature_engineering
DataFrame with ~150+ computed features (indicators, wavelets, MTF, regime)
    │
    ▼ Stage 4: initial_labeling
DataFrame + triple-barrier labels (label_h5, label_h10, label_h15, label_h20)
    │
    ▼ Stage 5: ga_optimize
Optimized barrier parameters
    │
    ▼ Stage 6: final_labels
DataFrame + optimized labels + sample weights + quality scores
    │
    ▼ Stage 7: create_splits
Train/val/test indices (chronological, with purge/embargo gaps)
    │
    ▼ Stage 7.5: feature_scaling
Scaled DataFrames (RobustScaler, fit on train only, clip ±5.0)
+ feature_scaler.pkl saved
    │
    ▼ Stage 7.6: build_datasets
Per-feature-set, per-horizon dataset parquet files
+ feature_set_manifest.json + dataset_manifest.json
    │
    ▼ Stages 7.7-9: validation & reporting
    │
    ════════════════════════════════════════
    │   PIPELINE OUTPUTS COMPLETE
    │   Downstream: Training / Adapter Phase
    ════════════════════════════════════════
    │
    ▼ UnifiedDataPreparation.prepare() OR AdapterFactory.prepare_data()
    │
    ├─ Split chronologically (purge/embargo)
    ├─ Route to adapter by MODEL_ADAPTER_MAP
    │   ├─ TabularAdapter  → 2D (n, features)
    │   ├─ SequenceAdapter → 3D (n, seq_len, features)
    │   └─ MultiStreamAdapter → 4D (n, n_tf, seq_len, features)
    ├─ AdapterScaler: fit on train, transform all (RobustScaler default)
    │   └─ Reshapes 3D/4D → 2D for sklearn, then restores shape
    └─ PreparedData with X_train, X_val, X_test, scaler, indices
```

### Two Scaling Stages

**Important**: There are TWO scaling points in the full pipeline:

1. **Pipeline Stage 7.5** (`FeatureScaler`): Scales the pipeline output parquet files. Saved as `feature_scaler.pkl`.
2. **Adapter Scaling** (`AdapterScaler`): Scales adapter-transformed numpy arrays. Applied via `UnifiedDataPreparation` when `apply_scaling=True`.

This means data may be **double-scaled** if both are applied. The `UnifiedDataPreparation.prepare()` method applies its own scaling via `AdapterScaler` after adapter transformation. If the input data has already been through Stage 7.5 scaling, there would be redundant scaling.

**Resolution**: In practice, the pipeline produces scaled parquet files at Stage 7.5. The adapter phase can be used with `apply_scaling=False` if consuming pre-scaled data. However, this is not enforced or documented clearly.

---

## Reproducibility Gaps

### GAP 1: Double Scaling Risk (MEDIUM)

**Issue**: Pipeline Stage 7.5 scales features in the parquet files. `UnifiedDataPreparation.prepare()` applies `AdapterScaler` again by default (`apply_scaling=True`). If training code consumes pre-scaled pipeline output through `UnifiedDataPreparation`, features get scaled twice.

**Impact**: Incorrect feature values for model training. Whether this actually happens depends on which code path is used for training.

**Mitigation**: Need to verify the actual training code path. If it uses `UnifiedDataPreparation` with pre-scaled data, `apply_scaling` must be `False`.

### GAP 2: Feature Column Auto-Detection Not Deterministic Across Environments (LOW-MEDIUM)

**Issue**: Both `BaseAdapter._get_feature_columns()` and pipeline's `_identify_feature_columns()` use heuristic exclusion lists (prefixes, exact names) to auto-detect features. If a new column is added to the DataFrame (e.g., a metadata column not in the exclusion list), it silently becomes a "feature."

**Impact**: Feature set mismatch between training and inference if DataFrame column sets differ.

**Mitigation**: `FeatureManifest` exists and can be used for explicit feature lists. `BaseAdapter.from_manifest()` factory method exists. However, it's not clear that all code paths use it.

### GAP 3: Adapter Scaler Not Serialized in Standard Training Path (MEDIUM)

**Issue**: `UnifiedDataPreparation` creates `AdapterScaler` instances stored in `self._scalers` dict. These are accessible via `get_scaler()` but there's no automatic serialization. The `AdapterScaler` class has `save()`/`load()` methods, but the responsibility to call them falls on the training orchestrator.

**Impact**: If the training code doesn't explicitly save the adapter scaler, the inference pipeline can't reproduce the exact same scaling.

**Mitigation**: The `ModelBundle` / inference system uses the Pipeline Stage 7.5 scaler (`feature_scaler.pkl`), which IS automatically saved. But this is a different scaler from the adapter-level one.

### GAP 4: Sequence Windowing Parameters Not in Bundle (LOW)

**Issue**: Sequence adapters use `sequence_length`, `stride`, and `symbol_column` parameters. These are available in `PipelineConfig` and `ModelContract`, but the `ModelBundle` only stores `sequence_length` in metadata. `stride` and `symbol_column` are not persisted.

**Impact**: At inference time, if the stride was non-default during training, the bundle doesn't record it. In practice, stride=1 is the default and typical usage, so impact is low.

### GAP 5: Feature Engineering Parameters Partially Captured (MEDIUM)

**Issue**: The `PreprocessingGraph` (`src/inference/preprocessing_graph.py`) captures indicator periods, wavelet config, MTF config, regime config, and scaling config. However, creating the graph requires manually calling `PreprocessingGraph.from_pipeline_config()` and saving it with the bundle.

**Impact**: If the preprocessing graph is not generated during training, inference can't reproduce the exact feature engineering. The feature manifest captures per-feature params but doesn't include the full computation code.

**Mitigation**: The `ModelBundle` has a `BUNDLE_PREPROCESSING_GRAPH_FILE` slot and `has_preprocessing_graph` metadata flag. The infrastructure exists but relies on the training orchestrator to populate it.

### GAP 6: Multi-Stream Adapter Data Source Ambiguity (LOW)

**Issue**: `MultiStreamAdapter` can load data from three sources: `additional_dfs`, raw MTF store, or legacy `data_dir`. At inference time, the bundle doesn't record which source was used or how to access multi-TF data.

**Impact**: 4D models (PatchTST, iTransformer) need multi-TF data at inference time. The inference code must know to provide the same timeframes. `ModelBundle` metadata stores `n_timeframes` and `requires_4d` but not the specific timeframe list.

**Mitigation**: `ModelContract` specifies `mtf_timeframes` per model, providing a default. But if custom timeframes were used during training, they need to be recorded.

### GAP 7: Pipeline vs Adapter Feature Exclusion List Inconsistency (LOW)

**Issue**: The pipeline's `_identify_feature_columns()` uses `EXCLUDED_COLUMNS` and `EXCLUDED_PREFIXES` from `src/data/pipeline/constants.py`. The adapter's `BaseAdapter._get_feature_columns()` has its own inline exclusion list. These are similar but not identical (e.g., adapter also excludes `mtf_raw_*` prefix, `target_*` prefix, `meta_*` prefix).

**Impact**: If both auto-detection mechanisms are used in the same flow, they could select slightly different feature sets. In practice, explicit `feature_columns` from manifests should be used to avoid this.

### GAP 8: No End-to-End Preprocessing Replay Mechanism (HIGH)

**Issue**: While individual components exist (FeatureManifest, PreprocessingGraph, FeatureScaler, AdapterScaler), there is no single function that takes raw OHLCV data and produces model-ready arrays by replaying the exact training preprocessing. The `PreprocessingGraph.transform()` method exists in concept but the actual feature computation code in `src/data/features/compute/` is not invoked through the graph at inference time.

**Impact**: Production inference currently requires either (a) pre-computed features in the same format as training, or (b) manual re-implementation of the feature engineering steps. This is the largest gap for going from research to production.

**Mitigation**: The `PreprocessingGraph` class captures the configuration, and the feature computation modules exist. What's missing is the orchestration layer that connects `PreprocessingGraph.transform()` to the actual feature computation code.

---

## Summary

| Component | Status | Files |
|-----------|--------|-------|
| Pipeline stages | Well-defined, 12 stages with dependencies | `stage_registry.py`, `runner.py` |
| Pipeline scaling | Saved, validated, leakage-checked | `stages/scaling/run.py` |
| Adapter routing | Clean registry pattern, 3 adapter types | `registry.py`, `factory.py` |
| Adapter transforms | Correct for all ranks (2D/3D/4D) | `tabular.py`, `sequence.py`, `multi_stream.py` |
| Adapter scaling | Exists but not auto-saved | `scaling.py` |
| Feature manifest | Rich metadata, reproducibility checking | `feature_manifest.py` |
| Preprocessing graph | Config capture exists, execution gap | `preprocessing_graph.py` |
| Feature selection | Optuna-based, 3 strategies | `selection.py` |
| Inference pipeline | Bundle-based, supports ensemble | `pipeline.py`, `bundle.py` |

### Critical Path for MGC Readiness

1. **Clarify double-scaling** (GAP 1): Document which code path is actually used in training
2. **Ensure preprocessing graph is populated** (GAP 5/8): Training orchestrator must call `PreprocessingGraph.from_pipeline_config()` and save it with the bundle
3. **Add feature columns to bundle** (GAP 2): Always use manifest-based feature columns, never auto-detect at inference time
4. **Add MTF timeframes to bundle metadata** (GAP 6): Record actual timeframes used during training
