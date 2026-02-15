# Inference Pipeline Architecture Audit

**Date:** 2026-02-15
**Scope:** `src/inference/` — all modules, data flow, and architecture

---

## Overview

The `src/inference/` package provides end-to-end inference capabilities for the ML Factory pipeline. It spans from model serialization (bundles) through prediction serving (HTTP API) and strategy evaluation (backtesting). The architecture is layered with clear separation of concerns:

**Layer 1 — Serialization:** `bundle.py`, `ensemble_bundle.py`, `preprocessing_graph.py`
**Layer 2 — Orchestration:** `orchestrator.py`, `pipeline.py`, `builder.py`
**Layer 3 — Execution:** `batch.py`, `server.py`
**Layer 4 — Evaluation:** `backtesting/` (backtest, costs, metrics, equity_curve, position_sizing, execution)
**Layer 5 — Monitoring:** `production/monitor.py`

**Total files:** 18 Python files across 3 directories
**Total exports:** ~90 symbols in `__init__.py`

---

## Module Map

### Core Inference Modules

| Module | Purpose | Key Classes | Lines |
|--------|---------|-------------|-------|
| `bundle.py` | Serializable model container | `ModelBundle`, `BundleMetadata`, `BundleManifest` | ~1107 |
| `ensemble_bundle.py` | Stacking ensemble container | `EnsembleBundle`, `EnsembleBundleMetadata`, `AlignmentConfig` | ~922 |
| `preprocessing_graph.py` | Train/serve parity preprocessing | `PreprocessingGraph`, `PreprocessingGraphConfig`, 6 sub-configs | ~885 |
| `orchestrator.py` | Unified inference entry point | `InferenceOrchestrator`, `PredictionResult` | ~785 |
| `pipeline.py` | High-level prediction interface | `InferencePipeline`, `InferenceResult`, `EnsembleResult` | ~452 |
| `builder.py` | Bridge training → bundles | `BundleBuilder`, `BundleBuildResult` | ~784 |
| `batch.py` | Large dataset processing | `BatchPredictor`, `BatchInference`, `BatchResult` | ~684 |
| `server.py` | HTTP API (FastAPI) | `ModelServer`, `ServerConfig`, request/response models | ~581 |

### Backtesting Modules

| Module | Purpose | Key Classes | Lines |
|--------|---------|-------------|-------|
| `backtest.py` | Main simulation loop | `Backtester`, `BacktestConfig`, `BacktestResult`, `Position` | ~833 |
| `costs.py` | Transaction cost modeling | `TransactionCosts`, `CostCalculator`, 4 slippage models | ~539 |
| `equity_curve.py` | Equity tracking + visualization | `EquityCurve`, `Trade` | ~663 |
| `metrics.py` | Performance metrics suite | `PerformanceMetrics`, 14 metric functions | ~700 |
| `position_sizing.py` | Position sizing algorithms | 6 sizer classes + `BetSizingPositioner` | ~631 |
| `execution.py` | Market hours + adverse selection | `MarketHoursFilter`, `CMECalendar` | ~233 |

### Production Monitoring

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `production/monitor.py` | Drift, freshness, performance tracking | `ProductionMonitor`, `ModelHealthMetrics` |

---

## Data Flow

### 1. Training → Bundle Creation Flow

```
TrainingRunResult (PHASE_3)
    ↓
BundleBuilder.build_from_training_result()
    ├── Extracts: model, scaler, feature_columns, calibrator
    ├── Creates: PreprocessingGraph from PipelineConfig
    ├── Creates: ModelBundle.from_training(...)
    └── Saves: bundle_dir/{metadata,features,scaler,model/,...}.json|pkl

EnsembleResult (PHASE_4)
    ↓
BundleBuilder.build_ensemble_bundle()
    ├── Extracts: meta-learner, stacking features, alignment config
    └── Saves: ensemble_dir/{metadata,stacking_features,base_bundles,...}
```

### 2. Bundle → Prediction Flow (Pre-computed Features)

```
X (DataFrame/ndarray)
    ↓
InferenceOrchestrator.predict(X)
    ↓
ModelBundle.predict(X)
    ├── _prepare_input(): validate columns, reorder, convert to ndarray
    ├── Scaler: transform (handles 2D/3D/4D shapes)
    ├── Model: model.predict(X_array)
    └── Calibrator: _apply_calibration() (optional)
    ↓
PredictionResult(class_predictions, class_probabilities, confidence)
```

### 3. Raw OHLCV → Prediction Flow (End-to-End)

```
raw_ohlcv_df [datetime, open, high, low, close, volume]
    ↓
InferenceOrchestrator.predict_from_raw(raw_df)
    ↓
PreprocessingGraph.transform(raw_df)
    ├── _validate_input(): check OHLCV columns
    ├── _apply_cleaning(): resample (1min → 5min)
    ├── _apply_features(): 25+ indicator functions
    ├── _apply_mtf(): MTFFeatureGenerator
    ├── _apply_regime(): regime detection
    ├── dropna()
    ├── _apply_scaling(): fitted scaler
    └── Select feature columns
    ↓
features_df
    ↓
ModelBundle.predict(features_df) → PredictionResult
```

### 4. Ensemble Prediction Flow

```
X (features)
    ↓
EnsembleBundle.predict_from_base_features(X)
    ├── Load base ModelBundles (lazy)
    ├── For each base model: bundle.predict(X) → probabilities
    ├── _stack_predictions(): concatenate + add derived features
    │   ├── If same length: _simple_stack() → hstack + confidence + agreement
    │   └── If different: OOFAligner alignment
    ├── Apply scaler (optional)
    └── meta_learner.predict(stacking_X) → PredictionResult
```

### 5. Batch Inference Flow

```
Large DataFrame or parquet path
    ↓
BatchPredictor.predict_batch()
    ├── Load data if path
    ├── Chunk into batches (default 10,000)
    ├── For each batch: pipeline.predict() → InferenceResult
    ├── Format to DataFrame with predictions + confidence
    ├── Track progress via callback
    └── Concatenate all → BatchResult
    ↓
predictions_df.to_parquet(output_path)
```

### 6. Parallel Ensemble Inference Flow

```
X (features) + list of ModelBundle
    ↓
BatchInference.predict_batch(X)
    ├── ThreadPoolExecutor(max_workers=n_jobs)
    ├── Submit all model predictions in parallel
    ├── Collect results as they complete
    ├── Stack probabilities (n_samples, n_models * n_classes)
    └── BatchInferenceResult with stacked_probabilities
    ↓
stacked_probabilities → EnsembleBundle.predict(base_predictions)
```

---

## Single vs Ensemble Inference

### Single Model

- **Entry point:** `InferenceOrchestrator.predict(X)` or `ModelBundle.predict(X)`
- **Flow:** Input → validate → scale → model.predict → calibrate → PredictionResult
- **Handles:** 2D (tabular), 3D (sequences for LSTM/GRU), 4D (multi-timeframe for PatchTST/iTransformer)
- **Scaling:** Per-dimension reshape-scale-reshape for 3D/4D inputs

### Ensemble

Two paths exist:

1. **EnsembleBundle (stacking):** Meta-learner trained on OOF predictions from base models. Uses `OOFAligner` for heterogeneous model alignment (different sequence lengths → different output lengths). Adds derived features (mean confidence, prediction agreement) to stacking input.

2. **InferencePipeline.predict_ensemble():** Simple voting (soft_vote, hard_vote, weighted) across multiple ModelBundles. No meta-learner, just probability averaging or majority voting.

3. **InferenceOrchestrator fallback:** When no ensemble bundle exists but multiple models are loaded, uses simple probability averaging.

---

## Gaps & Missing Pieces

### Critical for Universal Inference

1. **No automatic adapter selection at inference time.** The `PreprocessingGraph.transform()` generates a flat DataFrame, but sequence/4D models need adapter-specific reshaping (via `get_sequences_3d()` or `get_multi_resolution_4d()`). The `ModelBundle.predict()` validates shape but doesn't transform — caller must provide correct shape. This means **raw OHLCV → prediction for neural models is incomplete**: `predict_from_raw()` generates 2D features but neural models need 3D/4D tensors.

2. **PreprocessingGraph doesn't integrate with adapters.** The graph applies indicator computations but doesn't know about `ModelDataAdapter`, `SequenceAdapter`, or `TensorAdapter`. For true end-to-end inference with neural/transformer models, the graph needs to produce adapter-specific outputs.

3. **Feature selection not captured in PreprocessingGraph.** If Optuna feature selection was used during training (selecting a subset of features), the graph doesn't replay that selection — it relies on `feature_columns` in the bundle, but the graph generates ALL features and then filters. If feature selection happened before scaling, ordering could differ.

### Important but Not Blocking

4. **EnsembleBundle.predict() return type is `Any`.** Should be `PredictionResult` for type safety.

5. **InferenceOrchestrator imports PredictionResult from two locations.** Line 45 imports from `src.core.interfaces`, but `_predict_with_ensemble()` at line 530 imports from `src.models.base`. These could be different classes (or re-exports of the same). Potential name collision.

6. **ProductionMonitor accesses `bundle._metadata` (private).** Should use `bundle.metadata` (public attribute). See `monitor.py:165,168,295,297`.

7. **No streaming/real-time inference mode.** `BatchPredictor.predict_streaming()` exists but is batch-oriented (loads all data then yields chunks). True streaming from a live data feed isn't supported.

8. **Server doesn't support EnsembleBundle directly.** `ModelServer` wraps `InferencePipeline` (which uses `ModelBundle`), not `EnsembleBundle` or `InferenceOrchestrator`. The `/predict_ensemble` endpoint does simple voting, not stacking ensemble.

9. **No model versioning or A/B testing support.** Bundles have version numbers but no infrastructure for comparing model versions in production.

### Minor

10. **Backtester `run()` method re-imports `logging` inside the loop** (lines 664, 688, 699) — should use module-level logger.

11. **`_validate_predictions` in backtest.py loses datetime index** by calling `reset_index(drop=True)`. Timestamps are preserved via the "timestamp" column, but this could cause confusion.

---

## Dead Code / Issues

### Unused / Questionable

1. **`BundleManifest` loaded but not used during `ModelBundle.load()`** (line 625): The manifest is loaded and parsed but the result is immediately discarded — checksums are never validated on load.

2. **`EnsembleBundleManifest` similarly discarded during `EnsembleBundle.load()`** (line 524).

3. **`validate_distribution()` in `bundle.py`** references `self._training_stats` which is never populated. The method always returns `True, ["No training stats available"]`. Effectively dead code.

4. **`time.time()` called without using result** in `batch.py:195` (`time.time()` inside the batch loop, return value not captured).

5. **`time.perf_counter()` called without using result** in `server.py:378` (ensemble endpoint, `start_time` assigned but not used for timing).

6. **`PreprocessingGraph._apply_regime()`** is a no-op — just returns `df` unchanged. Regime features are already added in `_apply_features()`. The method exists but does nothing.

7. **CMECalendar defined twice** — once in `execution.py` as a fallback, and the real one imported from `src.data.pipeline.stages.sessions`. The local fallback shadows the import.

### Architectural Inconsistencies

8. **Two PredictionResult classes.** `orchestrator.py` imports from `src.core.interfaces` while `pipeline.py` and `_predict_with_ensemble()` import from `src.models.base`. Need to verify these are the same class or consolidate.

9. **InferencePipeline vs InferenceOrchestrator overlap.** Both provide similar functionality (load bundles, predict, predict ensemble). The `__init__.py` docstring says orchestrator is "THE single entry point" but pipeline is still fully functional and used by server/batch. This is a design tension — pipeline is lower-level, orchestrator is higher-level, but the boundary isn't always clear.

10. **`BundleBuilder._extract_model()` uses attribute probing.** Tries `model`, `_model`, `estimator`, `_estimator`, `get_model()` — fragile coupling to trainer internals.

---

## Dependencies Map

```
bundle.py → src.models.base (PredictionResult, BaseModel)
           → src.models.registry (ModelRegistry)
           → src.core.contracts (FeatureSpec, get_model_contract) [optional]

ensemble_bundle.py → src.core (OOFResult, PipelineConfig)
                    → src.data.adapters (OOFAligner)
                    → src.models.ensemble (get_meta_learner)

orchestrator.py → src.core (PipelineConfig)
                → src.core.interfaces (PredictionResult)
                → src.inference.bundle (ModelBundle)
                → src.inference.ensemble_bundle (EnsembleBundle)
                → src.models.ensemble (meta-learner classes)
                → src.models.base (PredictionResult) [in _predict_with_ensemble]

pipeline.py → src.inference.bundle (ModelBundle)
            → src.models.base (PredictionResult)

builder.py → src.core (PipelineConfig)
           → src.inference.bundle (ModelBundle)
           → src.inference.preprocessing_graph (PreprocessingGraph)

preprocessing_graph.py → src.data.pipeline.stages.features.* (25+ feature functions)
                       → src.data.pipeline.stages.mtf.generator (MTFFeatureGenerator)

batch.py → src.inference.pipeline (InferencePipeline)

server.py → src.inference.pipeline (InferencePipeline)
          → fastapi, uvicorn, pydantic [optional]
          → prometheus_client [optional]

production/monitor.py → src.inference.bundle (ModelBundle) [TYPE_CHECKING]
                      → src.validation.monitoring (FeatureDriftMonitor)

backtesting/backtest.py → backtesting.costs, equity_curve, execution, metrics, position_sizing
backtesting/execution.py → src.data.pipeline.stages.sessions (CMECalendar)
backtesting/position_sizing.py → src.models.training.meta_labeling.bet_sizing [optional]
```

---

## Summary Assessment

**Strengths:**
- Well-structured bundle serialization with manifest, metadata, checksums
- Comprehensive backtesting with realistic costs, slippage models, circuit breakers
- Multiple inference entry points (orchestrator, pipeline, batch, server)
- Stacking ensemble with OOF alignment for heterogeneous models
- PreprocessingGraph ensures train/serve parity for feature engineering
- Production monitoring with drift detection and alerting hooks

**Key Gap:**
The critical missing piece is **adapter integration in the inference path for neural/transformer models**. The `predict_from_raw()` flow produces 2D features, but 3D/4D models need adapter-specific reshaping. Currently, callers must manually prepare tensors for neural models, breaking the "universal" inference goal.

**Recommended Priority Actions:**
1. Integrate adapter selection into `ModelBundle.predict_from_raw()` based on `requires_sequences`/`requires_4d` metadata
2. Consolidate `PredictionResult` to one canonical location
3. Fix `ProductionMonitor` to use public `bundle.metadata` instead of `bundle._metadata`
4. Validate bundle checksums on load (currently skipped)
5. Remove dead code (`validate_distribution`, unused `time.time()` calls, no-op `_apply_regime`)
