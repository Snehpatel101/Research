# Training Orchestrator & Model Output Flow Audit

## Overview

The ML Factory training system has a clear three-tier orchestration hierarchy:

1. **MLFactory** (`src/factory.py`) - Top-level experiment runner using `ExperimentConfig`
2. **MLPipeline** (`src/orchestrator.py`) - Deprecated thin wrapper, delegates to orchestrator
3. **UnifiedTrainingOrchestrator** (`src/models/training/unified_orchestrator.py`) - THE core training engine using `PipelineConfig`

All roads lead to `UnifiedTrainingOrchestrator.train()`, which is the single entry point for all model training.

---

## Training Flow (Complete Orchestration)

### Entry Points

| Entry Point | Config Type | Status |
|-------------|------------|--------|
| `MLFactory.run()` | `ExperimentConfig` → converts to `PipelineConfig` | **Active** (recommended) |
| `MLPipeline.run()` | `PipelineConfig` | **Deprecated** (emits warning) |
| `UnifiedTrainingOrchestrator.train()` | `PipelineConfig` | **Core engine** |
| `train_pipeline()` convenience function | `PipelineConfig` | **Active** |
| `train_meta_labeling()` convenience function | `PipelineConfig` | **Active** |

### MLFactory Flow (4 phases)

```
MLFactory.run()
  ├── Phase 1: Data Pipeline (_run_data_pipeline)
  │     ├── Load raw OHLCV from parquet
  │     ├── FeatureEngineer.engineer_features()
  │     └── TripleBarrierLabeler.create_labels() per horizon
  ├── Phase 2: Training (_run_training)
  │     ├── config.to_pipeline_config()
  │     └── UnifiedTrainingOrchestrator(pipeline_config).train(df)
  ├── Phase 3: Evaluation (_run_evaluation)  [optional]
  │     └── Backtester using OOF predictions
  └── Phase 4: Bundling (_create_bundle)  [optional]
        └── BundleBuilder.build_from_training_result()
```

### UnifiedTrainingOrchestrator.train() Flow

```
train(df, additional_dfs, generate_financial_report)
  ├── Pre-training Validation
  │     ├── Contract validation (feature count vs model limits)
  │     ├── Leakage detection (correlation-based)
  │     └── Lookahead audit (resample config check)
  ├── Route by TrainingMode
  │     ├── STANDARD → _train_standard()
  │     ├── WALK_FORWARD → _train_walk_forward()
  │     ├── REGIME_AWARE → _train_regime_aware()
  │     └── META_LABELING → _train_meta_labeling()
  ├── Ensemble Building (if build_ensemble=True and >1 model)
  │     ├── EnsembleService.build_ensemble()
  │     ├── DiversityAnalyzer.analyze()
  │     └── Clear OOF predictions from memory
  ├── Save Results (ArtifactManager.save_all)
  ├── Generate Financial Reports [optional]
  └── Return TrainingRunResult
```

### Standard Training Mode Detail

```
_train_standard(df)
  FOR each horizon:
    ├── Separate models into boosting vs sequential
    ├── Boosting models (>=2): _train_boosting_parallel()
    │     ├── Prepare data with cache (all share rank 2)
    │     ├── ParallelTrainingService.train_models_parallel()
    │     ├── Calibrate each model
    │     ├── Generate OOF for each
    │     └── Store results
    ├── Sequential models (neural/transformer):
    │     FOR each model:
    │       ├── _prepare_with_cache(df, model_name)
    │       ├── _train_single_model()
    │       │     ├── ModelTrainingService.train_model()
    │       │     └── Calibrate if auto_calibrate=True
    │       └── Generate OOF if save_oof=True
    └── Clear PreparedData cache (free memory)
```

### Services (Delegated Components)

| Service | File | Responsibility |
|---------|------|---------------|
| `DataPreparer` | `services/data_preparer.py` | Wraps UnifiedDataPreparation, handles adapters |
| `ModelTrainingService` | `services/model_training.py` | Train individual models |
| `ParallelTrainingService` | `services/parallel_training.py` | Parallel boosting training |
| `OOFGenerationService` | `services/oof_generation.py` | Generate out-of-fold predictions |
| `EnsembleService` | `services/ensemble_service.py` | Build stacking ensembles |
| `ArtifactManager` | `services/artifact_persistence.py` | Save/load all artifacts |
| `HyperparameterTuningService` | `services/hyperparameter_tuning.py` | Optuna HPO |

---

## Artifacts Produced After Training

### Directory Structure (output_dir / run_id)

```
{output_dir}/run_{mode}_{timestamp}/
  ├── config.json                    # PipelineConfig serialized
  ├── metrics_summary.json           # All model metrics
  ├── models/                        # Trained model files
  │     ├── xgboost_h20.pkl
  │     ├── lightgbm_h20.pkl
  │     └── lstm_h20.pkl
  ├── oof/                           # OOF predictions (if save_oof=True)
  │     ├── xgboost_h20_oof.parquet
  │     └── lightgbm_h20_oof.parquet
  ├── oof_cache/                     # Temporary OOF cache
  ├── h20/                           # Per-horizon training artifacts
  ├── reports/                       # Financial reports (if enabled)
  │     └── {model_key}/
  └── bundles/                       # Inference bundles (if bundling enabled)
        ├── xgboost_h20/
        ├── lightgbm_h20/
        └── ensemble/
```

### ArtifactManager Saves

| Artifact | Format | Method |
|----------|--------|--------|
| Config | JSON (`config.json`) | `save_config()` |
| Metrics | JSON (`metrics_summary.json`) | `save_metrics()` |
| OOF predictions | Parquet (`{key}_oof.parquet`) | `save_oof_predictions()` |
| Trained models | Pickle (`{key}.pkl`) via `trainer.save()` | `save_models()` |

### In-Memory Results (TrainingRunResult)

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

@dataclass
class ModelTrainingResult:
    model_name: str
    horizon: int
    metrics: dict[str, float]        # val_f1, val_accuracy, etc.
    oof_prediction: OOFPrediction | None
    trainer: Any | None              # Trainer instance (has .model, .save())
    training_time_seconds: float
    n_features: int
    data_rank: int                   # 2, 3, or 4
```

---

## Ensemble Construction

### Flow

```
_build_ensemble(df)
  ├── Check _oof_predictions not empty
  ├── EnsembleService.build_ensemble(EnsembleRequest)
  │     ├── OOF alignment (AlignedOOFResult)
  │     ├── StackingDataset creation
  │     └── Meta-learner training (ridge_meta, logistic, etc.)
  ├── DiversityAnalyzer.analyze()
  │     ├── Q-statistic (Yule's Q)
  │     ├── Pairwise correlation
  │     ├── Disagreement rate
  │     └── Composite diversity score
  └── Return (aligned_oof, stacking_dataset, ensemble_result)
```

### Requirements for Ensemble
- `build_ensemble=True` in config
- More than 1 model trained
- OOF predictions generated (`save_oof=True`)
- OOF predictions stored in `_oof_predictions` dict

### Post-Ensemble Cleanup
- OOF predictions cleared from memory after ensemble building (Phase 37 fix)
- Each OOF dict entry is 50-200MB; clearing saves 750MB-1.5GB

---

## What Training Produces vs What Inference Needs

### Training Produces (TrainingRunResult)

| Component | Where | Format |
|-----------|-------|--------|
| Trained models | `model_results[key].trainer` | In-memory Trainer objects |
| Model metrics | `model_results[key].metrics` | Dict of floats |
| OOF predictions | `model_results[key].oof_prediction` | OOFPrediction (cleared after ensemble) |
| Ensemble meta-learner | `ensemble_result.trainer` | In-memory model |
| Stacking dataset | `stacking_dataset` | StackingDataset with aligned predictions |
| Aligned OOF | `aligned_oof` | AlignedOOFResult with common indices |
| Saved models on disk | `{output_dir}/models/*.pkl` | Pickle files via trainer.save() |
| Config | `{output_dir}/config.json` | JSON |

### Inference Needs (ModelBundle)

| Component | Required? | Source |
|-----------|-----------|--------|
| Trained model (BaseModel) | **YES** | Extracted from trainer via `_extract_model()` |
| Feature scaler | **YES** (for most) | Extracted from trainer via `_extract_scaler()` |
| Feature column names | **YES** | Extracted from trainer or fallback to generic |
| Prediction horizon | **YES** | From ModelTrainingResult.horizon |
| Probability calibrator | Optional | Extracted from trainer via `_extract_calibrator()` |
| Preprocessing graph | Optional | Created from PipelineConfig |
| FeatureSpec | Optional | Must be passed in explicitly |
| Symbol | Optional | From config |
| Training metrics | Optional | From ModelTrainingResult.metrics |

### For EnsembleBundle

| Component | Required? | Source |
|-----------|-----------|--------|
| Meta-learner | **YES** | From EnsembleResult |
| Base model bundle paths | **YES** | Created by BundleBuilder |
| Stacking feature names | **YES** | From AlignedOOFResult |
| Alignment config | **YES** | From AlignedOOFResult |
| Scaler for stacking | Optional | Not currently passed |

---

## Training → Inference Gap Analysis

### What Works (Bridge Exists)

1. **BundleBuilder.build_from_training_result()** - Bridges TrainingRunResult to ModelBundle
   - Extracts model, scaler, feature columns from trainer
   - Creates preprocessing graph from config
   - Saves as standardized bundle directory

2. **BundleBuilder.build_ensemble_bundle()** - Bridges EnsembleResult to ensemble metadata
   - Saves meta-learner, stacking dataset, aligned OOF info

3. **MLFactory Phase 4** - Automatically calls BundleBuilder after training

4. **ModelBundle.predict()** - Full inference with scaling, calibration
5. **ModelBundle.predict_from_raw()** - End-to-end raw OHLCV → prediction
6. **InferencePipeline** - Multi-model inference with soft/hard voting
7. **EnsembleBundle** - Stacking ensemble inference with alignment

### Gaps and Issues

#### GAP 1: BundleBuilder Extraction Fragility
- `_extract_model()` tries: `model`, `_model`, `estimator`, `_estimator`, `get_model()`
- `_extract_scaler()` tries: `scaler`, `_scaler`, `feature_scaler`, `_feature_scaler`
- `_extract_feature_columns()` tries: `feature_columns`, `_feature_columns`, `feature_names`, `_feature_names`
- `_extract_calibrator()` tries: `calibrator`, `_calibrator`, `prob_calibrator`
- **Risk**: If trainer attribute naming changes, extraction silently fails and model is skipped

#### GAP 2: FeatureSpec Not Auto-Populated
- BundleBuilder accepts `feature_specs` parameter but it must be passed in explicitly
- No automatic generation of FeatureSpec from training config
- Without FeatureSpec, inference can't guarantee same 5-dimension optimization params

#### GAP 3: Ensemble Bundle Incomplete Bridge
- `BundleBuilder.build_ensemble_bundle()` saves metadata/stacking dataset but the meta-learner extraction is fragile (tries `_ensemble` then `ensemble` attribute)
- The training orchestrator's ensemble uses `EnsembleService` which returns `EnsembleServiceResult`, but `build_ensemble_bundle()` expects `EnsembleResult` from `src/models/ensemble/orchestrator.py` - **type mismatch potential**

#### GAP 4: Walk-Forward and Regime Models Not Bundled
- Walk-forward training stores results but no walk-forward-specific bundling
- Regime-aware training stores per-regime models, but bundling doesn't handle regime routing
- Meta-labeling stores primary + meta model pair, but bundling expects single models

#### GAP 5: Preprocessing Graph Hardcoded Assumptions
- `BundleBuilder._create_preprocessing_graph()` hardcodes: source_timeframe="1min", target_timeframe="5min", scaler_type="robust"
- Should pull these from PipelineConfig or data pipeline config

#### GAP 6: No Automatic Bundle Building in Standard Training
- `UnifiedTrainingOrchestrator` does NOT call BundleBuilder
- Only `MLFactory` (Phase 4) calls it, and it's optional
- If using `train_pipeline()` directly, no bundles are created

#### GAP 7: Calibrator Attachment During Training (Partial)
- Calibrator IS stored on `Trainer.calibrator` and `BundleBuilder._extract_calibrator()` CAN find it via duck-typing
- However, the orchestrator also sets calibrator on the service result object (L993) which is LOST during `ModelTrainingResult` conversion (L912-920) since that dataclass has no `calibrator` field
- Net effect: calibrator transfer works for direct Trainer access but fails in the orchestrator conversion path

---

## Bridge Requirements (What's Needed)

### Critical (Must Have for Automated Training→Inference)

1. **Standardize Trainer Interface** - Define explicit protocol/interface for what trainer objects must expose (model, scaler, feature_columns, calibrator) instead of duck-typing with fallback chains

2. **Auto-Bundle After Training** - Add optional `build_bundle=True` to `UnifiedTrainingOrchestrator` so bundles are created as part of training, not as a separate step

3. **FeatureSpec Auto-Generation** - Generate FeatureSpec from PipelineConfig + training results automatically, so inference parity is guaranteed

### Important (Should Have)

4. **Regime Model Bundling** - Extend ModelBundle or create RegimeBundle that stores per-regime models with regime detection config

5. **Meta-Labeling Bundle** - Create MetaLabelingBundle that stores primary + meta model pair with threshold

6. **Walk-Forward Bundle** - Store walk-forward windows config alongside models

7. **Fix Calibrator Transfer in Orchestrator Path** - Calibrator IS on Trainer.calibrator (BundleBuilder finds it), but is LOST during orchestrator's ModelTrainingResult conversion (L912-920). Add `calibrator` field to `ModelTrainingResult` dataclass or ensure BundleBuilder always accesses Trainer directly.

### Nice to Have

8. **Preprocessing Graph from Config** - Pull preprocessing params from PipelineConfig instead of hardcoding

9. **Bundle Validation Integration** - Run bundle validation as part of training completion, not just on-demand

10. **Ensemble Type Alignment** - Ensure EnsembleService result type matches what BundleBuilder.build_ensemble_bundle() expects

---

## Existing "Export to Inference" Functionality

### What Exists

| Component | Location | Status |
|-----------|----------|--------|
| `BundleBuilder` | `src/inference/builder.py` | **Functional** - Creates ModelBundle from TrainingRunResult |
| `ModelBundle` | `src/inference/bundle.py` | **Functional** - Full save/load/predict with preprocessing graph |
| `EnsembleBundle` | `src/inference/ensemble_bundle.py` | **Functional** - Stacking ensemble with alignment |
| `InferencePipeline` | `src/inference/pipeline.py` | **Functional** - Multi-model orchestration |
| `PreprocessingGraph` | `src/inference/preprocessing_graph.py` | **Exists** - Train/serve parity |
| `ModelBundle.package_bundle()` | `src/inference/bundle.py` | **Functional** - Tar.gz packaging for deployment |
| `ModelBundle.extract_bundle()` | `src/inference/bundle.py` | **Functional** - Extract packaged bundles |
| `build_bundles()` | `src/inference/builder.py` | **Functional** - Convenience function |
| `build_from_run()` | `src/inference/builder.py` | **Functional** - Build from completed run directory |
| `MLFactory._create_bundle()` | `src/factory.py` | **Functional** - Auto-bundle in factory flow |

### What's Missing

| Gap | Description |
|-----|-------------|
| Direct orchestrator→bundle | No auto-bundling in UnifiedTrainingOrchestrator |
| Regime bundle | No regime-aware inference bundle |
| Meta-labeling bundle | No primary+meta model pair bundle |
| Walk-forward bundle | No window-aware inference bundle |
| FeatureSpec generation | No auto-generation from training config |
| Calibrator pipeline | Calibrator on Trainer works; lost in orchestrator ModelTrainingResult conversion path |

---

## Summary

The training→inference bridge is **architecturally sound** with BundleBuilder, ModelBundle, EnsembleBundle, and InferencePipeline providing a complete inference stack. The main gaps are:

1. **Fragile extraction** - Duck-typing instead of explicit interfaces for trainer components
2. **Manual bundle step** - Must explicitly call BundleBuilder after training (or use MLFactory)
3. **Special mode gaps** - Regime, meta-labeling, and walk-forward models lack specialized bundles
4. **Calibrator transfer** - Calibrator IS on Trainer.calibrator (works for direct access) but LOST in orchestrator's ModelTrainingResult conversion (L912-920)

The system is ~80% complete for a fully automated train→bundle→infer pipeline. The remaining 20% is mostly about standardizing interfaces and handling special training modes.
