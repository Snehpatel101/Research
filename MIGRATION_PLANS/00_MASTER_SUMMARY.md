# ML Factory Migration Plans - Master Summary

**Created:** 2026-01-18
**Purpose:** Cohesive implementation guide for all phases

---

## Executive Summary

This document provides a unified view of the ML Factory refactoring across all phases. The implementation follows a **clean-slate** approach with no backward compatibility constraints.

---

## Phase Status Overview

| Phase | Name | Status | Completeness |
|-------|------|--------|--------------|
| 0 | Foundation | ✅ Complete | 95% |
| 1 | Unified Features | ✅ Complete | 90% |
| 1B | Labeling & Optimization | ✅ Complete | 90% |
| 2 | Adapter Integration | ✅ Complete | 95% |
| 3 | Training Orchestration | ✅ Complete | 90% |
| 4 | Meta-Learners | ✅ Complete | 95% |
| 5 | Inference | ✅ Complete | 95% |

**Overall Status: ~92% Complete**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PHASE 0: FOUNDATION                          │
│  PipelineConfig │ Types/Enums │ Contracts │ Constants │ Validation  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────────┐    ┌─────────────────────┐    ┌───────────────────┐
│ PHASE 1: FEATURES │    │ PHASE 1B: OPTIMIZE  │    │                   │
│ 162 base features │    │ Labels (100 trials) │    │                   │
│ 12 families       │    │ Features (150 tr)   │    │                   │
│ Model strategies  │    │ Hyperparams (100×N) │    │                   │
└───────────────────┘    └─────────────────────┘    │                   │
        │                           │                │                   │
        └───────────────────────────┼────────────────┘                   │
                                    ▼                                    │
┌─────────────────────────────────────────────────────────────────────┐ │
│                      PHASE 2: ADAPTERS                               │ │
│  TabularAdapter (2D) │ SequenceAdapter (3D) │ MultiStreamAdapter (4D)│ │
│  AdapterFactory │ AdapterScaler │ UnifiedDataPreparation │ OOFAligner│ │
└─────────────────────────────────────────────────────────────────────┘ │
                                    │                                    │
                                    ▼                                    │
┌─────────────────────────────────────────────────────────────────────┐ │
│                    PHASE 3: TRAINING ORCHESTRATION                   │ │
│  UnifiedTrainingOrchestrator │ ModelTrainer │ CVOrchestrator         │ │
│  4 Training Modes │ 4 CV Methods │ OOF Generation                    │ │
└─────────────────────────────────────────────────────────────────────┘ │
                                    │                                    │
                                    ▼                                    │
┌─────────────────────────────────────────────────────────────────────┐ │
│                      PHASE 4: META-LEARNERS                          │ │
│  EnsembleOrchestrator │ HeterogeneousStackingBuilder │ OOF Alignment │ │
│  ridge_meta │ mlp_meta │ xgboost_meta │ calibrated_meta              │ │
└─────────────────────────────────────────────────────────────────────┘ │
                                    │                                    │
                                    ▼                                    │
┌─────────────────────────────────────────────────────────────────────┐ │
│                        PHASE 5: INFERENCE                            │◄┘
│  InferenceOrchestrator │ BundleBuilder │ ModelBundle │ EnsembleBundle│
│  PreprocessingGraph │ BatchPredictor │ ModelServer                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Entry Points

| Phase | Entry Point | Purpose |
|-------|-------------|---------|
| 0 | `PipelineConfig` | Single source of truth for all config |
| 1 | `compute_all_features()` | Feature computation |
| 1B | `OptimizationPipeline` | Label/feature/hyperparam optimization |
| 2 | `UnifiedDataPreparation` | Data prep (split + adapt + scale) |
| 3 | `UnifiedTrainingOrchestrator` | All training modes |
| 4 | `EnsembleOrchestrator` | Ensemble building |
| 5 | `InferenceOrchestrator` | All inference |

---

## Data Flow Summary

```
Raw OHLCV Data
    │
    ▼ [PHASE 1]
compute_all_features() → 162 features
    │
    ▼ [PHASE 1B]
LabelOptimizer → Optimal triple-barrier config
FeatureSelector → Selected features (~50-80)
FeaturePruner → Final features (~30-50)
HyperparameterOptimizer → Best params per model
    │
    ▼ [PHASE 2]
UnifiedDataPreparation.prepare()
├─ Split: train/val/test with purge/embargo
├─ Adapt: TabularAdapter (2D) / SequenceAdapter (3D) / MultiStreamAdapter (4D)
└─ Scale: Fit-on-train only
    │
    ▼ [PHASE 3]
UnifiedTrainingOrchestrator.train()
├─ For each model: CVOrchestrator → ModelTrainer → OOF
└─ TrainingRunResult with all artifacts
    │
    ▼ [PHASE 4]
EnsembleOrchestrator.train()
├─ OOFAligner → Align heterogeneous predictions
├─ HeterogeneousStackingBuilder → Stacking features
└─ Meta-learner training → EnsembleResult
    │
    ▼ [PHASE 5]
BundleBuilder.build_from_training_result()
├─ ModelBundle per model
├─ EnsembleBundle for stacking
└─ PreprocessingGraph for train/serve parity
    │
    ▼
InferenceOrchestrator.predict()
├─ predict(X_features)
├─ predict_from_raw(raw_ohlcv_df)
└─ predict_batch(data)
```

---

## Remaining Work Summary

### Critical (Must Fix)

| Phase | Issue | Priority |
|-------|-------|----------|
| 3 | Circular import in cross_validation | HIGH |
| 1 | MTF feature computation | MEDIUM |

### Important (Should Fix)

| Phase | Issue | Priority |
|-------|-------|----------|
| 0 | Config JSON serialization with enums | MEDIUM |
| 0 | AdapterResult feature bounds validation | MEDIUM |
| 3 | Regime-aware training completion | MEDIUM |
| 3 | Meta-labeling training completion | MEDIUM |
| 4 | OOFCache implementation | MEDIUM |

### Nice to Have

| Phase | Issue | Priority |
|-------|-------|----------|
| 1B | Parallelization support | LOW |
| 1B | Result caching | LOW |
| 5 | Bundle version migration | LOW |
| 5 | Full server load testing | LOW |

---

## Quick Start Guide

### 1. End-to-End Training Pipeline

```python
from src.core import PipelineConfig, production_config
from src.training import UnifiedTrainingOrchestrator

# Load config (or use preset)
config = production_config()
config.data_path = "./data/mes_1min.parquet"
config.output_dir = "./experiments/exp_001"

# Load data
df = pd.read_parquet(config.data_path)

# Train
orchestrator = UnifiedTrainingOrchestrator(config)
result = orchestrator.train(df)

print(f"Run ID: {result.run_id}")
print(f"Models: {list(result.model_results.keys())}")
print(f"Ensemble coverage: {result.aligned_oof.coverage:.2%}")
```

### 2. Inference from Trained Model

```python
from src.inference import InferenceOrchestrator

# Load from experiment
orchestrator = InferenceOrchestrator.from_experiment(config)

# Predict from features
result = orchestrator.predict(X_new)
print(f"Predictions: {result.class_predictions}")

# Or from raw OHLCV
result = orchestrator.predict_from_raw(raw_ohlcv_df)
```

### 3. Batch Processing

```python
from src.inference import InferenceOrchestrator

orchestrator = InferenceOrchestrator.from_bundle("./bundles/xgb_h20")
predictions_df = orchestrator.predict_batch(
    data=large_df,
    batch_size=10000,
    output_path="predictions.parquet"
)
```

---

## File Reference

| File | Purpose |
|------|---------|
| `PHASE_0_IMPLEMENTATION.md` | Foundation: types, contracts, config |
| `PHASE_1_IMPLEMENTATION.md` | Features: 162 features, 12 families |
| `PHASE_1B_IMPLEMENTATION.md` | Optimization: labels, features, hyperparams |
| `PHASE_2_IMPLEMENTATION.md` | Adapters: 2D/3D/4D, scaling, alignment |
| `PHASE_3_IMPLEMENTATION.md` | Training: modes, CV, OOF |
| `PHASE_4_IMPLEMENTATION.md` | Ensemble: meta-learners, stacking |
| `PHASE_5_IMPLEMENTATION.md` | Inference: bundles, preprocessing, serving |

---

## Versioning

| Component | Version |
|-----------|---------|
| Bundle | 1.1.0 |
| EnsembleBundle | 1.0.0 |
| PreprocessingGraph | 1.0.0 |
| PipelineConfig | 1.0 |

---

*Generated: 2026-01-18*
