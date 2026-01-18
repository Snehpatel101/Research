# PHASE 3: TRAINING ORCHESTRATION - Implementation Plan

**Status:** ✅ COMPLETE (90%)
**Last Updated:** 2026-01-18
**Dependencies:** PHASE_0, PHASE_1/1B, PHASE_2

---

## Executive Summary

PHASE_3 creates a unified training orchestration system with a single entry point for all training modes, CV methods, and OOF generation. The `UnifiedTrainingOrchestrator` is THE entry point.

---

## Current State Analysis

### Package Structure

```
src/training/
├── __init__.py               ✅ Complete
├── unified_orchestrator.py   ✅ Complete - UnifiedTrainingOrchestrator
├── model_trainer.py          ✅ Complete - ModelTrainer + TrainedModelArtifact
├── orchestrator.py           ✅ Complete - Legacy TrainingOrchestrator
├── config.py                 ✅ Complete - ExperimentConfig, ModelConfig
└── config_loader.py          ✅ Complete - ConfigLoader

src/cross_validation/
├── __init__.py               ✅ Complete
├── cv_orchestrator.py        ✅ Complete - CVOrchestrator (unified CV)
├── purged_kfold.py          ✅ Complete - PurgedKFold
├── cpcv.py                  ✅ Complete - CombinatorialPurgedCV
├── walk_forward.py          ✅ Complete - WalkForwardEvaluator
├── pbo.py                   ✅ Complete - PBO computation
└── oof_core.py              ✅ Complete - OOFPrediction, CoreOOFGenerator
```

---

## Implemented Components

### 1. UnifiedTrainingOrchestrator (`unified_orchestrator.py`)

**THE single entry point for all training.**

```python
# Key exports:
UnifiedTrainingOrchestrator  # Master controller
TrainingRunResult            # Complete training output
ModelTrainingResult          # Per-model training output
train_pipeline               # Convenience function

# Usage:
orchestrator = UnifiedTrainingOrchestrator(config)
result = orchestrator.train(df)

# Result contains:
#   - run_id: Unique identifier
#   - model_results: Dict[str, ModelTrainingResult]
#   - ensemble_result: Optional[ModelTrainingResult]
#   - stacking_dataset: Optional[StackingDataset]
#   - aligned_oof: Optional[AlignedOOFResult]
```

### 2. ModelTrainer (`model_trainer.py`)

**Per-model training with adapter integration.**

```python
# Key exports:
ModelTrainer          # Model trainer class
TrainedModelArtifact  # Training output
train_models          # Convenience function

# Usage:
trainer = ModelTrainer(config)
artifact = trainer.train_model("xgboost", df, horizon=20)
artifacts = trainer.train_all(df)

# Artifact contains:
#   - model: Trained model instance
#   - scaler: Fitted scaler
#   - feature_columns: List of features
#   - metrics: Training metrics
#   - oof_predictions: Optional OOF
```

### 3. CVOrchestrator (`cv_orchestrator.py`)

**Unified CV interface for all methods.**

```python
# Key exports:
CVOrchestrator      # Unified CV wrapper
PurgedKFold         # Time-series K-fold with purging
CombinatorialPurgedCV  # CPCV for robust validation
WalkForwardEvaluator   # Walk-forward validation
compute_pbo         # PBO computation

# Usage:
cv = CVOrchestrator.from_config(config)
for train_idx, test_idx in cv.split(X, y):
    model.fit(X[train_idx], y[train_idx])
```

### 4. Training Modes

| Mode | Description | Status |
|------|-------------|--------|
| `standard` | PurgedKFold with OOF generation | ✅ Implemented |
| `walk_forward` | Expanding/rolling windows | ✅ Implemented |
| `regime_aware` | Per-regime models | ⚠️ Partial |
| `meta_labeling` | Primary + meta-model | ⚠️ Partial |

### 5. CV Methods

| Method | Description | Status |
|--------|-------------|--------|
| `purged_kfold` | K-fold with purge/embargo | ✅ Implemented |
| `cpcv` | Combinatorial purged CV | ✅ Implemented |
| `walk_forward` | Expanding/rolling | ✅ Implemented |
| `pbo` | Probability of backtest overfitting | ✅ Implemented |

---

## Data Flow Diagram

```
PipelineConfig + DataFrame
        │
        ▼
UnifiedTrainingOrchestrator.train()
        │
        ├─ For each horizon:
        │   ├─ For each model:
        │   │   ├─ UnifiedDataPreparation.prepare() [PHASE_2]
        │   │   ├─ CVOrchestrator.split()
        │   │   ├─ ModelTrainer.train_model()
        │   │   └─ OOF generation (if save_oof=True)
        │   │
        │   └─ Store ModelTrainingResult
        │
        ├─ If build_ensemble:
        │   ├─ OOFAligner.align() [PHASE_2]
        │   ├─ HeterogeneousStackingBuilder.build() [PHASE_4]
        │   └─ Train meta-learner [PHASE_4]
        │
        └─ Save results and bundles [PHASE_5]
        │
        ▼
TrainingRunResult
```

---

## Remaining Tasks

### Task 3.1: Complete Regime-Aware Training ⚠️

**Gap:** Regime-aware training mode is partially implemented.

**Required:**
```python
def _train_regime_aware(self, df, models):
    # Detect regimes (volatility/trend)
    regimes = RegimeDetector(df).detect()

    # Train separate models per regime
    for regime in regimes.unique():
        regime_df = df[regimes == regime]
        for model in models:
            self._train_model(model, regime_df)
```

### Task 3.2: Complete Meta-Labeling Training ⚠️

**Gap:** Meta-labeling training mode needs full implementation.

**Required:**
```python
def _train_meta_labeling(self, df, models):
    # Train primary model
    primary = models[0]
    primary_trainer = self._train_model(primary, df)

    # Generate primary predictions
    primary_preds = primary_trainer.predict(df)

    # Train meta-model on primary predictions
    meta_model = self._train_meta_model(df, primary_preds)
```

### Task 3.3: Resolve Circular Import ⚠️

**Issue:** Circular import detected in cross_validation module.

**Chain:**
```
cross_validation/__init__
  → cv_feature_selection
    → oof_generator
      → oof_core
        → models.base
          → models/__init__
            → models.ensemble
              → heterogeneous_stacking
                → oof_core (CIRCULAR)
```

**Action Items:**
- [ ] Refactor imports to use TYPE_CHECKING
- [ ] Lazy load ensemble components
- [ ] Add integration test for import chain

---

## Integration Points

| Downstream Phase | Consumes |
|------------------|----------|
| PHASE_4 | `TrainingRunResult.oof_predictions` |
| PHASE_5 | `TrainedModelArtifact`, `TrainingRunResult` |

---

## Usage Examples

### Example 1: Standard Training
```python
from src.core import PipelineConfig
from src.training import UnifiedTrainingOrchestrator

config = PipelineConfig(
    symbol="MES",
    data_path="./data/mes.parquet",
    output_dir="./experiments/exp_001",
    models=["xgboost", "lightgbm", "lstm"],
    horizons=[20],
    build_ensemble=True,
    meta_learner="ridge_meta",
    save_oof=True,
)

orchestrator = UnifiedTrainingOrchestrator(config)
result = orchestrator.train(df)

print(f"Run ID: {result.run_id}")
print(f"Models trained: {list(result.model_results.keys())}")
print(f"Ensemble coverage: {result.aligned_oof.coverage}")
```

### Example 2: Walk-Forward Validation
```python
from src.core import TrainingMode

config = PipelineConfig(
    training_mode=TrainingMode.WALK_FORWARD,
    cv_method=CVMethod.WALK_FORWARD,
    models=["xgboost", "lstm"],
    ...
)

orchestrator = UnifiedTrainingOrchestrator(config)
result = orchestrator.train(df)
```

---

## Sign-off Criteria

- [x] UnifiedTrainingOrchestrator implemented
- [x] ModelTrainer with adapter integration
- [x] CVOrchestrator with 4 CV methods
- [x] OOF generation for heterogeneous models
- [x] TrainingRunResult with comprehensive output
- [ ] Regime-aware training complete
- [ ] Meta-labeling training complete
- [ ] Circular import resolved

**PHASE_3 Status: READY FOR PHASE_4**
