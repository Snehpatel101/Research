# SRC Improvement Plan: Unified ML Pipeline Architecture

**Date:** 2026-01-21  
**Author:** AI Engineering Analysis  
**Status:** Strategic Improvement Roadmap

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Core Problems](#core-problems)
4. [Target Architecture](#target-architecture)
5. [Pipeline Integration Strategy](#pipeline-integration-strategy)
6. [Module Reorganization Plan](#module-reorganization-plan)
7. [Feature Inventory](#feature-inventory)
8. [Implementation Roadmap](#implementation-roadmap)
9. [Migration Guide](#migration-guide)
10. [Success Criteria](#success-criteria)

---

## Executive Summary

### What We Have
A **powerful but fragmented** ML pipeline for financial price prediction:
- **23 models** across 4 families (boosting, classical, neural, ensemble)
- **180+ features** with multi-timeframe support
- **Advanced evaluation** (PurgedKFold, Walk-Forward, CPCV-PBO)
- **Production-ready inference** with bundling

### What We Need
A **unified, cohesive architecture** where:
```python
from src import MLPipeline

pipeline = MLPipeline(symbol="MES", models=["xgboost", "lstm"])
pipeline.run()  # ONE method runs EVERYTHING
```

### The Problem
The `src/` directory has **37 top-level modules** with significant duplication and unclear boundaries:

| Problem | Impact |
|---------|--------|
| Multiple pipeline implementations | Confusion, maintenance burden |
| Scattered configuration | No single source of truth |
| Feature computation in 5+ places | Inconsistency, bugs |
| Training logic in 4+ modules | Unclear ownership |
| 71,500 lines across 100+ files | Hard to navigate |

### The Goal
Consolidate into a **clean 16-stage unified pipeline** with:
- **One entry point** (`MLPipeline` or `MLFactory`)
- **One config system** (`UnifiedConfig`)
- **Clear module boundaries** (data → training → inference)
- **No duplication** (single implementation per concern)

---

## Current State Analysis

### Directory Structure (Current)

```
src/                           # 37 top-level modules!
├── adapters/                  # Data format adapters
├── backtesting/               # Backtesting engine
├── cli/                       # CLI commands
├── common/                    # Shared utilities
├── config/                    # Configuration (multiple systems!)
├── contracts/                 # Data/model contracts
├── coordination/              # Timeframe coordination
├── core/                      # Core abstractions
├── cross_validation/          # CV implementations
├── evaluation/                # Evaluators (CV, WF, CPCV-PBO)
├── features/                  # Feature computation (DUPLICATE!)
├── feature_selection/         # Feature selection
├── feature_store/             # Feature caching
├── inference/                 # Inference engine
├── labeling/                  # Triple-barrier labeling (DUPLICATE!)
├── ml_pipeline/               # MLPipeline unified (NEW)
├── models/                    # Model registry + implementations
├── monitoring/                # Drift detection
├── optimization/              # Optuna optimization
├── pipeline/                  # Data pipeline stages (MAIN)
│   ├── stages/                # 17 pipeline stages
│   ├── config/                # Pipeline-specific config
│   └── runner.py              # Pipeline runner
├── training/                  # Training orchestration (DUPLICATE!)
├── utils/                     # Utilities
├── validation/                # Data validation
├── factory.py                 # MLFactory entry point
└── pipeline_cli.py            # Pipeline CLI
```

### Problem: Multiple Implementations

| Concern | Locations | Problem |
|---------|-----------|---------|
| **Feature Computation** | `src/features/`, `src/pipeline/stages/features/`, `src/core/features/` | 3 implementations |
| **Labeling** | `src/labeling/`, `src/pipeline/stages/labeling/`, `src/pipeline/stages/final_labels/` | 3 implementations |
| **Training** | `src/training/`, `src/models/training/`, `src/models/trainer.py` | 3 implementations |
| **Configuration** | `src/config/`, `src/pipeline/config/`, `src/models/config/`, `src/ml_pipeline/config.py` | 4 systems |
| **Pipeline Orchestration** | `src/factory.py`, `src/ml_pipeline/unified.py`, `src/pipeline/runner.py` | 3 orchestrators |

### Largest Files (Complexity Hotspots)

| File | Lines | Purpose |
|------|-------|---------|
| `training/unified_orchestrator.py` | 1,599 | Training orchestration (TOO BIG) |
| `models/regime_evaluation.py` | 1,193 | Regime evaluation (TOO BIG) |
| `config/unified.py` | 1,116 | Unified config (acceptable) |
| `inference/preprocessing_graph.py` | 907 | Feature preprocessing |
| `feature_store/store.py` | 911 | Feature caching |
| `config/smart_config.py` | 925 | Smart config |
| `inference/ensemble_bundle.py` | 927 | Ensemble bundling |
| `ml_pipeline/state.py` | 887 | Pipeline state |

**Target:** ≤800 lines per file (650 preferred)

---

## Core Problems

### Problem 1: No Single Entry Point

**Current:** User must understand which orchestrator to use:
```python
# Option 1: MLFactory (src/factory.py)
from src.factory import MLFactory
factory = MLFactory(config)
result = factory.run(df)

# Option 2: MLPipeline (src/ml_pipeline/unified.py)
from src.ml_pipeline.unified import MLPipeline
pipeline = MLPipeline(config)
pipeline.run()

# Option 3: PipelineRunner (src/pipeline/runner.py)
from src.pipeline.runner import PipelineRunner
runner = PipelineRunner(config)
runner.run()

# Option 4: UnifiedTrainingOrchestrator (src/training/unified_orchestrator.py)
from src.training.unified_orchestrator import UnifiedTrainingOrchestrator
orchestrator = UnifiedTrainingOrchestrator(config)
orchestrator.train(df)
```

**Target:** ONE entry point:
```python
from src import MLPipeline
pipeline = MLPipeline(symbol="MES", models=["xgboost"])
pipeline.run()
```

### Problem 2: Configuration Sprawl

**Current:** 4+ configuration systems:

| System | Location | Purpose |
|--------|----------|---------|
| `UnifiedConfig` | `src/config/unified.py` | 20 dataclass sections |
| `PipelineConfig` | `src/pipeline/config/` | Pipeline-specific |
| `TrainerConfig` | `src/models/config/trainer_config.py` | Training-specific |
| `MLConfig` | `src/ml_pipeline/config.py` | MLPipeline config |
| Smart Config | `src/config/smart_config.py` | YAML-based config |
| Global Config | `src/config/global_config.py` | Global defaults |

**Target:** ONE config hierarchy:
```python
@dataclass
class MLConfig:
    """Single config for entire pipeline."""
    # Data
    symbol: str
    data_dir: Path
    
    # Features
    features: FeaturesConfig
    
    # Labeling
    labeling: LabelingConfig
    
    # Training
    training: TrainingConfig
    
    # Evaluation
    evaluation: EvaluationConfig
```

### Problem 3: Scattered Feature Logic

**Current:**
```
Feature computation in 5+ places:
├── src/features/compute/          # Feature computation
├── src/features/selection.py      # Feature selection
├── src/feature_selection/         # Another feature selection!
├── src/pipeline/stages/features/  # Pipeline feature stage
├── src/core/features/             # Core feature abstractions
└── src/feature_store/             # Feature caching
```

**Target:**
```
src/features/                      # ALL feature logic
├── compute/                       # Feature computation
├── selection/                     # Feature selection
├── store/                         # Feature caching
└── config.py                      # Feature configuration
```

### Problem 4: Training Fragmentation

**Current:**
```
Training logic in 4+ places:
├── src/training/unified_orchestrator.py  # Main orchestrator (1599 lines!)
├── src/training/model_trainer.py         # Model trainer
├── src/models/training/trainer.py        # Another trainer!
├── src/models/trainer.py                 # Re-export
└── src/models/training_utils.py          # Training utilities
```

**Target:**
```
src/training/                      # ALL training logic
├── orchestrator.py                # Main orchestrator (<800 lines)
├── trainer.py                     # Model trainer
├── modes/                         # Training modes
│   ├── standard.py
│   ├── walk_forward.py
│   ├── regime_aware.py
│   └── meta_labeling.py
└── config.py                      # Training configuration
```

### Problem 5: Pipeline vs. Module Confusion

**Question:** What is `src/pipeline/` vs the rest of `src/`?

**Current Answer (Confusing):**
- `src/pipeline/` = Data preparation stages (1-12)
- `src/training/` = Training stages (13-15)
- `src/inference/` = Deployment stage (16)
- `src/models/` = Model implementations
- `src/features/` = ???
- `src/labeling/` = ???

**Target Answer (Clear):**
- `src/pipeline/` = **Unified 16-stage orchestration**
- `src/data/` = Data loading, cleaning, validation
- `src/features/` = Feature engineering
- `src/training/` = Model training
- `src/inference/` = Model serving
- `src/models/` = Model implementations (no training logic)

---

## Target Architecture

### Proposed Directory Structure

```
src/
├── __init__.py                    # Exports: MLPipeline, MLConfig, ModelRegistry
│
├── pipeline/                      # 16-STAGE UNIFIED ORCHESTRATION
│   ├── __init__.py                # Exports: MLPipeline, run
│   ├── orchestrator.py            # MLPipeline class (main entry point)
│   ├── config.py                  # MLConfig dataclass
│   ├── state.py                   # PipelineState (checkpointing)
│   ├── phases/                    # Phase groupings
│   │   ├── data.py                # Stages 1-6: Data preparation
│   │   ├── optimization.py        # Stages 7-9: Optuna optimization
│   │   ├── preprocessing.py       # Stages 10-12: Splits, scaling, adapters
│   │   ├── training.py            # Stages 13-15: Training, stacking
│   │   └── deployment.py          # Stage 16: Bundling
│   └── stages/                    # Individual stage implementations
│       ├── ingest/
│       ├── clean/
│       ├── sessions/
│       ├── mtf/
│       ├── features/
│       ├── regime/
│       ├── labeling/
│       ├── splits/
│       ├── scaling/
│       └── ...
│
├── models/                        # MODEL IMPLEMENTATIONS ONLY
│   ├── __init__.py                # Exports: ModelRegistry, BaseModel
│   ├── registry.py                # Plugin registry
│   ├── base.py                    # BaseModel interface
│   ├── contracts.py               # ModelContract, DataContract
│   ├── boosting/                  # XGBoost, LightGBM, CatBoost
│   ├── neural/                    # LSTM, GRU, TCN, Transformer, etc.
│   ├── classical/                 # RF, Logistic, SVM
│   ├── ensemble/                  # Voting, Stacking, Blending
│   └── config/                    # Per-model configuration
│
├── training/                      # TRAINING EXECUTION
│   ├── __init__.py                # Exports: Trainer, TrainingResult
│   ├── trainer.py                 # Unified model trainer
│   ├── orchestrator.py            # Training orchestration (<800 lines)
│   ├── cv/                        # Cross-validation
│   │   ├── purged_kfold.py
│   │   ├── oof_generator.py
│   │   └── cv_runner.py
│   ├── modes/                     # Training modes
│   │   ├── standard.py
│   │   ├── walk_forward.py
│   │   ├── regime_aware.py
│   │   └── meta_labeling.py
│   └── config.py
│
├── inference/                     # INFERENCE ENGINE
│   ├── __init__.py                # Exports: InferenceOrchestrator, Bundle
│   ├── orchestrator.py            # Main inference controller
│   ├── bundle.py                  # ModelBundle, EnsembleBundle
│   ├── preprocessing.py           # PreprocessingGraph
│   ├── server.py                  # Flask server
│   └── batch.py                   # Batch inference
│
├── features/                      # FEATURE ENGINEERING (CONSOLIDATED)
│   ├── __init__.py                # Exports: compute_features, select_features
│   ├── compute/                   # Feature computation
│   │   ├── momentum.py
│   │   ├── volatility.py
│   │   ├── volume.py
│   │   ├── wavelets.py
│   │   ├── microstructure.py
│   │   └── mtf.py
│   ├── selection/                 # Feature selection
│   │   ├── selector.py
│   │   ├── optimization.py
│   │   └── pruning.py
│   ├── store/                     # Feature caching
│   │   ├── cache.py
│   │   └── store.py
│   └── config.py
│
├── labeling/                      # LABELING (CONSOLIDATED)
│   ├── __init__.py                # Exports: TripleBarrierLabeler
│   ├── triple_barrier.py          # Main implementation
│   ├── optimization.py            # Optuna label optimization
│   └── config.py
│
├── evaluation/                    # EVALUATION STRATEGIES
│   ├── __init__.py
│   ├── cv_evaluator.py
│   ├── walk_forward_evaluator.py
│   ├── cpcv_pbo_evaluator.py
│   └── metrics.py
│
├── backtesting/                   # BACKTESTING ENGINE
│   ├── __init__.py
│   ├── backtest.py
│   ├── equity_curve.py
│   ├── position_sizing.py
│   ├── costs.py
│   └── metrics.py
│
├── monitoring/                    # PRODUCTION MONITORING
│   ├── __init__.py
│   ├── drift_detector.py
│   └── alert_handler.py
│
├── cli/                           # COMMAND-LINE INTERFACE
│   ├── __init__.py
│   ├── main.py                    # Unified CLI entry point
│   ├── commands/
│   │   ├── run.py                 # ml run
│   │   ├── data.py                # ml data
│   │   ├── train.py               # ml train
│   │   ├── evaluate.py            # ml evaluate
│   │   └── status.py              # ml status
│   └── utils.py
│
├── config/                        # CONFIGURATION (CONSOLIDATED)
│   ├── __init__.py                # Exports: load_config, MLConfig
│   ├── unified.py                 # UnifiedConfig (refactored)
│   ├── loaders.py                 # Config loading utilities
│   ├── validators.py              # Config validation
│   └── defaults.py                # Default values
│
├── adapters/                      # DATA FORMAT ADAPTERS
│   ├── __init__.py
│   ├── tabular.py                 # 2D adapter
│   ├── sequence.py                # 3D adapter
│   ├── multi_resolution.py        # 4D adapter
│   └── preparation.py             # Data preparation
│
├── common/                        # SHARED UTILITIES
│   ├── __init__.py
│   ├── timeframes.py              # Canonical timeframes
│   ├── horizon_config.py
│   ├── split_ratios.py
│   └── manifest.py
│
└── utils/                         # GENERAL UTILITIES
    ├── __init__.py
    ├── checkpoint.py
    ├── memory.py
    ├── cache.py
    └── notebook.py
```

### Module Responsibilities

| Module | Responsibility | Should NOT Do |
|--------|---------------|---------------|
| `pipeline/` | Orchestrate 16-stage flow | Implement individual stages |
| `models/` | Model implementations | Training logic |
| `training/` | Training execution | Model implementation |
| `inference/` | Serving & bundling | Training |
| `features/` | Feature computation & selection | Model training |
| `labeling/` | Label generation & optimization | Feature engineering |
| `evaluation/` | Model evaluation | Training |
| `config/` | Configuration loading/validation | Business logic |
| `cli/` | User interface | Business logic |

---

## Pipeline Integration Strategy

### Current: `src/pipeline/` Directory

The `src/pipeline/` directory currently contains the **data preparation stages**:

```
src/pipeline/
├── runner.py              # PipelineRunner (orchestrates stages)
├── stage_registry.py      # Stage registry
├── data_config.py         # Data configuration
├── config_adapter.py      # Config adaptation
├── presets.py             # Pipeline presets
├── config/                # Pipeline-specific configs
│   ├── feature_sets/
│   ├── labels.py
│   ├── barriers_config.py
│   └── ...
├── stages/                # 17 stage implementations
│   ├── ingest/
│   ├── clean/
│   ├── sessions/
│   ├── mtf/
│   ├── features/
│   ├── regime/
│   ├── labeling/
│   ├── ga_optimize/
│   ├── final_labels/
│   ├── splits/
│   ├── scaling/
│   ├── datasets/
│   ├── meta_labeling/
│   └── ...
└── utils/
```

### Integration Approach

**Strategy: Elevate `pipeline/` to master orchestrator**

1. **Keep `src/pipeline/stages/`** as the source of truth for data preparation
2. **Create `src/pipeline/orchestrator.py`** as the unified entry point
3. **Import training from `src/training/`** (don't duplicate)
4. **Import inference from `src/inference/`** (don't duplicate)

### Unified Pipeline Orchestrator

```python
# src/pipeline/orchestrator.py

class MLPipeline:
    """
    THE single entry point for the entire ML pipeline.
    
    Orchestrates all 16 stages:
    - Stages 1-6: Data preparation (via PipelineRunner)
    - Stages 7-9: Optuna optimization (via OptimizationPipeline)
    - Stages 10-12: Preprocessing (via PipelineRunner)
    - Stages 13-15: Training (via UnifiedTrainingOrchestrator)
    - Stage 16: Bundling (via InferenceOrchestrator)
    """
    
    def __init__(self, config: MLConfig | dict | str):
        self.config = self._normalize_config(config)
        self.state = PipelineState(run_id=self._generate_run_id())
        
    def run(self) -> PipelineResult:
        """Run all 16 stages."""
        self.run_data()           # Stages 1-6
        self.run_optimization()   # Stages 7-9
        self.run_preprocessing()  # Stages 10-12
        self.run_training()       # Stages 13-15
        self.run_bundling()       # Stage 16
        return self.get_result()
    
    def run_data(self) -> DataResult:
        """Stages 1-6: Data preparation."""
        from src.pipeline.runner import PipelineRunner
        runner = PipelineRunner(self._to_pipeline_config())
        return runner.run()
    
    def run_optimization(self) -> OptimizationResult:
        """Stages 7-9: Optuna optimization."""
        from src.optimization.pipeline import OptimizationPipeline
        optimizer = OptimizationPipeline(self.config)
        return optimizer.run_full_optimization()
    
    def run_training(self) -> TrainingResult:
        """Stages 13-15: Training."""
        from src.training.orchestrator import UnifiedTrainingOrchestrator
        orchestrator = UnifiedTrainingOrchestrator(self.config)
        return orchestrator.train()
    
    def run_bundling(self) -> BundleResult:
        """Stage 16: Bundling."""
        from src.inference.builder import InferenceBuilder
        builder = InferenceBuilder(self.config)
        return builder.build_bundles()
```

### Integration Steps

#### Step 1: Consolidate Entry Points

| Current | Action | Target |
|---------|--------|--------|
| `src/factory.py` | Deprecate, redirect to `MLPipeline` | Remove after migration |
| `src/ml_pipeline/unified.py` | Merge into `pipeline/orchestrator.py` | Remove |
| `src/pipeline/runner.py` | Keep as data-only runner | `pipeline/phases/data.py` |
| `src/training/unified_orchestrator.py` | Keep, refactor to <800 lines | `training/orchestrator.py` |

#### Step 2: Merge Configuration

| Current | Action | Target |
|---------|--------|--------|
| `src/config/unified.py` | Keep as base | `config/unified.py` |
| `src/pipeline/config/` | Merge into unified | `config/pipeline/` |
| `src/ml_pipeline/config.py` | Merge into unified | Remove |
| `src/models/config/` | Keep for per-model configs | `models/config/` |

#### Step 3: Consolidate Features

| Current | Action | Target |
|---------|--------|--------|
| `src/features/` | Keep as primary | `features/` |
| `src/pipeline/stages/features/` | Import from `features/` | Wrapper only |
| `src/core/features/` | Merge into `features/` | Remove |
| `src/feature_selection/` | Merge into `features/selection/` | Remove |
| `src/feature_store/` | Move to `features/store/` | Remove |

#### Step 4: Consolidate Labeling

| Current | Action | Target |
|---------|--------|--------|
| `src/labeling/` | Keep as primary | `labeling/` |
| `src/pipeline/stages/labeling/` | Import from `labeling/` | Wrapper only |
| `src/pipeline/stages/final_labels/` | Merge into `labeling/` | Remove |
| `src/pipeline/stages/ga_optimize/` | Merge into `labeling/optimization.py` | Remove |

#### Step 5: Consolidate Training

| Current | Action | Target |
|---------|--------|--------|
| `src/training/unified_orchestrator.py` | Refactor to <800 lines | `training/orchestrator.py` |
| `src/training/model_trainer.py` | Keep | `training/trainer.py` |
| `src/models/training/trainer.py` | Merge into `training/trainer.py` | Remove |
| `src/models/trainer.py` | Remove (re-export only) | Remove |

---

## Module Reorganization Plan

### Phase 1: Quick Wins (1-2 days)

| Task | From | To | Impact |
|------|------|-----|--------|
| Create unified entry point | N/A | `src/__init__.py` | User-facing |
| Export `MLPipeline` | N/A | `from src import MLPipeline` | User-facing |
| Deprecation warnings | `factory.py`, etc. | N/A | Backward compat |

### Phase 2: Config Consolidation (2-3 days)

| Task | Effort | Risk |
|------|--------|------|
| Merge `MLConfig` into `UnifiedConfig` | Medium | Low |
| Remove `src/ml_pipeline/config.py` | Low | Low |
| Create config migration guide | Low | None |

### Phase 3: Feature Consolidation (3-5 days)

| Task | Effort | Risk |
|------|--------|------|
| Merge `feature_selection/` into `features/selection/` | Medium | Medium |
| Merge `feature_store/` into `features/store/` | Medium | Medium |
| Update all imports | High | Medium |
| Remove old directories | Low | Low |

### Phase 4: Training Consolidation (3-5 days)

| Task | Effort | Risk |
|------|--------|------|
| Refactor `unified_orchestrator.py` (1599→800 lines) | High | Medium |
| Merge `models/training/` into `training/` | Medium | Medium |
| Update all imports | High | Medium |

### Phase 5: Labeling Consolidation (2-3 days)

| Task | Effort | Risk |
|------|--------|------|
| Merge `pipeline/stages/labeling/` with `labeling/` | Medium | Low |
| Merge `pipeline/stages/ga_optimize/` | Medium | Low |
| Update pipeline stage to use consolidated module | Medium | Low |

### Phase 6: Cleanup (1-2 days)

| Task | Effort | Risk |
|------|--------|------|
| Remove deprecated modules | Low | Low |
| Update documentation | Medium | None |
| Run full test suite | Medium | Low |

---

## Feature Inventory

### Current Features (What We Have)

#### Models (23 total)
| Family | Models | Status |
|--------|--------|--------|
| **Boosting** | XGBoost, LightGBM, CatBoost | Complete |
| **Classical** | Random Forest, Logistic, SVM | Complete |
| **Neural** | LSTM, GRU, TCN, Transformer | Complete |
| **Advanced Neural** | PatchTST, iTransformer, TFT, N-BEATS, InceptionTime, ResNet1D | Complete |
| **Ensemble** | Voting, Stacking, Blending | Complete |
| **Meta-Learners** | Ridge, MLP, Calibrated, XGBoost | Complete |

#### Data Pipeline (17 stages)
| Stage | Purpose | Status |
|-------|---------|--------|
| Ingest | Load raw OHLCV | Complete |
| Clean | Resample, gaps | Complete |
| Sessions | Trading hours | Complete |
| MTF | 9 timeframes | Complete |
| Features | 180+ indicators | Complete |
| Regime | Vol/trend detection | Complete |
| GA Optimize | Label optimization | Complete |
| Final Labels | Apply optimized params | Complete |
| Splits | Train/val/test | Complete |
| Scaling | Train-only robust | Complete |
| Datasets | 2D/3D/4D adapters | Complete |
| Validation | Feature quality | Complete |
| Meta-labeling | Secondary model | Complete |

#### Feature Engineering
| Category | Count | Status |
|----------|-------|--------|
| Momentum | ~30 | Complete |
| Trend | ~25 | Complete |
| Volatility | ~20 | Complete |
| Volume | ~15 | Complete |
| Microstructure | ~20 | Complete |
| Wavelets | ~30 | Complete |
| Statistical | ~40 | Complete |
| **Total** | **~180** | Complete |

#### Evaluation Strategies
| Strategy | Purpose | Status |
|----------|---------|--------|
| PurgedKFold | Time-series CV | Complete |
| Walk-Forward | Rolling evaluation | Complete |
| CPCV-PBO | Backtest overfitting | Complete |
| OOF Generation | Stacking support | Complete |

#### Inference
| Component | Purpose | Status |
|-----------|---------|--------|
| ModelBundle | Single model | Complete |
| EnsembleBundle | Heterogeneous ensemble | Complete |
| PreprocessingGraph | Feature lineage | Complete |
| Flask Server | REST API | Complete |
| Batch Inference | Bulk predictions | Complete |

#### Backtesting
| Component | Purpose | Status |
|-----------|---------|--------|
| Backtest Engine | Strategy evaluation | Complete |
| Equity Curve | Performance tracking | Complete |
| Position Sizing | Kelly, fixed, etc. | Complete |
| Costs Model | Slippage, commission | Complete |
| Metrics | Sharpe, drawdown | Complete |

#### Monitoring
| Component | Purpose | Status |
|-----------|---------|--------|
| Drift Detection | Feature drift | Complete |
| Alert Handler | Notifications | Complete |

### Target Features (What We're Building)

#### Unified Pipeline API
```python
# One-liner usage
from src import MLPipeline

result = MLPipeline(symbol="MES", models=["xgboost"]).run()

# Detailed usage
pipeline = MLPipeline(
    symbol="MES",
    horizons=[5, 10, 15, 20],
    models=["xgboost", "lstm", "patchtst"],
    training_mode="standard",
    build_ensemble=True,
    meta_learner="ridge_meta",
    optimize_labels=True,
    optimize_features=True,
    optimize_hyperparameters=True,
)

# Run stages individually
pipeline.run_data()           # Stages 1-6
pipeline.run_optimization()   # Stages 7-9
pipeline.run_preprocessing()  # Stages 10-12
pipeline.run_training()       # Stages 13-15
pipeline.run_bundling()       # Stage 16

# Or run all
pipeline.run()

# Resume from checkpoint
pipeline.resume(from_stage=7)

# Get results
result = pipeline.get_result()
print(f"Best model: {result.best_model}")
print(f"Test F1: {result.test_metrics['f1']}")
```

#### Unified CLI
```bash
# Full pipeline
ml run --symbol MES --models xgboost,lstm --build-ensemble

# Individual phases
ml data --symbol MES
ml optimize --trials 100
ml train --models xgboost,lstm
ml evaluate --method walk_forward
ml bundle --model xgboost

# Status and resume
ml status --run-id 20260121_120000
ml resume --run-id 20260121_120000 --from-stage 7
```

---

## Implementation Roadmap

### Sprint 1: Foundation (Week 1)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Create `src/__init__.py` with unified exports | `from src import MLPipeline` works |
| 1 | Add deprecation warnings to old entry points | Backward compat |
| 2 | Create `src/pipeline/orchestrator.py` skeleton | MLPipeline class |
| 2 | Wire MLPipeline to existing implementations | run() works end-to-end |
| 3 | Create unified CLI structure | `ml run` works |
| 4-5 | Write integration tests | CI passes |

### Sprint 2: Config Consolidation (Week 2)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Merge MLConfig into UnifiedConfig | Single config |
| 3 | Update all config imports | No broken imports |
| 4 | Remove deprecated config modules | Clean codebase |
| 5 | Update documentation | Accurate docs |

### Sprint 3: Feature Consolidation (Week 3)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Merge feature_selection into features/ | Consolidated |
| 3 | Merge feature_store into features/store/ | Consolidated |
| 4 | Update pipeline stages to use consolidated | Working pipeline |
| 5 | Remove old directories | Clean codebase |

### Sprint 4: Training Consolidation (Week 4)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Refactor unified_orchestrator.py | <800 lines |
| 3 | Merge models/training into training/ | Consolidated |
| 4 | Update all imports | Working pipeline |
| 5 | Remove deprecated modules | Clean codebase |

### Sprint 5: Final Cleanup (Week 5)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Merge labeling modules | Consolidated |
| 2 | Final import cleanup | All imports work |
| 3 | Documentation update | Complete docs |
| 4 | Full test suite | All tests pass |
| 5 | Release | v2.0 |

---

## Migration Guide

### For Users

#### Before (v1.x)
```python
from src.factory import MLFactory
from src.core import PipelineConfig

config = PipelineConfig(...)
factory = MLFactory(config)
result = factory.run(df)
```

#### After (v2.0)
```python
from src import MLPipeline

pipeline = MLPipeline(symbol="MES", models=["xgboost"])
result = pipeline.run()
```

#### Backward Compatibility
```python
# This still works (with deprecation warning)
from src.factory import MLFactory  # DeprecationWarning

# Recommended migration
from src import MLPipeline
```

### For Developers

#### Adding a New Model
```python
# Location: src/models/{family}/{model_name}.py

from src.models import register, BaseModel

@register(name="my_model", family="boosting")
class MyModel(BaseModel):
    def fit(self, X_train, y_train, X_val, y_val, **kwargs):
        ...
    
    def predict(self, X):
        ...
```

#### Adding a New Feature
```python
# Location: src/features/compute/{category}.py

def compute_my_feature(df: pd.DataFrame) -> pd.Series:
    """Compute my custom feature."""
    return ...

# Register in src/features/compute/__init__.py
from .my_category import compute_my_feature
```

#### Adding a New Pipeline Stage
```python
# Location: src/pipeline/stages/{stage_name}/

# Create stage implementation
# Register in src/pipeline/stage_registry.py
```

---

## Success Criteria

### Quantitative Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Top-level modules | 37 | ≤15 |
| Lines in largest file | 1,599 | ≤800 |
| Configuration systems | 4+ | 1 |
| Entry points | 4 | 1 |
| Test coverage | Unknown | ≥80% |

### Qualitative Criteria

| Criterion | Description |
|-----------|-------------|
| **Single Entry Point** | `from src import MLPipeline` is the only way |
| **Clear Boundaries** | Each module has one responsibility |
| **No Duplication** | One implementation per concern |
| **Self-Documenting** | Module structure explains purpose |
| **Backward Compatible** | Old code works with deprecation warnings |

### User Experience Goals

```python
# Goal 1: One-liner training
result = MLPipeline(symbol="MES", models=["xgboost"]).run()

# Goal 2: Clear error messages
# "Stage 7 (Label Optimization) failed: No valid labels found"

# Goal 3: Easy resumption
pipeline.resume(from_stage=7)

# Goal 4: Discoverable API
help(MLPipeline)  # Shows all options clearly
```

### CLI Goals

```bash
# Goal 1: Simple commands
ml run --symbol MES --models xgboost

# Goal 2: Helpful errors
# "Error: Model 'xgbooost' not found. Did you mean 'xgboost'?"

# Goal 3: Progress visibility
# "Stage 7/16: Label Optimization [████████░░] 80% (80/100 trials)"
```

---

## Appendix: File Inventory

### Files to Keep (Primary Implementations)

| Path | Lines | Purpose |
|------|-------|---------|
| `src/pipeline/runner.py` | ~400 | Data pipeline runner |
| `src/pipeline/stages/*` | ~5000 | Pipeline stage implementations |
| `src/models/registry.py` | ~200 | Model registry |
| `src/models/base.py` | ~150 | BaseModel interface |
| `src/models/boosting/*` | ~600 | Boosting models |
| `src/models/neural/*` | ~2000 | Neural models |
| `src/models/ensemble/*` | ~1500 | Ensemble models |
| `src/training/unified_orchestrator.py` | ~1599 | Training (needs refactor) |
| `src/inference/orchestrator.py` | ~841 | Inference |
| `src/inference/bundle.py` | ~718 | Model bundling |
| `src/config/unified.py` | ~1116 | Unified config |
| `src/evaluation/*` | ~500 | Evaluators |
| `src/backtesting/*` | ~1500 | Backtesting |
| `src/cli/*` | ~600 | CLI commands |

### Files to Merge/Remove

| Path | Action | Target |
|------|--------|--------|
| `src/factory.py` | Deprecate | `src/pipeline/orchestrator.py` |
| `src/ml_pipeline/unified.py` | Merge | `src/pipeline/orchestrator.py` |
| `src/ml_pipeline/config.py` | Merge | `src/config/unified.py` |
| `src/models/training/trainer.py` | Merge | `src/training/trainer.py` |
| `src/models/trainer.py` | Remove | N/A (re-export only) |
| `src/feature_selection/*` | Merge | `src/features/selection/` |
| `src/feature_store/*` | Move | `src/features/store/` |
| `src/core/features/*` | Merge | `src/features/` |
| `src/core/datasets/*` | Merge | `src/adapters/` |

### Estimated Impact

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Top-level directories | 37 | 15 | -59% |
| Total Python files | 100+ | ~70 | -30% |
| Lines of code | 71,500 | ~60,000 | -16% |
| Config files | 30+ | ~25 | -17% |

---

## Conclusion

This improvement plan transforms the codebase from a **powerful but fragmented collection** into a **unified, cohesive ML pipeline**. The key changes:

1. **Single Entry Point**: `MLPipeline` is the one way to use the system
2. **Clear Module Boundaries**: Each directory has one responsibility
3. **No Duplication**: One implementation per concern
4. **Smaller Files**: All files under 800 lines
5. **Better UX**: Simple API, clear errors, easy resumption

**Estimated Effort**: 5 weeks (1 engineer)  
**Risk Level**: Medium (well-tested incremental migration)  
**ROI**: High (reduced maintenance, better onboarding, fewer bugs)

---

*Document Version: 1.0*  
*Last Updated: 2026-01-21*
