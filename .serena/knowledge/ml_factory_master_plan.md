# ML Factory Master Plan - Unified Pipeline Implementation

**Date:** 2026-01-18
**Status:** PLANNING COMPLETE - Ready for Implementation
**Architecture Version:** 4.0 (16-Stage Pipeline with Optuna Optimization)

---

## Executive Summary

### What We're Building
A **single unified ML pipeline** that transforms raw OHLCV data into production-ready model bundles through 16 automated stages with 4 Optuna optimization loops.

### The Goal
```python
from src.pipeline import MLPipeline

# ONE command runs EVERYTHING
pipeline = MLPipeline(symbol="MES", horizons=[20], models=["xgboost", "lstm", "patchtst"])
pipeline.run()  # 16 stages execute automatically
```

### Key Metrics
- **Total Stages:** 16 (Data: 1-6, Optimization: 7-9, Preprocessing: 10-12, Training: 13-15, Deployment: 16)
- **Total Optuna Trials:** 2,550 (Stage 7: 100, Stage 8: 100, Stage 9: 50, Stage 13: 2,300)
- **Total Models:** 23 (Boosting: 3, Classical: 3, Neural Basic: 4, Neural Advanced: 6, Ensemble: 3, Meta: 4)
- **Features:** ~180 raw → ~60-100 selected → ~30-60 pruned

---

## The Problem (Current State)

### Fragmentation Issues
1. **23+ scattered scripts** (`train_model.py`, `train_ensemble.py`, `train_meta_labeling.py`, etc.)
2. **Multiple disconnected src directories** (`phase1/`, `models/`, `training/`, `features/`, `feature_selection/`, etc.)
3. **No automatic handoff between phases** (user runs phase1, then manually runs training)
4. **Multiple config formats** (PipelineConfig, ExperimentConfig, TrainerConfig, CLI args)
5. **Advanced features in separate scripts** (meta-labeling, regime-aware, walk-forward)

### What's Missing
- Unified entry point (`MLPipeline`)
- Automatic stage transitions
- State management and checkpointing
- Single configuration system (`MLConfig`)

---

## The Solution (Proposed Architecture)

### 16-Stage Pipeline Overview

```
MLFactory (Entry Point)
│
├── PHASE A: DATA PREPARATION (Stages 1-6)
│   ├── Stage 1: Ingestion      → Load raw OHLCV
│   ├── Stage 2: Cleaning       → Gap handling, validation
│   ├── Stage 3: Sessions       → Trading hours filter
│   ├── Stage 4: MTF Upscaling  → 9 timeframes (1m-1h)
│   ├── Stage 5: Features       → 162 indicators (12 families)
│   └── Stage 6: Regime         → Market regime detection
│       [CHECKPOINT: data/features/{symbol}_features.parquet]
│
├── PHASE B: OPTUNA OPTIMIZATION (Stages 7-9)
│   ├── Stage 7: Label Optimization      → 100 trials (triple barrier params)
│   ├── Stage 8: Feature Selection       → 100 trials (binary include/exclude)
│   └── Stage 9: Feature Pruning         → 50 trials (importance-based removal)
│       [CHECKPOINT: data/optimized/{symbol}_optimized.parquet]
│
├── PHASE C: PREPROCESSING (Stages 10-12)
│   ├── Stage 10: Splits        → 70/15/15 + purge/embargo
│   ├── Stage 11: Scaling       → Train-only RobustScaler
│   └── Stage 12: Adaptation    → 2D/3D/4D tensors per model family
│       [CHECKPOINT: data/splits/scaled/{symbol}_{split}.parquet]
│
├── PHASE D: TRAINING (Stages 13-15)
│   ├── Stage 13: Hyperparameter Optimization → 100 trials × 23 models
│   ├── Stage 14: Training                    → PurgedKFold CV + OOF
│   └── Stage 15: Stacking                    → Meta-learner ensemble
│       [CHECKPOINT: experiments/runs/{run_id}/models/]
│
└── PHASE E: DEPLOYMENT (Stage 16)
    └── Stage 16: Bundling      → ModelBundle V1.1.0
        [OUTPUT: experiments/runs/{run_id}/bundles/{model}_bundle.pkl]
```

---

## Optuna Optimization Details

### Stage 7: Triple Barrier Label Optimization (100 trials)

**Purpose:** Optimize labeling parameters for downstream model performance

**Search Space:**
| Parameter | Range | Description |
|-----------|-------|-------------|
| `upper_mult` | [1.0, 4.0] | Upper barrier ATR multiplier (profit target) |
| `lower_mult` | [1.0, 4.0] | Lower barrier ATR multiplier (stop loss) |
| `horizon` | [5, 60] | Maximum holding period (bars) |
| `atr_period` | [7, 28] | ATR calculation lookback |

**Objective:** Maximize composite score (class balance 40% + barrier hit rate 30% + model F1 30%)

**Config:** `config/optimization/label_optimization.yaml`

---

### Stage 8: Feature Selection Optimization (100 trials)

**Purpose:** Select optimal feature subset from ~180 features

**Search Space:** Binary include/exclude for each of 162+ features (grouped by family)

**Strategies:**
1. Binary group selection (entire feature families)
2. Binary individual selection (per feature)
3. RFE-based selection
4. Importance-based selection
5. Correlation-based pruning

**Objective:** Maximize F1 with regularization for feature count

**Config:** `config/optimization/feature_selection.yaml`

---

### Stage 9: Feature Pruning Optimization (50 trials)

**Purpose:** Remove low-importance features from selected set

**Search Space:**
| Parameter | Range | Description |
|-----------|-------|-------------|
| `importance_threshold` | [0.001, 0.1] | Minimum importance to keep |
| `top_k_features` | [20, 100] | Maximum features to retain |
| `importance_method` | [gain, split, shap] | How to measure importance |

**Objective:** Maximize performance with minimal features

**Config:** `config/optimization/feature_pruning.yaml`

---

### Stage 13: Hyperparameter Optimization (2,300 trials)

**Purpose:** Optimize hyperparameters for each of 23 models

**Budget:** 100 trials per model × 23 models = 2,300 total trials

**Models Optimized:**
| Family | Models | Trials |
|--------|--------|--------|
| Boosting | XGBoost, LightGBM, CatBoost | 300 |
| Classical | Random Forest, Logistic, SVM | 300 |
| Neural Basic | LSTM, GRU, TCN, Transformer | 400 |
| Neural Advanced | PatchTST, iTransformer, TFT, N-BEATS, InceptionTime, ResNet1D | 600 |
| Ensemble | Voting, Stacking, Blending | 300 |
| Meta-Learners | Ridge Meta, MLP Meta, Calibrated, XGBoost Meta | 400 |

**Config:** `config/optimization/hyperparameter.yaml`

---

## Migration Map

### Directory Consolidation

```
CURRENT LOCATION                      →    PROPOSED LOCATION
────────────────────────────────────────────────────────────────

DATA PIPELINE
src/phase1/stages/ingest/             →    src/pipeline/phases/data.py::Stage1Ingestion
src/phase1/stages/clean/              →    src/pipeline/phases/data.py::Stage2Cleaning
src/phase1/stages/sessions/           →    src/pipeline/phases/data.py::Stage3Sessions
src/phase1/stages/mtf/                →    src/pipeline/phases/data.py::Stage4MTFUpscaling
src/phase1/stages/features/           →    src/pipeline/phases/data.py::Stage5Features
src/phase1/stages/regime/             →    src/pipeline/phases/data.py::Stage6Regime

OPTIMIZATION
src/phase1/stages/labeling/           →    src/pipeline/phases/optimization.py::Stage7LabelOptimization
src/labeling/                         →    (merged into above)
src/features/optimization.py          →    src/pipeline/phases/optimization.py::Stage8FeatureSelection
src/feature_selection/                →    src/pipeline/phases/optimization.py::Stage9FeaturePruning

TRAINING
src/phase1/stages/splits/             →    src/pipeline/phases/training.py::Stage10Splits
src/phase1/stages/scaling/            →    src/pipeline/phases/training.py::Stage11Scaling
src/phase1/stages/datasets/adapters/  →    src/pipeline/phases/training.py::Stage12Adaptation
src/optimization/                     →    src/pipeline/phases/training.py::Stage13HyperparameterOptimization
src/models/trainer.py                 →    src/pipeline/phases/training.py::Stage14Training
src/training/                         →    (merged into above)
src/cross_validation/oof_stacking.py  →    src/pipeline/phases/training.py::Stage15Stacking
src/models/ensemble/meta_learners/    →    (merged into above)

DEPLOYMENT
src/models/training/ (partial)        →    src/pipeline/phases/deployment.py::Stage16Bundling
```

### What Stays Separate (Not Migrated)

```
src/models/boosting/          # Model implementations (keep as-is)
src/models/classical/         # Model implementations (keep as-is)
src/models/neural/            # Model implementations (keep as-is)
src/models/ensemble/          # Model implementations (keep as-is)
src/models/base.py            # BaseModel interface (keep as-is)
src/models/registry.py        # Model registry (keep as-is)
src/cross_validation/         # CV infrastructure (keep as-is)
src/phase1/stages/features/   # Feature computation functions (keep, wrap)
```

### New Files to Create

```
src/pipeline/
├── __init__.py
├── unified.py              # MLPipeline master orchestrator (NEW)
├── config.py               # MLConfig unified configuration (NEW)
├── state.py                # PipelineState management (NEW)
└── phases/
    ├── __init__.py
    ├── data.py             # Stages 1-6 wrappers (NEW)
    ├── optimization.py     # Stages 7-9 wrappers (NEW)
    ├── training.py         # Stages 10-15 wrappers (NEW)
    └── deployment.py       # Stage 16 wrapper (NEW)

src/cli/
└── unified_cli.py          # Single 'ml' CLI (NEW)
```

---

## Model Architecture

### 23 Models Across 6 Families

```
FAMILY: BOOSTING (3 models) - Tabular 2D Adapter
├── XGBoost
├── LightGBM
└── CatBoost

FAMILY: CLASSICAL (3 models) - Tabular 2D Adapter
├── Random Forest
├── Logistic Regression
└── SVM

FAMILY: NEURAL BASIC (4 models) - Sequence 3D Adapter
├── LSTM
├── GRU
├── TCN
└── Transformer (basic encoder)

FAMILY: NEURAL ADVANCED (6 models) - Multi-Resolution 4D Adapter
├── PatchTST
├── iTransformer
├── TFT (Temporal Fusion Transformer)
├── N-BEATS
├── InceptionTime
└── ResNet1D

FAMILY: ENSEMBLE (3 models) - Aggregates base model outputs
├── Voting
├── Stacking
└── Blending

FAMILY: META-LEARNERS (4 models) - Stacks 3-4 heterogeneous bases
├── Ridge Meta
├── MLP Meta
├── Calibrated Meta
└── XGBoost Meta
```

### Adapter Mapping

| Model Family | Adapter | Input Shape | Feature Strategy |
|--------------|---------|-------------|------------------|
| Boosting | TabularAdapter | (N, ~60) | All engineered features |
| Classical | TabularAdapter | (N, ~60) | All engineered features |
| Neural Basic | SequenceAdapter | (N, T, ~60) | Single TF, windowed |
| Neural Advanced | MultiResolutionAdapter | (N, 9, T, 4) | Raw MTF OHLCV |
| Ensemble | N/A | Aggregated outputs | N/A |
| Meta-Learners | N/A | OOF predictions | N/A |

---

## Configuration Architecture

### Config File to Stage Mapping

```
config/
├── global.yaml                         → All stages (paths, symbols)
├── labeling.yaml                       → Stage 7 defaults
│
├── pipeline/
│   ├── training.yaml                   → Stages 1-3, 10-11, 14
│   └── cv.yaml                         → Stages 14-15 (PurgedKFold)
│
├── features/
│   ├── model_features.yaml             → Stages 5, 12
│   ├── mtf_strategies.yaml             → Stages 4, 12
│   └── selection_methods.yaml          → Stages 8-9
│
├── optimization/
│   ├── label_optimization.yaml         → Stage 7 (100 trials)
│   ├── feature_selection.yaml          → Stage 8 (100 trials)
│   ├── feature_pruning.yaml            → Stage 9 (50 trials)
│   └── hyperparameter.yaml             → Stage 13 (2,300 trials)
│
├── models/
│   └── {model_name}.yaml (×23)         → Stages 13-14
│
└── ensembles/
    ├── boosting_trio.yaml              → Stage 15
    ├── temporal_stack.yaml             → Stage 15
    └── meta_learner.yaml               → Stage 15
```

---

## Checkpoints and State Management

### Checkpoint Locations

| After Stage | Checkpoint Path | Contents |
|-------------|-----------------|----------|
| Stage 6 | `data/features/{symbol}_features.parquet` | ~180 features + regime labels |
| Stage 9 | `data/optimized/{symbol}_optimized.parquet` | ~30-60 pruned features + optimized labels |
| Stage 12 | `data/splits/scaled/{symbol}_{split}.parquet` | Train/val/test scaled splits |
| Stage 15 | `experiments/runs/{run_id}/models/` | All trained models + OOF predictions |
| Stage 16 | `experiments/runs/{run_id}/bundles/` | Production-ready ModelBundles |

### State Management

```python
@dataclass
class PipelineState:
    """Tracks pipeline execution state for checkpointing and resume."""
    run_id: str
    current_stage: int
    completed_stages: list[int]
    stage_outputs: dict[int, Any]
    optuna_studies: dict[str, optuna.Study]
    checkpoint_paths: dict[int, Path]

    def save(self, path: Path) -> None: ...
    def load(cls, path: Path) -> "PipelineState": ...
    def can_resume_from(self, stage: int) -> bool: ...
```

---

## Data Flow Summary

```
RAW DATA (data/raw/{symbol}_1m.parquet)
    │
    ▼ Stage 1-6: Data Preparation (~26 seconds)
FEATURES (data/features/{symbol}_features.parquet) - ~180 features
    │
    ▼ Stage 7-9: Optuna Optimization (~33 minutes, 250 trials)
OPTIMIZED (data/optimized/{symbol}_optimized.parquet) - ~30-60 features
    │
    ▼ Stage 10-12: Preprocessing (~4 seconds)
SPLITS (data/splits/scaled/{symbol}_{split}.parquet) - Train/Val/Test
    │
    ▼ Stage 13-15: Training (~4-8 hours, 2,300 trials + 23 models)
MODELS (experiments/runs/{run_id}/models/) - 23 trained models
    │
    ▼ Stage 16: Deployment (~2 minutes)
BUNDLES (experiments/runs/{run_id}/bundles/) - Production ModelBundles
```

---

## Implementation Priority

### Priority 1: Entry Points (Create First)
1. `src/pipeline/unified.py` - MLPipeline master orchestrator
2. `src/pipeline/config.py` - MLConfig unified configuration
3. `src/pipeline/state.py` - PipelineState management

### Priority 2: Phase Wrappers
1. `src/pipeline/phases/data.py` - Wraps Stages 1-6
2. `src/pipeline/phases/optimization.py` - Wraps Stages 7-9
3. `src/pipeline/phases/training.py` - Wraps Stages 10-15
4. `src/pipeline/phases/deployment.py` - Wraps Stage 16

### Priority 3: Already Complete (Verify & Test)
1. `src/models/base.py` - BaseModel interface (EXISTS)
2. `src/models/registry.py` - Model registry (EXISTS)
3. `src/phase1/stages/datasets/adapters/multi_resolution.py` - 619 lines (EXISTS)
4. `config/optimization/*.yaml` - All 4 optimization configs (EXISTS)

---

## Key Documentation References

| Document | Purpose |
|----------|---------|
| `docs/ARCHITECTURE.md` | Full architecture overview |
| `docs/DEPENDENCY_TREE.md` | Stage dependencies and migration map |
| `docs/implementation/UNIFIED_PIPELINE_ARCHITECTURE.md` | Detailed 16-stage specification |
| `docs/implementation/PHASE_*.md` | Per-phase implementation details |
| `docs/guides/HYPERPARAMETER_TUNING.md` | Optuna optimization guide |
| `docs/guides/FEATURE_OPTIMIZATION.md` | Feature selection/pruning guide |
| `config/optimization/README.md` | Optimization config reference |

---

## Success Criteria

### Definition of Done
1. **Single Entry Point:** `MLPipeline(symbol, horizons, models).run()` executes all 16 stages
2. **Automatic Transitions:** No manual intervention between stages
3. **Checkpointing:** Can resume from any completed stage
4. **State Persistence:** Run state survives process restarts
5. **Unified Config:** Single `MLConfig` controls all stages
6. **CLI Integration:** `ml train --symbol MES --horizons 20` works end-to-end

### Test Scenarios
1. Full pipeline run (all 16 stages, all 23 models)
2. Resume from Stage 7 (after data preparation)
3. Resume from Stage 13 (after optimization)
4. Single model run (e.g., only XGBoost)
5. Multi-symbol run (MES, MGC, MNQ)

---

**Last Updated:** 2026-01-18
**Status:** Ready for Implementation Review
