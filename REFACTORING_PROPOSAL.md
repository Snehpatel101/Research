# ML Factory Complete Refactoring Proposal

**Date:** 2026-01-16
**Analysis Team:** 5 Specialized Agents (Architecture, Config, Data, MLOps, State)
**Scope:** Major refactoring to unified pipeline with 9 first-class timeframes

---

## Executive Summary

Your ML factory currently contains **two pipelines in one codebase**:
1. Traditional single-timeframe pipeline (primary TF with bolt-on MTF)
2. Multi-timeframe aspirations that aren't fully realized

**This proposal redesigns the entire system into ONE unified, cohesive pipeline** where:
- ✅ All 9 timeframes are **equal first-class citizens**
- ✅ Models declaratively specify data needs, system provides
- ✅ **Single configuration surface** (85 config classes → 5 dataclasses)
- ✅ **Clean dependency graph** (no circular imports)
- ✅ **Automatic orchestration** (user says WHAT, system figures out HOW)
- ✅ **Smart resumption** (skip completed work across runs)

**Estimated Effort:** 6-8 weeks
**Expected Benefits:** 60% complexity reduction, 10x faster iteration, production-ready deployment

---

## Table of Contents

1. [Current Problems](#1-current-problems)
2. [Proposed Architecture](#2-proposed-architecture)
3. [Configuration Redesign](#3-configuration-redesign)
4. [Data Layer Design](#4-data-layer-design)
5. [Training Orchestration](#5-training-orchestration)
6. [State Management](#6-state-management)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [Migration Strategy](#8-migration-strategy)

---

## 1. Current Problems

### 1.1 Dual Pipeline Architecture

**Problem:** Two competing mental models in the codebase.

```
Pipeline 1 (Traditional):
  Ingest → Features → Labels → Train (assumes single "primary" TF)

Pipeline 2 (MTF Bolt-On):
  Separate MTF stage tries to retrofit multi-TF support
  MTF features pollute base TF artifacts
```

**Evidence:**
```python
# Current: MTF stage modifies base features
features_5min.parquet  # Base features
↓ [MTF stage runs]
features_5min.parquet  # Now contains MTF indicators mixed in
# Problem: Can't train single-TF model after MTF stage!
```

### 1.2 Circular Dependencies

**Problem:** `phase1 ↔ models` cycle blocks independent testing.

```python
# phase1 imports from models
from src.models.config.data_requirements import MODEL_DATA_REQUIREMENTS

# models imports from phase1
from src.phase1.stages.datasets.container import TimeSeriesDataContainer
from src.phase1.lineage import PipelineLineage
```

**Impact:** Cannot extract either package independently, tight coupling everywhere.

### 1.3 Configuration Sprawl

**Problem:** 85 configuration classes across 5 systems.

```
UnifiedConfig (1116 lines, 12 sections)
PipelineConfig (88 fields)
TrainerConfig (63 fields)
MLConfig (84 fields)
GlobalConfig (YAML-based)
+ 75 more config dataclasses
```

**User Pain:**
```python
# User wants: "Train LSTM on 5min with XGBoost on 15min, stack them"
# Current requirement: Understand 5 config systems, 200+ fields, 3 YAML files
```

### 1.4 Implicit Primary Timeframe

**Problem:** Everything assumes one "primary" timeframe.

```python
# Current storage - which TF is this?
data/features/features.parquet  # Unnamed TF (implicitly 5min)
data/labels/labels.parquet      # Unnamed TF
data/splits/scaled/train.parquet # Unnamed TF
```

**Result:** Cannot train models on different timeframes in same run.

### 1.5 No State Tracking Across Dimensions

**Problem:** State tracks phases but not per-TF/per-model execution.

```python
# Current: Can't answer
"Which timeframes have features computed?"
"Which models are trained for which TF-horizon combos?"
"Can I resume training after LSTM@5min-h20 failed?"
```

---

## 2. Proposed Architecture

### 2.1 Core Principle: One Source, N Artifacts

```
┌────────────────────────────────────────────────────────┐
│ SINGLE CANONICAL 1-MIN OHLCV (immutable source)       │
└────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────┐
│ 9 TIMEFRAME ARTIFACTS (first-class, cached, derived)  │
│  1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h            │
└────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────┐
│ FEATURE BANKS (per-TF, ~180 features each)            │
│  MES/features/1min/features.parquet                   │
│  MES/features/5min/features.parquet                   │
│  ... (9 independent feature sets)                     │
└────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────┐
│ LABEL BANKS (per-TF-horizon)                          │
│  MES/labels/1min/h5.parquet, h10.parquet, ...        │
│  MES/labels/5min/h5.parquet, h10.parquet, ...        │
│  ... (9 TFs × 4 horizons = 36 label sets)            │
└────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────┐
│ MODEL COMPOSITION (models declare needs, system provides)│
│  "XGBoost wants: 15min features + MTF(1m,5m,1h)"      │
│  → DataProvider.get_mtf_features("15min", ["1m"...])  │
│  "LSTM wants: 5min sequences, len=60"                 │
│  → DataProvider.get_sequences("5min", seq_len=60)     │
│  "PatchTST wants: Multi-stream 1m+5m+15m raw OHLCV"   │
│  → DataProvider.get_multistream(["1m","5m","15m"])    │
└────────────────────────────────────────────────────────┘
```

### 2.2 Module Reorganization

**New structure (breaks cycles, clear boundaries):**

```
src/
├── data/                       # Phase 1: OHLCV management
│   ├── ingestion.py           # Load raw 1-min
│   ├── resampler.py           # 1min → 9 timeframes
│   ├── provider.py            # DataProvider API (unified access)
│   ├── cache.py               # Intelligent caching
│   └── artifacts.py           # Manifest management
│
├── features/                   # Phase 2: Feature engineering
│   ├── bank.py                # FeatureBank (per-TF storage)
│   ├── composer.py            # FeatureComposer (model-specific sets)
│   ├── strategies.py          # MODEL_FEATURE_STRATEGIES (23 models)
│   ├── optimization.py        # Optuna feature pruning
│   └── families/              # Feature implementations
│       ├── momentum.py        # RSI, MACD, Stoch
│       ├── volatility.py      # ATR, Bollinger
│       ├── volume.py          # OBV, VWAP
│       ├── microstructure.py  # Spread, VPIN
│       └── wavelets.py        # CWT, DWT
│
├── labeling/                   # Phase 3: Label generation
│   ├── bank.py                # LabelBank (per-TF-horizon storage)
│   ├── triple_barrier.py      # Core labeling
│   ├── optimizer.py           # Optuna barrier optimization
│   └── quality.py             # Sample quality weighting
│
├── models/                     # Phase 4: Model declarations
│   ├── registry.py            # ModelRegistry (unchanged)
│   ├── requirements.py        # NEW: Declarative data requirements
│   ├── composer.py            # NEW: DatasetComposer
│   ├── boosting/              # Model implementations (unchanged)
│   ├── neural/
│   ├── classical/
│   └── ensemble/
│
├── training/                   # Phase 5: Orchestration
│   ├── orchestrator.py        # TrainingOrchestrator (main controller)
│   ├── config.py              # ExperimentConfig (5 dataclasses)
│   ├── planning.py            # ExecutionPlanner (config → plan)
│   ├── coordinators/          # Training mode plugins
│   │   ├── base.py            # StandardCoordinator
│   │   ├── ensemble.py        # EnsembleCoordinator
│   │   ├── walk_forward.py    # WalkForwardCoordinator
│   │   └── regime_aware.py    # RegimeAwareCoordinator
│   └── artifact_store.py      # Unified artifact management
│
└── pipeline/                   # Pipeline orchestration
    ├── graph_state.py         # Graph-based state management
    ├── graph_builder.py       # DAG construction from config
    └── progress.py            # Real-time dashboard
```

**Key changes:**
- ❌ **DELETE** `src/phase1/` (split into `data/`, `features/`, `labeling/`)
- ✅ **NEW** `src/data/` (shared OHLCV management, no model knowledge)
- ✅ **NEW** `src/features/` (independent of models, just produces feature banks)
- ✅ **NEW** `src/labeling/` (independent label banks)
- ✅ **REFACTOR** `src/models/` (add declarative requirements)
- ✅ **REFACTOR** `src/training/` (orchestrator + coordinators)

**Dependency graph (acyclic):**
```
data/     → No dependencies
features/ → Depends on data/ only
labeling/ → Depends on data/ only
models/   → Depends on features/ and labeling/ via interfaces
training/ → Depends on models/ and data/
pipeline/ → Depends on training/
```

---

## 3. Configuration Redesign

### 3.1 Single Configuration Surface

**Problem:** 85 config classes across 5 systems
**Solution:** 5 focused dataclasses with intelligent defaults

```python
# NEW: src/training/config.py

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal

@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str                           # "xgboost", "lstm", "patchtst"
    timeframe: str = "5min"            # Primary training TF
    sequence_length: Optional[int] = None  # For sequence models
    optimize_features: bool = False     # Run Optuna pruning?
    feature_opt_trials: int = 30
    hyperparams: Dict = field(default_factory=dict)

@dataclass
class ExperimentConfig:
    """Single source of truth for ML experiments."""

    # ========== REQUIRED ==========
    models: List[ModelConfig]           # Which models to train
    symbol: str                         # "MES", "MGC"

    # ========== DATA ==========
    horizons: List[int] = field(default_factory=lambda: [20])

    # ========== ENSEMBLE (optional) ==========
    build_ensemble: bool = False
    meta_learner: str = "ridge_meta"

    # ========== TRAINING MODE ==========
    training_mode: Literal["standard", "walk_forward", "regime_aware"] = "standard"

    # ========== ARTIFACTS ==========
    run_id: Optional[str] = None        # Auto-generated if None
    save_predictions: bool = True
    save_inference_bundle: bool = False
```

### 3.2 User Experience

**Example 1: Simple single model**
```python
config = ExperimentConfig(
    models=[ModelConfig(name="xgboost", timeframe="15min")],
    symbol="MES",
)
# System derives: Need 2D tabular, ~200 features, 15min TF
```

**Example 2: Heterogeneous ensemble**
```python
config = ExperimentConfig(
    models=[
        ModelConfig(name="xgboost", timeframe="15min", optimize_features=True),
        ModelConfig(name="lstm", timeframe="5min", sequence_length=60),
        ModelConfig(name="patchtst", timeframe="1min"),
    ],
    symbol="MES",
    build_ensemble=True,
    meta_learner="ridge_meta",
)

# System derives:
# - XGBoost: 2D from 15min, optimize from ~100 → ~60 features
# - LSTM: 3D sequences from 5min, len=60, optimize from ~80 → ~50 features
# - PatchTST: 4D multi-stream 1m+5m+15m, raw OHLCV (no optimization)
# - Need OOF generation for stacking
# - Need meta-learner training
```

**Example 3: Walk-forward validation**
```python
config = ExperimentConfig(
    models=[ModelConfig(name="lstm", timeframe="5min")],
    symbol="MES",
    training_mode="walk_forward",
)
# System derives: 5 expanding windows, standard purge/embargo
```

### 3.3 Configuration to Execution Plan

```python
# System automatically:
1. Resolves model requirements from MODEL_REQUIREMENTS registry
2. Validates data availability (checks if TFs exist)
3. Computes training order (dependency graph)
4. Plans ensemble (if requested)
5. Generates ExecutionPlan with all dependencies

# User never sees ExecutionPlan - it's internal
```

---

## 4. Data Layer Design

### 4.1 Directory Structure

```
data/
├── raw/
│   └── MES_1min.parquet              # Canonical source (immutable)
│
├── timeframes/                       # 9 first-class TFs
│   └── MES/
│       ├── 1min.parquet
│       ├── 5min.parquet
│       ├── 10min.parquet
│       ├── 15min.parquet
│       ├── 20min.parquet
│       ├── 25min.parquet
│       ├── 30min.parquet
│       ├── 45min.parquet
│       ├── 60min.parquet
│       └── _manifest.json            # Hashes, timestamps
│
├── features/                         # Per-TF feature banks
│   └── MES/
│       ├── 1min/
│       │   ├── features.parquet      # ~180 features
│       │   └── schema.json
│       ├── 5min/
│       │   ├── features.parquet
│       │   └── schema.json
│       └── ... (9 TFs)
│
├── labels/                           # Per-TF-horizon labels
│   └── MES/
│       ├── 1min/
│       │   ├── h5.parquet
│       │   ├── h10.parquet
│       │   ├── h15.parquet
│       │   └── h20.parquet
│       ├── 5min/
│       │   └── ... (4 horizons)
│       └── ... (9 TFs)
│
└── splits/                           # Per-TF-horizon splits
    └── MES/
        ├── 1min/
        │   └── h20/
        │       ├── train.parquet
        │       ├── val.parquet
        │       └── test.parquet
        └── ... (9 TFs × 4 horizons)
```

### 4.2 DataProvider API

```python
# src/data/provider.py

class DataProvider:
    """Unified data access for all 9 timeframes."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.feature_bank = FeatureBank(symbol)
        self.label_bank = LabelBank(symbol)

    # ========== SINGLE TIMEFRAME ==========
    def get_features(self, timeframe: str, families: List[str]) -> DataFrame:
        """Get features for one TF."""
        return self.feature_bank.query(timeframe, families)

    def get_labels(self, timeframe: str, horizon: int) -> DataFrame:
        """Get labels for TF-horizon."""
        return self.label_bank.query(timeframe, horizon)

    # ========== MULTI-TIMEFRAME ==========
    def get_mtf_features(
        self,
        primary: str,
        auxiliary: List[str]
    ) -> DataFrame:
        """Get primary features + MTF indicators aligned."""
        primary_feats = self.get_features(primary, ALL_FAMILIES)

        # Load and align auxiliary TFs
        for aux_tf in auxiliary:
            aux_feats = self.get_features(aux_tf, ALL_FAMILIES)
            aligned = self._align_to_primary(aux_feats, primary_tf=primary)
            primary_feats = pd.concat([primary_feats, aligned], axis=1)

        return primary_feats

    def get_multistream(
        self,
        timeframes: List[str],
        seq_len: int
    ) -> Dict[str, DataFrame]:
        """Get raw OHLCV streams for multi-stream models."""
        return {
            tf: self.get_features(tf, ["raw_ohlcv"])
            for tf in timeframes
        }

    # ========== TRAINING DATA ==========
    def get_training_data(
        self,
        timeframe: str,
        horizon: int,
        split: str
    ) -> Tuple[ndarray, ndarray, ndarray]:
        """Get (X, y, weights) for training."""
        # Loads from splits/{symbol}/{tf}/h{horizon}/{split}.parquet
        ...
```

### 4.3 Caching Strategy

**What gets cached:**
- ✅ Resampled timeframes (invalidate if 1min source changes)
- ✅ Features per TF (invalidate if OHLCV or config changes)
- ✅ Labels per TF-horizon (invalidate if features or config changes)
- ✅ In-memory LRU cache (2GB limit, disk spillover)

**Invalidation rules:**
```python
if source_hash != manifest["source_hash"]:
    invalidate_cascade(["timeframes", "features", "labels", "models"])

if feature_config_hash != manifest["feature_hash"]:
    invalidate_cascade(["features", "labels", "models"])

if label_config_hash != manifest["label_hash"]:
    invalidate_cascade(["labels", "models"])
```

---

## 5. Training Orchestration

### 5.1 Three-Layer Architecture

```
┌──────────────────────────────────────────────────────┐
│ Layer 1: TrainingOrchestrator                       │
│  - Takes ExperimentConfig                           │
│  - Builds ExecutionPlan                             │
│  - Selects appropriate Coordinator                  │
│  - Manages artifacts                                │
└──────────────────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────┐
│ Layer 2: Coordinators (pluggable strategies)        │
│  - StandardCoordinator: Train models independently  │
│  - EnsembleCoordinator: OOF + meta-learner          │
│  - WalkForwardCoordinator: Time-series CV           │
│  - RegimeAwareCoordinator: Per-regime models        │
└──────────────────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────┐
│ Layer 3: Providers (resource access)                │
│  - DataProvider: Timeframe-aware data delivery      │
│  - ModelFactory: Model instantiation                │
│  - ArtifactStore: Save/load trained models          │
└──────────────────────────────────────────────────────┘
```

### 5.2 Model Requirement Declaration

**Every model declares its needs:**

```python
# src/models/requirements.py

MODEL_REQUIREMENTS = {
    "xgboost": DataRequirement(
        primary_timeframe="15min",
        data_format="2d_tabular",
        baseline_features=["momentum", "volatility", "volume", "microstructure"],
        mtf_strategy="indicators",  # Add MTF indicator features
        mtf_timeframes=["1min", "5min", "60min"],
        optimize_features=True,
        min_features=20,
        max_features=120,
    ),
    "lstm": DataRequirement(
        primary_timeframe="5min",
        data_format="3d_sequence",
        baseline_features=["momentum", "volatility", "wavelets"],
        mtf_strategy="indicators",
        mtf_timeframes=["1min", "15min"],
        sequence_length=60,
        optimize_features=True,
        min_features=30,
        max_features=100,
    ),
    "patchtst": DataRequirement(
        primary_timeframe="1min",
        data_format="4d_multistream",
        baseline_features=["raw_ohlcv"],
        mtf_strategy="streams",  # Multi-stream ingestion
        mtf_timeframes=["1min", "5min", "15min"],
        sequence_length=96,
        optimize_features=False,  # No feature engineering
    ),
    # ... all 23 models
}
```

### 5.3 Automatic Data Composition

```python
# System automatically composes datasets:

# XGBoost needs 2D tabular with MTF indicators
→ DataProvider.get_mtf_features("15min", ["1min", "5min", "60min"])
→ Returns: ~200 features (base 15min + MTF indicators)

# LSTM needs 3D sequences
→ DataProvider.get_sequences("5min", seq_len=60)
→ Returns: (n_samples, 60, ~150)

# PatchTST needs 4D multi-stream
→ DataProvider.get_multistream(["1min", "5min", "15min"])
→ Returns: (n_samples, 3_streams, 96, 5_OHLCV)
```

### 5.4 Ensemble Workflow

```python
# When user requests ensemble:
config = ExperimentConfig(
    models=[...],  # Heterogeneous bases
    build_ensemble=True,
    meta_learner="ridge_meta",
)

# System automatically:
1. Trains all base models independently
2. Generates OOF predictions using PurgedKFold
3. Aligns OOF predictions to common index
4. Trains meta-learner on stacked OOF
5. Full retrain of base models
6. Saves ensemble bundle
```

---

## 6. State Management

### 6.1 Graph-Based State

**Replace phase-based state with DAG nodes:**

```python
# Current (phase-based):
PipelineState:
  phases:
    - data_generation: completed
    - feature_engineering: completed
    - training: in_progress

# Problem: Can't track per-TF or per-model status
```

**New (graph-based):**

```python
PipelineGraph:
  nodes:
    - data_source@MES: completed
    - timeframe@5min: completed
    - timeframe@15min: completed
    - features@5min: completed
    - features@15min: completed
    - labels@5min-h20: completed
    - labels@15min-h20: completed
    - xgboost@15min-h20: in_progress (epoch 45/100)
    - lstm@5min-h20: pending
    - ensemble[xgb+lstm]@h20: pending

  dependencies:
    xgboost@15min-h20: [features@15min, labels@15min-h20]
    lstm@5min-h20: [features@5min, labels@5min-h20]
    ensemble[...]: [xgboost@15min-h20, lstm@5min-h20]
```

### 6.2 Node Types

```python
class NodeType(Enum):
    DATA_SOURCE = "data_source"      # Raw 1-min OHLCV
    TIMEFRAME = "timeframe"          # Resampled TF (5min, 15min)
    FEATURES = "features"            # Feature engineering
    LABELS = "labels"                # TF-Horizon labeling
    MODEL = "model"                  # Trained model
    ENSEMBLE = "ensemble"            # Ensemble model
```

### 6.3 Smart Resumption

```python
# Run 1: Train XGBoost, LSTM training fails
orchestrator = StateAwareOrchestrator(config, run_id="run_001")
orchestrator.run()
# XGBoost: ✓ completed
# LSTM: ✗ failed (OOM at epoch 45)

# Fix issue, run 2: Resume from checkpoint
orchestrator = StateAwareOrchestrator.from_state("run_001")
orchestrator.resume()
# XGBoost: ⏭ skipped (already completed)
# LSTM: ▶ resumes from epoch 45
# Ensemble: ▶ starts after LSTM completes
```

### 6.4 Invalidation Cascade

```python
# User modifies feature engineering code
# System detects change via config hash

graph.invalidate_node("features@5min", reason="Config changed")

# Cascade invalidation:
features@5min: stale ❌
  ↓
labels@5min-h20: stale ❌ (depends on features@5min)
  ↓
lstm@5min-h20: stale ❌ (depends on labels@5min-h20)
  ↓
ensemble[xgb+lstm]: stale ❌ (depends on lstm@5min-h20)

# Next run: Only re-execute stale nodes
```

### 6.5 Progress Dashboard

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 PIPELINE EXECUTION DASHBOARD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Overall Progress: [████████████████████████░░░░░░░░░░░░░░░░░░░░░] 55.2%
  16/29 complete | 2 in progress | 0 failed

────────────────────────────────────────────────────────────────────────────
📋 Phase Breakdown:
────────────────────────────────────────────────────────────────────────────

✓ Data Preparation:
  [██████████████████████████████] 9/9 (100%)

✓ Timeframe Resampling:
  [██████████████████████████████] 9/9 (100%)

✓ Feature Engineering:
  [██████████████████████████████] 9/9 (100%)

⏳ Label Generation:
  [████████████████████░░░░░░░░░░] 12/16 (75%) | 1 in progress

⏳ Model Training:
  [████████░░░░░░░░░░░░░░░░░░░░░░] 2/5 (40%) | 1 in progress
  ├─ xgboost@15min-h20:  ✓ Complete (Sharpe: 0.82)
  ├─ lstm@5min-h20:      ⏳ 45% (epoch 45/100) [2m 15s]
  ├─ patchtst@1min-h20:  ○ Pending
  ├─ ensemble:           ○ Waiting for base models

────────────────────────────────────────────────────────────────────────────
⏳ Currently Executing:
────────────────────────────────────────────────────────────────────────────
  • labels@15min-h15 [15s]
    Computing triple-barrier labels...

  • lstm@5min-h20 (epoch 45/100) [2m 15s]
    Training: loss=0.023, val_acc=0.67

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 7. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Goal:** Build core infrastructure without breaking existing code.

**Week 1: Data Layer**
- [ ] Create `src/data/` package
- [ ] Implement `DataProvider` core (OHLCV retrieval)
- [ ] Implement `FeatureBank` (per-TF storage)
- [ ] Implement `LabelBank` (per-TF-horizon storage)
- [ ] Add manifest file generation
- [ ] Unit tests for all components

**Week 2: Configuration**
- [ ] Define `ExperimentConfig`, `ModelConfig` dataclasses
- [ ] Implement validation rules
- [ ] Build `ExecutionPlanner` (config → plan)
- [ ] Add backward compatibility layer (old configs → new)
- [ ] Integration tests

**Deliverables:**
- ✅ `src/data/provider.py` (~400 lines)
- ✅ `src/features/bank.py` (~300 lines)
- ✅ `src/labeling/bank.py` (~200 lines)
- ✅ `src/training/config.py` (~200 lines)
- ✅ `src/training/planning.py` (~250 lines)
- ✅ 100% test coverage for new modules

**Success Criteria:**
- DataProvider can load all 9 timeframes
- FeatureBank can store/retrieve per-TF features
- ExperimentConfig validates correctly
- Old configs can be converted to new format

---

### Phase 2: Orchestration (Weeks 3-4)

**Goal:** Build training orchestration with single-model support.

**Week 3: Core Orchestrator**
- [ ] Implement `TrainingOrchestrator`
- [ ] Implement `StandardCoordinator`
- [ ] Implement `ModelFactory` (uses existing registry)
- [ ] Implement `ArtifactStore`
- [ ] Integrate with DataProvider

**Week 4: Model Requirements**
- [ ] Define `DataRequirement` for all 23 models
- [ ] Implement requirement resolution
- [ ] Implement dataset composition
- [ ] Test with 3 representative models (XGBoost, LSTM, PatchTST)

**Deliverables:**
- ✅ `src/training/orchestrator.py` (~300 lines)
- ✅ `src/training/coordinators/base.py` (~250 lines)
- ✅ `src/models/requirements.py` (~400 lines)
- ✅ `src/models/composer.py` (~400 lines)
- ✅ End-to-end test: Train XGBoost on 15min

**Success Criteria:**
- Can train single model using new orchestrator
- All 23 models have requirement declarations
- Dataset composition works for 2D/3D/4D formats
- Artifacts saved to structured directory

---

### Phase 3: Graph State & Resumption (Weeks 5-6)

**Goal:** Implement graph-based state management.

**Week 5: Graph Infrastructure**
- [ ] Implement `PipelineGraph` with DAG
- [ ] Implement `GraphBuilder` (config → graph)
- [ ] Implement node state transitions
- [ ] Implement dependency tracking
- [ ] Implement serialization/deserialization

**Week 6: Smart Resumption**
- [ ] Implement `StateAwareOrchestrator`
- [ ] Implement config reconciliation
- [ ] Implement invalidation cascade
- [ ] Implement progress tracking
- [ ] Build terminal dashboard

**Deliverables:**
- ✅ `src/pipeline/graph_state.py` (~600 lines)
- ✅ `src/pipeline/graph_builder.py` (~300 lines)
- ✅ `src/pipeline/orchestrator.py` (~400 lines)
- ✅ `src/pipeline/progress.py` (~300 lines)
- ✅ Checkpoint/resume tests

**Success Criteria:**
- Graph correctly models all dependencies
- Can resume from any failure point
- Invalidation cascades correctly
- Dashboard shows real-time progress

---

### Phase 4: Ensemble & Advanced Features (Weeks 7-8)

**Goal:** Complete ensemble support and training modes.

**Week 7: Ensemble Coordinator**
- [ ] Implement `EnsembleCoordinator`
- [ ] Implement OOF generation with PurgedKFold
- [ ] Implement prediction alignment (heterogeneous bases)
- [ ] Implement meta-learner training
- [ ] Test with 3-model ensemble

**Week 8: Training Mode Plugins**
- [ ] Implement `WalkForwardCoordinator`
- [ ] Implement `RegimeAwareCoordinator`
- [ ] Implement `MetaLabelingCoordinator`
- [ ] Add inference bundle generation
- [ ] Documentation and examples

**Deliverables:**
- ✅ `src/training/coordinators/ensemble.py` (~350 lines)
- ✅ `src/training/coordinators/walk_forward.py` (~250 lines)
- ✅ `src/training/coordinators/regime_aware.py` (~250 lines)
- ✅ Complete notebook examples
- ✅ API documentation

**Success Criteria:**
- Heterogeneous ensembles work end-to-end
- Walk-forward validation produces correct windows
- Regime-aware training produces per-regime models
- Inference bundle can be deployed

---

### Phase 5: Migration & Cleanup (Weeks 9-10)

**Goal:** Migrate existing experiments, delete legacy code.

**Week 9: Migration**
- [ ] Data migration script (old → new structure)
- [ ] Config migration script
- [ ] State migration script
- [ ] Re-run 5 key experiments to validate
- [ ] Performance benchmarking

**Week 10: Cleanup**
- [ ] Delete `src/phase1/` directory
- [ ] Remove old config classes
- [ ] Remove backward compatibility shims
- [ ] Update all scripts to new API
- [ ] Final documentation update

**Deliverables:**
- ✅ Migration scripts with dry-run mode
- ✅ Migration guide for users
- ✅ Performance comparison (old vs new)
- ✅ Complete API documentation
- ✅ Updated notebooks

**Success Criteria:**
- All existing experiments migrated successfully
- No legacy code remaining
- Performance equal or better than old system
- All tests passing

---

## 8. Migration Strategy

### 8.1 Three-Phase Migration

**Phase 1: Dual Mode (Weeks 1-4)**
- New and old systems coexist
- Feature flag: `USE_NEW_PIPELINE = False` (default)
- New code doesn't break old experiments
- Users can opt-in to new system

**Phase 2: New Default (Weeks 5-8)**
- Feature flag: `USE_NEW_PIPELINE = True` (default)
- Old system still available via flag
- Migration tools ready
- Deprecation warnings in old code

**Phase 3: Legacy Removal (Weeks 9-10)**
- Delete old system
- Remove feature flag
- Complete migration

### 8.2 Backward Compatibility Facade

```python
# src/models/trainer.py (legacy facade)

class Trainer:
    """Legacy Trainer - delegates to new orchestrator."""

    def __init__(self, config: TrainerConfig):
        # Convert old config to new format
        exp_config = ExperimentConfig(
            models=[ModelConfig(
                name=config.model_name,
                timeframe=config.primary_timeframe,
            )],
            symbol=config.symbol,
            horizons=[config.horizon],
        )

        # Use new orchestrator internally
        self.orchestrator = TrainingOrchestrator(exp_config)

    def train(self) -> Dict:
        """Legacy train method."""
        results = self.orchestrator.run()
        return self._convert_results(results)

# Existing scripts work unchanged!
trainer = Trainer(old_config)
trainer.train()
```

### 8.3 Data Migration Script

```bash
# Migrate existing data to new structure
python scripts/migrate_data_layer.py \
  --symbol MES \
  --dry-run  # Preview changes

# Actually migrate
python scripts/migrate_data_layer.py \
  --symbol MES \
  --execute

# Output:
# ✓ Created data/timeframes/MES/
# ✓ Resampled 9 timeframes from 1min
# ✓ Created data/features/MES/5min/
# ✓ Migrated features → data/features/MES/5min/features.parquet
# ✓ Created data/labels/MES/5min/h20.parquet
# ✓ Migration complete (23 artifacts created)
```

---

## 9. Benefits Summary

### 9.1 Quantitative Improvements

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Config classes** | 85 | 5 | -94% |
| **Circular dependencies** | 1 | 0 | -100% |
| **Files > 1000 lines** | 4 | 0 | -100% |
| **Lines to train model** | ~50 (complex config) | ~10 (simple config) | -80% |
| **Pipeline iteration speed** | Baseline | 10x faster | +900% |
| **Resumption support** | Phase-level | Node-level | ✓ |
| **Multi-TF support** | Bolt-on | First-class | ✓ |

### 9.2 Qualitative Improvements

**Developer Experience:**
- ✅ Single config surface (beginner-friendly)
- ✅ Declarative model requirements (self-documenting)
- ✅ Automatic orchestration (less boilerplate)
- ✅ Smart resumption (save hours of iteration)
- ✅ Real-time dashboard (visibility into progress)

**Maintainability:**
- ✅ No circular dependencies (independent testing)
- ✅ Clear separation of concerns (modular refactoring)
- ✅ Plugin architecture (easy to extend)
- ✅ Graph-based state (audit trail)

**Production Readiness:**
- ✅ Inference bundle generation (deploy anywhere)
- ✅ Artifact lineage tracking (reproducibility)
- ✅ Cascade invalidation (prevent stale data bugs)
- ✅ Memory management (OOM recovery)

---

## 10. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Breaking existing experiments** | Medium | High | Dual-mode migration with backward compat facade |
| **Performance regression** | Low | Medium | Benchmark after each phase |
| **User adoption friction** | Medium | Medium | Comprehensive docs + migration scripts |
| **Data migration bugs** | Low | High | Dry-run mode + validation checks |
| **Timeline overrun** | Medium | Medium | Incremental delivery, MVP first |

---

## 11. Success Metrics

### 11.1 Technical Metrics

- [ ] Zero circular dependencies
- [ ] 100% test coverage for new modules
- [ ] All 23 models working with new system
- [ ] Performance ≥ old system (ideally 10x faster)
- [ ] All existing experiments migrated

### 11.2 User Metrics

- [ ] Time to train first model: < 5 minutes (config → results)
- [ ] Config complexity: ≤ 10 lines for typical experiment
- [ ] Error messages: 100% actionable (tell user how to fix)
- [ ] Documentation completeness: 100% API coverage

---

## 12. Open Questions & Decisions Needed

### 12.1 Architecture Decisions

1. **Parallel execution:** Should ready nodes execute in parallel?
   - **Recommendation:** Yes, add `parallel=True` flag to orchestrator
   - **Benefit:** Train multiple models simultaneously on multi-GPU

2. **Remote execution:** Should we support distributed training?
   - **Recommendation:** Phase 2 (after core refactor complete)
   - **Use case:** Multi-node GPU clusters

3. **Caching granularity:** Per-TF or per-feature-family?
   - **Recommendation:** Per-TF (simpler invalidation)
   - **Trade-off:** Slightly larger cache, but clearer semantics

### 12.2 Migration Decisions

4. **Deprecation timeline:** How long to keep old system?
   - **Recommendation:** 2 releases (4-6 weeks)
   - **Rationale:** Balance stability vs. maintenance burden

5. **Breaking changes:** Force upgrade or maintain dual systems?
   - **Recommendation:** Force upgrade after migration period
   - **Rationale:** Dual systems double maintenance cost

---

## 13. Next Steps

### Immediate Actions (This Week)

1. **Review this proposal** with team
2. **Approve architecture** decisions
3. **Assign owners** for each phase
4. **Set up project tracking** (GitHub milestones)
5. **Create feature branch** `refactor/unified-pipeline`

### Week 1 Kickoff

6. **Create new module stubs** (`data/`, `features/`, `labeling/`)
7. **Implement DataProvider** core functionality
8. **Write integration tests** for DataProvider
9. **Daily standups** to unblock issues
10. **Weekly demo** to stakeholders

---

## 14. Conclusion

This refactoring transforms your ML factory from **two competing pipelines** into **one cohesive system** where:

- **All 9 timeframes are equal** - No "primary TF" vs "MTF bolt-on"
- **Configuration is trivial** - 85 classes → 5 dataclasses, 10 lines of config
- **Models are declarative** - "I need X" → system provides X
- **State is queryable** - "What's done? What's next? What failed?"
- **Resumption is smart** - Skip completed work, resume from failures
- **Deployment is turnkey** - Inference bundle includes everything

**Expected outcome:** Production-ready ML factory with 60% less complexity, 10x faster iteration, and complete reproducibility.

**Estimated timeline:** 8-10 weeks with 1-2 engineers
**Risk level:** Medium (mitigated by incremental delivery + backward compat)
**ROI:** High (one-time cost, permanent benefits)

---

**Document Version:** 1.0
**Last Updated:** 2026-01-16
**Next Review:** After Phase 1 completion (Week 2)
