# ML Pipeline Fragmentation Analysis

**Generated:** 2026-01-19
**Status:** Critical architectural review

---

## EXECUTIVE SUMMARY

Your codebase has **3 major fragmentation issues**:

| Issue | Severity | Impact |
|-------|----------|--------|
| **Two Parallel Phase Systems** | 🔴 CRITICAL | Legacy phase1/ (19 stages) vs unified Phases 0-5 |
| **85+ Configuration Classes** | 🔴 CRITICAL | 5 competing config systems, no single source of truth |
| **Multiple Disconnected Implementations** | 🟠 HIGH | 4+ trainers, 5+ orchestrators, 8+ feature selectors |

---

## ROOT CAUSES OF LACK OF COHESION

### 1. Two Completely Separate Pipeline Architectures

```
LEGACY PIPELINE (src/phase1/)         UNIFIED PIPELINE (src/*)
├── 19 stages with full workflow      ├── Phases 0-5 in MIGRATION_PLANS/
├── stages/ingest/                    ├── src/core/ (Phase 0)
├── stages/clean/                     ├── src/features/ (Phase 1)
├── stages/features/                  ├── src/labeling/ (Phase 1B)
├── stages/labeling/                  ├── src/adapters/ (Phase 2)
├── stages/scaling/                   ├── src/training/ (Phase 3)
├── stages/meta_labeling/             ├── src/models/ensemble/ (Phase 4)
└── ... (13 more stages)              └── src/inference/ (Phase 5)
```

**Both exist in parallel!** The legacy `src/phase1/` is a complete working pipeline (21,643 lines), while `src/` contains a partially-migrated unified architecture.

### 2. Circular Dependencies

```
phase1/ imports from models/:
  └── from src.models.config.data_requirements import MODEL_DATA_REQUIREMENTS

models/training/trainer.py imports from phase1/:
  └── from src.phase1.stages.datasets.container import TimeSeriesDataContainer
  └── from src.phase1.lineage import PipelineLineage
```

This prevents clean extraction and forces both systems to coexist.

### 3. Configuration Sprawl (85+ Config Classes)

| System | Files | Status |
|--------|-------|--------|
| `PipelineConfig` (src/core/config.py) | 1 file | ✅ **INTENDED CANONICAL** |
| `UnifiedConfig` | 1116 lines, 12 nested sections | ⚠️ Overlaps |
| `TrainerConfig` | 63 fields | ⚠️ Overlaps |
| `MLConfig` | 84 fields | ⚠️ Overlaps |
| `Phase1PipelineConfig` | Multiple files | ⚠️ Legacy |

**Example duplication:**
```python
config.training.batch_size = 512       # UnifiedConfig
trainer_config.batch_size = 512        # TrainerConfig
ml_config.training_config.batch_size   # MLConfig
# Which is canonical?
```

---

## SPECIFIC PARALLEL FILES IDENTIFIED

### Trainers (4 implementations)

| File | Lines | Status |
|------|-------|--------|
| `src/models/training/trainer.py` | 786 | Main |
| `src/training/model_trainer.py` | 569 | Alternative |
| `src/training/regime_trainer.py` | 792 | Specialized |
| `src/training/modes/regime_aware.py` | 325 | **DUPLICATE** |

### Orchestrators (5 implementations)

| File | Purpose | Status |
|------|---------|--------|
| `src/training/orchestrator.py` | Original | Legacy |
| `src/training/unified_orchestrator.py` | **PHASE_3** | Recommended |
| `src/cross_validation/cv_orchestrator.py` | CV | Specialized |
| `src/inference/orchestrator.py` | Inference | Specialized |
| `src/models/ensemble/orchestrator.py` | Ensemble | Specialized |

### Feature Selection (8 implementations!)

| File | Status |
|------|--------|
| `src/models/feature_selection/manager.py` | Original |
| `src/feature_selection/manager.py` | **Consolidated** |
| `src/cross_validation/feature_selector.py` | **DEPRECATED** |
| `src/cross_validation/cv_feature_selection.py` | Per-fold |
| `src/features/selection.py` | Optuna-based (Phase 1B) |
| `src/training/feature_selector.py` | Config-based |
| `src/feature_selection/walk_forward.py` | Walk-forward |
| `src/phase1/utils/feature_selection.py` | Legacy |

### Triple-Barrier Labeling (2 implementations)

| File | Implementation |
|------|----------------|
| `src/labeling/triple_barrier.py` | Legacy (19KB) |
| `src/phase1/stages/labeling/triple_barrier.py` | Numba-optimized (646 lines) |

**Within same file, 95% code duplication:**
- `triple_barrier_numba()` (lines 34-174)
- `triple_barrier_numba_with_costs()` (lines 177-323) - only adds cost adjustment

---

## REPEATED CODE PATTERNS

| Pattern | Occurrences | Files |
|---------|-------------|-------|
| Empty DataFrame validation | 14 | Scattered across modules |
| Required columns validation | 3+ | `base.py`, `utils.py`, `validators.py` |
| Feature `add_*` functions | 58 | All feature modules |
| Array conversion (`.values`) | 27 | All labeling modules |
| Metadata dictionary creation | 6+ | All labelers |

---

## PHASE FILES INVENTORY

### Unified Phases (CURRENT/ACTIVE)

| Phase | File | Status | Purpose |
|-------|------|--------|---------|
| **PHASE_0** | `MIGRATION_PLANS/PHASE_0_IMPLEMENTATION.md` | 95% | Foundation: Core types, interfaces, constants |
| **PHASE_1** | `MIGRATION_PLANS/PHASE_1_IMPLEMENTATION.md` | 90% | Unified features: 162 base features, 12 families |
| **PHASE_1B** | `MIGRATION_PLANS/PHASE_1B_IMPLEMENTATION.md` | 90% | Labeling & Optuna: Triple-barrier, feature selection |
| **PHASE_2** | `MIGRATION_PLANS/PHASE_2_IMPLEMENTATION.md` | 95% | Adapters: Tabular, Sequence, Multi-stream |
| **PHASE_3** | `MIGRATION_PLANS/PHASE_3_IMPLEMENTATION.md` | 90% | Training: UnifiedTrainingOrchestrator, CV methods |
| **PHASE_4** | `MIGRATION_PLANS/PHASE_4_IMPLEMENTATION.md` | 95% | Meta-learners: Ensemble building, OOF alignment |
| **PHASE_5** | `MIGRATION_PLANS/PHASE_5_IMPLEMENTATION.md` | 95% | Inference: Bundles, PreprocessingGraph, deployment |

### Legacy Phases (docs/implementation/)

| Phase | File | Purpose |
|-------|------|---------|
| Phase 1 | `PHASE_1_INGESTION.md` | OHLCV ingestion & cleaning |
| Phase 2 | `PHASE_2_MTF_UPSCALING.md` | Multi-timeframe upscaling |
| Phase 3 | `PHASE_3_FEATURES.md` | Feature engineering |
| Phase 4 | `PHASE_4_LABELING.md` | Triple-barrier labeling |
| Phase 5 | `PHASE_5_ADAPTERS.md` | Data adapters |
| Phase 6 | `PHASE_6_TRAINING.md` | Model training |
| Phase 7 | `PHASE_7_META_LEARNER_STACKING.md` | Ensemble stacking |

### Phase Registry (src/ml_pipeline/phase_registry.py)

14 registered phases with dependency tracking:
1. `data_generation` (stage 1.0)
2. `data_cleaning` (stage 2.0)
3. `feature_engineering` (stage 3.0)
4. `initial_labeling` (stage 4.0)
5. `ga_optimize` (stage 5.0) - Can skip
6. `final_labels` (stage 6.0)
7. `create_splits` (stage 7.0)
8. `feature_scaling` (stage 7.5)
9. `build_datasets` (stage 7.6)
10. `validate_scaled` (stage 7.7)
11. `validate` (stage 8.0)
12. `generate_report` (stage 9.0)
13. `training` (stage 10.0)
14. `evaluation` (stage 11.0) - Can skip

---

## MIGRATION PATH TO UNIFIED PIPELINE

The codebase already has **MLFactory** (`src/factory.py`, 805 lines) intended as THE single entry point:

```python
class MLFactory:
    """
    THE single entry point for the entire ML pipeline.

    Orchestrates:
    - PHASE_1: Feature computation (162 base + 240 MTF features)
    - PHASE_1B: Labeling (triple barrier / Optuna optimization)
    - PHASE_2: Data preparation via adapters (2D/3D/4D)
    - PHASE_3: Training (standard, walk-forward, regime-aware, meta-labeling)
    - PHASE_4: Ensemble building (heterogeneous stacking)
    - PHASE_5: Bundling for inference
    """
```

### PHASE A: Configuration Consolidation

1. Make `PipelineConfig` the ONLY configuration class
2. Delete/deprecate `UnifiedConfig`, `TrainerConfig`, `MLConfig`
3. Add conversion methods for backward compatibility

### PHASE B: Delete Legacy Duplicates

```bash
# Files to DELETE after migrating references:
rm src/labeling/triple_barrier.py          # Use phase1/ version
rm src/training/modes/regime_aware.py       # Use regime_trainer.py
rm src/cross_validation/feature_selector.py # Explicitly DEPRECATED
rm src/models/feature_selection/            # Use src/feature_selection/
```

### PHASE C: Break Circular Dependencies

1. Extract `TimeSeriesDataContainer` to `src/core/`
2. Move `PipelineLineage` to `src/core/`
3. Remove `phase1/` → `models/` imports

### PHASE D: Connect Orphaned Components

```
Current:  Adapters → [GAP] → Trainer
Target:   Adapters → UnifiedTrainingOrchestrator → Trainer

Current:  Training → [GAP] → Inference
Target:   Training → InferenceBundle → InferencePipeline
```

### PHASE E: Centralize Validation Utilities

```python
# Create src/core/validation_utils.py
def require_non_empty_df(df: pd.DataFrame, context: str) -> None:
    if df.empty:
        raise ValueError(f"DataFrame is empty in {context}")

def require_columns(df: pd.DataFrame, columns: List[str]) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
```

---

## QUICK WINS (Immediate Impact)

| Action | Effort | Impact |
|--------|--------|--------|
| Delete `src/cross_validation/feature_selector.py` | 5 min | Remove confusion |
| Delete `src/training/modes/regime_aware.py` | 10 min | Remove duplicate |
| Merge triple_barrier Numba functions | 30 min | Remove 150 lines |
| Centralize validation utilities | 1 hour | Remove 14 duplicates |
| Mark legacy files with `# DEPRECATED` | 30 min | Clarify status |

---

## FINAL ARCHITECTURE TARGET

```
src/
├── factory.py                    # MLFactory - THE entry point
├── core/                         # Phase 0: Types, Config, Validation
│   ├── config.py                 # PipelineConfig (SINGLE CONFIG)
│   ├── types.py                  # All enums
│   ├── validation.py             # Centralized validation
│   └── container.py              # TimeSeriesDataContainer (moved)
├── features/                     # Phase 1: 162 features, 12 families
├── labeling/                     # Phase 1B: Triple-barrier (Numba)
├── adapters/                     # Phase 2: Tabular/Sequence/MultiStream
├── training/                     # Phase 3: UnifiedTrainingOrchestrator
├── models/ensemble/              # Phase 4: Meta-learners
└── inference/                    # Phase 5: Bundles, deployment

# DELETE:
src/phase1/                       # Replace with unified phases
src/models/feature_selection/     # Use src/feature_selection/
src/labeling/triple_barrier.py    # Duplicate
```

---

## KEY TAKEAWAYS

1. **`src/phase1/` is a COMPLETE legacy pipeline** (19 stages, 21K+ lines) that should be migrated to unified Phases 0-5, then deleted

2. **`MLFactory` (src/factory.py) is your unified entry point** - it's 90% complete but not fully connected to adapters and inference

3. **Critical duplicates to delete immediately:**
   - `src/training/modes/regime_aware.py` (duplicate of `regime_trainer.py`)
   - `src/cross_validation/feature_selector.py` (marked DEPRECATED)
   - `src/labeling/triple_barrier.py` (use `phase1/stages/labeling/` version)

4. **Configuration must consolidate to `PipelineConfig`** - currently 85+ config classes create maintenance nightmare

5. **Adapters are orphaned** - `AdapterFactory` exists with 3 adapter types (Tabular/Sequence/MultiStream) but `UnifiedTrainingOrchestrator` doesn't use them

---

## DATA FLOW: CURRENT vs TARGET

### Current (Fragmented)

```
Raw OHLCV
    ↓
[phase1/stages/ingest/] ──────────────────────┐
    ↓                                          │
[phase1/stages/clean/]                         │
    ↓                                          │ LEGACY
[phase1/stages/features/]                      │ PIPELINE
    ↓                                          │
[phase1/stages/labeling/]                      │
    ↓                                          │
[phase1/stages/scaling/] ─────────────────────┘
    ↓
    ╳ GAP ╳
    ↓
[src/training/] ──────────────────────────────┐
    ↓                                          │ UNIFIED
[src/models/ensemble/]                         │ (ORPHANED)
    ↓                                          │
    ╳ GAP ╳                                    │
    ↓                                          │
[src/inference/] ─────────────────────────────┘
```

### Target (Unified via MLFactory)

```
Raw OHLCV
    ↓
MLFactory.run(df)
    ├── PHASE_1: compute_all_features() → 162 features
    ├── PHASE_1B: TripleBarrierLabeler → labels
    ├── PHASE_2: AdapterFactory → 2D/3D/4D data
    ├── PHASE_3: UnifiedTrainingOrchestrator → trained models + OOF
    ├── PHASE_4: EnsembleOrchestrator → meta-learner
    └── PHASE_5: InferenceBundle → deployment artifact
    ↓
PipelineResult (single output object)
```

---

## METRICS

| Metric | Current | Target |
|--------|---------|--------|
| Config classes | 85+ | 1 (PipelineConfig) |
| Trainer implementations | 4 | 1 |
| Orchestrators | 5 | 3 (Training, Ensemble, Inference) |
| Feature selectors | 8 | 2 (manager + Optuna) |
| Pipeline entry points | 3+ | 1 (MLFactory) |
| Triple-barrier implementations | 2 | 1 |
| Validation utility copies | 14 | 1 |

---

## ABSOLUTE FILE PATHS FOR REFERENCE

### Critical Files to Keep
- `/home/jake/Desktop/Research/src/factory.py` - MLFactory (THE entry point)
- `/home/jake/Desktop/Research/src/core/config.py` - PipelineConfig (canonical config)
- `/home/jake/Desktop/Research/src/training/unified_orchestrator.py` - Training orchestrator
- `/home/jake/Desktop/Research/src/adapters/factory.py` - AdapterFactory
- `/home/jake/Desktop/Research/src/phase1/stages/labeling/triple_barrier.py` - Numba labeling

### Critical Files to DELETE
- `/home/jake/Desktop/Research/src/labeling/triple_barrier.py` - Duplicate
- `/home/jake/Desktop/Research/src/training/modes/regime_aware.py` - Duplicate
- `/home/jake/Desktop/Research/src/cross_validation/feature_selector.py` - Deprecated
- `/home/jake/Desktop/Research/src/models/feature_selection/` - Entire directory

### Migration Plans
- `/home/jake/Desktop/Research/MIGRATION_PLANS/PHASE_0_IMPLEMENTATION.md`
- `/home/jake/Desktop/Research/MIGRATION_PLANS/PHASE_1_IMPLEMENTATION.md`
- `/home/jake/Desktop/Research/MIGRATION_PLANS/PHASE_1B_IMPLEMENTATION.md`
- `/home/jake/Desktop/Research/MIGRATION_PLANS/PHASE_2_IMPLEMENTATION.md`
- `/home/jake/Desktop/Research/MIGRATION_PLANS/PHASE_3_IMPLEMENTATION.md`
- `/home/jake/Desktop/Research/MIGRATION_PLANS/PHASE_4_IMPLEMENTATION.md`
- `/home/jake/Desktop/Research/MIGRATION_PLANS/PHASE_5_IMPLEMENTATION.md`
