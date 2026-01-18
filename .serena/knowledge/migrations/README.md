# ML Factory Migration Plans

**Date:** 2026-01-18
**Status:** Planning Complete - Ready for Implementation

---

## Overview

These migration plans detail the transition from the current fragmented codebase to the unified 16-stage ML Factory pipeline architecture.

## Migration Documents

| Phase | Stages | Document | Status |
|-------|--------|----------|--------|
| A | 1-6 | `phase_a_data_migration.md` | Complete |
| B | 7-9 | `phase_b_optimization_migration.md` | Complete |
| C | 10-12 | `phase_c_preprocessing_migration.md` | Complete |
| D | 13-15 | `phase_d_training_migration.md` | Complete |
| E | 16 | `phase_e_deployment_migration.md` | Complete |

## Key Metrics

- **Total Source Lines Analyzed:** ~25,000 lines
- **Total Optuna Trials:** 2,550
- **Total Models:** 23
- **Estimated Migration Effort:** 15-20 days

## Migration Strategy

### Core Principle: WRAP, DON'T REWRITE

Each stage wrapper:
1. **Imports** existing implementation from current location
2. **Wraps** with standardized `run()` interface
3. **Validates** inputs/outputs with contracts
4. **Checkpoints** results for resume capability

### Directory Structure After Migration

```
src/pipeline/
├── __init__.py
├── unified.py          # MLPipeline orchestrator
├── config.py           # MLConfig
├── state.py            # PipelineState
└── phases/
    ├── __init__.py
    ├── base.py         # StageWrapper base class
    ├── data.py         # Stages 1-6 (wraps src/phase1/stages/*)
    ├── optimization.py # Stages 7-9 (wraps src/optimization/*, src/labeling/*)
    ├── training.py     # Stages 10-15 (wraps src/models/*, src/training/*)
    └── deployment.py   # Stage 16 (wraps src/inference/*)
```

## Implementation Order

1. **Core Infrastructure** (Priority 1)
   - `src/pipeline/unified.py` - Entry point
   - `src/pipeline/config.py` - Unified config
   - `src/pipeline/state.py` - State management
   - `src/pipeline/phases/base.py` - Stage base class

2. **Phase A: Data** (Priority 2)
   - 6 stage wrappers in `data.py`
   - ~16,690 lines of existing code to wrap

3. **Phase B: Optimization** (Priority 3)
   - 3 stage wrappers in `optimization.py`
   - Optuna integration with study persistence

4. **Phase C: Preprocessing** (Priority 4)
   - 3 stage wrappers (splits, scaling, adapters)
   - Leakage prevention validation

5. **Phase D: Training** (Priority 5)
   - 3 stage wrappers (hyperparams, training, stacking)
   - OOF generation and meta-learner integration

6. **Phase E: Deployment** (Priority 6)
   - ModelBundle V1.1.0 enhancement
   - Artifact packaging from all stages

## Critical Files

| Phase | Critical Files |
|-------|----------------|
| A | `src/phase1/stages/features/engineer.py`, `src/phase1/stages/clean/run.py` |
| B | `src/labeling/optimization.py`, `src/features/selection.py`, `src/features/pruning.py` |
| C | `src/phase1/stages/splits/core.py`, `src/adapters/registry.py` |
| D | `src/optimization/hyperparameters.py`, `src/cross_validation/oof_stacking.py` |
| E | `src/inference/bundle.py`, `src/inference/preprocessing_graph.py` |

---

**Last Updated:** 2026-01-18
