# Reconciled Refactoring Plan: Analysis & Unified Strategy

**Date:** 2026-01-21
**Status:** Conflict Analysis + Reconciled Implementation Plan

---

## Executive Summary

After deep analysis, **1.md and 2.md are NOT fully cohesive** - they share the same goals but propose **conflicting architectural decisions** for achieving them. This document reconciles the differences into a single actionable plan.

---

## Part 1: Cohesion Analysis

### What They AGREE On (Shared Goals)

| Goal | 1.md | 2.md | Status |
|------|------|------|--------|
| Single entry point | Yes | Yes | **Aligned** |
| Single config system | Yes (`UnifiedConfig`) | Yes (`UnifiedConfig`) | **Aligned** |
| Consolidate features | `features/` absorbs `feature_selection/`, `feature_store/` | Same | **Aligned** |
| Reduce module count | 37 → ~16 | 24 → 15 | **Aligned** |
| Refactor large files | <800 lines target | <800 lines target | **Aligned** |
| Merge small modules into `common/` | monitoring, validation, coordination | Same | **Aligned** |
| 5-week timeline | Yes | Yes | **Aligned** |

### What They CONFLICT On (Critical Differences)

| Aspect | 1.md Proposes | 2.md Proposes | Conflict Severity |
|--------|---------------|---------------|-------------------|
| **Entry Point Name** | `MLPipeline` | `MLFactory` | **HIGH** |
| **Entry Point Location** | `src/pipeline/orchestrator.py` | `src/factory.py` (existing) | **HIGH** |
| **Role of `src/pipeline/`** | Master 16-stage orchestrator | Data preparation only (Phases 1-5) | **HIGH** |
| **`cross_validation/`** | Merge into `training/cv/` | Keep as separate module | **MEDIUM** |
| **`monitoring/`** | Keep as separate module | Merge into `common/` | **LOW** |
| **`contracts/`** | Not mentioned | Keep as separate module | **LOW** |
| **`optimization/`** | Part of pipeline phases | Keep as separate module | **LOW** |

---

## Part 2: Conflict Deep Dive

### Critical Conflict #1: Entry Point Identity

**1.md Vision:**
```python
from src import MLPipeline

pipeline = MLPipeline(symbol="MES", models=["xgboost"])
pipeline.run()
```
- New class `MLPipeline` in `src/pipeline/orchestrator.py`
- Deprecate and remove `MLFactory`
- `pipeline/` becomes the "master" module

**2.md Vision:**
```python
from src import MLFactory, UnifiedConfig

config = UnifiedConfig(symbol="MES", models=["xgboost"])
factory = MLFactory(config)
factory.run(df)
```
- Keep existing `MLFactory` in `src/factory.py`
- `pipeline/` remains data-only
- More incremental change

**Impact:** These are mutually exclusive approaches. You cannot do both.

### Critical Conflict #2: Role of `src/pipeline/`

| Aspect | 1.md | 2.md |
|--------|------|------|
| Scope | All 16 stages (data → training → bundling) | Only data preparation (stages 1-12) |
| Contains | `orchestrator.py` (MLPipeline) + `phases/` + `stages/` | Only `runner.py` + `stages/` |
| Relationship to training | `pipeline/` imports and orchestrates `training/` | `factory.py` orchestrates both `pipeline/` and `training/` |

### Critical Conflict #3: API Signature

**1.md:** Config-light, keyword args
```python
MLPipeline(symbol="MES", models=["xgboost", "lstm"])
```

**2.md:** Config-heavy, explicit config object
```python
config = UnifiedConfig(...)
MLFactory(config)
```

---

## Part 3: Reconciliation Decision

### Recommended Approach: **Hybrid (1.md Architecture + 2.md Pragmatism)**

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Entry Point Name | `MLPipeline` | Clearer name, matches the concept |
| Entry Point Location | `src/pipeline/orchestrator.py` | Logical home for pipeline orchestration |
| `MLFactory` | Deprecate with shim | Backward compatibility for 2 versions |
| `src/pipeline/` Role | Master 16-stage orchestrator | 1.md's approach is cleaner architecturally |
| API Style | Both supported | `MLPipeline(symbol=...)` AND `MLPipeline(config=UnifiedConfig(...))` |
| Module Consolidation | Follow 1.md's structure | More thorough reorganization |
| `cross_validation/` | Merge into `training/cv/` | Training owns CV logic |
| `monitoring/` | Keep separate (not merge into `common/`) | Monitoring is a first-class concern |

### Backward Compatibility Shim

```python
# src/factory.py (deprecated, kept for 2 versions)
import warnings
from src.pipeline import MLPipeline

class MLFactory:
    """DEPRECATED: Use MLPipeline instead."""

    def __init__(self, config):
        warnings.warn(
            "MLFactory is deprecated. Use 'from src import MLPipeline' instead. "
            "Will be removed in v3.0.",
            DeprecationWarning,
            stacklevel=2
        )
        self._pipeline = MLPipeline(config=config)

    def run(self, df=None):
        return self._pipeline.run()
```

---

## Part 4: Unified Target Architecture

### Final Directory Structure (Reconciled)

```
src/
├── __init__.py                    # Exports: MLPipeline, UnifiedConfig, ModelRegistry
├── factory.py                     # DEPRECATED shim → MLPipeline
│
├── pipeline/                      # 16-STAGE UNIFIED ORCHESTRATION (from 1.md)
│   ├── __init__.py                # Exports: MLPipeline
│   ├── orchestrator.py            # MLPipeline class (THE entry point)
│   ├── state.py                   # PipelineState, checkpointing
│   ├── phases/                    # Phase groupings
│   │   ├── data.py                # Stages 1-6
│   │   ├── optimization.py        # Stages 7-9
│   │   ├── preprocessing.py       # Stages 10-12
│   │   ├── training.py            # Stages 13-15
│   │   └── deployment.py          # Stage 16
│   └── stages/                    # Individual stage implementations
│       └── [existing stages]
│
├── models/                        # MODEL IMPLEMENTATIONS ONLY
│   ├── registry.py
│   ├── base.py
│   ├── boosting/
│   ├── neural/
│   ├── classical/
│   └── ensemble/
│
├── training/                      # TRAINING EXECUTION
│   ├── orchestrator.py            # Refactored <800 lines
│   ├── trainer.py
│   ├── cv/                        # Absorbs cross_validation/
│   │   ├── purged_kfold.py
│   │   └── oof_generator.py
│   └── modes/
│       ├── standard.py
│       ├── walk_forward.py
│       └── regime_aware.py
│
├── features/                      # CONSOLIDATED (from both docs)
│   ├── compute/
│   ├── selection/                 # Absorbs feature_selection/
│   └── store/                     # Absorbs feature_store/
│
├── inference/                     # MODEL SERVING
├── labeling/                      # CONSOLIDATED
├── evaluation/                    # EVALUATION STRATEGIES
├── backtesting/                   # BACKTESTING ENGINE
├── optimization/                  # OPTUNA (kept separate)
├── monitoring/                    # KEPT SEPARATE (not merged)
├── config/                        # CONSOLIDATED CONFIG
├── adapters/                      # DATA FORMAT ADAPTERS
├── contracts/                     # KEPT (from 2.md)
├── cli/                           # COMMAND-LINE INTERFACE
├── common/                        # SHARED UTILITIES
│   ├── timeframes.py
│   ├── validation.py              # Absorbs validation/
│   └── coordination.py            # Absorbs coordination/
└── utils/
```

### Module Count Comparison

| State | Count | Notes |
|-------|-------|-------|
| Current | 37 (per 1.md) or 24 (per 2.md) | Discrepancy likely due to counting method |
| 1.md Target | ~16 | More aggressive consolidation |
| 2.md Target | 15 | Slightly different structure |
| **Reconciled Target** | **16** | Best of both |

### Modules to Remove/Merge (Final List)

| Current | Action | Target |
|---------|--------|--------|
| `ml_pipeline/` | REMOVE | → `pipeline/orchestrator.py` |
| `feature_selection/` | MERGE | → `features/selection/` |
| `feature_store/` | MERGE | → `features/store/` |
| `cross_validation/` | MERGE | → `training/cv/` |
| `core/` | SPLIT | config → `config/`, types → `common/` |
| `validation/` | MERGE | → `common/validation.py` |
| `coordination/` | MERGE | → `common/coordination.py` |

### Modules to KEEP Separate (Decision)

| Module | Reason |
|--------|--------|
| `monitoring/` | First-class production concern |
| `optimization/` | Clear single responsibility (Optuna) |
| `contracts/` | Domain-specific, isolated |
| `backtesting/` | Distinct from training |
| `evaluation/` | Distinct from training execution |

---

## Part 5: Reconciled Implementation Plan

### Phase 0: Preparation (Day 1)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 0.1 | Create feature branch `refactor/unified-pipeline` | Branch created |
| 0.2 | Document current import graph | `docs/import_graph_before.md` |
| 0.3 | Run full test suite, establish baseline | All tests pass |
| 0.4 | Create rollback plan | `docs/rollback.md` |

### Phase 1: Entry Point Unification (Days 2-4)

| Task | Description | Files Affected |
|------|-------------|----------------|
| 1.1 | Create `src/pipeline/orchestrator.py` with `MLPipeline` class | New file |
| 1.2 | Implement `MLPipeline.run()` delegating to existing runners | `orchestrator.py` |
| 1.3 | Create `src/pipeline/phases/` directory structure | New directory |
| 1.4 | Update `src/__init__.py` to export `MLPipeline` | `__init__.py` |
| 1.5 | Add deprecation shim to `src/factory.py` | `factory.py` |
| 1.6 | Add deprecation shim to `src/ml_pipeline/unified.py` | `unified.py` |
| 1.7 | Write integration tests for `MLPipeline` | `tests/integration/` |

**Exit Criteria:** `from src import MLPipeline; MLPipeline(...).run()` works end-to-end

### Phase 2: Configuration Consolidation (Days 5-7)

| Task | Description | Files Affected |
|------|-------------|----------------|
| 2.1 | Audit all config classes, create mapping document | `docs/config_mapping.md` |
| 2.2 | Extend `UnifiedConfig` to cover all fields from other configs | `config/unified.py` |
| 2.3 | Create `config/adapters.py` for backward compat | New file |
| 2.4 | Update `MLPipeline` to accept both styles | `orchestrator.py` |
| 2.5 | Deprecate `PipelineConfig`, `TrainerConfig`, `MLConfig` | Multiple files |
| 2.6 | Remove `src/ml_pipeline/config.py` | Delete file |

**Exit Criteria:** Single `UnifiedConfig` used everywhere, old configs still work with warnings

### Phase 3: Feature Consolidation (Days 8-12)

| Task | Description | Files Affected |
|------|-------------|----------------|
| 3.1 | Create `src/features/selection/` directory | New directory |
| 3.2 | Move `feature_selection/*.py` → `features/selection/` | ~10 files |
| 3.3 | Create `src/features/store/` directory | New directory |
| 3.4 | Move `feature_store/*.py` → `features/store/` | ~5 files |
| 3.5 | Update all imports across codebase | ~50+ files |
| 3.6 | Add deprecation shims in old locations | `feature_selection/__init__.py` |
| 3.7 | Delete empty directories | `feature_selection/`, `feature_store/` |
| 3.8 | Update pipeline stages to use consolidated imports | `pipeline/stages/features/` |

**Exit Criteria:** All feature-related code in `src/features/`, old imports work with warnings

### Phase 4: Training Consolidation (Days 13-17)

| Task | Description | Files Affected |
|------|-------------|----------------|
| 4.1 | Create `src/training/cv/` directory | New directory |
| 4.2 | Move `cross_validation/*.py` → `training/cv/` | ~5 files |
| 4.3 | Extract `training/cv_integration.py` from `unified_orchestrator.py` | ~350 lines |
| 4.4 | Extract `training/artifact_manager.py` from `unified_orchestrator.py` | ~300 lines |
| 4.5 | Extract `training/metrics_reporter.py` from `unified_orchestrator.py` | ~200 lines |
| 4.6 | Extract `training/mode_handlers.py` from `unified_orchestrator.py` | ~150 lines |
| 4.7 | Slim `unified_orchestrator.py` to <600 lines | `unified_orchestrator.py` |
| 4.8 | Merge `models/training/trainer.py` → `training/trainer.py` | Multiple files |
| 4.9 | Remove `src/models/trainer.py` (re-export only) | Delete file |
| 4.10 | Update all imports | ~30+ files |

**Exit Criteria:** `unified_orchestrator.py` <600 lines (extracting ~1000 lines total), all training logic in `src/training/`

### Phase 5: Small Module Consolidation (Days 18-20)

| Task | Description | Files Affected |
|------|-------------|----------------|
| 5.1 | Move `validation/*.py` → `common/validation.py` | ~3 files → 1 |
| 5.2 | Move `coordination/*.py` → `common/coordination.py` | ~2 files → 1 |
| 5.3 | Split `core/`: config → `config/`, types → `common/` | ~5 files |
| 5.4 | Delete empty `core/`, `validation/`, `coordination/` | Delete directories |
| 5.5 | Update all imports | ~20+ files |

**Exit Criteria:** 16 focused modules, no empty directories

### Phase 6: Cleanup & Documentation (Days 21-25)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 6.1 | Run full test suite | All tests pass |
| 6.2 | Fix any broken imports | Clean codebase |
| 6.3 | Update all docstrings | Accurate docs |
| 6.4 | Create migration guide | `docs/MIGRATION.md` |
| 6.5 | Update README | `README.md` |
| 6.6 | Create architecture diagram | `docs/architecture.png` |
| 6.7 | Tag release v2.0 | Git tag |

**Exit Criteria:** Clean, documented, tested codebase ready for release

---

## Part 6: Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking external scripts | High | High | 2-version deprecation period with clear warnings |
| Circular imports after merge | Medium | High | Careful import ordering, lazy imports, test after each phase |
| Test failures | Medium | Medium | Run tests after each task, not just each phase |
| Missing edge cases | Medium | Medium | Shadow test: run old and new in parallel |
| Documentation drift | Low | Medium | Update docs in same commit as code |

---

## Part 7: Success Criteria

### Quantitative

| Metric | Before | After | Target Met? |
|--------|--------|-------|-------------|
| Top-level modules | ~24* | 16 | Pending |
| Entry points | 4 | 1 (`MLPipeline`) | Pending |
| Config classes | 5+ | 1 (`UnifiedConfig`) | Pending |
| Feature modules | 3 | 1 (`features/`) | Pending |
| Max file size | 1,599 lines | <800 lines | Pending |
| `unified_orchestrator.py` | 1,599 lines | <600 lines | Pending |

*\*Note: 1.md claims 37 modules, 2.md claims 24. The 24 count (from 2.md's detailed enumeration) appears accurate. The 37 figure likely includes subdirectories or uses a different counting methodology.*

### Qualitative

- [ ] `from src import MLPipeline` is the obvious entry point
- [ ] New developers understand structure in <30 min
- [ ] Single import path for each concept
- [ ] No circular dependencies
- [ ] All existing tests pass
- [ ] Deprecation warnings guide migration clearly

---

## Part 8: Decision Log

| Decision | Options Considered | Chosen | Rationale |
|----------|-------------------|--------|-----------|
| Entry point name | MLPipeline, MLFactory | MLPipeline | Clearer semantics |
| Entry point location | `factory.py`, `pipeline/orchestrator.py` | `pipeline/orchestrator.py` | Pipeline is the master concept |
| `cross_validation/` fate | Keep separate, merge into training | Merge | Training owns CV |
| `monitoring/` fate | Keep separate, merge into common | Keep | First-class concern |
| Config unification | Keep multiple, full merge | Full merge with adapters | Clean architecture |
| Deprecation period | 1 version, 2 versions | 2 versions | User safety |

---

## Appendix: Quick Reference

### Before & After Imports

```python
# BEFORE (v1.x / v2.x)
from src.factory import MLFactory
from src.ml_pipeline import MLPipeline, MLConfig
from src.pipeline.runner import PipelineRunner
from src.feature_selection import PurgedSelector
from src.feature_store import FeatureCache
from src.cross_validation import PurgedKFold
from src.core import PipelineConfig

# AFTER (v3.0+)
from src import MLPipeline, UnifiedConfig
from src.features.selection import PurgedSelector
from src.features.store import FeatureCache
from src.training.cv import PurgedKFold
from src.config import UnifiedConfig
```

### CLI Changes

```bash
# BEFORE
python -m src.pipeline_cli run --symbol MES
python -m src.factory run --config config.yaml

# AFTER
ml run --symbol MES --models xgboost,lstm
ml data --symbol MES
ml train --models xgboost
ml status
```

---

*Last Updated: 2026-01-21*
*Status: Ready for Implementation*
