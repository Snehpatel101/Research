# ML Factory Source Improvement Plan

## Executive Summary

The ML Model Factory codebase is **functionally complete** with 23 models, 17 pipeline stages, and production-grade leakage prevention. However, the `src/` directory has grown organically to **24 top-level modules**, creating:

1. **Multiple entry points** (4 different orchestrators)
2. **Configuration sprawl** (4+ config systems)
3. **Feature logic fragmentation** (3 separate feature-related modules)
4. **Naming confusion** (`pipeline/` vs `ml_pipeline/` vs `core/`)

This document provides a structured plan to consolidate to **~15 focused modules** while preserving all functionality.

---

## Current State Analysis

### Top-Level Modules in src/ (24 total)

| Module | Purpose | Lines | Status |
|--------|---------|-------|--------|
| `adapters/` | 2D/3D/4D data format adapters | ~1,200 | ✅ Keep |
| `backtesting/` | Backtesting engine | ~800 | ✅ Keep |
| `cli/` | Command-line interface | ~400 | ✅ Keep |
| `common/` | Shared utilities, timeframes | ~600 | ✅ Keep |
| `config/` | Configuration (unified, global, smart) | ~1,500 | ⚠️ Consolidate |
| `contracts/` | Futures contract definitions | ~300 | ✅ Keep |
| `coordination/` | Multi-process coordination | ~400 | 🔄 Merge → common |
| `core/` | Core types, interfaces, config | ~1,800 | ⚠️ Overlaps with config |
| `cross_validation/` | PurgedKFold, OOF generation | ~1,200 | ✅ Keep |
| `evaluation/` | Model evaluation strategies | ~600 | ✅ Keep |
| `feature_selection/` | Feature selection algorithms | ~900 | 🔄 Merge → features |
| `feature_store/` | Feature caching/versioning | ~500 | 🔄 Merge → features |
| `features/` | Feature computation | ~700 | ⚠️ Absorb others |
| `inference/` | Model serving, bundles | ~1,500 | ✅ Keep |
| `labeling/` | Triple-barrier labeling | ~600 | ✅ Keep |
| `ml_pipeline/` | Unified training pipeline | ~1,200 | ⚠️ Redundant name |
| `models/` | Model registry, implementations | ~4,500 | ✅ Keep |
| `monitoring/` | Runtime monitoring | ~400 | 🔄 Merge → common |
| `optimization/` | Optuna optimization | ~800 | ✅ Keep |
| `pipeline/` | Data pipeline (17 stages) | ~5,000 | ⚠️ Rename clarity |
| `training/` | Training orchestration | ~2,500 | ⚠️ Large files |
| `utils/` | General utilities | ~500 | ✅ Keep |
| `validation/` | Data validation | ~400 | 🔄 Merge → common |

**Files at src/ root:**
- `factory.py` (573 lines) - MLFactory entry point
- `pipeline_cli.py` (~200 lines) - CLI wrapper
- `__init__.py` (94 lines) - Package exports

### Identified Problems

#### Problem 1: Multiple Entry Points

```
Current:
├── src/factory.py          → MLFactory.run()
├── src/ml_pipeline/unified.py → MLPipeline.run()
├── src/pipeline/runner.py  → PipelineRunner.run()
└── src/training/unified_orchestrator.py → UnifiedTrainingOrchestrator.run()
```

**Impact:** Confusion about which to use. `src/__init__.py` exports MLFactory, but other entry points remain.

#### Problem 2: Configuration Systems (4+)

| System | Location | Purpose |
|--------|----------|---------|
| `UnifiedConfig` | `src/config/unified.py` | 1,116 lines, comprehensive |
| `PipelineConfig` | `src/core/config.py` | Training-focused |
| `TrainerConfig` | `src/training/config.py` | Model training |
| `MLConfig` | `src/ml_pipeline/config.py` | ML pipeline specific |
| `SmartConfig` | `src/config/smart_config.py` | Auto-detection |

**Impact:** Users don't know which config to use. Internal code has adapters/converters.

#### Problem 3: Feature Logic Fragmentation

```
src/features/           # 7 files - computation, strategies, pruning
src/feature_selection/  # 10 files - selection algorithms
src/feature_store/      # 5 files - caching, versioning
src/pipeline/stages/features/  # Additional feature computation
```

**Impact:** Feature-related code in 4 places. No single "feature" module.

#### Problem 4: Naming Confusion

| Name | Actual Purpose |
|------|----------------|
| `pipeline/` | Data preparation pipeline (ingest→features→labels→splits) |
| `ml_pipeline/` | Training orchestration (calls pipeline/ + training/) |
| `core/` | Config + types (overlaps with config/) |

**Impact:** `pipeline/` sounds like the main pipeline, but it's just data prep.

#### Problem 5: Oversized Files

| File | Lines | Issue |
|------|-------|-------|
| `training/unified_orchestrator.py` | 1,599 | Too many responsibilities |
| `config/unified.py` | 1,116 | Monolithic config |
| `inference/preprocessing_graph.py` | 907 | Could split |

---

## Target Architecture

### Proposed Module Structure (15 modules)

```
src/
├── __init__.py              # MLFactory, PipelineConfig exports
├── factory.py               # THE entry point (MLFactory)
│
├── pipeline/                # Data Pipeline (Phases 1-5)
│   ├── stages/              # 17 stage implementations
│   ├── runner.py            # PipelineRunner
│   └── config.py            # Pipeline-specific config
│
├── training/                # Training (Phase 6)
│   ├── orchestrator.py      # UnifiedTrainingOrchestrator (refactored <800 lines)
│   ├── trainer.py           # Single model training
│   └── modes/               # Standard, regime-aware, meta-labeling
│
├── models/                  # Model Registry + Implementations
│   ├── registry.py
│   ├── base.py
│   ├── boosting/
│   ├── neural/
│   ├── classical/
│   └── ensemble/
│
├── features/                # CONSOLIDATED Feature Module
│   ├── compute/             # Feature computation
│   ├── selection/           # Feature selection (from feature_selection/)
│   ├── store/               # Feature caching (from feature_store/)
│   └── strategies.py        # Feature strategies
│
├── inference/               # Model Serving
│   ├── bundle.py
│   ├── orchestrator.py
│   └── preprocessing_graph.py
│
├── cross_validation/        # CV + OOF
│   ├── purged_kfold.py
│   └── oof_generator.py
│
├── config/                  # CONSOLIDATED Configuration
│   ├── unified.py           # Single UnifiedConfig
│   ├── defaults.py          # Default values
│   └── validators.py        # Schema validation
│
├── adapters/                # Data Format Adapters
├── labeling/                # Triple-Barrier Labeling
├── evaluation/              # Model Evaluation
├── optimization/            # Optuna Optimization
├── backtesting/             # Backtesting Engine
├── contracts/               # Futures Contracts
├── cli/                     # Command-Line Interface
└── common/                  # Shared Utilities
    ├── timeframes.py
    ├── manifest.py
    ├── monitoring.py        # (from monitoring/)
    ├── validation.py        # (from validation/)
    └── coordination.py      # (from coordination/)
```

### Modules to Remove/Merge

| Current | Action | Target |
|---------|--------|--------|
| `ml_pipeline/` | REMOVE | Functionality absorbed by `factory.py` |
| `feature_selection/` | MERGE | → `features/selection/` |
| `feature_store/` | MERGE | → `features/store/` |
| `core/` | SPLIT | Config → `config/`, Types → `common/` |
| `monitoring/` | MERGE | → `common/monitoring.py` |
| `validation/` | MERGE | → `common/validation.py` |
| `coordination/` | MERGE | → `common/coordination.py` |

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1)

**Goal:** Establish single entry point, deprecate alternatives.

| Task | File | Action |
|------|------|--------|
| 1.1 | `src/factory.py` | Ensure MLFactory is the canonical entry point |
| 1.2 | `src/ml_pipeline/unified.py` | Add deprecation warning, delegate to MLFactory |
| 1.3 | `src/__init__.py` | Update exports, add migration guide in docstring |
| 1.4 | Tests | Add integration tests for MLFactory |

**Deprecation Pattern:**
```python
# src/ml_pipeline/unified.py
import warnings
from src.factory import MLFactory

class MLPipeline:
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "MLPipeline is deprecated. Use MLFactory from src.factory instead. "
            "Will be removed in v3.0.",
            DeprecationWarning,
            stacklevel=2
        )
        self._factory = MLFactory(*args, **kwargs)
    
    def run(self, *args, **kwargs):
        return self._factory.run(*args, **kwargs)
```

### Phase 2: Config Consolidation (Week 2)

**Goal:** Single configuration system.

| Task | Action |
|------|--------|
| 2.1 | Audit all config classes, document field mappings |
| 2.2 | Extend `UnifiedConfig` to cover all use cases |
| 2.3 | Create config adapters for backward compatibility |
| 2.4 | Deprecate `PipelineConfig`, `TrainerConfig`, `MLConfig` |
| 2.5 | Update all internal code to use `UnifiedConfig` |

**Target API:**
```python
from src.config import UnifiedConfig  # THE config

config = UnifiedConfig(
    symbol="MES",
    models=["xgboost", "lstm"],
    # All settings in one place
)
```

### Phase 3: Feature Consolidation (Week 3)

**Goal:** Single `features/` module.

| Task | Action |
|------|--------|
| 3.1 | Create `features/selection/` subdirectory |
| 3.2 | Move `feature_selection/*.py` → `features/selection/` |
| 3.3 | Create `features/store/` subdirectory |
| 3.4 | Move `feature_store/*.py` → `features/store/` |
| 3.5 | Update all imports (use `ast_grep_replace`) |
| 3.6 | Delete empty `feature_selection/`, `feature_store/` |
| 3.7 | Add deprecation shims in old locations |

**Import Migration:**
```python
# Old
from src.feature_selection import PurgedSelector
from src.feature_store import FeatureCache

# New
from src.features.selection import PurgedSelector
from src.features.store import FeatureCache
```

### Phase 4: Training Refactor (Week 4)

**Goal:** Reduce `unified_orchestrator.py` from 1,599 → <800 lines.

| Task | Action |
|------|--------|
| 4.1 | Extract `training/cv_integration.py` (~300 lines) |
| 4.2 | Extract `training/artifact_manager.py` (~250 lines) |
| 4.3 | Extract `training/metrics_reporter.py` (~200 lines) |
| 4.4 | Slim `unified_orchestrator.py` to orchestration only |
| 4.5 | Ensure all tests pass |

### Phase 5: Cleanup & Merge Small Modules (Week 5)

**Goal:** Merge small modules into `common/`.

| Task | Action |
|------|--------|
| 5.1 | Move `monitoring/*.py` → `common/monitoring.py` |
| 5.2 | Move `validation/*.py` → `common/validation.py` |
| 5.3 | Move `coordination/*.py` → `common/coordination.py` |
| 5.4 | Split `core/`: config → `config/`, types → `common/` |
| 5.5 | Delete `ml_pipeline/` (after deprecation period) |
| 5.6 | Update all imports, run full test suite |

---

## Migration Guide

### For Users

**Before (v2.x):**
```python
from src.ml_pipeline import MLPipeline, MLConfig
from src.pipeline.runner import PipelineRunner

# Multiple ways to run
pipeline = MLPipeline(config)
# or
runner = PipelineRunner(config)
```

**After (v3.0):**
```python
from src import MLFactory, UnifiedConfig

config = UnifiedConfig(
    symbol="MES",
    models=["xgboost", "lstm"],
)
factory = MLFactory(config)
result = factory.run(df)
```

### For Developers

**Import Changes:**
```python
# Feature selection
- from src.feature_selection import PurgedSelector
+ from src.features.selection import PurgedSelector

# Feature store
- from src.feature_store import FeatureCache
+ from src.features.store import FeatureCache

# Config
- from src.core import PipelineConfig
+ from src.config import UnifiedConfig

# Validation
- from src.validation import DataValidator
+ from src.common.validation import DataValidator
```

---

## Success Criteria

### Quantitative

| Metric | Before | After |
|--------|--------|-------|
| Top-level modules | 24 | 15 |
| Entry points | 4 | 1 (MLFactory) |
| Config classes | 5+ | 1 (UnifiedConfig) |
| Feature modules | 3 | 1 (features/) |
| Max file size | 1,599 lines | <800 lines |

### Qualitative

- [ ] New developers understand structure in <30 min
- [ ] Single import path for each concept
- [ ] No circular dependencies
- [ ] All existing tests pass
- [ ] Deprecation warnings guide migration

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking external scripts | 2-version deprecation period with warnings |
| Circular imports after merge | Careful import ordering, lazy imports where needed |
| Test failures | Run full suite after each phase |
| Documentation drift | Update docs in same PR as code changes |

---

## Appendix: File Movement Manifest

### Phase 3: Feature Consolidation

```
src/feature_selection/walk_forward.py    → src/features/selection/walk_forward.py
src/feature_selection/config.py          → src/features/selection/config.py
src/feature_selection/filtering.py       → src/features/selection/filtering.py
src/feature_selection/purged_selector.py → src/features/selection/purged_selector.py
src/feature_selection/optimization.py    → src/features/selection/optimization.py
src/feature_selection/manager.py         → src/features/selection/manager.py
src/feature_selection/ohlcv_selector.py  → src/features/selection/ohlcv_selector.py
src/feature_selection/result.py          → src/features/selection/result.py
src/feature_selection/priority.py        → src/features/selection/priority.py
src/feature_selection/__init__.py        → src/features/selection/__init__.py

src/feature_store/lineage.py    → src/features/store/lineage.py
src/feature_store/store.py      → src/features/store/store.py
src/feature_store/cache.py      → src/features/store/cache.py
src/feature_store/versioning.py → src/features/store/versioning.py
src/feature_store/__init__.py   → src/features/store/__init__.py
```

### Phase 5: Small Module Merges

```
src/monitoring/*.py    → src/common/monitoring.py
src/validation/*.py    → src/common/validation.py
src/coordination/*.py  → src/common/coordination.py

src/core/config.py     → src/config/pipeline_config.py (then deprecate)
src/core/types.py      → src/common/types.py
src/core/interfaces.py → src/common/interfaces.py
src/core/constants.py  → src/common/constants.py
```

---

## Timeline Summary

| Week | Phase | Outcome |
|------|-------|---------|
| 1 | Foundation | Single entry point (MLFactory), deprecation warnings |
| 2 | Config | Single config system (UnifiedConfig) |
| 3 | Features | Consolidated features/ module |
| 4 | Training | Refactored orchestrator (<800 lines) |
| 5 | Cleanup | 15 modules, all imports updated |

**Total Effort:** ~5 weeks for 1 engineer

---

*Last Updated: 2026-01-21*
*Status: Planning Complete - Ready for Implementation*
