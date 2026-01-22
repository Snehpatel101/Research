# Module Consolidation Plan

**Status**: PLANNING (Do Not Execute)
**Author**: Architecture Team
**Created**: 2026-01-22
**Target Version**: 2.0.0

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Analysis](#2-current-state-analysis)
3. [Proposed Structure](#3-proposed-structure)
4. [Migration Steps](#4-migration-steps)
5. [Risk Assessment](#5-risk-assessment)
6. [What We Are NOT Doing](#6-what-we-are-not-doing)
7. [Success Criteria](#7-success-criteria)
8. [Appendix: Import Migration Guide](#appendix-import-migration-guide)

---

## 1. Executive Summary

### The Problem

The current codebase has **22 top-level modules** under `src/`, creating:

- **Unclear boundaries**: Where does `pipeline/` end and `features/` begin?
- **Circular dependency risks**: `cross_validation/` imports from `models/` which imports from `evaluation/`
- **Cognitive overload**: New developers must understand 22 different namespaces
- **Scattered functionality**: Feature-related code spans `features/`, `feature_store/`, `feature_selection/`, and `adapters/`
- **Inconsistent naming**: `common/` vs `utils/` vs `core/` for shared utilities

### The Solution

Consolidate into **7 cohesive domains** that align with the ML pipeline lifecycle:

```
src/
├── core/         # Foundation: types, contracts, configuration
├── data/         # Data lifecycle: ingestion, features, labeling
├── models/       # Model lifecycle: training, registry, ensemble
├── validation/   # Quality: cross-validation, evaluation, monitoring
├── optimization/ # Tuning: hyperparameters, feature selection
├── inference/    # Production: prediction, backtesting
└── cli/          # Interface: command-line tools
```

### Expected Benefits

| Metric | Before | After |
|--------|--------|-------|
| Top-level modules | 22 | 7 |
| Average import depth | 4+ levels | 2-3 levels |
| Cross-module dependencies | Complex web | Clear DAG |
| Onboarding time | ~2 weeks | ~1 week |

---

## 2. Current State Analysis

### Complete Module Inventory (22 Modules)

| # | Module | Purpose | Files | Dependencies |
|---|--------|---------|-------|--------------|
| 1 | `adapters/` | Data format adapters (tabular, sequence, multi-resolution) | 12 | core, features |
| 2 | `backtesting/` | Historical simulation and walk-forward testing | 7 | models, inference |
| 3 | `cli/` | Command-line interface and entry points | 10 | all modules |
| 4 | `common/` | Shared utilities (timeframes, split ratios) | 6 | none |
| 5 | `config/` | Configuration management and schemas | 10 | core |
| 6 | `contracts/` | Interface definitions and type contracts | 5 | none |
| 7 | `coordination/` | Multi-process coordination and locking | 4 | utils |
| 8 | `core/` | Core types, constants, paths, reproducibility | 14 | contracts |
| 9 | `cross_validation/` | CV strategies (purged, combinatorial, expanding) | 24 | core, models |
| 10 | `evaluation/` | Model evaluation metrics and reporting | 4 | models |
| 11 | `feature_selection/` | Feature importance and selection algorithms | 11 | features, models |
| 12 | `feature_store/` | Feature persistence and versioning | 6 | features, adapters |
| 13 | `features/` | Feature engineering and transformations | 9 | core, adapters |
| 14 | `inference/` | Prediction serving and batch inference | 10 | models |
| 15 | `labeling/` | Label generation and target engineering | 5 | core, features |
| 16 | `ml_pipeline/` | End-to-end pipeline orchestration | - | all modules |
| 17 | `models/` | Model definitions, training, registry | 22+ | core, features |
| 18 | `monitoring/` | Model drift and performance monitoring | 7 | models, inference |
| 19 | `optimization/` | Hyperparameter tuning (Optuna integration) | 5 | models, cross_validation |
| 20 | `pipeline/` | Data pipeline stages and transformations | 12 | adapters, features |
| 21 | `training/` | Training loops and utilities | 13 | models, core |
| 22 | `utils/` | General utilities and helpers | 8 | none |
| 23 | `validation/` | Data validation and schema checks | 7 | core |

### Dependency Graph (Simplified)

```
                    ┌─────────────────────────────────────┐
                    │              cli/                    │
                    └─────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
            ┌──────────────┐                   ┌──────────────┐
            │  inference/  │◄──────────────────│ backtesting/ │
            └──────────────┘                   └──────────────┘
                    │                                   │
                    ▼                                   │
            ┌──────────────┐                           │
            │   models/    │◄──────────────────────────┘
            └──────────────┘
                    │
        ┌───────────┼───────────┬───────────────┐
        ▼           ▼           ▼               ▼
┌──────────────┐ ┌────────┐ ┌───────────┐ ┌─────────────┐
│cross_valid/  │ │training│ │evaluation/│ │optimization/│
└──────────────┘ └────────┘ └───────────┘ └─────────────┘
        │           │           │               │
        └───────────┴───────────┴───────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  features/   │    │  labeling/   │    │feature_store/│
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
        ┌───────────────────┬───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  adapters/   │    │  pipeline/   │    │  validation/ │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                            ▼
            ┌───────────────┬───────────────┐
            ▼               ▼               ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │    core/     │ │   config/    │ │  contracts/  │
    └──────────────┘ └──────────────┘ └──────────────┘
            │               │               │
            └───────────────┴───────────────┘
                            │
                            ▼
            ┌───────────────┬───────────────┐
            ▼               ▼               ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │   common/    │ │    utils/    │ │coordination/ │
    └──────────────┘ └──────────────┘ └──────────────┘
```

### Pain Points by Category

**1. Scattered Feature Logic**
```
features/         # Feature engineering
feature_store/    # Feature persistence
feature_selection/# Feature importance
adapters/         # Data format conversion
```
These should be ONE cohesive `data/` domain.

**2. Fragmented Core Types**
```
core/       # Types, constants, paths
contracts/  # Interface definitions
common/     # Timeframes, split ratios
config/     # Configuration schemas
```
These should be ONE `core/` foundation.

**3. Validation Sprawl**
```
validation/       # Data validation
cross_validation/ # CV strategies
evaluation/       # Model evaluation
monitoring/       # Production monitoring
```
All quality-related - should be ONE `validation/` domain.

**4. Inference Duplication**
```
inference/    # Prediction serving
backtesting/  # Historical simulation
```
Both are about "using trained models" - consolidate to `inference/`.

---

## 3. Proposed Structure

### Target Architecture (7 Domains)

```
src/
├── core/                    # Foundation Layer
│   ├── __init__.py
│   ├── types/               # ← from core/types.py, contracts/
│   │   ├── base.py
│   │   ├── datasets.py
│   │   └── protocols.py
│   ├── config/              # ← from config/
│   │   ├── schema.py
│   │   └── loader.py
│   ├── constants.py         # ← from core/constants.py, common/
│   ├── paths.py             # ← from core/paths.py
│   ├── reproducibility.py   # ← from core/reproducibility.py
│   └── utils/               # ← from utils/, common/
│       ├── timing.py
│       ├── logging.py
│       └── coordination.py  # ← from coordination/
│
├── data/                    # Data Layer
│   ├── __init__.py
│   ├── pipeline/            # ← from pipeline/
│   │   ├── stages.py
│   │   └── orchestrator.py
│   ├── adapters/            # ← from adapters/
│   │   ├── base.py
│   │   ├── tabular.py
│   │   ├── sequence.py
│   │   └── multi_resolution.py
│   ├── features/            # ← from features/
│   │   ├── engineering.py
│   │   └── transformers.py
│   ├── labeling/            # ← from labeling/
│   │   └── generators.py
│   └── store/               # ← from feature_store/
│       ├── registry.py
│       └── versioning.py
│
├── models/                  # Model Layer (minimal changes)
│   ├── __init__.py
│   ├── base.py
│   ├── registry.py
│   ├── trainer.py
│   ├── classical/
│   ├── boosting/
│   ├── neural/
│   ├── ensemble/
│   ├── calibration/
│   └── tracking/
│
├── validation/              # Quality Layer
│   ├── __init__.py
│   ├── data/                # ← from validation/
│   │   └── schema.py
│   ├── cross_validation/    # ← from cross_validation/
│   │   ├── strategies.py
│   │   ├── purged.py
│   │   └── combinatorial.py
│   ├── evaluation/          # ← from evaluation/
│   │   ├── metrics.py
│   │   └── reports.py
│   └── monitoring/          # ← from monitoring/
│       ├── drift.py
│       └── performance.py
│
├── optimization/            # Optimization Layer
│   ├── __init__.py
│   ├── hyperparameter/      # ← from optimization/
│   │   ├── optuna_tuner.py
│   │   └── search_spaces.py
│   └── feature_selection/   # ← from feature_selection/
│       ├── importance.py
│       └── selectors.py
│
├── inference/               # Inference Layer
│   ├── __init__.py
│   ├── serving/             # ← from inference/
│   │   ├── predictor.py
│   │   └── batch.py
│   └── backtesting/         # ← from backtesting/
│       ├── simulator.py
│       └── walk_forward.py
│
└── cli/                     # Interface Layer (no changes)
    ├── __init__.py
    ├── commands/
    └── main.py
```

### Detailed Migration Mapping

#### 3.1 `core/` Domain

**Absorbs**: `core/`, `contracts/`, `config/`, `common/`, `utils/`, `coordination/`

| Source | Destination | Notes |
|--------|-------------|-------|
| `core/types.py` | `core/types/base.py` | Core type definitions |
| `core/constants.py` | `core/constants.py` | Keep as-is |
| `core/paths.py` | `core/paths.py` | Keep as-is |
| `core/reproducibility.py` | `core/reproducibility.py` | Keep as-is |
| `core/validation.py` | `core/types/validation.py` | Type validation |
| `core/interfaces.py` | `core/types/protocols.py` | Protocol definitions |
| `contracts/*` | `core/types/protocols.py` | Merge with interfaces |
| `config/*` | `core/config/` | Configuration submodule |
| `common/timeframes.py` | `core/constants.py` | Merge constants |
| `common/split_ratios.py` | `core/constants.py` | Merge constants |
| `utils/*` | `core/utils/` | General utilities |
| `coordination/*` | `core/utils/coordination.py` | Process coordination |

**Public API** (`core/__init__.py`):
```python
from core.types import DatasetConfig, ModelConfig, TrainingConfig
from core.config import load_config, ConfigSchema
from core.constants import TIMEFRAMES, SPLIT_RATIOS
from core.paths import ProjectPaths
from core.reproducibility import set_seed, get_reproducibility_info
```

#### 3.2 `data/` Domain

**Absorbs**: `pipeline/`, `adapters/`, `features/`, `labeling/`, `feature_store/`

| Source | Destination | Notes |
|--------|-------------|-------|
| `pipeline/*` | `data/pipeline/` | Pipeline orchestration |
| `adapters/base.py` | `data/adapters/base.py` | Base adapter class |
| `adapters/tabular.py` | `data/adapters/tabular.py` | Tabular adapter |
| `adapters/sequence.py` | `data/adapters/sequence.py` | Sequence adapter |
| `adapters/multi_resolution.py` | `data/adapters/multi_resolution.py` | Multi-res adapter |
| `features/*` | `data/features/` | Feature engineering |
| `labeling/*` | `data/labeling/` | Label generation |
| `feature_store/*` | `data/store/` | Feature persistence |

**Public API** (`data/__init__.py`):
```python
from data.pipeline import DataPipeline, PipelineStage
from data.adapters import TabularAdapter, SequenceAdapter, MultiResolutionAdapter
from data.features import FeatureEngineer, Transformer
from data.labeling import LabelGenerator
from data.store import FeatureStore, FeatureRegistry
```

#### 3.3 `models/` Domain

**Status**: Keep as-is (already well-organized)

The `models/` module is already well-structured with clear submodules:
- `classical/` - Traditional ML models
- `boosting/` - XGBoost, LightGBM, CatBoost
- `neural/` - PyTorch models
- `ensemble/` - Ensemble methods
- `calibration/` - Probability calibration
- `tracking/` - Experiment tracking

**Only change**: Absorb `training/` module.

| Source | Destination | Notes |
|--------|-------------|-------|
| `training/*` | `models/training/` | Already has this subdir |

#### 3.4 `validation/` Domain

**Absorbs**: `validation/`, `cross_validation/`, `evaluation/`, `monitoring/`

| Source | Destination | Notes |
|--------|-------------|-------|
| `validation/*` | `validation/data/` | Data validation |
| `cross_validation/strategies.py` | `validation/cv/strategies.py` | CV base |
| `cross_validation/purged.py` | `validation/cv/purged.py` | Purged CV |
| `cross_validation/combinatorial.py` | `validation/cv/combinatorial.py` | CPCV |
| `evaluation/metrics.py` | `validation/evaluation/metrics.py` | Evaluation metrics |
| `evaluation/reports.py` | `validation/evaluation/reports.py` | Report generation |
| `monitoring/drift.py` | `validation/monitoring/drift.py` | Drift detection |
| `monitoring/performance.py` | `validation/monitoring/performance.py` | Perf monitoring |

**Public API** (`validation/__init__.py`):
```python
from validation.cv import PurgedKFold, CombinatorialPurgedCV, ExpandingWindow
from validation.evaluation import evaluate_model, MetricsReport
from validation.monitoring import DriftDetector, PerformanceMonitor
from validation.data import validate_schema, DataValidator
```

#### 3.5 `optimization/` Domain

**Absorbs**: `optimization/`, `feature_selection/`

| Source | Destination | Notes |
|--------|-------------|-------|
| `optimization/*` | `optimization/hyperparameter/` | Optuna integration |
| `feature_selection/importance.py` | `optimization/feature_selection/importance.py` | Feature importance |
| `feature_selection/selectors.py` | `optimization/feature_selection/selectors.py` | Selection algorithms |

**Public API** (`optimization/__init__.py`):
```python
from optimization.hyperparameter import OptunaTuner, SearchSpace
from optimization.feature_selection import FeatureImportance, FeatureSelector
```

#### 3.6 `inference/` Domain

**Absorbs**: `inference/`, `backtesting/`

| Source | Destination | Notes |
|--------|-------------|-------|
| `inference/predictor.py` | `inference/serving/predictor.py` | Prediction serving |
| `inference/batch.py` | `inference/serving/batch.py` | Batch inference |
| `backtesting/simulator.py` | `inference/backtesting/simulator.py` | Backtest engine |
| `backtesting/walk_forward.py` | `inference/backtesting/walk_forward.py` | Walk-forward |

**Public API** (`inference/__init__.py`):
```python
from inference.serving import Predictor, BatchPredictor
from inference.backtesting import BacktestSimulator, WalkForwardTest
```

#### 3.7 `cli/` Domain

**Status**: Keep as-is (already isolated)

No changes needed. The CLI module is already well-isolated and serves as the user interface layer.

---

## 4. Migration Steps

### Overview

```
Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5
 (Shims)    (Move)     (Update)   (Deprecate)  (Remove)

 1 week     2 weeks    2 weeks     4 weeks     Release
```

### Phase 1: Create Re-export Shims (Week 1)

**Goal**: Enable new import paths without breaking existing code.

**Steps**:

1. Create new directory structure (empty `__init__.py` files)
2. Add re-exports from new locations to old locations
3. Add re-exports from old locations to new locations

**Example** - `data/__init__.py`:
```python
"""
Data domain - consolidates pipeline, adapters, features, labeling, feature_store.

New import paths (preferred):
    from data.pipeline import DataPipeline
    from data.adapters import TabularAdapter

Legacy import paths (deprecated, will be removed in v2.0):
    from pipeline import DataPipeline  # Still works
    from adapters import TabularAdapter  # Still works
"""

# Re-export from original locations (temporary)
from pipeline import DataPipeline, PipelineStage
from adapters import TabularAdapter, SequenceAdapter, MultiResolutionAdapter
from features import FeatureEngineer
from labeling import LabelGenerator
from feature_store import FeatureStore

__all__ = [
    "DataPipeline",
    "PipelineStage",
    "TabularAdapter",
    "SequenceAdapter",
    "MultiResolutionAdapter",
    "FeatureEngineer",
    "LabelGenerator",
    "FeatureStore",
]
```

**Verification**:
```bash
# Both should work
python -c "from data.pipeline import DataPipeline"
python -c "from pipeline import DataPipeline"
```

### Phase 2: Move Modules (Weeks 2-3)

**Goal**: Physically relocate files while maintaining backwards compatibility.

**For each module being moved**:

1. Copy files to new location
2. Update internal imports in copied files
3. Convert old module to re-export shim
4. Run full test suite
5. Commit with clear message

**Example** - Moving `adapters/` to `data/adapters/`:

```bash
# Step 1: Copy
cp -r src/adapters/* src/data/adapters/

# Step 2: Update imports in new files
# (Manual: change "from adapters.base" to "from data.adapters.base")

# Step 3: Convert old to shim
```

**Old `adapters/__init__.py`** (becomes shim):
```python
"""
DEPRECATED: Import from 'data.adapters' instead.

This module will be removed in v2.0.0.

Migration:
    # Old (deprecated)
    from adapters import TabularAdapter

    # New (preferred)
    from data.adapters import TabularAdapter
"""
import warnings

def __getattr__(name):
    warnings.warn(
        f"Importing '{name}' from 'adapters' is deprecated. "
        f"Use 'from data.adapters import {name}' instead. "
        "This will be removed in v2.0.0.",
        DeprecationWarning,
        stacklevel=2
    )
    from data import adapters
    return getattr(adapters, name)

# Keep direct imports working (without warning for now)
from data.adapters import (
    TabularAdapter,
    SequenceAdapter,
    MultiResolutionAdapter,
    # ... etc
)
```

### Phase 3: Update Internal Imports (Weeks 4-5)

**Goal**: All internal code uses new import paths.

**Steps**:

1. Run import analysis script to find all old imports
2. Batch update using automated tooling
3. Manual review of complex cases
4. Run test suite after each batch

**Automation Script**:
```python
#!/usr/bin/env python3
"""Update imports from old paths to new paths."""

import re
from pathlib import Path

IMPORT_MAPPINGS = {
    r"from adapters": "from data.adapters",
    r"from pipeline": "from data.pipeline",
    r"from features": "from data.features",
    r"from labeling": "from data.labeling",
    r"from feature_store": "from data.store",
    r"from cross_validation": "from validation.cv",
    r"from evaluation": "from validation.evaluation",
    r"from monitoring": "from validation.monitoring",
    r"from feature_selection": "from optimization.feature_selection",
    r"from backtesting": "from inference.backtesting",
    r"from contracts": "from core.types",
    r"from common": "from core",
    r"from utils": "from core.utils",
    r"from coordination": "from core.utils.coordination",
    r"from config": "from core.config",
}

def update_imports(file_path: Path) -> bool:
    """Update imports in a single file. Returns True if changes made."""
    content = file_path.read_text()
    original = content

    for old, new in IMPORT_MAPPINGS.items():
        content = re.sub(old, new, content)

    if content != original:
        file_path.write_text(content)
        return True
    return False
```

### Phase 4: Deprecation Warnings (Weeks 6-9)

**Goal**: Warn users about upcoming breaking changes.

**Steps**:

1. Enable deprecation warnings in all shim modules
2. Update documentation with migration guide
3. Announce deprecation timeline in release notes
4. Monitor for user feedback and issues

**Warning Implementation**:
```python
# In each deprecated module's __init__.py
import warnings

warnings.warn(
    "The 'adapters' module is deprecated and will be removed in v2.0.0. "
    "Please update your imports to use 'data.adapters' instead. "
    "See https://docs.example.com/migration for details.",
    DeprecationWarning,
    stacklevel=2
)
```

**Documentation Update**:
- Add migration guide to docs
- Update all code examples
- Add deprecation notices to API docs

### Phase 5: Remove Old Paths (Version 2.0.0)

**Goal**: Clean removal of deprecated modules.

**Steps**:

1. Delete shim modules
2. Update `__init__.py` files to remove backwards compat code
3. Run full test suite
4. Update version to 2.0.0
5. Release with clear changelog

**Checklist**:
- [ ] All tests pass with new imports only
- [ ] No internal code uses old imports
- [ ] Documentation fully updated
- [ ] Changelog documents breaking changes
- [ ] Migration guide is comprehensive

---

## 5. Risk Assessment

### High Risk Items

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Circular imports after consolidation | Build breaks | Medium | Map dependencies before moving; test incrementally |
| External users' code breaks | User frustration | High | Long deprecation period; clear migration guide |
| Test coverage gaps during migration | Regressions | Medium | Add integration tests before migration |
| Import path conflicts | Runtime errors | Low | Automated verification script |

### Medium Risk Items

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| IDE autocomplete breaks | Developer friction | Medium | Update `py.typed` markers; test with common IDEs |
| Documentation becomes stale | User confusion | Medium | Update docs in same PR as code changes |
| CI/CD pipelines break | Deploy failures | Low | Test pipeline changes in staging first |

### Low Risk Items

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Performance regression from re-exports | Slower imports | Low | Benchmark import times; use lazy imports |
| Git history becomes hard to follow | Maintenance burden | Low | Use `git mv` for moves; clear commit messages |

### Rollback Plan

If critical issues are discovered:

1. **Phase 1-2**: Simply delete new directories; shims are additive
2. **Phase 3**: Revert import update commits
3. **Phase 4-5**: Cannot easily rollback; ensure thorough testing

---

## 6. What We Are NOT Doing

### Why This Is a Plan, Not an Implementation

This document intentionally does **NOT** execute the migration because:

#### 1. High Risk of Breaking Changes

```python
# This import exists in potentially hundreds of files:
from adapters import TabularAdapter

# Changing it requires:
# - Finding all occurrences (internal + external)
# - Updating all occurrences atomically
# - Testing all affected code paths
```

A single missed import breaks the build.

#### 2. Requires Extensive Testing

Before executing this plan, we need:

- [ ] Full test coverage report (current: unknown)
- [ ] Integration tests for cross-module boundaries
- [ ] Import dependency graph verification
- [ ] Performance benchmarks for import times
- [ ] External user impact assessment

#### 3. Should Be Done Incrementally

The safest approach is:

1. **One domain at a time**: Start with lowest-risk (`cli/` - no changes needed)
2. **Feature-flagged**: Allow rollback via environment variable
3. **User-tested**: Beta period with early adopters
4. **Version-gated**: Clear major version boundary

#### 4. Requires Team Alignment

This migration affects:

- All developers (new import paths)
- All CI/CD pipelines (test configurations)
- All documentation (code examples)
- All external users (breaking change)

Requires sign-off from:
- [ ] Tech Lead
- [ ] DevOps
- [ ] Documentation owner
- [ ] Product (for version bump timing)

### What We ARE Doing

1. **Documenting** the target architecture
2. **Analyzing** the current state
3. **Planning** the migration phases
4. **Identifying** risks and mitigations
5. **Creating** a reference for future execution

---

## 7. Success Criteria

### Quantitative Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Top-level modules | 22 | 7 | `ls src/ \| wc -l` |
| Import depth (avg) | 4.2 | 2.5 | Static analysis |
| Circular dependencies | 3 | 0 | `pydeps --no-show` |
| Test coverage | TBD | >80% | `pytest --cov` |
| Import time | TBD | <2s | `python -X importtime` |

### Qualitative Criteria

- [ ] New developers can understand module structure in <30 minutes
- [ ] Each domain has clear, single responsibility
- [ ] No "where does this go?" questions for new code
- [ ] Documentation matches code structure
- [ ] IDE navigation works correctly

### Verification Commands

```bash
# Check module count
ls -d src/*/ | wc -l  # Should be 7

# Check for circular imports
python -c "import src" 2>&1 | grep -i circular  # Should be empty

# Check import time
python -X importtime -c "import src" 2>&1 | tail -20

# Run full test suite
pytest tests/ -v --tb=short

# Check deprecation warnings
python -W error::DeprecationWarning -c "import src"
```

---

## Appendix: Import Migration Guide

### Quick Reference

| Old Import | New Import |
|------------|------------|
| `from adapters import *` | `from data.adapters import *` |
| `from backtesting import *` | `from inference.backtesting import *` |
| `from common import *` | `from core import *` |
| `from config import *` | `from core.config import *` |
| `from contracts import *` | `from core.types import *` |
| `from coordination import *` | `from core.utils.coordination import *` |
| `from cross_validation import *` | `from validation.cv import *` |
| `from evaluation import *` | `from validation.evaluation import *` |
| `from feature_selection import *` | `from optimization.feature_selection import *` |
| `from feature_store import *` | `from data.store import *` |
| `from features import *` | `from data.features import *` |
| `from labeling import *` | `from data.labeling import *` |
| `from monitoring import *` | `from validation.monitoring import *` |
| `from pipeline import *` | `from data.pipeline import *` |
| `from training import *` | `from models.training import *` |
| `from utils import *` | `from core.utils import *` |
| `from validation import *` | `from validation.data import *` |

### Automated Migration Script

Save as `scripts/migrate_imports.py`:

```python
#!/usr/bin/env python3
"""
Migrate imports from old module paths to new consolidated paths.

Usage:
    python scripts/migrate_imports.py src/  # Dry run
    python scripts/migrate_imports.py src/ --apply  # Apply changes
"""

import argparse
import re
from pathlib import Path

MIGRATIONS = [
    (r"from adapters(\s+import|\.|$)", r"from data.adapters\1"),
    (r"from backtesting(\s+import|\.|$)", r"from inference.backtesting\1"),
    (r"from common(\s+import|\.|$)", r"from core\1"),
    (r"from config(\s+import|\.|$)", r"from core.config\1"),
    (r"from contracts(\s+import|\.|$)", r"from core.types\1"),
    (r"from coordination(\s+import|\.|$)", r"from core.utils.coordination\1"),
    (r"from cross_validation(\s+import|\.|$)", r"from validation.cv\1"),
    (r"from evaluation(\s+import|\.|$)", r"from validation.evaluation\1"),
    (r"from feature_selection(\s+import|\.|$)", r"from optimization.feature_selection\1"),
    (r"from feature_store(\s+import|\.|$)", r"from data.store\1"),
    (r"from features(\s+import|\.|$)", r"from data.features\1"),
    (r"from labeling(\s+import|\.|$)", r"from data.labeling\1"),
    (r"from monitoring(\s+import|\.|$)", r"from validation.monitoring\1"),
    (r"from pipeline(\s+import|\.|$)", r"from data.pipeline\1"),
    (r"from training(\s+import|\.|$)", r"from models.training\1"),
    (r"from utils(\s+import|\.|$)", r"from core.utils\1"),
    (r"from validation(\s+import|\.|$)", r"from validation.data\1"),
    # Also handle "import X" style
    (r"^import adapters$", r"import data.adapters as adapters"),
    (r"^import backtesting$", r"import inference.backtesting as backtesting"),
    # ... etc
]

def migrate_file(path: Path, apply: bool = False) -> list[tuple[int, str, str]]:
    """Migrate imports in a single file. Returns list of (line_num, old, new)."""
    changes = []
    lines = path.read_text().splitlines()
    new_lines = []

    for i, line in enumerate(lines, 1):
        new_line = line
        for pattern, replacement in MIGRATIONS:
            new_line = re.sub(pattern, replacement, new_line)

        if new_line != line:
            changes.append((i, line.strip(), new_line.strip()))
        new_lines.append(new_line)

    if apply and changes:
        path.write_text("\n".join(new_lines) + "\n")

    return changes

def main():
    parser = argparse.ArgumentParser(description="Migrate imports to new paths")
    parser.add_argument("directory", type=Path, help="Directory to process")
    parser.add_argument("--apply", action="store_true", help="Apply changes")
    args = parser.parse_args()

    total_changes = 0
    for path in args.directory.rglob("*.py"):
        changes = migrate_file(path, args.apply)
        if changes:
            print(f"\n{path}:")
            for line_num, old, new in changes:
                print(f"  L{line_num}: {old}")
                print(f"      -> {new}")
            total_changes += len(changes)

    action = "Applied" if args.apply else "Would apply"
    print(f"\n{action} {total_changes} changes")

if __name__ == "__main__":
    main()
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-22 | Architecture Team | Initial plan |

---

**Next Steps**:

1. Review this plan with the team
2. Create GitHub issues for each phase
3. Establish test coverage baseline
4. Begin Phase 1 (shims) in feature branch
5. Set timeline for v2.0.0 release
