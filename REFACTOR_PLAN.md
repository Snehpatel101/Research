# Organizational Refactor Plan

**Generated:** 2026-01-14
**Purpose:** Long-term clarity, extensibility, and maintainability
**Method:** 4 specialized parallel agents analyzing structure, config, files, and dead code

---

## Executive Summary

| Category | Items | Impact |
|----------|-------|--------|
| **Files to Split** | 4 large files | ~4,400 lines reorganized |
| **Modules to Merge** | 4 scattered locations | Single ownership |
| **Modules to Move** | 8 misplaced configs | Correct package ownership |
| **Dead Code to Delete** | 1-2 modules | ~50-1,300 lines removed |
| **Config Consolidation** | 23+ files | Single entry point |

---

## Quick Reference: What Changes

### DELETE (Safe)
- `src/utils/explainability.py` (53 lines) - orphaned, zero imports

### SPLIT (High Priority)
- `src/models/trainer.py` (1268 lines) → 4 files
- `src/cross_validation/cv_runner.py` (1107 lines) → 5 files
- `src/phase1/config/feature_sets.py` (1031 lines) → 3 files
- `src/models/neural/cnn.py` (1049 lines) → 3 files

### MERGE (Feature Selection)
- `src/cross_validation/feature_selector.py` +
- `src/feature_selection/` +
- `src/models/feature_selection/` +
- `src/phase1/utils/feature_selection.py`
- → Single `src/feature_selection/` package

### MOVE (Config Ownership)
- `MODEL_DATA_REQUIREMENTS` from phase1 → models
- Feature sets from phase1 → config/models
- All constants → `src/config/constants/`

---

## Phase 1: Immediate Cleanup (Zero Risk)

**Checkpoint: Git tag `refactor-phase1-start`**

### Step 1.1: Delete Confirmed Dead Code

```bash
# Verify no imports
grep -r "explainability" src/ scripts/ --include="*.py"
# Should return only the file itself

# Delete
rm src/utils/explainability.py
git add -A && git commit -m "chore: delete orphaned explainability.py"
```

**Verification:**
- [ ] All tests pass
- [ ] No import errors

**Rollback:** `git revert HEAD`

---

## Phase 2: Split Large Files (Medium Risk)

**Checkpoint: Git tag `refactor-phase2-start`**

### Step 2.1: Split `trainer.py` (1268 lines → 4 files)

**Current:** Single 1268-line file with mixed responsibilities

**Target Structure:**
```
src/models/training/
    __init__.py              # Re-exports for backward compatibility
    trainer.py               # Core Trainer class (~450 lines)
    evaluation.py            # _evaluate_test_set(), metrics (~250 lines)
    artifacts.py             # _save_*() methods (~300 lines)
    features.py              # Feature set resolution (~250 lines)
```

**Implementation:**
```python
# src/models/training/trainer.py
from .evaluation import TrainerEvaluationMixin
from .artifacts import TrainerArtifactsMixin
from .features import TrainerFeaturesMixin

class Trainer(TrainerFeaturesMixin, TrainerEvaluationMixin, TrainerArtifactsMixin):
    """Main trainer orchestrating model training."""
    ...
```

**Backward Compatibility:**
```python
# src/models/trainer.py (keep for imports)
from src.models.training.trainer import Trainer
from src.models.training.evaluation import *
from src.models.training.artifacts import *
__all__ = ["Trainer"]
```

**Verification:**
- [ ] `python -c "from src.models.trainer import Trainer"`
- [ ] `pytest tests/models/test_trainer.py -v`
- [ ] All 730 model tests pass

**Rollback:** `git checkout HEAD~1 -- src/models/`

---

### Step 2.2: Split `cv_runner.py` (1107 lines → 5 files)

**Target Structure:**
```
src/cross_validation/
    cv_runner.py             # Core CrossValidationRunner (~350 lines)
    cv_dataclasses.py        # FoldMetrics, CVResult (~120 lines)
    cv_tuner.py              # TimeSeriesOptunaTuner (~200 lines)
    cv_feature_selection.py  # Per-fold selection (~250 lines)
    cv_stacking.py           # Stacking dataset building (~200 lines)
```

**Verification:**
- [ ] `pytest tests/cross_validation/ -v`
- [ ] CV runner imports work

**Rollback:** `git checkout HEAD~1 -- src/cross_validation/`

---

### Step 2.3: Split `feature_sets.py` (1031 lines → 3 files)

**Target Structure:**
```
src/phase1/config/feature_sets/
    __init__.py              # Re-exports all
    definitions.py           # FEATURE_SET_DEFINITIONS (~700 lines)
    core.py                  # FeatureSetDefinition, aliases (~150 lines)
    validation.py            # Validation functions (~200 lines)
```

**Verification:**
- [ ] `from src.phase1.config.feature_sets import FEATURE_SET_DEFINITIONS`
- [ ] `pytest tests/phase_1_tests/config/ -v`

**Rollback:** `git checkout HEAD~1 -- src/phase1/config/`

---

### Step 2.4: Split `cnn.py` (1049 lines → 3 files)

**Target Structure:**
```
src/models/neural/
    cnn_base.py              # Shared CNN utilities (~200 lines)
    inceptiontime_model.py   # InceptionTime (~400 lines)
    resnet1d_model.py        # ResNet1D (~400 lines)
```

**Update Registry:** Ensure both models still register correctly.

**Verification:**
- [ ] `python -c "from src.models import ModelRegistry; print(ModelRegistry.list_all())"`
- [ ] `pytest tests/models/test_neural_models.py -v`

**Rollback:** `git checkout HEAD~1 -- src/models/neural/`

---

## Phase 3: Merge Feature Selection (High Complexity)

**Checkpoint: Git tag `refactor-phase3-start`**

### Step 3.1: Consolidate Feature Selection

**Current State (4 locations):**
```
src/cross_validation/feature_selector.py    # FeatureSelectionResult, WalkForward
src/feature_selection/                       # OHLCV, Purged selectors
src/models/feature_selection/                # Manager, PersistedSelection
src/phase1/utils/feature_selection.py        # Filtering functions
```

**Target State (1 location):**
```
src/feature_selection/
    __init__.py              # Unified exports
    result.py                # FeatureSelectionResult, PersistedFeatureSelection
    config.py                # FeatureSelectionConfig, ModelFamilyDefaults
    walk_forward.py          # WalkForwardFeatureSelector
    ohlcv_selector.py        # OHLCVFeatureSelector (existing)
    purged_selector.py       # PurgedFeatureSelector (existing)
    manager.py               # FeatureSelectionManager
    filtering.py             # filter_low_variance, filter_correlated
    priority.py              # FEATURE_PRIORITY dict
```

**Migration Steps:**
1. Create new files in `src/feature_selection/`
2. Move code with updated imports
3. Update `__init__.py` to re-export everything
4. Add deprecation warnings to old locations
5. Update all imports across codebase
6. Delete old files after verification

**Verification:**
- [ ] `from src.feature_selection import FeatureSelectionResult`
- [ ] `pytest tests/cross_validation/test_feature_selector.py -v`
- [ ] `pytest tests/feature_selection/ -v`
- [ ] All imports resolve correctly

**Rollback:** `git checkout HEAD~1 -- src/`

---

## Phase 4: Config Consolidation (Medium Complexity)

**Checkpoint: Git tag `refactor-phase4-start`**

### Step 4.1: Create Central Config Package

**Target Structure:**
```
src/config/
    __init__.py              # Central exports
    base.py                  # BaseConfig dataclass
    constants/
        __init__.py
        timeframes.py        # From src/common/timeframes.py
        horizons.py          # From src/common/horizon_config.py
        splits.py            # From src/common/split_ratios.py
    models/
        __init__.py
        trainer_config.py    # From src/models/config/
        data_requirements.py # From src/phase1/config/model_config.py
        feature_sets.py      # From src/phase1/config/feature_sets/
        loaders.py           # From src/models/config/
        merging.py           # From src/models/config/
    pipeline/
        __init__.py
        pipeline_config.py   # From src/phase1/pipeline_config.py
        barriers.py          # From src/phase1/config/
        labeling.py          # From src/phase1/config/
```

**Migration Steps:**
1. Create `src/config/` structure
2. Copy files to new locations (don't delete yet)
3. Update imports in new files
4. Add deprecation warnings to old locations
5. Update imports across codebase
6. Delete old files after verification

**Backward Compatibility:**
```python
# src/common/__init__.py
import warnings
warnings.warn("src.common is deprecated. Use src.config.constants", DeprecationWarning)
from src.config.constants import *
```

**Verification:**
- [ ] `from src.config import PipelineConfig, TrainerConfig`
- [ ] `from src.config.constants import CANONICAL_TIMEFRAMES`
- [ ] All pipeline stages work
- [ ] All model training works

**Rollback:** `git checkout HEAD~1 -- src/`

---

## Phase 5: Move Model Config (Low Risk)

**Checkpoint: Git tag `refactor-phase5-start`**

### Step 5.1: Move MODEL_DATA_REQUIREMENTS

**From:** `src/phase1/config/model_config.py`
**To:** `src/models/config/data_requirements.py`

**Includes:**
- `ModelFamily` enum
- `ModelDataRequirements` dataclass
- `MODEL_DATA_REQUIREMENTS` dict
- `EnsembleConfig` dataclass
- Helper functions

**Verification:**
- [ ] `from src.models.config import MODEL_DATA_REQUIREMENTS`
- [ ] `pytest tests/models/test_trainer.py -v`
- [ ] Model training works end-to-end

**Rollback:** `git revert HEAD`

---

## Phase 6: Naming Consistency (Low Risk)

**Checkpoint: Git tag `refactor-phase6-start`**

### Step 6.1: Rename `nbeats.py`

```bash
git mv src/models/neural/nbeats.py src/models/neural/nbeats_model.py
# Update imports in registry and tests
```

**Verification:**
- [ ] `python -c "from src.models.neural.nbeats_model import NBeatsModel"`
- [ ] Model still registers correctly

**Rollback:** `git revert HEAD`

---

## Execution Order Summary

| Phase | Risk | Duration | Checkpoint |
|-------|------|----------|------------|
| 1. Delete dead code | Zero | 5 min | `refactor-phase1-start` |
| 2. Split large files | Medium | 2-3 hours | `refactor-phase2-start` |
| 3. Merge feature selection | High | 4-6 hours | `refactor-phase3-start` |
| 4. Config consolidation | Medium | 3-4 hours | `refactor-phase4-start` |
| 5. Move model config | Low | 1 hour | `refactor-phase5-start` |
| 6. Naming consistency | Zero | 15 min | `refactor-phase6-start` |

**Total Estimated Time:** 1-2 days of focused work

---

## Rollback Strategy

Each phase creates a git tag before starting. To rollback any phase:

```bash
# List all checkpoints
git tag | grep refactor

# Rollback to specific checkpoint
git checkout refactor-phase2-start

# Or revert individual commits
git revert <commit-hash>
```

---

## Verification Checklist (Final)

After all phases complete:

- [ ] All 807+ tests pass (`pytest tests/ -q`)
- [ ] All src/ files compile (`find src -name "*.py" | xargs python -m py_compile`)
- [ ] No circular imports (`python -c "import src"`)
- [ ] Pipeline runs end-to-end (`./pipeline run --symbols MES --dry-run`)
- [ ] Model training works (`python scripts/train_model.py --model xgboost --horizon 20 --dry-run`)
- [ ] Import paths simplified (no warnings in production code)

---

## Post-Refactor Structure

```
src/
    config/                     # NEW: Central config package
        constants/              # Timeframes, horizons, splits
        models/                 # TrainerConfig, feature sets, requirements
        pipeline/               # PipelineConfig, barriers, labeling

    feature_selection/          # CONSOLIDATED: All feature selection
        result.py               # Canonical FeatureSelectionResult
        walk_forward.py         # Walk-forward selector
        ohlcv_selector.py       # OHLCV selector
        manager.py              # Training integration

    models/
        training/               # NEW: Split from trainer.py
            trainer.py          # Core trainer
            evaluation.py       # Metrics
            artifacts.py        # Saving
        config/                 # Model config (receives moved files)
        neural/
            inceptiontime_model.py  # Split from cnn.py
            resnet1d_model.py       # Split from cnn.py
            nbeats_model.py         # Renamed

    cross_validation/           # SIMPLIFIED: No feature selection code
        cv_runner.py            # Core runner
        cv_dataclasses.py       # Split out
        cv_tuner.py             # Split out

    phase1/                     # CONSUMER: Imports from above
        config/
            feature_sets/       # Split into subpackage
        stages/
        utils/                  # SIMPLIFIED: feature_selection.py removed

    common/                     # DEPRECATED: Re-exports from config/constants

    pipeline/                   # UNCHANGED: Orchestration only
```

---

## Benefits

1. **Single ownership** - Each concept has one canonical location
2. **Clear dependencies** - Config flows down, not across
3. **Manageable files** - All under 650-line target
4. **Extensible** - Easy to add new feature sets, models, selectors
5. **Testable** - Smaller files = easier unit testing
6. **Documented** - Structure reflects architecture

---

## Optional: Additional Cleanup

If monitoring module is not in scope:
```bash
rm -rf src/monitoring/
rm -rf tests/monitoring/
# ~1,300 lines removed
```

If notebook support is not needed:
```bash
rm src/utils/notebook.py
rm src/utils/colab_setup.py
rm tests/utils/test_notebook.py
# ~1,050 lines removed
```
