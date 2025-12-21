# AGGRESSIVE CODEBASE STANDARDIZATION PLAN
## Goal: Consolidate to Sneh's New Modular Pipeline Architecture

**Execution Date:** 2025-12-21
**Approach:** Option A - Aggressive Full Cleanup
**Timeline:** ~2-3 hours
**Risk Level:** Low (comprehensive test coverage exists)

---

## OVERVIEW

This plan standardizes the codebase to use **ONLY** the new modular pipeline architecture in `src/pipeline/`, eliminating all duplicate implementations and deprecated entry points.

### Target State:
- **Single orchestration layer:** `src/pipeline/` (DAG-based)
- **Single implementation layer:** `src/stages/` (actual logic)
- **Single entry point:** `./pipeline run` CLI
- **Single config source:** `config.py` (constants) + `pipeline_config.py` (runs)
- **Zero duplication:** No duplicate implementations
- **Clean legacy:** All deprecated code removed

---

## CRITICAL RULES

### Before Each Phase:
1. ✅ Read files to understand current state
2. ✅ Verify functionality equivalence before consolidation
3. ✅ Run tests after each change
4. ✅ Create git commit with descriptive message

### Rollback Plan:
- Each phase is a separate git commit
- If any phase fails: `git reset --hard HEAD~1`
- If multiple phases fail: `git log` to find last good commit

### Verification Requirements:
- All existing tests must pass
- Manual smoke test: `./pipeline run --symbols MES,MGC` (if data available)
- No import errors when importing from `src/`

---

## PHASE 1: ESTABLISH BASELINE ✅

**Objective:** Verify current state is working before making changes

### Actions:
1. Run full test suite to establish baseline
   ```bash
   pytest tests/ -v --tb=short > baseline_test_results.txt
   ```

2. Check for import errors in main modules
   ```bash
   python -c "import sys; sys.path.insert(0, 'src'); from pipeline import PipelineRunner"
   python -c "import sys; sys.path.insert(0, 'src'); from config import BARRIER_PARAMS"
   ```

3. Verify pipeline CLI works
   ```bash
   ./pipeline --help
   ./pipeline validate
   ```

### Success Criteria:
- Tests pass (or document known failures to exclude)
- No import errors
- CLI responds correctly

### Git Commit:
```bash
git add -A
git commit -m "Checkpoint: Baseline before standardization (all tests passing)"
```

---

## PHASE 2: CONSOLIDATE DATA CLEANING LOGIC 🔄

**Objective:** Merge `src/data_cleaning.py` into `src/stages/stage2_clean.py`

### Current State Analysis:

**File 1:** `src/data_cleaning.py` (188 lines)
- Functions: `clean_symbol_data()`, `resample_to_5min()`, `remove_gaps()`, `detect_outliers()`
- Used by: `run_phase1.py`, `src/pipeline/stages/data_cleaning.py`

**File 2:** `src/stages/stage2_clean.py` (566 lines)
- Class: `DataCleaner` with methods `clean_file()`, `_parse_timeframe()`, etc.
- Used by: Legacy imports

### Consolidation Strategy:

#### Step 2.1: Read both files completely
```bash
# Read to understand full functionality
Read src/data_cleaning.py
Read src/stages/stage2_clean.py
```

#### Step 2.2: Identify unique functionality
- Compare function signatures
- Find functions in `data_cleaning.py` NOT in `stage2_clean.py`
- Verify `stage2_clean.py` has superset of functionality

#### Step 2.3: Port missing functions (if any)
If `data_cleaning.py` has unique functions:
- Copy them to `stage2_clean.py`
- Ensure imports are correct
- Add docstrings if missing

#### Step 2.4: Update imports in pipeline wrapper
**File:** `src/pipeline/stages/data_cleaning.py`

Before:
```python
from data_cleaning import clean_symbol_data
```

After:
```python
from stages.stage2_clean import DataCleaner

def run(config):
    cleaner = DataCleaner(...)
    cleaner.clean_file(...)
```

#### Step 2.5: Verify functionality
```bash
# Run cleaning-related tests
pytest tests/test_stages.py::test_data_cleaning -v
pytest tests/test_pipeline.py -v
```

#### Step 2.6: Mark data_cleaning.py as deprecated
Add to top of `src/data_cleaning.py`:
```python
import warnings
warnings.warn(
    "data_cleaning.py is DEPRECATED. Use 'from stages.stage2_clean import DataCleaner' instead.",
    DeprecationWarning,
    stacklevel=2
)
```

### Success Criteria:
- `stage2_clean.py` has all functionality from `data_cleaning.py`
- Pipeline wrapper updated to use `stage2_clean`
- All tests pass
- Deprecation warning added

### Git Commit:
```bash
git add src/stages/stage2_clean.py src/pipeline/stages/data_cleaning.py src/data_cleaning.py
git commit -m "Consolidate: Merge data_cleaning.py into stages/stage2_clean.py"
```

---

## PHASE 3: CONSOLIDATE FEATURE ENGINEERING LOGIC 🔄

**Objective:** Merge `src/feature_engineering.py` into `src/stages/stage3_features.py`

### Current State Analysis:

**File 1:** `src/feature_engineering.py` (364 lines)
- Functions: `main()`, `calculate_sma()`, `calculate_ema()`, `calculate_rsi()`, etc.
- Used by: `run_phase1.py`, `src/pipeline/stages/feature_engineering.py`

**File 2:** `src/stages/stage3_features.py` (73 lines)
- Thin wrapper around `feature_engineering.py`
- Uses: `FeatureEngineer` class

### Consolidation Strategy:

#### Step 3.1: Read both files completely
```bash
Read src/feature_engineering.py
Read src/stages/stage3_features.py
```

#### Step 3.2: Identify unique functionality
- Compare all feature calculation functions
- Verify no feature calculators are missing

#### Step 3.3: Expand stage3_features.py
If `feature_engineering.py` has unique functions:
- Copy all feature calculation functions to `stage3_features.py`
- Organize into logical sections (trend, momentum, volatility)
- Ensure all imports present

#### Step 3.4: Update imports in pipeline wrapper
**File:** `src/pipeline/stages/feature_engineering.py`

Before:
```python
from feature_engineering import main
```

After:
```python
from stages.stage3_features import FeatureEngineer

def run(config):
    engineer = FeatureEngineer(...)
    engineer.process_file(...)
```

#### Step 3.5: Verify functionality
```bash
pytest tests/test_stages.py::test_feature_engineering -v
pytest tests/test_edge_cases.py -v
```

#### Step 3.6: Mark feature_engineering.py as deprecated
Add deprecation warning to `src/feature_engineering.py`

### Success Criteria:
- `stage3_features.py` has all functionality from `feature_engineering.py`
- Pipeline wrapper updated
- All tests pass
- Deprecation warning added

### Git Commit:
```bash
git add src/stages/stage3_features.py src/pipeline/stages/feature_engineering.py src/feature_engineering.py
git commit -m "Consolidate: Merge feature_engineering.py into stages/stage3_features.py"
```

---

## PHASE 4: CONSOLIDATE LABELING LOGIC 🔄

**Objective:** Merge `src/labeling.py` into `src/stages/stage4_labeling.py`

### Current State Analysis:

**File 1:** `src/labeling.py` (397 lines)
- Functions: `triple_barrier_numba()`, `apply_triple_barrier()`, etc.
- Used by: `run_phase1.py`

**File 2:** `src/stages/stage4_labeling.py` (361 lines)
- **ALREADY UPDATED IN PHASE A** - Uses `config.BARRIER_PARAMS`
- Functions: `triple_barrier_numba()`, `apply_triple_barrier()`, `process_symbol_labeling()`
- Used by: `src/pipeline/stages/labeling.py`

### Consolidation Strategy:

#### Step 4.1: Read both files completely
```bash
Read src/labeling.py
Read src/stages/stage4_labeling.py
```

#### Step 4.2: Compare implementations
- Verify `stage4_labeling.py` has same or better implementation
- Check for any unique functionality in `labeling.py`
- Confirm barrier parameter handling

#### Step 4.3: Port any missing functionality
If `labeling.py` has unique features:
- Add to `stage4_labeling.py`
- Ensure uses `config.get_barrier_params(symbol, horizon)`

#### Step 4.4: Verify pipeline already uses stage4
**File:** `src/pipeline/stages/labeling.py`
- Should already import from `stages.stage4_labeling`
- If not, update imports

#### Step 4.5: Verify functionality
```bash
pytest tests/test_phase1_stages.py::test_labeling -v
pytest tests/test_validation.py -v
```

#### Step 4.6: Mark labeling.py as deprecated
Add deprecation warning to `src/labeling.py`

### Success Criteria:
- `stage4_labeling.py` has all functionality from `labeling.py`
- Uses consolidated config from Phase A
- All tests pass
- Deprecation warning added

### Git Commit:
```bash
git add src/stages/stage4_labeling.py src/labeling.py
git commit -m "Consolidate: Verify labeling.py fully replaced by stages/stage4_labeling.py"
```

---

## PHASE 5: DELETE ORPHANED FILES 💀

**Objective:** Remove dead code with zero dependencies

### Files to Delete:

1. **`src/feature_scaling.py`** (1,029 lines)
   - Replaced by: `src/stages/feature_scaler.py`
   - Verify: `grep -r "from feature_scaling import" src/` returns nothing
   - Verify: `grep -r "import feature_scaling" src/` returns nothing

2. **`src/create_splits.py`** (109 lines)
   - Replaced by: `src/stages/stage7_splits.py`
   - Verify: No imports found

3. **`src/feature_selection.py`** (560 lines)
   - Only used in: `tests/test_phase1_stages_advanced.py` (optional test)
   - Can be deleted or moved to `src/utils/` if needed later

### Actions:

#### Step 5.1: Verify no active imports
```bash
# Check each file has no dependencies
grep -r "from feature_scaling import" src/ tests/
grep -r "import feature_scaling" src/ tests/
grep -r "from create_splits import" src/ tests/
grep -r "from feature_selection import" src/ tests/
```

#### Step 5.2: Delete files
```bash
rm src/feature_scaling.py
rm src/create_splits.py
rm src/feature_selection.py  # or mv to src/utils/
```

#### Step 5.3: Update tests if needed
If `test_phase1_stages_advanced.py` breaks:
- Comment out or skip tests that use `feature_selection`
- OR move `feature_selection.py` to `src/utils/`

#### Step 5.4: Verify tests still pass
```bash
pytest tests/ -v --ignore=tests/test_phase1_stages_advanced.py
```

### Success Criteria:
- 3 files deleted (~1,700 lines removed)
- No import errors
- Core tests pass

### Git Commit:
```bash
git add -A
git commit -m "Delete orphaned files: feature_scaling.py, create_splits.py, feature_selection.py"
```

---

## PHASE 6: UPDATE PIPELINE WRAPPERS 🔧

**Objective:** Ensure all `src/pipeline/stages/*.py` use consolidated code

### Files to Update:

1. **`src/pipeline/stages/data_cleaning.py`**
   - Should import from `stages.stage2_clean`
   - Already updated in Phase 2

2. **`src/pipeline/stages/feature_engineering.py`**
   - Should import from `stages.stage3_features`
   - Already updated in Phase 3

3. **`src/pipeline/stages/labeling.py`**
   - Should import from `stages.stage4_labeling`, `stages.stage6_final_labels`
   - Verify imports

### Actions:

#### Step 6.1: Verify all pipeline wrappers
```bash
# Check imports in each wrapper
grep -n "import" src/pipeline/stages/*.py
```

#### Step 6.2: Update any remaining legacy imports
For each wrapper in `src/pipeline/stages/`:
- Remove imports from `data_cleaning`, `feature_engineering`, `labeling`
- Add imports from `stages.stage2_clean`, `stages.stage3_features`, `stages.stage4_labeling`

#### Step 6.3: Verify pipeline runs
```bash
./pipeline validate
pytest tests/test_pipeline_system.py -v
```

### Success Criteria:
- All pipeline wrappers import from `src/stages/`
- No imports from deprecated files
- Pipeline validation passes

### Git Commit:
```bash
git add src/pipeline/stages/
git commit -m "Update pipeline wrappers to use consolidated code from src/stages/"
```

---

## PHASE 7: DELETE LEGACY ENTRYPOINTS 🗑️

**Objective:** Remove all deprecated `run_*.py` entry points

### Files to Delete:

1. **`src/run_phase1.py`** (~300 lines)
   - Legacy manual 6-stage runner
   - Replaced by: `./pipeline run`

2. **`src/run_phase1_complete.py`** (~200 lines)
   - Runs stages 7-9 only
   - Replaced by: `./pipeline run --from create_splits`

3. **`src/run_labeling_pipeline.py`** (~150 lines)
   - Runs stages 4-6 only
   - Replaced by: `./pipeline run --from labeling`

4. **`src/run_pipeline.py`** (~200 lines)
   - Runs stages 1-3 only
   - Incomplete, replaced by full pipeline

### Actions:

#### Step 7.1: Verify no critical code in legacy entrypoints
```bash
# Read each file to check for unique logic
Read src/run_phase1.py
Read src/run_phase1_complete.py
Read src/run_labeling_pipeline.py
Read src/run_pipeline.py
```

#### Step 7.2: Extract any unique utility functions
If any `run_*.py` file has utility functions not elsewhere:
- Move to appropriate `src/stages/` file
- Document in commit message

#### Step 7.3: Delete legacy entrypoints
```bash
rm src/run_phase1.py
rm src/run_phase1_complete.py
rm src/run_labeling_pipeline.py
rm src/run_pipeline.py
```

#### Step 7.4: Verify nothing breaks
```bash
pytest tests/test_pipeline.py -v
pytest tests/test_pipeline_system.py -v
```

### Success Criteria:
- 4 legacy entry points deleted (~850 lines)
- Any unique functionality preserved
- All tests pass

### Git Commit:
```bash
git add -A
git commit -m "Delete legacy entrypoints: run_phase1.py, run_phase1_complete.py, run_labeling_pipeline.py, run_pipeline.py"
```

---

## PHASE 8: CLEAN DEPRECATED CONFIG FIELDS ⚙️

**Objective:** Remove deprecated fields from `src/pipeline_config.py`

### Deprecated Fields:

From **Phase A**, these were marked deprecated:
- `barrier_k_up: float = 2.0  # DEPRECATED`
- `barrier_k_down: float = 2.0  # DEPRECATED`

These are no longer used (replaced by `config.BARRIER_PARAMS`)

### Actions:

#### Step 8.1: Read current state
```bash
Read src/pipeline_config.py
```

#### Step 8.2: Verify fields are truly unused
```bash
# Check if any code still references these fields
grep -r "\.barrier_k_up" src/ tests/
grep -r "\.barrier_k_down" src/ tests/
grep -r "barrier_k_up" src/ tests/ | grep -v "DEPRECATED"
```

#### Step 8.3: Remove deprecated fields
**File:** `src/pipeline_config.py`

Remove:
```python
barrier_k_up: float = 2.0  # DEPRECATED
barrier_k_down: float = 2.0  # DEPRECATED
```

Remove from validation (lines 286-290):
```python
if self.barrier_k_up <= 0:
    issues.append(...)
if self.barrier_k_down <= 0:
    issues.append(...)
```

Remove from `__str__()` (lines 359-360):
```python
- Barrier K-Up: {self.barrier_k_up} (DEPRECATED)
- Barrier K-Down: {self.barrier_k_down} (DEPRECATED)
```

#### Step 8.4: Update docstring
Add note to class docstring:
```python
"""
Complete configuration for Phase 1 pipeline.

Note: Barrier parameters are now in src/config.py as BARRIER_PARAMS.
Use config.get_barrier_params(symbol, horizon) for symbol-specific values.
"""
```

#### Step 8.5: Verify tests pass
```bash
pytest tests/test_pipeline_config.py -v
pytest tests/test_validation.py -v
```

### Success Criteria:
- Deprecated fields removed
- No references in codebase
- Tests pass
- Docstring updated

### Git Commit:
```bash
git add src/pipeline_config.py
git commit -m "Remove deprecated barrier_k_up/barrier_k_down from PipelineConfig"
```

---

## PHASE 9: DELETE DEPRECATED IMPLEMENTATION FILES 🗑️

**Objective:** Remove deprecated duplicate implementations

### Files to Delete:

Now that pipeline uses `src/stages/` and legacy entrypoints are gone:

1. **`src/data_cleaning.py`** (188 lines)
   - Marked deprecated in Phase 2
   - Functionality in `stages/stage2_clean.py`

2. **`src/feature_engineering.py`** (364 lines)
   - Marked deprecated in Phase 3
   - Functionality in `stages/stage3_features.py`

3. **`src/labeling.py`** (397 lines)
   - Marked deprecated in Phase 4
   - Functionality in `stages/stage4_labeling.py`

### Actions:

#### Step 9.1: Final verification - no imports remain
```bash
# Verify NO code imports from these deprecated files
grep -r "from data_cleaning import" src/
grep -r "from feature_engineering import" src/
grep -r "from labeling import" src/
grep -r "import data_cleaning" src/
grep -r "import feature_engineering" src/
grep -r "import labeling" src/
```

**Expected output:** Nothing (or only in deprecated `run_*.py` which we already deleted)

#### Step 9.2: Check tests don't directly import
```bash
grep -r "from data_cleaning import" tests/
grep -r "from feature_engineering import" tests/
grep -r "from labeling import" tests/
```

If tests import these:
- Update tests to import from `stages.*` instead

#### Step 9.3: Delete deprecated files
```bash
rm src/data_cleaning.py
rm src/feature_engineering.py
rm src/labeling.py
```

#### Step 9.4: Run full test suite
```bash
pytest tests/ -v
```

### Success Criteria:
- 3 deprecated files deleted (~949 lines)
- No import errors
- All tests pass

### Git Commit:
```bash
git add -A
git commit -m "Delete deprecated implementations: data_cleaning.py, feature_engineering.py, labeling.py"
```

---

## PHASE 10: DELETE COMPATIBILITY WRAPPER 🗑️

**Objective:** Remove `src/pipeline_runner.py` backward compatibility wrapper

### Current State:

**File:** `src/pipeline_runner.py` (40 lines)
- Re-exports from `pipeline/` for backward compatibility
- Purpose: Allow `from pipeline_runner import PipelineRunner` instead of `from pipeline import`

### Actions:

#### Step 10.1: Verify no imports from pipeline_runner
```bash
# Check if anything still imports from pipeline_runner
grep -r "from pipeline_runner import" src/ tests/
grep -r "import pipeline_runner" src/ tests/
```

**Expected:** Only `pipeline_runner.py` itself (the `if __name__ == "__main__"` block)

#### Step 10.2: Update any remaining imports
If any files import from `pipeline_runner`:
```python
# Before
from pipeline_runner import PipelineRunner

# After
from pipeline import PipelineRunner
```

#### Step 10.3: Delete wrapper
```bash
rm src/pipeline_runner.py
```

#### Step 10.4: Verify pipeline still works
```bash
./pipeline --help
./pipeline validate
python -c "from pipeline import PipelineRunner; print('✓ Direct import works')"
```

### Success Criteria:
- Wrapper deleted (40 lines)
- Direct imports from `pipeline` work
- CLI still functions

### Git Commit:
```bash
git add -A
git commit -m "Delete backward compatibility wrapper: pipeline_runner.py"
```

---

## PHASE 11: FINAL VERIFICATION 🧪

**Objective:** Comprehensive validation that everything works

### Test Suite:

#### Step 11.1: Run full test suite
```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Save results
pytest tests/ -v > final_test_results.txt
```

#### Step 11.2: Verify specific functionality
```bash
# Test imports work
python -c "
import sys
sys.path.insert(0, 'src')
from pipeline import PipelineRunner
from config import BARRIER_PARAMS, get_barrier_params
from stages.stage2_clean import DataCleaner
from stages.stage3_features import FeatureEngineer
from stages.stage4_labeling import process_symbol_labeling
print('✓ All imports successful')
"
```

#### Step 11.3: Verify CLI works
```bash
./pipeline --help
./pipeline validate
# ./pipeline run --symbols MES,MGC --description "Standardization test"  # If data available
```

#### Step 11.4: Check for broken imports
```bash
# Look for any import errors
python -c "
import sys
import os
sys.path.insert(0, 'src')
for module in ['config', 'pipeline_config', 'manifest', 'pipeline_cli']:
    __import__(module)
    print(f'✓ {module}')
"
```

#### Step 11.5: Verify file structure
```bash
# Expected structure
tree src/ -L 2 -I '__pycache__|*.pyc'
```

**Expected output:**
```
src/
├── config.py                    # Single source of truth
├── pipeline_config.py           # Run-specific config
├── manifest.py                  # Artifact tracking
├── pipeline_cli.py              # CLI entry point
├── generate_synthetic_data.py   # Utility
├── pipeline/                    # Orchestration layer
│   ├── __init__.py
│   ├── runner.py
│   ├── stage_registry.py
│   ├── utils.py
│   └── stages/                  # Thin wrappers
└── stages/                      # Implementation layer
    ├── __init__.py
    ├── stage1_ingest.py
    ├── stage2_clean.py
    ├── stage3_features.py
    ├── stage4_labeling.py
    ├── stage5_ga_optimize.py
    ├── stage6_final_labels.py
    ├── stage7_splits.py
    ├── stage8_validate.py
    ├── feature_scaler.py
    ├── generate_report.py
    ├── baseline_backtest.py
    └── time_series_cv.py
```

### Success Criteria:
- All tests pass
- All imports work
- CLI functional
- File structure clean
- No deprecated code remains

### Git Commit:
```bash
git add -A
git commit -m "Verification: All tests pass after full standardization"
```

---

## PHASE 12: UPDATE DOCUMENTATION 📚

**Objective:** Update README and guides to reflect new architecture

### Actions:

#### Step 12.1: Update README.md
**Sections to modify:**

1. **Quick Start** - Only show `./pipeline run`
   ```markdown
   ## Quick Start

   ```bash
   # Install dependencies
   pip install -r requirements.txt

   # Run the pipeline
   ./pipeline run --symbols MES,MGC

   # Check status
   ./pipeline status <run_id>
   ```
   ```

2. **Remove legacy commands**
   - Delete references to `run_phase1.py`, etc.
   - Remove old Python API examples using deprecated files

3. **Add architecture section**
   ```markdown
   ## Architecture

   The pipeline uses a modular architecture:

   - `src/pipeline/` - Orchestration layer (9-stage DAG)
   - `src/stages/` - Implementation layer (actual logic)
   - `src/config.py` - Central configuration
   - `./pipeline` - CLI entry point
   ```

#### Step 12.2: Update CLAUDE.md
**Sections to modify:**

1. **Remove deprecated build commands**
   - Delete references to `run_phase1.py`
   - Keep only `./pipeline` commands

2. **Update file structure**
   - Show new clean structure
   - Remove deleted files

3. **Update engineering rules compliance**
   - Note that duplication eliminated
   - Show file count reduced

#### Step 12.3: Create MIGRATION.md
**New file:** `docs/MIGRATION.md`

```markdown
# Migration Guide: Old to New Pipeline

## Summary of Changes

As of 2025-12-21, the codebase has been standardized to use ONLY the new modular pipeline architecture.

### What Was Removed

**Deleted Files:**
- `src/run_phase1.py` → Use `./pipeline run`
- `src/run_phase1_complete.py` → Use `./pipeline run --from create_splits`
- `src/run_labeling_pipeline.py` → Use `./pipeline run --from labeling`
- `src/run_pipeline.py` → Use `./pipeline run`
- `src/data_cleaning.py` → Use `from stages.stage2_clean import`
- `src/feature_engineering.py` → Use `from stages.stage3_features import`
- `src/labeling.py` → Use `from stages.stage4_labeling import`
- `src/feature_scaling.py` → Use `from stages.feature_scaler import`
- `src/create_splits.py` → Use `from stages.stage7_splits import`
- `src/pipeline_runner.py` → Use `from pipeline import`

**Deprecated Config Fields:**
- `PipelineConfig.barrier_k_up` → Use `config.BARRIER_PARAMS`
- `PipelineConfig.barrier_k_down` → Use `config.BARRIER_PARAMS`

### Migration Examples

#### Old Way (No Longer Works):
```python
from run_phase1 import main
main()
```

#### New Way:
```bash
./pipeline run --symbols MES,MGC
```

#### Old Import (No Longer Works):
```python
from data_cleaning import clean_symbol_data
```

#### New Import:
```python
from stages.stage2_clean import DataCleaner
cleaner = DataCleaner(...)
```

## Benefits

- ✅ Single source of truth for all logic
- ✅ Modular, testable architecture
- ✅ Clear separation of concerns
- ✅ No code duplication
- ✅ ~3,500 lines of dead code removed
```

#### Step 12.4: Update documentation index
**File:** `docs/guides/PIPELINE_CLI_GUIDE.md`

- Verify it only shows `./pipeline` commands
- Remove any references to legacy `run_*.py`

### Success Criteria:
- README updated
- CLAUDE.md updated
- Migration guide created
- No references to deleted files in docs

### Git Commit:
```bash
git add README.md CLAUDE.md docs/MIGRATION.md docs/guides/
git commit -m "Update documentation: Remove legacy references, add migration guide"
```

---

## FINAL SUMMARY 📊

### What Gets Deleted:

| Category | Files | Lines Removed |
|----------|-------|---------------|
| Orphaned code | 3 files | ~1,700 |
| Deprecated implementations | 3 files | ~949 |
| Legacy entrypoints | 4 files | ~850 |
| Compatibility wrapper | 1 file | ~40 |
| **TOTAL** | **11 files** | **~3,539 lines** |

### What Remains (Clean):

```
src/
├── config.py                    ← Single config source
├── pipeline_config.py           ← Run-specific config (cleaned)
├── manifest.py                  ← Artifact tracking
├── pipeline_cli.py              ← CLI entry point
├── generate_synthetic_data.py   ← Utility
├── pipeline/                    ← Orchestration (4 files)
│   ├── __init__.py
│   ├── runner.py
│   ├── stage_registry.py
│   ├── utils.py
│   └── stages/                  ← Wrappers (8 files)
└── stages/                      ← Implementation (12 files)
    ├── stage1_ingest.py
    ├── stage2_clean.py          ← Consolidated
    ├── stage3_features.py       ← Consolidated
    ├── stage4_labeling.py       ← Consolidated
    ├── stage5_ga_optimize.py
    ├── stage6_final_labels.py
    ├── stage7_splits.py
    ├── stage8_validate.py
    ├── feature_scaler.py
    ├── generate_report.py
    ├── baseline_backtest.py
    └── time_series_cv.py
```

### Benefits Achieved:

✅ **Zero duplication** - Each function exists in exactly one place
✅ **Single entry point** - Only `./pipeline run`
✅ **Clean architecture** - Orchestration vs implementation clearly separated
✅ **Centralized config** - `config.py` is single source of truth
✅ **Maintainability** - Clear module boundaries
✅ **Testability** - Each layer independently testable

---

## EXECUTION CHECKLIST ✅

Before starting:
- [ ] Commit current changes
- [ ] Ensure git status is clean
- [ ] Run baseline tests

Phase execution:
- [ ] Phase 1: Establish baseline ✅
- [ ] Phase 2: Consolidate data_cleaning.py
- [ ] Phase 3: Consolidate feature_engineering.py
- [ ] Phase 4: Consolidate labeling.py
- [ ] Phase 5: Delete orphaned files
- [ ] Phase 6: Update pipeline wrappers
- [ ] Phase 7: Delete legacy entrypoints
- [ ] Phase 8: Clean deprecated config fields
- [ ] Phase 9: Delete deprecated implementations
- [ ] Phase 10: Delete compatibility wrapper
- [ ] Phase 11: Final verification
- [ ] Phase 12: Update documentation

After completion:
- [ ] All tests pass
- [ ] Pipeline CLI works
- [ ] Documentation updated
- [ ] Create final commit: "Complete codebase standardization to modular pipeline"

---

## ROLLBACK PROCEDURE 🔄

If anything goes wrong:

```bash
# See recent commits
git log --oneline -15

# Rollback to last good commit
git reset --hard <commit-hash>

# OR rollback last N commits
git reset --hard HEAD~N

# Verify state
git status
pytest tests/ -v
```

---

**Ready to execute? Confirm to begin Phase 1.**
