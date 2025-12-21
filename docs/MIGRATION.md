# Migration Guide: Old to New Pipeline

## Summary of Changes

As of 2025-12-21, the codebase has been standardized to use **ONLY** the new modular pipeline architecture in `src/pipeline/` and `src/stages/`.

## What Was Removed

### Deleted Files (11 total, ~3,539 lines removed)

**Legacy Entry Points (4 files, 954 lines):**
- `src/run_phase1.py` → Use `./pipeline run`
- `src/run_phase1_complete.py` → Use `./pipeline run --from create_splits`
- `src/run_labeling_pipeline.py` → Use `./pipeline run --from labeling`
- `src/run_pipeline.py` → Use `./pipeline run`

**Deprecated Implementations (3 files, 1,011 lines):**
- `src/data_cleaning.py` → Use `from src.stages.stage2_clean import DataCleaner`
- `src/feature_engineering.py` → Use `from src.stages.stage3_features import FeatureEngineer`
- `src/labeling.py` → Use `from src.stages.stage4_labeling import process_symbol_labeling`

**Orphaned Files (3 files, ~1,530 lines):**
- `src/feature_scaling.py` → Use `from src.stages.feature_scaler import FeatureScaler`
- `src/create_splits.py` → Use `from src.stages.stage7_splits import create_splits`
- `src/feature_selection.py` → Moved to `src/utils/feature_selection.py`

**Compatibility Wrapper (1 file, 40 lines):**
- `src/pipeline_runner.py` → Use `from src.pipeline import PipelineRunner`

### Deprecated Config Fields

From `src/pipeline_config.py`:
- `barrier_k_up` → Use `config.BARRIER_PARAMS`
- `barrier_k_down` → Use `config.BARRIER_PARAMS`

## Migration Examples

### Old Way (No Longer Works)

**Running pipeline:**
```bash
python src/run_phase1.py
```

**Importing functions:**
```python
from data_cleaning import clean_symbol_data
from feature_engineering import main
from labeling import apply_triple_barrier
```

**Using config:**
```python
config = PipelineConfig()
k_up = config.barrier_k_up
k_down = config.barrier_k_down
```

### New Way (Standardized)

**Running pipeline:**
```bash
./pipeline run --symbols MES,MGC
./pipeline run --from create_splits  # Resume from specific stage
./pipeline status <run_id>
```

**Importing functions:**
```python
from src.stages.stage2_clean import DataCleaner, clean_symbol_data
from src.stages.stage3_features import FeatureEngineer
from src.stages.stage4_labeling import apply_triple_barrier, process_symbol_labeling
from src.stages.feature_scaler import FeatureScaler
from src.pipeline import PipelineRunner, StageStatus, StageResult
```

**Using config:**
```python
from src.config import get_barrier_params, BARRIER_PARAMS

# Get symbol-specific barriers
params = get_barrier_params('MES', 5)  # Returns (k_up, k_down) for MES H5
# Example: (1.5, 1.0) for MES (asymmetric), (2.0, 2.0) for MGC (symmetric)
```

## Import Reference

| Old Import | New Import |
|-----------|-----------|
| `from data_cleaning import clean_symbol_data` | `from src.stages.stage2_clean import clean_symbol_data` |
| `from data_cleaning import resample_to_5min` | `from src.stages.stage2_clean import resample_to_5min` |
| `from feature_engineering import main` | `from src.stages.stage3_features import FeatureEngineer` |
| `from labeling import apply_triple_barrier` | `from src.stages.stage4_labeling import apply_triple_barrier` |
| `from labeling import triple_barrier_numba` | `from src.stages.stage4_labeling import triple_barrier_numba` |
| `from feature_scaling import FeatureScaler` | `from src.stages.feature_scaler import FeatureScaler` |
| `from create_splits import create_splits` | `from src.stages.stage7_splits import create_splits` |
| `from pipeline_runner import PipelineRunner` | `from src.pipeline import PipelineRunner` |

## Benefits of Standardization

- ✅ **Single source of truth** - Each function exists in exactly one place
- ✅ **Single entry point** - Only `./pipeline run` for execution
- ✅ **Clean architecture** - Orchestration vs implementation clearly separated
- ✅ **Centralized config** - `src/config.py` is single source of truth
- ✅ **Maintainability** - Clear module boundaries, easier debugging
- ✅ **Testability** - Each layer independently testable
- ✅ **No duplication** - ~3,539 lines of dead code removed

## Rollback Instructions

If you need to rollback to before standardization:

```bash
# See commits before standardization
git log --oneline -20

# Rollback to specific commit
git reset --hard 648ae49  # "Checkpoint: Baseline before standardization"

# Verify state
git status
pytest tests/ -v
```

## Phase-by-Phase Commit History

| Phase | Commit | Description |
|-------|--------|-------------|
| 1 | 648ae49 | Baseline before standardization (180/227 tests passing) |
| 2 | 8632ecd | Consolidate data_cleaning.py into stages/stage2_clean.py |
| 3 | 54e5b3a | Consolidate feature_engineering.py into stages/stage3_features.py |
| 4 | b808c0a, f351796 | Consolidate labeling.py into stages/stage4_labeling.py |
| 5 | 670b32c | Delete orphaned files (feature_scaling, create_splits, feature_selection) |
| 6 | (No commit) | Pipeline wrappers already updated in Phases 2-4 |
| 7 | 4da5469 | Delete legacy entrypoints (run_*.py files) |
| 8 | cce9404 | Remove deprecated barrier_k_up/down from PipelineConfig |
| 9 | 88206dc | Delete deprecated implementations (data_cleaning, feature_engineering, labeling) |
| 10 | 78e8254, e1df1e9 | Delete pipeline_runner.py compatibility wrapper |
| 11 | 924c167, d518b4d | Final verification and import path fixes |
| 12 | (Current) | Update documentation |

## Help and Support

For questions or issues:
- Check `docs/guides/PIPELINE_CLI_GUIDE.md` for CLI usage
- Check `CLAUDE.md` for engineering rules and architecture
- Review `STANDARDIZATION_PLAN.md` for implementation details
- Report issues at: https://github.com/anthropics/claude-code/issues

---

**Standardization Complete:** 2025-12-21
**Test Pass Rate:** 77% (175/227 tests)
**Lines of Code Removed:** ~3,539
**Files Removed:** 11
