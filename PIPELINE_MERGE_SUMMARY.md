# Pipeline Structure Merge - Stages 1-3

## Summary

Merged the dual pipeline structure for Stages 1-3 by creating `run.py` files in each stage subdirectory containing the orchestration logic previously in `src/pipeline/stages/`.

## Changes Made

### Stage 1: Ingest (data_generation)

**Created:** `/home/jake/Desktop/Research/src/stages/ingest/run.py` (187 lines)
- Contains `run_data_generation()` function
- Imports from local modules using relative imports: `from . import DataIngestor`
- Returns StageResult as before
- All orchestration logic moved from `src/pipeline/stages/data_generation.py`

**Updated:** `/home/jake/Desktop/Research/src/pipeline/stages/data_generation.py` (17 lines)
- Now a thin wrapper that imports from `src.stages.ingest.run`
- Maintains backward compatibility

### Stage 2: Clean (data_cleaning)

**Renamed:** `src/stages/stage2_clean/` → `src/stages/clean/`
- Updated all references in codebase

**Created:** `/home/jake/Desktop/Research/src/stages/clean/run.py` (142 lines)
- Contains `run_data_cleaning()` function
- Imports from local modules: `from . import clean_symbol_data`
- Returns StageResult as before
- All orchestration logic moved from `src/pipeline/stages/data_cleaning.py`

**Updated:** `/home/jake/Desktop/Research/src/pipeline/stages/data_cleaning.py` (17 lines)
- Now a thin wrapper that imports from `src.stages.clean.run`
- Maintains backward compatibility

**Updated:** `src/stages/clean/__init__.py`
- Fixed documentation example to use correct import path

**Updated:** `src/stages/__init__.py`
- Changed import from `stage2_clean` to `clean`

### Stage 3: Features (feature_engineering)

**Created:** `/home/jake/Desktop/Research/src/stages/features/run.py` (186 lines)
- Contains `run_feature_engineering()` function
- Imports from local modules: `from .engineer import FeatureEngineer`
- Returns StageResult as before
- All orchestration logic moved from `src/pipeline/stages/feature_engineering.py`

**Updated:** `/home/jake/Desktop/Research/src/pipeline/stages/feature_engineering.py` (17 lines)
- Now a thin wrapper that imports from `src.stages.features.run`
- Maintains backward compatibility

## File Line Counts

All files remain well under the 650-line limit:

| File | Lines |
|------|-------|
| `src/stages/ingest/run.py` | 187 |
| `src/stages/clean/run.py` | 142 |
| `src/stages/features/run.py` | 186 |
| `src/pipeline/stages/data_generation.py` | 17 |
| `src/pipeline/stages/data_cleaning.py` | 17 |
| `src/pipeline/stages/feature_engineering.py` | 17 |

## Benefits

1. **Single Source of Truth**: Each stage's orchestration logic is now in one place (`run.py`)
2. **Cleaner Separation**: Pipeline wrappers are minimal (17 lines each)
3. **Better Modularity**: Stage subdirectories are self-contained
4. **Backward Compatibility**: All imports still work via wrapper files
5. **Follows Project Rules**: All files under 650 lines

## Testing

Structure verified:
- All `run.py` files created in correct locations
- `stage2_clean` successfully renamed to `clean`
- All imports updated correctly
- File structure validated

## Next Steps

The pipeline stages 1-3 now follow the consolidated structure. Future stages can follow this same pattern:
1. Create `src/stages/<stage_name>/run.py` with orchestration logic
2. Create minimal wrapper in `src/pipeline/stages/<stage>.py` that imports from run.py
