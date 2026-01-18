# Phase A: Data Preparation Migration (Stages 1-6)

**Status:** Planning Complete
**Estimated Effort:** 4 days

---

## Current State Summary

| Stage | Location | Lines | Entry Point |
|-------|----------|-------|-------------|
| 1: Ingestion | `src/phase1/stages/ingest/` | 1,051 | `run_data_generation()` |
| 2: Cleaning | `src/phase1/stages/clean/` | 3,214 | `run_data_cleaning()` |
| 3: Sessions | `src/phase1/stages/sessions/` | 1,922 | `SessionFilter.filter()` |
| 4: MTF | `src/phase1/stages/mtf/` | 975 | `add_mtf_features()` |
| 5: Features | `src/phase1/stages/features/` | 7,422 | `run_feature_engineering()` |
| 6: Regime | `src/phase1/stages/regime/` | 2,106 | `add_regime_features_to_dataframe()` |
| **Total** | | **16,690** | |

---

## Target State

### New File: `src/pipeline/phases/data.py`

```python
class Stage1Ingestion(StageWrapper):
    """Wraps run_data_generation() from src/phase1/stages/ingest/"""
    stage_number = 1

class Stage2Cleaning(StageWrapper):
    """Wraps run_data_cleaning() from src/phase1/stages/clean/"""
    stage_number = 2

class Stage3Sessions(StageWrapper):
    """Wraps SessionFilter from src/phase1/stages/sessions/"""
    stage_number = 3

class Stage4MTFUpscaling(StageWrapper):
    """Wraps MTFGenerator from src/phase1/stages/mtf/"""
    stage_number = 4

class Stage5Features(StageWrapper):
    """Wraps run_feature_engineering() from src/phase1/stages/features/"""
    stage_number = 5

class Stage6Regime(StageWrapper):
    """Wraps CompositeRegimeDetector from src/phase1/stages/regime/"""
    stage_number = 6

class DataPhase:
    """Orchestrates Stages 1-6 in sequence with checkpointing."""
    pass
```

---

## Migration Steps

### Step 1: Create Base Infrastructure (Day 1)
- Create `src/pipeline/phases/__init__.py`
- Create `src/pipeline/phases/base.py` with `StageWrapper` ABC

### Step 2: Implement Stage Wrappers (Days 2-3)
- Each wrapper imports and delegates to existing implementation
- No rewriting of core logic

### Step 3: Create DataPhase Orchestrator (Day 3)
- Chain stages with validation between each
- Add checkpoint saving after Stage 6

### Step 4: Integration (Day 4)
- Wire to MLFactory entry point
- Run regression tests

---

## Interface Contracts

| Stage | Input Type | Output Type |
|-------|-----------|-------------|
| 1 | `None` | `Stage1Output(raw_data, validation_results)` |
| 2 | `Stage1Output` | `Stage2Output(clean_data, metadata)` |
| 3 | `Stage2Output` | `Stage3Output(session_filtered_data)` |
| 4 | `Stage3Output` | `Stage4Output(mtf_data)` |
| 5 | `Stage4Output` | `Stage5Output(features_data, ~180 features)` |
| 6 | `Stage5Output` | `Stage6Output(regime_data, 9 regimes)` |

**Checkpoint:** `data/features/{symbol}_features.parquet`

---

## Testing Strategy

1. **Unit Tests:** Per-stage wrapper validation
2. **Integration Tests:** Full Phase A execution
3. **Regression Tests:** Run existing `tests/phase_1_tests/stages/` suite
4. **Backward Compatibility:** Verify wrapper output matches original function output

---

## Critical Files

1. `src/phase1/stages/features/engineer.py` (639 lines) - Pattern to follow
2. `src/phase1/stages/clean/run.py` (330 lines) - Entry point to wrap
3. `src/pipeline/utils.py` - StageResult utilities
4. `src/phase1/pipeline_config.py` (483 lines) - Config interface
