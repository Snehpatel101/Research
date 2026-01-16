# Configuration Refactoring Summary - 2026-01-15

## Executive Summary

**Goal:** Centralize configuration, make timeframes fully dynamic, implement per-model feature selection, and align codebase with OHLCV factory goals.

**Status:** Phase 1-3 complete (config centralization), Phase 4-6 in progress

---

## ✅ Completed Work (7/13 tasks - ~16 hours)

### Phase 1: Config Audit (2-3 hours)
- **Deliverable:** `docs/implementation/CONFIG_AUDIT_2026_01_15.md`
- Identified all hardcoded values across codebase
- Mapped competing configuration locations
- Designed centralized architecture

### Phase 2: Global Config System (2-3 hours)
- **Deliverables:**
  - `config/global.yaml` - Single source of truth (170+ lines)
  - `src/config/global_config.py` - Type-safe Python dataclasses
  - `tests/config/test_global_config.py` - 21 tests passing

**Key Features:**
- All defaults in one YAML file
- Lazy loading to avoid circular imports
- Singleton pattern with `get_global_config()`
- Full type safety with dataclasses

### Phase 3: Integration (4-5 hours)
**Files Updated:**
- `src/phase1/pipeline_config.py` - 15+ fields now use GlobalConfig
- `src/models/config/trainer_config.py` - 15+ fields now use GlobalConfig  
- `src/common/horizon_config.py` - Horizons/splits from GlobalConfig
- `src/config/__init__.py` - Exports GlobalConfig

**Test Results:**
- 106 config tests passing
- 21 global config tests passing
- Full backward compatibility maintained

---

## 🚧 In Progress (Task 8 - Per-Model Feature Selection)

### Goal
Each model should specify its own feature selection requirements in MODEL_DATA_REQUIREMENTS.

### Design
Extended `ModelDataRequirements` dataclass with:
```python
feature_selection_method: str = "mda"  # "mda", "mdi", "hybrid", "none"
feature_selection_n_features: int = 0  # 0 = use max_features
```

### Per-Model Defaults (Designed)
| Model | Method | N Features | Rationale |
|-------|--------|------------|-----------|
| XGBoost, LightGBM, CatBoost | mda | 50 | Tree-based benefit from MDA |
| LSTM, GRU | hybrid | 43 | Sequence models need stable features |
| TCN | hybrid | 60 | Longer sequences, more features |
| Transformer | none | 0 | Learns from raw features |
| PatchTST, iTransformer | mda | 30 | Patch/channel attention on selected features |
| TFT, N-BEATS | mda | 30-40 | Variable selection built-in |
| InceptionTime, ResNet1D | mda | 50 | CNN feature detectors |
| Random Forest | mda | 50 | Tree-based ensemble |
| Logistic, SVM | mda | 30 | Linear models need fewer features |
| Ensembles/Meta-learners | none | 0 | Use base model outputs |

### Remaining Work
1. Add feature_selection_method and feature_selection_n_features to all 23 model entries in `MODEL_DATA_REQUIREMENTS` (1-2 hours)
2. Update Trainer to read model-specific feature selection config (1 hour)
3. Add tests for per-model feature selection (1 hour)
4. Update documentation (1 hour)

**Total: 4-5 hours**

---

## ⏳ Remaining Tasks (5 tasks - ~25-30 hours)

### Task 9: Feature Optimization System (8-10 hours)
**Goal:** Automated feature selection optimization per model family

**Design:**
```python
# src/feature_selection/optimization.py
class FeatureOptimizer:
    def optimize_for_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        method: str = "optuna",  # or "genetic"
        n_trials: int = 100,
    ) -> list[str]:
        """Return optimized feature list for model."""
        ...
```

**Integration Points:**
- Use during cross-validation
- Cache results per (model, data, horizon) tuple
- Store in `experiments/feature_optimization/{model_name}/`

### Task 10: P0 - Lineage Unification (1-2 days)
**Goal:** Link training runs to pipeline artifacts

**Implementation:**
1. Add `pipeline_run_id` field to TrainerConfig
2. Add `data_checksums` to pipeline output
3. Validate checksums at training time
4. Store lineage in model artifacts

**Files to Create/Modify:**
- `src/phase1/pipeline_runner.py` - Add checksum computation
- `src/models/training/trainer.py` - Add lineage validation
- `src/models/artifacts.py` - Store lineage metadata

### Task 11: P0 - Timestamp Alignment Checks (1-2 days)
**Goal:** Ensure datetime alignment for heterogeneous stacking

**Implementation:**
1. Add datetime alignment validation in stacking dataset builder
2. Validate timestamp consistency across base model predictions
3. Add tests for timestamp alignment edge cases

**Files to Create/Modify:**
- `src/cross_validation/oof_stacking.py` - Add datetime validation
- `src/ensemble/stacking.py` - Enforce datetime matching
- `tests/ensemble/test_timestamp_alignment.py` - New test file

### Task 12: P0 - Standardized Evaluation Reports (4 hours)
**Goal:** Unified JSON + markdown artifacts per model run

**Implementation:**
1. Create evaluation report schema (JSON format)
2. Generate markdown summary from JSON
3. Save reports to `experiments/runs/{run_id}/reports/`

**Files to Create:**
- `src/models/evaluation/report_schema.py`
- `src/models/evaluation/report_generator.py`
- `templates/model_report.md.jinja2`

### Task 13: Documentation Updates (4 hours)
**Goal:** Update all docs with new config architecture

**Files to Update:**
- `docs/ARCHITECTURE.md` - Add global config section
- `docs/implementation/CONFIG_ARCHITECTURE.md` - New file
- `README.md` - Update configuration examples
- `CLAUDE.md` - Update config instructions

---

## Key Achievements

### Zero Hardcoded Defaults ✅
All business logic reads from `config/global.yaml`:
- Random seed
- Train/val/test splits
- Purge/embargo parameters
- All horizon settings
- Technical indicator periods
- Feature selection settings
- Training hyperparameters
- MTF configuration
- Optimization settings
- OOM recovery settings

### Single Source of Truth ✅
No more competing config definitions:
- `config/global.yaml` - Master config
- Python code uses `get_global_config()` 
- All defaults flow from one place

### Backward Compatible ✅
Existing code continues to work:
- Legacy constants still available
- Gradual migration supported
- No breaking changes

### Test Coverage ✅
- 127 config tests passing
- 21 global config tests
- 106 integration tests

---

## Next Steps (Priority Order)

1. **Complete Task 8** (4-5 hours) - Per-model feature selection
   - Add fields to all MODEL_DATA_REQUIREMENTS entries
   - Update Trainer to use model-specific config
   - Add tests

2. **Task 12** (4 hours) - Standardized reports
   - Quick win, high visibility
   - Improves experiment tracking

3. **Task 10** (1-2 days) - Lineage unification
   - P0 for reproducibility
   - Enables artifact tracing

4. **Task 11** (1-2 days) - Timestamp alignment
   - P0 for heterogeneous stacking
   - Prevents subtle bugs

5. **Task 9** (8-10 hours) - Feature optimization
   - Nice-to-have for now
   - Can be manual initially

6. **Task 13** (4 hours) - Documentation
   - Final polish

---

## Effort Summary

| Phase | Hours | Status |
|-------|-------|--------|
| Phase 1-3: Config Centralization | 8-11 | ✅ Complete |
| Task 8: Per-Model Features | 4-5 | 🚧 In Progress |
| Task 9: Feature Optimization | 8-10 | ⏳ Pending |
| Task 10: Lineage | 8-16 | ⏳ Pending |
| Task 11: Timestamps | 8-16 | ⏳ Pending |
| Task 12: Reports | 4 | ⏳ Pending |
| Task 13: Docs | 4 | ⏳ Pending |
| **Total** | **44-66 hours** | **~15% complete** |

---

## Recommendations

### Short Term (Next Session)
1. Finish Task 8 (per-model feature selection) - 4-5 hours
2. Implement Task 12 (standardized reports) - 4 hours
3. Run full test suite to validate

**Total: 8-9 hours**

### Medium Term (Next 2-3 Days)
4. Implement Task 10 (lineage unification) - 1-2 days
5. Implement Task 11 (timestamp alignment) - 1-2 days

**Total: 2-4 days**

### Long Term (Future)
6. Implement Task 9 (feature optimization) - 8-10 hours
7. Update documentation - 4 hours

**Total: 12-14 hours**

---

## Impact Assessment

### What's Working Now ✅
- All config defaults centralized in YAML
- PipelineConfig reads from GlobalConfig
- TrainerConfig reads from GlobalConfig
- Backward compatibility maintained
- 127 tests passing

### What's Still Needed ❌
- Per-model feature selection (in progress)
- Feature optimization framework
- Lineage tracking
- Timestamp alignment validation
- Standardized evaluation reports
- Updated documentation

### Risk Assessment
**Low Risk:**
- Config system is stable and tested
- Changes are additive, not breaking
- Backward compatibility maintained

**Medium Risk:**
- Feature optimization system is complex
- Lineage tracking touches many files
- Timestamp alignment needs careful testing

**Mitigation:**
- Incremental implementation
- Comprehensive testing at each step
- Keep existing code paths working

---

**Last Updated:** 2026-01-15 22:50 EST
**Next Review:** After Task 8 completion
