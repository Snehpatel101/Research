# ML Factory - Remaining Tasks

**Last Updated:** 2026-01-24
**Status:** Phases 0-6 Complete | All Models Implemented

---

## Completed (see COMPLETION.md)

| Phase | Impact |
|-------|--------|
| Phase 0 | -5,336 lines (deduplication) |
| Phase 1 | +616 lines (contract enforcement) |
| Phase 2 | +958 lines (4D infrastructure) |
| Phase 3 | +2,298 lines (5D Optuna) |
| Phase 4 | +50 lines (validation wiring) |
| Phase 5 | +1,281 lines (MLFactory + ExperimentConfig) |
| Phase 6 | +3,690 lines (6 advanced neural models) |

---

## Completed: Advanced Models (Phase 6)

| Model | Status |
|-------|--------|
| InceptionTime | ✅ |
| 1D ResNet | ✅ |
| PatchTST | ✅ |
| iTransformer | ✅ |
| TFT | ✅ |
| N-BEATS | ✅ |

---

### 3D Models (Sequence Adapter)

#### InceptionTime ✅
**Location:** `src/models/neural/inceptiontime_model.py`
**Contract:** `DataRank.SEQUENCE_3D`, `FeatureMode.ENGINEERED`
**Status:** Implemented (~500 lines)

#### 1D ResNet ✅
**Location:** `src/models/neural/resnet1d_model.py`
**Contract:** `DataRank.SEQUENCE_3D`, `FeatureMode.ENGINEERED`
**Status:** Implemented (~550 lines)

---

### 4D Models (MultiStream Adapter)

#### PatchTST ✅
**Location:** `src/models/neural/patchtst_model.py`
**Contract:** `DataRank.MULTI_TF_4D`, `FeatureMode.RAW`
**Status:** Implemented (~480 lines)

#### iTransformer ✅
**Location:** `src/models/neural/itransformer_model.py`
**Contract:** `DataRank.MULTI_TF_4D`, `FeatureMode.RAW`
**Status:** Implemented (~620 lines)

### 3D Models (Sequence Adapter) - Continued

#### TFT (Temporal Fusion Transformer) ✅
**Location:** `src/models/neural/tft_model.py`
**Contract:** `DataRank.SEQUENCE_3D`, `FeatureMode.ENGINEERED`
**Status:** Implemented (~780 lines)

#### N-BEATS ✅
**Location:** `src/models/neural/nbeats_model.py`
**Contract:** `DataRank.SEQUENCE_3D`, `FeatureMode.ENGINEERED`
**Status:** Implemented (~760 lines)

---

## Deferred (Low Priority)

| Task | Description |
|------|-------------|
| 5C | Unified deployment bundle (tar.gz format) |
| 4C | Ensemble diversity analysis integration |
| 4D | Deflated Sharpe Ratio post-Optuna validation |
| 4E | Bootstrap CIs in financial reports |
| 4F | Auto calibration in orchestrator |
| 4G | Bet sizing connection to backtest |
| - | MTF ablation flag |

---

## After Each Model - REQUIRED

### 1. Verify
```bash
python -c "from src.models.neural.<model> import <ModelClass>; print('OK')"
python -c "from src.core.contracts import get_model_contract; print(get_model_contract('<model>'))"
ruff check src/models/neural/<model>.py
black src/models/neural/<model>.py
```

### 2. Update Docs
1. Change ⬜ to ✅ in table above
2. Change ⬜ to ✅ in model header
3. Change ⬜ to ✅ in CLEANUP_PLAN.md table
4. After all 6 done → add "Advanced Models" to COMPLETION.md

---

*For completed phase details, see COMPLETION.md*
