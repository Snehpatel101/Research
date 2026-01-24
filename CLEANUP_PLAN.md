# Cleanup Plan: ML Factory - Remaining Work

**Status:** Phases 0-6 Complete | All Models Implemented
**Last Updated:** 2026-01-24

---

## Completed Phases (see COMPLETION.md)

| Phase | Description | Lines Changed |
|-------|-------------|---------------|
| 0 | Deduplication | -5,336 |
| 1 | Contract Enforcement | +616 |
| 2 | 4D Infrastructure | +958 |
| 3 | 5-Dimension Optuna | +2,298 |
| 4 | Validation Integration | +50 |
| 5 | Unified Entry Point | +1,281 |
| 6 | Advanced Models | +3,690 |

---

## Remaining Work

### Priority 1: Advanced Models (6 total) - COMPLETE

**Status:** All 6 models implemented and registered
**Completed:** 2026-01-24

| Model | Family | Data Rank | Adapter | Status |
|-------|--------|-----------|---------|--------|
| InceptionTime | CNN | 3D | Sequence | ✅ |
| 1D ResNet | CNN | 3D | Sequence | ✅ |
| PatchTST | Transformer | 4D | MultiStream | ✅ |
| iTransformer | Transformer | 4D | MultiStream | ✅ |
| TFT | Transformer | 3D | Sequence | ✅ |
| N-BEATS | MLP | 3D | Sequence | ✅ |

**Implementation Location:** `src/models/neural/`

---

### Priority 2: Deferred Items (Low Priority)

| Item | Description | Original Phase |
|------|-------------|----------------|
| 5C | Unified deployment bundle (tar.gz) | Phase 5 |
| 4C | Ensemble diversity analysis integration | Phase 4 |
| 4D | Deflated Sharpe Ratio post-Optuna | Phase 4 |
| 4E | Bootstrap CIs in financial reports | Phase 4 |
| 4F | Auto calibration in orchestrator | Phase 4 |
| 4G | Bet sizing connection | Phase 4 |
| - | MTF ablation flag | Phase 3 |

---

## Execution Protocol

### For Each Model:

```bash
# 1. Implement model at src/models/neural/<model_name>.py

# 2. Verify import works
python -c "from src.models.neural.<model_name> import <ModelClass>; print('OK')"

# 3. Verify contract registered
python -c "from src.core.contracts import get_model_contract; print(get_model_contract('<model_name>'))"

# 4. Lint
ruff check src/models/neural/<model_name>.py
black src/models/neural/<model_name>.py
```

### After Each Model - UPDATE DOCS:
1. Mark model ✅ in CLEANUP_PLAN.md table above
2. Mark model ✅ in CLEANUP_TASKS.md
3. After all 6 done, add "Advanced Models" entry to COMPLETION.md

### Order:
1. InceptionTime → 2. 1D ResNet (3D)
3. PatchTST → 4. iTransformer → 5. TFT → 6. N-BEATS (4D)

---

*For completed phase details, see COMPLETION.md*
