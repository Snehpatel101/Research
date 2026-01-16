# SNwH Implementation Plan

**Unified Multi-Timeframe Model Factory**

This folder contains the complete implementation plan to unify the ML pipeline so that:
- Every model works with every other model and every timeframe
- Heterogeneous ensembles (mixing tabular + neural + transformer) work by default
- All features are computed correctly, consistently, and without leakage
- The meta-learner inference layer is correct and integrates cleanly

---

## Quick Start

1. **Start here:** [00_INDEX.md](00_INDEX.md) - Master index with roadmap
2. **Understand gaps:** [SNWH_ARCHITECTURE_SYNTHESIS.md](SNWH_ARCHITECTURE_SYNTHESIS.md) - Gap analysis
3. **Implement phases:** Phase 0 → 1 → 2 → 3 → 4 → 5 (in order)
4. **Validate:** [SNWH_TESTING_STRATEGY.md](SNWH_TESTING_STRATEGY.md) - Test each phase

---

## Documents

| Document | Lines | Description |
|----------|-------|-------------|
| [00_INDEX.md](00_INDEX.md) | 327 | Master index - start here |
| [SNWH_ARCHITECTURE_SYNTHESIS.md](SNWH_ARCHITECTURE_SYNTHESIS.md) | 558 | Gap analysis, dependency graph, priority matrix |
| [SNWH_IMPLEMENTATION_PHASE_0.md](SNWH_IMPLEMENTATION_PHASE_0.md) | 1,305 | Canonical Contracts (DataContract, ModelContract, ArtifactManifest) |
| [SNWH_IMPLEMENTATION_PHASE_1.md](SNWH_IMPLEMENTATION_PHASE_1.md) | 704 | Configuration Layer (TrainerConfig, PerModelConfig, EnsemblePlan) |
| [SNWH_IMPLEMENTATION_PHASE_2.md](SNWH_IMPLEMENTATION_PHASE_2.md) | 1,091 | Adapter Architecture (TabularAdapter, SequenceAdapter, MultiStreamAdapter) |
| [SNWH_IMPLEMENTATION_PHASE_3.md](SNWH_IMPLEMENTATION_PHASE_3.md) | 907 | Timeframe Coordination (TimeframeCoordinator, per-model TF loading) |
| [SNWH_IMPLEMENTATION_PHASE_4.md](SNWH_IMPLEMENTATION_PHASE_4.md) | 691 | OOF Integrity (coverage alignment, heterogeneous stacking) |
| [SNWH_IMPLEMENTATION_PHASE_5.md](SNWH_IMPLEMENTATION_PHASE_5.md) | 860 | Feature Strategy Integration (MODEL_FEATURE_STRATEGIES wiring) |
| [SNWH_IMPLEMENTATION_SUMMARY.md](SNWH_IMPLEMENTATION_SUMMARY.md) | 295 | File listings, migration path, verification checklist |
| [SNWH_TESTING_STRATEGY.md](SNWH_TESTING_STRATEGY.md) | 2,294 | 28 test files, 82 test classes, 305 test methods |
| [SNWH_IMPLEMENTATION_PHASE_6.md](SNWH_IMPLEMENTATION_PHASE_6.md) | ~400 | Single Config System (delete 40+ YAML, 89% code reduction) |

**Total:** 12 documents, ~9,500 lines, ~300KB

---

## Critical Gaps to Fix

| Priority | Gap | Location | Impact |
|----------|-----|----------|--------|
| **P1** | Trainer loads single TF for all models | `trainer.py:314-317` | Blocks heterogeneous ensembles |
| **P1** | OOF coverage mismatch | `oof_sequence.py:194-206` | Breaks stacking alignment |
| **P2** | TrainerConfig missing per-model TF fields | `trainer_config.py:27-102` | Cannot configure per-model timeframes |
| **P2** | Default MTF = 7 TFs (not 9) | `constants.py:42-50` | Missing 1min, 5min |

---

## Implementation Summary

### Files to Create (15 new files)

| Package | Files | Phase |
|---------|-------|-------|
| `src/contracts/` | `__init__.py`, `data_contract.py`, `model_contract.py`, `artifact_manifest.py` | 0 |
| `src/adapters/` | `__init__.py`, `base.py`, `registry.py`, `tabular.py`, `sequence.py`, `multi_stream.py` | 2 |
| `src/coordination/` | `__init__.py`, `timeframe_coordinator.py`, `alignment.py` | 3 |
| `src/cross_validation/` | `oof_alignment.py` | 4 |
| `src/features/` | `strategy_manager.py`, `optimization.py` | 5 |

### Files to Modify (10 existing files)

| File | Phase | Changes |
|------|-------|---------|
| `src/models/config/trainer_config.py` | 1 | +8 fields for per-model config |
| `src/models/config/data_requirements.py` | 1 | +6 fields to ModelDataRequirements |
| `src/config/unified.py` | 1 | +ModelConfigSection |
| `src/models/training/trainer.py` | 3 | +_load_data_for_model() |
| `src/phase1/stages/mtf/constants.py` | 3 | 7→9 TFs |
| `src/cross_validation/oof_sequence.py` | 4 | +strict validation |
| `src/cross_validation/oof_stacking.py` | 4 | +HeterogeneousStackingBuilder |
| `src/cross_validation/oof_core.py` | 4 | +alignment metadata |
| `src/models/training/features.py` | 5 | +_get_strategy_features() |
| `src/features/strategies.py` | 5 | Sync with model YAMLs |

---

## Testing Summary

| Category | Files | Classes | Methods |
|----------|-------|---------|---------|
| Unit Tests | 15 | 45 | 180 |
| Integration Tests | 5 | 15 | 45 |
| Regression Tests | 4 | 12 | 40 |
| Property Tests | 4 | 10 | 40 |
| **Total** | **28** | **82** | **305** |

---

## Sprint Plan

| Week | Phases | Focus |
|------|--------|-------|
| 1 | 0, 1 | Foundation - contracts and config |
| 2 | 2, 3 | Data routing - adapters and timeframe coordination |
| 3 | 4, 5 | OOF integrity and feature strategy |
| 4 | 6, 7, 8 | Validation, testing, documentation |

---

## Models Supported

All 23 models will work with any timeframe and can be combined in ensembles:

**Tabular (6):** XGBoost, LightGBM, CatBoost, Random Forest, Logistic, SVM

**Neural (10):** LSTM, GRU, TCN, Transformer, InceptionTime, ResNet1D, N-BEATS, PatchTST, iTransformer, TFT

**Ensemble (3):** Voting, Stacking, Blending

**Meta-Learners (4):** Ridge Meta, MLP Meta, Calibrated Meta, XGBoost Meta

---

## Success Criteria

After implementation:
- [ ] Every model has a ModelContract with input_rank, primary_timeframe, mtf_mode
- [ ] Adapters correctly route 2D/3D/4D data to models
- [ ] Heterogeneous stacking (XGBoost + LSTM + PatchTST) trains successfully
- [ ] OOF coverage aligned across tabular and sequence models
- [ ] Per-model feature strategies applied automatically
- [ ] All 23 models work with all 9 timeframes

---

*Generated: 2026-01-16*
