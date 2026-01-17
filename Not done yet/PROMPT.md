# ML Factory Refactoring - Continuation Prompt

## Context

Hey Claude. I need you to analyze `src/` and my pipeline, then read the "Not done yet" folder. We were in the middle of creating a refactoring plan for my SRC PIPELINE based on `plan.md`. The problem is we ran out of credits.

After you understand the SRC PIPELINE, I need you to finish the implementation plan documents on how to specifically rework our pipeline so it's much smoother and everything flows, but also write high-level architecture docs that AI can easily ingest and consume.

**IMPORTANT:** We are taking ALL the features from `src/` and organizing the codebase for an ML Factory. We are NOT deleting anything - we are just refactoring. We are also NOT allowing any legacy code or backwards compatibility.

---

## Current Status

### Completed

| Document | Description |
|----------|-------------|
| `plan.md` | Master plan with feature inventory (22-23 models, 150+ features, training modes, CV methods, etc.) |
| `PHASE_0_FOUNDATION.md` | Core interfaces, types, constants, validation |
| `PHASE_1_UNIFIED_FEATURES.md` | FeatureRegistry with all 162 features across 12 families + model strategies |

### Missing (per plan.md)

- [ ] No high-level docs finished
- [ ] `PHASE_2_ADAPTER_INTEGRATION.md` (Not finished)
- [ ] `PHASE_3_TRAINING_ORCHESTRATION.md`
- [ ] `PHASE_4_META_LEARNERS.md`
- [ ] `PHASE_5_INFERENCE.md`

---

## Remaining Phases to Write

### 1. PHASE_3_TRAINING_ORCHESTRATION.md

Single entry point for all training. Should cover:

- `TrainingOrchestrator` class
- Integration of 3 training modes (standard, walk-forward, regime-aware, meta-labeling)
- CV methods (PurgedKFold, CPCV, PBO)
- OOF generation for tabular and sequence models
- Unified interface: `orchestrator.train(models=[...], mode="walk_forward")`

### 2. PHASE_4_META_LEARNERS.md

Heterogeneous ensemble + OOF alignment. Should cover:

- `HeterogeneousStackingBuilder`
- OOF alignment across different adapters (2D/3D/4D)
- 4 meta-learners: `ridge_meta`, `mlp_meta`, `xgboost_meta`, `calibrated_meta`
- `OOFCache` and `OOFAlignmentValidator`
- Training flow from heterogeneous bases to single meta-learner

### 3. PHASE_5_INFERENCE.md

Auto bundle creation + meta-learner inference. Should cover:

- `ModelBundle` with `PreprocessingGraph` for feature lineage
- `InferencePipeline` for single/ensemble predictions
- `BatchPredictor` for chunked processing
- Inference from raw OHLCV to prediction (no manual feature engineering at inference time)
- Bundle serialization/deserialization

---

## Format Requirements

Use the same detailed format as the existing phase docs:

- Code snippets with full implementations
- Implementation checklists with `- [ ]` items
- Data flow diagrams using ASCII art
- Reference the feature inventory in `plan.md`
- ALSO WRITE HIGH LEVEL docs to accompany each plan, feel free to expand the current documentation in Not done yet