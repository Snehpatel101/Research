# Serena Knowledge Base - ML Factory Documentation

This directory contains semantic knowledge base files for the ML Model Factory project. These files support Serena's code search and understanding capabilities.

---

## Master Plan (START HERE)

### `ml_factory_master_plan.md` ⭐ PRIMARY REFERENCE
**Purpose:** Complete unified pipeline implementation plan with all architectural decisions.

**Contents:**
- 16-stage pipeline architecture with Optuna optimization
- 4 Optuna loops: Label (100 trials), Feature Selection (100), Pruning (50), Hyperparams (2,300)
- 23 models across 6 families (Boosting, Classical, Neural Basic, Neural Advanced, Ensemble, Meta)
- Migration map: Current → Proposed locations
- Dependency tree: Stage inputs/outputs/configs
- Implementation priorities and success criteria

**Use when:** Understanding the complete plan, implementing new stages, reviewing architecture.

**Related Docs:**
- `docs/DEPENDENCY_TREE.md` - Full ASCII dependency tree
- `docs/implementation/UNIFIED_PIPELINE_ARCHITECTURE.md` - Detailed stage specifications
- `config/optimization/*.yaml` - Optuna configuration files

---

## Knowledge Base Files

### 1. `pipeline_implementation_status.md`
**Purpose:** Documents current implementation status, gaps, and priority tasks.

**Contents:**
- What works (Phases 1-7 status)
- What's missing (4 timeframes, multi-res adapter, 6 models, meta-learners)
- Priority tasks with effort estimates
- Implementation sequence and milestones

**Use when:** Understanding what's implemented vs. planned, prioritizing work.

---

### 2. `architecture_target.md`
**Purpose:** Defines target architecture and ONE unified pipeline principle.

**Contents:**
- Core architectural principle (ONE pipeline, not separate pipelines)
- Data flow (Phases 1-8)
- Model-family adapters (2D, 3D, 4D)
- Ensemble compatibility rules
- Extension points (adding models, features, timeframes)

**Use when:** Understanding system design, architectural decisions, adding new components.

---

### 3. `unified_pipeline_architecture.md`
**Purpose:** Deep dive into unified pipeline design and adapter pattern.

**Contents:**
- Single data flow vs. separate pipelines (anti-patterns)
- Adapter design principles (deterministic, shape-only transformations)
- Leakage prevention mechanisms (purge, embargo, shift(1), train-only scaling)
- Common misconceptions
- Design decisions and trade-offs

**Use when:** Understanding why adapters exist, how to implement new adapters, debugging data flow.

---

### 4. `mtf_strategies_clarification.md`
**Purpose:** Clarifies MTF terminology and dispels "strategy" misconception.

**Contents:**
- MTF is a capability, not a strategy
- Automatic MTF pipeline flow (Phases 2-5)
- Model-specific data consumption (2D, 3D, 4D)
- Deprecated "strategy" terminology
- MTF implementation gaps

**Use when:** Understanding MTF architecture, explaining to new contributors, planning MTF extensions.

---

## Related Files in `.serena/memories/`

### `project_overview.md`
High-level project summary with architecture overview, model families, and key parameters.

**Updated:** 2026-01-01 (aligned with unified pipeline architecture)

### `code_style_conventions.md`
Code formatting, naming conventions, docstrings, type hints, error handling.

### `critical_bugs.md`
Known bugs and issues (HMM lookahead bias, GA leakage, transaction costs).

### `editing_guidelines.md`
Guidelines for code editing, refactoring, and file organization.

---

## Knowledge Base Organization

```
.serena/
├── knowledge/              # Semantic knowledge base (architecture, design)
│   ├── README.md           # This file
│   ├── pipeline_implementation_status.md
│   ├── architecture_target.md
│   ├── unified_pipeline_architecture.md
│   └── mtf_strategies_clarification.md
│
├── memories/               # Project configuration and conventions
│   ├── project_overview.md
│   ├── code_style_conventions.md
│   ├── critical_bugs.md
│   ├── editing_guidelines.md
│   ├── planning_mode_guidelines.md
│   ├── sse_bridge_setup.md
│   ├── suggested_commands.md
│   └── task_completion_checklist.md
│
└── project.yml             # Serena configuration
```

---

## Quick Reference

### Key Architectural Principles
1. **ONE pipeline** - Single data flow, not separate pipelines
2. **Deterministic adapters** - Shape transformations only, no feature engineering
3. **Single source of truth** - Canonical dataset in `data/splits/scaled/`
4. **Leakage prevention** - Purge, embargo, shift(1), train-only scaling
5. **Plugin-based models** - Register via `@register` decorator

### 16-Stage Pipeline Architecture (V4.0)

```
PHASE A: DATA (Stages 1-6)           PHASE B: OPTUNA (Stages 7-9)
├── Stage 1: Ingestion               ├── Stage 7: Label Optimization (100 trials)
├── Stage 2: Cleaning                ├── Stage 8: Feature Selection (100 trials)
├── Stage 3: Sessions                └── Stage 9: Feature Pruning (50 trials)
├── Stage 4: MTF Upscaling
├── Stage 5: Features (180)          PHASE D: TRAINING (Stages 13-15)
└── Stage 6: Regime Detection        ├── Stage 13: Hyperparameter Opt (2,300 trials)
                                     ├── Stage 14: Training (23 models)
PHASE C: PREPROCESSING (10-12)       └── Stage 15: Stacking (Meta-learners)
├── Stage 10: Splits (70/15/15)
├── Stage 11: Scaling (RobustScaler) PHASE E: DEPLOYMENT (Stage 16)
└── Stage 12: Adaptation (2D/3D/4D)  └── Stage 16: Bundling (ModelBundle V1.1.0)
```

### Key Metrics
- **Total Optuna Trials:** 2,550
- **Total Models:** 23 (Boosting: 3, Classical: 3, Neural: 10, Ensemble: 3, Meta: 4)
- **Features:** 180 → 60-100 → 30-60 (after optimization)

### Implementation Priority (Next Steps)
1. **Create:** `src/pipeline/unified.py` - MLPipeline master orchestrator
2. **Create:** `src/pipeline/config.py` - MLConfig unified configuration
3. **Create:** `src/pipeline/state.py` - PipelineState management
4. **Create:** `src/pipeline/phases/*.py` - Stage wrappers

---

## Documentation Consistency

All knowledge base files align with:
- `docs/ARCHITECTURE.md` - Comprehensive architecture document
- `docs/CURRENT_VS_INTENDED_ARCHITECTURE.md` - Gap analysis
- `docs/archive/roadmaps/` - Implementation roadmaps
- `CLAUDE.md` - Project instructions for Claude Code

---

## Phase 1 Analysis Summary (2026-01-23)

The following deep analysis was performed on the codebase:

1. **Pipeline Architecture (Agent #1):** 12-stage orchestration system with clean run.py/core.py pattern (2/12 stages implement it)
2. **ML/Optimization (Agent #2):** Optuna TPE barrier optimization, 27% more sample-efficient than GA
3. **Configuration System (Agent #3):** Dual-layer config with 81+ classes, needs consolidation
4. **Data Store/Versioning (Agent #4):** 7,522 lines of provenance tracking infrastructure
5. **Features/Labeling (Agent #5):** 160+ features in 9 families with triple-barrier labeling
6. **Inference/Contracts (Agent #6):** Production-grade inference with 23 model contracts

Key findings documented in:
- `pipeline_implementation_status.md` - Updated with known issues
- `CONFIG_CONSOLIDATION_PLAN.md` - 81+ config classes to consolidate
- `TRAINER_CONSOLIDATION_PLAN.md` - 4 trainers to unify
- `ORCHESTRATOR_CONSOLIDATION_PLAN.md` - 5 orchestrators to 3

**Last Updated:** 2026-01-23
