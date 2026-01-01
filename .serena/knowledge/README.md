# Serena Knowledge Base - ML Factory Documentation

This directory contains semantic knowledge base files for the ML Model Factory project. These files support Serena's code search and understanding capabilities.

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

### Current Implementation Status
- **Phase 1 (Ingestion):** ✅ Complete
- **Phase 2 (MTF):** ⚠️ 5 of 9 timeframes
- **Phase 3 (Features):** ✅ Complete (~180 features)
- **Phase 4 (Labeling):** ✅ Complete
- **Phase 5 (Adapters):** ⚠️ 2D/3D done, 4D missing
- **Phase 6 (Models):** ⚠️ 13 of 19 models
- **Phase 7 (Cross-Val):** ✅ Complete
- **Phase 8 (Meta-Learners):** ❌ Not started

### Priority Tasks (Next 4-5 weeks)
1. Complete 9-timeframe MTF ladder (1-2 days)
2. Implement multi-resolution adapter (3 days)
3. Add advanced models (14-18 days)
4. Build meta-learners (5-7 days)

---

## Documentation Consistency

All knowledge base files align with:
- `docs/ARCHITECTURE.md` - Comprehensive architecture document
- `docs/CURRENT_VS_INTENDED_ARCHITECTURE.md` - Gap analysis
- `docs/archive/roadmaps/` - Implementation roadmaps
- `CLAUDE.md` - Project instructions for Claude Code

**Last Updated:** 2026-01-01
