# ML Factory Command System

**Tiered commands for structured, verified work on ML Factory.**

---

## Command Matrix (Visual)

```
                        LIGHT              MEDIUM             HEAVY
                      (1 file)           (1 task)          (1 phase)
    ┌──────────────┬──────────────────┬──────────────────┬──────────────────┐
    │   ANALYZE    │  /analysis-      │  /analysis-      │  /analysis-      │
    │              │  targeted(1c)    │  optimization(1b)│  full(1a)        │
    ├──────────────┼──────────────────┼──────────────────┼──────────────────┤
    │   VERIFY     │  /verify-        │  /verify-        │  /verify-        │
    │              │  claim(2a)       │  batch(2b)       │  contracts(2c)   │
    ├──────────────┼──────────────────┼──────────────────┼──────────────────┤
    │   DOCS       │  /docs-          │  /docs-          │  /docs-          │
    │              │  tasks(3a)       │  full(3b)        │  final(6a)       │
    ├──────────────┼──────────────────┼──────────────────┼──────────────────┤
    │   EXECUTE    │  /execute-       │  /execute-       │  /execute-       │
    │              │  surgical(4c)    │  standard(4a)    │  large(4b)       │
    ├──────────────┼──────────────────┼──────────────────┼──────────────────┤
    │   CHECK      │  /check-         │  /check-         │  /check-         │
    │              │  standard(5a)    │  deep(5b)        │  behavior(5c)    │
    └──────────────┴──────────────────┴──────────────────┴──────────────────┘
```

**Shorthand commands:** `/analyze`, `/verify`, `/execute`, `/docs`, `/check` (auto-select tier)

---

## Root Docs & Command Interaction

```
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                           ROOT DOCUMENTS                                │
    ├───────────────┬───────────────┬───────────────┬─────────────────────────┤
    │  DIRECTION    │ CLEANUP_PLAN  │ CLEANUP_TASKS │      COMPLETION         │
    │   (vision)    │   (roadmap)   │   (details)   │       (archive)         │
    │               │               │               │                         │
    │  Architecture │  Phase list   │  File:line    │  Completed work         │
    │  Blockers     │  Diagrams     │  Checklists   │  Disproven claims       │
    │  Trajectory   │  "What & why" │  "Where & how"│  Lessons learned        │
    └───────┬───────┴───────┬───────┴───────┬───────┴────────────┬────────────┘
            │               │               │                    │
            │       ┌───────┴───────┐       │                    │
            │       │ Always sync   │       │                    │
            │       │   together    │       │                    │
            │       └───────────────┘       │                    │
            │                               │                    │
    ┌───────┴───────────────────────────────┴────────────────────┴────────────┐
    │                             COMMANDS                                     │
    ├─────────────────┬─────────────────┬─────────────────┬────────────────────┤
    │   /analysis-*   │   /execute-*    │    /docs-*      │     /verify-*      │
    │                 │                 │                 │                    │
    │  Reads: ALL     │  Reads: TASKS   │  Updates: ALL   │  Checks:           │
    │                 │  Updates: PLAN  │                 │  COMPLETION first  │
    │                 │  + TASKS        │                 │                    │
    └─────────────────┴─────────────────┴─────────────────┴────────────────────┘
```

---

## Subagent Architecture

```
    COMMAND ──────────────► SUBAGENTS (isolated context)
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
           ▼                       ▼                       ▼
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │ codebase-       │   │ code-reviewer   │   │ Explore         │
    │ analyzer        │   │ contract-       │   │ (built-in)      │
    │                 │   │ verifier        │   │                 │
    │ Model: Opus     │   │ integration-    │   │ Modes:          │
    │                 │   │ checker         │   │ - quick         │
    │ Purpose:        │   │ doc-updater     │   │ - medium        │
    │ - Dead code     │   │                 │   │ - very thorough │
    │ - Performance   │   │ Model: Sonnet   │   │                 │
    │ - Architecture  │   │                 │   │                 │
    └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
             │                     │                     │
             └─────────────────────┴─────────────────────┘
                                   │
                                   ▼
                    Summary returned to main context
                         (context preserved)
```

### Subagent Reference

| Agent | Model | Tools | Purpose |
|-------|-------|-------|---------|
| `codebase-analyzer` | Opus | Read, Grep, Glob, Bash | Dead code, performance, architecture analysis |
| `code-reviewer` | Sonnet | Read, Grep, Glob | CLAUDE.md standards compliance |
| `contract-verifier` | Sonnet | Read, Grep, Glob | Type/schema validation |
| `integration-checker` | Sonnet | Read, Grep, Glob, Bash | Import/dependency analysis |
| `doc-updater` | Sonnet | Read, Edit, Write | Root document updates |

---

## Standard Workflows

### New Phase Work

```
┌────────────────────────────────────────────────────────────────────────────┐
│ 1. ANALYZE                                                                 │
│    /analysis-full(1a) [phase description]                                  │
│    - Reads all 4 root docs                                                 │
│    - Spawns 6 agents for comprehensive analysis                            │
│    - Updates CLEANUP_PLAN.md and CLEANUP_TASKS.md with plan                │
├────────────────────────────────────────────────────────────────────────────┤
│ 2. EXECUTE                                                                 │
│    /execute-large(4b) [phase]                                              │
│    - Reads CLEANUP_PLAN.md + CLEANUP_TASKS.md                              │
│    - Implements in phases with verification                                │
│    - Runs ruff, verifies imports                                           │
│    - Updates docs via doc-updater                                          │
├────────────────────────────────────────────────────────────────────────────┤
│ 3. VERIFY                                                                  │
│    /check-deep(5b) [what was changed]                                      │
│    - 4-way verification (code, contracts, integration, runtime)            │
│    - Returns PASS/FAIL per category                                        │
├────────────────────────────────────────────────────────────────────────────┤
│ 4. CLOSE OUT                                                               │
│    /docs-final(6a) [phase]                                                 │
│    - Archives to COMPLETION.md                                             │
│    - Removes completed items from PLAN + TASKS                             │
│    - Updates DIRECTION.md if architecture changed                          │
└────────────────────────────────────────────────────────────────────────────┘
```

### Quick Fix

```
/execute-surgical(4c) [description]
```

Single focused change, minimal scope, auto-updates docs.

### Investigate an Issue

```
1. /verify-claim(2a) [claim]           # First: is this even real?
2. /analysis-targeted(1c) [issue]      # Understand dependencies
3. /execute-surgical(4c) [fix]         # Fix it
4. /check-standard(5a) [area]          # Verify fix
```

---

## Command Quick Reference

### Analyze Commands

| Command | Agents | Reads | Purpose |
|---------|--------|-------|---------|
| `/analysis-targeted(1c)` | 1 | TASKS, COMPLETION | Focused investigation of specific item |
| `/analysis-optimization(1b)` | 3 | DIRECTION, PLAN | Find improvements ranked by impact/effort |
| `/analysis-full(1a)` | 6 | ALL | Comprehensive analysis, creates plan |

### Verify Commands

| Command | Agents | Reads | Purpose |
|---------|--------|-------|---------|
| `/verify-claim(2a)` | 1 | COMPLETION | Single claim verification |
| `/verify-batch(2b)` | 3 | COMPLETION | Multiple items, parallel verification |
| `/verify-contracts(2c)` | 3 | DIRECTION | Deep contract compliance check |

### Docs Commands

| Command | Agents | Updates | Purpose |
|---------|--------|---------|---------|
| `/docs-tasks(3a)` | 1 | PLAN, TASKS | Quick task status update |
| `/docs-full(3b)` | 1 | ALL | Update all 4 docs |
| `/docs-final(6a)` | 1 | ALL | Close out phase, archive to COMPLETION |

### Execute Commands

| Command | Agents | Reads | Purpose |
|---------|--------|-------|---------|
| `/execute-surgical(4c)` | 1 | TASKS | Single focused fix |
| `/execute-standard(4a)` | 1 | TASKS | Normal task with verification |
| `/execute-large(4b)` | 1+ | PLAN, TASKS | Phase-wide implementation |

### Check Commands

| Command | Agents | Purpose |
|---------|--------|---------|
| `/check-standard(5a)` | 1 | Basic validation (ruff, imports) |
| `/check-deep(5b)` | 4 | Comprehensive 4-way check |
| `/check-behavior(5c)` | 2 | Execution tracing, edge cases |

---

## Tier Selection Guide

| Scope | Tier | Example Commands |
|-------|------|------------------|
| **1 file, 1 fix** | Light | `/execute-surgical(4c)`, `/verify-claim(2a)` |
| **1 task, 2-5 files** | Medium | `/execute-standard(4a)`, `/verify-batch(2b)` |
| **1 phase, 10+ files** | Heavy | `/execute-large(4b)`, `/analysis-full(1a)` |

---

## Project Rules

| Rule | Rationale |
|------|-----------|
| **Check COMPLETION.md first** | Many claims already disproven |
| **PLAN + TASKS sync together** | They mirror each other |
| **Delete, don't adapt** | No compatibility layers |
| **Verify before delete** | Use `/verify-claim` first |
| **Run linters always** | `ruff check src/`, `black src/` |

---

## Anti-Patterns

| Don't | Do Instead |
|-------|------------|
| Use heavy command for 1 fix | `/execute-surgical(4c)` |
| Update PLAN without TASKS | Always sync both |
| Skip COMPLETION.md check | Check first, avoid rework |
| Investigate without verifying | `/verify-claim(2a)` first |
| Skip post-change verification | Run `/check-standard(5a)` |

---

## Verification Commands (Manual)

```bash
# Linting (required)
ruff check src/
ruff check src/ --fix

# Formatting (required)
black src/
black --check src/

# Import checks
python -c "from src.core.types import DataRank, ModelFamily; print('OK')"
python -c "from src.core.contracts import get_model_contract; print('OK')"
python -c "from src.data.adapters import get_adapter; print('OK')"

# Single definition checks (should each return 1)
grep -r "class DataRank" src/ --include="*.py" | wc -l
grep -r "class ModelFamily" src/ --include="*.py" | wc -l
```

---

*See CLAUDE.md for project standards and workflow patterns.*
*See .claude/commands/ for individual command implementations.*
*See .claude/agents/ for subagent definitions.*
