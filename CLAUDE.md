# CLAUDE.md - ML Factory

**What this is:** Context file for AI assistants working on ML Factory.

---

## How We Work

1. **Read root docs first** - DIRECTION.md for vision, CLEANUP_PLAN.md for current phase
2. **Get user approval before changes** - Propose plans, wait for confirmation
3. **Update docs as you go** - Every change updates CLEANUP_TASKS and eventually COMPLETION
4. **Delete, don't adapt** - Remove duplicates rather than maintaining compatibility layers
5. **Clean code always** - Run linters, format code, no shortcuts

---

## The Four Root Documents

| Document | Purpose | Update Trigger |
|----------|---------|----------------|
| **DIRECTION.md** | Architecture vision, what we're building, blockers, trajectory | Major architectural decisions |
| **CLEANUP_PLAN.md** | Phase roadmap with architecture diagrams and rationale | Phase completes or priorities change |
| **CLEANUP_TASKS.md** | Same phases as PLAN but with specific file:line tasks | Starting/completing any task |
| **COMPLETION.md** | Running archive of all completed work | After each phase completes |

### CLEANUP_PLAN vs CLEANUP_TASKS

These two documents **mirror each other** - same phases, different detail levels:

| CLEANUP_PLAN.md | CLEANUP_TASKS.md |
|-----------------|------------------|
| Phase overview | Same phases |
| Architecture diagrams | Specific file locations |
| "What and why" | "Where and how" |
| Execution order | Task checklists |
| Validation criteria | Verification commands |

**Always update both together.** Plan changes → Tasks change.

### Update Flow

```
User Request
     ↓
Read DIRECTION.md + CLEANUP_PLAN.md (understand context)
     ↓
Propose approach → Get user approval
     ↓
Execute changes → Update CLEANUP_TASKS.md
     ↓
Phase complete → Move summary to COMPLETION.md
     ↓
If architecture changed → Update DIRECTION.md
```

### Rules

- **CLEANUP_PLAN.md and CLEANUP_TASKS.md update together** - They're mirrors
- **Check COMPLETION.md before investigating** - Many issues already resolved/disproven
- **DIRECTION.md changes require user approval** - It's the architectural source of truth

---

## What is ML Factory?

**ML Factory** = Config-driven system for building production ML ensembles for financial time-series prediction.

### The Goal

Put data in, get optimized trading model out. No data leakage, reproducible results, realistic financial metrics.

### Core Flow

```
Raw OHLCV → Pipeline (12 stages) → Features + Labels → Adapters → Models → Ensemble
```

### Key Guarantees

| Guarantee | How |
|-----------|-----|
| No data leakage | Purge/embargo in all CV splits |
| No lookahead | All MTF operations use `shift(1)` |
| Reproducible | Same config = same output |
| Realistic metrics | Transaction costs, slippage included |

### Model Support

All 12 models are production-ready:

| Category | Models |
|----------|--------|
| **Boosting** | XGBoost, LightGBM, CatBoost |
| **Neural RNN** | LSTM, GRU |
| **Neural CNN** | TCN, InceptionTime, 1D ResNet |
| **Transformer** | PatchTST, iTransformer, TFT |
| **MLP** | N-BEATS |

---

## Code Quality

**Clean code is always better.** Run linters before committing.

### Linting & Formatting

```bash
# Linting (required - must pass)
ruff check src/
ruff check src/ --fix  # Auto-fix what's possible

# Formatting (required)
black src/
black --check src/  # Check without modifying

# Type checking (informational - many false positives from stubs)
mypy src/ --ignore-missing-imports
```

### Standards

| Tool | Purpose | Config |
|------|---------|--------|
| **ruff** | Linting + import sorting | `pyproject.toml` |
| **black** | Code formatting | Default settings |
| **mypy** | Type checking | Ignore missing imports |

### Before Every Commit

1. `ruff check src/ --fix` - Fix linting issues
2. `black src/` - Format code
3. Import verification (see below)
4. No new pyright errors (existing stub issues OK)

### Clean Code Principles

- **Delete dead code** - Don't comment it out, delete it
- **One definition per concept** - No duplicates
- **Imports from canonical locations** - Re-export for compatibility
- **No magic numbers without context** - Use constants or document inline
- **Functions do one thing** - If it needs "and" to describe, split it

---

## Project Structure

```
src/
├── core/           # Types, contracts, base interfaces
├── data/           # Adapters, features, pipeline, labeling
├── models/         # All model implementations + training
├── optimization/   # Optuna, feature selection
├── validation/     # Leakage detection, lookahead audit, CV
├── inference/      # Backtesting, prediction
├── config/         # Configuration classes
└── cli/            # Command-line interface
```

### Canonical Locations

| Thing | Location |
|-------|----------|
| All enums/types | `src/core/types.py` |
| Model contracts | `src/core/contracts/` |
| Adapters | `src/data/adapters/` |
| Feature selection | `src/optimization/feature_selection/` |
| Validation | `src/validation/` |

---

## Current Status

**Phases 0-6: COMPLETE**
- Phase 0: Removed ~5,336 lines of duplicate code
- Phase 1: Contract enforcement with blocking validation
- Phase 2: 4D data infrastructure for transformers
- Phase 3: Enhanced adapter error handling
- Phase 4: Feature manifest with lineage tracking
- Phase 5: Performance optimizations
- Phase 6: Deprecation cleanup and orchestrator consolidation

**Phases 24-50: COMPLETE**
- See CLEANUP_PLAN.md and COMPLETION.md for full details

**Phases 51-52 (Phase 3 Master Plan): COMPLETE — 26/26 tasks**
- Phase 51: Deploy artifact system, single-call production inference, TrainerProtocol, adapter routing
- Phase 52: UniversalInferencePipeline, special mode bundles (WalkForward, Regime, MetaLabeling), safe_pickle_load migration (16 sites), neural architecture versioning

**Phase 53: COMPLETE — Security Hardening & SymbolConfig**
- Complete safe_pickle_load migration (0 joblib.load remaining, 36 safe sites)
- SymbolConfig standalone class (src/config/symbol.py) with MES/MGC/MNQ presets
- Explicit resample anti-lookahead params on all inference sites
- 12-model training smoke test: ALL PASS

**Phase 54: COMPLETE — E2E Pipeline Bug Fixes (5 bugs)**
- Trainer.save() added for model persistence
- Per-model feature selection (replaces global truncation that caused conflicts)
- 4D multi-stream data wired through MLFactory for PatchTST/iTransformer
- Timeframe key normalization (1h → 60min)
- Empty test split guard + date-range filtering for additional_dfs
- Optuna flags propagated when n_trials=0
- Full E2E verified: 10/12 models PASS (2 queued behind CPU time)

**Phase 55: COMPLETE — Deploy Manifest Fix**
- Bundle metadata `model_name` now correctly set (was "unknown" for boosting models)
- Deploy manifest `primary_model` selects best model by macro_f1
- Verified: all 6 bundles + manifest have correct model names

**Phase 56: COMPLETE — Backtest Pipeline Fix**
- Fixed `_extract_predictions()` — majority vote from AlignedOOFResult, proper OOFPrediction API
- Fixed timestamp column mismatch (datetime→timestamp rename for Backtester merge)
- Full verification: standard pipeline (7/7), backtest (5 trades), MTF (28 columns), Optuna (3 trials)

**Phase 57: COMPLETE — 4D OOF Generation (Cross-Family Ensembles)**
- Added `_generate_4d_oof()` to OOFGenerationService for transformer models (PatchTST, iTransformer, TFT)
- 4D data split by sample index using PurgedKFold (no re-windowing — samples already windowed)
- Enables cross-family ensembles: boosting (2D) + transformer (4D) working together
- Verified: xgboost+patchtst ensemble PASS, boosting-only regression PASS

**Phase 58: COMPLETE — Feature Selection Pipeline Overhaul**
- Wired low-variance and correlation pre-filters into orchestrator
- Per-model feature selection (respects model contract max_features)
- Features saved per-model to bundles

**Phase 59: COMPLETE — MDA Feature Ranking + Test Split Fix**
- Replaced variance ranking with MDA (permutation importance) in orchestrator
- MDA is target-aware: ranks features by predictive power, not just spread
- Fallback to variance if MDA fails (no labels, too few rows, CV error)
- Fixed test split crash: embargo_bars > remaining data caused KeyError
- Guard in trainer.py skips test eval gracefully when no test split exists

**See CLEANUP_PLAN.md for full phase details.**

---

## Workflow Patterns

### Making Changes

1. **Spawn specialized agents** for analysis (3-7 depending on scope)
2. **Verify findings** before acting (batch verification for large changes)
3. **Execute with handoffs** (sequential agents, each passes context to next)
4. **Verify after each step** (spawn verification subagent)
5. **Update documentation** (CLEANUP_TASKS during, COMPLETION after)

### When to Ask User

- Architectural decisions
- Deleting more than one file
- Changes to core interfaces
- Anything not in current phase scope
- When unsure about intent

### Verification Commands

```bash
# Import checks (should all succeed)
python -c "from src.core.types import DataRank, ModelFamily; print('OK')"
python -c "from src.core.contracts import get_model_contract; print('OK')"
python -c "from src.data.adapters import get_adapter; print('OK')"

# Single definition checks (should each return 1)
grep -r "class DataRank" src/ --include="*.py" | wc -l
grep -r "class ModelFamily" src/ --include="*.py" | wc -l

# Dead import checks (should return 0)
grep -r "from src\.coordination" src/ --include="*.py" | wc -l
grep -r "from src\.feature_selection" src/ --include="*.py" | wc -l
```

---

## Don'ts

| Don't | Do Instead |
|-------|------------|
| Make changes without reading docs | Read DIRECTION + CLEANUP_PLAN first |
| Execute without user approval | Propose approach, wait for confirmation |
| Skip linting | Run `ruff check` and `black` before commit |
| Create duplicate definitions | Import from canonical location |
| Comment out dead code | Delete it completely |
| Ignore validation failures | Fix or document as exception |

---

## Documented Exceptions

| Exception | Reason | Status |
|-----------|--------|--------|
| Dual AdapterResult | Circular import prevention | Bidirectional properties added |
| Pyright pandas errors | Type stub limitations | Not blocking, document when seen |

---

## Commands

**See COMMANDS.md** for the full command system reference including:
- Visual command matrix (tiered by scope)
- Subagent architecture and reference
- Standard workflows for phases and quick fixes
- Anti-patterns to avoid

Quick reference:
| Category | Light | Medium | Heavy |
|----------|-------|--------|-------|
| Analyze | `/analysis-targeted(1c)` | `/analysis-optimization(1b)` | `/analysis-full(1a)` |
| Verify | `/verify-claim(2a)` | `/verify-batch(2b)` | `/verify-contracts(2c)` |
| Execute | `/execute-surgical(4c)` | `/execute-standard(4a)` | `/execute-large(4b)` |
| Docs | `/docs-tasks(3a)` | `/docs-full(3b)` | `/docs-final(6a)` |
| Check | `/check-standard(5a)` | `/check-deep(5b)` | `/check-behavior(5c)` |

---

## Templates

Templates for DIRECTION, CLEANUP_PLAN, CLEANUP_TASKS, and COMPLETION are in:
`X ( IN PROGRESS DOCS) X/TEMPLATES/`

Use these when starting fresh or resetting documentation.

---

*Last updated: 2026-02-16*
*See CLEANUP_PLAN.md for current phase*
*See COMMANDS.md for command reference*
