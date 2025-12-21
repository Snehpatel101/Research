# Ensemble Price Prediction Pipeline

## Goal

Keep the codebase modular, readable, and easy to extend as we build a pipeline that can train and evaluate many different model families.

---

## Engineering Rules (Non-Negotiables)

### Architecture and Modularity
We do not build monoliths. Responsibilities must be split into small, composable modules with clear contracts and minimal coupling. Each module should do one thing well and expose a narrow, well-documented interface. Prefer dependency injection and explicit wiring over hidden globals or implicit side effects.

### File and Complexity Limits
No single file may exceed **650 lines**. If a module is growing, it's a signal that boundaries are wrong and responsibilities need to be refactored. Keep functions short, keep layers separated, and keep the cognitive load low.

### Fail Fast, Fail Hard
We would rather crash early than silently continue in an invalid state. Inputs are validated at the boundary. Assumptions are enforced with explicit checks. If something is wrong, we stop and surface a clear error message that points to the cause.

### Less Code is Better
Simpler implementations win. Prefer straightforward, boring solutions over clever abstractions. Avoid premature generalization. If a feature can be expressed with fewer moving parts, do that. Complexity must earn its place.

### No Exception Swallowing
Do not paper over failures with try/except. We do not swallow errors or "recover" by guessing. Use explicit validation, explicit return types, and explicit preconditions. If a dependency can fail, make that failure visible in the function contract and test it. Exceptions are allowed to propagate naturally so failures are obvious and diagnosable.

### Clear Validation
Every boundary validates what it receives: configuration, CLI inputs, dataset schemas, feature matrices, labels, and model parameters. Validation errors must be actionable, specific, and consistent. Prefer schema-based validation and typed structures over ad hoc checks.

### Clear Tests
Every module ships with tests that prove the contract. Unit tests cover pure logic. Integration tests cover pipeline wiring and data flow. Regression tests lock down previously fixed issues. Tests should be deterministic, fast, and easy to run locally and in CI.

### Definition of Done
A change is complete only when:
- Implementation is modular
- Stays within file limits (650 lines)
- Validates inputs at boundaries
- Backed by tests that clearly demonstrate correctness

---

## Auto-Activated Agents

These agents trigger **automatically** based on context - no manual invocation needed:

| When You're Doing... | Agent Auto-Activates | What It Does |
|---------------------|---------------------|--------------|
| Building pipeline stages | `ml-engineer` | Creates DAG-based ML workflows |
| Writing feature code | `data-engineer` | Spark optimization, data pipelines |
| Designing labeling logic | `quant-analyst` | Trading strategies, risk metrics |
| Creating data classes | `python-pro` | Modern Python patterns, Pydantic |
| Adding tests | `tdd-orchestrator` | Red-green-refactor cycles |
| Debugging issues | `debugger` | Error investigation |
| Optimizing performance | `performance-engineer` | Profiling, caching |

---

## Build Commands

```bash
# Build new pipeline stage
/ml-pipeline-workflow "create stage for [description]"

# Build feature engineering
/data-engineering:spark-optimization "optimize feature calculation for [task]"

# Build labeling system
/quantitative-trading:quant-analyst "implement triple-barrier labeling"

# Build validation
/tdd-workflows:tdd-cycle "add validation for [component]"
```

---

## Sequential Build Flow

When building new functionality, agents chain automatically:

```
You say: "Build a new resampling stage"
         ↓
    ml-engineer activates (pipeline design)
         ↓
    data-engineer activates (implementation)
         ↓
    python-pro activates (code patterns)
         ↓
    tdd-orchestrator activates (tests)
```

---

## Context Auto-Save

Context saves automatically at:
- Stage completion → `runs/{run_id}/artifacts/pipeline_state.json`
- Checkpoint → `_save_state()` in PipelineRunner

Restore: `/context-restore --project research --mode full`

---

## Pipeline Structure

```
src/stages/
├── stage1_ingest.py    → Load raw data
├── stage2_clean.py     → Resample 1min→5min
├── stage3_features.py  → 50+ indicators
├── stage4_labeling.py  → Triple-barrier
├── stage5_ga_optimize.py → DEAP optimization
├── stage6_final_labels.py
├── stage7_splits.py    → Train/val/test
└── stage8_validate.py  → Quality checks
```

---

## Quick Commands

```bash
./pipeline run --symbols MES,MGC
./pipeline rerun <id> --from <stage>
./pipeline status <id>
```

---

## Key Params

```python
SYMBOLS = ['MES', 'MGC']
LABEL_HORIZONS = [5, 20]  # H1 excluded (transaction costs > profit)
TRAIN/VAL/TEST = 70/15/15
PURGE_BARS = 60   # = max_bars for H20 (prevents label leakage)
EMBARGO_BARS = 288  # ~1 day buffer
```

---

## Phase 1 Analysis Summary (2025-12-20)

### Overall Score: 7.5/10 (Production-Ready)

**Strengths:**
- Triple-barrier labeling with symbol-specific asymmetric barriers (MES: 1.5:1.0)
- GA optimization with transaction cost penalties
- Proper purge (60) and embargo (288) for leakage prevention
- Quality-based sample weighting (0.5x-1.5x)

**Critical Fixes Applied:**
- `PURGE_BARS` increased from 20 to 60 (= max_bars for H20)
- Path traversal vulnerability fixed in stage1_ingest.py
- DataIngestor now integrated in pipeline

**Remaining Issues:**
- `TimeSeriesDataset` needed for Phase 2 model training
- 7 files still use `logging.basicConfig` (should use NullHandler)
- No regime-adaptive barriers implemented

**Expected Performance:**
| Horizon | Sharpe | Win Rate | Max DD |
|---------|--------|----------|--------|
| H5 | 0.3-0.8 | 45-50% | 10-25% |
| H20 | 0.5-1.2 | 48-55% | 8-18% |
