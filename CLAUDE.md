# Ensemble Price Prediction Pipeline

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

## Context Auto-Save

Context saves automatically at:
- Stage completion → `runs/{run_id}/artifacts/pipeline_state.json`
- Checkpoint → `_save_state()` in PipelineRunner

Restore: `/context-restore --project research --mode full`

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

## Quick Commands

```bash
./pipeline run --symbols MES,MGC
./pipeline rerun <id> --from <stage>
./pipeline status <id>
```

## Key Params

```python
SYMBOLS = ['MES', 'MGC']
LABEL_HORIZONS = [1, 5, 20]
TRAIN/VAL/TEST = 70/15/15
PURGE_BARS = 60  # = max_bars for H20 (prevents label leakage)
EMBARGO_BARS = 288
```
