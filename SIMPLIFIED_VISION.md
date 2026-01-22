# Simplified Vision: ONE Config, ONE Orchestrator

**Date:** 2026-01-21
**Status:** Architecture Proposal (REQUIRES RECONCILIATION)

> ⚠️ **WARNING:** This document contains inaccuracies about "dead code" that have been corrected below.
> See the "CORRECTION" notes throughout. This proposal needs reconciliation with RECONCILED_PLAN.md.

---

## The User's Question

> "Why can't we have ONE SINGLE config inside SRC that we can edit that has all the configs? MLPipeline should ORCHESTRATE every single thing inside SRC."

## The Answer: We CAN. Here's How.

---

## Current State (The Problem)

> **CORRECTION:** The original "dead code" claims below were INACCURATE. Analysis of the actual codebase shows:
> - `MLConfig`: Used by MLPipeline, exported in `__all__`
> - `UnifiedConfig`: Documented as primary interface in `src/config/__init__.py` (1117 lines, 16 sections)
> - `SmartConfig`: 925-line UX layer for beginners, actively maintained
>
> These are NOT dead code. The actual problem is **overlapping configs**, not unused ones.

```
30+ Config Classes (overlapping, confusing):
├── PipelineConfig (x2!)  ← TWO classes with same name in different locations!
│   ├── src/core/config.py (625 lines) - Orchestration
│   └── src/pipeline/data_config.py (350 lines) - Data prep
├── UnifiedConfig         ← Comprehensive (1117 lines, 16 sections) - ACTIVELY USED
├── GlobalConfig          ← Loads YAML defaults (needed)
├── TrainerConfig         ← Per-model settings (internal, needed)
├── DataConfig            ← Internal pipeline (hidden, needed)
├── MLConfig              ← ML pipeline specific - ACTIVELY USED
├── SmartConfig           ← Beginner-friendly API - ACTIVELY USED
└── [20+ more specialized configs across modules]

4 Entry Points (confusing):
├── MLFactory          ← Primary entry point (940 lines)
├── MLPipeline         ← Deprecated, delegates to MLFactory
├── PipelineRunner     ← Phase 1 data only
└── UnifiedTrainingOrchestrator ← Training only
```

---

## Target State (The Solution)

```
1 Config (PipelineConfig - extended):
└── PipelineConfig     ← THE ONE CONFIG
    ├── symbol, data_path, output_dir
    ├── models, horizons, build_ensemble
    ├── training_mode, cv_method
    ├── labeling settings
    ├── optimization flags
    ├── feature settings
    └── ALL other settings

1 Orchestrator (MLPipeline):
└── MLPipeline         ← THE ONE ORCHESTRATOR
    ├── run()          ← Does EVERYTHING
    ├── run_data()     ← Phase 1-2 only
    ├── run_train()    ← Phase 3-4 only
    ├── run_bundle()   ← Phase 5 only
    ├── run_evaluate() ← Phase 6 only
    └── run_backtest() ← Phase 7 only
```

---

## The Simple API

```python
from src import MLPipeline, PipelineConfig

# ONE config with everything
config = PipelineConfig(
    # Required
    symbol="MES",
    data_path="./data/mes_1min.parquet",
    output_dir="./experiments/run_001",

    # Models
    models=["xgboost", "lightgbm", "lstm"],
    horizons=[5, 10, 15, 20],
    build_ensemble=True,
    meta_learner="ridge_meta",

    # Training
    training_mode="standard",  # or walk_forward, regime_aware, meta_labeling
    cv_method="purged_kfold",
    n_splits=5,

    # Labeling
    labeling_method="triple_barrier",
    upper_mult=2.0,
    lower_mult=2.0,

    # Optimization
    optimize_labels=True,
    optimize_features=True,
    optimize_hyperparams=True,
)

# ONE orchestrator that does EVERYTHING
pipeline = MLPipeline(config)
result = pipeline.run()  # Runs all 28 operations across 8 phases

# Or run phases individually
pipeline.run_data()      # Data prep
pipeline.run_train()     # Model training
pipeline.run_bundle()    # Create inference bundles
pipeline.run_evaluate()  # CV, walk-forward, CPCV-PBO
pipeline.run_backtest()  # Strategy backtesting
```

---

## What MLPipeline.run() Does (All 28 Operations)

```
Phase 1: Data Preparation
├── 1.1 Ingest (load parquet)
├── 1.2 Clean (resample, fill gaps)
├── 1.3 Sessions (trading hours filter)
├── 1.4 MTF (multi-timeframe features)
├── 1.5 Features (180+ indicators)
├── 1.6 Regime (volatility/trend detection)
├── 1.7 Labeling (triple-barrier)
├── 1.8 Label Optimization (Optuna)
├── 1.9 Splits (train/val/test)
├── 1.10 Scaling (robust scaler)
└── 1.11 Validation (data quality)

Phase 2: Data Adapters
├── 2.1 Tabular (2D for boosting)
├── 2.2 Sequence (3D for LSTM/GRU)
└── 2.3 Multi-Resolution (4D for transformers)

Phase 3: Training
├── 3.1 Standard training (all models)
├── 3.2 Walk-forward training (if enabled)
├── 3.3 Regime-aware training (if enabled)
└── 3.4 Meta-labeling (if enabled)

Phase 4: Ensemble
├── 4.1 OOF generation
├── 4.2 OOF alignment
└── 4.3 Stacking meta-learner

Phase 5: Bundling
├── 5.1 Model bundles (per-model)
├── 5.2 Preprocessing graph
└── 5.3 Ensemble bundle

Phase 6: Evaluation
├── 6.1 CV evaluation
├── 6.2 Walk-forward evaluation
└── 6.3 CPCV-PBO (backtest overfitting)

Phase 7: Inference (optional)
├── 7.1 Single prediction
├── 7.2 Batch prediction
└── 7.3 Model serving

Phase 8: Backtesting (optional)
├── 8.1 Strategy backtest
├── 8.2 Equity curve
└── 8.3 Performance metrics
```

---

## Implementation: What to Change

### Step 1: Consolidate Configs (Day 1)

> **CORRECTION:** Do NOT delete these files - they are actively used!
> The correct approach is to CONSOLIDATE, not delete.

```bash
# WRONG - These are NOT dead code:
# rm src/ml_pipeline/config.py       # MLConfig - USED by MLPipeline
# rm src/config/smart_config.py      # SmartConfig - USED as beginner API

# CORRECT - Add deprecation warnings and consolidate:
# 1. Choose ONE canonical config (PipelineConfig or UnifiedConfig - TEAM DECISION)
# 2. Add deprecation warnings to non-canonical configs
# 3. Create adapters so old configs delegate to canonical one
# 4. Update all internal code to use canonical config
```

### Step 2: Extend PipelineConfig (Day 2-3)

Add any missing fields from the deleted configs:

```python
# src/core/config.py - add missing fields
@dataclass
class PipelineConfig:
    # ... existing fields ...

    # Add from MLConfig (if not already present)
    walk_forward_window: int = 5000
    walk_forward_step: int = 1000
    cpcv_n_splits: int = 5
    cpcv_n_tests: int = 2

    # Add from UnifiedConfig sections (if not already present)
    # Most already exist - just verify completeness
```

### Step 3: Create MLPipeline Orchestrator (Day 4-5)

```python
# src/pipeline/orchestrator.py

class MLPipeline:
    """THE orchestrator for the entire ML pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._factory = MLFactory(config)

    def run(self, df: pd.DataFrame = None) -> PipelineResult:
        """Run the COMPLETE pipeline."""
        # Load data if not provided
        if df is None:
            df = self._load_data()

        # Run all phases
        result = self._factory.run(df)

        # Run evaluation (not in MLFactory currently)
        if self.config.run_evaluation:
            self._run_evaluation(result)

        # Run backtesting (not in MLFactory currently)
        if self.config.run_backtest:
            self._run_backtest(result)

        return result

    def run_data(self) -> DataResult:
        """Phase 1-2: Data preparation and adapters."""
        ...

    def run_train(self) -> TrainingResult:
        """Phase 3-4: Training and ensemble."""
        ...

    def run_bundle(self) -> BundleResult:
        """Phase 5: Create inference bundles."""
        ...

    def run_evaluate(self) -> EvaluationResult:
        """Phase 6: Evaluation."""
        ...

    def run_backtest(self) -> BacktestResult:
        """Phase 7-8: Backtesting."""
        ...
```

### Step 4: Update Exports (Day 5)

```python
# src/__init__.py

from src.core.config import PipelineConfig
from src.pipeline.orchestrator import MLPipeline
from src.pipeline.result import PipelineResult

# Deprecated (with warnings)
from src.factory import MLFactory  # DeprecationWarning

__all__ = ["MLPipeline", "PipelineConfig", "PipelineResult"]
```

### Step 5: Deprecate Old Entry Points (Day 6)

```python
# src/factory.py
import warnings

class MLFactory:
    def __init__(self, config):
        warnings.warn(
            "MLFactory is deprecated. Use MLPipeline instead:\n"
            "  from src import MLPipeline\n"
            "  pipeline = MLPipeline(config)\n"
            "  result = pipeline.run()",
            DeprecationWarning,
            stacklevel=2,
        )
        self._impl = ...
```

---

## What This Achieves

| Before | After |
|--------|-------|
| 30+ overlapping config classes | 1 canonical config (PipelineConfig) + internal helpers |
| 4 entry points | 1 entry point (MLPipeline) |
| User confusion | Simple, clear API |
| Overlapping responsibilities | Clear boundaries |

> **NOTE:** Achieving "1 config" doesn't mean deleting configs - it means having ONE
> user-facing config that internally delegates to specialized configs as needed.

---

## Timeline

| Day | Task |
|-----|------|
| 1 | Delete dead config code (MLConfig, SmartConfig) |
| 2-3 | Audit PipelineConfig, add any missing fields |
| 4-5 | Create MLPipeline orchestrator with full coverage |
| 5 | Update src/__init__.py exports |
| 6 | Add deprecation warnings to old entry points |
| 7 | Update documentation and tests |

**Total: 1 week**

---

## The Final API

```python
from src import MLPipeline, PipelineConfig

config = PipelineConfig(
    symbol="MES",
    data_path="./data/mes.parquet",
    output_dir="./experiments",
    models=["xgboost", "lstm"],
    build_ensemble=True,
)

result = MLPipeline(config).run()

print(f"Best model: {result.best_model}")
print(f"Ensemble F1: {result.ensemble_metrics['f1']}")
```

**That's it. ONE config. ONE orchestrator. EVERYTHING runs.**

---

## ⚠️ CRITICAL: Reconciliation Required

This document proposes `PipelineConfig` as THE ONE config. However, `RECONCILED_PLAN.md` proposes `UnifiedConfig` as THE ONE config.

| Document | Canonical Config | Rationale |
|----------|-----------------|-----------|
| SIMPLIFIED_VISION.md | `PipelineConfig` | Already used by MLFactory, simpler |
| RECONCILED_PLAN.md | `UnifiedConfig` | Already comprehensive (1117 lines, 16 sections) |

**TEAM DECISION REQUIRED:**
1. **Option A (This doc):** Extend `PipelineConfig`, deprecate `UnifiedConfig`
2. **Option B (RECONCILED_PLAN):** Keep `UnifiedConfig`, deprecate `PipelineConfig`

**Factors to Consider:**
- `UnifiedConfig` already has `to_trainer_config()` and `to_pipeline_config()` adapters
- `PipelineConfig` is already the type MLFactory accepts
- Changing canonical config affects ALL existing code and scripts

**Timeline Mismatch:**
- This document: 1 week (optimistic, may be unrealistic)
- RECONCILED_PLAN.md: 7 weeks (more realistic, includes tooling and buffer)

---

*Last Updated: 2026-01-21*
*Status: DRAFT - Corrections applied, awaiting team decision*
