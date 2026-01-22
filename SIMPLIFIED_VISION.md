# Simplified Vision: ONE Config, ONE Orchestrator

**Date:** 2026-01-21
**Status:** Final Architecture Decision

---

## The User's Question

> "Why can't we have ONE SINGLE config inside SRC that we can edit that has all the configs? MLPipeline should ORCHESTRATE every single thing inside SRC."

## The Answer: We CAN. Here's How.

---

## Current State (The Problem)

```
7 Config Classes (only 2 actually used):
├── PipelineConfig     ← THE ONE (actually used by MLFactory)
├── GlobalConfig       ← Loads YAML defaults (needed)
├── TrainerConfig      ← Per-model settings (internal, needed)
├── DataConfig         ← Internal pipeline (hidden, needed)
├── MLConfig           ← DEAD CODE (0 imports in production)
├── UnifiedConfig      ← DEAD CODE (1117 lines, 0 imports)
└── SmartConfig        ← DEAD CODE (train() never called)

4 Entry Points (confusing):
├── MLFactory          ← Primary entry point
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

### Step 1: Delete Dead Code (Day 1)

```bash
# Remove unused config classes
rm src/ml_pipeline/config.py       # MLConfig - dead
rm src/config/smart_config.py      # SmartConfig - dead
# Keep UnifiedConfig for now, but mark deprecated
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
| 7 config classes | 1 config class (PipelineConfig) |
| 4 entry points | 1 entry point (MLPipeline) |
| User confusion | Simple, clear API |
| Dead code everywhere | Clean codebase |

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
