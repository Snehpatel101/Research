# Phase B: Optuna Optimization Migration (Stages 7-9)

**Status:** Planning Complete
**Estimated Effort:** 8.5 days

---

## Current State Summary

### Stage 7: Label Optimization (100 trials)
| Location | Lines | Purpose |
|----------|-------|---------|
| `src/labeling/optimization.py` | ~700 | Primary LabelOptimizer with TPE |
| `src/optimization/labels.py` | ~540 | Alternative implementation |
| `src/phase1/stages/labeling/triple_barrier.py` | ~650 | Core labeling with numba |

### Stage 8: Feature Selection (100 trials)
| Location | Lines | Purpose |
|----------|-------|---------|
| `src/features/selection.py` | ~700 | Full-featured selector |
| `src/features/optimization.py` | ~400 | Model-specific optimization |
| `src/optimization/features.py` | ~720 | Combined selection/pruning |

### Stage 9: Feature Pruning (50 trials)
| Location | Lines | Purpose |
|----------|-------|---------|
| `src/features/pruning.py` | ~800 | Comprehensive pruner |
| `src/optimization/features.py` | ~150 | Pruning method |

---

## Target State

### New File: `src/pipeline/phases/optimization.py`

```python
class Stage7LabelOptimization(OptimizationStage):
    """100 trials to optimize triple-barrier parameters."""
    stage_number = 7
    n_trials = 100
    # Search: upper_mult, lower_mult, horizon, atr_period
    # Objective: class_balance(40%) + barrier_hit_rate(30%) + model_f1(30%)

class Stage8FeatureSelection(OptimizationStage):
    """100 trials for binary feature include/exclude."""
    stage_number = 8
    n_trials = 100
    # Search: Binary for 162+ features (by group)
    # Objective: F1 with regularization

class Stage9FeaturePruning(OptimizationStage):
    """50 trials for importance-based pruning."""
    stage_number = 9
    n_trials = 50
    # Search: importance_threshold, top_k, method
    # Objective: Performance with minimal features
```

---

## Optuna Integration

### Study Creation Pattern
```python
def create_study(self, study_name: str, storage: str = None):
    sampler = optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=10,
        multivariate=True,
    )
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5)

    return optuna.create_study(
        study_name=study_name,
        storage=storage or f"sqlite:///experiments/optuna/{study_name}.db",
        load_if_exists=True,  # Resume support
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
```

### Resume Pattern
```python
def run(self, df: pd.DataFrame, resume: bool = True):
    study = self.create_study(load_if_exists=resume)
    completed = len([t for t in study.trials if t.state == COMPLETE])
    remaining = max(0, self.n_trials - completed)

    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
```

---

## Config Mapping

| Config File | Stage | Key Parameters |
|-------------|-------|----------------|
| `config/optimization/label_optimization.yaml` | 7 | upper_mult, lower_mult, horizon, atr_period |
| `config/optimization/feature_selection.yaml` | 8 | selection_mode, min/max features |
| `config/optimization/feature_pruning.yaml` | 9 | importance_threshold, method |

---

## Interface Contracts

| Stage | Input | Output |
|-------|-------|--------|
| 7 | Features from Stage 6 (~180) | `Stage7Output(labeled_df, barrier_params, study)` |
| 8 | Labeled df (~180 features) | `Stage8Output(selected_df, ~60-100 features, mask)` |
| 9 | Selected df (~60-100) | `Stage9Output(pruned_df, ~30-60 features, rankings)` |

**Checkpoint:** `data/optimized/{symbol}_optimized.parquet`

---

## Migration Steps

1. **Base infrastructure** (1 day): OptimizationStage base class
2. **Stage 7 migration** (2 days): Wrap LabelOptimizer
3. **Stage 8 migration** (2 days): Unify 4 selection implementations
4. **Stage 9 migration** (1.5 days): Wrap FeaturePruner
5. **Integration testing** (1.5 days): End-to-end optimization
6. **Documentation** (0.5 days)

---

## Critical Files

1. `src/labeling/optimization.py` - Primary Stage 7 source
2. `src/features/selection.py` - Primary Stage 8 source
3. `src/features/pruning.py` - Primary Stage 9 source
4. `src/phase1/stages/labeling/triple_barrier.py` - Labeling core
5. `config/optimization/label_optimization.yaml` - Config pattern
