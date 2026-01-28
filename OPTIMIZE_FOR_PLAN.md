# Implementation Plan: Wire OPTIMIZE_FOR Through Pipeline

**Date:** 2026-01-27
**Status:** PLAN (not yet implemented)
**Source:** 3 specialized agents — metric mapping, design, plan assembly

---

## Problem

`OPTIMIZE_FOR = "sharpe_ratio"` in the notebook is **silently ignored**. The user's chosen metric never reaches the optimization layer.

```
ExperimentConfig.training.optuna.metric = "sharpe_ratio"   ← user sets this
         ↓
to_pipeline_config()                                        ← DROPS metric
         ↓
PipelineConfig                                              ← HAS NO metric field
         ↓
OptimizationPipeline                                        ← hardcodes "f1_weighted"
         ↓
HyperparameterOptimizer(scoring="f1_weighted")              ← ignores user's choice
FeatureOptimizer(scoring="f1_weighted")                     ← ignores user's choice
```

---

## Solution: 7 Changes in 6 Files + 1 New File

### Change 1: Add `optuna_metric` to PipelineConfig

**File:** `src/core/config.py:201`
**After:** `optuna_random_state: int = DEFAULT_OPTUNA_RANDOM_STATE`
**Add:**
```python
optuna_metric: str = "f1_weighted"  # Optimization metric (from OptunaConfig.metric)
```

---

### Change 2: Pass metric in `to_pipeline_config()`

**File:** `src/config/experiment.py:~421` (inside `to_pipeline_config()`)
**After:** `optuna_random_state=self.training.optuna.random_state,`
**Add:**
```python
optuna_metric=self.training.optuna.metric,
```

---

### Change 3: Create shared scoring module

**File:** `src/optimization/scoring.py` (NEW — ~60 lines)

Create `get_score_fn(metric_name: str) -> Callable[[ndarray, ndarray], float]` with cases:

| Metric | Implementation |
|--------|---------------|
| `f1_weighted` | `f1_score(y_true, y_pred, average="weighted")` |
| `f1_macro` | `f1_score(y_true, y_pred, average="macro")` |
| `accuracy` | `accuracy_score(y_true, y_pred)` |
| `precision` | `precision_score(y_true, y_pred, average="weighted")` |
| `recall` | `recall_score(y_true, y_pred, average="weighted")` |
| `sharpe_ratio` | Proxy: simulate PnL from classification, compute annualized Sharpe |
| `sortino_ratio` | Proxy: same as Sharpe but penalize downside only |
| `profit_factor` | Proxy: `sum(wins) / abs(sum(losses))` from classification PnL |

Trading metrics use a **classification proxy**: correct prediction = +1 return, incorrect = -1. This matches the existing `five_dimension_objective.py:441-487` default Sharpe logic.

`roc_auc` and `log_loss` raise `ValueError` — they require probabilities, not class predictions.

---

### Change 4: Thread metric through OptimizationPipeline

**File:** `src/optimization/pipeline.py`

| Location | Current | Change To |
|----------|---------|-----------|
| `__init__` (~line 238) | no `scoring` param | Add `scoring: str = "f1_weighted"`, store `self.scoring` |
| `from_config()` (~line 264) | no metric read | Add `scoring=config.optuna_metric` |
| Line 422-428 | `FeatureOptimizer(...)` no scoring | Add `scoring=self.scoring` |
| Line 494 | `scoring="f1_weighted"` hardcoded | `scoring=self.scoring` |
| Line 652 | `scoring="f1_weighted"` hardcoded | `scoring=self.scoring` |

---

### Change 5: Update FeatureOptimizer hardcoded call

**File:** `src/optimization/features.py:659`
**Current:** `permutation_importance(..., scoring="f1_weighted")`
**Change:** `permutation_importance(..., scoring=self.scoring)`

---

### Change 6: Replace duplicated dispatchers with shared module

**File:** `src/optimization/hyperparameters.py:540-549`
**File:** `src/optimization/features.py:255-269`

Both have identical `_get_score_fn()` methods. Replace body with:
```python
from src.optimization.scoring import get_score_fn
return get_score_fn(self.scoring)
```

---

### Change 7: Add imports to `__init__.py`

**File:** `src/optimization/__init__.py`
**Add:** `from src.optimization.scoring import get_score_fn`

---

## Wiring Chain After Implementation

```
ExperimentConfig.training.optuna.metric = "sharpe_ratio"    [EXISTS]
         ↓  (Change 2)
to_pipeline_config() passes optuna_metric=...
         ↓  (Change 1)
PipelineConfig.optuna_metric = "sharpe_ratio"               [NEW FIELD]
         ↓  (Change 4)
OptimizationPipeline(scoring="sharpe_ratio")                [NEW PARAM]
         ↓
    ├── HyperparameterOptimizer(scoring="sharpe_ratio")     [ALREADY ACCEPTS]
    │       └── _get_score_fn() → scoring.get_score_fn()    (Change 6)
    │
    └── FeatureOptimizer(scoring="sharpe_ratio")            [ALREADY ACCEPTS]
            ├── _get_score_fn() → scoring.get_score_fn()    (Change 6)
            └── permutation_importance(scoring=self.scoring) (Change 5)
```

---

## What Does NOT Change

- **`OptunaConfig`** — already has `metric` field + validation whitelist
- **`HyperparameterOptimizer.__init__`** — already accepts `scoring` param
- **`FeatureOptimizer.__init__`** — already accepts `scoring` param
- **`five_dimension_objective.py`** — already accepts `metric_fn` callable (separate path)
- **Notebook** — already has NOTE about OPTIMIZE_FOR; remove it after this fix
- **Training layer metric keys** — `val_f1`, `accuracy` dict keys are evaluation outputs, not optimization targets

---

## Execution Order

1. Create `src/optimization/scoring.py` (Change 3) — no dependencies
2. Add `optuna_metric` to `PipelineConfig` (Change 1) — no dependencies
3. Update `to_pipeline_config()` (Change 2) — depends on Change 1
4. Update `OptimizationPipeline` (Change 4) — depends on Change 1
5. Update `FeatureOptimizer` hardcoded call (Change 5) — no dependencies
6. Replace both dispatchers (Change 6) — depends on Change 3
7. Update `__init__.py` exports (Change 7) — depends on Change 3

**Changes 1, 2, 3, 5 can run in parallel. Then 4, 6, 7.**

---

## Validation

```bash
# Verify new field exists
python -c "from src.core.config import PipelineConfig; c = PipelineConfig(); print(c.optuna_metric)"

# Verify wiring
python -c "
from src.config.experiment import ExperimentConfig
e = ExperimentConfig()
e.training.optuna.metric = 'sharpe_ratio'
p = e.to_pipeline_config()
print(f'optuna_metric={p.optuna_metric}')  # Should print sharpe_ratio
"

# Verify scoring module
python -c "
from src.optimization.scoring import get_score_fn
import numpy as np
fn = get_score_fn('sharpe_ratio')
print(fn(np.array([0,1,1,0]), np.array([0,1,0,0])))  # Should return float
"

# Verify no hardcoded f1_weighted remains in pipeline.py
grep -n 'scoring="f1_weighted"' src/optimization/pipeline.py  # Should return 0 lines

# Ruff check
ruff check src/optimization/scoring.py src/optimization/pipeline.py src/core/config.py
```

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Trading metrics as classification proxy may mislead | MEDIUM | Document that proxy Sharpe != backtest Sharpe |
| `permutation_importance` may not work with custom scorer | LOW | sklearn accepts callable scorers |
| Breaking change if code reads `scoring` differently | LOW | Default is `"f1_weighted"` (no behavior change) |

---

## Estimated Scope

- **6 modified files** + **1 new file** (`scoring.py`)
- **~110 lines changed/added total**
- **Default behavior unchanged** — `"f1_weighted"` remains the default everywhere

---

*This is a plan document. No code has been modified.*
