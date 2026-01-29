# Notebook Investigation: POSITION_SIZING & Optuna

**Date:** 2026-01-27
**Source:** Parallel agent investigation of `src/` pipeline vs notebook config

---

## Finding 1: POSITION_SIZING Wiring

### Verdict: TRIVIAL to wire, but reveals a bug

**Current state:** `POSITION_SIZING` is defined in Cell 2 but not consumed. `EvaluationSection` has no `position_sizing` field.

**The fix is ~10 lines:**
1. Add `position_sizing: str = "fixed"` to `EvaluationSection` in `src/config/experiment.py`
2. Update `to_backtest_config()` to pass it through
3. Update `from_dict()` and `to_dict()` for serialization
4. Add one line to notebook Cell 5: `position_sizing=POSITION_SIZING`

### Bug Discovered: BacktestConfig Mismatch

There are **two incompatible BacktestConfig classes**:

| Config | Location | Field Name |
|--------|----------|------------|
| Canonical | `src/config/inference.py:265` | `position_sizing: str = "fixed"` |
| Backtester local | `src/inference/backtesting/backtest.py:59` | `position_sizing_method: str = "fixed_contracts"` |

`factory.py:570` calls `to_backtest_config()` which returns the canonical config, but `Backtester.__init__()` accesses `self.config.position_sizing_method` (the local field name). This means **the field name doesn't match** — the canonical config says `position_sizing`, the backtester expects `position_sizing_method`.

**Recommendation:** Fix the field name mismatch first, then wire POSITION_SIZING through. Separate PR.

---

## Finding 2: Optuna in the Notebook

### Verdict: OPTUNA_TRIALS and OPTIMIZE_FOR are currently IGNORED

### How Optuna Actually Works

The 5-dimension optimizer (`src/optimization/five_dimension_objective.py`) optimizes:

| Dimension | What It Tunes |
|-----------|---------------|
| 1. Triple Barrier | profit/loss thresholds, max holding bars |
| 2. Feature Selection | which features to include from base set |
| 3. Feature Parameters | RSI period, ATR window, BB std, etc. |
| 4. Feature Timeframes | which timeframe each feature uses |
| 5. Model Hyperparameters | learning rate, depth, layers, dropout, etc. |

### The Broken Pipeline

```
ExperimentConfig.training.optuna.n_trials = 50  (from notebook)
          ↓
ExperimentConfig.to_pipeline_config()  ← DOES NOT PASS optuna fields
          ↓
PipelineConfig.hyperparam_trials = 100  (hardcoded default, ignores 50)
```

`to_pipeline_config()` at `src/config/experiment.py:388-414` maps models, horizons, splits, etc. but **skips all Optuna fields**. The `PipelineConfig` in `src/core/config.py` uses its own hardcoded defaults:

| PipelineConfig Field | Default | Notebook Setting | Used? |
|---------------------|---------|-----------------|-------|
| `hyperparam_trials` | 100 | `OPTUNA_TRIALS = 50` | NO |
| `label_optimization_trials` | 100 | not exposed | N/A |
| `feature_selection_trials` | 100 | not exposed | N/A |
| `feature_pruning_trials` | 50 | not exposed | N/A |

### OPTIMIZE_FOR ("sharpe_ratio") — Also Ignored

- `OptunaConfig.metric` accepts any string — no validation
- The metric field is **never read** by the training pipeline after conversion to `PipelineConfig`
- `OptunaConfig.validate()` checks `n_trials`, `sampler`, `pruner`, `direction` — but **not `metric`**

### What Power Users Can't Control (but OptunaConfig supports)

| Field | Default | Not Exposed in Notebook |
|-------|---------|------------------------|
| `sampler` | "tpe" | TPE, random, CMA-ES, grid |
| `pruner` | "median" | median, hyperband, none, percentile |
| `direction` | "maximize" | maximize or minimize |
| `n_jobs` | 1 | parallel trial execution |
| `timeout` | 0 (none) | seconds limit |
| `study_name` | None | for persistence |
| `storage` | None | database URL for persistence |

---

## Summary: What Needs Fixing

### To Make the Notebook Honest

| Issue | Severity | Fix |
|-------|----------|-----|
| `OPTUNA_TRIALS` ignored | HIGH | Wire `to_pipeline_config()` to pass `optuna.n_trials → hyperparam_trials` |
| `OPTIMIZE_FOR` ignored | HIGH | Wire metric through conversion, or connect to objective function |
| `POSITION_SIZING` unused | LOW | Add to `EvaluationSection`, fix BacktestConfig mismatch |
| BacktestConfig field mismatch | MEDIUM | Unify `position_sizing` vs `position_sizing_method` |
| No metric validation | LOW | Add whitelist to `OptunaConfig.validate()` |

### Key Files to Modify

| File | What to Change |
|------|---------------|
| `src/config/experiment.py:388-414` | `to_pipeline_config()` — pass optuna fields |
| `src/config/experiment.py:128-137` | `EvaluationSection` — add `position_sizing` |
| `src/config/experiment.py:443-453` | `to_backtest_config()` — pass `position_sizing` |
| `src/core/config.py` | `PipelineConfig` — accept optuna fields from ExperimentConfig |
| `src/inference/backtesting/backtest.py:59` | Align field name with canonical BacktestConfig |

### The Notebook Itself Is Fine

The notebook's structure and user experience are correct. The problem is upstream in the config conversion layer — `to_pipeline_config()` drops Optuna settings on the floor. Once the pipeline wiring is fixed, the notebook's `OPTUNA_TRIALS` and `OPTIMIZE_FOR` will work as intended without any notebook changes.

---

*This document is investigative. No code was modified.*
