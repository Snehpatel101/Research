# CHILL — Verified Issues Registry

**What this is:** The verified list of real bugs found by 5 parallel diagnostic agents and confirmed by 3 parallel verification agents. These are the issues.

**Last updated:** 2026-02-27

---

## Verification Summary

- 15 claims investigated
- 2 claims **FALSE** (already fixed or already guarded)
- 2 claims **PARTIALLY TRUE**
- 11 claims **VERIFIED TRUE**

---

## FALSE — Already Fixed / Not Real

| # | Claim | Why False |
|---|-------|-----------|
| 1 | TCN seq_len=120 mismatch | Fixed in Phase 80. Default is now 64 (`tcn_model.py:211`), SequenceConfig default is 60 (`data.py:309`). |
| 3 | Invalid labels (-99) leak into training | `filter_invalid_labels()` is called at 5+ sites: `training_ops.py:446`, `model_training.py:133`, `hyperparameter_tuning.py:107`, `pipeline.py:390`, `trainer.py:594`. |

---

## VERIFIED TRUE — These Are The Issues

### CRITICAL

#### Issue 2: Walk-Forward Drops All But First Horizon
- **File:** `src/models/training/training_ops.py:400`
- **Code:** `horizon = self.config.horizons[0]` — no loop
- **Impact:** `HORIZONS=[10, 20]` silently drops H20. Standard, regime-aware, and meta-labeling modes all loop correctly (`for horizon in self.config.horizons:`). Walk-forward does not.
- **Fix:** Wrap the walk-forward body in `for horizon in self.config.horizons:` like the other modes.

#### Issue 5: Optuna Timeout Is Dead Code
- **File:** `src/validation/cv/cv_tuner.py:248-253`
- **Code:** `study.optimize(objective, n_trials=self.n_trials, ...)` — no `timeout=` parameter
- **Impact:** User sets `OPTUNA_MAX_TIME=7200` (2hr safety cap) but it never reaches `study.optimize()`. `TimeSeriesOptunaTuner.__init__` (`cv_tuner.py:38-47`) doesn't even accept a timeout parameter. No safety net if trials hang.
- **Fix:** Add `timeout` param to `TimeSeriesOptunaTuner.__init__`, forward to `study.optimize(timeout=self.timeout)`.

#### Issue 8: Walk-Forward + 4D Models Lose DatetimeIndex
- **File:** `src/models/training/training_ops.py:494-496`
- **Code:** `idx = pd.RangeIndex(n_all)` replaces DatetimeIndex
- **Impact:** `WalkForwardEvaluator.split()` at `walk_forward.py:308` checks `isinstance(X.index, pd.DatetimeIndex)` — with RangeIndex, label-aware purging (anti-leakage protection) is silently disabled for 4D models (PatchTST, iTransformer, TFT).
- **Fix:** Preserve original DatetimeIndex through the flatten/unflatten cycle instead of replacing with RangeIndex.

#### Issue 9: Walk-Forward Ignores Optuna Best Params
- **File:** `src/models/training/modes/walk_forward.py:427-442`
- **Code:** `_model_config` only gets `max_epochs`, `batch_size`, `early_stopping_patience` — no Optuna best_params
- **Impact:** Standard mode applies Optuna results (`model_training.py:326-328`). Walk-forward creates models with default hyperparameters only. Tuned params are silently discarded.
- **Fix:** Merge Optuna best_params into `_model_config` before `ModelRegistry.create()`.

### HIGH

#### Issue 4: Feature Selection Config Is Decorative (Orchestrator Path)
- **File:** `src/models/training/feature_selection.py:101-113, 237-289`
- **What's hardcoded vs what config says:**

| Setting | Config Value | Hardcoded Value | Location |
|---------|-------------|-----------------|----------|
| CV splits | `selection_cv_splits=5` | `n_splits=3` | `feature_selection.py:102` |
| Min frequency | `selection_min_frequency=0.6` | `min_feature_frequency=0.01` | `feature_selection.py:113` |
| N features | `selection_n_features=60` | `contract.max_features` (200 for boosting) | `feature_selection.py:237,288` |

- **Impact:** User's feature selection settings in Cell 2 have zero effect on the orchestrator's MDA ranking. The secondary Trainer path (`features.py:129,133,548`) does use config values, but the orchestrator path is the primary code path.
- **Fix:** Read `selection_cv_splits`, `selection_min_frequency`, and `selection_n_features` from config instead of hardcoding.

#### Issue 6: Optuna n_startup_trials Is Dead Code
- **File:** `src/validation/cv/cv_tuner.py:147`
- **Code:** `sampler=TPESampler(seed=42)` — no `n_startup_trials`
- **Impact:** User sets `OPTUNA_N_STARTUP_TRIALS=10` but it never reaches TPESampler. Happens to match Optuna's default (10), so no visible effect currently. But if user changes to 20 or 5, nothing happens.
- **Fix:** Forward `n_startup_trials` to `TPESampler(seed=..., n_startup_trials=...)`.

#### Issue 7: TICK_VALUE and CONFORMAL Config Completely Unwired
- **TICK_VALUE:**
  - Defined in notebook Cell 2 as `TICK_VALUE = 0.10`
  - No `tick_value` field exists on `EvaluationSection` (`experiment.py:132-146`)
  - Factory gets tick_value from `SymbolConfig.from_symbol("MGC")` which returns `1.00` (`symbol.py:107`)
  - The notebook value 0.10 is also **wrong** — it's the tick_size, not tick_value. MGC: tick_size=0.10, tick_value=1.00
  - **Fix:** Remove `TICK_VALUE` from notebook or rename to `TICK_SIZE` with a comment explaining it's informational only.

- **CONFORMAL:**
  - `CONFORMAL_ENABLED`, `CONFORMAL_ALPHA`, `CONFORMAL_METHOD` defined in Cell 2
  - No conformal fields on `ExperimentConfig` or any section
  - `ConformalConfig` exists at `training.py:178` but is never composed into the experiment config hierarchy
  - Pipeline never runs conformal prediction — these variables are purely cosmetic
  - Two incompatible `ConformalConfig` classes: `training.py:208` accepts `["naive", "aps", "raps"]`, `conformal.py:66` accepts `("lac", "aps", "naive")`
  - **Fix:** Either wire `ConformalConfig` into `ExperimentConfig` and the pipeline, or remove the dead variables from the notebook with a comment explaining conformal is post-hoc only.

#### Issue 10: Three Duplicate WalkForwardConfig Classes
- **Canonical:** `src/config/cv.py:280` — has `retrain_on_each_window=True`, inherits `BaseConfig`
- **Operational:** `src/validation/cv/walk_forward.py:34` — plain dataclass, no `retrain_on_each_window`
- **Trainer:** `src/models/training/modes/walk_forward.py:50` — `gap_bars=0`, `embargo_bars=0` (different defaults!)
- **Impact:** Walk-forward trainer imports from the non-canonical duplicate. `retrain_on_each_window` on canonical class is dead code. `WalkForwardTrainerConfig` defaults to 0 embargo/gap (silently disables anti-leakage if not explicitly overridden).
- **Fix:** Consolidate to one WalkForwardConfig. Delete duplicates.

#### Issue 14: tick_value Hardcoded for Non-MES in Financial Reports
- **File:** `src/models/training/feature_selection.py:545`
- **Code:** `tick_value=1.25 if self.config.symbol == "MES" else 0.10`
- **Impact:** MGC gets tick_value=0.10 instead of correct 1.00 (10x underestimate). MNQ gets 0.10 instead of 0.50 (5x underestimate). Transaction cost calculations in financial reports are wrong for all non-MES symbols.
- **Fix:** Replace ternary with `SymbolConfig.from_symbol(self.config.symbol).tick_value`.

### MEDIUM

#### Issue 11: cv_method Not Exposed in Notebook
- **File:** `src/config/experiment.py:108`
- **Default:** `cv_method: str = "purged_kfold"`
- **Impact:** Phase 66 added CPCV support but users have no way to enable it through the notebook. No `CV_METHOD` variable in Cell 2.
- **Fix:** Add `CV_METHOD = "purged_kfold"` to Cell 2 with comment about CPCV option.

#### Issue 12: TrainingSection batch_size Default Still 256
- **File:** `src/config/experiment.py:123`
- **Default:** `batch_size: int = 256`
- **Impact:** Phase 68 changed default to 512 conceptually but never updated the dataclass. Notebook overrides to 512 so low impact for notebook users. Affects programmatic API users.
- **Fix:** Change default to `batch_size: int = 512`.

#### Issue 13: Feature Selection Is Global, Not Per-Window (Lookahead Bias)
- **File:** `src/models/training/unified_orchestrator.py:399` calls `_pre_training_validation(df)` once on ALL data
- **File:** `src/models/training/feature_selection.py:209-297` runs MDA on full dataset
- **Impact:** Features are ranked using future data that wouldn't be available in earlier walk-forward windows. Mild lookahead bias. Per-window infrastructure exists (`WalkForwardFeatureSelector`) but isn't wired into walk-forward training.
- **Fix:** Re-run feature selection per walk-forward window using only the training portion.

---

## Non-Fatal Warnings (Not Bugs)

| Warning | Explanation |
|---------|-------------|
| `torch.nn.utils.weight_norm is deprecated` | Fixed in Phase 83 source but Colab may run older checkout |
| `permute(sparse_coo): input.dim() = 3 != len(dims) = 4` | `torch.compile(mode="max-autotune")` tries invalid kernel candidates during autotuning, discards them. Expected behavior. |
| `PreparedData validation issues: y_test contains 5 invalid labels (-99)` | Warning fires during validation but labels ARE filtered before training at 5+ guard sites. Not a bug. |
| `Receptive field (61) < sequence length (120)` | May appear if checkpoint resumes old config. Current defaults are correct (seq_len=64). |
| `TypedStorage is deprecated` | PyTorch internal deprecation warning. No action needed. |

---

## Priority Order for Fixes

1. **Walk-forward multi-horizon** (Issue 2) — silent data loss, easy fix
2. **Feature selection wiring** (Issue 4) — user config ignored, moderate fix
3. **Optuna timeout** (Issue 5) — no safety net, easy fix
4. **Walk-forward DatetimeIndex** (Issue 8) — anti-leakage disabled, moderate fix
5. **Walk-forward Optuna params** (Issue 9) — tuned params discarded, moderate fix
6. **tick_value hardcode** (Issue 14) — wrong financial reports, easy fix
7. **Optuna n_startup_trials** (Issue 6) — dead config, easy fix
8. **Duplicate WalkForwardConfig** (Issue 10) — architectural debt, moderate fix
9. **TICK_VALUE/CONFORMAL cleanup** (Issue 7) — cosmetic, easy fix
10. **cv_method exposure** (Issue 11) — missing feature, easy fix
11. **batch_size default** (Issue 12) — stale default, trivial fix
12. **Per-window feature selection** (Issue 13) — lookahead bias, complex fix

---

*This document was generated by 5 parallel diagnostic agents and verified by 3 parallel verification agents.*
*All file:line references verified against source code on 2026-02-27.*
