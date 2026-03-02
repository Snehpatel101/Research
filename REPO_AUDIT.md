# Repository Audit — ML Factory for Intraday Trading

**Date:** March 1, 2026
**Branch:** `main` (HEAD: `a6a6262`)
**Notebook:** `notebooks/ml_factory_colab.ipynb` (34 cells, 2194 lines)
**Scope:** Commit history, notebook ↔ src alignment, config consistency, production readiness

---

## Executive Summary

The repo is an ambitious, well-structured ML pipeline for intraday futures trading (MGC, MES). The notebook (`ml_factory_colab.ipynb`) is cleanly organized with 34 cells covering setup → config → data → training → eval → deploy. All notebook imports resolve to real `src/` modules. However, there are **5 bugs**, **4 serious config inconsistencies**, **3 repo hygiene issues**, and **several production-readiness gaps** documented below.

**Priority ranking:** 🔴 = bug/will-break, 🟡 = inconsistency/silent-wrong-result, 🟢 = hygiene/improvement

---

## 1. 🔴 BUGS (Will Break or Silently Corrupt Results)

### 1.1 `batch_size` Default Mismatch in `from_dict()`

**File:** `src/config/experiment.py`

- **Line 123:** `TrainingSection.batch_size` defaults to `512`
- **Line 290:** `from_dict()` fallback defaults to `256`

```python
# Line 123
batch_size: int = 512

# Line 290
batch_size=training_section_dict.get("batch_size", 256),  # BUG: should be 512
```

**Impact:** If you load an experiment config from YAML/dict without specifying `batch_size`, you silently get 256 instead of 512. This changes training behavior (gradient noise, convergence speed, GPU utilization) without any warning.

**Fix:** Change line 290 to `256` → `512` or better, reference `TrainingSection.batch_size` default.

---

### 1.2 Dual `ConformalConfig` Classes — Incompatible APIs

**Files:**
- `src/config/training.py:193` — `ConformalConfig(enabled, alpha, method, calibration_size)`
- `src/models/calibration/conformal.py:48` — `ConformalConfig(confidence_level, method, ...)`

**The notebook (Cell 7 — Calibration):**
```python
from src.config.training import ConformalConfig  # has alpha
from src.models.calibration.conformal import ConformalPredictor  # expects confidence_level
```

**Impact:** `ConformalPredictor` internally uses its own `ConformalConfig`, which expects `confidence_level` (e.g., 0.90) not `alpha` (e.g., 0.1). The notebook creates a `ConformalConfig` from `training.py` and displays it, but if anyone tries to pass it to `ConformalPredictor`, it would silently use default values or crash.

**Fix:** Unify to one `ConformalConfig`. The `training.py` version should be the canonical one, and `conformal.py` should import from there, mapping `alpha` → `confidence_level = 1 - alpha`.

---

### 1.3 Notebook `TICK_VALUE` ≠ `SymbolConfig.from_symbol("MGC").tick_value`

**Notebook Cell 2:**
```python
TICK_VALUE = 0.10   # Display only — pipeline uses SymbolConfig.from_symbol()
```

**`src/config/symbol.py`:**
```python
SymbolConfig.from_symbol("MGC"):
    tick_value = 1.00    # $1.00 per tick
    tick_size  = 0.10    # $0.10 minimum price increment
```

**Impact:** The notebook displays transaction cost as `SLIPPAGE_TICKS * TICK_VALUE = 1.0 * 0.10 = $0.10` per trade. But the actual pipeline uses `tick_value=1.00`, meaning real slippage is `$1.00` per trade — **10x higher**. The user sees a misleading cost summary. The notebook comment says "Display only" but users will use this number to evaluate strategy viability.

**Fix:** Either:
1. Dynamically compute from `SymbolConfig.from_symbol(SYMBOL)` in the notebook, or
2. Update `TICK_VALUE = 1.00` and add comment that `tick_size=0.10` is the minimum price increment, not the tick value.

---

### 1.4 `REPO_DIR` Detection is Fragile (Cell 1)

**Code:**
```python
REPO_DIR = os.path.dirname(os.path.abspath("."))
if not os.path.exists(os.path.join(REPO_DIR, "src", "factory.py")):
    REPO_DIR = os.getcwd()
```

**Issue:** `os.path.abspath(".")` returns the CWD, and `os.path.dirname(CWD)` returns its parent. This works when Jupyter's CWD is `notebooks/` (parent = repo root) and the fallback catches the repo-root case. But:
- If CWD is `notebooks/data/` → REPO_DIR = `notebooks/` → fallback to CWD = `notebooks/data/` → **fails**
- If CWD is `~/Desktop/` → REPO_DIR = `~/` → fallback to `~/Desktop/` → **fails**

**Fix:** Use `__file__`-based detection or walk upward looking for `src/factory.py`:
```python
# Robust: walk up from notebook file location
_here = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
for _p in [_here, _here.parent, _here.parent.parent]:
    if (_p / "src" / "factory.py").exists():
        REPO_DIR = str(_p)
        break
```

---

### 1.5 `embargo_bars` Default Mismatch — Notebook vs Config vs Global YAML

| Source | `purge_bars` | `embargo_bars` |
|--------|-------------|----------------|
| Notebook Cell 2 | **10** | **60** |
| `ExperimentConfig` default | 60 | **1440** |
| `config/global.yaml` | N/A (`purge_multiplier: 3.0`) | **1440** (`min_embargo_bars`) |
| `WalkForwardConfig` in Cell 5 | `PURGE_BARS` (10) | `EMBARGO_BARS` (60) |

**Impact:** The notebook overrides both values to much smaller numbers (10/60 vs 60/1440). On 5-minute data:
- `purge_bars=10` = 50 min purge. **Likely too small** for H20 horizon (20 bars × 5min = 100 min). The purge should be ≥ the horizon to prevent label leakage.
- `embargo_bars=60` = 5 hours embargo. Reasonable, but dramatically different from the ExperimentConfig default of 1440 (5 days).

Anyone who creates an `ExperimentConfig` without the notebook gets 1440 embargo bars — a completely different leakage envelope. The global.yaml also says `min_embargo_bars: 1440`.

**Fix:**
1. Set `PURGE_BARS >= max(HORIZONS)` (i.e., ≥ 20 for H20 on 5min data).
2. Align `ExperimentConfig` defaults with what the notebook uses, or add a validation warning when `purge_bars < max(horizons)`.

---

## 2. 🟡 INCONSISTENCIES (Silent Wrong Results)

### 2.1 `TARGET_TIMEFRAME` is Set But Never Passed to Pipeline

**Notebook Cell 2:**
```python
TARGET_TIMEFRAME = "5min"
```

**Notebook Cell 5 (config construction):**
```python
mtf=MTFConfig(
    ...
    primary_timeframe=TARGET_TIMEFRAME,
)
```

This is correctly wired to `MTFConfig.primary_timeframe`. **However**, the actual data resampling from 1min → 5min bars happens *inside the pipeline* based on this setting. If the pipeline ignores `primary_timeframe` in certain code paths (e.g., direct parquet loading), the data may remain at 1min resolution while features are computed at 5min. **Verify the pipeline actually resamples based on this field.**

---

### 2.2 Experiment Name Doesn't Match Model Selection

```python
EXPERIMENT_NAME = "mgc_h100_xcb_tcn_pst"
```

But `USE_CATBOOST = False` — the "xcb" (XGBoost/CatBoost?) in the name is misleading. The actual enabled models are XGBoost, LightGBM, TCN, PatchTST. Minor but causes confusion when reviewing experiment results directories.

---

### 2.3 Walk-Forward Validation Cells Reference `fold_info` — Depends on Implementation

**Notebook Cells 25 (Model Diagnostics) and 29 (Walk-Forward Analysis)** extract per-fold metrics via:
```python
fi = getattr(oof, 'fold_info', [])
```

This assumes the `OOFPrediction` object has a `fold_info` attribute populated with per-fold F1/accuracy dicts. If the pipeline doesn't populate this (which depends on whether walk-forward vs standard CV returns fold-level data), these cells silently show "no data" instead of erroring.

**Not a bug per se, but a fragile coupling.** Add a clear warning if `fold_info` is empty when `TRAINING_MODE == "walk_forward"`.

---

### 2.4 Notebook Conformal Cell Creates Config But Never Uses It

**Cell 7 (Calibration & Conformal Prediction):**
```python
conf_config = ConformalConfig(enabled=True, alpha=CONFORMAL_ALPHA, method=CONFORMAL_METHOD)
```
This config is instantiated and then... nothing happens with it. The cell just prints a message saying "Conformal prediction is configured." It never calls `ConformalPredictor.fit()` or `predict_sets()`. This is **dead code** that gives a false sense of functionality.

---

## 3. 🟢 REPO HYGIENE ISSUES

### 3.1 Git Repo is 982 MB — Bloated with Binary Data

```
.git/  → 982 MB
data/raw/  → 162 MB of tracked .parquet files (18 files)
```

**18 parquet files** are committed directly to git, including a 77 MB `SI_1m_validated.parquet` and 27 MB `MGC_1m_5year.parquet`. Every clone downloads ~1 GB.

**Fix:** Use **Git LFS** (`git lfs track "*.parquet"`) or **DVC** (you already have a `.dvc/` directory!). The `.dvc/` directory exists but data files are tracked directly in git anyway — this suggests DVC was set up but not used.

### 3.2 Root Directory has 11 Orphan Markdown Files

```
AUDIT_2026-02-26.md   CHILL.md          CLAUDE.md
CLEANUP_PLAN.md       CLEANUP_TASKS.md  COMPLETION.md
DIRECTION.md          EndtoEndtests.md  FEATURE_SELECTION_AUDIT.md
HARDCODEFIXES.md      SNEH.md
```

Plus `docs/` has 7 more `.md` files, `X ( IN PROGRESS DOCS) X/` has more, and `THINGS TO HANDLE AFTER THE REPO IS ORGANIZED./XX/` has 5 more. That's **23+ markdown docs** scattered across the repo.

**Fix:** Consolidate into `docs/` with a clear index. Delete stale/completed documents. A production repo should have: `README.md`, `CHANGELOG.md`, `docs/`, and nothing else at root.

### 3.3 93 Phases of Commits — No Semantic Versioning

The last 30 commits show "Phase 80" through "Phase 93". Each phase is a `fix:` or `feat:` commit. There are no tags, releases, or version numbers. This makes it impossible to:
1. Roll back to a known-good state
2. Track which phase introduced a regression
3. Deploy a specific version

**Fix:** Add git tags for milestone phases (e.g., `v0.1.0` = Phase 80, `v0.2.0` = Phase 90).

---

## 4. PRODUCTION READINESS GAPS

### 4.1 No CI/CD Pipeline

The repo has a `.github/` directory but no visible workflows for:
- Running tests on push/PR
- Linting (ruff config exists in `.ruff_cache/` but no CI integration)
- Type checking (`.mypy_cache/` exists but no CI integration)

**Impact:** Regressions can be introduced silently. Tests exist (223 collected) but aren't enforced.

### 4.2 No `README.md` at Root

The root `README.md` was deleted in a previous commit (Phase 85+). The repo has no entry point documentation.

### 4.3 Test Coverage is Narrow

**9 test files, 223 tests** — but they cover:
- Triple barrier labeling ✅
- Purged K-Fold ✅
- Leakage detection ✅
- Financial metrics ✅
- Model smoke tests ✅
- Lookahead audit ✅

**Missing test coverage:**
- `src/factory.py` (MLFactory) — the main entry point
- `src/config/experiment.py` — config construction and serialization
- `src/inference/deploy.py` — deploy artifact creation/loading
- `src/models/calibration/conformal.py` — conformal prediction
- Walk-forward validation end-to-end
- Notebook cell execution (no integration test)

### 4.4 No Logging Configuration

The notebook doesn't configure Python's `logging` module. All `src/` modules use `logging.getLogger(__name__)`, but without a handler configured, log messages are silently dropped. In production, this means:
- Training warnings are invisible
- OOM recovery attempts aren't logged
- Drift detection alerts aren't captured

**Fix:** Add to Cell 1:
```python
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
```

### 4.5 Conformal Prediction is Incomplete

The notebook advertises conformal prediction prominently (Cell 1 docs, Cell 2 config, Cell 7 display) but:
1. The calibration cell (Cell 7) creates a config and prints instructions but **never runs** conformal prediction
2. There's no integration between the factory's training pipeline and conformal prediction
3. No test coverage for conformal prediction

This feature is **scaffolded but not wired end-to-end**.

### 4.6 Leakage Detection Depends on Saved Files

Cell 8 (Leakage Detection) looks for `features.parquet` and `labels.npy` in the output directory. If the pipeline doesn't save these files (which depends on implementation), leakage detection silently skips with a polite message. This is a critical safety check that should be mandatory, not optional.

---

## 5. COMMIT HISTORY OBSERVATIONS

### Recent Phases (80–93) Show Rapid Iteration

| Phase | Focus |
|-------|-------|
| 80 | Critical fixes: label balance, memory, early stopping, TCN |
| 81 | Dead notebook cells (calibration, leakage, feature importance) |
| 82 | Checkpoint resume for 4D models |
| 83 | Feature selection min_frequency wiring |
| 84 | Logloss metrics + binary classification |
| 85 | Full 8-agent audit fixes |
| 86 | Backtest barrier alignment (ATR-based stop/profit) |
| 87 | DSR metric gate |
| 88 | Thread-safe label cache, execution guards |
| 89 | Pipeline speed optimizations (3-5x) |
| 90 | CUDA memory guards |
| 91 | Gradient checkpointing + TFT flash attention |
| 92 | Optuna robustness + sequential ensemble |
| 93 | Per-symbol ADX regime thresholds |

**Pattern:** Most phases are fixes to previously broken features. This suggests rapid AI-assisted development without sufficient testing between phases. The Phase 81 commit ("fix 5 dead notebook cells") indicates cells were added but never executed.

### Concerns

1. **No revert history** — if Phase 92's Optuna changes broke something, there's no way to identify the regression without reading the diff
2. **Coupled commits** — "Phase 86" modifies both `execution.py` and `costs.py` in the same commit. These should be separate atomic commits
3. **Missing test commits** — none of the Phase 80–93 commits include new tests for the features they add/fix

---

## 6. NOTEBOOK STRUCTURE ANALYSIS

### Cell Map (34 cells)

| # | Type | Title | Status |
|---|------|-------|--------|
| 1 | MD | Title + Quick Start docs | ✅ Good |
| 2 | Code | Setup (clone, install, imports) | 🟡 REPO_DIR fragile |
| 3 | Code | Configuration (all params) | 🟡 TICK_VALUE wrong |
| 4 | Code | Validate configuration | ✅ Good |
| 5 | Code | Load & preview data | ✅ Good |
| 6 | MD | EDA header | ✅ Good |
| 7 | Code | EDA: price, volume, returns | ✅ Good |
| 8 | Code | Run ML Factory | ✅ Core pipeline |
| 9 | Code | Results & visualization | ✅ Good |
| 10 | MD | Calibration header | ✅ Good |
| 11 | Code | Calibration & conformal | 🟡 Dead conformal code |
| 12 | MD | Leakage header | ✅ Good |
| 13 | Code | Leakage detection | 🟡 Depends on saved files |
| 14 | Code | Deploy artifact | ✅ Good |
| 15 | MD | Model comparison header | ✅ Good |
| 16 | Code | Model comparison chart | ✅ Good |
| 17 | MD | Feature importance header | ✅ Good |
| 18 | Code | Feature importance viz | ✅ Good |
| 19 | MD | Barrier alignment docs | ✅ Good |
| 20 | MD | Backtest header | ✅ Good |
| 21 | Code | Backtest equity + drawdown | ✅ Good |
| 22 | MD | Trading analytics header | ✅ Good |
| 23 | Code | Trading analytics (monthly, P&L, streaks) | ✅ Good |
| 24 | MD | Model diagnostics header | ✅ Good |
| 25 | Code | Confusion matrix + calibration + fold stability | ✅ Good |
| 26 | MD | Multi-model insights header | ✅ Good |
| 27 | Code | Radar chart + agreement matrix | ✅ Good |
| 28 | MD | Walk-forward header | ✅ Good |
| 29 | Code | Walk-forward window analysis | ✅ Good |
| 30 | MD | Interpreting results guide | ✅ Good |
| 31 | MD | Save & export header | ✅ Good |
| 32 | Code | Google Drive save | ✅ Good |
| 33 | Code | Inference-only export | ✅ Good |
| 34 | Code | Save & download results | ✅ Good |

**Overall:** The notebook is well-organized with clear headers and good error handling (try/except in all cells). The main issues are the 5 bugs above, not structural problems.

---

## 7. RECOMMENDED FIXES (Priority Order)

### Immediate (Before Next Run)

1. **Fix `PURGE_BARS`** — Set to `>= max(HORIZONS)` (i.e., ≥ 20). Current value of 10 allows label leakage for H20.
2. **Fix `TICK_VALUE`** — Change to `1.00` or dynamically load from `SymbolConfig.from_symbol(SYMBOL).tick_value`.
3. **Fix `batch_size` default in `from_dict()`** — Change 256 → 512 in `experiment.py:290`.
4. **Unify `ConformalConfig`** — Remove duplicate in `conformal.py`, import from `training.py`.

### Short-Term (This Week)

5. **Add root `README.md`** — A production repo needs an entry point.
6. **Wire conformal prediction end-to-end** — Or remove it from the notebook docs if it's not ready.
7. **Add logging config** to Cell 1 — `logging.basicConfig(level=logging.INFO)`.
8. **Add git tags** for key phases — `git tag v0.2.0-phase93`.
9. **Add `purge_bars >= max(horizons)` validation** to `ExperimentConfig.__post_init__()`.

### Medium-Term (This Month)

10. **Move parquets to Git LFS or DVC** — 982 MB git repo is unsustainable.
11. **Consolidate 23+ markdown docs** into `docs/` with an index.
12. **Add CI/CD** — GitHub Actions for pytest + ruff + mypy on push.
13. **Add integration test** — One test that runs the notebook end-to-end on a small dataset.
14. **Add tests for** `MLFactory`, `ExperimentConfig`, `deploy.py`, `conformal.py`.

---

## 8. WHAT'S WORKING WELL

- **All 34 notebook cells have proper error handling** (try/except with traceback)
- **All notebook imports resolve to real `src/` modules** — no broken imports
- **Config architecture is clean** — `ExperimentConfig` composes `DataSection`, `TrainingSection`, etc.
- **Feature selection is properly wired** — per-model MDA with configurable frequency threshold
- **Walk-forward validation is properly configured** — embargo + purge passed through
- **Transaction costs are passed to backtest** — commission and slippage properly wired
- **Deploy artifact system is complete** — manifest-based loading with validation
- **OOM recovery is implemented** — batch reduction with retry logic
- **Gradient checkpointing for large models** — TFT flash attention support
- **Per-symbol ADX regime thresholds** — MGC/MES have different trending thresholds
- **223 tests collected successfully** — no collection errors

---

*Generated by automated audit — March 1, 2026*
