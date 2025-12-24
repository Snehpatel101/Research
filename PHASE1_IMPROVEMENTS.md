# Phase 1 Improvements: Validated Roadmap

> Last validated: 2025-12-24 (2nd pass after user fixes)
> Scope: Data pipeline only (ingest → splits → validation). Model training is Phase 2+.

---

## Executive Summary

Two sequential agents revalidated the codebase after user fixes.

**Current State:**
| Category | Count |
|----------|-------|
| FIXED | 4 |
| PARTIALLY FIXED | 1 |
| STILL OPEN | 1 |
| NOT A BUG | 1 |

---

## Part A: Bug Status After Fixes

### A1. CLI Stage Mapping ✅ FIXED

**Was:** `pipeline rerun --from labeling` failed.

**Now:** `src/cli/run_commands.py:414-434` includes all 12 stages:
- `initial_labeling`, `ga_optimize`, `final_labels`
- `feature_scaling`, `build_datasets`, `validate_scaled`

**No action needed.**

---

### A2. Status Command Stage Count ⚠️ STILL OPEN

**Problem:** Status hardcodes `total = 6` when pipeline has 12 stages.

**Location:** `src/cli/status_commands.py:386`

**Current:**
```python
total = 6  # Total stages
```

**Fix:**
```python
total = len(all_stages)  # Dynamic from registry
```

---

### A3. PipelineConfig Project Root ⚠️ PARTIALLY FIXED

**__post_init__ (line 123):** ✅ FIXED - Uses `.parent.parent.parent` (3 levels)

**load_from_run_id (line 355):** ❌ STILL BROKEN - Uses `.parent.parent` (2 levels → resolves to `src/`)

**Fix line 355:**
```python
project_root = Path(__file__).parent.parent.parent.resolve()  # 3 levels
```

---

### A4. Labeling Report Path ✅ FIXED

**Was:** Path mismatch between write and read.

**Now:** Both `core.py:315` and `run.py:216` use consistent `output_dir`/`config.results_dir`.

**No action needed.**

---

### A5. Horizons Hardcoded ✅ FIXED

**Was:** Hardcoded `[5, 20]` in report generator.

**Now:** `core.py:347` accepts `horizons: list[int]` parameter and iterates dynamically.

**No action needed.**

---

### A6. get_regime_adjusted_barriers ✅ FIXED

**Was:** Function missing, always fell back to defaults.

**Now:** Fully implemented in `src/phase1/config/regime_config.py:62-115` with regime multipliers. Exported in `config/__init__.py:33`.

**No action needed.**

---

### A7. Duplicate Purge/Embargo ✅ NOT A BUG

**Analysis:** `runtime.py:33` calls `auto_scale_purge_embargo()` from `horizon_config.py`. Proper separation of function definition and usage.

**No action needed.**

---

## Part B: Documentation Issues Found

Second agent reviewed all docs for accuracy:

| Document | Score | Issues |
|----------|-------|--------|
| `docs/phase1/README.md` | 7/10 | Feature count claim (129) is dynamic |
| `docs/README.md` | 6/10 | Embargo formula incomplete |
| `docs/getting-started/QUICKSTART.md` | 8/10 | Minor |
| `docs/getting-started/PIPELINE_CLI.md` | 9/10 | Accurate |
| `docs/reference/ARCHITECTURE.md` | 9/10 | Accurate |
| `docs/reference/FEATURES.md` | 8/10 | Accurate |
| **CLAUDE.md** | **5/10** | **Critical issues** |

### CLAUDE.md Critical Fixes Needed

| Line | Current (WRONG) | Should Be |
|------|-----------------|-----------|
| ~238 | `src/stages/stage1_ingest.py` | `src/phase1/stages/ingest/` |
| ~293 | `LABEL_HORIZONS = [5, 20]` | `[5, 10, 15, 20]` |
| ~296 | `EMBARGO_BARS = 288` | `1440` (auto-scaled) |

---

## Part C: Feature Improvements (Phase 1 Scope)

### C1. Replace DEAP GA with Optuna TPE [HIGH IMPACT]

**Problem:** GA converges slower, 27% less efficient than Bayesian methods.

**Evidence:** [Nature study](https://www.nature.com/articles/s41598-025-29383-7)

**Implementation:**
```python
# src/phase1/stages/ga_optimize/ → optuna_optimize/
import optuna

def optimize_barriers(symbol: str, df: pd.DataFrame, config: dict) -> dict:
    def objective(trial):
        pt = trial.suggest_float('profit_take', 0.001, 0.02)
        sl = trial.suggest_float('stop_loss', 0.001, 0.02)
        return -sharpe  # Minimize negative sharpe

    study = optuna.create_study(sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=100)
    return study.best_params
```

**Effort:** Low

---

### C2. Add Wavelet Decomposition Features [HIGH IMPACT]

**Problem:** Pure resampling misses non-stationary patterns.

**Evidence:** [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2667096825000424)

**Implementation:**
```python
# src/phase1/stages/features/wavelets.py (NEW)
import pywt

def add_wavelet_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['close', 'volume']:
        coeffs = pywt.wavedec(df[col].values, 'db4', level=3)
        df[f'{col}_wavelet_approx'] = pywt.waverec([coeffs[0]] + [None]*3, 'db4')[:len(df)]
        for i, detail in enumerate(coeffs[1:], 1):
            reconstructed = pywt.waverec([None] + [None]*(i-1) + [detail] + [None]*(3-i), 'db4')
            df[f'{col}_wavelet_d{i}'] = reconstructed[:len(df)]
    return df
```

**Effort:** Low

---

### C3. Add Microstructure Proxy Features [MEDIUM IMPACT]

**Implementation:**
```python
# src/phase1/stages/features/microstructure.py (NEW)
def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    # Amihud illiquidity proxy
    df['amihud'] = df['close'].pct_change().abs() / (df['volume'] + 1e-10)
    # Roll spread estimator
    delta_price = df['close'].diff()
    df['roll_spread'] = 2 * np.sqrt(-delta_price.cov(delta_price.shift(1)))
    # Kyle's lambda proxy
    df['kyle_lambda'] = df['close'].pct_change().abs() / df['volume'].rolling(20).mean()
    return df
```

**Effort:** Low

---

### C4. Enable Cross-Asset Features [MEDIUM IMPACT]

**Current:** Implemented but disabled.

**Fix:** Enable in config (trivial change).

---

## Part D: Out of Scope (Phase 2+)

| Improvement | Phase | Reason |
|-------------|-------|--------|
| Meta-labeling | 2 | Requires trained primary model |
| CPCV/PBO validation | 3 | Cross-validation framework |
| Data augmentation | 2 | Model training concern |
| Concept drift monitoring | 3/5 | Production monitoring |
| Autoencoder features | 2 | Learned features need model |

---

## Priority Execution Order

### Remaining Bugs (2 items)
1. **A2** - Fix status stage count (`total = 6` → dynamic)
2. **A3** - Fix `load_from_run_id` project_root (add one more `.parent`)

### Documentation (3 items)
3. Fix CLAUDE.md stage paths
4. Fix CLAUDE.md horizons `[5,20]` → `[5,10,15,20]`
5. Fix CLAUDE.md embargo `288` → `1440`

### Feature Enhancements (4 items)
6. **C1** - Replace GA with Optuna (high impact)
7. **C2** - Add wavelet features (high impact)
8. **C4** - Enable cross-asset (trivial)
9. **C3** - Add microstructure features (medium impact)

---

## Validation Checklist

```bash
# 1. Pipeline runs end-to-end
./pipeline run --symbols MES,MGC

# 2. Status shows correct count (after A2 fix)
./pipeline status <run_id>
# Should show: X/12 stages

# 3. Tests pass
python -m pytest tests/phase_1_tests/ -q

# 4. load_from_run_id works (after A3 fix)
python -c "from src.phase1.pipeline_config import PipelineConfig; c = PipelineConfig.load_from_run_id('test'); print(c.project_root)"
```

---

## Agent Validation Summary

**Python Pro Agent (Revalidation):**
- 4/7 FIXED
- 1/7 PARTIALLY FIXED (A3 - one location remains)
- 1/7 STILL OPEN (A2 - status count)
- 1/7 NOT A BUG (purge/embargo separation is correct)

**Backend Architect Agent (Documentation):**
- CLAUDE.md needs 3 critical updates
- Other docs score 6-9/10 (acceptable)
- Stage names and CLI are accurate
- Architecture diagrams are accurate
