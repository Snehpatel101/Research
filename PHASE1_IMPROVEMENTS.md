# Phase 1 Improvements: Validated Roadmap

> Last validated: 2025-12-24
> Scope: Data pipeline only (ingest → splits → validation). Model training is Phase 2+.

---

## Executive Summary

Two specialized agents validated the previous static analysis findings against the actual codebase.

**Results:**
- 7 findings CONFIRMED (need fixing)
- 1 finding FALSE POSITIVE (intentional design)
- 4 CRITICAL inconsistencies identified

---

## Part A: Bug Fixes (Must Fix Before Phase 2)

### A1. CLI Stage Mapping Broken [HIGH]

**Problem:** `pipeline rerun --from labeling` fails because stage aliases don't match actual stage names.

**Location:** `src/cli/run_commands.py:414-423`

**Current (broken):**
```python
stage_map = {
    'labeling': 'labeling',      # Stage doesn't exist
    'labels': 'labeling',        # Stage doesn't exist
}
```

**Actual stages in registry:** `initial_labeling`, `ga_optimize`, `final_labels`

**Fix:**
```python
stage_map = {
    'labeling': 'initial_labeling',
    'labels': 'final_labels',
    'initial_labeling': 'initial_labeling',
    'final_labels': 'final_labels',
    'ga': 'ga_optimize',
    'ga_optimize': 'ga_optimize',
    'scaling': 'feature_scaling',
    'datasets': 'build_datasets',
    'validate': 'validate_scaled',
}
```

---

### A2. Status Command Reports Wrong Stage Count [MEDIUM]

**Problem:** Status shows "3/6 stages" when actually "6/12 stages" complete.

**Locations:**
- `src/cli/status_commands.py:130-137` - Wrong stage list
- `src/cli/status_commands.py:383` - Hardcoded `total = 6`

**Fix:** Update `all_stages` to match actual 12 stages from `src/pipeline/runner.py:113-128`

---

### A3. PipelineConfig Project Root Wrong [HIGH]

**Problem:** Default project_root resolves to `src/` instead of project root.

**Location:** `src/phase1/pipeline_config.py:122-123`

**Current (broken):**
```python
self.project_root = Path(__file__).parent.parent.resolve()
# Resolves to: Research/src/  (WRONG)
```

**Fix:**
```python
self.project_root = Path(__file__).parent.parent.parent.resolve()
# Resolves to: Research/  (CORRECT)
```

Also fix line 354-355 (same issue).

---

### A4. Labeling Report Path Mismatch [MEDIUM]

**Problem:** Report is written to one path, but artifact collection looks in another.

**Locations:**
- `src/phase1/stages/final_labels/core.py:395-396` - Writes to `src/phase1/results/`
- `src/phase1/stages/final_labels/run.py:217-219` - Looks in `config.results_dir/`

**Fix:** Use `config.results_dir` consistently:
```python
# In core.py, accept results_dir parameter instead of computing it
def generate_labeling_report(df, results_dir: Path, ...):
    report_path = results_dir / 'labeling_report.md'
```

---

### A5. Horizons Hardcoded [5, 20] in Report [MEDIUM]

**Problem:** Report generator ignores configured horizons `[5, 10, 15, 20]`.

**Location:** `src/phase1/stages/final_labels/core.py:335`

**Current:**
```python
for horizon in [5, 20]:  # Hardcoded!
```

**Fix:**
```python
# Detect from DataFrame columns or accept as parameter
label_cols = [c for c in df.columns if c.startswith('label_h')]
horizons = sorted(set(int(c.replace('label_h', '')) for c in label_cols))
for horizon in horizons:
```

---

## Part B: Known Limitations (Document, Don't Block)

### B1. get_regime_adjusted_barriers Not Implemented

**Status:** Placeholder exists, falls back to defaults.

**Location:** `src/phase1/stages/labeling/adaptive_barriers.py:144-153`

**Impact:** Regime-adaptive labeling always uses default barriers. Not blocking for Phase 1.

**Phase 1 Action:** Document as known limitation.
**Phase 2 Action:** Implement HMM regime detection + barrier adjustment.

---

### B2. Two Purge/Embargo Implementations Exist

**Locations:**
- `src/phase1/config/features.py:95-129` - Uses actual max_bars
- `src/common/horizon_config.py:208-271` - Uses multiplier formula

**Impact:** Both converge for default horizons. Potential confusion for custom configurations.

**Phase 1 Action:** Document which is authoritative.
**Phase 2 Action:** Consolidate into single source.

---

### B3. Advanced Regime Features Use Fallback Pattern [FALSE POSITIVE]

**Location:** `src/phase1/stages/features/regime.py:147-168`

**Analysis:** This is INTENTIONAL defensive programming. Warning is logged, fallback is valid.

**Action:** No fix needed. This is correct behavior.

---

## Part C: Feature Improvements (Phase 1 Scope)

### C1. Replace DEAP GA with Optuna TPE [HIGH IMPACT]

**Current:** DEAP genetic algorithm for barrier optimization.

**Problem:** GA converges slower, harder to reproduce, 27% less efficient than Bayesian methods.

**Evidence:** [Nature study](https://www.nature.com/articles/s41598-025-29383-7) - Hybrid BayGA outperforms pure GA.

**Implementation:**
```python
# src/phase1/stages/ga_optimize/ → optuna_optimize/
import optuna

def optimize_barriers(symbol: str, df: pd.DataFrame, config: dict) -> dict:
    def objective(trial):
        pt = trial.suggest_float('profit_take', 0.001, 0.02)
        sl = trial.suggest_float('stop_loss', 0.001, 0.02)
        # ... apply barriers, compute fitness
        return -sharpe  # Minimize negative sharpe

    study = optuna.create_study(sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=100)
    return study.best_params
```

**Effort:** Low (same interface, better algorithm)

---

### C2. Add Wavelet Decomposition Features [HIGH IMPACT]

**Current:** MTF resampling captures different timeframes.

**Problem:** Pure resampling misses non-stationary patterns wavelets capture.

**Evidence:** [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2667096825000424) - Wavelets "especially suitable for multi-scale analysis."

**Implementation:**
```python
# src/phase1/stages/features/wavelets.py (NEW)
import pywt

def add_wavelet_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Add discrete wavelet transform features."""
    for col in ['close', 'volume']:
        # Daubechies wavelet, 3 levels
        coeffs = pywt.wavedec(df[col].values, 'db4', level=3)
        df[f'{col}_wavelet_approx'] = pywt.waverec([coeffs[0]] + [None]*3, 'db4')[:len(df)]
        for i, detail in enumerate(coeffs[1:], 1):
            reconstructed = pywt.waverec([None] + [None]*(i-1) + [detail] + [None]*(3-i), 'db4')
            df[f'{col}_wavelet_d{i}'] = reconstructed[:len(df)]
    return df
```

**Effort:** Low (add new feature module)

---

### C3. Add Microstructure Proxy Features [MEDIUM IMPACT]

**Current:** No microstructure features.

**Problem:** Missing volume-price relationship signals available from OHLCV.

**Implementation:**
```python
# src/phase1/stages/features/microstructure.py (NEW)
def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add microstructure proxy features from OHLCV."""
    # Amihud illiquidity (proxy)
    df['amihud'] = df['close'].pct_change().abs() / (df['volume'] + 1e-10)

    # Roll spread estimator (proxy)
    delta_price = df['close'].diff()
    df['roll_spread'] = 2 * np.sqrt(-delta_price.cov(delta_price.shift(1)))

    # Kyle's lambda proxy (price impact)
    df['kyle_lambda'] = df['close'].pct_change().abs() / df['volume'].rolling(20).mean()

    return df
```

**Effort:** Low (add new feature module)

---

### C4. Enable Cross-Asset Features [MEDIUM IMPACT]

**Current:** Implemented but disabled by default.

**Problem:** MES-MGC correlation provides predictive signal.

**Fix:** Enable in default config:
```python
# src/phase1/pipeline_config.py
enable_cross_asset_features: bool = True  # Changed from False
```

**Effort:** Trivial (config change)

---

## Part D: Out of Scope (Phase 2+)

These improvements are NOT Phase 1:

| Improvement | Phase | Reason |
|-------------|-------|--------|
| Meta-labeling | 2 | Requires trained primary model |
| CPCV/PBO validation | 3 | Cross-validation framework |
| Data augmentation | 2 | Model training concern |
| Concept drift monitoring | 3/5 | Production monitoring |
| Autoencoder features | 2 | Learned features need model |
| Foundation models | 2+ | Model family work |

---

## Priority Execution Order

### Immediate (Before Any More Runs)
1. **A3** - Fix PipelineConfig project_root
2. **A1** - Fix CLI stage mapping

### Before Phase 2
3. **A2** - Fix status stage count
4. **A4** - Fix labeling report path
5. **A5** - Fix hardcoded horizons

### Phase 1 Enhancements (Optional)
6. **C1** - Replace GA with Optuna (high impact, low effort)
7. **C2** - Add wavelet features (high impact, low effort)
8. **C4** - Enable cross-asset features (trivial)
9. **C3** - Add microstructure features (medium impact, low effort)

### Document Only
10. **B1** - Document regime barrier limitation
11. **B2** - Document authoritative purge/embargo source

---

## Validation Checklist

After fixes, verify:

```bash
# 1. CLI rerun works
./pipeline rerun <run_id> --from initial_labeling

# 2. Status shows correct stages
./pipeline status <run_id>
# Should show: X/12 stages complete

# 3. Pipeline runs end-to-end
./pipeline run --symbols MES,MGC

# 4. Report includes all horizons
cat runs/<run_id>/results/labeling_report.md
# Should show H5, H10, H15, H20

# 5. Tests pass
python -m pytest tests/phase_1_tests/ -q
```

---

## Agent Validation Notes

**Python Expert Agent (Static Analysis):**
- 7/8 findings confirmed
- #8 (fragile imports) is FALSE POSITIVE - intentional design

**Backend Architect Agent (Consistency):**
- 4 critical inconsistencies identified
- Path consistency is proper
- Stage registry is properly aligned (CLI is not)

Both agents agree: **Fix A1-A5 before Phase 2.**
