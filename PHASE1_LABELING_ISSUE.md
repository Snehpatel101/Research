# Phase 1 Labeling Issue - Critical Finding

**Date:** 2025-12-27
**Run ID:** 20251227_213131
**Status:** 🔴 **CRITICAL - Requires Investigation**

---

## Issue Summary

Phase 1 pipeline completed successfully (12/12 stages), but **all generated labels are NEUTRAL (class 0)**. No LONG or SHORT labels were created.

---

## Label Distribution Analysis

### Actual Distribution (Horizon 20):

| Split | Total Samples | Valid | Invalid (-99) | LONG (1) | NEUTRAL (0) | SHORT (-1) |
|-------|---------------|-------|---------------|----------|-------------|------------|
| Train | 13,758 | 13,758 | 0 | **0 (0.0%)** | **13,758 (100%)** | **0 (0.0%)** |
| Val | 1,461 | 1,461 | 0 | **0 (0.0%)** | **1,461 (100%)** | **0 (0.0%)** |
| Test | 1,521 | 1,481 | 40 | **0 (0.0%)** | **1,481 (100%)** | **0 (0.0%)** |

### Expected Distribution (from CLAUDE.md):

According to Phase 1 documentation, we should see approximately:
- **LONG:** 30-40% (barrier hit upward)
- **NEUTRAL:** 20-40% (timeout without hitting barrier)
- **SHORT:** 30-40% (barrier hit downward)

---

## Impact on Model Training

### Observed Symptoms:

1. **Perfect validation metrics (1.0000)** - meaningless, just predicting majority class
2. **Random Forest training failure** - `log_loss` requires multiple classes
3. **XGBoost/LSTM/Transformer all achieve 1.0000 accuracy** - trivial task

### Why This Happens:

All models predict NEUTRAL (class 0) for all samples → 100% accuracy since all labels are NEUTRAL.

---

## Root Cause Investigation

### Potential Issues in Phase 1:

1. **Triple-Barrier Logic** (`src/phase1/stages/labeling/`)
   - Barriers may be set too wide (never get hit)
   - Or barrier thresholds calculated incorrectly

2. **GA Optimization** (`src/phase1/stages/ga_optimize/`)
   - Optuna may have optimized barriers to extreme values
   - Check: `config/ga_results/optimization_summary.json`

3. **MGC-Specific Barriers** (`src/phase1/config/barriers_config.py`)
   - MGC uses **symmetric barriers** (1.0:1.0 k_up:k_down)
   - May not be appropriate for actual MGC price movement

4. **Barrier Scaling**
   - ATR-based barriers might be calculated incorrectly
   - 5-minute ATR may be too large relative to price movements

---

## Verification Steps

### 1. Check GA Optimization Results

```bash
cat config/ga_results/optimization_summary.json | jq '.best_params'
```

**Expected:** Reasonable barrier multipliers (k_up: 0.5-2.0, k_down: 0.5-2.0)
**If Found:** Extreme values (k_up: 10+, k_down: 10+) → barriers never get hit

### 2. Check Raw Labeling Output

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/labeling/MGC_initial_labels.parquet')
print(df['label_h20'].value_counts())
"
```

**Expected:** Mix of -1, 0, 1
**If Found:** All 0 → confirms issue in triple-barrier stage

### 3. Check Barrier Sizes

```python
from src.phase1.config.barriers_config import get_barrier_params
params = get_barrier_params('MGC', horizon=20)
print(f"k_up: {params['k_up']}, k_down: {params['k_down']}")
print(f"max_bars: {params['max_bars_ahead']}")
```

### 4. Manual Barrier Calculation

```python
# Example: If ATR = $10 and k_up = 1.5
# Upper barrier = current_price + (1.5 * $10) = current_price + $15
# For MGC ~$1800, this is a +0.83% move to hit barrier
#
# Question: Is 20-bar (100 minutes at 5min) enough time to see +0.83% move?
```

---

## Recommended Fixes

### Option 1: Adjust Barrier Parameters

Reduce barrier widths for MGC (gold is mean-reverting, needs tighter barriers):

```python
# In src/phase1/config/barriers_config.py
BARRIER_PARAMS = {
    'MGC': {
        20: {
            'k_up': 0.5,    # Reduce from 1.0
            'k_down': 0.5,  # Reduce from 1.0
            'max_bars_ahead': 60  # Increase timeout
        }
    }
}
```

### Option 2: Re-run GA Optimization

Force GA to explore tighter barrier ranges:

```python
# In GA optimization stage
param_ranges = {
    'k_up': (0.2, 2.0),    # Previously might have been (0.5, 5.0)
    'k_down': (0.2, 2.0),
    'max_bars': (20, 100)
}
```

### Option 3: Use Fixed Percentage Barriers

Instead of ATR-based, use fixed percentage:

```python
# Upper barrier = price * (1 + 0.003)  # +0.3%
# Lower barrier = price * (1 - 0.003)  # -0.3%
```

---

## Next Steps

1. ✅ **Document issue** (this file)
2. ⬜ **Investigate GA optimization results**
3. ⬜ **Check raw labeling output**
4. ⬜ **Test with adjusted barrier parameters**
5. ⬜ **Re-run Phase 1 with fixes**
6. ⬜ **Verify label distribution after fix**

---

## Files to Investigate

```
config/ga_results/optimization_summary.json     # Check optimized parameters
src/phase1/config/barriers_config.py            # MGC barrier configuration
src/phase1/stages/labeling/triple_barrier.py    # Core labeling logic
src/phase1/stages/ga_optimize/run.py            # GA optimization
runs/20251227_213131/logs/pipeline.log          # Full pipeline logs
```

---

## Expected vs Actual

### Expected Label Distribution (Healthy):
```
LONG:    ████████████  35%
NEUTRAL: ████████      25%
SHORT:   ████████████  40%
```

### Actual Label Distribution (Current):
```
LONG:                  0%
NEUTRAL: ████████████████████████████████  100%
SHORT:                 0%
```

---

## Impact on Project

- ⚠️ **Model training is technically working** but learning a trivial task
- ⚠️ **All perfect metrics are meaningless** (predicting majority class)
- 🔴 **Cannot evaluate model performance** until labeling is fixed
- 🔴 **Phase 2-4 blocked** by Phase 1 data quality issue

**Priority:** HIGH - Fix before continuing with model experiments
