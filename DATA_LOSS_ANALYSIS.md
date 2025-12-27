# Data Loss Analysis: 2 Commits Ago vs Current

**Date:** 2025-12-27
**Comparison:** Commit 30e2454 (Dec 24) vs 7a6e410 (Dec 27)
**Status:** 🔴 **CRITICAL - 44% Sample Reduction**

---

## Executive Summary

The current pipeline produces **44% fewer samples** (35,388 → 19,740) than 2 commits ago, despite starting **70 days earlier**. Root cause: **overly aggressive NaN handling** in feature engineering dropping 86% of data.

---

## Numerical Comparison

### Sample Counts

| Metric | 2 Commits Ago | Current | Change |
|--------|---------------|---------|--------|
| **Total Samples** | 35,388 | 19,740 | **-15,648 (-44.2%)** |
| Train | 24,711 | 13,758 | -10,953 (-44.3%) |
| Val | 3,808 | 1,461 | -2,347 (-61.6%) |
| Test | 3,869 | 1,521 | -2,348 (-60.7%) |

### Date Ranges

| Metric | 2 Commits Ago | Current | Change |
|--------|---------------|---------|--------|
| **Start Date** | 2020-03-12 | 2020-01-02 | **+70 days earlier** |
| **End Date** | 2021-12-01 | 2021-12-01 | Same |
| **Duration** | 629 days | 699 days | +70 days |
| **Run ID** | 20251224_210811 | 20251227_213131 | - |

### The Paradox

**Despite starting 70 days earlier (Jan-Mar 2020 data), we have 44% FEWER samples!**

---

## Root Cause: Feature Engineering NaN Handling

### Data Loss Through Pipeline Stages

| Stage | Rows | Loss | % Retention |
|-------|------|------|-------------|
| Raw 1-minute data | 690,000 | - | 100.0% |
| Clean 5-minute data | 140,994 | -549,006 | 20.4% |
| **Feature engineering** | **19,740** | **-121,254** | **14.0%** |

**Critical Finding:** Feature engineering stage drops **86% of data** (140,994 → 19,740 rows)

### Evidence from Pipeline Logs

```
2025-12-27 21:38:22,417 - INFO - NaN cleanup: 163 cols -> 162 cols (-1 dropped)
2025-12-27 21:38:22,417 - INFO - NaN cleanup: 140,994 rows -> 19,740 rows (-121,254 dropped, 86.0%)
```

**NaN Handling Strategy:** Drop ANY row that has ANY NaN in ANY feature column

---

## Why So Much NaN?

### Expected NaN (Lookback Windows)

| Feature | Lookback | NaN Rows |
|---------|----------|----------|
| Daily MTF | 1 day | ~288 bars |
| SMA(200) | 200 bars | 200 bars |
| Wavelet decomposition | Window 64 | 64 bars |
| EMA(50) | 50 bars | 50 bars |
| 4h MTF | 4 hours | ~48 bars |

**Expected max lookback:** ~288 bars → ~99.8% retention

### Actual Results

- **Expected valid rows:** ~140,706 (99.8% retention)
- **Actual valid rows:** 19,740 (14.0% retention)
- **Unexplained loss:** 120,966 rows (85.6%)

---

## Hypothesis: Scattered NaN Throughout Dataset

The 86% data loss suggests NaN is not just at the beginning (lookback) but **scattered throughout the entire dataset**.

### Possible Causes:

1. **Wavelet Edge Effects**
   - Wavelet decomposition creates NaN at chunk boundaries
   - With window=64, could affect many rows

2. **MTF Alignment Issues**
   - Multi-timeframe features may have NaN when timeframes don't align
   - Daily features may have gaps during market closures

3. **Autocorrelation Failures**
   - Log shows: `return_autocorr_lag20` has 100% NaN (dropped)
   - Other autocorr features have 50-90% NaN

4. **Bollinger Band Issues**
   - `bb_width`, `bb_position`, `close_bb_zscore` have 50-90% NaN
   - May indicate calculation failures or insufficient data

### From Pipeline Logs:

```
WARNING - Columns with 100% NaN (1): ['return_autocorr_lag20']
INFO - Columns with 50-90% NaN (10):
  ['bb_width', 'bb_position', 'close_bb_zscore',
   'return_autocorr_lag1', 'return_autocorr_lag2']... and 5 more
```

**If 10 columns have 50-90% NaN, dropping rows with ANY NaN will decimate the dataset!**

---

## Comparison: Old vs New Pipeline

### Old Pipeline (2 Commits Ago)

- **Input:** 140,994 clean 5-min bars
- **Feature set:** Unknown (likely simpler)
- **Final samples:** 35,388 (25% retention)
- **Inference:** Simpler features or better NaN handling

### New Pipeline (Current)

- **Input:** 140,994 clean 5-min bars (same)
- **Feature set:** 150+ features (MTF, wavelets, microstructure)
- **Final samples:** 19,740 (14% retention)
- **Inference:** Richer features create more NaN

**The new pipeline generates MORE features but LOSES more data in the process**

---

## Impact Assessment

### Data Quality ✅

- Remaining 19,740 samples are 100% valid (no NaN)
- Proper lookahead bias prevention (dropping rows with future NaN)

### Data Quantity ❌

- 44% fewer samples than before
- Training on smaller dataset → potentially worse model performance
- Lost Jan-Mar 2020 data despite earlier start date

### Label Distribution 🔴

- **CRITICAL:** ALL labels are NEUTRAL (100%)
- No LONG or SHORT labels (separate issue)
- Makes sample count reduction even worse

---

## Recommended Solutions

### Option 1: Forward-Fill NaN for MTF Features

Instead of dropping rows with NaN in MTF features, forward-fill:

```python
# For MTF features only (not base features)
mtf_cols = [col for col in df.columns if any(tf in col for tf in ['15min', '30min', '1h', '4h', 'daily'])]
df[mtf_cols] = df[mtf_cols].fillna(method='ffill')
```

**Expected improvement:** Retain ~100,000+ rows instead of 19,740

### Option 2: Fix Broken Features

Investigate why autocorrelation and BB features have 50-100% NaN:

- `return_autocorr_lag20` → 100% NaN (completely broken)
- `bb_width`, `bb_position` → 50-90% NaN (partially broken)

Fix or remove these features.

**Expected improvement:** Eliminate major NaN sources

### Option 3: Selective NaN Handling

Different strategies for different feature types:

- **Base features (SMA, EMA, RSI):** Drop rows (prevent lookahead)
- **MTF features:** Forward-fill (MTF lags are expected)
- **Wavelet features:** Interpolate edge effects
- **Broken features:** Fix or remove

**Expected improvement:** Maximize data retention while preventing bias

### Option 4: Reduce Feature Richness

Disable features that create excessive NaN:

```bash
./pipeline run --symbols MGC \
  --disable-wavelets \
  --mtf-timeframes 15min,1h  # Remove daily/4h \
  --horizons 20
```

**Expected improvement:** Retain more data, simpler feature set

---

## Recommendations

### Immediate (High Priority)

1. **Investigate broken features** (autocorr_lag20, BB features)
2. **Test MTF forward-fill** strategy
3. **Document NaN sources** per feature type

### Short-term

1. **Implement selective NaN handling** (Option 3)
2. **Re-run pipeline** with fixes
3. **Compare sample counts** to old pipeline

### Long-term

1. **Add NaN diagnostics** to pipeline reporting
2. **Track NaN by feature** in logs
3. **Configure NaN strategy** per feature type

---

## Conclusion

**The 44% sample reduction is caused by overly aggressive NaN handling, not by actual data loss.**

The new pipeline generates richer features (150+ including MTF, wavelets, microstructure) but many of these features have scattered NaN throughout the dataset. The current "drop any row with any NaN" strategy decimates the data.

**Trade-off:**
- **Old pipeline:** More data (35,388 samples), simpler features
- **New pipeline:** Less data (19,740 samples), richer features

**Solution:** Implement selective NaN handling to retain data while preventing lookahead bias.

---

## Files to Investigate

```
runs/20251227_213131/logs/pipeline.log           # Full pipeline logs
src/phase1/stages/features/nan_handling.py       # NaN cleanup logic
src/phase1/stages/features/autocorrelation.py    # Broken autocorr features
src/phase1/stages/features/price_features.py     # BB calculation
src/phase1/stages/mtf/                           # MTF feature generation
```
