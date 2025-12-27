# Label Distribution Bug - Root Cause Analysis & Fix

## Executive Summary

**Issue**: Severe data distribution mismatch between training and validation sets, with validation showing 0.4% Short labels instead of expected ~48%.

**Root Cause**: Pipeline state inconsistency - stale indices from Dec 18 (2.4M samples) vs. current scaled data from Dec 24-25 (35K samples).

**Impact**: Models trained on this data will have poor F1 scores for Short class due to extreme class imbalance.

---

## Detailed Analysis

### 1. Data Flow Investigation

```
Raw Data (combined_final_labeled.parquet)
  ├── 2,398,649 rows total
  ├── Balanced labels: 48.4% Short, 1.7% Neutral, 49.9% Long
  └── Date range: 2008-11-27 to 2025-07-16

Indices (created Dec 18, 2024)
  ├── train_indices.npy: 1,679,034 samples
  ├── val_indices.npy:     359,509 samples
  └── test_indices.npy:    359,510 samples

Scaled Data (created Dec 25, 2024)
  ├── train_scaled.parquet: 24,711 rows  (98.5% DATA LOSS!)
  ├── val_scaled.parquet:    3,808 rows  (98.9% DATA LOSS!)
  ├── test_scaled.parquet:   3,869 rows  (98.9% DATA LOSS!)
  ├── Imbalanced labels: 0.3-0.4% Short, 40-43% Neutral, 56-59% Long
  └── Date range: 2020-03-12 to 2021-12-01 (filtered!)
```

### 2. Root Cause

**Timeline of Events:**

1. **Dec 18**: Full pipeline run on 2.4M samples → created indices
2. **Dec 20**: Final labels generated for full dataset
3. **Dec 24-25**: Pipeline re-run with filtered data (35K samples from 2020-2021)
4. **Dec 25**: Scaling stage created new scaled files
5. **Result**: Indices reference full dataset, but scaled data is from filtered subset

**The Mismatch:**
- `split_config.json` (Dec 25): `total_samples: 35,388`
- Index files (Dec 18): Reference 2,398,649 samples
- Current scaled data uses the Dec 24 filtered dataset
- Old indices are incompatible with current data

### 3. Secondary Issue: Quality Score Bias

Even in the full dataset, there's a significant quality score imbalance:

```
Label Type | Mean Quality | % with Quality >= 0.5
-----------|--------------|---------------------
Short      | 0.441        | 31.1%
Neutral    | 0.359        |  6.6%
Long       | 0.668        | 98.4%
```

This bias in the labeling algorithm causes Long labels to have 3x higher quality scores than Short labels.

---

## Solution Options

### Option A: Re-run Pipeline with Full Dataset (RECOMMENDED)

This ensures all downstream models have access to the full 2.4M samples.

```bash
# 1. Backup current state
mkdir -p /Users/sneh/research/data/backups/20251226_pre_fix
cp -r /Users/sneh/research/data/splits /Users/sneh/research/data/backups/20251226_pre_fix/

# 2. Delete stale indices and scaled data
rm /Users/sneh/research/data/splits/*_indices.npy
rm -rf /Users/sneh/research/data/splits/scaled

# 3. Re-run splits stage
./pipeline run --start-from create_splits --symbols MES,MGC

# 4. Verify indices match combined data
python scripts/diagnose_label_distribution.py
```

**Pros:**
- Uses full dataset (2.4M samples)
- Maximum training data available
- Consistent with CLAUDE.md expectations

**Cons:**
- Requires re-running downstream stages
- Longer training times

### Option B: Regenerate Indices for Filtered Dataset

Use the existing 35K filtered dataset and regenerate matching indices.

```bash
# 1. Create filtered combined file
python scripts/create_filtered_dataset.py \
  --start-date "2020-03-12" \
  --end-date "2021-12-01" \
  --output /Users/sneh/research/data/final/combined_filtered_2020_2021.parquet

# 2. Re-run splits on filtered data
./pipeline run --start-from create_splits \
  --input-data combined_filtered_2020_2021.parquet

# 3. Re-run scaling
./pipeline run --start-from feature_scaling
```

**Pros:**
- Faster training (smaller dataset)
- Maintains current scaled data

**Cons:**
- Discards 98.5% of available data
- May not generalize to other time periods

### Option C: Fix Quality Score Bias in Labeling (LONG-TERM)

Address the root cause of quality score imbalance in the triple-barrier labeling algorithm.

See `/Users/sneh/research/docs/QUALITY_SCORE_BIAS_FIX.md` for details.

---

## Immediate Action Required

Run diagnostic to confirm current state:

```bash
cd /Users/sneh/research
python3 scripts/diagnose_label_distribution.py
```

Then choose and execute one of the solution options above.

---

## Prevention

1. **Add pipeline state validation** to detect index/data mismatches
2. **Version control indices** with corresponding data files
3. **Add checksums** to split_config.json to validate data integrity
4. **Atomic updates** - replace indices and data together, or not at all

## Implementation Notes

- File: `/Users/sneh/research/scripts/diagnose_label_distribution.py`
- Evidence: Diagnostic output shows 98.5% data loss
- Fix requires: Pipeline re-run from create_splits stage
- Estimated time: 30-60 minutes for full pipeline re-run

---

**Status**: Issue diagnosed, awaiting user decision on solution option.
**Priority**: HIGH - affects all model training
**Assigned**: User decision required
