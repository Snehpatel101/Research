# Scaling Validation Fix Report

**Date:** 2025-12-24
**Issue:** `data/splits/scaled/scaling_metadata.json` showed 4 validation failures
**Status:** ✓ RESOLVED

---

## Problem Summary

The scaling validation reported 4 issues:
1. `obv` had 1,598 NaN values in validation split
2. `obv_sma_20` had 1,886 NaN values in validation split
3. `obv` had 4,008 NaN values in test split
4. `obv_sma_20` had 4,884 NaN values in test split

Initial metadata:
```json
"scaling_validation": {
    "is_valid": false,
    "warnings_count": 0,
    "issues_count": 4
}
```

---

## Root Cause Analysis

### Investigation Steps

1. **Checked original data** - OBV columns had ZERO NaNs before scaling
2. **Reproduced the issue** - NaNs were created during the transform step
3. **Identified warning** - Runtime warning: `invalid value encountered in log1p`
4. **Traced code path** - Found issue in `/home/jake/Desktop/Research/src/stages/scaling/scalers.py`

### Root Cause

The `should_log_transform()` function (line 56) incorrectly flagged OBV features for log transformation:

```python
def should_log_transform(feature_name: str, category: FeatureCategory) -> bool:
    """Log transform recommended for price/volume features"""
    if category in [FeatureCategory.PRICE_LEVEL, FeatureCategory.VOLUME]:
        if not any(x in feature_name.lower() for x in ['ratio', 'pct', 'zscore', 'to_']):
            return True  # ← Problem: OBV is VOLUME category
    return False
```

**Why this failed:**
- OBV (On-Balance Volume) is a cumulative volume indicator: `OBV = previous_OBV + (volume if close>close_prev else -volume)`
- OBV can be **negative** when cumulative selling volume exceeds buying volume
- Log transform requires positive values: `np.log1p(negative_value)` → **NaN**
- Example: `np.log1p(-229685.0)` → `nan` (with RuntimeWarning)

---

## Solution

### Code Changes

**File:** `/home/jake/Desktop/Research/src/stages/scaling/scalers.py`

**Change:** Added explicit check to exclude OBV features from log transformation

```python
def should_log_transform(feature_name: str, category: FeatureCategory) -> bool:
    """
    Determine if a feature should have log transform applied.

    Log transform is recommended for:
    - Price level features (SMA, EMA, etc.)
    - Volume features (but NOT OBV which can be negative)
    - Features with high positive skewness
    """
    # OBV can be negative (cumulative buying - selling volume), so never log-transform it
    if 'obv' in feature_name.lower():
        return False

    if category in [FeatureCategory.PRICE_LEVEL, FeatureCategory.VOLUME]:
        # Check if it's a raw price/volume feature (not a ratio)
        if not any(x in feature_name.lower() for x in ['ratio', 'pct', 'zscore', 'to_']):
            return True
    return False
```

### Affected Features

The fix prevents log transformation for:
- `obv` - On-Balance Volume indicator
- `obv_sma_20` - 20-period SMA of OBV

These features now use RobustScaler **without** log transformation.

---

## Validation

### Before Fix
```
Val split:
  obv: 1,598 NaN values
  obv_sma_20: 1,886 NaN values

Test split:
  obv: 4,008 NaN values
  obv_sma_20: 4,884 NaN values
```

### After Fix
```
Scaling Validation Results:
{
  "is_valid": true,
  "warnings_count": 0,
  "issues_count": 0
}

Checking 129 features for NaN/Inf...
train: NaN=0, Inf=0
val: NaN=0, Inf=0
test: NaN=0, Inf=0

✓ SUCCESS: No NaN or Inf values in any scaled features!
```

### Test Results
```python
# Verified fix works correctly
obv_should_log = should_log_transform('obv', FeatureCategory.VOLUME)
assert obv_should_log == False  # ✓ Pass

obv_sma_should_log = should_log_transform('obv_sma_20', FeatureCategory.PRICE_LEVEL)
assert obv_sma_should_log == False  # ✓ Pass
```

---

## Impact Assessment

### What Changed
- **Scaler behavior:** OBV features now use `RobustScaler` directly (no log transform)
- **Data integrity:** All 4 validation issues resolved
- **Downstream impact:** Phase 2 models can now consume scaled data without NaN handling

### What Stayed the Same
- Other volume features (e.g., `volume`, `volume_15m`) still use log transform (they're always positive)
- Price level features still use log transform where appropriate
- Overall scaling strategy unchanged for 127/129 features

### Files Modified
1. `/home/jake/Desktop/Research/src/stages/scaling/scalers.py` - Added OBV exclusion logic
2. `/home/jake/Desktop/Research/data/splits/scaled/train_scaled.parquet` - Regenerated
3. `/home/jake/Desktop/Research/data/splits/scaled/val_scaled.parquet` - Regenerated
4. `/home/jake/Desktop/Research/data/splits/scaled/test_scaled.parquet` - Regenerated
5. `/home/jake/Desktop/Research/data/splits/scaled/feature_scaler.pkl` - Regenerated
6. `/home/jake/Desktop/Research/data/splits/scaled/scaling_metadata.json` - Updated validation status

---

## Recommendations

### Immediate Actions
- ✓ Fix applied and validated
- ✓ Scaled data regenerated
- ✓ Validation passes

### Future Considerations

1. **Add feature range validation** - Detect if a feature can be negative before applying log transform
2. **Add unit tests** - Test `should_log_transform()` for edge cases:
   - Features with negative values
   - Cumulative indicators (OBV, MFI if it can be negative)
   - Zero-crossing features

3. **Documentation** - Update feature engineering docs to note:
   - OBV is a signed cumulative indicator
   - Features that can be negative should not use log transforms

4. **Consider alternative transforms for OBV:**
   - Asinh transform: `np.arcsinh(x)` handles negative values
   - Signed log: `sign(x) * log(1 + abs(x))`
   - Current approach (RobustScaler only) is acceptable

---

## Conclusion

**Issue:** Log transformation of OBV features created NaN values
**Cause:** OBV can be negative, but log transform requires positive values
**Fix:** Excluded OBV features from log transformation in `should_log_transform()`
**Result:** All 4 validation issues resolved, 0 NaN/Inf values in scaled data

Phase 2 model training can proceed with clean, validated scaled datasets.
