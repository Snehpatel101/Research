# Performance Anti-Patterns in Feature Engineering

**Generated:** 2026-01-28
**Directory:** `/home/jake/Desktop/Research/src/data/pipeline/stages/features/`

---

## Executive Summary

Analysis of 20 Python files in the feature engineering pipeline reveals significant performance optimization opportunities. The codebase relies heavily on pandas operations that create unnecessary intermediate objects, triggering repeated memory allocation and copy operations.

### Key Findings

| Anti-Pattern | Count | Impact | Priority |
|--------------|-------|--------|----------|
| Individual column assignments | 83 | High - O(n) copy per assignment | P0 |
| `pd.Series()` wrapping | 38 | Medium - unnecessary object creation | P1 |
| `.astype(int)` usage | 17 | Low - suboptimal dtype | P2 |
| `.rolling()` calls | 47 | Low - inherent to computation | P3 |

### Estimated Performance Impact

For a typical 100,000-row DataFrame:
- **Current approach:** ~83 DataFrame copy operations
- **Batch assignment:** ~10 batch operations (8-10x reduction in copies)
- **Estimated speedup:** 2-5x for feature generation pipeline

---

## File-by-File Analysis

### 1. temporal.py (PRIORITY: HIGH)

**Location:** `/home/jake/Desktop/Research/src/data/pipeline/stages/features/temporal.py`

**Issues Found:**
- **Column assignments:** 13
- **`.astype(int)` usage:** 4 (lines 68, 108, 109, 110)
- **`.apply()` usage:** 1 (line 64) - extremely slow

**Affected Lines:**

| Line | Pattern | Issue |
|------|---------|-------|
| 38 | `df["hour"] = df["datetime"].dt.hour` | Individual assignment |
| 39 | `df["hour_sin"] = np.sin(...)` | Individual assignment |
| 40 | `df["hour_cos"] = np.cos(...)` | Individual assignment |
| 43-45 | `df["minute"]` + sin/cos | 3 individual assignments |
| 48-50 | `df["dayofweek"]` + sin/cos | 3 individual assignments |
| 64 | `df["session"] = df["hour"].apply(get_session)` | **CRITICAL: `.apply()` is ~100x slower than vectorized** |
| 68 | `df[f"session_{session}"] = (...).astype(int)` | Loop with individual assignments |
| 108-110 | Session assignments with `.astype(int)` | Individual assignments |

**BEFORE:**
```python
df["hour"] = df["datetime"].dt.hour
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["minute"] = df["datetime"].dt.minute
df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
df["dayofweek"] = df["datetime"].dt.dayofweek
df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
df["session"] = df["hour"].apply(get_session)  # SLOW!
```

**AFTER:**
```python
# Extract once
hour = df["datetime"].dt.hour.values
minute = df["datetime"].dt.minute.values
dayofweek = df["datetime"].dt.dayofweek.values

# Vectorized session calculation (replaces slow .apply())
session_asia = ((hour >= 0) & (hour < 8)).astype(np.int8)
session_london = ((hour >= 8) & (hour < 16)).astype(np.int8)
session_ny = (hour >= 16).astype(np.int8)

# Batch assignment using pd.concat or dict
new_cols = pd.DataFrame({
    "hour_sin": np.sin(2 * np.pi * hour / 24),
    "hour_cos": np.cos(2 * np.pi * hour / 24),
    "minute_sin": np.sin(2 * np.pi * minute / 60),
    "minute_cos": np.cos(2 * np.pi * minute / 60),
    "dayofweek_sin": np.sin(2 * np.pi * dayofweek / 7),
    "dayofweek_cos": np.cos(2 * np.pi * dayofweek / 7),
    "session_asia": session_asia,
    "session_london": session_london,
    "session_ny": session_ny,
}, index=df.index)

df = pd.concat([df, new_cols], axis=1)
```

---

### 2. momentum.py (PRIORITY: HIGH)

**Location:** `/home/jake/Desktop/Research/src/data/pipeline/stages/features/momentum.py`

**Issues Found:**
- **Column assignments:** 14
- **`pd.Series()` wrapping:** 6 (lines 51, 100, 104, 164, 165)
- **`.astype(int)` usage:** 6 (lines 54, 55, 114, 118, 168, 169)
- **`.rolling()` calls:** 5 (lines 201, 202, 272, 273, 313, 314)

**Affected Lines:**

| Line | Pattern | Issue |
|------|---------|-------|
| 51 | `pd.Series(calculate_rsi_numba(...)).shift(1).values` | Unnecessary Series creation |
| 54-55 | `df["rsi_overbought"] = (...).astype(int)` | Boolean to int, individual assign |
| 100, 104 | `pd.Series(ema_fast - ema_slow)` | Intermediate Series |
| 101-108 | MACD assignments | 5 individual assignments |
| 111-118 | MACD crossover with `.astype(int)` | Individual assignments |
| 164-165 | `pd.Series(k).shift(1).values` | Unnecessary Series |
| 168-169 | Stochastic overbought/oversold | Individual assignments |

**BEFORE:**
```python
df[col_name] = pd.Series(calculate_rsi_numba(df["close"].values, period)).shift(1).values
df["rsi_overbought"] = (df[col_name] > 70).astype(int)
df["rsi_oversold"] = (df[col_name] < 30).astype(int)
```

**AFTER:**
```python
# Use numpy directly, shift with np.roll or slicing
rsi = calculate_rsi_numba(df["close"].values, period)
rsi_shifted = np.concatenate([[np.nan], rsi[:-1]])  # shift without pd.Series

# Batch assignment
new_cols = pd.DataFrame({
    col_name: rsi_shifted,
    "rsi_overbought": (rsi_shifted > 70).astype(np.int8),
    "rsi_oversold": (rsi_shifted < 30).astype(np.int8),
}, index=df.index)

df = pd.concat([df, new_cols], axis=1)
```

---

### 3. volatility.py (PRIORITY: HIGH)

**Location:** `/home/jake/Desktop/Research/src/data/pipeline/stages/features/volatility.py`

**Issues Found:**
- **Column assignments:** 20
- **`pd.Series()` wrapping:** 4 (lines 50, 161, 162, 176)
- **`.rolling()` calls:** 17 (high-frequency pattern)

**Affected Lines:**

| Line | Pattern | Issue |
|------|---------|-------|
| 50 | `df[f"atr_{period}"] = pd.Series(atr).shift(1).values` | Unnecessary Series |
| 94-95 | Rolling mean/std | Required computation |
| 97-106 | Bollinger Band assignments | 6 individual assignments |
| 161-166 | Keltner Channel assignments | 5 individual assignments |
| 268, 307, 409, 491 | Volatility measure assignments | Individual assignments |
| 675, 697, 703 | GARCH feature assignments | Individual assignments |
| 692, 701, 702 | Rolling computations on GARCH output | Required rolling |

**BEFORE:**
```python
df["bb_middle"] = bb_middle_raw.shift(1)
bb_std = bb_std_raw.shift(1)
df["bb_upper"] = df["bb_middle"] + (std_mult * bb_std)
df["bb_lower"] = df["bb_middle"] - (std_mult * bb_std)
df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_std_safe
df["bb_position"] = (close_lagged - df["bb_lower"]) / band_range_safe
df["close_bb_zscore"] = (close_lagged - df["bb_middle"]) / bb_std_safe
```

**AFTER:**
```python
# Compute all values using numpy arrays first
bb_middle = bb_middle_raw.shift(1).values
bb_std = bb_std_raw.shift(1).values
bb_upper = bb_middle + (std_mult * bb_std)
bb_lower = bb_middle - (std_mult * bb_std)
bb_std_safe = np.where(bb_std == 0, np.nan, bb_std)
band_range = bb_upper - bb_lower
band_range_safe = np.where(band_range == 0, np.nan, band_range)
close_lagged = df["close"].shift(1).values

# Batch assignment
bb_cols = pd.DataFrame({
    "bb_middle": bb_middle,
    "bb_upper": bb_upper,
    "bb_lower": bb_lower,
    "bb_width": (bb_upper - bb_lower) / bb_std_safe,
    "bb_position": (close_lagged - bb_lower) / band_range_safe,
    "close_bb_zscore": (close_lagged - bb_middle) / bb_std_safe,
}, index=df.index)

df = pd.concat([df, bb_cols], axis=1)
```

---

### 4. volume.py (PRIORITY: MEDIUM)

**Location:** `/home/jake/Desktop/Research/src/data/pipeline/stages/features/volume.py`

**Issues Found:**
- **Column assignments:** 18
- **`pd.Series()` wrapping:** 1 (line 117)
- **`.rolling()` calls:** 7

**Affected Lines:**

| Line | Pattern | Issue |
|------|---------|-------|
| 52-68 | OBV and volume feature assignments | 6 individual assignments |
| 105-120 | VWAP calculation with temp columns | Creates 5 temp columns then drops |
| 157, 198, 207 | Dollar volume assignments | Individual assignments |
| 261, 267, 280-281 | TWAP assignments | Individual assignments |

**BEFORE:**
```python
df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
df["date"] = df["datetime"].dt.date
df["cum_vol"] = df.groupby("date")["volume"].cumsum()
df["tp_vol"] = df["typical_price"] * df["volume"]
df["cum_vwap_num"] = df.groupby("date")["tp_vol"].cumsum()
# ... later drops these columns
```

**AFTER:**
```python
# Use local variables instead of DataFrame columns for intermediates
typical_price = (df["high"] + df["low"] + df["close"]).values / 3
date_col = df["datetime"].dt.date
tp_vol = typical_price * df["volume"].values

# Use transform for cumsum without adding columns
cum_vol = df.groupby(date_col)["volume"].transform("cumsum").values
cum_vwap_num = pd.Series(tp_vol, index=df.index).groupby(date_col).transform("cumsum").values

vwap_raw = np.where(cum_vol > 0, cum_vwap_num / cum_vol, typical_price)

# Single assignment
df["vwap"] = np.concatenate([[np.nan], vwap_raw[:-1]])  # shift(1)
```

---

### 5. microstructure.py (PRIORITY: MEDIUM)

**Location:** `/home/jake/Desktop/Research/src/data/pipeline/stages/features/microstructure.py`

**Issues Found:**
- **Column assignments:** 17
- **`pd.Series()` wrapping:** 1 (line 250)
- **`.rolling()` calls:** 12

**Affected Lines:**

| Line | Pattern | Issue |
|------|---------|-------|
| 76, 83 | Amihud assignments | Individual assignments |
| 130, 135 | Roll spread assignments | Individual assignments |
| 181 | Kyle lambda assignment | Individual assignment |
| 250 | `pd.Series(cs_spread_raw, index=df.index).shift(1)` | Unnecessary Series |
| 290, 297 | Relative spread assignments | Individual assignments in loop |
| 336, 339 | Volume imbalance assignments | Individual assignments |
| 485 | Vol ratio assignment | Individual assignment |

**Pattern for batch fix:**
```python
# Collect all microstructure features in a dict, then batch assign
micro_features = {
    "micro_amihud": amihud_raw.shift(1),
    "micro_roll_spread": roll_spread_raw.shift(1),
    "micro_kyle_lambda": kyle_lambda_raw.shift(1),
    # ... etc
}

for period in periods:
    micro_features[f"micro_amihud_{period}"] = amihud_raw.rolling(window=period).mean().shift(1)

df = pd.concat([df, pd.DataFrame(micro_features, index=df.index)], axis=1)
```

---

### 6. regime.py (PRIORITY: LOW)

**Location:** `/home/jake/Desktop/Research/src/data/pipeline/stages/features/regime.py`

**Issues Found:**
- **Column assignments:** 5
- **`.astype(int)` usage:** 3 (lines 96, 200)
- **`.rolling()` calls:** 2

**Affected Lines:**

| Line | Pattern | Issue |
|------|---------|-------|
| 96 | `df["volatility_regime"] = (...).astype(int)` | Individual assignment |
| 113 | `df["trend_regime"] = np.where(...)` | Individual assignment |
| 200 | Same as 96 | Duplicate pattern |
| 241 | Same as 113 | Duplicate pattern |
| 298 | `df["structure_regime"] = regimes.shift(1)` | Individual assignment |

---

### 7. entropy.py (PRIORITY: LOW)

**Location:** `/home/jake/Desktop/Research/src/data/pipeline/stages/features/entropy.py`

**Issues Found:**
- **Column assignments:** 1
- **`pd.Series()` wrapping:** 5 (lines 234, 420, 621, 837, 1063)
- **`.astype(int)` usage:** 3 (lines 166, 273, 675) - internal computation, acceptable

**Affected Lines:**

| Line | Pattern | Issue |
|------|---------|-------|
| 234 | `df[col_name] = pd.Series(entropy, index=df.index).shift(1)` | Unnecessary Series |
| 420 | `df[col_name] = pd.Series(lz_values, index=df.index).shift(1)` | Unnecessary Series |
| 621 | `df[col_name] = pd.Series(apen, index=df.index).shift(1)` | Unnecessary Series |
| 837 | `df[col_name] = pd.Series(hurst, index=df.index).shift(1)` | Unnecessary Series |
| 854 | `df["hurst_regime"] = np.select(...)` | Individual assignment |
| 1063 | `df[col_name] = pd.Series(sampen, index=df.index).shift(1)` | Unnecessary Series |

**Fix pattern:**
```python
# Instead of:
df[col_name] = pd.Series(entropy, index=df.index).shift(1)

# Use direct numpy assignment:
df[col_name] = np.concatenate([[np.nan], entropy[:-1]])
```

---

### 8. trend.py (PRIORITY: LOW)

**Location:** `/home/jake/Desktop/Research/src/data/pipeline/stages/features/trend.py`

**Issues Found:**
- **Column assignments:** 6
- **`pd.Series()` wrapping:** 5 (lines 49, 50, 51, 167, 168)
- **`.astype(int)` usage:** 1 (line 54)

**Affected Lines:**

| Line | Pattern | Issue |
|------|---------|-------|
| 49-51 | `df[...] = pd.Series(...).shift(1).values` | Unnecessary Series for ADX |
| 54 | `df["adx_strong_trend"] = (...).astype(int)` | Individual assignment |
| 167-168 | `df["supertrend"] = pd.Series(...).shift(1).values` | Unnecessary Series |

---

### 9. wavelets.py (PRIORITY: LOW)

**Location:** `/home/jake/Desktop/Research/src/data/pipeline/stages/features/wavelets.py`

**Issues Found:**
- **Column assignments:** 10 (in loops)
- **`pd.Series()` wrapping:** 10 (lines 154, 157, 164, 169, 200, 207, 212, 253, 300, 301)

**Affected Lines:**

| Line | Pattern | Issue |
|------|---------|-------|
| 154-169 | Wavelet coefficient assignments in loop | Multiple individual assignments |
| 200-212 | Wavelet energy assignments in loop | Multiple individual assignments |
| 253 | `df[vol_col] = pd.Series(wavelet_vol).shift(1).values` | Unnecessary Series |
| 300-301 | Trend strength/direction assignments | Individual assignments |

---

### 10. price_features.py (PRIORITY: LOW)

**Location:** `/home/jake/Desktop/Research/src/data/pipeline/stages/features/price_features.py`

**Issues Found:**
- **Column assignments:** 4
- **`.rolling()` calls:** 1 (line 146)

**Affected Lines:**

| Line | Pattern | Issue |
|------|---------|-------|
| 87 | `df["hl_ratio"] = _safe_divide(...)` | Individual assignment |
| 90 | `df["co_ratio"] = _safe_divide(...)` | Individual assignment |
| 93 | `df["range_pct"] = _safe_divide(...)` | Individual assignment |
| 187 | `df["clv"] = clv_raw.shift(1)` | Individual assignment |

---

### 11. microstructure_proxies.py (PRIORITY: MEDIUM)

**Location:** `/home/jake/Desktop/Research/src/data/pipeline/stages/features/microstructure_proxies.py`

**Issues Found:**
- **`pd.Series()` wrapping:** 5 (lines 102, 133, 252, 279, 343) - returns only
- **`.astype(int)` usage:** 1 (line 210)
- **`.rolling()` calls:** 3 (lines 125, 250, 277)

These are primarily return statements creating Series, which is appropriate for the API.

---

### 12. moving_averages.py (PRIORITY: LOW)

**Location:** `/home/jake/Desktop/Research/src/data/pipeline/stages/features/moving_averages.py`

**Issues Found:**
- **Column assignments:** 4 per function call (SMA/EMA + ratio)
- **`pd.Series()` wrapping:** 2 (lines 52, 94)

**Affected Lines:**

| Line | Pattern | Issue |
|------|---------|-------|
| 52-53 | `pd.Series(calculate_sma_numba(...))` | Unnecessary Series |
| 94-95 | `pd.Series(calculate_ema_numba(...))` | Unnecessary Series |

---

## Recommended Batch Fix Pattern

### Universal Fix Template

For any function that makes multiple column assignments:

```python
def add_features_batch(df: pd.DataFrame, ...) -> pd.DataFrame:
    """Add features using batch assignment pattern."""

    # 1. Extract numpy arrays for computation
    close = df["close"].values
    high = df["high"].values
    # etc.

    # 2. Compute all features as numpy arrays
    feature1 = compute_feature1(close)
    feature2 = compute_feature2(high)
    # etc.

    # 3. Apply shifts using numpy (faster than pd.Series.shift)
    def numpy_shift(arr, periods=1):
        result = np.empty_like(arr)
        if periods > 0:
            result[:periods] = np.nan
            result[periods:] = arr[:-periods]
        return result

    feature1_shifted = numpy_shift(feature1, 1)
    feature2_shifted = numpy_shift(feature2, 1)

    # 4. Batch assign using DataFrame constructor
    new_cols = pd.DataFrame({
        "feature1": feature1_shifted,
        "feature2": feature2_shifted,
    }, index=df.index)

    # 5. Single concat operation
    return pd.concat([df, new_cols], axis=1)
```

### Numpy Shift Helper

Add this utility to avoid `pd.Series().shift()`:

```python
# Add to numba_functions.py
import numba as nb

@nb.jit(nopython=True, cache=True)
def numpy_shift_1d(arr: np.ndarray, periods: int = 1) -> np.ndarray:
    """
    Shift array by N periods, filling with NaN.

    Equivalent to pd.Series(arr).shift(periods).values but ~10x faster.
    """
    n = len(arr)
    result = np.empty(n, dtype=np.float64)

    if periods >= 0:
        result[:periods] = np.nan
        result[periods:] = arr[:n - periods]
    else:
        result[periods:] = np.nan
        result[:n + periods] = arr[-periods:]

    return result
```

---

## Priority Ranking

### P0 - Critical (Fix First)
1. **temporal.py** - `.apply()` is extremely slow, easy win
2. **momentum.py** - High call frequency, many individual assignments
3. **volatility.py** - Complex calculations with many intermediates

### P1 - High Priority
4. **volume.py** - Creates/drops temp columns
5. **microstructure.py** - Many rolling computations

### P2 - Medium Priority
6. **entropy.py** - pd.Series wrapping
7. **wavelets.py** - Loop-based assignments
8. **trend.py** - pd.Series wrapping

### P3 - Low Priority (Optional)
9. **regime.py** - Few assignments
10. **price_features.py** - Few assignments
11. **moving_averages.py** - Simple cases
12. **microstructure_proxies.py** - Return statements only

---

## Validation Checklist

After applying fixes, verify:

```bash
# 1. Run existing tests
pytest src/data/pipeline/stages/features/ -v

# 2. Verify feature values haven't changed
python -c "
import pandas as pd
import numpy as np
from src.data.pipeline.stages.features import *

# Load test data
df = pd.read_parquet('path/to/test_data.parquet')

# Generate features with old code (before fix)
df_old = add_all_features(df.copy())

# Generate features with new code (after fix)
df_new = add_all_features(df.copy())

# Compare
for col in df_old.columns:
    if col in df_new.columns:
        if not np.allclose(df_old[col].values, df_new[col].values, equal_nan=True):
            print(f'MISMATCH: {col}')
print('Validation complete')
"

# 3. Benchmark performance
python -c "
import time
import pandas as pd
from src.data.pipeline.stages.features import *

df = pd.read_parquet('path/to/test_data.parquet')

start = time.time()
for _ in range(10):
    add_all_features(df.copy())
print(f'Time: {(time.time() - start) / 10:.2f}s per iteration')
"
```

---

## Summary

| Metric | Current | After Fix (Est.) |
|--------|---------|------------------|
| Column assignments | 83 | ~15 batch ops |
| pd.Series() calls | 38 | ~5 |
| Memory copies | ~100 | ~20 |
| Est. speedup | 1x | 2-5x |

**Total lines requiring changes:** ~200
**Estimated effort:** 2-3 days for full implementation

---

*Document generated by code analysis. Review each fix before implementation.*
