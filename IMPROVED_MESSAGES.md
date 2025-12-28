# Improved Messages - Quick Reference

## Coverage Warnings (Issue #11)

### ✅ Normal Coverage (87.5% - Expected)

**OLD MESSAGE (Confusing):**
```
WARNING: lstm: Low coverage 87.5% (1250 missing).
         This is expected for sequence models with seq_len=60.
```

**NEW MESSAGE (Clear & Informative):**
```
INFO: lstm: Coverage 87.5% (1250 samples missing) - EXPECTED for seq_len=60 with 2 segments.
      Expected coverage: ~88.0%, actual is within normal range.
```

**Why this is better:**
- ✅ Uses INFO level (not WARNING) for expected behavior
- ✅ Shows expected coverage (88.0%) vs actual (87.5%)
- ✅ Explains why coverage is below 100% (2 segments, seq_len=60)
- ✅ Clearly states this is normal behavior
- ✅ No action required from user

---

### ⚠️ Unexpectedly Low Coverage (60% - Problem!)

**OLD MESSAGE (Misleading):**
```
WARNING: lstm: Low coverage 60.0% (4000 missing).
         This is expected for sequence models with seq_len=60.
```

**NEW MESSAGE (Actionable & Clear):**
```
WARNING: lstm: Coverage 60% is UNEXPECTEDLY LOW (expected ~85.0% for seq_len=60, 2 segments).
         Missing 4000 samples (25.0% below expected).
         Investigate: possible data issues or excessive gaps.
```

**Why this is better:**
- ✅ Uses WARNING level for abnormal behavior
- ✅ Clearly states "UNEXPECTEDLY LOW" (not "expected")
- ✅ Shows how far below expected (25% below)
- ✅ Provides actionable guidance (investigate data issues)
- ✅ Helps user diagnose the problem

---

## Boundary Detection (Issue #12)

### Symbol-Based Detection

**MESSAGE:**
```
DEBUG: Found 3 symbols, 2 boundaries
INFO: Generating sequence OOF for lstm (seq_len=60, boundary_detection=symbol_column)
```

**What this tells you:**
- ✅ Symbol column exists and is being used
- ✅ Data has 3 unique symbols with 2 boundaries between them
- ✅ Sequences won't cross symbol boundaries

---

### DateTime Gap Detection (Single-Symbol Pipeline)

**OLD MESSAGE (Silent):**
```
DEBUG: Symbol column 'symbol' not found, disabling isolation
INFO: Generating sequence OOF for lstm (seq_len=60, symbol_isolated=False)
```

**NEW MESSAGE (Informative):**
```
DEBUG: Symbol column 'symbol' not found, attempting gap detection
INFO: Detected 104 time gaps (using boundary detection).
     Bar resolution: 0 days 00:05:00, gap threshold: 0 days 00:10:00
INFO: Generating sequence OOF for lstm (seq_len=60, boundary_detection=datetime_gaps)
```

**Why this is better:**
- ✅ Shows fallback to gap detection (not just "disabled")
- ✅ Reports number of gaps found (104)
- ✅ Shows bar resolution (5 min) and gap threshold (10 min)
- ✅ Clear indication that boundary detection is active
- ✅ User knows sequences won't span gaps

---

### No Boundaries Detected

**MESSAGE:**
```
DEBUG: No significant time gaps detected (resolution: 0 days 00:05:00)
INFO: Generating sequence OOF for lstm (seq_len=60, boundary_detection=datetime_gaps)
```

**What this tells you:**
- ✅ Gap detection was attempted but found no significant gaps
- ✅ Data appears continuous (good data quality)
- ✅ Sequences can span the entire dataset

---

### No DatetimeIndex Available

**MESSAGE:**
```
DEBUG: Symbol column 'symbol' not found, attempting gap detection
DEBUG: No DatetimeIndex found, disabling boundary detection
INFO: Generating sequence OOF for lstm (seq_len=60, boundary_detection=none)
```

**What this tells you:**
- ✅ No symbol column or DatetimeIndex available
- ✅ Boundary detection is disabled (sequences can span anything)
- ✅ Consider adding DatetimeIndex for better gap handling

---

## Coverage Formula Explained

### Expected Coverage Calculation

```
n_segments = n_boundaries + 1
expected_missing = n_segments * seq_len
expected_coverage = max(0.0, 1.0 - (expected_missing / n_samples))
```

### Examples

| n_samples | seq_len | n_boundaries | n_segments | expected_missing | expected_coverage |
|-----------|---------|--------------|------------|------------------|-------------------|
| 10,000    | 60      | 0            | 1          | 60               | 99.4%             |
| 10,000    | 60      | 1            | 2          | 120              | 98.8%             |
| 10,000    | 60      | 4            | 5          | 300              | 97.0%             |
| 10,000    | 120     | 0            | 1          | 120              | 98.8%             |
| 10,000    | 120     | 4            | 5          | 600              | 94.0%             |

**Key Insights:**
- ✅ More boundaries = lower expected coverage
- ✅ Longer sequences = lower expected coverage
- ✅ Coverage < 100% is NORMAL and EXPECTED
- ✅ Only warn when significantly below expected (>5%)

---

## Gap Detection Formula Explained

### Gap Threshold Calculation

```python
median_diff = time_diffs.median()  # Normal bar spacing
gap_threshold = median_diff * GAP_DETECTION_MULTIPLIER  # 2.0x
```

### Examples

| Bar Resolution | Median Diff | Gap Threshold | What Gets Detected        |
|----------------|-------------|---------------|---------------------------|
| 1 min          | 1 min       | 2 min         | Gaps > 2 min              |
| 5 min          | 5 min       | 10 min        | Gaps > 10 min             |
| 15 min         | 15 min      | 30 min        | Gaps > 30 min             |
| 1 hour         | 1 hour      | 2 hours       | Gaps > 2 hours            |
| 1 day          | 1 day       | 2 days        | Gaps > 2 days             |

**What gets detected as gaps:**
- ✅ Weekends (for intraday data)
- ✅ Holidays (market closures)
- ✅ Missing data periods
- ✅ Intraday gaps (lunch breaks, after-hours)

**What doesn't get detected (by design):**
- ✅ Single missing bar (< 2x threshold)
- ✅ Normal market hours gaps (within threshold)
- ✅ Short interruptions (avoid false positives)

---

## Message Type Guidelines

### INFO Level (Normal Operation)
- ✅ Expected behavior and normal operation
- ✅ Configuration details
- ✅ Summary statistics within normal range
- ✅ Successful completion

**Example:**
```
INFO: Coverage 87.5% (1250 samples missing) - EXPECTED for seq_len=60 with 2 segments.
```

### WARNING Level (Action Required)
- ⚠️ Unexpected behavior or potential problems
- ⚠️ Values significantly outside expected range
- ⚠️ Data quality issues
- ⚠️ Configuration problems

**Example:**
```
WARNING: Coverage 60% is UNEXPECTEDLY LOW (expected ~85.0% for seq_len=60, 2 segments).
```

### DEBUG Level (Detailed Trace)
- 🔍 Implementation details
- 🔍 Fallback logic activation
- 🔍 Internal state changes
- 🔍 Diagnostic information

**Example:**
```
DEBUG: Symbol column 'symbol' not found, attempting gap detection
DEBUG: No significant time gaps detected (resolution: 0 days 00:05:00)
```

---

## Quick Troubleshooting Guide

### "Coverage is UNEXPECTEDLY LOW"

**Possible causes:**
1. Data quality issues (excessive missing data)
2. Too many gaps in the data
3. Incorrect DatetimeIndex (non-uniform spacing)
4. Too many symbols/segments for the sequence length

**How to investigate:**
1. Check `n_segments` in the message (many segments = lower coverage)
2. Look for gap detection messages (many gaps = more boundaries)
3. Verify your data has uniform time spacing
4. Consider reducing `seq_len` or increasing data quality

**Example diagnosis:**
```
WARNING: Coverage 60% is UNEXPECTEDLY LOW (expected ~85.0% for seq_len=60, 2 segments).
INFO: Detected 104 time gaps (using boundary detection).
```
→ **Diagnosis:** Data has 104 gaps, creating many small segments. Each segment loses 60 samples at start, resulting in low coverage.

---

### "No significant time gaps detected" but expect gaps

**Possible causes:**
1. DatetimeIndex is missing or incorrect
2. Gaps are smaller than 2x threshold
3. Data has been forward-filled or interpolated

**How to investigate:**
1. Check if index is a DatetimeIndex: `isinstance(X.index, pd.DatetimeIndex)`
2. Check gap threshold in message (might be too large)
3. Examine your data preprocessing (are gaps being filled?)
4. Consider adjusting `GAP_DETECTION_MULTIPLIER` constant if needed

---

### "Boundary detection: none" when you expect detection

**Possible causes:**
1. No symbol column and no DatetimeIndex
2. Index is RangeIndex or Int64Index (not DatetimeIndex)
3. Data loaded without parsing dates

**How to fix:**
1. Ensure your DataFrame has a DatetimeIndex
2. When loading data: `pd.read_parquet(..., parse_dates=True)`
3. Or convert index: `df.index = pd.to_datetime(df.index)`

---

## Summary

### Key Improvements
- 🎯 **Accurate warnings:** Only warn when there's a real problem
- 📊 **Context-aware:** Messages explain expected vs actual behavior
- 🔍 **Transparent:** Shows what detection methods are active
- 💡 **Actionable:** Provides guidance on what to investigate
- 📈 **Educational:** Helps users understand sequence model behavior

### User Benefits
- ✅ No more confusion about "expected" warnings
- ✅ Clear distinction between normal and problematic situations
- ✅ Better understanding of sequence model coverage
- ✅ Easier troubleshooting of data quality issues
- ✅ Confidence that gap detection is working
