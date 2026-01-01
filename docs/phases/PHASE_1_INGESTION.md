# Phase 1: Canonical OHLCV Ingestion

**Status:** ✅ Complete
**Effort:** 2 days (completed)
**Dependencies:** None (entry point)

---

## Goal

Ingest, validate, and prepare canonical 1-minute OHLCV data from raw files, ensuring data quality and establishing the foundation for all downstream processing.

**Output:** Clean, validated 1-minute OHLCV dataset ready for feature engineering and MTF upscaling.

---

## Current Status

### Implemented
- ✅ Raw data ingestion from `data/raw/{symbol}_1m.parquet` or `.csv`
- ✅ OHLCV schema validation (required columns, data types)
- ✅ Timestamp parsing and timezone handling (UTC)
- ✅ Duplicate removal (keep last)
- ✅ Gap detection and reporting
- ✅ Business hours filtering (futures sessions)
- ✅ Data quality metrics (completeness, gap analysis)
- ✅ Single-symbol isolation (one contract per run)

### Not Needed
- ❌ Gap filling (preserve authentic market gaps)
- ❌ Multi-symbol ingestion (single-contract architecture)
- ❌ Cross-symbol correlation (out of scope)

---

## Data Contracts

### Input Specification

**File Location:** `data/raw/{SYMBOL}_1m.parquet` or `data/raw/{SYMBOL}_1m.csv`

**Required Columns:**
```python
{
    "timestamp": datetime64[ns],  # UTC timezone
    "open": float64,              # Opening price
    "high": float64,              # High price
    "low": float64,               # Low price
    "close": float64,             # Closing price
    "volume": float64             # Trading volume
}
```

**Constraints:**
- `timestamp` must be monotonically increasing
- `high >= max(open, close)`
- `low <= min(open, close)`
- `volume >= 0`
- No NaN values in OHLCV columns

**Supported Symbols:**
- `MES` (E-mini S&P 500 futures)
- `MGC` (E-mini Gold futures)
- Any single futures contract with 1-minute bars

### Output Specification

**File Location:** `data/processed/{symbol}_1m_clean.parquet`

**Schema:** Same as input with additional metadata columns:
```python
{
    "timestamp": datetime64[ns],
    "open": float64,
    "high": float64,
    "low": float64,
    "close": float64,
    "volume": float64,
    "session_start": datetime64[ns],  # Session start time
    "session_end": datetime64[ns],    # Session end time
    "is_regular_hours": bool          # True if regular trading hours
}
```

**Quality Guarantees:**
- No duplicates
- Sorted by timestamp
- Valid OHLCV constraints
- Session boundaries identified
- Gaps documented in metadata

---

## Implementation Tasks

### Task 1.1: Raw Data Loading
**File:** `src/phase1/stages/ingest/data_ingestor.py`

**Implementation:**
```python
class DataIngestor:
    def load_raw_data(self, symbol: str) -> pd.DataFrame:
        """Load raw OHLCV from parquet or CSV."""
        # 1. Resolve path: data/raw/{symbol}_1m.parquet
        # 2. Read with pandas (parquet preferred, fallback to CSV)
        # 3. Parse timestamp column to datetime64[ns]
        # 4. Set timestamp as index
        # 5. Validate schema (all required columns present)
        # 6. Return DataFrame
```

**Validation:**
- Schema validation (required columns)
- Data type validation
- Raise `ValidationError` if constraints violated

### Task 1.2: Data Validation
**File:** `src/phase1/stages/ingest/data_ingestor.py`

**Implementation:**
```python
class DataIngestor:
    def validate_ohlcv(self, df: pd.DataFrame) -> None:
        """Validate OHLCV constraints."""
        # 1. Check high >= max(open, close)
        # 2. Check low <= min(open, close)
        # 3. Check volume >= 0
        # 4. Check no NaN in OHLCV columns
        # 5. Check timestamp monotonicity
        # 6. Raise ValidationError with details if fails
```

**Error Handling:**
- Clear error messages indicating which rows violate constraints
- Report first N violations (not all, to avoid spam)

### Task 1.3: Duplicate Removal
**File:** `src/phase1/stages/clean/data_cleaner.py`

**Implementation:**
```python
class DataCleaner:
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate timestamps, keep last."""
        # 1. Find duplicate timestamps
        # 2. Log count of duplicates
        # 3. Keep='last' (most recent data)
        # 4. Return deduplicated DataFrame
```

**Rationale:** `keep='last'` assumes later data is more accurate (corrections).

### Task 1.4: Gap Detection
**File:** `src/phase1/stages/clean/data_cleaner.py`

**Implementation:**
```python
class DataCleaner:
    def detect_gaps(self, df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
        """Detect gaps larger than expected frequency."""
        # 1. Calculate time deltas between consecutive timestamps
        # 2. Identify gaps > 1 minute (or expected frequency)
        # 3. Log gap locations and durations
        # 4. Store gap metadata for reporting
        # 5. Return df unchanged (gaps preserved)
```

**Gap Policy:** Preserve gaps (authentic market data), do not fill.

### Task 1.5: Session Filtering
**File:** `src/phase1/stages/sessions/session_filter.py`

**Implementation:**
```python
class SessionFilter:
    def filter_sessions(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Filter to trading hours for symbol."""
        # 1. Load session config for symbol (e.g., MES: 17:00-16:00 CT)
        # 2. Mark is_regular_hours column
        # 3. Optionally filter to regular hours (config-driven)
        # 4. Add session_start, session_end metadata
        # 5. Return filtered DataFrame
```

**Session Configuration:**
```yaml
# config/sessions.yaml
MES:
  timezone: "America/Chicago"
  regular_hours:
    start: "17:00"  # Sunday-Thursday
    end: "16:00"    # Next day
  holidays: []      # Holiday calendar
```

### Task 1.6: Quality Reporting
**File:** `src/phase1/stages/validation/data_validator.py`

**Implementation:**
```python
class DataValidator:
    def generate_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data quality metrics."""
        # 1. Calculate completeness (% non-NaN)
        # 2. Count gaps and total gap duration
        # 3. Count duplicates removed
        # 4. Calculate date range
        # 5. Calculate bar count
        # 6. Return metrics dict
```

**Metrics:**
- Total bars
- Date range (start, end)
- Completeness percentage
- Gap count and total duration
- Duplicates removed
- OHLCV constraint violations (should be 0)

---

## Testing Requirements

### Unit Tests
**File:** `tests/phase1/test_data_ingestor.py`

```python
def test_load_raw_data_parquet():
    """Test loading from parquet file."""
    # 1. Create synthetic 1m OHLCV parquet
    # 2. Load with DataIngestor
    # 3. Assert schema correct
    # 4. Assert timestamp index

def test_validate_ohlcv_valid():
    """Test validation passes for valid data."""
    # 1. Create valid OHLCV DataFrame
    # 2. Call validate_ohlcv
    # 3. Assert no errors raised

def test_validate_ohlcv_invalid_high():
    """Test validation fails when high < close."""
    # 1. Create invalid OHLCV (high < close)
    # 2. Call validate_ohlcv
    # 3. Assert ValidationError raised

def test_remove_duplicates():
    """Test duplicate removal keeps last."""
    # 1. Create DataFrame with duplicate timestamps
    # 2. Call remove_duplicates
    # 3. Assert duplicates removed
    # 4. Assert last value kept

def test_detect_gaps():
    """Test gap detection finds missing bars."""
    # 1. Create DataFrame with 10-minute gap
    # 2. Call detect_gaps
    # 3. Assert gap detected and logged
    # 4. Assert gap metadata recorded
```

### Integration Tests
**File:** `tests/phase1/test_ingestion_pipeline.py`

```python
def test_end_to_end_ingestion():
    """Test full ingestion pipeline."""
    # 1. Create raw test data file (data/raw/TEST_1m.parquet)
    # 2. Run DataIngestor.run(symbol="TEST")
    # 3. Assert output file created (data/processed/TEST_1m_clean.parquet)
    # 4. Assert schema correct
    # 5. Assert quality metrics generated
    # 6. Cleanup test files
```

### Regression Tests
**File:** `tests/phase1/test_ingestion_regression.py`

```python
def test_duplicate_timestamps_regression():
    """Regression test for duplicate timestamp bug."""
    # Previously: duplicates caused downstream errors
    # Now: duplicates removed, last value kept
    # Test specific case that caused original bug

def test_invalid_high_low_regression():
    """Regression test for invalid high/low data."""
    # Previously: invalid OHLCV constraints not caught
    # Now: validation fails with clear error
    # Test specific invalid data case
```

---

## Artifacts

### Data Files
- `data/raw/{symbol}_1m.parquet` - Raw input data
- `data/processed/{symbol}_1m_clean.parquet` - Clean output data
- `data/processed/{symbol}_1m_quality_report.json` - Quality metrics

### Logs
- `logs/phase1/ingestion_{timestamp}.log` - Ingestion run log
- `logs/phase1/validation_{timestamp}.log` - Validation results

### Reports
```json
// data/processed/{symbol}_1m_quality_report.json
{
  "symbol": "MES",
  "date_range": {
    "start": "2023-01-01T00:00:00Z",
    "end": "2023-12-31T23:59:00Z"
  },
  "total_bars": 525600,
  "completeness_pct": 99.8,
  "gaps": {
    "count": 12,
    "total_duration_minutes": 240
  },
  "duplicates_removed": 5,
  "ohlcv_violations": 0,
  "session_stats": {
    "regular_hours_bars": 450000,
    "extended_hours_bars": 75600
  }
}
```

---

## Configuration

**File:** `config/pipeline.yaml`

```yaml
phase1:
  ingestion:
    data_dir: "data/raw"
    output_dir: "data/processed"
    supported_formats: ["parquet", "csv"]
    required_columns: ["timestamp", "open", "high", "low", "close", "volume"]

  validation:
    check_ohlcv_constraints: true
    check_monotonicity: true
    allow_gaps: true        # Don't fail on gaps
    max_gap_minutes: 1440   # Warn if gap > 1 day

  cleaning:
    remove_duplicates: true
    duplicate_keep: "last"

  sessions:
    filter_to_regular_hours: false  # Keep all hours
    session_config_path: "config/sessions.yaml"
```

---

## Dependencies

**External:**
- `pandas >= 2.0.0` - DataFrame operations
- `pyarrow >= 10.0.0` - Parquet support
- `pyyaml >= 6.0` - Config loading

**Internal:**
- None (entry point to pipeline)

---

## Next Steps

**After Phase 1 completion:**
1. ✅ Clean 1-minute OHLCV dataset ready
2. ➡️ Proceed to **Phase 2: MTF Upscaling** to create multi-timeframe views
3. ➡️ Feature engineering will use both 1-minute and MTF data

**Validation Checklist:**
- [ ] Raw data file exists and is readable
- [ ] Schema validation passes
- [ ] OHLCV constraints validated
- [ ] Duplicates removed
- [ ] Gaps detected and logged
- [ ] Session metadata added
- [ ] Quality report generated
- [ ] Output file created in `data/processed/`

---

## Error Handling

### Common Issues

**Issue 1: Missing raw data file**
```
Error: File not found: data/raw/MES_1m.parquet
Solution: Ensure raw data file exists with correct naming convention
```

**Issue 2: Invalid OHLCV constraints**
```
Error: OHLCV validation failed - 15 rows with high < close
Solution: Inspect raw data, fix upstream data provider issues
```

**Issue 3: Duplicate timestamps**
```
Warning: 5 duplicate timestamps found, keeping last
Solution: Normal - duplicate removal is automatic
```

**Issue 4: Large gaps detected**
```
Warning: Gap detected: 2023-07-04 00:00 to 2023-07-05 00:00 (1440 minutes)
Solution: Expected for holidays/weekends - gaps preserved
```

### Validation Failures

**Critical failures (pipeline stops):**
- Missing required columns
- Invalid data types
- OHLCV constraint violations
- Non-monotonic timestamps

**Warnings (pipeline continues):**
- Gaps detected
- Duplicates removed
- Extended hours data present

---

## Performance

**Benchmarks (MES 1-year data, ~500K bars):**
- Load parquet: ~0.5 seconds
- Validation: ~1.0 seconds
- Duplicate removal: ~0.2 seconds
- Gap detection: ~0.5 seconds
- Session filtering: ~0.8 seconds
- **Total Phase 1 runtime: ~3 seconds**

**Memory:** ~50 MB for 1-year 1-minute data

**Scalability:**
- ✅ Tested with 5 years data (~2.5M bars) - ~15 seconds
- ✅ Single-symbol isolation keeps memory bounded
- ✅ No cross-symbol operations (no memory scaling issues)

---

## References

**Code Files:**
- `src/phase1/stages/ingest/data_ingestor.py` - Data loading
- `src/phase1/stages/clean/data_cleaner.py` - Cleaning operations
- `src/phase1/stages/sessions/session_filter.py` - Session filtering
- `src/phase1/stages/validation/data_validator.py` - Validation logic

**Config Files:**
- `config/pipeline.yaml` - Pipeline configuration
- `config/sessions.yaml` - Trading session definitions

**Tests:**
- `tests/phase1/` - All Phase 1 tests

**Documentation:**
- `docs/reference/PIPELINE_FLOW.md` - Visual pipeline overview
- `docs/QUICK_REFERENCE.md` - Command examples
