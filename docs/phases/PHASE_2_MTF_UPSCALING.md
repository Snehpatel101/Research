# Phase 2: Multi-Timeframe Upscaling

**Status:** ⚠️ Partial (5 of 9 timeframes implemented)
**Effort:** 3 days (2 days remaining)
**Dependencies:** Phase 1 (clean 1-minute OHLCV)

---

## Goal

Create multi-timeframe (MTF) views of the canonical 1-minute OHLCV data by upscaling to higher timeframes, enabling both indicator-based features and raw multi-resolution tensor ingestion for advanced models.

**Output:** Aligned MTF datasets (1min, 5min, 15min, 30min, 1h, 4h, daily) ready for feature engineering and model-specific adapters.

---

## Current Status

### Implemented (Strategy 2: MTF Indicators)
- ✅ 5-minute base resampling from 1-minute data
- ✅ MTF upscaling to **5 timeframes**: 15min, 30min, 1h, 4h, daily
- ✅ Proper alignment (shift(1) to prevent lookahead)
- ✅ OHLCV aggregation rules (first open, max high, min low, last close, sum volume)
- ✅ MTF indicator features (~30 features from 5 timeframes)

### Not Yet Implemented
- ❌ **Complete 9-timeframe ladder**: 1min, 5min, 10min, 15min, 20min, 25min, 30min, 45min, 1h
- ❌ Strategy 1 (single-TF baselines for ablation studies)
- ❌ Strategy 3 (raw MTF OHLCV tensors for sequence models)

**Roadmap:** See `docs/archive/roadmaps/MTF_IMPLEMENTATION_ROADMAP.md` for detailed 9-timeframe plan.

---

## Architecture: Three MTF Strategies

### Strategy 1: Single-TF (Baselines)
**Purpose:** Ablation study to measure MTF value
**Data:** One timeframe only (e.g., only 5-minute data)
**Models:** All model families
**Status:** ❌ Not implemented

### Strategy 2: MTF Indicators (Current)
**Purpose:** Indicator features from multiple timeframes
**Data:** ~180 indicator-derived features (150 base + 30 MTF)
**Models:** Tabular (XGBoost, LightGBM, CatBoost, RF, Logistic, SVM)
**Status:** ⚠️ Partial (5 of 9 timeframes)

### Strategy 3: MTF Ingestion (Future)
**Purpose:** Raw multi-resolution OHLCV for temporal learning
**Data:** 4D tensors with raw OHLCV from 9 timeframes
**Models:** Sequence (LSTM, GRU, TCN, Transformer, PatchTST, iTransformer, TFT, N-BEATS)
**Status:** ❌ Not implemented

---

## Data Contracts

### Input Specification

**File Location:** `data/processed/{symbol}_1m_clean.parquet` (from Phase 1)

**Schema:**
```python
{
    "timestamp": datetime64[ns],
    "open": float64,
    "high": float64,
    "low": float64,
    "close": float64,
    "volume": float64
}
```

### Output Specification (Current: 5 Timeframes)

**File Locations:**
- `data/processed/{symbol}_5m.parquet`
- `data/processed/{symbol}_15m.parquet`
- `data/processed/{symbol}_30m.parquet`
- `data/processed/{symbol}_1h.parquet`
- `data/processed/{symbol}_4h.parquet`
- `data/processed/{symbol}_1d.parquet`

**Schema:** Same as input (OHLCV columns)

**Alignment Guarantee:**
- All MTF dataframes share common timestamps (outer join on 5-minute base)
- Forward-fill for alignment (e.g., 1h data aligned to every 5-minute bar)
- `shift(1)` applied to prevent lookahead bias

### Output Specification (Intended: 9 Timeframes)

**Additional Files:**
- `data/processed/{symbol}_1m.parquet` (passthrough from Phase 1)
- `data/processed/{symbol}_10m.parquet`
- `data/processed/{symbol}_20m.parquet`
- `data/processed/{symbol}_25m.parquet`
- `data/processed/{symbol}_45m.parquet`

**Total:** 9 aligned timeframe views

---

## Implementation Tasks

### Task 2.1: Base Resampling (5-minute)
**File:** `src/phase1/stages/clean/data_cleaner.py`

**Status:** ✅ Complete

**Implementation:**
```python
class DataCleaner:
    def resample_to_5min(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample 1-min to 5-min OHLCV."""
        # 1. Resample to 5-minute frequency
        # 2. Aggregate: first(open), max(high), min(low), last(close), sum(volume)
        # 3. Drop incomplete bars (start/end of dataset)
        # 4. Return resampled DataFrame
```

**Validation:**
- 5-minute bars properly aligned to clock (00, 05, 10, ...)
- No partial bars at boundaries
- Volume sums correctly

### Task 2.2: MTF Upscaling (Current 5 Timeframes)
**File:** `src/phase1/stages/mtf/mtf_scaler.py`

**Status:** ✅ Complete (5 timeframes)

**Implementation:**
```python
class MTFScaler:
    def create_mtf_views(self, base_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create multiple timeframe views from 5-minute base."""
        # Timeframes: 15min, 30min, 1h, 4h, 1d
        # 1. For each target timeframe:
        #    a. Resample base_df to target frequency
        #    b. Aggregate OHLCV (same rules as 5-min)
        #    c. Align to 5-minute index (forward-fill)
        #    d. Apply shift(1) to prevent lookahead
        # 2. Return dict of {timeframe: DataFrame}
```

**Aggregation Rules:**
```python
OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum"
}
```

**Lookahead Prevention:**
```python
# After resampling and alignment
df_mtf = df_mtf.shift(1)  # Use previous bar's value
```

### Task 2.3: Extend to 9 Timeframes (TODO)
**File:** `src/phase1/stages/mtf/mtf_scaler.py`

**Status:** ❌ Not implemented

**Implementation:**
```python
class MTFScaler:
    # Add support for 9-timeframe ladder
    TIMEFRAMES = [
        "1min",   # Passthrough from Phase 1
        "5min",   # Base timeframe
        "10min",  # NEW
        "15min",  # Current
        "20min",  # NEW
        "25min",  # NEW
        "30min",  # Current
        "45min",  # NEW
        "1h"      # Current
    ]

    def create_full_mtf_ladder(self, df_1m: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create all 9 timeframe views."""
        # 1. Passthrough 1-minute (already clean)
        # 2. Resample to 5-minute base
        # 3. For each higher timeframe:
        #    a. Resample from appropriate source (1m or 5m)
        #    b. Align to 5-minute index
        #    c. Apply shift(1)
        # 4. Return dict with all 9 timeframes
```

**Effort:** 1 day (straightforward extension of existing logic)

### Task 2.4: MTF Indicator Features (Current)
**File:** `src/phase1/stages/features/feature_engineer.py`

**Status:** ✅ Complete (but only 5 timeframes)

**Implementation:**
```python
class FeatureEngineer:
    def add_mtf_indicators(
        self,
        base_df: pd.DataFrame,
        mtf_dfs: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Add MTF indicator features to base DataFrame."""
        # For each MTF timeframe:
        # 1. Calculate indicators (RSI, MACD, ATR, etc.)
        # 2. Align to base 5-minute index
        # 3. Prefix with timeframe (e.g., '1h_rsi_14')
        # 4. Concat to base DataFrame
        # Return: base_df with ~30 MTF indicator columns
```

**MTF Indicators:**
- RSI (14-period)
- MACD (12, 26, 9)
- ATR (14-period)
- Bollinger Bands (20, 2)
- Moving averages (SMA, EMA)

**Total MTF Features:** ~30 (6 indicators × 5 timeframes)

### Task 2.5: MTF Raw Tensors (Strategy 3 - TODO)
**File:** `src/phase1/stages/mtf/mtf_tensor_builder.py`

**Status:** ❌ Not implemented

**Implementation:**
```python
class MTFTensorBuilder:
    def build_mtf_tensors(
        self,
        mtf_dfs: Dict[str, pd.DataFrame],
        lookback_config: Dict[str, int]
    ) -> np.ndarray:
        """Build 4D multi-resolution OHLCV tensors.

        Args:
            mtf_dfs: Dict of {timeframe: DataFrame}
            lookback_config: Dict of {timeframe: num_bars}
                Example: {'5min': 60, '15min': 20, '30min': 10, '1h': 5}

        Returns:
            4D array: (n_samples, n_timeframes, max_lookback, 4)
                - n_samples: number of prediction points
                - n_timeframes: 9 timeframes
                - max_lookback: longest lookback window
                - 4: OHLC features (volume separate or normalized)
        """
        # 1. For each prediction timestamp:
        #    a. For each timeframe:
        #       - Extract lookback window (e.g., last 60 bars for 5-min)
        #       - Pad if necessary (start of dataset)
        #       - Stack into tensor
        # 2. Return 4D array
```

**Effort:** 2-3 days (new capability, needs design review)

**Use Case:** PatchTST, iTransformer, TFT, N-BEATS models consume raw MTF tensors

---

## Testing Requirements

### Unit Tests
**File:** `tests/phase1/test_mtf_scaler.py`

```python
def test_resample_5min():
    """Test 1-min to 5-min resampling."""
    # 1. Create 1-min OHLCV (5 bars)
    # 2. Resample to 5-min
    # 3. Assert single bar with correct OHLC aggregation

def test_mtf_alignment():
    """Test MTF alignment to base 5-min index."""
    # 1. Create 5-min base (12 bars = 1 hour)
    # 2. Create 1h MTF view (1 bar)
    # 3. Align 1h to 5-min index
    # 4. Assert all 12 rows have same 1h value (forward-fill)

def test_lookahead_prevention():
    """Test shift(1) prevents lookahead."""
    # 1. Create 5-min data with known pattern
    # 2. Create 1h MTF with shift(1)
    # 3. Assert current bar uses previous 1h value
    # 4. Assert first bar is NaN (no previous value)

def test_9_timeframe_ladder():
    """Test full 9-timeframe creation."""
    # 1. Create 1-min data (1 day)
    # 2. Call create_full_mtf_ladder()
    # 3. Assert 9 DataFrames returned
    # 4. Assert all aligned to 5-min base
    # 5. Assert proper aggregation for each timeframe
```

### Integration Tests
**File:** `tests/phase1/test_mtf_pipeline.py`

```python
def test_end_to_end_mtf():
    """Test full MTF pipeline."""
    # 1. Load clean 1-min data (from Phase 1)
    # 2. Run MTF upscaling
    # 3. Assert all timeframe files created
    # 4. Assert no lookahead (shift validation)
    # 5. Assert alignment (common timestamps)
```

---

## Artifacts

### Data Files (Current)
- `data/processed/{symbol}_5m.parquet` - Base timeframe
- `data/processed/{symbol}_15m.parquet`
- `data/processed/{symbol}_30m.parquet`
- `data/processed/{symbol}_1h.parquet`
- `data/processed/{symbol}_4h.parquet`
- `data/processed/{symbol}_1d.parquet`

### Data Files (Intended)
- Add: `data/processed/{symbol}_1m.parquet` (passthrough)
- Add: `data/processed/{symbol}_10m.parquet`
- Add: `data/processed/{symbol}_20m.parquet`
- Add: `data/processed/{symbol}_25m.parquet`
- Add: `data/processed/{symbol}_45m.parquet`

### Metadata
- `data/processed/{symbol}_mtf_alignment_report.json` - Alignment validation

```json
{
  "base_timeframe": "5min",
  "mtf_timeframes": ["15min", "30min", "1h", "4h", "1d"],
  "alignment": {
    "common_start": "2023-01-01T00:00:00Z",
    "common_end": "2023-12-31T23:55:00Z",
    "base_bars": 105120,
    "aligned": true
  },
  "lookahead_check": {
    "shift_applied": true,
    "first_bar_nan": true
  }
}
```

---

## Configuration

**File:** `config/pipeline.yaml`

```yaml
phase2:
  mtf:
    base_timeframe: "5min"
    timeframes: ["15min", "30min", "1h", "4h", "1d"]  # Current
    # Intended: ["1min", "5min", "10min", "15min", "20min", "25min", "30min", "45min", "1h"]

    aggregation:
      open: "first"
      high: "max"
      low: "min"
      close: "last"
      volume: "sum"

    alignment:
      method: "forward_fill"
      apply_shift: true      # Prevent lookahead
      shift_periods: 1

    validation:
      check_alignment: true
      check_lookahead: true
      allow_first_nan: true  # Expected after shift
```

---

## Strategy Comparison

| Aspect | Strategy 1 (Single-TF) | Strategy 2 (MTF Indicators) | Strategy 3 (MTF Ingestion) |
|--------|------------------------|-----------------------------|-----------------------------|
| **Data Type** | One timeframe | Indicator features from 9 TFs | Raw OHLCV from 9 TFs |
| **Model Families** | All (baseline) | Tabular (Boosting + Classical) | Sequence (Neural + CNN + Advanced) |
| **Feature Count** | ~150 | ~180 (150 base + 30 MTF) | Variable (raw bars) |
| **Input Shape** | 2D: `(N, 150)` | 2D: `(N, 180)` | 4D: `(N, 9, T, 4)` |
| **Status** | ❌ Not implemented | ⚠️ Partial (5 TFs) | ❌ Not implemented |
| **Effort** | 0.5 days (config flag) | 1 day (extend to 9 TFs) | 3 days (new tensor builder) |

**Rollout Plan:**
1. Complete Strategy 2 (extend to 9 timeframes) - **1 day**
2. Implement Strategy 1 (config flag to disable MTF) - **0.5 days**
3. Implement Strategy 3 (MTF tensor builder) - **3 days**

---

## Dependencies

**Internal:**
- Phase 1 (clean 1-minute OHLCV)

**External:**
- `pandas >= 2.0.0` - Resampling operations
- `numpy >= 1.24.0` - Tensor operations (Strategy 3)

---

## Next Steps

**After Phase 2 completion:**
1. ✅ MTF datasets (9 timeframes) ready for feature engineering
2. ➡️ Proceed to **Phase 3: Feature Engineering** to calculate indicators on all timeframes
3. ➡️ Strategy 3 implementation enables advanced sequence models (Phase 6)

**Validation Checklist:**
- [ ] All 9 timeframes created (current: 5)
- [ ] Proper OHLCV aggregation verified
- [ ] Alignment to base timeframe validated
- [ ] Lookahead prevention confirmed (shift(1) applied)
- [ ] First bar NaN after shift (expected)
- [ ] MTF alignment report generated

---

## Performance

**Benchmarks (MES 1-year data, ~500K 1-min bars):**
- 5-minute resampling: ~1 second
- MTF upscaling (5 timeframes): ~2 seconds
- Alignment and shift: ~1 second
- **Total Phase 2 runtime: ~4 seconds**

**Memory:** ~80 MB (base + 5 MTF views)

**Scalability:**
- ✅ 9-timeframe extension adds ~1 second and ~30 MB
- ✅ Strategy 3 tensors: ~200 MB (4D arrays)

---

## References

**Code Files:**
- `src/phase1/stages/clean/data_cleaner.py` - 5-minute resampling
- `src/phase1/stages/mtf/mtf_scaler.py` - MTF upscaling (current)
- `src/phase1/stages/mtf/mtf_tensor_builder.py` - Strategy 3 (TODO)

**Config Files:**
- `config/pipeline.yaml` - MTF configuration

**Documentation:**
- `docs/archive/roadmaps/MTF_IMPLEMENTATION_ROADMAP.md` - Detailed 9-TF plan
- `docs/archive/guides/MTF_STRATEGY_GUIDE.md` - Strategy comparison

**Tests:**
- `tests/phase1/test_mtf_scaler.py` - Unit tests
- `tests/phase1/test_mtf_pipeline.py` - Integration tests
