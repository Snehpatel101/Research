# Phase 2: Configurable Primary Timeframe & MTF Strategies

**Status:** ⚠️ Partial (hardcoded to 5min base, MTF optional enrichment not configurable)
**Effort:** 3 days (2 days remaining)
**Dependencies:** Phase 1 (clean 1-minute OHLCV)

---

## Goal

Enable flexible primary timeframe selection (5min, 10min, 15min, 1h, etc.) for training, with optional multi-timeframe enrichment strategies. The primary timeframe is NOT hardcoded to 5-minutes - users can train on any timeframe.

**Output:** Configurable primary timeframe + optional MTF enrichment (indicators or raw multi-stream data) ready for feature engineering and model-specific adapters.

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

**Roadmap:** See `docs/archive/implementation/MTF_IMPLEMENTATION_ROADMAP.md` for detailed 9-timeframe plan.

---

## ⚠️ CRITICAL GAPS

### Gap 1: 9-Timeframe Infrastructure Exists But Only 5 TFs Used by Default (1-2 days)
**Status:** ⏳ Partial Implementation
**Impact:** Users cannot easily access all 9 timeframes without manual config changes
**What's Actually There:**
- ✅ All 9 TFs defined in `src/phase1/stages/mtf/constants.py` (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)
- ✅ Infrastructure supports all 9 timeframes
- ❌ `DEFAULT_MTF_TIMEFRAMES` hardcoded to only 5: `["15min", "30min", "1h", "4h", "daily"]`
- ❌ No config option to easily enable all 9 TFs

**Required Changes:**
1. Update `config/pipeline.yaml` to include toggle for 9-TF mode
2. Add validation that warns users about default vs full ladder
3. Update `MTFFeatureGenerator` to use config-driven timeframe selection
4. Document performance implications (5 TFs vs 9 TFs)

**Files to Modify:**
- `src/phase1/stages/mtf/constants.py` - Add `FULL_MTF_LADDER` constant
- `src/phase1/stages/mtf/generator.py` - Support config-driven TF selection
- `config/pipeline.yaml` - Add `use_full_mtf_ladder: bool` option
- `docs/guides/MTF_CONFIGURATION.md` - Document 5-TF vs 9-TF tradeoffs

**Blockers:** None (infrastructure exists)
**Estimate:** 1-2 days (config wiring + validation + docs)

### Gap 2: No Per-Model MTF Strategy Selection (2-3 days)
**Status:** ❌ Not Implemented
**Impact:** Cannot train CatBoost with MTF indicators while TCN uses single-TF in same experiment
**What's Missing:**
- No per-model MTF strategy configuration
- Cannot mix Strategy 1 (single-TF) + Strategy 2 (MTF indicators) + Strategy 3 (MTF ingestion) in one experiment
- Each model must use same MTF strategy globally

**Required Changes:**
1. Extend `TrainerConfig` to include `mtf_strategy` per model
2. Modify model router to load different feature sets based on strategy
3. Add validation that model family supports chosen strategy
4. Update training scripts to accept per-model MTF config

**Example Config (Target):**
```yaml
models:
  catboost:
    mtf_strategy: "mtf_indicators"  # Use MTF features
    primary_tf: "15min"
  tcn:
    mtf_strategy: "single_tf"  # No MTF, just 5min
    primary_tf: "5min"
  patchtst:
    mtf_strategy: "mtf_ingestion"  # Multi-stream OHLCV
    primary_tf: "1min"
```

**Files to Create:**
- `src/models/config/mtf_strategy.py` - MTF strategy enum and validator
- `src/phase1/stages/datasets/loaders.py` - Strategy-specific data loaders

**Files to Modify:**
- `src/models/trainer.py` - Route to strategy-specific adapters
- `src/models/config/trainer_config.py` - Add `mtf_strategy` field

**Blockers:** None
**Estimate:** 2-3 days (config system + routing + validation + tests)

### Gap 3: MTF Ingestion (Strategy 3) Not Wired to Multi-Res Adapter (1 day)
**Status:** ⚠️ Adapter Exists, Not Connected
**Impact:** Advanced models (PatchTST, iTransformer, TFT) cannot train on multi-stream OHLCV
**Surprise Finding:** Multi-resolution 4D adapter IS implemented (`src/phase1/stages/datasets/adapters/multi_resolution.py`, 619 lines)
**What's Missing:**
- Adapter not registered in model trainer routing logic
- No integration tests for 4D adapter
- Not documented in Phase 5 or Phase 6 docs
- No example configs showing how to use it

**Required Changes:**
1. Wire `MultiResolution4DAdapter` into `ModelTrainer.prepare_data()`
2. Add family="advanced" routing logic
3. Create example config for PatchTST using 4D adapter
4. Add integration tests

**Files to Modify:**
- `src/models/trainer.py` - Add routing for family="advanced"
- `docs/implementation/PHASE_5_ADAPTERS.md` - Update status to ✅ Complete
- `config/models/patchtst.yaml` - Example config using 4D adapter

**Files to Create:**
- `tests/phase1/test_multi_resolution_adapter.py` - Integration tests

**Blockers:** None (adapter fully implemented, just needs wiring)
**Estimate:** 1 day (routing + tests + docs)

### Gap 4: Configurable Primary Timeframe Per Model (1-2 days)
**Status:** ❌ Hardcoded to 5min
**Impact:** Cannot train CatBoost on 15min while TCN trains on 5min
**What's Missing:**
- Primary timeframe selection hardcoded in pipeline
- No per-model timeframe configuration
- All models must train on same primary TF

**Required Changes:**
1. Add `primary_timeframe` to per-model config
2. Modify data pipeline to resample canonical 1-min to requested primary TF
3. Ensure proper alignment and feature derivation per TF
4. Update feature engineering to work on any primary TF

**Files to Modify:**
- `src/phase1/stages/clean/data_cleaner.py` - Parameterize resampling TF
- `src/phase1/stages/features/feature_engineer.py` - TF-agnostic features
- `src/models/config/trainer_config.py` - Add `primary_timeframe` field

**Blockers:** Requires Gap 2 (per-model config) to be useful
**Estimate:** 1-2 days (pipeline changes + validation)

**Days of Work Remaining:** 5-8 days (Gaps 1-4 combined)

---

## Configurable Primary Timeframe

### Primary Timeframe Selection (Not Yet Configurable)

**Current:** Hardcoded to 5-minute base timeframe
**Intended:** User specifies primary training timeframe per experiment

**Available Options:**
- 5min, 10min, 15min, 20min, 25min, 30min, 45min, 1h
- Each experiment chooses ONE primary timeframe
- All features computed on primary timeframe

**Configuration (Future):**
```yaml
phase2:
  primary_timeframe: "15min"  # User-configurable
  mtf_strategy: "single_tf"   # OR "mtf_indicators", "mtf_ingestion"
```

---

## Architecture: Three MTF Strategies (Optional Enrichment)

### Strategy 1: Single-TF (Baseline - No MTF Enrichment)
**Purpose:** Train on chosen primary timeframe without MTF features
**Data:** Features from ONE timeframe only (e.g., only 15-minute)
**Models:** All model families
**Status:** ❌ Not implemented (simple config flag)

### Strategy 2: MTF Indicators (Optional Enrichment for Tabular Models)
**Purpose:** Add indicator features from OTHER timeframes to enrich primary TF
**Data:** Primary TF features + indicators from other TFs (~180 total)
**Models:** Tabular models (Boosting, Classical)
**Status:** ⚠️ Partial (hardcoded to 5min base + 5 other TFs; intended: configurable primary + flexible MTF)

### Strategy 3: MTF Ingestion (Optional Multi-Stream for Sequence Models)
**Purpose:** Feed multi-stream raw OHLCV bars from multiple timeframes
**Data:** Multi-stream inputs (e.g., separate 5min, 15min, 1h streams)
**Models:** Sequence models (Neural, CNN, Transformer, MLP)
**Status:** ❌ Not implemented

**Key Insight:** Models can mix-and-match strategies in same experiment (e.g., CatBoost uses MTF indicators, TCN uses single-TF)

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

### Task 2.3: Make Primary Timeframe Configurable (TODO)
**File:** `src/phase1/stages/mtf/mtf_scaler.py`

**Status:** ❌ Not implemented

**Implementation:**
```python
class MTFScaler:
    def __init__(self, primary_timeframe: str = "5min"):
        """
        Args:
            primary_timeframe: Primary training timeframe (5min, 10min, 15min, 1h, etc.)
        """
        self.primary_timeframe = primary_timeframe

        # Available timeframes for resampling
        self.available_timeframes = [
            "1min", "5min", "10min", "15min", "20min",
            "25min", "30min", "45min", "1h", "4h", "1d"
        ]

    def create_primary_tf_data(
        self,
        df_1m: pd.DataFrame
    ) -> pd.DataFrame:
        """Resample 1-min data to user-specified primary timeframe."""
        # 1. Resample to primary timeframe
        # 2. Aggregate OHLCV
        # 3. Return primary TF dataframe

    def create_mtf_enrichment(
        self,
        df_primary: pd.DataFrame,
        strategy: str = "single_tf"
    ) -> Dict[str, pd.DataFrame]:
        """Create MTF enrichment views based on strategy.

        Args:
            df_primary: Primary timeframe data
            strategy: 'single_tf' (no MTF), 'mtf_indicators', or 'mtf_ingestion'

        Returns:
            Dict of {timeframe: DataFrame} for MTF enrichment (empty if single_tf)
        """
        if strategy == "single_tf":
            return {}  # No MTF enrichment

        elif strategy == "mtf_indicators":
            # Create higher timeframes for indicator features
            return self._create_indicator_mtf(df_primary)

        elif strategy == "mtf_ingestion":
            # Create multi-stream raw OHLCV for sequence models
            return self._create_multi_stream_mtf(df_primary)
```

**Effort:** 1-2 days (refactor existing MTF logic to support configurable primary TF)

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
- `docs/archive/implementation/MTF_IMPLEMENTATION_ROADMAP.md` - Detailed 9-TF plan
- `docs/archive/guides/MTF_STRATEGY_GUIDE.md` - Strategy comparison

**Tests:**
- `tests/phase1/test_mtf_scaler.py` - Unit tests
- `tests/phase1/test_mtf_pipeline.py` - Integration tests
