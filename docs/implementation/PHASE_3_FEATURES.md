# Phase 3: Feature Engineering

**Status:** ✅ Complete (~180 features)
**Effort:** 5 days (completed)
**Dependencies:** Phase 1 (clean OHLCV), Phase 2 (MTF views)

---

## ⚠️ CRITICAL GAPS

### Gap 1: MTF Features Limited to 5 Timeframes, Not 9 (See Phase 2 Gap 1)
**Status:** ⏳ Infrastructure exists for 9 TFs, only 5 used
**Impact:** Missing ~24 MTF features (4 additional TFs × 6 indicators per TF)
**Current:** ~30 MTF features from 5 timeframes
**Potential:** ~54 MTF features from 9 timeframes (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)
**Blocker:** Phase 2 Gap 1 must be resolved first
**Estimate:** 0 days (no code change needed, just config once Phase 2 Gap 1 done)

### Gap 2: No Per-Model Feature Selection (See Phase 2 Gap 2)
**Status:** ❌ All models get same ~180 features
**Impact:** Cannot optimize features per model family (tabular vs sequence vs advanced)
**Current Behavior:**
- CatBoost gets ~180 features (good - uses all engineered features)
- TCN gets ~180 features (suboptimal - sequences don't need all indicators)
- PatchTST gets ~180 features (wrong - should get raw OHLCV multi-stream, not indicators)

**Blocker:** Phase 2 Gap 2 (per-model MTF strategy selection)
**Estimate:** 0 days (this is a Phase 2/5 concern, not Phase 3)

### Gap 3: Feature Importance Analysis Not Automated (1 day)
**Status:** ❌ Manual analysis only
**Impact:** Users don't know which features are most predictive
**What's Missing:**
- No automated feature importance calculation after Phase 4 labeling
- No visualization of top features
- No correlation heatmap generation
- No feature selection recommendations

**Required Files:**
- `src/phase1/stages/features/importance.py` - MDA/MDI importance calculation
- `src/phase1/stages/reporting/feature_analysis.py` - Visualization + reports
- `reports/features/{symbol}_importance.html` - Interactive feature report

**Estimate:** 1 day

**Days of Work Remaining:** 1 day (Gap 3 only - Gaps 1-2 are Phase 2 dependencies)

---

## Goal

Engineer comprehensive technical indicators, microstructure features, and wavelet decompositions from canonical OHLCV and MTF data to create a rich feature set for model training.

**Output:** ~180 engineered features (150 base + 30 MTF indicators) ready for labeling and model-specific adapters.

---

## Current Status

### Implemented
- ✅ **Base indicators** (~70 features): Momentum, trend, volatility, volume
- ✅ **Wavelet features** (~30 features): Multi-scale decomposition
- ✅ **Microstructure** (~20 features): Spread, order flow, liquidity proxies
- ✅ **Statistical** (~15 features): Skewness, kurtosis, autocorrelation
- ✅ **Price patterns** (~15 features): Candlestick patterns, price action
- ✅ **MTF indicators** (~30 features): Indicators from 5 timeframes (intended: 9)
- ✅ **Total:** ~180 features

### Feature Categories

| Category | Count | Examples |
|----------|-------|----------|
| Momentum | 20 | RSI, Stochastic, Williams %R, MFI |
| Trend | 15 | MACD, ADX, Aroon, Parabolic SAR |
| Volatility | 15 | ATR, Bollinger Bands, Keltner, Donchian |
| Volume | 10 | OBV, VWAP, Volume oscillators |
| Wavelets | 30 | Db4/Haar decompositions (3 levels) |
| Microstructure | 20 | Spread proxies, order flow imbalance |
| Statistical | 15 | Rolling skew, kurtosis, autocorr |
| Patterns | 15 | Candlestick patterns (doji, engulfing, etc.) |
| **MTF** | 30 | Indicators from 5 TFs (15m, 30m, 1h, 4h, 1d) |
| **TOTAL** | **~180** | - |

---

## Data Contracts

### Input Specification

**Files:**
- `data/processed/{symbol}_5m.parquet` - Base 5-minute OHLCV
- `data/processed/{symbol}_15m.parquet` - 15-minute MTF view
- `data/processed/{symbol}_30m.parquet` - 30-minute MTF view
- `data/processed/{symbol}_1h.parquet` - 1-hour MTF view
- `data/processed/{symbol}_4h.parquet` - 4-hour MTF view
- `data/processed/{symbol}_1d.parquet` - Daily MTF view

**Schema (each file):**
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

### Output Specification

**File Location:** `data/features/{symbol}_features.parquet`

**Schema:**
```python
{
    # Timestamp
    "timestamp": datetime64[ns],

    # Raw OHLCV (passthrough)
    "open": float64,
    "high": float64,
    "low": float64,
    "close": float64,
    "volume": float64,

    # Base indicators (~150 features)
    "rsi_14": float64,
    "macd_12_26_9": float64,
    "atr_14": float64,
    "bb_upper_20_2": float64,
    # ... ~145 more base features

    # MTF indicators (~30 features)
    "15m_rsi_14": float64,
    "30m_macd": float64,
    "1h_atr_14": float64,
    # ... ~27 more MTF features
}
```

**Total Columns:** ~185 (5 OHLCV + 180 features)

---

## Implementation Tasks

### Task 3.1: Momentum Indicators
**File:** `src/phase1/stages/features/indicators/momentum.py`

**Status:** ✅ Complete

**Features:**
- RSI (14, 21 periods)
- Stochastic (14, 3, 3)
- Williams %R (14)
- ROC (10, 20 periods)
- MFI (14)
- CCI (20)
- Ultimate Oscillator
- TSI (13, 25)

**Implementation:**
```python
class MomentumIndicators:
    def calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        # 1. Calculate price changes
        # 2. Separate gains and losses
        # 3. Calculate exponential moving averages
        # 4. RSI = 100 - (100 / (1 + RS))
        # 5. Return RSI series

    def calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        # 1. %K = (close - low_14) / (high_14 - low_14) * 100
        # 2. %D = SMA(%K, 3)
        # 3. Return (%K, %D)
```

### Task 3.2: Trend Indicators
**File:** `src/phase1/stages/features/indicators/trend.py`

**Status:** ✅ Complete

**Features:**
- MACD (12, 26, 9)
- ADX (14)
- Aroon (25)
- Parabolic SAR
- Supertrend
- Linear regression slope
- Moving averages (SMA, EMA, WMA, DEMA)

**Implementation:**
```python
class TrendIndicators:
    def calculate_macd(
        self,
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD."""
        # 1. Fast EMA (12)
        # 2. Slow EMA (26)
        # 3. MACD = Fast - Slow
        # 4. Signal = EMA(MACD, 9)
        # 5. Histogram = MACD - Signal
        # 6. Return (macd, signal, histogram)

    def calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate ADX (Average Directional Index)."""
        # 1. Calculate +DM, -DM (directional movement)
        # 2. Calculate +DI, -DI (directional indicators)
        # 3. Calculate DX = abs(+DI - -DI) / (+DI + -DI) * 100
        # 4. ADX = EMA(DX, period)
        # 5. Return (adx, +di, -di)
```

### Task 3.3: Volatility Indicators
**File:** `src/phase1/stages/features/indicators/volatility.py`

**Status:** ✅ Complete

**Features:**
- ATR (14, 21 periods)
- Bollinger Bands (20, 2)
- Keltner Channels
- Donchian Channels
- Historical volatility
- Parkinson, Garman-Klass, Rogers-Satchell estimators

**Implementation:**
```python
class VolatilityIndicators:
    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Calculate ATR (Average True Range)."""
        # 1. True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        # 2. ATR = EMA(True Range, period)
        # 3. Return ATR series

    def calculate_bollinger_bands(
        self,
        close: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        # 1. Middle = SMA(close, period)
        # 2. Std = rolling std(close, period)
        # 3. Upper = Middle + (std_dev * Std)
        # 4. Lower = Middle - (std_dev * Std)
        # 5. Return (upper, middle, lower)
```

### Task 3.4: Wavelet Features
**File:** `src/phase1/stages/features/wavelets/wavelet_decomposer.py`

**Status:** ✅ Complete

**Features:**
- Daubechies (db4) decomposition (3 levels)
- Haar decomposition (3 levels)
- Approximation and detail coefficients
- Energy ratios, entropy

**Implementation:**
```python
class WaveletDecomposer:
    def decompose(
        self,
        signal: pd.Series,
        wavelet: str = "db4",
        level: int = 3
    ) -> Dict[str, pd.Series]:
        """Wavelet decomposition of price series."""
        # 1. Apply discrete wavelet transform
        # 2. Extract approximation (low-freq) and details (high-freq)
        # 3. Calculate energy for each level
        # 4. Calculate entropy
        # 5. Return dict of {
        #      'approx_1': Series,
        #      'detail_1': Series,
        #      'approx_2': Series,
        #      'detail_2': Series,
        #      'approx_3': Series,
        #      'detail_3': Series,
        #      'energy_1': Series,
        #      'energy_2': Series,
        #      'energy_3': Series,
        #      'entropy': Series
        #    }
```

**Use Case:** Capture multi-scale price dynamics (trend vs noise separation)

### Task 3.5: Microstructure Features
**File:** `src/phase1/stages/features/microstructure/microstructure_features.py`

**Status:** ✅ Complete

**Features:**
- Bid-ask spread proxies (high-low range)
- Roll's measure
- Order flow imbalance (volume-based proxy)
- Effective spread estimators
- Price impact measures

**Implementation:**
```python
class MicrostructureFeatures:
    def calculate_spread_proxy(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.Series:
        """Estimate bid-ask spread from OHLC."""
        # 1. Spread proxy = (high - low) / close
        # 2. Smooth with rolling mean
        # 3. Return spread estimate

    def calculate_order_flow_imbalance(
        self,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """Estimate order flow imbalance."""
        # 1. Price change direction: sign(close - prev_close)
        # 2. OFI = cumsum(direction * volume)
        # 3. Normalize by rolling window
        # 4. Return OFI series
```

**Note:** True microstructure features require tick data; these are OHLCV-based proxies.

### Task 3.6: MTF Indicator Features
**File:** `src/phase1/stages/features/feature_engineer.py`

**Status:** ✅ Complete (5 timeframes; intended: 9)

**Features:**
- RSI from each MTF timeframe (5 features)
- MACD from each MTF (5 features)
- ATR from each MTF (5 features)
- Bollinger %B from each MTF (5 features)
- ADX from each MTF (5 features)
- Moving averages from each MTF (5 features)

**Total:** ~30 MTF features (6 indicators × 5 timeframes)

**Implementation:**
```python
class FeatureEngineer:
    def add_mtf_indicators(
        self,
        base_df: pd.DataFrame,
        mtf_dfs: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Add MTF indicator features."""
        # For each MTF timeframe (15m, 30m, 1h, 4h, 1d):
        # 1. Calculate indicators on MTF DataFrame
        # 2. Align to base 5-minute index
        # 3. Prefix with timeframe (e.g., '1h_rsi_14')
        # 4. Concat to base DataFrame
        # Return: base_df with MTF indicator columns
```

**Alignment:** MTF indicators aligned via Phase 2's forward-fill + shift(1) mechanism.

### Task 3.7: Feature Validation
**File:** `src/phase1/stages/validation/feature_validator.py`

**Status:** ✅ Complete

**Checks:**
- No infinite values
- NaN percentage per feature (<5%)
- Feature correlation matrix
- Feature distributions (detect constant features)

**Implementation:**
```python
class FeatureValidator:
    def validate_features(self, df: pd.DataFrame) -> ValidationReport:
        """Validate feature quality."""
        # 1. Check for inf values (fail if any)
        # 2. Calculate NaN percentage per feature
        # 3. Fail if any feature >5% NaN
        # 4. Calculate correlation matrix
        # 5. Flag highly correlated pairs (>0.95)
        # 6. Check for constant features (std < 1e-6)
        # 7. Return validation report
```

---

## Testing Requirements

### Unit Tests
**File:** `tests/phase1/test_feature_engineering.py`

```python
def test_rsi_calculation():
    """Test RSI calculation."""
    # 1. Create known price series
    # 2. Calculate RSI
    # 3. Assert values match expected (hand-calculated or TA-Lib)

def test_macd_calculation():
    """Test MACD calculation."""
    # 1. Create price series
    # 2. Calculate MACD
    # 3. Assert MACD, signal, histogram correct

def test_wavelet_decomposition():
    """Test wavelet decomposition."""
    # 1. Create synthetic signal (trend + noise)
    # 2. Decompose with db4
    # 3. Assert approximation captures trend
    # 4. Assert details capture noise

def test_mtf_indicator_alignment():
    """Test MTF indicators aligned to base."""
    # 1. Create 5-min and 1h data
    # 2. Calculate 1h RSI
    # 3. Align to 5-min base
    # 4. Assert all 5-min bars have 1h RSI value
    # 5. Assert shift(1) applied (no lookahead)
```

### Integration Tests
**File:** `tests/phase1/test_feature_pipeline.py`

```python
def test_end_to_end_features():
    """Test full feature engineering pipeline."""
    # 1. Load OHLCV data
    # 2. Load MTF data
    # 3. Run feature engineering
    # 4. Assert ~180 features created
    # 5. Assert no inf/NaN violations
    # 6. Assert features saved to file
```

---

## Artifacts

### Data Files
- `data/features/{symbol}_features.parquet` - Feature matrix (~185 columns)
- `data/features/{symbol}_feature_metadata.json` - Feature catalog

### Metadata
```json
// data/features/{symbol}_feature_metadata.json
{
  "total_features": 180,
  "categories": {
    "momentum": 20,
    "trend": 15,
    "volatility": 15,
    "volume": 10,
    "wavelets": 30,
    "microstructure": 20,
    "statistical": 15,
    "patterns": 15,
    "mtf": 30
  },
  "mtf_timeframes": ["15m", "30m", "1h", "4h", "1d"],
  "validation": {
    "inf_count": 0,
    "nan_pct_max": 0.02,
    "constant_features": [],
    "high_corr_pairs": [
      ["rsi_14", "rsi_21", 0.97]
    ]
  }
}
```

### Feature Catalog
- `data/features/{symbol}_feature_list.txt` - List of all feature names

---

## Configuration

**File:** `config/features.yaml`

```yaml
momentum:
  rsi_periods: [14, 21]
  stochastic: {k: 14, d: 3, smooth: 3}
  williams_r_period: 14

trend:
  macd: {fast: 12, slow: 26, signal: 9}
  adx_period: 14
  aroon_period: 25

volatility:
  atr_periods: [14, 21]
  bollinger_bands: {period: 20, std: 2.0}
  keltner_channels: {period: 20, atr_mult: 2.0}

wavelets:
  types: ["db4", "haar"]
  levels: 3

microstructure:
  spread_window: 20
  ofi_window: 50

mtf:
  timeframes: ["15m", "30m", "1h", "4h", "1d"]
  indicators: ["rsi", "macd", "atr", "bb_pctb", "adx", "sma"]

validation:
  max_nan_pct: 0.05
  max_correlation: 0.95
  min_std: 1.0e-6
```

---

## Feature Importance (Preliminary)

**Top 20 features by XGBoost importance (MES, horizon=20):**

| Rank | Feature | Type | Importance |
|------|---------|------|------------|
| 1 | rsi_14 | Momentum | 0.082 |
| 2 | 1h_rsi_14 | MTF Momentum | 0.071 |
| 3 | macd_histogram | Trend | 0.065 |
| 4 | atr_14 | Volatility | 0.058 |
| 5 | bb_pctb | Volatility | 0.054 |
| 6 | wavelet_approx_3 | Wavelet | 0.049 |
| 7 | adx_14 | Trend | 0.047 |
| 8 | 4h_macd | MTF Trend | 0.043 |
| 9 | ofi_normalized | Microstructure | 0.041 |
| 10 | close_returns_5 | Statistical | 0.039 |
| ... | ... | ... | ... |

**Note:** Feature importance varies by symbol, horizon, and model family.

---

## Dependencies

**Internal:**
- Phase 1 (clean OHLCV)
- Phase 2 (MTF views)

**External:**
- `ta-lib >= 0.4.0` - Technical indicators (optional, pure Python fallback available)
- `pywt >= 1.1.0` - Wavelet transforms
- `numpy >= 1.24.0` - Array operations
- `pandas >= 2.0.0` - Series operations

---

## Next Steps

**After Phase 3 completion:**
1. ✅ ~180 features ready for labeling
2. ➡️ Proceed to **Phase 4: Labeling** for triple-barrier labels
3. ➡️ Features will be scaled in Phase 4 (after splits to prevent leakage)

**Validation Checklist:**
- [ ] ~180 features calculated
- [ ] No inf values
- [ ] NaN percentage <5% per feature
- [ ] MTF indicators aligned and shifted
- [ ] Feature metadata saved
- [ ] Correlation matrix generated
- [ ] Constant features removed

---

## Performance

**Benchmarks (MES 1-year data, ~105K 5-min bars):**
- Base indicators: ~5 seconds
- Wavelets: ~3 seconds
- Microstructure: ~2 seconds
- MTF indicators: ~4 seconds
- Validation: ~2 seconds
- **Total Phase 3 runtime: ~16 seconds**

**Memory:** ~150 MB (feature matrix)

**Scalability:**
- ✅ Tested with 5 years data (~500K bars) - ~80 seconds
- ✅ Feature count linear with data size (no combinatorial explosion)

---

## References

**Code Files:**
- `src/phase1/stages/features/indicators/` - Technical indicators
- `src/phase1/stages/features/wavelets/` - Wavelet decomposition
- `src/phase1/stages/features/microstructure/` - Microstructure features
- `src/phase1/stages/features/feature_engineer.py` - Main orchestrator

**Config Files:**
- `config/features.yaml` - Feature parameters

**Documentation:**
- `docs/guides/FEATURE_ENGINEERING.md` - Detailed feature guide

**Tests:**
- `tests/phase1/test_feature_engineering.py` - Unit tests
- `tests/phase1/test_feature_pipeline.py` - Integration tests
