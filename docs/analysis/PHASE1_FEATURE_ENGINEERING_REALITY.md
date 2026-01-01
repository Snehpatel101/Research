# Phase 1 Feature Engineering Reality Check

**Date:** 2026-01-01
**Analyst:** Claude (Data Science Agent)
**Mission:** Validate documentation claims vs. actual implementation

---

## Executive Summary

**Finding:** All models receive identical ~168 indicator-derived features. No model-specific data preparation exists. MTF infrastructure supports mode-based routing but is unused for model differentiation.

**Status:**
- ✅ **Strategy 2 (MTF Indicators):** Partially implemented (5 of 9 timeframes)
- ❌ **Strategy 1 (Single-TF):** Not implemented (no baseline without MTF)
- ❌ **Strategy 3 (MTF Ingestion):** Not implemented (no multi-resolution tensors)

---

## 1. Feature Engineering Analysis

### 1.1 Actual Feature Count Breakdown

```
Base Indicators (5min):        ~64 features
  - Returns, ratios:             6
  - Moving averages (SMA/EMA):   7
  - Momentum (RSI, MACD, etc):  14
  - Volatility (ATR, BB, etc):  15
  - Volume/VWAP:                 5
  - Trend (ADX, Supertrend):     5
  - Temporal/Regime:             5
  - Autocorr, CLV, moments:      7

Microstructure (proxy):        10 features
  - Spread, Amihud, Roll, etc

Wavelets (db4, level=3):       ~24 features
  - 4 decomposition bands × 5 components (price)
  - 4 decomposition bands × 1 component (volume)

MTF Features (5 timeframes):   70 features
  - 5 OHLCV bars per TF:        25 bars
  - 9 indicators per TF:        45 indicators

─────────────────────────────────────────────
TOTAL:                         ~168 features
```

**Source:** `src/phase1/stages/features/engineer.py::engineer_features()`

### 1.2 Feature Generation Process

**All models receive:**
1. Base 5min indicators (momentum, volatility, volume, trend)
2. Microstructure proxy features (spread, liquidity, impact)
3. Wavelet decomposition features (multi-scale price/volume)
4. MTF indicator features from 5 timeframes (15min, 30min, 1h, 4h, daily)

**No conditional generation:** Feature pipeline is model-agnostic. The `FeatureEngineer` class has no awareness of model types.

**Code Evidence:**
```python
# src/phase1/stages/features/engineer.py, line 185-414
def engineer_features(self, df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, dict]:
    """Each symbol is processed independently - no cross-symbol correlation."""

    # Add ALL features unconditionally
    df = add_returns(df, self.feature_metadata)
    df = add_sma(df, ...)
    df = add_wavelets(df, ...)
    df = add_mtf_features(df, ...)  # MTF for ALL models
    # ...
    return df, feature_report
```

---

## 2. MTF Implementation Deep Dive

### 2.1 MTF Mode Infrastructure (Exists but Unused)

**MTFMode Enum:**
```python
# src/phase1/stages/mtf/constants.py
class MTFMode(str, Enum):
    BARS = 'bars'           # Only OHLCV at higher TFs
    INDICATORS = 'indicators'  # Only indicators at higher TFs
    BOTH = 'both'           # Both OHLCV + indicators
```

**Current Usage:** `DEFAULT_MTF_MODE = MTFMode.BOTH` → All models get both bars and indicators.

**Potential for Model-Specific Routing:**
```python
# INFRASTRUCTURE EXISTS but is not used for model differentiation
generator = MTFFeatureGenerator(
    base_timeframe='5min',
    mode=MTFMode.INDICATORS  # Could be: BARS for sequence models, INDICATORS for tabular
)
```

**Reality:** Pipeline always uses `mode='both'`, no differentiation by model family.

### 2.2 Timeframe Discrepancy: 5 vs 9

**Intended (docs):** 9-timeframe ladder
```
['1min', '5min', '10min', '15min', '20min', '25min', '30min', '45min', '1h']
```

**Actual (code):** 5 timeframes
```python
# src/phase1/stages/mtf/constants.py, line 46
DEFAULT_MTF_TIMEFRAMES = ['15min', '30min', '1h', '4h', 'daily']
```

**Missing:** `1min, 5min, 10min, 20min, 25min, 45min` (6 timeframes)

**Note:** 20min and 25min are not in `MTF_TIMEFRAMES` dict, so cannot be added without constant updates.

**Impact:** Lower temporal resolution than intended. Docs claim 9-TF fine-grained ladder, reality is 5 coarser timeframes.

### 2.3 MTF Bars: Flattened vs. Multi-Resolution

**Current MTF Bars:** Flattened 1D arrays
```python
# Example: 4h timeframe generates these columns
['open_4h', 'high_4h', 'low_4h', 'close_4h', 'volume_4h']

# Shape per sample: (5,) - just 5 scalar features
```

**Strategy 3 Requirement:** Multi-resolution tensors
```python
# For InceptionTime, PatchTST, iTransformer, etc.
{
    '5min':  np.ndarray(shape=(n_samples, 60, 4)),  # 60 bars of 5min OHLC
    '15min': np.ndarray(shape=(n_samples, 20, 4)),  # 20 bars of 15min OHLC
    '30min': np.ndarray(shape=(n_samples, 10, 4)),  # 10 bars of 30min OHLC
    '1h':    np.ndarray(shape=(n_samples, 5, 4)),   # 5 bars of 1h OHLC
}

# Shape per sample: (n_bars, 4) - preserves temporal structure
```

**Missing Infrastructure:**
- No multi-resolution sequence builder in `SequenceDataset`
- No hierarchical temporal encoding
- No alignment of different-resolution windows to same prediction point

**What Exists:**
```python
# src/phase1/stages/datasets/sequences.py
# Only creates 3D sequences from SAME resolution indicators
X_seq = features[start:end, :]  # (seq_len, n_features)
# Where n_features = 168 indicator-derived features (all same 5min base)
```

### 2.4 MTF Temporal Alignment (Anti-Lookahead)

**Implementation:** ✅ Correct - uses `shift(1)` on higher TFs

```python
# src/phase1/stages/mtf/generator.py, line 330-332
# ANTI-LOOKAHEAD: Shift MTF data by 1 period
df_mtf_shifted = df_mtf_idx.shift(1)

# This ensures a 4h bar at 12:00 is only visible to 5min bars from 12:00 onwards
# The 12:00-16:00 in-progress bar is NOT visible (would cause lookahead)
```

**Validation:** `validate_no_lookahead()` checks first row is NaN for all MTF features.

---

## 3. TimeSeriesDataContainer Analysis

### 3.1 Data Serving Capabilities

**Current Interface:**
```python
# src/phase1/stages/datasets/container.py
container = TimeSeriesDataContainer.from_parquet_dir(...)

# Tabular models (2D)
X_train, y_train, w_train = container.get_sklearn_arrays("train")
# Shape: (n_samples, 168)

# Sequence models (3D)
train_dataset = container.get_pytorch_sequences("train", seq_len=60)
# Returns SequenceDataset that yields: (60, 168) per sample
```

**Key Observation:** Both data types use the SAME 168 features. The only difference is reshaping:
- Tabular: `(n_samples, 168)`
- Sequence: `(n_samples, 60, 168)` - sliding window of last 60 bars

**No Model-Specific Feature Subsets:**
- No `feature_mask` parameter
- No `feature_columns` filtering by model type
- No separate data views (e.g., "tabular_features", "sequence_features")

### 3.2 Missing Infrastructure for Model-Specific Pipelines

**What Would Be Needed:**

```python
# Hypothetical model-specific data preparation
container = TimeSeriesDataContainer.from_parquet_dir(
    path="data/splits/scaled",
    horizon=20,
    model_type="lstm"  # ❌ NOT SUPPORTED
)

# Tabular models: get indicator features
X_tabular, y, w = container.get_tabular_features(
    split="train",
    feature_set="indicators"  # ❌ NOT SUPPORTED
)

# Sequence models: get multi-resolution OHLCV
X_sequences, y, w = container.get_mtf_sequences(
    split="train",
    timeframes=['5min', '15min', '30min', '1h'],  # ❌ NOT SUPPORTED
    bars_per_tf={'5min': 60, '15min': 20, ...}
)
```

**Reality:** Container is model-agnostic. It serves the same unified feature set to all models.

---

## 4. Feature Selection Capability

### 4.1 WalkForwardFeatureSelector (Exists but Unused in Pipeline)

**Location:** `src/cross_validation/feature_selector.py`

**Capabilities:**
- ✅ MDI/MDA/hybrid importance methods
- ✅ Walk-forward selection (no lookahead)
- ✅ Stability scoring (features selected across multiple folds)
- ✅ Clustered importance (handles multicollinearity)

**Usage:** Only in `scripts/run_cv.py` with `--select-features` flag. NOT integrated into main pipeline.

**Evidence:**
```bash
# Grep for feature selector usage in pipeline
$ grep -r "FeatureSelector\|select_features" src/phase1/
# No results in phase1 pipeline
```

**Implication:** All 168 features are used by all models. No automatic feature selection or model-specific subsets.

### 4.2 Infrastructure Gaps for Model-Specific Selection

**What's Missing:**
1. **Model-aware feature selection:** No routing like "Boosting models use indicators, RNNs use raw OHLCV"
2. **Family-specific feature sets:** No precomputed lists like `TABULAR_FEATURES`, `SEQUENCE_FEATURES`
3. **Dynamic feature filtering:** Container doesn't support `feature_mask` or `include_only` parameters

**Potential Architecture:**
```python
# Not implemented - conceptual design
FEATURE_STRATEGIES = {
    "tabular": {
        "families": ["boosting", "classical"],
        "feature_types": ["indicators", "mtf_indicators"],
        "exclude": ["mtf_bars"]  # Don't need raw OHLCV
    },
    "sequence": {
        "families": ["neural", "cnn", "advanced"],
        "feature_types": ["mtf_bars", "base_ohlcv"],
        "exclude": ["indicators"]  # Want raw bars, not derived
    }
}
```

---

## 5. Data Leakage Validation

### 5.1 MTF Temporal Leakage: ✅ PREVENTED

**Shift Logic:**
```python
# src/phase1/stages/mtf/generator.py, line 330-338
df_mtf_shifted = df_mtf_idx.shift(1)  # Use COMPLETED bars only
aligned = df_mtf_shifted.reindex(df_base_idx.index, method='ffill')
```

**Effect:** A 4h bar completed at 12:00 is shifted forward, so it's visible starting at 16:00 (next bar). This prevents using in-progress bar data.

**Validation:** `validate_no_lookahead()` checks first valid index > 0 for all MTF columns.

### 5.2 Purge/Embargo Leakage: ✅ PREVENTED

**Split Logic:**
```python
# src/phase1/stages/splits/core.py, line 171-173
purge_bars: int = 60       # 3 × max_horizon (20) = 60 bars
embargo_bars: int = 1440   # ~5 days at 5min (1440 bars)
```

**Purpose:**
- **Purge:** Remove samples near split boundaries where labels may overlap
- **Embargo:** Buffer period to prevent serial correlation leakage

**Calculation:**
- Purge: `max_horizon * 3 = 20 * 3 = 60` bars (~5 hours at 5min)
- Embargo: Fixed 1440 bars (~5 trading days at 5min)

**Effect:** Train/val/test splits have ~5-day gaps, preventing label leakage from adjacent periods.

### 5.3 Label Lookahead: ✅ PREVENTED

**Triple-Barrier Logic:**
```python
# Labels computed from FUTURE prices (horizon bars ahead)
# But labels are ASSIGNED to the current bar (prediction point)
# Example: label_h20 at t=100 uses prices from t=101 to t=120
```

**Validation:** `label_end_time_h{horizon}` column tracks when label is known, used for purged CV.

---

## 6. What SHOULD Be Refactored

### 6.1 Enable Model-Specific MTF Modes

**Current State:** All models get `MTFMode.BOTH` (bars + indicators)

**Proposed Refactor:**
```python
# src/phase1/stages/features/run.py
def run_feature_engineering(config, manifest, model_family=None):
    # Determine MTF mode based on model family
    if model_family in ["boosting", "classical"]:
        mtf_mode = MTFMode.INDICATORS  # Tabular models get indicator features
    elif model_family in ["neural", "cnn", "advanced"]:
        mtf_mode = MTFMode.BARS  # Sequence models get raw OHLCV bars
    else:
        mtf_mode = MTFMode.BOTH  # Default or ensemble

    engineer = FeatureEngineer(
        ...,
        mtf_mode=mtf_mode  # Pass model-aware mode
    )
```

**Benefit:** Reduces feature dimensionality for each model type, prevents boosting models from wasting features on raw OHLCV.

### 6.2 Implement Strategy 3: Multi-Resolution Sequence Builder

**Missing Component:** `MultiResolutionSequenceDataset`

**Requirements:**
```python
# Hypothetical implementation
class MultiResolutionSequenceDataset(Dataset):
    def __init__(self, df, timeframes, bars_per_tf, ...):
        self.timeframes = timeframes
        self.bars_per_tf = bars_per_tf  # {'5min': 60, '15min': 20, ...}
        # Pre-compute aligned windows for each timeframe

    def __getitem__(self, idx):
        # Return dict of multi-resolution tensors
        X_multi = {
            '5min':  self._get_window('5min', idx, 60),   # (60, 4)
            '15min': self._get_window('15min', idx, 20),  # (20, 4)
            '30min': self._get_window('30min', idx, 10),  # (10, 4)
            '1h':    self._get_window('1h', idx, 5),      # (5, 4)
        }
        y = self.labels[idx]
        w = self.weights[idx]
        return X_multi, y, w
```

**Models That Need This:** InceptionTime, PatchTST, iTransformer, TFT, N-BEATS

**Current Workaround:** These models would receive flattened MTF bars as part of 168 features, losing multi-resolution structure.

### 6.3 Complete 9-Timeframe Ladder

**Current:** 5 timeframes (15min, 30min, 1h, 4h, daily)
**Intended:** 9 timeframes (1min, 5min, 10min, 15min, 20min, 25min, 30min, 45min, 1h)

**Required Changes:**
1. Add missing timeframes to `MTF_TIMEFRAMES` dict:
   ```python
   # src/phase1/stages/mtf/constants.py
   MTF_TIMEFRAMES = {
       # ... existing ...
       '20min': 20,  # ← ADD
       '25min': 25,  # ← ADD
   }
   ```

2. Update `DEFAULT_MTF_TIMEFRAMES`:
   ```python
   DEFAULT_MTF_TIMEFRAMES = [
       '5min', '10min', '15min', '20min', '25min', '30min', '45min', '1h'
   ]  # Exclude 1min (same as base), keep 9-TF ladder
   ```

3. Update documentation to reflect actual vs. intended.

### 6.4 Add Model-Aware Feature Selection to Pipeline

**Current:** Feature selection only in `scripts/run_cv.py`, not in main pipeline.

**Proposed Integration:**
```python
# src/phase1/stages/datasets/run.py
def build_datasets(config, manifest):
    container = TimeSeriesDataContainer.from_parquet_dir(...)

    # If feature selection enabled
    if config.enable_feature_selection:
        from src.cross_validation.feature_selector import WalkForwardFeatureSelector

        selector = WalkForwardFeatureSelector(
            n_features_to_select=config.n_features,
            selection_method=config.selection_method
        )

        # Select features using train data only
        X_train, y_train, w_train = container.get_sklearn_arrays("train")
        cv_splits = create_cv_splits(...)
        result = selector.select_features_walkforward(X_train, y_train, cv_splits, w_train)

        # Update container with selected features
        container.config.feature_columns = result.stable_features

    return container
```

**Benefit:** Automatic feature reduction before training, prevents overfitting on 168 features.

### 6.5 Create Separate Feature Sets for Tabular vs. Sequence

**Concept:** Pre-partition features into model-appropriate subsets

```python
# src/phase1/utils/feature_sets.py
TABULAR_FEATURES = [
    "return_1", "return_5", "rsi_14", "macd_hist",
    # ... all indicator-derived features
    "sma_20_15m", "rsi_14_1h",  # MTF indicators
]

SEQUENCE_RAW_FEATURES = [
    "open", "high", "low", "close", "volume",  # Base 5min OHLCV
    "open_15m", "high_15m", ...,  # MTF bars (raw OHLCV at higher TFs)
]

# Container usage
X_tabular = container.get_features("train", feature_set="tabular")
X_sequences = container.get_features("train", feature_set="sequence_raw")
```

**Alternative:** Feature groups by type
```python
FEATURE_GROUPS = {
    "base_ohlcv": ["open", "high", "low", "close", "volume"],
    "base_indicators": ["return_1", "rsi_14", "macd_hist", ...],
    "mtf_bars": ["open_15m", "high_15m", ..., "open_4h", ...],
    "mtf_indicators": ["sma_20_15m", "rsi_14_1h", ...],
    "microstructure": ["micro_spread", "micro_amihud", ...],
    "wavelets": ["wavelet_cA3_price", ...],
}
```

---

## 7. What's Missing Entirely

### 7.1 Strategy 1: Single-TF Baselines

**Definition:** Models trained on ONE timeframe only (no MTF features)

**Purpose:** Establish baseline performance before adding MTF complexity

**Current State:** NOT IMPLEMENTED - all models get MTF features by default

**Required:** CLI flag `--disable-mtf` or `--single-tf` to skip MTF stage

### 7.2 Strategy 3: MTF Ingestion (Multi-Resolution Tensors)

**Status:** ❌ NOT IMPLEMENTED

**What's Missing:**
1. Multi-resolution sequence builder (`MultiResolutionSequenceDataset`)
2. Hierarchical temporal alignment (align 5min/15min/30min/1h windows to same prediction point)
3. Model architectures that consume multi-resolution inputs (InceptionTime, PatchTST, etc.)

**Current Limitation:** Sequence models receive 3D tensor of shape `(seq_len, 168)` where 168 includes flattened MTF bars. This loses the multi-scale temporal structure.

### 7.3 Model-Specific Data Preparation Hooks

**Concept:** Allow models to customize data preparation

```python
# Hypothetical BaseModel extension
class BaseModel(ABC):
    @classmethod
    def prepare_data(cls, container: TimeSeriesDataContainer, split: str) -> tuple:
        """Override to customize data preparation for this model."""
        # Default: return all features
        return container.get_sklearn_arrays(split)

    @abstractmethod
    def fit(self, X, y, sample_weights, ...): ...

# Example: XGBoost uses only indicator features
class XGBoostModel(BaseModel):
    @classmethod
    def prepare_data(cls, container, split):
        # Filter to indicator features only
        feature_mask = [f for f in container.feature_columns
                       if not f.endswith(('_15m', '_30m', '_1h', '_4h', '_1d'))]
        X, y, w = container.get_sklearn_arrays(split)
        X_filtered = X[:, feature_mask]
        return X_filtered, y, w
```

**Status:** NOT IMPLEMENTED - all models use generic `prepare_training_data()` utility

---

## 8. Infrastructure That Exists But Is Unused

### 8.1 MTFMode Enum and Mode-Based Routing

**Location:** `src/phase1/stages/mtf/constants.py`, `generator.py`

**Capability:** Can generate BARS only, INDICATORS only, or BOTH

**Current Usage:** Always `MTFMode.BOTH`

**Potential:** Could be used to route tabular → INDICATORS, sequence → BARS

### 8.2 WalkForwardFeatureSelector

**Location:** `src/cross_validation/feature_selector.py`

**Capability:**
- Walk-forward feature selection (prevents lookahead)
- MDI/MDA/hybrid importance
- Clustered importance (handles multicollinearity)
- Stability scoring

**Current Usage:** Only via `scripts/run_cv.py --select-features`, NOT in main pipeline

**Potential:** Could be integrated into `build_datasets` stage to auto-reduce 168 features before training

### 8.3 Model Family and Sequence Requirements Properties

**Location:** All model classes via `BaseModel`

**Properties:**
```python
@property
def model_family(self) -> str:
    return "boosting"  # or "neural", "classical", "ensemble"

@property
def requires_sequences(self) -> bool:
    return False  # or True for LSTM, GRU, etc.
```

**Current Usage:** Only for determining 2D vs 3D data shape

**Potential:** Could be used to route model families to different feature sets or MTF modes

---

## 9. Recommendations

### 9.1 Immediate Actions (Low-Hanging Fruit)

1. **Document 5 vs 9 timeframe reality**
   - Update `CLAUDE.md` and `docs/CURRENT_VS_INTENDED_ARCHITECTURE.md`
   - Clarify that 9-TF ladder is roadmap, not current state
   - List missing timeframes: 1min, 5min, 10min, 20min, 25min, 45min

2. **Add 20min and 25min to MTF_TIMEFRAMES dict**
   - Currently missing from constants, cannot be used even if requested
   - Simple addition to `src/phase1/stages/mtf/constants.py`

3. **Enable model-specific MTF modes via config**
   - Add `mtf_mode_by_family` to PipelineConfig
   - Route boosting/classical → INDICATORS, neural/cnn/advanced → BARS
   - Requires minor changes to `run_feature_engineering()`

### 9.2 Short-Term Enhancements (1-2 weeks)

1. **Integrate feature selection into pipeline**
   - Add `enable_feature_selection` flag to PipelineConfig
   - Run WalkForwardFeatureSelector during `build_datasets` stage
   - Save selected features to manifest for reproducibility

2. **Create feature group definitions**
   - Add `src/phase1/utils/feature_groups.py`
   - Define: `BASE_INDICATORS`, `MTF_INDICATORS`, `MTF_BARS`, `MICROSTRUCTURE`, `WAVELETS`
   - Allow models to request specific groups

3. **Implement Strategy 1 (Single-TF baseline)**
   - Add `--disable-mtf` CLI flag
   - Skip MTF stage in feature engineering
   - Train baseline models without MTF complexity

### 9.3 Medium-Term Refactor (3-4 weeks)

1. **Implement Strategy 3 (MTF Ingestion)**
   - Build `MultiResolutionSequenceDataset`
   - Add hierarchical temporal alignment
   - Create separate data path for sequence models

2. **Add model-specific data preparation hooks**
   - Extend `BaseModel` with `prepare_data()` classmethod
   - Allow each model to filter/transform features
   - Maintain backward compatibility with existing models

3. **Complete 9-timeframe ladder**
   - Update `DEFAULT_MTF_TIMEFRAMES` to full 9-TF ladder
   - Benchmark performance impact (more features = slower training)
   - Make configurable via `mtf_timeframes` parameter

---

## 10. Conclusion

### Summary of Findings

| Aspect | Documentation Claim | Reality |
|--------|-------------------|---------|
| **Feature Count** | ~180 features | ~168 features ✅ |
| **MTF Timeframes** | 9 timeframes | 5 timeframes ❌ |
| **MTF Modes** | Model-specific (Strategy 2/3) | All models get BOTH ❌ |
| **Sequence Data** | Multi-resolution tensors (Strategy 3) | Flattened indicators ❌ |
| **Feature Selection** | Walk-forward selection | Available but unused ❌ |
| **Model-Specific Prep** | Different pipelines | Unified pipeline ✅/❌ |
| **Data Leakage** | Prevented (MTF shift, purge/embargo) | ✅ Correct |

### Key Insights

1. **Universal Feature Set:** All 13 models (6 tabular + 4 neural + 3 classical) receive the same ~168 indicator-derived features. No differentiation by model type.

2. **MTF Mode Unused:** Infrastructure exists (`MTFMode.BARS`, `INDICATORS`, `BOTH`) but is not leveraged for model-specific routing. Always defaults to `BOTH`.

3. **Timeframe Gap:** Documentation claims 9-timeframe MTF ladder, reality is 5 timeframes. Missing: 1min, 5min, 10min, 20min, 25min, 45min.

4. **Strategy Maturity:**
   - Strategy 1 (Single-TF): ❌ Not implemented
   - Strategy 2 (MTF Indicators): ⚠️ Partial (5 of 9 TFs, all models)
   - Strategy 3 (MTF Ingestion): ❌ Not implemented (no multi-resolution tensors)

5. **Leakage Prevention:** ✅ Excellent - MTF shift(1), purge/embargo, label_end_times all correctly implemented.

6. **Feature Selection:** Exists (`WalkForwardFeatureSelector`) but is NOT integrated into main pipeline. Only accessible via separate CV script.

### Refactoring Priority

**High Priority (Enable Model Differentiation):**
1. Model-specific MTF modes (BARS for sequence, INDICATORS for tabular)
2. Feature group definitions (allow models to request subsets)
3. Integrate feature selection into pipeline

**Medium Priority (Complete MTF Implementation):**
1. 9-timeframe ladder (add missing timeframes)
2. Multi-resolution sequence builder (Strategy 3)
3. Single-TF baseline support (Strategy 1)

**Low Priority (Nice-to-Have):**
1. Model-specific data preparation hooks
2. Dynamic feature filtering in TimeSeriesDataContainer
3. Automatic feature importance tracking per model

### Final Verdict

**The pipeline is well-engineered and leakage-free, but currently serves a one-size-fits-all feature set to all models.** The infrastructure exists to enable model-specific pipelines (MTFMode, feature groups, selection), but these capabilities are not yet wired together for differentiated data preparation.

**Recommended Path Forward:** Implement model-aware MTF routing first (quick win), then build multi-resolution sequences for advanced models (larger effort). This aligns with the roadmap in `docs/implementation/MTF_IMPLEMENTATION_ROADMAP.md`.
