# Current vs. Intended Architecture Analysis (2025-12-30)

## Executive Summary

**The Problem:** Documentation claims "unified pipeline with shared features" when the INTENDED design is "unified pipeline with MODEL-SPECIFIC feature/data selection."

**Current State:** All models receive the same ~180 features (base + MTF indicators), just in different shapes (2D vs 3D).

**Intended State:** Different model families should receive DIFFERENT data types based on their architectural requirements:
- **Tabular models** → Indicator-derived features (RSI, MACD, wavelets, etc.)
- **Sequence models** → Raw MTF OHLCV bars for multi-resolution temporal learning

**Gap:** Model-specific MTF strategies (from `MTF_IMPLEMENTATION_ROADMAP.md`) are NOT implemented.

---

## The Intended Architecture

### Core Principle: One Pipeline, Model-Specific Data

```
                    Single 1-Minute Raw OHLCV Dataset
                                  ↓
                    [ Resample to Multiple Timeframes ]
                    1m → 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h
                                  ↓
              ┌─────────────────────────────────────────┐
              │                                         │
              ▼                                         ▼
    [ Strategy 2: MTF Indicators ]        [ Strategy 3: MTF Ingestion ]
    For Tabular Models                    For Sequence Models
              │                                         │
              ▼                                         ▼
    Generate ~150 indicators              Keep raw OHLCV bars
    on each timeframe:                    across multiple timeframes:
    - RSI, MACD, ATR                      - 5min OHLC shape: (T, 4)
    - Wavelets, microstructure            - 15min OHLC shape: (T/3, 4)
    - Bollinger Bands                     - 30min OHLC shape: (T/6, 4)
    Flatten to single vector:             - 1h OHLC shape: (T/12, 4)
    shape (T, ~180)                       Multi-resolution tensor input
              │                                         │
              ▼                                         ▼
       XGBoost, LightGBM                    LSTM, GRU, TCN, Transformer
       CatBoost, RandomForest               (Multi-scale temporal networks)
```

### Three MTF Strategies (from MTF_IMPLEMENTATION_ROADMAP.md)

| Strategy | Data Type | Model Families | Status |
|----------|-----------|----------------|--------|
| **Strategy 1: Single-TF** | One timeframe, no MTF | All models (baselines) | ❌ Not implemented |
| **Strategy 2: MTF Indicators** | Indicator features from multiple TFs | Tabular (XGBoost, LightGBM, RF) | ⚠️ Partially implemented* |
| **Strategy 3: MTF Ingestion** | Raw OHLCV bars from multiple TFs as tensors | Sequence (LSTM, TCN, Transformer) | ❌ Not implemented |

\* Phase 1 generates MTF indicators, but there's no model-specific selection mechanism.

---

## What's Currently Implemented

### Phase 1: Universal Feature Generation

**Location:** `src/phase1/stages/`

**Process:**
1. Ingest 1-min raw OHLCV
2. Resample to 5-min base timeframe
3. Generate ~150 base features (all indicator-based)
4. MTF upscaling to 5 higher timeframes (15min, 30min, 1h, 4h, daily)
5. Generate ~30 MTF features (OHLCV + indicators from higher TFs)
6. **Total output:** ~180 features, ALL indicator-derived

**MTF Mode:**
```python
# src/phase1/stages/mtf/constants.py
class MTFMode(str, Enum):
    BARS = 'bars'           # Only OHLCV from higher TFs
    INDICATORS = 'indicators'  # Only indicators from higher TFs
    BOTH = 'both'           # Both (DEFAULT)
```

**Current default:** `MTFMode.BOTH` → ALL models get OHLCV + indicators

### Phase 2: Shape Transformation Only

**Location:** `src/models/data_preparation.py`

```python
def prepare_training_data(container, requires_sequences, sequence_length=60):
    if requires_sequences:
        # Sequence models: Get 3D windows
        train_dataset = container.get_pytorch_sequences("train", seq_len=60)
        # Shape: (n_samples, 60, 180) - same 180 features, windowed
    else:
        # Tabular models: Get 2D arrays
        X_train, y_train, w_train = container.get_sklearn_arrays("train")
        # Shape: (n_samples, 180) - same 180 features, flat
```

**Key observation:** Same 180 features, different shapes. No feature selection based on model type.

### What Models Actually Receive Today

| Model Family | Input Shape | Features | Data Type |
|--------------|-------------|----------|-----------|
| XGBoost, LightGBM, CatBoost | (n_samples, 180) | All ~180 features | Indicators + MTF indicators |
| Random Forest, Logistic, SVM | (n_samples, 180) | All ~180 features | Indicators + MTF indicators |
| LSTM, GRU, TCN, Transformer | (n_samples, 60, 180) | All ~180 features (windowed) | Indicators + MTF indicators |

**Problem:** Sequence models receive indicator-derived features when they should ideally receive raw OHLCV bars for multi-resolution temporal learning.

---

## What's Missing: Model-Specific MTF Strategies

### Gap 1: No Model-Specific Feature Selection

**Current:** All models get all ~180 features

**Needed:**
```python
# This doesn't exist yet:
def select_features_for_model(model_family: str, container: TimeSeriesDataContainer):
    if model_family == "boosting":
        # Strategy 2: MTF Indicators
        # Use indicator-derived features from multiple timeframes
        return container.get_mtf_indicators(["15min", "30min", "1h", "4h", "daily"])

    elif model_family == "neural":
        # Strategy 3: MTF Ingestion
        # Use raw OHLCV bars from multiple timeframes as separate tensors
        return container.get_multi_resolution_bars(["5min", "15min", "1h"])

    elif model_family == "classical":
        # Strategy 1: Single-TF
        # Use only base timeframe features
        return container.get_single_timeframe_features("5min")

    else:
        return container.get_sklearn_arrays("train")  # fallback
```

### Gap 2: No Multi-Resolution Tensor Inputs

**Current:** Sequence models get `(n_samples, seq_len, n_features)` with flattened features

**Needed (Strategy 3):**
```python
# Multi-resolution input for sequence models
{
    '5min':  torch.Tensor(n_samples, 60, 4),    # 60 bars, OHLCV
    '15min': torch.Tensor(n_samples, 20, 4),    # 20 bars (60/3), OHLCV
    '1h':    torch.Tensor(n_samples, 5, 4),     # 5 bars (60/12), OHLCV
}
# Models like TFT, PatchTST, TimesNet can process these jointly
```

### Gap 3: No Configurable Training Timeframe

**Current:** Hardcoded to 5-min base

**Needed:** `training_timeframe` config parameter to support:
- 1min, 5min, 10min, 15min, 20min, 25min, 30min, 45min, 1h
- Each timeframe affects which MTF timeframes are "higher" vs "base"

---

## Why This Matters

### Problem 1: Suboptimal Model Performance

**Tabular models (XGBoost, etc.):**
- ✅ **Good:** Indicator-derived features are what they expect
- ⚠️ **Suboptimal:** May receive redundant or correlated MTF features without selection
- **Fix:** Implement feature selection per model (MDA/MDI on model-specific importance)

**Sequence models (LSTM, etc.):**
- ❌ **Bad:** Receiving pre-computed indicators loses raw temporal structure
- ❌ **Missing:** Multi-resolution learning (e.g., attend to 5min + 1h simultaneously)
- **Fix:** Implement Strategy 3 (MTF Ingestion with raw bars)

### Problem 2: Misleading Documentation

**Current docs claim:**
> "All models consume identical preprocessed datasets"

**This implies:** Fair model comparison on the same data

**Reality:**
- All models get the same features, but sequence models SHOULD get different data
- We're comparing models on data that's optimized for tabular models (indicators)
- Sequence models are handicapped by not receiving raw temporal data

### Problem 3: Research Framing is Wrong

**Current framing:** "Research project comparing 13 models on shared features"

**Correct framing:** "ML factory with model-specific data pipelines, currently using universal indicator pipeline (Phase 1 complete), model-specific pipelines (Phase 2+) not yet implemented"

---

## Documentation Corrections Needed

### CLAUDE.md

**Section to ADD:**
```markdown
## Data Pipeline Architecture: Current vs. Intended

### Current Implementation (Phase 1 Complete)

**Universal Feature Pipeline:**
- All models receive ~180 indicator-derived features
- MTF indicators from 5 timeframes (15min, 30min, 1h, 4h, daily)
- Data served in model-appropriate shapes:
  - Tabular models: 2D arrays `(n_samples, 180)`
  - Sequence models: 3D windows `(n_samples, 60, 180)`

**Limitation:** Sequence models receive indicators when they should receive raw OHLCV bars.

### Intended Architecture (Roadmap)

**Model-Specific MTF Strategies:**
- **Tabular models** (XGBoost, LightGBM, etc.) → Strategy 2: MTF Indicators
  - Indicator-derived features from multiple timeframes
  - Feature selection based on model-specific importance
- **Sequence models** (LSTM, GRU, TCN, Transformer) → Strategy 3: MTF Ingestion
  - Raw OHLCV bars from multiple timeframes as multi-resolution tensors
  - Enables cross-scale temporal learning
- **All models** → Strategy 1: Single-TF (baseline option)
  - Train on single timeframe without MTF

**Status:** Strategies 1 & 3 not implemented. See `docs/roadmaps/MTF_IMPLEMENTATION_ROADMAP.md`
```

**Sections to UPDATE:**

| Current (Wrong) | Corrected |
|----------------|-----------|
| "Shared Data Contract - All models consume identical preprocessed datasets" | "Unified Pipeline - One 1-min dataset → model-specific feature/data selection (⚠️ currently all models get same indicators, roadmap has model-specific strategies)" |
| "150+ features including wavelets and microstructure" | "150+ base features + ~30 MTF features = ~180 total (all indicator-derived, raw MTF bars not yet implemented for sequence models)" |
| "Multi-timeframe analysis (5min to daily)" | "Multi-timeframe indicators (15min to daily) - Strategy 2 partially implemented. Strategy 3 (multi-resolution raw bars) not yet implemented." |

### README.md

**Add warning section:**
```markdown
## ⚠️ Current MTF Limitations

**All models currently receive the same indicator-derived features.**

The intended architecture (per MTF_IMPLEMENTATION_ROADMAP.md) includes:
- **Strategy 2:** MTF Indicators (for tabular models) - ✅ Partially implemented
- **Strategy 3:** MTF Ingestion (for sequence models) - ❌ Not implemented

**Sequence models** (LSTM, GRU, TCN, Transformer) should receive raw OHLCV bars from multiple timeframes for multi-resolution learning, but currently receive pre-computed indicators instead.

See `docs/roadmaps/MTF_IMPLEMENTATION_ROADMAP.md` for implementation plan (6-8 weeks estimated).
```

### PHASE_1.md

**Add section:**
```markdown
## MTF Feature Generation (Current Implementation)

**Mode:** `MTFMode.BOTH` (default) - generates OHLCV + indicators from higher timeframes

**Output:** ~30 MTF features added to ~150 base features = ~180 total

**All features are indicator-derived**, including:
- MTF OHLCV bars (open_15min, high_1h, close_4h, volume_daily)
- MTF indicators (rsi_15min, macd_1h, atr_4h)

**Limitation:** Raw multi-resolution bars are not preserved for sequence models. All timeframes are flattened into a single feature vector.

**Future:** Strategy 3 (MTF Ingestion) will provide multi-resolution tensor inputs for sequence models. See `docs/roadmaps/MTF_IMPLEMENTATION_ROADMAP.md`.
```

### PHASE_2.md

**Add section:**
```markdown
## Model Data Requirements

### Tabular Models (2D Input)
**Models:** XGBoost, LightGBM, CatBoost, Random Forest, Logistic, SVM

**Input:** `(n_samples, n_features)` via `container.get_sklearn_arrays()`

**Features:** All ~180 indicator-derived features

**Optimal Data (Strategy 2):** MTF indicators from multiple timeframes ✅ Currently implemented

### Sequence Models (3D Input)
**Models:** LSTM, GRU, TCN, Transformer

**Input:** `(n_samples, seq_len, n_features)` via `container.get_pytorch_sequences()`

**Current Features:** All ~180 indicator-derived features (windowed)

**Optimal Data (Strategy 3):** Multi-resolution raw OHLCV bars ❌ Not implemented
- Example: `{'5min': (T, 60, 4), '15min': (T, 20, 4), '1h': (T, 5, 4)}`
- Enables models like TFT, PatchTST to learn cross-scale patterns

**Current Limitation:** Sequence models receive indicators when they should receive raw bars.
```

---

## Summary

### ✅ What's Correctly Implemented
1. **Universal feature pipeline** generating ~180 indicator-derived features
2. **MTF upscaling** with anti-lookahead (shift + forward-fill)
3. **Dual data access patterns** (2D vs 3D shapes)
4. **13 models across 4 families** (all functional)

### ❌ What's Misleading in Docs
1. **"Shared Data Contract"** → Implies intentional design, but it's a temporary limitation
2. **"All models consume identical datasets"** → Obscures the fact that sequence models should get different data
3. **"Multi-timeframe analysis"** → Implies full MTF support, but Strategy 3 is missing

### ⚠️ What's Missing from Implementation
1. **Strategy 1:** Single-timeframe training (no MTF)
2. **Strategy 3:** Multi-resolution tensor inputs for sequence models
3. **Model-specific feature selection** (automatic filtering based on model family)
4. **Configurable training_timeframe** (currently hardcoded to 5min)

### 📋 Action Items for Documentation
1. Add "Current vs. Intended Architecture" section to CLAUDE.md
2. Add MTF limitations warning to README.md
3. Update PHASE_1.md to clarify MTF feature types (indicators vs. raw bars)
4. Update PHASE_2.md to document model data requirements and current limitations
5. Link all docs to `MTF_IMPLEMENTATION_ROADMAP.md` for future work

---

## References

- `docs/roadmaps/MTF_IMPLEMENTATION_ROADMAP.md` - 6-8 week implementation plan for 3 MTF strategies
- `src/phase1/stages/mtf/` - Current MTF indicator generation (Strategy 2 partial)
- `src/models/data_preparation.py` - Current universal data access (no model-specific logic)
