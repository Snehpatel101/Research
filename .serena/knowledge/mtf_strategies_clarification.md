# Multi-Timeframe (MTF) Strategies - Clarification

## Critical Clarification: MTF is a Capability, Not a Strategy

**IMPORTANT:** The term "MTF strategies" is misleading. MTF upscaling (Phase 2) is a **core pipeline capability** that generates multi-timeframe views of OHLCV data. It is NOT a user-selectable "strategy."

**What users actually select:**
1. **Which model to train** (XGBoost, LSTM, PatchTST, etc.)
2. **Which hyperparameters to use** (seq_len, learning_rate, etc.)

**What the pipeline does automatically:**
1. Generate 9 MTF views (1min → 1h) in Phase 2
2. Compute indicator features from MTF views in Phase 3
3. Serve data in model-appropriate format via adapters in Phase 5

---

## MTF Pipeline Flow (Automatic)

### Phase 2: MTF Upscaling (Automatic)
**Input:** `data/processed/{symbol}_1m_clean.parquet`
**Output:** 9 MTF views
```
1min → 5min → 10min → 15min → 20min → 25min → 30min → 45min → 1h
```

**Operations:**
- Resample to higher timeframes (OHLCV aggregation)
- Align to 5-minute base index (forward-fill)
- Apply shift(1) to prevent lookahead

**Status:** ⚠️ 5 of 9 timeframes implemented (15min, 30min, 1h, 4h, daily)

---

### Phase 3: Feature Engineering (Automatic)
**Input:** Base OHLCV + 9 MTF views
**Output:** ~180 features

**Indicator Features (~150):**
- RSI, MACD, ATR, Bollinger, ADX (from 5-minute base)
- Wavelets, microstructure, statistical features

**MTF Features (~30):**
- Key indicators (RSI, MACD, ATR) from 9 timeframes
- Trend alignment indicators across timeframes
- Volatility ratios between timeframes

**Status:** ✅ Complete (but only 5 timeframes, intended: 9)

---

### Phase 5: Model-Family Adapters (Automatic)
**Input:** Canonical dataset (~180 features)
**Output:** Model-specific format

| Model Family | Adapter | Output Shape | Data Type |
|--------------|---------|--------------|-----------|
| Tabular (Boosting, Classical) | Tabular | 2D `(N, 180)` | Indicator features |
| Sequence (Neural, CNN) | Sequence | 3D `(N, T, 180)` | Indicator features in windows |
| Advanced (PatchTST, iTransformer, TFT) | Multi-Res | 4D `(N, 9, T, 4)` | Raw OHLCV from 9 timeframes |

**Status:**
- ✅ Tabular adapter (2D)
- ✅ Sequence adapter (3D)
- ❌ Multi-res adapter (4D) - planned

---

## Model-Specific Data Consumption

### Tabular Models (2D)
**Models:** XGBoost, LightGBM, CatBoost, Random Forest, Logistic, SVM

**Data format:** 2D arrays `(N, 180)`
- All ~180 features as flat vectors
- Includes MTF indicators (e.g., RSI_15min, MACD_1h)

**MTF benefit:** Tabular models can learn feature interactions across timeframes (e.g., "buy when RSI_5min < 30 AND RSI_1h > 50")

**Adapter:** Tabular adapter (Phase 5)

---

### Sequence Models (3D)
**Models:** LSTM, GRU, TCN, Transformer, InceptionTime, 1D ResNet

**Data format:** 3D windows `(N, seq_len, 180)`
- Lookback windows (e.g., 30 bars)
- All ~180 features at each timestep
- Includes MTF indicators as time-series

**MTF benefit:** Sequence models can learn temporal dynamics of MTF indicators (e.g., "RSI_1h crosses above 50 while RSI_5min is trending up")

**Adapter:** Sequence adapter (Phase 5)

---

### Multi-Resolution Models (4D)
**Models:** PatchTST, iTransformer, TFT, N-BEATS

**Data format:** 4D tensors `(N, 9, T, 4)`
- N: samples
- 9: timeframes (1min, 5min, 10min, ..., 1h)
- T: lookback window (varies by timeframe)
- 4: OHLC features

**MTF benefit:** Multi-resolution models directly learn from raw OHLCV bars across timeframes, capturing multi-scale temporal patterns (e.g., "5-min bar pattern within 1h context")

**Adapter:** Multi-res adapter (Phase 5) - ❌ not implemented

**Example 4D tensor structure:**
```python
# Shape: (N, 9, T, 4)
X = np.array([
    [  # Sample 0
        [[o1, h1, l1, c1], ..., [o_T, h_T, l_T, c_T]],  # 1-min bars (T=60)
        [[o1, h1, l1, c1], ..., [o_T, h_T, l_T, c_T]],  # 5-min bars (T=60)
        [[o1, h1, l1, c1], ..., [o_T, h_T, l_T, c_T]],  # 10-min bars (T=30)
        ...
        [[o1, h1, l1, c1], ..., [o_T, h_T, l_T, c_T]],  # 1h bars (T=5)
    ],
    ...
])
```

---

## Historical "Strategy" Terminology (Deprecated)

### Old Terminology: "MTF Strategies"
The old roadmap (`docs/archive/roadmaps/MTF_IMPLEMENTATION_ROADMAP.md`) used "strategy" terminology:

- **Strategy 1:** Single-TF (no MTF)
- **Strategy 2:** MTF Indicators (current implementation)
- **Strategy 3:** MTF Raw Ingestion (multi-resolution models)

**Problem with this terminology:**
- Implies user choice ("which strategy should I use?")
- Confusing (MTF is automatic, not optional)
- Suggests separate pipelines (wrong - ONE pipeline)

---

### New Terminology: Model-Specific Data Consumption
**Correct framing:**
1. **Pipeline generates MTF views** (Phase 2, automatic)
2. **Pipeline computes MTF features** (Phase 3, automatic)
3. **Adapters serve data in model-appropriate format** (Phase 5, automatic)

**User only chooses:**
- Which model to train (XGBoost, LSTM, PatchTST, etc.)
- Which hyperparameters to use

**Pipeline handles the rest automatically.**

---

## Single-TF "Strategy" (Ablation Study)

### Purpose
**Ablation study:** Measure MTF value by training models WITHOUT MTF features.

### Implementation
**Not a separate pipeline** - just a config flag:
```yaml
# config/pipeline.yaml
phase3:
  features:
    enable_mtf: false  # Disable MTF features for ablation
```

**Effect:**
- Phase 2 still runs (MTF views generated)
- Phase 3 skips MTF feature computation (~150 base features only)
- Phase 5 adapters serve base features only

**Use case:**
- Baseline comparison ("how much does MTF help?")
- Faster training (fewer features)
- Debugging (simpler feature set)

**Status:** ❌ Not implemented (simple config flag)

---

## MTF Implementation Gaps

### Gap 1: Missing Timeframes (Phase 2)
**Current:** 5 timeframes (15min, 30min, 1h, 4h, daily)
**Intended:** 9 timeframes (1min, 5min, 10min, 15min, 20min, 25min, 30min, 45min, 1h)

**Impact:**
- MTF features incomplete (~30 features from 5 TFs, intended: ~30 from 9 TFs)
- Multi-resolution adapter blocked (requires 9 TFs)
- Advanced models blocked (PatchTST, iTransformer, TFT)

**Effort:** 1-2 days (add 4 missing timeframes)

---

### Gap 2: Multi-Resolution Adapter (Phase 5)
**Current:** Only 2D (tabular) and 3D (sequence) adapters exist

**Missing:** 4D multi-resolution adapter
- Shape: `(N, 9, T, 4)` (raw OHLCV from 9 timeframes)
- Input: Raw MTF OHLCV (not indicator features)
- Output: `TimeSeriesDataContainer` with 4D data

**Impact:**
- Advanced models cannot be trained (PatchTST, iTransformer, TFT, N-BEATS)
- Missing multi-scale temporal learning capability

**Effort:** 3 days (create multi-res adapter, update data container)

---

### Gap 3: Advanced Models (Phase 6)
**Current:** 13 models (boosting, neural, classical, ensemble)

**Missing:** 6 advanced models
- CNN (2): InceptionTime, 1D ResNet (3D input)
- Transformers (3): PatchTST, iTransformer, TFT (4D input)
- MLP (1): N-BEATS (3D or 4D input)

**Impact:**
- Missing SOTA time-series models
- Cannot leverage multi-resolution temporal learning

**Effort:** 14-18 days (see `docs/archive/roadmaps/ADVANCED_MODELS_ROADMAP.md`)

---

## Common Questions

### Q1: "Should I use Strategy 2 or Strategy 3?"
**A:** This is a misleading question. You don't choose a "strategy."

**Correct question:** "Which model should I train?"
- **Tabular model (XGBoost, etc.):** Gets indicator features (2D) - automatic
- **Sequence model (LSTM, etc.):** Gets indicator features in windows (3D) - automatic
- **Multi-res model (PatchTST, etc.):** Gets raw OHLCV from 9 TFs (4D) - automatic

---

### Q2: "How do I enable MTF features?"
**A:** MTF features are **always enabled** (part of Phase 3).

If you want to **disable** MTF features (for ablation study):
```yaml
# config/pipeline.yaml
phase3:
  features:
    enable_mtf: false  # Disable MTF features
```

---

### Q3: "Why are there only 5 timeframes instead of 9?"
**A:** Implementation gap. Phase 2 is partially complete.

**Intended:** 9 timeframes (1min → 1h)
**Current:** 5 timeframes (15min, 30min, 1h, 4h, daily)

**Fix:** Add missing timeframes (5min, 10min, 20min, 25min, 45min) to Phase 2

---

### Q4: "Can I train models without MTF data?"
**A:** Yes, use the single-TF config flag (when implemented):
```yaml
phase3:
  features:
    enable_mtf: false
```

**Result:** Models train on ~150 base features only (no MTF features)

---

### Q5: "Do all models use the same MTF data?"
**A:** No. Models consume MTF data in different formats:

| Model Family | MTF Data Format |
|--------------|-----------------|
| Tabular | MTF indicators as flat features (2D) |
| Sequence | MTF indicators in windows (3D) |
| Multi-Res | Raw OHLCV from 9 TFs (4D) |

**But all models train on the SAME canonical dataset** (adapters just reshape it).

---

## Summary

### Key Takeaways
1. **MTF is automatic** - not a user-selectable strategy
2. **ONE pipeline** - generates MTF views, computes features, serves via adapters
3. **Model-specific formats** - adapters serve data in model-appropriate shapes
4. **Current gaps** - 4 timeframes missing, multi-res adapter not implemented

### Deprecated Terminology
- ❌ "MTF Strategy 1/2/3" (implies user choice)
- ❌ "Separate MTF pipelines" (wrong architecture)

### Correct Terminology
- ✅ "MTF upscaling" (Phase 2 capability)
- ✅ "Model-family adapters" (Phase 5 transformations)
- ✅ "Model-specific data consumption" (how models use MTF data)

---

**Last Updated:** 2026-01-01
