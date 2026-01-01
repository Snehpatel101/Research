# Unified Pipeline Architecture

## Core Concept

**ONE canonical pipeline** that ingests OHLCV data and serves all model families through deterministic adapters.

**NOT separate pipelines** - this is a critical architectural principle.

---

## What "Unified Pipeline" Means

### Single Data Flow
```
Raw OHLCV → Clean → Configurable TF → Features → Labels → Splits → Scaling → CANONICAL DATASET
                          |                                                            ↓
                    (MTF optional)                                      ┌──────────────┴──────────────┐
                                                                        ↓              ↓              ↓
                                                                  Tabular (2D)  Sequence (3D)  Multi-Res (4D)
                                                                  Adapter       Adapter        Adapter
                                                                        ↓              ↓              ↓
                                                                  Boosting/     Neural/        PatchTST/
                                                                  Classical     CNN            iTransformer
                                                                        └──────────────┴──────────────┘
                                                                                       ↓
                                                                        HETEROGENEOUS ENSEMBLE STACKING
                                                                        (3-4 bases → Meta-Learner → Final)
```

### Key Properties

1. **Configurable primary timeframe:** 5m/10m/15m/1h per experiment
2. **Optional MTF enrichment:** Single-TF, MTF indicators, or MTF ingestion
3. **Single source of truth:** Canonical dataset stored once in `data/splits/scaled/`
4. **Deterministic adapters:** Pure transformations (no feature engineering, no randomness)
5. **Reproducibility:** All models consume identical features/labels
6. **Heterogeneous ensembles:** 3-4 base families -> meta-learner stacking

---

## Pipeline Stages (Phases 1-5)

### Phase 1: Data Ingestion
**Input:** `data/raw/{symbol}_1m.parquet`
**Output:** `data/processed/{symbol}_1m_clean.parquet`

**Operations:**
- Schema validation (OHLCV columns, data types)
- Duplicate removal (keep last)
- Gap detection (preserved, not filled)
- Session filtering (regular vs extended hours)

**Leakage protection:** None needed (raw data only)

---

### Phase 2: Multi-Timeframe Upscaling (Optional)
**Input:** `data/processed/{symbol}_1m_clean.parquet`
**Output:** `data/processed/{symbol}_{timeframe}.parquet` (variable based on config)

**Configurable Primary Timeframe:**
- Primary TF is configurable per experiment: 5m/10m/15m/1h
- Not hardcoded to 5-min

**MTF Strategies:**
- **Strategy 1: Single-TF** - No MTF, train on primary timeframe only
- **Strategy 2: MTF Indicators** - Add indicator features from multiple TFs
- **Strategy 3: MTF Ingestion** - Raw OHLCV bars from multiple TFs (for sequence models)

**Operations (when MTF enabled):**
- Resample to higher timeframes (OHLCV aggregation)
- Align to primary timeframe index (forward-fill)
- Apply shift(1) to prevent lookahead

**Leakage protection:** shift(1) on all MTF features (prevents lookahead)

**Status:** ⚠️ 5 of 9 timeframes implemented for Strategy 2

---

### Phase 3: Feature Engineering
**Input:** Base OHLCV + MTF views
**Output:** `data/features/{symbol}_features.parquet` (~180 features)

**Operations:**
- Base indicators (~150): RSI, MACD, ATR, Bollinger, ADX
- Wavelets (~30): Db4/Haar decomposition (3 levels)
- Microstructure (~20): Spread proxies, order flow
- MTF indicators (~30): Indicators from 5 timeframes (intended: 9)

**Leakage protection:** All indicators respect shift(1) from Phase 2

**Status:** ✅ Complete (but MTF features only from 5 timeframes)

---

### Phase 4: Triple-Barrier Labeling + Splits + Scaling
**Input:** `data/features/{symbol}_features.parquet`
**Output:** `data/splits/scaled/{symbol}_{split}.parquet` (train/val/test)

**Operations:**
1. Triple-barrier labeling (profit/loss/time barriers)
2. Quality weighting (0.5x-1.5x based on barrier touches)
3. Time-series splits (70/15/15) with purge (60) + embargo (1440)
4. Robust scaling (train-only fit, transform all splits)

**Leakage protection:**
- **Purge (60 bars):** Remove overlapping labels between splits
- **Embargo (1440 bars):** Prevent serial correlation leakage
- **Train-only scaling:** Scaler never sees validation/test statistics

**Status:** ✅ Complete

---

### Phase 5: Model-Family Adapters
**Input:** `data/splits/scaled/` (canonical splits)
**Output:** `TimeSeriesDataContainer` (in-memory, model-specific shape)

**Operations:**
- **Tabular adapter:** Extract 2D arrays `(N, 180)`
- **Sequence adapter:** Create 3D windows `(N, seq_len, 180)`
- **Multi-res adapter:** Build 4D tensors `(N, 9, T, 4)` (planned)

**Leakage protection:** None needed (adapters are pure transformations)

**Status:**
- ✅ Tabular adapter (2D)
- ✅ Sequence adapter (3D)
- ❌ Multi-res adapter (4D) - planned

---

## Adapter Design Principles

### 1. Deterministic Transformations
**Adapters are pure functions:**
```python
def tabular_adapter(canonical_df: pd.DataFrame) -> TimeSeriesDataContainer:
    """Pure transformation: same input → same output."""
    X = canonical_df[feature_cols].values  # (N, 180)
    y = canonical_df["label"].values        # (N,)
    return TimeSeriesDataContainer(X_train=X, y_train=y, ...)
```

**Properties:**
- No randomness (no shuffling, no sampling, no noise)
- No feature engineering (features already computed)
- No state (no instance variables, no side effects)
- Reproducible (same input → same output)

---

### 2. Feature Selection + Shape Transformations
**Adapters select model-specific features AND transform to appropriate shape:**

| Adapter | Features | Output Shape | Transformation |
|---------|----------|--------------|----------------|
| Tabular | ~200 engineered (with MTF) | `(N, 200)` | Select features from canonical |
| Sequence | ~150 base (single TF) | `(N, T, 150)` | Select features + create windows |
| Multi-Res | Raw OHLCV streams | `(N, 9, T, 4)` | Load raw MTF OHLCV |

**Example: Sequence Adapter**
```python
# Input: canonical 1-min OHLCV + model config
canonical_1m = load_canonical("data/raw/MES_1m.parquet")
config = load_model_config("tcn")  # primary_tf=5min, mtf_strategy=single_tf

# Select features based on config
feature_selector = FeatureSelector(config)
df = feature_selector.select_features(canonical_1m)  # (N, 150)

# Create 3D windows (lookback = 60 bars)
X = create_windows(df.values, seq_len=60)  # (N-60, 60, 150)
y = df["label"].values[60:]  # (N-60,) - skip first 60 samples

# Output: TimeSeriesDataContainer with 3D data
return TimeSeriesDataContainer(X_train=X, y_train=y, ...)
```

**Key:** Adapters perform BOTH feature selection (different features per model) AND shape transformation (2D/3D/4D).

**See:** `.serena/knowledge/per_model_feature_selection.md` for feature selection logic

---

### 3. No Feature Engineering in Adapters
**Adapters do NOT compute features:**

```python
# ❌ BAD: Computing features in adapter
def bad_adapter(canonical_df):
    # DON'T DO THIS - feature engineering belongs in Phase 3
    canonical_df["new_feature"] = calculate_rsi(canonical_df["close"])
    return TimeSeriesDataContainer(...)

# ✅ GOOD: Only shape transformation
def good_adapter(canonical_df):
    # Just reshape - features already computed in Phase 3
    X = canonical_df[feature_cols].values  # Extract existing features
    X_windows = create_windows(X, seq_len=30)  # Reshape to 3D
    return TimeSeriesDataContainer(X_train=X_windows, ...)
```

**Why:**
- Feature engineering in adapters → inconsistency between models
- Feature engineering in Phase 3 → all models get same features

---

### 4. Adapter Compatibility Matrix

| Model Family | Input Shape | Adapter | Status |
|--------------|-------------|---------|--------|
| Boosting | 2D `(N, 180)` | Tabular | ✅ |
| Classical | 2D `(N, 180)` | Tabular | ✅ |
| Neural | 3D `(N, T, 180)` | Sequence | ✅ |
| CNN | 3D `(N, T, 180)` | Sequence | ✅ (adapter ready, models not implemented) |
| Advanced | 4D `(N, 9, T, 4)` | Multi-Res | ❌ (adapter not implemented) |
| MLP | 3D/4D | Sequence/Multi-Res | ⚠️ (depends on model config) |

---

## Why NOT Separate Pipelines?

### Anti-Pattern: Separate Pipelines
```
# ❌ BAD: Separate pipelines for each model family
Raw OHLCV → [Tabular Pipeline] → Tabular Dataset → Boosting Models
Raw OHLCV → [Sequence Pipeline] → Sequence Dataset → Neural Models
Raw OHLCV → [Multi-Res Pipeline] → Multi-Res Dataset → Advanced Models
```

**Problems:**
1. **Inconsistency:** Different pipelines → different features → unfair comparison
2. **Storage:** 3× storage (same data stored 3 times)
3. **Maintenance:** 3× maintenance burden (bug in one pipeline → fix 3 times)
4. **Divergence:** Pipelines drift over time (different bugs, different features)
5. **Reproducibility:** Hard to ensure all pipelines use same data

---

### Correct Pattern: ONE Pipeline + Adapters
```
# ✅ GOOD: Single pipeline with adapters
Raw OHLCV → [CANONICAL PIPELINE] → Canonical Dataset
                                           ↓
                                    ┌──────┴──────┐
                                    ↓      ↓      ↓
                               Tabular Sequence Multi-Res
                               Adapter Adapter  Adapter
                                    ↓      ↓      ↓
                               Boosting Neural Advanced
```

**Benefits:**
1. **Consistency:** Same features/labels for all models
2. **Storage:** 1× storage (canonical data stored once)
3. **Maintenance:** 1× maintenance (fix once, applies to all)
4. **Reproducibility:** Deterministic adapters guarantee consistency
5. **Fairness:** All models compete on equal footing

---

## Common Misconceptions

### Misconception 1: "Adapters are model-specific pipelines"
**FALSE.** Adapters are lightweight shape transformations, not pipelines.

- **Pipeline:** Ingests raw data, computes features, creates labels (complex, stateful)
- **Adapter:** Reshapes existing features (simple, stateless)

---

### Misconception 2: "Different models need different features"
**TRUE.** Different models get different feature sets tailored to their inductive biases.

**Reason:** Tabular models (boosting, classical) benefit from engineered MTF indicators, while sequence models (LSTM, TCN) learn from temporal context without MTF enrichment, and transformers (PatchTST, TFT) need raw multi-resolution OHLCV bars.

**Examples:**
- **CatBoost (tabular):** ~200 features (indicators + wavelets + MTF indicators from 1m/5m/15m/1h)
- **TCN (sequence):** ~150 features (indicators + wavelets, single primary TF, no MTF)
- **PatchTST (transformer):** Raw OHLCV bars from 3 timeframe streams (1m/5m/15m)

**Key:** Feature selection is deterministic and reproducible (same model config → same features), but different models intentionally receive different features for diversity.

**See:** `.serena/knowledge/per_model_feature_selection.md` for detailed feature selection strategies

---

### Misconception 3: "Adapters introduce leakage"
**FALSE.** Adapters are deterministic transformations.

Leakage is prevented in **Phases 1-4** (purge, embargo, shift(1), train-only scaling). Adapters just reshape the already-leakage-free data.

---

### Misconception 4: "We need separate MTF strategies"
**FALSE.** MTF upscaling (Phase 2) is a **capability**, not a strategy.

- **Phase 2:** Generates 9 MTF views (1min → 1h)
- **Phase 3:** Computes indicators from MTF views
- **Phase 5:** Adapters serve MTF features (2D) or raw MTF OHLCV (4D)

All models have access to MTF data. The only difference is **how they consume it** (indicators vs raw bars).

---

## Design Decisions

### Why Adapters Instead of Separate Datasets?

**Decision:** Store canonical dataset once, adapt on-the-fly.

**Rationale:**
- **Reproducibility:** All models train on identical features/labels
- **Storage:** Store data once (canonical), not per model family
- **Consistency:** No divergence (adapters are deterministic)

**Trade-off:** Adapters add ~2 seconds overhead (acceptable for consistency gain).

---

### Why Train-Only Scaling?

**Decision:** Fit scaler on train split only, transform all splits.

**Rationale:**
- **Prevents leakage:** Scaler never sees validation/test statistics
- **Realistic:** Mimics production (scaler fit on historical data)
- **Standard practice:** Industry standard for time-series ML

**Trade-off:** Validation/test may have values outside train range (handled via robust scaler).

---

### Why Purge + Embargo?

**Decision:** Remove 60 bars (purge) + 1440 bars (embargo) between splits.

**Rationale:**
- **Purge:** Labels look forward `horizon` bars; purge 3× to ensure no overlap
- **Embargo:** Financial data has serial correlation; 5 days (~1440 bars) prevents temporal leakage
- **Evidence:** Proven effective in "Advances in Financial Machine Learning" (de Prado)

**Trade-off:** Lose ~10% of data, but prevents overfitting.

---

## Future Extensions

### Multi-Resolution Adapter (4D)
**Goal:** Enable PatchTST, iTransformer, TFT, N-BEATS

**Implementation:**
```python
def multi_resolution_adapter(
    mtf_data: dict[str, pd.DataFrame],  # 9 timeframes
    lookback_window: int = 60,
) -> TimeSeriesDataContainer:
    """
    Build 4D tensor from raw MTF OHLCV.

    Args:
        mtf_data: Dict of {timeframe: OHLCV DataFrame}
        lookback_window: Lookback window in bars

    Returns:
        TimeSeriesDataContainer with 4D data (N, 9, T, 4)
    """
    # Load raw OHLCV for all 9 timeframes
    ohlc_tensors = []
    for tf in [1, 5, 10, 15, 20, 25, 30, 45, 60]:
        df = mtf_data[f"{tf}min"]
        ohlc = df[["open", "high", "low", "close"]].values  # (N, 4)
        windows = create_windows(ohlc, seq_len=lookback_window)  # (N, T, 4)
        ohlc_tensors.append(windows)

    # Stack into 4D tensor (N, 9, T, 4)
    X = np.stack(ohlc_tensors, axis=1)

    return TimeSeriesDataContainer(X_train=X, ...)
```

**Status:** ❌ Not implemented (requires Phase 2 completion: 9 timeframes)

---

**Last Updated:** 2026-01-01
