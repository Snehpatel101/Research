# Multi-Timeframe (MTF) Feature Configurations

**Complete reference for multi-timeframe feature engineering strategies**

This document specifies exact configurations for generating, selecting, and optimizing multi-timeframe features in the ML Model Factory.

---

## Table of Contents

- [Overview](#overview)
- [MTF Architecture](#mtf-architecture)
- [Timeframe Configurations](#timeframe-configurations)
- [MTF Feature Generation Strategies](#mtf-feature-generation-strategies)
- [MTF Feature Selection](#mtf-feature-selection)
- [Performance Optimization](#performance-optimization)
- [Usage Examples](#usage-examples)

---

## Overview

### What are MTF Features?

**Multi-Timeframe (MTF) features** provide the model with price action context from higher timeframes, capturing trends and patterns that are invisible at the base timeframe.

**Example:**
- Base: 5min bars (short-term noise and microstructure)
- MTF 15min: Intraday patterns and trends
- MTF 60min: Hourly support/resistance and momentum
- MTF Daily: Long-term trend and regime

### Why MTF?

**Benefits:**
1. **Multi-scale context:** Model sees both short-term and long-term patterns
2. **Trend detection:** Higher timeframes reveal trends obscured by base TF noise
3. **Regime awareness:** Daily/4h features capture market regime shifts
4. **Support/resistance:** MTF levels act as dynamic barriers

**Trade-offs:**
- Increases feature count significantly (+40-60 features per MTF timeframe)
- Risk of multicollinearity (MTF features correlated with base TF)
- Computational cost (resampling + indicator calculation)

**When to use:**
- Sufficient training data (>50k bars at base TF)
- Models can handle high-dimensional input (boosting, neural networks)
- Cross-timeframe patterns are predictive (most markets)

**When to skip:**
- Very limited data (<10k bars)
- Shallow models prone to overfitting (logistic regression)
- Pure high-frequency strategies (sub-minute trading)

---

## MTF Architecture

### Two-Component System

**1. MTF OHLCV Bars (5 features per TF)**
- Resampled OHLCV data from higher timeframe
- Example: `open_15m`, `high_15m`, `low_15m`, `close_15m`, `volume_15m`

**2. MTF Indicators (15-20 features per TF)**
- Technical indicators computed on MTF OHLCV
- Example: `rsi_14_15m`, `sma_50_15m`, `atr_14_15m`

### Implementation

**Source:** `src/phase1/stages/mtf/generator.py`

**Key Functions:**
- `resample_to_higher_timeframe()`: Resample OHLCV bars
- `compute_mtf_indicators()`: Calculate indicators on MTF data
- `add_mtf_features()`: Main entry point for MTF feature generation

---

## Timeframe Configurations

### Supported Timeframes

**All timeframes must be integer multiples of base timeframe.**

**Mapping table:**

| Timeframe String | Minutes | Multiplier (from 5min) | Bars per Day (6.5h) |
|------------------|---------|------------------------|---------------------|
| `5min` (base) | 5 | 1x | 78 |
| `10min` | 10 | 2x | 39 |
| `15min` | 15 | 3x | 26 |
| `20min` | 20 | 4x | 19.5 |
| `30min` | 30 | 6x | 13 |
| `45min` | 45 | 9x | 8.67 |
| `60min` or `1h` | 60 | 12x | 6.5 |
| `4h` or `240min` | 240 | 48x | 1.625 |
| `daily` or `1d` | 1440 | 288x | 1 |

**Base TF constraint:** MTF timeframes > base timeframe only.
- Valid: 5min base → [15min, 60min, daily]
- Invalid: 5min base → [1min, 5min] (not higher)

### Recommended MTF Strategies

**Strategy 1: Intraday Trading (5min base)**

| Purpose | Timeframes | Total MTF Features |
|---------|------------|-------------------|
| Short-term patterns | Base: 5min | 0 (base only) |
| Intraday trends | MTF: [15min] | ~25 |
| Hourly momentum | MTF: [15min, 60min] | ~50 |
| Daily regime | MTF: [15min, 60min, daily] | ~75 |

**Recommendation:** `[15min, 60min]` for balanced intraday context.

**Strategy 2: Swing Trading (15min base)**

| Purpose | Timeframes | Total MTF Features |
|---------|------------|-------------------|
| Short-term patterns | Base: 15min | 0 |
| Multi-hour trends | MTF: [60min] | ~25 |
| Daily regime | MTF: [60min, daily] | ~50 |

**Recommendation:** `[60min, daily]` for swing context.

**Strategy 3: Position Trading (60min base)**

| Purpose | Timeframes | Total MTF Features |
|---------|------------|-------------------|
| Hourly patterns | Base: 60min | 0 |
| Daily trends | MTF: [daily] | ~25 |

**Recommendation:** `[daily]` for long-term regime.

**Strategy 4: Minimal (Fast Training)**

| Purpose | Timeframes | Total MTF Features |
|---------|------------|-------------------|
| Base TF only | Base: 5min | 0 |

**Recommendation:** No MTF for fast prototyping or limited data.

### Data Requirements

**Minimum bars required per MTF timeframe:**

| MTF Timeframe | Min Base Bars (5min) | Equivalent Duration |
|---------------|---------------------|---------------------|
| 15min | 150 | ~12.5 hours |
| 30min | 300 | ~25 hours |
| 60min | 600 | ~50 hours (~2 days) |
| 4h | 2400 | ~200 hours (~8 days) |
| daily | 14400 | ~1200 hours (~50 days) |

**Formula:**
```python
min_base_bars = mtf_min_bars * (mtf_timeframe_minutes / base_timeframe_minutes)
```

**Default `mtf_min_bars = 30`** (ensures sufficient lookback for longest indicator).

---

## MTF Feature Generation Strategies

### Strategy 1: OHLCV Only

**Config:**
```python
{
    "enable_mtf": True,
    "mtf_timeframes": ["15min", "60min"],
    "mtf_include_ohlcv": True,
    "mtf_include_indicators": False
}
```

**Features Generated:**
- 15min: `open_15m`, `high_15m`, `low_15m`, `close_15m`, `volume_15m`
- 60min: `open_1h`, `high_1h`, `low_1h`, `close_1h`, `volume_1h`
- **Total:** 10 features

**When to Use:**
- Minimal MTF context needed
- Model learns patterns from raw prices
- Fastest MTF generation

### Strategy 2: Indicators Only

**Config:**
```python
{
    "enable_mtf": True,
    "mtf_timeframes": ["15min", "60min"],
    "mtf_include_ohlcv": False,
    "mtf_include_indicators": True
}
```

**Features Generated:**
- 15min: `rsi_14_15m`, `sma_50_15m`, `atr_14_15m`, ...
- 60min: `rsi_14_1h`, `sma_50_1h`, `atr_14_1h`, ...
- **Total:** ~30-40 features

**When to Use:**
- Want processed indicators, not raw prices
- Reduce feature count slightly
- Leverage pre-computed indicators

### Strategy 3: Full (OHLCV + Indicators) - RECOMMENDED

**Config:**
```python
{
    "enable_mtf": True,
    "mtf_timeframes": ["15min", "60min"],
    "mtf_include_ohlcv": True,
    "mtf_include_indicators": True
}
```

**Features Generated:**
- 15min: 5 OHLCV + ~15 indicators = ~20 features
- 60min: 5 OHLCV + ~15 indicators = ~20 features
- **Total:** ~40 features

**When to Use:**
- Default recommendation
- Maximum MTF information
- Model can handle dimensionality

### Strategy 4: Selective Indicators

**Config:**
```python
{
    "enable_mtf": True,
    "mtf_timeframes": ["15min", "60min"],
    "mtf_include_ohlcv": True,
    "mtf_include_indicators": True,
    "mtf_indicators": [  # Custom indicator subset
        "rsi_14",
        "sma_20",
        "sma_50",
        "atr_14",
        "hvol_20",
        "macd_hist"
    ]
}
```

**Features Generated:**
- 15min: 5 OHLCV + 6 indicators = 11 features
- 60min: 5 OHLCV + 6 indicators = 11 features
- **Total:** ~22 features

**When to Use:**
- Want specific MTF indicators
- Reduce dimensionality
- Know which MTF indicators are predictive

**Note:** This is a future enhancement. Current implementation uses default indicator set.

---

## MTF Feature Selection

### Challenge: MTF Multicollinearity

**Problem:** MTF features are highly correlated with base TF features.

**Example:**
- `close` (5min) ≈ `close_15m` (smoothed version)
- `rsi_14` (5min) ≈ `rsi_14_15m` (similar information)
- Correlation coefficients often >0.8

**Solution:** Use **Clustered MDA feature selection** to handle correlation.

### Recommended Selection Strategy

**Step 1: Generate All MTF Features**
```python
engineer = FeatureEngineer(
    enable_mtf=True,
    mtf_timeframes=["15min", "60min"],
    mtf_include_ohlcv=True,
    mtf_include_indicators=True
)
df_features, report = engineer.engineer_features(df, symbol)
```

**Total features:** ~175 base + ~40 MTF = ~215 features

**Step 2: Clustered MDA Selection**
```python
from src.cross_validation import WalkForwardFeatureSelector

selector = WalkForwardFeatureSelector(
    n_features_to_select=60,  # Increased for MTF
    selection_method="mda",
    use_clustered_importance=True,  # CRITICAL for MTF
    max_clusters=30,  # More clusters for more features
    min_feature_frequency=0.6
)

result = selector.select_features_walkforward(X, y, cv_splits)
stable_features = result.stable_features
```

**Why Clustered MDA?**
- Groups correlated features (e.g., `rsi_14` and `rsi_14_15m`) into clusters
- Computes importance per cluster (not inflated by redundancy)
- Distributes importance within cluster
- Ensures diverse feature selection across base + MTF

**Expected Outcome:**
- ~60 features selected from 215 total
- Mix of base TF and MTF features
- Reduced redundancy

### MTF-Specific Selection Heuristics

**Feature Importance Patterns:**

| Feature Type | Typical Importance | Reason |
|--------------|-------------------|--------|
| Base TF OHLCV | Medium | Core price action |
| Base TF indicators | High | Direct predictive signals |
| MTF OHLCV | Medium | Provides trend context |
| MTF indicators | Medium-High | Trend + regime detection |

**Common Selected MTF Features:**
1. `close_15m`, `close_1h` (trend direction)
2. `rsi_14_15m` (momentum regime)
3. `sma_50_1h` (hourly trend)
4. `atr_14_1h` (volatility regime)
5. `hvol_20_1h` (realized vol regime)
6. `macd_hist_15m` (intraday momentum shift)

**Common Rejected MTF Features (redundant):**
- `open_15m`, `high_15m`, `low_15m` (less informative than close)
- Short-term MTF indicators (e.g., `return_1_15m`) when base has `return_1`
- Duplicate signals across MTF timeframes

---

## Performance Optimization

### Computational Costs

**MTF generation time (5min → 15min + 60min, full indicators):**

| Dataset Size (5min bars) | MTF Generation Time | Speedup Tips |
|---------------------------|---------------------|--------------|
| 10k bars | ~5 sec | Fast already |
| 50k bars | ~15 sec | Use Numba-optimized functions |
| 100k bars | ~30 sec | Parallelize MTF timeframes |
| 500k bars | ~2 min | Cache MTF data if reusing |

**Bottlenecks:**
1. Resampling: Fast (pandas built-in)
2. Indicator calculation: Medium (Numba-accelerated)
3. Lagging/shifting: Fast (vectorized)

### Optimization Strategies

**1. Disable Unnecessary MTF Timeframes**

```python
# SLOW: Too many MTF timeframes
mtf_timeframes = ["10min", "15min", "20min", "30min", "45min", "60min", "4h", "daily"]

# FAST: Selective timeframes
mtf_timeframes = ["15min", "60min"]  # 75% fewer computations
```

**2. Skip Indicators if OHLCV Sufficient**

```python
# SLOW: Full indicators (~20 features per TF)
mtf_include_indicators = True

# FAST: OHLCV only (5 features per TF)
mtf_include_indicators = False
```

**3. Parallelize MTF Timeframes**

```python
# Future enhancement: Parallel MTF generation
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    mtf_futures = {
        executor.submit(resample_and_compute, df, tf): tf
        for tf in mtf_timeframes
    }
```

**4. Cache MTF Data for Reuse**

```python
# Save MTF features to disk if reusing
mtf_cache_path = output_dir / f"{symbol}_mtf_{timeframe}.parquet"

if mtf_cache_path.exists():
    df_mtf = pd.read_parquet(mtf_cache_path)
else:
    df_mtf = add_mtf_features(df, ...)
    df_mtf.to_parquet(mtf_cache_path)
```

### Memory Optimization

**Memory usage (100k base bars, 2 MTF timeframes, full indicators):**

| Component | Memory |
|-----------|--------|
| Base OHLCV | ~10 MB |
| Base features (~175) | ~140 MB |
| MTF features (~40) | ~32 MB |
| **Total** | **~182 MB** |

**Tips:**
- Use `dtype=float32` instead of `float64` (50% memory reduction)
- Drop MTF intermediate columns after merge
- Process symbols sequentially (not all in memory)

---

## Usage Examples

### Example 1: Default MTF Setup (5min base)

**Config:**
```python
from src.phase1.stages.features import FeatureEngineer

engineer = FeatureEngineer(
    input_dir="data/clean",
    output_dir="data/features",
    timeframe="5min",
    enable_mtf=True,
    mtf_timeframes=["15min", "60min"],
    mtf_include_ohlcv=True,
    mtf_include_indicators=True
)

df_features, report = engineer.engineer_features(df, symbol="MES")

print(f"Base features: {report['features_added'] - report['mtf_feature_count']}")
print(f"MTF features: {report['mtf_feature_count']}")
print(f"Total: {report['final_columns']}")
```

**Output:**
```
Base features: 175
MTF features: 42
Total: 222
```

### Example 2: Minimal MTF (15min only, OHLCV only)

**Config:**
```python
engineer = FeatureEngineer(
    input_dir="data/clean",
    output_dir="data/features",
    timeframe="5min",
    enable_mtf=True,
    mtf_timeframes=["15min"],  # Single TF
    mtf_include_ohlcv=True,    # OHLCV only
    mtf_include_indicators=False  # Skip indicators
)

df_features, report = engineer.engineer_features(df, symbol="MES")
```

**Output:**
```
MTF features: 5 (open_15m, high_15m, low_15m, close_15m, volume_15m)
```

### Example 3: Aggressive MTF (Multiple TFs, Full Indicators)

**Config:**
```python
engineer = FeatureEngineer(
    input_dir="data/clean",
    output_dir="data/features",
    timeframe="5min",
    enable_mtf=True,
    mtf_timeframes=["15min", "30min", "60min", "4h", "daily"],
    mtf_include_ohlcv=True,
    mtf_include_indicators=True
)

df_features, report = engineer.engineer_features(df, symbol="MES")
```

**Output:**
```
MTF features: ~100-125 (5 TFs × ~20-25 features each)
Total features: ~275-300
```

**Recommendation:** Use **Clustered MDA** with `max_clusters=50` for feature selection.

### Example 4: MTF with Feature Selection

**Full Pipeline:**
```python
# Step 1: Generate all features including MTF
engineer = FeatureEngineer(
    timeframe="5min",
    enable_mtf=True,
    mtf_timeframes=["15min", "60min"]
)
df_features, _ = engineer.engineer_features(df, "MES")

print(f"Total features: {len(df_features.columns)}")  # ~220

# Step 2: Prepare for selection
feature_cols = [c for c in df_features.columns if c not in ['datetime', 'label_20']]
X = df_features[feature_cols]
y = df_features['label_20']

# Step 3: Walk-forward feature selection with clustering
from src.cross_validation import WalkForwardFeatureSelector, PurgedKFold

cv = PurgedKFold(n_splits=5, purge_bars=60, embargo_bars=1440)
cv_splits = list(cv.split(X, y))

selector = WalkForwardFeatureSelector(
    n_features_to_select=60,
    selection_method="mda",
    use_clustered_importance=True,
    max_clusters=30,
    min_feature_frequency=0.6
)

result = selector.select_features_walkforward(X, y, cv_splits)
stable_features = result.stable_features

print(f"Selected features: {len(stable_features)}")  # ~60
print(f"MTF features in selection: {sum('_15m' in f or '_1h' in f for f in stable_features)}")

# Step 4: Train model on selected features
X_selected = X[stable_features]
```

**Expected Output:**
```
Total features: 222
Selected features: 57
MTF features in selection: 18 (32% of selected)
```

---

## YAML Configuration Templates

### Template 1: Default MTF Strategy

```yaml
# config/features/mtf_strategies.yaml

mtf_default:
  description: "Balanced intraday MTF context (5min base)"
  base_timeframe: "5min"
  enable_mtf: true
  mtf_timeframes:
    - "15min"
    - "60min"
  mtf_include_ohlcv: true
  mtf_include_indicators: true
  expected_mtf_features: 40-50
  data_requirement: "Minimum 50k 5min bars (~4 months)"

mtf_minimal:
  description: "Minimal MTF (15min only, fast training)"
  base_timeframe: "5min"
  enable_mtf: true
  mtf_timeframes:
    - "15min"
  mtf_include_ohlcv: true
  mtf_include_indicators: true
  expected_mtf_features: 20-25
  data_requirement: "Minimum 10k 5min bars (~2 weeks)"

mtf_aggressive:
  description: "Maximum MTF context (multiple TFs)"
  base_timeframe: "5min"
  enable_mtf: true
  mtf_timeframes:
    - "15min"
    - "30min"
    - "60min"
    - "4h"
    - "daily"
  mtf_include_ohlcv: true
  mtf_include_indicators: true
  expected_mtf_features: 100-125
  data_requirement: "Minimum 100k 5min bars (~1 year)"

mtf_disabled:
  description: "No MTF (base TF only, fast prototyping)"
  base_timeframe: "5min"
  enable_mtf: false
  mtf_timeframes: []
  expected_mtf_features: 0
```

### Template 2: MTF Feature Selection

```yaml
# config/features/mtf_feature_selection.yaml

mtf_selection:
  description: "Feature selection config for MTF datasets"

  # Use clustered MDA for MTF correlation
  method: mda
  use_clustered_importance: true
  max_clusters: 30

  # Select more features to accommodate MTF
  n_features_to_select: 60  # vs 50 for base-only

  # Stricter stability threshold for MTF
  min_feature_frequency: 0.6

  # Random Forest params
  n_estimators: 100
  max_depth: 5
  n_repeats: 10

mtf_selection_fast:
  description: "Fast MTF feature selection (prototyping)"
  method: mdi  # Faster method
  n_features_to_select: 40
  min_feature_frequency: 0.5
  n_estimators: 50

mtf_selection_robust:
  description: "Robust MTF feature selection (production)"
  method: mda
  use_clustered_importance: true
  max_clusters: 40  # More clusters for more features
  n_features_to_select: 70
  min_feature_frequency: 0.7  # Stricter stability
  n_estimators: 100
```

### Template 3: Loading MTF Config

```python
import yaml
from pathlib import Path

# Load MTF strategy
config_path = Path("config/features/mtf_strategies.yaml")
with open(config_path) as f:
    mtf_config = yaml.safe_load(f)

# Use default strategy
params = mtf_config['mtf_default']

engineer = FeatureEngineer(
    input_dir="data/clean",
    output_dir="data/features",
    timeframe=params['base_timeframe'],
    enable_mtf=params['enable_mtf'],
    mtf_timeframes=params['mtf_timeframes'],
    mtf_include_ohlcv=params['mtf_include_ohlcv'],
    mtf_include_indicators=params['mtf_include_indicators']
)
```

---

## Common Pitfalls

### 1. MTF Timeframe Too Close to Base

**Problem:**
```python
mtf_timeframes = ["10min"]  # Only 2x base TF
```

**Issue:** Minimal additional information, high correlation.

**Solution:**
```python
mtf_timeframes = ["15min", "60min"]  # 3x and 12x base TF
```

**Guideline:** Use MTF timeframes ≥3x base TF for meaningful context.

### 2. Too Many MTF Timeframes

**Problem:**
```python
mtf_timeframes = ["10min", "15min", "20min", "30min", "45min", "60min", "4h", "daily"]
```

**Issue:** Excessive features (~160 MTF), high correlation, slow generation.

**Solution:**
```python
mtf_timeframes = ["15min", "60min"]  # Selective, meaningful jumps
```

**Guideline:** 2-3 MTF timeframes is usually sufficient.

### 3. Insufficient Data for High MTF Timeframes

**Problem:**
```python
mtf_timeframes = ["daily"]
# But only 5k 5min bars (~1 week data)
```

**Issue:** Only ~7 daily bars → insufficient for daily indicators.

**Solution:**
```python
# Check data requirement before adding MTF
min_required = 30 * (1440 / 5)  # 30 daily bars in 5min terms
if len(df) >= min_required:
    mtf_timeframes.append("daily")
```

**Guideline:** Ensure ≥30 bars at highest MTF timeframe.

### 4. Not Using Clustered MDA

**Problem:**
```python
# Standard MDA on MTF dataset
selector = WalkForwardFeatureSelector(
    selection_method="mda",
    use_clustered_importance=False  # WRONG for MTF
)
```

**Issue:** Correlated MTF features inflate importance, poor selection.

**Solution:**
```python
selector = WalkForwardFeatureSelector(
    selection_method="mda",
    use_clustered_importance=True,  # CRITICAL for MTF
    max_clusters=30
)
```

---

## See Also

- [Feature Catalog](./FEATURE_CATALOG.md) - All features including MTF
- [Feature Selection Configs](./FEATURE_SELECTION_CONFIGS.md) - Selection methods
- [Model Feature Requirements](./MODEL_FEATURE_REQUIREMENTS.md) - Model-specific needs
- [Performance Tuning Guide](/docs/guides/PERFORMANCE_TUNING.md) - Optimization strategies
